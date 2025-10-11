"""
Time Series Change Point Detection using PELT

Implements AR, VAR, and other time series models for change point detection.
AR and VAR are essentially linear regression with lagged variables.
"""

import numpy as np
from typing import Union, Dict
from fastcpd.pelt_lm import _fastcpd_lm


def _create_ar_data(data: np.ndarray, order: int) -> np.ndarray:
    """Create lagged data matrix for AR(p) model.

    AR(p): y_t = φ_1 * y_{t-1} + φ_2 * y_{t-2} + ... + φ_p * y_{t-p} + ε_t

    This is linear regression where:
    - Response: y_t (current value)
    - Predictors: y_{t-1}, y_{t-2}, ..., y_{t-p} (lagged values)

    Parameters
    ----------
    data : np.ndarray
        Univariate time series data, shape (n,) or (n, 1)
    order : int
        AR order p (number of lags)

    Returns
    -------
    np.ndarray
        Data matrix [y, X] where:
        - y has shape (n-p, 1): response starting from index p
        - X has shape (n-p, p): lagged predictors
        Combined shape: (n-p, p+1)

    Notes
    -----
    Matches R's fastcpd implementation (lines 323-331):
    ```r
    } else if (family == "ar") {
      p <- order
      fastcpd_family <- "gaussian"
      y <- data_[p + seq_len(nrow(data_) - p), ]
      x <- matrix(NA, nrow(data_) - p, p)
      for (p_i in seq_len(p)) {
        x[, p_i] <- data_[(p - p_i) + seq_len(nrow(data_) - p), ]
      }
      data_ <- cbind(y, x)
    }
    ```
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = len(data)
    p = int(order)

    if n <= p:
        raise ValueError(f"Data length {n} must be greater than AR order {p}")

    # Response: y_t for t = p+1, p+2, ..., n
    # In Python 0-indexing: data[p:n]
    y = data[p:n]

    # Predictors: lagged values
    # X[:, 0] = y_{t-1} (lag 1)
    # X[:, 1] = y_{t-2} (lag 2)
    # ...
    # X[:, p-1] = y_{t-p} (lag p)
    X = np.zeros((n - p, p))
    for lag in range(1, p + 1):
        # For lag i, we want y_{t-i}
        # At time t (index t in y), we want data[t-i]
        # Since y starts at index p, time t corresponds to data[p + t]
        # We want data[p + t - lag] = data[p - lag + t]
        X[:, lag - 1] = data[p - lag:n - lag, 0]

    # Combine [y, X] as in linear regression
    return np.column_stack([y, X])


def _create_var_data(data: np.ndarray, order: int) -> np.ndarray:
    """Create lagged data matrix for VAR(p) model.

    VAR(p): Multivariate AR where each variable is regressed on lags of all variables.

    For d variables: Y_t = Φ_1 Y_{t-1} + Φ_2 Y_{t-2} + ... + Φ_p Y_{t-p} + ε_t

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series data, shape (n, d)
    order : int
        VAR order p (number of lags)

    Returns
    -------
    np.ndarray
        Data matrix [Y, X] where:
        - Y has shape (n-p, d): multivariate response
        - X has shape (n-p, p*d): stacked lagged predictors
        Combined shape: (n-p, d + p*d)

    Notes
    -----
    Matches R's fastcpd implementation (lines 336-346):
    ```r
    } else if (family == "var") {
      p <- order * p_response^2
      fastcpd_family <- "mgaussian"
      vanilla_percentage <- 1
      y <- data_[order + seq_len(nrow(data_) - order), ]
      x <- matrix(NA, nrow(data_) - order, order * ncol(data_))
      for (p_i in seq_len(order)) {
        x[, (p_i - 1) * ncol(data_) + seq_len(ncol(data_))] <-
          data_[(order - p_i) + seq_len(nrow(data_) - order), ]
      }
      data_ <- cbind(y, x)
    }
    ```
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n, d = data.shape
    p = int(order)

    if n <= p:
        raise ValueError(f"Data length {n} must be greater than VAR order {p}")

    # Response: Y_t for t = p+1, p+2, ..., n
    Y = data[p:n]

    # Predictors: stacked lagged values
    # X = [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    # Shape: (n-p, p*d)
    X = np.zeros((n - p, p * d))
    for lag in range(1, p + 1):
        # For lag i, columns (i-1)*d : i*d contain Y_{t-i}
        start_col = (lag - 1) * d
        end_col = lag * d
        X[:, start_col:end_col] = data[p - lag:n - lag, :]

    # Combine [Y, X]
    return np.column_stack([Y, X])


def _fastcpd_ar(
    data: np.ndarray,
    order: int,
    beta: Union[str, float],
    trim: float = 0.025
) -> dict:
    """AR(p) change point detection using PELT.

    AR(p) model is just linear regression with lagged predictors.
    This function creates the lagged data matrix and calls linear regression PELT.

    Parameters
    ----------
    data : np.ndarray
        Univariate time series, shape (n,) or (n, 1)
    order : int
        AR order p
    beta : Union[str, float]
        Penalty criterion ("BIC", "MBIC", "MDL" or numeric)
    trim : float
        Boundary trimming proportion (default 0.025)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cp_set': Detected change points (adjusted back to original indices)
        - 'raw_cp_set': Raw change points before post-processing
        - 'order': AR order used

    Notes
    -----
    Change points are detected in the lagged data space (indices 0 to n-p-1),
    then adjusted by adding `order` to get indices in original data space.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = len(data)
    p = int(order)

    # Create lagged data matrix [y, X]
    ar_data = _create_ar_data(data, p)

    # Run linear regression PELT
    # p_response=1 for univariate AR
    result = _fastcpd_lm(ar_data, beta, trim, p_response=1)

    # Adjust change points: add `order` back to get original indices
    # This is because we dropped the first `order` observations
    cp_set_adjusted = result['cp_set'] + p
    raw_cp_set_adjusted = result['raw_cp_set'] + p

    return {
        'cp_set': cp_set_adjusted,
        'raw_cp_set': raw_cp_set_adjusted,
        'order': p
    }


def _fastcpd_var(
    data: np.ndarray,
    order: int,
    beta: Union[str, float],
    trim: float = 0.025
) -> dict:
    """VAR(p) change point detection using PELT.

    VAR(p) model is multivariate linear regression with lagged predictors.
    This function creates the lagged data matrix and calls linear regression PELT.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series, shape (n, d)
    order : int
        VAR order p
    beta : Union[str, float]
        Penalty criterion ("BIC", "MBIC", "MDL" or numeric)
    trim : float
        Boundary trimming proportion (default 0.025)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cp_set': Detected change points (adjusted back to original indices)
        - 'raw_cp_set': Raw change points before post-processing
        - 'order': VAR order used
        - 'p_response': Number of response variables

    Notes
    -----
    Change points are detected in the lagged data space (indices 0 to n-p-1),
    then adjusted by adding `order` to get indices in original data space.

    R uses vanilla_percentage=1 (PELT only) for VAR.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        raise ValueError("VAR requires multivariate data (at least 2 variables)")

    n, d = data.shape
    p = int(order)

    # Create lagged data matrix [Y, X]
    var_data = _create_var_data(data, p)

    # Run multivariate linear regression PELT
    # p_response=d for multivariate VAR with d variables
    result = _fastcpd_lm(var_data, beta, trim, p_response=d)

    # Adjust change points: add `order` back to get original indices
    cp_set_adjusted = result['cp_set'] + p
    raw_cp_set_adjusted = result['raw_cp_set'] + p

    return {
        'cp_set': cp_set_adjusted,
        'raw_cp_set': raw_cp_set_adjusted,
        'order': p,
        'p_response': d
    }
