"""PELT implementation for linear regression.

This module implements PELT (Pruned Exact Linear Time) for linear regression
change point detection, matching the R fastcpd implementation.
"""

import numpy as np
from typing import List, Tuple, Union
from sklearn.linear_model import LinearRegression


def variance_lm(data: np.ndarray, d: int = 1, outlier_iqr: float = np.inf) -> Union[float, np.ndarray]:
    """Estimate variance for linear models with change points.

    Matches R's variance_lm function (variance_estimation.R lines 56-123).

    Parameters:
        data: Data matrix with response as first column(s) and predictors as remaining columns
        d: Dimension of response variable (default 1)
        outlier_iqr: IQR multiplier for outlier detection (default Inf = no filtering)

    Returns:
        Variance estimate (scalar for d=1, matrix for d>1)

    Algorithm:
        1. For each block of size (ncol - d + 1):
           - Fit linear model on block
           - Fit linear model on lagged block
           - Estimate variance from difference of coefficients
        2. Average estimates across blocks
        3. Remove outliers using IQR threshold
    """
    data = np.asarray(data, dtype=np.float64)
    n, n_cols = data.shape
    block_size = n_cols - d + 1

    if block_size < 2:
        # Not enough data for variance estimation
        return 1.0 if d == 1 else np.eye(d)

    estimators = []

    for i in range(n - block_size):
        block_index = np.arange(block_size) + i
        block_index_lagged = np.arange(block_size) + i + 1

        # Extract blocks
        y_block = data[block_index, :d]
        X_block = data[block_index, d:]

        y_block_lagged = data[block_index_lagged, :d]
        X_block_lagged = data[block_index_lagged, d:]

        try:
            # Fit linear models
            # block_slope = (X'X)^{-1} X'y
            XtX = X_block.T @ X_block
            block_slope = np.linalg.solve(XtX, X_block.T @ y_block)

            XtX_lagged = X_block_lagged.T @ X_block_lagged
            block_lagged_slope = np.linalg.solve(XtX_lagged, X_block_lagged.T @ y_block_lagged)

            # Variance estimate from difference
            XtX_inv = np.linalg.inv(XtX)
            XtX_inv_lagged = np.linalg.inv(XtX_lagged)

            # Cross term (matches R lines 87-91)
            # R: crossprod(x_block[-1, ], x_block_lagged[-block_size, ])
            # R's x_block[-1, ] means "exclude row 1" → Python's X_block[1:]
            # R's x_block_lagged[-block_size, ] means "exclude last row" → Python's X_block_lagged[:-1]
            cross_term_x = X_block[1:].T @ X_block_lagged[:-1]
            cross_term = XtX_inv @ XtX_inv_lagged @ cross_term_x

            # Delta numerator and denominator (R lines 92-105)
            delta_numerator = (block_slope - block_lagged_slope).T @ (block_slope - block_lagged_slope)

            delta_denominator = np.zeros((d, d))
            for j in range(d):
                for k in range(d):
                    if j != k:
                        delta_denominator[j, k] += np.dot(
                            block_slope[:, j] - block_lagged_slope[:, k],
                            block_slope[:, j] - block_lagged_slope[:, k]
                        )

            delta_denominator += np.sum(np.diag(XtX_inv + XtX_inv_lagged - 2 * cross_term))

            variance_estimate = delta_numerator / delta_denominator
            estimators.append(variance_estimate)

        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix or other error - skip this block
            continue

    if len(estimators) == 0:
        # All blocks failed - return default
        return 1.0 if d == 1 else np.eye(d)

    # Average estimates
    if d == 1:
        # Scalar case
        estimators_array = np.array([est[0, 0] if est.ndim > 0 else est for est in estimators])
        estimators_array = estimators_array[~np.isnan(estimators_array)]

        if len(estimators_array) == 0:
            return 1.0

        # Remove outliers
        q75 = np.percentile(estimators_array, 75)
        iqr = np.percentile(estimators_array, 75) - np.percentile(estimators_array, 25)
        outlier_threshold = q75 + outlier_iqr * iqr
        filtered = estimators_array[estimators_array < outlier_threshold]

        return np.mean(filtered) if len(filtered) > 0 else np.mean(estimators_array)
    else:
        # Matrix case
        return np.mean(estimators, axis=0)


def _fastcpd_lm(
    data: np.ndarray,
    beta: Union[str, float],
    trim: float,
    p_response: int = 1,
) -> dict:
    """Linear regression change point detection using PELT.

    Matches R's fastcpd with family="lm".

    Parameters:
        data: Input data (n, d+1) with first column(s) as response, rest as predictors
        beta: Penalty value or criterion ("BIC", "MBIC", "MDL")
        trim: Boundary trim proportion
        p_response: Number of response variables (default 1)

    Returns:
        Dictionary with change points and other results
    """
    data = np.asarray(data, dtype=np.float64)
    n, n_cols = data.shape

    # Number of predictors and effective parameter count (match R)
    n_predictors = n_cols - p_response
    p = n_predictors if p_response == 1 else n_predictors * p_response

    # Estimate variance (matches R lines 391-392)
    if p_response == 1:
        variance_est = variance_lm(data, d=p_response)
    else:
        variance_est = variance_lm(data, d=p_response)
        # For multivariate, ensure positive definite
        variance_est = _nearest_pd(variance_est)

    # Calculate beta if string (matches R lines 401-420)
    if isinstance(beta, str):
        if beta == "MBIC":
            beta_val = (p + 2) * np.log(n) / 2
        elif beta == "BIC":
            beta_val = (p + 1) * np.log(n) / 2
        elif beta == "MDL":
            beta_val = (p + 2) * np.log2(n) / 2
        else:
            raise ValueError(f"Unknown beta criterion: {beta}")

        # CRITICAL: Multiply beta by variance for UNIVARIATE Gaussian only (R lines 82-84)
        # R only does this for fastcpd_family == "gaussian" (univariate)
        # NOT for "mgaussian" (multivariate/VAR)
        if p_response == 1:
            beta_val = beta_val * variance_est
        # For multivariate (VAR), beta is NOT adjusted by variance/determinant
    else:
        beta_val = float(beta)

    # Run PELT algorithm
    change_points_raw = _pelt_lm(data, beta_val, p_response, variance_est)

    # Post-process (boundary removal and merging)
    change_points = _postprocess_lm(change_points_raw, n, trim)

    return {
        'raw_cp_set': np.array(change_points_raw),
        'cp_set': np.array(change_points),
        'cost_values': np.array([]),
        'residuals': np.array([]).reshape(-1, 1),
        'thetas': np.array([]),
        'data': data,
        'family': 'lm',
    }


def _pelt_lm(
    data: np.ndarray,
    beta: float,
    p_response: int,
    variance_est: Union[float, np.ndarray],
) -> List[int]:
    """PELT algorithm for linear regression.

    Parameters:
        data: Input data (n, d+1)
        beta: Penalty value (already adjusted by variance)
        p_response: Number of response variables
        variance_est: Variance estimate

    Returns:
        List of change point indices
    """
    n = data.shape[0]
    p_predictors = data.shape[1] - p_response

    # Initialize
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    R = np.zeros(n + 1, dtype=int)
    cp_list = [0]

    # Main PELT loop
    for t in range(1, n + 1):
        candidates = []

        for tau in cp_list:
            if tau < t:
                # Segment [tau, t)
                segment_data = data[tau:t]
                segment_length = t - tau

                # Minimum segment length
                if segment_length >= p_predictors + 1:
                    # Compute cost for this segment
                    cost = _segment_cost_lm(segment_data, p_response, variance_est)
                else:
                    cost = 0

                total_cost = F[tau] + cost + beta
                candidates.append((total_cost, tau))

        if candidates:
            min_cost, min_tau = min(candidates)
            F[t] = min_cost
            R[t] = min_tau

            # Pruning
            pruned = []
            for tau in cp_list:
                if tau < t:
                    # Recompute cost for pruning check
                    segment_data = data[tau:t]
                    segment_length = t - tau

                    if segment_length >= p_predictors + 1:
                        cost = _segment_cost_lm(segment_data, p_response, variance_est)
                    else:
                        cost = 0

                    if F[tau] + cost <= F[t]:
                        pruned.append(tau)
                else:
                    pruned.append(tau)

            cp_list = pruned + [t]

    # Backtrack
    change_points = []
    curr = n
    while curr > 0:
        prev = R[curr]
        if prev > 0:
            change_points.append(prev)
        curr = prev

    return sorted(change_points)


def _segment_cost_lm(
    segment_data: np.ndarray,
    p_response: int,
    variance_est: Union[float, np.ndarray],
) -> float:
    """Compute cost for a linear regression segment.

    Uses Gaussian negative log-likelihood:
        cost = 0.5 * RSS / variance

    For multivariate:
        cost = 0.5 * n * log(det(Sigma)) + 0.5 * trace(Sigma^{-1} * S)

    where S is the residual covariance matrix.

    Parameters:
        segment_data: Segment data (n_seg, p+d)
        p_response: Number of response variables
        variance_est: Variance estimate

    Returns:
        Cost value
    """
    n_seg = segment_data.shape[0]

    # Extract response and predictors
    y = segment_data[:, :p_response]
    X = segment_data[:, p_response:]

    try:
        # Fit linear regression
        if p_response == 1:
            # Univariate response
            y = y.ravel()

            # OLS: theta = (X'X)^{-1} X'y
            theta = np.linalg.solve(X.T @ X, X.T @ y)
            y_pred = X @ theta
            residuals = y - y_pred

            # Match R: use deviance ~ RSS/phi with phi=1; adjust beta by sigma upstream
            rss = np.sum(residuals ** 2)
            cost = 0.5 * rss

        else:
            # Multivariate response
            # OLS: Theta = (X'X)^{-1} X'Y
            Theta = np.linalg.solve(X.T @ X, X.T @ y)
            Y_pred = X @ Theta
            residuals = y - Y_pred

            # Residual covariance
            S = residuals.T @ residuals / n_seg

            # Multivariate Gaussian negative log-likelihood
            # cost = 0.5 * n * log(det(Sigma)) + 0.5 * trace(Sigma^{-1} * S * n)
            sign, logdet = np.linalg.slogdet(variance_est)
            if sign <= 0:
                return np.inf

            cost = 0.5 * n_seg * logdet + 0.5 * n_seg * np.trace(np.linalg.solve(variance_est, S))

        return cost

    except (np.linalg.LinAlgError, ValueError):
        # Singular matrix - return infinity
        return np.inf


def _postprocess_lm(change_points: List[int], n: int, trim: float) -> List[int]:
    """Post-process change points (boundary removal and merging).

    Matches R's post-processing.

    Parameters:
        change_points: Raw change points
        n: Total number of observations
        trim: Trim proportion

    Returns:
        Post-processed change points
    """
    if not change_points:
        return []

    cp = list(change_points)

    # Remove boundary CPs
    trim_threshold = trim * n
    cp = [c for c in cp if trim_threshold <= c <= (1 - trim) * n]

    if not cp:
        return []

    # Merge close CPs
    cp = sorted(set([0] + cp))
    diffs = np.diff(cp)
    close_indices = np.where(diffs < trim_threshold)[0]

    if len(close_indices) > 0:
        merged = []
        skip_next = set()

        for i in range(len(cp)):
            if i in skip_next:
                continue
            if i in close_indices:
                merged_cp = int(np.floor((cp[i] + cp[i+1]) / 2))
                merged.append(merged_cp)
                skip_next.add(i + 1)
            elif i - 1 not in close_indices:
                merged.append(cp[i])

        cp = merged

    # Remove 0
    cp = [c for c in cp if c > 0]

    return sorted(cp)


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Find nearest positive definite matrix.

    Matches R's Matrix::nearPD function (simplified version).

    Parameters:
        A: Input matrix

    Returns:
        Nearest positive definite matrix
    """
    # Symmetrize
    B = (A + A.T) / 2

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Force positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-10)

    # Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
