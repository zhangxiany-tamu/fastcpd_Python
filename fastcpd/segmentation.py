"""Convenience functions for common change point detection scenarios."""

from typing import Optional, Union
import numpy as np
from fastcpd.fastcpd import fastcpd, FastcpdResult


def mean(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect mean changes in univariate or multivariate data.

    Parameters:
        data: Input data of shape (n,) for univariate or (n, d) for multivariate
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import mean
        >>>
        >>> # Univariate mean change
        >>> data = np.concatenate([np.random.normal(0, 1, 300),
        ...                        np.random.normal(5, 1, 400)])
        >>> result = mean(data)
        >>> print(result.cp_set)

        >>> # Multivariate mean change
        >>> data = np.concatenate([
        ...     np.random.multivariate_normal([0, 0], np.eye(2), 300),
        ...     np.random.multivariate_normal([5, 5], np.eye(2), 400)
        ... ])
        >>> result = mean(data)
    """
    return fastcpd(data, family="mean", beta=beta, **kwargs)


def variance(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect variance changes in data.

    Parameters:
        data: Input data
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results
    """
    return fastcpd(data, family="variance", beta=beta, **kwargs)


def meanvariance(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect mean and/or variance changes in data.

    Parameters:
        data: Input data
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results
    """
    return fastcpd(data, family="meanvariance", beta=beta, **kwargs)


def ar(
    data: Union[np.ndarray, list],
    p: int = 1,
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in AR(p) model parameters.

    Parameters:
        data: Input time series data
        p: Order of AR model
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import ar
        >>>
        >>> # Simulate AR(2) data with change
        >>> np.random.seed(42)
        >>> n = 500
        >>> data1 = np.zeros(n // 2)
        >>> for i in range(2, n // 2):
        ...     data1[i] = 0.6 * data1[i-1] - 0.3 * data1[i-2] + np.random.normal()
        >>> data2 = np.zeros(n // 2)
        >>> for i in range(2, n // 2):
        ...     data2[i] = -0.4 * data2[i-1] + 0.5 * data2[i-2] + np.random.normal()
        >>> data = np.concatenate([data1, data2])
        >>> result = ar(data, p=2)
        >>> print(result.cp_set)
    """
    return fastcpd(data, family="ar", p=p, beta=beta, order=[p], **kwargs)


def arma(
    data: Union[np.ndarray, list],
    p: int = 1,
    q: int = 1,
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in ARMA(p,q) model parameters.

    Parameters:
        data: Input time series data
        p: AR order
        q: MA order
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results
    """
    return fastcpd(data, family="arma", beta=beta, order=[p, q], **kwargs)


def garch(
    data: Union[np.ndarray, list],
    p: int = 1,
    q: int = 1,
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in GARCH(p,q) model parameters.

    Parameters:
        data: Input time series data
        p: GARCH order
        q: ARCH order
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results
    """
    return fastcpd(data, family="garch", beta=beta, order=[p, q], **kwargs)


def var(
    data: Union[np.ndarray, list],
    p: int = 1,
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in VAR(p) model parameters.

    Parameters:
        data: Input multivariate time series data of shape (n, d)
        p: Order of VAR model
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results
    """
    return fastcpd(data, family="var", p=p, beta=beta, order=[p], **kwargs)


def linear_regression(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in linear regression parameters.

    The first column of data is treated as the response variable,
    and the remaining columns are treated as predictors.

    Parameters:
        data: Input data of shape (n, d+1) where first column is response
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import linear_regression
        >>>
        >>> # Simulate linear regression with change
        >>> n = 500
        >>> X = np.random.randn(n, 2)
        >>> y1 = 2 * X[:n//2, 0] + 3 * X[:n//2, 1] + np.random.randn(n//2)
        >>> y2 = -1 * X[n//2:, 0] + 5 * X[n//2:, 1] + np.random.randn(n//2)
        >>> y = np.concatenate([y1, y2])
        >>> data = np.column_stack([y, X])
        >>> result = linear_regression(data)
        >>> print(result.cp_set)
    """
    return fastcpd(data, family="lm", beta=beta, **kwargs)


def logistic_regression(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in logistic regression parameters.

    Uses scikit-learn's highly optimized LogisticRegression (Cython/C implementation).
    The first column of data is the binary response (0/1), remaining columns are predictors.

    Parameters:
        data: Input data of shape (n, d+1) where first column is binary response
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import logistic_regression
        >>>
        >>> # Simulate logistic regression with change
        >>> n = 500
        >>> X = np.random.randn(n, 2)
        >>> # First segment: strong positive effect
        >>> prob1 = 1 / (1 + np.exp(-(2*X[:n//2, 0] + 3*X[:n//2, 1])))
        >>> y1 = (np.random.rand(n//2) < prob1).astype(float)
        >>> # Second segment: negative effect
        >>> prob2 = 1 / (1 + np.exp(-(-1*X[n//2:, 0] + 2*X[n//2:, 1])))
        >>> y2 = (np.random.rand(n//2) < prob2).astype(float)
        >>> y = np.concatenate([y1, y2])
        >>> data = np.column_stack([y, X])
        >>> result = logistic_regression(data)
        >>> print(result.cp_set)
    """
    return fastcpd(data, family="binomial", beta=beta, **kwargs)


def poisson_regression(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    **kwargs
) -> FastcpdResult:
    """Detect changes in Poisson regression parameters.

    Uses scikit-learn's PoissonRegressor with optimized LBFGS solver.
    The first column of data is the count response, remaining columns are predictors.

    Parameters:
        data: Input data of shape (n, d+1) where first column is count response
        beta: Penalty for number of change points
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import poisson_regression
        >>>
        >>> # Simulate Poisson regression with change
        >>> n = 500
        >>> X = np.random.randn(n, 2)
        >>> # First segment
        >>> lambda1 = np.exp(0.5*X[:n//2, 0] + 0.8*X[:n//2, 1])
        >>> y1 = np.random.poisson(lambda1)
        >>> # Second segment
        >>> lambda2 = np.exp(-0.3*X[n//2:, 0] + 1.2*X[n//2:, 1])
        >>> y2 = np.random.poisson(lambda2)
        >>> y = np.concatenate([y1, y2])
        >>> data = np.column_stack([y, X])
        >>> result = poisson_regression(data)
        >>> print(result.cp_set)
    """
    return fastcpd(data, family="poisson", beta=beta, **kwargs)


def lasso(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    alpha: float = 1.0,
    cv: bool = False,
    **kwargs
) -> FastcpdResult:
    """Detect changes in LASSO regression parameters.

    Uses scikit-learn's highly optimized coordinate descent algorithm.
    The first column of data is the response, remaining columns are predictors.

    Parameters:
        data: Input data of shape (n, d+1) where first column is response
        beta: Penalty for number of change points
        alpha: L1 regularization strength (lambda). Ignored if cv=True.
        cv: If True, use cross-validation to select alpha (slower but automatic)
        **kwargs: Additional arguments passed to fastcpd()

    Returns:
        FastcpdResult: Change point detection results

    Examples:
        >>> import numpy as np
        >>> from fastcpd.segmentation import lasso
        >>>
        >>> # Simulate sparse regression with change
        >>> n = 500
        >>> p = 20
        >>> X = np.random.randn(n, p)
        >>> # First segment: only first 3 features matter
        >>> y1 = 2*X[:n//2, 0] + 3*X[:n//2, 1] - 1.5*X[:n//2, 2] + np.random.randn(n//2)
        >>> # Second segment: different sparse coefficients
        >>> y2 = -1*X[n//2:, 5] + 2*X[n//2:, 8] + np.random.randn(n//2)
        >>> y = np.concatenate([y1, y2])
        >>> data = np.column_stack([y, X])
        >>> result = lasso(data, alpha=0.1)
        >>> print(result.cp_set)
    """
    kwargs['lasso_alpha'] = alpha
    kwargs['lasso_cv'] = cv
    return fastcpd(data, family="lasso", beta=beta, **kwargs)
