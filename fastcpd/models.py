"""High-performance model implementations using scikit-learn.

This module provides optimized GLM and LASSO implementations using scikit-learn's
highly optimized Cython/C code, which is faster and more reliable than porting R code.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelResult:
    """Result from a model fit."""
    coefficients: np.ndarray
    residuals: np.ndarray
    deviance: float
    converged: bool = True


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    warm_start_coef: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-4
) -> ModelResult:
    """Fit logistic regression using scikit-learn.

    Uses scikit-learn's optimized LogisticRegression which is implemented
    in Cython and uses liblinear/lbfgs - much faster than R's glm.fit.

    Parameters:
        X: Design matrix (n_samples, n_features)
        y: Binary response (n_samples,)
        warm_start_coef: Initial coefficients for warm start
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        ModelResult with coefficients, residuals, and deviance
    """
    from sklearn.linear_model import LogisticRegression

    # Fit model
    model = LogisticRegression(
        penalty=None,  # No regularization (equivalent to glm)
        fit_intercept=False,  # X should already include intercept
        max_iter=max_iter,
        tol=tol,
        solver='lbfgs',  # Fast Newton-like method
        warm_start=warm_start_coef is not None
    )

    if warm_start_coef is not None and hasattr(model, 'coef_'):
        model.coef_ = warm_start_coef.reshape(1, -1)

    model.fit(X, y)

    # Get predictions
    y_pred = model.predict_proba(X)[:, 1]

    # Compute deviance (binomial deviance)
    # D = -2 * sum(y*log(p) + (1-y)*log(1-p))
    # NLL = D / 2 (to match R's fastglm implementation)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    deviance = -2 * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    nll = deviance / 2  # Negative log-likelihood (matches R)

    # Compute residuals
    residuals = y - y_pred

    return ModelResult(
        coefficients=model.coef_.flatten(),
        residuals=residuals,
        deviance=nll,  # Use NLL not full deviance
        converged=model.n_iter_ < max_iter
    )


def fit_poisson_regression(
    X: np.ndarray,
    y: np.ndarray,
    warm_start_coef: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-4
) -> ModelResult:
    """Fit Poisson regression using scikit-learn.

    Uses scikit-learn's PoissonRegressor with optimized LBFGS solver.

    Parameters:
        X: Design matrix (n_samples, n_features)
        y: Count response (n_samples,)
        warm_start_coef: Initial coefficients
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        ModelResult with coefficients, residuals, and deviance
    """
    from sklearn.linear_model import PoissonRegressor

    model = PoissonRegressor(
        alpha=0,  # No regularization
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start_coef is not None
    )

    if warm_start_coef is not None and hasattr(model, 'coef_'):
        model.coef_ = warm_start_coef

    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)

    # Compute Poisson deviance
    # D = 2 * sum(y*log(y/mu) - (y-mu)) where mu = y_pred
    # NLL = D / 2 (to match R's fastglm implementation)
    epsilon = 1e-15
    y_pred = np.maximum(y_pred, epsilon)
    # Avoid log(0) by only computing for y > 0
    y_safe = np.where(y > 0, y, 1)  # Replace 0 with 1 to avoid log(0)
    deviance = 2 * np.sum(
        np.where(y > 0, y * np.log(y_safe / y_pred), 0) - (y - y_pred)
    )
    nll = deviance / 2  # Negative log-likelihood (matches R)

    # Compute residuals
    residuals = y - y_pred

    return ModelResult(
        coefficients=model.coef_,
        residuals=residuals,
        deviance=nll,  # Use NLL not full deviance
        converged=model.n_iter_ < max_iter
    )


def fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    warm_start_coef: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> ModelResult:
    """Fit LASSO regression using scikit-learn.

    Uses scikit-learn's highly optimized coordinate descent algorithm,
    which is faster than R's glmnet in many cases.

    Parameters:
        X: Design matrix (n_samples, n_features)
        y: Response (n_samples,)
        alpha: L1 regularization strength (lambda in R terminology)
        warm_start_coef: Initial coefficients
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        ModelResult with coefficients, residuals, and loss
    """
    from sklearn.linear_model import Lasso

    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start_coef is not None,
        selection='random',  # Faster convergence
        positive=False
    )

    if warm_start_coef is not None and hasattr(model, 'coef_'):
        model.coef_ = warm_start_coef

    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)

    # Compute residuals and loss
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    penalty = alpha * np.sum(np.abs(model.coef_))
    loss = mse + penalty  # Total objective

    return ModelResult(
        coefficients=model.coef_,
        residuals=residuals,
        deviance=loss,
        converged=model.n_iter_ < max_iter
    )


def fit_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> ModelResult:
    """Fit LASSO with cross-validated alpha selection.

    Uses LassoCV for automatic regularization parameter tuning.

    Parameters:
        X: Design matrix (n_samples, n_features)
        y: Response (n_samples,)
        cv: Number of cross-validation folds
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        ModelResult with coefficients, residuals, and loss
    """
    from sklearn.linear_model import LassoCV

    model = LassoCV(
        cv=cv,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        selection='random',
        n_jobs=-1  # Use all CPUs
    )

    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)

    # Compute residuals and loss
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    penalty = model.alpha_ * np.sum(np.abs(model.coef_))
    loss = mse + penalty

    return ModelResult(
        coefficients=model.coef_,
        residuals=residuals,
        deviance=loss,
        converged=True
    )


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray
) -> ModelResult:
    """Fit ordinary least squares using numpy (fastest for OLS).

    This is faster than scikit-learn for simple OLS since it's just
    a matrix solve operation.

    Parameters:
        X: Design matrix (n_samples, n_features)
        y: Response (n_samples,)

    Returns:
        ModelResult with coefficients, residuals, and RSS
    """
    # Use numpy's least squares (uses LAPACK, very fast)
    coef, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Compute residuals
    y_pred = X @ coef
    residuals = y - y_pred

    # RSS (residual sum of squares)
    rss = np.sum(residuals ** 2)

    return ModelResult(
        coefficients=coef,
        residuals=residuals,
        deviance=rss,
        converged=True
    )


# Try to use Numba-accelerated versions if available
try:
    from fastcpd.glm_numba import fit_binomial_fast, fit_poisson_fast, HAS_NUMBA
    if HAS_NUMBA:
        # Use Numba JIT versions (5-10x faster)
        GLM_FITTERS = {
            'binomial': fit_binomial_fast,
            'poisson': fit_poisson_fast,
            'lm': fit_linear_regression,
        }
        print("⚡ Using Numba-accelerated GLM fitters (5-10x faster)")
    else:
        # Fall back to sklearn
        GLM_FITTERS = {
            'binomial': fit_logistic_regression,
            'poisson': fit_poisson_regression,
            'lm': fit_linear_regression,
        }
except ImportError:
    # Fall back to sklearn if glm_numba not available
    GLM_FITTERS = {
        'binomial': fit_logistic_regression,
        'poisson': fit_poisson_regression,
        'lm': fit_linear_regression,
    }

LASSO_FITTER = fit_lasso
LASSO_CV_FITTER = fit_lasso_cv
