import pandas as pd
import numpy as np
from patsy import dmatrices

# Assume fastcpd_impl and helper functions are exposed via pybind11
from fastcpd.interface import fastcpd_impl
import fastcpd.variance_estimation
from math import log, log2


def pelt(
    formula: str = 'y ~ . - 1',
    data: np.ndarray = None,
    beta: object = 'MBIC',
    cost_adjustment: str = 'MBIC',
    family: str = None,
    cost=None,
    cost_gradient=None,
    cost_hessian=None,
    line_search=(1,),
    lower=None,
    upper=None,
    pruning_coef: float = None,
    segment_count: int = 10,
    trim: float = 0.05,
    momentum_coef: float = 0,
    multiple_epochs=lambda x: 0,
    epsilon: float = 1e-10,
    order=(0, 0, 0),
    p: int = None,
    variance_estimation=None,
    cp_only: bool = False,
    vanilla_percentage: float = 0,
    warm_start: bool = False,
    **kwargs
):
    """
    Find change points efficiently.

    Parameters:
    -----------
    formula : str
        A formula string specifying the model to be fitted. The optional
        response variable should be on the LHS, covariates on the RHS.
        Intercept should be removed by appending '- 1'. By default,
        an intercept column is added as in R's lm().
    data : pandas.DataFrame
        DataFrame of shape (T, d) containing the data to be segmented.
        Each row is a data point z_t in R^d.
    beta : {'BIC', 'MBIC', 'MDL'} or float
        Penalty criterion for the number of change points. String uses
        built-in criteria; numeric uses provided value.
    cost_adjustment : {'BIC', 'MBIC', 'MDL'} or None
        Cost adjustment criterion modifying the cost function.
    family : str or None
        Family of change point model. One of:
        'mean', 'variance', 'meanvariance', 'lm', 'binomial', 'poisson',
        'lasso', 'ar', 'arma', 'arima', 'garch', 'var', 'custom'.
        None is treated as 'custom'.
    cost : callable or None
        Custom cost function: cost(data) or cost(data, theta).
    cost_gradient : callable or None
        Gradient of custom cost: cost_gradient(data, theta).
    cost_hessian : callable or None
        Hessian of custom cost: cost_hessian(data, theta).
    line_search : tuple of floats
        Values for line search step sizes.
    lower : array-like or None
        Lower bound for parameters after each update.
    upper : array-like or None
        Upper bound for parameters after each update.
    pruning_coef : float
        Pruning coefficient for PELT algorithm.
    segment_count : int
        Initial guess for number of segments.
    trim : float
        Trimming proportion for boundary change points.
    momentum_coef : float
        Momentum coefficient for parameter updates.
    multiple_epochs : callable
        Function(segment_length) -> int additional epochs.
    epsilon : float
        Epsilon for numerical stability.
    order : tuple of int
        Order for AR, VAR, ARIMA, GARCH models.
    p : int or None
        Number of covariates. If None, inferred from data.
    variance_estimation : array-like or None
        Pre-specified variance/covariance matrix.
    cp_only : bool
        If True, only change points are returned.
    vanilla_percentage : float
        Interpolation parameter between PELT and SeGD.
    warm_start : bool
        Use previous segment's params as initial values.
    **kwargs :
        Additional model-specific parameters.

    Returns:
    --------
    result : object
        A FastCPDResult object containing change points, costs,
        residuals, and parameter estimates.
    """
    family = family.lower() if family is not None else 'custom'
    assert family in ('mean', 'variance', 'meanvariance')
    assert cost_adjustment in ('BIC', 'MBIC', 'MDL')

    if variance_estimation is not None:
        variance_estimation = np.asarray(variance_estimation)
    elif family == 'mean':
        variance_estimation = fastcpd.variance_estimation.mean(data)

    if pruning_coef is None:
        pruning_coef = 0.0
    if cost_adjustment == "MBIC":
        pruning_coef += data.shape[1] * log(2)
    elif cost_adjustment == "MDL":
        pruning_coef += data.shape[1] * log2(2)

    result = fastcpd_impl(
        data,
        variance_estimation
    )
    return result


def mean(data, **kwargs):
    """
    Find change points efficiently in mean change models.

    Parameters:
    -----------
    data : array-like or pandas.DataFrame
        Univariate data for mean change detection.
    **kwargs :
        Additional arguments passed to pelt().

    Returns:
    --------
    result
    """
    return pelt(data=data, family='mean', **kwargs)


def variance(data, **kwargs):
    """
    Find change points efficiently in variance change models.
    """
    return pelt(data=data, family='variance', **kwargs)


def meanvariance(data, **kwargs):
    """
    Find change points efficiently in mean and/or variance change models.
    """
    return pelt(data=data, family='meanvariance', **kwargs)
