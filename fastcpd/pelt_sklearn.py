"""Pure Python PELT implementation using scikit-learn models.

This module implements the PELT (Pruned Exact Linear Time) algorithm for
GLM and LASSO models using scikit-learn's highly optimized implementations.
"""

import numpy as np
from typing import Union, List
from fastcpd.models import GLM_FITTERS, LASSO_FITTER, LASSO_CV_FITTER, ModelResult
from fastcpd.pelt_sen import _postprocess_changepoints


def _fastcpd_sklearn(
    data: np.ndarray,
    beta: Union[str, float],
    cost_adjustment: str,
    family: str,
    segment_count: int,
    trim: float,
    warm_start: bool,
    lasso_alpha: float = 1.0,
    lasso_cv: bool = False,
):
    """Run PELT algorithm using scikit-learn models.

    This is a pure Python implementation that uses scikit-learn's optimized
    GLM and LASSO implementations, which are often faster than the C++ versions
    for these specific models.

    Parameters:
        data: Input data (n, d+1) with first column as response
        beta: Penalty value or criterion
        cost_adjustment: Cost adjustment type
        family: Model family (binomial, poisson, lasso)
        segment_count: Initial segment count
        trim: Boundary trim proportion
        warm_start: Whether to use warm start
        lasso_alpha: LASSO regularization parameter
        lasso_cv: Whether to use CV for LASSO

    Returns:
        FastcpdResult object
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_obs, n_cols = data.shape

    # Calculate beta if string
    if isinstance(beta, str):
        p_param = n_cols - 1  # Number of predictors
        if beta == "MBIC":
            beta_val = (p_param + 2) * np.log(n_obs) / 2
        elif beta == "BIC":
            beta_val = p_param * np.log(n_obs) / 2
        elif beta == "MDL":
            beta_val = (p_param / 2) * np.log(n_obs)
        else:
            raise ValueError(f"Unknown beta criterion: {beta}")
    else:
        beta_val = beta

    # Run PELT algorithm
    change_points_raw = _pelt_sklearn(
        data, family, beta_val, warm_start, lasso_alpha, lasso_cv
    )

    # Post-process: remove boundary CPs and merge close CPs (matching R)
    change_points = _postprocess_changepoints(change_points_raw, n_obs, trim)

    # Estimate parameters for each segment
    segments = _get_segments(change_points, n_obs)
    thetas = []
    residuals_list = []

    for start, end in segments:
        segment_data = data[start:end+1, :]
        if segment_data.shape[0] > 1:
            X = segment_data[:, 1:]
            y = segment_data[:, 0]

            # Add intercept column (sklearn needs it, fit_intercept=False)
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

            # Fit model
            if family == 'lasso':
                if lasso_cv:
                    result = LASSO_CV_FITTER(X_with_intercept, y)
                else:
                    result = LASSO_FITTER(X_with_intercept, y, alpha=lasso_alpha)
            else:
                fitter = GLM_FITTERS.get(family)
                if fitter is None:
                    raise ValueError(f"Unknown family: {family}")
                result = fitter(X_with_intercept, y)

            thetas.append(result.coefficients)
            residuals_list.append(result.residuals)
        else:
            thetas.append(np.zeros(n_cols - 1))
            residuals_list.append(np.zeros(1))

    # Combine results
    # Stack thetas as matrix (may have different lengths if some segments are empty)
    max_len = max(len(t) for t in thetas) if thetas else 0
    thetas_padded = [np.pad(t, (0, max_len - len(t))) for t in thetas]
    thetas_mat = np.array(thetas_padded) if thetas_padded else np.array([])

    # Stack residuals
    all_residuals = np.concatenate(residuals_list) if residuals_list else np.array([])

    # Calculate costs (approximate)
    costs = np.array([_segment_cost(data[s:e+1, :], family, lasso_alpha)
                      for s, e in segments])

    # Return dictionary instead of FastcpdResult to avoid circular import
    return {
        'raw_cp_set': np.array(change_points_raw),  # Before post-processing
        'cp_set': np.array(change_points),           # After post-processing
        'cost_values': costs,
        'residuals': all_residuals.reshape(-1, 1),
        'thetas': thetas_mat,
        'data': data,
        'family': family,
    }


def _pelt_sklearn(
    data: np.ndarray,
    family: str,
    beta: float,
    warm_start: bool,
    lasso_alpha: float,
    lasso_cv: bool,
) -> List[int]:
    """PELT algorithm using scikit-learn models.

    Parameters:
        data: Input data (n, d+1)
        family: Model family
        beta: Penalty value
        warm_start: Whether to use warm start
        lasso_alpha: LASSO alpha
        lasso_cv: LASSO CV flag

    Returns:
        List of change point indices
    """
    n = data.shape[0]
    X = data[:, 1:]
    y = data[:, 0]

    # Initialize
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    R = np.zeros(n + 1, dtype=int)
    cp_list = [0]

    # PELT main loop
    for t in range(1, n + 1):
        candidates = []
        cost_cache = {}  # Cache costs to avoid recomputation in pruning

        for tau in cp_list:
            if tau < t:
                # Cost for segment [tau, t)
                X_seg = X[tau:t, :]
                y_seg = y[tau:t]
                segment_length = X_seg.shape[0]

                # Match R behavior: only fit if segment is long enough
                # R requirement: segment_length >= parameters_count for GLM
                # parameters_count = X_seg.shape[1] + 1 (intercept)
                min_length = X_seg.shape[1] + 1 if family != 'lasso' else 3

                if segment_length >= min_length:
                    try:
                        # Add intercept column (sklearn needs it, fit_intercept=False)
                        X_seg_with_intercept = np.column_stack([np.ones(segment_length), X_seg])

                        # Fit model
                        if family == 'lasso':
                            if lasso_cv:
                                result = LASSO_CV_FITTER(X_seg_with_intercept, y_seg, max_iter=500)
                            else:
                                result = LASSO_FITTER(X_seg_with_intercept, y_seg, alpha=lasso_alpha, max_iter=500)
                        else:
                            fitter = GLM_FITTERS.get(family)
                            if fitter is None:
                                continue
                            result = fitter(X_seg_with_intercept, y_seg, max_iter=100)

                        cost = result.deviance
                    except:
                        cost = np.inf
                else:
                    # Match R: return cost = 0 for segments too small to fit
                    cost = 0

                cost_cache[tau] = cost  # Cache for pruning
                total_cost = F[tau] + cost + beta
                candidates.append((total_cost, tau))

        if candidates:
            F[t], R[t] = min(candidates)

            # Pruning: only keep candidates where F[tau] + cost <= F[t]
            # Use cached costs to avoid recomputation
            pruned = []
            for tau in cp_list:
                if tau < t:
                    # Use cached cost from candidates loop
                    cost = cost_cache[tau]

                    # FIXED: Remove beta from condition to match R
                    # R checks: candidates[i] <= F[t] + beta
                    # where candidates[i] = F[tau] + cost + beta
                    # So: F[tau] + cost + beta <= F[t] + beta
                    # Simplifies to: F[tau] + cost <= F[t]
                    if F[tau] + cost <= F[t]:
                        pruned.append(tau)
                else:
                    pruned.append(tau)

            cp_list = pruned + [t]

    # Backtrack to find change points
    change_points = []
    curr = n
    while curr > 0:
        prev = R[curr]
        if prev > 0:
            change_points.append(prev)
        curr = prev

    return sorted(change_points)


def _get_segments(change_points: List[int], n: int) -> List[tuple]:
    """Convert change points to segment boundaries.

    Parameters:
        change_points: List of change point indices
        n: Total number of observations

    Returns:
        List of (start, end) tuples for each segment
    """
    if not change_points:
        return [(0, n - 1)]

    segments = []
    boundaries = [0] + sorted(change_points) + [n]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        if end >= start:
            segments.append((start, end))

    return segments


def _segment_cost(
    segment_data: np.ndarray,
    family: str,
    lasso_alpha: float = 1.0
) -> float:
    """Calculate cost for a single segment.

    Parameters:
        segment_data: Data for this segment (n, d+1)
        family: Model family
        lasso_alpha: LASSO alpha

    Returns:
        Cost value
    """
    X = segment_data[:, 1:]
    y = segment_data[:, 0]
    segment_length = X.shape[0]

    # Match R behavior: only fit if segment is long enough
    min_length = X.shape[1] + 1 if family != 'lasso' else 3

    if segment_length < min_length:
        return 0.0

    # Add intercept column (sklearn needs it, fit_intercept=False)
    X_with_intercept = np.column_stack([np.ones(segment_length), X])

    try:
        if family == 'lasso':
            result = LASSO_FITTER(X_with_intercept, y, alpha=lasso_alpha, max_iter=500)
        else:
            fitter = GLM_FITTERS.get(family)
            if fitter is None:
                return 0.0
            result = fitter(X_with_intercept, y, max_iter=100)

        return result.deviance
    except:
        return 0.0
