"""Vanilla PELT for GARCH models - Pure Python implementation.

This uses standard PELT with arch package GARCH fitting for each segment,
similar to the ARMA vanilla PELT approach.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def _fit_garch_segment(data: np.ndarray, p: int, q: int) -> Tuple[float, np.ndarray]:
    """Fit GARCH model to a segment using arch package.

    Returns:
        (nll, theta) where nll is negative log-likelihood and theta are parameters
    """
    try:
        from arch import arch_model

        # Need enough data points
        if len(data) < max(p, q) + 5:
            # Return large cost for too-short segments
            return 1e10, np.zeros(p + q + 1)

        # Fit GARCH(p, q) model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # arch_model expects returns/residuals, not raw prices
            # For change point detection, we work with the data as-is
            model = arch_model(data, vol='GARCH', p=p, q=q, mean='Zero', rescale=False)
            result = model.fit(disp='off', show_warning=False)

        # Extract parameters: [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
        theta = np.zeros(p + q + 1)
        params = result.params.values

        theta[0] = params[0]  # omega (constant)
        if p > 0:
            theta[1:p+1] = params[1:p+1]  # alpha coefficients
        if q > 0:
            theta[p+1:p+q+1] = params[p+1:p+q+1]  # beta coefficients

        # Negative log-likelihood
        nll = -result.loglikelihood

        return nll, theta

    except Exception as e:
        # If fitting fails, return large cost
        return 1e10, np.zeros(p + q + 1)


def _pelt_garch_vanilla(
    data: np.ndarray,
    p: int,
    q: int,
    beta: float,
    trim: float = 0.025
) -> List[int]:
    """PELT for GARCH with pruning - pure Python.

    Parameters:
        data: Time series data (1D array)
        p: GARCH order (ARCH terms)
        q: GARCH order (GARCH terms)
        beta: Penalty parameter
        trim: Minimum segment length as fraction of n

    Returns:
        List of change point indices
    """
    n = len(data)
    min_segment_length = max(int(trim * n), max(p, q) + 5)

    # Dynamic programming arrays
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    R = np.zeros(n + 1, dtype=int)

    # Cost cache
    cost_cache = {}

    # Active set of candidate change points (pruning)
    candidates_set = [0]

    # Main PELT loop
    for t in range(min_segment_length, n + 1):
        candidates = []
        costs_for_pruning = {}

        # Only try candidates in active set (PRUNING)
        for tau in candidates_set:
            # Check if this segment is long enough
            if t - tau < min_segment_length:
                continue

            # Compute cost for segment [tau, t-1]
            cache_key = (tau, t-1)
            if cache_key not in cost_cache:
                segment_data = data[tau:t]
                nll, _ = _fit_garch_segment(segment_data, p, q)
                cost_cache[cache_key] = nll

            cost = cost_cache[cache_key]
            costs_for_pruning[tau] = cost
            total_cost = F[tau] + cost + beta
            candidates.append((total_cost, tau))

        if candidates:
            # Choose minimum cost
            F[t], R[t] = min(candidates)

            # PRUNING: Keep only candidates where F[tau] + cost(tau,t) <= F[t]
            pruned_candidates = []
            for tau in candidates_set:
                if tau in costs_for_pruning:
                    if F[tau] + costs_for_pruning[tau] <= F[t]:
                        pruned_candidates.append(tau)
                else:
                    pruned_candidates.append(tau)

            # Add current time point as new candidate
            candidates_set = pruned_candidates + [t]

    # Backtrack to find change points
    change_points = []
    curr = n
    while curr > 0:
        prev = R[curr]
        if prev > 0:
            change_points.append(prev)
        curr = prev

    return sorted(change_points)


def _postprocess_changepoints(change_points: List[int], n: int, trim: float = 0.025) -> List[int]:
    """Post-process change points (remove boundary points, merge close ones)."""
    if not change_points:
        return []

    cp = list(change_points)
    trim_threshold = trim * n

    # Remove boundary change points
    cp = [c for c in cp if trim_threshold <= c <= (1 - trim) * n]

    if not cp:
        return []

    # Merge close change points
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
                # Merge with next
                merged_cp = int(np.floor((cp[i] + cp[i+1]) / 2))
                merged.append(merged_cp)
                skip_next.add(i + 1)
            elif i - 1 not in close_indices:
                merged.append(cp[i])

        cp = merged

    # Remove 0 if present
    cp = [c for c in cp if c > 0]

    return sorted(cp)


def _fastcpd_garch_vanilla(
    data: np.ndarray,
    order: List[int],
    beta: float,
    trim: float = 0.025,
    **kwargs
) -> Dict:
    """GARCH change point detection using vanilla PELT.

    This is a pure Python implementation that uses arch package for fitting,
    following the same approach as ARMA vanilla PELT.

    Parameters:
        data: Time series data (typically returns or residuals)
        order: [p, q] for GARCH(p,q)
        beta: Penalty parameter
        trim: Trimming parameter

    Returns:
        Dictionary with change points and related information
    """
    # Flatten if needed
    if data.ndim > 1:
        data = data.flatten()

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    p, q = order

    # Run vanilla PELT
    change_points_raw = _pelt_garch_vanilla(data, p, q, beta, trim)

    # Post-process
    change_points = _postprocess_changepoints(change_points_raw, n, trim)

    # Fit final segments to get parameters and residuals
    segments = []
    if len(change_points) == 0:
        segments = [(0, n-1)]
    else:
        boundaries = [0] + sorted(change_points) + [n]
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i+1] - 1))

    thetas = []
    residuals_list = []

    for start, end in segments:
        segment_data = data[start:end+1]
        if len(segment_data) >= max(p, q) + 5:
            _, theta = _fit_garch_segment(segment_data, p, q)
            thetas.append(theta)

            # Compute residuals
            try:
                from arch import arch_model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = arch_model(segment_data, vol='GARCH', p=p, q=q,
                                      mean='Zero', rescale=False)
                    result = model.fit(disp='off', show_warning=False)
                    residuals_list.append(result.resid)
            except:
                residuals_list.append(np.zeros(len(segment_data)))
        else:
            thetas.append(np.zeros(p + q + 1))
            residuals_list.append(np.zeros(len(segment_data)))

    # Combine results
    if thetas:
        max_len = max(len(t) for t in thetas)
        thetas_padded = [np.pad(t, (0, max_len - len(t))) for t in thetas]
        thetas_mat = np.array(thetas_padded)
    else:
        thetas_mat = np.array([])

    all_residuals = np.concatenate(residuals_list) if residuals_list else np.array([])

    return {
        'raw_cp_set': np.array(change_points_raw),
        'cp_set': np.array(change_points),
        'cost_values': np.array([]),
        'residuals': all_residuals.reshape(-1, 1),
        'thetas': thetas_mat,
        'data': data.reshape(-1, 1),
        'family': 'garch',
        'order': order,
    }
