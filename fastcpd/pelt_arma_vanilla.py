"""Vanilla PELT for ARMA models - Pure Python implementation.

This uses standard PELT with statsmodels ARIMA fitting for each segment,
avoiding the numerical issues of SeGD with Hessian accumulation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


def _fit_arma_segment(data: np.ndarray, p: int, q: int) -> Tuple[float, np.ndarray]:
    """Fit ARMA model to a segment using statsmodels.

    Returns:
        (nll, theta) where nll is negative log-likelihood and theta are parameters
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Need enough data points
        if len(data) < max(p, q) + 3:
            # Return large cost for too-short segments
            return 1e10, np.zeros(p + q + 1)

        # Fit ARIMA(p,0,q) model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(data, order=(p, 0, q), trend='n')
            result = model.fit(method='statespace', low_memory=True)

        # Extract parameters
        theta = np.zeros(p + q + 1)
        if p > 0:
            theta[:p] = result.arparams
        if q > 0:
            theta[p:p+q] = result.maparams
        theta[p + q] = result.params[-1]  # sigma^2

        # Negative log-likelihood
        nll = -result.llf

        return nll, theta

    except Exception as e:
        # If fitting fails, return large cost
        return 1e10, np.zeros(p + q + 1)


def _pelt_arma_vanilla(
    data: np.ndarray,
    p: int,
    q: int,
    beta: float,
    trim: float = 0.025
) -> List[int]:
    """PELT for ARMA with pruning - faster and stable.

    Parameters:
        data: Time series data (1D array)
        p: AR order
        q: MA order
        beta: Penalty parameter
        trim: Minimum segment length as fraction of n

    Returns:
        List of change point indices
    """
    n = len(data)
    min_segment_length = max(int(trim * n), max(p, q) + 3)

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
                nll, _ = _fit_arma_segment(segment_data, p, q)
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


def _fastcpd_arma_vanilla(
    data: np.ndarray,
    order: List[int],
    beta: float,
    trim: float = 0.025,
    **kwargs
) -> Dict:
    """ARMA change point detection using vanilla PELT.

    This is a pure Python implementation that uses statsmodels for fitting,
    avoiding the numerical issues of SeGD.

    Parameters:
        data: Time series data
        order: [p, q] for ARMA(p,q)
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
    change_points_raw = _pelt_arma_vanilla(data, p, q, beta, trim)

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
        if len(segment_data) >= max(p, q) + 3:
            _, theta = _fit_arma_segment(segment_data, p, q)
            thetas.append(theta)

            # Compute residuals
            try:
                from statsmodels.tsa.arima.model import ARIMA
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(segment_data, order=(p, 0, q), trend='n')
                    result = model.fit(method='statespace', low_memory=True)
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
        'family': 'arma',
        'order': order,
    }
