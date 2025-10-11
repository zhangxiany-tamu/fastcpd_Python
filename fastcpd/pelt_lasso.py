"""PELT/SEN implementation for LASSO regression.

This module implements both pure PELT and SeGD (SEN) for LASSO,
matching the R implementation in SeGD-Lasso.R.
"""

import numpy as np
from typing import List, Tuple, Union
from sklearn.linear_model import LassoCV
from fastcpd.sen_glm import SENLASSO, SENState


def _fastcpd_lasso_sen(
    data: np.ndarray,
    beta: Union[str, float],
    vanilla_percentage: float,
    trim: float,
    segment_count: int = 10,
    epsilon: float = 1e-5,
) -> dict:
    """LASSO change point detection with SEN support.

    Matches R's CP() function in SeGD-Lasso.R.

    Parameters:
        data: Input data (n, p+1) with columns [X1, ..., Xp, y]
        beta: Penalty value or criterion
        vanilla_percentage: Fraction using PELT (0=pure SEN, 1=pure PELT)
        trim: Boundary trim proportion
        segment_count: Number of pre-segmentation segments (B in R)
        epsilon: Regularization for numerical stability

    Returns:
        Dictionary with change points and other results
    """
    data = np.asarray(data, dtype=np.float64)
    n, n_cols = data.shape
    p = n_cols - 1  # Number of features

    X = data[:, :p]
    y = data[:, p]

    # Calculate beta if string
    if isinstance(beta, str):
        if beta == "MBIC":
            beta_val = (p + 2) * np.log(n) / 2
        elif beta == "BIC":
            beta_val = p * np.log(n) / 2
        elif beta == "MDL":
            beta_val = (p / 2) * np.log(n)
        else:
            raise ValueError(f"Unknown beta criterion: {beta}")
    else:
        beta_val = float(beta)

    # Pre-segmentation to get initial values (matching R lines 127-140)
    index = np.repeat(np.arange(segment_count), n // segment_count)
    if len(index) < n:
        index = np.concatenate([index, np.full(n - len(index), segment_count - 1)])

    coef_init = np.zeros((segment_count, p))
    err_sd = np.zeros(segment_count)
    act_num = np.zeros(segment_count)

    for i in range(segment_count):
        seg_mask = (index == i)
        X_seg = X[seg_mask]
        y_seg = y[seg_mask]

        if X_seg.shape[0] > p:
            # Fit LASSO with CV
            lasso_cv = LassoCV(cv=5, max_iter=1000)
            lasso_cv.fit(X_seg, y_seg)
            coef_init[i] = lasso_cv.coef_

            # Compute residual standard deviation
            y_pred = X_seg @ coef_init[i]
            residuals = y_seg - y_pred
            err_sd[i] = np.sqrt(np.mean(residuals ** 2))

            # Count active coefficients
            act_num[i] = np.sum(np.abs(coef_init[i]) > 0)

    err_sd_mean = np.mean(err_sd)
    act_num_mean = np.mean(act_num)

    # CRITICAL: Adjust beta like R (line 140)
    beta_adjusted = (act_num_mean + 1) * beta_val

    print(f"LASSO pre-segmentation: err_sd_mean={err_sd_mean:.4f}, act_num_mean={act_num_mean:.1f}")
    print(f"Beta adjustment: {beta_val:.4f} -> {beta_adjusted:.4f} (multiplier={act_num_mean+1:.2f})")

    # Run PELT/SEN algorithm
    change_points_raw = _pelt_lasso(
        X, y, beta_adjusted, vanilla_percentage,
        coef_init, index, err_sd_mean, epsilon, segment_count, p
    )

    # Post-process (matching R lines 195-207)
    change_points = _postprocess_lasso(change_points_raw, n, trim)

    return {
        'raw_cp_set': np.array(change_points_raw),
        'cp_set': np.array(change_points),
        'cost_values': np.array([]),
        'residuals': np.array([]).reshape(-1, 1),
        'thetas': np.array([]),
        'data': data,
        'family': 'lasso',
    }


def _pelt_lasso(
    X: np.ndarray,
    y: np.ndarray,
    beta: float,
    vanilla_percentage: float,
    coef_init: np.ndarray,
    index: np.ndarray,
    err_sd_mean: float,
    epsilon: float,
    segment_count: int,
    p: int,
) -> List[int]:
    """PELT algorithm for LASSO with SEN support.

    Matches R's CP() main loop (lines 149-193).
    """
    n = X.shape[0]

    # Initialize
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    R = np.zeros(n + 1, dtype=int)
    cp_list = [0]

    # SEN state arrays (matching R: coef, cum_coef, cmatrix)
    sen_coefs: List[np.ndarray] = []
    sen_cum_coefs: List[np.ndarray] = []
    sen_cmatrices: List[np.ndarray] = []

    # Initialize first state (matching R lines 142-147)
    X1 = X[0]
    coef1 = coef_init[0]
    cum_coef1 = coef1.copy()
    cmatrix1 = np.outer(X1, X1) + epsilon * np.eye(p)

    sen_coefs.append(coef1)
    sen_cum_coefs.append(cum_coef1)
    sen_cmatrices.append(cmatrix1)

    vanilla_threshold = int(vanilla_percentage * n)
    sen = SENLASSO(p, epsilon)

    # Main loop (matching R lines 149-193)
    for t in range(1, n):  # R uses 2:n, we use 1:n (0-indexed)
        m = len(cp_list)
        cval = np.full(m, np.nan)

        # Update existing candidates (matching R lines 154-167)
        for i in range(m - 1):  # R: 1:(m-1)
            coef_c = sen_coefs[i]
            cum_coef_c = sen_cum_coefs[i]
            cmatrix_c = sen_cmatrices[i]

            k = cp_list[i]  # tau in our notation
            segment_length = t - k + 1

            # Lambda depends on segment length (R line 161)
            lambda_val = err_sd_mean * np.sqrt(2 * np.log(p) / segment_length)

            # Update with new observation (R: cost_lasso_update)
            X_t = X[t]
            y_t = y[t]

            # Manual update matching R lines 88-102
            mu = np.dot(X_t, coef_c)
            cmatrix_new = cmatrix_c + np.outer(X_t, X_t)
            lik_dev = -(y_t - mu) * X_t

            try:
                coef_new = coef_c - np.linalg.solve(cmatrix_new, lik_dev)
            except np.linalg.LinAlgError:
                coef_new = coef_c - lik_dev / np.trace(cmatrix_new)

            # Soft thresholding (R line 100)
            nc = np.linalg.norm(cmatrix_new, ord='fro')
            coef_new = sen.soft_threshold(coef_new, lambda_val / nc)
            cum_coef_new = cum_coef_c + coef_new

            # Store updated states
            sen_coefs[i] = coef_new
            sen_cum_coefs[i] = cum_coef_new
            sen_cmatrices[i] = cmatrix_new

            # Compute cost (R line 166)
            if segment_length >= 2:
                theta_avg = cum_coef_new / segment_length
                cval[i] = _neg_log_lik_lasso(
                    X[k:t+1], y[k:t+1], theta_avg, lambda_val
                )
            else:
                cval[i] = 0

        # New candidate at time t has cost 0 (R line 171)
        cval[m - 1] = 0

        # Find minimum (R lines 181-185)
        # obj and cval have same length m = len(cp_list)
        obj = cval + F[np.array(cp_list)] + beta
        min_idx = np.argmin(obj)
        min_val = obj[min_idx]
        F[t + 1] = min_val
        R[t + 1] = cp_list[min_idx]

        # Pruning (R line 186)
        ind2 = (cval + F[np.array(cp_list)]) <= min_val
        cp_list_pruned = [cp for i, cp in enumerate(cp_list) if ind2[i]]

        # Prune SEN states
        sen_coefs = [sen_coefs[i] for i in range(len(ind2)) if ind2[i]]
        sen_cum_coefs = [sen_cum_coefs[i] for i in range(len(ind2)) if ind2[i]]
        sen_cmatrices = [sen_cmatrices[i] for i in range(len(ind2)) if ind2[i]]

        # Add new candidate at time t (R line 191: set <- c(set[ind2], t))
        cp_list = cp_list_pruned + [t]

        # Add new candidate states (R lines 173-179)
        X_t = X[t]
        coef_new_cand = coef_init[index[min(t, n-1)]]
        cum_coef_new_cand = coef_new_cand.copy()
        cmatrix_new_cand = np.outer(X_t, X_t) + epsilon * np.eye(p)

        sen_coefs.append(coef_new_cand)
        sen_cum_coefs.append(cum_coef_new_cand)
        sen_cmatrices.append(cmatrix_new_cand)

    # Backtrack (R lines 197)
    cp = []
    curr = n
    while curr > 0:
        prev = R[curr]
        if prev > 0:
            cp.append(prev)
        curr = prev

    return sorted(cp)


def _neg_log_lik_lasso(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    lambda_val: float
) -> float:
    """Compute penalized negative log-likelihood for LASSO.

    Matches R's neg_log_lik function (lines 105-113).
    """
    residuals = y - X @ theta
    L = np.sum(residuals ** 2) / 2 + lambda_val * np.sum(np.abs(theta))
    return L


def _postprocess_lasso(change_points: List[int], n: int, trim: float) -> List[int]:
    """Post-process change points for LASSO.

    Matches R lines 195-207.
    """
    if not change_points:
        return []

    cp = list(change_points)

    # Remove boundary CPs (lines 199-202)
    trim_threshold = trim * n
    cp = [c for c in cp if trim_threshold <= c <= (1 - trim) * n]

    if not cp:
        return []

    # Merge close CPs (lines 204-207)
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
