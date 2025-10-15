"""Nonparametric cost functions with PELT.

Implements rank-based and RBF-kernel-based nonparametric costs with a fast
PELT driver. For kernel costs, defaults to Random Fourier Features (RFF)
for scalability while keeping an exact small-n path available in the future.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union
import numpy as np


def _postprocess_changepoints(change_points: List[int], n: int, trim: float) -> List[int]:
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
    close_idx = np.where(diffs < trim_threshold)[0]
    if len(close_idx) > 0:
        merged: List[int] = []
        skip: set[int] = set()
        for i in range(len(cp)):
            if i in skip:
                continue
            if i in close_idx:
                merged_cp = int(np.floor((cp[i] + cp[i + 1]) / 2))
                merged.append(merged_cp)
                skip.add(i + 1)
            elif i - 1 not in close_idx:
                merged.append(cp[i])
        cp = merged
    # Remove 0
    cp = [c for c in cp if c > 0]
    return sorted(cp)


def _enforce_min_segment_length(cp: np.ndarray, n_total: int, min_len: Optional[int]) -> np.ndarray:
    if not min_len or min_len <= 0 or cp.size == 0:
        return cp
    cp_list = list(map(int, sorted(cp.tolist())))
    changed = True
    while changed and cp_list:
        changed = False
        boundaries = [0] + cp_list + [n_total]
        seg_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        for i, seg_len in enumerate(seg_lengths):
            if seg_len < min_len and (i > 0 and i < len(seg_lengths) - 1):
                # Remove the closer CP
                del cp_list[i - 1 if seg_lengths[i - 1] < seg_lengths[i + 1] else i]
                changed = True
                break
    return np.array(cp_list, dtype=float)


def _pelt_generic_cost(
    n: int,
    beta: float,
    cost_fn: Callable[[int, int], float],
    min_seg_len: int,
) -> List[int]:
    """Generic PELT with pruning for a supplied segment cost function.

    cost_fn expects half-open indices [a, b), and the driver will supply a in [0, t)
    and b = t for t from 1..n.
    """
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    R = np.zeros(n + 1, dtype=int)
    cp_list: List[int] = [0]

    for t in range(1, n + 1):
        candidates: List[Tuple[float, int]] = []
        costs_cache: dict[int, float] = {}
        for tau in cp_list:
            seg_len = t - tau
            if seg_len < min_seg_len:
                c = 0.0
            else:
                c = cost_fn(tau, t)
            costs_cache[tau] = c
            candidates.append((F[tau] + c + beta, tau))

        if candidates:
            min_val, min_tau = min(candidates)
            F[t] = min_val
            R[t] = min_tau

            # Prune: keep tau where F[tau] + cost <= F[t]
            pruned: List[int] = []
            for tau in cp_list:
                if tau < t:
                    if F[tau] + costs_cache.get(tau, np.inf) <= F[t]:
                        pruned.append(tau)
                else:
                    pruned.append(tau)
            cp_list = pruned + [t]

    # Backtrack
    cp: List[int] = []
    curr = n
    while curr > 0:
        prev = R[curr]
        if prev > 0:
            cp.append(prev)
        curr = prev
    return sorted(cp)


# ---------- Rank-based cost ----------


def _precompute_rank_signal(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centered rank signal r_t and Sigma_r inverse.

    r_{t,j} = rank(y_{t,j}) - (T+1)/2. Sigma_r = (1/T) sum (r_t + 1/2)^T (r_t + 1/2).
    Returns (r, Sigma_r_inv) with small ridge regularization if needed.
    """
    n, d = y.shape
    r = np.empty_like(y, dtype=np.float64)
    for j in range(d):
        # ranks in 1..n
        order = np.argsort(y[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        r[:, j] = ranks - (n + 1) / 2.0

    # Sigma_r
    rp = r + 0.5  # r_t + 1/2
    Sigma = (rp.T @ rp) / float(n)
    # Regularize if nearly singular
    eps = 1e-10 * np.trace(Sigma) / Sigma.shape[0]
    Sigma += eps * np.eye(Sigma.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)
    return r, Sigma_inv


def _rank_cost_factory(y: np.ndarray) -> Tuple[Callable[[int, int], float], int]:
    """Return cost_fn and min segment length for rank-based cost.

    cost = (L) * mean(r)'.Sigma_inv.mean(r), where L = b-a and mean over segment.
    Precompute prefix sums of r for O(1) segment mean.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, d = y.shape
    r, Sigma_inv = _precompute_rank_signal(y)
    pref = np.zeros((n + 1, d), dtype=np.float64)
    pref[1:] = np.cumsum(r, axis=0)

    def cost_fn(a: int, b: int) -> float:
        L = b - a
        if L <= 0:
            return 0.0
        mean_r = (pref[b] - pref[a]) / float(L)
        return float(L) * float(mean_r.T @ Sigma_inv @ mean_r)

    min_len = max(2, d + 1)
    return cost_fn, min_len


# ---------- RBF kernel cost (RFF) ----------


def _pairwise_median_distance(y: np.ndarray, max_samples: int = 10000) -> float:
    n = y.shape[0]
    idx = np.arange(n)
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
    z = y[idx]
    # sample pairs
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    # use upper triangle excluding diagonal
    tri = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(tri) if tri.size else 1.0
    return float(med)


def _rff_features(y: np.ndarray, m: int, gamma: float, seed: int = 0) -> np.ndarray:
    """Random Fourier Features for RBF kernel.

    k(x, y) = exp(-gamma ||x - y||^2). Features: phi(x) = sqrt(2/m)*cos(Wx + b),
    W ~ N(0, 2*gamma I), b ~ Uniform[0, 2pi]. Then E[phi(x)^T phi(y)] = k(x,y).
    """
    rng = np.random.default_rng(seed)
    d = y.shape[1]
    W = rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(m, d))
    b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(m,))
    Z = y @ W.T + b[None, :]
    phi = np.sqrt(2.0 / m) * np.cos(Z)
    return phi.astype(np.float64)


def _rbf_cost_factory(
    y: np.ndarray,
    gamma: Optional[float] = None,
    feature_dim: int = 256,
    seed: int = 0,
) -> Tuple[Callable[[int, int], float], int, float]:
    """Return cost_fn and min segment length for RBF cost using RFF.

    Using c_rbf(L) = L - (1/L) ||sum_{t in seg} phi(y_t)||^2, since k(x,x)=1,
    and inner products approximated via RFF.
    Returns (cost_fn, min_len, chosen_gamma).
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, d = y.shape
    if gamma is None:
        med = _pairwise_median_distance(y)
        if med <= 0:
            med = 1.0
        gamma = 1.0 / max(med, 1e-12)
    phi = _rff_features(y, m=feature_dim, gamma=gamma, seed=seed)
    pref_phi = np.zeros((n + 1, feature_dim), dtype=np.float64)
    pref_phi[1:] = np.cumsum(phi, axis=0)

    def cost_fn(a: int, b: int) -> float:
        L = b - a
        if L <= 0:
            return 0.0
        s = pref_phi[b] - pref_phi[a]
        return float(L) - float((s @ s) / float(L))

    min_len = 2
    return cost_fn, min_len, float(gamma)


# ---------- Public runners ----------


def run_rank(
    data: Union[np.ndarray, List[List[float]], List[float]],
    beta: float,
    trim: float = 0.02,
    min_segment_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(data, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]
    cost_fn, min_len_rank = _rank_cost_factory(y)
    min_len = max(min_segment_length or 1, min_len_rank)
    cp_raw = _pelt_generic_cost(n=n, beta=beta, cost_fn=cost_fn, min_seg_len=min_len)
    cp = _postprocess_changepoints(cp_raw, n, trim)
    cp = _enforce_min_segment_length(np.array(cp, dtype=float), n, min_segment_length)
    return np.array(cp_raw, dtype=float), cp


def run_rbf(
    data: Union[np.ndarray, List[List[float]], List[float]],
    beta: float,
    trim: float = 0.02,
    min_segment_length: Optional[int] = None,
    gamma: Optional[float] = None,
    feature_dim: int = 256,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(data, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]
    cost_fn, min_len_rbf, gamma_used = _rbf_cost_factory(y, gamma=gamma, feature_dim=feature_dim, seed=seed)
    min_len = max(min_segment_length or 1, min_len_rbf)
    cp_raw = _pelt_generic_cost(n=n, beta=beta, cost_fn=cost_fn, min_seg_len=min_len)
    cp = _postprocess_changepoints(cp_raw, n, trim)
    cp = _enforce_min_segment_length(np.array(cp, dtype=float), n, min_segment_length)
    return np.array(cp_raw, dtype=float), cp, gamma_used


def calibrate_beta_n_bkps(
    solver: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    target_k: int,
    beta_min: float = 0.0,
    beta_max: float = 1e9,
    max_iter: int = 20,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Bisection on beta to reach target number of change points.

    The solver(beta) must return (raw_cp, cp_set). Assumes cp_count is non-increasing in beta.
    """
    best_beta = None
    best_cp = None
    best_raw = None
    lo, hi = beta_min, beta_max
    # Expand hi if needed
    raw, cp = solver(lo)
    if len(cp) < target_k:
        return lo, raw, cp
    raw, cp = solver(hi)
    # Expand hi if needed (guarded)
    expand = 0
    while len(cp) > target_k and hi < 1e16 and expand < 8:
        hi *= 10
        raw, cp = solver(hi)
        expand += 1

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        raw, cp = solver(mid)
        k = len(cp)
        best_beta, best_raw, best_cp = mid, raw, cp
        if k == target_k:
            break
        if k > target_k:
            lo = mid
        else:
            hi = mid
    return float(best_beta), np.asarray(best_raw, dtype=float), np.asarray(best_cp, dtype=float)
