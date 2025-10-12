"""Synthetic data generation for change point detection.

This module provides functions to generate synthetic time series data with
various types of changes, including:

- Multiple model types (mean, variance, regression, GLM, LASSO, ARMA, GARCH)
- Rich metadata (SNR, difficulty scores, true parameters)
- Realistic parameter generation
- Multiple change patterns (jump, drift, coefficient changes)
- Reproducible with seed control

All functions return dictionaries with 'data', 'changepoints', and 'metadata'
for comprehensive analysis and benchmarking.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple


def _draw_changepoints(n_samples: int, n_changepoints: int,
                      min_segment_length: Optional[int] = None,
                      seed: Optional[int] = None) -> np.ndarray:
    """Draw random change point locations using Dirichlet distribution.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points to generate
        min_segment_length: Minimum samples per segment (default: n_samples//(n_changepoints+1)//2)
        seed: Random seed

    Returns:
        Sorted array of change point indices
    """
    rng = np.random.default_rng(seed=seed)

    if min_segment_length is None:
        min_segment_length = max(5, n_samples // (n_changepoints + 1) // 2)

    # Use Dirichlet to get segment lengths
    alpha = np.ones(n_changepoints + 1) / (n_changepoints + 1) * 2000
    segment_proportions = rng.dirichlet(alpha)
    segment_lengths = (segment_proportions * n_samples).astype(int)

    # Ensure minimum segment length
    while np.any(segment_lengths < min_segment_length):
        too_small = segment_lengths < min_segment_length
        deficit = min_segment_length - segment_lengths[too_small]
        segment_lengths[too_small] = min_segment_length

        # Take from largest segments
        surplus_idx = np.argmax(segment_lengths)
        segment_lengths[surplus_idx] -= deficit.sum()

    # Adjust last segment to match exactly n_samples
    segment_lengths[-1] = n_samples - segment_lengths[:-1].sum()

    # Convert to change points
    changepoints = np.cumsum(segment_lengths)[:-1]

    return changepoints


def make_mean_change(n_samples: int = 500,
                    n_changepoints: int = 3,
                    n_dim: int = 1,
                    mean_deltas: Optional[List[float]] = None,
                    noise_std: float = 1.0,
                    change_type: str = 'jump',
                    seed: Optional[int] = None) -> Dict:
    """Generate data with mean changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        n_dim: Data dimensionality
        mean_deltas: List of mean shifts for each segment (auto-generated if None)
        noise_std: Noise standard deviation
        change_type: 'jump' (step change) or 'drift' (gradual)
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array of shape (n_samples, n_dim)
            - changepoints: array of CP indices
            - true_means: list of segment means
            - metadata: dict with SNR, deltas, difficulty, etc.

    Examples:
        >>> data_dict = make_mean_change(n_samples=500, n_changepoints=3)
        >>> data = data_dict['data']
        >>> cps = data_dict['changepoints']
        >>> print(f"SNR: {data_dict['metadata']['snr']:.2f}")
    """
    rng = np.random.default_rng(seed=seed)

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate mean deltas if not provided
    if mean_deltas is None:
        # Realistic deltas: between 1 and 5 std devs
        mean_deltas = rng.uniform(1.0, 5.0, size=n_dim) * noise_std
        # Random sign
        mean_deltas *= rng.choice([-1, 1], size=n_dim)

    mean_deltas = np.atleast_1d(mean_deltas)
    if len(mean_deltas) == 1 and n_dim > 1:
        mean_deltas = np.tile(mean_deltas, n_dim)

    # Generate segment means
    true_means = []
    current_mean = np.zeros(n_dim)
    true_means.append(current_mean.copy())

    for _ in range(n_segments - 1):
        # Jump or accumulate
        jump = rng.uniform(0.5, 1.5, n_dim) * mean_deltas
        jump *= rng.choice([-1, 1], n_dim)
        current_mean += jump
        true_means.append(current_mean.copy())

    # Generate data
    data = np.zeros((n_samples, n_dim))

    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        segment_length = end - start

        if change_type == 'jump':
            # Step change
            segment_mean = true_means[i]
            data[start:end] = segment_mean + rng.normal(0, noise_std, (segment_length, n_dim))

        elif change_type == 'drift':
            # Gradual drift
            if i < n_segments - 1:
                # Drift from current to next mean
                mean_start = true_means[i]
                mean_end = true_means[i + 1]
                drift = np.linspace(mean_start, mean_end, segment_length)
                data[start:end] = drift + rng.normal(0, noise_std, (segment_length, n_dim))
            else:
                # Last segment stays constant
                data[start:end] = true_means[i] + rng.normal(0, noise_std, (segment_length, n_dim))
        else:
            raise ValueError(f"change_type must be 'jump' or 'drift', got {change_type}")

    # Calculate metadata
    signal_power = np.var([m for means in true_means for m in means])
    noise_power = noise_std ** 2
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    # Difficulty: lower SNR = harder
    difficulty = 1.0 / (1.0 + snr / 10)  # 0 = easy (high SNR), 1 = hard (low SNR)

    metadata = {
        'mean_deltas': mean_deltas.tolist(),
        'segment_lengths': [int(end - start) for start, end in
                           zip(segment_boundaries[:-1], segment_boundaries[1:])],
        'snr_db': float(snr),
        'difficulty': float(difficulty),
        'change_type': change_type,
        'noise_std': noise_std
    }

    return {
        'data': data if n_dim > 1 else data.flatten(),
        'changepoints': changepoints.tolist(),
        'true_means': [m.tolist() if hasattr(m, 'tolist') else m for m in true_means],
        'metadata': metadata
    }


def make_variance_change(n_samples: int = 500,
                        n_changepoints: int = 3,
                        n_dim: int = 1,
                        variance_ratios: Optional[List[float]] = None,
                        base_var: float = 1.0,
                        change_type: str = 'multiplicative',
                        seed: Optional[int] = None) -> Dict:
    """Generate data with variance changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        n_dim: Data dimensionality
        variance_ratios: List of variance multipliers (auto-generated if None)
        base_var: Baseline variance
        change_type: 'multiplicative' or 'additive'
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array of shape (n_samples, n_dim)
            - changepoints: array of CP indices
            - true_variances: list of segment variances
            - metadata: dict with variance_ratios, kurtosis, etc.

    Examples:
        >>> data_dict = make_variance_change(n_samples=500, n_changepoints=2)
        >>> print(data_dict['metadata']['variance_ratios'])
    """
    rng = np.random.default_rng(seed=seed)

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate variance ratios if not provided
    if variance_ratios is None:
        # Ratios between 0.5 and 4.0 for noticeable changes
        variance_ratios = rng.uniform(0.5, 4.0, n_segments)
        variance_ratios[0] = 1.0  # Start with base variance

    variance_ratios = np.atleast_1d(variance_ratios)

    # Generate segment variances
    if change_type == 'multiplicative':
        true_variances = [base_var * ratio for ratio in variance_ratios]
    elif change_type == 'additive':
        true_variances = [base_var + ratio for ratio in variance_ratios]
    else:
        raise ValueError(f"change_type must be 'multiplicative' or 'additive', got {change_type}")

    # Generate data
    data = np.zeros((n_samples, n_dim))

    kurtosis_per_segment = []
    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        segment_length = end - start
        std = np.sqrt(true_variances[i])

        segment_data = rng.normal(0, std, (segment_length, n_dim))
        data[start:end] = segment_data

        # Calculate kurtosis (for metadata)
        from scipy import stats
        kurt = stats.kurtosis(segment_data.flatten())
        kurtosis_per_segment.append(float(kurt))

    metadata = {
        'variance_ratios': variance_ratios.tolist(),
        'true_variances': [float(v) for v in true_variances],
        'kurtosis_per_segment': kurtosis_per_segment,
        'segment_lengths': [int(end - start) for start, end in
                           zip(segment_boundaries[:-1], segment_boundaries[1:])],
        'change_type': change_type,
        'base_var': base_var
    }

    return {
        'data': data if n_dim > 1 else data.flatten(),
        'changepoints': changepoints.tolist(),
        'true_variances': [float(v) for v in true_variances],
        'metadata': metadata
    }


def make_regression_change(n_samples: int = 500,
                          n_changepoints: int = 3,
                          n_features: int = 3,
                          coef_changes: Union[str, List[np.ndarray]] = 'random',
                          noise_std: float = 0.5,
                          correlation: float = 0.0,
                          seed: Optional[int] = None) -> Dict:
    """Generate linear regression data with coefficient changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        n_features: Number of covariates
        coef_changes: 'random', 'sign_flip', 'magnitude', or list of coefficient arrays
        noise_std: Error term std deviation
        correlation: Covariate correlation (0 to 0.9)
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array (n_samples, n_features+1) where [:, 0] is y and [:, 1:] is X
            - changepoints: array of CP indices
            - true_coefs: array (n_segments, n_features) of coefficients per segment
            - X: covariate matrix (n_samples, n_features)
            - y: response vector (n_samples,)
            - metadata: dict with R², condition number, effect size

    Examples:
        >>> data_dict = make_regression_change(n_samples=300, n_changepoints=2, n_features=3)
        >>> X = data_dict['X']
        >>> y = data_dict['y']
        >>> print(data_dict['metadata']['r_squared_per_segment'])
    """
    rng = np.random.default_rng(seed=seed)

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate covariates with optional correlation
    if correlation > 0:
        # Generate correlated covariates
        mean = np.zeros(n_features)
        cov = np.eye(n_features) * (1 - correlation) + correlation
        X = rng.multivariate_normal(mean, cov, size=n_samples)
    else:
        X = rng.normal(0, 1, (n_samples, n_features))

    # Generate coefficients
    if isinstance(coef_changes, str):
        if coef_changes == 'random':
            # Random coefficients in range [-3, 3]
            true_coefs = rng.uniform(-3, 3, (n_segments, n_features))

        elif coef_changes == 'sign_flip':
            # Start with random, then flip signs
            coef_base = rng.uniform(1, 3, n_features)
            true_coefs = []
            for i in range(n_segments):
                if i % 2 == 0:
                    true_coefs.append(coef_base)
                else:
                    true_coefs.append(-coef_base)
            true_coefs = np.array(true_coefs)

        elif coef_changes == 'magnitude':
            # Same signs, different magnitudes
            signs = rng.choice([-1, 1], n_features)
            true_coefs = []
            for i in range(n_segments):
                magnitudes = rng.uniform(0.5, 3.0, n_features)
                true_coefs.append(signs * magnitudes)
            true_coefs = np.array(true_coefs)

        else:
            raise ValueError(f"coef_changes must be 'random', 'sign_flip', 'magnitude', or list of arrays")
    else:
        true_coefs = np.array(coef_changes)

    # Generate response
    y = np.zeros(n_samples)
    r_squared_per_segment = []

    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        X_seg = X[start:end]
        y_pred = X_seg @ true_coefs[i]
        noise = rng.normal(0, noise_std, end - start)
        y_seg = y_pred + noise
        y[start:end] = y_seg

        # Calculate R²
        ss_res = np.sum(noise ** 2)
        ss_tot = np.sum((y_seg - y_seg.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared_per_segment.append(float(r2))

    # Calculate condition number
    condition_number = np.linalg.cond(X)

    # Calculate effect size (how different are coefficients between segments)
    if n_segments > 1:
        coef_diffs = np.linalg.norm(np.diff(true_coefs, axis=0), axis=1)
        effect_size = float(np.mean(coef_diffs))
    else:
        effect_size = 0.0

    metadata = {
        'r_squared_per_segment': r_squared_per_segment,
        'condition_number': float(condition_number),
        'effect_size': effect_size,
        'correlation': correlation,
        'noise_std': noise_std,
        'n_features': n_features
    }

    # Combine y and X as in fastcpd convention
    data = np.column_stack([y, X])

    return {
        'data': data,
        'changepoints': changepoints.tolist(),
        'true_coefs': true_coefs.tolist(),
        'X': X,
        'y': y,
        'metadata': metadata
    }


def make_arma_change(n_samples: int = 500,
                    n_changepoints: int = 3,
                    orders: Optional[List[Tuple[int, int]]] = None,
                    sigma_change: bool = False,
                    innovation: str = 'normal',
                    seed: Optional[int] = None) -> Dict:
    """Generate ARMA time series with parameter changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        orders: List of (p,q) tuples for each segment (auto-generated if None)
        sigma_change: If True, innovation variance also changes
        innovation: 'normal', 't', or 'skew_normal'
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array (n_samples,)
            - changepoints: array of CP indices
            - true_params: list of dicts with 'ar', 'ma', 'sigma' for each segment
            - metadata: dict with stationarity checks, ACF, PACF

    Examples:
        >>> data_dict = make_arma_change(n_samples=500, orders=[(1,1), (2,0)])
        >>> print(data_dict['metadata']['is_stationary'])
    """
    rng = np.random.default_rng(seed=seed)

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate ARMA orders if not provided
    if orders is None:
        max_p, max_q = 2, 2
        orders = [(rng.integers(0, max_p + 1), rng.integers(0, max_q + 1))
                 for _ in range(n_segments)]

    if len(orders) != n_segments:
        orders = list(orders) * (n_segments // len(orders) + 1)
        orders = orders[:n_segments]

    # Generate ARMA parameters
    true_params = []
    data = np.zeros(n_samples)

    from statsmodels.tsa.arima_process import arma_generate_sample

    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        p, q = orders[i]

        # Generate stationary AR coefficients
        if p > 0:
            # Ensure stationarity: roots outside unit circle
            ar_coefs = rng.uniform(-0.8, 0.8, p) / (p + 1)
        else:
            ar_coefs = np.array([])

        # Generate invertible MA coefficients
        if q > 0:
            ma_coefs = rng.uniform(-0.8, 0.8, q) / (q + 1)
        else:
            ma_coefs = np.array([])

        # Variance
        if sigma_change:
            sigma = rng.uniform(0.5, 2.0)
        else:
            sigma = 1.0

        # Generate innovations
        segment_length = end - start
        if innovation == 'normal':
            innovations = rng.normal(0, sigma, segment_length)
        elif innovation == 't':
            innovations = rng.standard_t(5, segment_length) * sigma
        elif innovation == 'skew_normal':
            # Approximate skew normal
            innovations = rng.gamma(2, sigma, segment_length) - 2 * sigma
        else:
            raise ValueError(f"innovation must be 'normal', 't', or 'skew_normal'")

        # Generate ARMA series using statsmodels
        # Note: arma_generate_sample uses [1, -ar1, -ar2, ...] convention
        ar_params = np.r_[1, -ar_coefs] if p > 0 else np.array([1])
        ma_params = np.r_[1, ma_coefs] if q > 0 else np.array([1])

        try:
            segment_data = arma_generate_sample(ar_params, ma_params,
                                               segment_length,
                                               sigma=sigma,
                                               distrvs=lambda size: innovations[:size])
            data[start:end] = segment_data
        except:
            # Fallback: just use innovations if generation fails
            data[start:end] = innovations

        true_params.append({
            'ar': ar_coefs.tolist(),
            'ma': ma_coefs.tolist(),
            'sigma': float(sigma)
        })

    # Metadata: check stationarity
    is_stationary = []
    is_invertible = []

    for i, params in enumerate(true_params):
        p, q = orders[i]

        # Check stationarity (AR roots)
        if p > 0:
            ar_poly = np.r_[1, -np.array(params['ar'])]
            roots = np.roots(ar_poly)
            is_stat = np.all(np.abs(roots) > 1.0)
        else:
            is_stat = True
        is_stationary.append(is_stat)

        # Check invertibility (MA roots)
        if q > 0:
            ma_poly = np.r_[1, np.array(params['ma'])]
            roots = np.roots(ma_poly)
            is_inv = np.all(np.abs(roots) > 1.0)
        else:
            is_inv = True
        is_invertible.append(is_inv)

    metadata = {
        'orders': orders,
        'is_stationary': is_stationary,
        'is_invertible': is_invertible,
        'innovation_type': innovation,
        'segment_lengths': [int(end - start) for start, end in
                           zip(segment_boundaries[:-1], segment_boundaries[1:])]
    }

    return {
        'data': data,
        'changepoints': changepoints.tolist(),
        'true_params': true_params,
        'metadata': metadata
    }


def make_glm_change(n_samples: int = 500,
                   n_changepoints: int = 3,
                   n_features: int = 3,
                   family: str = 'binomial',
                   coef_changes: Union[str, List[np.ndarray]] = 'random',
                   trials: Optional[int] = None,
                   correlation: float = 0.0,
                   seed: Optional[int] = None) -> Dict:
    """Generate GLM data with coefficient changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        n_features: Number of covariates
        family: 'binomial' or 'poisson'
        coef_changes: 'random', 'sign_flip', or list of coefficient arrays
        trials: Number of trials for binomial (default: 1 for logistic regression)
        correlation: Covariate correlation (0 to 0.9)
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array (n_samples, n_features+1) where [:, 0] is y and [:, 1:] is X
            - changepoints: array of CP indices
            - true_coefs: array (n_segments, n_features) of coefficients per segment
            - X: covariate matrix (n_samples, n_features)
            - y: response vector (n_samples,)
            - metadata: dict with AUC (binomial), overdispersion (poisson), etc.

    Examples:
        >>> data_dict = make_glm_change(n_samples=400, family='binomial', n_features=3)
        >>> X = data_dict['X']
        >>> y = data_dict['y']
        >>> print(data_dict['metadata']['separation_per_segment'])
    """
    rng = np.random.default_rng(seed=seed)

    if family not in ['binomial', 'poisson']:
        raise ValueError(f"family must be 'binomial' or 'poisson', got {family}")

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate covariates
    if correlation > 0:
        mean = np.zeros(n_features)
        cov = np.eye(n_features) * (1 - correlation) + correlation
        X = rng.multivariate_normal(mean, cov, size=n_samples)
    else:
        X = rng.normal(0, 1, (n_samples, n_features))

    # Generate coefficients
    if isinstance(coef_changes, str):
        if coef_changes == 'random':
            true_coefs = rng.uniform(-2, 2, (n_segments, n_features))
        elif coef_changes == 'sign_flip':
            coef_base = rng.uniform(0.5, 2.0, n_features)
            true_coefs = []
            for i in range(n_segments):
                if i % 2 == 0:
                    true_coefs.append(coef_base)
                else:
                    true_coefs.append(-coef_base)
            true_coefs = np.array(true_coefs)
        else:
            raise ValueError(f"coef_changes must be 'random', 'sign_flip', or list of arrays")
    else:
        true_coefs = np.array(coef_changes)

    # Generate response
    y = np.zeros(n_samples, dtype=int if family == 'binomial' else float)

    if family == 'binomial':
        if trials is None:
            trials = 1  # Logistic regression

        separation_per_segment = []

        for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
            X_seg = X[start:end]
            eta = X_seg @ true_coefs[i]
            prob = 1 / (1 + np.exp(-eta))

            # Generate binomial outcomes
            if trials == 1:
                y_seg = rng.binomial(1, prob)
            else:
                y_seg = rng.binomial(trials, prob)

            y[start:end] = y_seg

            # Calculate separation (ROC AUC for trials=1)
            if trials == 1:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_seg, prob)
                except:
                    auc = 0.5
                separation_per_segment.append(float(auc))
            else:
                separation_per_segment.append(None)

        metadata = {
            'family': 'binomial',
            'trials': trials,
            'separation_per_segment': separation_per_segment,
            'correlation': correlation,
            'n_features': n_features
        }

    elif family == 'poisson':
        overdispersion_per_segment = []

        for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
            X_seg = X[start:end]
            eta = X_seg @ true_coefs[i]
            lambda_mean = np.exp(eta)

            # Generate Poisson outcomes
            y_seg = rng.poisson(lambda_mean)
            y[start:end] = y_seg

            # Calculate overdispersion (variance / mean)
            overdispersion = float(np.var(y_seg) / (np.mean(y_seg) + 1e-10))
            overdispersion_per_segment.append(overdispersion)

        metadata = {
            'family': 'poisson',
            'overdispersion_per_segment': overdispersion_per_segment,
            'correlation': correlation,
            'n_features': n_features
        }

    # Combine y and X
    data = np.column_stack([y, X])

    return {
        'data': data,
        'changepoints': changepoints.tolist(),
        'true_coefs': true_coefs.tolist(),
        'X': X,
        'y': y,
        'metadata': metadata
    }


def make_garch_change(n_samples: int = 500,
                     n_changepoints: int = 3,
                     orders: Optional[List[Tuple[int, int]]] = None,
                     volatility_regimes: Optional[List[str]] = None,
                     seed: Optional[int] = None) -> Dict:
    """Generate GARCH time series with volatility regime changes.

    Args:
        n_samples: Total number of samples
        n_changepoints: Number of change points
        orders: List of (p,q) tuples for each segment (auto-generated if None)
        volatility_regimes: List of 'low', 'medium', 'high' for each segment
        seed: Random seed

    Returns:
        Dictionary with:
            - data: array (n_samples,) of returns
            - changepoints: array of CP indices
            - true_params: list of dicts with 'omega', 'alpha', 'beta' for each segment
            - volatility: array (n_samples,) of conditional volatility
            - metadata: dict with volatility ratios, kurtosis

    Examples:
        >>> data_dict = make_garch_change(n_samples=600, n_changepoints=2)
        >>> returns = data_dict['data']
        >>> vol = data_dict['volatility']
        >>> print(data_dict['metadata']['avg_volatility_per_segment'])
    """
    rng = np.random.default_rng(seed=seed)

    # Generate change points
    changepoints = _draw_changepoints(n_samples, n_changepoints, seed=seed)
    segment_boundaries = np.concatenate([[0], changepoints, [n_samples]])
    n_segments = n_changepoints + 1

    # Generate GARCH orders if not provided
    if orders is None:
        # Default to GARCH(1,1)
        orders = [(1, 1)] * n_segments

    if len(orders) != n_segments:
        orders = list(orders) * (n_segments // len(orders) + 1)
        orders = orders[:n_segments]

    # Generate volatility regimes
    if volatility_regimes is None:
        regime_choices = ['low', 'medium', 'high']
        volatility_regimes = rng.choice(regime_choices, n_segments, replace=True).tolist()

    regime_params = {
        'low': {'omega': 0.01, 'alpha_scale': 0.05, 'beta_scale': 0.90},
        'medium': {'omega': 0.05, 'alpha_scale': 0.10, 'beta_scale': 0.85},
        'high': {'omega': 0.10, 'alpha_scale': 0.20, 'beta_scale': 0.75}
    }

    # Generate GARCH parameters and data
    data = np.zeros(n_samples)
    volatility = np.zeros(n_samples)
    true_params = []
    avg_vol_per_segment = []

    for i, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        p, q = orders[i]
        regime = volatility_regimes[i]
        regime_param = regime_params[regime]

        # Generate GARCH parameters
        omega = regime_param['omega']

        if p > 0:
            alpha = rng.uniform(0.5, 1.5, p) * regime_param['alpha_scale']
        else:
            alpha = np.array([])

        if q > 0:
            beta = rng.uniform(0.5, 1.5, q) * regime_param['beta_scale']
        else:
            beta = np.array([])

        # Ensure persistence < 1 for stationarity
        persistence = np.sum(alpha) + np.sum(beta)
        if persistence >= 0.98:
            scale_factor = 0.95 / (persistence + 1e-10)
            alpha = alpha * scale_factor
            beta = beta * scale_factor
            persistence = np.sum(alpha) + np.sum(beta)

        # Simulate GARCH process
        segment_length = end - start
        seg_vol = np.zeros(segment_length)
        seg_returns = np.zeros(segment_length)

        # Initialize
        seg_vol[0] = np.sqrt(omega / (1 - persistence + 1e-10))

        for t in range(segment_length):
            if t > 0:
                # GARCH variance equation
                variance = omega

                # ARCH terms
                for j in range(min(p, t)):
                    variance += alpha[j] * seg_returns[t-1-j]**2

                # GARCH terms
                for j in range(min(q, t)):
                    variance += beta[j] * seg_vol[t-1-j]**2

                seg_vol[t] = np.sqrt(max(variance, 1e-10))

            # Generate return
            seg_returns[t] = seg_vol[t] * rng.standard_normal()

        data[start:end] = seg_returns
        volatility[start:end] = seg_vol

        true_params.append({
            'omega': float(omega),
            'alpha': alpha.tolist(),
            'beta': beta.tolist(),
            'regime': regime
        })

        avg_vol_per_segment.append(float(np.mean(seg_vol)))

    # Calculate metadata
    from scipy import stats
    kurtosis_per_segment = []

    for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        kurt = stats.kurtosis(data[start:end])
        kurtosis_per_segment.append(float(kurt))

    # Volatility ratios
    vol_ratios = [avg_vol_per_segment[i] / avg_vol_per_segment[0]
                  for i in range(n_segments)]

    metadata = {
        'orders': orders,
        'volatility_regimes': volatility_regimes,
        'avg_volatility_per_segment': avg_vol_per_segment,
        'volatility_ratios': vol_ratios,
        'kurtosis_per_segment': kurtosis_per_segment,
        'segment_lengths': [int(end - start) for start, end in
                           zip(segment_boundaries[:-1], segment_boundaries[1:])]
    }

    return {
        'data': data,
        'changepoints': changepoints.tolist(),
        'true_params': true_params,
        'volatility': volatility,
        'metadata': metadata
    }


def add_annotation_noise(true_changepoints: Union[List, np.ndarray],
                        n_annotators: int = 5,
                        noise_std: float = 5.0,
                        agreement_rate: float = 0.8,
                        seed: Optional[int] = None) -> List[List[int]]:
    """Simulate multiple human annotators with varying agreement.

    Useful for testing covering metric and multi-annotator scenarios.

    Args:
        true_changepoints: Ground truth change points
        n_annotators: Number of annotators to simulate
        noise_std: Std of Gaussian noise added to CP locations
        agreement_rate: Probability each annotator includes each CP
        seed: Random seed

    Returns:
        List of lists, each sublist is one annotator's change points

    Examples:
        >>> true_cps = [100, 200, 300]
        >>> annotators = add_annotation_noise(true_cps, n_annotators=5)
        >>> print(f"Annotator 1: {annotators[0]}")
        >>> print(f"Annotator 2: {annotators[1]}")
    """
    rng = np.random.default_rng(seed=seed)
    true_cps = np.atleast_1d(true_changepoints)

    annotator_cps = []

    for _ in range(n_annotators):
        cps = []
        for true_cp in true_cps:
            # Decide if annotator includes this CP
            if rng.random() < agreement_rate:
                # Add noise to location
                noisy_cp = true_cp + rng.normal(0, noise_std)
                noisy_cp = int(np.clip(noisy_cp, 0, np.inf))
                cps.append(noisy_cp)

        # Sort and remove duplicates
        cps = sorted(set(cps))
        annotator_cps.append(cps)

    return annotator_cps
