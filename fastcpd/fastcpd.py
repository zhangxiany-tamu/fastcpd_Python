"""Main fastcpd function and result class."""

from typing import Callable, Optional, Union, List
import numpy as np
from dataclasses import dataclass

# Try to import C++ extension
_fastcpd_impl = None
_import_error = None

try:
    # Use absolute import with __package__ to avoid circular import issues
    import fastcpd._fastcpd_impl as _fastcpd_impl
except (ImportError, ModuleNotFoundError) as e:
    _import_error = str(e)
except Exception as e:
    _import_error = f"Unexpected error: {str(e)}"

from fastcpd.pelt_sklearn import _fastcpd_sklearn


@dataclass
class FastcpdResult:
    """Result from fastcpd change point detection.

    Attributes:
        raw_cp_set: Raw change point indices
        cp_set: Final change point indices
        cost_values: Cost values for each segment
        residuals: Residuals from the model
        thetas: Parameter estimates for each segment
        data: Original data
        family: Model family used
    """
    raw_cp_set: np.ndarray
    cp_set: np.ndarray
    cost_values: np.ndarray
    residuals: np.ndarray
    thetas: np.ndarray
    data: np.ndarray
    family: str

    def __repr__(self) -> str:
        n_cp = len(self.cp_set)
        return f"FastcpdResult(n_changepoints={n_cp}, family='{self.family}')"

    def plot(self):
        """Plot the data with detected change points."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        fig, ax = plt.subplots(figsize=(12, 6))

        if self.data.ndim == 1:
            ax.plot(self.data, 'o-', markersize=3, label='Data')
        else:
            for i in range(min(3, self.data.shape[1])):  # Plot up to 3 dimensions
                ax.plot(self.data[:, i], 'o-', markersize=3, label=f'Dim {i+1}')

        # Plot change points
        for cp in self.cp_set:
            ax.axvline(x=cp, color='r', linestyle='--', alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Change Point Detection ({self.family} model)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax


def _variance_mean_rice(data: np.ndarray) -> np.ndarray:
    """Rice estimator for variance in mean change models (matches R).

    For univariate data returns shape (1, 1). For multivariate, returns (d, d).
    Adds a tiny jitter on the diagonal if not positive definite.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < 2:
        return np.eye(data.shape[1], dtype=np.float64)
    diffs = np.diff(data, axis=0)
    # Average outer product of differences divided by 2
    cov = (diffs.T @ diffs) / (2.0 * diffs.shape[0])
    # Ensure symmetric
    cov = (cov + cov.T) * 0.5
    # Ensure positive definiteness with minimal jitter if needed
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eps = 1e-10
        eye = np.eye(cov.shape[0], dtype=np.float64)
        while True:
            try:
                np.linalg.cholesky(cov + eps * eye)
                cov = cov + eps * eye
                break
            except np.linalg.LinAlgError:
                eps *= 10.0
                if eps > 1e-2:
                    # Fallback to identity if still not PD
                    cov = eye
                    break
    return cov


def fastcpd(
    data: Union[np.ndarray, list],
    beta: Union[str, float] = "MBIC",
    cost_adjustment: str = "MBIC",
    family: str = "mean",
    epsilon: float = 1e-10,
    segment_count: int = 10,
    trim: float = 0.02,
    momentum_coef: float = 0.0,
    p: int = 0,
    order: Optional[list] = None,
    cost: Optional[Callable] = None,
    cost_gradient: Optional[Callable] = None,
    cost_hessian: Optional[Callable] = None,
    cp_only: bool = False,
    vanilla_percentage: Union[float, str] = 0.0,
    warm_start: bool = True,
    lower: Optional[list] = None,
    upper: Optional[list] = None,
    line_search: Optional[list] = None,
    variance_estimate: Optional[np.ndarray] = None,
    p_response: int = 1,
    pruning_coef: float = 0.0,
    multiple_epochs: Optional[Callable[[int], int]] = None,
    lasso_alpha: float = 1.0,
    lasso_cv: bool = False,
    min_segment_length: Optional[int] = None,
) -> FastcpdResult:
    """Fast change point detection using sequential gradient descent.

    Parameters:
        data: Input data array of shape (n, d) where n is number of observations
            and d is the dimensionality. For univariate data, shape can be (n,).
        beta: Penalty for number of change points. Can be "BIC", "MBIC", "MDL" or a numeric value.
            Default is "MBIC" which uses (p + 2) * log(n) / 2.
        cost_adjustment: Cost adjustment criterion. Can be "BIC", "MBIC", "MDL" or None.
        family: Model family. Options:
            - "mean": Mean change detection
            - "variance": Variance change detection
            - "meanvariance": Mean and/or variance change
            - "ar": AR(p) model
            - "arma": ARMA(p,q) model
            - "garch": GARCH(p,q) model
            - "var": VAR(p) model
            - "custom": Custom cost function
        epsilon: Small constant to avoid numerical issues
        segment_count: Initial number of segments for warm start
        trim: Proportion to trim from boundaries
        momentum_coef: Momentum coefficient for gradient descent
        p: Number of parameters (for AR models)
        order: Order parameters for time series models, e.g., [p, q] for ARMA(p,q)
        cost: Custom cost function (for family="custom")
        cost_gradient: Gradient of custom cost function
        cost_hessian: Hessian of custom cost function
        cp_only: Whether to return only change points (no parameters)
        vanilla_percentage: Fraction of data to process with vanilla PELT (0 to 1)
        warm_start: Whether to use warm start initialization
        lower: Lower bounds for parameters
        upper: Upper bounds for parameters
        line_search: Line search coefficients
        variance_estimate: Known variance-covariance matrix
        p_response: Dimension of response variable
        pruning_coef: Pruning coefficient for optimization
        multiple_epochs: Function to determine number of epochs

    Returns:
        FastcpdResult: Object containing change points and related information

    Examples:
        >>> import numpy as np
        >>> from fastcpd import fastcpd
        >>>
        >>> # Mean change detection
        >>> data = np.concatenate([np.random.normal(0, 1, 300),
        ...                        np.random.normal(5, 1, 400),
        ...                        np.random.normal(2, 1, 300)])
        >>> result = fastcpd(data, family="mean")
        >>> print(result.cp_set)

        >>> # Multivariate mean change
        >>> data = np.concatenate([
        ...     np.random.multivariate_normal([0, 0, 0], np.eye(3), 300),
        ...     np.random.multivariate_normal([5, 5, 5], np.eye(3), 400),
        ...     np.random.multivariate_normal([2, 2, 2], np.eye(3), 300)
        ... ])
        >>> result = fastcpd(data, family="mean")
        >>> result.plot()
    """
    def _auto_vanilla(n_obs_val: int, fam: str) -> float:
        # Adaptive default based on data size and family
        if fam in ["mean", "variance", "meanvariance", "ar", "var", "arma", "garch"]:
            # Time series and moment families default to PELT for stability/accuracy
            return 1.0
        if n_obs_val <= 200:
            return 1.0
        if n_obs_val <= 500:
            return 0.5
        return 0.0

    def _enforce_min_segment_length(cp: np.ndarray, n_total: int, min_len: int) -> np.ndarray:
        if min_len is None or min_len <= 0:
            return cp
        cp_list = list(map(int, sorted(cp.tolist())))
        changed = True
        while changed and cp_list:
            changed = False
            boundaries = [0] + cp_list + [n_total]
            seg_lengths = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
            # Find any short segment (exclude first and last only if they are bounded by CPs)
            for i, seg_len in enumerate(seg_lengths):
                if seg_len < min_len and (i > 0 and i < len(seg_lengths) - 1):
                    # Remove the change point that creates this short segment
                    # Short segment sits between cp_list[i-1] and cp_list[i]
                    # Remove the nearer CP to merge with larger neighbor; choose to drop right CP
                    del cp_list[i-1 if seg_lengths[i-1] < seg_lengths[i+1] else i]
                    changed = True
                    break
        return np.array(cp_list, dtype=float)
    # For time series families (AR, VAR, ARMA, GARCH), use Python implementations
    if family in ['ar', 'var', 'arma', 'garch']:
        from fastcpd.pelt_ts import _fastcpd_ar, _fastcpd_var

        # Get order parameter
        if order is None or len(order) == 0:
            raise ValueError(f"{family.upper()} model requires 'order' parameter")

        if family == 'ar' or family == 'var':
            ts_order = order[0] if isinstance(order, list) else order

            if family == 'ar':
                result_dict = _fastcpd_ar(data, ts_order, beta, trim)
            else:  # family == 'var'
                result_dict = _fastcpd_var(data, ts_order, beta, trim)

            return FastcpdResult(
                raw_cp_set=result_dict['raw_cp_set'],
                cp_set=result_dict['cp_set'],
                cost_values=np.array([]),  # Not returned by TS models
                residuals=np.array([]),
                thetas=np.array([]),
                data=data,
                family=family,
            )
        elif family == 'arma':
            from fastcpd.pelt_arma_vanilla import _fastcpd_arma_vanilla

            # ARMA requires [p, q] order
            if not isinstance(order, list) or len(order) != 2:
                raise ValueError("ARMA model requires order=[p, q]")

            # Calculate beta if string
            if isinstance(beta, str):
                p_param = sum(order) + 1  # p + q + 1 (including σ²)
                if beta == "MBIC":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = (p_param + 2) * np.log(n_obs) / 2
                elif beta == "BIC":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = p_param * np.log(n_obs) / 2
                elif beta == "MDL":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = (p_param / 2) * np.log(n_obs)
                else:
                    raise ValueError(f"Unknown beta criterion: {beta}")

            # Use vanilla PELT with statsmodels (pure Python, no R dependency)
            result_dict = _fastcpd_arma_vanilla(data, order, beta, trim)

            result_obj = FastcpdResult(
                raw_cp_set=result_dict['raw_cp_set'],
                cp_set=result_dict['cp_set'],
                cost_values=result_dict['cost_values'],
                residuals=result_dict['residuals'],
                thetas=result_dict['thetas'],
                data=result_dict['data'],
                family=result_dict['family'],
            )
            # Optional post-processing: enforce minimum segment length
            if min_segment_length and min_segment_length > 0:
                result_obj.cp_set = _enforce_min_segment_length(result_obj.cp_set, len(result_obj.data), min_segment_length)
            return result_obj
        elif family == 'garch':
            from fastcpd.pelt_garch_vanilla import _fastcpd_garch_vanilla

            # GARCH requires [p, q] order
            if not isinstance(order, list) or len(order) != 2:
                raise ValueError("GARCH model requires order=[p, q]")

            # Calculate beta if string
            if isinstance(beta, str):
                p_param = sum(order) + 1  # p + q + 1 (omega + alpha + beta)
                if beta == "MBIC":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = (p_param + 2) * np.log(n_obs) / 2
                elif beta == "BIC":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = p_param * np.log(n_obs) / 2
                elif beta == "MDL":
                    data_arr = np.asarray(data)
                    n_obs = len(data_arr) if data_arr.ndim == 1 else data_arr.shape[0]
                    beta = (p_param / 2) * np.log(n_obs)
                else:
                    raise ValueError(f"Unknown beta criterion: {beta}")

            # Use vanilla PELT with arch package (pure Python, no R dependency)
            result_dict = _fastcpd_garch_vanilla(data, order, beta, trim)

            result_obj = FastcpdResult(
                raw_cp_set=result_dict['raw_cp_set'],
                cp_set=result_dict['cp_set'],
                cost_values=result_dict['cost_values'],
                residuals=result_dict['residuals'],
                thetas=result_dict['thetas'],
                data=result_dict['data'],
                family=result_dict['family'],
            )
            if min_segment_length and min_segment_length > 0:
                result_obj.cp_set = _enforce_min_segment_length(result_obj.cp_set, len(result_obj.data), min_segment_length)
            return result_obj

    # For GLM, LASSO, and linear regression families, use Python implementations
    if family in ['binomial', 'poisson', 'lasso', 'lm']:
        # Compute adaptive vanilla if requested
        data_arr_for_len = np.asarray(data)
        n_obs_for_len = len(data_arr_for_len) if data_arr_for_len.ndim == 1 else data_arr_for_len.shape[0]
        if isinstance(vanilla_percentage, str) and vanilla_percentage.lower() == 'auto':
            vanilla_val = _auto_vanilla(n_obs_for_len, family)
        else:
            vanilla_val = float(vanilla_percentage)
        # Linear regression has its own implementation (needs variance estimation)
        if family == 'lm':
            from fastcpd.pelt_lm import _fastcpd_lm
            result_dict = _fastcpd_lm(
                data, beta, trim, p_response
            )
        # LASSO has its own implementation (different SEN update with lambda)
        elif family == 'lasso':
            from fastcpd.pelt_lasso import _fastcpd_lasso_sen
            result_dict = _fastcpd_lasso_sen(
                data, beta, vanilla_val, trim, segment_count
            )
        # Binomial/Poisson: Use SEN implementation if vanilla_percentage is specified
        elif vanilla_val != 1.0:
            from fastcpd.pelt_sen import _fastcpd_sen
            result_dict = _fastcpd_sen(
                data, beta, cost_adjustment, family, segment_count, trim,
                warm_start, vanilla_val
            )
        else:
            # Use pure PELT implementation (vanilla_percentage = 1.0)
            result_dict = _fastcpd_sklearn(
                data, beta, cost_adjustment, family, segment_count, trim,
                warm_start, lasso_alpha, lasso_cv
            )
        result_obj = FastcpdResult(
            raw_cp_set=result_dict['raw_cp_set'],
            cp_set=result_dict['cp_set'],
            cost_values=result_dict['cost_values'],
            residuals=result_dict['residuals'],
            thetas=result_dict['thetas'],
            data=result_dict['data'],
            family=result_dict['family'],
        )
        if min_segment_length and min_segment_length > 0:
            result_obj.cp_set = _enforce_min_segment_length(result_obj.cp_set, len(result_obj.data), min_segment_length)
        return result_obj

    # Convert data to numpy array
    data = np.asarray(data, dtype=np.float64)

    # Handle 1D data
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.ndim != 2:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")

    n_obs, n_dim = data.shape

    # For mean/variance families, p should default to the number of dimensions
    if p == 0:
        if family == 'mean':
            p = n_dim
        elif family == 'variance':
            # variance has covariance matrix parameters (n_dim^2)
            p = n_dim * n_dim
        elif family == 'meanvariance':
            # meanvariance has mean (n_dim) + covariance matrix (n_dim^2) parameters
            p = n_dim + n_dim * n_dim

    # For mean/variance families without SEN implementation, use PELT; allow 'auto'
    if isinstance(vanilla_percentage, str) and vanilla_percentage.lower() == 'auto':
        vanilla_percentage = 1.0
    if family in ['mean', 'variance', 'meanvariance'] and float(vanilla_percentage) < 1.0:
        vanilla_percentage = 1.0

    # Calculate beta if string
    if isinstance(beta, str):
        p_param = p if p > 0 else n_dim
        if beta == "MBIC":
            beta = (p_param + 2) * np.log(n_obs) / 2
        elif beta == "BIC":
            beta = p_param * np.log(n_obs) / 2
        elif beta == "MDL":
            beta = (p_param / 2) * np.log(n_obs)
        else:
            raise ValueError(f"Unknown beta criterion: {beta}")

    # Set defaults
    if order is None:
        order = []
    if lower is None:
        lower = []
    if upper is None:
        upper = []
    if line_search is None:
        line_search = []
    # Variance handling to match fastcpd R implementation
    if family == 'mean' and (variance_estimate is None or np.asarray(variance_estimate).size == 0):
        variance_estimate = _variance_mean_rice(data)
    elif variance_estimate is None:
        variance_estimate = np.array([], dtype=np.float64).reshape(0, 0)

    # Ensure variance_estimate is 2D
    if variance_estimate.size > 0 and variance_estimate.ndim == 1:
        variance_estimate = variance_estimate.reshape(-1, 1)
    elif variance_estimate.size == 0:
        # Ensure empty array is (0, 0) not (1, 0) or (0, 1)
        variance_estimate = variance_estimate.reshape(0, 0)

    # Make data C-contiguous
    data = np.ascontiguousarray(data)
    variance_estimate = np.ascontiguousarray(variance_estimate)

    # Check if C++ extension is available
    if _fastcpd_impl is None:
        error_details = f" Import error: {_import_error}" if '_import_error' in globals() else ""
        raise ImportError(
            f"C++ extension '_fastcpd_impl' is not available, which is required for family='{family}'. "
            f"This usually means the package was not properly installed.{error_details} "
            "Try reinstalling with: pip install --force-reinstall fastcpd"
        )

    # Call C++ implementation
    result = _fastcpd_impl.fastcpd_impl(
        data=data,
        beta=beta,
        cost_adjustment=cost_adjustment,
        segment_count=segment_count,
        trim=trim,
        momentum_coef=momentum_coef,
        multiple_epochs_function=multiple_epochs,
        family=family,
        epsilon=epsilon,
        p=p,
        order=order,
        cost_pelt=cost,
        cost_sen=cost,
        cost_gradient=cost_gradient,
        cost_hessian=cost_hessian,
        cp_only=cp_only,
        vanilla_percentage=float(vanilla_percentage),
        warm_start=warm_start,
        lower=lower,
        upper=upper,
        line_search=line_search,
        variance_estimate=variance_estimate,
        p_response=p_response,
        pruning_coef=pruning_coef,
        r_progress=False,
    )

    # Convert result to numpy arrays
    result_obj = FastcpdResult(
        raw_cp_set=np.array(result.raw_cp_set),
        cp_set=np.array(result.cp_set),
        cost_values=np.array(result.cost_values),
        residuals=np.array(result.residuals),
        thetas=np.array(result.thetas),
        data=data,
        family=family,
    )
    if min_segment_length and min_segment_length > 0:
        result_obj.cp_set = _enforce_min_segment_length(result_obj.cp_set, len(result_obj.data), min_segment_length)
    return result_obj
