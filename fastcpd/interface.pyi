import numpy as np
from numpy.typing import NDArray

__all__ = ['fastcpd_impl']

def fastcpd_impl(
    beta: float,
    cost_adjustment: str,
    cp_only: bool,
    data: NDArray[np.float64],
    epsilon: float,
    family: str,
    line_search: NDArray[np.float64],
    lower: NDArray[np.float64],
    momentum_coef: float,
    order: NDArray[np.int64],
    p: int,
    p_response: int,
    pruning_coef: float,
    segment_count: int,
    trim: float,
    upper: NDArray[np.float64],
    vanilla_percentage: float,
    variance_estimate: NDArray[np.float64],
    warm_start: bool,
) -> NDArray[np.float64]:
    ...
