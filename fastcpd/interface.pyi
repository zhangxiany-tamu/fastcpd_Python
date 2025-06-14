import numpy as np
from numpy.typing import NDArray

__all__ = ['fastcpd_impl']

def fastcpd_impl(data: NDArray[np.float64], variance_estimate: NDArray[np.float64]) -> NDArray[np.float64]:
    ...
