import numpy as np


def mean(data):
    """
    Variance estimation for mean change models (Rice estimator).

    data : array-like, shape (n, p)
      Each row is a p-vector observation.

    Returns
    -------
    ndarray, shape (p, p)
      Estimated variance-covariance matrix.
    """
    data_matrix = np.asarray(data)
    diffs = data_matrix[1:] - data_matrix[:-1]
    return np.mean(diffs[:, :, None] * diffs[:, None, :], axis=0) / 2


def median(data):
    """
    Variance estimation for median change models (Rice estimator).

    data : array-like, shape (n,)
      Univariate series.

    Returns
    -------
    float
      Estimated variance.
    """
    data_flat = np.asarray(data).ravel()
    return 2 * (2 * np.mean(np.abs(np.diff(data_flat))) / 3) ** 2
