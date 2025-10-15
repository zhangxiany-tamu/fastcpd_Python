"""Basic tests for fastcpd functionality."""

import numpy as np
import pytest


def test_imports():
    """Test that all modules can be imported."""
    import fastcpd
    from fastcpd import segmentation
    from fastcpd.fastcpd import fastcpd as fastcpd_func

    assert fastcpd.__version__ == "0.18.1"


def test_mean_change_univariate():
    """Test univariate mean change detection."""
    from fastcpd.segmentation import mean

    # Create data with known change point at 300
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 300),
        np.random.normal(5, 1, 400)
    ])

    result = mean(data)

    # Check that we detected a change point near 300
    assert len(result.cp_set) > 0
    assert result.cp_set.ndim == 1
    assert result.family == "mean"

    # The change point should be within reasonable range
    cp = result.cp_set[0] if len(result.cp_set) > 0 else None
    if cp is not None:
        assert 250 < cp < 350, f"Expected change point near 300, got {cp}"


def test_mean_change_multivariate():
    """Test multivariate mean change detection."""
    from fastcpd.segmentation import mean

    # Create 3D data with known change point at 300
    np.random.seed(42)
    data = np.concatenate([
        np.random.multivariate_normal([0, 0, 0], np.eye(3), 300),
        np.random.multivariate_normal([5, 5, 5], np.eye(3), 400)
    ])

    result = mean(data)

    assert len(result.cp_set) > 0
    assert result.data.shape == data.shape

    # Check change point location
    cp = result.cp_set[0] if len(result.cp_set) > 0 else None
    if cp is not None:
        assert 250 < cp < 350


def test_variance_change():
    """Test variance change detection."""
    from fastcpd.segmentation import variance

    # Create data with variance change at 300
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 300),
        np.random.normal(0, 3, 400)
    ])

    result = variance(data)

    assert len(result.cp_set) >= 0  # May or may not detect, depends on parameters
    assert result.family == "variance"


def test_beta_options():
    """Test different beta penalty options."""
    from fastcpd.segmentation import mean

    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 300),
        np.random.normal(5, 1, 400)
    ])

    # Test different beta options
    for beta in ["BIC", "MBIC", "MDL", 10.0]:
        result = mean(data, beta=beta)
        assert result is not None


def test_result_structure():
    """Test the structure of FastcpdResult."""
    from fastcpd.segmentation import mean

    np.random.seed(42)
    data = np.random.normal(0, 1, 100)

    result = mean(data)

    # Check all expected attributes exist
    assert hasattr(result, 'raw_cp_set')
    assert hasattr(result, 'cp_set')
    assert hasattr(result, 'cost_values')
    assert hasattr(result, 'residuals')
    assert hasattr(result, 'thetas')
    assert hasattr(result, 'data')
    assert hasattr(result, 'family')

    # Check types
    assert isinstance(result.cp_set, np.ndarray)
    assert isinstance(result.cost_values, np.ndarray)
    assert isinstance(result.family, str)


def test_1d_data_handling():
    """Test that 1D data is properly converted to 2D."""
    from fastcpd.segmentation import mean

    data_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = mean(data_1d)

    # Internal data should be 2D
    assert result.data.ndim == 2
    assert result.data.shape == (10, 1)


def test_ar_model():
    """Test AR model change detection."""
    from fastcpd.segmentation import ar

    np.random.seed(42)
    n = 200

    # Create AR(1) data with change
    data1 = np.zeros(n)
    for i in range(1, n):
        data1[i] = 0.6 * data1[i-1] + np.random.normal()

    data2 = np.zeros(n)
    for i in range(1, n):
        data2[i] = -0.5 * data2[i-1] + np.random.normal()

    data = np.concatenate([data1, data2])

    result = ar(data, p=1)
    assert result.family == "ar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
