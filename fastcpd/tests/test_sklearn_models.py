"""Tests for GLM and LASSO models using scikit-learn."""

import numpy as np
import pytest


def test_logistic_regression():
    """Test logistic regression change detection."""
    from fastcpd.segmentation import logistic_regression

    np.random.seed(42)
    n = 500

    # Generate logistic regression data with change
    X = np.random.randn(n, 2)

    # First segment: strong positive coefficients
    prob1 = 1 / (1 + np.exp(-(2*X[:n//2, 0] + 3*X[:n//2, 1])))
    y1 = (np.random.rand(n//2) < prob1).astype(float)

    # Second segment: different coefficients
    prob2 = 1 / (1 + np.exp(-(-1*X[n//2:, 0] + 2*X[n//2:, 1])))
    y2 = (np.random.rand(n//2) < prob2).astype(float)

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Detect change points
    result = logistic_regression(data)

    assert result.family == "binomial"
    assert len(result.cp_set) > 0

    # Check that detected change point is roughly at n//2
    if len(result.cp_set) > 0:
        cp = result.cp_set[0]
        assert 200 < cp < 300, f"Expected change point near 250, got {cp}"


def test_poisson_regression():
    """Test Poisson regression change detection."""
    from fastcpd.segmentation import poisson_regression

    np.random.seed(42)
    n = 500

    # Generate Poisson regression data with change
    X = np.random.randn(n, 2)

    # First segment
    lambda1 = np.exp(0.5*X[:n//2, 0] + 0.8*X[:n//2, 1])
    y1 = np.random.poisson(lambda1)

    # Second segment with different coefficients
    lambda2 = np.exp(-0.3*X[n//2:, 0] + 1.2*X[n//2:, 1])
    y2 = np.random.poisson(lambda2)

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Detect change points
    result = poisson_regression(data)

    assert result.family == "poisson"
    assert len(result.cp_set) > 0

    # Check change point location
    if len(result.cp_set) > 0:
        cp = result.cp_set[0]
        assert 200 < cp < 300, f"Expected change point near 250, got {cp}"


def test_lasso():
    """Test LASSO regression change detection."""
    from fastcpd.segmentation import lasso

    np.random.seed(42)
    n = 500
    p = 20  # High-dimensional

    # Generate sparse regression data with change
    X = np.random.randn(n, p)

    # First segment: only first 3 features matter
    y1 = 2*X[:n//2, 0] + 3*X[:n//2, 1] - 1.5*X[:n//2, 2] + np.random.randn(n//2)*0.5

    # Second segment: different sparse coefficients
    y2 = -1*X[n//2:, 5] + 2*X[n//2:, 8] + np.random.randn(n//2)*0.5

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Detect change points with LASSO
    result = lasso(data, alpha=0.1)

    assert result.family == "lasso"
    assert len(result.cp_set) > 0

    # Check change point location
    if len(result.cp_set) > 0:
        cp = result.cp_set[0]
        assert 200 < cp < 300, f"Expected change point near 250, got {cp}"


def test_lasso_cv():
    """Test LASSO with cross-validation."""
    from fastcpd.segmentation import lasso

    np.random.seed(42)
    n = 300
    p = 10

    # Generate data
    X = np.random.randn(n, p)
    y1 = X[:n//2, 0] + X[:n//2, 1] + np.random.randn(n//2)*0.3
    y2 = -X[n//2:, 3] + X[n//2:, 4] + np.random.randn(n//2)*0.3

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Use cross-validation for alpha selection
    result = lasso(data, cv=True)

    assert result.family == "lasso"
    # CV version may find different number of change points


def test_model_result_structure():
    """Test that model results have correct structure."""
    from fastcpd.models import fit_logistic_regression, fit_poisson_regression, fit_lasso

    np.random.seed(42)
    n = 100

    # Test logistic regression
    X = np.random.randn(n, 2)
    y_binary = (np.random.rand(n) > 0.5).astype(float)

    result = fit_logistic_regression(X, y_binary)
    assert hasattr(result, 'coefficients')
    assert hasattr(result, 'residuals')
    assert hasattr(result, 'deviance')
    assert hasattr(result, 'converged')
    assert result.coefficients.shape == (2,)
    assert result.residuals.shape == (n,)

    # Test Poisson regression
    y_count = np.random.poisson(3, size=n)

    result = fit_poisson_regression(X, y_count)
    assert result.coefficients.shape == (2,)
    assert result.residuals.shape == (n,)

    # Test LASSO
    y_continuous = X @ np.array([1.0, 2.0]) + np.random.randn(n)*0.1

    result = fit_lasso(X, y_continuous, alpha=0.1)
    assert result.coefficients.shape == (2,)
    assert result.residuals.shape == (n,)


def test_warm_start():
    """Test warm start functionality."""
    from fastcpd.models import fit_logistic_regression

    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = (np.random.rand(n) > 0.5).astype(float)

    # First fit without warm start
    result1 = fit_logistic_regression(X, y)

    # Second fit with warm start
    result2 = fit_logistic_regression(X, y, warm_start_coef=result1.coefficients)

    # Coefficients should be similar (same data, same model)
    np.testing.assert_allclose(result1.coefficients, result2.coefficients, rtol=1e-3)


def test_sklearn_vs_ols():
    """Test that OLS gives same results as linear regression."""
    from fastcpd.models import fit_linear_regression

    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    true_coef = np.array([2.0, -1.5])
    y = X @ true_coef + np.random.randn(n)*0.1

    result = fit_linear_regression(X, y)

    # Check coefficients are close to true values
    np.testing.assert_allclose(result.coefficients, true_coef, atol=0.3)

    # Check residuals
    y_pred = X @ result.coefficients
    expected_residuals = y - y_pred
    np.testing.assert_allclose(result.residuals, expected_residuals, rtol=1e-6)


def test_high_dimensional_lasso():
    """Test LASSO in high-dimensional setting (p > n)."""
    from fastcpd.segmentation import lasso

    np.random.seed(42)
    n = 100
    p = 200  # p > n

    X = np.random.randn(n, p)
    # Only 5 features are truly relevant
    true_coef = np.zeros(p)
    true_coef[[0, 10, 50, 100, 150]] = [2, -1.5, 3, -2, 1]

    y_seg1 = X[:n//2, :] @ true_coef + np.random.randn(n//2)*0.5

    # Different sparse pattern in second segment
    true_coef2 = np.zeros(p)
    true_coef2[[5, 15, 60, 110, 160]] = [-2, 1.5, -3, 2, -1]
    y_seg2 = X[n//2:, :] @ true_coef2 + np.random.randn(n//2)*0.5

    y = np.concatenate([y_seg1, y_seg2])
    data = np.column_stack([y, X])

    # Should still work with LASSO regularization
    result = lasso(data, alpha=0.5)

    assert result.family == "lasso"
    # With strong regularization, may detect change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
