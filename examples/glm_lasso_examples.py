#!/usr/bin/env python3
"""
Examples for GLM and LASSO models using scikit-learn.

These examples demonstrate how fastcpd uses scikit-learn's highly optimized
implementations for GLM and LASSO models, which are often faster than
equivalent implementations in R or other languages.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def example_logistic_regression():
    """Example: Logistic regression change detection."""
    print("=" * 60)
    print("Example: Logistic Regression (Binary Classification)")
    print("=" * 60)

    from fastcpd.segmentation import logistic_regression

    # Generate data with change in logistic regression parameters
    n = 600
    X = np.random.randn(n, 2)

    # Segment 1: Strong positive relationship
    linear_pred1 = 2*X[:n//3, 0] + 3*X[:n//3, 1]
    prob1 = 1 / (1 + np.exp(-linear_pred1))
    y1 = (np.random.rand(n//3) < prob1).astype(float)

    # Segment 2: Weak negative relationship
    linear_pred2 = -0.5*X[n//3:2*n//3, 0] + 0.8*X[n//3:2*n//3, 1]
    prob2 = 1 / (1 + np.exp(-linear_pred2))
    y2 = (np.random.rand(n//3) < prob2).astype(float)

    # Segment 3: Strong negative relationship
    linear_pred3 = -2*X[2*n//3:, 0] - 1.5*X[2*n//3:, 1]
    prob3 = 1 / (1 + np.exp(-linear_pred3))
    y3 = (np.random.rand(n//3) < prob3).astype(float)

    y = np.concatenate([y1, y2, y3])
    data = np.column_stack([y, X])

    # Detect change points
    print("Using scikit-learn's LogisticRegression (Cython/LBFGS solver)...")
    result = logistic_regression(data)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change points: [200, 400]")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot response
    axes[0].scatter(range(n), y, c=y, cmap='coolwarm', alpha=0.5, s=10)
    for cp in result.cp_set:
        axes[0].axvline(x=cp, color='r', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Response (0/1)')
    axes[0].set_title('Logistic Regression Change Detection')

    # Plot predictor 1
    axes[1].scatter(range(n), X[:, 0], c=y, cmap='coolwarm', alpha=0.5, s=10)
    for cp in result.cp_set:
        axes[1].axvline(x=cp, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Predictor 1')

    # Plot predictor 2
    axes[2].scatter(range(n), X[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
    for cp in result.cp_set:
        axes[2].axvline(x=cp, color='r', linestyle='--', linewidth=2)
    axes[2].set_ylabel('Predictor 2')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig("glm_logistic.png", dpi=150)
    print("Saved: glm_logistic.png\n")


def example_poisson_regression():
    """Example: Poisson regression for count data."""
    print("=" * 60)
    print("Example: Poisson Regression (Count Data)")
    print("=" * 60)

    from fastcpd.segmentation import poisson_regression

    # Generate count data with change
    n = 500
    X = np.random.randn(n, 2)

    # Segment 1: Low counts
    log_lambda1 = 0.3*X[:n//2, 0] + 0.5*X[:n//2, 1]
    y1 = np.random.poisson(np.exp(log_lambda1))

    # Segment 2: High counts
    log_lambda2 = 1.5*X[n//2:, 0] + 1.2*X[n//2:, 1]
    y2 = np.random.poisson(np.exp(log_lambda2))

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Detect change points
    print("Using scikit-learn's PoissonRegressor (LBFGS solver)...")
    result = poisson_regression(data)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change point: [250]")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(range(n), y, alpha=0.5, s=10)
    for cp in result.cp_set:
        plt.axvline(x=cp, color='r', linestyle='--', linewidth=2, label='Change point')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Poisson Regression Change Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig("glm_poisson.png", dpi=150)
    print("Saved: glm_poisson.png\n")


def example_lasso():
    """Example: LASSO for sparse regression with change."""
    print("=" * 60)
    print("Example: LASSO (Sparse High-Dimensional Regression)")
    print("=" * 60)

    from fastcpd.segmentation import lasso

    # Generate high-dimensional sparse data
    n = 500
    p = 50  # Many features, but only a few are relevant

    X = np.random.randn(n, p)

    # Segment 1: Only features [0, 5, 10] are active
    true_coef1 = np.zeros(p)
    true_coef1[[0, 5, 10]] = [3, -2, 1.5]
    y1 = X[:n//2, :] @ true_coef1 + np.random.randn(n//2)*0.5

    # Segment 2: Different sparse pattern - features [15, 25, 35]
    true_coef2 = np.zeros(p)
    true_coef2[[15, 25, 35]] = [-2.5, 3, -1]
    y2 = X[n//2:, :] @ true_coef2 + np.random.randn(n//2)*0.5

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Detect change with LASSO
    print("Using scikit-learn's Lasso (coordinate descent solver)...")
    result = lasso(data, alpha=0.2)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change point: [250]")
    print(f"Number of features: {p}")

    # Show which features are selected in each segment
    print("\nSparse coefficient patterns:")
    for i, theta in enumerate(result.thetas):
        if len(theta) > 0:
            active = np.where(np.abs(theta) > 0.1)[0]
            print(f"  Segment {i+1}: Active features {active.tolist()}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot response
    axes[0].plot(y, 'o-', markersize=2)
    for cp in result.cp_set:
        axes[0].axvline(x=cp, color='r', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Response')
    axes[0].set_title('LASSO Regression Change Detection')

    # Plot coefficient heatmap
    if result.thetas.size > 0:
        im = axes[1].imshow(result.thetas.T, aspect='auto', cmap='RdBu_r',
                           vmin=-3, vmax=3)
        axes[1].set_xlabel('Segment')
        axes[1].set_ylabel('Feature Index')
        axes[1].set_title('Sparse Coefficients per Segment')
        plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig("lasso_sparse.png", dpi=150)
    print("\nSaved: lasso_sparse.png\n")


def example_lasso_cv():
    """Example: LASSO with automatic cross-validation."""
    print("=" * 60)
    print("Example: LASSO with Cross-Validation (Automatic α)")
    print("=" * 60)

    from fastcpd.segmentation import lasso

    # Generate data
    n = 400
    p = 30
    X = np.random.randn(n, p)

    # Change in sparse coefficients
    true_coef1 = np.zeros(p)
    true_coef1[[0, 3, 7]] = [2, -1.5, 1]
    y1 = X[:n//2, :] @ true_coef1 + np.random.randn(n//2)*0.3

    true_coef2 = np.zeros(p)
    true_coef2[[10, 15, 20]] = [-2, 1.8, -1.2]
    y2 = X[n//2:, :] @ true_coef2 + np.random.randn(n//2)*0.3

    y = np.concatenate([y1, y2])
    data = np.column_stack([y, X])

    # Use cross-validation to select alpha automatically
    print("Using LassoCV with 5-fold cross-validation...")
    result = lasso(data, cv=True)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change point: [200]")
    print("Note: α (regularization strength) selected automatically via CV")

    plt.figure(figsize=(12, 6))
    plt.plot(y, 'o-', markersize=2)
    for cp in result.cp_set:
        plt.axvline(x=cp, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title('LASSO-CV Change Detection (Automatic Regularization)')
    plt.tight_layout()
    plt.savefig("lasso_cv.png", dpi=150)
    print("Saved: lasso_cv.png\n")


def compare_speed():
    """Compare speed of scikit-learn implementations."""
    print("=" * 60)
    print("Speed Comparison: scikit-learn vs Pure Python")
    print("=" * 60)

    import time
    from fastcpd.segmentation import logistic_regression, lasso

    n = 1000
    p_glm = 5
    p_lasso = 50

    # Logistic regression data
    X_glm = np.random.randn(n, p_glm)
    y_binary = (np.random.rand(n) > 0.5).astype(float)
    data_glm = np.column_stack([y_binary, X_glm])

    # LASSO data
    X_lasso = np.random.randn(n, p_lasso)
    y_continuous = X_lasso @ np.random.randn(p_lasso) + np.random.randn(n)
    data_lasso = np.column_stack([y_continuous, X_lasso])

    # Time logistic regression
    start = time.perf_counter()
    result_glm = logistic_regression(data_glm)
    time_glm = time.perf_counter() - start

    # Time LASSO
    start = time.perf_counter()
    result_lasso = lasso(data_lasso, alpha=0.1)
    time_lasso = time.perf_counter() - start

    print(f"\nLogistic Regression (n={n}, p={p_glm}): {time_glm:.4f}s")
    print(f"LASSO (n={n}, p={p_lasso}): {time_lasso:.4f}s")

    print("\nNote: These use scikit-learn's highly optimized Cython implementations:")
    print("  - LogisticRegression: liblinear/LBFGS (C/C++)")
    print("  - Lasso: Coordinate descent (Cython)")
    print("  - Often faster than R's glm.fit or glmnet for these models\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GLM and LASSO Examples with scikit-learn")
    print("=" * 60 + "\n")

    example_logistic_regression()
    example_poisson_regression()
    example_lasso()
    example_lasso_cv()
    compare_speed()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
