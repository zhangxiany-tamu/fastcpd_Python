#!/usr/bin/env python3
"""
Quickstart examples for fastcpd-python.

This script demonstrates basic usage of the fastcpd package for
various change point detection scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)


def example_univariate_mean_change():
    """Example 1: Univariate mean change detection."""
    print("=" * 60)
    print("Example 1: Univariate Mean Change Detection")
    print("=" * 60)

    from fastcpd.segmentation import mean

    # Generate data with mean changes at positions 300 and 700
    data = np.concatenate([
        np.random.normal(0, 1, 300),    # Mean = 0
        np.random.normal(5, 1, 400),    # Mean = 5
        np.random.normal(2, 1, 300),    # Mean = 2
    ])

    # Detect change points
    result = mean(data)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change points: [300, 700]")
    print(f"Number of segments: {len(result.cp_set) + 1}")

    # Visualize
    result.plot()
    plt.title("Univariate Mean Change Detection")
    plt.savefig("example1_mean_change.png", dpi=150, bbox_inches='tight')
    print("Saved plot to: example1_mean_change.png\n")


def example_multivariate_mean_change():
    """Example 2: Multivariate mean change detection."""
    print("=" * 60)
    print("Example 2: Multivariate Mean Change Detection (3D)")
    print("=" * 60)

    from fastcpd.segmentation import mean

    # Generate 3D data with mean changes
    data = np.concatenate([
        np.random.multivariate_normal([0, 0, 0], np.eye(3), 300),
        np.random.multivariate_normal([5, 5, 5], np.eye(3), 400),
        np.random.multivariate_normal([2, 2, 2], np.eye(3), 300),
    ])

    # Detect change points
    result = mean(data)

    print(f"Data shape: {data.shape}")
    print(f"Detected change points: {result.cp_set}")
    print(f"True change points: [300, 700]")

    # Plot first dimension
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i in range(3):
        axes[i].plot(data[:, i], 'o-', markersize=2)
        for cp in result.cp_set:
            axes[i].axvline(x=cp, color='r', linestyle='--', alpha=0.7)
        axes[i].set_ylabel(f'Dimension {i+1}')
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time')
    plt.suptitle('Multivariate Mean Change Detection')
    plt.tight_layout()
    plt.savefig("example2_multivariate_mean.png", dpi=150, bbox_inches='tight')
    print("Saved plot to: example2_multivariate_mean.png\n")


def example_variance_change():
    """Example 3: Variance change detection."""
    print("=" * 60)
    print("Example 3: Variance Change Detection")
    print("=" * 60)

    from fastcpd.segmentation import variance

    # Generate data with variance changes
    data = np.concatenate([
        np.random.normal(0, 1, 300),    # SD = 1
        np.random.normal(0, 3, 400),    # SD = 3
        np.random.normal(0, 0.5, 300),  # SD = 0.5
    ])

    # Detect change points
    result = variance(data)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change points: [300, 700]")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data, 'o-', markersize=2)
    for cp in result.cp_set:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7, label='Change point' if cp == result.cp_set[0] else '')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Variance Change Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("example3_variance_change.png", dpi=150, bbox_inches='tight')
    print("Saved plot to: example3_variance_change.png\n")


def example_ar_model():
    """Example 4: AR model change detection."""
    print("=" * 60)
    print("Example 4: AR(2) Model Change Detection")
    print("=" * 60)

    from fastcpd.segmentation import ar

    # Generate AR(2) data with parameter change
    n = 300

    # First segment: AR(2) with coefficients [0.6, -0.3]
    data1 = np.zeros(n)
    for i in range(2, n):
        data1[i] = 0.6 * data1[i-1] - 0.3 * data1[i-2] + np.random.normal(0, 0.5)

    # Second segment: AR(2) with coefficients [-0.4, 0.5]
    data2 = np.zeros(n)
    for i in range(2, n):
        data2[i] = -0.4 * data2[i-1] + 0.5 * data2[i-2] + np.random.normal(0, 0.5)

    data = np.concatenate([data1, data2])

    # Detect change points
    result = ar(data, p=2)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change point: [300]")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data, 'o-', markersize=2)
    for cp in result.cp_set:
        plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('AR(2) Model Change Detection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("example4_ar_model.png", dpi=150, bbox_inches='tight')
    print("Saved plot to: example4_ar_model.png\n")


def example_linear_regression():
    """Example 5: Linear regression change detection."""
    print("=" * 60)
    print("Example 5: Linear Regression Change Detection")
    print("=" * 60)

    from fastcpd.segmentation import linear_regression

    # Generate linear regression data with parameter change
    n = 500
    X = np.random.randn(n, 2)

    # First segment: y = 2*x1 + 3*x2 + noise
    y1 = 2 * X[:n//2, 0] + 3 * X[:n//2, 1] + np.random.randn(n//2)

    # Second segment: y = -1*x1 + 5*x2 + noise
    y2 = -1 * X[n//2:, 0] + 5 * X[n//2:, 1] + np.random.randn(n//2)

    y = np.concatenate([y1, y2])

    # Prepare data (first column is response)
    data = np.column_stack([y, X])

    # Detect change points
    result = linear_regression(data)

    print(f"Detected change points: {result.cp_set}")
    print(f"True change point: [250]")
    print(f"Parameter estimates per segment:")
    for i, theta in enumerate(result.thetas):
        if len(theta) > 0:
            print(f"  Segment {i+1}: {theta}")

    # Plot residuals
    plt.figure(figsize=(12, 6))
    if result.residuals.size > 0:
        plt.plot(result.residuals.flatten(), 'o-', markersize=2)
        for cp in result.cp_set:
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Linear Regression Change Detection - Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("example5_linear_regression.png", dpi=150, bbox_inches='tight')
    print("Saved plot to: example5_linear_regression.png\n")


def example_comparison_beta():
    """Example 6: Effect of different beta penalties."""
    print("=" * 60)
    print("Example 6: Comparing Different Beta Penalties")
    print("=" * 60)

    from fastcpd.segmentation import mean

    # Generate data with 2 change points
    data = np.concatenate([
        np.random.normal(0, 1, 300),
        np.random.normal(5, 1, 400),
        np.random.normal(2, 1, 300),
    ])

    # Try different penalties
    penalties = ["BIC", "MBIC", "MDL", 5.0, 20.0]
    results = {}

    for beta in penalties:
        result = mean(data, beta=beta)
        results[str(beta)] = result.cp_set
        beta_str = f"{beta:8s}" if isinstance(beta, str) else f"{beta:8.1f}"
        print(f"Beta = {beta_str}: {len(result.cp_set)} change points at {result.cp_set}")

    print("\nNote: Smaller beta tends to detect more change points")
    print("      Larger beta favors simpler models with fewer change points\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("fastcpd-python Quickstart Examples")
    print("=" * 60 + "\n")

    # Run all examples
    example_univariate_mean_change()
    example_multivariate_mean_change()
    example_variance_change()
    example_ar_model()
    example_linear_regression()
    example_comparison_beta()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
