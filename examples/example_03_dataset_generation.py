"""
Example 3: Synthetic Dataset Generation
========================================

Demonstrates comprehensive dataset generation for benchmarking and testing.
Includes UNIQUE GLM and GARCH datasets not available in ruptures!
"""

import numpy as np
import matplotlib.pyplot as plt
from fastcpd.datasets import (
    make_mean_change,
    make_variance_change,
    make_regression_change,
    make_arma_change,
    make_glm_change,
    make_garch_change
)

print("=" * 70)
print("Example 3: Synthetic Dataset Generation")
print("=" * 70)
print()

# ============================================================================
# Dataset 1: Mean Change
# ============================================================================
print("Dataset 1: Mean Change")
print("-" * 70)

data_dict = make_mean_change(
    n_samples=500,
    n_changepoints=3,
    n_dim=1,
    noise_std=1.0,
    change_type='jump',
    seed=42
)

print(f"Data shape:       {data_dict['data'].shape}")
print(f"True CPs:         {data_dict['changepoints']}")
true_means_str = [f'{m:.2f}' if isinstance(m, (int, float)) else str(m) for m in data_dict['true_means']]
print(f"True means:       {true_means_str}")
print(f"SNR:              {data_dict['metadata']['snr_db']:.2f} dB")
print(f"Difficulty:       {data_dict['metadata']['difficulty']:.3f} (0=easy, 1=hard)")
print(f"Segment lengths:  {data_dict['metadata']['segment_lengths']}")
print()

# ============================================================================
# Dataset 2: Variance Change
# ============================================================================
print("Dataset 2: Variance Change")
print("-" * 70)

data_dict = make_variance_change(
    n_samples=500,
    n_changepoints=3,
    variance_ratios=[1.0, 4.0, 0.5, 2.0],  # Different variances per segment
    base_var=1.0,
    seed=42
)

print(f"Data shape:          {data_dict['data'].shape}")
print(f"True CPs:            {data_dict['changepoints']}")
print(f"True variances:      {[f'{v:.2f}' for v in data_dict['true_variances']]}")
print(f"Variance ratios:     {[f'{r:.2f}' for r in data_dict['metadata']['variance_ratios']]}")
print(f"Kurtosis per segment:{[f'{k:.2f}' for k in data_dict['metadata']['kurtosis_per_segment']]}")
print()

# ============================================================================
# Dataset 3: Linear Regression Change
# ============================================================================
print("Dataset 3: Linear Regression with Coefficient Changes")
print("-" * 70)

data_dict = make_regression_change(
    n_samples=500,
    n_changepoints=3,
    n_features=5,
    coef_changes='random',
    noise_std=0.5,
    correlation=0.3,
    seed=42
)

print(f"Data shape:      {data_dict['data'].shape} (y + X)")
print(f"X shape:         {data_dict['X'].shape}")
print(f"y shape:         {data_dict['y'].shape}")
print(f"True CPs:        {data_dict['changepoints']}")
print(f"R² per segment:  {[f'{r:.3f}' for r in data_dict['metadata']['r_squared_per_segment']]}")
print(f"Condition number:{data_dict['metadata']['condition_number']:.2f}")
print(f"Effect size:     {data_dict['metadata']['effect_size']:.3f}")
print()
print("True coefficients per segment:")
for i, coefs in enumerate(data_dict['true_coefs']):
    print(f"  Segment {i+1}: {[f'{c:.2f}' for c in coefs]}")
print()

# ============================================================================
# Dataset 4: ARMA Time Series Change
# ============================================================================
print("Dataset 4: ARMA Time Series with Parameter Changes")
print("-" * 70)

data_dict = make_arma_change(
    n_samples=500,
    n_changepoints=3,
    orders=[(1, 1), (2, 0), (0, 2), (1, 1)],  # Different ARMA orders
    sigma_change=True,
    innovation='normal',
    seed=42
)

print(f"Data shape:     {data_dict['data'].shape}")
print(f"True CPs:       {data_dict['changepoints']}")
print(f"ARMA orders:    {data_dict['metadata']['orders']}")
print(f"Stationary:     {data_dict['metadata']['is_stationary']}")
print(f"Invertible:     {data_dict['metadata']['is_invertible']}")
print()
print("Parameters per segment:")
for i, params in enumerate(data_dict['true_params']):
    print(f"  Segment {i+1}:")
    print(f"    AR:  {params['ar']}")
    print(f"    MA:  {params['ma']}")
    print(f"    σ:   {params['sigma']:.3f}")
print()

# ============================================================================
# Dataset 5: GLM Change (UNIQUE! Not in ruptures!)
# ============================================================================
print("Dataset 5: GLM with Coefficient Changes (BINOMIAL)")
print("-" * 70)
print("🌟 UNIQUE to fastcpd! Not available in ruptures!")
print()

data_dict = make_glm_change(
    n_samples=500,
    n_changepoints=3,
    n_features=3,
    family='binomial',
    trials=1,  # Logistic regression
    seed=42
)

print(f"Data shape:              {data_dict['data'].shape}")
print(f"True CPs:                {data_dict['changepoints']}")
print(f"Family:                  {data_dict['metadata']['family']}")
print(f"Separation (AUC) per seg:{[f'{s:.3f}' if s is not None else 'N/A' for s in data_dict['metadata']['separation_per_segment']]}")
print(f"Response range:          [{data_dict['y'].min()}, {data_dict['y'].max()}]")
print()

# Poisson GLM
data_dict_poisson = make_glm_change(
    n_samples=500,
    n_changepoints=2,
    n_features=3,
    family='poisson',
    seed=42
)

print("GLM with Coefficient Changes (POISSON)")
print(f"Family:                  {data_dict_poisson['metadata']['family']}")
print(f"Overdispersion per seg:  {[f'{o:.3f}' for o in data_dict_poisson['metadata']['overdispersion_per_segment']]}")
print(f"Response range:          [{data_dict_poisson['y'].min()}, {data_dict_poisson['y'].max()}]")
print()

# ============================================================================
# Dataset 6: GARCH Change (UNIQUE! Not in ruptures!)
# ============================================================================
print("Dataset 6: GARCH Volatility Regime Changes")
print("-" * 70)
print("🌟 UNIQUE to fastcpd! Not available in ruptures!")
print()

data_dict = make_garch_change(
    n_samples=600,
    n_changepoints=2,
    orders=[(1, 1), (1, 1), (1, 1)],
    volatility_regimes=['low', 'high', 'low'],
    seed=42
)

print(f"Data shape:             {data_dict['data'].shape}")
print(f"True CPs:               {data_dict['changepoints']}")
print(f"Volatility regimes:     {data_dict['metadata']['volatility_regimes']}")
print(f"Avg vol per segment:    {[f'{v:.4f}' for v in data_dict['metadata']['avg_volatility_per_segment']]}")
print(f"Volatility ratios:      {[f'{r:.2f}' for r in data_dict['metadata']['volatility_ratios']]}")
print(f"Kurtosis per segment:   {[f'{k:.2f}' for k in data_dict['metadata']['kurtosis_per_segment']]}")
print()
print("GARCH parameters per segment:")
for i, params in enumerate(data_dict['true_params']):
    print(f"  Segment {i+1} ({params['regime']}):")
    print(f"    ω (omega): {params['omega']:.4f}")
    print(f"    α (alpha): {params['alpha']}")
    print(f"    β (beta):  {params['beta']}")
    persistence = sum(params['alpha']) + sum(params['beta'])
    print(f"    Persistence: {persistence:.4f} (< 1 for stationarity)")
print()

# ============================================================================
# Visualization Example
# ============================================================================
print("=" * 70)
print("VISUALIZATION EXAMPLE")
print("=" * 70)
print()

# Generate data for visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Synthetic Dataset Generation Examples', fontsize=16)

# 1. Mean change
data_dict = make_mean_change(n_samples=300, n_changepoints=3, seed=42)
axes[0, 0].plot(data_dict['data'], linewidth=0.8, alpha=0.7)
for cp in data_dict['changepoints']:
    axes[0, 0].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Mean Change')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# 2. Variance change
data_dict = make_variance_change(n_samples=300, n_changepoints=3, seed=42)
axes[0, 1].plot(data_dict['data'], linewidth=0.8, alpha=0.7)
for cp in data_dict['changepoints']:
    axes[0, 1].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title('Variance Change')
axes[0, 1].set_ylabel('Value')
axes[0, 1].grid(True, alpha=0.3)

# 3. Regression
data_dict = make_regression_change(n_samples=300, n_changepoints=2, n_features=3, seed=42)
axes[1, 0].plot(data_dict['y'], linewidth=0.8, alpha=0.7)
for cp in data_dict['changepoints']:
    axes[1, 0].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Linear Regression')
axes[1, 0].set_ylabel('Response')
axes[1, 0].grid(True, alpha=0.3)

# 4. ARMA
data_dict = make_arma_change(n_samples=300, n_changepoints=2, seed=42)
axes[1, 1].plot(data_dict['data'], linewidth=0.8, alpha=0.7)
for cp in data_dict['changepoints']:
    axes[1, 1].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('ARMA Time Series')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3)

# 5. GLM (Binomial)
data_dict = make_glm_change(n_samples=300, n_changepoints=2, family='binomial', seed=42)
axes[2, 0].scatter(range(len(data_dict['y'])), data_dict['y'], s=5, alpha=0.5)
for cp in data_dict['changepoints']:
    axes[2, 0].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[2, 0].set_title('GLM (Binomial) 🌟')
axes[2, 0].set_ylabel('Response')
axes[2, 0].set_xlabel('Sample')
axes[2, 0].grid(True, alpha=0.3)

# 6. GARCH
data_dict = make_garch_change(n_samples=300, n_changepoints=2, seed=42)
axes[2, 1].plot(data_dict['data'], linewidth=0.8, alpha=0.7, label='Returns')
axes[2, 1].plot(data_dict['volatility'], linewidth=1.5, alpha=0.8, color='orange', label='Volatility')
for cp in data_dict['changepoints']:
    axes[2, 1].axvline(cp, color='red', linestyle='--', linewidth=2)
axes[2, 1].set_title('GARCH (Volatility) 🌟')
axes[2, 1].set_ylabel('Value')
axes[2, 1].set_xlabel('Sample')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/xianyangzhang/My Drive/fastcpd-python/examples/datasets_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: examples/datasets_visualization.png")
print()

# ============================================================================
# Key Takeaways
# ============================================================================
print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print()
print("1. All datasets return rich metadata (SNR, R², AUC, kurtosis, etc.)")
print("2. Reproducible with seed parameter")
print("3. GLM datasets: UNIQUE to fastcpd! 🌟")
print("4. GARCH datasets: UNIQUE to fastcpd! 🌟")
print("5. Perfect for benchmarking and testing algorithms")
print()
print("Dataset Comparison:")
print("  fastcpd:  7 generators (mean, variance, regression, ARMA, GLM, GARCH, +multi-annotator)")
print("  ruptures: 4 generators (constant, linear, normal, wavy)")
print()
print("Unique Features:")
print("  ✅ GLM data generation (binomial, poisson)")
print("  ✅ GARCH volatility regimes")
print("  ✅ Rich metadata (SNR, R², stationarity, etc.)")
print("  ✅ Custom coefficient changes (random, sign_flip, magnitude)")
print("  ✅ Correlation control for covariates")
print()
