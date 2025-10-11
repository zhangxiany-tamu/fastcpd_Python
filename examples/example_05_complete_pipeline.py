"""
Example 5: Complete Detection & Evaluation Pipeline
====================================================

Demonstrates end-to-end workflow:
1. Generate synthetic data with known change points
2. Run fastcpd detection
3. Evaluate with comprehensive metrics
4. Visualize results

This is a copy-paste ready template for your own analyses.
"""

import numpy as np
from fastcpd import fastcpd
from fastcpd.datasets import make_mean_change, make_glm_change, make_garch_change
from fastcpd.metrics import evaluate_all, covering_metric
from fastcpd.visualization import plot_detection, plot_metric_comparison

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Complete Change Point Detection Pipeline")
print("=" * 80)
print()

# ============================================================================
# Example 1: Mean Change Detection
# ============================================================================
print("Example 1: Mean Change Detection")
print("-" * 80)

# Step 1: Generate data
data_dict = make_mean_change(
    n_samples=500,
    n_changepoints=3,
    noise_std=1.0,
    change_type='jump',
    seed=42
)

print(f"Generated data:")
print(f"  n_samples: {len(data_dict['data'])}")
print(f"  True change points: {data_dict['changepoints']}")
print(f"  SNR: {data_dict['metadata']['snr_db']:.2f} dB")
print()

# Step 2: Run detection
result = fastcpd(data_dict['data'], family='mean', beta='MBIC')

print(f"Detection results:")
print(f"  Detected change points: {result.cp_set.tolist()}")
print()

# Step 3: Evaluate
metrics = evaluate_all(
    true_cps=data_dict['changepoints'],
    pred_cps=result.cp_set.tolist(),
    n_samples=len(data_dict['data']),
    margin=10
)

print(f"Evaluation metrics:")
print(f"  Precision: {metrics['point_metrics']['precision']:.3f}")
print(f"  Recall: {metrics['point_metrics']['recall']:.3f}")
print(f"  F1 Score: {metrics['point_metrics']['f1_score']:.3f}")
print(f"  Hausdorff: {metrics['distance_metrics']['hausdorff']:.1f}")
print(f"  ARI: {metrics['segmentation_metrics']['adjusted_rand_index']:.3f}")
print()

# Step 4: Visualize
import matplotlib.pyplot as plt
fig, axes = plot_detection(
    data=data_dict['data'],
    true_cps=data_dict['changepoints'],
    pred_cps=result.cp_set.tolist(),
    metric_result=metrics,
    title="Mean Change Detection"
)
plt.savefig('mean_detection_result.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: mean_detection_result.png")
print()

# ============================================================================
# Example 2: GLM (Binomial) Detection
# ============================================================================
print("Example 2: GLM (Binomial) Detection")
print("-" * 80)

# Generate GLM data
data_dict = make_glm_change(
    n_samples=400,
    n_changepoints=2,
    n_features=3,
    family='binomial',
    seed=42
)

print(f"Generated GLM data:")
print(f"  True change points: {data_dict['changepoints']}")
print(f"  Family: {data_dict['metadata']['family']}")
print()

# Detect
result = fastcpd(data_dict['data'], family='binomial', beta='MBIC')

print(f"Detection results:")
print(f"  Detected change points: {result.cp_set.tolist()}")
print()

# Evaluate
metrics = evaluate_all(
    true_cps=data_dict['changepoints'],
    pred_cps=result.cp_set.tolist(),
    n_samples=len(data_dict['data']),
    margin=10
)

print(f"Evaluation summary:")
print(f"  F1 Score: {metrics['point_metrics']['f1_score']:.3f}")
print(f"  Annotation Error: {metrics['distance_metrics']['annotation_error_mae']:.2f}")
print()

# ============================================================================
# Example 3: Algorithm Comparison
# ============================================================================
print("Example 3: Comparing Multiple Detection Methods")
print("-" * 80)

# Generate data
data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=123)
true_cps = data_dict['changepoints']
data = data_dict['data']

# Run multiple detectors with different parameters
result_pelt = fastcpd(data, family='mean', beta='MBIC', vanilla_percentage=1.0)
result_segd = fastcpd(data, family='mean', beta='MBIC', vanilla_percentage=0.0)
result_hybrid = fastcpd(data, family='mean', beta='MBIC', vanilla_percentage=0.5)

# Evaluate each
n_samples = len(data)
results_dict = {}

for name, result in [('PELT', result_pelt), ('SeGD', result_segd), ('Hybrid', result_hybrid)]:
    metrics = evaluate_all(true_cps, result.cp_set.tolist(), n_samples, margin=10)
    results_dict[name] = metrics
    print(f"{name:8s}: F1={metrics['point_metrics']['f1_score']:.3f}, "
          f"Precision={metrics['point_metrics']['precision']:.3f}, "
          f"Recall={metrics['point_metrics']['recall']:.3f}")

print()

# Visualize comparison
fig, ax = plot_metric_comparison(
    results_dict,
    metrics=['precision', 'recall', 'f1_score'],
    figsize=(10, 6)
)
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison saved to: algorithm_comparison.png")
print()

# ============================================================================
# Example 4: Multi-Annotator Scenario
# ============================================================================
print("Example 4: Multi-Annotator Evaluation")
print("-" * 80)

from fastcpd.datasets import add_annotation_noise
from fastcpd.visualization import plot_annotators

# True change points
true_cps = [100, 200, 300]

# Simulate 5 human annotators
annotators = add_annotation_noise(
    true_changepoints=true_cps,
    n_annotators=5,
    noise_std=5.0,
    agreement_rate=0.8,
    seed=42
)

print("Annotators' change points:")
for i, ann_cps in enumerate(annotators, 1):
    print(f"  Annotator {i}: {ann_cps}")
print()

# Algorithm prediction
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100),
    np.random.normal(2, 1, 100),
    np.random.normal(-2, 1, 100)
])
result = fastcpd(data, family='mean', beta='MBIC')
pred_cps = result.cp_set.tolist()

print(f"Algorithm detected: {pred_cps}")
print()

# Evaluate with covering metric
covering_result = covering_metric(annotators, pred_cps, margin=10)

print(f"Covering metric results:")
print(f"  Covering score: {covering_result['covering_score']:.3f}")
print(f"  Recall per annotator: {[f'{r:.2f}' for r in covering_result['recall_per_annotator']]}")
print(f"  Std recall: {covering_result['std_recall']:.3f}")
print()

# Visualize
fig, ax = plot_annotators(data, annotators, pred_cps)
plt.savefig('multi_annotator_result.png', dpi=300, bbox_inches='tight')
print("Multi-annotator plot saved to: multi_annotator_result.png")
print()

# ============================================================================
# Example 5: Time Series (GARCH)
# ============================================================================
print("Example 5: GARCH Volatility Detection")
print("-" * 80)

# Generate GARCH data
data_dict = make_garch_change(
    n_samples=600,
    n_changepoints=2,
    volatility_regimes=['low', 'high', 'low'],
    seed=42
)

print(f"Generated GARCH data:")
print(f"  True change points: {data_dict['changepoints']}")
print(f"  Volatility regimes: {data_dict['metadata']['volatility_regimes']}")
print(f"  Avg volatility: {[f'{v:.3f}' for v in data_dict['metadata']['avg_volatility_per_segment']]}")
print()

# Detect (GARCH uses vanilla PELT)
result = fastcpd(data_dict['data'], family='garch', order=[1, 1], beta=2.0)

print(f"Detection results:")
print(f"  Detected change points: {result.cp_set.tolist()}")
print()

# Evaluate
metrics = evaluate_all(
    true_cps=data_dict['changepoints'],
    pred_cps=result.cp_set.tolist(),
    n_samples=len(data_dict['data']),
    margin=10
)

print(f"Evaluation:")
print(f"  F1 Score: {metrics['point_metrics']['f1_score']:.3f}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("Pipeline Summary")
print("=" * 80)
print()
print("✅ Successfully demonstrated:")
print("  1. Mean change detection with visualization")
print("  2. GLM (binomial) detection")
print("  3. Algorithm comparison (PELT vs SeGD vs Hybrid)")
print("  4. Multi-annotator evaluation with covering metric")
print("  5. GARCH volatility detection")
print()
print("✅ Files created:")
print("  - mean_detection_result.png")
print("  - algorithm_comparison.png")
print("  - multi_annotator_result.png")
print()
print("📚 Key takeaways:")
print("  - Use evaluate_all() for comprehensive metrics")
print("  - Compare multiple algorithms for robustness")
print("  - Covering metric for multi-annotator scenarios")
print("  - Visualizations reveal patterns metrics might miss")
print()
print("🚀 Next steps:")
print("  - Adapt this pipeline to your data")
print("  - Tune beta parameter for your application")
print("  - Choose appropriate margin for evaluation")
print()
print("=" * 80)

# Optional: Show all plots
plt.show()
