"""
Example 2: Multi-Annotator Evaluation
======================================

Demonstrates the covering metric for multiple annotators.

Use case: When multiple human annotators provide ground truth labels,
we want to know how well the algorithm agrees with ALL of them, not just
their union or average.
"""

import numpy as np
from fastcpd.metrics import covering_metric, evaluate_all
from fastcpd.datasets import add_annotation_noise

print("=" * 70)
print("Example 2: Multi-Annotator Evaluation")
print("=" * 70)
print()

# ============================================================================
# Scenario 1: Simulating Multiple Annotators
# ============================================================================
print("Scenario 1: Simulating Realistic Annotator Disagreement")
print("-" * 70)

# Ground truth
true_cps = [100, 200, 300]
print(f"Ground truth CPs: {true_cps}")
print()

# Simulate 5 annotators with varying agreement
annotators = add_annotation_noise(
    true_cps,
    n_annotators=5,
    noise_std=8.0,      # 8 samples std deviation
    agreement_rate=0.8,  # 80% chance each annotator includes each CP
    seed=42
)

print("Annotator CPs:")
for i, ann_cps in enumerate(annotators, 1):
    print(f"  Annotator {i}: {ann_cps}")
print()

# ============================================================================
# Scenario 2: Evaluating Against Multiple Annotators
# ============================================================================
print("Scenario 2: Evaluating Detection Against All Annotators")
print("-" * 70)

# Algorithm's prediction
pred_cps = [98, 202, 295]
print(f"Algorithm's prediction: {pred_cps}")
print()

# Calculate covering metric
result = covering_metric(annotators, pred_cps, margin=10, n_samples=400)

print("Covering Metric Results:")
print(f"  Covering Score:     {result['covering_score']:.4f} (mean recall across annotators)")
print(f"  Mean Recall:        {result['mean_recall']:.4f}")
print(f"  Std Recall:         {result['std_recall']:.4f}")
print(f"  Min Recall:         {result['min_recall']:.4f}")
print(f"  Max Recall:         {result['max_recall']:.4f}")
print()

print("Recall per Annotator:")
for i, recall in enumerate(result['recall_per_annotator'], 1):
    print(f"  Annotator {i}: {recall:.4f}")
print()

print("Interpretation:")
print(f"  The algorithm achieves {result['covering_score']:.1%} recall on average")
print(f"  across all {result['n_annotators']} annotators.")
print(f"  Std = {result['std_recall']:.4f} indicates {'consistent' if result['std_recall'] < 0.1 else 'variable'} performance.")
print()

# ============================================================================
# Scenario 3: Comparison with Union-based Evaluation
# ============================================================================
print("Scenario 3: Covering vs Union-based Evaluation")
print("-" * 70)

# Union of all annotators (traditional approach)
union_cps = sorted(set(cp for ann in annotators for cp in ann))
print(f"Union of all annotators: {union_cps}")
print()

# Evaluate against union
from fastcpd.metrics import precision_recall

union_result = precision_recall(union_cps, pred_cps, margin=10)
print("Union-based Evaluation:")
print(f"  Precision: {union_result['precision']:.4f}")
print(f"  Recall:    {union_result['recall']:.4f}")
print(f"  F1 Score:  {union_result['f1_score']:.4f}")
print()

print("Covering-based Evaluation:")
print(f"  Covering Score: {result['covering_score']:.4f}")
print()

print("Why Covering is Better:")
print("  - Union treats all annotators' CPs as equally valid")
print("  - Covering measures agreement with EACH annotator")
print("  - Covering reveals if algorithm works for some annotators but not others")
print("  - More robust to outlier annotators")
print()

# ============================================================================
# Scenario 4: High Agreement Annotators
# ============================================================================
print("Scenario 4: High Agreement Among Annotators")
print("-" * 70)

# Simulate annotators with high agreement
high_agreement_annotators = add_annotation_noise(
    true_cps,
    n_annotators=5,
    noise_std=3.0,       # Low noise
    agreement_rate=0.95,  # High agreement
    seed=42
)

print("High Agreement Annotators:")
for i, ann_cps in enumerate(high_agreement_annotators, 1):
    print(f"  Annotator {i}: {ann_cps}")
print()

result_high = covering_metric(high_agreement_annotators, pred_cps, margin=10, n_samples=400)
print(f"Covering Score:     {result_high['covering_score']:.4f}")
print(f"Std Recall:         {result_high['std_recall']:.4f}")
print()

print("Interpretation:")
print("  High agreement → Low std → Easier to satisfy all annotators")
print()

# ============================================================================
# Scenario 5: Low Agreement Annotators
# ============================================================================
print("Scenario 5: Low Agreement Among Annotators")
print("-" * 70)

# Simulate annotators with low agreement
low_agreement_annotators = add_annotation_noise(
    true_cps,
    n_annotators=5,
    noise_std=15.0,      # High noise
    agreement_rate=0.6,   # Low agreement
    seed=42
)

print("Low Agreement Annotators:")
for i, ann_cps in enumerate(low_agreement_annotators, 1):
    print(f"  Annotator {i}: {ann_cps}")
print()

result_low = covering_metric(low_agreement_annotators, pred_cps, margin=10, n_samples=400)
print(f"Covering Score:     {result_low['covering_score']:.4f}")
print(f"Std Recall:         {result_low['std_recall']:.4f}")
print()

print("Interpretation:")
print("  Low agreement → High std → Harder to satisfy all annotators")
print("  Some annotators may have very different change point locations")
print()

# ============================================================================
# Scenario 6: Using evaluate_all with Multiple Annotators
# ============================================================================
print("Scenario 6: Comprehensive Evaluation with Multiple Annotators")
print("-" * 70)

result_all = evaluate_all(annotators, pred_cps, n_samples=400, margin=10)
print(result_all['summary'])
print()

# ============================================================================
# Key Takeaways
# ============================================================================
print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print()
print("1. Covering Metric: Measures agreement with each annotator")
print("2. More informative than union-based evaluation")
print("3. Useful when annotators have different expertise")
print("4. Reveals annotator-specific performance")
print("5. Use add_annotation_noise() to simulate realistic scenarios")
print()
print("Real-world Applications:")
print("  - Medical image analysis (multiple radiologists)")
print("  - Video segmentation (multiple labelers)")
print("  - Any scenario with subjective change point locations")
print()
