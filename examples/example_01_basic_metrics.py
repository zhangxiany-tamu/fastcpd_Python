"""
Example 1: Basic Metrics Usage
================================

Demonstrates basic usage of evaluation metrics for change point detection.
Shows precision, recall, F1 score, and other standard metrics.
"""

import numpy as np
from fastcpd.metrics import (
    precision_recall,
    f_beta_score,
    hausdorff_distance,
    annotation_error,
    adjusted_rand_index,
    evaluate_all
)

print("=" * 70)
print("Example 1: Basic Metrics Usage")
print("=" * 70)
print()

# ============================================================================
# Scenario 1: Perfect Detection
# ============================================================================
print("Scenario 1: Perfect Detection")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [100, 200, 300]

result = precision_recall(true_cps, pred_cps, margin=10)
print(f"True CPs:      {true_cps}")
print(f"Predicted CPs: {pred_cps}")
print(f"Precision:     {result['precision']:.4f}")
print(f"Recall:        {result['recall']:.4f}")
print(f"F1 Score:      {result['f1_score']:.4f}")
print()

# ============================================================================
# Scenario 2: Detection with Small Errors
# ============================================================================
print("Scenario 2: Detection with Small Errors (within margin)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [98, 205, 295]  # All within margin=10

result = precision_recall(true_cps, pred_cps, margin=10)
print(f"True CPs:      {true_cps}")
print(f"Predicted CPs: {pred_cps}")
print(f"Precision:     {result['precision']:.4f}")
print(f"Recall:        {result['recall']:.4f}")
print(f"F1 Score:      {result['f1_score']:.4f}")
print(f"Matched pairs: {result['matched_pairs']}")
print()

# ============================================================================
# Scenario 3: False Positives (Over-detection)
# ============================================================================
print("Scenario 3: False Positives (Over-detection)")
print("-" * 70)

true_cps = [100, 200]
pred_cps = [100, 150, 200, 250]  # Extra CPs at 150 and 250

result = precision_recall(true_cps, pred_cps, margin=10)
print(f"True CPs:         {true_cps}")
print(f"Predicted CPs:    {pred_cps}")
print(f"Precision:        {result['precision']:.4f} (2 correct out of 4)")
print(f"Recall:           {result['recall']:.4f} (found all true CPs)")
print(f"F1 Score:         {result['f1_score']:.4f}")
print(f"True Positives:   {result['true_positives']}")
print(f"False Positives:  {result['false_positives']}")
print(f"False Negatives:  {result['false_negatives']}")
print()

# ============================================================================
# Scenario 4: False Negatives (Under-detection)
# ============================================================================
print("Scenario 4: False Negatives (Under-detection)")
print("-" * 70)

true_cps = [100, 200, 300, 400]
pred_cps = [100, 200]  # Missed CPs at 300 and 400

result = precision_recall(true_cps, pred_cps, margin=10)
print(f"True CPs:         {true_cps}")
print(f"Predicted CPs:    {pred_cps}")
print(f"Precision:        {result['precision']:.4f} (all predictions correct)")
print(f"Recall:           {result['recall']:.4f} (missed 2 out of 4)")
print(f"F1 Score:         {result['f1_score']:.4f}")
print(f"Unmatched true:   {result['unmatched_true']}")
print()

# ============================================================================
# Scenario 5: F-beta Score (Precision vs Recall Trade-off)
# ============================================================================
print("Scenario 5: F-beta Score (Weighting Precision vs Recall)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [100, 200, 250, 300]  # One false positive at 250

result = f_beta_score(true_cps, pred_cps, beta=1.0, margin=10)
print(f"True CPs:      {true_cps}")
print(f"Predicted CPs: {pred_cps}")
print(f"F1  (beta=1.0): {result['f1_score']:.4f} (balanced)")
print(f"F2  (beta=2.0): {result['f2_score']:.4f} (favor recall)")
print(f"F0.5(beta=0.5): {result['f0_5_score']:.4f} (favor precision)")
print()
print("Interpretation:")
print("  - F2 > F1: When recall is good, emphasizing it gives higher score")
print("  - F0.5 < F1: When precision is lower, de-emphasizing it gives lower score")
print()

# ============================================================================
# Scenario 6: Hausdorff Distance
# ============================================================================
print("Scenario 6: Hausdorff Distance (Worst-case Error)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [105, 200, 350]  # Errors: 5, 0, 50

result = hausdorff_distance(true_cps, pred_cps)
print(f"True CPs:         {true_cps}")
print(f"Predicted CPs:    {pred_cps}")
print(f"Hausdorff:        {result['hausdorff']:.1f} (max distance)")
print(f"Forward distance: {result['forward_distance']:.1f} (true -> pred)")
print(f"Backward distance:{result['backward_distance']:.1f} (pred -> true)")
print(f"Closest pairs:    {result['closest_pairs']}")
print()
print("Note: Hausdorff = 50 because CP at 300 is 50 away from nearest pred (350)")
print()

# ============================================================================
# Scenario 7: Annotation Error (Average Localization Error)
# ============================================================================
print("Scenario 7: Annotation Error (Localization Accuracy)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [105, 195, 310]  # Errors: 5, 5, 10

result = annotation_error(true_cps, pred_cps, method='mae')
print(f"True CPs:      {true_cps}")
print(f"Predicted CPs: {pred_cps}")
print(f"MAE:           {result['error']:.2f}")
print(f"Median Error:  {result['median_error']:.2f}")
print(f"Max Error:     {result['max_error']:.2f}")
print(f"Std Error:     {result['std_error']:.2f}")
print(f"Errors per CP: {result['errors_per_cp']}")
print()

# ============================================================================
# Scenario 8: Adjusted Rand Index (Segmentation Agreement)
# ============================================================================
print("Scenario 8: Adjusted Rand Index (Segmentation Similarity)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [105, 195, 305]  # Slightly offset

result = adjusted_rand_index(true_cps, pred_cps, n_samples=400)
print(f"True CPs:      {true_cps}")
print(f"Predicted CPs: {pred_cps}")
print(f"ARI:           {result['ari']:.4f}")
print(f"Rand Index:    {result['rand_index']:.4f}")
print()
print("Note: ARI measures similarity of segmentations")
print("      ARI=1.0 means perfect agreement, ARI=0 means random agreement")
print()

# ============================================================================
# Scenario 9: Comprehensive Evaluation
# ============================================================================
print("Scenario 9: Comprehensive Evaluation (All Metrics)")
print("-" * 70)

true_cps = [100, 200, 300]
pred_cps = [98, 205, 295]

result = evaluate_all(true_cps, pred_cps, n_samples=400, margin=10)
print(result['summary'])
print()

# ============================================================================
# Key Takeaways
# ============================================================================
print("=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print()
print("1. Precision/Recall: Trade-off between false positives and false negatives")
print("2. F-beta: Adjust beta to weight precision (beta<1) or recall (beta>1)")
print("3. Hausdorff: Worst-case distance, sensitive to outliers")
print("4. Annotation Error: Average localization accuracy")
print("5. ARI: Overall segmentation similarity")
print("6. evaluate_all(): One-stop comprehensive evaluation")
print()
print("Use margin parameter to set tolerance for 'close enough' matches")
print()
