"""Comprehensive tests for fastcpd.metrics module."""

import numpy as np
import pytest
from fastcpd.metrics import (
    precision_recall,
    f_beta_score,
    hausdorff_distance,
    annotation_error,
    adjusted_rand_index,
    covering_metric,
    evaluate_all
)


class TestPrecisionRecall:
    """Test precision_recall function."""

    def test_perfect_match(self):
        """Test perfect match between true and predicted CPs."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200, 300]

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
        assert result['true_positives'] == 3
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 0

    def test_with_margin(self):
        """Test matching within margin."""
        true_cps = [100, 200, 300]
        pred_cps = [105, 195, 305]  # All within 10

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0

    def test_false_positives(self):
        """Test with extra predicted CPs."""
        true_cps = [100, 200]
        pred_cps = [100, 150, 200]  # 150 is false positive

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 2/3  # 2 correct out of 3
        assert result['recall'] == 1.0     # Found all true CPs
        assert result['true_positives'] == 2
        assert result['false_positives'] == 1
        assert result['false_negatives'] == 0

    def test_false_negatives(self):
        """Test with missed true CPs."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200]  # Missed 300

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 1.0   # All predictions correct
        assert result['recall'] == 2/3      # Found 2 out of 3
        assert result['true_positives'] == 2
        assert result['false_positives'] == 0
        assert result['false_negatives'] == 1

    def test_empty_predictions(self):
        """Test with no predictions."""
        true_cps = [100, 200]
        pred_cps = []

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 1.0  # No false positives
        assert result['recall'] == 0.0      # Missed all true CPs
        assert result['f1_score'] == 0.0

    def test_empty_true(self):
        """Test with no true CPs."""
        true_cps = []
        pred_cps = [100, 200]

        result = precision_recall(true_cps, pred_cps, margin=10)

        assert result['precision'] == 0.0
        assert result['recall'] == 1.0  # No CPs to miss

    def test_both_empty(self):
        """Test with both empty."""
        result = precision_recall([], [], margin=10)

        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0


class TestFBetaScore:
    """Test f_beta_score function."""

    def test_f1_equals_precision_recall(self):
        """F1 (beta=1) should match precision_recall result."""
        true_cps = [100, 200, 300]
        pred_cps = [105, 150, 200]

        pr_result = precision_recall(true_cps, pred_cps, margin=10)
        f1_result = f_beta_score(true_cps, pred_cps, beta=1.0, margin=10)

        assert abs(pr_result['f1_score'] - f1_result['f_beta']) < 1e-10

    def test_f2_favors_recall(self):
        """F2 (beta=2) should weight recall higher."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200, 250, 300]  # High recall (1.0), lower precision (0.75)

        f1 = f_beta_score(true_cps, pred_cps, beta=1.0, margin=10)
        f2 = f_beta_score(true_cps, pred_cps, beta=2.0, margin=10)

        # F2 should be higher when recall is high despite lower precision
        assert f2['f_beta'] >= f1['f_beta']

    def test_f05_favors_precision(self):
        """F0.5 should weight precision higher."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200]  # Perfect precision (1.0), lower recall (0.67)

        f1 = f_beta_score(true_cps, pred_cps, beta=1.0, margin=10)
        f05 = f_beta_score(true_cps, pred_cps, beta=0.5, margin=10)

        # F0.5 should be higher when precision is high despite lower recall
        assert f05['f_beta'] >= f1['f_beta']


class TestHausdorffDistance:
    """Test hausdorff_distance function."""

    def test_identical_sets(self):
        """Identical sets should have 0 distance."""
        cps1 = [100, 200, 300]
        cps2 = [100, 200, 300]

        result = hausdorff_distance(cps1, cps2)

        assert result['hausdorff'] == 0.0
        assert result['forward_distance'] == 0.0
        assert result['backward_distance'] == 0.0

    def test_directed_distance(self):
        """Test directed Hausdorff."""
        cps1 = [100, 200]
        cps2 = [100, 250]  # 200 -> 250 is 50 away

        result = hausdorff_distance(cps1, cps2, directed=True)

        assert result['forward_distance'] == 50.0  # max(0, 50) from cps1 to cps2
        assert result['backward_distance'] == 50.0  # max(0, 50) from cps2 to cps1
        assert result['hausdorff'] == 50.0

    def test_asymmetric(self):
        """Hausdorff can be asymmetric."""
        cps1 = [100]
        cps2 = [100, 500]  # Extra CP far away

        result = hausdorff_distance(cps1, cps2, directed=True)

        # cps1 -> cps2: each in cps1 close to something in cps2
        assert result['forward_distance'] == 0.0
        # cps2 -> cps1: 500 is 400 away from nearest (100)
        assert result['backward_distance'] == 400.0

    def test_empty_sets(self):
        """Empty sets handled gracefully."""
        # One empty set returns inf
        result = hausdorff_distance([], [100, 200])
        assert result['hausdorff'] == np.inf

        result = hausdorff_distance([100], [])
        assert result['hausdorff'] == np.inf

        # Both empty returns 0
        result = hausdorff_distance([], [])
        assert result['hausdorff'] == 0.0


class TestAnnotationError:
    """Test annotation_error function."""

    def test_perfect_match(self):
        """Perfect match should have 0 error."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200, 300]

        result = annotation_error(true_cps, pred_cps, method='mae')

        assert result['error'] == 0.0
        assert result['mean_error'] == 0.0
        assert result['median_error'] == 0.0

    def test_mae_calculation(self):
        """Test MAE calculation."""
        true_cps = [100, 200, 300]
        pred_cps = [105, 195, 310]  # Errors: 5, 5, 10

        result = annotation_error(true_cps, pred_cps, method='mae')

        expected_mae = (5 + 5 + 10) / 3
        assert abs(result['error'] - expected_mae) < 1e-10
        assert abs(result['mean_error'] - expected_mae) < 1e-10

    def test_mse_calculation(self):
        """Test MSE calculation."""
        true_cps = [100, 200, 300]
        pred_cps = [110, 190, 300]  # Errors: 10, 10, 0

        result_mse = annotation_error(true_cps, pred_cps, method='mse')
        result_rmse = annotation_error(true_cps, pred_cps, method='rmse')

        expected_mse = (100 + 100 + 0) / 3
        expected_rmse = np.sqrt(expected_mse)

        assert abs(result_mse['error'] - expected_mse) < 1e-10
        assert abs(result_rmse['error'] - expected_rmse) < 1e-10

    def test_different_lengths(self):
        """Different lengths should return nan."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200]

        result = annotation_error(true_cps, pred_cps, method='mae')

        # With different lengths, only 2 pairs can be matched, errors calculated for those
        assert len(result['errors_per_cp']) == 2
        assert not np.isnan(result['error'])

    def test_both_empty(self):
        """Both empty should return nan."""
        result = annotation_error([], [], method='mae')

        assert np.isnan(result['error'])
        assert np.isnan(result['mean_error'])


class TestAdjustedRandIndex:
    """Test adjusted_rand_index function."""

    def test_identical_segmentation(self):
        """Identical segmentations should give ARI=1."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200, 300]

        result = adjusted_rand_index(true_cps, pred_cps, n_samples=400)

        assert abs(result['ari'] - 1.0) < 1e-10

    def test_completely_different(self):
        """Completely different should give lower ARI."""
        true_cps = [100, 200, 300]
        pred_cps = [50, 150, 250]

        result = adjusted_rand_index(true_cps, pred_cps, n_samples=400)

        # ARI should be lower than perfect match
        result_perfect = adjusted_rand_index(true_cps, true_cps, n_samples=400)
        assert result['ari'] < result_perfect['ari']

    def test_no_changepoints(self):
        """No changepoints in both should give ARI=1."""
        result = adjusted_rand_index([], [], n_samples=400)

        assert abs(result['ari'] - 1.0) < 1e-10

    def test_offset_changepoints(self):
        """Slightly offset CPs should give reasonable ARI."""
        true_cps = [100, 200, 300]
        pred_cps = [105, 205, 305]  # Offset by 5

        result = adjusted_rand_index(true_cps, pred_cps, n_samples=400)

        # Should still have high agreement
        assert result['ari'] > 0.8


class TestCoveringMetric:
    """Test covering_metric function for multiple annotators."""

    def test_single_annotator_perfect(self):
        """Single annotator with perfect match."""
        true_cps_list = [[100, 200, 300]]
        pred_cps = [100, 200, 300]

        result = covering_metric(true_cps_list, pred_cps, margin=10)

        assert result['covering_score'] == 1.0
        assert result['recall_per_annotator'][0] == 1.0

    def test_multiple_annotators(self):
        """Multiple annotators with varying agreement."""
        true_cps_list = [
            [100, 200, 300],     # Annotator 1
            [105, 195, 305],     # Annotator 2 (close)
            [100, 250]           # Annotator 3 (different)
        ]
        pred_cps = [100, 200, 300]

        result = covering_metric(true_cps_list, pred_cps, margin=10, n_samples=400)

        # Should have high recall for first two annotators
        assert result['recall_per_annotator'][0] == 1.0
        assert result['recall_per_annotator'][1] == 1.0

        # Covering should be average recall
        assert 0 < result['covering_score'] <= 1.0

    def test_empty_annotators(self):
        """Gracefully handle edge cases."""
        result = covering_metric([[]], [100, 200], margin=10)

        assert 'covering_score' in result
        assert 'recall_per_annotator' in result


class TestEvaluateAll:
    """Test evaluate_all comprehensive evaluation."""

    def test_returns_all_metrics(self):
        """Should return all metrics in one call."""
        true_cps = [100, 200, 300]
        pred_cps = [105, 195, 305]

        result = evaluate_all(true_cps, pred_cps, n_samples=400, margin=10)

        # Check all metric categories present
        assert 'point_metrics' in result
        assert 'distance_metrics' in result
        assert 'segmentation_metrics' in result
        assert 'summary' in result

        # Check sub-keys
        assert 'precision' in result['point_metrics']
        assert 'recall' in result['point_metrics']
        assert 'adjusted_rand_index' in result['segmentation_metrics']

    def test_perfect_detection(self):
        """Perfect detection should score well on all metrics."""
        true_cps = [100, 200, 300]
        pred_cps = [100, 200, 300]

        result = evaluate_all(true_cps, pred_cps, n_samples=400, margin=10)

        assert result['point_metrics']['precision'] == 1.0
        assert result['point_metrics']['recall'] == 1.0
        assert result['point_metrics']['f1_score'] == 1.0
        assert result['distance_metrics']['hausdorff'] == 0.0
        assert result['distance_metrics']['annotation_error_mae'] == 0.0
        assert abs(result['segmentation_metrics']['adjusted_rand_index'] - 1.0) < 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_changepoint(self):
        """Single CP should work."""
        result = precision_recall([100], [105], margin=10)
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0

    def test_large_margin(self):
        """Large margin should match more."""
        true_cps = [100, 200]
        pred_cps = [150, 250]

        # Small margin: no match
        r1 = precision_recall(true_cps, pred_cps, margin=10)
        assert r1['precision'] == 0.0

        # Large margin: matches
        r2 = precision_recall(true_cps, pred_cps, margin=100)
        assert r2['precision'] == 1.0

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        true_cps = np.array([100, 200, 300])
        pred_cps = np.array([100, 200, 300])

        result = precision_recall(true_cps, pred_cps, margin=10)
        assert result['precision'] == 1.0

    def test_unsorted_inputs(self):
        """Should require sorted inputs."""
        from fastcpd.metrics import ChangePointMetricsError

        true_cps = [300, 100, 200]
        pred_cps = [200, 300, 100]

        # Should raise error for unsorted inputs
        with pytest.raises(ChangePointMetricsError):
            precision_recall(true_cps, pred_cps, margin=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
