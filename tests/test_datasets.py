"""Comprehensive tests for fastcpd.datasets module."""

import numpy as np
import pytest
from fastcpd.datasets import (
    _draw_changepoints,
    make_mean_change,
    make_variance_change,
    make_regression_change,
    make_arma_change,
    make_glm_change,
    make_garch_change,
    add_annotation_noise
)


class TestDrawChangepoints:
    """Test _draw_changepoints helper."""

    def test_correct_number(self):
        """Should generate correct number of CPs."""
        cps = _draw_changepoints(n_samples=1000, n_changepoints=3, seed=42)

        assert len(cps) == 3
        assert cps[0] > 0
        assert cps[-1] < 1000

    def test_sorted_order(self):
        """CPs should be sorted."""
        cps = _draw_changepoints(n_samples=1000, n_changepoints=5, seed=42)

        assert np.all(np.diff(cps) > 0)

    def test_min_segment_length(self):
        """Should respect minimum segment length."""
        min_len = 50
        cps = _draw_changepoints(n_samples=500, n_changepoints=3,
                                min_segment_length=min_len, seed=42)

        boundaries = np.concatenate([[0], cps, [500]])
        segment_lengths = np.diff(boundaries)

        assert np.all(segment_lengths >= min_len)

    def test_reproducibility(self):
        """Same seed should give same result."""
        cps1 = _draw_changepoints(n_samples=1000, n_changepoints=3, seed=42)
        cps2 = _draw_changepoints(n_samples=1000, n_changepoints=3, seed=42)

        assert np.array_equal(cps1, cps2)


class TestMakeMeanChange:
    """Test make_mean_change function."""

    def test_basic_generation(self):
        """Basic data generation works."""
        data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'data' in data_dict
        assert 'changepoints' in data_dict
        assert 'true_means' in data_dict
        assert 'metadata' in data_dict

    def test_data_shape(self):
        """Data shape matches specifications."""
        # 1D data
        data_dict = make_mean_change(n_samples=500, n_changepoints=3,
                                    n_dim=1, seed=42)
        assert data_dict['data'].shape == (500,)

        # 3D data
        data_dict = make_mean_change(n_samples=500, n_changepoints=3,
                                    n_dim=3, seed=42)
        assert data_dict['data'].shape == (500, 3)

    def test_correct_number_of_segments(self):
        """Should have n_changepoints + 1 segments."""
        data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

        assert len(data_dict['true_means']) == 4
        assert len(data_dict['changepoints']) == 3

    def test_metadata_present(self):
        """Metadata should contain expected keys."""
        data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

        metadata = data_dict['metadata']
        assert 'mean_deltas' in metadata
        assert 'segment_lengths' in metadata
        assert 'snr_db' in metadata
        assert 'difficulty' in metadata
        assert 'change_type' in metadata

    def test_jump_vs_drift(self):
        """Jump and drift should produce different patterns."""
        data_jump = make_mean_change(n_samples=500, n_changepoints=2,
                                    change_type='jump', seed=42)
        data_drift = make_mean_change(n_samples=500, n_changepoints=2,
                                     change_type='drift', seed=42)

        # Both should have same structure
        assert data_jump['data'].shape == data_drift['data'].shape

        # Drift should have gradual changes
        assert data_drift['metadata']['change_type'] == 'drift'

    def test_custom_deltas(self):
        """Custom mean deltas should be used."""
        custom_deltas = [5.0, -5.0, 5.0]
        data_dict = make_mean_change(n_samples=500, n_changepoints=3,
                                    n_dim=3, mean_deltas=custom_deltas, seed=42)

        assert len(data_dict['metadata']['mean_deltas']) == 3

    def test_reproducibility(self):
        """Same seed produces same data."""
        data1 = make_mean_change(n_samples=500, n_changepoints=3, seed=42)
        data2 = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

        assert np.allclose(data1['data'], data2['data'])


class TestMakeVarianceChange:
    """Test make_variance_change function."""

    def test_basic_generation(self):
        """Basic generation works."""
        data_dict = make_variance_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'data' in data_dict
        assert 'changepoints' in data_dict
        assert 'true_variances' in data_dict
        assert 'metadata' in data_dict

    def test_variance_ratios(self):
        """Variance ratios affect data."""
        # Low variance
        data_low = make_variance_change(n_samples=500, n_changepoints=2,
                                       variance_ratios=[1.0, 0.5, 0.5], seed=42)
        # High variance
        data_high = make_variance_change(n_samples=500, n_changepoints=2,
                                        variance_ratios=[1.0, 3.0, 3.0], seed=42)

        # High variance data should have larger spread
        assert np.var(data_high['data']) > np.var(data_low['data'])

    def test_kurtosis_metadata(self):
        """Kurtosis should be calculated per segment."""
        data_dict = make_variance_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'kurtosis_per_segment' in data_dict['metadata']
        assert len(data_dict['metadata']['kurtosis_per_segment']) == 4

    def test_multiplicative_vs_additive(self):
        """Multiplicative and additive modes should differ."""
        data_mult = make_variance_change(n_samples=500, n_changepoints=2,
                                        change_type='multiplicative', seed=42)
        data_add = make_variance_change(n_samples=500, n_changepoints=2,
                                       change_type='additive', seed=42)

        assert data_mult['metadata']['change_type'] == 'multiplicative'
        assert data_add['metadata']['change_type'] == 'additive'


class TestMakeRegressionChange:
    """Test make_regression_change function."""

    def test_basic_generation(self):
        """Basic generation works."""
        data_dict = make_regression_change(n_samples=500, n_changepoints=3,
                                          n_features=3, seed=42)

        assert 'data' in data_dict
        assert 'changepoints' in data_dict
        assert 'true_coefs' in data_dict
        assert 'X' in data_dict
        assert 'y' in data_dict

    def test_data_shape(self):
        """Data shapes are correct."""
        data_dict = make_regression_change(n_samples=500, n_changepoints=3,
                                          n_features=5, seed=42)

        assert data_dict['X'].shape == (500, 5)
        assert data_dict['y'].shape == (500,)
        assert data_dict['data'].shape == (500, 6)  # y + X

    def test_coefficient_changes(self):
        """Different coefficient change types."""
        # Random
        data_rand = make_regression_change(n_samples=500, n_changepoints=2,
                                          coef_changes='random', seed=42)
        assert len(data_rand['true_coefs']) == 3

        # Sign flip
        data_flip = make_regression_change(n_samples=500, n_changepoints=2,
                                          coef_changes='sign_flip', seed=42)
        assert len(data_flip['true_coefs']) == 3

        # Magnitude
        data_mag = make_regression_change(n_samples=500, n_changepoints=2,
                                         coef_changes='magnitude', seed=42)
        assert len(data_mag['true_coefs']) == 3

    def test_r_squared_metadata(self):
        """R² should be calculated."""
        data_dict = make_regression_change(n_samples=500, n_changepoints=3,
                                          n_features=3, seed=42)

        assert 'r_squared_per_segment' in data_dict['metadata']
        assert len(data_dict['metadata']['r_squared_per_segment']) == 4

    def test_correlation_effect(self):
        """Correlation should affect condition number."""
        data_uncor = make_regression_change(n_samples=500, n_changepoints=2,
                                           correlation=0.0, seed=42)
        data_cor = make_regression_change(n_samples=500, n_changepoints=2,
                                         correlation=0.5, seed=42)

        # Correlated features should have higher condition number
        assert data_cor['metadata']['condition_number'] > data_uncor['metadata']['condition_number']


class TestMakeArmaChange:
    """Test make_arma_change function."""

    def test_basic_generation(self):
        """Basic generation works."""
        data_dict = make_arma_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'data' in data_dict
        assert 'changepoints' in data_dict
        assert 'true_params' in data_dict
        assert 'metadata' in data_dict

    def test_data_shape(self):
        """Data is 1D time series."""
        data_dict = make_arma_change(n_samples=500, n_changepoints=3, seed=42)

        assert data_dict['data'].shape == (500,)

    def test_custom_orders(self):
        """Custom ARMA orders should be used."""
        orders = [(1, 1), (2, 0), (0, 2), (1, 1)]
        data_dict = make_arma_change(n_samples=500, n_changepoints=3,
                                    orders=orders, seed=42)

        assert data_dict['metadata']['orders'] == orders

    def test_stationarity_checks(self):
        """Stationarity should be checked."""
        data_dict = make_arma_change(n_samples=500, n_changepoints=3, seed=42)

        metadata = data_dict['metadata']
        assert 'is_stationary' in metadata
        assert 'is_invertible' in metadata
        assert len(metadata['is_stationary']) == 4

    def test_innovation_types(self):
        """Different innovation types should work."""
        # Normal
        data_norm = make_arma_change(n_samples=500, n_changepoints=2,
                                    innovation='normal', seed=42)
        assert data_norm['metadata']['innovation_type'] == 'normal'

        # Student's t
        data_t = make_arma_change(n_samples=500, n_changepoints=2,
                                 innovation='t', seed=42)
        assert data_t['metadata']['innovation_type'] == 't'

        # Skew normal
        data_skew = make_arma_change(n_samples=500, n_changepoints=2,
                                    innovation='skew_normal', seed=42)
        assert data_skew['metadata']['innovation_type'] == 'skew_normal'


class TestMakeGlmChange:
    """Test make_glm_change function."""

    def test_binomial_generation(self):
        """Binomial GLM generation works."""
        data_dict = make_glm_change(n_samples=500, n_changepoints=3,
                                   family='binomial', seed=42)

        assert data_dict['metadata']['family'] == 'binomial'
        assert 'separation_per_segment' in data_dict['metadata']

        # y should be binary for logistic regression
        assert set(np.unique(data_dict['y'])).issubset({0, 1})

    def test_poisson_generation(self):
        """Poisson GLM generation works."""
        data_dict = make_glm_change(n_samples=500, n_changepoints=3,
                                   family='poisson', seed=42)

        assert data_dict['metadata']['family'] == 'poisson'
        assert 'overdispersion_per_segment' in data_dict['metadata']

        # y should be non-negative integers
        assert np.all(data_dict['y'] >= 0)
        assert np.all(data_dict['y'] == data_dict['y'].astype(int))

    def test_data_structure(self):
        """Data structure is correct."""
        data_dict = make_glm_change(n_samples=500, n_changepoints=3,
                                   n_features=5, family='binomial', seed=42)

        assert data_dict['X'].shape == (500, 5)
        assert data_dict['y'].shape == (500,)
        assert data_dict['data'].shape == (500, 6)  # y + X

    def test_binomial_trials(self):
        """Binomial with trials > 1."""
        data_dict = make_glm_change(n_samples=500, n_changepoints=2,
                                   family='binomial', trials=10, seed=42)

        assert data_dict['metadata']['trials'] == 10
        # y should be in [0, 10]
        assert np.all(data_dict['y'] >= 0)
        assert np.all(data_dict['y'] <= 10)

    def test_coefficient_changes(self):
        """Different coefficient change types."""
        # Random
        data_rand = make_glm_change(n_samples=500, n_changepoints=2,
                                   family='poisson', coef_changes='random', seed=42)
        assert len(data_rand['true_coefs']) == 3

        # Sign flip
        data_flip = make_glm_change(n_samples=500, n_changepoints=2,
                                   family='poisson', coef_changes='sign_flip', seed=42)
        assert len(data_flip['true_coefs']) == 3


class TestMakeGarchChange:
    """Test make_garch_change function."""

    def test_basic_generation(self):
        """Basic generation works."""
        data_dict = make_garch_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'data' in data_dict
        assert 'changepoints' in data_dict
        assert 'true_params' in data_dict
        assert 'volatility' in data_dict
        assert 'metadata' in data_dict

    def test_data_shape(self):
        """Data and volatility are 1D."""
        data_dict = make_garch_change(n_samples=500, n_changepoints=3, seed=42)

        assert data_dict['data'].shape == (500,)
        assert data_dict['volatility'].shape == (500,)

    def test_volatility_regimes(self):
        """Volatility regimes are respected."""
        regimes = ['low', 'high', 'low', 'high']
        data_dict = make_garch_change(n_samples=600, n_changepoints=3,
                                     volatility_regimes=regimes, seed=42)

        assert data_dict['metadata']['volatility_regimes'] == regimes

        # High volatility segments should have higher average volatility
        vols = data_dict['metadata']['avg_volatility_per_segment']
        assert vols[1] > vols[0]  # high > low

    def test_garch_orders(self):
        """Custom GARCH orders should work."""
        orders = [(1, 1), (2, 1), (1, 2)]
        data_dict = make_garch_change(n_samples=500, n_changepoints=2,
                                     orders=orders, seed=42)

        assert data_dict['metadata']['orders'] == orders

    def test_kurtosis_metadata(self):
        """Kurtosis should be calculated."""
        data_dict = make_garch_change(n_samples=500, n_changepoints=3, seed=42)

        assert 'kurtosis_per_segment' in data_dict['metadata']
        # GARCH should have heavy tails (kurtosis > 0)

    def test_persistence(self):
        """GARCH persistence should be < 1."""
        data_dict = make_garch_change(n_samples=500, n_changepoints=3, seed=42)

        for params in data_dict['true_params']:
            alpha_sum = sum(params['alpha'])
            beta_sum = sum(params['beta'])
            persistence = alpha_sum + beta_sum

            assert persistence < 1.0


class TestAddAnnotationNoise:
    """Test add_annotation_noise function."""

    def test_basic_functionality(self):
        """Basic annotation noise generation."""
        true_cps = [100, 200, 300]
        annotators = add_annotation_noise(true_cps, n_annotators=5, seed=42)

        assert len(annotators) == 5
        assert all(isinstance(a, list) for a in annotators)

    def test_agreement_rate(self):
        """Agreement rate affects number of CPs."""
        true_cps = [100, 200, 300]

        # High agreement: most annotators include most CPs
        annotators_high = add_annotation_noise(true_cps, n_annotators=10,
                                              agreement_rate=0.9, seed=42)
        avg_cps_high = np.mean([len(a) for a in annotators_high])

        # Low agreement: fewer CPs per annotator
        annotators_low = add_annotation_noise(true_cps, n_annotators=10,
                                             agreement_rate=0.5, seed=42)
        avg_cps_low = np.mean([len(a) for a in annotators_low])

        assert avg_cps_high > avg_cps_low

    def test_noise_std(self):
        """Noise std affects CP locations."""
        true_cps = [100, 200, 300]

        # Low noise
        annotators_low = add_annotation_noise(true_cps, n_annotators=10,
                                             noise_std=2.0, agreement_rate=1.0, seed=42)

        # High noise
        annotators_high = add_annotation_noise(true_cps, n_annotators=10,
                                              noise_std=20.0, agreement_rate=1.0, seed=42)

        # High noise should have more variation from true CPs
        var_low = np.var([cp for a in annotators_low for cp in a])
        var_high = np.var([cp for a in annotators_high for cp in a])

        assert var_high > var_low

    def test_sorted_output(self):
        """Annotator CPs should be sorted."""
        true_cps = [100, 200, 300]
        annotators = add_annotation_noise(true_cps, n_annotators=5, seed=42)

        for a in annotators:
            if len(a) > 1:
                assert a == sorted(a)

    def test_reproducibility(self):
        """Same seed produces same result."""
        true_cps = [100, 200, 300]

        ann1 = add_annotation_noise(true_cps, n_annotators=5, seed=42)
        ann2 = add_annotation_noise(true_cps, n_annotators=5, seed=42)

        for a1, a2 in zip(ann1, ann2):
            assert a1 == a2


class TestDatasetIntegration:
    """Integration tests combining datasets and detection."""

    def test_mean_change_detection(self):
        """Mean change data should be detectable."""
        from fastcpd import fastcpd

        # Generate data with clear mean changes
        data_dict = make_mean_change(n_samples=300, n_changepoints=2,
                                    noise_std=0.5, seed=42)

        # Detect change points
        result = fastcpd(data_dict['data'], family='mean', beta=1.0)

        # Should detect some change points
        assert len(result.cp_set) > 0

    def test_variance_change_detection(self):
        """Variance change data should be detectable."""
        from fastcpd import fastcpd

        data_dict = make_variance_change(n_samples=300, n_changepoints=2,
                                        variance_ratios=[1.0, 4.0, 1.0], seed=42)

        result = fastcpd(data_dict['data'], family='variance', beta=1.0)

        assert len(result.cp_set) > 0

    def test_glm_detection(self):
        """GLM data should be detectable."""
        from fastcpd import fastcpd

        data_dict = make_glm_change(n_samples=400, n_changepoints=2,
                                   family='binomial', n_features=3, seed=42)

        result = fastcpd(data_dict['data'], family='binomial', beta=3.0)

        assert len(result.cp_set) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
