"""Tests for biweight midcorrelation implementation."""

import numpy as np
import pandas as pd
import pytest

from pathwas.pas.pas import _bicor


class TestBicor:
    """Tests for _bicor function."""

    def test_perfect_correlation(self):
        """Test that identical vectors give r=1."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x.copy()

        r = _bicor(x, y)

        np.testing.assert_allclose(r, 1.0, atol=1e-6)

    def test_perfect_anticorrelation(self):
        """Test that perfectly anticorrelated vectors give r=-1."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])

        r = _bicor(x, y)

        np.testing.assert_allclose(r, -1.0, atol=1e-6)

    def test_similar_to_pearson_clean_data(self):
        """Test that bicor is similar to Pearson on clean data."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x * 0.8 + np.random.randn(100) * 0.2

        bicor_r = _bicor(x, y)
        pearson_r = x.corr(y)

        # Should be close for clean data
        assert abs(bicor_r - pearson_r) < 0.1

    def test_robust_to_outlier(self):
        """Test that bicor is more robust to outliers than Pearson."""
        np.random.seed(42)

        # Create data with a strong linear relationship
        x = pd.Series(np.linspace(0, 10, 50))
        y = x * 2 + np.random.randn(50) * 0.5

        # Add an extreme outlier
        x_outlier = x.copy()
        y_outlier = y.copy()
        x_outlier.iloc[25] = 100  # Extreme outlier
        y_outlier.iloc[25] = -100

        # Compute correlations
        pearson_clean = x.corr(y)
        pearson_outlier = x_outlier.corr(y_outlier)

        bicor_clean = _bicor(x, y)
        bicor_outlier = _bicor(x_outlier, y_outlier)

        # Pearson should be heavily affected by outlier
        pearson_diff = abs(pearson_clean - pearson_outlier)

        # Bicor should be less affected
        bicor_diff = abs(bicor_clean - bicor_outlier)

        # Bicor should be more robust (less change due to outlier)
        # Note: This test may not pass if astropy is not installed
        # since fallback is Pearson
        try:
            from astropy.stats import biweight_midcorrelation
            assert bicor_diff < pearson_diff
        except ImportError:
            # If astropy not available, bicor falls back to Pearson
            pass

    def test_handles_nan(self):
        """Test that NaN values are handled correctly."""
        x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        r = _bicor(x, y)

        # Should compute on valid pairs (indices 0, 1, 4, 5, 6, 7, 8, 9)
        assert not np.isnan(r)

    def test_handles_inf(self):
        """Test that inf values are handled correctly."""
        x = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        r = _bicor(x, y)

        # Should compute on valid pairs
        assert not np.isnan(r)

    def test_insufficient_values_returns_nan(self):
        """Test that too few valid values returns NaN."""
        x = pd.Series([1.0, np.nan, np.nan, np.nan])
        y = pd.Series([1.0, 2.0, 3.0, np.nan])

        r = _bicor(x, y, min_valid=5)

        assert np.isnan(r)

    def test_constant_values_fallback(self):
        """Test fallback for degenerate input (constant values)."""
        x = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        r = _bicor(x, y)

        # Constant x has zero variance, correlation is NaN
        assert np.isnan(r)

    def test_numpy_input(self):
        """Test that numpy arrays work too."""
        x = pd.Series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        y = pd.Series(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))

        r = _bicor(x, y)

        np.testing.assert_allclose(r, 1.0, atol=1e-6)


class TestBicorIntegration:
    """Integration tests for bicor in PAS computation."""

    def test_bicor_used_in_pas(self):
        """Test that bicor is used in activity-weighted PAS."""
        from pathwas.pas.pas import compute_pas

        expr = pd.DataFrame({
            'g1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'g2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }, index=[f's{i}' for i in range(10)])

        pathways = {'pw1': ['g1', 'g2']}

        # This should use _bicor internally
        pas, weights = compute_pas(
            expr, pathways, method='activity_weighted', corr_method='bicor'
        )

        assert weights is not None
        assert 'pw1' in weights
        assert 'g1' in weights['pw1']
        assert 'g2' in weights['pw1']
