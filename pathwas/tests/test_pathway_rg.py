"""Tests for pathway-level genetic correlation."""

import numpy as np
import pandas as pd
import pytest

from pathwas.association.pathway_rg import (
    compute_ld_matrix,
    compute_genetic_correlation,
    compute_genetic_correlation_with_components,
    jackknife_genetic_correlation,
    GeneticCorrelationResult,
)


class TestComputeLdMatrix:
    """Tests for compute_ld_matrix function."""

    def test_identity_ld_uncorrelated(self):
        """Uncorrelated standardized genotypes should give near-identity LD."""
        np.random.seed(42)
        n_samples = 1000
        n_snps = 10

        # Generate uncorrelated genotypes
        X = np.random.randn(n_samples, n_snps)
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

        R = compute_ld_matrix(X)

        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(R), 1.0, atol=0.05)

        # Off-diagonal should be ~0
        off_diag = R - np.diag(np.diag(R))
        assert np.abs(off_diag).max() < 0.15

    def test_perfect_correlation(self):
        """Perfectly correlated SNPs should give r close to 1."""
        np.random.seed(42)
        n_samples = 100

        # Two identical SNPs
        X = np.random.randn(n_samples, 1)
        X = np.hstack([X, X])  # Duplicate column
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

        R = compute_ld_matrix(X)

        # All entries should be close to 1 (N-1 divisor gives slight variance)
        np.testing.assert_allclose(R, np.ones((2, 2)), atol=0.02)

    def test_minimum_samples(self):
        """Should raise with fewer than 2 samples."""
        X = np.random.randn(1, 5)
        with pytest.raises(ValueError, match="at least 2 samples"):
            compute_ld_matrix(X)


class TestComputeGeneticCorrelation:
    """Tests for genetic correlation computation."""

    def test_perfect_correlation(self):
        """Identical effects should give rg=1."""
        beta = np.array([0.5, 0.3, -0.2])
        gamma = beta.copy()
        R = np.eye(3)

        rg = compute_genetic_correlation(beta, gamma, R)
        np.testing.assert_allclose(rg, 1.0)

    def test_perfect_anticorrelation(self):
        """Opposite effects should give rg=-1."""
        beta = np.array([0.5, 0.3, -0.2])
        gamma = -beta
        R = np.eye(3)

        rg = compute_genetic_correlation(beta, gamma, R)
        np.testing.assert_allclose(rg, -1.0)

    def test_zero_correlation(self):
        """Orthogonal effects should give rg=0."""
        beta = np.array([1.0, 0.0])
        gamma = np.array([0.0, 1.0])
        R = np.eye(2)

        rg = compute_genetic_correlation(beta, gamma, R)
        np.testing.assert_allclose(rg, 0.0)

    def test_with_ld_structure(self):
        """Test with non-identity LD matrix."""
        beta = np.array([0.5, 0.3])
        gamma = np.array([0.4, 0.2])

        # Correlated LD
        R = np.array([[1.0, 0.5],
                      [0.5, 1.0]])

        rg = compute_genetic_correlation(beta, gamma, R)

        # Manually compute
        cov_g = beta @ R @ gamma
        var_pas = beta @ R @ beta
        var_trait = gamma @ R @ gamma
        expected_rg = cov_g / np.sqrt(var_pas * var_trait)

        np.testing.assert_allclose(rg, expected_rg)

    def test_zero_variance_returns_nan(self):
        """Zero variance should return NaN."""
        beta = np.array([0.0, 0.0])  # Zero variance
        gamma = np.array([0.5, 0.3])
        R = np.eye(2)

        rg = compute_genetic_correlation(beta, gamma, R)
        assert np.isnan(rg)

    def test_pandas_input(self):
        """Should work with pandas Series."""
        beta = pd.Series([0.5, 0.3, -0.2], index=["rs1", "rs2", "rs3"])
        gamma = pd.Series([0.4, 0.2, -0.1], index=["rs1", "rs2", "rs3"])
        R = np.eye(3)

        rg = compute_genetic_correlation(beta, gamma, R)
        assert not np.isnan(rg)

    def test_dimension_mismatch_raises(self):
        """Mismatched dimensions should raise."""
        beta = np.array([0.5, 0.3])
        gamma = np.array([0.4, 0.2, 0.1])  # Wrong length
        R = np.eye(2)

        with pytest.raises(ValueError, match="length"):
            compute_genetic_correlation(beta, gamma, R)


class TestComputeGeneticCorrelationWithComponents:
    """Tests for compute_genetic_correlation_with_components."""

    def test_returns_all_components(self):
        beta = np.array([0.5, 0.3])
        gamma = np.array([0.4, 0.2])
        R = np.eye(2)

        result = compute_genetic_correlation_with_components(beta, gamma, R)

        assert "rg" in result
        assert "cov_g" in result
        assert "var_g_pas" in result
        assert "var_g_trait" in result

    def test_variance_components_correct(self):
        beta = np.array([0.5, 0.3])
        gamma = np.array([0.4, 0.2])
        R = np.eye(2)

        result = compute_genetic_correlation_with_components(beta, gamma, R)

        expected_var_pas = beta @ R @ beta
        expected_var_trait = gamma @ R @ gamma
        expected_cov = beta @ R @ gamma

        np.testing.assert_allclose(result["var_g_pas"], expected_var_pas)
        np.testing.assert_allclose(result["var_g_trait"], expected_var_trait)
        np.testing.assert_allclose(result["cov_g"], expected_cov)


class TestJackknifeGeneticCorrelation:
    """Tests for jackknife standard error estimation."""

    def test_basic_jackknife(self):
        """Test basic jackknife computation."""
        np.random.seed(42)

        # Create correlated effects
        n_snps = 30
        true_beta = np.random.randn(n_snps) * 0.3
        true_gamma = true_beta * 0.8 + np.random.randn(n_snps) * 0.1

        R = np.eye(n_snps)
        block_ids = np.repeat(np.arange(5), 6)  # 5 blocks of 6 SNPs

        result = jackknife_genetic_correlation(true_beta, true_gamma, R, block_ids)

        assert isinstance(result, GeneticCorrelationResult)
        assert not np.isnan(result.rg)
        assert not np.isnan(result.se)
        assert result.n_snps == n_snps
        assert result.n_blocks_used > 0

    def test_significant_correlation(self):
        """Test that significant correlation gives small p-value."""
        np.random.seed(42)

        n_snps = 50
        true_beta = np.random.randn(n_snps) * 0.5
        # Strongly correlated gamma
        true_gamma = true_beta * 0.9 + np.random.randn(n_snps) * 0.05

        R = np.eye(n_snps)
        block_ids = np.repeat(np.arange(10), 5)

        result = jackknife_genetic_correlation(true_beta, true_gamma, R, block_ids)

        # rg should be high
        assert result.rg > 0.8

        # p-value should be very small
        assert result.p < 0.01

    def test_few_blocks_warning(self):
        """Test handling of few blocks."""
        beta = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.4, 0.2, 0.1])
        R = np.eye(3)
        block_ids = np.array([0, 0, 0])  # All in one block

        result = jackknife_genetic_correlation(beta, gamma, R, block_ids)

        # Should still compute rg but SE should be NaN
        assert not np.isnan(result.rg)
        assert np.isnan(result.se)

    def test_result_to_dict(self):
        """Test that result can be converted to dict."""
        np.random.seed(42)
        n_snps = 20
        beta = np.random.randn(n_snps)
        gamma = np.random.randn(n_snps)
        R = np.eye(n_snps)
        block_ids = np.repeat(np.arange(4), 5)

        result = jackknife_genetic_correlation(beta, gamma, R, block_ids)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "rg" in result_dict
        assert "se" in result_dict
        assert "p" in result_dict

    def test_edge_case_min_snps_per_block(self):
        """Test min_snps_per_block parameter."""
        np.random.seed(42)
        n_snps = 12
        beta = np.random.randn(n_snps)
        gamma = np.random.randn(n_snps)
        R = np.eye(n_snps)
        # 4 blocks of 3 SNPs each
        block_ids = np.repeat(np.arange(4), 3)

        # With high min_snps, fewer blocks should be usable
        result = jackknife_genetic_correlation(
            beta, gamma, R, block_ids, min_snps_per_block=10
        )

        # Should still work but with fewer usable blocks
        assert result.n_snps == n_snps
