"""Tests for ancestry/LD mismatch QC module."""

import numpy as np
import pandas as pd
import pytest

from pathwas.qc.ancestry_mismatch import (
    ReliabilityLevel,
    AncestryQCResult,
    compute_maf_correlation,
    compute_ld_concordance,
    classify_reliability,
    run_ancestry_qc,
)


class TestComputeMafCorrelation:
    """Tests for compute_maf_correlation function."""

    def test_perfect_correlation(self):
        """Identical MAFs should give r=1."""
        maf1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=["rs1", "rs2", "rs3", "rs4", "rs5"])
        maf2 = maf1.copy()

        r, n = compute_maf_correlation(maf1, maf2)

        np.testing.assert_allclose(r, 1.0)
        assert n == 5

    def test_partial_overlap(self):
        """Test with partial SNP overlap (needs at least 3 for correlation)."""
        maf1 = pd.Series([0.1, 0.2, 0.3, 0.4], index=["rs1", "rs2", "rs3", "rs5"])
        maf2 = pd.Series([0.1, 0.2, 0.4, 0.3], index=["rs1", "rs2", "rs4", "rs5"])

        r, n = compute_maf_correlation(maf1, maf2)

        assert n == 3  # rs1, rs2, rs5 overlap
        assert not np.isnan(r)

    def test_no_overlap(self):
        """Test with no overlapping SNPs."""
        maf1 = pd.Series([0.1, 0.2], index=["rs1", "rs2"])
        maf2 = pd.Series([0.3, 0.4], index=["rs3", "rs4"])

        r, n = compute_maf_correlation(maf1, maf2)

        assert n == 0
        assert np.isnan(r)

    def test_handles_nan(self):
        """Test handling of NaN values."""
        maf1 = pd.Series([0.1, 0.2, np.nan, 0.4], index=["rs1", "rs2", "rs3", "rs4"])
        maf2 = pd.Series([0.1, 0.2, 0.3, 0.4], index=["rs1", "rs2", "rs3", "rs4"])

        r, n = compute_maf_correlation(maf1, maf2)

        assert n == 3  # rs3 excluded due to NaN


class TestComputeLdConcordance:
    """Tests for compute_ld_concordance function."""

    def test_identical_ld(self):
        """Identical LD matrices should give r=1."""
        R = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        snp_ids = ["rs1", "rs2", "rs3"]

        r, n = compute_ld_concordance(R, R, snp_ids, snp_ids)

        np.testing.assert_allclose(r, 1.0)
        assert n == 3  # 3 pairs in upper triangle

    def test_different_ld(self):
        """Different LD matrices should give r < 1."""
        R1 = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.4],
            [0.6, 0.4, 1.0]
        ])
        R2 = np.array([
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0]
        ])
        snp_ids = ["rs1", "rs2", "rs3"]

        r, n = compute_ld_concordance(R1, R2, snp_ids, snp_ids)

        assert r < 1.0
        assert n == 3

    def test_partial_overlap(self):
        """Test with partial SNP overlap between matrices (needs >= 3 for correlation)."""
        R1 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        R2 = np.array([[1.0, 0.5, 0.3, 0.1], [0.5, 1.0, 0.2, 0.15],
                       [0.3, 0.2, 1.0, 0.25], [0.1, 0.15, 0.25, 1.0]])

        snp_ids1 = ["rs1", "rs2", "rs3"]
        snp_ids2 = ["rs1", "rs2", "rs3", "rs4"]

        r, n = compute_ld_concordance(R1, R2, snp_ids1, snp_ids2)

        assert n == 3  # Three pairs: (rs1-rs2), (rs1-rs3), (rs2-rs3)

    def test_too_few_snps(self):
        """Test with too few overlapping SNPs."""
        R1 = np.array([[1.0]])
        R2 = np.array([[1.0]])
        snp_ids = ["rs1"]

        r, n = compute_ld_concordance(R1, R2, snp_ids, snp_ids)

        assert np.isnan(r)
        assert n == 0  # No pairs with single SNP


class TestClassifyReliability:
    """Tests for classify_reliability function."""

    def test_high_reliability(self):
        """Test high reliability classification."""
        level = classify_reliability(maf_corr_training_ldref=0.98, ld_concordance=0.95)
        assert level == ReliabilityLevel.HIGH

    def test_moderate_reliability(self):
        """Test moderate reliability classification."""
        level = classify_reliability(maf_corr_training_ldref=0.90, ld_concordance=0.80)
        assert level == ReliabilityLevel.MODERATE

    def test_low_reliability(self):
        """Test low reliability classification."""
        level = classify_reliability(maf_corr_training_ldref=0.70, ld_concordance=0.50)
        assert level == ReliabilityLevel.LOW

    def test_nan_values(self):
        """Test handling of NaN values."""
        # Both NaN -> LOW
        level = classify_reliability(maf_corr_training_ldref=np.nan, ld_concordance=np.nan)
        assert level == ReliabilityLevel.LOW

        # Only MAF available and high -> HIGH
        level = classify_reliability(maf_corr_training_ldref=0.98, ld_concordance=np.nan)
        assert level == ReliabilityLevel.HIGH


class TestRunAncestryQc:
    """Tests for run_ancestry_qc function."""

    @pytest.fixture
    def well_matched_data(self):
        """Create well-matched panel data."""
        np.random.seed(42)
        n_snps = 100
        snp_ids = [f"rs{i}" for i in range(n_snps)]

        maf_training = pd.Series(np.random.uniform(0.05, 0.5, n_snps), index=snp_ids)
        # Very similar MAFs
        maf_ldref = maf_training + np.random.normal(0, 0.02, n_snps)
        maf_ldref = maf_ldref.clip(0.01, 0.99)

        return maf_training, maf_ldref

    @pytest.fixture
    def mismatched_data(self):
        """Create mismatched panel data."""
        np.random.seed(42)
        n_snps = 100
        snp_ids = [f"rs{i}" for i in range(n_snps)]

        maf_training = pd.Series(np.random.uniform(0.05, 0.5, n_snps), index=snp_ids)
        # Completely different MAFs
        maf_ldref = pd.Series(np.random.uniform(0.05, 0.5, n_snps), index=snp_ids)

        return maf_training, maf_ldref

    def test_well_matched_high_reliability(self, well_matched_data):
        """Test that well-matched data gives high reliability."""
        maf_training, maf_ldref = well_matched_data

        result = run_ancestry_qc(maf_training, maf_ldref)

        assert result.maf_corr_training_ldref > 0.9
        assert result.reliability in [ReliabilityLevel.HIGH, ReliabilityLevel.MODERATE]

    def test_mismatched_low_reliability(self, mismatched_data):
        """Test that mismatched data gives low reliability."""
        maf_training, maf_ldref = mismatched_data

        result = run_ancestry_qc(maf_training, maf_ldref)

        # Random MAFs should have low correlation
        assert abs(result.maf_corr_training_ldref) < 0.5
        assert result.reliability == ReliabilityLevel.LOW

    def test_with_gwas_maf(self, well_matched_data):
        """Test including GWAS MAF."""
        maf_training, maf_ldref = well_matched_data
        maf_gwas = maf_training + np.random.normal(0, 0.03, len(maf_training))
        maf_gwas = maf_gwas.clip(0.01, 0.99)

        result = run_ancestry_qc(maf_training, maf_ldref, maf_gwas=maf_gwas)

        assert not np.isnan(result.maf_corr_training_gwas)
        assert not np.isnan(result.maf_corr_ldref_gwas)

    def test_with_ld_matrices(self, well_matched_data):
        """Test including LD matrices."""
        maf_training, maf_ldref = well_matched_data
        n_snps = len(maf_training)

        # Create similar LD matrices
        R = np.eye(n_snps) + 0.1 * np.random.randn(n_snps, n_snps)
        R = (R + R.T) / 2  # Symmetrize
        np.fill_diagonal(R, 1.0)

        R_training = R
        R_ldref = R + 0.05 * np.random.randn(n_snps, n_snps)
        R_ldref = (R_ldref + R_ldref.T) / 2
        np.fill_diagonal(R_ldref, 1.0)

        snp_ids = list(maf_training.index)

        result = run_ancestry_qc(
            maf_training, maf_ldref,
            R_training=R_training,
            R_ldref=R_ldref,
            snp_ids_training=snp_ids,
            snp_ids_ldref=snp_ids,
        )

        assert not np.isnan(result.ld_concordance)
        assert result.n_ld_pairs_compared > 0

    def test_result_to_dict(self, well_matched_data):
        """Test that result can be converted to dict."""
        maf_training, maf_ldref = well_matched_data
        result = run_ancestry_qc(maf_training, maf_ldref)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "maf_corr_training_ldref" in result_dict
        assert "reliability" in result_dict

    def test_result_summary(self, well_matched_data):
        """Test that result summary is generated."""
        maf_training, maf_ldref = well_matched_data
        result = run_ancestry_qc(maf_training, maf_ldref)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Reliability" in summary
