"""Tests for the harmonization module."""

import numpy as np
import pandas as pd
import pytest

from pathwas.io.harmonize import (
    is_palindromic,
    harmonize_weights_gwas,
    prepare_pathway_vectors,
)


class TestIsPalindromic:
    """Tests for is_palindromic function."""

    def test_palindromic_pairs(self):
        assert is_palindromic("A", "T") is True
        assert is_palindromic("T", "A") is True
        assert is_palindromic("C", "G") is True
        assert is_palindromic("G", "C") is True

    def test_non_palindromic_pairs(self):
        assert is_palindromic("A", "C") is False
        assert is_palindromic("A", "G") is False
        assert is_palindromic("T", "C") is False
        assert is_palindromic("T", "G") is False
        assert is_palindromic("C", "A") is False
        assert is_palindromic("C", "T") is False
        assert is_palindromic("G", "A") is False
        assert is_palindromic("G", "T") is False

    def test_case_insensitive(self):
        assert is_palindromic("a", "t") is True
        assert is_palindromic("A", "t") is True
        assert is_palindromic("a", "T") is True


class TestHarmonizeWeightsGwas:
    """Tests for harmonize_weights_gwas function."""

    @pytest.fixture
    def weights_df(self):
        """Create test weights DataFrame."""
        return pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "effect_allele_pas": ["A", "C", "A", "G", "A"],
            "other_allele_pas": ["G", "T", "C", "T", "T"],  # rs5 is palindromic
            "beta_pas": [0.5, -0.3, 0.2, 0.1, 0.4],
            "pathway": ["pw1", "pw1", "pw1", "pw1", "pw1"],
        })

    @pytest.fixture
    def gwas_df_matching(self):
        """Create GWAS DataFrame with matching alleles."""
        return pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "effect_allele_gwas": ["A", "C", "A", "G", "A"],
            "other_allele_gwas": ["G", "T", "C", "T", "T"],
            "beta_trait": [1.0, 2.0, 3.0, 4.0, 5.0],
            "se": [0.1, 0.2, 0.3, 0.4, 0.5],
            "z": [10.0, 10.0, 10.0, 10.0, 10.0],
        })

    @pytest.fixture
    def gwas_df_reversed(self):
        """Create GWAS DataFrame with some reversed alleles."""
        return pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            # rs2 has reversed alleles (T/C instead of C/T)
            "effect_allele_gwas": ["A", "T", "A", "G", "A"],
            "other_allele_gwas": ["G", "C", "C", "T", "T"],
            "beta_trait": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [10.0, 10.0, 10.0, 10.0, 10.0],
        })

    @pytest.fixture
    def gwas_df_mismatched(self):
        """Create GWAS DataFrame with mismatched alleles."""
        return pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            # rs3 has completely different alleles
            "effect_allele_gwas": ["A", "C", "T", "G", "A"],
            "other_allele_gwas": ["G", "T", "G", "T", "T"],
            "beta_trait": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [10.0, 10.0, 10.0, 10.0, 10.0],
        })

    def test_matching_alleles(self, weights_df, gwas_df_matching):
        """Test harmonization with matching alleles."""
        harm_weights, harm_gwas = harmonize_weights_gwas(weights_df, gwas_df_matching)

        # rs5 should be dropped (palindromic A/T)
        assert len(harm_weights) == 4
        assert "rs5" not in harm_weights["snp_id"].values

        # Beta values should be unchanged
        rs1_beta = harm_gwas[harm_weights["snp_id"] == "rs1"]["beta_trait"].values[0]
        assert rs1_beta == 1.0

    def test_reversed_alleles_flipped(self, weights_df, gwas_df_reversed):
        """Test that reversed alleles cause beta/z flip."""
        harm_weights, harm_gwas = harmonize_weights_gwas(weights_df, gwas_df_reversed)

        # rs2 should have flipped sign
        idx = harm_weights["snp_id"] == "rs2"
        assert idx.any()
        rs2_beta = harm_gwas.loc[idx.values, "beta_trait"].values[0]
        assert rs2_beta == -2.0  # Original was 2.0, should be flipped

        rs2_z = harm_gwas.loc[idx.values, "z"].values[0]
        assert rs2_z == -10.0  # Z also flipped

    def test_mismatched_alleles_dropped(self, weights_df, gwas_df_mismatched):
        """Test that mismatched alleles are dropped."""
        harm_weights, harm_gwas = harmonize_weights_gwas(weights_df, gwas_df_mismatched)

        # rs3 should be dropped (mismatched alleles)
        assert "rs3" not in harm_weights["snp_id"].values
        # rs5 should be dropped (palindromic)
        assert "rs5" not in harm_weights["snp_id"].values

        # Only rs1, rs2, rs4 should remain
        assert len(harm_weights) == 3

    def test_same_snp_order(self, weights_df, gwas_df_matching):
        """Test that output tables have same SNP order."""
        harm_weights, harm_gwas = harmonize_weights_gwas(weights_df, gwas_df_matching)

        # Same length
        assert len(harm_weights) == len(harm_gwas)

        # SNP order should match
        assert list(harm_weights["snp_id"]) == list(harm_gwas["snp_id"])

    def test_no_overlapping_snps(self, weights_df):
        """Test handling of no overlapping SNPs."""
        gwas_no_overlap = pd.DataFrame({
            "snp_id": ["rs100", "rs200"],
            "effect_allele_gwas": ["A", "C"],
            "other_allele_gwas": ["G", "T"],
            "beta_trait": [1.0, 2.0],
        })

        harm_weights, harm_gwas = harmonize_weights_gwas(weights_df, gwas_no_overlap)

        assert len(harm_weights) == 0
        assert len(harm_gwas) == 0

    def test_missing_columns_raises(self, weights_df, gwas_df_matching):
        """Test that missing required columns raise errors."""
        weights_missing = weights_df.drop(columns=["beta_pas"])
        with pytest.raises(ValueError, match="missing required column"):
            harmonize_weights_gwas(weights_missing, gwas_df_matching)

        gwas_missing = gwas_df_matching.drop(columns=["beta_trait"])
        with pytest.raises(ValueError, match="missing required column"):
            harmonize_weights_gwas(weights_df, gwas_missing)


class TestPreparePathwayVectors:
    """Tests for prepare_pathway_vectors function."""

    def test_extract_single_pathway(self):
        """Test extracting vectors for a single pathway."""
        harm_weights = pd.DataFrame({
            "pathway": ["pw1", "pw1", "pw2"],
            "snp_id": ["rs1", "rs2", "rs1"],
            "beta_pas": [0.5, 0.3, 0.1],
            "effect_allele": ["A", "C", "A"],
            "other_allele": ["G", "T", "G"],
        })

        harm_gwas = pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs1"],
            "beta_trait": [1.0, 2.0, 1.5],
        })

        beta_p, gamma = prepare_pathway_vectors(harm_weights, harm_gwas, "pw1")

        assert len(beta_p) == 2
        assert len(gamma) == 2
        assert set(beta_p.index) == {"rs1", "rs2"}

    def test_missing_pathway_raises(self):
        """Test that requesting missing pathway raises error."""
        harm_weights = pd.DataFrame({
            "pathway": ["pw1"],
            "snp_id": ["rs1"],
            "beta_pas": [0.5],
        })

        harm_gwas = pd.DataFrame({
            "snp_id": ["rs1"],
            "beta_trait": [1.0],
        })

        with pytest.raises(ValueError, match="not found"):
            prepare_pathway_vectors(harm_weights, harm_gwas, "pw_missing")
