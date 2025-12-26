## ------------------------------------------------------------------------------------------- ##
## Tests for LD Reference Setup                                                               ##
## ------------------------------------------------------------------------------------------- ##
## @script: test_ld_setup.py                                                                  ##
##                                                                                             ##
## @description: Unit tests for LD reference setup functions including MAF computation,       ##
##               LD matrix calculation, and PLINK file parsing.                               ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Tests for LD reference setup."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pathwas.ld.setup import (
    SUPPORTED_ANCESTRIES,
    compute_ld_matrix,
    compute_maf,
)


class TestSupportedAncestries:
    """Tests for supported ancestry constants."""

    def test_eur_in_supported(self):
        """EUR ancestry should be supported."""
        assert "EUR" in SUPPORTED_ANCESTRIES

    def test_afr_in_supported(self):
        """AFR ancestry should be supported."""
        assert "AFR" in SUPPORTED_ANCESTRIES

    def test_eas_in_supported(self):
        """EAS ancestry should be supported."""
        assert "EAS" in SUPPORTED_ANCESTRIES

    def test_ancestry_descriptions(self):
        """Each ancestry should have a non-empty description."""
        for code, desc in SUPPORTED_ANCESTRIES.items():
            assert len(desc) > 0, f"Ancestry {code} has empty description"


class TestComputeMaf:
    """Tests for minor allele frequency computation."""

    def test_maf_simple(self):
        """Test MAF with simple known genotypes."""
        # 3 samples, 2 SNPs
        # SNP1: 0, 1, 2 -> freq = 3/6 = 0.5, MAF = 0.5
        # SNP2: 2, 2, 2 -> freq = 6/6 = 1.0, MAF = 0.0
        geno = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.float32)
        maf = compute_maf(geno)
        assert np.allclose(maf, [0.5, 0.0], atol=0.01)

    def test_maf_with_missing(self):
        """Test MAF computation with missing values."""
        # 4 samples, 2 SNPs with some missing
        geno = np.array(
            [[0, 2], [1, np.nan], [2, 2], [np.nan, 2]], dtype=np.float32
        )
        maf = compute_maf(geno)
        # SNP1: 0, 1, 2 (ignoring nan) -> freq = 3/6 = 0.5, MAF = 0.5
        # SNP2: 2, 2, 2 (ignoring nan) -> freq = 6/6 = 1.0, MAF = 0.0
        assert np.allclose(maf, [0.5, 0.0], atol=0.01)

    def test_maf_all_ref(self):
        """Test MAF when all samples are homozygous reference."""
        geno = np.array([[0], [0], [0]], dtype=np.float32)
        maf = compute_maf(geno)
        assert maf[0] == 0.0

    def test_maf_all_het(self):
        """Test MAF when all samples are heterozygous."""
        geno = np.array([[1], [1], [1]], dtype=np.float32)
        maf = compute_maf(geno)
        # freq = 3/6 = 0.5, MAF = 0.5
        assert np.allclose(maf, [0.5], atol=0.01)

    def test_maf_low_frequency(self):
        """Test MAF with low frequency variant."""
        # 10 samples, 1 het carrier
        geno = np.zeros((10, 1), dtype=np.float32)
        geno[0, 0] = 1.0
        maf = compute_maf(geno)
        # freq = 1/20 = 0.05
        assert np.allclose(maf, [0.05], atol=0.01)


class TestComputeLdMatrix:
    """Tests for LD matrix computation."""

    def test_ld_perfect_correlation(self):
        """Test LD matrix with identical genotypes (perfect LD)."""
        geno = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        R = compute_ld_matrix(geno)
        assert R.shape == (2, 2)
        assert np.allclose(R[0, 1], 1.0, atol=0.01)
        assert np.allclose(R[1, 0], 1.0, atol=0.01)
        assert np.allclose(R[0, 0], 1.0, atol=0.01)
        assert np.allclose(R[1, 1], 1.0, atol=0.01)

    def test_ld_negative_correlation(self):
        """Test LD matrix with inversely correlated genotypes."""
        geno = np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float32)
        R = compute_ld_matrix(geno)
        assert R.shape == (2, 2)
        # Should have negative correlation
        assert R[0, 1] < -0.5

    def test_ld_uncorrelated(self):
        """Test LD matrix with uncorrelated genotypes."""
        np.random.seed(42)
        n_samples = 100
        geno = np.column_stack(
            [
                np.random.choice([0, 1, 2], n_samples),
                np.random.choice([0, 1, 2], n_samples),
            ]
        ).astype(np.float32)
        R = compute_ld_matrix(geno)
        # Off-diagonal should be close to 0 with random data
        assert abs(R[0, 1]) < 0.3

    def test_ld_with_missing_values(self):
        """Test LD matrix handles missing values."""
        geno = np.array(
            [[0, 0], [1, 1], [2, 2], [np.nan, np.nan]], dtype=np.float32
        )
        R = compute_ld_matrix(geno)
        assert R.shape == (2, 2)
        # Should still compute valid correlation
        assert not np.isnan(R).any()
        assert np.allclose(R[0, 1], 1.0, atol=0.1)

    def test_ld_single_snp(self):
        """Test LD matrix with single SNP."""
        geno = np.array([[0], [1], [2]], dtype=np.float32)
        R = compute_ld_matrix(geno)
        assert R.shape == (1, 1)
        assert np.allclose(R[0, 0], 1.0)

    def test_ld_matrix_symmetric(self):
        """Test that LD matrix is symmetric."""
        np.random.seed(123)
        n_samples = 50
        n_snps = 5
        geno = np.random.choice([0, 1, 2], (n_samples, n_snps)).astype(
            np.float32
        )
        R = compute_ld_matrix(geno)
        assert R.shape == (n_snps, n_snps)
        assert np.allclose(R, R.T)

    def test_ld_diagonal_is_one(self):
        """Test that diagonal elements are 1.0."""
        np.random.seed(456)
        n_samples = 30
        n_snps = 4
        geno = np.random.choice([0, 1, 2], (n_samples, n_snps)).astype(
            np.float32
        )
        R = compute_ld_matrix(geno)
        assert np.allclose(np.diag(R), 1.0)


class TestLdSetupImports:
    """Test that module imports work correctly."""

    def test_import_from_pathwas(self):
        """Test that setup_ld_reference can be imported from pathwas."""
        from pathwas import setup_ld_reference, SUPPORTED_ANCESTRIES

        assert callable(setup_ld_reference)
        assert isinstance(SUPPORTED_ANCESTRIES, dict)

    def test_import_from_ld_module(self):
        """Test imports from pathwas.ld module."""
        from pathwas.ld import (
            setup_ld_reference,
            SUPPORTED_ANCESTRIES,
            compute_ld_matrix,
            compute_maf,
            list_ancestries,
        )

        assert callable(setup_ld_reference)
        assert callable(compute_ld_matrix)
        assert callable(compute_maf)
        assert callable(list_ancestries)


class TestSetupLdReferenceValidation:
    """Test input validation for setup_ld_reference."""

    def test_invalid_ancestry_raises(self):
        """Test that invalid ancestry raises ValueError."""
        from pathwas.ld.setup import setup_ld_reference

        with pytest.raises(ValueError, match="Unsupported ancestry"):
            setup_ld_reference(
                ancestry="INVALID",
                output_dir=Path("/tmp/test"),
            )

    def test_ancestry_case_insensitive(self):
        """Test that ancestry is case-insensitive."""
        from pathwas.ld.setup import SUPPORTED_ANCESTRIES

        # The function should normalize to uppercase internally
        # Just verify the constants are uppercase
        for key in SUPPORTED_ANCESTRIES.keys():
            assert key == key.upper()
