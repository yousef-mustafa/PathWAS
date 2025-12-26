"""Tests for expression preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from pathwas.io import (
    normalize_library_size,
    log_transform,
    zscore_genes,
    filter_low_expression,
    filter_low_count_samples,
    preprocess_expression,
)


class TestNormalizeLibrarySize:
    """Tests for library size normalization."""

    @pytest.fixture
    def raw_counts(self):
        """Create simple raw count data."""
        np.random.seed(42)
        counts = np.random.poisson(lam=100, size=(10, 5))
        return pd.DataFrame(
            counts,
            index=[f"sample_{i}" for i in range(10)],
            columns=[f"gene_{i}" for i in range(5)],
        )

    def test_cpm_normalization(self, raw_counts):
        cpm = normalize_library_size(raw_counts, mode="cpm")

        assert cpm.shape == raw_counts.shape
        assert list(cpm.index) == list(raw_counts.index)
        assert list(cpm.columns) == list(raw_counts.columns)

        # Each row should sum to 1e6 (or close to it)
        row_sums = cpm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1e6, rtol=1e-10)

    def test_cpm_preserves_relative_expression(self, raw_counts):
        cpm = normalize_library_size(raw_counts, mode="cpm")

        # Relative expression within each sample should be preserved
        for sample in raw_counts.index:
            raw_ratios = raw_counts.loc[sample] / raw_counts.loc[sample].sum()
            cpm_ratios = cpm.loc[sample] / cpm.loc[sample].sum()
            np.testing.assert_allclose(raw_ratios, cpm_ratios)

    def test_tpm_normalization(self, raw_counts):
        gene_lengths = pd.Series(
            [1000, 2000, 500, 1500, 3000],
            index=[f"gene_{i}" for i in range(5)],
        )

        tpm = normalize_library_size(raw_counts, mode="tpm", gene_lengths=gene_lengths)

        assert tpm.shape == raw_counts.shape

        # Each row should sum to 1e6
        row_sums = tpm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1e6, rtol=1e-10)

    def test_tpm_requires_gene_lengths(self, raw_counts):
        with pytest.raises(ValueError, match="gene_lengths required"):
            normalize_library_size(raw_counts, mode="tpm")

    def test_none_mode_returns_copy(self, raw_counts):
        result = normalize_library_size(raw_counts, mode=None)

        pd.testing.assert_frame_equal(result, raw_counts)
        # Should be a copy, not the same object
        assert result is not raw_counts

    def test_unknown_mode_raises(self, raw_counts):
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize_library_size(raw_counts, mode="unknown")

    def test_zero_total_counts_handled(self):
        """Test that samples with zero total counts are handled gracefully."""
        counts = pd.DataFrame(
            [[0, 0, 0], [100, 200, 300]],
            index=["zero_sample", "normal_sample"],
            columns=["gene_1", "gene_2", "gene_3"],
        )

        cpm = normalize_library_size(counts, mode="cpm")

        # Zero sample should have all zeros
        assert (cpm.loc["zero_sample"] == 0).all()
        # Normal sample should be normalized
        assert cpm.loc["normal_sample"].sum() == pytest.approx(1e6)


class TestLogTransform:
    """Tests for log transformation."""

    @pytest.fixture
    def expression(self):
        return pd.DataFrame(
            [[1, 10, 100], [2, 20, 200]],
            index=["s1", "s2"],
            columns=["g1", "g2", "g3"],
        )

    def test_log2_transform(self, expression):
        log_expr = log_transform(expression, base="log2", pseudocount=1.0)

        expected = np.log2(expression + 1.0)
        pd.testing.assert_frame_equal(log_expr, expected)

    def test_log10_transform(self, expression):
        log_expr = log_transform(expression, base="log10", pseudocount=1.0)

        expected = np.log10(expression + 1.0)
        pd.testing.assert_frame_equal(log_expr, expected)

    def test_ln_transform(self, expression):
        log_expr = log_transform(expression, base="ln", pseudocount=1.0)

        expected = np.log(expression + 1.0)
        pd.testing.assert_frame_equal(log_expr, expected)

    def test_custom_pseudocount(self, expression):
        log_expr = log_transform(expression, base="log2", pseudocount=0.5)

        expected = np.log2(expression + 0.5)
        pd.testing.assert_frame_equal(log_expr, expected)

    def test_unknown_base_raises(self, expression):
        with pytest.raises(ValueError, match="Unknown log base"):
            log_transform(expression, base="log3")


class TestZscoreGenes:
    """Tests for z-score normalization."""

    def test_zscore_basic(self):
        expression = pd.DataFrame(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            index=["s1", "s2", "s3"],
            columns=["g1", "g2"],
        )

        zscore = zscore_genes(expression)

        # Each column should have mean ~0 and std ~1
        assert zscore.shape == expression.shape
        np.testing.assert_allclose(zscore.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(zscore.std(axis=0, ddof=1), 1, atol=1e-10)

    def test_zscore_zero_variance(self):
        """Test that zero-variance genes are set to 0."""
        expression = pd.DataFrame(
            [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]],  # g1 has zero variance
            index=["s1", "s2", "s3"],
            columns=["g1", "g2"],
        )

        zscore = zscore_genes(expression)

        # Zero-variance gene should be all zeros
        assert (zscore["g1"] == 0).all()
        # Non-zero variance gene should be normalized
        assert (zscore["g2"] == 0).all()  # All same value -> all 0 after zscore


class TestFilterLowExpression:
    """Tests for filtering lowly expressed genes."""

    @pytest.fixture
    def counts_with_low_genes(self):
        """Create data with some lowly expressed genes."""
        # 10 samples, 5 genes
        # gene_4 has very low expression in most samples
        return pd.DataFrame(
            {
                "gene_0": [100, 150, 200, 120, 180, 160, 140, 130, 170, 190],
                "gene_1": [200, 250, 300, 220, 280, 260, 240, 230, 270, 290],
                "gene_2": [50, 60, 70, 55, 65, 62, 58, 57, 63, 68],
                "gene_3": [300, 350, 400, 320, 380, 360, 340, 330, 370, 390],
                "gene_4": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Very low expression
            },
            index=[f"sample_{i}" for i in range(10)],
        )

    def test_filter_removes_low_genes(self, counts_with_low_genes):
        filtered = filter_low_expression(
            counts_with_low_genes,
            min_cpm=1.0,
            min_samples_fraction=0.5,
        )

        # gene_4 should be filtered out
        assert "gene_4" not in filtered.columns
        assert len(filtered.columns) == 4

    def test_filter_preserves_samples(self, counts_with_low_genes):
        filtered = filter_low_expression(counts_with_low_genes)

        assert len(filtered) == len(counts_with_low_genes)
        assert list(filtered.index) == list(counts_with_low_genes.index)

    def test_filter_with_cpm_input(self, counts_with_low_genes):
        # Convert to CPM first
        cpm = normalize_library_size(counts_with_low_genes, mode="cpm")

        filtered = filter_low_expression(
            cpm,
            min_cpm=1.0,
            counts_are_cpm=True,
        )

        assert len(filtered.columns) <= len(cpm.columns)


class TestFilterLowCountSamples:
    """Tests for filtering low-count samples."""

    def test_filter_removes_low_samples(self):
        counts = pd.DataFrame(
            {
                "gene_0": [100, 5, 150],
                "gene_1": [200, 3, 250],
                "gene_2": [300, 2, 350],
            },
            index=["normal_1", "low_count", "normal_2"],
        )

        filtered = filter_low_count_samples(counts, min_total_counts=100)

        assert "low_count" not in filtered.index
        assert len(filtered) == 2

    def test_filter_preserves_genes(self):
        counts = pd.DataFrame(
            {
                "gene_0": [100, 200],
                "gene_1": [150, 250],
            },
            index=["s1", "s2"],
        )

        filtered = filter_low_count_samples(counts, min_total_counts=100)

        assert list(filtered.columns) == list(counts.columns)


class TestPreprocessExpression:
    """Tests for the full preprocessing pipeline."""

    @pytest.fixture
    def raw_counts(self):
        np.random.seed(42)
        counts = np.random.poisson(lam=100, size=(20, 10))
        return pd.DataFrame(
            counts,
            index=[f"sample_{i}" for i in range(20)],
            columns=[f"gene_{i}" for i in range(10)],
        )

    def test_full_pipeline(self, raw_counts):
        processed = preprocess_expression(
            raw_counts,
            normalization="cpm",
            log_transform_expr=True,
            zscore=True,
        )

        assert processed.shape[0] == raw_counts.shape[0]
        # May have fewer genes after filtering
        assert processed.shape[1] <= raw_counts.shape[1]

    def test_pipeline_no_filtering(self, raw_counts):
        processed = preprocess_expression(
            raw_counts,
            normalization="cpm",
            log_transform_expr=True,
            zscore=True,
            filter_genes=False,
            filter_samples=False,
        )

        assert processed.shape == raw_counts.shape

    def test_pipeline_minimal(self, raw_counts):
        """Test pipeline with minimal processing."""
        processed = preprocess_expression(
            raw_counts,
            normalization=None,
            log_transform_expr=False,
            zscore=False,
            filter_genes=False,
            filter_samples=False,
        )

        pd.testing.assert_frame_equal(processed, raw_counts)

    def test_pipeline_with_sample_filter(self, raw_counts):
        # Add a low-count sample
        raw_counts.loc["low_sample"] = 1

        processed = preprocess_expression(
            raw_counts,
            filter_samples=True,
            min_total_counts=100,
            filter_genes=False,
            normalization="cpm",
        )

        assert "low_sample" not in processed.index
