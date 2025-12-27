#!/usr/bin/env python
"""
Test pathwas.io.expression module API.

This script tests the expression data preprocessing functions including:
- CPM/TPM normalization
- Log transformation
- Z-score normalization
- Low expression filtering
- Full preprocessing pipeline

Usage:
    python scripts/tests/test_expression_api.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathwas.io.expression import (
    normalize_library_size,
    log_transform,
    zscore_genes,
    preprocess_expression,
)


def test_cpm_normalization():
    """Test CPM normalization."""
    print("Testing CPM normalization...")

    # Create test data with known values
    expr = pd.DataFrame({
        'GENE1': [100, 200, 150],
        'GENE2': [50, 100, 75],
        'GENE3': [200, 400, 300],
    }, index=['sample1', 'sample2', 'sample3'])

    # Apply CPM
    cpm = normalize_library_size(expr, mode='cpm')

    # Verify: each row should sum to 1e6
    row_sums = cpm.sum(axis=1)
    assert np.allclose(row_sums, 1e6), f"Row sums should be 1e6, got {row_sums.values}"

    # Verify shape preserved
    assert cpm.shape == expr.shape, f"Shape mismatch: {cpm.shape} vs {expr.shape}"

    # Verify index and columns preserved
    assert list(cpm.index) == list(expr.index)
    assert list(cpm.columns) == list(expr.columns)

    print("  - Row sums equal 1e6: PASS")
    print("  - Shape preserved: PASS")
    print("  - Index/columns preserved: PASS")
    print("  [PASS] CPM normalization")
    return True


def test_tpm_normalization():
    """Test TPM normalization."""
    print("Testing TPM normalization...")

    # Create test data
    expr = pd.DataFrame({
        'GENE1': [100, 200, 150],
        'GENE2': [50, 100, 75],
        'GENE3': [200, 400, 300],
    }, index=['sample1', 'sample2', 'sample3'])

    # Gene lengths in bp
    gene_lengths = pd.Series([1000, 2000, 500], index=['GENE1', 'GENE2', 'GENE3'])

    # Apply TPM
    tpm = normalize_library_size(expr, mode='tpm', gene_lengths=gene_lengths)

    # Verify: each row should sum to 1e6
    row_sums = tpm.sum(axis=1)
    assert np.allclose(row_sums, 1e6), f"Row sums should be 1e6, got {row_sums.values}"

    # TPM should account for gene length
    # Shorter genes should have higher TPM per read
    # GENE3 is shortest (500bp), should have highest TPM/count ratio
    rpk_ratio = (tpm / expr).iloc[0]
    # GENE3 should have highest ratio (shortest gene)
    assert rpk_ratio['GENE3'] > rpk_ratio['GENE1'], "Shorter genes should have higher TPM/count ratio"

    print("  - Row sums equal 1e6: PASS")
    print("  - Gene length accounted for: PASS")
    print("  [PASS] TPM normalization")
    return True


def test_log_transform():
    """Test log transformation."""
    print("Testing log transformation...")

    expr = pd.DataFrame({
        'GENE1': [0, 1, 10, 100],
        'GENE2': [1, 2, 20, 200],
    })

    # Test log2
    log2_expr = log_transform(expr, base='log2', pseudocount=1.0)

    # log2(0+1) = 0, log2(1+1) = 1
    assert np.isclose(log2_expr.iloc[0, 0], 0.0), f"log2(0+1) should be 0, got {log2_expr.iloc[0, 0]}"
    assert np.isclose(log2_expr.iloc[1, 0], 1.0), f"log2(1+1) should be 1, got {log2_expr.iloc[1, 0]}"

    # log2(10+1) should be log2(11) ≈ 3.46
    expected_log11 = np.log2(11)
    assert np.isclose(log2_expr.iloc[2, 0], expected_log11, atol=0.01), \
        f"log2(10+1) should be {expected_log11:.3f}, got {log2_expr.iloc[2, 0]}"

    print("  - log2(0+1) = 0: PASS")
    print("  - log2(1+1) = 1: PASS")
    print("  - log2(10+1) correct: PASS")

    # Test log10
    log10_expr = log_transform(expr, base='log10', pseudocount=1.0)
    assert np.isclose(log10_expr.iloc[3, 0], np.log10(101), atol=0.01), "log10 failed"
    print("  - log10 transform: PASS")

    # Test ln
    ln_expr = log_transform(expr, base='ln', pseudocount=1.0)
    assert np.isclose(ln_expr.iloc[2, 0], np.log(11), atol=0.01), "ln failed"
    print("  - ln transform: PASS")

    print("  [PASS] Log transformation")
    return True


def test_zscore():
    """Test z-score normalization."""
    print("Testing z-score normalization...")

    expr = pd.DataFrame({
        'GENE1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'GENE2': [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    zscore = zscore_genes(expr, ddof=1)

    # Each column should have mean ~0 and std ~1
    col_means = zscore.mean()
    col_stds = zscore.std(ddof=1)

    assert np.allclose(col_means, 0, atol=1e-10), f"Mean should be 0, got {col_means.values}"
    assert np.allclose(col_stds, 1, atol=1e-10), f"Std should be 1, got {col_stds.values}"

    # Verify shape preserved
    assert zscore.shape == expr.shape

    print("  - Column means are 0: PASS")
    print("  - Column stds are 1: PASS")
    print("  [PASS] Z-score normalization")
    return True


def test_preprocessing_pipeline():
    """Test full preprocessing pipeline."""
    print("Testing full preprocessing pipeline...")

    # Create test data
    np.random.seed(42)
    n_samples, n_genes = 50, 100

    # Generate realistic count data
    expr = pd.DataFrame(
        np.random.negative_binomial(n=10, p=0.1, size=(n_samples, n_genes)),
        index=[f'sample_{i}' for i in range(n_samples)],
        columns=[f'GENE{i}' for i in range(n_genes)]
    )

    # Add some low-expression genes (should be filtered)
    expr.iloc[:, :5] = 0  # First 5 genes have zero expression

    # Run full pipeline
    processed = preprocess_expression(
        expr,
        normalization='cpm',
        log_transform_expr=True,
        log_base='log2',
        zscore=True,
        filter_genes=True,
        min_cpm=1.0,
        min_samples_fraction=0.5,
    )

    # Check that low-expression genes were filtered
    assert processed.shape[1] < n_genes, "Low expression genes should be filtered"
    print(f"  - Filtered {n_genes - processed.shape[1]} low-expression genes: PASS")

    # Check that samples are preserved
    assert processed.shape[0] == n_samples, "Sample count should be preserved"
    print("  - Sample count preserved: PASS")

    # Check that z-score was applied (mean ~0, std ~1 per gene)
    col_means = processed.mean()
    col_stds = processed.std(ddof=1)
    assert np.allclose(col_means, 0, atol=0.1), f"Column means should be ~0"
    assert np.allclose(col_stds, 1, atol=0.1), f"Column stds should be ~1"
    print("  - Z-score applied correctly: PASS")

    print("  [PASS] Full preprocessing pipeline")
    return True


def test_with_synthetic_data():
    """Test with actual synthetic data if available."""
    print("Testing with synthetic data...")

    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "expression.csv"

    if not data_path.exists():
        print("  [SKIP] Synthetic data not generated yet")
        print(f"         Run: python scripts/generate_synthetic_expression.py")
        return None

    # Load synthetic data
    expr = pd.read_csv(data_path, index_col=0)
    print(f"  - Loaded expression: {expr.shape[0]} samples x {expr.shape[1]} genes")

    # Test preprocessing
    processed = preprocess_expression(
        expr,
        normalization='cpm',
        log_transform_expr=True,
        zscore=True,
        filter_genes=True,
        min_cpm=1.0,
    )

    print(f"  - After preprocessing: {processed.shape[0]} samples x {processed.shape[1]} genes")

    # Verify no NaN or Inf values
    assert not processed.isna().any().any(), "No NaN values should be present"
    assert not np.isinf(processed.values).any(), "No Inf values should be present"
    print("  - No NaN/Inf values: PASS")

    # Verify z-score normalization
    col_means = processed.mean()
    assert np.allclose(col_means, 0, atol=0.1), "Column means should be ~0 after z-score"
    print("  - Z-score normalization verified: PASS")

    print("  [PASS] Synthetic data test")
    return True


def main():
    print("=" * 60)
    print("PathWAS Expression Module API Tests")
    print("=" * 60)
    print()

    tests = [
        ("CPM Normalization", test_cpm_normalization),
        ("TPM Normalization", test_tpm_normalization),
        ("Log Transformation", test_log_transform),
        ("Z-score Normalization", test_zscore),
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Synthetic Data", test_with_synthetic_data),
    ]

    results = []
    for name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {name}: {status}")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
