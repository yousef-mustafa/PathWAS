#!/usr/bin/env python
"""
Test pathwas.pas module API.

This script tests the PAS computation functions including:
- Mean/sum/median expression methods
- Activity-weighted method with correlation
- Normalization options
- Biweight midcorrelation (bicor)

Usage:
    python scripts/tests/test_pas_api.py
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathwas.pas.pas import compute_pas


def test_pas_mean_method():
    """Test mean expression PAS method."""
    print("Testing mean expression PAS method...")

    # Create test expression data
    np.random.seed(42)
    n_samples, n_genes = 50, 20

    expr = pd.DataFrame(
        np.random.randn(n_samples, n_genes),
        index=[f'sample_{i}' for i in range(n_samples)],
        columns=[f'GENE{i}' for i in range(n_genes)]
    )

    # Define test pathways
    pathways = {
        'pathway_A': ['GENE0', 'GENE1', 'GENE2', 'GENE3', 'GENE4'],
        'pathway_B': ['GENE5', 'GENE6', 'GENE7', 'GENE8', 'GENE9'],
        'pathway_C': ['GENE10', 'GENE11', 'GENE12'],
    }

    # Compute PAS with mean method
    pas, weights = compute_pas(expr, pathways, method='mean')

    # Verify shape
    assert pas.shape == (n_samples, 3), f"Expected (50, 3), got {pas.shape}"
    print(f"  - Shape correct ({pas.shape}): PASS")

    # Verify columns match pathways
    assert list(pas.columns) == list(pathways.keys())
    print("  - Column names match pathways: PASS")

    # Verify weights is None for mean method
    assert weights is None, "Weights should be None for mean method"
    print("  - Weights is None for mean: PASS")

    # Verify values are correct (manual calculation for pathway_A)
    expected_mean = expr[['GENE0', 'GENE1', 'GENE2', 'GENE3', 'GENE4']].mean(axis=1)
    assert np.allclose(pas['pathway_A'], expected_mean), "Mean calculation incorrect"
    print("  - Mean values correct: PASS")

    print("  [PASS] Mean expression method")
    return True


def test_pas_sum_method():
    """Test sum expression PAS method."""
    print("Testing sum expression PAS method...")

    np.random.seed(42)
    expr = pd.DataFrame(
        np.random.randn(30, 15),
        index=[f'sample_{i}' for i in range(30)],
        columns=[f'GENE{i}' for i in range(15)]
    )

    pathways = {
        'pw1': ['GENE0', 'GENE1', 'GENE2'],
        'pw2': ['GENE3', 'GENE4', 'GENE5'],
    }

    pas, weights = compute_pas(expr, pathways, method='sum')

    # Verify sum calculation
    expected_sum = expr[['GENE0', 'GENE1', 'GENE2']].sum(axis=1)
    assert np.allclose(pas['pw1'], expected_sum), "Sum calculation incorrect"
    print("  - Sum values correct: PASS")

    print("  [PASS] Sum expression method")
    return True


def test_pas_median_method():
    """Test median expression PAS method."""
    print("Testing median expression PAS method...")

    np.random.seed(42)
    expr = pd.DataFrame(
        np.random.randn(30, 15),
        index=[f'sample_{i}' for i in range(30)],
        columns=[f'GENE{i}' for i in range(15)]
    )

    pathways = {
        'pw1': ['GENE0', 'GENE1', 'GENE2'],
    }

    pas, weights = compute_pas(expr, pathways, method='median')

    # Verify median calculation
    expected_median = expr[['GENE0', 'GENE1', 'GENE2']].median(axis=1)
    assert np.allclose(pas['pw1'], expected_median), "Median calculation incorrect"
    print("  - Median values correct: PASS")

    print("  [PASS] Median expression method")
    return True


def test_pas_activity_weighted():
    """Test activity-weighted PAS method."""
    print("Testing activity-weighted PAS method...")

    np.random.seed(42)
    n_samples = 50
    n_genes = 20

    expr = pd.DataFrame(
        np.random.randn(n_samples, n_genes),
        index=[f'sample_{i}' for i in range(n_samples)],
        columns=[f'GENE{i}' for i in range(n_genes)]
    )

    pathways = {
        'pathway_A': ['GENE0', 'GENE1', 'GENE2', 'GENE3', 'GENE4'],
        'pathway_B': ['GENE5', 'GENE6', 'GENE7', 'GENE8', 'GENE9'],
    }

    # Test with different correlation methods
    for corr_method in ['pearson', 'spearman', 'bicor']:
        pas, weights = compute_pas(
            expr, pathways,
            method='activity_weighted',
            corr_method=corr_method
        )

        # Verify weights are returned
        assert weights is not None, f"Weights should be returned for activity_weighted ({corr_method})"

        # Verify weights structure
        assert 'pathway_A' in weights, "Weights should contain pathway keys"
        assert all(g in weights['pathway_A'] for g in pathways['pathway_A']), \
            "Weights should contain all genes"

        # Verify PAS shape
        assert pas.shape == (n_samples, 2)

        print(f"  - {corr_method} correlation: PASS")

    print("  [PASS] Activity-weighted method")
    return True


def test_pas_normalization():
    """Test PAS normalization options."""
    print("Testing PAS normalization...")

    np.random.seed(42)
    expr = pd.DataFrame(
        np.random.randn(30, 15) + 5,  # Add offset to make non-zero centered
        index=[f'sample_{i}' for i in range(30)],
        columns=[f'GENE{i}' for i in range(15)]
    )

    pathways = {
        'pw1': ['GENE0', 'GENE1', 'GENE2'],
        'pw2': ['GENE3', 'GENE4', 'GENE5'],
    }

    # Test normalize_samples (z-score across samples per pathway)
    pas_norm, _ = compute_pas(expr, pathways, method='mean', normalize_samples=True)
    col_means = pas_norm.mean(axis=0)
    col_stds = pas_norm.std(axis=0, ddof=1)

    assert np.allclose(col_means, 0, atol=1e-10), f"Column means should be ~0, got {col_means.values}"
    assert np.allclose(col_stds, 1, atol=1e-10), f"Column stds should be ~1, got {col_stds.values}"
    print("  - normalize_samples: PASS")

    # Test normalize_pathways (z-score across pathways per sample)
    pas_norm2, _ = compute_pas(expr, pathways, method='mean', normalize_pathways=True)
    row_means = pas_norm2.mean(axis=1)
    row_stds = pas_norm2.std(axis=1, ddof=1)

    assert np.allclose(row_means, 0, atol=1e-10), f"Row means should be ~0, got {row_means.values}"
    assert np.allclose(row_stds, 1, atol=1e-10), f"Row stds should be ~1, got {row_stds.values}"
    print("  - normalize_pathways: PASS")

    print("  [PASS] PAS normalization")
    return True


def test_missing_genes():
    """Test behavior with missing genes in pathways."""
    print("Testing missing gene handling...")

    np.random.seed(42)
    expr = pd.DataFrame(
        np.random.randn(30, 10),
        index=[f'sample_{i}' for i in range(30)],
        columns=[f'GENE{i}' for i in range(10)]
    )

    # Pathway includes genes not in expression data
    pathways = {
        'pw1': ['GENE0', 'GENE1', 'MISSING_GENE'],
        'pw2': ['GENE2', 'NONEXISTENT'],
    }

    # Should handle gracefully (only use available genes)
    pas, _ = compute_pas(expr, pathways, method='mean')

    # Should still produce output
    assert pas.shape[0] == 30, "Should produce output for all samples"
    assert 'pw1' in pas.columns, "Should include pathway with some valid genes"

    # Values should be based only on available genes
    expected = expr[['GENE0', 'GENE1']].mean(axis=1)
    assert np.allclose(pas['pw1'], expected), "Should use only available genes"
    print("  - Missing genes handled correctly: PASS")

    print("  [PASS] Missing gene handling")
    return True


def test_bicor_correlation():
    """Test biweight midcorrelation."""
    print("Testing bicor correlation...")

    # Import the internal bicor function
    from pathwas.pas.pas import _bicor

    np.random.seed(42)

    # Create correlated data
    x = pd.Series(np.random.randn(100))
    y = x + np.random.randn(100) * 0.1  # Highly correlated
    z = pd.Series(np.random.randn(100))  # Uncorrelated

    corr_xy = _bicor(x, y)
    corr_xz = _bicor(x, z)

    assert corr_xy > 0.9, f"Expected high correlation, got {corr_xy}"
    print(f"  - bicor(x, y) = {corr_xy:.3f} (expected >0.9): PASS")

    assert abs(corr_xz) < 0.3, f"Expected low correlation, got {corr_xz}"
    print(f"  - bicor(x, z) = {corr_xz:.3f} (expected ~0): PASS")

    # Test with outliers
    x_outlier = x.copy()
    x_outlier.iloc[0] = 100  # Add outlier

    # Bicor should be robust to outliers
    corr_robust = _bicor(x_outlier, y)
    assert corr_robust > 0.8, f"Bicor should be robust to outliers, got {corr_robust}"
    print(f"  - Robust to outliers ({corr_robust:.3f}): PASS")

    print("  [PASS] Bicor correlation")
    return True


def test_with_synthetic_data():
    """Test with actual synthetic data if available."""
    print("Testing with synthetic data...")

    expr_path = Path(__file__).parent.parent.parent / "data" / "raw" / "expression.csv"
    pathway_path = Path(__file__).parent.parent.parent / "data" / "raw" / "pathways.json"

    if not expr_path.exists() or not pathway_path.exists():
        print("  [SKIP] Synthetic data not generated yet")
        return None

    import json

    # Load data
    expr = pd.read_csv(expr_path, index_col=0)
    with open(pathway_path) as f:
        pathways = json.load(f)

    print(f"  - Loaded expression: {expr.shape[0]} samples x {expr.shape[1]} genes")
    print(f"  - Loaded pathways: {len(pathways)}")

    # Compute PAS
    pas, weights = compute_pas(
        expr, pathways,
        method='activity_weighted',
        corr_method='bicor',
        normalize_samples=True
    )

    print(f"  - Computed PAS: {pas.shape[0]} samples x {pas.shape[1]} pathways")

    # Verify no NaN values
    assert not pas.isna().any().any(), "No NaN values should be present"
    print("  - No NaN values: PASS")

    # Verify normalization
    assert np.allclose(pas.mean(), 0, atol=0.1), "PAS should be centered"
    print("  - PAS is centered: PASS")

    print("  [PASS] Synthetic data test")
    return True


def main():
    print("=" * 60)
    print("PathWAS PAS Module API Tests")
    print("=" * 60)
    print()

    tests = [
        ("Mean Method", test_pas_mean_method),
        ("Sum Method", test_pas_sum_method),
        ("Median Method", test_pas_median_method),
        ("Activity-Weighted Method", test_pas_activity_weighted),
        ("PAS Normalization", test_pas_normalization),
        ("Missing Genes", test_missing_genes),
        ("Bicor Correlation", test_bicor_correlation),
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
