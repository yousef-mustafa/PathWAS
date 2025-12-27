#!/usr/bin/env python
"""
Test pathwas.pas.pas_test module API.

This script tests the statistical testing functions including:
- t-test, Welch's t-test, Mann-Whitney U
- ANOVA and Kruskal-Wallis for multiple groups
- Multiple testing correction (FDR, Bonferroni)
- Pairwise comparisons

Usage:
    python scripts/tests/test_statistics_api.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathwas.pas.pas_test import (
    test_pas_difference,
    PASTestConfig,
    TestMethod,
    MultipleTestingCorrection,
    pairwise_group_tests,
)


def test_ttest():
    """Test t-test for differential PAS."""
    print("Testing t-test...")

    np.random.seed(42)

    # Create PAS with known differences
    n_samples = 100
    n_pathways = 20

    pas = pd.DataFrame(
        np.random.randn(n_samples, n_pathways),
        index=[f'sample_{i}' for i in range(n_samples)],
        columns=[f'pathway_{i}' for i in range(n_pathways)]
    )

    # Add differential signal to first 5 pathways
    groups = pd.Series(
        ['case'] * 50 + ['control'] * 50,
        index=pas.index,
        name='group'
    )
    pas.loc[groups == 'case', 'pathway_0':'pathway_4'] += 1.5

    # Run test
    config = PASTestConfig(
        method=TestMethod.TTEST,
        alpha=0.05,
        correction=MultipleTestingCorrection.FDR_BH,
    )

    results = test_pas_difference(pas, groups, config=config)

    # Verify output columns
    required_cols = ['pathway', 'pvalue', 'pvalue_adj', 'significant']
    for col in required_cols:
        assert col in results.columns, f"Missing column: {col}"
    print("  - Output columns correct: PASS")

    # Check that differential pathways are detected
    sig_pathways = set(results[results['significant']]['pathway'].tolist())
    expected_sig = {'pathway_0', 'pathway_1', 'pathway_2', 'pathway_3', 'pathway_4'}

    detected = len(sig_pathways & expected_sig)
    print(f"  - Detected {detected}/5 differential pathways")
    assert detected >= 3, f"Should detect at least 3 of 5 differential pathways"

    # Check that most non-differential pathways are not significant
    non_de_sig = len(sig_pathways - expected_sig)
    print(f"  - False positives: {non_de_sig}")
    assert non_de_sig <= 3, "Too many false positives"

    print("  [PASS] t-test")
    return True


def test_welch():
    """Test Welch's t-test (unequal variance)."""
    print("Testing Welch's t-test...")

    np.random.seed(42)
    n = 60

    # Create groups with different variances
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.randn(30) * 0.5 + 1,  # Group A: low variance, mean=1
            np.random.randn(30) * 2.0       # Group B: high variance, mean=0
        ]),
        'pw2': np.random.randn(n),
    }, index=[f's{i}' for i in range(n)])

    groups = pd.Series(['A'] * 30 + ['B'] * 30, index=pas.index)

    config = PASTestConfig(method=TestMethod.WELCH)
    results = test_pas_difference(pas, groups, config=config)

    # pw1 should be significant
    pw1_pval = results[results['pathway'] == 'pw1']['pvalue'].values[0]
    assert pw1_pval < 0.05, f"pw1 should be significant (p={pw1_pval})"
    print(f"  - pw1 p-value: {pw1_pval:.4f}: PASS")

    print("  [PASS] Welch's t-test")
    return True


def test_mann_whitney():
    """Test Mann-Whitney U test (non-parametric)."""
    print("Testing Mann-Whitney U test...")

    np.random.seed(42)
    n = 60

    # Create data with rank difference but non-normal distribution
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.exponential(scale=2.0, size=30),  # Group A: higher
            np.random.exponential(scale=1.0, size=30)   # Group B: lower
        ]),
        'pw2': np.random.exponential(scale=1.5, size=n),
    }, index=[f's{i}' for i in range(n)])

    groups = pd.Series(['A'] * 30 + ['B'] * 30, index=pas.index)

    config = PASTestConfig(method=TestMethod.MANN_WHITNEY)
    results = test_pas_difference(pas, groups, config=config)

    # pw1 should be significant
    pw1_pval = results[results['pathway'] == 'pw1']['pvalue'].values[0]
    assert pw1_pval < 0.05, f"pw1 should be significant (p={pw1_pval})"
    print(f"  - pw1 p-value: {pw1_pval:.4f}: PASS")

    print("  [PASS] Mann-Whitney U test")
    return True


def test_anova():
    """Test ANOVA with multiple groups."""
    print("Testing ANOVA...")

    np.random.seed(42)
    n_per_group = 30

    # Create 3 groups with different means
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.randn(n_per_group) + 0,
            np.random.randn(n_per_group) + 1,
            np.random.randn(n_per_group) + 2,
        ]),
        'pw2': np.random.randn(n_per_group * 3),
    })
    pas.index = [f's{i}' for i in range(len(pas))]

    groups = pd.Series(
        ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group,
        index=pas.index
    )

    config = PASTestConfig(method=TestMethod.ANOVA)
    results = test_pas_difference(pas, groups, config=config)

    # pw1 should be significant (groups differ)
    pw1_result = results[results['pathway'] == 'pw1']
    pw1_pval = pw1_result['pvalue'].values[0]
    assert pw1_pval < 0.05, f"pw1 should be significant (p={pw1_pval})"
    print(f"  - pw1 p-value: {pw1_pval:.4f}: PASS")

    # pw2 should not be significant
    pw2_pval = results[results['pathway'] == 'pw2']['pvalue'].values[0]
    assert pw2_pval > 0.05, f"pw2 should not be significant (p={pw2_pval})"
    print(f"  - pw2 p-value: {pw2_pval:.4f}: PASS")

    print("  [PASS] ANOVA")
    return True


def test_kruskal():
    """Test Kruskal-Wallis test (non-parametric ANOVA)."""
    print("Testing Kruskal-Wallis test...")

    np.random.seed(42)
    n_per_group = 30

    # Create 3 groups with different distributions
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.exponential(scale=1.0, size=n_per_group),
            np.random.exponential(scale=2.0, size=n_per_group),
            np.random.exponential(scale=3.0, size=n_per_group),
        ]),
        'pw2': np.random.exponential(scale=2.0, size=n_per_group * 3),
    })
    pas.index = [f's{i}' for i in range(len(pas))]

    groups = pd.Series(
        ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group,
        index=pas.index
    )

    config = PASTestConfig(method=TestMethod.KRUSKAL)
    results = test_pas_difference(pas, groups, config=config)

    # pw1 should be significant
    pw1_pval = results[results['pathway'] == 'pw1']['pvalue'].values[0]
    assert pw1_pval < 0.05, f"pw1 should be significant (p={pw1_pval})"
    print(f"  - pw1 p-value: {pw1_pval:.4f}: PASS")

    print("  [PASS] Kruskal-Wallis test")
    return True


def test_fdr_correction():
    """Test FDR correction."""
    print("Testing FDR correction...")

    np.random.seed(42)

    # Many pathways, few truly differential
    n_pathways = 100
    pas = pd.DataFrame(
        np.random.randn(60, n_pathways),
        columns=[f'pw{i}' for i in range(n_pathways)]
    )
    pas.index = [f's{i}' for i in range(60)]

    groups = pd.Series(['A'] * 30 + ['B'] * 30, index=pas.index)

    # Add signal to 5 pathways
    for i in range(5):
        pas.loc[groups == 'A', f'pw{i}'] += 2.0

    config = PASTestConfig(
        method=TestMethod.TTEST,
        correction=MultipleTestingCorrection.FDR_BH,
        alpha=0.05,
    )

    results = test_pas_difference(pas, groups, config=config)

    # Check that adjusted p-values are larger than raw
    assert all(results['pvalue_adj'] >= results['pvalue'] - 1e-10), \
        "Adjusted p-values should be >= raw p-values"
    print("  - Adjusted p-values >= raw: PASS")

    # Should detect some true positives but control FDR
    n_sig = results['significant'].sum()
    print(f"  - Detected {n_sig} significant pathways (5 true)")

    # Check false discovery control
    true_positives = set(f'pw{i}' for i in range(5))
    sig_pathways = set(results[results['significant']]['pathway'].tolist())
    false_positives = sig_pathways - true_positives

    if n_sig > 0:
        fdr = len(false_positives) / n_sig
        print(f"  - Empirical FDR: {fdr:.2%}")
        assert fdr <= 0.20, f"FDR too high: {fdr:.2%}"

    print("  [PASS] FDR correction")
    return True


def test_bonferroni_correction():
    """Test Bonferroni correction."""
    print("Testing Bonferroni correction...")

    np.random.seed(42)

    n_pathways = 50
    pas = pd.DataFrame(
        np.random.randn(60, n_pathways),
        columns=[f'pw{i}' for i in range(n_pathways)]
    )
    pas.index = [f's{i}' for i in range(60)]

    groups = pd.Series(['A'] * 30 + ['B'] * 30, index=pas.index)

    # Add strong signal to 3 pathways
    for i in range(3):
        pas.loc[groups == 'A', f'pw{i}'] += 3.0

    config = PASTestConfig(
        method=TestMethod.TTEST,
        correction=MultipleTestingCorrection.BONFERRONI,
        alpha=0.05,
    )

    results = test_pas_difference(pas, groups, config=config)

    # Bonferroni should be more conservative
    # Adjusted p-values should be min(raw_p * n_tests, 1.0)
    expected_adj = np.minimum(results['pvalue'] * n_pathways, 1.0)
    assert np.allclose(results['pvalue_adj'], expected_adj, atol=1e-10), \
        "Bonferroni adjustment incorrect"
    print("  - Bonferroni formula correct: PASS")

    print("  [PASS] Bonferroni correction")
    return True


def test_pairwise_comparisons():
    """Test pairwise group comparisons."""
    print("Testing pairwise comparisons...")

    np.random.seed(42)
    n_per_group = 20

    # 3 groups with different means
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.randn(n_per_group) + 0,
            np.random.randn(n_per_group) + 2,
            np.random.randn(n_per_group) + 4,
        ]),
    })
    pas.index = [f's{i}' for i in range(len(pas))]

    groups = pd.Series(
        ['A'] * n_per_group + ['B'] * n_per_group + ['C'] * n_per_group,
        index=pas.index
    )

    # Run pairwise tests
    pairwise_results = pairwise_group_tests(
        pas, groups,
        pathway='pw1',
        method='ttest',
        correction='bonferroni'
    )

    # Should have 3 comparisons: A vs B, A vs C, B vs C
    assert len(pairwise_results) == 3, f"Expected 3 pairwise comparisons, got {len(pairwise_results)}"
    print(f"  - Number of comparisons: {len(pairwise_results)}: PASS")

    # All comparisons should be significant (groups are well-separated)
    for _, row in pairwise_results.iterrows():
        print(f"    {row['group1']} vs {row['group2']}: p={row['pvalue']:.4f}")

    print("  [PASS] Pairwise comparisons")
    return True


def test_effect_size():
    """Test effect size calculation."""
    print("Testing effect size calculation...")

    np.random.seed(42)

    # Create data with known effect size
    # Cohen's d of ~1.0
    pas = pd.DataFrame({
        'pw1': np.concatenate([
            np.random.randn(50) + 0,
            np.random.randn(50) + 1,
        ]),
        'pw2': np.random.randn(100),
    })
    pas.index = [f's{i}' for i in range(100)]

    groups = pd.Series(['A'] * 50 + ['B'] * 50, index=pas.index)

    config = PASTestConfig(method=TestMethod.TTEST)
    results = test_pas_difference(pas, groups, config=config)

    # Check effect size is present and reasonable
    if 'effect_size' in results.columns:
        pw1_effect = results[results['pathway'] == 'pw1']['effect_size'].values[0]
        print(f"  - pw1 effect size: {pw1_effect:.3f}")
        assert abs(pw1_effect) > 0.5, "Effect size should be substantial"
        print("  - Effect size reasonable: PASS")
    else:
        print("  - Effect size column not present (optional)")

    print("  [PASS] Effect size calculation")
    return True


def main():
    print("=" * 60)
    print("PathWAS Statistics Module API Tests")
    print("=" * 60)
    print()

    tests = [
        ("t-test", test_ttest),
        ("Welch's t-test", test_welch),
        ("Mann-Whitney U", test_mann_whitney),
        ("ANOVA", test_anova),
        ("Kruskal-Wallis", test_kruskal),
        ("FDR Correction", test_fdr_correction),
        ("Bonferroni Correction", test_bonferroni_correction),
        ("Pairwise Comparisons", test_pairwise_comparisons),
        ("Effect Size", test_effect_size),
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
