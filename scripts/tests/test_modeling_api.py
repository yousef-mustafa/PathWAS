#!/usr/bin/env python
"""
Test pathwas.modeling module API.

This script tests the modeling classes including:
- RidgePathwayModel
- BayesMixturePathwayModel
- ModelConfig
- Model fitting, prediction, and weight extraction

Usage:
    python scripts/tests/test_modeling_api.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathwas.modeling.base import ModelConfig
from pathwas.modeling.ridge import RidgePathwayModel
from pathwas.modeling.bayes_mixture import BayesMixturePathwayModel


def test_ridge_model():
    """Test ridge regression model."""
    print("Testing Ridge model...")

    np.random.seed(42)
    n_samples = 100
    n_snps = 500
    n_pathways = 5

    # Generate synthetic data with known relationship
    genotypes = np.random.randn(n_samples, n_snps)
    true_effects = np.random.randn(n_snps, n_pathways) * 0.1
    noise = np.random.randn(n_samples, n_pathways) * 0.5

    pas = pd.DataFrame(
        genotypes @ true_effects + noise,
        columns=[f'pathway_{i}' for i in range(n_pathways)],
        index=[f'sample_{i}' for i in range(n_samples)]
    )
    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Fit model
    config = ModelConfig(lambda_=0.1)
    model = RidgePathwayModel(config)
    model.fit(pas, genotypes, snp_ids)

    # Check weights output
    weights = model.get_weights()
    assert len(weights) == n_snps * n_pathways, f"Expected {n_snps * n_pathways} weights, got {len(weights)}"
    assert 'beta_pas' in weights.columns, "Missing beta_pas column"
    assert 'pathway' in weights.columns, "Missing pathway column"
    assert 'snp_id' in weights.columns, "Missing snp_id column"
    print(f"  - Weights shape: {len(weights)} entries: PASS")

    # Check metrics output
    metrics = model.get_metrics()
    assert len(metrics) == n_pathways, f"Expected {n_pathways} metrics rows, got {len(metrics)}"
    assert 'r_squared' in metrics.columns, "Missing r_squared column"
    assert 'pathway' in metrics.columns, "Missing pathway column"
    print(f"  - Metrics shape: {len(metrics)} pathways: PASS")

    # Check R² is reasonable
    mean_r2 = metrics['r_squared'].mean()
    print(f"  - Mean R²: {mean_r2:.3f}")
    assert mean_r2 > 0.1, f"R² too low: {mean_r2}"
    assert mean_r2 < 1.0, f"R² too high: {mean_r2}"

    # Check prediction
    predictions = model.predict(genotypes)
    assert predictions.shape == pas.shape, f"Prediction shape mismatch: {predictions.shape} vs {pas.shape}"
    print(f"  - Prediction shape correct: PASS")

    # Verify predictions are correlated with actual PAS
    for pathway in pas.columns:
        corr = np.corrcoef(predictions[pathway], pas[pathway])[0, 1]
        assert corr > 0.3, f"Low correlation for {pathway}: {corr}"
    print("  - Predictions correlated with actual: PASS")

    print("  [PASS] Ridge model")
    return True


def test_ridge_with_covariates():
    """Test ridge model with covariate adjustment."""
    print("Testing Ridge model with covariates...")

    np.random.seed(42)
    n_samples = 100
    n_snps = 200

    genotypes = np.random.randn(n_samples, n_snps)

    # Covariates that affect PAS
    covariates = pd.DataFrame({
        'age': np.random.randn(n_samples) * 10 + 50,
        'sex': np.random.randint(0, 2, n_samples).astype(float),
        'pc1': np.random.randn(n_samples),
        'pc2': np.random.randn(n_samples),
    }, index=[f'sample_{i}' for i in range(n_samples)])

    # PAS affected by both genetics and covariates
    genetic_effect = genotypes @ (np.random.randn(n_snps) * 0.1)
    covariate_effect = covariates['age'] * 0.02 + covariates['sex'] * 0.3
    noise = np.random.randn(n_samples) * 0.5

    pas = pd.DataFrame({
        'pathway_1': genetic_effect + covariate_effect + noise,
    }, index=covariates.index)

    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Fit with covariates
    config = ModelConfig(lambda_=0.1)
    model = RidgePathwayModel(config)
    model.fit(pas, genotypes, snp_ids, covariates=covariates)

    # Check that model fitted
    weights = model.get_weights()
    assert len(weights) > 0, "No weights returned"
    print(f"  - Weights extracted: {len(weights)} entries: PASS")

    metrics = model.get_metrics()
    r2 = metrics['r_squared'].values[0]
    print(f"  - R² with covariates: {r2:.3f}")

    print("  [PASS] Ridge with covariates")
    return True


def test_ridge_lambda_effect():
    """Test effect of lambda on ridge regression."""
    print("Testing lambda effect on Ridge model...")

    np.random.seed(42)
    n_samples = 100
    n_snps = 200

    genotypes = np.random.randn(n_samples, n_snps)
    true_effects = np.random.randn(n_snps) * 0.2
    pas = pd.DataFrame({
        'pathway_1': genotypes @ true_effects + np.random.randn(n_samples) * 0.3,
    }, index=[f'sample_{i}' for i in range(n_samples)])
    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Test with different lambda values
    lambdas = [0.001, 0.1, 1.0, 10.0]
    weight_norms = []

    for lambda_ in lambdas:
        config = ModelConfig(lambda_=lambda_)
        model = RidgePathwayModel(config)
        model.fit(pas, genotypes, snp_ids)

        weights = model.get_weights()
        weight_norm = np.sqrt((weights['beta_pas'] ** 2).sum())
        weight_norms.append(weight_norm)

        metrics = model.get_metrics()
        r2 = metrics['r_squared'].values[0]
        print(f"  - lambda={lambda_}: ||w||={weight_norm:.4f}, R²={r2:.3f}")

    # Higher lambda should give smaller weights
    for i in range(len(lambdas) - 1):
        assert weight_norms[i] >= weight_norms[i+1], \
            f"Weight norm should decrease with lambda: {weight_norms}"

    print("  - Weight norms decrease with lambda: PASS")

    print("  [PASS] Lambda effect")
    return True


def test_bayes_mixture_model():
    """Test Bayesian mixture model."""
    print("Testing Bayesian mixture model...")

    np.random.seed(42)
    n_samples = 100
    n_snps = 200

    # Sparse true effects (most are zero)
    genotypes = np.random.randn(n_samples, n_snps)
    true_effects = np.zeros(n_snps)
    true_effects[:10] = np.random.randn(10) * 0.5  # Only 10 causal SNPs

    pas = pd.DataFrame({
        'pathway_1': genotypes @ true_effects + np.random.randn(n_samples) * 0.3,
    }, index=[f'sample_{i}' for i in range(n_samples)])

    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Fit model
    config = ModelConfig(
        extra_params={
            'tau_grid': [0.0, 0.01, 0.1, 1.0],
            'max_iter': 50,
            'tol': 1e-4,
        }
    )
    model = BayesMixturePathwayModel(config)
    model.fit(pas, genotypes, snp_ids)

    # Check weights output
    weights = model.get_weights()
    assert 'pip' in weights.columns, "Missing PIP column"
    assert 'beta_pas' in weights.columns, "Missing beta_pas column"
    print(f"  - Weights with PIPs: {len(weights)} entries: PASS")

    # Check PIPs
    pips = model.get_pips()
    assert pips.shape == (n_snps, 1), f"PIP shape mismatch: {pips.shape}"
    assert all(pips.values >= 0) and all(pips.values <= 1), "PIPs should be in [0, 1]"
    print(f"  - PIPs in valid range: PASS")

    # Check metrics
    metrics = model.get_metrics()
    assert 'sparsity' in metrics.columns, "Missing sparsity column"
    assert 'r_squared' in metrics.columns, "Missing r_squared column"

    sparsity = metrics['sparsity'].values[0]
    r2 = metrics['r_squared'].values[0]
    print(f"  - Sparsity: {sparsity:.3f}")
    print(f"  - R²: {r2:.3f}")

    # Sparsity should be high (few causal SNPs)
    assert sparsity > 0.8, f"Sparsity too low: {sparsity}"
    print("  - Model identifies sparse structure: PASS")

    print("  [PASS] Bayesian mixture model")
    return True


def test_bayes_pip_ranking():
    """Test that PIPs correctly rank causal SNPs."""
    print("Testing PIP ranking of causal SNPs...")

    np.random.seed(42)
    n_samples = 150
    n_snps = 100

    # Create data with known causal SNPs
    genotypes = np.random.randn(n_samples, n_snps)
    true_effects = np.zeros(n_snps)

    # First 5 SNPs are strongly causal
    causal_indices = [0, 1, 2, 3, 4]
    true_effects[causal_indices] = np.random.uniform(0.3, 0.5, len(causal_indices))

    pas = pd.DataFrame({
        'pathway_1': genotypes @ true_effects + np.random.randn(n_samples) * 0.2,
    }, index=[f'sample_{i}' for i in range(n_samples)])

    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Fit model
    config = ModelConfig(
        extra_params={
            'tau_grid': [0.0, 0.001, 0.01, 0.1, 1.0],
            'max_iter': 100,
        }
    )
    model = BayesMixturePathwayModel(config)
    model.fit(pas, genotypes, snp_ids)

    # Get PIPs
    pips = model.get_pips()

    # Check that causal SNPs have higher PIPs
    causal_pips = pips.iloc[causal_indices, 0].mean()
    non_causal_pips = pips.iloc[5:, 0].mean()

    print(f"  - Mean PIP (causal): {causal_pips:.3f}")
    print(f"  - Mean PIP (non-causal): {non_causal_pips:.3f}")

    assert causal_pips > non_causal_pips, "Causal SNPs should have higher PIPs"
    print("  - Causal SNPs ranked higher: PASS")

    # Check top-ranked SNPs include causal ones
    top_5_indices = pips['pathway_1'].nlargest(5).index.tolist()
    top_5_numeric = [int(idx.replace('rs', '')) for idx in top_5_indices]
    overlap = len(set(top_5_numeric) & set(causal_indices))
    print(f"  - Causal SNPs in top 5: {overlap}/5")
    assert overlap >= 2, f"Too few causal SNPs in top 5: {overlap}"

    print("  [PASS] PIP ranking")
    return True


def test_model_prediction():
    """Test model prediction on new data."""
    print("Testing model prediction...")

    np.random.seed(42)
    n_train = 80
    n_test = 20
    n_snps = 200
    n_pathways = 3

    # Generate training data
    genotypes_train = np.random.randn(n_train, n_snps)
    true_effects = np.random.randn(n_snps, n_pathways) * 0.15

    pas_train = pd.DataFrame(
        genotypes_train @ true_effects + np.random.randn(n_train, n_pathways) * 0.3,
        columns=[f'pathway_{i}' for i in range(n_pathways)],
        index=[f'train_{i}' for i in range(n_train)]
    )
    snp_ids = [f'rs{i}' for i in range(n_snps)]

    # Generate test data (same true effects)
    genotypes_test = np.random.randn(n_test, n_snps)
    pas_test_actual = pd.DataFrame(
        genotypes_test @ true_effects + np.random.randn(n_test, n_pathways) * 0.3,
        columns=[f'pathway_{i}' for i in range(n_pathways)],
        index=[f'test_{i}' for i in range(n_test)]
    )

    # Fit model
    config = ModelConfig(lambda_=0.1)
    model = RidgePathwayModel(config)
    model.fit(pas_train, genotypes_train, snp_ids)

    # Predict on test data
    pas_test_pred = model.predict(genotypes_test)

    # Check prediction shape
    assert pas_test_pred.shape == (n_test, n_pathways), \
        f"Prediction shape wrong: {pas_test_pred.shape}"
    print(f"  - Prediction shape correct: PASS")

    # Check prediction accuracy
    for pathway in pas_train.columns:
        corr = np.corrcoef(pas_test_pred[pathway], pas_test_actual[pathway])[0, 1]
        print(f"  - {pathway} test correlation: {corr:.3f}")
        assert corr > 0.2, f"Low prediction accuracy for {pathway}"

    print("  - Predictions accurate on test set: PASS")

    print("  [PASS] Model prediction")
    return True


def main():
    print("=" * 60)
    print("PathWAS Modeling Module API Tests")
    print("=" * 60)
    print()

    tests = [
        ("Ridge Model", test_ridge_model),
        ("Ridge with Covariates", test_ridge_with_covariates),
        ("Lambda Effect", test_ridge_lambda_effect),
        ("Bayesian Mixture Model", test_bayes_mixture_model),
        ("PIP Ranking", test_bayes_pip_ranking),
        ("Model Prediction", test_model_prediction),
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
