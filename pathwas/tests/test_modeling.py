"""Tests for the modeling subpackage."""

import numpy as np
import pandas as pd
import pytest

from pathwas.modeling import (
    ModelConfig,
    PathwayModel,
    RidgePathwayModel,
    create_model,
    available_models,
    register_model,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        config = ModelConfig()
        assert config.lambda_ == 1.0
        assert config.model_name == "ridge"
        assert config.extra_params == {}

    def test_custom_config(self):
        config = ModelConfig(lambda_=0.5, model_name="custom")
        assert config.lambda_ == 0.5
        assert config.model_name == "custom"


class TestCreateModel:
    """Tests for model factory."""

    def test_create_ridge_model(self):
        model = create_model("ridge", lambda_=0.1)
        assert isinstance(model, RidgePathwayModel)
        assert model.config.lambda_ == 0.1

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown_model")

    def test_available_models(self):
        models = available_models()
        assert "ridge" in models


class TestRidgePathwayModel:
    """Tests for RidgePathwayModel."""

    @pytest.fixture
    def simple_data(self):
        """Create simple simulated data with known SNP effects."""
        np.random.seed(42)
        n_samples = 100
        n_snps = 20
        n_pathways = 3

        # Standardized genotype matrix
        X = np.random.randn(n_samples, n_snps)
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

        # True SNP effects for each pathway
        true_betas = {
            "pw1": np.random.randn(n_snps) * 0.5,
            "pw2": np.random.randn(n_snps) * 0.3,
            "pw3": np.random.randn(n_snps) * 0.1,
        }

        # Generate PAS with noise
        pas_data = {}
        for pw, beta in true_betas.items():
            signal = X @ beta
            noise = np.random.randn(n_samples) * 0.5
            pas_data[pw] = signal + noise

        pas_matrix = pd.DataFrame(
            pas_data,
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        snp_ids = [f"rs{i}" for i in range(n_snps)]

        return pas_matrix, X, snp_ids, true_betas

    def test_fit_without_covariates(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        model.fit(pas_matrix, X, snp_ids)

        assert model._fitted
        assert len(model._pathways) == 3
        assert len(model._snp_ids) == 20

    def test_get_weights_shape(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        model.fit(pas_matrix, X, snp_ids)

        weights = model.get_weights()

        # Should have pathway * snp rows
        assert len(weights) == 3 * 20
        assert set(weights.columns) >= {"pathway", "snp_id", "beta_pas", "model_name", "lambda"}

    def test_get_metrics(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        model.fit(pas_matrix, X, snp_ids)

        metrics = model.get_metrics()

        assert len(metrics) == 3  # One row per pathway
        assert set(metrics.columns) >= {"pathway", "model_name", "n_snps", "n_samples", "lambda", "r_squared"}

        # R² should be finite when there's signal
        for r2 in metrics["r_squared"]:
            assert np.isfinite(r2)

    def test_fit_with_covariates(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        # Create simple covariates
        covariates = pd.DataFrame(
            {
                "age": np.random.randn(len(pas_matrix)),
                "sex": np.random.choice([0, 1], len(pas_matrix)),
            },
            index=pas_matrix.index,
        )

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        model.fit(pas_matrix, X, snp_ids, covariates)

        assert model._fitted

    def test_sample_mismatch_raises(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        # Create misaligned data
        X_wrong = X[:50]  # Different number of samples

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        with pytest.raises(ValueError, match="Sample count mismatch"):
            model.fit(pas_matrix, X_wrong, snp_ids)

    def test_snp_id_mismatch_raises(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        wrong_snp_ids = snp_ids[:10]  # Wrong number of SNP IDs

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        with pytest.raises(ValueError, match="SNP ID count"):
            model.fit(pas_matrix, X, wrong_snp_ids)

    def test_predict(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = RidgePathwayModel(ModelConfig(lambda_=0.1))
        model.fit(pas_matrix, X, snp_ids)

        # Predict on new samples
        X_new = np.random.randn(10, 20)
        X_new = (X_new - X_new.mean(axis=0)) / X_new.std(axis=0, ddof=0)

        predictions = model.predict(X_new)

        assert predictions.shape == (10, 3)
        assert set(predictions.columns) == set(pas_matrix.columns)

    def test_unfitted_model_raises(self):
        model = RidgePathwayModel(ModelConfig())

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_weights()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_metrics()

    def test_recovery_of_true_effects(self, simple_data):
        """Test that ridge recovers approximately correct effects."""
        pas_matrix, X, snp_ids, true_betas = simple_data

        # Use small regularization
        model = RidgePathwayModel(ModelConfig(lambda_=0.01))
        model.fit(pas_matrix, X, snp_ids)

        weights = model.get_weights()

        # Check correlation between estimated and true effects for pw1
        pw1_weights = weights[weights["pathway"] == "pw1"].set_index("snp_id")["beta_pas"]
        pw1_true = pd.Series(true_betas["pw1"], index=snp_ids)

        # Should have positive correlation (effects in right direction)
        corr = pw1_weights.corr(pw1_true)
        assert corr > 0.5, f"Correlation too low: {corr}"
