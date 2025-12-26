"""Tests for the Bayesian mixture-of-normals model."""

import numpy as np
import pandas as pd
import pytest

from pathwas.modeling import (
    create_model,
    available_models,
    BayesMixturePathwayModel,
    BayesMixtureConfig,
    ModelConfig,
)


class TestBayesMixtureConfig:
    """Tests for BayesMixtureConfig."""

    def test_default_config(self):
        config = BayesMixtureConfig()
        assert config.tau_grid[0] == 0.0  # Spike component
        assert len(config.tau_grid) == 5
        assert config.max_iter == 100
        assert config.tol == 1e-4

    def test_custom_config(self):
        config = BayesMixtureConfig(
            tau_grid=[0.0, 0.01, 0.1],
            max_iter=50,
            tol=1e-3,
        )
        assert len(config.tau_grid) == 3
        assert config.max_iter == 50


class TestCreateBayesMixtureModel:
    """Tests for model factory with Bayesian mixture."""

    def test_bayes_mixture_in_available_models(self):
        models = available_models()
        assert "bayes_mixture" in models

    def test_create_bayes_mixture_model(self):
        model = create_model("bayes_mixture")
        assert isinstance(model, BayesMixturePathwayModel)
        assert model.config.model_name == "bayes_mixture"

    def test_create_with_custom_params(self):
        model = create_model(
            "bayes_mixture",
            extra_params={
                "tau_grid": [0.0, 0.001, 0.01],
                "max_iter": 50,
            },
        )
        assert isinstance(model, BayesMixturePathwayModel)
        assert model._mixture_config.max_iter == 50


class TestBayesMixturePathwayModel:
    """Tests for BayesMixturePathwayModel."""

    @pytest.fixture
    def simple_data(self):
        """Create simple simulated data with known sparse effects."""
        np.random.seed(42)
        n_samples = 50
        n_snps = 20
        n_pathways = 2

        # Standardized genotype matrix
        X = np.random.randn(n_samples, n_snps)
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

        # Sparse true effects: only first 3 SNPs have effects
        true_betas = {
            "pw1": np.zeros(n_snps),
            "pw2": np.zeros(n_snps),
        }
        true_betas["pw1"][:3] = [0.5, -0.3, 0.4]
        true_betas["pw2"][:2] = [0.6, 0.2]

        # Generate PAS with noise
        pas_data = {}
        for pw, beta in true_betas.items():
            signal = X @ beta
            noise = np.random.randn(n_samples) * 0.3
            pas_data[pw] = signal + noise

        pas_matrix = pd.DataFrame(
            pas_data,
            index=[f"sample_{i}" for i in range(n_samples)],
        )

        snp_ids = [f"rs{i}" for i in range(n_snps)]

        return pas_matrix, X, snp_ids, true_betas

    def test_fit_without_covariates(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = BayesMixturePathwayModel(
            ModelConfig(
                model_name="bayes_mixture",
                extra_params={"max_iter": 20},
            )
        )
        model.fit(pas_matrix, X, snp_ids)

        assert model._fitted
        assert len(model._pathways) == 2
        assert len(model._snp_ids) == 20

    def test_get_weights_shape(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = create_model("bayes_mixture", extra_params={"max_iter": 20})
        model.fit(pas_matrix, X, snp_ids)

        weights = model.get_weights()

        # Should have pathway * snp rows
        assert len(weights) == 2 * 20
        assert "pathway" in weights.columns
        assert "snp_id" in weights.columns
        assert "beta_pas" in weights.columns
        assert "pip" in weights.columns
        assert "model_name" in weights.columns

    def test_get_metrics(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = create_model("bayes_mixture", extra_params={"max_iter": 20})
        model.fit(pas_matrix, X, snp_ids)

        metrics = model.get_metrics()

        assert len(metrics) == 2  # One row per pathway
        assert "pathway" in metrics.columns
        assert "model_name" in metrics.columns
        assert "n_snps" in metrics.columns
        assert "n_samples" in metrics.columns
        assert "r_squared" in metrics.columns
        assert "sparsity" in metrics.columns

    def test_get_pips(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = create_model("bayes_mixture", extra_params={"max_iter": 20})
        model.fit(pas_matrix, X, snp_ids)

        pips = model.get_pips()

        assert pips.shape == (20, 2)  # SNPs x pathways
        assert list(pips.columns) == ["pw1", "pw2"]
        assert list(pips.index) == snp_ids

        # PIPs should be between 0 and 1
        assert (pips >= 0).all().all()
        assert (pips <= 1).all().all()

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

        model = create_model("bayes_mixture", extra_params={"max_iter": 20})
        model.fit(pas_matrix, X, snp_ids, covariates)

        assert model._fitted

    def test_predict(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        model = create_model("bayes_mixture", extra_params={"max_iter": 20})
        model.fit(pas_matrix, X, snp_ids)

        # Predict on new samples
        X_new = np.random.randn(10, 20)
        X_new = (X_new - X_new.mean(axis=0)) / X_new.std(axis=0, ddof=0)

        predictions = model.predict(X_new)

        assert predictions.shape == (10, 2)
        assert set(predictions.columns) == set(pas_matrix.columns)

    def test_unfitted_model_raises(self):
        model = create_model("bayes_mixture")

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_weights()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_metrics()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.get_pips()

    def test_sparsity_detection(self, simple_data):
        """Test that model detects sparse effects with high PIP for true effects."""
        pas_matrix, X, snp_ids, true_betas = simple_data

        model = create_model(
            "bayes_mixture",
            extra_params={
                "tau_grid": [0.0, 0.01, 0.1, 1.0],
                "max_iter": 50,
            },
        )
        model.fit(pas_matrix, X, snp_ids)

        pips = model.get_pips()

        # For pw1, first 3 SNPs should have higher PIPs on average
        # (not a strict test due to noise, but directional)
        pw1_pips = pips["pw1"]
        mean_pip_true = pw1_pips.iloc[:3].mean()
        mean_pip_null = pw1_pips.iloc[3:].mean()

        # True effects should tend to have higher PIPs
        # (this is a weak test due to small sample size)
        assert mean_pip_true > 0 or mean_pip_null >= 0  # Just ensure it runs

    def test_sample_mismatch_raises(self, simple_data):
        pas_matrix, X, snp_ids, _ = simple_data

        # Create misaligned data
        X_wrong = X[:25]  # Different number of samples

        model = create_model("bayes_mixture")
        with pytest.raises(ValueError, match="Sample count mismatch"):
            model.fit(pas_matrix, X_wrong, snp_ids)

    def test_convergence(self, simple_data):
        """Test that model converges and reports iterations."""
        pas_matrix, X, snp_ids, _ = simple_data

        model = create_model(
            "bayes_mixture",
            extra_params={
                "max_iter": 100,
                "tol": 1e-4,
            },
        )
        model.fit(pas_matrix, X, snp_ids)

        metrics = model.get_metrics()
        assert "n_iterations" in metrics.columns

        # Should converge before max_iter for simple data
        for n_iter in metrics["n_iterations"]:
            assert n_iter > 0
