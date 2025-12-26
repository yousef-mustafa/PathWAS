## ------------------------------------------------------------------------------------------- ##
## Bayesian Mixture-of-Normals SNP-to-PAS Model                                               ##
## ------------------------------------------------------------------------------------------- ##
## @script: bayes_mixture.py                                                                  ##
##                                                                                            ##
## @description: Implements a Bayesian mixture-of-normals linear model for SNP→PAS effects.  ##
##               Uses an approximate EM procedure to estimate posterior mean effects and     ##
##               posterior inclusion probabilities (PIPs).                                   ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Bayesian mixture-of-normals model for SNP-to-PAS effects."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ModelConfig, PathwayModel


@dataclass
class BayesMixtureConfig:
    """Configuration for Bayesian mixture model.

    Attributes
    ----------
    tau_grid : list of float
        Variance scales for mixture components. tau_grid[0] should be 0 (spike).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for ELBO change.
    init_pi : list of float, optional
        Initial mixture weights. If None, uses uniform.
    """

    tau_grid: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01, 0.1, 1.0])
    max_iter: int = 100
    tol: float = 1e-4
    init_pi: Optional[List[float]] = None


class BayesMixturePathwayModel(PathwayModel):
    """Bayesian mixture-of-normals model for computing SNP-to-PAS weights.

    This model places a mixture-of-normals prior on SNP effects:
        beta_j ~ sum_k pi_k * Normal(0, tau_k^2)
    where component k=0 has tau_0=0 (spike at zero for sparsity).

    The fitting procedure uses an approximate EM algorithm:
    1. E-step: Compute posterior responsibilities for each SNP-component pair.
    2. M-step: Update beta, pi, and sigma^2.

    Parameters
    ----------
    config : ModelConfig
        Configuration with extra_params containing BayesMixtureConfig settings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config.model_name = "bayes_mixture"

        # Parse extra params for mixture config
        extra = config.extra_params or {}
        self._mixture_config = BayesMixtureConfig(
            tau_grid=extra.get("tau_grid", [0.0, 0.001, 0.01, 0.1, 1.0]),
            max_iter=extra.get("max_iter", 100),
            tol=extra.get("tol", 1e-4),
            init_pi=extra.get("init_pi", None),
        )

        self._weights: Dict[str, np.ndarray] = {}
        self._pips: Dict[str, np.ndarray] = {}
        self._metrics: Dict[str, Dict] = {}
        self._snp_ids: List[str] = []
        self._pathways: List[str] = []
        self._n_samples: int = 0
        self._n_snps: int = 0

    def fit(
        self,
        pas_matrix: pd.DataFrame,
        genotype_matrix: np.ndarray,
        snp_ids: List[str],
        covariates: Optional[pd.DataFrame] = None,
    ) -> "BayesMixturePathwayModel":
        """Train Bayesian mixture models for all pathways.

        Parameters
        ----------
        pas_matrix : pd.DataFrame
            PAS matrix (samples x pathways).
        genotype_matrix : np.ndarray
            Standardized genotype matrix (N x M).
        snp_ids : list of str
            SNP identifiers for genotype columns.
        covariates : pd.DataFrame, optional
            Covariates for residualization.

        Returns
        -------
        BayesMixturePathwayModel
            Fitted model.
        """
        self._validate_inputs(pas_matrix, genotype_matrix, snp_ids, covariates)

        self._snp_ids = list(snp_ids)
        self._pathways = list(pas_matrix.columns)
        self._n_samples, self._n_snps = genotype_matrix.shape

        logging.info(
            "Fitting Bayesian mixture model: %d samples, %d SNPs, %d pathways",
            self._n_samples,
            self._n_snps,
            len(self._pathways),
        )

        # Step 1: Residualize PAS on covariates
        residual_pas = self._residualize(pas_matrix, covariates)

        # Prepare genotype matrix
        X = genotype_matrix.astype(np.float64)

        # Precompute X^T X diagonal for efficiency
        xtx_diag = np.sum(X ** 2, axis=0)

        # Step 2: Fit model for each pathway
        for pathway in self._pathways:
            y = residual_pas[pathway].values.astype(np.float64)

            beta, pip, metrics = self._fit_single_pathway(X, y, xtx_diag)

            self._weights[pathway] = beta
            self._pips[pathway] = pip
            self._metrics[pathway] = metrics

            logging.debug(
                "Pathway %s: R² = %.4f, sparsity = %.4f",
                pathway,
                metrics["r_squared"],
                metrics["sparsity"],
            )

        self._fitted = True
        logging.info("Bayesian mixture model fitting complete")
        return self

    def _fit_single_pathway(
        self,
        X: np.ndarray,
        y: np.ndarray,
        xtx_diag: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fit the mixture model for a single pathway.

        Parameters
        ----------
        X : np.ndarray
            Genotype matrix (N x M).
        y : np.ndarray
            Residualized PAS values (length N).
        xtx_diag : np.ndarray
            Diagonal of X^T X (length M).

        Returns
        -------
        tuple
            (beta, pip, metrics) - posterior mean effects, PIPs, and metrics.
        """
        N, M = X.shape
        tau_grid = np.array(self._mixture_config.tau_grid)
        K = len(tau_grid)

        # Initialize
        beta = np.zeros(M)
        sigma2 = np.var(y) if np.var(y) > 0 else 1.0

        if self._mixture_config.init_pi is not None:
            pi = np.array(self._mixture_config.init_pi)
        else:
            # Slightly favor spike component
            pi = np.ones(K) / K
            pi[0] = 0.5
            pi = pi / pi.sum()

        # Responsibilities: gamma[j, k] = P(component k | beta_j)
        gamma = np.zeros((M, K))

        # EM iterations
        for iteration in range(self._mixture_config.max_iter):
            beta_old = beta.copy()

            # E-step: compute responsibilities
            gamma = self._compute_responsibilities(beta, sigma2, tau_grid, pi)

            # M-step: update beta using coordinate descent
            beta = self._update_beta(X, y, beta, gamma, tau_grid, sigma2, xtx_diag)

            # M-step: update pi
            pi = gamma.mean(axis=0)
            pi = np.clip(pi, 1e-10, 1.0)
            pi = pi / pi.sum()

            # M-step: update sigma^2
            residuals = y - X @ beta
            sigma2 = np.sum(residuals ** 2) / N
            sigma2 = max(sigma2, 1e-10)

            # Check convergence
            beta_change = np.max(np.abs(beta - beta_old))
            if beta_change < self._mixture_config.tol:
                logging.debug("Converged at iteration %d", iteration + 1)
                break

        # Compute final PIPs: 1 - P(component 0)
        gamma_final = self._compute_responsibilities(beta, sigma2, tau_grid, pi)
        pip = 1.0 - gamma_final[:, 0]

        # Compute metrics
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        sparsity = np.mean(pip < 0.5)  # Fraction with PIP < 0.5

        metrics = {
            "model_name": "bayes_mixture",
            "n_snps": M,
            "n_samples": N,
            "r_squared": r_squared,
            "sparsity": sparsity,
            "sigma2": sigma2,
            "n_iterations": iteration + 1,
        }

        return beta, pip, metrics

    def _compute_responsibilities(
        self,
        beta: np.ndarray,
        sigma2: float,
        tau_grid: np.ndarray,
        pi: np.ndarray,
    ) -> np.ndarray:
        """Compute posterior responsibilities for component membership.

        Parameters
        ----------
        beta : np.ndarray
            Current SNP effects (length M).
        sigma2 : float
            Residual variance.
        tau_grid : np.ndarray
            Variance scales for each component.
        pi : np.ndarray
            Mixture weights.

        Returns
        -------
        np.ndarray
            Responsibilities matrix (M x K).
        """
        M = len(beta)
        K = len(tau_grid)
        log_gamma = np.zeros((M, K))

        for k in range(K):
            tau2_k = tau_grid[k]

            if tau2_k == 0:
                # Spike component: only assign high prob if beta is near zero
                # Use a very small variance for numerical stability
                log_gamma[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (beta ** 2) / 1e-10
            else:
                # Normal component
                log_gamma[:, k] = (
                    np.log(pi[k] + 1e-300)
                    - 0.5 * np.log(2 * np.pi * tau2_k)
                    - 0.5 * (beta ** 2) / tau2_k
                )

        # Normalize using log-sum-exp trick
        log_gamma_max = np.max(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma - log_gamma_max)
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-300)

        return gamma

    def _update_beta(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        tau_grid: np.ndarray,
        sigma2: float,
        xtx_diag: np.ndarray,
    ) -> np.ndarray:
        """Update beta using coordinate descent with mixture prior.

        Parameters
        ----------
        X : np.ndarray
            Genotype matrix (N x M).
        y : np.ndarray
            Residualized PAS values.
        beta : np.ndarray
            Current SNP effects.
        gamma : np.ndarray
            Responsibilities (M x K).
        tau_grid : np.ndarray
            Variance scales.
        sigma2 : float
            Residual variance.
        xtx_diag : np.ndarray
            Diagonal of X^T X.

        Returns
        -------
        np.ndarray
            Updated beta values.
        """
        M = len(beta)
        beta_new = beta.copy()

        # Compute expected prior precision for each SNP
        # E[1/tau^2] under the mixture
        expected_precision = np.zeros(M)
        for k, tau2_k in enumerate(tau_grid):
            if tau2_k > 0:
                expected_precision += gamma[:, k] / tau2_k
            else:
                # Spike: use very large precision
                expected_precision += gamma[:, k] * 1e10

        # Coordinate descent
        residual = y - X @ beta_new

        for j in range(M):
            # Add back contribution of current beta_j
            residual = residual + X[:, j] * beta_new[j]

            # Compute posterior mean for beta_j
            xj_residual = X[:, j] @ residual
            denominator = xtx_diag[j] + sigma2 * expected_precision[j]

            if denominator > 1e-10:
                beta_new[j] = xj_residual / denominator
            else:
                beta_new[j] = 0.0

            # Update residual
            residual = residual - X[:, j] * beta_new[j]

        return beta_new

    def _residualize(
        self, pas_matrix: pd.DataFrame, covariates: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Residualize PAS on covariates using OLS with intercept.

        Parameters
        ----------
        pas_matrix : pd.DataFrame
            Original PAS matrix.
        covariates : pd.DataFrame, optional
            Covariate matrix.

        Returns
        -------
        pd.DataFrame
            Residualized PAS matrix.
        """
        if covariates is None or covariates.empty:
            # No covariates - just demean
            return pas_matrix - pas_matrix.mean()

        # Build design matrix with intercept
        C = covariates.values.astype(np.float64)
        ones = np.ones((C.shape[0], 1))
        X_cov = np.hstack([ones, C])

        residuals = {}
        for pathway in pas_matrix.columns:
            y = pas_matrix[pathway].values.astype(np.float64)
            # OLS: beta = (X'X)^-1 X'y
            beta_cov, _, _, _ = np.linalg.lstsq(X_cov, y, rcond=None)
            y_pred = X_cov @ beta_cov
            residuals[pathway] = y - y_pred

        return pd.DataFrame(residuals, index=pas_matrix.index)

    def get_weights(self) -> pd.DataFrame:
        """Return SNP-to-PAS weights in long format.

        Returns
        -------
        pd.DataFrame
            Columns: pathway, snp_id, beta_pas, model_name, pip.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting weights")

        records = []
        for pathway in self._pathways:
            beta_p = self._weights[pathway]
            pip_p = self._pips[pathway]
            for i, snp_id in enumerate(self._snp_ids):
                records.append(
                    {
                        "pathway": pathway,
                        "snp_id": snp_id,
                        "beta_pas": beta_p[i],
                        "model_name": self.config.model_name,
                        "pip": pip_p[i],
                    }
                )
        return pd.DataFrame(records)

    def get_metrics(self) -> pd.DataFrame:
        """Return per-pathway model metrics.

        Returns
        -------
        pd.DataFrame
            Columns: pathway, model_name, n_snps, n_samples, r_squared, sparsity.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting metrics")

        records = []
        for pathway in self._pathways:
            m = self._metrics[pathway]
            records.append(
                {
                    "pathway": pathway,
                    "model_name": m["model_name"],
                    "n_snps": m["n_snps"],
                    "n_samples": m["n_samples"],
                    "r_squared": m["r_squared"],
                    "sparsity": m["sparsity"],
                    "sigma2": m["sigma2"],
                    "n_iterations": m["n_iterations"],
                }
            )
        return pd.DataFrame(records)

    def get_pips(self) -> pd.DataFrame:
        """Return posterior inclusion probabilities.

        Returns
        -------
        pd.DataFrame
            DataFrame with pathways as columns, SNPs as rows.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting PIPs")

        pip_data = {pathway: self._pips[pathway] for pathway in self._pathways}
        return pd.DataFrame(pip_data, index=self._snp_ids)

    def predict(self, genotype_matrix: np.ndarray) -> pd.DataFrame:
        """Predict PAS values from genotypes.

        Parameters
        ----------
        genotype_matrix : np.ndarray
            Standardized genotype matrix (N_new x M).

        Returns
        -------
        pd.DataFrame
            Predicted PAS values (N_new x n_pathways).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if genotype_matrix.shape[1] != self._n_snps:
            raise ValueError(
                f"Genotype matrix has {genotype_matrix.shape[1]} SNPs, "
                f"expected {self._n_snps}"
            )

        predictions = {}
        for pathway in self._pathways:
            beta_p = self._weights[pathway]
            predictions[pathway] = genotype_matrix @ beta_p

        return pd.DataFrame(predictions)
