## ------------------------------------------------------------------------------------------- ##
## Ridge SNP-to-PAS Model                                                                     ##
## ------------------------------------------------------------------------------------------- ##
## @script: ridge.py                                                                          ##
##                                                                                            ##
## @description: Implements ridge regression for modeling SNP effects on pathway             ##
##               activation scores. Uses dual formulation with GRM for efficiency.           ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Ridge regression model for SNP-to-PAS effects."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import linalg

from .base import ModelConfig, PathwayModel


class RidgePathwayModel(PathwayModel):
    """Ridge regression model for computing SNP-to-PAS weights.

    This implementation uses the dual formulation with a genetic relationship
    matrix (GRM) for computational efficiency when the number of SNPs exceeds
    the number of samples.

    The model workflow:
    1. Residualize each PAS column on covariates (OLS with intercept).
    2. Build GRM K = X X^T / P (X = N x P standardized genotypes).
    3. Solve dual ridge: alpha_p = (K + lambda*I)^-1 * r_p for each pathway.
    4. Recover primal coefficients: beta_p = X^T * alpha_p.

    Parameters
    ----------
    config : ModelConfig
        Configuration with at least lambda_ specified.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config.model_name = "ridge"
        self._weights: Dict[str, np.ndarray] = {}
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
    ) -> "RidgePathwayModel":
        """Train ridge models for all pathways.

        Parameters
        ----------
        pas_matrix : pd.DataFrame
            PAS matrix (samples x pathways).
        genotype_matrix : np.ndarray
            Standardized genotype matrix (N x P).
        snp_ids : list of str
            SNP identifiers for genotype columns.
        covariates : pd.DataFrame, optional
            Covariates for residualization.

        Returns
        -------
        RidgePathwayModel
            Fitted model.
        """
        self._validate_inputs(pas_matrix, genotype_matrix, snp_ids, covariates)

        self._snp_ids = list(snp_ids)
        self._pathways = list(pas_matrix.columns)
        self._n_samples, self._n_snps = genotype_matrix.shape

        logging.info(
            "Fitting ridge model: %d samples, %d SNPs, %d pathways, lambda=%.4f",
            self._n_samples,
            self._n_snps,
            len(self._pathways),
            self.config.lambda_,
        )

        # Step 1: Residualize PAS on covariates
        residual_pas = self._residualize(pas_matrix, covariates)

        # Step 2: Build GRM K = X X^T / P
        X = genotype_matrix.astype(np.float64)
        K = X @ X.T / self._n_snps

        # Step 3: Solve dual system (K + lambda*I)^-1
        # Precompute the inverse/factorization to reuse across pathways
        K_reg = K + self.config.lambda_ * np.eye(self._n_samples)
        try:
            # Use Cholesky for symmetric positive definite
            L = linalg.cholesky(K_reg, lower=True)
            use_cholesky = True
        except linalg.LinAlgError:
            # Fall back to general solver if not PD
            logging.warning("K + lambda*I is not positive definite, using LU decomposition")
            lu_piv = linalg.lu_factor(K_reg)
            use_cholesky = False

        # Step 4: For each pathway, solve and compute coefficients
        for pathway in self._pathways:
            r_p = residual_pas[pathway].values.astype(np.float64)

            # Solve (K + lambda*I) * alpha = r
            if use_cholesky:
                alpha_p = linalg.cho_solve((L, True), r_p)
            else:
                alpha_p = linalg.lu_solve(lu_piv, r_p)

            # Recover primal coefficients: beta_p = X^T * alpha_p
            beta_p = X.T @ alpha_p

            # Compute in-sample predictions and R²
            y_pred = X @ beta_p
            ss_res = np.sum((r_p - y_pred) ** 2)
            ss_tot = np.sum((r_p - r_p.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            self._weights[pathway] = beta_p
            self._metrics[pathway] = {
                "model_name": self.config.model_name,
                "n_snps": self._n_snps,
                "n_samples": self._n_samples,
                "lambda": self.config.lambda_,
                "r_squared": r_squared,
            }
            logging.debug("Pathway %s: R² = %.4f", pathway, r_squared)

        self._fitted = True
        logging.info("Ridge model fitting complete")
        return self

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
            Columns: pathway, snp_id, beta_pas, model_name, lambda.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before getting weights")

        records = []
        for pathway in self._pathways:
            beta_p = self._weights[pathway]
            for i, snp_id in enumerate(self._snp_ids):
                records.append(
                    {
                        "pathway": pathway,
                        "snp_id": snp_id,
                        "beta_pas": beta_p[i],
                        "model_name": self.config.model_name,
                        "lambda": self.config.lambda_,
                    }
                )
        return pd.DataFrame(records)

    def get_metrics(self) -> pd.DataFrame:
        """Return per-pathway model metrics.

        Returns
        -------
        pd.DataFrame
            Columns: pathway, model_name, n_snps, n_samples, lambda, r_squared.
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
                    "lambda": m["lambda"],
                    "r_squared": m["r_squared"],
                }
            )
        return pd.DataFrame(records)

    def predict(self, genotype_matrix: np.ndarray) -> pd.DataFrame:
        """Predict PAS values from genotypes.

        Parameters
        ----------
        genotype_matrix : np.ndarray
            Standardized genotype matrix (N_new x P).

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
