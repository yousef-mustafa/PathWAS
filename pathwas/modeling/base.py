## ------------------------------------------------------------------------------------------- ##
## Modeling Base Classes                                                                      ##
## ------------------------------------------------------------------------------------------- ##
## @script: base.py                                                                           ##
##                                                                                            ##
## @description: Defines generic modeling interfaces for SNP-to-PAS models.                  ##
##               PathwayModel provides the abstract base class that all model                ##
##               implementations must follow. ModelConfig holds hyperparameters.             ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Base classes for SNP-to-PAS modeling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Configuration for pathway models.

    Attributes
    ----------
    lambda_ : float
        Ridge regularization parameter (penalty strength).
    model_name : str
        Name identifier for the model type.
    extra_params : dict
        Additional model-specific hyperparameters.
    """

    lambda_: float = 1.0
    model_name: str = "ridge"
    extra_params: Dict[str, Any] = field(default_factory=dict)


class PathwayModel(ABC):
    """Abstract base class for SNP-to-PAS models.

    All pathway model implementations must subclass this and implement
    the fit(), get_weights(), and get_metrics() methods.

    Parameters
    ----------
    config : ModelConfig
        Model configuration containing hyperparameters.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        pas_matrix: pd.DataFrame,
        genotype_matrix: np.ndarray,
        snp_ids: List[str],
        covariates: Optional[pd.DataFrame] = None,
    ) -> "PathwayModel":
        """Train models for all pathways.

        Parameters
        ----------
        pas_matrix : pd.DataFrame
            PAS matrix with samples as rows (index = sample IDs) and
            pathways as columns.
        genotype_matrix : np.ndarray
            Genotype matrix of shape (N, P) where N = samples, P = SNPs.
            Assumed to be standardized (mean 0, var 1 per SNP).
        snp_ids : list of str
            List of SNP identifiers corresponding to columns of genotype_matrix.
        covariates : pd.DataFrame, optional
            Covariate matrix with samples as rows (aligned to pas_matrix).
            If provided, PAS will be residualized on these covariates.

        Returns
        -------
        PathwayModel
            The fitted model (self).
        """
        pass

    @abstractmethod
    def get_weights(self) -> pd.DataFrame:
        """Return SNP-to-PAS weights in long format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: pathway, snp_id, beta_pas, model_name,
            and key hyperparameters (e.g., lambda).
        """
        pass

    @abstractmethod
    def get_metrics(self) -> pd.DataFrame:
        """Return per-pathway model metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: pathway, model_name, n_snps, n_samples,
            lambda, and in-sample performance metric (e.g., R²).
        """
        pass

    def _validate_inputs(
        self,
        pas_matrix: pd.DataFrame,
        genotype_matrix: np.ndarray,
        snp_ids: List[str],
        covariates: Optional[pd.DataFrame] = None,
    ) -> None:
        """Validate input dimensions and alignment.

        Raises
        ------
        ValueError
            If inputs are misaligned or have inconsistent dimensions.
        """
        n_samples_pas = pas_matrix.shape[0]
        n_samples_geno, n_snps = genotype_matrix.shape

        if n_samples_pas != n_samples_geno:
            raise ValueError(
                f"Sample count mismatch: PAS has {n_samples_pas} samples, "
                f"genotype matrix has {n_samples_geno} samples."
            )

        if len(snp_ids) != n_snps:
            raise ValueError(
                f"SNP ID count ({len(snp_ids)}) does not match "
                f"genotype matrix columns ({n_snps})."
            )

        if covariates is not None:
            if covariates.shape[0] != n_samples_pas:
                raise ValueError(
                    f"Covariate sample count ({covariates.shape[0]}) does not match "
                    f"PAS sample count ({n_samples_pas})."
                )
            # Check sample alignment by index
            if not pas_matrix.index.equals(covariates.index):
                raise ValueError(
                    "PAS matrix and covariates have different sample indices."
                )
