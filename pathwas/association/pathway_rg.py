## ------------------------------------------------------------------------------------------- ##
## Pathway-Level Genetic Correlation                                                          ##
## ------------------------------------------------------------------------------------------- ##
## @script: pathway_rg.py                                                                     ##
##                                                                                            ##
## @description: Computes true genetic correlation between PAS genetics and trait genetics   ##
##               per pathway using SNP effects and LD structure.                             ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway-level genetic correlation estimation with jackknife standard errors."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class GeneticCorrelationResult:
    """Result container for genetic correlation estimation.

    Attributes
    ----------
    rg : float
        Genetic correlation estimate.
    se : float
        Standard error (from jackknife, if computed).
    z : float
        Z-score (rg / se).
    p : float
        Two-sided p-value.
    n_snps : int
        Number of SNPs used.
    n_blocks_used : int
        Number of blocks used in jackknife (0 if not computed).
    cov_g : float
        Genetic covariance.
    var_g_pas : float
        Genetic variance of PAS.
    var_g_trait : float
        Genetic variance of trait.
    """

    rg: float
    se: float
    z: float
    p: float
    n_snps: int
    n_blocks_used: int
    cov_g: float
    var_g_pas: float
    var_g_trait: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "rg": self.rg,
            "se": self.se,
            "z": self.z,
            "p": self.p,
            "n_snps": self.n_snps,
            "n_blocks_used": self.n_blocks_used,
            "cov_g": self.cov_g,
            "var_g_pas": self.var_g_pas,
            "var_g_trait": self.var_g_trait,
        }


def compute_ld_matrix(genotypes: np.ndarray) -> np.ndarray:
    """Compute LD correlation matrix from standardized genotypes.

    Parameters
    ----------
    genotypes : np.ndarray
        Standardized genotype matrix (N samples x M SNPs).
        Assumed to have mean 0 and variance 1 per SNP.

    Returns
    -------
    np.ndarray
        LD correlation matrix R of shape (M x M).
        R = X^T X / (N - 1)
    """
    n_samples = genotypes.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute LD")

    R = genotypes.T @ genotypes / (n_samples - 1)
    return R


def compute_genetic_correlation(
    beta_p: Union[pd.Series, np.ndarray],
    gamma: Union[pd.Series, np.ndarray],
    R: np.ndarray,
) -> float:
    """Compute genetic correlation between PAS and trait.

    The genetic correlation is defined as:
        rg = (beta_p^T R gamma) / sqrt((beta_p^T R beta_p)(gamma^T R gamma))

    Parameters
    ----------
    beta_p : pd.Series or np.ndarray
        SNP-to-PAS effects for pathway p.
    gamma : pd.Series or np.ndarray
        SNP-to-trait effects.
    R : np.ndarray
        LD correlation matrix (M x M).

    Returns
    -------
    float
        Genetic correlation. Returns NaN if variances are zero or negative.
    """
    # Convert to numpy arrays if needed
    if isinstance(beta_p, pd.Series):
        beta_p = beta_p.values
    if isinstance(gamma, pd.Series):
        gamma = gamma.values

    beta_p = np.asarray(beta_p, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)

    # Validate dimensions
    m = len(beta_p)
    if len(gamma) != m:
        raise ValueError(f"beta_p length ({m}) != gamma length ({len(gamma)})")
    if R.shape != (m, m):
        raise ValueError(f"R shape {R.shape} does not match SNP count {m}")

    # Compute genetic covariance and variances
    cov_g = float(beta_p @ R @ gamma)
    var_g_pas = float(beta_p @ R @ beta_p)
    var_g_trait = float(gamma @ R @ gamma)

    # Handle edge cases
    if var_g_pas <= 0 or var_g_trait <= 0:
        logging.warning("Non-positive genetic variance(s): Var_PAS=%.4f, Var_trait=%.4f",
                       var_g_pas, var_g_trait)
        return np.nan

    rg = cov_g / np.sqrt(var_g_pas * var_g_trait)
    return rg


def compute_genetic_correlation_with_components(
    beta_p: Union[pd.Series, np.ndarray],
    gamma: Union[pd.Series, np.ndarray],
    R: np.ndarray,
) -> Dict[str, float]:
    """Compute genetic correlation with variance components.

    Parameters
    ----------
    beta_p : pd.Series or np.ndarray
        SNP-to-PAS effects for pathway p.
    gamma : pd.Series or np.ndarray
        SNP-to-trait effects.
    R : np.ndarray
        LD correlation matrix (M x M).

    Returns
    -------
    dict
        Dictionary with keys: rg, cov_g, var_g_pas, var_g_trait.
    """
    if isinstance(beta_p, pd.Series):
        beta_p = beta_p.values
    if isinstance(gamma, pd.Series):
        gamma = gamma.values

    beta_p = np.asarray(beta_p, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)

    cov_g = float(beta_p @ R @ gamma)
    var_g_pas = float(beta_p @ R @ beta_p)
    var_g_trait = float(gamma @ R @ gamma)

    if var_g_pas <= 0 or var_g_trait <= 0:
        rg = np.nan
    else:
        rg = cov_g / np.sqrt(var_g_pas * var_g_trait)

    return {
        "rg": rg,
        "cov_g": cov_g,
        "var_g_pas": var_g_pas,
        "var_g_trait": var_g_trait,
    }


def jackknife_genetic_correlation(
    beta_p: Union[pd.Series, np.ndarray],
    gamma: Union[pd.Series, np.ndarray],
    R: np.ndarray,
    block_ids: np.ndarray,
    min_snps_per_block: int = 5,
) -> GeneticCorrelationResult:
    """Estimate genetic correlation with jackknife standard errors.

    Jackknife procedure:
    1. Compute full rg using all SNPs.
    2. For each unique block b:
       - Subset SNPs where block_ids != b.
       - Recompute rg_b on that subset.
    3. Compute jackknife variance:
       Var(rg) = (B-1)/B * sum_b (rg_b - mean_rg)^2
       SE = sqrt(Var)
       Z = rg_full / SE
       p = two-sided normal p-value

    Parameters
    ----------
    beta_p : pd.Series or np.ndarray
        SNP-to-PAS effects.
    gamma : pd.Series or np.ndarray
        SNP-to-trait effects.
    R : np.ndarray
        LD correlation matrix (M x M).
    block_ids : np.ndarray
        Block ID for each SNP (length M).
    min_snps_per_block : int
        Minimum SNPs remaining after dropping a block to compute rg.

    Returns
    -------
    GeneticCorrelationResult
        Result container with rg, se, z, p, and metadata.
    """
    if isinstance(beta_p, pd.Series):
        beta_p = beta_p.values
    if isinstance(gamma, pd.Series):
        gamma = gamma.values

    beta_p = np.asarray(beta_p, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    block_ids = np.asarray(block_ids)

    m = len(beta_p)
    if len(gamma) != m or len(block_ids) != m:
        raise ValueError("beta_p, gamma, and block_ids must have the same length")
    if R.shape != (m, m):
        raise ValueError(f"R shape {R.shape} does not match SNP count {m}")

    # Compute full rg
    full_result = compute_genetic_correlation_with_components(beta_p, gamma, R)
    rg_full = full_result["rg"]

    # Get unique blocks
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    # Jackknife: leave one block out at a time
    rg_jackknife = []
    valid_blocks = 0

    for block in unique_blocks:
        mask = block_ids != block
        n_remaining = mask.sum()

        if n_remaining < min_snps_per_block:
            logging.debug("Block %s: only %d SNPs remain, skipping", block, n_remaining)
            continue

        # Subset data
        beta_sub = beta_p[mask]
        gamma_sub = gamma[mask]
        R_sub = R[np.ix_(mask, mask)]

        # Compute rg for subset
        rg_b = compute_genetic_correlation(beta_sub, gamma_sub, R_sub)
        if not np.isnan(rg_b):
            rg_jackknife.append(rg_b)
            valid_blocks += 1

    # Compute jackknife variance and SE
    if valid_blocks < 2:
        logging.warning("Too few valid blocks (%d) for jackknife SE estimation", valid_blocks)
        se = np.nan
        z = np.nan
        p = np.nan
    else:
        rg_array = np.array(rg_jackknife)
        mean_rg = rg_array.mean()
        B = len(rg_array)

        # Jackknife variance: (B-1)/B * sum((rg_b - mean)^2)
        var_rg = (B - 1) / B * np.sum((rg_array - mean_rg) ** 2)
        se = np.sqrt(var_rg)

        if se > 0 and not np.isnan(rg_full):
            z = rg_full / se
            # Two-sided p-value
            p = 2 * scipy_stats.norm.sf(abs(z))
        else:
            z = np.nan
            p = np.nan

    return GeneticCorrelationResult(
        rg=rg_full,
        se=se,
        z=z,
        p=p,
        n_snps=m,
        n_blocks_used=valid_blocks,
        cov_g=full_result["cov_g"],
        var_g_pas=full_result["var_g_pas"],
        var_g_trait=full_result["var_g_trait"],
    )


def compute_pathway_rg_for_all(
    weights_df: pd.DataFrame,
    gamma: pd.Series,
    R: np.ndarray,
    block_ids: Optional[np.ndarray] = None,
    pathway_col: str = "pathway",
    snp_col: str = "snp_id",
    beta_col: str = "beta_pas",
) -> pd.DataFrame:
    """Compute genetic correlations for all pathways.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Long-format weights with pathway, snp_id, beta_pas columns.
    gamma : pd.Series
        SNP-to-trait effects indexed by SNP ID.
    R : np.ndarray
        LD correlation matrix ordered as gamma.index.
    block_ids : np.ndarray, optional
        Block IDs for jackknife. If None, no SE is computed.
    pathway_col : str
        Column name for pathway identifier.
    snp_col : str
        Column name for SNP identifier.
    beta_col : str
        Column name for PAS effect.

    Returns
    -------
    pd.DataFrame
        Results for all pathways with rg, se, z, p, n_snps.
    """
    snp_order = gamma.index.tolist()
    pathways = weights_df[pathway_col].unique()
    results = []

    for pathway in pathways:
        pw_mask = weights_df[pathway_col] == pathway
        pw_weights = weights_df[pw_mask].set_index(snp_col)[beta_col]

        # Align to gamma index
        pw_weights = pw_weights.reindex(snp_order, fill_value=0.0)
        beta_p = pw_weights.values

        if block_ids is not None:
            result = jackknife_genetic_correlation(beta_p, gamma.values, R, block_ids)
        else:
            rg = compute_genetic_correlation(beta_p, gamma.values, R)
            result = GeneticCorrelationResult(
                rg=rg,
                se=np.nan,
                z=np.nan,
                p=np.nan,
                n_snps=len(gamma),
                n_blocks_used=0,
                cov_g=np.nan,
                var_g_pas=np.nan,
                var_g_trait=np.nan,
            )

        row = result.to_dict()
        row["pathway"] = pathway
        results.append(row)

    return pd.DataFrame(results)
