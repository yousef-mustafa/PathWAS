## ------------------------------------------------------------------------------------------- ##
## Ancestry / LD Mismatch QC                                                                  ##
## ------------------------------------------------------------------------------------------- ##
## @script: ancestry_mismatch.py                                                              ##
##                                                                                            ##
## @description: Quality control utilities for detecting ancestry and LD mismatches between  ##
##               training panel, LD reference, and GWAS cohorts.                             ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Ancestry and LD mismatch QC utilities."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class ReliabilityLevel(Enum):
    """Reliability classification for ancestry/LD match."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class AncestryQCResult:
    """Results from ancestry/LD mismatch QC.

    Attributes
    ----------
    maf_corr_training_ldref : float
        Pearson correlation between training panel and LD reference MAFs.
    maf_corr_training_gwas : float
        Pearson correlation between training panel and GWAS MAFs (NaN if unavailable).
    maf_corr_ldref_gwas : float
        Pearson correlation between LD reference and GWAS MAFs (NaN if unavailable).
    ld_concordance : float
        Correlation between training and LD reference pairwise r² values.
    n_snps_compared : int
        Number of SNPs used in MAF comparison.
    n_ld_pairs_compared : int
        Number of SNP pairs used in LD comparison.
    reliability : ReliabilityLevel
        Overall reliability classification.
    """

    maf_corr_training_ldref: float
    maf_corr_training_gwas: float
    maf_corr_ldref_gwas: float
    ld_concordance: float
    n_snps_compared: int
    n_ld_pairs_compared: int
    reliability: ReliabilityLevel

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "maf_corr_training_ldref": self.maf_corr_training_ldref,
            "maf_corr_training_gwas": self.maf_corr_training_gwas,
            "maf_corr_ldref_gwas": self.maf_corr_ldref_gwas,
            "ld_concordance": self.ld_concordance,
            "n_snps_compared": self.n_snps_compared,
            "n_ld_pairs_compared": self.n_ld_pairs_compared,
            "reliability": self.reliability.value,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Ancestry/LD Mismatch QC Summary",
            f"=" * 40,
            f"MAF correlation (training vs LD ref): {self.maf_corr_training_ldref:.4f}",
            f"MAF correlation (training vs GWAS):   {self.maf_corr_training_gwas:.4f}",
            f"MAF correlation (LD ref vs GWAS):     {self.maf_corr_ldref_gwas:.4f}",
            f"LD concordance:                       {self.ld_concordance:.4f}",
            f"SNPs compared:                        {self.n_snps_compared}",
            f"LD pairs compared:                    {self.n_ld_pairs_compared}",
            f"Reliability:                          {self.reliability.value.upper()}",
        ]
        return "\n".join(lines)


def compute_maf_correlation(
    maf1: pd.Series,
    maf2: pd.Series,
) -> Tuple[float, int]:
    """Compute Pearson correlation between two MAF vectors.

    Parameters
    ----------
    maf1 : pd.Series
        MAF values indexed by SNP ID.
    maf2 : pd.Series
        MAF values indexed by SNP ID.

    Returns
    -------
    Tuple[float, int]
        (correlation, n_snps) - Pearson r and number of overlapping SNPs.
    """
    # Align on common SNPs
    common_snps = list(set(maf1.index) & set(maf2.index))
    if len(common_snps) < 3:
        return np.nan, len(common_snps)

    m1 = maf1.reindex(common_snps).values
    m2 = maf2.reindex(common_snps).values

    # Remove NaNs
    valid = ~(np.isnan(m1) | np.isnan(m2))
    if valid.sum() < 3:
        return np.nan, valid.sum()

    r, _ = scipy_stats.pearsonr(m1[valid], m2[valid])
    return r, int(valid.sum())


def compute_ld_concordance(
    R_training: np.ndarray,
    R_ldref: np.ndarray,
    snp_ids_training: List[str],
    snp_ids_ldref: List[str],
    max_pairs: int = 10000,
) -> Tuple[float, int]:
    """Compute concordance between training and LD reference pairwise r².

    Parameters
    ----------
    R_training : np.ndarray
        LD correlation matrix from training panel.
    R_ldref : np.ndarray
        LD correlation matrix from LD reference.
    snp_ids_training : List[str]
        SNP IDs in R_training order.
    snp_ids_ldref : List[str]
        SNP IDs in R_ldref order.
    max_pairs : int
        Maximum number of pairs to compare (for efficiency).

    Returns
    -------
    Tuple[float, int]
        (correlation, n_pairs) - Correlation between pairwise r² values.
    """
    # Find common SNPs
    common_snps = list(set(snp_ids_training) & set(snp_ids_ldref))
    if len(common_snps) < 3:
        return np.nan, 0

    # Get indices for each matrix
    idx_training = [snp_ids_training.index(s) for s in common_snps]
    idx_ldref = [snp_ids_ldref.index(s) for s in common_snps]

    # Subset matrices
    R_train_sub = R_training[np.ix_(idx_training, idx_training)]
    R_ldref_sub = R_ldref[np.ix_(idx_ldref, idx_ldref)]

    # Extract upper triangle (excluding diagonal)
    n = len(common_snps)
    triu_idx = np.triu_indices(n, k=1)

    r2_training = R_train_sub[triu_idx] ** 2
    r2_ldref = R_ldref_sub[triu_idx] ** 2

    n_pairs = len(r2_training)

    # Subsample if too many pairs
    if n_pairs > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pairs, max_pairs, replace=False)
        r2_training = r2_training[idx]
        r2_ldref = r2_ldref[idx]
        n_pairs = max_pairs

    if n_pairs < 3:
        return np.nan, n_pairs

    # Compute correlation
    r, _ = scipy_stats.pearsonr(r2_training, r2_ldref)
    return r, n_pairs


def classify_reliability(
    maf_corr_training_ldref: float,
    ld_concordance: float,
    maf_threshold_high: float = 0.95,
    maf_threshold_moderate: float = 0.85,
    ld_threshold_high: float = 0.90,
    ld_threshold_moderate: float = 0.75,
) -> ReliabilityLevel:
    """Classify reliability based on MAF and LD concordance.

    Parameters
    ----------
    maf_corr_training_ldref : float
        MAF correlation between training and LD reference.
    ld_concordance : float
        LD concordance measure.
    maf_threshold_high : float
        MAF correlation threshold for HIGH reliability.
    maf_threshold_moderate : float
        MAF correlation threshold for MODERATE reliability.
    ld_threshold_high : float
        LD concordance threshold for HIGH reliability.
    ld_threshold_moderate : float
        LD concordance threshold for MODERATE reliability.

    Returns
    -------
    ReliabilityLevel
        Reliability classification.
    """
    # Handle NaN values
    maf_ok = not np.isnan(maf_corr_training_ldref)
    ld_ok = not np.isnan(ld_concordance)

    if not maf_ok and not ld_ok:
        return ReliabilityLevel.LOW

    # Use available metrics
    if maf_ok and ld_ok:
        if maf_corr_training_ldref >= maf_threshold_high and ld_concordance >= ld_threshold_high:
            return ReliabilityLevel.HIGH
        elif maf_corr_training_ldref >= maf_threshold_moderate and ld_concordance >= ld_threshold_moderate:
            return ReliabilityLevel.MODERATE
        else:
            return ReliabilityLevel.LOW
    elif maf_ok:
        if maf_corr_training_ldref >= maf_threshold_high:
            return ReliabilityLevel.HIGH
        elif maf_corr_training_ldref >= maf_threshold_moderate:
            return ReliabilityLevel.MODERATE
        else:
            return ReliabilityLevel.LOW
    else:  # ld_ok only
        if ld_concordance >= ld_threshold_high:
            return ReliabilityLevel.HIGH
        elif ld_concordance >= ld_threshold_moderate:
            return ReliabilityLevel.MODERATE
        else:
            return ReliabilityLevel.LOW


def run_ancestry_qc(
    maf_training: pd.Series,
    maf_ldref: pd.Series,
    maf_gwas: Optional[pd.Series] = None,
    R_training: Optional[np.ndarray] = None,
    R_ldref: Optional[np.ndarray] = None,
    snp_ids_training: Optional[List[str]] = None,
    snp_ids_ldref: Optional[List[str]] = None,
) -> AncestryQCResult:
    """Run full ancestry/LD mismatch QC.

    Parameters
    ----------
    maf_training : pd.Series
        MAF values from training panel indexed by SNP ID.
    maf_ldref : pd.Series
        MAF values from LD reference indexed by SNP ID.
    maf_gwas : pd.Series, optional
        MAF values from GWAS indexed by SNP ID.
    R_training : np.ndarray, optional
        LD correlation matrix from training panel.
    R_ldref : np.ndarray, optional
        LD correlation matrix from LD reference.
    snp_ids_training : List[str], optional
        SNP IDs in R_training order (required if R_training provided).
    snp_ids_ldref : List[str], optional
        SNP IDs in R_ldref order (required if R_ldref provided).

    Returns
    -------
    AncestryQCResult
        QC results with concordance metrics and reliability flag.
    """
    # MAF correlations
    maf_corr_train_ldref, n_snps = compute_maf_correlation(maf_training, maf_ldref)
    logging.info("MAF correlation (training vs LD ref): %.4f (%d SNPs)",
                maf_corr_train_ldref, n_snps)

    if maf_gwas is not None:
        maf_corr_train_gwas, _ = compute_maf_correlation(maf_training, maf_gwas)
        maf_corr_ldref_gwas, _ = compute_maf_correlation(maf_ldref, maf_gwas)
        logging.info("MAF correlation (training vs GWAS): %.4f", maf_corr_train_gwas)
        logging.info("MAF correlation (LD ref vs GWAS): %.4f", maf_corr_ldref_gwas)
    else:
        maf_corr_train_gwas = np.nan
        maf_corr_ldref_gwas = np.nan

    # LD concordance
    n_ld_pairs = 0
    if R_training is not None and R_ldref is not None:
        if snp_ids_training is None or snp_ids_ldref is None:
            raise ValueError("snp_ids must be provided with LD matrices")
        ld_conc, n_ld_pairs = compute_ld_concordance(
            R_training, R_ldref, snp_ids_training, snp_ids_ldref
        )
        logging.info("LD concordance: %.4f (%d pairs)", ld_conc, n_ld_pairs)
    else:
        ld_conc = np.nan
        logging.info("LD concordance: not computed (matrices not provided)")

    # Classify reliability
    reliability = classify_reliability(maf_corr_train_ldref, ld_conc)
    logging.info("Reliability classification: %s", reliability.value)

    return AncestryQCResult(
        maf_corr_training_ldref=maf_corr_train_ldref,
        maf_corr_training_gwas=maf_corr_train_gwas,
        maf_corr_ldref_gwas=maf_corr_ldref_gwas,
        ld_concordance=ld_conc,
        n_snps_compared=n_snps,
        n_ld_pairs_compared=n_ld_pairs,
        reliability=reliability,
    )
