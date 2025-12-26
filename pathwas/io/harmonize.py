## ------------------------------------------------------------------------------------------- ##
## GWAS and PathQTL Harmonization                                                             ##
## ------------------------------------------------------------------------------------------- ##
## @script: harmonize.py                                                                      ##
##                                                                                            ##
## @description: Harmonizes SNP-to-PAS weights (pathQTL) with GWAS summary statistics.       ##
##               Handles allele alignment, flipping, and filtering of palindromic/mismatched ##
##               SNPs.                                                                        ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Harmonization of pathQTL weights with GWAS summary statistics."""

import logging
from typing import Tuple, Set

import pandas as pd


# Palindromic allele pairs
_PALINDROMIC_PAIRS: Set[Tuple[str, str]] = {
    ("A", "T"),
    ("T", "A"),
    ("C", "G"),
    ("G", "C"),
}


def is_palindromic(allele1: str, allele2: str) -> bool:
    """Check if an allele pair is palindromic (strand ambiguous).

    Parameters
    ----------
    allele1 : str
        First allele.
    allele2 : str
        Second allele.

    Returns
    -------
    bool
        True if the allele pair is palindromic (A/T, T/A, C/G, G/C).
    """
    return (allele1.upper(), allele2.upper()) in _PALINDROMIC_PAIRS


def harmonize_weights_gwas(
    weights_df: pd.DataFrame,
    gwas_df: pd.DataFrame,
    snp_col: str = "snp_id",
    effect_allele_pas: str = "effect_allele_pas",
    other_allele_pas: str = "other_allele_pas",
    beta_pas_col: str = "beta_pas",
    pathway_col: str = "pathway",
    effect_allele_gwas: str = "effect_allele_gwas",
    other_allele_gwas: str = "other_allele_gwas",
    beta_trait_col: str = "beta_trait",
    se_col: str = "se",
    z_col: str = "z",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Harmonize pathQTL weights with GWAS summary statistics.

    Performs the following:
    1. Inner-join by SNP ID.
    2. Drop palindromic SNPs (A/T, T/A, C/G, G/C).
    3. For each SNP:
       - If PAS and GWAS alleles match: keep as-is.
       - If reversed (effect <-> other): flip sign of beta_trait (and z if present).
       - Otherwise (mismatched alleles): drop the SNP.

    Parameters
    ----------
    weights_df : pd.DataFrame
        PathQTL weights with columns: snp_id, effect_allele_pas, other_allele_pas,
        beta_pas, pathway.
    gwas_df : pd.DataFrame
        GWAS summary stats with columns: snp_id, effect_allele_gwas, other_allele_gwas,
        beta_trait, and optionally se, z.
    snp_col : str
        Column name for SNP identifier.
    effect_allele_pas : str
        Column name for effect allele in weights.
    other_allele_pas : str
        Column name for other allele in weights.
    beta_pas_col : str
        Column name for PAS effect size.
    pathway_col : str
        Column name for pathway identifier.
    effect_allele_gwas : str
        Column name for effect allele in GWAS.
    other_allele_gwas : str
        Column name for other allele in GWAS.
    beta_trait_col : str
        Column name for trait effect size in GWAS.
    se_col : str
        Column name for standard error in GWAS (optional).
    z_col : str
        Column name for Z-score in GWAS (optional).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (harmonized_weights, harmonized_gwas)
        - harmonized_weights: SNPs retained with pathway, snp_id, beta_pas, effect alleles.
        - harmonized_gwas: Same SNP order, with beta_trait (and z) aligned to same
          effect allele as beta_pas.
    """
    # Validate required columns
    required_weights = [snp_col, effect_allele_pas, other_allele_pas, beta_pas_col, pathway_col]
    for col in required_weights:
        if col not in weights_df.columns:
            raise ValueError(f"weights_df missing required column: {col}")

    required_gwas = [snp_col, effect_allele_gwas, other_allele_gwas, beta_trait_col]
    for col in required_gwas:
        if col not in gwas_df.columns:
            raise ValueError(f"gwas_df missing required column: {col}")

    has_se = se_col in gwas_df.columns
    has_z = z_col in gwas_df.columns

    # Inner join on SNP ID
    merged = weights_df.merge(gwas_df, on=snp_col, how="inner", suffixes=("", "_gwas"))
    logging.info("After inner join: %d SNP-pathway pairs", len(merged))

    if merged.empty:
        logging.warning("No overlapping SNPs between weights and GWAS")
        return pd.DataFrame(), pd.DataFrame()

    # Track which rows to keep
    keep_mask = []
    flip_mask = []

    for idx, row in merged.iterrows():
        ea_pas = row[effect_allele_pas].upper()
        oa_pas = row[other_allele_pas].upper()
        ea_gwas = row[effect_allele_gwas].upper()
        oa_gwas = row[other_allele_gwas].upper()

        # Check palindromic
        if is_palindromic(ea_pas, oa_pas):
            keep_mask.append(False)
            flip_mask.append(False)
            continue

        # Check allele match
        if ea_pas == ea_gwas and oa_pas == oa_gwas:
            # Direct match
            keep_mask.append(True)
            flip_mask.append(False)
        elif ea_pas == oa_gwas and oa_pas == ea_gwas:
            # Reversed alleles - need to flip
            keep_mask.append(True)
            flip_mask.append(True)
        else:
            # Mismatched alleles
            keep_mask.append(False)
            flip_mask.append(False)

    merged["_keep"] = keep_mask
    merged["_flip"] = flip_mask

    n_palindromic = (~merged["_keep"]).sum() - len([f for f in flip_mask if not f and not keep_mask[flip_mask.index(f)]])
    n_dropped = (~merged["_keep"]).sum()
    n_flipped = sum(flip_mask)
    logging.info("Dropped %d SNPs (palindromic or mismatched alleles)", n_dropped)
    logging.info("Flipped %d SNPs to align effect alleles", n_flipped)

    # Filter to kept rows
    harmonized = merged[merged["_keep"]].copy()

    if harmonized.empty:
        logging.warning("No SNPs remain after harmonization")
        return pd.DataFrame(), pd.DataFrame()

    # Flip GWAS effects where needed
    flip_indices = harmonized["_flip"]
    harmonized.loc[flip_indices, beta_trait_col] *= -1
    if has_z:
        harmonized.loc[flip_indices, z_col] *= -1

    # Build output DataFrames
    weights_cols = [pathway_col, snp_col, beta_pas_col, effect_allele_pas, other_allele_pas]
    harmonized_weights = harmonized[weights_cols].copy()
    harmonized_weights = harmonized_weights.rename(
        columns={
            effect_allele_pas: "effect_allele",
            other_allele_pas: "other_allele",
        }
    )

    gwas_cols = [snp_col, beta_trait_col]
    if has_se:
        gwas_cols.append(se_col)
    if has_z:
        gwas_cols.append(z_col)
    harmonized_gwas = harmonized[gwas_cols].copy()

    # Reset indices to ensure alignment
    harmonized_weights = harmonized_weights.reset_index(drop=True)
    harmonized_gwas = harmonized_gwas.reset_index(drop=True)

    logging.info("Harmonization complete: %d SNP-pathway pairs retained", len(harmonized_weights))

    return harmonized_weights, harmonized_gwas


def prepare_pathway_vectors(
    harmonized_weights: pd.DataFrame,
    harmonized_gwas: pd.DataFrame,
    pathway: str,
    snp_col: str = "snp_id",
    pathway_col: str = "pathway",
    beta_pas_col: str = "beta_pas",
    beta_trait_col: str = "beta_trait",
    z_col: str = "z",
) -> Tuple[pd.Series, pd.Series]:
    """Extract aligned beta vectors for a single pathway.

    Parameters
    ----------
    harmonized_weights : pd.DataFrame
        Harmonized weights from harmonize_weights_gwas.
    harmonized_gwas : pd.DataFrame
        Harmonized GWAS from harmonize_weights_gwas.
    pathway : str
        Pathway identifier to extract.
    snp_col : str
        Column name for SNP identifier.
    pathway_col : str
        Column name for pathway identifier.
    beta_pas_col : str
        Column name for PAS effect size.
    beta_trait_col : str
        Column name for trait effect size.
    z_col : str
        Column name for Z-score (used if beta_trait not available).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (beta_p, gamma) - SNP-to-PAS effects and SNP-to-trait effects,
        both indexed by SNP ID.
    """
    mask = harmonized_weights[pathway_col] == pathway
    pathway_weights = harmonized_weights[mask].copy()

    if pathway_weights.empty:
        raise ValueError(f"Pathway '{pathway}' not found in harmonized weights")

    # Get corresponding GWAS rows
    indices = mask[mask].index
    pathway_gwas = harmonized_gwas.loc[indices].copy()

    beta_p = pd.Series(
        pathway_weights[beta_pas_col].values,
        index=pathway_weights[snp_col].values,
        name="beta_pas",
    )

    if beta_trait_col in pathway_gwas.columns:
        gamma = pd.Series(
            pathway_gwas[beta_trait_col].values,
            index=pathway_weights[snp_col].values,
            name="gamma",
        )
    elif z_col in pathway_gwas.columns:
        gamma = pd.Series(
            pathway_gwas[z_col].values,
            index=pathway_weights[snp_col].values,
            name="gamma",
        )
    else:
        raise ValueError("Neither beta_trait nor z found in harmonized GWAS")

    return beta_p, gamma
