## ------------------------------------------------------------------------------------------- ##
## Expression Data Preprocessing Utilities                                                    ##
## ------------------------------------------------------------------------------------------- ##
## @script: expression.py                                                                     ##
##                                                                                            ##
## @description: Utilities for preprocessing gene expression data before PAS computation.    ##
##               Includes library-size normalization (CPM/TPM), log transformation, and      ##
##               z-score standardization.                                                    ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Expression data preprocessing utilities."""

import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


def normalize_library_size(
    expression: pd.DataFrame,
    mode: Literal["cpm", "tpm", None] = "cpm",
    gene_lengths: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Normalize expression data by library size.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix with samples as rows (index) and genes as columns.
        Values should be raw counts.
    mode : {"cpm", "tpm", None}
        Normalization mode:
        - "cpm": Counts Per Million (divide by total counts, multiply by 1e6)
        - "tpm": Transcripts Per Million (requires gene_lengths)
        - None: Return expression unchanged
    gene_lengths : pd.Series, optional
        Gene lengths in base pairs, indexed by gene ID.
        Required if mode="tpm".

    Returns
    -------
    pd.DataFrame
        Normalized expression matrix with same shape and indexing.

    Raises
    ------
    ValueError
        If mode="tpm" but gene_lengths is not provided.

    Examples
    --------
    >>> expr_cpm = normalize_library_size(raw_counts, mode="cpm")
    >>> expr_tpm = normalize_library_size(raw_counts, mode="tpm", gene_lengths=lengths)
    """
    if mode is None:
        return expression.copy()

    mode = mode.lower()

    if mode == "cpm":
        return _counts_per_million(expression)
    elif mode == "tpm":
        if gene_lengths is None:
            raise ValueError("gene_lengths required for TPM normalization")
        return _transcripts_per_million(expression, gene_lengths)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Use 'cpm', 'tpm', or None.")


def _counts_per_million(expression: pd.DataFrame) -> pd.DataFrame:
    """Compute Counts Per Million (CPM).

    For each sample, divides counts by total counts and multiplies by 1e6.

    Parameters
    ----------
    expression : pd.DataFrame
        Raw count matrix (samples x genes).

    Returns
    -------
    pd.DataFrame
        CPM-normalized expression.
    """
    total_counts = expression.sum(axis=1)

    # Handle zero total counts
    total_counts = total_counts.replace(0, np.nan)

    cpm = expression.div(total_counts, axis=0) * 1e6

    # Replace NaN with 0 for samples with zero total counts
    cpm = cpm.fillna(0)

    logging.debug("CPM normalization complete: %d samples, %d genes",
                 expression.shape[0], expression.shape[1])

    return cpm


def _transcripts_per_million(
    expression: pd.DataFrame,
    gene_lengths: pd.Series,
) -> pd.DataFrame:
    """Compute Transcripts Per Million (TPM).

    TPM = (counts / gene_length_kb) / sum(counts / gene_length_kb) * 1e6

    Parameters
    ----------
    expression : pd.DataFrame
        Raw count matrix (samples x genes).
    gene_lengths : pd.Series
        Gene lengths in base pairs, indexed by gene ID.

    Returns
    -------
    pd.DataFrame
        TPM-normalized expression.
    """
    # Align gene lengths to expression columns
    common_genes = list(set(expression.columns) & set(gene_lengths.index))

    if len(common_genes) == 0:
        raise ValueError("No common genes between expression and gene_lengths")

    if len(common_genes) < len(expression.columns):
        logging.warning(
            "Only %d of %d genes have length information, others will be dropped",
            len(common_genes),
            len(expression.columns),
        )

    expr_subset = expression[common_genes]
    lengths = gene_lengths.reindex(common_genes)

    # Convert to kilobases
    lengths_kb = lengths / 1000.0

    # Handle zero lengths
    lengths_kb = lengths_kb.replace(0, np.nan)

    # Compute reads per kilobase (RPK)
    rpk = expr_subset.div(lengths_kb, axis=1)

    # Compute scaling factor per sample
    rpk_sum = rpk.sum(axis=1)
    rpk_sum = rpk_sum.replace(0, np.nan)

    # Compute TPM
    tpm = rpk.div(rpk_sum, axis=0) * 1e6

    # Replace NaN with 0
    tpm = tpm.fillna(0)

    logging.debug("TPM normalization complete: %d samples, %d genes",
                 expression.shape[0], len(common_genes))

    return tpm


def log_transform(
    expression: pd.DataFrame,
    base: Literal["log2", "log10", "ln"] = "log2",
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Apply log transformation to expression data.

    Computes log(expression + pseudocount).

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (samples x genes).
    base : {"log2", "log10", "ln"}
        Logarithm base to use.
    pseudocount : float
        Value added before log transformation to avoid log(0).

    Returns
    -------
    pd.DataFrame
        Log-transformed expression matrix.

    Examples
    --------
    >>> expr_log = log_transform(expr_cpm, base="log2", pseudocount=1.0)
    """
    expr_shifted = expression + pseudocount

    if base == "log2":
        log_expr = np.log2(expr_shifted)
    elif base == "log10":
        log_expr = np.log10(expr_shifted)
    elif base == "ln":
        log_expr = np.log(expr_shifted)
    else:
        raise ValueError(f"Unknown log base: {base}. Use 'log2', 'log10', or 'ln'.")

    logging.debug("Log transformation (base=%s, pseudocount=%.2f) complete",
                 base, pseudocount)

    return log_expr


def zscore_genes(
    expression: pd.DataFrame,
    ddof: int = 1,
) -> pd.DataFrame:
    """Z-score normalize expression across samples for each gene.

    For each gene (column), subtracts mean and divides by standard deviation.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (samples x genes).
    ddof : int
        Delta degrees of freedom for standard deviation calculation.

    Returns
    -------
    pd.DataFrame
        Z-score normalized expression matrix.

    Examples
    --------
    >>> expr_zscore = zscore_genes(expr_log)
    """
    mean_expr = expression.mean(axis=0)
    std_expr = expression.std(axis=0, ddof=ddof)

    # Handle zero variance genes
    std_expr = std_expr.replace(0, np.nan)

    zscore = (expression - mean_expr) / std_expr

    # Replace NaN with 0 for zero-variance genes
    zscore = zscore.fillna(0)

    n_zero_var = (std_expr.isna()).sum()
    if n_zero_var > 0:
        logging.warning("%d genes have zero variance and were set to 0", n_zero_var)

    logging.debug("Z-score normalization complete: %d samples, %d genes",
                 expression.shape[0], expression.shape[1])

    return zscore


def filter_low_expression(
    expression: pd.DataFrame,
    min_cpm: float = 1.0,
    min_samples_fraction: float = 0.5,
    counts_are_cpm: bool = False,
) -> pd.DataFrame:
    """Filter out lowly expressed genes.

    Removes genes that do not have at least `min_cpm` in at least
    `min_samples_fraction` of samples.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (samples x genes). Should be raw counts or CPM.
    min_cpm : float
        Minimum CPM threshold for a gene to be considered expressed.
    min_samples_fraction : float
        Fraction of samples that must meet the min_cpm threshold.
    counts_are_cpm : bool
        If True, expression values are already CPM-normalized.
        If False, will convert to CPM first for threshold comparison.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix with low-expression genes removed.

    Examples
    --------
    >>> expr_filtered = filter_low_expression(raw_counts, min_cpm=1.0, min_samples_fraction=0.5)
    """
    if counts_are_cpm:
        expr_cpm = expression
    else:
        expr_cpm = _counts_per_million(expression)

    n_samples = expression.shape[0]
    min_samples = int(np.ceil(n_samples * min_samples_fraction))

    # Count samples meeting threshold for each gene
    samples_above_threshold = (expr_cpm >= min_cpm).sum(axis=0)

    # Keep genes passing filter
    keep_genes = samples_above_threshold >= min_samples
    filtered = expression.loc[:, keep_genes]

    n_removed = len(expression.columns) - len(filtered.columns)
    logging.info(
        "Filtered %d of %d genes (%.1f%%) with CPM < %.1f in > %.0f%% of samples",
        n_removed,
        len(expression.columns),
        100 * n_removed / len(expression.columns) if len(expression.columns) > 0 else 0,
        min_cpm,
        100 * (1 - min_samples_fraction),
    )

    return filtered


def filter_low_count_samples(
    expression: pd.DataFrame,
    min_total_counts: int = 1000,
) -> pd.DataFrame:
    """Filter out samples with very low total counts.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (samples x genes).
    min_total_counts : int
        Minimum total counts required for a sample.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix with low-count samples removed.

    Examples
    --------
    >>> expr_filtered = filter_low_count_samples(raw_counts, min_total_counts=10000)
    """
    total_counts = expression.sum(axis=1)
    keep_samples = total_counts >= min_total_counts
    filtered = expression.loc[keep_samples, :]

    n_removed = len(expression) - len(filtered)
    if n_removed > 0:
        logging.warning(
            "Removed %d of %d samples (%.1f%%) with total counts < %d",
            n_removed,
            len(expression),
            100 * n_removed / len(expression),
            min_total_counts,
        )

    return filtered


def preprocess_expression(
    expression: pd.DataFrame,
    normalization: Literal["cpm", "tpm", None] = "cpm",
    gene_lengths: Optional[pd.Series] = None,
    log_transform_expr: bool = True,
    log_base: Literal["log2", "log10", "ln"] = "log2",
    log_pseudocount: float = 1.0,
    zscore: bool = True,
    filter_genes: bool = True,
    min_cpm: float = 1.0,
    min_samples_fraction: float = 0.5,
    filter_samples: bool = False,
    min_total_counts: int = 1000,
) -> pd.DataFrame:
    """Complete expression preprocessing pipeline.

    Applies the following steps in order:
    1. (Optional) Filter low-count samples
    2. (Optional) Filter lowly expressed genes
    3. (Optional) Library size normalization (CPM or TPM)
    4. (Optional) Log transformation
    5. (Optional) Z-score across samples per gene

    Parameters
    ----------
    expression : pd.DataFrame
        Raw count expression matrix (samples x genes).
    normalization : {"cpm", "tpm", None}
        Library size normalization mode.
    gene_lengths : pd.Series, optional
        Gene lengths for TPM normalization.
    log_transform_expr : bool
        Whether to apply log transformation.
    log_base : {"log2", "log10", "ln"}
        Base for log transformation.
    log_pseudocount : float
        Pseudocount for log transformation.
    zscore : bool
        Whether to z-score normalize genes.
    filter_genes : bool
        Whether to filter lowly expressed genes.
    min_cpm : float
        Minimum CPM for gene filtering.
    min_samples_fraction : float
        Fraction of samples for gene filtering.
    filter_samples : bool
        Whether to filter low-count samples.
    min_total_counts : int
        Minimum total counts for sample filtering.

    Returns
    -------
    pd.DataFrame
        Preprocessed expression matrix.

    Examples
    --------
    >>> processed = preprocess_expression(
    ...     raw_counts,
    ...     normalization="cpm",
    ...     log_transform_expr=True,
    ...     zscore=True
    ... )
    """
    expr = expression.copy()

    # Step 1: Filter low-count samples
    if filter_samples:
        expr = filter_low_count_samples(expr, min_total_counts=min_total_counts)

    # Step 2: Filter lowly expressed genes (before normalization)
    if filter_genes:
        expr = filter_low_expression(
            expr,
            min_cpm=min_cpm,
            min_samples_fraction=min_samples_fraction,
        )

    # Step 3: Library size normalization
    if normalization is not None:
        expr = normalize_library_size(expr, mode=normalization, gene_lengths=gene_lengths)

    # Step 4: Log transformation
    if log_transform_expr:
        expr = log_transform(expr, base=log_base, pseudocount=log_pseudocount)

    # Step 5: Z-score normalization
    if zscore:
        expr = zscore_genes(expr)

    logging.info(
        "Expression preprocessing complete: %d samples, %d genes",
        expr.shape[0],
        expr.shape[1],
    )

    return expr
