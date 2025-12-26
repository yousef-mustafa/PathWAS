## ------------------------------------------------------------------------------------------- ##
## Pathway Activation Score Computation                                                        ##
## ------------------------------------------------------------------------------------------- ##
## @script: pas.py                                                                             ##
##                                                                                             ##
## @description: Implements multiple strategies to compute pathway activation scores           ##
##               including mean, sum, median and an activity weighted approach.                ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Computation of Pathway Activation Scores (PAS)."""

from typing import Callable, Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import logging


class PASMethodRegistry:
    """Registry for different Pathway Activation Score computation methods."""

    def __init__(self):
        self._methods: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def register(self, name: str):
        def decorator(func: Callable[[pd.DataFrame], pd.Series]):
            self._methods[name] = func
            return func
        return decorator

    def get(self, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        return self._methods[name]

    def available(self) -> Iterable[str]:
        return self._methods.keys()


registry = PASMethodRegistry()


@registry.register('mean')
def mean_expression(expr: pd.DataFrame) -> pd.Series:
    """Compute mean expression for genes in a pathway."""
    return expr.mean(axis=1)


@registry.register('sum')
def sum_expression(expr: pd.DataFrame) -> pd.Series:
    """Compute summed expression for genes in a pathway."""
    return expr.sum(axis=1)


@registry.register('median')
def median_expression(expr: pd.DataFrame) -> pd.Series:
    """Compute median expression for genes in a pathway."""
    return expr.median(axis=1)


def _bicor(x: pd.Series, y: pd.Series, min_valid: int = 5) -> float:
    """Compute biweight midcorrelation between two vectors.

    Uses astropy.stats.biweight_midcorrelation if available, otherwise falls
    back to Pearson correlation.

    Parameters
    ----------
    x : pd.Series
        First vector.
    y : pd.Series
        Second vector.
    min_valid : int
        Minimum number of valid (non-NaN, non-inf) values required.

    Returns
    -------
    float
        Biweight midcorrelation coefficient, or NaN if insufficient valid data.
    """
    # Convert to numpy and remove NaN/inf pairs
    x_vals = np.asarray(x, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)

    # Find valid pairs (neither NaN nor inf)
    valid_mask = (
        np.isfinite(x_vals) &
        np.isfinite(y_vals)
    )

    x_valid = x_vals[valid_mask]
    y_valid = y_vals[valid_mask]

    # Check minimum valid values
    if len(x_valid) < min_valid:
        logging.debug("Insufficient valid values for bicor: %d < %d", len(x_valid), min_valid)
        return np.nan

    # Try to use astropy's biweight_midcorrelation
    try:
        from astropy.stats import biweight_midcorrelation
        try:
            result = biweight_midcorrelation(x_valid, y_valid)
            return float(result)
        except (ValueError, ZeroDivisionError) as e:
            # Degenerate input (e.g., constant values) - fall back to Pearson
            logging.debug("Bicor failed on degenerate input, falling back to Pearson: %s", e)
            return float(np.corrcoef(x_valid, y_valid)[0, 1])
    except ImportError:
        # astropy not available - fall back to Pearson
        logging.debug("astropy not available, using Pearson correlation as fallback")
        return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Return correlation coefficient between two gene expression vectors."""

    method = method.lower()
    if method == "pearson":
        return x.corr(y, method="pearson")
    if method == "spearman":
        return x.corr(y, method="spearman")
    if method == "bicor":
        return _bicor(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


def _activity_weighted(expr: pd.DataFrame, corr_method: str) -> Tuple[pd.Series, Dict[str, float]]:
    """Compute activity weighted PAS for a single pathway."""

    genes = expr.columns
    if len(genes) == 1:
        # Single gene pathway - weight is 1
        weight = {genes[0]: 1.0}
        pas = expr[genes[0]].copy()
        return pas, weight

    weights: Dict[str, float] = {}
    for g in genes:
        others = [h for h in genes if h != g]
        corrs = [_corr(expr[g], expr[o], corr_method) for o in others]
        # mean absolute correlation
        weights[g] = float(np.mean(np.abs(corrs))) if corrs else 1.0

    # Weighted sum normalised by total weight
    weight_series = pd.Series(weights)
    pas = (expr * weight_series).sum(axis=1) / weight_series.sum()
    logging.debug("Computed weights for %d genes", len(weights))
    return pas, weights


def compute_pas(
    expression_df: pd.DataFrame,
    pathway_dict: Dict[str, Iterable[str]],
    method: str = "activity_weighted",
    corr_method: str = "bicor",
    normalize_samples: bool = False,
    normalize_pathways: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Dict[str, float]]]]:
    """Compute pathway activation scores (PAS).

    Parameters
    ----------
    expression_df : pandas.DataFrame
        Gene expression matrix with samples in rows and genes in columns. Values
        are assumed to be ``log2(TPM + 1)`` transformed.
    pathway_dict : dict
        Mapping of pathway names to lists of gene symbols.
    method : {{"sum", "mean", "median", "activity_weighted"}}, optional
        Aggregation method used to compute PAS. ``activity_weighted`` computes
        gene weights based on correlation within each pathway.
    corr_method : {{"pearson", "spearman", "bicor"}}, optional
        Correlation method used when ``method="activity_weighted"``.
    normalize_samples : bool, optional
        If ``True``, z-score PAS across samples for each pathway.
    normalize_pathways : bool, optional
        If ``True``, z-score PAS across pathways for each sample.

    Returns
    -------
    Tuple[pandas.DataFrame, Optional[Dict[str, Dict[str, float]]]]
        PAS matrix (samples x pathways) and, if ``activity_weighted`` is used,
        a mapping of pathway -> gene -> weight.
    """

    method = method.lower()
    logging.info("Computing PAS using method %s", method)
    scores: Dict[str, pd.Series] = {}
    weights: Dict[str, Dict[str, float]] = {}

    for pw, genes in pathway_dict.items():
        genes = [g for g in genes if g in expression_df.columns]
        if not genes:
            continue
        logging.debug("Processing pathway %s with %d genes", pw, len(genes))
        sub_expr = expression_df[genes]

        if method == "sum":
            pas_vec = sub_expr.sum(axis=1)
        elif method == "mean":
            pas_vec = sub_expr.mean(axis=1)
        elif method == "median":
            pas_vec = sub_expr.median(axis=1)
        elif method == "activity_weighted":
            pas_vec, w = _activity_weighted(sub_expr, corr_method)
            weights[pw] = w
        else:
            raise ValueError(f"Unknown PAS method: {method}")

        scores[pw] = pas_vec

    pas_df = pd.DataFrame(scores)

    if normalize_samples and not pas_df.empty:
        pas_df = (pas_df - pas_df.mean(axis=0)) / pas_df.std(axis=0, ddof=0)
        logging.debug("Normalized PAS across samples")

    if normalize_pathways and not pas_df.empty:
        pas_df = ((pas_df.T - pas_df.T.mean(axis=0)) / pas_df.T.std(axis=0, ddof=0)).T
        logging.debug("Normalized PAS across pathways")

    return pas_df, (weights if method == "activity_weighted" else None)