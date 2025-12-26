## ------------------------------------------------------------------------------------------- ##
## PAS Statistical Testing                                                                     ##
## ------------------------------------------------------------------------------------------- ##
## @script: pas_test.py                                                                        ##
##                                                                                             ##
## @description: Statistical analyses to detect significantly different PAS across groups.    ##
##               Supports t-tests, ANOVA, Kruskal-Wallis, and covariate-adjusted models.      ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Statistical testing for Pathway Activation Scores (PAS) across groups."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class TestMethod(Enum):
    """Available statistical test methods."""
    TTEST = "ttest"
    WELCH = "welch"
    MANN_WHITNEY = "mann_whitney"
    ANOVA = "anova"
    KRUSKAL = "kruskal"
    LINEAR = "linear"
    LOGISTIC = "logistic"


class MultipleTestingCorrection(Enum):
    """Multiple testing correction methods."""
    NONE = "none"
    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"


@dataclass
class PASTestResult:
    """Results from a PAS statistical test.

    Attributes
    ----------
    pathway : str
        Name of the pathway tested.
    statistic : float
        Test statistic value.
    pvalue : float
        Raw p-value.
    pvalue_adjusted : float
        Adjusted p-value after multiple testing correction.
    effect_size : float
        Effect size measure (Cohen's d, eta-squared, etc.).
    method : str
        Statistical test method used.
    group_means : Dict[str, float]
        Mean PAS for each group.
    group_stds : Dict[str, float]
        Standard deviation of PAS for each group.
    n_samples : Dict[str, int]
        Number of samples per group.
    significant : bool
        Whether the result is significant at alpha threshold.
    """
    pathway: str
    statistic: float
    pvalue: float
    pvalue_adjusted: float = np.nan
    effect_size: float = np.nan
    method: str = ""
    group_means: Dict[str, float] = field(default_factory=dict)
    group_stds: Dict[str, float] = field(default_factory=dict)
    n_samples: Dict[str, int] = field(default_factory=dict)
    significant: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PASTestConfig:
    """Configuration for PAS statistical testing.

    Attributes
    ----------
    method : TestMethod
        Statistical test method to use.
    alpha : float
        Significance threshold (default: 0.05).
    correction : MultipleTestingCorrection
        Multiple testing correction method.
    min_samples_per_group : int
        Minimum samples required per group.
    paired : bool
        Whether to use paired tests (for t-test).
    parametric : bool
        Whether to assume parametric distribution.
    """
    method: TestMethod = TestMethod.TTEST
    alpha: float = 0.05
    correction: MultipleTestingCorrection = MultipleTestingCorrection.FDR_BH
    min_samples_per_group: int = 3
    paired: bool = False
    parametric: bool = True


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    group1 : np.ndarray
        Values for first group.
    group2 : np.ndarray
        Values for second group.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Calculate eta-squared effect size for ANOVA.

    Parameters
    ----------
    groups : List[np.ndarray]
        List of value arrays for each group.

    Returns
    -------
    float
        Eta-squared effect size.
    """
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    ss_total = np.sum((all_values - grand_mean) ** 2)

    if ss_total == 0:
        return np.nan

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total


def _apply_fdr_correction(
    pvalues: np.ndarray,
    method: MultipleTestingCorrection,
) -> np.ndarray:
    """Apply multiple testing correction to p-values.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of raw p-values.
    method : MultipleTestingCorrection
        Correction method to apply.

    Returns
    -------
    np.ndarray
        Adjusted p-values.
    """
    if method == MultipleTestingCorrection.NONE:
        return pvalues.copy()

    n = len(pvalues)
    if n == 0:
        return pvalues.copy()

    # Handle NaN values
    valid_mask = ~np.isnan(pvalues)
    adjusted = np.full_like(pvalues, np.nan)

    if not np.any(valid_mask):
        return adjusted

    valid_pvals = pvalues[valid_mask]
    n_valid = len(valid_pvals)

    if method == MultipleTestingCorrection.BONFERRONI:
        adjusted[valid_mask] = np.minimum(valid_pvals * n_valid, 1.0)

    elif method == MultipleTestingCorrection.FDR_BH:
        # Benjamini-Hochberg procedure
        sorted_idx = np.argsort(valid_pvals)
        sorted_pvals = valid_pvals[sorted_idx]
        ranks = np.arange(1, n_valid + 1)

        adjusted_sorted = sorted_pvals * n_valid / ranks
        # Ensure monotonicity
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        # Restore original order
        adjusted_valid = np.empty_like(sorted_pvals)
        adjusted_valid[sorted_idx] = adjusted_sorted
        adjusted[valid_mask] = adjusted_valid

    elif method == MultipleTestingCorrection.FDR_BY:
        # Benjamini-Yekutieli procedure
        sorted_idx = np.argsort(valid_pvals)
        sorted_pvals = valid_pvals[sorted_idx]
        ranks = np.arange(1, n_valid + 1)
        c_m = np.sum(1.0 / np.arange(1, n_valid + 1))

        adjusted_sorted = sorted_pvals * n_valid * c_m / ranks
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        adjusted_valid = np.empty_like(sorted_pvals)
        adjusted_valid[sorted_idx] = adjusted_sorted
        adjusted[valid_mask] = adjusted_valid

    return adjusted


def _run_ttest(
    values: np.ndarray,
    groups: np.ndarray,
    group_labels: List[str],
    config: PASTestConfig,
) -> Tuple[float, float, float]:
    """Run t-test for two groups.

    Returns (statistic, pvalue, effect_size).
    """
    if len(group_labels) != 2:
        raise ValueError("t-test requires exactly 2 groups")

    g1_mask = groups == group_labels[0]
    g2_mask = groups == group_labels[1]

    g1_vals = values[g1_mask]
    g2_vals = values[g2_mask]

    if len(g1_vals) < config.min_samples_per_group or len(g2_vals) < config.min_samples_per_group:
        return np.nan, np.nan, np.nan

    if config.method == TestMethod.WELCH:
        stat, pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
    elif config.paired:
        if len(g1_vals) != len(g2_vals):
            raise ValueError("Paired t-test requires equal sample sizes")
        stat, pval = stats.ttest_rel(g1_vals, g2_vals)
    else:
        stat, pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=True)

    effect = _cohens_d(g1_vals, g2_vals)
    return float(stat), float(pval), effect


def _run_mann_whitney(
    values: np.ndarray,
    groups: np.ndarray,
    group_labels: List[str],
    config: PASTestConfig,
) -> Tuple[float, float, float]:
    """Run Mann-Whitney U test for two groups.

    Returns (statistic, pvalue, effect_size).
    """
    if len(group_labels) != 2:
        raise ValueError("Mann-Whitney test requires exactly 2 groups")

    g1_mask = groups == group_labels[0]
    g2_mask = groups == group_labels[1]

    g1_vals = values[g1_mask]
    g2_vals = values[g2_mask]

    if len(g1_vals) < config.min_samples_per_group or len(g2_vals) < config.min_samples_per_group:
        return np.nan, np.nan, np.nan

    stat, pval = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')

    # Compute rank-biserial correlation as effect size
    n1, n2 = len(g1_vals), len(g2_vals)
    r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial correlation

    return float(stat), float(pval), float(r)


def _run_anova(
    values: np.ndarray,
    groups: np.ndarray,
    group_labels: List[str],
    config: PASTestConfig,
) -> Tuple[float, float, float]:
    """Run one-way ANOVA for multiple groups.

    Returns (statistic, pvalue, effect_size).
    """
    group_data = []
    for label in group_labels:
        g_vals = values[groups == label]
        if len(g_vals) < config.min_samples_per_group:
            return np.nan, np.nan, np.nan
        group_data.append(g_vals)

    stat, pval = stats.f_oneway(*group_data)
    effect = _eta_squared(group_data)

    return float(stat), float(pval), effect


def _run_kruskal(
    values: np.ndarray,
    groups: np.ndarray,
    group_labels: List[str],
    config: PASTestConfig,
) -> Tuple[float, float, float]:
    """Run Kruskal-Wallis H test for multiple groups.

    Returns (statistic, pvalue, effect_size).
    """
    group_data = []
    for label in group_labels:
        g_vals = values[groups == label]
        if len(g_vals) < config.min_samples_per_group:
            return np.nan, np.nan, np.nan
        group_data.append(g_vals)

    stat, pval = stats.kruskal(*group_data)

    # Epsilon-squared effect size for Kruskal-Wallis
    n = len(values)
    effect = (stat - len(group_labels) + 1) / (n - len(group_labels)) if n > len(group_labels) else np.nan

    return float(stat), float(pval), effect


def _run_linear_regression(
    pas_values: np.ndarray,
    group_indicator: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Run linear regression with optional covariates.

    Parameters
    ----------
    pas_values : np.ndarray
        PAS values (dependent variable).
    group_indicator : np.ndarray
        Group indicator (0/1 for binary, or continuous).
    covariates : np.ndarray, optional
        Covariate matrix (n_samples x n_covariates).

    Returns
    -------
    Tuple[float, float, float, Dict]
        (t-statistic, p-value, effect_size (beta), additional_info).
    """
    from scipy.linalg import lstsq

    n = len(pas_values)

    # Build design matrix
    X = np.column_stack([np.ones(n), group_indicator])
    if covariates is not None:
        X = np.column_stack([X, covariates])

    # Fit linear model using least squares
    try:
        beta, residuals, rank, s = lstsq(X, pas_values)
    except Exception as e:
        logging.warning("Linear regression failed: %s", e)
        return np.nan, np.nan, np.nan, {}

    # Compute residuals and standard errors
    y_pred = X @ beta
    residuals = pas_values - y_pred
    mse = np.sum(residuals ** 2) / (n - X.shape[1])

    # Standard error of coefficients
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * mse)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, {}

    # t-statistic and p-value for group coefficient (index 1)
    t_stat = beta[1] / se[1] if se[1] > 0 else np.nan
    df = n - X.shape[1]
    pval = 2 * stats.t.sf(np.abs(t_stat), df) if not np.isnan(t_stat) else np.nan

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((pas_values - np.mean(pas_values)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    additional_info = {
        "coefficients": dict(zip(["intercept", "group"] + [f"cov_{i}" for i in range(X.shape[1] - 2)], beta)),
        "std_errors": dict(zip(["intercept", "group"] + [f"cov_{i}" for i in range(X.shape[1] - 2)], se)),
        "r_squared": r_squared,
        "df": df,
    }

    return float(t_stat), float(pval), float(beta[1]), additional_info


def test_pas_difference(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    config: Optional[PASTestConfig] = None,
    covariates: Optional[pd.DataFrame] = None,
    pathways: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Test for significant differences in PAS across groups.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix with samples as rows and pathways as columns.
    group_labels : pd.Series
        Group assignment for each sample. Index must match pas_df index.
    config : PASTestConfig, optional
        Test configuration. Defaults to t-test with FDR correction.
    covariates : pd.DataFrame, optional
        Covariate matrix for adjusted analysis. Index must match pas_df index.
    pathways : List[str], optional
        Subset of pathways to test. Defaults to all pathways.

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns: pathway, statistic, pvalue, pvalue_adj,
        effect_size, method, significant, and per-group statistics.
    """
    if config is None:
        config = PASTestConfig()

    # Align indices
    common_samples = pas_df.index.intersection(group_labels.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples between PAS matrix and group labels")

    pas_df = pas_df.loc[common_samples]
    group_labels = group_labels.loc[common_samples]

    if covariates is not None:
        common_samples = common_samples.intersection(covariates.index)
        covariates = covariates.loc[common_samples]
        pas_df = pas_df.loc[common_samples]
        group_labels = group_labels.loc[common_samples]

    # Get unique groups
    unique_groups = sorted(group_labels.unique())
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups, got {n_groups}")

    # Select pathways to test
    if pathways is None:
        pathways = list(pas_df.columns)
    else:
        pathways = [p for p in pathways if p in pas_df.columns]

    logging.info("Testing %d pathways across %d groups using %s",
                 len(pathways), n_groups, config.method.value)

    # Determine which test to run
    if config.method in (TestMethod.TTEST, TestMethod.WELCH):
        if n_groups != 2:
            logging.warning("t-test selected but %d groups found. Switching to ANOVA.", n_groups)
            config.method = TestMethod.ANOVA if config.parametric else TestMethod.KRUSKAL

    results: List[PASTestResult] = []

    for pathway in pathways:
        values = pas_df[pathway].values
        groups = group_labels.values

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        values = values[valid_mask]
        groups = groups[valid_mask]

        # Compute group statistics
        group_means = {}
        group_stds = {}
        n_samples = {}
        for g in unique_groups:
            g_vals = values[groups == g]
            group_means[str(g)] = float(np.mean(g_vals)) if len(g_vals) > 0 else np.nan
            group_stds[str(g)] = float(np.std(g_vals, ddof=1)) if len(g_vals) > 1 else np.nan
            n_samples[str(g)] = len(g_vals)

        # Run the appropriate test
        additional_info = {}

        if config.method == TestMethod.LINEAR and covariates is not None:
            # Linear regression with covariates
            # Create binary group indicator (first group = 0, second = 1)
            if n_groups == 2:
                group_indicator = (groups == unique_groups[1]).astype(float)
            else:
                # For multiple groups, use dummy encoding
                group_indicator = (groups == unique_groups[1]).astype(float)
                logging.warning("Linear model with >2 groups uses only first two groups")

            cov_values = covariates.loc[pas_df.index[valid_mask]].values
            stat, pval, effect, additional_info = _run_linear_regression(
                values, group_indicator, cov_values
            )

        elif config.method in (TestMethod.TTEST, TestMethod.WELCH):
            stat, pval, effect = _run_ttest(values, groups, unique_groups, config)

        elif config.method == TestMethod.MANN_WHITNEY:
            stat, pval, effect = _run_mann_whitney(values, groups, unique_groups, config)

        elif config.method == TestMethod.ANOVA:
            stat, pval, effect = _run_anova(values, groups, unique_groups, config)

        elif config.method == TestMethod.KRUSKAL:
            stat, pval, effect = _run_kruskal(values, groups, unique_groups, config)

        else:
            stat, pval, effect = np.nan, np.nan, np.nan

        results.append(PASTestResult(
            pathway=pathway,
            statistic=stat,
            pvalue=pval,
            effect_size=effect,
            method=config.method.value,
            group_means=group_means,
            group_stds=group_stds,
            n_samples=n_samples,
            additional_info=additional_info,
        ))

    # Apply multiple testing correction
    pvalues = np.array([r.pvalue for r in results])
    adjusted_pvalues = _apply_fdr_correction(pvalues, config.correction)

    for i, result in enumerate(results):
        result.pvalue_adjusted = adjusted_pvalues[i]
        result.significant = result.pvalue_adjusted < config.alpha

    # Convert to DataFrame
    records = []
    for r in results:
        record = {
            "pathway": r.pathway,
            "statistic": r.statistic,
            "pvalue": r.pvalue,
            "pvalue_adj": r.pvalue_adjusted,
            "effect_size": r.effect_size,
            "method": r.method,
            "significant": r.significant,
        }
        # Add per-group statistics
        for g in unique_groups:
            record[f"mean_{g}"] = r.group_means.get(str(g), np.nan)
            record[f"std_{g}"] = r.group_stds.get(str(g), np.nan)
            record[f"n_{g}"] = r.n_samples.get(str(g), 0)
        records.append(record)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("pvalue").reset_index(drop=True)

    n_sig = results_df["significant"].sum()
    logging.info("Found %d significant pathways at alpha=%.3f (after %s correction)",
                 n_sig, config.alpha, config.correction.value)

    return results_df


def test_pas_with_covariates(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    covariates: pd.DataFrame,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Test PAS differences adjusting for covariates using linear regression.

    This is a convenience function that wraps test_pas_difference with
    appropriate settings for covariate-adjusted analysis.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix (samples x pathways).
    group_labels : pd.Series
        Group assignment for each sample.
    covariates : pd.DataFrame
        Covariate matrix (samples x covariates).
    alpha : float
        Significance threshold.
    correction : str
        Multiple testing correction: 'none', 'bonferroni', 'fdr_bh', 'fdr_by'.

    Returns
    -------
    pd.DataFrame
        Test results with covariate-adjusted statistics.
    """
    correction_map = {
        "none": MultipleTestingCorrection.NONE,
        "bonferroni": MultipleTestingCorrection.BONFERRONI,
        "fdr_bh": MultipleTestingCorrection.FDR_BH,
        "fdr_by": MultipleTestingCorrection.FDR_BY,
    }

    config = PASTestConfig(
        method=TestMethod.LINEAR,
        alpha=alpha,
        correction=correction_map.get(correction.lower(), MultipleTestingCorrection.FDR_BH),
    )

    return test_pas_difference(pas_df, group_labels, config=config, covariates=covariates)


def pairwise_group_tests(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    pathway: str,
    method: str = "ttest",
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """Perform pairwise tests between all groups for a specific pathway.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix (samples x pathways).
    group_labels : pd.Series
        Group assignment for each sample.
    pathway : str
        Pathway to test.
    method : str
        Test method: 'ttest', 'welch', 'mann_whitney'.
    correction : str
        Multiple testing correction method.

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results.
    """
    if pathway not in pas_df.columns:
        raise ValueError(f"Pathway '{pathway}' not found in PAS matrix")

    # Align indices
    common_samples = pas_df.index.intersection(group_labels.index)
    pas_values = pas_df.loc[common_samples, pathway]
    groups = group_labels.loc[common_samples]

    unique_groups = sorted(groups.unique())
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("Need at least 2 groups for pairwise tests")

    results = []

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            g1, g2 = unique_groups[i], unique_groups[j]
            g1_vals = pas_values[groups == g1].values
            g2_vals = pas_values[groups == g2].values

            if len(g1_vals) < 2 or len(g2_vals) < 2:
                stat, pval = np.nan, np.nan
            elif method.lower() == "welch":
                stat, pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
            elif method.lower() == "mann_whitney":
                stat, pval = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            else:
                stat, pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=True)

            results.append({
                "group1": g1,
                "group2": g2,
                "statistic": float(stat),
                "pvalue": float(pval),
                "mean_diff": float(np.mean(g1_vals) - np.mean(g2_vals)),
                "effect_size": _cohens_d(g1_vals, g2_vals),
            })

    results_df = pd.DataFrame(results)

    # Apply correction
    if len(results_df) > 0:
        correction_map = {
            "none": MultipleTestingCorrection.NONE,
            "bonferroni": MultipleTestingCorrection.BONFERRONI,
            "fdr_bh": MultipleTestingCorrection.FDR_BH,
            "fdr_by": MultipleTestingCorrection.FDR_BY,
        }
        corr_method = correction_map.get(correction.lower(), MultipleTestingCorrection.BONFERRONI)
        results_df["pvalue_adj"] = _apply_fdr_correction(
            results_df["pvalue"].values, corr_method
        )

    return results_df


def summarize_test_results(
    results_df: pd.DataFrame,
    top_n: int = 20,
) -> str:
    """Generate a summary of PAS test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from test_pas_difference.
    top_n : int
        Number of top results to include.

    Returns
    -------
    str
        Formatted summary string.
    """
    n_tested = len(results_df)
    n_significant = results_df["significant"].sum() if "significant" in results_df else 0

    summary = [
        "=" * 60,
        "PAS DIFFERENTIAL ANALYSIS SUMMARY",
        "=" * 60,
        f"Pathways tested: {n_tested}",
        f"Significant pathways: {n_significant}",
        "",
    ]

    if n_significant > 0:
        summary.append(f"Top {min(top_n, n_significant)} significant pathways:")
        summary.append("-" * 60)

        sig_results = results_df[results_df["significant"]].head(top_n)
        for _, row in sig_results.iterrows():
            summary.append(
                f"  {row['pathway']}: p_adj={row['pvalue_adj']:.2e}, "
                f"effect={row['effect_size']:.3f}"
            )

    summary.append("=" * 60)
    return "\n".join(summary)
