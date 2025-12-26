## ------------------------------------------------------------------------------------------- ##
## PAS Visualization                                                                           ##
## ------------------------------------------------------------------------------------------- ##
## @script: pas_viz.py                                                                         ##
##                                                                                             ##
## @description: Visualization utilities for Pathway Activation Scores (PAS).                 ##
##               Supports heatmaps, boxplots, and other visualizations with                   ##
##               customizable parameters for titles, clustering, and styling.                 ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Visualization utilities for Pathway Activation Scores (PAS)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


class ColorPalette(Enum):
    """Available color palettes for visualizations."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    COOLWARM = "coolwarm"
    RDBU = "RdBu_r"
    RDYLBU = "RdYlBu_r"
    SPECTRAL = "Spectral_r"
    BLUES = "Blues"
    REDS = "Reds"


class ClusterMethod(Enum):
    """Hierarchical clustering methods."""
    AVERAGE = "average"
    COMPLETE = "complete"
    SINGLE = "single"
    WARD = "ward"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"


class ClusterMetric(Enum):
    """Distance metrics for clustering."""
    EUCLIDEAN = "euclidean"
    CORRELATION = "correlation"
    COSINE = "cosine"
    MANHATTAN = "cityblock"
    CHEBYSHEV = "chebyshev"


@dataclass
class HeatmapConfig:
    """Configuration for PAS heatmap visualization.

    Attributes
    ----------
    title : str
        Plot title.
    figsize : Tuple[float, float]
        Figure size in inches (width, height).
    cmap : str
        Colormap name.
    center : float, optional
        Value to center the colormap at.
    vmin : float, optional
        Minimum value for color scale.
    vmax : float, optional
        Maximum value for color scale.
    cluster_rows : bool
        Whether to cluster rows (pathways).
    cluster_cols : bool
        Whether to cluster columns (samples).
    cluster_method : ClusterMethod
        Hierarchical clustering method.
    cluster_metric : ClusterMetric
        Distance metric for clustering.
    row_colors : Dict[str, str], optional
        Colors for row annotations.
    col_colors : pd.DataFrame, optional
        Column annotation colors.
    show_row_labels : bool
        Whether to show row labels.
    show_col_labels : bool
        Whether to show column labels.
    row_label_size : int
        Font size for row labels.
    col_label_size : int
        Font size for column labels.
    cbar_label : str
        Colorbar label.
    dpi : int
        Resolution for saved figures.
    dendrogram_ratio : float
        Ratio of dendrogram size to main plot.
    """
    title: str = "PAS Heatmap"
    figsize: Tuple[float, float] = (12, 10)
    cmap: str = "RdBu_r"
    center: Optional[float] = 0.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cluster_rows: bool = True
    cluster_cols: bool = True
    cluster_method: ClusterMethod = ClusterMethod.AVERAGE
    cluster_metric: ClusterMetric = ClusterMetric.CORRELATION
    row_colors: Optional[Dict[str, str]] = None
    col_colors: Optional[pd.DataFrame] = None
    show_row_labels: bool = True
    show_col_labels: bool = False
    row_label_size: int = 8
    col_label_size: int = 8
    cbar_label: str = "PAS (z-score)"
    dpi: int = 150
    dendrogram_ratio: float = 0.15
    robust: bool = False  # Use robust color scaling (percentile-based)
    linewidths: float = 0.0
    linecolor: str = "white"


@dataclass
class BoxplotConfig:
    """Configuration for PAS boxplot visualization.

    Attributes
    ----------
    title : str
        Plot title.
    figsize : Tuple[float, float]
        Figure size in inches.
    palette : str
        Color palette name.
    orient : str
        Orientation: 'v' (vertical) or 'h' (horizontal).
    show_points : bool
        Whether to overlay individual data points.
    point_size : float
        Size of overlay points.
    show_violin : bool
        Whether to show violin plots instead of boxplots.
    show_mean : bool
        Whether to show mean markers.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    rotate_labels : int
        Rotation angle for x-axis labels.
    order : List[str], optional
        Custom order for groups.
    dpi : int
        Resolution for saved figures.
    notch : bool
        Whether to show notched boxes.
    width : float
        Width of boxes.
    fliersize : float
        Size of outlier markers.
    """
    title: str = "PAS Distribution by Group"
    figsize: Tuple[float, float] = (10, 6)
    palette: str = "Set2"
    orient: str = "v"
    show_points: bool = True
    point_size: float = 3.0
    show_violin: bool = False
    show_mean: bool = True
    xlabel: str = "Group"
    ylabel: str = "PAS"
    rotate_labels: int = 45
    order: Optional[List[str]] = None
    dpi: int = 150
    notch: bool = False
    width: float = 0.7
    fliersize: float = 3.0
    alpha: float = 0.8
    jitter: float = 0.2


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def _check_seaborn():
    """Check if seaborn is available."""
    if not HAS_SEABORN:
        raise ImportError(
            "seaborn is required for visualization. "
            "Install it with: pip install seaborn"
        )


def plot_pas_heatmap(
    pas_df: pd.DataFrame,
    config: Optional[HeatmapConfig] = None,
    group_labels: Optional[pd.Series] = None,
    pathways: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Figure]:
    """Create a heatmap visualization of PAS values.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix with samples as rows and pathways as columns.
    config : HeatmapConfig, optional
        Visualization configuration.
    group_labels : pd.Series, optional
        Group labels for column annotation.
    pathways : List[str], optional
        Subset of pathways to include.
    output_path : str or Path, optional
        Path to save the figure. If None, returns the figure.

    Returns
    -------
    Figure or None
        Matplotlib figure if output_path is None, otherwise None.
    """
    _check_matplotlib()
    _check_seaborn()

    if config is None:
        config = HeatmapConfig()

    # Subset pathways if specified
    if pathways is not None:
        pathways = [p for p in pathways if p in pas_df.columns]
        pas_df = pas_df[pathways]

    # Transpose so pathways are rows
    data = pas_df.T

    # Prepare column colors based on group labels
    col_colors = None
    if group_labels is not None:
        common_samples = data.columns.intersection(group_labels.index)
        group_labels = group_labels.loc[common_samples]
        data = data[common_samples]

        # Create color mapping for groups
        unique_groups = group_labels.unique()
        palette = sns.color_palette(config.cmap if len(unique_groups) <= 10 else "husl", len(unique_groups))
        group_color_map = dict(zip(unique_groups, palette))
        col_colors = group_labels.map(group_color_map)
        col_colors.name = "Group"

    # Handle robust color scaling
    vmin, vmax = config.vmin, config.vmax
    if config.robust and vmin is None and vmax is None:
        vmin = np.percentile(data.values[~np.isnan(data.values)], 2)
        vmax = np.percentile(data.values[~np.isnan(data.values)], 98)

    # Create clustermap
    try:
        g = sns.clustermap(
            data,
            figsize=config.figsize,
            cmap=config.cmap,
            center=config.center,
            vmin=vmin,
            vmax=vmax,
            row_cluster=config.cluster_rows,
            col_cluster=config.cluster_cols,
            method=config.cluster_method.value if config.cluster_rows or config.cluster_cols else None,
            metric=config.cluster_metric.value if config.cluster_rows or config.cluster_cols else None,
            col_colors=col_colors,
            xticklabels=config.show_col_labels,
            yticklabels=config.show_row_labels,
            dendrogram_ratio=config.dendrogram_ratio,
            linewidths=config.linewidths,
            linecolor=config.linecolor,
            cbar_kws={"label": config.cbar_label},
        )
    except Exception as e:
        logging.warning("Clustering failed, creating heatmap without clustering: %s", e)
        g = sns.clustermap(
            data,
            figsize=config.figsize,
            cmap=config.cmap,
            center=config.center,
            vmin=vmin,
            vmax=vmax,
            row_cluster=False,
            col_cluster=False,
            col_colors=col_colors,
            xticklabels=config.show_col_labels,
            yticklabels=config.show_row_labels,
            cbar_kws={"label": config.cbar_label},
        )

    # Set title
    g.fig.suptitle(config.title, y=1.02, fontsize=14, fontweight='bold')

    # Adjust label sizes
    if config.show_row_labels:
        g.ax_heatmap.tick_params(axis='y', labelsize=config.row_label_size)
    if config.show_col_labels:
        g.ax_heatmap.tick_params(axis='x', labelsize=config.col_label_size)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

    # Add legend for groups if applicable
    if group_labels is not None and col_colors is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=group_color_map[g])
                   for g in unique_groups]
        g.ax_heatmap.legend(
            handles, unique_groups,
            title="Group",
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
        )

    plt.tight_layout()

    if output_path is not None:
        g.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        logging.info("Saved heatmap to %s", output_path)
        plt.close(g.fig)
        return None

    return g.fig


def plot_pas_boxplot(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    pathway: str,
    config: Optional[BoxplotConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Figure]:
    """Create a boxplot of PAS values for a specific pathway across groups.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix with samples as rows and pathways as columns.
    group_labels : pd.Series
        Group labels for each sample.
    pathway : str
        Pathway to visualize.
    config : BoxplotConfig, optional
        Visualization configuration.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    Figure or None
        Matplotlib figure if output_path is None.
    """
    _check_matplotlib()
    _check_seaborn()

    if config is None:
        config = BoxplotConfig()

    if pathway not in pas_df.columns:
        raise ValueError(f"Pathway '{pathway}' not found in PAS matrix")

    # Align data
    common_samples = pas_df.index.intersection(group_labels.index)
    pas_values = pas_df.loc[common_samples, pathway]
    groups = group_labels.loc[common_samples]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        "PAS": pas_values.values,
        "Group": groups.values,
    })

    # Set order
    order = config.order
    if order is None:
        order = sorted(groups.unique())

    fig, ax = plt.subplots(figsize=config.figsize)

    if config.show_violin:
        # Violin plot
        sns.violinplot(
            data=plot_data,
            x="Group",
            y="PAS",
            order=order,
            palette=config.palette,
            ax=ax,
            inner="box" if not config.show_points else None,
            alpha=config.alpha,
        )
        if config.show_points:
            sns.stripplot(
                data=plot_data,
                x="Group",
                y="PAS",
                order=order,
                color="black",
                size=config.point_size,
                alpha=0.5,
                ax=ax,
                jitter=config.jitter,
            )
    else:
        # Boxplot
        sns.boxplot(
            data=plot_data,
            x="Group",
            y="PAS",
            order=order,
            palette=config.palette,
            ax=ax,
            notch=config.notch,
            width=config.width,
            fliersize=config.fliersize,
        )
        if config.show_points:
            sns.stripplot(
                data=plot_data,
                x="Group",
                y="PAS",
                order=order,
                color="black",
                size=config.point_size,
                alpha=0.5,
                ax=ax,
                jitter=config.jitter,
            )

    # Show mean if requested
    if config.show_mean:
        for i, group in enumerate(order):
            group_mean = plot_data[plot_data["Group"] == group]["PAS"].mean()
            ax.scatter([i], [group_mean], marker='D', color='red', s=50, zorder=5)

    # Set title and labels
    title = config.title if config.title != "PAS Distribution by Group" else f"{pathway}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(config.xlabel, fontsize=12)
    ax.set_ylabel(config.ylabel, fontsize=12)

    # Rotate x labels
    if config.rotate_labels != 0:
        plt.xticks(rotation=config.rotate_labels, ha='right')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        logging.info("Saved boxplot to %s", output_path)
        plt.close(fig)
        return None

    return fig


def plot_pas_boxplot_multi(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    pathways: List[str],
    config: Optional[BoxplotConfig] = None,
    ncols: int = 3,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Figure]:
    """Create a grid of boxplots for multiple pathways.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix with samples as rows and pathways as columns.
    group_labels : pd.Series
        Group labels for each sample.
    pathways : List[str]
        List of pathways to visualize.
    config : BoxplotConfig, optional
        Visualization configuration.
    ncols : int
        Number of columns in the grid.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    Figure or None
        Matplotlib figure if output_path is None.
    """
    _check_matplotlib()
    _check_seaborn()

    if config is None:
        config = BoxplotConfig()

    # Filter valid pathways
    pathways = [p for p in pathways if p in pas_df.columns]
    n_pathways = len(pathways)

    if n_pathways == 0:
        raise ValueError("No valid pathways found in PAS matrix")

    # Calculate grid dimensions
    nrows = int(np.ceil(n_pathways / ncols))
    figsize = (config.figsize[0] * ncols / 2, config.figsize[1] * nrows / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Align data
    common_samples = pas_df.index.intersection(group_labels.index)
    pas_df = pas_df.loc[common_samples]
    groups = group_labels.loc[common_samples]

    order = config.order
    if order is None:
        order = sorted(groups.unique())

    for idx, pathway in enumerate(pathways):
        ax = axes[idx]

        plot_data = pd.DataFrame({
            "PAS": pas_df[pathway].values,
            "Group": groups.values,
        })

        if config.show_violin:
            sns.violinplot(
                data=plot_data,
                x="Group",
                y="PAS",
                order=order,
                palette=config.palette,
                ax=ax,
                inner="box",
            )
        else:
            sns.boxplot(
                data=plot_data,
                x="Group",
                y="PAS",
                order=order,
                palette=config.palette,
                ax=ax,
                width=config.width,
                fliersize=config.fliersize,
            )

        if config.show_points:
            sns.stripplot(
                data=plot_data,
                x="Group",
                y="PAS",
                order=order,
                color="black",
                size=config.point_size * 0.7,
                alpha=0.4,
                ax=ax,
                jitter=config.jitter,
            )

        ax.set_title(pathway, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("PAS" if idx % ncols == 0 else "")
        ax.tick_params(axis='x', rotation=config.rotate_labels)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Hide unused axes
    for idx in range(n_pathways, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(config.title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        logging.info("Saved multi-boxplot to %s", output_path)
        plt.close(fig)
        return None

    return fig


def plot_pas_volcano(
    results_df: pd.DataFrame,
    effect_col: str = "effect_size",
    pvalue_col: str = "pvalue_adj",
    pathway_col: str = "pathway",
    alpha: float = 0.05,
    effect_threshold: float = 0.5,
    title: str = "PAS Differential Analysis",
    figsize: Tuple[float, float] = (10, 8),
    top_n_labels: int = 10,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Figure]:
    """Create a volcano plot of PAS differential analysis results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from test_pas_difference.
    effect_col : str
        Column name for effect size.
    pvalue_col : str
        Column name for p-values.
    pathway_col : str
        Column name for pathway names.
    alpha : float
        Significance threshold.
    effect_threshold : float
        Effect size threshold for highlighting.
    title : str
        Plot title.
    figsize : Tuple[float, float]
        Figure size.
    top_n_labels : int
        Number of top significant pathways to label.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    Figure or None
        Matplotlib figure if output_path is None.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate -log10(p-value)
    log_pval = -np.log10(results_df[pvalue_col].replace(0, 1e-300))
    effect = results_df[effect_col]

    # Categorize points
    sig_up = (results_df[pvalue_col] < alpha) & (effect > effect_threshold)
    sig_down = (results_df[pvalue_col] < alpha) & (effect < -effect_threshold)
    sig_mid = (results_df[pvalue_col] < alpha) & (~sig_up) & (~sig_down)
    not_sig = results_df[pvalue_col] >= alpha

    # Plot points
    ax.scatter(effect[not_sig], log_pval[not_sig], c='gray', alpha=0.5, s=30, label='Not significant')
    ax.scatter(effect[sig_mid], log_pval[sig_mid], c='blue', alpha=0.7, s=40, label='Significant')
    ax.scatter(effect[sig_up], log_pval[sig_up], c='red', alpha=0.8, s=50, label='Up')
    ax.scatter(effect[sig_down], log_pval[sig_down], c='green', alpha=0.8, s=50, label='Down')

    # Add threshold lines
    ax.axhline(-np.log10(alpha), color='gray', linestyle='--', linewidth=1)
    ax.axvline(effect_threshold, color='gray', linestyle=':', linewidth=1)
    ax.axvline(-effect_threshold, color='gray', linestyle=':', linewidth=1)

    # Label top pathways
    sig_results = results_df[results_df[pvalue_col] < alpha].nsmallest(top_n_labels, pvalue_col)
    for _, row in sig_results.iterrows():
        ax.annotate(
            row[pathway_col],
            (row[effect_col], -np.log10(row[pvalue_col])),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.9,
        )

    ax.set_xlabel(f"Effect Size ({effect_col})", fontsize=12)
    ax.set_ylabel(f"-log10({pvalue_col})", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info("Saved volcano plot to %s", output_path)
        plt.close(fig)
        return None

    return fig


def plot_pathway_correlation(
    pas_df: pd.DataFrame,
    method: str = "spearman",
    title: str = "Pathway Correlation Matrix",
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = "coolwarm",
    cluster: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Figure]:
    """Plot correlation matrix between pathways.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix with samples as rows and pathways as columns.
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'.
    title : str
        Plot title.
    figsize : Tuple[float, float]
        Figure size.
    cmap : str
        Colormap name.
    cluster : bool
        Whether to cluster pathways.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    Figure or None
        Matplotlib figure if output_path is None.
    """
    _check_matplotlib()
    _check_seaborn()

    # Compute correlation matrix
    corr = pas_df.corr(method=method)

    if cluster:
        g = sns.clustermap(
            corr,
            figsize=figsize,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
        )
        g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        fig = g.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": f"{method.capitalize()} Correlation"},
        )
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info("Saved correlation plot to %s", output_path)
        plt.close(fig)
        return None

    return fig


def create_pas_report_figures(
    pas_df: pd.DataFrame,
    group_labels: pd.Series,
    results_df: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = ".",
    prefix: str = "pas",
    top_n_pathways: int = 10,
) -> Dict[str, Path]:
    """Generate a set of standard PAS visualization figures.

    Parameters
    ----------
    pas_df : pd.DataFrame
        PAS matrix.
    group_labels : pd.Series
        Group labels.
    results_df : pd.DataFrame, optional
        Results from test_pas_difference.
    output_dir : str or Path
        Output directory for figures.
    prefix : str
        Filename prefix.
    top_n_pathways : int
        Number of top pathways to visualize.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping figure names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Heatmap
    heatmap_path = output_dir / f"{prefix}_heatmap.png"
    try:
        plot_pas_heatmap(
            pas_df,
            group_labels=group_labels,
            config=HeatmapConfig(title="PAS Heatmap", cluster_rows=True, cluster_cols=True),
            output_path=heatmap_path,
        )
        figures["heatmap"] = heatmap_path
    except Exception as e:
        logging.warning("Failed to create heatmap: %s", e)

    # Get top pathways to visualize
    if results_df is not None and "pvalue_adj" in results_df.columns:
        top_pathways = results_df.nsmallest(top_n_pathways, "pvalue_adj")["pathway"].tolist()
    else:
        # Use pathways with highest variance
        top_pathways = pas_df.var().nlargest(top_n_pathways).index.tolist()

    # Multi-boxplot for top pathways
    if len(top_pathways) > 0:
        boxplot_path = output_dir / f"{prefix}_boxplots.png"
        try:
            plot_pas_boxplot_multi(
                pas_df,
                group_labels,
                top_pathways,
                config=BoxplotConfig(title="Top Differential Pathways"),
                output_path=boxplot_path,
            )
            figures["boxplots"] = boxplot_path
        except Exception as e:
            logging.warning("Failed to create boxplots: %s", e)

    # Volcano plot if results available
    if results_df is not None:
        volcano_path = output_dir / f"{prefix}_volcano.png"
        try:
            plot_pas_volcano(
                results_df,
                title="PAS Differential Analysis",
                output_path=volcano_path,
            )
            figures["volcano"] = volcano_path
        except Exception as e:
            logging.warning("Failed to create volcano plot: %s", e)

    # Correlation plot
    corr_path = output_dir / f"{prefix}_correlation.png"
    try:
        plot_pathway_correlation(
            pas_df,
            title="Pathway Correlation Matrix",
            output_path=corr_path,
        )
        figures["correlation"] = corr_path
    except Exception as e:
        logging.warning("Failed to create correlation plot: %s", e)

    logging.info("Generated %d figures in %s", len(figures), output_dir)
    return figures
