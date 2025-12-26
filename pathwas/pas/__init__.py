## ------------------------------------------------------------------------------------------- ##
## PAS Subpackage Initialization                                                              ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes pathway activation score computation, statistical testing,          ##
##               and visualization utilities.                                                ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway Activation Score (PAS) computation, testing, and visualization subpackage."""

# Core PAS computation
from .pas import compute_pas, registry, PASMethodRegistry, _bicor

# Statistical testing
from .pas_test import (
    TestMethod,
    MultipleTestingCorrection,
    PASTestResult,
    PASTestConfig,
    test_pas_difference,
    test_pas_with_covariates,
    pairwise_group_tests,
    summarize_test_results,
)

# Visualization
from .pas_viz import (
    ColorPalette,
    ClusterMethod,
    ClusterMetric,
    HeatmapConfig,
    BoxplotConfig,
    plot_pas_heatmap,
    plot_pas_boxplot,
    plot_pas_boxplot_multi,
    plot_pas_volcano,
    plot_pathway_correlation,
    create_pas_report_figures,
)

__all__ = [
    # Core PAS
    "compute_pas",
    "registry",
    "PASMethodRegistry",
    "_bicor",
    # Testing
    "TestMethod",
    "MultipleTestingCorrection",
    "PASTestResult",
    "PASTestConfig",
    "test_pas_difference",
    "test_pas_with_covariates",
    "pairwise_group_tests",
    "summarize_test_results",
    # Visualization
    "ColorPalette",
    "ClusterMethod",
    "ClusterMetric",
    "HeatmapConfig",
    "BoxplotConfig",
    "plot_pas_heatmap",
    "plot_pas_boxplot",
    "plot_pas_boxplot_multi",
    "plot_pas_volcano",
    "plot_pathway_correlation",
    "create_pas_report_figures",
]
