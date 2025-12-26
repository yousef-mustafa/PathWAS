## ------------------------------------------------------------------------------------------- ##
## Association Subpackage Initialization                                                       ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                        ##
##                                                                                             ##
## @description: Exposes modules for association analyses within the pathWAS toolkit.          ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Association submodule for pathWAS."""

from .genetic_correlation import generate_pas_summary_stats, run_ldsc_rg
from .pathway_test import aggregate_variants, association_test, test_pathway_twas
from .pathway_rg import (
    compute_ld_matrix,
    compute_genetic_correlation,
    compute_genetic_correlation_with_components,
    jackknife_genetic_correlation,
    compute_pathway_rg_for_all,
    GeneticCorrelationResult,
)

__all__ = [
    "generate_pas_summary_stats",
    "run_ldsc_rg",
    "aggregate_variants",
    "association_test",
    "test_pathway_twas",
    "compute_ld_matrix",
    "compute_genetic_correlation",
    "compute_genetic_correlation_with_components",
    "jackknife_genetic_correlation",
    "compute_pathway_rg_for_all",
    "GeneticCorrelationResult",
]