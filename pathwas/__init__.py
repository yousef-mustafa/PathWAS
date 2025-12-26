## ------------------------------------------------------------------------------------------- ##
## PathWAS Package Initialization                                                              ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                        ##
##                                                                                             ##
## @description: Exposes the main interfaces for pathway-wide association study workflows,     ##
##               including LD pruning, covariance computation, PAS calculation and testing.    ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway-wide association study toolkit."""

from .io.vcf_processing import ld_prune
from .io.covariance import load_vcf_as_matrix, compute_covariance
from .pas.pas import compute_pas, registry as pas_registry
from .association.pathway_test import (
    aggregate_variants,
    association_test,
    test_pathway_twas,
)
from .logging.logging_util import configure_logging
from .io.data_prep import convert_gene_ids, convert_gene_list, load_msigdb_library
from .association.genetic_correlation import generate_pas_summary_stats, run_ldsc_rg
from .ld.setup import setup_ld_reference, SUPPORTED_ANCESTRIES

__all__ = [
    'ld_prune',
    'load_vcf_as_matrix',
    'compute_covariance',
    'compute_pas',
    'pas_registry',
    'aggregate_variants',
    'association_test',
    'test_pathway_twas',
    'convert_gene_ids',
    'convert_gene_list',
    'load_msigdb_library',
    'generate_pas_summary_stats',
    'run_ldsc_rg',
    'configure_logging',
    'setup_ld_reference',
    'SUPPORTED_ANCESTRIES',
]