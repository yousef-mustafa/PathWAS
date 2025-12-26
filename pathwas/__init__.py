## ------------------------------------------------------------------------------------------- ##
## PathWAS Package Initialization                                                              ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                        ##
##                                                                                             ##
## @description: Exposes the main interfaces for pathway-wide association study workflows,     ##
##               including LD pruning, covariance computation, PAS calculation, modeling,      ##
##               harmonization, association testing, and QC.                                   ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway-wide association study toolkit."""

# IO utilities
from .io.vcf_processing import ld_prune
from .io.covariance import load_vcf_as_matrix, compute_covariance
from .io.data_prep import convert_gene_ids, convert_gene_list, load_msigdb_library
from .io.harmonize import harmonize_weights_gwas, prepare_pathway_vectors, is_palindromic

# PAS computation
from .pas.pas import compute_pas, registry as pas_registry

# Association testing
from .association.pathway_test import (
    aggregate_variants,
    association_test,
    test_pathway_twas,
)
from .association.genetic_correlation import generate_pas_summary_stats, run_ldsc_rg
from .association.pathway_rg import (
    compute_ld_matrix,
    compute_genetic_correlation,
    jackknife_genetic_correlation,
    compute_pathway_rg_for_all,
    GeneticCorrelationResult,
)

# Modeling
from .modeling import (
    ModelConfig,
    PathwayModel,
    RidgePathwayModel,
    create_model,
    register_model,
    available_models,
)

# LD reference support
from .ld import (
    LDReference,
    load_ld_reference,
    load_ld_for_snps,
    align_ld_to_vectors,
    save_block_ld,
)

# QC utilities
from .qc import (
    ReliabilityLevel,
    AncestryQCResult,
    run_ancestry_qc,
    compute_maf_correlation,
)

__all__ = [
    # IO
    'ld_prune',
    'load_vcf_as_matrix',
    'compute_covariance',
    'convert_gene_ids',
    'convert_gene_list',
    'load_msigdb_library',
    'harmonize_weights_gwas',
    'prepare_pathway_vectors',
    'is_palindromic',
    # PAS
    'compute_pas',
    'pas_registry',
    # Association
    'aggregate_variants',
    'association_test',
    'test_pathway_twas',
    'generate_pas_summary_stats',
    'run_ldsc_rg',
    'compute_ld_matrix',
    'compute_genetic_correlation',
    'jackknife_genetic_correlation',
    'compute_pathway_rg_for_all',
    'GeneticCorrelationResult',
    # Modeling
    'ModelConfig',
    'PathwayModel',
    'RidgePathwayModel',
    'create_model',
    'register_model',
    'available_models',
    # LD
    'LDReference',
    'load_ld_reference',
    'load_ld_for_snps',
    'align_ld_to_vectors',
    'save_block_ld',
    # QC
    'ReliabilityLevel',
    'AncestryQCResult',
    'run_ancestry_qc',
    'compute_maf_correlation',
]
