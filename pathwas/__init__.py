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

# PAS statistical testing
from .pas.pas_test import (
    TestMethod,
    MultipleTestingCorrection,
    PASTestConfig,
    test_pas_difference,
    test_pas_with_covariates,
    pairwise_group_tests,
    summarize_test_results,
)

# PAS visualization
from .pas.pas_viz import (
    HeatmapConfig,
    BoxplotConfig,
    plot_pas_heatmap,
    plot_pas_boxplot,
    plot_pas_boxplot_multi,
    plot_pas_volcano,
    plot_pathway_correlation,
    create_pas_report_figures,
)

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
    BayesMixturePathwayModel,
    BayesMixtureConfig,
    create_model,
    register_model,
    available_models,
)

# Expression preprocessing
from .io.expression import (
    normalize_library_size,
    log_transform,
    zscore_genes,
    filter_low_expression,
    filter_low_count_samples,
    preprocess_expression,
)

# Gene set loading
from .io.gene_sets import (
    load_gene_sets,
    load_gmt,
    load_tabular_gene_sets,
    save_gene_sets_gmt,
    gene_sets_to_dataframe,
    filter_gene_sets_by_size,
    intersect_gene_sets_with_genes,
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

# Experiment orchestration
from .experiment import (
    run_experiment,
    load_config,
    merge_config,
    ensure_experiments_dir,
    setup_experiment_dir,
    validate_config,
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
    # PAS testing
    'TestMethod',
    'MultipleTestingCorrection',
    'PASTestConfig',
    'test_pas_difference',
    'test_pas_with_covariates',
    'pairwise_group_tests',
    'summarize_test_results',
    # PAS visualization
    'HeatmapConfig',
    'BoxplotConfig',
    'plot_pas_heatmap',
    'plot_pas_boxplot',
    'plot_pas_boxplot_multi',
    'plot_pas_volcano',
    'plot_pathway_correlation',
    'create_pas_report_figures',
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
    'BayesMixturePathwayModel',
    'BayesMixtureConfig',
    'create_model',
    'register_model',
    'available_models',
    # Expression preprocessing
    'normalize_library_size',
    'log_transform',
    'zscore_genes',
    'filter_low_expression',
    'filter_low_count_samples',
    'preprocess_expression',
    # Gene set loading
    'load_gene_sets',
    'load_gmt',
    'load_tabular_gene_sets',
    'save_gene_sets_gmt',
    'gene_sets_to_dataframe',
    'filter_gene_sets_by_size',
    'intersect_gene_sets_with_genes',
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
    # Experiment orchestration
    'run_experiment',
    'load_config',
    'merge_config',
    'ensure_experiments_dir',
    'setup_experiment_dir',
    'validate_config',
]
