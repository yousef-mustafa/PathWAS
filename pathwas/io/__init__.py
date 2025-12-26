## ------------------------------------------------------------------------------------------- ##
## IO Subpackage Initialization                                                               ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes input/output utilities for VCF processing, covariance computation,  ##
##               data preparation, and harmonization.                                        ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""IO subpackage for data loading, processing, and harmonization."""

from .vcf_processing import ld_prune
from .covariance import load_vcf_as_matrix, compute_covariance
from .data_prep import convert_gene_ids, convert_gene_list, load_msigdb_library
from .harmonize import harmonize_weights_gwas, prepare_pathway_vectors, is_palindromic

# Expression preprocessing utilities
from .expression import (
    normalize_library_size,
    log_transform,
    zscore_genes,
    filter_low_expression,
    filter_low_count_samples,
    preprocess_expression,
)

# Gene set loading utilities
from .gene_sets import (
    load_gene_sets,
    load_gmt,
    load_tabular_gene_sets,
    save_gene_sets_gmt,
    gene_sets_to_dataframe,
    filter_gene_sets_by_size,
    intersect_gene_sets_with_genes,
)

__all__ = [
    # VCF and covariance
    "ld_prune",
    "load_vcf_as_matrix",
    "compute_covariance",
    # Data prep
    "convert_gene_ids",
    "convert_gene_list",
    "load_msigdb_library",
    # Harmonization
    "harmonize_weights_gwas",
    "prepare_pathway_vectors",
    "is_palindromic",
    # Expression preprocessing
    "normalize_library_size",
    "log_transform",
    "zscore_genes",
    "filter_low_expression",
    "filter_low_count_samples",
    "preprocess_expression",
    # Gene set loading
    "load_gene_sets",
    "load_gmt",
    "load_tabular_gene_sets",
    "save_gene_sets_gmt",
    "gene_sets_to_dataframe",
    "filter_gene_sets_by_size",
    "intersect_gene_sets_with_genes",
]
