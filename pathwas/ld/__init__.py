## ------------------------------------------------------------------------------------------- ##
## LD Subpackage Initialization                                                               ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes LD reference panel utilities for loading external LD resources.     ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""LD reference panel support for external panels like 1000 Genomes."""

from .reference import (
    SNPInfo,
    LDBlock,
    LDReference,
    load_snp_manifest,
    load_block_definitions,
    assign_snps_to_blocks,
    load_ld_reference,
    load_block_ld_matrix,
    load_ld_for_snps,
    align_ld_to_vectors,
    save_block_ld,
)

__all__ = [
    "SNPInfo",
    "LDBlock",
    "LDReference",
    "load_snp_manifest",
    "load_block_definitions",
    "assign_snps_to_blocks",
    "load_ld_reference",
    "load_block_ld_matrix",
    "load_ld_for_snps",
    "align_ld_to_vectors",
    "save_block_ld",
]
