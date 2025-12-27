## ------------------------------------------------------------------------------------------- ##
## LD Reference Module                                                                        ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                             ##
## @description: Exposes LD reference setup and management functionality for PathWAS.         ##
##               Enables downloading and configuring LD panels from 1000 Genomes data.        ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""LD Reference Module.

Provides functionality for downloading and managing LD (linkage disequilibrium)
reference panels from 1000 Genomes data.
"""

from pathwas.ld.setup import (
    SUPPORTED_ANCESTRIES,
    compute_ld_matrix,
    compute_maf,
    download_ld_blocks,
    download_plink_files,
    list_ancestries,
    read_plink_genotypes,
    setup_ld_reference,
)

__all__ = [
    "SUPPORTED_ANCESTRIES",
    "setup_ld_reference",
    "download_plink_files",
    "download_ld_blocks",
    "read_plink_genotypes",
    "compute_ld_matrix",
    "compute_maf",
    "list_ancestries",
]
