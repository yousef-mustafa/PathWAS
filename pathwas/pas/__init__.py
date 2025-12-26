## ------------------------------------------------------------------------------------------- ##
## PAS Subpackage Initialization                                                              ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes pathway activation score computation utilities.                     ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway Activation Score (PAS) computation subpackage."""

from .pas import compute_pas, registry, PASMethodRegistry, _bicor

__all__ = [
    "compute_pas",
    "registry",
    "PASMethodRegistry",
    "_bicor",
]
