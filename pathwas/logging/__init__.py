## ------------------------------------------------------------------------------------------- ##
## PathWAS Logging Module                                                                     ##
## ------------------------------------------------------------------------------------------- ##
## @script: __init__.py                                                                       ##
##                                                                                            ##
## @description: Exposes logging configuration utilities for PathWAS.                         ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""PathWAS logging module."""

from .logging_util import configure_logging

__all__ = ["configure_logging"]
