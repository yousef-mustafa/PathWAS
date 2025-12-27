## ------------------------------------------------------------------------------------------- ##
## PathWAS Logging Utilities                                                                 ##
## ------------------------------------------------------------------------------------------- ##
## @script: logging_util.py                                                                  ##
##                                                                                            ##
## @description: Centralized logging configuration for the PathWAS package.                   ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Logging configuration utilities for PathWAS."""

import logging
import sys
from typing import Optional

_configured = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for PathWAS.

    Sets up a consistent logging format across the package. This function is
    idempotent - calling it multiple times will not create duplicate handlers.

    Parameters
    ----------
    level : int
        Logging level (default: logging.INFO). Common values:
        - logging.DEBUG (10): Detailed debugging information
        - logging.INFO (20): General information
        - logging.WARNING (30): Warning messages
        - logging.ERROR (40): Error messages
    log_file : str, optional
        Path to log file. If None, logs to stderr only.
    format_string : str
        Format string for log messages. Default includes timestamp,
        logger name, level, and message.

    Examples
    --------
    >>> from pathwas import configure_logging
    >>> import logging
    >>> configure_logging(level=logging.DEBUG)  # Enable debug output
    >>> configure_logging(log_file="pathwas.log")  # Also log to file
    """
    global _configured
    if _configured:
        return

    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )

    _configured = True


def reset_logging() -> None:
    """Reset logging configuration.

    Primarily used for testing. Allows configure_logging to be called again.
    """
    global _configured
    _configured = False
