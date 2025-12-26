## ------------------------------------------------------------------------------------------- ##
## PathWAS Command Line Interface                                                             ##
## ------------------------------------------------------------------------------------------- ##
## @script: cli.py                                                                            ##
##                                                                                             ##
## @description: Unified command-line interface for PathWAS experiments. Supports config      ##
##               files, direct CLI arguments, and hybrid mode with overrides.                 ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""PathWAS command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .experiment import (
    load_config,
    merge_config,
    build_config_from_args,
    run_experiment,
    validate_config,
    ensure_experiments_dir,
    DEFAULT_CONFIG,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for PathWAS CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="pathwas",
        description=(
            "PathWAS: Pathway-Wide Association Studies\n\n"
            "Run pathway-level transcriptome-wide association analyses using "
            "config files and/or command-line arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Config-driven mode:
  pathwas --config configs/ad_resilience_1kg.yaml

  # Config with CLI overrides:
  pathwas --config configs/base.yaml --model bayes_mixture --experiment-name my_exp

  # Argument-only mode:
  pathwas --expression data/expr.tsv --genotypes data/geno.bed --pathways hallmark

For more information, see: https://github.com/yousef-mustafa/pathWAS
        """,
    )

    # ----------------------------- Config file -------------------------------- #
    parser.add_argument(
        "--config", "-c",
        type=str,
        metavar="PATH",
        help="Path to YAML or JSON configuration file",
    )

    # ----------------------------- Experiment metadata ------------------------ #
    exp_group = parser.add_argument_group("Experiment metadata")
    exp_group.add_argument(
        "--experiment-name", "-n",
        type=str,
        metavar="NAME",
        help="Name for the experiment (used as subdirectory name under experiments/)",
    )
    exp_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment directory if it exists",
    )
    exp_group.add_argument(
        "--note",
        type=str,
        metavar="TEXT",
        help="Free-text description of the experiment",
    )

    # ----------------------------- Data inputs -------------------------------- #
    data_group = parser.add_argument_group("Data inputs")
    data_group.add_argument(
        "--expression", "-e",
        type=str,
        metavar="PATH",
        help="Path to gene expression matrix (samples x genes, CSV/TSV)",
    )
    data_group.add_argument(
        "--genotypes", "-g",
        type=str,
        metavar="PATH",
        help="Path to genotype data (PLINK .bed/.bim/.fam, VCF, or CSV)",
    )
    data_group.add_argument(
        "--covariates",
        type=str,
        metavar="PATH",
        help="Path to covariates file (samples x covariates, CSV/TSV)",
    )

    # ----------------------------- Pathway settings --------------------------- #
    pathway_group = parser.add_argument_group("Pathway settings")
    pathway_group.add_argument(
        "--pathways", "-p",
        type=str,
        choices=["kegg", "reactome", "wikipathways", "go", "go_bp", "go_mf",
                 "go_cc", "hallmark", "custom"],
        default="hallmark",
        help="Pathway database source (default: hallmark)",
    )
    pathway_group.add_argument(
        "--pathways-file",
        type=str,
        metavar="PATH",
        help="Path to custom gene sets JSON file (required if --pathways=custom)",
    )

    # ----------------------------- Preprocessing ------------------------------ #
    preproc_group = parser.add_argument_group("Preprocessing")
    preproc_group.add_argument(
        "--expr-normalization",
        type=str,
        choices=["CPM", "TPM", "none"],
        default="none",
        help="Expression normalization method (default: none)",
    )
    preproc_group.add_argument(
        "--expr-log-transform",
        action="store_true",
        help="Apply log2(x + 1) transformation to expression",
    )
    preproc_group.add_argument(
        "--expr-zscore-genes",
        action="store_true",
        help="Z-score normalize expression across samples for each gene",
    )

    # ----------------------------- Modeling ----------------------------------- #
    model_group = parser.add_argument_group("Modeling")
    model_group.add_argument(
        "--model", "-m",
        type=str,
        choices=["ridge", "bayes_mixture", "elastic_net", "lasso"],
        default="ridge",
        help="Modeling method for SNP-to-PAS (default: ridge)",
    )
    model_group.add_argument(
        "--lambda",
        type=float,
        dest="lambda_param",
        metavar="VALUE",
        help="Regularization parameter lambda (for ridge/lasso/elastic_net)",
    )
    model_group.add_argument(
        "--mixture-components",
        type=int,
        metavar="N",
        help="Number of mixture components (for bayes_mixture model)",
    )
    model_group.add_argument(
        "--alpha",
        type=float,
        metavar="VALUE",
        help="Elastic net mixing parameter (0=ridge, 1=lasso)",
    )

    # ----------------------------- LD / Association --------------------------- #
    ld_group = parser.add_argument_group("LD and association settings")
    ld_group.add_argument(
        "--ld-root",
        type=str,
        metavar="PATH",
        help="Root directory for LD reference panel files",
    )
    ld_group.add_argument(
        "--ld-ancestry",
        type=str,
        default="EUR_1KG",
        metavar="TAG",
        help="LD ancestry tag (e.g., EUR_1KG, AFR_1KG; default: EUR_1KG)",
    )

    # ----------------------------- Output settings ---------------------------- #
    output_group = parser.add_argument_group("Output settings")
    output_group.add_argument(
        "--output-dir",
        type=str,
        metavar="PATH",
        help="Base directory for experiments (default: experiments/)",
    )
    output_group.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Do not save intermediate files (preprocessed expression, etc.)",
    )

    # ----------------------------- Verbosity ---------------------------------- #
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (use -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Merge CLI arguments as overrides into an existing config.

    Parameters
    ----------
    config : dict
        Base configuration dictionary (loaded from file or defaults).
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration with CLI overrides applied.
    """
    overrides: dict = {"experiment": {}, "data": {}, "pathways": {},
                       "preprocessing": {}, "modeling": {"model_params": {}},
                       "association": {}, "output": {}}

    # Experiment metadata
    if args.experiment_name is not None:
        overrides["experiment"]["name"] = args.experiment_name
    if args.overwrite:
        overrides["experiment"]["overwrite"] = True
    if args.note is not None:
        overrides["experiment"]["description"] = args.note

    # Data inputs
    if args.expression is not None:
        overrides["data"]["expression"] = args.expression
    if args.genotypes is not None:
        overrides["data"]["genotypes"] = args.genotypes
    if args.covariates is not None:
        overrides["data"]["covariates"] = args.covariates

    # Pathway settings
    if args.pathways is not None:
        overrides["pathways"]["source"] = args.pathways
    if args.pathways_file is not None:
        overrides["pathways"]["file"] = args.pathways_file

    # Preprocessing
    if args.expr_normalization is not None:
        overrides["preprocessing"]["normalization"] = args.expr_normalization
    if args.expr_log_transform:
        overrides["preprocessing"]["log_transform"] = True
    if args.expr_zscore_genes:
        overrides["preprocessing"]["zscore_genes"] = True

    # Modeling
    if args.model is not None:
        overrides["modeling"]["model_type"] = args.model
    if args.lambda_param is not None:
        overrides["modeling"]["model_params"]["lambda"] = args.lambda_param
    if args.mixture_components is not None:
        overrides["modeling"]["model_params"]["mixture_components"] = args.mixture_components
    if args.alpha is not None:
        overrides["modeling"]["model_params"]["alpha"] = args.alpha

    # LD / Association
    if args.ld_root is not None:
        overrides["association"]["ld_root"] = args.ld_root
    if args.ld_ancestry is not None:
        overrides["association"]["ld_ancestry"] = args.ld_ancestry

    # Output settings
    if args.output_dir is not None:
        overrides["output"]["base_dir"] = args.output_dir
    if args.no_save_intermediate:
        overrides["output"]["save_intermediate"] = False

    # Clean up empty dicts
    def remove_empty(d: dict) -> dict:
        return {k: remove_empty(v) if isinstance(v, dict) else v
                for k, v in d.items() if v or v is False or v == 0}

    overrides = remove_empty(overrides)
    return merge_config(config, overrides)


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging based on CLI verbosity arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[list] = None) -> int:
    """Main entry point for PathWAS CLI.

    Parameters
    ----------
    argv : list, optional
        Command-line arguments. Defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args)

    # Ensure experiments directory exists
    ensure_experiments_dir()

    try:
        # Build configuration
        if args.config:
            # Config-driven mode: load config and apply CLI overrides
            logging.info("Loading configuration from %s", args.config)
            config = load_config(args.config)
            config = merge_cli_overrides(config, args)
        else:
            # Argument-only mode: build config from CLI args
            logging.info("Building configuration from command-line arguments")
            config = build_config_from_args(args)
            # Apply any additional overrides
            config = merge_cli_overrides(config, args)

        # Validate configuration
        errors = validate_config(config)
        if errors:
            for error in errors:
                logging.error("Configuration error: %s", error)
            parser.print_usage()
            print("\nConfiguration errors found. Use --config or provide required arguments.")
            print("Required: --expression (at minimum)")
            return 1

        # Run experiment
        logging.info("Starting PathWAS experiment")
        result = run_experiment(config)

        exp_dir = result.get("experiment_dir", "unknown")
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {exp_dir}")
        print(f"\nKey outputs:")
        print(f"  - Configuration: {exp_dir}/config.yaml")
        print(f"  - Metadata: {exp_dir}/experiment_metadata.json")
        print(f"  - Report: {exp_dir}/experiment_report.html")

        return 0

    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        return 1
    except FileExistsError as e:
        logging.error("%s", e)
        print("\nTip: Use --overwrite to replace an existing experiment directory.")
        return 1
    except ValueError as e:
        logging.error("Invalid value: %s", e)
        return 1
    except ImportError as e:
        logging.error("Missing dependency: %s", e)
        print("\nInstall missing dependencies with: pip install -e .[all]")
        return 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
