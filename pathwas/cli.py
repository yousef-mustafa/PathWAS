## ------------------------------------------------------------------------------------------- ##
## PathWAS Command Line Interface                                                             ##
## ------------------------------------------------------------------------------------------- ##
## @script: cli.py                                                                            ##
##                                                                                             ##
## @description: Provide a command line interface for LD pruning, covariance calculation,      ##
##               pathway activation scoring and association testing.                           ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

import argparse
import json
import logging

import pandas as pd

from pathlib import Path

from .io.vcf_processing import ld_prune
from .io.covariance import load_vcf_as_matrix, compute_covariance
from .pas.pas import compute_pas
from .association.pathway_test import aggregate_variants, association_test
from .logging.logging_util import configure_logging
from .io.data_prep import (
    convert_gene_ids,
    convert_gene_list,
    load_msigdb_library,
)
from .ld.setup import (
    SUPPORTED_ANCESTRIES,
    list_ancestries,
    setup_ld_reference,
)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Pathway-level association analysis")
    subparsers = parser.add_subparsers(dest='command')

    prune_p = subparsers.add_parser('ld-prune', help='LD prune a VCF using PLINK')
    prune_p.add_argument('vcf')
    prune_p.add_argument('--out', default='ld_pruned')
    prune_p.add_argument('--plink', default='plink')

    cov_p = subparsers.add_parser('cov', help='Compute variant covariance from VCF')
    cov_p.add_argument('vcf')
    cov_p.add_argument('--out', default='covariance.csv')

    pas_p = subparsers.add_parser('pas', help='Compute pathway activation scores')
    pas_p.add_argument('expression', help='Gene expression matrix (samples x genes)')
    pas_p.add_argument('--pathways', help='JSON file mapping pathway name to list of genes')
    pas_p.add_argument('--msigdb', help='MSigDB gene set library name to use')
    pas_p.add_argument('--method', default='activity_weighted', choices=['sum', 'mean', 'median', 'activity_weighted'])
    pas_p.add_argument('--corr-method', default='bicor', choices=['pearson', 'spearman', 'bicor'])
    pas_p.add_argument('--normalize-samples', action='store_true', help='Z-score PAS across samples')
    pas_p.add_argument('--normalize-pathways', action='store_true', help='Z-score PAS across pathways')
    pas_p.add_argument('--out', default='pas.csv')

    test_p = subparsers.add_parser('test', help='Association test of pathways')
    test_p.add_argument('pas', help='Pathway activation scores CSV')
    test_p.add_argument('genotypes', help='Genotype matrix CSV (samples x variants)')
    test_p.add_argument('pathways', help='JSON mapping pathway name to genes')
    test_p.add_argument('--out', default='results.csv')

    # LD reference setup subcommand
    ld_p = subparsers.add_parser(
        'setup-ld',
        help='Download and set up LD reference panel from 1000 Genomes'
    )
    ld_p.add_argument(
        '--ancestry',
        choices=list(SUPPORTED_ANCESTRIES.keys()),
        help='Population ancestry code (EUR, AFR, or EAS)'
    )
    ld_p.add_argument(
        '--output-dir',
        type=str,
        default='./ld_reference',
        help='Directory to save LD reference files (default: ./ld_reference)'
    )
    ld_p.add_argument(
        '--chromosomes',
        type=str,
        help='Comma-separated list of chromosomes to process (default: 1-22)'
    )
    ld_p.add_argument(
        '--maf-threshold',
        type=float,
        default=0.01,
        help='Minimum minor allele frequency (default: 0.01)'
    )
    ld_p.add_argument(
        '--keep-downloads',
        action='store_true',
        help='Keep intermediate PLINK files after processing'
    )
    ld_p.add_argument(
        '--list',
        action='store_true',
        dest='list_ancestries',
        help='List available ancestries and exit'
    )

    args = parser.parse_args()

    if args.command == 'ld-prune':
        logging.info("Running LD pruning on %s", args.vcf)
        pruned_vcf = ld_prune(args.vcf, plink_path=args.plink, out_dir=args.out)
        logging.info("Pruned VCF written to %s", pruned_vcf)
        print(pruned_vcf)
    elif args.command == 'cov':
        logging.info("Computing covariance from %s", args.vcf)
        gt = load_vcf_as_matrix(args.vcf)
        cov = compute_covariance(gt)
        cov.to_csv(args.out)
        logging.info("Covariance matrix written to %s", args.out)
    elif args.command == 'pas':
        logging.info("Computing PAS using %s", args.expression)
        expr = pd.read_csv(args.expression, index_col=0)
        expr = convert_gene_ids(expr)

        if args.msigdb:
            logging.info("Fetching pathways from MSigDB library %s", args.msigdb)
            pathways = load_msigdb_library(args.msigdb)
            pathways = {pw: convert_gene_list(gs) for pw, gs in pathways.items()}
        else:
            if not args.pathways:
                parser.error("Either --msigdb or --pathways is required")
            with open(args.pathways) as fh:
                pathways = json.load(fh)
            pathways = {pw: convert_gene_list(gs) for pw, gs in pathways.items()}

        pas, _ = compute_pas(
            expr,
            pathways,
            method=args.method,
            corr_method=args.corr_method,
            normalize_samples=args.normalize_samples,
            normalize_pathways=args.normalize_pathways,
        )
        pas.to_csv(args.out)
        logging.info("PAS written to %s", args.out)
    elif args.command == 'test':
        logging.info("Running association test")
        pas = pd.read_csv(args.pas, index_col=0)
        genotypes = pd.read_csv(args.genotypes, index_col=0)
        with open(args.pathways) as fh:
            pathways = json.load(fh)
        variants = aggregate_variants(genotypes, pathways)
        res = association_test(pas, variants)
        res.to_csv(args.out, index=False)
        logging.info("Association results written to %s", args.out)
    elif args.command == 'setup-ld':
        if args.list_ancestries:
            list_ancestries()
        elif args.ancestry:
            # Parse chromosomes if provided
            chromosomes = None
            if args.chromosomes:
                try:
                    chromosomes = [int(c.strip()) for c in args.chromosomes.split(',')]
                except ValueError:
                    parser.error("Chromosomes must be comma-separated integers")

            success = setup_ld_reference(
                ancestry=args.ancestry,
                output_dir=Path(args.output_dir),
                chromosomes=chromosomes,
                maf_threshold=args.maf_threshold,
                keep_downloads=args.keep_downloads,
            )
            if not success:
                logging.error("LD reference setup failed")
                exit(1)
        else:
            print("Error: --ancestry is required unless using --list")
            print()
            ld_p.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()