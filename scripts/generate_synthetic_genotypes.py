#!/usr/bin/env python
"""
Generate Synthetic Genotype Data for PathWAS Testing.

This script generates realistic synthetic VCF and genotype matrix files with:
- Hardy-Weinberg equilibrium genotypes
- Configurable minor allele frequencies
- LD block structure
- Proper VCF formatting

Output files:
- genotypes.vcf.gz: Compressed VCF file
- genotypes.csv: Simple matrix format for CLI testing
- variant_info.csv: Variant metadata

Usage:
    python scripts/generate_synthetic_genotypes.py \\
        --n-samples 200 \\
        --n-variants-per-chr 1000 \\
        --chromosomes 1,2,22 \\
        --maf-min 0.01 \\
        --maf-max 0.5 \\
        --ld-block-size 50 \\
        --seed 42 \\
        --output-dir data/raw \\
        --sample-ids data/raw/sample_metadata.csv
"""

import argparse
import gzip
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Chromosome lengths (GRCh38, in Mb) for realistic positions
CHROMOSOME_LENGTHS = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}

# Reference and alternate alleles
BASES = ["A", "C", "G", "T"]


def generate_allele_frequencies(
    n_variants: int,
    maf_min: float = 0.01,
    maf_max: float = 0.5,
    distribution: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate minor allele frequencies following a realistic distribution.

    Args:
        n_variants: Number of variants
        maf_min: Minimum MAF
        maf_max: Maximum MAF
        distribution: "uniform" or "beta" (skewed toward rare variants)
        seed: Random seed

    Returns:
        Array of MAFs
    """
    np.random.seed(seed)

    if distribution == "uniform":
        mafs = np.random.uniform(maf_min, maf_max, size=n_variants)
    elif distribution == "beta":
        # Beta distribution skewed toward rare variants
        # Shape parameters chosen to give realistic MAF distribution
        mafs = np.random.beta(a=0.5, b=2.0, size=n_variants)
        mafs = mafs * (maf_max - maf_min) + maf_min
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return mafs


def generate_genotypes_hwe(
    n_samples: int,
    maf: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate genotypes in Hardy-Weinberg equilibrium.

    Under HWE, genotype frequencies are:
    - P(AA) = (1-p)^2
    - P(Aa) = 2*p*(1-p)
    - P(aa) = p^2

    where p is the minor allele frequency.

    Args:
        n_samples: Number of samples
        maf: Minor allele frequency
        seed: Random seed

    Returns:
        Array of genotypes (0, 1, or 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # HWE genotype frequencies
    p = maf
    q = 1 - p

    freq_aa = q * q  # Homozygous reference
    freq_Aa = 2 * p * q  # Heterozygous
    freq_AA = p * p  # Homozygous alternate

    # Generate genotypes
    genotypes = np.random.choice(
        [0, 1, 2],
        size=n_samples,
        p=[freq_aa, freq_Aa, freq_AA]
    )

    return genotypes


def add_ld_structure(
    genotypes: np.ndarray,
    block_size: int = 50,
    r2_within_block: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    """
    Add LD correlation structure within blocks.

    Creates blocks of variants where variants are correlated with
    a "tag" SNP at the beginning of each block.

    Args:
        genotypes: Genotype matrix (samples × variants)
        block_size: Number of variants per LD block
        r2_within_block: Target r² within blocks
        seed: Random seed

    Returns:
        Modified genotype matrix with LD structure
    """
    np.random.seed(seed)

    n_samples, n_variants = genotypes.shape
    genotypes_ld = genotypes.copy().astype(float)

    n_blocks = n_variants // block_size

    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_variants)

        # First variant in block is the "tag" SNP
        tag_snp = genotypes_ld[:, start_idx].copy()

        # Correlate other variants with tag SNP
        for var_idx in range(start_idx + 1, end_idx):
            # How much to weight the tag SNP vs random
            r = np.sqrt(r2_within_block) * np.random.uniform(0.5, 1.0)

            # Create correlated genotype
            original = genotypes_ld[:, var_idx]

            # Blend with tag SNP
            blended = r * tag_snp + (1 - r) * original

            # Round to valid genotypes with added noise
            noise = np.random.normal(0, 0.1, size=n_samples)
            genotypes_ld[:, var_idx] = np.clip(np.round(blended + noise), 0, 2)

    return genotypes_ld.astype(int)


def generate_variant_positions(
    n_variants: int,
    chromosome: str,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic variant positions for a chromosome.

    Args:
        n_variants: Number of variants
        chromosome: Chromosome name
        seed: Random seed

    Returns:
        Array of sorted positions
    """
    np.random.seed(seed)

    chr_length = CHROMOSOME_LENGTHS.get(str(chromosome), 100000000)

    # Leave some buffer at ends
    min_pos = 10000
    max_pos = chr_length - 10000

    # Generate positions with minimum spacing
    positions = np.sort(np.random.choice(
        range(min_pos, max_pos),
        size=n_variants,
        replace=False
    ))

    return positions


def generate_ref_alt_alleles(
    n_variants: int,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Generate reference and alternate alleles.

    Args:
        n_variants: Number of variants
        seed: Random seed

    Returns:
        Tuple of (ref_alleles, alt_alleles)
    """
    np.random.seed(seed)

    refs = []
    alts = []

    for _ in range(n_variants):
        ref = np.random.choice(BASES)
        # Alt must be different from ref
        alt_choices = [b for b in BASES if b != ref]
        alt = np.random.choice(alt_choices)
        refs.append(ref)
        alts.append(alt)

    return refs, alts


def write_vcf(
    genotypes: np.ndarray,
    variant_info: pd.DataFrame,
    sample_ids: List[str],
    output_path: Path,
    compress: bool = True,
) -> None:
    """
    Write genotypes to VCF format.

    Args:
        genotypes: Genotype matrix (samples × variants)
        variant_info: DataFrame with CHROM, POS, ID, REF, ALT, AF
        sample_ids: List of sample IDs
        output_path: Output file path
        compress: Whether to gzip the output
    """
    # Prepare VCF content
    lines = []

    # Header
    lines.append("##fileformat=VCFv4.2")
    lines.append(f"##fileDate={datetime.now().strftime('%Y%m%d')}")
    lines.append("##source=PathWAS_synthetic_generator")
    lines.append("##reference=GRCh38")
    lines.append('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">')
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

    # Contig lines
    for chrom, length in CHROMOSOME_LENGTHS.items():
        lines.append(f"##contig=<ID={chrom},length={length}>")

    # Column header
    header_cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    header_cols.extend(sample_ids)
    lines.append("\t".join(header_cols))

    # Data lines
    for var_idx in range(len(variant_info)):
        row = variant_info.iloc[var_idx]

        # Basic columns
        cols = [
            str(row["CHROM"]),
            str(row["POS"]),
            str(row["ID"]),
            str(row["REF"]),
            str(row["ALT"]),
            ".",  # QUAL
            "PASS",  # FILTER
            f"AF={row['AF']:.4f}",  # INFO
            "GT",  # FORMAT
        ]

        # Genotype columns
        for sample_idx in range(len(sample_ids)):
            gt_value = genotypes[sample_idx, var_idx]
            if gt_value == 0:
                gt = "0/0"
            elif gt_value == 1:
                gt = "0/1"
            else:
                gt = "1/1"
            cols.append(gt)

        lines.append("\t".join(cols))

    # Write file
    content = "\n".join(lines) + "\n"

    if compress:
        with gzip.open(output_path, "wt") as f:
            f.write(content)
    else:
        with open(output_path, "w") as f:
            f.write(content)


def write_genotype_matrix(
    genotypes: np.ndarray,
    variant_info: pd.DataFrame,
    sample_ids: List[str],
    output_path: Path,
) -> None:
    """
    Write genotypes to simple CSV matrix format.

    Args:
        genotypes: Genotype matrix (samples × variants)
        variant_info: DataFrame with variant info
        sample_ids: List of sample IDs
        output_path: Output file path
    """
    # Create DataFrame with sample IDs as index and variant IDs as columns
    df = pd.DataFrame(
        genotypes,
        index=sample_ids,
        columns=variant_info["ID"].values,
    )

    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic genotype data for PathWAS testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sample parameters
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples"
    )
    parser.add_argument(
        "--sample-ids", type=str, default=None,
        help="Path to CSV with sample IDs (uses sample_id or first column as index)"
    )

    # Variant parameters
    parser.add_argument(
        "--n-variants-per-chr", type=int, default=1000,
        help="Number of variants per chromosome"
    )
    parser.add_argument(
        "--chromosomes", type=str, default="1,2,22",
        help="Comma-separated list of chromosomes to include"
    )

    # MAF parameters
    parser.add_argument(
        "--maf-min", type=float, default=0.01,
        help="Minimum minor allele frequency"
    )
    parser.add_argument(
        "--maf-max", type=float, default=0.5,
        help="Maximum minor allele frequency"
    )
    parser.add_argument(
        "--maf-distribution", type=str, default="uniform",
        choices=["uniform", "beta"],
        help="MAF distribution (beta is skewed toward rare variants)"
    )

    # LD parameters
    parser.add_argument(
        "--ld-block-size", type=int, default=50,
        help="Size of LD blocks (number of variants)"
    )
    parser.add_argument(
        "--ld-r2", type=float, default=0.8,
        help="Target r² within LD blocks"
    )
    parser.add_argument(
        "--no-ld", action="store_true",
        help="Disable LD structure"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Don't compress VCF output"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Parse chromosomes
    chromosomes = [c.strip() for c in args.chromosomes.split(",")]

    logger.info("=" * 60)
    logger.info("Generating Synthetic Genotype Data")
    logger.info("=" * 60)
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Variants per chromosome: {args.n_variants_per_chr}")
    logger.info(f"Chromosomes: {chromosomes}")
    logger.info(f"MAF range: {args.maf_min} - {args.maf_max}")
    logger.info(f"LD block size: {args.ld_block_size}")
    logger.info(f"Seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample IDs
    if args.sample_ids:
        logger.info(f"Loading sample IDs from {args.sample_ids}...")
        sample_df = pd.read_csv(args.sample_ids, index_col=0)
        sample_ids = list(sample_df.index)
        if len(sample_ids) != args.n_samples:
            logger.warning(f"Sample file has {len(sample_ids)} samples, using that instead of {args.n_samples}")
            args.n_samples = len(sample_ids)
    else:
        sample_ids = [f"sample_{i:03d}" for i in range(args.n_samples)]

    # Generate variants for each chromosome
    all_genotypes = []
    all_variant_info = []

    for chrom_idx, chrom in enumerate(chromosomes):
        logger.info(f"Generating variants for chromosome {chrom}...")

        # Generate MAFs
        mafs = generate_allele_frequencies(
            n_variants=args.n_variants_per_chr,
            maf_min=args.maf_min,
            maf_max=args.maf_max,
            distribution=args.maf_distribution,
            seed=args.seed + chrom_idx * 1000,
        )

        # Generate positions
        positions = generate_variant_positions(
            n_variants=args.n_variants_per_chr,
            chromosome=chrom,
            seed=args.seed + chrom_idx * 1000 + 1,
        )

        # Generate alleles
        refs, alts = generate_ref_alt_alleles(
            n_variants=args.n_variants_per_chr,
            seed=args.seed + chrom_idx * 1000 + 2,
        )

        # Generate genotypes in HWE
        genotypes = np.zeros((args.n_samples, args.n_variants_per_chr), dtype=int)
        for var_idx, maf in enumerate(mafs):
            genotypes[:, var_idx] = generate_genotypes_hwe(
                n_samples=args.n_samples,
                maf=maf,
                seed=args.seed + chrom_idx * 1000 + var_idx + 3,
            )

        # Add LD structure
        if not args.no_ld:
            genotypes = add_ld_structure(
                genotypes=genotypes,
                block_size=args.ld_block_size,
                r2_within_block=args.ld_r2,
                seed=args.seed + chrom_idx * 1000 + 100000,
            )

            # Recalculate MAFs after LD modification
            mafs = genotypes.mean(axis=0) / 2

        # Create variant IDs
        var_ids = [f"chr{chrom}_{pos}" for pos in positions]

        # Create variant info DataFrame
        variant_info = pd.DataFrame({
            "CHROM": chrom,
            "POS": positions,
            "ID": var_ids,
            "REF": refs,
            "ALT": alts,
            "AF": mafs,
        })

        all_genotypes.append(genotypes)
        all_variant_info.append(variant_info)

        logger.info(f"  Generated {len(variant_info)} variants")
        logger.info(f"  Mean MAF: {mafs.mean():.3f}")

    # Combine all chromosomes
    logger.info("Combining chromosomes...")
    genotypes_combined = np.hstack(all_genotypes)
    variant_info_combined = pd.concat(all_variant_info, ignore_index=True)

    n_total_variants = len(variant_info_combined)
    logger.info(f"Total variants: {n_total_variants}")

    # Write VCF
    logger.info("Writing VCF file...")
    vcf_suffix = ".vcf.gz" if not args.no_compress else ".vcf"
    vcf_path = output_dir / f"genotypes{vcf_suffix}"
    write_vcf(
        genotypes=genotypes_combined,
        variant_info=variant_info_combined,
        sample_ids=sample_ids,
        output_path=vcf_path,
        compress=not args.no_compress,
    )
    logger.info(f"  VCF: {vcf_path}")

    # Write genotype matrix
    logger.info("Writing genotype matrix...")
    matrix_path = output_dir / "genotypes.csv"
    write_genotype_matrix(
        genotypes=genotypes_combined,
        variant_info=variant_info_combined,
        sample_ids=sample_ids,
        output_path=matrix_path,
    )
    logger.info(f"  Matrix: {matrix_path}")

    # Write variant info
    logger.info("Writing variant info...")
    var_info_path = output_dir / "variant_info.csv"
    variant_info_combined.to_csv(var_info_path, index=False)
    logger.info(f"  Variant info: {var_info_path}")

    # Summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)
    logger.info(f"Total samples: {args.n_samples}")
    logger.info(f"Total variants: {n_total_variants}")
    logger.info(f"Mean MAF: {variant_info_combined['AF'].mean():.3f}")
    logger.info(f"MAF range: {variant_info_combined['AF'].min():.3f} - {variant_info_combined['AF'].max():.3f}")

    # Calculate approximate size
    vcf_size = vcf_path.stat().st_size / (1024 * 1024)
    matrix_size = matrix_path.stat().st_size / (1024 * 1024)
    logger.info(f"VCF size: {vcf_size:.2f} MB")
    logger.info(f"Matrix size: {matrix_size:.2f} MB")


if __name__ == "__main__":
    main()
