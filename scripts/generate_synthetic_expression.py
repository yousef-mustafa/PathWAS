#!/usr/bin/env python
"""
Generate Synthetic Gene Expression Data for PathWAS Testing.

This script generates realistic synthetic gene expression data with:
- Negative binomial distribution (mimics RNA-seq counts)
- Configurable differential expression between groups
- Optional batch effects
- Correlation structure within pathways

Output files:
- expression.csv: Expression matrix (samples × genes)
- sample_metadata.csv: Sample information with group labels
- gene_info.csv: Gene metadata (symbol, ensembl_id, length)

Usage:
    python scripts/generate_synthetic_expression.py \\
        --n-samples 200 \\
        --n-genes 5000 \\
        --n-groups 2 \\
        --samples-per-group 100 \\
        --n-de-genes 500 \\
        --de-fold-change 2.0 \\
        --add-batch-effect \\
        --seed 42 \\
        --output-dir data/raw
"""

import argparse
import json
import logging
import sys
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

# Common HGNC gene symbols for realistic gene names
# This is a subset - in practice would use full HGNC database
COMMON_GENE_SYMBOLS = [
    # Housekeeping genes
    "GAPDH", "ACTB", "B2M", "HPRT1", "RPL13A", "SDHA", "TBP", "UBC", "YWHAZ", "RPLP0",
    # Signaling pathways
    "AKT1", "AKT2", "AKT3", "MTOR", "PIK3CA", "PIK3CB", "PIK3CD", "PTEN", "TSC1", "TSC2",
    "MAPK1", "MAPK3", "MAPK8", "MAPK14", "RAF1", "BRAF", "KRAS", "NRAS", "HRAS", "EGFR",
    "ERBB2", "ERBB3", "ERBB4", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "PDGFRA", "PDGFRB", "KIT",
    # Cell cycle
    "CCNA1", "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCND2", "CCND3", "CCNE1", "CCNE2",
    "CDK1", "CDK2", "CDK4", "CDK6", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "RB1", "E2F1",
    # Apoptosis
    "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BID", "BAD", "CASP3", "CASP8", "CASP9",
    "TP53", "MDM2", "MDM4", "PARP1", "XIAP", "BIRC2", "BIRC3", "BIRC5", "DIABLO", "CYCS",
    # DNA repair
    "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "TP53BP1", "RAD51", "MLH1", "MSH2",
    "MSH6", "PMS2", "XRCC1", "XRCC4", "LIG4", "PRKDC", "DCLRE1C", "ERCC1", "XPA", "XPC",
    # Metabolism
    "HK1", "HK2", "PFKM", "PKM", "LDHA", "LDHB", "PDK1", "PDK2", "PDK3", "PDK4",
    "GLUT1", "GLUT4", "G6PD", "TKT", "TALDO1", "PGK1", "ENO1", "ENO2", "ALDOA", "ALDOB",
    # Immune response
    "IL1A", "IL1B", "IL2", "IL4", "IL6", "IL10", "IL12A", "IL12B", "IL17A", "IL23A",
    "IFNG", "IFNA1", "IFNB1", "TNF", "TNFRSF1A", "TNFRSF1B", "CD4", "CD8A", "CD8B", "CD28",
    "CTLA4", "PDCD1", "CD274", "LAG3", "TIM3", "TIGIT", "BTLA", "ICOS", "OX40", "GITR",
    # Transcription factors
    "MYC", "MYCN", "MAX", "FOS", "JUN", "JUNB", "JUND", "NFKB1", "NFKB2", "RELA",
    "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6", "HIF1A", "EPAS1", "VHL",
    # Wnt pathway
    "WNT1", "WNT2", "WNT3", "WNT3A", "WNT5A", "WNT7A", "WNT10B", "FZD1", "FZD2", "FZD7",
    "LRP5", "LRP6", "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3A", "GSK3B", "TCF7", "LEF1",
    # Notch pathway
    "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1", "DLL3", "DLL4",
    "HES1", "HES5", "HEY1", "HEY2", "RBPJ", "MAML1", "MAML2", "MAML3", "NUMB", "NUMBL",
    # Hedgehog pathway
    "SHH", "IHH", "DHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3", "SUFU",
    # TGF-beta pathway
    "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5",
    "SMAD6", "SMAD7", "BMP1", "BMP2", "BMP4", "BMP7", "BMPR1A", "BMPR1B", "BMPR2", "ACVR1",
    # Adhesion molecules
    "CDH1", "CDH2", "CDH3", "CDH5", "CDH11", "ITGA1", "ITGA2", "ITGA3", "ITGA4", "ITGA5",
    "ITGAV", "ITGB1", "ITGB2", "ITGB3", "ITGB4", "ICAM1", "VCAM1", "PECAM1", "SELE", "SELP",
    # Cytoskeleton
    "TUBB", "TUBA1A", "TUBA1B", "TUBA4A", "ACTC1", "ACTG1", "VIM", "DES", "KRT8", "KRT18",
    "KRT19", "GFAP", "NEFL", "NEFM", "NEFH", "MYH1", "MYH2", "MYH7", "TPM1", "TPM2",
    # Extracellular matrix
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A1", "COL5A1", "COL6A1", "FN1", "LAMA1", "LAMB1",
    "LAMC1", "VTN", "TNC", "THBS1", "THBS2", "SPARC", "DCN", "BGN", "LUM", "ASPN",
    # Proteases and inhibitors
    "MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP14", "ADAM10", "ADAM17", "TIMP1", "TIMP2",
    "TIMP3", "SERPINE1", "SERPINB2", "PLAU", "PLAUR", "CTSA", "CTSB", "CTSD", "CTSL", "CTSS",
    # Growth factors
    "EGF", "VEGFA", "VEGFB", "VEGFC", "FGF1", "FGF2", "FGF7", "FGF10", "PDGFA", "PDGFB",
    "HGF", "IGF1", "IGF2", "NGF", "BDNF", "GDNF", "AREG", "EREG", "BTC", "TGFA",
    # Receptors
    "ESR1", "ESR2", "AR", "PGR", "GR", "INSR", "IGF1R", "IGF2R", "LEPR", "ADIPOR1",
    "ADIPOR2", "GHSR", "MC4R", "NPY1R", "NPY2R", "ADRB1", "ADRB2", "ADRB3", "ADRA1A", "ADRA2A",
    # Kinases
    "SRC", "ABL1", "ABL2", "LCK", "FYN", "YES1", "BTK", "SYK", "JAK1", "JAK2",
    "JAK3", "TYK2", "FLT3", "CSF1R", "MET", "RON", "RET", "ALK", "ROS1", "NTRK1",
    # Phosphatases
    "PTPN1", "PTPN2", "PTPN6", "PTPN11", "PTPN22", "PTPRC", "PTPRF", "PTPRJ", "PTPRK", "CDC25A",
    "CDC25B", "CDC25C", "PPP1CA", "PPP1CB", "PPP2CA", "PPP2CB", "PPP3CA", "PHLPP1", "PHLPP2", "DUSP1",
    # Chromatin modifiers
    "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "SIRT1", "SIRT2", "SIRT3", "KAT2A", "KAT2B",
    "EP300", "CREBBP", "KMT2A", "KMT2D", "EZH2", "SUZ12", "EED", "DNMT1", "DNMT3A", "DNMT3B",
    # RNA processing
    "SRSF1", "SRSF2", "SRSF3", "HNRNPA1", "HNRNPC", "HNRNPK", "RBFOX2", "PTBP1", "FUS", "TARDBP",
    "EIF4A1", "EIF4E", "EIF4G1", "EIF2S1", "AGO1", "AGO2", "DICER1", "DROSHA", "XRN1", "XRN2",
    # Ubiquitin system
    "UBA1", "UBE2D1", "UBE2N", "UBE2C", "FBXW7", "SKP2", "BTRC", "CUL1", "CUL3", "RNF4",
    "TRIM21", "TRIM25", "TRIM32", "USP7", "USP14", "UCHL1", "PSMA1", "PSMB1", "PSMC1", "PSMD1",
    # Autophagy
    "BECN1", "ATG5", "ATG7", "ATG12", "ATG16L1", "LC3B", "SQSTM1", "LAMP1", "LAMP2", "CTSD",
    "ULK1", "ULK2", "VPS34", "ATG14", "RUBCN", "AMBRA1", "UVRAG", "ATG9A", "WIPI1", "WIPI2",
    # Metabolism enzymes
    "ACACA", "FASN", "SCD", "HMGCR", "HMGCS1", "MVK", "MVD", "FDFT1", "SQLE", "DHCR7",
    "CYP19A1", "CYP1A1", "CYP1B1", "CYP2D6", "CYP3A4", "GSTM1", "GSTP1", "NQO1", "GPX1", "SOD1",
    # Transporters
    "SLC2A1", "SLC2A4", "SLC7A5", "SLC7A11", "SLC16A1", "ABCB1", "ABCC1", "ABCG2", "ATP1A1", "ATP1B1",
    "SLC1A5", "SLC3A2", "SLC6A14", "CFTR", "SLC12A2", "AQP1", "AQP4", "SLC4A1", "SLC9A1", "SLC25A1",
]


def generate_gene_list(n_genes: int, seed: int = 42) -> List[str]:
    """
    Generate a list of gene symbols.

    Uses real HGNC symbols where available, generates synthetic ones for the rest.

    Args:
        n_genes: Number of genes to generate
        seed: Random seed for reproducibility

    Returns:
        List of gene symbols
    """
    np.random.seed(seed)

    genes = []

    # Use real gene symbols first
    n_real = min(n_genes, len(COMMON_GENE_SYMBOLS))
    genes.extend(COMMON_GENE_SYMBOLS[:n_real])

    # Generate synthetic gene names for remaining
    n_synthetic = n_genes - n_real
    if n_synthetic > 0:
        # Generate names like GENE0001, GENE0002, etc.
        synthetic_genes = [f"GENE{i:04d}" for i in range(n_synthetic)]
        genes.extend(synthetic_genes)

    return genes


def generate_gene_info(genes: List[str], seed: int = 42) -> pd.DataFrame:
    """
    Generate gene metadata.

    Args:
        genes: List of gene symbols
        seed: Random seed

    Returns:
        DataFrame with gene info (symbol, ensembl_id, length, chromosome)
    """
    np.random.seed(seed)

    n_genes = len(genes)

    # Generate Ensembl-like IDs
    ensembl_ids = [f"ENSG{i:011d}" for i in range(n_genes)]

    # Generate realistic gene lengths (bp) - log-normal distribution
    # Most genes are 1-10kb, some are very long
    lengths = np.random.lognormal(mean=8.5, sigma=1.0, size=n_genes).astype(int)
    lengths = np.clip(lengths, 500, 2000000)  # Clip to realistic range

    # Assign to chromosomes
    # Probabilities: 22 autosomes equal, X twice as likely, Y half as likely
    chr_probs = np.array([1.0] * 22 + [2.0, 0.5])
    chr_probs = chr_probs / chr_probs.sum()  # Normalize to sum to 1
    chromosomes = np.random.choice(
        [str(i) for i in range(1, 23)] + ["X", "Y"],
        size=n_genes,
        p=chr_probs
    )

    # Generate GC content
    gc_content = np.random.beta(5, 5, size=n_genes) * 0.4 + 0.3  # Range 0.3-0.7

    gene_info = pd.DataFrame({
        "symbol": genes,
        "ensembl_id": ensembl_ids,
        "length": lengths,
        "chromosome": chromosomes,
        "gc_content": gc_content.round(3),
    })

    return gene_info


def generate_base_expression(
    n_samples: int,
    n_genes: int,
    mean_count: float = 100.0,
    dispersion: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate base expression using negative binomial distribution.

    The negative binomial distribution models RNA-seq count data well,
    capturing the overdispersion typical of sequencing experiments.

    Args:
        n_samples: Number of samples
        n_genes: Number of genes
        mean_count: Average count per gene (before normalization)
        dispersion: Overdispersion parameter (higher = more variance)
        seed: Random seed

    Returns:
        Expression matrix (samples × genes)
    """
    np.random.seed(seed)

    # Gene-specific mean expression (some genes highly expressed, most lower)
    gene_means = np.random.gamma(shape=2, scale=mean_count / 2, size=n_genes)

    # Sample-specific library size factors (sequencing depth variation)
    size_factors = np.random.lognormal(mean=0, sigma=0.3, size=n_samples)

    # Generate counts using negative binomial
    expression = np.zeros((n_samples, n_genes))

    for i in range(n_samples):
        for j in range(n_genes):
            # Mean for this sample-gene combination
            mu = gene_means[j] * size_factors[i]

            # Dispersion determines variance: Var = mu + mu^2 * dispersion
            # For negative binomial: p = 1/(1 + mu*dispersion), n = 1/dispersion
            if dispersion > 0:
                p = 1 / (1 + mu * dispersion)
                n = 1 / dispersion
                expression[i, j] = np.random.negative_binomial(n=max(n, 0.01), p=min(p, 0.99))
            else:
                expression[i, j] = np.random.poisson(mu)

    return expression


def add_differential_expression(
    expression: np.ndarray,
    group_indices: Dict[str, List[int]],
    de_gene_indices: List[int],
    fold_changes: Dict[str, float],
    seed: int = 42,
) -> np.ndarray:
    """
    Add differential expression for specified genes.

    Args:
        expression: Base expression matrix (samples × genes)
        group_indices: Dict mapping group name to sample indices
        de_gene_indices: Indices of differentially expressed genes
        fold_changes: Dict mapping group name to fold change (relative to first group)
        seed: Random seed

    Returns:
        Modified expression matrix with differential expression
    """
    np.random.seed(seed)

    expr_modified = expression.copy()

    # Get reference group (first group, fold change = 1)
    groups = list(group_indices.keys())
    reference_group = groups[0]

    for group, indices in group_indices.items():
        if group == reference_group:
            continue

        fc = fold_changes.get(group, 1.0)

        for gene_idx in de_gene_indices:
            # Apply fold change with some noise
            noise = np.random.normal(0, 0.1, size=len(indices))
            multiplier = fc * np.exp(noise)
            expr_modified[indices, gene_idx] *= multiplier

    return expr_modified


def add_batch_effects(
    expression: np.ndarray,
    batch_assignments: np.ndarray,
    effect_size: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Add batch effects to expression data.

    Args:
        expression: Expression matrix (samples × genes)
        batch_assignments: Array of batch labels for each sample
        effect_size: Magnitude of batch effect
        seed: Random seed

    Returns:
        Expression matrix with batch effects
    """
    np.random.seed(seed)

    expr_modified = expression.copy()
    n_genes = expression.shape[1]

    unique_batches = np.unique(batch_assignments)

    # Generate batch-specific effects for each gene
    for batch in unique_batches:
        batch_mask = batch_assignments == batch

        # Each batch has a gene-specific multiplicative effect
        batch_effects = np.random.lognormal(mean=0, sigma=effect_size, size=n_genes)

        expr_modified[batch_mask, :] *= batch_effects

    return expr_modified


def add_pathway_correlation(
    expression: np.ndarray,
    pathway_gene_indices: List[List[int]],
    correlation_strength: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Add correlation structure within pathway genes.

    Args:
        expression: Expression matrix (samples × genes)
        pathway_gene_indices: List of lists, each containing gene indices in a pathway
        correlation_strength: How strongly to correlate genes (0-1)
        seed: Random seed

    Returns:
        Expression matrix with pathway correlation
    """
    np.random.seed(seed)

    expr_modified = expression.copy()
    n_samples = expression.shape[0]

    for gene_indices in pathway_gene_indices:
        if len(gene_indices) < 2:
            continue

        # Get expression for pathway genes
        pathway_expr = expr_modified[:, gene_indices]

        # Generate a shared latent factor
        latent_factor = np.random.randn(n_samples)

        # Add latent factor to each gene (scaled by gene's expression level)
        for i, gene_idx in enumerate(gene_indices):
            gene_std = pathway_expr[:, i].std()
            if gene_std > 0:
                contribution = latent_factor * gene_std * correlation_strength
                expr_modified[:, gene_idx] += contribution

    # Ensure non-negative
    expr_modified = np.maximum(expr_modified, 0)

    return expr_modified


def generate_sample_metadata(
    n_samples: int,
    n_groups: int,
    samples_per_group: Optional[List[int]] = None,
    group_names: Optional[List[str]] = None,
    add_batch: bool = False,
    n_batches: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate sample metadata with group assignments.

    Args:
        n_samples: Total number of samples
        n_groups: Number of groups
        samples_per_group: List of samples per group (if None, equal split)
        group_names: Names for groups (if None, uses 'case', 'control', etc.)
        add_batch: Whether to add batch information
        n_batches: Number of batches
        seed: Random seed

    Returns:
        DataFrame with sample metadata
    """
    np.random.seed(seed)

    # Generate group names
    if group_names is None:
        if n_groups == 2:
            group_names = ["case", "control"]
        else:
            group_names = [f"group_{i+1}" for i in range(n_groups)]

    # Calculate samples per group
    if samples_per_group is None:
        base = n_samples // n_groups
        remainder = n_samples % n_groups
        samples_per_group = [base + (1 if i < remainder else 0) for i in range(n_groups)]

    # Generate group assignments
    groups = []
    for i, (name, n) in enumerate(zip(group_names, samples_per_group)):
        groups.extend([name] * n)

    # Generate sample IDs
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]

    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "group": groups,
    })

    # Add batch information
    if add_batch:
        # Assign batches (try to balance across groups)
        batches = np.random.choice([f"batch_{i+1}" for i in range(n_batches)], size=n_samples)
        metadata["batch"] = batches

    # Add simulated covariates
    metadata["age"] = np.random.normal(50, 15, size=n_samples).clip(18, 90).astype(int)
    metadata["sex"] = np.random.choice(["M", "F"], size=n_samples)

    metadata = metadata.set_index("sample_id")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic gene expression data for PathWAS testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sample parameters
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Total number of samples"
    )
    parser.add_argument(
        "--n-genes", type=int, default=5000,
        help="Number of genes"
    )
    parser.add_argument(
        "--n-groups", type=int, default=2,
        help="Number of groups (e.g., case/control)"
    )
    parser.add_argument(
        "--samples-per-group", type=int, nargs="+", default=None,
        help="Samples per group (if not specified, equal split)"
    )
    parser.add_argument(
        "--group-names", type=str, nargs="+", default=None,
        help="Names for groups"
    )

    # Expression parameters
    parser.add_argument(
        "--mean-count", type=float, default=100.0,
        help="Mean count per gene"
    )
    parser.add_argument(
        "--dispersion", type=float, default=0.5,
        help="Overdispersion parameter"
    )

    # Differential expression
    parser.add_argument(
        "--n-de-genes", type=int, default=500,
        help="Number of differentially expressed genes"
    )
    parser.add_argument(
        "--de-fold-change", type=float, default=2.0,
        help="Fold change for DE genes"
    )

    # Batch effects
    parser.add_argument(
        "--add-batch-effect", action="store_true",
        help="Add batch effects"
    )
    parser.add_argument(
        "--n-batches", type=int, default=2,
        help="Number of batches"
    )
    parser.add_argument(
        "--batch-effect-size", type=float, default=0.5,
        help="Magnitude of batch effect"
    )

    # Pathway correlation
    parser.add_argument(
        "--n-correlated-pathways", type=int, default=10,
        help="Number of pathways with correlated genes"
    )
    parser.add_argument(
        "--pathway-size", type=int, default=30,
        help="Average genes per correlated pathway"
    )
    parser.add_argument(
        "--correlation-strength", type=float, default=0.5,
        help="Strength of within-pathway correlation"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.samples_per_group is not None:
        if len(args.samples_per_group) != args.n_groups:
            parser.error(f"--samples-per-group must have {args.n_groups} values")
        if sum(args.samples_per_group) != args.n_samples:
            parser.error(f"Sum of --samples-per-group must equal --n-samples ({args.n_samples})")

    logger.info("=" * 60)
    logger.info("Generating Synthetic Expression Data")
    logger.info("=" * 60)
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Genes: {args.n_genes}")
    logger.info(f"Groups: {args.n_groups}")
    logger.info(f"DE genes: {args.n_de_genes}")
    logger.info(f"Seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate gene list and info
    logger.info("Generating gene information...")
    genes = generate_gene_list(args.n_genes, seed=args.seed)
    gene_info = generate_gene_info(genes, seed=args.seed)

    # Generate sample metadata
    logger.info("Generating sample metadata...")
    metadata = generate_sample_metadata(
        n_samples=args.n_samples,
        n_groups=args.n_groups,
        samples_per_group=args.samples_per_group,
        group_names=args.group_names,
        add_batch=args.add_batch_effect,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    # Generate base expression
    logger.info("Generating base expression...")
    expression = generate_base_expression(
        n_samples=args.n_samples,
        n_genes=args.n_genes,
        mean_count=args.mean_count,
        dispersion=args.dispersion,
        seed=args.seed,
    )

    # Add differential expression
    if args.n_de_genes > 0:
        logger.info(f"Adding differential expression ({args.n_de_genes} genes, FC={args.de_fold_change})...")

        # Select DE genes
        np.random.seed(args.seed + 1)
        de_gene_indices = np.random.choice(args.n_genes, size=args.n_de_genes, replace=False)

        # Get group indices
        group_indices = {}
        for group in metadata["group"].unique():
            group_indices[group] = list(metadata[metadata["group"] == group].index)
            group_indices[group] = [list(metadata.index).index(s) for s in group_indices[group]]

        # Set fold changes
        groups = list(group_indices.keys())
        fold_changes = {groups[0]: 1.0}  # Reference group
        for i, group in enumerate(groups[1:], 1):
            # Alternate up/down regulation
            fc = args.de_fold_change if i % 2 == 1 else 1.0 / args.de_fold_change
            fold_changes[group] = fc

        expression = add_differential_expression(
            expression=expression,
            group_indices=group_indices,
            de_gene_indices=de_gene_indices.tolist(),
            fold_changes=fold_changes,
            seed=args.seed + 2,
        )

        # Save DE gene info
        de_genes = [genes[i] for i in de_gene_indices]
        de_info = pd.DataFrame({
            "gene": de_genes,
            "de_gene_index": de_gene_indices,
        })
        de_info.to_csv(output_dir / "de_genes.csv", index=False)
        logger.info(f"  Saved DE gene list to {output_dir / 'de_genes.csv'}")

    # Add batch effects
    if args.add_batch_effect:
        logger.info(f"Adding batch effects ({args.n_batches} batches)...")
        batch_assignments = metadata["batch"].values
        expression = add_batch_effects(
            expression=expression,
            batch_assignments=batch_assignments,
            effect_size=args.batch_effect_size,
            seed=args.seed + 3,
        )

    # Add pathway correlation
    if args.n_correlated_pathways > 0:
        logger.info(f"Adding pathway correlation ({args.n_correlated_pathways} pathways)...")

        np.random.seed(args.seed + 4)
        pathway_gene_indices = []
        for _ in range(args.n_correlated_pathways):
            # Random pathway size
            size = max(5, int(np.random.normal(args.pathway_size, args.pathway_size / 4)))
            indices = np.random.choice(args.n_genes, size=min(size, args.n_genes // 2), replace=False)
            pathway_gene_indices.append(indices.tolist())

        expression = add_pathway_correlation(
            expression=expression,
            pathway_gene_indices=pathway_gene_indices,
            correlation_strength=args.correlation_strength,
            seed=args.seed + 5,
        )

    # Create expression DataFrame
    expr_df = pd.DataFrame(
        expression,
        index=metadata.index,
        columns=genes,
    )

    # Ensure integer counts (for realistic RNA-seq data)
    expr_df = expr_df.round().astype(int)
    expr_df = expr_df.clip(lower=0)

    # Save files
    logger.info("Saving output files...")

    # Expression matrix
    expr_path = output_dir / "expression.csv"
    expr_df.to_csv(expr_path)
    logger.info(f"  Expression matrix: {expr_path} ({expr_df.shape[0]} samples × {expr_df.shape[1]} genes)")

    # Sample metadata
    meta_path = output_dir / "sample_metadata.csv"
    metadata.to_csv(meta_path)
    logger.info(f"  Sample metadata: {meta_path}")

    # Gene info
    gene_path = output_dir / "gene_info.csv"
    gene_info.to_csv(gene_path, index=False)
    logger.info(f"  Gene info: {gene_path}")

    # Summary statistics
    # Convert numpy types to Python types for JSON serialization
    group_sizes = {str(k): int(v) for k, v in metadata["group"].value_counts().items()}
    summary = {
        "n_samples": int(args.n_samples),
        "n_genes": int(args.n_genes),
        "n_groups": int(args.n_groups),
        "group_sizes": group_sizes,
        "n_de_genes": int(args.n_de_genes),
        "de_fold_change": float(args.de_fold_change),
        "has_batch_effect": bool(args.add_batch_effect),
        "mean_expression": float(expr_df.values.mean()),
        "median_expression": float(np.median(expr_df.values)),
        "seed": int(args.seed),
    }

    summary_path = output_dir / "expression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary: {summary_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)

    # Print summary
    logger.info(f"Mean expression: {summary['mean_expression']:.2f}")
    logger.info(f"Median expression: {summary['median_expression']:.2f}")
    for group, count in summary["group_sizes"].items():
        logger.info(f"  {group}: {count} samples")


if __name__ == "__main__":
    main()
