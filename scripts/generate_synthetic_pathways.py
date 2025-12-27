#!/usr/bin/env python
"""
Generate Synthetic Pathway Definitions for PathWAS Testing.

This script generates realistic pathway definitions that map to generated genes:
- Configurable number of pathways
- Variable pathway sizes (15-200 genes)
- Some overlapping genes between pathways
- Both JSON and GMT output formats

Output files:
- pathways.json: JSON format pathway definitions
- pathways.gmt: GMT format (Gene Matrix Transposed)

Usage:
    python scripts/generate_synthetic_pathways.py \\
        --n-pathways 50 \\
        --min-genes 15 \\
        --max-genes 200 \\
        --gene-list data/raw/gene_info.csv \\
        --output-dir data/raw
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Realistic pathway name prefixes (mimicking KEGG, Reactome, GO)
PATHWAY_PREFIXES = [
    # Signaling pathways
    "MAPK_SIGNALING", "PI3K_AKT_SIGNALING", "WNT_SIGNALING", "NOTCH_SIGNALING",
    "HEDGEHOG_SIGNALING", "TGFB_SIGNALING", "NFKB_SIGNALING", "JAK_STAT_SIGNALING",
    "HIPPO_SIGNALING", "CALCIUM_SIGNALING", "CAMP_SIGNALING", "MTOR_SIGNALING",

    # Metabolism
    "GLYCOLYSIS", "GLUCONEOGENESIS", "TCA_CYCLE", "OXIDATIVE_PHOSPHORYLATION",
    "FATTY_ACID_METABOLISM", "LIPID_BIOSYNTHESIS", "AMINO_ACID_METABOLISM",
    "PURINE_METABOLISM", "PYRIMIDINE_METABOLISM", "PENTOSE_PHOSPHATE_PATHWAY",

    # Cell cycle and death
    "CELL_CYCLE_G1_S", "CELL_CYCLE_G2_M", "MITOTIC_SPINDLE", "DNA_REPLICATION",
    "APOPTOSIS", "AUTOPHAGY", "NECROPTOSIS", "FERROPTOSIS", "SENESCENCE",

    # DNA/RNA
    "DNA_REPAIR", "HOMOLOGOUS_RECOMBINATION", "MISMATCH_REPAIR", "BASE_EXCISION_REPAIR",
    "TRANSCRIPTION_REGULATION", "RNA_SPLICING", "RNA_TRANSPORT", "RIBOSOME_BIOGENESIS",

    # Immune response
    "INNATE_IMMUNITY", "ADAPTIVE_IMMUNITY", "T_CELL_ACTIVATION", "B_CELL_ACTIVATION",
    "CYTOKINE_SIGNALING", "INTERFERON_RESPONSE", "COMPLEMENT_SYSTEM", "ANTIGEN_PRESENTATION",
    "INFLAMMATORY_RESPONSE", "CHEMOKINE_SIGNALING",

    # Development
    "EMBRYONIC_DEVELOPMENT", "TISSUE_MORPHOGENESIS", "STEM_CELL_MAINTENANCE",
    "EPITHELIAL_MESENCHYMAL_TRANSITION", "ANGIOGENESIS", "NEUROGENESIS",

    # Cellular processes
    "PROTEIN_FOLDING", "PROTEASOME", "ENDOCYTOSIS", "EXOCYTOSIS", "VESICLE_TRANSPORT",
    "CYTOSKELETON_ORGANIZATION", "CELL_ADHESION", "CELL_MIGRATION", "EXTRACELLULAR_MATRIX",

    # Disease-related
    "P53_PATHWAY", "RB_PATHWAY", "HYPOXIA_RESPONSE", "OXIDATIVE_STRESS",
    "UNFOLDED_PROTEIN_RESPONSE", "DRUG_METABOLISM",
]

# Pathway descriptions
PATHWAY_DESCRIPTIONS = {
    "SIGNALING": "Signal transduction pathway involved in cellular communication",
    "METABOLISM": "Metabolic pathway regulating energy and biosynthesis",
    "CELL_CYCLE": "Pathway controlling cell division and proliferation",
    "APOPTOSIS": "Programmed cell death pathway",
    "AUTOPHAGY": "Cellular degradation and recycling pathway",
    "DNA_REPAIR": "Pathway maintaining genomic integrity",
    "IMMUNE": "Immune response and defense pathway",
    "DEVELOPMENT": "Developmental and differentiation pathway",
    "TRANSPORT": "Intracellular transport pathway",
    "ADHESION": "Cell-cell and cell-matrix adhesion pathway",
}


def generate_pathway_name() -> str:
    """Generate a random pathway name."""
    prefix = np.random.choice(PATHWAY_PREFIXES)
    suffix = np.random.choice(["PATHWAY", "PROCESS", "REGULATION", "RESPONSE", "ACTIVATION", ""])

    if suffix:
        return f"{prefix}_{suffix}"
    return prefix


def generate_pathway_description(name: str) -> str:
    """Generate a description for a pathway based on its name."""
    for key, desc in PATHWAY_DESCRIPTIONS.items():
        if key in name.upper():
            return desc
    return "Biological pathway"


def generate_pathway_sizes(
    n_pathways: int,
    min_genes: int = 15,
    max_genes: int = 200,
    seed: int = 42,
) -> List[int]:
    """
    Generate pathway sizes following a realistic distribution.

    Most pathways are medium-sized, with fewer very small or very large ones.

    Args:
        n_pathways: Number of pathways
        min_genes: Minimum genes per pathway
        max_genes: Maximum genes per pathway
        seed: Random seed

    Returns:
        List of pathway sizes
    """
    np.random.seed(seed)

    # Log-normal distribution centered around medium size
    mean_size = (min_genes + max_genes) / 2
    sigma = 0.5

    sizes = np.random.lognormal(mean=np.log(mean_size), sigma=sigma, size=n_pathways)
    sizes = np.clip(sizes, min_genes, max_genes).astype(int)

    return sizes.tolist()


def generate_pathways(
    available_genes: List[str],
    n_pathways: int = 50,
    min_genes: int = 15,
    max_genes: int = 200,
    overlap_fraction: float = 0.2,
    de_genes: Optional[List[str]] = None,
    de_enrichment: float = 0.3,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate synthetic pathway definitions.

    Args:
        available_genes: List of available gene symbols
        n_pathways: Number of pathways to generate
        min_genes: Minimum genes per pathway
        max_genes: Maximum genes per pathway
        overlap_fraction: Fraction of genes that can overlap between pathways
        de_genes: List of differentially expressed genes to enrich in some pathways
        de_enrichment: Fraction of DE genes to include in enriched pathways
        seed: Random seed

    Returns:
        Dict mapping pathway names to gene lists
    """
    np.random.seed(seed)

    n_genes = len(available_genes)
    gene_set = set(available_genes)

    # Generate pathway sizes
    sizes = generate_pathway_sizes(n_pathways, min_genes, max_genes, seed)

    # Track gene usage for controlled overlap
    gene_usage_count = {gene: 0 for gene in available_genes}
    max_pathway_per_gene = max(3, int(n_pathways * overlap_fraction))

    pathways = {}
    used_names: Set[str] = set()

    for i in range(n_pathways):
        # Generate unique pathway name
        attempts = 0
        while attempts < 100:
            name = generate_pathway_name()
            # Add number suffix if needed for uniqueness
            if name not in used_names:
                break
            name = f"{name}_{i+1}"
            if name not in used_names:
                break
            attempts += 1

        used_names.add(name)

        # Get target size
        target_size = sizes[i]

        # Get available genes (not over-used)
        available_for_pathway = [
            g for g in available_genes
            if gene_usage_count[g] < max_pathway_per_gene
        ]

        # If we have DE genes and this pathway should be enriched
        pathway_genes = []
        if de_genes and np.random.random() < 0.3:  # 30% of pathways enriched
            n_de_to_include = min(
                int(target_size * de_enrichment),
                len(de_genes),
            )
            de_available = [g for g in de_genes if g in available_for_pathway]
            if de_available:
                de_sample = np.random.choice(
                    de_available,
                    size=min(n_de_to_include, len(de_available)),
                    replace=False
                ).tolist()
                pathway_genes.extend(de_sample)

        # Fill remaining with random genes
        remaining_needed = target_size - len(pathway_genes)
        non_pathway_available = [
            g for g in available_for_pathway
            if g not in pathway_genes
        ]

        if len(non_pathway_available) >= remaining_needed:
            additional = np.random.choice(
                non_pathway_available,
                size=remaining_needed,
                replace=False
            ).tolist()
        else:
            additional = non_pathway_available

        pathway_genes.extend(additional)

        # Update usage counts
        for gene in pathway_genes:
            gene_usage_count[gene] += 1

        pathways[name] = pathway_genes

    return pathways


def write_gmt(pathways: Dict[str, List[str]], output_path: Path) -> None:
    """
    Write pathways to GMT (Gene Matrix Transposed) format.

    GMT format: pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...

    Args:
        pathways: Dict mapping pathway names to gene lists
        output_path: Output file path
    """
    lines = []

    for name, genes in pathways.items():
        description = generate_pathway_description(name)
        line = "\t".join([name, description] + genes)
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_json(pathways: Dict[str, List[str]], output_path: Path) -> None:
    """
    Write pathways to JSON format.

    Args:
        pathways: Dict mapping pathway names to gene lists
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        json.dump(pathways, f, indent=2)


def compute_pathway_stats(pathways: Dict[str, List[str]]) -> Dict:
    """
    Compute summary statistics for pathways.

    Args:
        pathways: Dict mapping pathway names to gene lists

    Returns:
        Dict with statistics
    """
    sizes = [len(genes) for genes in pathways.values()]
    all_genes = set()
    for genes in pathways.values():
        all_genes.update(genes)

    # Compute overlap statistics
    gene_pathway_count = {}
    for genes in pathways.values():
        for gene in genes:
            gene_pathway_count[gene] = gene_pathway_count.get(gene, 0) + 1

    overlapping_genes = sum(1 for count in gene_pathway_count.values() if count > 1)

    return {
        "n_pathways": len(pathways),
        "total_unique_genes": len(all_genes),
        "mean_pathway_size": np.mean(sizes),
        "median_pathway_size": np.median(sizes),
        "min_pathway_size": min(sizes),
        "max_pathway_size": max(sizes),
        "genes_in_multiple_pathways": overlapping_genes,
        "fraction_overlapping": overlapping_genes / len(all_genes) if all_genes else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic pathway definitions for PathWAS testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pathway parameters
    parser.add_argument(
        "--n-pathways", type=int, default=50,
        help="Number of pathways to generate"
    )
    parser.add_argument(
        "--min-genes", type=int, default=15,
        help="Minimum genes per pathway"
    )
    parser.add_argument(
        "--max-genes", type=int, default=200,
        help="Maximum genes per pathway"
    )
    parser.add_argument(
        "--overlap-fraction", type=float, default=0.2,
        help="Fraction of genes that can appear in multiple pathways"
    )

    # Gene list
    parser.add_argument(
        "--gene-list", type=str, default=None,
        help="Path to gene info CSV (uses 'symbol' column or first column)"
    )
    parser.add_argument(
        "--n-genes", type=int, default=5000,
        help="Number of genes if no gene list provided"
    )

    # DE gene enrichment
    parser.add_argument(
        "--de-genes", type=str, default=None,
        help="Path to DE genes CSV for pathway enrichment"
    )
    parser.add_argument(
        "--de-enrichment", type=float, default=0.3,
        help="Fraction of DE genes to include in enriched pathways"
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

    logger.info("=" * 60)
    logger.info("Generating Synthetic Pathway Definitions")
    logger.info("=" * 60)
    logger.info(f"Pathways: {args.n_pathways}")
    logger.info(f"Size range: {args.min_genes} - {args.max_genes} genes")
    logger.info(f"Overlap fraction: {args.overlap_fraction}")
    logger.info(f"Seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate gene list
    if args.gene_list:
        logger.info(f"Loading genes from {args.gene_list}...")
        gene_df = pd.read_csv(args.gene_list)

        # Try to find the gene symbol column
        if "symbol" in gene_df.columns:
            genes = gene_df["symbol"].tolist()
        elif "gene" in gene_df.columns:
            genes = gene_df["gene"].tolist()
        else:
            genes = gene_df.iloc[:, 0].tolist()

        logger.info(f"  Loaded {len(genes)} genes")
    else:
        logger.info(f"Generating {args.n_genes} gene names...")
        genes = [f"GENE{i:04d}" for i in range(args.n_genes)]

    # Load DE genes if provided
    de_genes = None
    if args.de_genes:
        logger.info(f"Loading DE genes from {args.de_genes}...")
        de_df = pd.read_csv(args.de_genes)
        if "gene" in de_df.columns:
            de_genes = de_df["gene"].tolist()
        else:
            de_genes = de_df.iloc[:, 0].tolist()
        logger.info(f"  Loaded {len(de_genes)} DE genes")

    # Generate pathways
    logger.info("Generating pathways...")
    pathways = generate_pathways(
        available_genes=genes,
        n_pathways=args.n_pathways,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        overlap_fraction=args.overlap_fraction,
        de_genes=de_genes,
        de_enrichment=args.de_enrichment,
        seed=args.seed,
    )

    # Compute statistics
    stats = compute_pathway_stats(pathways)

    # Write outputs
    logger.info("Writing output files...")

    # JSON format
    json_path = output_dir / "pathways.json"
    write_json(pathways, json_path)
    logger.info(f"  JSON: {json_path}")

    # GMT format
    gmt_path = output_dir / "pathways.gmt"
    write_gmt(pathways, gmt_path)
    logger.info(f"  GMT: {gmt_path}")

    # Summary
    summary_path = output_dir / "pathways_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Summary: {summary_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)
    logger.info(f"Pathways: {stats['n_pathways']}")
    logger.info(f"Unique genes covered: {stats['total_unique_genes']}")
    logger.info(f"Mean pathway size: {stats['mean_pathway_size']:.1f}")
    logger.info(f"Size range: {stats['min_pathway_size']} - {stats['max_pathway_size']}")
    logger.info(f"Genes in multiple pathways: {stats['genes_in_multiple_pathways']} ({stats['fraction_overlapping']:.1%})")


if __name__ == "__main__":
    main()
