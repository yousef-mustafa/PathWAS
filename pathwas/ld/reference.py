## ------------------------------------------------------------------------------------------- ##
## LD Reference Panel Support                                                                 ##
## ------------------------------------------------------------------------------------------- ##
## @script: reference.py                                                                      ##
##                                                                                            ##
## @description: Utilities for loading and using external LD reference panels (e.g., 1KG).   ##
##               Defines a standardized directory layout and provides loaders for            ##
##               SNP manifests, block definitions, and per-block LD matrices.                ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""LD reference panel loading and management."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


# Expected directory layout for LD resources:
# <root_path>/<ancestry>/
#   snp_manifest.tsv        - SNP metadata (snp_id, chrom, pos, effect_allele, other_allele, maf)
#   block_definitions.bed   - BED-like block file (chrom, start, end, block_id)
#   blocks/
#     block_<id>.npz        - Per-block LD matrix and SNP list


@dataclass
class SNPInfo:
    """SNP metadata from manifest."""

    snp_id: str
    chrom: str
    pos: int
    effect_allele: str
    other_allele: str
    maf: float


@dataclass
class LDBlock:
    """LD block information."""

    block_id: str
    chrom: str
    start: int
    end: int
    snp_ids: List[str] = field(default_factory=list)


@dataclass
class LDReference:
    """Container for LD reference panel data.

    Attributes
    ----------
    ancestry : str
        Ancestry label (e.g., "EUR", "AFR").
    snp_manifest : pd.DataFrame
        SNP metadata with columns: snp_id, chrom, pos, effect_allele, other_allele, maf.
    block_definitions : pd.DataFrame
        Block definitions with columns: chrom, start, end, block_id.
    snp_to_block : dict
        Mapping from snp_id to block_id.
    root_path : Path
        Root path to LD resources.
    """

    ancestry: str
    snp_manifest: pd.DataFrame
    block_definitions: pd.DataFrame
    snp_to_block: Dict[str, str]
    root_path: Path

    @property
    def snp_ids(self) -> List[str]:
        """List of all SNP IDs in the reference."""
        return self.snp_manifest["snp_id"].tolist()

    @property
    def n_snps(self) -> int:
        """Number of SNPs in the reference."""
        return len(self.snp_manifest)

    @property
    def n_blocks(self) -> int:
        """Number of LD blocks."""
        return len(self.block_definitions)

    def get_maf(self, snp_id: str) -> Optional[float]:
        """Get MAF for a SNP."""
        mask = self.snp_manifest["snp_id"] == snp_id
        if mask.any():
            return float(self.snp_manifest.loc[mask, "maf"].values[0])
        return None

    def get_block_id(self, snp_id: str) -> Optional[str]:
        """Get block ID for a SNP."""
        return self.snp_to_block.get(snp_id)


def load_snp_manifest(manifest_path: Union[str, Path]) -> pd.DataFrame:
    """Load SNP manifest file.

    Expected columns: snp_id, chrom, pos, effect_allele, other_allele, maf

    Parameters
    ----------
    manifest_path : str or Path
        Path to snp_manifest.tsv file.

    Returns
    -------
    pd.DataFrame
        SNP manifest DataFrame.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"SNP manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, sep="\t")
    required_cols = ["snp_id", "chrom", "pos", "effect_allele", "other_allele", "maf"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"SNP manifest missing columns: {missing}")

    # Ensure types
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    df["maf"] = df["maf"].astype(float)

    logging.info("Loaded SNP manifest with %d SNPs from %s", len(df), manifest_path)
    return df


def load_block_definitions(block_path: Union[str, Path]) -> pd.DataFrame:
    """Load block definitions file.

    Expected format: BED-like with columns: chrom, start, end, block_id

    Parameters
    ----------
    block_path : str or Path
        Path to block_definitions.bed file.

    Returns
    -------
    pd.DataFrame
        Block definitions DataFrame.
    """
    block_path = Path(block_path)
    if not block_path.exists():
        raise FileNotFoundError(f"Block definitions not found: {block_path}")

    df = pd.read_csv(block_path, sep="\t", header=None)
    if df.shape[1] < 4:
        raise ValueError("Block definitions must have at least 4 columns: chrom, start, end, block_id")

    df.columns = ["chrom", "start", "end", "block_id"] + list(df.columns[4:])
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["block_id"] = df["block_id"].astype(str)

    logging.info("Loaded %d LD blocks from %s", len(df), block_path)
    return df


def assign_snps_to_blocks(
    snp_manifest: pd.DataFrame,
    block_definitions: pd.DataFrame,
) -> Dict[str, str]:
    """Assign SNPs to LD blocks based on genomic position.

    Parameters
    ----------
    snp_manifest : pd.DataFrame
        SNP manifest with chrom, pos columns.
    block_definitions : pd.DataFrame
        Block definitions with chrom, start, end, block_id columns.

    Returns
    -------
    dict
        Mapping from snp_id to block_id.
    """
    snp_to_block = {}

    for _, block_row in block_definitions.iterrows():
        block_chrom = str(block_row["chrom"])
        block_start = int(block_row["start"])
        block_end = int(block_row["end"])
        block_id = str(block_row["block_id"])

        # Find SNPs in this block
        mask = (
            (snp_manifest["chrom"] == block_chrom) &
            (snp_manifest["pos"] >= block_start) &
            (snp_manifest["pos"] < block_end)
        )

        for snp_id in snp_manifest.loc[mask, "snp_id"]:
            snp_to_block[snp_id] = block_id

    logging.info("Assigned %d SNPs to blocks", len(snp_to_block))
    return snp_to_block


def load_ld_reference(
    root_path: Union[str, Path],
    ancestry: str,
) -> LDReference:
    """Load LD reference panel for a given ancestry.

    Expected directory structure:
        <root_path>/<ancestry>/
            snp_manifest.tsv
            block_definitions.bed
            blocks/
                block_<id>.npz

    Parameters
    ----------
    root_path : str or Path
        Root path to LD resources.
    ancestry : str
        Ancestry label (e.g., "EUR", "AFR").

    Returns
    -------
    LDReference
        Loaded LD reference panel.
    """
    root_path = Path(root_path)
    ancestry_path = root_path / ancestry

    if not ancestry_path.exists():
        raise FileNotFoundError(f"Ancestry directory not found: {ancestry_path}")

    # Load manifest and blocks
    manifest_path = ancestry_path / "snp_manifest.tsv"
    block_path = ancestry_path / "block_definitions.bed"

    snp_manifest = load_snp_manifest(manifest_path)
    block_definitions = load_block_definitions(block_path)

    # Assign SNPs to blocks
    snp_to_block = assign_snps_to_blocks(snp_manifest, block_definitions)

    return LDReference(
        ancestry=ancestry,
        snp_manifest=snp_manifest,
        block_definitions=block_definitions,
        snp_to_block=snp_to_block,
        root_path=ancestry_path,
    )


def load_block_ld_matrix(
    ld_ref: LDReference,
    block_id: str,
) -> Tuple[np.ndarray, List[str]]:
    """Load LD matrix for a specific block.

    Parameters
    ----------
    ld_ref : LDReference
        LD reference panel.
    block_id : str
        Block identifier.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (R, snp_ids) - LD correlation matrix and ordered SNP IDs.
    """
    block_path = ld_ref.root_path / "blocks" / f"block_{block_id}.npz"

    if not block_path.exists():
        raise FileNotFoundError(f"Block LD file not found: {block_path}")

    data = np.load(block_path, allow_pickle=True)
    R = data["ld_matrix"]
    snp_ids = data["snp_ids"].tolist()

    logging.debug("Loaded LD matrix for block %s: %d x %d", block_id, R.shape[0], R.shape[1])
    return R, snp_ids


def load_ld_for_snps(
    ld_ref: LDReference,
    snp_ids: List[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load LD matrices for a set of SNPs, aligning across blocks.

    Parameters
    ----------
    ld_ref : LDReference
        LD reference panel.
    snp_ids : List[str]
        List of SNP IDs to include.

    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray]
        (R, ordered_snp_ids, block_ids)
        - R: Block-diagonal LD matrix.
        - ordered_snp_ids: SNP IDs in matrix order.
        - block_ids: Block ID for each SNP.
    """
    # Filter to SNPs in the reference
    available_snps = set(ld_ref.snp_to_block.keys())
    matched_snps = [s for s in snp_ids if s in available_snps]

    if not matched_snps:
        raise ValueError("No requested SNPs found in LD reference")

    # Group SNPs by block
    block_to_snps: Dict[str, List[str]] = {}
    for snp_id in matched_snps:
        block_id = ld_ref.snp_to_block[snp_id]
        if block_id not in block_to_snps:
            block_to_snps[block_id] = []
        block_to_snps[block_id].append(snp_id)

    # Load each block and build block-diagonal matrix
    all_snp_ids = []
    all_block_ids = []
    block_matrices = []

    for block_id in sorted(block_to_snps.keys()):
        try:
            R_block, block_snp_order = load_block_ld_matrix(ld_ref, block_id)
        except FileNotFoundError:
            logging.warning("Block %s LD file not found, skipping", block_id)
            continue

        # Get requested SNPs in this block
        requested = set(block_to_snps[block_id])
        block_snp_set = set(block_snp_order)

        # Align: keep only SNPs that are both requested and in the block file
        keep_snps = [s for s in block_snp_order if s in requested]
        if not keep_snps:
            continue

        # Subset the LD matrix
        keep_idx = [block_snp_order.index(s) for s in keep_snps]
        R_sub = R_block[np.ix_(keep_idx, keep_idx)]

        block_matrices.append(R_sub)
        all_snp_ids.extend(keep_snps)
        all_block_ids.extend([block_id] * len(keep_snps))

    if not block_matrices:
        raise ValueError("No LD data could be loaded for requested SNPs")

    # Build block-diagonal matrix
    n_total = len(all_snp_ids)
    R_full = np.zeros((n_total, n_total), dtype=np.float64)

    offset = 0
    for R_block in block_matrices:
        n_block = R_block.shape[0]
        R_full[offset:offset + n_block, offset:offset + n_block] = R_block
        offset += n_block

    block_ids_array = np.array(all_block_ids)

    logging.info("Loaded LD for %d SNPs across %d blocks", n_total, len(block_matrices))
    return R_full, all_snp_ids, block_ids_array


def align_ld_to_vectors(
    R: np.ndarray,
    ld_snp_ids: List[str],
    beta_p: pd.Series,
    gamma: pd.Series,
) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    """Align LD matrix to match beta and gamma vectors.

    Parameters
    ----------
    R : np.ndarray
        LD correlation matrix.
    ld_snp_ids : List[str]
        SNP IDs in R matrix order.
    beta_p : pd.Series
        SNP-to-PAS effects indexed by SNP ID.
    gamma : pd.Series
        SNP-to-trait effects indexed by SNP ID.

    Returns
    -------
    Tuple[np.ndarray, pd.Series, pd.Series]
        (R_aligned, beta_aligned, gamma_aligned)
        All with matching SNP order.
    """
    # Find common SNPs
    common_snps = list(set(ld_snp_ids) & set(beta_p.index) & set(gamma.index))

    if not common_snps:
        raise ValueError("No common SNPs between LD, beta_p, and gamma")

    # Order by LD matrix
    snp_order = [s for s in ld_snp_ids if s in common_snps]

    # Subset LD matrix
    ld_idx = [ld_snp_ids.index(s) for s in snp_order]
    R_aligned = R[np.ix_(ld_idx, ld_idx)]

    # Align beta and gamma
    beta_aligned = beta_p.reindex(snp_order)
    gamma_aligned = gamma.reindex(snp_order)

    logging.info("Aligned %d SNPs for analysis", len(snp_order))
    return R_aligned, beta_aligned, gamma_aligned


def save_block_ld(
    output_path: Union[str, Path],
    block_id: str,
    ld_matrix: np.ndarray,
    snp_ids: List[str],
) -> None:
    """Save LD matrix for a block to NPZ format.

    Parameters
    ----------
    output_path : str or Path
        Directory to save block file.
    block_id : str
        Block identifier.
    ld_matrix : np.ndarray
        LD correlation matrix.
    snp_ids : List[str]
        Ordered SNP IDs.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"block_{block_id}.npz"
    np.savez(file_path, ld_matrix=ld_matrix, snp_ids=np.array(snp_ids))
    logging.info("Saved block %s LD matrix (%d x %d) to %s",
                block_id, ld_matrix.shape[0], ld_matrix.shape[1], file_path)
