## ------------------------------------------------------------------------------------------- ##
## LD Reference Setup Module                                                                  ##
## ------------------------------------------------------------------------------------------- ##
## @script: setup.py                                                                          ##
##                                                                                             ##
## @description: Downloads and configures LD reference panels from 1000 Genomes data.         ##
##               Enables SNP-based pathway analyses like pathway TWAS and genetic correlation.##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""LD Reference Setup Module.

Downloads and configures LD reference panels from 1000 Genomes data.
"""

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
SUPPORTED_ANCESTRIES: Dict[str, str] = {
    "EUR": "European (503 samples from CEU, TSI, FIN, GBR, IBS)",
    "AFR": "African (661 samples from YRI, LWK, GWD, MSL, ESN, ASW, ACB)",
    "EAS": "East Asian (504 samples from CHB, JPT, CHS, CDX, KHV)",
}

# URL patterns for 1000 Genomes PLINK files (via LDSC/Broad Institute)
PLINK_BASE_URL = "https://storage.googleapis.com/broad-alkesgroup-public/LDSCORE"

# LD block definitions from Berisa & Pickrell (2016)
LD_BLOCK_URLS = {
    "EUR": "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7f/EUR/fourier_ls-all.bed",
    "AFR": "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7f/AFR/fourier_ls-all.bed",
    "EAS": "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7f/ASN/fourier_ls-all.bed",
}

# Maximum SNPs per LD block before subsampling (memory management)
MAX_SNPS_PER_BLOCK = 2000


def setup_ld_reference(
    ancestry: str,
    output_dir: Path,
    chromosomes: Optional[List[int]] = None,
    maf_threshold: float = 0.01,
    keep_downloads: bool = False,
) -> bool:
    """
    Download and set up LD reference panel.

    Downloads 1000 Genomes PLINK files, computes LD matrices for each
    LD block, and saves in PathWAS-compatible format.

    Parameters
    ----------
    ancestry : str
        Population ancestry code: EUR, AFR, or EAS
    output_dir : Path
        Directory to save LD reference files
    chromosomes : list of int, optional
        Chromosomes to process (default: 1-22)
    maf_threshold : float
        Minimum minor allele frequency (default: 0.01)
    keep_downloads : bool
        Keep intermediate PLINK files (default: False)

    Returns
    -------
    bool
        True if successful

    Creates
    -------
    output_dir/{ancestry}/
        snp_manifest.tsv : SNP information (ID, chr, pos, alleles, MAF)
        block_definitions.bed : LD block boundaries
        blocks/block_{id}.npz : LD matrix and SNP IDs for each block
    """
    ancestry = ancestry.upper()
    if ancestry not in SUPPORTED_ANCESTRIES:
        raise ValueError(
            f"Unsupported ancestry '{ancestry}'. "
            f"Supported: {list(SUPPORTED_ANCESTRIES.keys())}"
        )

    if chromosomes is None:
        chromosomes = list(range(1, 23))

    output_dir = Path(output_dir)
    ancestry_dir = output_dir / ancestry
    ancestry_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = ancestry_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = ancestry_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Setting up LD reference for {ancestry}")
    print(f"Output: {ancestry_dir}")
    print("=" * 60)
    print()

    # Step 1: Download LD block definitions
    print("[Step 1/4] Downloading LD block definitions...")
    ld_blocks_file = download_ld_blocks(ancestry, downloads_dir)
    if ld_blocks_file is None:
        logging.error("Failed to download LD block definitions")
        return False
    ld_blocks = pd.read_csv(
        ld_blocks_file,
        sep="\t",
        names=["chrom", "start", "stop"],
        skiprows=1,
    )
    # Normalize chromosome names (remove 'chr' prefix if present)
    ld_blocks["chrom"] = ld_blocks["chrom"].astype(str).str.replace("chr", "")
    ld_blocks["chrom"] = pd.to_numeric(ld_blocks["chrom"], errors="coerce")
    ld_blocks = ld_blocks.dropna(subset=["chrom"])
    ld_blocks["chrom"] = ld_blocks["chrom"].astype(int)
    print(f"  Loaded {len(ld_blocks)} LD blocks")
    print()

    # Step 2: Download genotype files
    print("[Step 2/4] Downloading genotype files...")
    plink_prefixes = {}
    for chrom in tqdm(chromosomes, desc="Downloading chromosomes"):
        prefix = download_plink_files(ancestry, chrom, downloads_dir)
        if prefix is not None:
            plink_prefixes[chrom] = prefix
            logging.info("Downloaded chromosome %d", chrom)
        else:
            logging.warning("Failed to download chromosome %d", chrom)
    print(f"  Downloaded {len(plink_prefixes)} chromosome(s)")
    print()

    if not plink_prefixes:
        logging.error("No genotype files downloaded")
        return False

    # Step 3: Compute LD matrices
    print("[Step 3/4] Computing LD matrices...")
    all_snps = []
    block_id = 0

    for chrom in tqdm(sorted(plink_prefixes.keys()), desc="Processing chromosomes"):
        prefix = plink_prefixes[chrom]

        try:
            genotypes, bim, fam = read_plink_genotypes(prefix)
        except Exception as e:
            logging.error("Failed to read PLINK files for chr%d: %s", chrom, e)
            continue

        n_samples, n_snps = genotypes.shape
        print(f"  {n_samples} samples, {n_snps} SNPs")

        # Compute MAF and filter
        maf = compute_maf(genotypes)
        maf_mask = maf >= maf_threshold
        genotypes = genotypes[:, maf_mask]
        bim = bim[maf_mask].copy()
        maf = maf[maf_mask]
        bim["maf"] = maf
        print(f"  {len(bim)} SNPs after MAF filter (>= {maf_threshold})")

        # Get blocks for this chromosome
        chrom_blocks = ld_blocks[ld_blocks["chrom"] == chrom]

        for _, block_row in chrom_blocks.iterrows():
            block_start = block_row["start"]
            block_end = block_row["stop"]

            # Get SNPs in this block
            in_block = (bim["pos"].values >= block_start) & (
                bim["pos"].values < block_end
            )
            block_snp_indices = np.where(in_block)[0]

            if len(block_snp_indices) == 0:
                continue

            # Subsample if too many SNPs
            if len(block_snp_indices) > MAX_SNPS_PER_BLOCK:
                np.random.seed(42)
                block_snp_indices = np.sort(
                    np.random.choice(
                        block_snp_indices, MAX_SNPS_PER_BLOCK, replace=False
                    )
                )

            block_geno = genotypes[:, block_snp_indices]
            block_bim = bim.iloc[block_snp_indices]

            # Compute LD matrix
            ld_matrix = compute_ld_matrix(block_geno)

            # Save block
            block_file = blocks_dir / f"block_{block_id}.npz"
            snp_ids = block_bim["snp"].values
            np.savez_compressed(
                block_file,
                ld_matrix=ld_matrix.astype(np.float32),
                snp_ids=snp_ids,
            )

            block_id += 1

        # Add SNPs to manifest
        bim["chrom"] = chrom
        all_snps.append(
            bim[["snp", "chrom", "pos", "a1", "a2", "maf"]].rename(
                columns={
                    "snp": "snp_id",
                    "a1": "effect_allele",
                    "a2": "other_allele",
                }
            )
        )

        print(f"  Created {block_id} blocks, {sum(len(s) for s in all_snps)} SNPs")

    print()

    # Step 4: Save reference files
    print("[Step 4/4] Saving reference files...")

    # Save SNP manifest
    if all_snps:
        snp_manifest = pd.concat(all_snps, ignore_index=True)
        manifest_file = ancestry_dir / "snp_manifest.tsv"
        snp_manifest.to_csv(manifest_file, sep="\t", index=False)
        print(f"  Saved SNP manifest: {len(snp_manifest)} SNPs")
    else:
        logging.error("No SNPs processed")
        return False

    # Save block definitions
    blocks_file = ancestry_dir / "block_definitions.bed"
    ld_blocks.to_csv(blocks_file, sep="\t", index=False, header=False)
    print(f"  Saved block definitions: {len(ld_blocks)} blocks")

    # Cleanup
    if not keep_downloads:
        shutil.rmtree(downloads_dir, ignore_errors=True)

    print()
    print("=" * 60)
    print("LD REFERENCE SETUP COMPLETE")
    print("=" * 60)
    print(f"Ancestry: {ancestry}")
    print(f"Location: {ancestry_dir}")
    print(f"SNPs: {len(snp_manifest):,}")
    print(f"Blocks: {block_id}")
    print()
    print("Use with PathWAS:")
    print(f"  pathwas --ld-root {output_dir} --ld-ancestry {ancestry} ...")

    return True


def download_plink_files(
    ancestry: str,
    chrom: int,
    dest_dir: Path,
) -> Optional[Path]:
    """
    Download PLINK .bed/.bim/.fam files for one chromosome.

    Parameters
    ----------
    ancestry : str
        Population ancestry code (EUR, AFR, or EAS)
    chrom : int
        Chromosome number (1-22)
    dest_dir : Path
        Directory to save downloaded files

    Returns
    -------
    Path or None
        PLINK prefix if successful, None otherwise
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Map ancestry to 1000G naming convention
    ancestry_map = {"EUR": "EUR", "AFR": "AFR", "EAS": "EAS"}
    pop = ancestry_map.get(ancestry.upper(), ancestry.upper())

    prefix = dest_dir / f"1000G.{pop}.{chrom}"
    extensions = ["bed", "bim", "fam"]

    all_exist = all((prefix.with_suffix(f".{ext}")).exists() for ext in extensions)
    if all_exist:
        logging.info("PLINK files for chr%d already exist, skipping", chrom)
        return prefix

    for ext in extensions:
        url = f"{PLINK_BASE_URL}/1000G_Phase3_plinkfiles/1000G.{pop}.{chrom}.{ext}"
        dest_file = prefix.with_suffix(f".{ext}")

        if dest_file.exists():
            continue

        print(f"  Downloading chr{chrom}.{ext}...")
        success = _download_file(url, dest_file)
        if not success:
            logging.error("Failed to download %s", url)
            return None

    return prefix


def download_ld_blocks(
    ancestry: str,
    dest_dir: Path,
) -> Optional[Path]:
    """
    Download LD block definitions.

    Parameters
    ----------
    ancestry : str
        Population ancestry code (EUR, AFR, or EAS)
    dest_dir : Path
        Directory to save downloaded file

    Returns
    -------
    Path or None
        Path to block definitions file if successful, None otherwise
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ancestry = ancestry.upper()
    if ancestry not in LD_BLOCK_URLS:
        logging.error("No LD block URL for ancestry %s", ancestry)
        return None

    url = LD_BLOCK_URLS[ancestry]
    dest_file = dest_dir / f"ld_blocks_{ancestry}.bed"

    if dest_file.exists():
        logging.info("LD blocks file already exists, skipping download")
        return dest_file

    success = _download_file(url, dest_file)
    if success:
        return dest_file
    return None


def _download_file(url: str, dest: Path, max_retries: int = 3) -> bool:
    """
    Download a file with retry logic.

    Parameters
    ----------
    url : str
        URL to download
    dest : Path
        Destination path
    max_retries : int
        Maximum number of retry attempts

    Returns
    -------
    bool
        True if successful
    """
    # Try wget first (shows progress)
    if shutil.which("wget"):
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", str(dest), url],
                    check=True,
                    capture_output=False,
                )
                return True
            except subprocess.CalledProcessError:
                if attempt < max_retries - 1:
                    logging.warning("Download attempt %d failed, retrying...", attempt + 1)
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        # wget failed, fall through to urllib

    # Fallback to urllib
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning("Download attempt %d failed: %s", attempt + 1, e)
                import time
                time.sleep(2 ** attempt)
            continue

    return False


def read_plink_genotypes(
    plink_prefix: Path,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Read PLINK binary files.

    Parameters
    ----------
    plink_prefix : Path
        PLINK file prefix (without .bed/.bim/.fam extension)

    Returns
    -------
    genotypes : np.ndarray
        Shape (n_samples, n_snps), values 0/1/2/NaN for missing
    bim : pd.DataFrame
        SNP information (chrom, snp, cm, pos, a1, a2)
    fam : pd.DataFrame
        Sample information (fid, iid, father, mother, sex, pheno)
    """
    plink_prefix = Path(plink_prefix)

    # Read .bim file (SNP info)
    bim_file = plink_prefix.with_suffix(".bim")
    bim = pd.read_csv(
        bim_file,
        sep="\t",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
    )

    # Read .fam file (sample info)
    fam_file = plink_prefix.with_suffix(".fam")
    fam = pd.read_csv(
        fam_file,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "pheno"],
    )

    n_snps = len(bim)
    n_samples = len(fam)

    # Read .bed file (binary genotypes)
    bed_file = plink_prefix.with_suffix(".bed")
    genotypes = _read_bed_file(bed_file, n_samples, n_snps)

    return genotypes, bim, fam


def _read_bed_file(bed_file: Path, n_samples: int, n_snps: int) -> np.ndarray:
    """
    Read PLINK .bed binary file using vectorized operations.

    PLINK .bed format (SNP-major mode):
    - 3-byte magic number: 0x6c 0x1b 0x01
    - Each SNP stored as ceil(n_samples/4) bytes
    - 2 bits per sample: 00=hom_ref(0), 01=missing, 10=het(1), 11=hom_alt(2)

    Parameters
    ----------
    bed_file : Path
        Path to .bed file
    n_samples : int
        Number of samples
    n_snps : int
        Number of SNPs

    Returns
    -------
    np.ndarray
        Genotype matrix (n_samples, n_snps), values 0/1/2/NaN
    """
    bytes_per_snp = (n_samples + 3) // 4

    with open(bed_file, "rb") as f:
        # Check magic number
        magic = f.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"Invalid PLINK .bed file: {bed_file}")

        # Read genotype data
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape to (n_snps, bytes_per_snp)
    raw = raw.reshape((n_snps, bytes_per_snp))

    # Vectorized decoding of 2-bit genotypes
    # Expand each byte into 4 genotype codes
    genotypes = np.zeros((n_snps, n_samples), dtype=np.float32)

    for bit_pos in range(4):
        # Extract 2-bit codes for this position across all bytes
        codes = (raw >> (bit_pos * 2)) & 0x03

        # Calculate sample indices for this bit position
        sample_indices = np.arange(bytes_per_snp) * 4 + bit_pos
        valid_mask = sample_indices < n_samples

        if not valid_mask.any():
            continue

        valid_indices = sample_indices[valid_mask]
        valid_codes = codes[:, valid_mask]

        # Map codes: 0->0 (hom ref), 1->NaN (missing), 2->1 (het), 3->2 (hom alt)
        genotypes[:, valid_indices] = np.where(
            valid_codes == 0, 0.0,
            np.where(valid_codes == 2, 1.0,
            np.where(valid_codes == 3, 2.0, np.nan))
        )

    return genotypes.T  # Return (n_samples, n_snps)


def compute_ld_matrix(genotypes: np.ndarray) -> np.ndarray:
    """
    Compute LD correlation matrix from genotypes.

    Handles missing values via mean imputation.
    Returns Pearson correlation matrix.

    Parameters
    ----------
    genotypes : np.ndarray
        Genotype matrix (n_samples, n_snps), values 0/1/2/NaN

    Returns
    -------
    np.ndarray
        LD correlation matrix (n_snps, n_snps)
    """
    n_samples, n_snps = genotypes.shape

    # Mean imputation for missing values
    geno_imputed = genotypes.copy()
    for j in range(n_snps):
        col = geno_imputed[:, j]
        mask = np.isnan(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0.0
            geno_imputed[mask, j] = col_mean

    # Center and scale
    means = np.mean(geno_imputed, axis=0)
    stds = np.std(geno_imputed, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero

    geno_scaled = (geno_imputed - means) / stds

    # Compute correlation matrix
    ld_matrix = np.dot(geno_scaled.T, geno_scaled) / n_samples

    return ld_matrix


def compute_maf(genotypes: np.ndarray) -> np.ndarray:
    """
    Compute minor allele frequency for each SNP.

    Parameters
    ----------
    genotypes : np.ndarray
        Genotype matrix (n_samples, n_snps), values 0/1/2/NaN

    Returns
    -------
    np.ndarray
        Minor allele frequency for each SNP
    """
    n_snps = genotypes.shape[1]
    maf = np.zeros(n_snps)

    for j in range(n_snps):
        col = genotypes[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            maf[j] = 0.0
            continue

        # Allele frequency = mean dosage / 2
        freq = np.nanmean(col) / 2.0

        # Minor allele frequency
        maf[j] = min(freq, 1.0 - freq)

    return maf


def list_ancestries() -> None:
    """Print available ancestries and their descriptions."""
    print()
    print("Available LD reference ancestries:")
    print()
    for code, desc in SUPPORTED_ANCESTRIES.items():
        print(f"  {code}: {desc}")
    print()
    print("Data source: 1000 Genomes Phase 3 (via LDSC)")
    print("LD blocks: Berisa & Pickrell (2016)")
    print()
