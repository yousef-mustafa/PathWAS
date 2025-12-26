## ------------------------------------------------------------------------------------------- ##
## Pathway Gene Set Loading Utilities                                                         ##
## ------------------------------------------------------------------------------------------- ##
## @script: gene_sets.py                                                                      ##
##                                                                                            ##
## @description: Utilities for loading pathway gene sets from multiple databases              ##
##               (KEGG, Reactome, WikiPathways, GO, MSigDB Hallmark) and custom files.       ##
##                                                                                            ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                         ##
## ------------------------------------------------------------------------------------------- ##

"""Pathway gene set loading utilities."""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Union

import pandas as pd


# Standard source types
SourceType = Literal["kegg", "reactome", "wikipathways", "go", "hallmark", "custom"]


def load_gene_sets(
    source: SourceType,
    file_path: Optional[Union[str, Path]] = None,
    gene_sets: Optional[Dict[str, Iterable[str]]] = None,
    organism: str = "human",
    go_namespace: str = "BP",
    **kwargs,
) -> Dict[str, Set[str]]:
    """Load gene sets from various sources.

    Parameters
    ----------
    source : {"kegg", "reactome", "wikipathways", "go", "hallmark", "custom"}
        Source of gene sets.
    file_path : str or Path, optional
        Path to gene set file (GMT or TSV format).
        Required for built-in sources if no bundled data is available.
    gene_sets : dict, optional
        For source="custom", a dictionary mapping pathway names to gene lists.
    organism : str
        Organism identifier (e.g., "human", "mouse"). Used for filtering.
    go_namespace : str
        GO namespace for source="go" (e.g., "BP", "MF", "CC").
    **kwargs
        Additional arguments passed to source-specific loaders.

    Returns
    -------
    dict
        Dictionary mapping pathway_name -> set of gene IDs.

    Raises
    ------
    ValueError
        If required arguments are missing or source is unknown.

    Examples
    --------
    >>> # Load from GMT file
    >>> gene_sets = load_gene_sets("kegg", file_path="c2.cp.kegg.v7.gmt")
    >>>
    >>> # Load custom gene sets from dict
    >>> custom = {"pathway1": ["GENE1", "GENE2"], "pathway2": ["GENE3", "GENE4"]}
    >>> gene_sets = load_gene_sets("custom", gene_sets=custom)
    >>>
    >>> # Load from GMT file for any source
    >>> gene_sets = load_gene_sets("hallmark", file_path="h.all.v7.gmt")
    """
    source = source.lower()

    if source == "custom":
        return _load_custom_gene_sets(file_path=file_path, gene_sets=gene_sets)

    if file_path is None:
        raise ValueError(
            f"file_path is required for source='{source}'. "
            "Provide a path to a GMT or TSV file containing gene sets."
        )

    # Determine file format and load
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()

    if file_ext == ".gmt":
        all_gene_sets = load_gmt(file_path)
    elif file_ext in [".tsv", ".txt", ".csv"]:
        all_gene_sets = load_tabular_gene_sets(file_path)
    else:
        # Try GMT format first, fall back to tabular
        try:
            all_gene_sets = load_gmt(file_path)
        except Exception:
            all_gene_sets = load_tabular_gene_sets(file_path)

    # Apply source-specific filtering
    if source == "kegg":
        all_gene_sets = _filter_kegg(all_gene_sets, organism=organism)
    elif source == "reactome":
        all_gene_sets = _filter_reactome(all_gene_sets, organism=organism)
    elif source == "wikipathways":
        all_gene_sets = _filter_wikipathways(all_gene_sets, organism=organism)
    elif source == "go":
        all_gene_sets = _filter_go(all_gene_sets, namespace=go_namespace)
    elif source == "hallmark":
        all_gene_sets = _filter_hallmark(all_gene_sets)
    else:
        raise ValueError(
            f"Unknown source: '{source}'. "
            "Use 'kegg', 'reactome', 'wikipathways', 'go', 'hallmark', or 'custom'."
        )

    logging.info("Loaded %d gene sets from %s", len(all_gene_sets), source)
    return all_gene_sets


def load_gmt(file_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """Load gene sets from GMT (Gene Matrix Transposed) format.

    GMT format: one line per gene set
        pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...

    Parameters
    ----------
    file_path : str or Path
        Path to GMT file.

    Returns
    -------
    dict
        Dictionary mapping pathway_name -> set of gene IDs.

    Examples
    --------
    >>> gene_sets = load_gmt("pathways.gmt")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"GMT file not found: {file_path}")

    gene_sets = {}

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            pathway_name = parts[0]
            # parts[1] is description, skip it
            genes = set(parts[2:])

            # Remove empty strings
            genes = {g for g in genes if g}

            if genes:
                gene_sets[pathway_name] = genes

    logging.debug("Loaded %d gene sets from GMT file: %s", len(gene_sets), file_path)
    return gene_sets


def load_tabular_gene_sets(
    file_path: Union[str, Path],
    pathway_col: str = "pathway",
    gene_col: str = "gene",
    sep: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """Load gene sets from tabular (TSV/CSV) format.

    Expected format: two columns with pathway and gene IDs.
    Each row maps one gene to one pathway.

    Parameters
    ----------
    file_path : str or Path
        Path to tabular file.
    pathway_col : str
        Column name for pathway identifiers.
    gene_col : str
        Column name for gene identifiers.
    sep : str, optional
        Column separator. If None, inferred from file extension.

    Returns
    -------
    dict
        Dictionary mapping pathway_name -> set of gene IDs.

    Examples
    --------
    >>> gene_sets = load_tabular_gene_sets("pathways.tsv", pathway_col="pathway", gene_col="gene")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Gene set file not found: {file_path}")

    # Infer separator
    if sep is None:
        if file_path.suffix.lower() == ".csv":
            sep = ","
        else:
            sep = "\t"

    df = pd.read_csv(file_path, sep=sep)

    # Check for required columns
    if pathway_col not in df.columns:
        # Try common alternatives
        for alt in ["pathway", "Pathway", "PATHWAY", "term", "Term", "set", "Set"]:
            if alt in df.columns:
                pathway_col = alt
                break
        else:
            raise ValueError(f"Could not find pathway column. Columns: {list(df.columns)}")

    if gene_col not in df.columns:
        for alt in ["gene", "Gene", "GENE", "symbol", "Symbol", "gene_id", "gene_symbol"]:
            if alt in df.columns:
                gene_col = alt
                break
        else:
            raise ValueError(f"Could not find gene column. Columns: {list(df.columns)}")

    gene_sets = {}
    for pathway, group in df.groupby(pathway_col):
        genes = set(group[gene_col].dropna().astype(str))
        if genes:
            gene_sets[pathway] = genes

    logging.debug("Loaded %d gene sets from tabular file: %s", len(gene_sets), file_path)
    return gene_sets


def _load_custom_gene_sets(
    file_path: Optional[Union[str, Path]] = None,
    gene_sets: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict[str, Set[str]]:
    """Load custom gene sets from file or dict.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to GMT or TSV file.
    gene_sets : dict, optional
        Dictionary mapping pathway_name -> list of genes.

    Returns
    -------
    dict
        Dictionary mapping pathway_name -> set of gene IDs.
    """
    if gene_sets is not None:
        # Convert to sets
        return {name: set(genes) for name, genes in gene_sets.items()}

    if file_path is not None:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".gmt":
            return load_gmt(file_path)
        else:
            return load_tabular_gene_sets(file_path)

    raise ValueError("Either file_path or gene_sets must be provided for custom source")


def _filter_kegg(
    gene_sets: Dict[str, Set[str]],
    organism: str = "human",
) -> Dict[str, Set[str]]:
    """Filter gene sets for KEGG pathways.

    Parameters
    ----------
    gene_sets : dict
        All loaded gene sets.
    organism : str
        Organism to filter for.

    Returns
    -------
    dict
        Filtered gene sets.
    """
    # KEGG pathway names often contain organism prefix (e.g., "hsa" for human)
    organism_prefixes = {
        "human": ["hsa", "KEGG_", "HSA_"],
        "mouse": ["mmu", "MMU_"],
        "rat": ["rno", "RNO_"],
    }

    prefixes = organism_prefixes.get(organism.lower(), [])

    if not prefixes:
        # No filtering
        return gene_sets

    filtered = {}
    for name, genes in gene_sets.items():
        # Check if pathway name matches organism
        name_upper = name.upper()
        if any(name_upper.startswith(p.upper()) or p.upper() in name_upper for p in prefixes):
            filtered[name] = genes
        elif "KEGG" in name_upper:
            # Generic KEGG pathway, include
            filtered[name] = genes

    if not filtered:
        # If no matches, return all (user may have custom naming)
        logging.warning("No KEGG pathways matched organism '%s', returning all", organism)
        return gene_sets

    return filtered


def _filter_reactome(
    gene_sets: Dict[str, Set[str]],
    organism: str = "human",
) -> Dict[str, Set[str]]:
    """Filter gene sets for Reactome pathways.

    Parameters
    ----------
    gene_sets : dict
        All loaded gene sets.
    organism : str
        Organism to filter for.

    Returns
    -------
    dict
        Filtered gene sets.
    """
    # Reactome pathways often include organism in name
    organism_patterns = {
        "human": ["homo sapiens", "h sapiens", "human"],
        "mouse": ["mus musculus", "m musculus", "mouse"],
    }

    patterns = organism_patterns.get(organism.lower(), [])

    if not patterns:
        return gene_sets

    filtered = {}
    for name, genes in gene_sets.items():
        name_lower = name.lower()
        if any(p in name_lower for p in patterns):
            filtered[name] = genes
        elif "REACTOME" in name.upper():
            # Include generic Reactome pathways
            filtered[name] = genes

    if not filtered:
        logging.warning("No Reactome pathways matched organism '%s', returning all", organism)
        return gene_sets

    return filtered


def _filter_wikipathways(
    gene_sets: Dict[str, Set[str]],
    organism: str = "human",
) -> Dict[str, Set[str]]:
    """Filter gene sets for WikiPathways.

    Parameters
    ----------
    gene_sets : dict
        All loaded gene sets.
    organism : str
        Organism to filter for.

    Returns
    -------
    dict
        Filtered gene sets.
    """
    # WikiPathways uses WP prefix with species
    organism_patterns = {
        "human": ["_Homo_sapiens", "_human", "WP"],
        "mouse": ["_Mus_musculus", "_mouse"],
    }

    patterns = organism_patterns.get(organism.lower(), [])

    if not patterns:
        return gene_sets

    filtered = {}
    for name, genes in gene_sets.items():
        if any(p in name for p in patterns):
            filtered[name] = genes
        elif "WIKIPATHWAYS" in name.upper() or "WP" in name.upper():
            filtered[name] = genes

    if not filtered:
        logging.warning("No WikiPathways matched organism '%s', returning all", organism)
        return gene_sets

    return filtered


def _filter_go(
    gene_sets: Dict[str, Set[str]],
    namespace: str = "BP",
) -> Dict[str, Set[str]]:
    """Filter gene sets for Gene Ontology.

    Parameters
    ----------
    gene_sets : dict
        All loaded gene sets.
    namespace : str
        GO namespace: "BP" (Biological Process), "MF" (Molecular Function),
        "CC" (Cellular Component).

    Returns
    -------
    dict
        Filtered gene sets.
    """
    namespace_patterns = {
        "BP": ["GO_BP", "GOBP", "BIOLOGICAL_PROCESS", "GO:"],
        "MF": ["GO_MF", "GOMF", "MOLECULAR_FUNCTION"],
        "CC": ["GO_CC", "GOCC", "CELLULAR_COMPONENT"],
    }

    patterns = namespace_patterns.get(namespace.upper(), [])

    if not patterns:
        return gene_sets

    filtered = {}
    for name, genes in gene_sets.items():
        name_upper = name.upper()
        if any(p in name_upper for p in patterns):
            filtered[name] = genes

    if not filtered:
        logging.warning("No GO terms matched namespace '%s', returning all", namespace)
        return gene_sets

    return filtered


def _filter_hallmark(gene_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Filter gene sets for MSigDB Hallmark.

    Parameters
    ----------
    gene_sets : dict
        All loaded gene sets.

    Returns
    -------
    dict
        Filtered gene sets (Hallmark sets start with "HALLMARK_").
    """
    filtered = {}
    for name, genes in gene_sets.items():
        if name.upper().startswith("HALLMARK"):
            filtered[name] = genes

    if not filtered:
        logging.warning("No Hallmark gene sets found, returning all")
        return gene_sets

    return filtered


def save_gene_sets_gmt(
    gene_sets: Dict[str, Set[str]],
    file_path: Union[str, Path],
    descriptions: Optional[Dict[str, str]] = None,
) -> None:
    """Save gene sets to GMT format.

    Parameters
    ----------
    gene_sets : dict
        Dictionary mapping pathway_name -> set of gene IDs.
    file_path : str or Path
        Output file path.
    descriptions : dict, optional
        Dictionary mapping pathway_name -> description.

    Examples
    --------
    >>> save_gene_sets_gmt(gene_sets, "output.gmt")
    """
    file_path = Path(file_path)
    descriptions = descriptions or {}

    with open(file_path, "w") as f:
        for pathway, genes in gene_sets.items():
            desc = descriptions.get(pathway, "")
            genes_str = "\t".join(sorted(genes))
            f.write(f"{pathway}\t{desc}\t{genes_str}\n")

    logging.info("Saved %d gene sets to %s", len(gene_sets), file_path)


def gene_sets_to_dataframe(
    gene_sets: Dict[str, Set[str]],
) -> pd.DataFrame:
    """Convert gene sets dict to long-format DataFrame.

    Parameters
    ----------
    gene_sets : dict
        Dictionary mapping pathway_name -> set of gene IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: pathway, gene.

    Examples
    --------
    >>> df = gene_sets_to_dataframe(gene_sets)
    >>> df.head()
          pathway   gene
    0  pathway_1  GENE1
    1  pathway_1  GENE2
    """
    records = []
    for pathway, genes in gene_sets.items():
        for gene in genes:
            records.append({"pathway": pathway, "gene": gene})

    # Ensure columns exist even for empty input
    if not records:
        return pd.DataFrame(columns=["pathway", "gene"])

    return pd.DataFrame(records)


def filter_gene_sets_by_size(
    gene_sets: Dict[str, Set[str]],
    min_size: int = 5,
    max_size: Optional[int] = 500,
) -> Dict[str, Set[str]]:
    """Filter gene sets by size.

    Parameters
    ----------
    gene_sets : dict
        Dictionary mapping pathway_name -> set of gene IDs.
    min_size : int
        Minimum number of genes.
    max_size : int, optional
        Maximum number of genes. If None, no upper limit is applied.

    Returns
    -------
    dict
        Filtered gene sets.

    Examples
    --------
    >>> filtered = filter_gene_sets_by_size(gene_sets, min_size=10, max_size=200)
    """
    if max_size is None:
        filtered = {
            name: genes
            for name, genes in gene_sets.items()
            if len(genes) >= min_size
        }
    else:
        filtered = {
            name: genes
            for name, genes in gene_sets.items()
            if min_size <= len(genes) <= max_size
        }

    n_removed = len(gene_sets) - len(filtered)
    if n_removed > 0:
        max_str = str(max_size) if max_size is not None else "unlimited"
        logging.info(
            "Filtered %d gene sets by size (min=%d, max=%s), %d remaining",
            n_removed,
            min_size,
            max_str,
            len(filtered),
        )

    return filtered


def intersect_gene_sets_with_genes(
    gene_sets: Dict[str, Set[str]],
    available_genes: Iterable[str],
) -> Dict[str, Set[str]]:
    """Intersect gene sets with available genes.

    Keeps only genes that are in the available set, useful for matching
    gene sets to expression matrix columns.

    Parameters
    ----------
    gene_sets : dict
        Dictionary mapping pathway_name -> set of gene IDs.
    available_genes : iterable
        Set of available gene IDs (e.g., from expression matrix).

    Returns
    -------
    dict
        Gene sets with only available genes.

    Examples
    --------
    >>> matched = intersect_gene_sets_with_genes(gene_sets, expression.columns)
    """
    available = set(available_genes)

    intersected = {}
    total_before = 0
    total_after = 0

    for name, genes in gene_sets.items():
        total_before += len(genes)
        matched = genes & available
        total_after += len(matched)

        if matched:
            intersected[name] = matched

    logging.info(
        "Intersected gene sets with available genes: %d/%d genes matched, %d/%d pathways retained",
        total_after,
        total_before,
        len(intersected),
        len(gene_sets),
    )

    return intersected
