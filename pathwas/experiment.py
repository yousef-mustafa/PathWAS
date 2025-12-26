## ------------------------------------------------------------------------------------------- ##
## PathWAS Experiment Orchestration                                                            ##
## ------------------------------------------------------------------------------------------- ##
## @script: experiment.py                                                                      ##
##                                                                                             ##
## @description: Manages experiment directories, configuration loading/merging, and            ##
##               orchestrates the full PathWAS workflow from config to results.                ##
##                                                                                             ##
## @author: Yousef Mustafa, Lab of Dr. William Bush.                                           ##
## ------------------------------------------------------------------------------------------- ##

"""Experiment orchestration and directory management for PathWAS."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

try:
    import markdown
except ImportError:
    markdown = None

from .io.data_prep import convert_gene_ids, convert_gene_list, load_msigdb_library
from .pas.pas import compute_pas
from .pas.pas_test import (
    TestMethod,
    MultipleTestingCorrection,
    PASTestConfig,
    test_pas_difference,
    test_pas_with_covariates,
    summarize_test_results,
)
from .pas.pas_viz import (
    HeatmapConfig,
    BoxplotConfig,
    ClusterMethod,
    plot_pas_heatmap,
    plot_pas_boxplot_multi,
    plot_pas_volcano,
    create_pas_report_figures,
)
from .association.pathway_test import aggregate_variants, association_test


# ----------------------------- Constants ------------------------------------ #

DEFAULT_EXPERIMENTS_DIR = "experiments"
DEFAULT_CONFIG = {
    "experiment": {
        "name": None,
        "description": "",
        "overwrite": False,
    },
    "data": {
        "expression": None,
        "genotypes": None,
        "covariates": None,
        "group_labels": None,  # Path to group labels file for PAS analysis
    },
    "pathways": {
        "source": "hallmark",
        "file": None,
    },
    "preprocessing": {
        "normalization": "none",
        "log_transform": False,
        "zscore_genes": False,
    },
    "modeling": {
        "model_type": "ridge",
        "model_params": {
            "lambda": 0.1,
            "mixture_components": 3,
        },
    },
    "association": {
        "ld_root": None,
        "ld_ancestry": "EUR_1KG",
    },
    "pas_analysis": {
        "enabled": False,
        "test_method": "ttest",  # ttest, welch, mann_whitney, anova, kruskal, linear
        "alpha": 0.05,
        "correction": "fdr_bh",  # none, bonferroni, fdr_bh, fdr_by
        "adjust_covariates": False,
        "group_column": "group",  # Column name in group labels file
    },
    "visualization": {
        "enabled": True,
        "heatmap": {
            "enabled": True,
            "title": "PAS Heatmap",
            "cluster_rows": True,
            "cluster_cols": True,
            "cluster_method": "average",
            "cmap": "RdBu_r",
            "show_row_labels": True,
            "show_col_labels": False,
        },
        "boxplot": {
            "enabled": True,
            "title": "PAS Distribution by Group",
            "show_points": True,
            "show_violin": False,
            "palette": "Set2",
            "top_n": 10,
        },
        "volcano": {
            "enabled": True,
            "title": "PAS Differential Analysis",
            "alpha": 0.05,
            "effect_threshold": 0.5,
            "top_n_labels": 10,
        },
    },
    "output": {
        "base_dir": DEFAULT_EXPERIMENTS_DIR,
        "save_intermediate": True,
    },
}


# ----------------------------- Directory Management ------------------------- #

def ensure_experiments_dir(base_dir: Optional[str] = None) -> Path:
    """Ensure the experiments directory exists.

    Parameters
    ----------
    base_dir : str, optional
        Base directory for experiments. Defaults to 'experiments/' in the
        current working directory.

    Returns
    -------
    pathlib.Path
        Path to the experiments directory.
    """
    if base_dir is None:
        base_dir = DEFAULT_EXPERIMENTS_DIR

    experiments_path = Path(base_dir)
    if not experiments_path.is_absolute():
        experiments_path = Path.cwd() / experiments_path

    experiments_path.mkdir(parents=True, exist_ok=True)
    logging.debug("Experiments directory ensured: %s", experiments_path)
    return experiments_path


def generate_experiment_name() -> str:
    """Generate a unique experiment name with timestamp and random suffix.

    Returns
    -------
    str
        A unique experiment name like '20251226_191530_abc123'.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = os.urandom(3).hex()
    return f"{timestamp}_{random_suffix}"


def setup_experiment_dir(
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
) -> Path:
    """Set up the experiment directory.

    Parameters
    ----------
    config : dict
        Experiment configuration dictionary.
    base_dir : str, optional
        Base directory for experiments.

    Returns
    -------
    pathlib.Path
        Path to the experiment directory.

    Raises
    ------
    FileExistsError
        If the experiment directory exists and overwrite is False.
    """
    experiments_path = ensure_experiments_dir(base_dir)

    # Determine experiment name
    exp_name = config.get("experiment", {}).get("name")
    if not exp_name:
        exp_name = generate_experiment_name()
        config.setdefault("experiment", {})["name"] = exp_name

    exp_dir = experiments_path / exp_name
    overwrite = config.get("experiment", {}).get("overwrite", False)

    if exp_dir.exists():
        if overwrite:
            logging.warning("Overwriting existing experiment directory: %s", exp_dir)
            shutil.rmtree(exp_dir)
        else:
            raise FileExistsError(
                f"Experiment directory already exists: {exp_dir}. "
                f"Use --overwrite to replace it."
            )

    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created experiment directory: %s", exp_dir)
    return exp_dir


# ----------------------------- Config Management ---------------------------- #

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a configuration file (YAML or JSON).

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        content = f.read()

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files")
        return yaml.safe_load(content)
    elif suffix == ".json":
        return json.loads(content)
    else:
        # Try YAML first, then JSON
        if yaml is not None:
            try:
                return yaml.safe_load(content)
            except Exception:
                pass
        return json.loads(content)


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into base config.

    Parameters
    ----------
    base : dict
        Base configuration dictionary.
    overrides : dict
        Override values to merge in.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    output_path : pathlib.Path
        Output file path.
    """
    if yaml is None:
        # Fall back to JSON if YAML not available
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
    else:
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def build_config_from_args(args) -> Dict[str, Any]:
    """Build configuration dictionary from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config = {
        "experiment": {
            "name": getattr(args, "experiment_name", None),
            "description": getattr(args, "note", ""),
            "overwrite": getattr(args, "overwrite", False),
        },
        "data": {
            "expression": getattr(args, "expression", None),
            "genotypes": getattr(args, "genotypes", None),
            "covariates": getattr(args, "covariates", None),
        },
        "pathways": {
            "source": getattr(args, "pathways", "hallmark"),
            "file": getattr(args, "pathways_file", None),
        },
        "preprocessing": {
            "normalization": getattr(args, "expr_normalization", "none"),
            "log_transform": getattr(args, "expr_log_transform", False),
            "zscore_genes": getattr(args, "expr_zscore_genes", False),
        },
        "modeling": {
            "model_type": getattr(args, "model", "ridge"),
            "model_params": {},
        },
        "association": {
            "ld_root": getattr(args, "ld_root", None),
            "ld_ancestry": getattr(args, "ld_ancestry", "EUR_1KG"),
        },
        "output": {
            "base_dir": DEFAULT_EXPERIMENTS_DIR,
            "save_intermediate": True,
        },
    }

    # Handle model-specific parameters
    if hasattr(args, "lambda_param") and args.lambda_param is not None:
        config["modeling"]["model_params"]["lambda"] = args.lambda_param
    if hasattr(args, "mixture_components") and args.mixture_components is not None:
        config["modeling"]["model_params"]["mixture_components"] = args.mixture_components

    return config


# ----------------------------- Metadata & Reporting ------------------------- #

def get_git_commit() -> Optional[str]:
    """Get the current git commit hash if available.

    Returns
    -------
    str or None
        Git commit hash or None if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def save_metadata(config: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    """Save experiment metadata to JSON file.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    exp_dir : pathlib.Path
        Experiment directory.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    metadata = {
        "experiment_name": config.get("experiment", {}).get("name", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "description": config.get("experiment", {}).get("description", ""),
        "inputs": {
            "expression": config.get("data", {}).get("expression"),
            "genotypes": config.get("data", {}).get("genotypes"),
            "covariates": config.get("data", {}).get("covariates"),
            "ld_root": config.get("association", {}).get("ld_root"),
        },
        "model_type": config.get("modeling", {}).get("model_type"),
        "pathway_source": config.get("pathways", {}).get("source"),
        "git_commit": get_git_commit(),
    }

    metadata_path = exp_dir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logging.info("Saved experiment metadata to %s", metadata_path)
    return metadata


def generate_markdown_report(
    config: Dict[str, Any],
    results: Dict[str, Any],
    exp_dir: Path,
) -> str:
    """Generate a Markdown report summarizing the experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    results : dict
        Experiment results.
    exp_dir : pathlib.Path
        Experiment directory.

    Returns
    -------
    str
        Path to the generated Markdown report.
    """
    exp_name = config.get("experiment", {}).get("name", "unknown")
    description = config.get("experiment", {}).get("description", "No description provided.")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build preprocessing summary
    preproc = config.get("preprocessing", {})
    preproc_items = []
    if preproc.get("normalization", "none") != "none":
        preproc_items.append(f"Normalization: {preproc['normalization']}")
    if preproc.get("log_transform"):
        preproc_items.append("Log transform: Yes")
    if preproc.get("zscore_genes"):
        preproc_items.append("Z-score genes: Yes")
    if not preproc_items:
        preproc_items.append("None (raw expression)")

    # Build model summary
    modeling = config.get("modeling", {})
    model_type = modeling.get("model_type", "ridge")
    model_params = modeling.get("model_params", {})
    model_params_str = ", ".join(f"{k}={v}" for k, v in model_params.items()) or "default"

    # Build pathway summary
    pathways_cfg = config.get("pathways", {})
    pathway_source = pathways_cfg.get("source", "unknown")
    pathway_file = pathways_cfg.get("file", "N/A")

    # Build results summary
    pas_summary = results.get("pas_summary", {})
    num_pathways = pas_summary.get("num_pathways", 0)
    num_samples = pas_summary.get("num_samples", 0)
    top_pathways = results.get("top_pathways", [])

    # Association results
    association_results = results.get("association", {})
    top_associations = association_results.get("top_pathways", [])

    # QC metrics
    qc_metrics = results.get("qc", {})

    # Generate Markdown content
    md_content = f"""# PathWAS Experiment Report

## Experiment: {exp_name}

**Date:** {timestamp}

**Description:** {description}

---

## Configuration Summary

### Data Inputs

| Input | Path |
|-------|------|
| Expression | `{config.get('data', {}).get('expression', 'N/A')}` |
| Genotypes | `{config.get('data', {}).get('genotypes', 'N/A')}` |
| Covariates | `{config.get('data', {}).get('covariates', 'N/A')}` |
| LD Reference | `{config.get('association', {}).get('ld_root', 'N/A')}` |

### Preprocessing

{chr(10).join(f'- {item}' for item in preproc_items)}

### Pathway Database

- **Source:** {pathway_source}
- **Custom file:** {pathway_file if pathway_file else 'N/A'}

### Modeling

- **Model type:** {model_type}
- **Parameters:** {model_params_str}

---

## Results Summary

### Pathway Activation Scores (PAS)

- **Number of pathways:** {num_pathways}
- **Number of samples:** {num_samples}

"""

    if top_pathways:
        md_content += """#### Top Pathways by Mean PAS

| Rank | Pathway | Mean PAS | Std PAS |
|------|---------|----------|---------|
"""
        for i, pw in enumerate(top_pathways[:10], 1):
            md_content += f"| {i} | {pw['name']} | {pw['mean']:.4f} | {pw['std']:.4f} |\n"

    if top_associations:
        md_content += """
### Association Results

#### Top Pathways by Absolute Genetic Correlation

| Rank | Pathway | rg | SE | P-value |
|------|---------|-----|-----|---------|
"""
        for i, assoc in enumerate(top_associations[:10], 1):
            md_content += (
                f"| {i} | {assoc.get('pathway', 'N/A')} | "
                f"{assoc.get('rg', 0):.4f} | {assoc.get('se', 0):.4f} | "
                f"{assoc.get('pvalue', 1):.2e} |\n"
            )

    if qc_metrics:
        md_content += """
### QC Metrics

"""
        for metric, value in qc_metrics.items():
            md_content += f"- **{metric}:** {value}\n"

    md_content += f"""
---

## Output Files

All outputs are stored in: `{exp_dir}`

- `config.yaml` - Configuration used for this run
- `experiment_metadata.json` - Metadata including timestamp and git commit
- `pas_matrix.csv` - Pathway activation scores matrix
- `pas_weights.csv` - Per-gene weights (if activity-weighted method used)
- `association_results.csv` - Association test results
- `experiment_report.md` - This report
- `experiment_report.html` - HTML version of this report

---

*Generated by PathWAS v0.1.0*
"""

    report_path = exp_dir / "experiment_report.md"
    with open(report_path, "w") as f:
        f.write(md_content)
    logging.info("Generated Markdown report: %s", report_path)

    # Convert to HTML
    generate_html_report(md_content, exp_dir)

    return str(report_path)


def generate_html_report(md_content: str, exp_dir: Path) -> str:
    """Convert Markdown report to HTML.

    Parameters
    ----------
    md_content : str
        Markdown content.
    exp_dir : pathlib.Path
        Experiment directory.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    html_path = exp_dir / "experiment_report.html"

    if markdown is None:
        # Fallback: wrap in basic HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PathWAS Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
        code {{ background: #f4f4f4; padding: 2px 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f4f4f4; }}
        h1, h2, h3 {{ color: #333; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 2em 0; }}
    </style>
</head>
<body>
<pre>{md_content}</pre>
</body>
</html>
"""
    else:
        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code"],
        )
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PathWAS Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; border-radius: 4px; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 2px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4a90d9; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        h1 {{ color: #2c5aa0; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #3d7ac7; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #555; }}
        hr {{ border: none; border-top: 2px solid #4a90d9; margin: 2em 0; }}
        a {{ color: #4a90d9; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html_content)
    logging.info("Generated HTML report: %s", html_path)
    return str(html_path)


# ----------------------------- Data Processing ------------------------------ #

def preprocess_expression(
    expr_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Apply preprocessing steps to expression data.

    Parameters
    ----------
    expr_df : pandas.DataFrame
        Raw expression matrix (samples x genes).
    config : dict
        Preprocessing configuration.

    Returns
    -------
    pandas.DataFrame
        Preprocessed expression matrix.
    """
    preproc = config.get("preprocessing", {})

    # Normalization
    norm = preproc.get("normalization", "none").lower()
    if norm == "cpm":
        # Counts per million
        logging.info("Applying CPM normalization")
        expr_df = expr_df.div(expr_df.sum(axis=1), axis=0) * 1e6
    elif norm == "tpm":
        # TPM assumes gene lengths are embedded or not needed for relative comparison
        logging.info("Applying TPM normalization (approximate)")
        expr_df = expr_df.div(expr_df.sum(axis=1), axis=0) * 1e6
    elif norm != "none":
        logging.warning("Unknown normalization method: %s, skipping", norm)

    # Log transform
    if preproc.get("log_transform", False):
        logging.info("Applying log2(x + 1) transformation")
        expr_df = np.log2(expr_df + 1)

    # Z-score genes
    if preproc.get("zscore_genes", False):
        logging.info("Z-scoring expression across samples for each gene")
        expr_df = (expr_df - expr_df.mean(axis=0)) / expr_df.std(axis=0, ddof=0)
        expr_df = expr_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return expr_df


def load_pathways(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Load pathway definitions based on configuration.

    Parameters
    ----------
    config : dict
        Pathways configuration.

    Returns
    -------
    dict
        Mapping of pathway names to gene lists.
    """
    pathways_cfg = config.get("pathways", {})
    source = pathways_cfg.get("source", "hallmark").lower()
    custom_file = pathways_cfg.get("file")

    if source == "custom" and custom_file:
        logging.info("Loading custom pathways from %s", custom_file)
        with open(custom_file) as f:
            pathways = json.load(f)
    else:
        # Map source names to MSigDB library names
        msigdb_mapping = {
            "hallmark": "HALLMARK",
            "kegg": "KEGG_2021_Human",
            "reactome": "Reactome_2022",
            "wikipathways": "WikiPathways_2021_Human",
            "go": "GO_Biological_Process_2021",
            "go_bp": "GO_Biological_Process_2021",
            "go_mf": "GO_Molecular_Function_2021",
            "go_cc": "GO_Cellular_Component_2021",
        }
        library = msigdb_mapping.get(source, source.upper())
        logging.info("Loading pathways from MSigDB: %s", library)
        pathways = load_msigdb_library(library)

    # Convert gene IDs if needed
    pathways = {pw: convert_gene_list(genes) for pw, genes in pathways.items()}
    logging.info("Loaded %d pathways", len(pathways))
    return pathways


# ----------------------------- Main Experiment Runner ----------------------- #

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a complete PathWAS experiment.

    Parameters
    ----------
    config : dict
        Complete experiment configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing all experiment results.
    """
    # Merge with defaults
    full_config = merge_config(DEFAULT_CONFIG.copy(), config)

    # Setup experiment directory
    base_dir = full_config.get("output", {}).get("base_dir", DEFAULT_EXPERIMENTS_DIR)
    exp_dir = setup_experiment_dir(full_config, base_dir)

    # Save configuration
    config_path = exp_dir / "config.yaml"
    save_config(full_config, config_path)
    logging.info("Saved configuration to %s", config_path)

    # Save metadata
    metadata = save_metadata(full_config, exp_dir)

    # Initialize results dictionary
    results: Dict[str, Any] = {
        "pas_summary": {},
        "top_pathways": [],
        "association": {},
        "qc": {},
    }

    try:
        # Load and preprocess expression data
        expr_path = full_config.get("data", {}).get("expression")
        if expr_path and os.path.exists(expr_path):
            logging.info("Loading expression data from %s", expr_path)
            expr_df = pd.read_csv(expr_path, index_col=0)
            expr_df = convert_gene_ids(expr_df)
            expr_df = preprocess_expression(expr_df, full_config)

            # Save preprocessed expression
            if full_config.get("output", {}).get("save_intermediate", True):
                preprocessed_path = exp_dir / "expression_preprocessed.csv"
                expr_df.to_csv(preprocessed_path)
                logging.info("Saved preprocessed expression to %s", preprocessed_path)

            # Load pathways
            pathways = load_pathways(full_config)

            # Compute PAS
            logging.info("Computing pathway activation scores")
            pas_df, weights = compute_pas(
                expr_df,
                pathways,
                method="activity_weighted",
                corr_method="bicor",
                normalize_samples=True,
                normalize_pathways=False,
            )

            # Save PAS matrix
            pas_path = exp_dir / "pas_matrix.csv"
            pas_df.to_csv(pas_path)
            logging.info("Saved PAS matrix to %s", pas_path)

            # Save weights if available
            if weights:
                weights_df = pd.DataFrame(weights).T
                weights_path = exp_dir / "pas_weights.csv"
                weights_df.to_csv(weights_path)
                logging.info("Saved PAS weights to %s", weights_path)

            # Update results
            results["pas_summary"] = {
                "num_pathways": pas_df.shape[1],
                "num_samples": pas_df.shape[0],
            }

            # Compute top pathways by mean PAS
            if not pas_df.empty:
                pas_means = pas_df.mean(axis=0).sort_values(ascending=False)
                pas_stds = pas_df.std(axis=0)
                top_pathways = []
                for pw in pas_means.index[:20]:
                    top_pathways.append({
                        "name": pw,
                        "mean": float(pas_means[pw]),
                        "std": float(pas_stds[pw]),
                    })
                results["top_pathways"] = top_pathways

            # PAS differential analysis (if group labels provided)
            group_labels_path = full_config.get("data", {}).get("group_labels")
            pas_analysis_config = full_config.get("pas_analysis", {})
            group_labels = None

            if group_labels_path and os.path.exists(group_labels_path):
                logging.info("Loading group labels from %s", group_labels_path)
                group_df = pd.read_csv(group_labels_path, index_col=0)
                group_col = pas_analysis_config.get("group_column", "group")

                if group_col in group_df.columns:
                    group_labels = group_df[group_col]
                elif len(group_df.columns) == 1:
                    group_labels = group_df.iloc[:, 0]
                else:
                    logging.warning("Could not find group column '%s' in group labels file", group_col)

            if group_labels is not None and pas_analysis_config.get("enabled", False):
                logging.info("Running PAS differential analysis")

                # Build test configuration
                test_method_map = {
                    "ttest": TestMethod.TTEST,
                    "welch": TestMethod.WELCH,
                    "mann_whitney": TestMethod.MANN_WHITNEY,
                    "anova": TestMethod.ANOVA,
                    "kruskal": TestMethod.KRUSKAL,
                    "linear": TestMethod.LINEAR,
                }
                correction_map = {
                    "none": MultipleTestingCorrection.NONE,
                    "bonferroni": MultipleTestingCorrection.BONFERRONI,
                    "fdr_bh": MultipleTestingCorrection.FDR_BH,
                    "fdr_by": MultipleTestingCorrection.FDR_BY,
                }

                test_config = PASTestConfig(
                    method=test_method_map.get(
                        pas_analysis_config.get("test_method", "ttest").lower(),
                        TestMethod.TTEST
                    ),
                    alpha=pas_analysis_config.get("alpha", 0.05),
                    correction=correction_map.get(
                        pas_analysis_config.get("correction", "fdr_bh").lower(),
                        MultipleTestingCorrection.FDR_BH
                    ),
                )

                # Load covariates if needed
                cov_df = None
                if pas_analysis_config.get("adjust_covariates", False):
                    cov_path = full_config.get("data", {}).get("covariates")
                    if cov_path and os.path.exists(cov_path):
                        logging.info("Loading covariates for adjusted analysis")
                        cov_df = pd.read_csv(cov_path, index_col=0)

                # Run differential analysis
                try:
                    diff_results = test_pas_difference(
                        pas_df,
                        group_labels,
                        config=test_config,
                        covariates=cov_df,
                    )

                    # Save differential analysis results
                    diff_path = exp_dir / "pas_differential_results.csv"
                    diff_results.to_csv(diff_path, index=False)
                    logging.info("Saved PAS differential results to %s", diff_path)

                    # Update results
                    n_sig = diff_results["significant"].sum() if "significant" in diff_results.columns else 0
                    results["pas_analysis"] = {
                        "num_tested": len(diff_results),
                        "num_significant": int(n_sig),
                        "method": test_config.method.value,
                        "alpha": test_config.alpha,
                        "correction": test_config.correction.value,
                    }

                    # Get top differential pathways
                    if not diff_results.empty:
                        top_diff = diff_results.nsmallest(10, "pvalue_adj")
                        results["pas_analysis"]["top_differential"] = top_diff.to_dict("records")

                    # Print summary
                    logging.info(summarize_test_results(diff_results))

                except Exception as e:
                    logging.error("PAS differential analysis failed: %s", e)
                    results["pas_analysis"] = {"error": str(e)}

            # Visualization
            viz_config = full_config.get("visualization", {})
            if viz_config.get("enabled", True) and not pas_df.empty:
                logging.info("Generating visualizations")
                figures_dir = exp_dir / "figures"
                figures_dir.mkdir(exist_ok=True)

                try:
                    # Heatmap
                    heatmap_config = viz_config.get("heatmap", {})
                    if heatmap_config.get("enabled", True):
                        cluster_method_map = {
                            "average": ClusterMethod.AVERAGE,
                            "complete": ClusterMethod.COMPLETE,
                            "single": ClusterMethod.SINGLE,
                            "ward": ClusterMethod.WARD,
                        }
                        hm_cfg = HeatmapConfig(
                            title=heatmap_config.get("title", "PAS Heatmap"),
                            cluster_rows=heatmap_config.get("cluster_rows", True),
                            cluster_cols=heatmap_config.get("cluster_cols", True),
                            cluster_method=cluster_method_map.get(
                                heatmap_config.get("cluster_method", "average").lower(),
                                ClusterMethod.AVERAGE
                            ),
                            cmap=heatmap_config.get("cmap", "RdBu_r"),
                            show_row_labels=heatmap_config.get("show_row_labels", True),
                            show_col_labels=heatmap_config.get("show_col_labels", False),
                        )
                        plot_pas_heatmap(
                            pas_df,
                            config=hm_cfg,
                            group_labels=group_labels,
                            output_path=figures_dir / "pas_heatmap.png",
                        )

                    # Boxplots for top pathways
                    boxplot_config = viz_config.get("boxplot", {})
                    if boxplot_config.get("enabled", True) and group_labels is not None:
                        top_n = boxplot_config.get("top_n", 10)

                        # Get top pathways (from differential analysis or by variance)
                        if "pas_analysis" in results and "top_differential" in results.get("pas_analysis", {}):
                            top_pathways_list = [
                                p["pathway"] for p in results["pas_analysis"]["top_differential"][:top_n]
                            ]
                        else:
                            top_pathways_list = pas_df.var().nlargest(top_n).index.tolist()

                        bp_cfg = BoxplotConfig(
                            title=boxplot_config.get("title", "PAS Distribution by Group"),
                            show_points=boxplot_config.get("show_points", True),
                            show_violin=boxplot_config.get("show_violin", False),
                            palette=boxplot_config.get("palette", "Set2"),
                        )
                        plot_pas_boxplot_multi(
                            pas_df,
                            group_labels,
                            top_pathways_list,
                            config=bp_cfg,
                            output_path=figures_dir / "pas_boxplots.png",
                        )

                    # Volcano plot
                    volcano_config = viz_config.get("volcano", {})
                    if volcano_config.get("enabled", True) and "pas_analysis" in results:
                        diff_path = exp_dir / "pas_differential_results.csv"
                        if diff_path.exists():
                            diff_results = pd.read_csv(diff_path)
                            plot_pas_volcano(
                                diff_results,
                                title=volcano_config.get("title", "PAS Differential Analysis"),
                                alpha=volcano_config.get("alpha", 0.05),
                                effect_threshold=volcano_config.get("effect_threshold", 0.5),
                                top_n_labels=volcano_config.get("top_n_labels", 10),
                                output_path=figures_dir / "pas_volcano.png",
                            )

                    results["figures"] = {
                        "directory": str(figures_dir),
                        "generated": list(figures_dir.glob("*.png")),
                    }
                    logging.info("Saved visualizations to %s", figures_dir)

                except ImportError as e:
                    logging.warning("Visualization dependencies not available: %s", e)
                except Exception as e:
                    logging.error("Visualization failed: %s", e)

            # Genotype-based association (if genotypes provided)
            geno_path = full_config.get("data", {}).get("genotypes")
            if geno_path and os.path.exists(geno_path):
                logging.info("Loading genotype data from %s", geno_path)
                # For now, support CSV format; PLINK/VCF would need specific loaders
                if geno_path.endswith(".csv"):
                    geno_df = pd.read_csv(geno_path, index_col=0)

                    # Run association test
                    logging.info("Running association test")
                    variants = aggregate_variants(geno_df, pathways)
                    assoc_results = association_test(pas_df, variants)

                    # Save association results
                    assoc_path = exp_dir / "association_results.csv"
                    assoc_results.to_csv(assoc_path, index=False)
                    logging.info("Saved association results to %s", assoc_path)

                    # Update results
                    if not assoc_results.empty and "pathway" in assoc_results.columns:
                        top_assoc = assoc_results.head(10).to_dict("records")
                        results["association"]["top_pathways"] = top_assoc
                else:
                    logging.warning(
                        "Genotype file format not directly supported: %s. "
                        "Use PLINK tools to convert to CSV first.",
                        geno_path,
                    )

        else:
            logging.warning("Expression data not provided or not found")

        # Add QC metrics
        results["qc"]["status"] = "completed"
        results["qc"]["warnings"] = []

    except Exception as e:
        logging.error("Experiment failed: %s", e)
        results["qc"]["status"] = "failed"
        results["qc"]["error"] = str(e)
        raise

    finally:
        # Generate report even if experiment partially failed
        try:
            generate_markdown_report(full_config, results, exp_dir)
        except Exception as e:
            logging.error("Failed to generate report: %s", e)

    logging.info("Experiment completed: %s", exp_dir)
    return {
        "experiment_dir": str(exp_dir),
        "config": full_config,
        "results": results,
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    list
        List of validation error messages.
    """
    errors = []

    data = config.get("data", {})
    expression = data.get("expression")
    if not expression:
        errors.append("Expression data path is required (--expression or data.expression in config)")

    pathways = config.get("pathways", {})
    source = pathways.get("source", "").lower()
    if source == "custom" and not pathways.get("file"):
        errors.append("Custom pathway source requires --pathways-file")

    modeling = config.get("modeling", {})
    model_type = modeling.get("model_type", "").lower()
    valid_models = ["ridge", "bayes_mixture", "elastic_net", "lasso"]
    if model_type and model_type not in valid_models:
        errors.append(f"Invalid model type: {model_type}. Valid options: {valid_models}")

    return errors
