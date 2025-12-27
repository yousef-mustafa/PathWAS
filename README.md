# pathWAS: Analysis Framework for Pathway-Wide Association Studies (PWAS)

**pathWAS** is an open-source Python package for conducting **pathway-level analogs of transcriptome-wide association studies (TWAS)**. It enables researchers to model genetically regulated pathway activity and test its association with complex traits using both **individual-level** and **summary-level** data.

---

## Overview

Traditional TWAS focuses on associating genetically predicted **gene expression** with traits of interest. `pathWAS` generalizes this approach to the **pathway level** by:

- Modeling **SNP-to-pathway activation scores (PAS)** using gene expression and eQTL data
- Supporting multiple methods for computing PAS, including a proprietary **activity-weighted sum** method
- Testing for **trait associations** using both individual-level and summary-level data
- Enabling **genetic correlation** testing between pathway activity and phenotypes via LD Score Regression

---

## Features

- PAS computation using mean, median, sum, or activity-weighted methods
- Support for KEGG, GO, MSigDB, and custom gene sets
- SNP-to-PAS model building using ridge regression or Bayesian mixture models
- Individual-level or summary-based association tests (TWAS-style Z-scores)
- Genetic correlation estimation via LDSC
- LD reference panel setup from 1000 Genomes data
- Config-driven experiment management with automatic documentation
- Modular, testable architecture

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yousef-mustafa/pathWAS.git
cd pathWAS

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Quick Start

```python
import pandas as pd
from pathwas.pas.pas import compute_pas
from pathwas.io.data_prep import load_msigdb_library, convert_gene_ids, convert_gene_list

# Load expression data (samples x genes)
expression = pd.read_csv("expression.csv", index_col=0)
expression = convert_gene_ids(expression)  # Convert to gene symbols

# Load pathways from MSigDB
pathways = load_msigdb_library("KEGG_2021_Human")
pathways = {pw: convert_gene_list(genes) for pw, genes in pathways.items()}

# Compute pathway activation scores
pas_matrix, metadata = compute_pas(
    expression,
    pathways,
    method="activity_weighted",
    corr_method="bicor",
    normalize_samples=True,
)

# Save results
pas_matrix.to_csv("pas_results.csv")
```

---

## Command Line Interface

PathWAS provides a comprehensive CLI for running analyses without writing Python code.

### Available Commands

| Command | Description |
|---------|-------------|
| `pathwas pas` | Compute Pathway Activation Scores from expression data |
| `pathwas test` | Run association tests between PAS and genotypes |
| `pathwas setup-ld` | Download and configure LD reference panels |
| `pathwas ld-prune` | LD prune a VCF file using PLINK |
| `pathwas cov` | Compute variant covariance matrix from VCF |

### PAS Computation

```bash
# Using MSigDB gene sets (requires internet)
pathwas pas expression.csv \
    --msigdb KEGG_2021_Human \
    --method activity_weighted \
    --corr-method bicor \
    --normalize-samples \
    --out pas_results.csv

# Using custom pathway definitions
pathwas pas expression.csv \
    --pathways my_pathways.json \
    --method mean \
    --out pas_results.csv
```

**Options:**
- `--msigdb`: MSigDB library name (e.g., `KEGG_2021_Human`, `Reactome_2022`, `GO_Biological_Process_2023`)
- `--pathways`: Path to JSON file mapping pathway names to gene lists
- `--method`: Aggregation method (`sum`, `mean`, `median`, `activity_weighted`)
- `--corr-method`: Correlation method for activity_weighted (`pearson`, `spearman`, `bicor`)
- `--normalize-samples`: Z-score PAS across samples
- `--normalize-pathways`: Z-score PAS across pathways
- `--out`: Output file path

### Association Testing

```bash
pathwas test pas.csv genotypes.csv pathways.json --out results.csv
```

### LD Reference Setup

```bash
# List available ancestries
pathwas setup-ld --list

# Download European reference (quick test with chr22 only)
pathwas setup-ld --ancestry EUR --chromosomes 22 --output-dir ./ld_reference

# Download full reference (all autosomes)
pathwas setup-ld --ancestry EUR --output-dir ./ld_reference --maf-threshold 0.01
```

**Supported Ancestries:**
- `EUR`: European (503 samples from CEU, TSI, FIN, GBR, IBS)
- `AFR`: African (661 samples from YRI, LWK, GWD, MSL, ESN, ASW, ACB)
- `EAS`: East Asian (504 samples from CHB, JPT, CHS, CDX, KHV)

---

## Configuration Files

PathWAS experiments can be configured via YAML files for reproducibility.

### Example Configuration

```yaml
# experiment_config.yaml
experiment:
  name: ukbb_pathway_analysis_v1
  description: "Pathway analysis of UK Biobank Alzheimer's cohort"
  overwrite: false

data:
  expression: /path/to/expression.csv
  genotypes: /path/to/genotypes.csv
  covariates: /path/to/covariates.csv
  group_labels: /path/to/groups.csv

pathways:
  source: hallmark  # kegg, reactome, wikipathways, go, hallmark, custom
  # file: /path/to/custom.json  # Required if source is 'custom'

preprocessing:
  normalization: cpm  # cpm, tpm, or none
  log_transform: true
  zscore_genes: false

modeling:
  model_type: ridge  # ridge or bayes_mixture
  model_params:
    lambda: 0.1

pas_analysis:
  enabled: true
  test_method: ttest  # ttest, welch, mann_whitney, anova, kruskal, linear
  alpha: 0.05
  correction: fdr_bh  # none, bonferroni, fdr_bh, fdr_by
  adjust_covariates: false

visualization:
  enabled: true
  heatmap:
    enabled: true
    cluster_rows: true
    cluster_cols: true
    cmap: RdBu_r
  boxplot:
    enabled: true
    top_n: 10
  volcano:
    enabled: true
    top_n_labels: 10

output:
  base_dir: experiments
  save_intermediate: true
```

### Configuration Parameters Reference

| Section | Parameter | Type | Description |
|---------|-----------|------|-------------|
| `experiment.name` | string | Experiment directory name (auto-generated if not provided) |
| `experiment.overwrite` | bool | Whether to overwrite existing experiment |
| `data.expression` | path | Gene expression matrix (samples x genes) |
| `data.genotypes` | path | Genotype data (CSV format) |
| `data.covariates` | path | Optional covariate matrix |
| `data.group_labels` | path | Group assignments for differential analysis |
| `pathways.source` | string | Gene set database: `kegg`, `reactome`, `wikipathways`, `go`, `hallmark`, `custom` |
| `modeling.model_type` | string | SNP-to-PAS model: `ridge` or `bayes_mixture` |
| `modeling.model_params.lambda` | float | Ridge regularization strength |
| `pas_analysis.test_method` | string | Statistical test for group comparisons |
| `pas_analysis.correction` | string | Multiple testing correction method |

---

## Python API

### Step 1: Compute PAS

```python
from pathwas.pas.pas import compute_pas

pas_matrix, metadata = compute_pas(
    expression_df,
    pathway_definitions,
    method="activity_weighted",
    corr_method="bicor",
)
```

### Step 2: Build SNP-to-PAS Models

```python
from pathwas.modeling import create_model, RidgePathwayModel

# Using factory function
model = create_model("ridge", lambda_=0.1)
model.fit(pas_matrix, genotype_matrix, snp_ids)

# Or directly
from pathwas.modeling.ridge import RidgePathwayModel
from pathwas.modeling.base import ModelConfig

config = ModelConfig(lambda_=0.1)
model = RidgePathwayModel(config)
model.fit(pas_matrix, genotype_matrix, snp_ids)

# Get weights and metrics
weights = model.get_weights()
metrics = model.get_metrics()
```

### Step 3: Perform Pathway-TWAS (summary-level)

```python
from pathwas.association.pathway_test import test_pathway_twas

z, var_pas = test_pathway_twas(snp_weights, gwas_zscores, ld_matrix)
```

### Step 4: Estimate Genetic Correlation

```python
from pathwas.association.genetic_correlation import run_ldsc_rg

run_ldsc_rg(
    "pas.sumstats.gz",
    "trait.sumstats.gz",
    "ref_ld_chr/",
    "w_ld_chr/",
    "output/pathway_trait_rg",
)
```

---

## Experiment Documentation

PathWAS emphasizes reproducibility. Each experiment is saved with full configuration, metadata, and results.

### The Description Field

When running experiments, use the `description` field to document the scientific context, data provenance, and analysis rationale. This becomes part of the permanent experiment record.

**Example: Documenting a Complex Analysis**

```python
from pathwas.experiment import run_experiment

config = {
    "experiment": {
        "name": "magenta_ad_kegg_v3",
        "description": """
        This experiment contrasts with previous GWAS-based pathway analyses
        of Alzheimer's disease using UK Biobank data.

        EXPRESSION DATA:
        - Source: GTEx v8 brain cortex samples (n=205)
        - Preprocessing: TPM normalized, log2(x+1) transformed
        - Processed by: Nick Wheeler (2024-01-15)

        GENOTYPE DATA:
        - Source: UK Biobank imputed genotypes (v3)
        - QC: MAF > 0.01, HWE p > 1e-6, call rate > 0.95
        - LD pruned: r^2 < 0.2, 500kb window

        HYPOTHESIS:
        We expect immune-related pathways (complement cascade, cytokine signaling)
        to show stronger genetic correlation with AD than metabolic pathways,
        based on recent microglia GWAS findings (Bellenguez et al., 2022).
        """,
        "overwrite": False,
    },
    "data": {
        "expression": "/storage/projects/magentaAD/expression_data/gtex_brain_tpm.csv",
        "genotypes": "/storage/projects/magentaAD/genotypes/ukbb_qc_pruned.csv",
        "covariates": "/storage/projects/magentaAD/covariates/age_sex_pcs.csv",
    },
    "pathways": {
        "source": "kegg",
    },
    # ... rest of config
}

results = run_experiment(config)
```

### What Gets Saved

Each experiment directory contains:
```
experiments/magenta_ad_kegg_v3/
├── config.yaml              # Full configuration (including description)
├── metadata.json            # Timestamps, git commit, environment info
├── pas.csv                  # Pathway activation scores
├── pas_differential_results.csv
├── association_results.csv
├── figures/
│   ├── pas_heatmap.png
│   ├── pas_boxplots.png
│   └── pas_volcano.png
└── report.md                # Auto-generated analysis report
```

This ensures every analysis is fully documented and reproducible months or years later.

---

## LD Reference Setup

PathWAS requires LD (linkage disequilibrium) reference panels for SNP-based pathway analyses. The `setup-ld` command downloads and configures reference data from 1000 Genomes.

### Supported Populations

| Code | Population | Samples |
|------|------------|---------|
| EUR | European (CEU, TSI, FIN, GBR, IBS) | 503 |
| AFR | African (YRI, LWK, GWD, MSL, ESN, ASW, ACB) | 661 |
| EAS | East Asian (CHB, JPT, CHS, CDX, KHV) | 504 |

### Output Structure

```
ld_reference/
└── EUR/
    ├── snp_manifest.tsv      # SNP metadata (ID, chr, pos, alleles, MAF)
    ├── block_definitions.bed  # LD block boundaries
    └── blocks/
        ├── block_0.npz       # LD matrix and SNP IDs per block
        ├── block_1.npz
        └── ...
```

### Python API

```python
from pathwas.ld import setup_ld_reference, SUPPORTED_ANCESTRIES

# View supported populations
print(SUPPORTED_ANCESTRIES)

# Download and configure LD reference
setup_ld_reference(
    ancestry="EUR",
    output_dir="./ld_reference",
    chromosomes=[21, 22],  # Optional: specific chromosomes
    maf_threshold=0.01,    # Minimum MAF filter
)
```

---

## Project Structure

```text
pathWAS/
├── pathwas/                    # Core package
│   ├── __init__.py             # Package exports
│   ├── cli.py                  # Command-line interface
│   ├── experiment.py           # Experiment orchestration
│   ├── pas/                    # PAS computation
│   │   ├── pas.py              # Core PAS methods (mean, sum, activity-weighted)
│   │   ├── pas_test.py         # Statistical testing (t-test, ANOVA, etc.)
│   │   └── pas_viz.py          # Visualization (heatmaps, boxplots, volcano)
│   ├── modeling/               # SNP-to-PAS models
│   │   ├── base.py             # Abstract base class
│   │   ├── ridge.py            # Ridge regression (dual formulation)
│   │   └── bayes_mixture.py    # Bayesian mixture model
│   ├── association/            # Trait association
│   │   ├── pathway_test.py     # Pathway-TWAS Z-scores
│   │   ├── genetic_correlation.py  # LDSC integration
│   │   └── pathway_rg.py       # Pathway genetic correlation
│   ├── io/                     # Input/output utilities
│   │   ├── data_prep.py        # Gene ID conversion, MSigDB loading
│   │   ├── expression.py       # Expression preprocessing (CPM, TPM, filtering)
│   │   ├── gene_sets.py        # Gene set loading (GMT, KEGG, GO, etc.)
│   │   ├── harmonize.py        # GWAS/pathQTL allele harmonization
│   │   ├── covariance.py       # Covariance matrix computation
│   │   └── vcf_processing.py   # VCF/PLINK utilities
│   ├── ld/                     # LD reference panels
│   │   ├── setup.py            # Download 1000 Genomes data
│   │   └── reference.py        # Load and query LD matrices
│   ├── qc/                     # Quality control
│   │   └── ancestry_mismatch.py  # Ancestry/LD mismatch detection
│   ├── logging/                # Logging utilities
│   │   └── logging_util.py     # Centralized logging configuration
│   └── tests/                  # Unit tests
├── configs/                    # Example configuration files
│   └── example_config.yaml
├── Dockerfile                  # Container definition
├── requirements.txt            # Pinned dependencies
├── setup.py                    # Package installation
└── README.md
```

---

## Documentation

Full documentation and tutorials (coming soon).

---

## Contributing

We welcome contributions! To get started:

1. Fork the repo
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit and push your changes
4. Open a pull request

---

## License

Distributed under the MIT License. See LICENSE for details.

---

## Authors

Yousef Mustafa, MS, PhD (lead developer)

Contributions welcome!

---

## Acknowledgments

This project is inspired by methodologies from:

- PrediXcan / S-PrediXcan
- MAGMA / GSEA
- WGCNA
- LDSC
