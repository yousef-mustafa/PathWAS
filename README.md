# PathWAS: Analysis Framework for Pathway-Wide Association Studies (PWAS)

**PathWAS** is an open-source Python package for conducting **pathway-level analogs of transcriptome-wide association studies (TWAS)**. It enables researchers to model genetically regulated pathway activity and test its association with complex traits using both **individual-level** and **summary-level** data.

---

## Overview

Traditional TWAS focuses on associating genetically predicted **gene expression** with traits of interest. `PathWAS` generalizes this approach to the **pathway level** by:

- Modeling **SNP-to-pathway activation scores (PAS)** using gene expression and eQTL data
- Supporting multiple methods for computing PAS, including a proprietary **activity-weighted sum** method
- Testing for **trait associations** using both individual-level and summary-level data
- Enabling **genetic correlation** testing between pathway activity and phenotypes
- Providing quality control for ancestry and LD mismatch detection

---

## Features

### Core Functionality
- **PAS Computation**: Mean, median, sum, or activity-weighted methods with robust biweight midcorrelation
- **Gene Set Support**: KEGG, GO, MSigDB, and custom gene sets via gseapy
- **SNP-to-PAS Modeling**: Ridge regression with extensible factory for future model types

### Association Analysis
- **Pathway-TWAS**: Summary-level association tests with LD matrix support
- **Genetic Correlation**: True genetic correlation (rg) with jackknife standard errors
- **LDSC Integration**: Genetic correlation estimation via LD Score Regression

### Data Harmonization
- **GWAS/pathQTL Alignment**: Automatic allele matching, flipping, and palindromic SNP filtering
- **LD Reference Support**: Standardized format for external LD panels (e.g., 1000 Genomes)

### Quality Control
- **Ancestry Mismatch Detection**: MAF and LD concordance metrics
- **Reliability Scoring**: HIGH/MODERATE/LOW classification for result interpretation

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yousef-mustafa/PathWAS.git
cd PathWAS

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# For robust correlation (optional)
pip install -e ".[robust]"

# For development (includes pytest)
pip install -e ".[dev]"
```

---

## Project Structure

```text
PathWAS/
├── pathwas/                    # Core package
│   ├── pas/                    # PAS computation methods
│   │   └── pas.py              # Activity-weighted, mean, median, sum methods
│   ├── modeling/               # SNP → PAS modeling
│   │   ├── base.py             # Abstract model interface
│   │   └── ridge.py            # Ridge regression implementation
│   ├── association/            # Trait association & genetic correlation
│   │   ├── pathway_test.py     # Pathway-TWAS tests
│   │   ├── pathway_rg.py       # Genetic correlation with jackknife
│   │   └── genetic_correlation.py  # LDSC integration
│   ├── io/                     # Input/output utilities
│   │   ├── harmonize.py        # GWAS/pathQTL harmonization
│   │   ├── covariance.py       # VCF covariance computation
│   │   └── data_prep.py        # Gene ID conversion, MSigDB loading
│   ├── ld/                     # LD reference panel support
│   │   └── reference.py        # External LD panel loading
│   ├── qc/                     # Quality control
│   │   └── ancestry_mismatch.py # Ancestry/LD mismatch detection
│   ├── tests/                  # Unit tests
│   └── cli.py                  # Command-line interface
├── requirements.txt            # Python dependencies
├── setup.py                    # Package configuration
└── README.md
```

---

## Example Usage

### Step 1: Compute PAS

```python
from pathwas import compute_pas

# Load expression data and pathway definitions
pas_matrix, weights = compute_pas(
    expression_df,
    pathway_dict,
    method="activity_weighted",
    corr_method="bicor"  # Robust biweight midcorrelation
)
```

### Step 2: Build SNP → PAS Models

```python
from pathwas import create_model

# Create and fit a ridge model
model = create_model("ridge", lambda_=0.1)
model.fit(pas_matrix, genotype_matrix, snp_ids, covariates)

# Get SNP weights and metrics
weights_df = model.get_weights()  # Long-format: pathway, snp_id, beta_pas
metrics_df = model.get_metrics()  # Per-pathway: R², n_snps, lambda
```

### Step 3: Harmonize with GWAS

```python
from pathwas import harmonize_weights_gwas

# Align pathQTL weights with GWAS summary statistics
harmonized_weights, harmonized_gwas = harmonize_weights_gwas(
    weights_df,
    gwas_df
)
# Handles allele flipping, drops palindromic/mismatched SNPs
```

### Step 4: Compute Genetic Correlation

```python
from pathwas import jackknife_genetic_correlation, compute_ld_matrix

# Compute LD matrix from reference genotypes
R = compute_ld_matrix(reference_genotypes)

# Estimate genetic correlation with standard errors
result = jackknife_genetic_correlation(
    beta_pas,      # SNP-to-PAS effects
    gamma,         # SNP-to-trait effects
    R,             # LD correlation matrix
    block_ids      # Block assignments for jackknife
)

print(f"rg = {result.rg:.3f} (SE = {result.se:.3f}, p = {result.p:.2e})")
```

### Step 5: Quality Control

```python
from pathwas import run_ancestry_qc

# Check ancestry/LD mismatch between panels
qc_result = run_ancestry_qc(
    maf_training,   # MAF from training panel
    maf_ldref,      # MAF from LD reference
    maf_gwas=maf_gwas  # Optional: MAF from GWAS
)

print(qc_result.summary())
# Reliability: HIGH/MODERATE/LOW
```

### Step 6: Pathway-TWAS (Summary-Level)

```python
from pathwas import test_pathway_twas

# Test pathway association using summary statistics
z_pathway, var_pas = test_pathway_twas(
    snp_weights,
    gwas_zscores,
    ld_matrix
)
```

---

## LD Reference Format

PathWAS supports external LD reference panels with the following structure:

```text
<root_path>/<ancestry>/
├── snp_manifest.tsv         # SNP metadata (snp_id, chrom, pos, alleles, maf)
├── block_definitions.bed    # LD block boundaries
└── blocks/
    ├── block_1.npz          # Per-block LD matrices
    ├── block_2.npz
    └── ...
```

Load with:
```python
from pathwas import load_ld_reference, load_ld_for_snps

ld_ref = load_ld_reference("/path/to/ld", "EUR")
R, snp_order, block_ids = load_ld_for_snps(ld_ref, my_snp_list)
```

---

## Running Tests

```bash
# Run all tests
pytest pathwas/tests/ -v

# Run with coverage
pytest pathwas/tests/ --cov=pathwas --cov-report=term-missing
```

---

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `compute_pas()` | Compute pathway activation scores |
| `create_model()` | Create SNP-to-PAS model (ridge, etc.) |
| `harmonize_weights_gwas()` | Align pathQTL with GWAS |
| `compute_genetic_correlation()` | Compute rg between PAS and trait |
| `jackknife_genetic_correlation()` | rg with jackknife SE |
| `test_pathway_twas()` | Summary-level pathway-TWAS |
| `run_ancestry_qc()` | Ancestry/LD mismatch QC |
| `load_ld_reference()` | Load external LD panel |

### Key Classes

| Class | Description |
|-------|-------------|
| `RidgePathwayModel` | Ridge regression for SNP-to-PAS |
| `ModelConfig` | Model hyperparameters |
| `LDReference` | LD reference panel container |
| `GeneticCorrelationResult` | rg, SE, Z, p-value result |
| `AncestryQCResult` | QC metrics and reliability |

---

## Contributing

We welcome contributions! To get started:

1. Fork the repo
2. Create a feature branch (`git checkout -b my-feature`)
3. Write tests for new functionality
4. Commit and push your changes
5. Open a pull request

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Authors

- **Yousef Mustafa, MS, PhD** (lead developer)

Contributions welcome!

---

## Acknowledgments

This project is inspired by methodologies from:

- PrediXcan / S-PrediXcan
- MAGMA / GSEA
- WGCNA
- LDSC
