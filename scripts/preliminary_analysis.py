#!/usr/bin/env python3.11
"""
PathWAS Preliminary Analysis
-----------------------------
Runs a step-by-step test of the PathWAS pipeline using the existing synthetic
data in data/raw/. Checks each module independently so failures are isolated.

Run from the PathWAS root directory:
    python3.11 scripts/preliminary_analysis.py
"""

import sys
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        print(f"  {FAIL} {label}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
section("1. Data Loading")

expr_df = check("Load expression matrix (200 samples × ~360 genes)", lambda:
    pd.read_csv(ROOT / "data/raw/expression.csv", index_col=0)
)
meta_df = check("Load sample metadata", lambda:
    pd.read_csv(ROOT / "data/raw/sample_metadata.csv", index_col=0)
)
pathways = check("Load pathway definitions (JSON)", lambda:
    json.load(open(ROOT / "data/raw/pathways.json"))
)

if expr_df is not None:
    print(f"         Expression shape: {expr_df.shape}")
if meta_df is not None:
    print(f"         Metadata shape:   {meta_df.shape}")
    print(f"         Groups:           {meta_df['group'].value_counts().to_dict()}")
if pathways is not None:
    print(f"         Pathways loaded:  {len(pathways)}")
    first = next(iter(pathways))
    print(f"         Example pathway:  '{first}' ({len(pathways[first])} genes)")


# ------------------------------------------------------------------
# 2. Gene ID conversion
# ------------------------------------------------------------------
section("2. Gene ID Conversion")

from pathwas.io.data_prep import convert_gene_ids

expr_converted = check("Convert gene IDs (HGNC → Ensembl, with fallback)", lambda:
    convert_gene_ids(expr_df.copy()) if expr_df is not None else None
)
if expr_converted is not None:
    print(f"         Columns before: {expr_df.shape[1]}, after: {expr_converted.shape[1]}")


# ------------------------------------------------------------------
# 3. Expression preprocessing
# ------------------------------------------------------------------
section("3. Expression Preprocessing")

from pathwas.io.expression import normalize_library_size

expr_cpm = check("CPM normalization", lambda:
    normalize_library_size(expr_df.copy(), mode="cpm") if expr_df is not None else None
)
if expr_cpm is not None:
    col_sums = expr_cpm.sum(axis=1)
    print(f"         Column sums (should be ~1e6): min={col_sums.min():.0f}, max={col_sums.max():.0f}")

expr_log = check("Log1p transform after CPM", lambda:
    np.log1p(expr_cpm) if expr_cpm is not None else None
)


# ------------------------------------------------------------------
# 4. Pathway Activation Score computation
# ------------------------------------------------------------------
section("4. PAS Computation")

from pathwas.pas.pas import compute_pas

work_expr = expr_log if expr_log is not None else expr_df

pas_mean = check("PAS — mean method", lambda:
    compute_pas(work_expr, pathways, method="mean") if work_expr is not None else None
)

pas_aw = check("PAS — activity_weighted method", lambda:
    compute_pas(work_expr, pathways, method="activity_weighted") if work_expr is not None else None
)

if pas_mean is not None:
    pas_df_mean, weights_mean = pas_mean
    print(f"         PAS matrix shape: {pas_df_mean.shape}")

if pas_aw is not None:
    pas_df_aw, weights_aw = pas_aw
    print(f"         PAS matrix shape: {pas_df_aw.shape}")
    print(f"         PAS value range:  {pas_df_aw.values.min():.3f} – {pas_df_aw.values.max():.3f}")


# ------------------------------------------------------------------
# 5. Differential PAS testing
# ------------------------------------------------------------------
section("5. Differential PAS Testing (case vs control)")

from pathwas.pas.pas_test import (
    TestMethod, MultipleTestingCorrection, PASTestConfig,
    test_pas_difference, summarize_test_results
)

pas_input = pas_df_aw if (pas_aw is not None) else (pas_df_mean if pas_mean is not None else None)

if pas_input is not None and meta_df is not None:
    shared = pas_input.index.intersection(meta_df.index)
    pas_aligned = pas_input.loc[shared]
    groups = meta_df.loc[shared, "group"]
    group_labels = groups  # must be a Series (index-aligned)

    test_config = PASTestConfig(
        method=TestMethod.WELCH,
        alpha=0.05,
        correction=MultipleTestingCorrection.FDR_BH,
    )

    diff_results = check("Welch t-test with FDR-BH correction across all pathways", lambda:
        test_pas_difference(pas_aligned, group_labels, test_config)
    )

    if diff_results is not None:
        sig = diff_results[diff_results["significant"] == True]
        print(f"         Total pathways tested:  {len(diff_results)}")
        print(f"         Significant (FDR<0.05): {len(sig)}")
        if len(sig) > 0:
            top = sig.sort_values("pvalue").iloc[0]
            print(f"         Top pathway:           {top['pathway']}")
            print(f"           p={top['pvalue']:.2e}, FDR={top['pvalue_adj']:.2e}, d={top['effect_size']:.3f}")
        check("summarize_test_results (string report)", lambda:
            print("         " + summarize_test_results(diff_results).replace("\n", "\n         ")) or True
        )
else:
    print(f"  {SKIP} No PAS data available")


# ------------------------------------------------------------------
# 6. Full experiment via config
# ------------------------------------------------------------------
section("6. Config-Driven Experiment (test_pas_differential.yaml)")

from pathwas.experiment import load_config, run_experiment

config_path = ROOT / "configs/test_pas_differential.yaml"

full_results = None
if config_path.exists():
    config = check("Load config YAML", lambda: load_config(str(config_path)))

    if config is not None:
        # Set output to results/ so we don't clutter data/
        config.setdefault("output", {})["base_dir"] = str(ROOT / "results/experiments")
        config["experiment"]["overwrite"] = True

        full_results = check("Run full experiment pipeline", lambda:
            run_experiment(config)
        )

        if full_results is not None:
            out_dir = ROOT / "results/experiments" / config["experiment"]["name"]
            print(f"         Output directory: {out_dir}")
            if out_dir.exists():
                files = list(out_dir.rglob("*"))
                print(f"         Files written: {len(files)}")
                for f in sorted(files)[:10]:
                    print(f"           {f.relative_to(out_dir)}")
else:
    print(f"  {SKIP} Config not found: {config_path}")


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
section("Summary")

steps = [
    ("Data loading",          expr_df is not None and meta_df is not None and pathways is not None),
    ("Gene ID conversion",    expr_converted is not None),
    ("CPM normalization",     expr_cpm is not None),
    ("PAS (mean)",            pas_mean is not None),
    ("PAS (activity_weighted)", pas_aw is not None),
    ("Differential testing",  diff_results is not None if 'diff_results' in dir() else False),
    ("Full experiment",       full_results is not None),
]

for label, ok in steps:
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
