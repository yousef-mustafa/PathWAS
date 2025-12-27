#!/usr/bin/env python
"""
Test PathWAS CLI integration.

This script tests the CLI commands with synthetic data including:
- pas command for PAS computation
- test command for association testing
- cov command for covariance computation
- Experiment running with config files

Usage:
    python scripts/tests/test_cli_integration.py
"""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def run_command(cmd: list, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_DIR) + ':' + env.get('PYTHONPATH', '')

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        capture_output=capture,
        text=True,
    )

    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    return result


def test_cli_help():
    """Test CLI help command."""
    print("Testing CLI help...")

    result = run_command(['python', '-m', 'pathwas.cli', '--help'], check=False)

    # Should show help (exit 0) or at least run
    if result.returncode != 0:
        # Some CLIs return non-zero for --help
        assert 'usage' in result.stdout.lower() or 'usage' in result.stderr.lower(), \
            "Help should show usage"

    print("  - CLI help works: PASS")
    print("  [PASS] CLI help")
    return True


def test_pas_command():
    """Test the 'pas' CLI command."""
    print("Testing 'pas' command...")

    expr_path = PROJECT_DIR / "data" / "raw" / "expression.csv"
    pathway_path = PROJECT_DIR / "data" / "raw" / "pathways.json"

    if not expr_path.exists() or not pathway_path.exists():
        print("  [SKIP] Synthetic data not generated yet")
        return None

    # Create temp output file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run PAS computation
        cmd = [
            'python', '-m', 'pathwas.cli', 'pas',
            str(expr_path),
            '--pathways', str(pathway_path),
            '--method', 'mean',
            '--out', str(output_path)
        ]

        result = run_command(cmd)
        print(f"  - PAS command executed: PASS")

        # Verify output exists
        assert output_path.exists(), "Output file not created"
        print("  - Output file created: PASS")

        # Load and verify output
        pas_df = pd.read_csv(output_path, index_col=0)
        assert pas_df.shape[0] > 0, "PAS output is empty"
        assert pas_df.shape[1] > 0, "No pathways in output"
        print(f"  - PAS output shape: {pas_df.shape}: PASS")

        # Verify no NaN values
        assert not pas_df.isna().any().any(), "NaN values in PAS output"
        print("  - No NaN values: PASS")

    finally:
        if output_path.exists():
            output_path.unlink()

    print("  [PASS] 'pas' command")
    return True


def test_pas_activity_weighted():
    """Test activity-weighted PAS via CLI."""
    print("Testing activity-weighted PAS via CLI...")

    expr_path = PROJECT_DIR / "data" / "raw" / "expression.csv"
    pathway_path = PROJECT_DIR / "data" / "raw" / "pathways.json"

    if not expr_path.exists() or not pathway_path.exists():
        print("  [SKIP] Synthetic data not generated yet")
        return None

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        output_path = Path(f.name)

    try:
        cmd = [
            'python', '-m', 'pathwas.cli', 'pas',
            str(expr_path),
            '--pathways', str(pathway_path),
            '--method', 'activity_weighted',
            '--corr-method', 'bicor',
            '--normalize-samples',
            '--out', str(output_path)
        ]

        result = run_command(cmd)
        print("  - Activity-weighted command executed: PASS")

        # Verify output
        pas_df = pd.read_csv(output_path, index_col=0)

        # Check normalization was applied (mean ~0)
        col_means = pas_df.mean()
        assert all(abs(col_means) < 0.1), "normalize-samples should center columns"
        print("  - Normalization applied: PASS")

    finally:
        if output_path.exists():
            output_path.unlink()

    print("  [PASS] Activity-weighted PAS")
    return True


def test_experiment_config():
    """Test running experiment with config file."""
    print("Testing experiment with config...")

    config_path = PROJECT_DIR / "configs" / "test_pas_basic.yaml"
    expr_path = PROJECT_DIR / "data" / "raw" / "expression.csv"
    pathway_path = PROJECT_DIR / "data" / "raw" / "pathways.json"

    if not all(p.exists() for p in [config_path, expr_path, pathway_path]):
        print("  [SKIP] Required files not found")
        return None

    try:
        # Import and run experiment
        from pathwas.experiment import run_experiment, load_config

        config = load_config(str(config_path))
        results = run_experiment(config)

        # Verify results
        assert 'experiment_dir' in results, "Missing experiment_dir in results"
        print(f"  - Experiment directory: {results['experiment_dir']}")

        exp_dir = Path(results['experiment_dir'])
        assert exp_dir.exists(), "Experiment directory not created"
        print("  - Experiment directory created: PASS")

        # Check for expected output files
        expected_files = ['config.yaml', 'pas.csv']
        for fname in expected_files:
            fpath = exp_dir / fname
            if fpath.exists():
                print(f"  - {fname} exists: PASS")

    except ImportError as e:
        print(f"  [SKIP] Import error: {e}")
        return None
    except Exception as e:
        print(f"  [FAIL] Experiment failed: {e}")
        return False

    print("  [PASS] Experiment config")
    return True


def test_experiment_differential():
    """Test experiment with differential analysis."""
    print("Testing experiment with differential analysis...")

    config_path = PROJECT_DIR / "configs" / "test_pas_differential.yaml"

    if not config_path.exists():
        print("  [SKIP] Config file not found")
        return None

    try:
        from pathwas.experiment import run_experiment, load_config

        config = load_config(str(config_path))
        results = run_experiment(config)

        exp_dir = Path(results['experiment_dir'])

        # Check for differential analysis outputs
        expected_outputs = ['differential_results.csv', 'pas.csv']
        for fname in expected_outputs:
            fpath = exp_dir / fname
            if fpath.exists():
                print(f"  - {fname} exists: PASS")

        # Check results contain differential analysis
        if 'results' in results and 'differential' in results.get('results', {}):
            diff_results = results['results']['differential']
            print(f"  - Differential analysis in results: {len(diff_results)} pathways")

    except ImportError as e:
        print(f"  [SKIP] Import error: {e}")
        return None
    except Exception as e:
        print(f"  [FAIL] Experiment failed: {e}")
        traceback.print_exc()
        return False

    print("  [PASS] Differential experiment")
    return True


def test_data_generation_scripts():
    """Test that data generation scripts run successfully."""
    print("Testing data generation scripts...")

    scripts = [
        ('generate_synthetic_expression.py', ['--n-samples', '50', '--n-genes', '100', '--seed', '999']),
        ('generate_synthetic_genotypes.py', ['--n-samples', '50', '--n-variants-per-chr', '100', '--chromosomes', '22', '--seed', '999']),
        ('generate_synthetic_pathways.py', ['--n-pathways', '10', '--seed', '999']),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for script_name, extra_args in scripts:
            script_path = PROJECT_DIR / "scripts" / script_name

            if not script_path.exists():
                print(f"  [SKIP] {script_name} not found")
                continue

            try:
                cmd = [
                    'python', str(script_path),
                    '--output-dir', str(tmpdir),
                ] + extra_args

                # For pathways, need gene list
                if 'pathways' in script_name and (tmpdir / 'gene_info.csv').exists():
                    cmd.extend(['--gene-list', str(tmpdir / 'gene_info.csv')])

                # For genotypes, need sample IDs
                if 'genotypes' in script_name and (tmpdir / 'sample_metadata.csv').exists():
                    cmd.extend(['--sample-ids', str(tmpdir / 'sample_metadata.csv')])

                result = run_command(cmd, check=True)
                print(f"  - {script_name}: PASS")

            except Exception as e:
                print(f"  - {script_name}: FAIL ({e})")
                return False

        # Verify files were created
        expected_files = ['expression.csv', 'sample_metadata.csv', 'gene_info.csv']
        for fname in expected_files:
            if (tmpdir / fname).exists():
                print(f"  - {fname} generated: PASS")

    print("  [PASS] Data generation scripts")
    return True


def test_full_pipeline():
    """Test full pipeline from data generation to analysis."""
    print("Testing full pipeline...")

    # Check all required data exists
    required_files = [
        PROJECT_DIR / "data" / "raw" / "expression.csv",
        PROJECT_DIR / "data" / "raw" / "sample_metadata.csv",
        PROJECT_DIR / "data" / "raw" / "pathways.json",
    ]

    if not all(f.exists() for f in required_files):
        print("  [SKIP] Required data files not found")
        return None

    try:
        from pathwas.io.expression import preprocess_expression
        from pathwas.io.gene_sets import load_gene_sets, intersect_gene_sets_with_genes
        from pathwas.pas.pas import compute_pas
        from pathwas.pas.pas_test import test_pas_difference, PASTestConfig, TestMethod

        # Load data
        expr = pd.read_csv(required_files[0], index_col=0)
        metadata = pd.read_csv(required_files[1], index_col=0)
        pathways = load_gene_sets(source='custom', file_path=required_files[2])

        print(f"  - Loaded expression: {expr.shape}")
        print(f"  - Loaded metadata: {len(metadata)} samples")
        print(f"  - Loaded pathways: {len(pathways)}")

        # Preprocess expression
        expr_processed = preprocess_expression(
            expr,
            normalization='cpm',
            log_transform_expr=True,
            zscore=True,
        )
        print(f"  - Preprocessed expression: {expr_processed.shape}")

        # Intersect pathways with available genes
        available_genes = set(expr_processed.columns)
        pathways_filtered = intersect_gene_sets_with_genes(
            pathways, available_genes, min_genes=5
        )
        print(f"  - Filtered pathways: {len(pathways_filtered)}")

        # Compute PAS
        pas, weights = compute_pas(
            expr_processed, pathways_filtered,
            method='activity_weighted',
            normalize_samples=True
        )
        print(f"  - Computed PAS: {pas.shape}")

        # Run differential analysis
        if 'group' in metadata.columns:
            groups = metadata.loc[pas.index, 'group']
            config = PASTestConfig(method=TestMethod.TTEST)
            results = test_pas_difference(pas, groups, config=config)
            n_sig = results['significant'].sum()
            print(f"  - Differential analysis: {n_sig} significant pathways")

        print("  - Full pipeline completed successfully")

    except Exception as e:
        print(f"  [FAIL] Pipeline failed: {e}")
        traceback.print_exc()
        return False

    print("  [PASS] Full pipeline")
    return True


def main():
    print("=" * 60)
    print("PathWAS CLI Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("CLI Help", test_cli_help),
        ("PAS Command", test_pas_command),
        ("Activity-Weighted PAS", test_pas_activity_weighted),
        ("Data Generation Scripts", test_data_generation_scripts),
        ("Experiment Config", test_experiment_config),
        ("Differential Experiment", test_experiment_differential),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = []
    for name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {name}: {status}")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
