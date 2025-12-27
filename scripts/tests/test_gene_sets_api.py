#!/usr/bin/env python
"""
Test pathwas.io.gene_sets module API.

This script tests gene set loading and manipulation including:
- GMT file parsing
- JSON pathway loading
- Pathway filtering by size
- Gene intersection with expression data

Usage:
    python scripts/tests/test_gene_sets_api.py
"""

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathwas.io.gene_sets import (
    load_gene_sets,
    load_gmt,
    filter_gene_sets_by_size,
    intersect_gene_sets_with_genes,
    save_gene_sets_gmt,
    gene_sets_to_dataframe,
)


def test_load_gmt():
    """Test loading GMT format files."""
    print("Testing GMT loading...")

    # Create a temporary GMT file
    gmt_content = """PATHWAY_A\tDescription A\tGENE1\tGENE2\tGENE3\tGENE4\tGENE5
PATHWAY_B\tDescription B\tGENE3\tGENE4\tGENE5\tGENE6\tGENE7
PATHWAY_C\tDescription C\tGENE8\tGENE9
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.gmt', delete=False) as f:
        f.write(gmt_content)
        gmt_path = Path(f.name)

    try:
        # Load GMT
        pathways = load_gmt(gmt_path)

        # Verify structure
        assert isinstance(pathways, dict), "Should return dict"
        assert len(pathways) == 3, f"Expected 3 pathways, got {len(pathways)}"
        print(f"  - Loaded {len(pathways)} pathways: PASS")

        # Verify pathway contents
        assert 'PATHWAY_A' in pathways, "Missing PATHWAY_A"
        assert pathways['PATHWAY_A'] == {'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'}
        print("  - PATHWAY_A genes correct: PASS")

        # Check overlapping genes handled
        assert 'GENE3' in pathways['PATHWAY_A'] and 'GENE3' in pathways['PATHWAY_B']
        print("  - Overlapping genes preserved: PASS")

        # Check small pathway
        assert pathways['PATHWAY_C'] == {'GENE8', 'GENE9'}
        print("  - Small pathway correct: PASS")

    finally:
        gmt_path.unlink()

    print("  [PASS] GMT loading")
    return True


def test_load_json_pathways():
    """Test loading JSON format pathway files."""
    print("Testing JSON pathway loading...")

    import json

    # Create temporary JSON file
    pathways_data = {
        'pathway_1': ['GENE1', 'GENE2', 'GENE3'],
        'pathway_2': ['GENE4', 'GENE5'],
        'pathway_3': ['GENE1', 'GENE6', 'GENE7', 'GENE8'],
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pathways_data, f)
        json_path = Path(f.name)

    try:
        # Load using load_gene_sets with custom source
        pathways = load_gene_sets(source='custom', file_path=json_path)

        # Verify structure
        assert len(pathways) == 3, f"Expected 3 pathways, got {len(pathways)}"
        print(f"  - Loaded {len(pathways)} pathways: PASS")

        # Verify contents
        assert 'pathway_1' in pathways
        assert 'GENE1' in pathways['pathway_1']
        print("  - Pathway contents correct: PASS")

    finally:
        json_path.unlink()

    print("  [PASS] JSON loading")
    return True


def test_filter_by_size():
    """Test filtering pathways by size."""
    print("Testing pathway size filtering...")

    pathways = {
        'small': {'G1', 'G2'},  # 2 genes
        'medium': {'G3', 'G4', 'G5', 'G6', 'G7'},  # 5 genes
        'large': {'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17'},  # 10 genes
        'xlarge': set(f'G{i}' for i in range(100, 130)),  # 30 genes
    }

    # Filter to medium size only
    filtered = filter_gene_sets_by_size(pathways, min_genes=3, max_genes=15)

    assert 'small' not in filtered, "Small pathway should be filtered"
    assert 'medium' in filtered, "Medium pathway should be kept"
    assert 'large' in filtered, "Large pathway should be kept"
    assert 'xlarge' not in filtered, "XLarge pathway should be filtered"
    print("  - Size filtering works: PASS")

    # Edge cases
    filtered_min = filter_gene_sets_by_size(pathways, min_genes=5)
    assert 'small' not in filtered_min
    assert 'medium' in filtered_min
    print("  - Min-only filtering: PASS")

    filtered_max = filter_gene_sets_by_size(pathways, max_genes=5)
    assert 'small' in filtered_max
    assert 'medium' in filtered_max
    assert 'large' not in filtered_max
    print("  - Max-only filtering: PASS")

    print("  [PASS] Size filtering")
    return True


def test_intersect_with_genes():
    """Test intersecting pathways with available genes."""
    print("Testing gene intersection...")

    pathways = {
        'pathway_1': {'GENE1', 'GENE2', 'GENE3', 'GENE4'},
        'pathway_2': {'GENE5', 'GENE6', 'GENE7'},
        'pathway_3': {'GENE_NOT_IN_DATA1', 'GENE_NOT_IN_DATA2'},
    }

    # Available genes in expression data
    available_genes = {'GENE1', 'GENE2', 'GENE3', 'GENE5', 'GENE6', 'GENE8'}

    # Intersect
    intersected = intersect_gene_sets_with_genes(pathways, available_genes)

    # pathway_1 should have 3 genes (GENE4 not in available)
    assert intersected['pathway_1'] == {'GENE1', 'GENE2', 'GENE3'}
    print("  - Intersection removes missing genes: PASS")

    # pathway_2 should have 2 genes
    assert intersected['pathway_2'] == {'GENE5', 'GENE6'}
    print("  - Multiple pathways handled: PASS")

    # pathway_3 should be empty (no overlap)
    # Depending on implementation, might be removed or empty
    if 'pathway_3' in intersected:
        assert len(intersected['pathway_3']) == 0
    print("  - Empty pathway handled: PASS")

    # Test with minimum size requirement
    intersected_filtered = intersect_gene_sets_with_genes(
        pathways, available_genes, min_genes=2
    )
    assert 'pathway_1' in intersected_filtered  # Has 3 genes
    assert 'pathway_2' in intersected_filtered  # Has 2 genes
    print("  - Minimum size after intersection: PASS")

    print("  [PASS] Gene intersection")
    return True


def test_save_gmt():
    """Test saving gene sets to GMT format."""
    print("Testing GMT saving...")

    pathways = {
        'PATHWAY_X': {'GENE_A', 'GENE_B', 'GENE_C'},
        'PATHWAY_Y': {'GENE_D', 'GENE_E'},
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.gmt', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Save
        save_gene_sets_gmt(pathways, output_path)

        # Reload and verify
        reloaded = load_gmt(output_path)

        assert len(reloaded) == 2
        assert 'PATHWAY_X' in reloaded
        assert reloaded['PATHWAY_X'] == {'GENE_A', 'GENE_B', 'GENE_C'}
        print("  - Round-trip GMT save/load: PASS")

    finally:
        output_path.unlink()

    print("  [PASS] GMT saving")
    return True


def test_gene_sets_to_dataframe():
    """Test converting gene sets to DataFrame format."""
    print("Testing DataFrame conversion...")

    pathways = {
        'pathway_1': {'GENE1', 'GENE2', 'GENE3'},
        'pathway_2': {'GENE4', 'GENE5'},
    }

    df = gene_sets_to_dataframe(pathways)

    # Verify structure
    assert 'pathway' in df.columns, "Missing pathway column"
    assert 'gene' in df.columns, "Missing gene column"
    print("  - DataFrame columns correct: PASS")

    # Verify row count (3 + 2 = 5 rows)
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    print("  - Row count correct: PASS")

    # Verify all genes present
    all_genes = set(df['gene'])
    expected_genes = {'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'}
    assert all_genes == expected_genes
    print("  - All genes present: PASS")

    print("  [PASS] DataFrame conversion")
    return True


def test_with_synthetic_data():
    """Test with actual synthetic pathway file if available."""
    print("Testing with synthetic data...")

    import json

    json_path = Path(__file__).parent.parent.parent / "data" / "raw" / "pathways.json"
    gmt_path = Path(__file__).parent.parent.parent / "data" / "raw" / "pathways.gmt"
    gene_path = Path(__file__).parent.parent.parent / "data" / "raw" / "gene_info.csv"

    if not json_path.exists():
        print("  [SKIP] Synthetic data not generated yet")
        return None

    # Load JSON pathways
    pathways = load_gene_sets(source='custom', file_path=json_path)
    print(f"  - Loaded {len(pathways)} pathways from JSON")

    # Basic validation
    assert len(pathways) > 0, "No pathways loaded"

    # Check pathway sizes
    sizes = [len(genes) for genes in pathways.values()]
    print(f"  - Pathway sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # Test GMT loading if available
    if gmt_path.exists():
        pathways_gmt = load_gmt(gmt_path)
        assert len(pathways_gmt) == len(pathways), "GMT and JSON pathway counts should match"
        print("  - GMT and JSON match: PASS")

    # Test intersection with expression genes if available
    if gene_path.exists():
        gene_df = pd.read_csv(gene_path)
        if 'symbol' in gene_df.columns:
            available_genes = set(gene_df['symbol'])
        else:
            available_genes = set(gene_df.iloc[:, 0])

        intersected = intersect_gene_sets_with_genes(pathways, available_genes)

        # All pathways should have genes after intersection
        empty_pathways = sum(1 for genes in intersected.values() if len(genes) == 0)
        print(f"  - Pathways with genes after intersection: {len(intersected) - empty_pathways}/{len(intersected)}")

    print("  [PASS] Synthetic data test")
    return True


def main():
    print("=" * 60)
    print("PathWAS Gene Sets Module API Tests")
    print("=" * 60)
    print()

    tests = [
        ("GMT Loading", test_load_gmt),
        ("JSON Loading", test_load_json_pathways),
        ("Size Filtering", test_filter_by_size),
        ("Gene Intersection", test_intersect_with_genes),
        ("GMT Saving", test_save_gmt),
        ("DataFrame Conversion", test_gene_sets_to_dataframe),
        ("Synthetic Data", test_with_synthetic_data),
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
