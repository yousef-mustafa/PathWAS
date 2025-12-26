"""Tests for gene set loading utilities."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pathwas.io import (
    load_gmt,
    load_tabular_gene_sets,
    save_gene_sets_gmt,
    gene_sets_to_dataframe,
    filter_gene_sets_by_size,
    intersect_gene_sets_with_genes,
)


class TestLoadGmt:
    """Tests for GMT file loading."""

    @pytest.fixture
    def gmt_file(self, tmp_path):
        """Create a temporary GMT file."""
        gmt_content = """PATHWAY_A\tDescription A\tGENE1\tGENE2\tGENE3
PATHWAY_B\tDescription B\tGENE2\tGENE4\tGENE5\tGENE6
PATHWAY_C\thttp://example.com\tGENE7
"""
        gmt_path = tmp_path / "test.gmt"
        gmt_path.write_text(gmt_content)
        return gmt_path

    def test_load_gmt_basic(self, gmt_file):
        gene_sets = load_gmt(gmt_file)

        assert len(gene_sets) == 3
        assert "PATHWAY_A" in gene_sets
        assert "PATHWAY_B" in gene_sets
        assert "PATHWAY_C" in gene_sets

    def test_load_gmt_genes_correct(self, gmt_file):
        gene_sets = load_gmt(gmt_file)

        assert gene_sets["PATHWAY_A"] == {"GENE1", "GENE2", "GENE3"}
        assert gene_sets["PATHWAY_B"] == {"GENE2", "GENE4", "GENE5", "GENE6"}
        assert gene_sets["PATHWAY_C"] == {"GENE7"}

    def test_load_gmt_returns_sets(self, gmt_file):
        gene_sets = load_gmt(gmt_file)

        for pathway, genes in gene_sets.items():
            assert isinstance(genes, set)

    def test_load_gmt_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.gmt"
        empty_file.write_text("")

        gene_sets = load_gmt(empty_file)
        assert gene_sets == {}

    def test_load_gmt_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gmt(tmp_path / "nonexistent.gmt")


class TestLoadTabularGeneSets:
    """Tests for loading gene sets from tabular files."""

    @pytest.fixture
    def tsv_file(self, tmp_path):
        """Create a temporary TSV file."""
        tsv_content = """pathway\tgene
PATHWAY_A\tGENE1
PATHWAY_A\tGENE2
PATHWAY_B\tGENE3
PATHWAY_B\tGENE4
PATHWAY_B\tGENE5
"""
        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text(tsv_content)
        return tsv_path

    def test_load_tsv_basic(self, tsv_file):
        gene_sets = load_tabular_gene_sets(tsv_file)

        assert len(gene_sets) == 2
        assert "PATHWAY_A" in gene_sets
        assert "PATHWAY_B" in gene_sets

    def test_load_tsv_genes_correct(self, tsv_file):
        gene_sets = load_tabular_gene_sets(tsv_file)

        assert gene_sets["PATHWAY_A"] == {"GENE1", "GENE2"}
        assert gene_sets["PATHWAY_B"] == {"GENE3", "GENE4", "GENE5"}

    def test_load_csv_custom_columns(self, tmp_path):
        csv_content = """set_name,gene_id
SET1,A
SET1,B
SET2,C
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        gene_sets = load_tabular_gene_sets(
            csv_path,
            pathway_col="set_name",
            gene_col="gene_id",
        )

        assert gene_sets["SET1"] == {"A", "B"}
        assert gene_sets["SET2"] == {"C"}


class TestSaveGeneSetsGmt:
    """Tests for saving gene sets to GMT format."""

    def test_save_and_reload(self, tmp_path):
        gene_sets = {
            "PATHWAY_A": {"GENE1", "GENE2", "GENE3"},
            "PATHWAY_B": {"GENE4", "GENE5"},
        }

        gmt_path = tmp_path / "output.gmt"
        save_gene_sets_gmt(gene_sets, gmt_path)

        # Reload and verify
        loaded = load_gmt(gmt_path)

        assert len(loaded) == 2
        assert loaded["PATHWAY_A"] == gene_sets["PATHWAY_A"]
        assert loaded["PATHWAY_B"] == gene_sets["PATHWAY_B"]

    def test_save_with_descriptions(self, tmp_path):
        gene_sets = {
            "PATHWAY_A": {"GENE1", "GENE2"},
        }
        descriptions = {
            "PATHWAY_A": "Test pathway description",
        }

        gmt_path = tmp_path / "output.gmt"
        save_gene_sets_gmt(gene_sets, gmt_path, descriptions=descriptions)

        # Read raw file to check description
        content = gmt_path.read_text()
        assert "Test pathway description" in content

    def test_save_empty_gene_sets(self, tmp_path):
        gene_sets = {}

        gmt_path = tmp_path / "empty.gmt"
        save_gene_sets_gmt(gene_sets, gmt_path)

        loaded = load_gmt(gmt_path)
        assert loaded == {}


class TestGeneSetsToDataframe:
    """Tests for converting gene sets to DataFrame."""

    def test_basic_conversion(self):
        gene_sets = {
            "PATHWAY_A": {"GENE1", "GENE2"},
            "PATHWAY_B": {"GENE3"},
        }

        df = gene_sets_to_dataframe(gene_sets)

        assert len(df) == 3
        assert "pathway" in df.columns
        assert "gene" in df.columns

    def test_dataframe_contents(self):
        gene_sets = {
            "PW1": {"G1", "G2"},
        }

        df = gene_sets_to_dataframe(gene_sets)

        assert set(df["pathway"]) == {"PW1"}
        assert set(df["gene"]) == {"G1", "G2"}

    def test_empty_gene_sets(self):
        df = gene_sets_to_dataframe({})
        assert len(df) == 0
        assert "pathway" in df.columns
        assert "gene" in df.columns


class TestFilterGeneSetsBySize:
    """Tests for filtering gene sets by size."""

    @pytest.fixture
    def gene_sets(self):
        return {
            "tiny": {"G1", "G2"},  # 2 genes
            "small": {"G1", "G2", "G3", "G4", "G5"},  # 5 genes
            "medium": set(f"G{i}" for i in range(50)),  # 50 genes
            "large": set(f"G{i}" for i in range(600)),  # 600 genes
        }

    def test_filter_by_min_size(self, gene_sets):
        filtered = filter_gene_sets_by_size(gene_sets, min_size=5, max_size=None)

        assert "tiny" not in filtered
        assert "small" in filtered
        assert "medium" in filtered
        assert "large" in filtered

    def test_filter_by_max_size(self, gene_sets):
        filtered = filter_gene_sets_by_size(gene_sets, min_size=0, max_size=500)

        assert "tiny" in filtered
        assert "small" in filtered
        assert "medium" in filtered
        assert "large" not in filtered

    def test_filter_by_both(self, gene_sets):
        filtered = filter_gene_sets_by_size(gene_sets, min_size=5, max_size=100)

        assert "tiny" not in filtered
        assert "small" in filtered
        assert "medium" in filtered
        assert "large" not in filtered

    def test_no_filter(self, gene_sets):
        filtered = filter_gene_sets_by_size(gene_sets, min_size=0, max_size=None)

        assert len(filtered) == len(gene_sets)


class TestIntersectGeneSetsWithGenes:
    """Tests for intersecting gene sets with available genes."""

    @pytest.fixture
    def gene_sets(self):
        return {
            "PATHWAY_A": {"GENE1", "GENE2", "GENE3"},
            "PATHWAY_B": {"GENE2", "GENE4", "GENE5"},
            "PATHWAY_C": {"GENE6", "GENE7"},  # No overlap with available
        }

    def test_basic_intersection(self, gene_sets):
        available = {"GENE1", "GENE2", "GENE3", "GENE4"}

        intersected = intersect_gene_sets_with_genes(gene_sets, available)

        assert intersected["PATHWAY_A"] == {"GENE1", "GENE2", "GENE3"}
        assert intersected["PATHWAY_B"] == {"GENE2", "GENE4"}
        # PATHWAY_C should be empty or removed
        if "PATHWAY_C" in intersected:
            assert len(intersected["PATHWAY_C"]) == 0

    def test_with_list_input(self, gene_sets):
        available = ["GENE1", "GENE2"]

        intersected = intersect_gene_sets_with_genes(gene_sets, available)

        assert intersected["PATHWAY_A"] == {"GENE1", "GENE2"}

    def test_no_overlap(self):
        gene_sets = {"PW": {"A", "B", "C"}}
        available = {"X", "Y", "Z"}

        intersected = intersect_gene_sets_with_genes(gene_sets, available)

        if "PW" in intersected:
            assert len(intersected["PW"]) == 0

    def test_empty_available(self, gene_sets):
        intersected = intersect_gene_sets_with_genes(gene_sets, set())

        for pathway in intersected:
            assert len(intersected[pathway]) == 0


class TestGeneSetLoadingIntegration:
    """Integration tests for gene set loading workflow."""

    def test_load_filter_save_reload(self, tmp_path):
        """Test complete workflow: load -> filter -> save -> reload."""
        # Create initial GMT
        gmt_content = """SMALL\tdesc\tG1\tG2
MEDIUM\tdesc\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10
LARGE\tdesc\t""" + "\t".join(f"G{i}" for i in range(100))

        input_path = tmp_path / "input.gmt"
        input_path.write_text(gmt_content)

        # Load
        gene_sets = load_gmt(input_path)

        # Filter
        filtered = filter_gene_sets_by_size(gene_sets, min_size=5, max_size=50)

        # Save
        output_path = tmp_path / "output.gmt"
        save_gene_sets_gmt(filtered, output_path)

        # Reload and verify
        reloaded = load_gmt(output_path)

        assert "SMALL" not in reloaded  # Too small
        assert "MEDIUM" in reloaded
        assert "LARGE" not in reloaded  # Too large
