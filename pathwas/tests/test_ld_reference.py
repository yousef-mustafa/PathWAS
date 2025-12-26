"""Tests for LD reference panel loading."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from pathwas.ld.reference import (
    load_snp_manifest,
    load_block_definitions,
    assign_snps_to_blocks,
    load_ld_reference,
    load_block_ld_matrix,
    load_ld_for_snps,
    align_ld_to_vectors,
    save_block_ld,
    LDReference,
)


class TestLoadSnpManifest:
    """Tests for load_snp_manifest function."""

    def test_load_valid_manifest(self, tmp_path):
        """Test loading a valid manifest file."""
        manifest_path = tmp_path / "snp_manifest.tsv"
        manifest_path.write_text(
            "snp_id\tchrom\tpos\teffect_allele\tother_allele\tmaf\n"
            "rs1\t1\t1000\tA\tG\t0.25\n"
            "rs2\t1\t2000\tC\tT\t0.10\n"
        )

        df = load_snp_manifest(manifest_path)

        assert len(df) == 2
        assert list(df.columns) == ["snp_id", "chrom", "pos", "effect_allele", "other_allele", "maf"]
        assert df["maf"].dtype == float

    def test_missing_columns_raises(self, tmp_path):
        """Test that missing columns raise error."""
        manifest_path = tmp_path / "snp_manifest.tsv"
        manifest_path.write_text(
            "snp_id\tchrom\tpos\n"
            "rs1\t1\t1000\n"
        )

        with pytest.raises(ValueError, match="missing columns"):
            load_snp_manifest(manifest_path)

    def test_file_not_found(self, tmp_path):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_snp_manifest(tmp_path / "nonexistent.tsv")


class TestLoadBlockDefinitions:
    """Tests for load_block_definitions function."""

    def test_load_valid_blocks(self, tmp_path):
        """Test loading valid block definitions."""
        block_path = tmp_path / "block_definitions.bed"
        block_path.write_text(
            "1\t0\t1000000\tblock_1\n"
            "1\t1000000\t2000000\tblock_2\n"
        )

        df = load_block_definitions(block_path)

        assert len(df) == 2
        assert "block_id" in df.columns
        assert df["start"].dtype == int
        assert df["end"].dtype == int

    def test_insufficient_columns(self, tmp_path):
        """Test that fewer than 4 columns raises error."""
        block_path = tmp_path / "block_definitions.bed"
        block_path.write_text("1\t0\t1000000\n")  # Missing block_id

        with pytest.raises(ValueError, match="at least 4 columns"):
            load_block_definitions(block_path)


class TestAssignSnpsToBlocks:
    """Tests for assign_snps_to_blocks function."""

    def test_correct_assignment(self):
        """Test that SNPs are correctly assigned to blocks."""
        snp_manifest = pd.DataFrame({
            "snp_id": ["rs1", "rs2", "rs3", "rs4"],
            "chrom": ["1", "1", "1", "2"],
            "pos": [500, 1500, 2500, 500],
        })

        block_definitions = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "start": [0, 1000, 0],
            "end": [1000, 2000, 1000],
            "block_id": ["b1", "b2", "b3"],
        })

        mapping = assign_snps_to_blocks(snp_manifest, block_definitions)

        assert mapping["rs1"] == "b1"  # 500 in [0, 1000)
        assert mapping["rs2"] == "b2"  # 1500 in [1000, 2000)
        assert mapping["rs4"] == "b3"  # chr2
        assert "rs3" not in mapping  # 2500 not in any block


class TestLDReferenceIntegration:
    """Integration tests for LD reference loading."""

    @pytest.fixture
    def fake_ld_resource(self, tmp_path):
        """Create a fake LD resource directory."""
        ancestry = "EUR"
        ancestry_path = tmp_path / ancestry
        ancestry_path.mkdir()

        # Create SNP manifest
        manifest_path = ancestry_path / "snp_manifest.tsv"
        manifest_path.write_text(
            "snp_id\tchrom\tpos\teffect_allele\tother_allele\tmaf\n"
            "rs1\t1\t500\tA\tG\t0.25\n"
            "rs2\t1\t1500\tC\tT\t0.10\n"
            "rs3\t1\t2500\tG\tA\t0.30\n"
            "rs4\t2\t500\tT\tC\t0.15\n"
        )

        # Create block definitions
        block_path = ancestry_path / "block_definitions.bed"
        block_path.write_text(
            "1\t0\t1000\tblock_1\n"
            "1\t1000\t2000\tblock_2\n"
            "1\t2000\t3000\tblock_3\n"
            "2\t0\t1000\tblock_4\n"
        )

        # Create blocks directory and LD matrices
        blocks_dir = ancestry_path / "blocks"
        blocks_dir.mkdir()

        # Block 1: rs1 only
        R1 = np.array([[1.0]])
        np.savez(blocks_dir / "block_block_1.npz", ld_matrix=R1, snp_ids=np.array(["rs1"]))

        # Block 2: rs2 only
        R2 = np.array([[1.0]])
        np.savez(blocks_dir / "block_block_2.npz", ld_matrix=R2, snp_ids=np.array(["rs2"]))

        # Block 3: rs3 only
        R3 = np.array([[1.0]])
        np.savez(blocks_dir / "block_block_3.npz", ld_matrix=R3, snp_ids=np.array(["rs3"]))

        # Block 4: rs4 only
        R4 = np.array([[1.0]])
        np.savez(blocks_dir / "block_block_4.npz", ld_matrix=R4, snp_ids=np.array(["rs4"]))

        return tmp_path, ancestry

    def test_load_ld_reference(self, fake_ld_resource):
        """Test loading complete LD reference."""
        root_path, ancestry = fake_ld_resource

        ld_ref = load_ld_reference(root_path, ancestry)

        assert isinstance(ld_ref, LDReference)
        assert ld_ref.ancestry == ancestry
        assert ld_ref.n_snps == 4
        assert ld_ref.n_blocks == 4
        assert len(ld_ref.snp_to_block) == 4

    def test_get_maf(self, fake_ld_resource):
        """Test getting MAF for a SNP."""
        root_path, ancestry = fake_ld_resource
        ld_ref = load_ld_reference(root_path, ancestry)

        maf = ld_ref.get_maf("rs1")
        assert maf == 0.25

        maf_none = ld_ref.get_maf("rs_nonexistent")
        assert maf_none is None

    def test_load_block_ld_matrix(self, fake_ld_resource):
        """Test loading LD matrix for a block."""
        root_path, ancestry = fake_ld_resource
        ld_ref = load_ld_reference(root_path, ancestry)

        R, snp_ids = load_block_ld_matrix(ld_ref, "block_1")

        assert R.shape == (1, 1)
        assert snp_ids == ["rs1"]

    def test_missing_ancestry_raises(self, fake_ld_resource):
        """Test that missing ancestry directory raises error."""
        root_path, _ = fake_ld_resource

        with pytest.raises(FileNotFoundError):
            load_ld_reference(root_path, "AFR")


class TestSaveBlockLd:
    """Tests for save_block_ld function."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading block LD."""
        block_id = "test_block"
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        snp_ids = ["rs1", "rs2"]

        save_block_ld(tmp_path, block_id, R, snp_ids)

        # Load it back
        data = np.load(tmp_path / f"block_{block_id}.npz", allow_pickle=True)
        np.testing.assert_allclose(data["ld_matrix"], R)
        assert list(data["snp_ids"]) == snp_ids


class TestAlignLdToVectors:
    """Tests for align_ld_to_vectors function."""

    def test_correct_alignment(self):
        """Test that vectors are correctly aligned to LD matrix."""
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        ld_snp_ids = ["rs1", "rs2"]

        beta_p = pd.Series([0.3, 0.5], index=["rs2", "rs1"])  # Different order
        gamma = pd.Series([0.2, 0.4], index=["rs1", "rs2"])

        R_aligned, beta_aligned, gamma_aligned = align_ld_to_vectors(
            R, ld_snp_ids, beta_p, gamma
        )

        # Should be ordered as ld_snp_ids
        assert list(beta_aligned.index) == ["rs1", "rs2"]
        assert list(gamma_aligned.index) == ["rs1", "rs2"]
        assert beta_aligned["rs1"] == 0.5
        assert beta_aligned["rs2"] == 0.3

    def test_no_common_snps_raises(self):
        """Test that no common SNPs raises error."""
        R = np.array([[1.0]])
        ld_snp_ids = ["rs1"]

        beta_p = pd.Series([0.5], index=["rs2"])
        gamma = pd.Series([0.3], index=["rs3"])

        with pytest.raises(ValueError, match="No common SNPs"):
            align_ld_to_vectors(R, ld_snp_ids, beta_p, gamma)
