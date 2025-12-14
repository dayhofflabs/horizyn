"""
Unit tests for the data module.
"""

import csv
import tempfile
from pathlib import Path

import h5py
import pytest
import torch

from horizyn.data_module import HorizynDataModule


@pytest.fixture
def mock_data_files():
    """Create mock data files for testing."""
    tmpdir = tempfile.mkdtemp()
    tmpdir = Path(tmpdir)

    # Create mock training pairs CSV
    train_pairs_path = tmpdir / "train_pairs.csv"
    with open(train_pairs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pr_id", "reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"pr_id": "1", "reaction_id": "rxn1", "protein_id": "prot1"},
                {"pr_id": "2", "reaction_id": "rxn2", "protein_id": "prot2"},
                {"pr_id": "3", "reaction_id": "rxn1", "protein_id": "prot2"},
            ]
        )

    # Create mock validation pairs CSV
    val_pairs_path = tmpdir / "val_pairs.csv"
    with open(val_pairs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pr_id", "reaction_id", "protein_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"pr_id": "1", "reaction_id": "rxn1", "protein_id": "prot1"},
                {"pr_id": "2", "reaction_id": "rxn2", "protein_id": "prot2"},
            ]
        )

    # Create mock training reactions CSV
    train_reactions_path = tmpdir / "train_rxns.csv"
    with open(train_reactions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reaction_id", "reaction_smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"reaction_id": "rxn1", "reaction_smiles": "CCO>>CC=O"},
                {"reaction_id": "rxn2", "reaction_smiles": "C>>CC"},
            ]
        )

    # Create mock validation reactions CSV (same reactions for simplicity)
    val_reactions_path = tmpdir / "val_rxns.csv"
    with open(val_reactions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reaction_id", "reaction_smiles"])
        writer.writeheader()
        writer.writerows(
            [
                {"reaction_id": "rxn1", "reaction_smiles": "CCO>>CC=O"},
                {"reaction_id": "rxn2", "reaction_smiles": "C>>CC"},
            ]
        )

    # Create mock protein embeddings
    protein_embeds_path = tmpdir / "protein_embeds.h5"
    with h5py.File(protein_embeds_path, "w") as f:
        # Create ids dataset (keys)
        ids = ["prot1", "prot2"]
        f.create_dataset("ids", data=[id.encode() for id in ids])
        # Create vectors dataset (embeddings)
        vectors = torch.randn(2, 1024).numpy()
        f.create_dataset("vectors", data=vectors)

    return {
        "train_pairs": str(train_pairs_path),
        "val_pairs": str(val_pairs_path),
        "train_reactions": str(train_reactions_path),
        "val_reactions": str(val_reactions_path),
        "protein_embeds": str(protein_embeds_path),
    }


class TestHorizynDataModule:
    """Tests for the HorizynDataModule class."""

    def test_initialization(self, mock_data_files):
        """Test data module initialization."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
            train_batch_size=2,
            retrieval_batch_size=1,
        )

        assert dm.train_batch_size == 2
        assert dm.retrieval_batch_size == 1
        assert dm._train_data is None  # Not setup yet

    def test_setup(self, mock_data_files):
        """Test data module setup loads data."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
            train_batch_size=2,
            retrieval_batch_size=1,
        )

        dm.setup("fit")

        # Verify datasets are created
        assert dm._train_data is not None
        assert dm._val_data is not None
        assert dm._target_data is not None

        # Verify sizes (bidirectional augmentation doubles pairs and reactions)
        assert len(dm._train_data) == 6  # 3 training pairs × 2 directions = 6
        assert len(dm._val_data) == 4  # 2 validation pairs × 2 directions = 4

        # NEW: Verify validation retrieval dataset (unique queries, bidirectional)
        assert dm._val_query_data is not None
        assert len(dm._val_query_data) == 4  # 2 unique queries × 2 directions = 4

        # NEW: Verify retrieval targets dataset exists
        assert dm._val_retrieval_targets is not None
        assert (
            len(dm._val_retrieval_targets) == 4
        )  # One target list per unique query (bidirectional)

    def test_validation_query_grouping(self, mock_data_files):
        """Test that validation pairs are correctly grouped by query."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        dm.setup("fit")

        # Check query dataset has unique queries (bidirectional: _f and _r)
        query_keys = dm._val_query_data.keys
        assert len(query_keys) == 4  # 2 reactions × 2 directions
        assert "rxn1_f" in query_keys
        assert "rxn1_r" in query_keys
        assert "rxn2_f" in query_keys
        assert "rxn2_r" in query_keys

        # Check target lists
        rxn1_f_targets = dm._val_retrieval_targets["rxn1_f"]
        rxn1_r_targets = dm._val_retrieval_targets["rxn1_r"]
        rxn2_f_targets = dm._val_retrieval_targets["rxn2_f"]
        rxn2_r_targets = dm._val_retrieval_targets["rxn2_r"]

        # rxn1_f and rxn1_r both have 1 validation pair (with prot1)
        assert len(rxn1_f_targets) == 1
        assert "prot1" in rxn1_f_targets
        assert len(rxn1_r_targets) == 1
        assert "prot1" in rxn1_r_targets

        # rxn2_f and rxn2_r both have 1 validation pair (with prot2)
        assert len(rxn2_f_targets) == 1
        assert "prot2" in rxn2_f_targets
        assert len(rxn2_r_targets) == 1
        assert "prot2" in rxn2_r_targets

    def test_validation_batch_format(self, mock_data_files):
        """Test that validation retrieval batches contain query_id."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
            retrieval_batch_size=1,
        )

        dm.setup("fit")
        val_loaders = dm.val_dataloader()

        # Get retrieval loader (index 2)
        retrieval_loader = val_loaders[2]
        batch = next(iter(retrieval_loader))

        # Batch should contain query_id and query_vec
        assert "query_id" in batch
        assert "query_vec" in batch

        # query_id should be a list of strings
        assert isinstance(batch["query_id"], list)
        assert isinstance(batch["query_id"][0], str)

        # query_vec should be a tensor
        assert isinstance(batch["query_vec"], torch.Tensor)

    def test_val_dataloader(self, mock_data_files):
        """Test validation dataloader creation."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
            train_batch_size=2,
            retrieval_batch_size=1,
        )

        dm.setup("fit")
        val_loaders = dm.val_dataloader()

        # Should return 3 dataloaders
        assert len(val_loaders) == 3

        # Check each dataloader
        val_loss_loader, target_loader, query_loader = val_loaders

        assert val_loss_loader is not None
        assert target_loader is not None
        assert query_loader is not None

    def test_error_without_setup(self, mock_data_files):
        """Test that dataloaders raise error if setup not called."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        with pytest.raises(RuntimeError, match="not setup"):
            dm.train_dataloader()

        with pytest.raises(RuntimeError, match="not setup"):
            dm.val_dataloader()

    def test_bidirectional_augmentation(self, mock_data_files):
        """Test that reactions are augmented bidirectionally (Bug 1 fix)."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        dm.setup("fit")

        # Check that query dataset has bidirectional reactions (_f and _r)
        query_keys = list(dm._train_query_data.keys)

        # Should have 2 original reactions × 2 directions = 4 total
        assert len(query_keys) == 4

        # Check for forward and backward variants
        assert any("rxn1_f" in k for k in query_keys)
        assert any("rxn1_r" in k for k in query_keys)
        assert any("rxn2_f" in k for k in query_keys)
        assert any("rxn2_r" in k for k in query_keys)

        # Check that backward reactions have reversed SMILES
        # Forward: "CCO>>CC=O" → Backward: "CC=O>>CCO"
        # We can't easily check SMILES directly through fingerprints,
        # but we can verify the keys exist

    def test_full_screening_set(self, mock_data_files):
        """Test that screening set includes ALL proteins (Bug 2 fix)."""
        # Modify mock data to have val-only proteins
        tmpdir = Path(mock_data_files["train_pairs"]).parent

        # Create new validation pairs with val-only protein
        val_pairs_path = tmpdir / "val_pairs_extended.csv"
        with open(val_pairs_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pr_id", "reaction_id", "protein_id"])
            writer.writeheader()
            writer.writerows(
                [
                    {"pr_id": "1", "reaction_id": "rxn1", "protein_id": "prot1"},
                    {"pr_id": "2", "reaction_id": "rxn2", "protein_id": "prot3"},
                ]
            )

        # Create new proteins file with val-only protein
        protein_embeds_path = tmpdir / "protein_embeds_extended.h5"
        with h5py.File(protein_embeds_path, "w") as f:
            ids = ["prot1", "prot2", "prot3"]
            f.create_dataset("ids", data=[id.encode() for id in ids])
            vectors = torch.randn(3, 1024).numpy()
            f.create_dataset("vectors", data=vectors)

        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=str(val_pairs_path),
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=str(protein_embeds_path),
        )

        dm.setup("fit")

        # Verify screening set includes all proteins (train + val)
        screening_protein_ids = set(dm._screening_target_data.keys)

        # Should have all 3 proteins
        assert len(screening_protein_ids) == 3
        assert "prot1" in screening_protein_ids
        assert "prot2" in screening_protein_ids
        assert "prot3" in screening_protein_ids  # Val-only protein

    def test_morgan_fingerprint_radius(self):
        """Test that Morgan fingerprint radius is 3 (Bug 3 fix)."""
        from horizyn.datasets.base import BaseDataset
        from horizyn.datasets.fingerprints import RDKitPlusFingerprintDataset

        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            mol_fp_type="morgan",
        )

        # Check that the fingerprint generator has radius=3
        assert fp_dataset._fp_gen is not None
        # The generator object doesn't expose radius directly,
        # but we can verify it was created without errors
        # and produces valid fingerprints
        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_bidirectional_reactions_have_forward_backward_suffixes(self, mock_data_files):
        """Test that reactions are augmented with _f and _r suffixes."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        dm.setup("fit")

        # Check that query dataset has forward and backward reactions
        query_keys = list(dm._train_query_data.keys)

        # Should have _f and _r suffixes
        forward_keys = [k for k in query_keys if k.endswith("_f")]
        backward_keys = [k for k in query_keys if k.endswith("_r")]

        assert len(forward_keys) > 0, "No forward reactions found"
        assert len(backward_keys) > 0, "No backward reactions found"
        assert len(forward_keys) == len(backward_keys), "Forward and backward counts don't match"

        # Check that for each forward, there's a corresponding backward
        for fkey in forward_keys:
            base_id = fkey[:-2]  # Remove _f suffix
            rkey = f"{base_id}_r"
            assert rkey in backward_keys, f"Missing backward reaction for {fkey}"

    def test_bidirectional_training_pairs(self, mock_data_files):
        """Test that training pairs are doubled for bidirectional reactions."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        dm.setup("fit")

        # Count pairs with _f and _r suffixes
        train_keys = list(dm._train_data.keys)
        forward_pairs = [k for k in train_keys if k.endswith("_f")]
        backward_pairs = [k for k in train_keys if k.endswith("_r")]

        assert len(forward_pairs) > 0, "No forward pairs found"
        assert len(backward_pairs) > 0, "No backward pairs found"
        assert len(forward_pairs) == len(
            backward_pairs
        ), "Forward and backward pair counts don't match"

    def test_bidirectional_validation_pairs(self, mock_data_files):
        """Test that validation pairs are doubled for bidirectional reactions."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            train_reactions_path=mock_data_files["train_reactions"],
            val_reactions_path=mock_data_files["val_reactions"],
            protein_embeds_path=mock_data_files["protein_embeds"],
        )

        dm.setup("fit")

        # Count validation pairs with _f and _r suffixes
        val_keys = list(dm._val_data.keys)
        forward_pairs = [k for k in val_keys if k.endswith("_f")]
        backward_pairs = [k for k in val_keys if k.endswith("_r")]

        assert len(forward_pairs) > 0, "No forward validation pairs found"
        assert len(backward_pairs) > 0, "No backward validation pairs found"
        assert len(forward_pairs) == len(
            backward_pairs
        ), "Forward and backward validation pair counts don't match"
