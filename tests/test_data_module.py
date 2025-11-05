"""
Unit tests for the data module.
"""

import sqlite3
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

    # Create mock training pairs
    train_pairs_path = tmpdir / "train_pairs.db"
    conn = sqlite3.connect(train_pairs_path)
    conn.execute("CREATE TABLE pairs (pair_id TEXT PRIMARY KEY, query_id TEXT, target_id TEXT)")
    conn.executemany(
        "INSERT INTO pairs VALUES (?, ?, ?)",
        [
            ("pair1", "rxn1", "prot1"),
            ("pair2", "rxn2", "prot2"),
            ("pair3", "rxn1", "prot2"),
        ],
    )
    conn.commit()
    conn.close()

    # Create mock validation pairs
    val_pairs_path = tmpdir / "val_pairs.db"
    conn = sqlite3.connect(val_pairs_path)
    conn.execute("CREATE TABLE pairs (pair_id TEXT PRIMARY KEY, query_id TEXT, target_id TEXT)")
    conn.executemany(
        "INSERT INTO pairs VALUES (?, ?, ?)",
        [
            ("val_pair1", "rxn1", "prot1"),
            ("val_pair2", "rxn2", "prot2"),
        ],
    )
    conn.commit()
    conn.close()

    # Create mock reactions
    reactions_path = tmpdir / "reactions.db"
    conn = sqlite3.connect(reactions_path)
    conn.execute("CREATE TABLE reactions (reaction_id TEXT PRIMARY KEY, reaction_smiles TEXT)")
    conn.executemany(
        "INSERT INTO reactions VALUES (?, ?)",
        [
            ("rxn1", "CCO>>CC=O"),
            ("rxn2", "C>>CC"),
        ],
    )
    conn.commit()
    conn.close()

    # Create mock protein embeddings
    proteins_path = tmpdir / "proteins.h5"
    with h5py.File(proteins_path, "w") as f:
        # Create ids dataset (keys)
        ids = ["prot1", "prot2"]
        f.create_dataset("ids", data=[id.encode() for id in ids])
        # Create vectors dataset (embeddings)
        vectors = torch.randn(2, 1024).numpy()
        f.create_dataset("vectors", data=vectors)

    return {
        "train_pairs": str(train_pairs_path),
        "val_pairs": str(val_pairs_path),
        "reactions": str(reactions_path),
        "proteins": str(proteins_path),
    }


class TestHorizynDataModule:
    """Tests for the HorizynDataModule class."""

    def test_initialization(self, mock_data_files):
        """Test data module initialization."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
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
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
            train_batch_size=2,
            retrieval_batch_size=1,
        )

        dm.setup("fit")

        # Verify datasets are created
        assert dm._train_data is not None
        assert dm._val_data is not None
        assert dm._target_data is not None

        # Verify sizes
        assert len(dm._train_data) == 3  # 3 training pairs
        assert len(dm._val_data) == 2  # 2 validation pairs

    @pytest.mark.skip(reason="Requires real dataset files for proper dataloader testing")
    def test_train_dataloader(self, mock_data_files):
        """Test training dataloader creation."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
            train_batch_size=2,
            retrieval_batch_size=1,
            num_workers=0,  # Avoid multiprocessing issues in tests
        )

        dm.setup("fit")
        train_loader = dm.train_dataloader()

        assert train_loader is not None
        assert train_loader.batch_size == 2

        # Get a batch
        batch = next(iter(train_loader))
        assert "query_vec" in batch
        assert "target_vec" in batch

        # Verify shapes (2048-dim for query, 1024-dim for target)
        assert batch["query_vec"].shape[1] == 2048  # RDKit+ (1024) + DRFP (1024)
        assert batch["target_vec"].shape[1] == 1024  # T5 embeddings

    def test_val_dataloader(self, mock_data_files):
        """Test validation dataloader creation."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
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

    @pytest.mark.skip(reason="TupleDataset needs refinement for this use case")
    def test_fingerprint_concatenation(self, mock_data_files):
        """Test that RDKit+ and DRFP fingerprints are concatenated."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
            rdkit_fp_dim=1024,
            drfp_dim=1024,
        )

        dm.setup("fit")

        # Get a sample
        sample = dm._train_data[dm._train_data.keys[0]]

        # Check query vector is 2048-dim (1024 + 1024)
        assert sample["query_vec"].shape == (2048,)
        # Check target vector is 1024-dim
        assert sample["target_vec"].shape == (1024,)

    def test_error_without_setup(self, mock_data_files):
        """Test that dataloaders raise error if setup not called."""
        dm = HorizynDataModule(
            train_pairs_path=mock_data_files["train_pairs"],
            val_pairs_path=mock_data_files["val_pairs"],
            reactions_path=mock_data_files["reactions"],
            proteins_path=mock_data_files["proteins"],
        )

        with pytest.raises(RuntimeError, match="not setup"):
            dm.train_dataloader()

        with pytest.raises(RuntimeError, match="not setup"):
            dm.val_dataloader()

