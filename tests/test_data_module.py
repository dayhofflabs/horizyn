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

    # Create mock training pairs (using standardized schema)
    train_pairs_path = tmpdir / "train_pairs.db"
    conn = sqlite3.connect(train_pairs_path)
    conn.execute(
        "CREATE TABLE protein_to_reaction (pr_id INTEGER PRIMARY KEY, reaction_id TEXT, protein_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO protein_to_reaction VALUES (?, ?, ?)",
        [
            (1, "rxn1", "prot1"),
            (2, "rxn2", "prot2"),
            (3, "rxn1", "prot2"),
        ],
    )
    conn.commit()
    conn.close()

    # Create mock validation pairs (using standardized schema)
    val_pairs_path = tmpdir / "val_pairs.db"
    conn = sqlite3.connect(val_pairs_path)
    conn.execute(
        "CREATE TABLE protein_to_reaction (pr_id INTEGER PRIMARY KEY, reaction_id TEXT, protein_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO protein_to_reaction VALUES (?, ?, ?)",
        [
            (1, "rxn1", "prot1"),
            (2, "rxn2", "prot2"),
        ],
    )
    conn.commit()
    conn.close()

    # Create mock reactions (using standardized schema)
    reactions_path = tmpdir / "reactions.db"
    conn = sqlite3.connect(reactions_path)
    conn.execute("CREATE TABLE reaction (reaction_id TEXT PRIMARY KEY, reaction_smiles TEXT)")
    conn.executemany(
        "INSERT INTO reaction VALUES (?, ?)",
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
