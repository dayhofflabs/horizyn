"""
Unit tests for CSV and HDF5 dataset classes.
"""

import csv
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from horizyn.datasets import EmbedDataset, CSVDataset


class TestCSVDataset:
    """Tests for the CSVDataset class."""

    @pytest.fixture
    def sample_csv_reactions(self, tmp_path):
        """Create a sample reactions CSV for testing."""
        csv_path = tmp_path / "reactions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reaction_id", "reaction_smiles", "ec_number"])
            writer.writerow(["rxn1", "CCO.O>>CC=O", "1.1.1.1"])
            writer.writerow(["rxn2", "C.O>>CO", "2.2.2.2"])
            writer.writerow(["rxn3", "CC>>C=C", "3.3.3.3"])
        return csv_path

    @pytest.fixture
    def sample_csv_pairs(self, tmp_path):
        """Create a sample pairs CSV for testing."""
        csv_path = tmp_path / "pairs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pr_id", "reaction_id", "protein_id"])
            writer.writerow(["0", "rxn1", "prot1"])
            writer.writerow(["1", "rxn2", "prot2"])
            writer.writerow(["2", "rxn1", "prot3"])
        return csv_path

    def test_initialization(self, sample_csv_reactions):
        """Test basic initialization."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        assert len(dataset) == 3
        assert "rxn1" in dataset.keys
        assert "rxn2" in dataset.keys
        assert "rxn3" in dataset.keys

    def test_getitem_single_column(self, sample_csv_reactions):
        """Test accessing data with a single column."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns="reaction_smiles",
        )

        sample = dataset["rxn1"]
        assert sample == {"reaction_smiles": "CCO.O>>CC=O"}

    def test_getitem_multiple_columns(self, sample_csv_reactions):
        """Test accessing data with multiple columns."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles", "ec_number"],
        )

        sample = dataset["rxn2"]
        assert sample == {"reaction_smiles": "C.O>>CO", "ec_number": "2.2.2.2"}

    def test_all_columns(self, sample_csv_reactions):
        """Test loading all columns (except key_column)."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=None,  # Load all columns
        )

        sample = dataset["rxn1"]
        assert "reaction_smiles" in sample
        assert "ec_number" in sample
        assert "reaction_id" not in sample  # key_column excluded

    def test_rename_columns(self, sample_csv_reactions):
        """Test column renaming."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
            rename_map={"reaction_smiles": "smiles"},
        )

        sample = dataset["rxn1"]
        assert "smiles" in sample
        assert "reaction_smiles" not in sample

    def test_file_not_found(self, tmp_path):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            CSVDataset(
                file_path=str(tmp_path / "nonexistent.csv"),
                key_column="reaction_id",
            )

    def test_key_column_not_found(self, sample_csv_reactions):
        """Test that missing key_column raises error."""
        with pytest.raises(ValueError, match="Key column.*not found"):
            CSVDataset(
                file_path=str(sample_csv_reactions),
                key_column="nonexistent_column",
            )

    def test_column_not_found(self, sample_csv_reactions):
        """Test that missing column raises error."""
        with pytest.raises(ValueError, match="Columns.*not found"):
            CSVDataset(
                file_path=str(sample_csv_reactions),
                key_column="reaction_id",
                columns=["nonexistent_column"],
            )

    def test_key_not_found(self, sample_csv_reactions):
        """Test that accessing non-existent key raises error."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        with pytest.raises(KeyError, match="not found"):
            dataset["nonexistent_rxn"]

    def test_pairs_csv(self, sample_csv_pairs):
        """Test loading a pairs CSV (typical use case)."""
        dataset = CSVDataset(
            file_path=str(sample_csv_pairs),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
        )

        assert len(dataset) == 3

        pair0 = dataset["0"]
        assert pair0 == {"reaction_id": "rxn1", "protein_id": "prot1"}

        pair1 = dataset["1"]
        assert pair1 == {"reaction_id": "rxn2", "protein_id": "prot2"}

    def test_iteration(self, sample_csv_reactions):
        """Test that dataset is iterable."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        items = [dataset[key] for key in dataset.keys]
        assert len(items) == 3

    def test_integer_indexing(self, sample_csv_reactions):
        """Test that dataset supports integer indexing."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        # Access by integer index
        sample0 = dataset[0]
        sample1 = dataset[1]
        sample2 = dataset[2]

        # Should return same data as accessing by key
        assert sample0 == dataset[dataset.keys[0]]
        assert sample1 == dataset[dataset.keys[1]]
        assert sample2 == dataset[dataset.keys[2]]

    def test_integer_indexing_out_of_bounds(self, sample_csv_reactions):
        """Test that out of bounds integer index raises IndexError."""
        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        with pytest.raises(IndexError, match="out of bounds"):
            dataset[10]

        with pytest.raises(IndexError, match="out of bounds"):
            dataset[-1]

    def test_integer_indexing_dataloader_compatible(self, sample_csv_reactions):
        """Test that integer indexing makes dataset compatible with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = CSVDataset(
            file_path=str(sample_csv_reactions),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        # Should be able to create a DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Should be able to iterate
        batches = list(loader)
        assert len(batches) == 2  # 3 items, batch size 2 -> 2 batches


class TestEmbedDataset:
    """Tests for the EmbedDataset class."""

    @pytest.fixture
    def sample_hdf5(self, tmp_path):
        """Create a sample HDF5 file for testing."""
        h5_path = tmp_path / "test_embeddings.h5"

        with h5py.File(str(h5_path), "w") as f:
            # Create ids dataset
            ids = np.array([b"prot1", b"prot2", b"prot3"], dtype="S10")
            f.create_dataset("ids", data=ids)

            # Create vectors dataset (3 proteins, 512-dim embeddings)
            vectors = np.random.randn(3, 512).astype(np.float32)
            f.create_dataset("vectors", data=vectors)

        return h5_path, vectors

    def test_initialization(self, sample_hdf5):
        """Test basic initialization."""
        h5_path, _ = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        assert len(dataset) == 3
        assert "prot1" in dataset.keys
        assert "prot2" in dataset.keys
        assert "prot3" in dataset.keys
        assert dataset.vec_dim == 512
        assert dataset.num_vecs == 3

    def test_getitem(self, sample_hdf5):
        """Test accessing embeddings."""
        h5_path, original_vectors = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        embed1 = dataset["prot1"]
        assert embed1.shape == (512,)
        assert torch.allclose(embed1, torch.from_numpy(original_vectors[0]), atol=1e-5)

    def test_in_memory_true(self, sample_hdf5):
        """Test in-memory loading."""
        h5_path, _ = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        assert dataset.in_memory is True
        assert dataset.data is not None
        assert dataset.data.shape == (3, 512)
        assert dataset.file is None  # File should be closed

    def test_in_memory_false(self, sample_hdf5):
        """Test on-the-fly loading from HDF5."""
        h5_path, original_vectors = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=False)

        assert dataset.in_memory is False
        assert dataset.data is None
        assert dataset.file is not None  # File should be open

        # Should still be able to access data
        embed1 = dataset["prot1"]
        assert torch.allclose(embed1, torch.from_numpy(original_vectors[0]), atol=1e-5)

    def test_dtype(self, sample_hdf5):
        """Test different data types."""
        h5_path, _ = sample_hdf5

        # Test float32 (default)
        dataset_f32 = EmbedDataset(file_path=str(h5_path), in_memory=True, dtype=torch.float32)
        assert dataset_f32["prot1"].dtype == torch.float32

        # Test float64
        dataset_f64 = EmbedDataset(file_path=str(h5_path), in_memory=True, dtype=torch.float64)
        assert dataset_f64["prot1"].dtype == torch.float64

    def test_file_not_found(self, tmp_path):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            EmbedDataset(file_path=str(tmp_path / "nonexistent.h5"))

    def test_missing_ids_dataset(self, tmp_path):
        """Test that missing 'ids' dataset raises error."""
        h5_path = tmp_path / "missing_ids.h5"

        with h5py.File(str(h5_path), "w") as f:
            vectors = np.random.randn(3, 512).astype(np.float32)
            f.create_dataset("vectors", data=vectors)
            # No 'ids' dataset

        with pytest.raises(KeyError, match="ids.*not found"):
            EmbedDataset(file_path=str(h5_path))

    def test_missing_vectors_dataset(self, tmp_path):
        """Test that missing 'vectors' dataset raises error."""
        h5_path = tmp_path / "missing_vectors.h5"

        with h5py.File(str(h5_path), "w") as f:
            ids = np.array([b"prot1", b"prot2"], dtype="S10")
            f.create_dataset("ids", data=ids)
            # No 'vectors' dataset

        with pytest.raises(KeyError, match="vectors.*not found"):
            EmbedDataset(file_path=str(h5_path))

    def test_mismatched_lengths(self, tmp_path):
        """Test that mismatched ids/vectors lengths raise error."""
        h5_path = tmp_path / "mismatched.h5"

        with h5py.File(str(h5_path), "w") as f:
            ids = np.array([b"prot1", b"prot2"], dtype="S10")
            f.create_dataset("ids", data=ids)

            vectors = np.random.randn(3, 512).astype(np.float32)  # 3 != 2
            f.create_dataset("vectors", data=vectors)

        with pytest.raises(ValueError, match="Mismatch"):
            EmbedDataset(file_path=str(h5_path))

    def test_key_not_found(self, sample_hdf5):
        """Test that accessing non-existent key raises error."""
        h5_path, _ = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        with pytest.raises(KeyError):
            dataset["nonexistent_prot"]

    def test_all_embeddings_accessible(self, sample_hdf5):
        """Test that all embeddings can be accessed."""
        h5_path, original_vectors = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        for i, key in enumerate(dataset.keys):
            embed = dataset[key]
            expected = torch.from_numpy(original_vectors[i]).to(torch.float32)
            assert torch.allclose(embed, expected, atol=1e-5)

    def test_large_embedding_dim(self, tmp_path):
        """Test with larger embedding dimensions (e.g., T5)."""
        h5_path = tmp_path / "large_embeds.h5"

        with h5py.File(str(h5_path), "w") as f:
            ids = np.array([b"prot1", b"prot2"], dtype="S10")
            f.create_dataset("ids", data=ids)

            # T5-like dimension (1024)
            vectors = np.random.randn(2, 1024).astype(np.float32)
            f.create_dataset("vectors", data=vectors)

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        assert dataset.vec_dim == 1024
        assert dataset["prot1"].shape == (1024,)

    def test_string_ids(self, tmp_path):
        """Test with different ID formats (strings vs bytes)."""
        h5_path = tmp_path / "string_ids.h5"

        with h5py.File(str(h5_path), "w") as f:
            # Use variable-length string type
            ids = np.array(["protein_001", "protein_002"], dtype=h5py.string_dtype())
            f.create_dataset("ids", data=ids)

            vectors = np.random.randn(2, 512).astype(np.float32)
            f.create_dataset("vectors", data=vectors)

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        assert "protein_001" in dataset.keys
        assert "protein_002" in dataset.keys
        assert dataset["protein_001"].shape == (512,)

    def test_integer_indexing(self, sample_hdf5):
        """Test that dataset supports integer indexing."""
        h5_path, original_vectors = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        # Access by integer index
        embed0 = dataset[0]
        embed1 = dataset[1]
        embed2 = dataset[2]

        # Should return same data as accessing by key
        assert torch.equal(embed0, dataset[dataset.keys[0]])
        assert torch.equal(embed1, dataset[dataset.keys[1]])
        assert torch.equal(embed2, dataset[dataset.keys[2]])

        # Should match original vectors
        assert torch.allclose(embed0, torch.from_numpy(original_vectors[0]), atol=1e-5)
        assert torch.allclose(embed1, torch.from_numpy(original_vectors[1]), atol=1e-5)
        assert torch.allclose(embed2, torch.from_numpy(original_vectors[2]), atol=1e-5)

    def test_integer_indexing_out_of_bounds(self, sample_hdf5):
        """Test that out of bounds integer index raises IndexError."""
        h5_path, _ = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        with pytest.raises(IndexError, match="out of bounds"):
            dataset[10]

        with pytest.raises(IndexError, match="out of bounds"):
            dataset[-1]

    def test_integer_indexing_dataloader_compatible(self, sample_hdf5):
        """Test that integer indexing makes dataset compatible with DataLoader."""
        from torch.utils.data import DataLoader

        h5_path, _ = sample_hdf5

        dataset = EmbedDataset(file_path=str(h5_path), in_memory=True)

        # Should be able to create a DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Should be able to iterate
        batches = list(loader)
        assert len(batches) == 2  # 3 items, batch size 2 -> 2 batches

        # Check batch structure
        first_batch = batches[0]
        assert first_batch.shape == (2, 512)  # batch_size x vec_dim


class TestCSVAndHDF5Integration:
    """Integration tests combining CSV and HDF5 datasets."""

    def test_realistic_workflow(self, tmp_path):
        """Test a realistic workflow with pairs, reactions, and embeddings."""
        # Create pairs CSV
        pairs_csv = tmp_path / "pairs.csv"
        with open(pairs_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pr_id", "reaction_id", "protein_id"])
            writer.writerow(["0", "rxn1", "prot1"])
            writer.writerow(["1", "rxn2", "prot2"])

        # Create embeddings HDF5
        embeds_h5 = tmp_path / "embeddings.h5"
        with h5py.File(str(embeds_h5), "w") as f:
            ids = np.array([b"prot1", b"prot2"], dtype="S10")
            f.create_dataset("ids", data=ids)
            vectors = np.random.randn(2, 512).astype(np.float32)
            f.create_dataset("vectors", data=vectors)

        # Load datasets
        pairs = CSVDataset(
            file_path=str(pairs_csv),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
        )

        embeds = EmbedDataset(file_path=str(embeds_h5), in_memory=True)

        # Test accessing related data
        pair0 = pairs["0"]
        target_id = pair0["protein_id"]
        embedding = embeds[target_id]

        assert embedding.shape == (512,)
