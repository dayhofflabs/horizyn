"""
Integration tests using the full SwissProt dataset.

These tests verify data integrity with the production dataset without expensive
operations like fingerprint computation or full training runs. All tests complete
in < 5 seconds and are included in default test runs.

Run tests:
    pytest tests/integration/test_swissprot.py -v
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture
def check_swissprot_data():
    """Skip test if SwissProt data is not available."""
    swissprot_dir = Path("data/swissprot")
    required_files = [
        "reactions.db",
        "proteins_t5_embeddings.h5",
        "train_pairs.db",
        "val_pairs.db",
    ]

    for filename in required_files:
        filepath = swissprot_dir / filename
        if not filepath.exists():
            pytest.skip(
                f"SwissProt data not found: {filename}. "
                "Run 'python scripts/download_data.py' to download."
            )


class TestSwissProtFast:
    """
    Tests using full SwissProt dataset.

    These tests verify data integrity and basic loading without expensive
    operations like fingerprint computation or training. They run in < 5 seconds
    each and are included in default test runs.
    """

    def test_swissprot_config_is_valid(self, check_swissprot_data):
        """Test that sota.yaml config is valid and points to SwissProt data."""
        config_path = Path("configs/sota.yaml")
        assert config_path.exists(), "sota.yaml config not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check all required sections
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert "logging" in config

        # Verify paths point to swissprot
        data_cfg = config["data"]
        assert "swissprot" in data_cfg["reactions_path"]
        assert "swissprot" in data_cfg["proteins_path"]
        assert "swissprot" in data_cfg["train_pairs_path"]
        assert "swissprot" in data_cfg["val_pairs_path"]

        # Verify all files exist
        for key in ["reactions_path", "proteins_path", "train_pairs_path", "val_pairs_path"]:
            filepath = Path(data_cfg[key])
            assert filepath.exists(), f"Missing SwissProt file: {filepath}"

    def test_swissprot_files_exist_and_not_empty(self, check_swissprot_data):
        """Test that all SwissProt files exist and are not empty."""
        swissprot_dir = Path("data/swissprot")
        required_files = [
            "reactions.db",
            "proteins_t5_embeddings.h5",
            "train_pairs.db",
            "val_pairs.db",
        ]

        for filename in required_files:
            filepath = swissprot_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"

    def test_swissprot_database_schemas(self, check_swissprot_data):
        """
        Test that SwissProt SQLite databases have expected schemas.

        This is a fast test that just checks table structure without loading data.
        Runtime: < 1 second
        """
        import sqlite3

        # Check reactions database
        reactions_db = Path("data/swissprot/reactions.db")
        with sqlite3.connect(str(reactions_db)) as conn:
            cursor = conn.cursor()

            # Check reaction table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reaction'")
            assert cursor.fetchone() is not None, "reaction table not found"

            # Check expected columns exist
            cursor.execute("PRAGMA table_info(reaction)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "reaction_id" in columns
            assert "reaction_smiles" in columns

        # Check train_pairs database
        train_pairs_db = Path("data/swissprot/train_pairs.db")
        with sqlite3.connect(str(train_pairs_db)) as conn:
            cursor = conn.cursor()

            # Check protein_to_reaction table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='protein_to_reaction'"
            )
            assert cursor.fetchone() is not None, "protein_to_reaction table not found"

            # Check expected columns exist
            cursor.execute("PRAGMA table_info(protein_to_reaction)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "pr_id" in columns
            assert "reaction_id" in columns
            assert "protein_id" in columns

    def test_swissprot_dataset_sizes(self, check_swissprot_data):
        """
        Test that SwissProt datasets have expected sizes.

        This is a fast test that queries row counts without loading data.
        Runtime: < 2 seconds
        """
        import sqlite3

        import h5py

        # Check training pairs count
        train_pairs_db = Path("data/swissprot/train_pairs.db")
        with sqlite3.connect(str(train_pairs_db)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM protein_to_reaction")
            train_count = cursor.fetchone()[0]

            # SwissProt has ~257k training pairs
            assert (
                200_000 < train_count < 300_000
            ), f"Training pairs count {train_count} outside expected range (200k-300k)"

        # Check validation pairs count
        val_pairs_db = Path("data/swissprot/val_pairs.db")
        with sqlite3.connect(str(val_pairs_db)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM protein_to_reaction")
            val_count = cursor.fetchone()[0]

            # SwissProt has ~36k validation pairs
            assert (
                30_000 < val_count < 50_000
            ), f"Validation pairs count {val_count} outside expected range (30k-50k)"

        # Check protein embeddings count
        proteins_h5 = Path("data/swissprot/proteins_t5_embeddings.h5")
        with h5py.File(str(proteins_h5), "r") as f:
            assert "ids" in f, "ids dataset not found in HDF5"
            assert "vectors" in f, "vectors dataset not found in HDF5"

            protein_count = len(f["ids"])
            # SwissProt has ~200k+ proteins
            assert (
                150_000 < protein_count < 300_000
            ), f"Protein count {protein_count} outside expected range (150k-300k)"

            # Check embedding dimensions
            embed_shape = f["vectors"].shape
            assert embed_shape[1] == 1024, f"Expected 1024-dim embeddings, got {embed_shape[1]}"
