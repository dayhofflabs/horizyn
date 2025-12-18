"""
Integration tests using the full SOTA dataset.

These tests verify data integrity with the production dataset without expensive
operations like fingerprint computation or full training runs. All tests complete
in < 5 seconds and are included in default test runs.

Run tests:
    pytest tests/integration/test_sota.py -v
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture
def check_sota_data():
    """Skip test if SOTA data is not available."""
    sota_dir = Path("data/sota")
    required_files = [
        "train_rxns.csv",
        "test_rxns.csv",
        "prots_t5.h5",
        "train_pairs.csv",
        "test_pairs.csv",
    ]

    for filename in required_files:
        filepath = sota_dir / filename
        if not filepath.exists():
            pytest.skip(
                f"SOTA data not found: {filename}. "
                "Run 'python scripts/download_data.py' to download."
            )


class TestSOTAFast:
    """
    Tests using full SOTA dataset.

    These tests verify data integrity and basic loading without expensive
    operations like fingerprint computation or training. They run in < 5 seconds
    each and are included in default test runs.
    """

    def test_sota_config_is_valid(self, check_sota_data):
        """Test that sota.yaml config is valid and points to SOTA data."""
        config_path = Path("configs/sota.yaml")
        assert config_path.exists(), "sota.yaml config not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check all required sections
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert "logging" in config

        # Verify paths point to sota
        data_cfg = config["data"]
        assert "sota" in data_cfg["train_reactions_path"]
        assert "sota" in data_cfg["test_reactions_path"]
        assert "sota" in data_cfg["protein_embeds_path"]
        assert "sota" in data_cfg["train_pairs_path"]
        assert "sota" in data_cfg["test_pairs_path"]

        # Verify all files exist
        for key in [
            "train_reactions_path",
            "test_reactions_path",
            "protein_embeds_path",
            "train_pairs_path",
            "test_pairs_path",
        ]:
            filepath = Path(data_cfg[key])
            assert filepath.exists(), f"Missing SOTA file: {filepath}"

    def test_sota_files_exist_and_not_empty(self, check_sota_data):
        """Test that all SOTA files exist and are not empty."""
        sota_dir = Path("data/sota")
        required_files = [
            "train_rxns.csv",
            "test_rxns.csv",
            "prots_t5.h5",
            "train_pairs.csv",
            "test_pairs.csv",
        ]

        for filename in required_files:
            filepath = sota_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"

    def test_sota_csv_schemas(self, check_sota_data):
        """
        Test that SOTA CSV files have expected schemas.

        This is a fast test that just checks column structure without loading all data.
        Runtime: < 1 second
        """
        import csv

        # Check reactions CSV
        train_rxns_csv = Path("data/sota/train_rxns.csv")
        with open(train_rxns_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Check expected columns exist
            assert "reaction_id" in fieldnames
            assert "reaction_smiles" in fieldnames

        # Check train_pairs CSV
        train_pairs_csv = Path("data/sota/train_pairs.csv")
        with open(train_pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Check expected columns exist
            assert "pr_id" in fieldnames
            assert "reaction_id" in fieldnames
            assert "protein_id" in fieldnames

    def test_sota_dataset_sizes(self, check_sota_data):
        """
        Test that SOTA datasets have expected sizes.

        This is a fast test that queries row counts without loading data.
        Runtime: < 2 seconds
        """
        import csv

        import h5py

        # Check training pairs count
        train_pairs_csv = Path("data/sota/train_pairs.csv")
        with open(train_pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            train_count = sum(1 for _ in reader)

            # SOTA has ~257k training pairs
            assert (
                200_000 < train_count < 300_000
            ), f"Training pairs count {train_count} outside expected range (200k-300k)"

        # Check test pairs count
        test_pairs_csv = Path("data/sota/test_pairs.csv")
        with open(test_pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            test_count = sum(1 for _ in reader)

            # SOTA has ~36k test pairs
            assert (
                30_000 < test_count < 50_000
            ), f"Test pairs count {test_count} outside expected range (30k-50k)"

        # Check protein embeddings count
        proteins_h5 = Path("data/sota/prots_t5.h5")
        with h5py.File(str(proteins_h5), "r") as f:
            assert "ids" in f, "ids dataset not found in HDF5"
            assert "vectors" in f, "vectors dataset not found in HDF5"

            protein_count = len(f["ids"])
            # SOTA has ~200k+ proteins
            assert (
                150_000 < protein_count < 300_000
            ), f"Protein count {protein_count} outside expected range (150k-300k)"

            # Check embedding dimensions
            embed_shape = f["vectors"].shape
            assert embed_shape[1] == 1024, f"Expected 1024-dim embeddings, got {embed_shape[1]}"
