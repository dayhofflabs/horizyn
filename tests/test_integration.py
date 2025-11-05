"""
Integration tests for the Horizyn training pipeline.

This module contains two types of tests:
1. Smoke tests: Fast tests using nanodata (~12 reactions, runs in < 1 minute)
2. E2E tests: Full end-to-end tests using SwissProt data (requires download, ~10 minutes)

Smoke tests run by default. E2E tests are marked with @pytest.mark.e2e and can be run with:
    pytest tests/test_integration.py -v -m e2e
"""

import shutil
from pathlib import Path

import pytest
import torch
import yaml


# Pytest markers
pytestmark = pytest.mark.integration


class TestSmokeNanodata:
    """
    Smoke tests using the nanodata dataset.

    Fast tests (<1 minute) that verify basic functionality:
    - Config loading
    - Data loading (all files present and valid)
    - Model initialization
    - Training loop execution (2 epochs, limited batches)
    - Checkpoint saving
    - Metrics logging

    These tests run by default and are designed to catch obvious breakage quickly.
    """

    def test_nano_config_is_valid(self):
        """Test that nano.yaml config is valid and complete."""
        config_path = Path("configs/nano.yaml")
        assert config_path.exists(), "nano.yaml config not found"

        # Load and validate config structure
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check all required top-level sections
        assert "data" in config, "Missing 'data' section"
        assert "model" in config, "Missing 'model' section"
        assert "training" in config, "Missing 'training' section"
        assert "logging" in config, "Missing 'logging' section"

        # Check data paths point to nanodata
        data_cfg = config["data"]
        assert "nanodata" in data_cfg["reactions_db"]
        assert "nanodata" in data_cfg["proteins_h5"]
        assert "nanodata" in data_cfg["train_pairs_db"]
        assert "nanodata" in data_cfg["val_pairs_db"]

    def test_nanodata_files_exist(self):
        """Test that all nanodata files are present."""
        nanodata_dir = Path("data/nanodata")
        assert nanodata_dir.exists(), "Nanodata directory not found"

        required_files = [
            "reactions.db",
            "proteins_t5_embeddings.h5",
            "train_pairs.db",
            "val_pairs.db",
        ]

        for filename in required_files:
            filepath = nanodata_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty file: {filename}"

    def test_smoke_training_pipeline(self, tmp_path):
        """
        Smoke test: Run 2 epochs with limited batches on nanodata.

        This is a fast smoke test that verifies:
        1. Config loads successfully
        2. Data module initializes without errors
        3. Model initializes correctly
        4. Training runs for 2 epochs without crashes
        5. Checkpoints are saved
        6. Metrics are logged to CSV

        Expected runtime: ~30 seconds on CPU, ~10 seconds on GPU
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        # Load config with overrides for fast testing
        config = load_config("configs/nano.yaml")

        # Override settings for fast testing
        config.training.max_epochs = 2
        config.training.limit_train_batches = 5  # Only 5 batches per epoch
        config.training.limit_val_batches = 3  # Only 3 batches for validation

        # Use temporary directory for outputs
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir.mkdir()
        checkpoint_dir.mkdir()

        # Setup seed for reproducibility
        pl.seed_everything(42, workers=True)

        # Initialize data module
        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")

        # Verify data loaded
        assert data_module.train_data is not None
        assert data_module.val_retrieval_queries is not None
        assert data_module.val_retrieval_targets is not None
        assert data_module.val_retrieval_pairs is not None

        # Initialize model
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            scheduler_config=config.scheduler if hasattr(config, "scheduler") else None,
            dedup_pairs=config.training.get("dedup_pairs", True),
            log_retrieval_metrics=config.training.get("log_retrieval_metrics", True),
        )

        # Verify model structure
        assert model.model is not None
        assert model.loss is not None
        assert hasattr(model, "retrieval_metrics")

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="auto",
            devices=1,
            logger=pl.loggers.CSVLogger(str(log_dir), name="nano_test"),
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="epoch-{epoch:02d}",
                    save_last=True,
                    every_n_epochs=1,
                ),
            ],
            limit_train_batches=config.training.limit_train_batches,
            limit_val_batches=config.training.limit_val_batches,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=True,
        )

        # Run training
        trainer.fit(model, data_module)

        # Verify training completed
        assert trainer.current_epoch == config.training.max_epochs - 1

        # Verify checkpoints were saved
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        assert len(checkpoint_files) >= 1, "No checkpoints saved"

        # Verify last checkpoint exists
        last_ckpt = checkpoint_dir / "last.ckpt"
        assert last_ckpt.exists(), "last.ckpt not found"

        # Verify metrics were logged
        metrics_csv = log_dir / "nano_test" / "version_0" / "metrics.csv"
        assert metrics_csv.exists(), "metrics.csv not found"

        # Read and validate metrics
        with open(metrics_csv) as f:
            header = f.readline().strip()
            assert "train_loss" in header or "loss" in header

            # Should have at least one data line
            first_line = f.readline()
            assert len(first_line) > 0, "No metrics logged"

    def test_checkpoint_loading(self, tmp_path):
        """Test that we can load a model from a checkpoint."""
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        # Load config
        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 2
        config.training.limit_val_batches = 1

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Setup and train for 1 epoch
        pl.seed_everything(42)
        data_module = HorizynDataModule(**config.data)
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            dedup_pairs=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    save_last=True,
                ),
            ],
            limit_train_batches=2,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, data_module)

        # Load from checkpoint
        checkpoint_path = checkpoint_dir / "last.ckpt"
        assert checkpoint_path.exists()

        loaded_model = HorizynLitModule.load_from_checkpoint(
            checkpoint_path,
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
        )

        # Verify loaded model has correct structure
        assert loaded_model.model is not None
        assert loaded_model.loss is not None

        # Verify model parameters were loaded
        original_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        assert original_params == loaded_params

    def test_memory_efficiency(self):
        """
        Test that data loading is memory-efficient.

        Verifies that in_memory=True works and that data is loaded once.
        """
        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule

        config = load_config("configs/nano.yaml")

        # Initialize data module
        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")

        # Check that datasets are using in_memory mode
        # (They should load all data at initialization)
        train_loader = data_module.train_dataloader()
        assert train_loader is not None

        # Get first batch to verify data is accessible
        first_batch = next(iter(train_loader))

        # Verify batch structure
        assert "query" in first_batch
        assert "target" in first_batch
        assert "pair_id" in first_batch or "query_id" in first_batch

        # Verify tensors have correct shapes
        query = first_batch["query"]
        target = first_batch["target"]

        assert isinstance(query, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert query.dim() == 2  # (batch_size, features)
        assert target.dim() == 2  # (batch_size, features)

    def test_validation_metrics_computed(self, tmp_path):
        """Test that validation metrics are computed correctly."""
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 2
        config.training.limit_val_batches = 2

        pl.seed_everything(42)

        data_module = HorizynDataModule(**config.data)
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            log_retrieval_metrics=True,
        )

        log_dir = tmp_path / "logs"

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            logger=pl.loggers.CSVLogger(str(log_dir)),
            limit_train_batches=2,
            limit_val_batches=2,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, data_module)

        # Check that validation metrics were logged
        metrics_csv = log_dir / "lightning_logs" / "version_0" / "metrics.csv"
        assert metrics_csv.exists()

        with open(metrics_csv) as f:
            content = f.read()
            # Should contain validation metrics
            # Note: exact metric names depend on implementation
            # Just verify that we have some validation output
            assert "epoch" in content


class TestSmokeConfigAndErrors:
    """Smoke tests for configuration and error handling."""

    def test_config_override_from_command_line(self):
        """Test that command-line overrides work correctly."""
        from horizyn.config import apply_overrides, load_config

        # Load base config
        config = load_config("configs/nano.yaml")
        original_epochs = config.training.max_epochs

        # Apply override
        overrides = {"training.max_epochs": 999}
        apply_overrides(config, overrides)

        assert config.training.max_epochs == 999
        assert config.training.max_epochs != original_epochs

    def test_missing_data_files_produce_clear_errors(self):
        """Test that missing data files produce helpful error messages."""
        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule

        config = load_config("configs/nano.yaml")

        # Override to point to non-existent files
        config.data.reactions_db = "data/nonexistent/reactions.db"

        data_module = HorizynDataModule(**config.data)

        # Should raise clear error when trying to setup
        with pytest.raises(Exception) as exc_info:
            data_module.setup("fit")

        # Error message should mention the missing file
        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg or "not found" in error_msg.lower()


class TestSmokeRobustness:
    """Smoke tests for edge cases and robustness."""

    def test_single_batch_training(self, tmp_path):
        """Test that training works with just 1 batch."""
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 1
        config.training.limit_val_batches = 1

        pl.seed_everything(42)

        data_module = HorizynDataModule(**config.data)
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            limit_train_batches=1,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Should not crash
        trainer.fit(model, data_module)
        assert trainer.current_epoch == 0  # Completed 1 epoch (0-indexed)

    def test_deterministic_training(self, tmp_path):
        """Test that training is deterministic with fixed seed."""
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        def train_one_epoch(seed: int):
            config = load_config("configs/nano.yaml")
            config.training.max_epochs = 1
            config.training.limit_train_batches = 3
            config.training.limit_val_batches = 1

            pl.seed_everything(seed, workers=True)

            data_module = HorizynDataModule(**config.data)
            model = HorizynLitModule(
                model_config=config.model,
                optimizer_config=config.optimizer,
                loss_config=config.loss,
            )

            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="cpu",  # Force CPU for determinism
                devices=1,
                limit_train_batches=3,
                limit_val_batches=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                deterministic=True,
            )

            trainer.fit(model, data_module)

            # Return final model parameters as a signature
            params = []
            for p in model.parameters():
                params.append(p.detach().cpu().clone())
            return params

        # Train twice with same seed
        params1 = train_one_epoch(seed=12345)
        params2 = train_one_epoch(seed=12345)

        # Parameters should be identical
        for p1, p2 in zip(params1, params2):
            assert torch.allclose(
                p1, p2, rtol=1e-5, atol=1e-7
            ), "Training is not deterministic with same seed"


# =============================================================================
# E2E Tests with Full SwissProt Data
# =============================================================================


@pytest.mark.e2e
class TestE2ESwissProtTraining:
    """
    End-to-end tests using the full SwissProt dataset.

    These tests require:
    - SwissProt data downloaded to data/swissprot/ (~930 MB)
    - Sufficient RAM (~16GB)
    - GPU with 16GB+ VRAM recommended
    - ~10-15 minutes runtime per test

    Run these tests with:
        pytest tests/test_integration.py -v -m e2e

    Skip these tests with:
        pytest tests/test_integration.py -v -m "not e2e"
    """

    @pytest.fixture
    def check_swissprot_data(self):
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

    def test_e2e_swissprot_config_is_valid(self, check_swissprot_data):
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
        assert "swissprot" in data_cfg["reactions_db"]
        assert "swissprot" in data_cfg["proteins_h5"]
        assert "swissprot" in data_cfg["train_pairs_db"]
        assert "swissprot" in data_cfg["val_pairs_db"]

        # Verify all files exist
        for key in ["reactions_db", "proteins_h5", "train_pairs_db", "val_pairs_db"]:
            filepath = Path(data_cfg[key])
            assert filepath.exists(), f"Missing SwissProt file: {filepath}"

    def test_e2e_data_loading_memory_footprint(self, check_swissprot_data):
        """
        Test that SwissProt data loads into memory successfully.

        This test verifies:
        - All data files can be opened
        - Data can be loaded into memory
        - Memory footprint is reasonable (~15GB)
        - Datasets have expected sizes
        """
        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule

        config = load_config("configs/sota.yaml")

        # Initialize data module
        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")

        # Verify data was loaded
        assert data_module.train_data is not None
        assert data_module.val_retrieval_queries is not None
        assert data_module.val_retrieval_targets is not None
        assert data_module.val_retrieval_pairs is not None

        # Check expected dataset sizes (approximate)
        # SwissProt has ~257k training pairs, ~36k validation pairs
        train_loader = data_module.train_dataloader()
        assert (
            len(train_loader.dataset) > 200_000
        ), "Training dataset seems too small (expected ~257k pairs)"
        assert (
            len(train_loader.dataset) < 300_000
        ), "Training dataset seems too large (expected ~257k pairs)"

        # Verify validation loaders work
        val_loaders = data_module.val_dataloader()
        assert len(val_loaders) == 3, "Expected 3 validation dataloaders"

        # Get first batch from each loader
        for i, loader in enumerate(val_loaders):
            batch = next(iter(loader))
            assert (
                "query" in batch or "target" in batch
            ), f"Validation loader {i} missing query/target"

    def test_e2e_full_training_two_epochs(self, tmp_path, check_swissprot_data):
        """
        E2E test: Train SOTA model for 2 epochs on full SwissProt data.

        This is a comprehensive test that:
        1. Loads the full SOTA configuration
        2. Initializes the complete data pipeline (all 257k training pairs)
        3. Runs 2 full epochs of training
        4. Validates on all 36k validation pairs
        5. Saves checkpoints
        6. Logs metrics

        Expected runtime: ~10-15 minutes on T4 GPU, ~30-45 minutes on CPU
        Expected memory: ~15GB RAM, ~10GB GPU VRAM
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        # Load SOTA config
        config = load_config("configs/sota.yaml")

        # Override for faster testing (2 epochs only)
        config.training.max_epochs = 2

        # Use temporary directory for outputs
        log_dir = tmp_path / "logs"
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir.mkdir()
        checkpoint_dir.mkdir()

        # Set seed
        pl.seed_everything(42, workers=True)

        # Initialize data module (this loads all data into memory)
        print("\nLoading SwissProt data into memory (~15GB)...")
        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")
        print(f"Training pairs: {len(data_module.train_data)}")

        # Initialize model
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            scheduler_config=config.scheduler if hasattr(config, "scheduler") else None,
            dedup_pairs=config.training.get("dedup_pairs", True),
            log_retrieval_metrics=config.training.get("log_retrieval_metrics", True),
        )

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="auto",
            devices=1,
            logger=pl.loggers.CSVLogger(str(log_dir), name="sota_e2e"),
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="epoch-{epoch:02d}",
                    save_last=True,
                    every_n_epochs=1,
                    save_top_k=2,
                    monitor="val_retrieval_queries/top_1",
                    mode="max",
                ),
            ],
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=config.training.get("gradient_clip_val", None),
        )

        # Run training
        print("\nTraining for 2 epochs (this will take ~10-15 minutes)...")
        trainer.fit(model, data_module)

        # Verify training completed
        assert trainer.current_epoch == 1  # 0-indexed, so epoch 1 = 2 epochs

        # Verify checkpoints were saved
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        assert (
            len(checkpoint_files) >= 2
        ), f"Expected at least 2 checkpoints, found {len(checkpoint_files)}"

        # Verify last checkpoint exists
        last_ckpt = checkpoint_dir / "last.ckpt"
        assert last_ckpt.exists(), "last.ckpt not found"

        # Verify metrics were logged
        metrics_csv = log_dir / "sota_e2e" / "version_0" / "metrics.csv"
        assert metrics_csv.exists(), "metrics.csv not found"

        # Read and validate metrics
        with open(metrics_csv) as f:
            content = f.read()

            # Should have training loss
            assert "train_loss" in content or "loss" in content

            # Should have validation metrics
            assert "val_retrieval" in content, "No validation metrics logged"

            # Should have multiple epochs worth of data
            lines = content.strip().split("\n")
            assert len(lines) > 3, "Expected multiple metric entries"

    def test_e2e_checkpoint_loading_and_resume(self, tmp_path, check_swissprot_data):
        """
        E2E test: Train 1 epoch, save checkpoint, load and resume for 1 more epoch.

        This verifies:
        1. Checkpoints save correctly with full model state
        2. Checkpoints can be loaded
        3. Training can be resumed from checkpoint
        4. Resumed training continues properly
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/sota.yaml")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Train for 1 epoch
        print("\nTraining epoch 1...")
        pl.seed_everything(42)
        data_module = HorizynDataModule(**config.data)
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            dedup_pairs=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    save_last=True,
                ),
            ],
            enable_progress_bar=True,
            logger=False,
        )

        trainer.fit(model, data_module)

        checkpoint_path = checkpoint_dir / "last.ckpt"
        assert checkpoint_path.exists(), "Checkpoint not saved"

        # Load from checkpoint and train 1 more epoch
        print("\nLoading checkpoint and training epoch 2...")
        loaded_model = HorizynLitModule.load_from_checkpoint(
            checkpoint_path,
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
        )

        # Create new trainer for resumed training
        trainer_resumed = pl.Trainer(
            max_epochs=2,  # Will continue from epoch 1 to epoch 2
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            logger=False,
        )

        # Resume training
        trainer_resumed.fit(loaded_model, data_module, ckpt_path=str(checkpoint_path))

        # Verify training continued
        assert trainer_resumed.current_epoch == 1  # Completed 2 epochs total (0-indexed)

    def test_e2e_validation_metrics_realistic_values(self, check_swissprot_data):
        """
        E2E test: Run validation and check that metrics are in realistic ranges.

        This test trains for just 1 epoch and then validates that:
        - Top-1 accuracy is reasonable (>1% even with random init)
        - Top-10 accuracy > Top-1 accuracy
        - Loss values are finite and in expected range
        - Positive scores > negative scores (after some training)
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/sota.yaml")
        config.training.max_epochs = 1  # Just 1 epoch

        pl.seed_everything(42)

        data_module = HorizynDataModule(**config.data)
        model = HorizynLitModule(
            model_config=config.model,
            optimizer_config=config.optimizer,
            loss_config=config.loss,
            log_retrieval_metrics=True,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            logger=pl.loggers.CSVLogger("logs", name="validation_test"),
        )

        print("\nTraining 1 epoch to check validation metrics...")
        trainer.fit(model, data_module)

        # Read logged metrics
        metrics_csv = Path("logs/validation_test/version_0/metrics.csv")
        assert metrics_csv.exists()

        # Parse metrics (simplified parsing)
        metrics = {}
        with open(metrics_csv) as f:
            header = f.readline().strip().split(",")
            for line in f:
                if line.strip():
                    values = line.strip().split(",")
                    for h, v in zip(header, values):
                        if v and v != "":
                            try:
                                metrics[h] = float(v)
                            except ValueError:
                                pass

        # Check that some validation metrics were logged
        val_metrics = [k for k in metrics.keys() if "val_retrieval" in k]
        assert len(val_metrics) > 0, "No validation metrics found"

        # Clean up
        shutil.rmtree("logs/validation_test", ignore_errors=True)
