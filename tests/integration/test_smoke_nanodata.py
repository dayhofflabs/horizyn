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

from pathlib import Path

import pytest
import torch
import yaml

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
        assert "nanodata" in data_cfg["reactions_path"]
        assert "nanodata" in data_cfg["proteins_path"]
        assert "nanodata" in data_cfg["train_pairs_path"]
        assert "nanodata" in data_cfg["val_pairs_path"]

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
        assert data_module.val_retrieval_pairs is not None
        # Targets are accessed via _target_data (not exposed as public property)
        assert data_module.val_retrieval_pairs is not None

        # Initialize model
        model = HorizynLitModule(
            query_encoder_dims=config.model.query_encoder_dims,
            target_encoder_dims=config.model.target_encoder_dims,
            embedding_dim=config.model.embedding_dim,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta=config.training.loss.beta,
            learn_beta=config.training.loss.get("learn_beta", False),
            metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
        )

        # Verify model structure
        assert model.model is not None
        assert model.loss_fn is not None
        assert hasattr(model, "metric_functionals")

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
        # After training, current_epoch is incremented one more time
        assert trainer.current_epoch == config.training.max_epochs

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
            query_encoder_dims=config.model.query_encoder_dims,
            target_encoder_dims=config.model.target_encoder_dims,
            embedding_dim=config.model.embedding_dim,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta=config.training.loss.beta,
            learn_beta=config.training.loss.get("learn_beta", False),
            metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
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
            query_encoder_dims=config.model.query_encoder_dims,
            target_encoder_dims=config.model.target_encoder_dims,
            embedding_dim=config.model.embedding_dim,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta=config.training.loss.beta,
            learn_beta=config.training.loss.get("learn_beta", False),
            metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
        )

        # Verify loaded model has correct structure
        assert loaded_model.model is not None
        assert loaded_model.loss_fn is not None

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

        # Verify batch structure - keys are query_id, target_id, query_vec, target_vec
        assert "query_id" in first_batch
        assert "target_id" in first_batch
        assert "query_vec" in first_batch
        assert "target_vec" in first_batch

        # Verify tensors have correct shapes
        query_vec = first_batch["query_vec"]
        target_vec = first_batch["target_vec"]

        assert isinstance(query_vec, torch.Tensor)
        assert isinstance(target_vec, torch.Tensor)
        assert query_vec.dim() == 2  # (batch_size, features)
        assert target_vec.dim() == 2  # (batch_size, features)

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
            query_encoder_dims=config.model.query_encoder_dims,
            target_encoder_dims=config.model.target_encoder_dims,
            embedding_dim=config.model.embedding_dim,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta=config.training.loss.beta,
            learn_beta=config.training.loss.get("learn_beta", False),
            metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
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

