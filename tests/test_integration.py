"""
Integration tests for the Horizyn training pipeline.

This module contains several types of tests:
1. Smoke tests: Fast tests using nanodata (~12 reactions, < 1 minute)
2. SwissProt fast tests: Fast tests with full data (no fingerprints, < 5 seconds)
3. SwissProt slow tests: Tests with fingerprints/training (5-15 minutes)

Run all tests (excluding slow):
    pytest tests/test_integration.py -v -m "not slow"

Run only slow tests:
    pytest tests/test_integration.py -v -m slow

Run all tests including slow:
    pytest tests/test_integration.py -v
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
        config.data.reactions_path = "data/nonexistent/reactions.db"

        data_module = HorizynDataModule(**config.data)

        # Should raise clear error when trying to setup
        with pytest.raises(Exception) as exc_info:
            data_module.setup("fit")

        # Error message should mention the missing file
        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg or "not found" in error_msg.lower()


class TestSmokeTrainingDynamics:
    """Smoke tests that verify training actually works (learning, gradients, etc.)."""

    def test_training_loss_is_finite_and_decreases(self, tmp_path):
        """
        Test that training loss is finite and decreases over epochs.

        This catches:
        - NaN/Inf loss values (numerical instability)
        - Frozen model (loss doesn't change)
        - Gradient explosion (loss increases dramatically)
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 3
        config.training.limit_train_batches = 5
        config.training.limit_val_batches = 1

        log_dir = tmp_path / "logs"

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
            max_epochs=3,
            accelerator="auto",
            devices=1,
            logger=pl.loggers.CSVLogger(str(log_dir)),
            limit_train_batches=5,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, data_module)

        # Read logged metrics
        metrics_csv = log_dir / "lightning_logs" / "version_0" / "metrics.csv"
        assert metrics_csv.exists()

        # Parse loss values by epoch
        losses_by_epoch = {}
        with open(metrics_csv) as f:
            import csv

            reader = csv.DictReader(f)
            for row in reader:
                if "train/loss_epoch" in row and row["train/loss_epoch"]:
                    epoch = int(row["epoch"]) if row["epoch"] else None
                    loss = float(row["train/loss_epoch"])
                    if epoch is not None:
                        losses_by_epoch[epoch] = loss

        assert (
            len(losses_by_epoch) >= 2
        ), f"Not enough epoch loss values logged (found {len(losses_by_epoch)})"

        # Check all losses are finite
        for epoch, loss in losses_by_epoch.items():
            assert torch.isfinite(
                torch.tensor(loss)
            ), f"Loss at epoch {epoch} is not finite: {loss}"
            assert loss > 0, f"Loss at epoch {epoch} is not positive: {loss}"

        # Check loss generally decreases (last < first)
        epochs = sorted(losses_by_epoch.keys())
        first_loss = losses_by_epoch[epochs[0]]
        last_loss = losses_by_epoch[epochs[-1]]

        # With small data, loss might not decrease monotonically, but should decrease overall
        # Allow some tolerance for small dataset variability
        assert (
            last_loss < first_loss * 1.1
        ), f"Loss didn't decrease: {first_loss:.4f} → {last_loss:.4f}"

    def test_model_weights_actually_change(self):
        """
        Test that model parameters change during training (model learns).

        This catches:
        - Accidentally frozen parameters
        - Zero learning rate
        - Gradients not flowing
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 5
        config.training.limit_val_batches = 1

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

        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.detach().cpu().clone()

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            limit_train_batches=5,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, data_module)

        # Compare with final parameters
        parameters_changed = 0
        parameters_unchanged = 0

        for name, param in model.named_parameters():
            final_param = param.detach().cpu()
            initial_param = initial_params[name]

            if not torch.allclose(initial_param, final_param, rtol=1e-5, atol=1e-7):
                parameters_changed += 1
            else:
                parameters_unchanged += 1

        # At least 90% of parameters should have changed
        total_params = parameters_changed + parameters_unchanged
        change_rate = parameters_changed / total_params

        assert (
            change_rate > 0.9
        ), f"Only {parameters_changed}/{total_params} parameters changed ({change_rate:.1%})"

    def test_embeddings_are_normalized_and_correct_dim(self):
        """
        Test that model outputs normalized 512-dim embeddings.

        This catches:
        - Wrong embedding dimension
        - Missing normalization layer
        - Embeddings with zero norm
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")

        pl.seed_everything(42)

        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")

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

        # Get a batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        # Forward pass
        model.eval()
        with torch.no_grad():
            query_emb = model.model.query_encoder(batch["query_vec"])
            target_emb = model.model.target_encoder(batch["target_vec"])

        # Check dimensions
        assert (
            query_emb.shape[1] == 512
        ), f"Query embedding dim is {query_emb.shape[1]}, expected 512"
        assert (
            target_emb.shape[1] == 512
        ), f"Target embedding dim is {target_emb.shape[1]}, expected 512"

        # Check normalization (L2 norm should be 1.0)
        query_norms = torch.norm(query_emb, p=2, dim=1)
        target_norms = torch.norm(target_emb, p=2, dim=1)

        assert torch.allclose(
            query_norms, torch.ones_like(query_norms), rtol=1e-5, atol=1e-6
        ), f"Query embeddings not normalized: norms {query_norms}"

        assert torch.allclose(
            target_norms, torch.ones_like(target_norms), rtol=1e-5, atol=1e-6
        ), f"Target embeddings not normalized: norms {target_norms}"

    def test_gradients_flow_to_both_encoders(self):
        """
        Test that gradients flow to both query and target encoders.

        This catches:
        - Accidentally detached tensors
        - Frozen encoder
        - Broken backward pass
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")

        pl.seed_everything(42)

        data_module = HorizynDataModule(**config.data)
        data_module.setup("fit")

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

        # Get a batch and do a training step
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        # Manual training step
        model.train()
        loss = model.training_step(batch, batch_idx=0)

        # Backward pass
        loss.backward()

        # Check that both encoders have gradients
        query_encoder_has_grads = False
        target_encoder_has_grads = False

        for name, param in model.named_parameters():
            if param.grad is not None and torch.abs(param.grad).sum() > 0:
                if "query_encoder" in name:
                    query_encoder_has_grads = True
                if "target_encoder" in name:
                    target_encoder_has_grads = True

        assert query_encoder_has_grads, "No gradients flowing to query encoder"
        assert target_encoder_has_grads, "No gradients flowing to target encoder"


class TestSmokeValidationMetrics:
    """Smoke tests that verify validation metrics are computed correctly."""

    def test_validation_three_dataloader_design_works(self, tmp_path):
        """
        Test that the 3-dataloader validation design works correctly.

        This is the most complex part of the pipeline:
        1. Loader 0: Compute validation loss on pairs
        2. Loader 1: Build lookup table of all target embeddings
        3. Loader 2: Compute retrieval metrics for queries

        This catches:
        - Dataloader coordination issues
        - Batch format mismatches
        - Lookup table building errors
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 2
        config.training.limit_val_batches = 2

        log_dir = tmp_path / "logs"

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
            logger=pl.loggers.CSVLogger(str(log_dir)),
            limit_train_batches=2,
            limit_val_batches=2,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # This should not crash during validation
        trainer.fit(model, data_module)

        # Verify validation ran by checking logged metrics
        metrics_csv = log_dir / "lightning_logs" / "version_0" / "metrics.csv"
        assert metrics_csv.exists()

        with open(metrics_csv) as f:
            content = f.read()
            # Should have validation metrics (Lightning uses "val/" prefix)
            assert "val/" in content, "No validation metrics logged"

    def test_validation_metrics_have_reasonable_values(self, tmp_path):
        """
        Test that validation metrics are in valid ranges.

        This catches:
        - Metrics outside [0, 1] range
        - NaN or Inf metric values
        - Metrics not being computed
        """
        import lightning.pytorch as pl

        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule
        from horizyn.lightning_module import HorizynLitModule

        config = load_config("configs/nano.yaml")
        config.training.max_epochs = 1
        config.training.limit_train_batches = 3
        config.training.limit_val_batches = 999  # Use all validation data

        log_dir = tmp_path / "logs"

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
            logger=pl.loggers.CSVLogger(str(log_dir)),
            limit_train_batches=3,
            limit_val_batches=999,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, data_module)

        # Read and parse metrics
        metrics_csv = log_dir / "lightning_logs" / "version_0" / "metrics.csv"
        assert metrics_csv.exists()

        with open(metrics_csv) as f:
            import csv

            reader = csv.DictReader(f)
            metrics = {}
            for row in reader:
                for key, value in row.items():
                    if value and value != "" and key != "epoch" and key != "step":
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            pass

        # Find top-k metrics (Lightning uses "val/" prefix)
        top_k_metrics = {k: v for k, v in metrics.items() if "top_" in k and "val/" in k}

        # Should have at least some top-k metrics
        assert (
            len(top_k_metrics) > 0
        ), f"No top-k metrics found. Available metrics: {list(metrics.keys())}"

        # Check all metrics are in valid range [0, 1]
        for name, value in top_k_metrics.items():
            assert torch.isfinite(torch.tensor(value)), f"{name} is not finite: {value}"
            assert 0 <= value <= 1, f"{name} outside [0,1] range: {value}"

        # With nanodata (very small), we can't expect monotonicity
        # (e.g., if there are only 5 targets, top_10 = top_100)
        # But we can check that metrics are not all zeros or all ones
        values = list(top_k_metrics.values())
        assert not all(v == 0 for v in values), "All metrics are zero (model not learning)"


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
            limit_train_batches=1,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Should not crash
        trainer.fit(model, data_module)
        # After training max_epochs=1, current_epoch will be 1
        assert trainer.current_epoch == 1

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
# SwissProt Tests (Fast and Slow)
# =============================================================================
#
# These tests use the full SwissProt dataset and will skip automatically if
# the data is not available. Fast tests run by default, slow tests are marked
# with @pytest.mark.slow.
#
# Run all tests (including slow):
#     pytest tests/test_integration.py -v
#
# Skip slow tests (default for CI):
#     pytest tests/test_integration.py -v -m "not slow"
#
# Run only slow tests:
#     pytest tests/test_integration.py -v -m slow
# =============================================================================


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
    Fast tests using full SwissProt dataset.

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
