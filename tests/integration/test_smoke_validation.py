"""Smoke tests that verify validation metrics are computed correctly."""

import pytest
import torch


pytestmark = pytest.mark.integration


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

