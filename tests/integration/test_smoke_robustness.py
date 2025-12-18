"""Smoke tests for edge cases and robustness."""

import pytest
import torch


pytestmark = pytest.mark.integration


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
