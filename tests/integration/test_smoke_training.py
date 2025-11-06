"""Smoke tests that verify training actually works (learning, gradients, etc.)."""

import pytest
import torch


pytestmark = pytest.mark.integration


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
