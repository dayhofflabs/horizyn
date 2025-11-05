"""
Unit tests for the Lightning module.
"""

import pytest
import torch

from horizyn.lightning_module import HorizynLitModule


class TestHorizynLitModule:
    """Tests for the HorizynLitModule class."""

    def test_initialization_default(self):
        """Test lightning module initialization with default parameters."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[2048, 4096, 512],
            target_encoder_dims=[1024, 4096, 512],
        )

        assert lit_module.model is not None
        assert lit_module.loss_fn is not None
        assert lit_module.learning_rate == 1e-4
        assert lit_module.weight_decay == 0.01
        assert lit_module.hparams.embedding_dim == 512
        assert lit_module.hparams.beta == 10.0
        assert lit_module.hparams.learn_beta is False

    def test_initialization_custom(self):
        """Test lightning module initialization with custom parameters."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[100, 200],
            target_encoder_dims=[50, 200],
            embedding_dim=200,
            beta=5.0,
            learn_beta=True,
            learning_rate=1e-3,
            weight_decay=0.1,
            metric_ks=[1, 10],
        )

        assert lit_module.hparams.embedding_dim == 200
        assert lit_module.hparams.beta == 5.0
        assert lit_module.hparams.learn_beta is True
        assert lit_module.learning_rate == 1e-3
        assert lit_module.weight_decay == 0.1
        assert lit_module.hparams.metric_ks == [1, 10]

    def test_forward(self):
        """Test forward pass through the model."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
        )

        query_vec = torch.randn(4, 128)
        target_vec = torch.randn(4, 256)

        query_embeds, target_embeds = lit_module(query_vec, target_vec)

        assert query_embeds.shape == (4, 64)
        assert target_embeds.shape == (4, 64)

        # Check embeddings are normalized
        query_norms = torch.norm(query_embeds, dim=1)
        target_norms = torch.norm(target_embeds, dim=1)
        assert torch.allclose(query_norms, torch.ones(4), atol=1e-5)
        assert torch.allclose(target_norms, torch.ones(4), atol=1e-5)

    def test_compute_cosine_distances(self):
        """Test cosine distance computation."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128],
            target_encoder_dims=[64, 128],
            embedding_dim=128,
        )

        # Create normalized vectors
        query_vecs = torch.randn(3, 128)
        query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

        target_vecs = torch.randn(5, 128)
        target_vecs = torch.nn.functional.normalize(target_vecs, dim=1)

        dists = lit_module._compute_cosine_distances(query_vecs, target_vecs)

        # Check shape
        assert dists.shape == (3, 5)

        # Check distance is in valid range [0, 2]
        assert torch.all(dists >= 0)
        assert torch.all(dists <= 2)

    def test_deduplicate_inputs(self):
        """Test input deduplication."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128],
            target_encoder_dims=[64, 128],
        )

        vecs = torch.randn(5, 64)
        ids = ["a", "b", "a", "c", "b"]

        unique_vecs, inverse_indices = lit_module._deduplicate_inputs(vecs, ids)

        # Should have 3 unique vectors (a, b, c)
        assert unique_vecs.shape == (3, 64)
        assert inverse_indices.shape == (5,)

        # Check that we can reconstruct original order
        reconstructed = unique_vecs[inverse_indices]
        # Check that duplicates have the same vector
        assert torch.allclose(reconstructed[0], reconstructed[2])  # Both "a"
        assert torch.allclose(reconstructed[1], reconstructed[4])  # Both "b"

    def test_training_step(self):
        """Test training step computation."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
        )

        # Create mock batch
        batch = {
            "query_vec": torch.randn(8, 128),
            "target_vec": torch.randn(8, 256),
            "query_id": ["q1", "q2", "q3", "q1", "q2", "q4", "q5", "q6"],
            "target_id": ["t1", "t2", "t1", "t2", "t3", "t4", "t5", "t6"],
        }

        # Run training step
        loss = lit_module.training_step(batch, batch_idx=0)

        # Check loss is a scalar
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_training_step_with_learnable_beta(self):
        """Test training step with learnable beta."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
            learn_beta=True,
        )

        batch = {
            "query_vec": torch.randn(4, 128),
            "target_vec": torch.randn(4, 256),
            "query_id": ["q1", "q2", "q3", "q4"],
            "target_id": ["t1", "t2", "t3", "t4"],
        }

        loss = lit_module.training_step(batch, batch_idx=0)

        assert loss.dim() == 0
        assert loss.item() >= 0

        # Beta should be a learnable parameter
        assert lit_module.loss_fn.learn_beta is True
        assert lit_module.loss_fn.logbeta.requires_grad is True

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128],
            target_encoder_dims=[64, 128],
            learning_rate=1e-3,
            weight_decay=0.05,
        )

        optimizer = lit_module.configure_optimizers()

        # Check optimizer type
        assert isinstance(optimizer, torch.optim.AdamW)

        # Check learning rate
        assert optimizer.defaults["lr"] == 1e-3

        # Check weight decay
        assert optimizer.defaults["weight_decay"] == 0.05

    def test_validation_loss_step(self):
        """Test validation loss computation."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
        )

        batch = {
            "query_vec": torch.randn(8, 128),
            "target_vec": torch.randn(8, 256),
            "query_id": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"],
            "target_id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
        }

        loss = lit_module._validation_loss_step(batch, batch_idx=0)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_validation_lookup_step(self):
        """Test target lookup table building."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
        )

        # Initialize target lookup table
        lit_module.target_lookup_table = torch.zeros(10, 64)
        lit_module.num_targets = 10

        # Mock trainer for world_size
        class MockTrainer:
            world_size = 1

            class MockStrategy:
                @staticmethod
                def barrier():
                    pass

                @staticmethod
                def all_gather(data, **kwargs):
                    """Mock all_gather that just returns the data as-is for single process."""
                    return data

            strategy = MockStrategy()

        lit_module.trainer = MockTrainer()

        batch = {
            "target_vec": torch.randn(3, 256),
            "target_lookup_row_idx": torch.tensor([0, 5, 9]),
        }

        lit_module._validation_lookup_step(batch, batch_idx=0)

        # Check that lookup table was updated
        assert not torch.all(lit_module.target_lookup_table == 0)

        # Check that the specified rows were updated
        assert not torch.all(lit_module.target_lookup_table[0] == 0)
        assert not torch.all(lit_module.target_lookup_table[5] == 0)
        assert not torch.all(lit_module.target_lookup_table[9] == 0)

        # Check that unspecified rows are still zero
        assert torch.all(lit_module.target_lookup_table[1] == 0)
        assert torch.all(lit_module.target_lookup_table[7] == 0)

    def test_validation_retrieval_step(self):
        """Test retrieval metrics computation."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[128, 256, 64],
            target_encoder_dims=[256, 256, 64],
            embedding_dim=64,
            metric_ks=[1, 5, 10],
        )

        # Create target lookup table
        lit_module.target_lookup_table = torch.randn(20, 64)
        lit_module.target_lookup_table = torch.nn.functional.normalize(
            lit_module.target_lookup_table, dim=1
        )

        batch = {
            "query_vec": torch.randn(4, 128),
            "target_id": [
                torch.tensor([0, 3]),  # Query 0 should retrieve targets 0 and 3
                torch.tensor([5]),  # Query 1 should retrieve target 5
                torch.tensor([10, 11, 12]),  # Query 2 should retrieve targets 10-12
                torch.tensor([19]),  # Query 3 should retrieve target 19
            ],
        }

        # Run retrieval step (no exceptions should be raised)
        lit_module._validation_retrieval_step(batch, batch_idx=0)

    def test_sota_configuration(self):
        """Test SOTA model configuration."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[2048, 4096, 512],
            target_encoder_dims=[1024, 4096, 512],
            embedding_dim=512,
            beta=10.0,
            learn_beta=False,
            learning_rate=1e-4,
            weight_decay=0.01,
            metric_ks=[1, 5, 10, 50],
        )

        # Check model dimensions
        assert lit_module.model.query_encoder.input_dim == 2048
        assert lit_module.model.query_encoder.output_dim == 512
        assert lit_module.model.target_encoder.input_dim == 1024
        assert lit_module.model.target_encoder.output_dim == 512
        assert lit_module.hparams.embedding_dim == 512

        # Check loss configuration
        assert lit_module.loss_fn.beta_init == 10.0
        assert lit_module.loss_fn.learn_beta is False

        # Check optimizer configuration
        assert lit_module.learning_rate == 1e-4
        assert lit_module.weight_decay == 0.01

        # Check metrics
        expected_metrics = [
            "top_1",
            "top_5",
            "top_10",
            "top_50",
            "mrr",
        ]
        for metric_name in expected_metrics:
            assert metric_name in lit_module.metric_functionals

    def test_gradient_flow(self):
        """Test gradient flow through training step."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128, 32],
            target_encoder_dims=[64, 128, 32],
            embedding_dim=32,
        )

        batch = {
            "query_vec": torch.randn(4, 64),
            "target_vec": torch.randn(4, 64),
            "query_id": ["q1", "q2", "q3", "q4"],
            "target_id": ["t1", "t2", "t3", "t4"],
        }

        loss = lit_module.training_step(batch, batch_idx=0)
        loss.backward()

        # Check that gradients exist
        for name, param in lit_module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_device_placement(self):
        """Test model works on CPU (GPU test would require GPU)."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128],
            target_encoder_dims=[64, 128],
        )

        # Move to CPU
        lit_module = lit_module.to("cpu")

        batch = {
            "query_vec": torch.randn(2, 64),
            "target_vec": torch.randn(2, 64),
            "query_id": ["q1", "q2"],
            "target_id": ["t1", "t2"],
        }

        loss = lit_module.training_step(batch, batch_idx=0)

        assert loss.device.type == "cpu"

    def test_hyperparameter_saving(self):
        """Test that hyperparameters are saved."""
        lit_module = HorizynLitModule(
            query_encoder_dims=[64, 128],
            target_encoder_dims=[64, 128],
            embedding_dim=128,
            beta=5.0,
            learning_rate=1e-3,
        )

        # Check that hyperparameters are saved
        assert hasattr(lit_module, "hparams")
        assert lit_module.hparams.query_encoder_dims == [64, 128]
        assert lit_module.hparams.target_encoder_dims == [64, 128]
        assert lit_module.hparams.embedding_dim == 128
        assert lit_module.hparams.beta == 5.0
        assert lit_module.hparams.learning_rate == 1e-3
