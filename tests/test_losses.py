"""
Unit tests for loss functions.
"""

import pytest
import torch
from horizyn.losses import FullBatchNCELoss, FullBatchMLNCELoss


class TestFullBatchNCELoss:
    """Tests for the FullBatchNCELoss base class."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = FullBatchNCELoss()
        assert loss_fn.beta_init == 1.0
        assert loss_fn.learn_beta is False
        assert loss_fn.beta_min == -float("inf")
        assert loss_fn.beta_max == float("inf")
        assert torch.allclose(loss_fn.beta, torch.tensor(1.0))

    def test_initialization_custom_beta(self):
        """Test initialization with custom beta value."""
        loss_fn = FullBatchNCELoss(beta=10.0)
        assert loss_fn.beta_init == 10.0
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0), rtol=1e-5)

    def test_beta_stored_in_log_space(self):
        """Test that beta is stored in log space."""
        loss_fn = FullBatchNCELoss(beta=5.0)
        expected_logbeta = torch.log(torch.tensor(5.0))
        assert torch.allclose(loss_fn.logbeta, expected_logbeta)

    def test_beta_property(self):
        """Test beta property converts from log space."""
        loss_fn = FullBatchNCELoss(beta=3.0)
        assert torch.allclose(loss_fn.beta, torch.tensor(3.0))

    def test_learn_beta_false(self):
        """Test that beta is not learnable when learn_beta is False."""
        loss_fn = FullBatchNCELoss(beta=2.0, learn_beta=False)
        assert loss_fn.logbeta.requires_grad is False

    def test_learn_beta_true(self):
        """Test that beta is learnable when learn_beta is True."""
        loss_fn = FullBatchNCELoss(beta=2.0, learn_beta=True)
        assert loss_fn.logbeta.requires_grad is True

    def test_beta_clipping_hook_registered(self):
        """Test that beta clipping hook is registered when learn_beta is True."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=1.0, beta_max=10.0)
        # Check that a forward pre-hook is registered
        assert len(loss_fn._forward_pre_hooks) > 0

    def test_beta_clipping_hook_not_registered(self):
        """Test that beta clipping hook is not registered when learn_beta is False."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=False, beta_min=1.0, beta_max=10.0)
        assert len(loss_fn._forward_pre_hooks) == 0

    def test_beta_clipping_min(self):
        """Test that beta is clipped to minimum value."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=2.0, beta_max=10.0)
        # Manually set beta below minimum
        loss_fn.logbeta.data = torch.log(torch.tensor(0.5))
        
        # Trigger the clipping hook via a forward pass
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])
        
        try:
            loss_fn(dists, query_idx, target_idx)
        except NotImplementedError:
            pass  # Expected since forward is not implemented
        
        # Beta should be clipped to beta_min
        assert torch.allclose(loss_fn.beta, torch.tensor(2.0), atol=1e-5)

    def test_beta_clipping_max(self):
        """Test that beta is clipped to maximum value."""
        loss_fn = FullBatchNCELoss(beta=5.0, learn_beta=True, beta_min=1.0, beta_max=8.0)
        # Manually set beta above maximum
        loss_fn.logbeta.data = torch.log(torch.tensor(20.0))
        
        # Trigger the clipping hook via a forward pass
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])
        
        try:
            loss_fn(dists, query_idx, target_idx)
        except NotImplementedError:
            pass  # Expected since forward is not implemented
        
        # Beta should be clipped to beta_max
        assert torch.allclose(loss_fn.beta, torch.tensor(8.0), atol=1e-5)

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        loss_fn = FullBatchNCELoss()
        dists = torch.randn(2, 2)
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])
        
        with pytest.raises(NotImplementedError):
            loss_fn(dists, query_idx, target_idx)


class TestFullBatchMLNCELoss:
    """Tests for the FullBatchMLNCELoss class."""

    def test_initialization(self):
        """Test basic initialization."""
        loss_fn = FullBatchMLNCELoss(beta=10.0, learn_beta=False)
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0))
        assert loss_fn.learn_beta is False

    def test_forward_single_pair(self):
        """Test forward pass with a single positive pair."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)
        
        # Simple 2x2 distance matrix
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0])
        target_idx = torch.tensor([0])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        # Loss should be a scalar
        assert loss.dim() == 0
        # Loss should be positive (we're minimizing distance + partition)
        assert loss.item() > 0

    def test_forward_multiple_pairs(self):
        """Test forward pass with multiple positive pairs."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)
        
        # 3x3 distance matrix
        dists = torch.tensor([
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1]
        ])
        query_idx = torch.tensor([0, 1, 2])
        target_idx = torch.tensor([0, 1, 2])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_forward_multi_label(self):
        """Test forward pass with multiple targets per query (multi-label)."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)
        
        # 2x3 distance matrix
        dists = torch.tensor([
            [0.1, 0.2, 0.9],
            [0.9, 0.9, 0.1]
        ])
        # Query 0 has two positive targets (0 and 1)
        query_idx = torch.tensor([0, 0, 1])
        target_idx = torch.tensor([0, 1, 2])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_forward_shape_consistency(self):
        """Test that forward works with different batch sizes."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        
        # Test with various sizes
        for num_queries, num_targets, num_pairs in [(5, 10, 5), (10, 5, 10), (20, 20, 30)]:
            dists = torch.randn(num_queries, num_targets).abs()
            query_idx = torch.randint(0, num_queries, (num_pairs,))
            target_idx = torch.randint(0, num_targets, (num_pairs,))
            
            loss = loss_fn(dists, query_idx, target_idx)
            assert loss.dim() == 0

    def test_gradient_flow_through_dists(self):
        """Test that gradients flow through the distance matrix."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=False)
        
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]], requires_grad=True)
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        loss = loss_fn(dists, query_idx, target_idx)
        loss.backward()
        
        assert dists.grad is not None
        assert not torch.allclose(dists.grad, torch.zeros_like(dists))

    def test_gradient_flow_through_beta(self):
        """Test that gradients flow through beta when learnable."""
        loss_fn = FullBatchMLNCELoss(beta=5.0, learn_beta=True)
        
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        loss = loss_fn(dists, query_idx, target_idx)
        loss.backward()
        
        assert loss_fn.logbeta.grad is not None
        assert not torch.allclose(loss_fn.logbeta.grad, torch.tensor(0.0))

    def test_beta_effect_on_loss(self):
        """Test that beta affects the loss (not necessarily monotonically)."""
        dists = torch.tensor([[0.2, 0.8], [0.8, 0.2]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        # Different beta values should produce different losses
        loss_fn_low = FullBatchMLNCELoss(beta=1.0)
        loss_fn_high = FullBatchMLNCELoss(beta=10.0)
        
        loss_low = loss_fn_low(dists, query_idx, target_idx)
        loss_high = loss_fn_high(dists, query_idx, target_idx)
        
        # Both should be positive
        assert loss_low.item() > 0
        assert loss_high.item() > 0
        # Losses should be different (beta changes the scale)
        assert not torch.allclose(loss_low, loss_high, rtol=0.1)

    def test_learned_beta_updates(self):
        """Test that learned beta can be updated via gradient descent."""
        loss_fn = FullBatchMLNCELoss(beta=1.0, learn_beta=True)
        optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1)
        
        initial_beta = loss_fn.beta.item()
        
        # Perform a few optimization steps
        for _ in range(5):
            dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
            query_idx = torch.tensor([0, 1])
            target_idx = torch.tensor([0, 1])
            
            optimizer.zero_grad()
            loss = loss_fn(dists, query_idx, target_idx)
            loss.backward()
            optimizer.step()
        
        final_beta = loss_fn.beta.item()
        
        # Beta should have changed
        assert abs(final_beta - initial_beta) > 1e-3

    def test_beta_clipping_during_training(self):
        """Test that beta is clipped during training when constraints are set."""
        loss_fn = FullBatchMLNCELoss(
            beta=5.0, learn_beta=True, beta_min=1.0, beta_max=10.0
        )
        
        # Manually set beta to violate constraints
        loss_fn.logbeta.data = torch.log(torch.tensor(0.1))  # Below min
        
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        # Beta should be clipped to minimum
        assert loss_fn.beta.item() >= 1.0 - 1e-5

    def test_device_placement(self, device):
        """Test that loss function works on both CPU and GPU."""
        loss_fn = FullBatchMLNCELoss(beta=1.0).to(device)
        
        dists = torch.tensor([[0.1, 0.9], [0.9, 0.1]]).to(device)
        query_idx = torch.tensor([0, 1]).to(device)
        target_idx = torch.tensor([0, 1]).to(device)
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert loss.device.type == device.type

    def test_loss_decreases_with_better_distances(self):
        """Test that loss is lower when positive pairs have smaller distances."""
        loss_fn = FullBatchMLNCELoss(beta=10.0)
        
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        # Good distances: positive pairs are close
        dists_good = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        loss_good = loss_fn(dists_good, query_idx, target_idx)
        
        # Bad distances: positive pairs are far
        dists_bad = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        loss_bad = loss_fn(dists_bad, query_idx, target_idx)
        
        # Loss should be lower for good distances
        assert loss_good.item() < loss_bad.item()

    def test_all_pairs_positive(self):
        """Test edge case where all pairs are positive."""
        loss_fn = FullBatchMLNCELoss(beta=1.0)
        
        dists = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        # All 4 pairs are positive
        query_idx = torch.tensor([0, 0, 1, 1])
        target_idx = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_numerical_stability_extreme_distances(self):
        """Test numerical stability with very large and small distances."""
        loss_fn = FullBatchMLNCELoss(beta=100.0)
        
        # Mix of very small and large distances
        dists = torch.tensor([[1e-6, 10.0], [10.0, 1e-6]])
        query_idx = torch.tensor([0, 1])
        target_idx = torch.tensor([0, 1])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        # Should not be NaN or inf
        assert torch.isfinite(loss)

    def test_consistency_with_cosine_distance(self):
        """Test that loss works with cosine distance (1 - cosine_similarity)."""
        loss_fn = FullBatchMLNCELoss(beta=10.0)
        
        # Normalized embeddings
        query_embeds = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        target_embeds = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        
        # Cosine distance
        cosine_sim = torch.mm(query_embeds, target_embeds.t())
        dists = 1.0 - cosine_sim
        
        query_idx = torch.tensor([0, 1, 2])
        target_idx = torch.tensor([0, 1, 2])
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_sota_configuration(self):
        """Test with SOTA paper configuration."""
        # From swissprot_sota.yaml
        loss_fn = FullBatchMLNCELoss(
            beta=10.0,
            learn_beta=False,
            beta_min=0.01,
            beta_max=100.0,
        )
        
        # Simulate a batch with query and target encoders
        batch_size = 16
        query_dim = 2048  # RDKit+ (1024) + DRFP (1024)
        target_dim = 1024  # T5 embeddings
        embed_dim = 512
        
        # Mock embeddings (normalized)
        query_embeds = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=1)
        target_embeds = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=1)
        
        # Cosine distance
        dists = 1.0 - torch.mm(query_embeds, target_embeds.t())
        
        # Assume diagonal pairs are positive (simple case)
        query_idx = torch.arange(batch_size)
        target_idx = torch.arange(batch_size)
        
        loss = loss_fn(dists, query_idx, target_idx)
        
        assert torch.isfinite(loss)
        assert loss.item() > 0
        # Verify beta is as configured
        assert torch.allclose(loss_fn.beta, torch.tensor(10.0))

