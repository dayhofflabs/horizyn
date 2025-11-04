"""
Unit tests for retrieval metrics.
"""

import pytest
import torch
from horizyn.metrics import (
    RetrievalMetric,
    top_k_hit_rate,
    mean_reciprocal_rank,
    positive_score,
    negative_score,
    create_retrieval_metrics,
)


class TestTopKHitRate:
    """Tests for the top_k_hit_rate function."""

    def test_perfect_prediction_top1(self):
        """Test with perfect top-1 prediction."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])  # Target at index 1 (highest score)
        
        result = top_k_hit_rate(scores, target_idx, k=1)
        assert result == 1.0

    def test_miss_top1(self):
        """Test with target not in top-1."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])  # Target at index 2 (3rd highest)
        
        result = top_k_hit_rate(scores, target_idx, k=1)
        assert result == 0.0

    def test_hit_in_top3(self):
        """Test with target in top-3."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])  # Target at index 2
        
        result = top_k_hit_rate(scores, target_idx, k=3)
        assert result == 1.0

    def test_multiple_targets_one_hit(self):
        """Test with multiple targets where one is in top-K."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 3, -1])  # Targets at indices 0 (low) and 3 (high)
        
        result = top_k_hit_rate(scores, target_idx, k=2)
        assert result == 1.0  # Index 3 is in top-2

    def test_multiple_targets_no_hit(self):
        """Test with multiple targets where none are in top-K."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 4, -1])  # Targets at indices 0 and 4 (both low)
        
        result = top_k_hit_rate(scores, target_idx, k=2)
        assert result == 0.0  # Neither 0 nor 4 are in top-2

    def test_empty_targets(self):
        """Test with no valid targets (all padding)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])
        
        result = top_k_hit_rate(scores, target_idx, k=3)
        assert result == 0.0

    def test_k_larger_than_items(self):
        """Test with k larger than number of items."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([0, -1, -1])
        
        result = top_k_hit_rate(scores, target_idx, k=100)
        assert result == 1.0  # All items considered

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])
        
        with pytest.raises(ValueError, match="expects 1D tensors"):
            top_k_hit_rate(scores, target_idx, k=1)


class TestMeanReciprocalRank:
    """Tests for the mean_reciprocal_rank function."""

    def test_first_rank(self):
        """Test when target has rank 1 (highest score)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])  # Target at index 1 (rank 1)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert torch.allclose(result, torch.tensor(1.0))

    def test_second_rank(self):
        """Test when target has rank 2."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([3, -1, -1])  # Target at index 3 (rank 2)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.5))

    def test_third_rank(self):
        """Test when target has rank 3."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])  # Target at index 2 (rank 3)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert torch.allclose(result, torch.tensor(1.0 / 3.0))

    def test_multiple_targets_best_rank(self):
        """Test with multiple targets (should use best rank)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 3, -1])  # Targets at indices 0 (rank 5) and 3 (rank 2)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.5))  # Best rank is 2, so 1/2

    def test_empty_targets(self):
        """Test with no valid targets."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert result == 0.0

    def test_last_rank(self):
        """Test when target has the worst rank."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, -1, -1])  # Target at index 0 (rank 5, lowest)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.2))

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])
        
        with pytest.raises(ValueError, match="expects 1D tensors"):
            mean_reciprocal_rank(scores, target_idx)


class TestPositiveScore:
    """Tests for the positive_score function."""

    def test_single_positive(self):
        """Test with a single positive item."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])
        
        result = positive_score(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.9))

    def test_multiple_positives(self):
        """Test with multiple positive items."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 3, -1])
        
        result = positive_score(scores, target_idx)
        assert torch.allclose(result, torch.tensor((0.9 + 0.8) / 2))

    def test_all_positives(self):
        """Test when all items are positive."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 1, 2, 3, 4])
        
        result = positive_score(scores, target_idx)
        assert torch.allclose(result, scores.mean())

    def test_empty_targets(self):
        """Test with no valid targets."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])
        
        result = positive_score(scores, target_idx)
        assert result == 0.0

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])
        
        with pytest.raises(ValueError, match="expects 1D tensors"):
            positive_score(scores, target_idx)


class TestNegativeScore:
    """Tests for the negative_score function."""

    def test_single_negative(self):
        """Test with mostly positive items."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 2, 3, 4, -1])  # Index 0 is negative
        
        result = negative_score(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.1))

    def test_multiple_negatives(self):
        """Test with multiple negative items."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 3, -1])  # Indices 0, 2, 4 are negative
        
        result = negative_score(scores, target_idx)
        assert torch.allclose(result, torch.tensor((0.1 + 0.3 + 0.2) / 3))

    def test_all_negatives(self):
        """Test when all items are negative (no positives)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])
        
        result = negative_score(scores, target_idx)
        assert torch.allclose(result, scores.mean())

    def test_no_negatives(self):
        """Test when all items are positive (no negatives)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 1, 2, 3, 4])
        
        result = negative_score(scores, target_idx)
        assert result == 0.0

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])
        
        with pytest.raises(ValueError, match="expects 1D tensors"):
            negative_score(scores, target_idx)


class TestRetrievalMetric:
    """Tests for the RetrievalMetric wrapper class."""

    def test_initialization(self):
        """Test basic initialization."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 10},
            reduction="mean",
        )
        assert metric.metric_kwargs == {"k": 10}
        assert metric.reduction == "mean"

    def test_invalid_reduction(self):
        """Test that invalid reduction raises an error."""
        with pytest.raises(ValueError, match="reduction must be"):
            RetrievalMetric(
                metric_functional=top_k_hit_rate,
                reduction="invalid",
            )

    def test_single_sample(self):
        """Test with a single sample (1D tensors)."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 3},
        )
        
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])
        
        result = metric(scores, target_idx)
        assert result == 1.0

    def test_batch_with_mean_reduction(self):
        """Test with batch of samples and mean reduction."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 2},
            reduction="mean",
        )
        
        scores = torch.tensor([
            [0.1, 0.9, 0.3, 0.8, 0.2],
            [0.9, 0.1, 0.8, 0.3, 0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5]
        ])
        target_idx = torch.tensor([
            [1, -1, -1],  # Hit (index 1 is rank 1)
            [2, -1, -1],  # Hit (index 2 is rank 2)
            [0, -1, -1]   # Miss (index 0 is rank 5)
        ])
        
        result = metric(scores, target_idx)
        # Expected: (1.0 + 1.0 + 0.0) / 3 = 0.667
        assert torch.allclose(result, torch.tensor(2.0 / 3.0))

    def test_batch_without_reduction(self):
        """Test with batch of samples without reduction."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 1},
            reduction=None,
        )
        
        scores = torch.tensor([
            [0.1, 0.9, 0.3],
            [0.9, 0.1, 0.3],
        ])
        target_idx = torch.tensor([
            [1, -1],  # Hit
            [2, -1]   # Miss
        ])
        
        result = metric(scores, target_idx)
        assert result.shape == (2,)
        assert result[0] == 1.0
        assert result[1] == 0.0

    def test_batch_dimension_mismatch_error(self):
        """Test that batch scores with 1D targets raises an error."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 1},
        )
        
        scores = torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.3]])
        target_idx = torch.tensor([1, 2])  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="target_idx must also be 2D"):
            metric(scores, target_idx)

    def test_invalid_dimension_error(self):
        """Test that 3D tensors raise an error."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 1},
        )
        
        scores = torch.randn(2, 3, 4)
        target_idx = torch.randint(0, 4, (2, 3, 2))
        
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            metric(scores, target_idx)

    def test_with_mrr_metric(self):
        """Test RetrievalMetric with MRR functional."""
        metric = RetrievalMetric(
            metric_functional=mean_reciprocal_rank,
            reduction="mean",
        )
        
        scores = torch.tensor([
            [0.1, 0.9, 0.3],
            [0.9, 0.1, 0.3],
        ])
        target_idx = torch.tensor([
            [1, -1],  # Rank 1, MRR = 1.0
            [2, -1]   # Rank 2, MRR = 0.5
        ])
        
        result = metric(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.75))  # (1.0 + 0.5) / 2


class TestCreateRetrievalMetrics:
    """Tests for the create_retrieval_metrics factory function."""

    def test_default_metrics(self):
        """Test default metric creation."""
        metrics = create_retrieval_metrics()
        
        assert "top_1" in metrics
        assert "top_10" in metrics
        assert "top_100" in metrics
        assert "top_1000" in metrics
        assert "mrr" in metrics
        assert "pos_score" not in metrics
        assert "neg_score" not in metrics

    def test_custom_top_k(self):
        """Test with custom top_k values."""
        metrics = create_retrieval_metrics(top_k=[5, 20])
        
        assert "top_5" in metrics
        assert "top_20" in metrics
        assert "top_1" not in metrics

    def test_without_mrr(self):
        """Test excluding MRR."""
        metrics = create_retrieval_metrics(include_mrr=False)
        
        assert "mrr" not in metrics
        assert "top_1" in metrics

    def test_with_pos_neg_scores(self):
        """Test including pos/neg score metrics."""
        metrics = create_retrieval_metrics(pos_score=True, neg_score=True)
        
        assert "pos_score" in metrics
        assert "neg_score" in metrics

    def test_sota_configuration(self):
        """Test with SOTA config parameters."""
        metrics = create_retrieval_metrics(
            top_k=[1, 10, 100, 1000],
            include_mrr=True,
            pos_score=True,
            neg_score=True,
        )
        
        assert len(metrics) == 7  # 4 top_k + mrr + pos + neg
        assert all(isinstance(m, RetrievalMetric) for m in metrics.values())

    def test_metric_functionality(self):
        """Test that created metrics work correctly."""
        metrics = create_retrieval_metrics(top_k=[1, 10], include_mrr=True)
        
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])
        
        # All metrics should return scalar tensors
        results = {name: metric(scores, target_idx) for name, metric in metrics.items()}
        
        assert results["top_1"] == 1.0
        assert results["top_10"] == 1.0
        assert results["mrr"] == 1.0  # Rank 1


class TestMetricsDevicePlacement:
    """Tests for device placement (CPU/GPU compatibility)."""

    def test_top_k_device(self, device):
        """Test top_k_hit_rate works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, -1, -1]).to(device)
        
        result = top_k_hit_rate(scores, target_idx, k=1)
        assert result.device.type == device.type

    def test_mrr_device(self, device):
        """Test mean_reciprocal_rank works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, -1, -1]).to(device)
        
        result = mean_reciprocal_rank(scores, target_idx)
        assert result.device.type == device.type

    def test_positive_score_device(self, device):
        """Test positive_score works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, -1, -1]).to(device)
        
        result = positive_score(scores, target_idx)
        assert result.device.type == device.type

    def test_negative_score_device(self, device):
        """Test negative_score works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, -1, -1]).to(device)
        
        result = negative_score(scores, target_idx)
        assert result.device.type == device.type


class TestMetricsEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_item(self):
        """Test with only one item."""
        scores = torch.tensor([0.5])
        target_idx = torch.tensor([0])
        
        assert top_k_hit_rate(scores, target_idx, k=1) == 1.0
        assert mean_reciprocal_rank(scores, target_idx) == 1.0
        assert positive_score(scores, target_idx) == 0.5
        assert negative_score(scores, target_idx) == 0.0

    def test_large_batch(self):
        """Test with a large batch."""
        batch_size = 100
        num_items = 1000
        
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 10},
            reduction="mean",
        )
        
        scores = torch.randn(batch_size, num_items)
        target_idx = torch.randint(0, num_items, (batch_size, 5))
        target_idx[target_idx >= 5] = -1  # Add padding
        
        result = metric(scores, target_idx)
        assert result.dim() == 0  # Scalar
        assert 0.0 <= result <= 1.0

    def test_all_same_scores(self):
        """Test when all scores are identical."""
        scores = torch.ones(10)
        target_idx = torch.tensor([5, -1, -1])
        
        # With all same scores, ranking is arbitrary but metrics should still work
        result_top_k = top_k_hit_rate(scores, target_idx, k=5)
        result_mrr = mean_reciprocal_rank(scores, target_idx)
        result_pos = positive_score(scores, target_idx)
        result_neg = negative_score(scores, target_idx)
        
        assert torch.isfinite(result_top_k)
        assert torch.isfinite(result_mrr)
        assert result_pos == 1.0
        assert result_neg == 1.0

