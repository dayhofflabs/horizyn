"""
Unit tests for retrieval metrics.
"""

import pytest
import torch

from horizyn.metrics import (
    RetrievalMetric,
    average_precision,
    create_retrieval_metrics,
    mean_reciprocal_rank,
    negative_score,
    positive_score,
    r_precision,
    top_k_hit_rate,
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

    def test_out_of_range_target_raises(self):
        """Targets beyond num_items should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([3, -1, -1])  # 3 out of range
        with pytest.raises(ValueError, match="out-of-range"):
            top_k_hit_rate(scores, target_idx, k=3)

    def test_wrong_dtype_target_raises(self):
        """Non-long dtype for target indices should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])  # float
        with pytest.raises(ValueError, match="dtype torch.long"):
            top_k_hit_rate(scores, target_idx, k=2)


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

    def test_mrr_out_of_range_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([5, -1])
        with pytest.raises(ValueError, match="out-of-range"):
            mean_reciprocal_rank(scores, target_idx)

    def test_mrr_wrong_dtype_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="dtype torch.long"):
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

    def test_positive_score_out_of_range_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([3, -1])
        with pytest.raises(ValueError, match="out-of-range"):
            positive_score(scores, target_idx)

    def test_positive_score_wrong_dtype_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="dtype torch.long"):
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

    def test_negative_score_out_of_range_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([2, 3])  # 3 out of range
        with pytest.raises(ValueError, match="out-of-range"):
            negative_score(scores, target_idx)

    def test_negative_score_wrong_dtype_target_raises(self):
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="dtype torch.long"):
            negative_score(scores, target_idx)


class TestRPrecision:
    """Tests for the r_precision function."""

    def test_perfect_r_precision(self):
        """Test with all relevant items in top-R."""
        # 2 relevant items at indices 1 and 3, which are highest scores
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 3, -1])  # R = 2

        result = r_precision(scores, target_idx)
        # Top-2 are indices 1 (0.9) and 3 (0.8), both are relevant
        assert torch.allclose(result, torch.tensor(1.0))

    def test_half_r_precision(self):
        """Test with half of relevant items in top-R."""
        # 2 relevant items at indices 1 and 0
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 0, -1])  # R = 2

        result = r_precision(scores, target_idx)
        # Top-2 are indices 1 (0.9) and 3 (0.8), only index 1 is relevant
        assert torch.allclose(result, torch.tensor(0.5))

    def test_zero_r_precision(self):
        """Test with no relevant items in top-R."""
        # 2 relevant items at indices 0 and 4 (lowest scores)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 4, -1])  # R = 2

        result = r_precision(scores, target_idx)
        # Top-2 are indices 1 (0.9) and 3 (0.8), neither is relevant
        assert torch.allclose(result, torch.tensor(0.0))

    def test_single_relevant(self):
        """Test with single relevant item."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])  # R = 1

        result = r_precision(scores, target_idx)
        # Top-1 is index 1, which is relevant
        assert torch.allclose(result, torch.tensor(1.0))

    def test_single_relevant_not_top(self):
        """Test with single relevant item not at top."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])  # R = 1

        result = r_precision(scores, target_idx)
        # Top-1 is index 1, but relevant is index 2
        assert torch.allclose(result, torch.tensor(0.0))

    def test_all_relevant(self):
        """Test when all items are relevant (R = num_items)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 1, 2, 3, 4])

        result = r_precision(scores, target_idx)
        # Top-5 are all items, all are relevant
        assert torch.allclose(result, torch.tensor(1.0))

    def test_many_relevant(self):
        """Test with 3 relevant items out of 5."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 3, 2, -1])  # R = 3

        result = r_precision(scores, target_idx)
        # Top-3 are indices 1 (0.9), 3 (0.8), 2 (0.3)
        # All 3 relevant items (1, 3, 2) are in top-3
        assert torch.allclose(result, torch.tensor(1.0))

    def test_empty_targets(self):
        """Test with no valid targets (all padding)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])

        result = r_precision(scores, target_idx)
        assert result == 0.0

    def test_r_larger_than_items(self):
        """Test with R larger than number of items."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([0, 1, 2, -1, -1, -1, -1, -1, -1, -1])  # R = 3

        result = r_precision(scores, target_idx)
        # Top-3 (all items) are all relevant
        assert torch.allclose(result, torch.tensor(1.0))

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])

        with pytest.raises(ValueError, match="expects 1D tensors"):
            r_precision(scores, target_idx)

    def test_out_of_range_target_raises(self):
        """Targets beyond num_items should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([3, -1, -1])  # 3 out of range

        with pytest.raises(ValueError, match="out-of-range"):
            r_precision(scores, target_idx)

    def test_wrong_dtype_target_raises(self):
        """Non-long dtype for target indices should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])  # float

        with pytest.raises(ValueError, match="dtype torch.long"):
            r_precision(scores, target_idx)


class TestAveragePrecision:
    """Tests for the average_precision function."""

    def test_perfect_ranking(self):
        """Test with all relevant items at the top."""
        # 2 relevant items at indices 1 and 3, which have highest scores
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 3, -1])

        result = average_precision(scores, target_idx)
        # Rank 1: index 1 (relevant), P@1 = 1/1 = 1.0
        # Rank 2: index 3 (relevant), P@2 = 2/2 = 1.0
        # AP = (1.0 + 1.0) / 2 = 1.0
        assert torch.allclose(result, torch.tensor(1.0))

    def test_mixed_ranking(self):
        """Test with relevant items mixed in ranking."""
        # Relevant at indices 1 and 2
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, 2, -1])

        result = average_precision(scores, target_idx)
        # Ranking by score: 1 (0.9), 3 (0.8), 2 (0.3), 4 (0.2), 0 (0.1)
        # Rank 1: index 1 (relevant), P@1 = 1/1 = 1.0
        # Rank 3: index 2 (relevant), P@3 = 2/3 = 0.667
        # AP = (1.0 + 0.667) / 2 = 0.833
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-4)

    def test_worst_ranking(self):
        """Test with all relevant items at the bottom."""
        # Relevant at indices 0 and 4 (lowest scores)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 4, -1])

        result = average_precision(scores, target_idx)
        # Ranking: 1 (0.9), 3 (0.8), 2 (0.3), 4 (0.2), 0 (0.1)
        # Rank 4: index 4 (relevant), P@4 = 1/4 = 0.25
        # Rank 5: index 0 (relevant), P@5 = 2/5 = 0.4
        # AP = (0.25 + 0.4) / 2 = 0.325
        expected = (1.0 / 4.0 + 2.0 / 5.0) / 2.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-4)

    def test_single_relevant_at_top(self):
        """Test with single relevant item at rank 1."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([1, -1, -1])

        result = average_precision(scores, target_idx)
        # Rank 1: index 1 (relevant), P@1 = 1.0
        # AP = 1.0 / 1 = 1.0
        assert torch.allclose(result, torch.tensor(1.0))

    def test_single_relevant_in_middle(self):
        """Test with single relevant item in middle of ranking."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([2, -1, -1])

        result = average_precision(scores, target_idx)
        # Ranking: 1 (0.9), 3 (0.8), 2 (0.3), 4 (0.2), 0 (0.1)
        # Rank 3: index 2 (relevant), P@3 = 1/3
        # AP = (1/3) / 1 = 0.333
        expected = 1.0 / 3.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-4)

    def test_single_relevant_at_bottom(self):
        """Test with single relevant item at worst rank."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, -1, -1])  # Index 0 has lowest score

        result = average_precision(scores, target_idx)
        # Rank 5: index 0 (relevant), P@5 = 1/5 = 0.2
        # AP = 0.2 / 1 = 0.2
        assert torch.allclose(result, torch.tensor(0.2))

    def test_all_relevant(self):
        """Test when all items are relevant."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([0, 1, 2, 3, 4])

        result = average_precision(scores, target_idx)
        # Every position is relevant
        # P@1=1, P@2=1, P@3=1, P@4=1, P@5=1
        # AP = 5/5 = 1.0
        assert torch.allclose(result, torch.tensor(1.0))

    def test_three_relevant_items(self):
        """Test with 3 relevant items at various ranks."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.7])
        target_idx = torch.tensor([1, 4, 0, -1])

        result = average_precision(scores, target_idx)
        # Ranking: 1 (0.9), 3 (0.8), 4 (0.7), 2 (0.3), 0 (0.1)
        # Rank 1: index 1 (relevant), P@1 = 1/1 = 1.0
        # Rank 3: index 4 (relevant), P@3 = 2/3 = 0.667
        # Rank 5: index 0 (relevant), P@5 = 3/5 = 0.6
        # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-4)

    def test_empty_targets(self):
        """Test with no valid targets (all padding)."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        target_idx = torch.tensor([-1, -1, -1])

        result = average_precision(scores, target_idx)
        assert result == 0.0

    def test_dimension_error(self):
        """Test that 2D tensors raise an error."""
        scores = torch.tensor([[0.1, 0.9, 0.3]])
        target_idx = torch.tensor([1])

        with pytest.raises(ValueError, match="expects 1D tensors"):
            average_precision(scores, target_idx)

    def test_out_of_range_target_raises(self):
        """Targets beyond num_items should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([3, -1, -1])  # 3 out of range

        with pytest.raises(ValueError, match="out-of-range"):
            average_precision(scores, target_idx)

    def test_wrong_dtype_target_raises(self):
        """Non-long dtype for target indices should raise an error."""
        scores = torch.tensor([0.1, 0.9, 0.3])
        target_idx = torch.tensor([1.0, -1.0])  # float

        with pytest.raises(ValueError, match="dtype torch.long"):
            average_precision(scores, target_idx)


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

        scores = torch.tensor(
            [[0.1, 0.9, 0.3, 0.8, 0.2], [0.9, 0.1, 0.8, 0.3, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        target_idx = torch.tensor(
            [
                [1, -1, -1],  # Hit (index 1 is rank 1)
                [2, -1, -1],  # Hit (index 2 is rank 2)
                [0, -1, -1],  # Miss (index 0 is rank 5)
            ]
        )

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

        scores = torch.tensor(
            [
                [0.1, 0.9, 0.3],
                [0.9, 0.1, 0.3],
            ]
        )
        target_idx = torch.tensor([[1, -1], [2, -1]])  # Hit  # Miss

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

    def test_batch_first_dim_mismatch_error(self):
        """Mismatch between batch dimension of scores and target_idx should raise."""
        metric = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": 1},
        )
        scores = torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.3]])
        target_idx = torch.tensor([[1, -1]])  # batch size 1 vs 2
        with pytest.raises(ValueError, match="Batch size mismatch"):
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

        scores = torch.tensor(
            [
                [0.1, 0.9, 0.3],
                [0.9, 0.1, 0.3],
            ]
        )
        target_idx = torch.tensor([[1, -1], [2, -1]])  # Rank 1, MRR = 1.0  # Rank 2, MRR = 0.5

        result = metric(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.75))  # (1.0 + 0.5) / 2

    def test_with_r_precision_metric(self):
        """Test RetrievalMetric with r_precision functional."""
        metric = RetrievalMetric(
            metric_functional=r_precision,
            reduction="mean",
        )

        scores = torch.tensor(
            [
                [0.1, 0.9, 0.3, 0.8, 0.2],  # Relevant: 1, 3 (both in top-2)
                [0.1, 0.9, 0.3, 0.8, 0.2],  # Relevant: 0, 4 (neither in top-2)
            ]
        )
        target_idx = torch.tensor(
            [
                [1, 3, -1],  # R=2, both in top-2, R-precision = 1.0
                [0, 4, -1],  # R=2, neither in top-2, R-precision = 0.0
            ]
        )

        result = metric(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.5))  # (1.0 + 0.0) / 2

    def test_with_average_precision_metric(self):
        """Test RetrievalMetric with average_precision functional."""
        metric = RetrievalMetric(
            metric_functional=average_precision,
            reduction="mean",
        )

        scores = torch.tensor(
            [
                [0.1, 0.9, 0.3, 0.8, 0.2],  # Relevant: 1, 3 (ranks 1, 2)
                [0.1, 0.9, 0.3, 0.8, 0.2],  # Relevant: 0 (rank 5)
            ]
        )
        target_idx = torch.tensor(
            [
                [1, 3, -1],  # AP = (1/1 + 2/2) / 2 = 1.0
                [0, -1, -1],  # AP = (1/5) / 1 = 0.2
            ]
        )

        result = metric(scores, target_idx)
        assert torch.allclose(result, torch.tensor(0.6))  # (1.0 + 0.2) / 2


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

    def test_with_r_precision(self):
        """Test including R-precision metric."""
        metrics = create_retrieval_metrics(include_r_precision=True)

        assert "r_precision" in metrics
        assert isinstance(metrics["r_precision"], RetrievalMetric)

    def test_with_avg_precision(self):
        """Test including Average Precision metric."""
        metrics = create_retrieval_metrics(include_avg_precision=True)

        assert "avg_precision" in metrics
        assert isinstance(metrics["avg_precision"], RetrievalMetric)

    def test_paper_table_configuration(self):
        """Test with full paper table configuration (all metrics)."""
        metrics = create_retrieval_metrics(
            top_k=[1, 10, 100, 1000],
            include_mrr=True,
            include_r_precision=True,
            include_avg_precision=True,
            pos_score=False,
            neg_score=False,
        )

        assert len(metrics) == 7  # 4 top_k + mrr + r_precision + avg_precision
        assert "top_1" in metrics
        assert "top_10" in metrics
        assert "top_100" in metrics
        assert "top_1000" in metrics
        assert "mrr" in metrics
        assert "r_precision" in metrics
        assert "avg_precision" in metrics

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

    def test_r_precision_device(self, device):
        """Test r_precision works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, 3, -1]).to(device)

        result = r_precision(scores, target_idx)
        assert result.device.type == device.type

    def test_average_precision_device(self, device):
        """Test average_precision works on both CPU and GPU."""
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2]).to(device)
        target_idx = torch.tensor([1, 3, -1]).to(device)

        result = average_precision(scores, target_idx)
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
        assert r_precision(scores, target_idx) == 1.0
        assert average_precision(scores, target_idx) == 1.0

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
        result_r_prec = r_precision(scores, target_idx)
        result_avg_prec = average_precision(scores, target_idx)

        assert torch.isfinite(result_top_k)
        assert torch.isfinite(result_mrr)
        assert result_pos == 1.0
        assert result_neg == 1.0
        assert torch.isfinite(result_r_prec)
        assert torch.isfinite(result_avg_prec)
