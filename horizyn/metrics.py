"""
Retrieval metrics for evaluating contrastive learning models.

This module implements common information retrieval metrics used to evaluate
dual-encoder models on protein-reaction retrieval tasks.
"""

from typing import Callable, Dict, List, Optional, Any
import torch
from torch import Tensor


class RetrievalMetric:
    """
    Wrapper class for retrieval metrics that handles both single-sample and batch processing.

    This class provides a unified interface for applying retrieval metrics to either
    individual samples or batches of samples, with automatic reduction of batch results.
    It handles 1D/2D tensor conversions and applies the metric function independently
    to each sample in a batch.

    Attributes:
        metric_functional (Callable): The underlying metric computation function.
        metric_kwargs (dict): Additional keyword arguments for the metric function.
        reduction (str | None): How to reduce batch results ("mean" or None).

    Example:
        >>> # Create a Top-10 hit rate metric
        >>> top10_metric = RetrievalMetric(
        ...     metric_functional=top_k_hit_rate,
        ...     metric_kwargs={"k": 10},
        ...     reduction="mean"
        ... )
        >>>
        >>> # Single sample usage
        >>> scores = torch.tensor([0.9, 0.8, 0.4, 0.3, 0.1])
        >>> targets = torch.tensor([0, 2])  # Indices 0 and 2 are relevant
        >>> result = top10_metric(scores, targets)  # Returns 1.0 (hit in top 10)
        >>>
        >>> # Batch usage
        >>> scores_batch = torch.tensor([
        ...     [0.9, 0.8, 0.4, 0.3, 0.1],
        ...     [0.2, 0.9, 0.3, 0.4, 0.1]
        ... ])
        >>> targets_batch = torch.tensor([
        ...     [0, 2, -1],  # Sample 1: targets at indices 0,2 (padding: -1)
        ...     [1, -1, -1]  # Sample 2: target at index 1
        ... ])
        >>> result = top10_metric(scores_batch, targets_batch)  # Returns mean
    """

    def __init__(
        self,
        metric_functional: Callable[[Tensor, Tensor], Tensor],
        metric_kwargs: Optional[Dict[str, Any]] = None,
        reduction: Optional[str] = "mean",
    ):
        """
        Initialize a retrieval metric wrapper.

        Args:
            metric_functional: Function that computes the metric. Should accept:
                - scores: Tensor of prediction scores (1D)
                - target_idx: Tensor of relevant item indices, padded with -1 (1D)
                And return a scalar Tensor.
            metric_kwargs: Additional keyword arguments to pass to metric_functional.
                Defaults to empty dict.
            reduction: How to reduce multiple scores in a batch:
                - "mean": Return mean of all scores (default)
                - None: Return tensor of individual scores
                Defaults to "mean".

        Raises:
            ValueError: If reduction is not one of ["mean", None].
        """
        self.metric_functional = metric_functional
        self.metric_kwargs = metric_kwargs or {}
        
        if reduction not in ["mean", None]:
            raise ValueError(f"reduction must be 'mean' or None, got {reduction}")
        self.reduction = reduction

    def __call__(self, scores: Tensor, target_idx: Tensor) -> Tensor:
        """
        Compute the metric for scores and target indices.

        Args:
            scores: Prediction scores. Can be:
                - 1D tensor of shape (num_items,) for a single query
                - 2D tensor of shape (batch_size, num_items) for multiple queries
            target_idx: Indices of relevant items. Can be:
                - 1D tensor of shape (num_targets,) for single query, padded with -1
                - 2D tensor of shape (batch_size, max_targets) for batch, padded with -1

        Returns:
            Scalar tensor with the computed metric value (if reduction="mean"),
            or 1D tensor of shape (batch_size,) with individual metric values
            (if reduction=None).

        Raises:
            ValueError: If scores and target_idx dimensions don't match.
        """
        # Handle single sample case
        if scores.dim() == 1:
            return self.metric_functional(scores, target_idx, **self.metric_kwargs)
        
        # Handle batch case
        if scores.dim() == 2:
            batch_size = scores.shape[0]
            
            # Ensure target_idx is also 2D
            if target_idx.dim() == 1:
                raise ValueError(
                    "For batch scores (2D), target_idx must also be 2D. "
                    f"Got scores.shape={scores.shape}, target_idx.shape={target_idx.shape}"
                )
            
            # Compute metric for each sample in the batch
            metric_values = []
            for i in range(batch_size):
                value = self.metric_functional(
                    scores[i], target_idx[i], **self.metric_kwargs
                )
                metric_values.append(value)
            
            # Stack results
            result = torch.stack(metric_values)
            
            # Apply reduction
            if self.reduction == "mean":
                return result.mean()
            else:
                return result
        
        raise ValueError(
            f"scores must be 1D or 2D, got shape {scores.shape}"
        )


def top_k_hit_rate(scores: Tensor, target_idx: Tensor, k: int = 100) -> Tensor:
    """
    Compute Top-K hit rate (binary: 1 if any target in top-K, else 0).

    This metric returns 1.0 if at least one relevant item appears in the top-K
    predictions, and 0.0 otherwise. It's a binary metric useful for measuring
    whether the model can rank any relevant item highly.

    Args:
        scores: Prediction scores of shape (num_items,). Higher scores indicate
            higher relevance/similarity.
        target_idx: Indices of relevant items, shape (num_targets,). Should be
            padded with -1 for unused positions. For example, if targets are at
            indices 2 and 5, use torch.tensor([2, 5, -1, -1, ...]).
        k: Number of top predictions to consider. Defaults to 100.

    Returns:
        Tensor containing 1.0 if any relevant item is in top-K, else 0.0.

    Example:
        >>> scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        >>> target_idx = torch.tensor([2, -1, -1])  # Target at index 2
        >>> top_k_hit_rate(scores, target_idx, k=3)
        tensor(1.)  # Index 2 is in top-3 (indices 1, 3, 2)
        >>> top_k_hit_rate(scores, target_idx, k=2)
        tensor(0.)  # Index 2 is NOT in top-2 (indices 1, 3)
    """
    if scores.dim() != 1 or target_idx.dim() != 1:
        raise ValueError(
            "top_k_hit_rate expects 1D tensors. "
            f"Got scores.dim()={scores.dim()}, target_idx.dim()={target_idx.dim()}"
        )
    
    # Filter out padding (-1)
    valid_targets = target_idx[target_idx >= 0]
    
    if len(valid_targets) == 0:
        # No valid targets, return 0
        return torch.tensor(0.0, device=scores.device)
    
    # Clamp k to number of items
    num_items = scores.shape[0]
    k_clamped = min(k, num_items)
    
    # Get indices of top-K predictions
    top_k_indices = torch.topk(scores, k=k_clamped, largest=True).indices
    
    # Check if any valid target is in top-K
    # torch.isin returns a boolean tensor indicating which targets are in top_k_indices
    hit = torch.isin(valid_targets, top_k_indices).any()
    
    return torch.tensor(1.0 if hit else 0.0, device=scores.device)


def mean_reciprocal_rank(scores: Tensor, target_idx: Tensor) -> Tensor:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR is the average of the reciprocal ranks of the first relevant item for
    each query. The rank is the position (1-indexed) of the item in the sorted
    list of scores. If multiple targets exist, only the highest-ranked target
    is considered.

    For a single query, MRR = 1 / (rank of first relevant item).

    Args:
        scores: Prediction scores of shape (num_items,). Higher scores indicate
            higher relevance/similarity.
        target_idx: Indices of relevant items, shape (num_targets,). Should be
            padded with -1 for unused positions.

    Returns:
        Tensor containing the reciprocal rank of the first relevant item.
        Returns 0.0 if no relevant items exist or none are found in ranking.

    Example:
        >>> scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        >>> target_idx = torch.tensor([2, 0, -1])  # Targets at indices 2 and 0
        >>> mean_reciprocal_rank(scores, target_idx)
        tensor(0.3333)  # Index 2 is rank 3 (after indices 1 and 3), 1/3 ≈ 0.333
    """
    if scores.dim() != 1 or target_idx.dim() != 1:
        raise ValueError(
            "mean_reciprocal_rank expects 1D tensors. "
            f"Got scores.dim()={scores.dim()}, target_idx.dim()={target_idx.dim()}"
        )
    
    # Filter out padding (-1)
    valid_targets = target_idx[target_idx >= 0]
    
    if len(valid_targets) == 0:
        # No valid targets, return 0
        return torch.tensor(0.0, device=scores.device)
    
    # Get the sorted indices (descending order of scores)
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Find the ranks (1-indexed) of all valid targets
    # Create a mapping from index to rank
    ranks = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    ranks[sorted_indices] = torch.arange(1, scores.shape[0] + 1, device=scores.device)
    
    # Get ranks of valid targets
    target_ranks = ranks[valid_targets]
    
    # Find the minimum rank (best rank)
    best_rank = target_ranks.min().float()
    
    # Return reciprocal rank
    return 1.0 / best_rank


def positive_score(scores: Tensor, target_idx: Tensor) -> Tensor:
    """
    Compute the mean score of positive (relevant) items.

    This metric measures how highly the model scores relevant items on average.
    Higher values indicate the model assigns high similarity/relevance scores
    to the correct targets.

    Args:
        scores: Prediction scores of shape (num_items,).
        target_idx: Indices of relevant items, shape (num_targets,). Should be
            padded with -1 for unused positions.

    Returns:
        Tensor containing the mean score of relevant items. Returns 0.0 if
        no relevant items exist.

    Example:
        >>> scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        >>> target_idx = torch.tensor([1, 3, -1])  # Targets at indices 1 and 3
        >>> positive_score(scores, target_idx)
        tensor(0.8500)  # (0.9 + 0.8) / 2
    """
    if scores.dim() != 1 or target_idx.dim() != 1:
        raise ValueError(
            "positive_score expects 1D tensors. "
            f"Got scores.dim()={scores.dim()}, target_idx.dim()={target_idx.dim()}"
        )
    
    # Filter out padding (-1)
    valid_targets = target_idx[target_idx >= 0]
    
    if len(valid_targets) == 0:
        # No valid targets, return 0
        return torch.tensor(0.0, device=scores.device)
    
    # Get scores of positive items
    pos_scores = scores[valid_targets]
    
    return pos_scores.mean()


def negative_score(scores: Tensor, target_idx: Tensor) -> Tensor:
    """
    Compute the mean score of negative (non-relevant) items.

    This metric measures how the model scores non-relevant items on average.
    Lower values indicate the model correctly assigns low similarity scores
    to incorrect targets.

    Args:
        scores: Prediction scores of shape (num_items,).
        target_idx: Indices of relevant items, shape (num_targets,). Should be
            padded with -1 for unused positions. All other indices are considered
            negative.

    Returns:
        Tensor containing the mean score of non-relevant items. Returns 0.0 if
        all items are relevant (no negatives exist).

    Example:
        >>> scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        >>> target_idx = torch.tensor([1, 3, -1])  # Targets at indices 1 and 3
        >>> negative_score(scores, target_idx)
        tensor(0.2000)  # (0.1 + 0.3 + 0.2) / 3
    """
    if scores.dim() != 1 or target_idx.dim() != 1:
        raise ValueError(
            "negative_score expects 1D tensors. "
            f"Got scores.dim()={scores.dim()}, target_idx.dim()={target_idx.dim()}"
        )
    
    # Filter out padding (-1)
    valid_targets = target_idx[target_idx >= 0]
    
    # Create a mask for negative items (all items not in valid_targets)
    num_items = scores.shape[0]
    neg_mask = torch.ones(num_items, dtype=torch.bool, device=scores.device)
    
    if len(valid_targets) > 0:
        neg_mask[valid_targets] = False
    
    # Get scores of negative items
    neg_scores = scores[neg_mask]
    
    if len(neg_scores) == 0:
        # All items are positive, return 0
        return torch.tensor(0.0, device=scores.device)
    
    return neg_scores.mean()


def create_retrieval_metrics(
    top_k: Optional[List[int]] = None,
    include_mrr: bool = True,
    pos_score: bool = False,
    neg_score: bool = False,
) -> Dict[str, RetrievalMetric]:
    """
    Create a dictionary of retrieval metrics for evaluation.

    This factory function creates a collection of common retrieval metrics
    wrapped in RetrievalMetric instances for easy batch processing.

    Args:
        top_k: List of k values for Top-K hit rate metrics. If None, defaults
            to [1, 10, 100, 1000]. Defaults to None.
        include_mrr: Whether to include Mean Reciprocal Rank metric. Defaults to True.
        pos_score: Whether to include mean positive score metric. Defaults to False.
        neg_score: Whether to include mean negative score metric. Defaults to False.

    Returns:
        Dictionary mapping metric names to RetrievalMetric instances:
            - f"top_{k}": Top-K hit rate for each k value
            - "mrr": Mean Reciprocal Rank (if include_mrr=True)
            - "pos_score": Mean score of positive items (if pos_score=True)
            - "neg_score": Mean score of negative items (if neg_score=True)

    Example:
        >>> # Create metrics matching SOTA config
        >>> metrics = create_retrieval_metrics(
        ...     top_k=[1, 10, 100, 1000],
        ...     include_mrr=True,
        ...     pos_score=True,
        ...     neg_score=True
        ... )
        >>> # Use with predictions
        >>> scores = torch.randn(128, 1000)  # 128 queries, 1000 targets
        >>> targets = torch.randint(0, 1000, (128, 5))  # Up to 5 targets per query
        >>> targets[targets >= 5] = -1  # Pad unused positions
        >>> results = {name: metric(scores, targets) for name, metric in metrics.items()}
    """
    if top_k is None:
        top_k = [1, 10, 100, 1000]
    
    metrics = {}
    
    # Top-K hit rate metrics
    for k in top_k:
        metrics[f"top_{k}"] = RetrievalMetric(
            metric_functional=top_k_hit_rate,
            metric_kwargs={"k": k},
            reduction="mean",
        )
    
    # Mean Reciprocal Rank
    if include_mrr:
        metrics["mrr"] = RetrievalMetric(
            metric_functional=mean_reciprocal_rank,
            reduction="mean",
        )
    
    # Positive/negative score metrics
    if pos_score:
        metrics["pos_score"] = RetrievalMetric(
            metric_functional=positive_score,
            reduction="mean",
        )
    
    if neg_score:
        metrics["neg_score"] = RetrievalMetric(
            metric_functional=negative_score,
            reduction="mean",
        )
    
    return metrics

