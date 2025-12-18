"""
Batch collation functions for PyTorch DataLoader.
"""

from typing import Any, Dict, List

import torch
from torch.utils.data._utils.collate import default_collate


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of dictionaries into a single dictionary of batched tensors.

    This function takes a list of dictionaries (one per sample) and combines them
    into a single dictionary where each value is a batched tensor. It uses PyTorch's
    default_collate for the actual batching.

    This is the standard collation function for the SOTA model, which uses
    dictionaries containing query and target vectors.

    Args:
        batch: List of dictionaries, one per sample. Each dict should have the
            same keys and compatible tensor values.

    Returns:
        Dictionary with same keys as input dicts, where each value is a batched
        tensor of shape [batch_size, ...].

    Example:
        >>> batch = [
        ...     {"query_vec": torch.tensor([1, 2, 3]), "target_vec": torch.tensor([4, 5, 6])},
        ...     {"query_vec": torch.tensor([7, 8, 9]), "target_vec": torch.tensor([10, 11, 12])}
        ... ]
        >>> result = dict_collate_fn(batch)
        >>> result["query_vec"].shape
        torch.Size([2, 3])
    """
    if not batch:
        return {}

    return default_collate(batch)


__all__ = [
    "default_collate",
    "dict_collate_fn",
]
