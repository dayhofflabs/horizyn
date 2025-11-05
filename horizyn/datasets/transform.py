"""
Data transformation classes for dataset pipelines.
"""

from typing import Any, Dict, List

import torch


class ConcatTensorTransform:
    """
    Concatenate multiple tensors from a dictionary into a single tensor.

    This transform takes a dictionary containing multiple tensors and
    concatenates them along a specified dimension. It's used to combine
    different fingerprint types (e.g., RDKit+ and DRFP) into a single vector.

    Args:
        labels: List of keys in the dictionary to concatenate.
        dim: Dimension along which to concatenate. Defaults to 0.

    Example:
        >>> data = {"rdkit": torch.tensor([1, 2, 3]), "drfp": torch.tensor([4, 5, 6])}
        >>> transform = ConcatTensorTransform(labels=["rdkit", "drfp"])
        >>> result = transform("key1", data)
        >>> result.shape
        torch.Size([6])
    """

    def __init__(self, labels: List[str], dim: int = 0):
        self.labels = labels
        self.dim = dim

    def __call__(self, key: Any, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate tensors from the dictionary.

        Args:
            key: Key for the sample (unused, for compatibility).
            data: Dictionary containing tensors to concatenate.

        Returns:
            Concatenated tensor.

        Raises:
            KeyError: If a label is not in the dictionary.
            ValueError: If tensors cannot be concatenated.
        """
        # Extract tensors in order
        tensors = []
        for label in self.labels:
            if label not in data:
                raise KeyError(
                    f"Label '{label}' not found in data. "
                    f"Available keys: {list(data.keys())}"
                )
            tensors.append(data[label])

        # Concatenate
        try:
            return torch.cat(tensors, dim=self.dim)
        except Exception as e:
            shapes = [t.shape for t in tensors]
            raise ValueError(
                f"Failed to concatenate tensors with shapes {shapes} "
                f"along dim {self.dim}: {e}"
            ) from e

