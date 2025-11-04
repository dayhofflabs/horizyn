"""
Base dataset classes providing key-based access and transforms.
"""

from typing import Any, Callable, Generic, List, Optional, TypeVar, Union
import torch
from torch.utils.data import Dataset

# Generic type for dataset keys (strings, ints, or floats)
K = TypeVar("K", str, int, float)


class BaseDataset(Dataset, Generic[K]):
    """
    Base dataset class with key-based access and optional transforms.

    This dataset provides a flexible interface for accessing data using keys
    (which can be strings, ints, or floats) and supports optional data transforms.
    It serves as the foundation for all other dataset classes in Horizyn.

    Key Features:
        - Key-based access: Access data using meaningful keys (not just indices)
        - Key-to-index mapping: Optional mapping from keys to contiguous indices
        - Transform support: Apply transformations to data on-the-fly
        - Array data storage: Store data in memory as lists, numpy arrays, or tensors

    Attributes:
        _keys (List[K] | None): List of dataset keys.
        _array_data (List | torch.Tensor | None): Optional array-like data storage.
        _key_to_idx (dict[K, int] | None): Optional key-to-index mapping.
        transforms (Callable | None): Optional transform function(s).

    Example:
        >>> # Create dataset with keys and data
        >>> keys = ["sample1", "sample2", "sample3"]
        >>> data = torch.randn(3, 10)
        >>> dataset = BaseDataset(keys=keys, array_data=data)
        >>> len(dataset)
        3
        >>> sample = dataset["sample1"]  # Access by key
        >>> sample = dataset[0]  # Access by index (if key_to_idx exists)
    """

    def __init__(
        self,
        keys: Optional[List[K]] = None,
        array_data: Optional[Union[List, torch.Tensor]] = None,
        use_key_to_idx: bool = False,
        transforms: Optional[Callable[[K, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the base dataset.

        Args:
            keys: List of keys corresponding to data points. Must be unique.
                Defaults to None.
            array_data: Array-like data (list or tensor). If provided with keys,
                must have the same length. Defaults to None.
            use_key_to_idx: Whether to create a key-to-index mapping for integer
                indexing. Automatically created if both keys and array_data are
                provided. Defaults to False.
            transforms: Optional callable to transform data. Should accept (key, data)
                and return transformed data. Defaults to None.
            **kwargs: Additional keyword arguments (for subclass compatibility).

        Raises:
            ValueError: If duplicate keys are found.
            ValueError: If keys and array_data lengths don't match.
            ValueError: If use_key_to_idx is True but no keys are provided.
        """
        super().__init__()

        self._keys = keys
        self._array_data = array_data
        self._key_to_idx = None
        self.transforms = transforms

        # Validate keys are unique
        if self._keys is not None:
            if len(self._keys) != len(set(self._keys)):
                duplicates = [k for k in self._keys if self._keys.count(k) > 1]
                raise ValueError(
                    f"Keys must be unique. Found duplicates: {set(duplicates)}"
                )

        # Validate keys and data match
        if self._keys is not None and self._array_data is not None:
            if len(self._keys) != len(self._array_data):
                raise ValueError(
                    f"Keys and data must have same length. "
                    f"Got {len(self._keys)} keys and {len(self._array_data)} data points."
                )

        # Create key-to-index mapping if requested or if both keys and data exist
        if self._keys is not None and (use_key_to_idx or self._array_data is not None):
            self._key_to_idx = {key: idx for idx, key in enumerate(self._keys)}
        elif use_key_to_idx:
            raise ValueError("Cannot create key-to-idx mapping without keys.")

    @property
    def keys(self) -> List[K]:
        """
        Get the list of dataset keys.

        Returns:
            List of keys in the dataset.

        Raises:
            AttributeError: If keys are not set.
        """
        if self._keys is None:
            raise AttributeError("Keys are not set.")
        return self._keys

    @property
    def array_data(self) -> Union[List, torch.Tensor]:
        """
        Get the array data stored in the dataset.

        Returns:
            The array-like data.

        Raises:
            AttributeError: If array data is not set.
        """
        if self._array_data is None:
            raise AttributeError("Array data is not set.")
        return self._array_data

    @property
    def key_to_idx(self) -> dict[K, int]:
        """
        Get the key-to-index mapping.

        Returns:
            Dictionary mapping keys to integer indices.

        Raises:
            AttributeError: If key-to-idx mapping is not set.
        """
        if self._key_to_idx is None:
            raise AttributeError("Key-to-idx mapping is not set.")
        return self._key_to_idx

    def _get_idx(self, key: K) -> int:
        """
        Convert a key to an integer index.

        Args:
            key: The key to convert.

        Returns:
            The corresponding integer index.

        Raises:
            KeyError: If the key is not found in the mapping.
        """
        if self._key_to_idx is None:
            raise AttributeError("Key-to-idx mapping is not available.")
        
        if key not in self._key_to_idx:
            raise KeyError(f"Key {key} not found in dataset.")
        
        return self._key_to_idx[key]

    def _apply_transforms(self, key: K, data: Any) -> Any:
        """
        Apply transforms to the data.

        Args:
            key: The key associated with the data.
            data: The data to transform.

        Returns:
            Transformed data.
        """
        if self.transforms is not None:
            return self.transforms(key, data)
        return data

    def __len__(self) -> int:
        """
        Get the dataset length.

        Returns:
            Number of samples in the dataset.

        Raises:
            ValueError: If no keys or array data are set.
        """
        if self._keys is not None:
            return len(self._keys)
        elif self._array_data is not None:
            return len(self._array_data)
        else:
            raise ValueError("Cannot determine length: no keys or array data set.")

    def __getitem__(self, key: K) -> Any:
        """
        Get a data sample by key or index.

        Supports both key-based and integer-based indexing (if key_to_idx exists).

        Args:
            key: The key or index of the sample to retrieve. Can be:
                - A string/float key from the keys list
                - An integer index (if key_to_idx mapping exists)

        Returns:
            The data sample (potentially transformed).

        Raises:
            ValueError: If array data is not set.
            KeyError: If the key is not found.
            IndexError: If using integer index and it's out of bounds.
        """
        if self._array_data is None:
            raise ValueError("Cannot get item: array data is not set.")

        # Handle integer indexing if key_to_idx exists
        if isinstance(key, int) and self._key_to_idx is not None:
            if key < 0 or key >= len(self):
                raise IndexError(
                    f"Index {key} is out of bounds for dataset of length {len(self)}."
                )
            # Get the actual key from the index
            actual_key = self._keys[key] if self._keys is not None else key
            idx = key
        else:
            # Key-based access
            actual_key = key
            idx = self._get_idx(key)

        sample = self._array_data[idx]
        return self._apply_transforms(actual_key, sample)


class WrapperDataset(BaseDataset[K]):
    """
    Dataset wrapper that applies transforms to another dataset.

    This class wraps an existing dataset and allows applying transformations
    without modifying the original dataset. It's useful for composing datasets
    with different preprocessing pipelines.

    Attributes:
        dataset (BaseDataset[K]): The wrapped dataset.

    Example:
        >>> # Create base dataset
        >>> base_dataset = BaseDataset(keys=["a", "b"], array_data=torch.randn(2, 10))
        >>>
        >>> # Wrap with transform
        >>> def normalize_transform(key, data):
        ...     return (data - data.mean()) / data.std()
        >>>
        >>> wrapped = WrapperDataset(base_dataset, transforms=normalize_transform)
        >>> normalized_sample = wrapped["a"]  # Gets transformed data
    """

    def __init__(
        self,
        dataset: BaseDataset[K],
        transforms: Optional[Callable[[K, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the wrapper dataset.

        Args:
            dataset: The dataset to wrap.
            transforms: Optional transform function to apply. Should accept
                (key, data) and return transformed data. Defaults to None.
            **kwargs: Additional keyword arguments (for compatibility).
        """
        # Initialize without calling super().__init__() to avoid conflicts
        Dataset.__init__(self)
        self.dataset = dataset
        self.transforms = transforms

    @property
    def keys(self) -> List[K]:
        """Get keys from the wrapped dataset."""
        return self.dataset.keys

    @property
    def array_data(self) -> Union[List, torch.Tensor]:
        """Get array data from the wrapped dataset."""
        return self.dataset.array_data

    @property
    def key_to_idx(self) -> dict[K, int]:
        """Get key-to-idx mapping from the wrapped dataset."""
        return self.dataset.key_to_idx

    def __len__(self) -> int:
        """Get length from the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, key: K) -> Any:
        """
        Get a data sample from the wrapped dataset and apply transforms.

        Args:
            key: The key or index of the sample to retrieve.

        Returns:
            The transformed data sample.
        """
        # Get data from wrapped dataset
        data = self.dataset[key]
        
        # Apply this wrapper's transforms
        if self.transforms is not None:
            data = self.transforms(key, data)
        
        return data

