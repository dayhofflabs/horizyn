"""
Dataset classes for combining multiple datasets.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from horizyn.datasets.base import BaseDataset, K

logger = logging.getLogger(__name__)


class MergeDataset(BaseDataset[K]):
    """
    Dataset that merges multiple datasets based on the intersection of their keys.

    This dataset combines data from multiple source datasets, returning a dictionary
    containing data from all datasets for each key. Only keys that exist in ALL
    datasets are included in the merged dataset.

    Attributes:
        datasets (Dict[str, BaseDataset[K]]): Dictionary mapping names to datasets.
        add_prefix (bool): Whether to prepend dataset names to feature keys.

    Example:
        >>> # Create two datasets
        >>> reactions = BaseDataset(
        ...     keys=["rxn1", "rxn2", "rxn3"],
        ...     array_data=torch.randn(3, 1024)
        ... )
        >>> proteins = BaseDataset(
        ...     keys=["rxn1", "rxn2", "rxn4"],  # Note: rxn3 vs rxn4 difference
        ...     array_data=torch.randn(3, 512)
        ... )
        >>>
        >>> # Merge datasets (only rxn1 and rxn2 will be in result)
        >>> merged = MergeDataset({
        ...     "reaction": reactions,
        ...     "protein": proteins
        ... })
        >>> len(merged)  # 2 (intersection of keys)
        >>> sample = merged["rxn1"]  # {"reaction": ..., "protein": ...}
    """

    def __init__(
        self,
        datasets: Dict[str, BaseDataset[K]],
        add_prefix: bool = False,
        transforms: Optional[Callable[[K, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the MergeDataset.

        Args:
            datasets: Dictionary mapping dataset names to dataset objects.
                The resulting dataset will use the intersection of all keys.
            add_prefix: Whether to prepend dataset names to result keys
                (e.g., "reaction_smiles" instead of "smiles"). Defaults to False.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If no common keys are found between datasets.
            ValueError: If datasets dict is empty.
        """
        if not datasets:
            raise ValueError("Must provide at least one dataset.")

        self.datasets = datasets
        self.add_prefix = add_prefix

        # Find intersection of all dataset keys
        key_sets = [set(dataset.keys) for dataset in datasets.values()]
        common_keys = set.intersection(*key_sets)

        if len(common_keys) == 0:
            raise ValueError(
                f"No common keys found between datasets. "
                f"Dataset key counts: {[(name, len(ds.keys)) for name, ds in datasets.items()]}"
            )

        # Sort keys for consistency
        keys = sorted(list(common_keys))

        # Initialize base dataset with merged keys
        super().__init__(keys=keys, transforms=transforms, **kwargs)

    def __getitem__(self, key: K) -> Dict[str, Any]:
        """
        Get samples from all datasets for the given key.

        Args:
            key: The key to retrieve from all datasets.

        Returns:
            Dictionary containing data from all datasets. If a dataset returns
            a dict, its contents are merged into the result. If add_prefix=True,
            dataset names are prepended to keys. Non-dict results use the dataset
            name as the key.

        Raises:
            ValueError: If duplicate keys are found when merging dict results.
        """
        sample = {}

        for name, dataset in self.datasets.items():
            result = dataset[key]

            # If result is a dict, merge its contents
            if isinstance(result, dict):
                if self.add_prefix:
                    # Add prefix to all keys
                    for k, v in result.items():
                        sample[f"{name}_{k}"] = v
                else:
                    # Merge directly, checking for duplicates
                    for k, v in result.items():
                        if k in sample:
                            raise ValueError(
                                f"Duplicate key '{k}' found when merging datasets. "
                                f"Consider using add_prefix=True."
                            )
                        sample[k] = v
            else:
                # Use dataset name as key
                sample[name] = result

        return self._apply_transforms(key, sample)


class TupleDataset(BaseDataset[K]):
    """
    Dataset that aggregates data from multiple datasets using tuples of keys.

    This dataset uses a "tuple dataset" that provides mappings of keys to other
    datasets. For each sample, it retrieves the tuple of keys and fetches data
    from the corresponding datasets.

    This is useful for contrastive learning where you have pairs/tuples of
    samples from different datasets (e.g., reaction-protein pairs).

    By default, this dataset gracefully handles missing data by filtering out
    pairs that reference non-existent keys in source datasets. This allows
    training to proceed even with incomplete or evolving data. Set skip_missing=False
    for strict validation that raises errors on missing keys.

    Attributes:
        tuple_dataset (BaseDataset[K]): Dataset providing tuples of keys.
        key_name_to_dataset (Dict[str, BaseDataset]): Mapping from key names to datasets.
        rename_map (Dict[str, str]): Optional mapping to rename output keys.
        add_prefix (bool): Whether to prepend dataset names to feature keys.
        skip_missing (bool): Whether to skip pairs with missing keys (default: True).

    Example:
        >>> # Create source datasets
        >>> reactions = BaseDataset(
        ...     keys=["rxn1", "rxn2", "rxn3"],
        ...     array_data=torch.randn(3, 1024)
        ... )
        >>> proteins = BaseDataset(
        ...     keys=["prot1", "prot2", "prot3"],
        ...     array_data=torch.randn(3, 512)
        ... )
        >>>
        >>> # Create tuple dataset defining pairs
        >>> pairs = BaseDataset(
        ...     keys=["pair1", "pair2", "pair3"],
        ...     array_data=[
        ...         {"query_id": "rxn1", "target_id": "prot2"},
        ...         {"query_id": "rxn3", "target_id": "prot1"},
        ...         {"query_id": "rxn99", "target_id": "prot1"}  # rxn99 doesn't exist
        ...     ]
        ... )
        >>>
        >>> # Create tuple dataset (will filter out pair3)
        >>> tuple_ds = TupleDataset(
        ...     tuple_dataset=pairs,
        ...     key_name_to_dataset={
        ...         "query_id": reactions,
        ...         "target_id": proteins
        ...     }
        ... )
        >>> len(tuple_ds)  # 2 (pair3 filtered out)
        >>> sample = tuple_ds["pair1"]  # {"query_id": <rxn1 data>, "target_id": <prot2 data>}
    """

    def __init__(
        self,
        tuple_dataset: BaseDataset[K],
        key_name_to_dataset: Dict[str, BaseDataset],
        rename_map: Optional[Dict[str, str]] = None,
        add_prefix: bool = False,
        skip_missing: bool = True,
        transforms: Optional[Callable[[K, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the TupleDataset.

        Args:
            tuple_dataset: Dataset where each entry is a dict of keys to other datasets.
                Example: {"query_id": "rxn1", "target_id": "prot5"}
            key_name_to_dataset: Dictionary mapping key names (from tuple_dataset)
                to the actual datasets to fetch from.
            rename_map: Optional dictionary to rename keys in the output.
                Example: {"query_id": "reaction", "target_id": "protein"}
                Defaults to None (no renaming).
            add_prefix: Whether to prepend source dataset names to feature keys.
                Defaults to False.
            skip_missing: Whether to skip pairs with missing keys in source datasets.
                If True, filters out invalid pairs with a warning. If False, will
                raise KeyError when missing keys are accessed. Defaults to True.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.tuple_dataset = tuple_dataset
        self.key_name_to_dataset = key_name_to_dataset
        self.rename_map = rename_map or {}
        self.add_prefix = add_prefix
        self.skip_missing = skip_missing

        # Filter out invalid pairs if skip_missing is enabled
        if skip_missing:
            valid_keys = self._filter_valid_keys()
        else:
            valid_keys = tuple_dataset.keys

        # Initialize with filtered keys and create key_to_idx mapping
        super().__init__(keys=valid_keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def _filter_valid_keys(self) -> List[K]:
        """
        Filter tuple dataset keys to only include pairs with valid references.

        Checks that all keys referenced in tuple_dict exist in their corresponding
        source datasets. Logs warnings about missing keys.

        Returns:
            List of valid keys (subset of tuple_dataset.keys)
        """
        valid_keys = []
        invalid_by_reason = {}  # Track which dataset is missing which keys

        # Iterate by index for efficiency
        for idx in range(len(self.tuple_dataset)):
            key = self.tuple_dataset.keys[idx]
            tuple_dict = self.tuple_dataset[idx]

            # Check if all referenced keys exist in their datasets
            is_valid = True
            for key_name, dataset in self.key_name_to_dataset.items():
                if key_name not in tuple_dict:
                    # This is a configuration error, not a data issue
                    raise KeyError(
                        f"Key '{key_name}' not found in tuple_dict for pair '{key}'. "
                        f"Available keys: {list(tuple_dict.keys())}"
                    )

                dataset_key = tuple_dict[key_name]

                # Check if the key exists in the dataset (all keys are strings)
                if dataset_key not in dataset.keys:
                    is_valid = False
                    # Track for logging
                    reason = f"{key_name}={dataset_key}"
                    if reason not in invalid_by_reason:
                        invalid_by_reason[reason] = []
                    invalid_by_reason[reason].append(key)
                    break  # No need to check other keys once we found one missing

            if is_valid:
                valid_keys.append(key)

        # Log summary of filtered pairs
        n_total = len(self.tuple_dataset.keys)
        n_valid = len(valid_keys)
        n_invalid = n_total - n_valid

        if n_invalid > 0:
            logger.warning(
                f"Filtered out {n_invalid}/{n_total} pairs ({n_invalid/n_total:.1%}) "
                f"due to missing keys in source datasets"
            )

            # Log details about what was missing (limit output)
            for reason, pair_keys in list(invalid_by_reason.items())[:10]:
                logger.warning(
                    f"  Missing {reason}: {len(pair_keys)} pairs " f"(e.g., {pair_keys[0]})"
                )

            if len(invalid_by_reason) > 10:
                logger.warning(f"  ... and {len(invalid_by_reason) - 10} more missing keys")

        return valid_keys

    @property
    def array_data(self) -> Union[List, torch.Tensor]:
        """
        Get array data from the tuple dataset.

        Note: This returns the original tuple_dataset's array_data, which may include
        filtered-out entries. Use keys property to get only valid keys.
        """
        return self.tuple_dataset.array_data

    @property
    def key_to_idx(self) -> dict[K, int]:
        """
        Get key-to-idx mapping.

        Returns a mapping based on the filtered keys, not the original tuple_dataset.
        This ensures integer indexing works correctly with filtered datasets.
        """
        # Return the mapping from BaseDataset, which is based on filtered keys
        return super().key_to_idx

    def __getitem__(self, key: K) -> Dict[str, Any]:
        """
        Get samples from multiple datasets using the tuple of keys.

        Args:
            key: Key to index into the tuple dataset.

        Returns:
            Dictionary containing data from all datasets specified in the tuple.
            Keys are renamed according to rename_map if provided.

        Raises:
            TypeError: If tuple_dataset doesn't return a dict.
            KeyError: If tuple_dataset is missing a required key name.
        """
        # Handle integer indexing - convert to actual key
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key

        # Get the tuple (dict of keys) from the tuple dataset
        tuple_dict = self.tuple_dataset[actual_key]

        if not isinstance(tuple_dict, dict):
            raise TypeError(
                f"tuple_dataset must return a dict of keys, got {type(tuple_dict)}. "
                f"Value: {tuple_dict}"
            )

        # Start with the original tuple_dict (preserves query_id, target_id, etc.)
        sample = dict(tuple_dict)

        # Fetch data from each dataset using the keys from tuple_dict
        for key_name, dataset in self.key_name_to_dataset.items():
            if key_name not in tuple_dict:
                raise KeyError(
                    f"Key '{key_name}' not found in tuple_dict. "
                    f"Available keys: {list(tuple_dict.keys())}"
                )

            # Get the key to use for this dataset
            dataset_key = tuple_dict[key_name]

            # Fetch data from the dataset
            result = dataset[dataset_key]

            # Apply rename if specified
            output_name = self.rename_map.get(key_name, key_name)

            # Handle dict vs non-dict results
            if isinstance(result, dict):
                if self.add_prefix:
                    # Add prefix to all nested keys
                    sample.update({f"{output_name}_{k}": v for k, v in result.items()})
                else:
                    # Merge directly
                    sample.update(result)
            else:
                # Use (possibly renamed) key name
                sample[output_name] = result

        return self._apply_transforms(actual_key, sample)
