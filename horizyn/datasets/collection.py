"""
Dataset classes for combining multiple datasets.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import torch
from horizyn.datasets.base import BaseDataset, K


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

    Attributes:
        tuple_dataset (BaseDataset[K]): Dataset providing tuples of keys.
        key_name_to_dataset (Dict[str, BaseDataset]): Mapping from key names to datasets.
        rename_map (Dict[str, str]): Optional mapping to rename output keys.
        add_prefix (bool): Whether to prepend dataset names to feature keys.

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
        ...     keys=["pair1", "pair2"],
        ...     array_data=[
        ...         {"query_id": "rxn1", "target_id": "prot2"},
        ...         {"query_id": "rxn3", "target_id": "prot1"}
        ...     ]
        ... )
        >>>
        >>> # Create tuple dataset
        >>> tuple_ds = TupleDataset(
        ...     tuple_dataset=pairs,
        ...     key_name_to_dataset={
        ...         "query_id": reactions,
        ...         "target_id": proteins
        ...     }
        ... )
        >>> sample = tuple_ds["pair1"]  # {"query_id": <rxn1 data>, "target_id": <prot2 data>}
    """

    def __init__(
        self,
        tuple_dataset: BaseDataset[K],
        key_name_to_dataset: Dict[str, BaseDataset],
        rename_map: Optional[Dict[str, str]] = None,
        add_prefix: bool = False,
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
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.tuple_dataset = tuple_dataset
        self.key_name_to_dataset = key_name_to_dataset
        self.rename_map = rename_map or {}
        self.add_prefix = add_prefix

        # Initialize with keys from tuple dataset
        super().__init__(keys=tuple_dataset.keys, transforms=transforms, **kwargs)

    @property
    def array_data(self) -> Union[List, torch.Tensor]:
        """Get array data from the tuple dataset."""
        return self.tuple_dataset.array_data

    @property
    def key_to_idx(self) -> dict[K, int]:
        """Get key-to-idx mapping from the tuple dataset."""
        return self.tuple_dataset.key_to_idx

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
        # Get the tuple (dict of keys) from the tuple dataset
        tuple_dict = self.tuple_dataset[key]

        if not isinstance(tuple_dict, dict):
            raise TypeError(
                f"tuple_dataset must return a dict of keys, got {type(tuple_dict)}. "
                f"Value: {tuple_dict}"
            )

        sample = {}

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

        return self._apply_transforms(key, sample)

