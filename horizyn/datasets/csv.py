"""
Dataset for loading data from CSV files.
"""

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

from horizyn.datasets.base import BaseDataset


class CSVDataset(BaseDataset[str]):
    """
    Dataset for loading data from CSV files.

    This dataset loads tabular data from CSV files into memory for fast access
    during training. It's designed for loading reaction-protein pair files
    and reaction SMILES data.

    Attributes:
        file_path (str): Path to the CSV file.
        key_column (str): Column name to use as the dataset key.
        columns (list[str]): List of column names to load as features.
        rename_map (dict[str, str]): Mapping to rename columns in output.
        _data (dict[str, dict[str, Any]]): In-memory data cache.

    Note:
        All keys are converted to strings, even if the key column contains integers.
        This eliminates ambiguity between integer indices (for DataLoader) and keys.
        - Integer access: `dataset[0]` → array index (first item)
        - String access: `dataset["809274"]` → key lookup

    Example:
        >>> # Load reaction-protein pairs
        >>> pairs_dataset = CSVDataset(
        ...     file_path="data/train_pairs.csv",
        ...     key_column="pr_id",
        ...     columns=["reaction_id", "protein_id"],
        ... )
        >>> pair = pairs_dataset["0"]  # {"reaction_id": "Rh_10008", "protein_id": "P12345"}
        >>>
        >>> # Load reaction SMILES
        >>> reactions_dataset = CSVDataset(
        ...     file_path="data/train_rxns.csv",
        ...     key_column="reaction_id",
        ...     columns=["reaction_smiles"],
        ... )
        >>> rxn = reactions_dataset["Rh_10008"]  # {"reaction_smiles": "CCO.O>>CC=O"}
    """

    def __init__(
        self,
        file_path: str,
        key_column: str,
        columns: Sequence[str] | str | None = None,
        rename_map: dict[str, str] | None = None,
        transforms: Callable[[str, Any], Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the CSVDataset.

        Args:
            file_path: Path to the CSV file. Must exist.
            key_column: Column name to use as the key for accessing data.
                This column should contain unique values.
            columns: Column names to load. If None, loads all columns except
                the key_column. Can be a single string or sequence of strings.
                Defaults to None.
            rename_map: Optional dictionary to rename columns in the output.
                Example: {"reaction_id": "query_id"}. Defaults to None.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the key_column doesn't exist in the CSV.
            ValueError: If specified columns don't exist in the CSV.
            ValueError: If no data is found in the CSV.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.key_column = key_column
        self.rename_map = rename_map or {}

        # Load CSV using standard library
        with open(self.file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []

        if len(rows) == 0:
            raise ValueError(f"No data found in CSV file: {file_path}")

        # Verify key column exists
        if key_column not in header:
            raise ValueError(
                f"Key column '{key_column}' not found in CSV. "
                f"Available columns: {header}"
            )

        # Determine columns to load
        if columns is None:
            self.columns = [c for c in header if c != key_column]
        elif isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = list(columns)

        # Verify all columns exist
        missing_cols = set(self.columns) - set(header)
        if missing_cols:
            raise ValueError(
                f"Columns {missing_cols} not found in CSV. "
                f"Available columns: {header}"
            )

        # Build data dictionary with string keys
        keys = []
        self._data: dict[str, dict[str, Any]] = {}

        for row in rows:
            key = str(row[key_column])
            keys.append(key)

            row_data = {}
            for col in self.columns:
                output_name = self.rename_map.get(col, col)
                row_data[output_name] = row[col]

            self._data[key] = row_data

        # Initialize base dataset
        super().__init__(keys=keys, transforms=transforms, **kwargs)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        """
        Get a data sample by key or integer index.

        Args:
            key: String key from the key column, or integer index (0 to len-1).
                 All CSV keys are strings (even if originally integers).

        Returns:
            Dictionary mapping column names (or renamed names) to values.

        Raises:
            KeyError: If the key is not found in the dataset.
            IndexError: If integer index is out of bounds.
        """
        # Handle integer indexing (for DataLoader)
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            actual_key = self.keys[key]
        else:
            actual_key = key

        if actual_key not in self._data:
            raise KeyError(f"Key '{actual_key}' not found in dataset")

        sample = self._data[actual_key]
        return self._apply_transforms(actual_key, sample)

