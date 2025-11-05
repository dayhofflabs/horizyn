"""
Dataset for loading data from SQLite databases.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import sqlite3
from pathlib import Path
from horizyn.datasets.base import BaseDataset


class SQLDataset(BaseDataset[str]):
    """
    Dataset for loading data from SQLite database files.

    This dataset loads tabular data from SQLite databases with support for
    in-memory caching. It's designed for loading reaction-protein pair files
    and reaction SMILES data from SQLite tables.

    The dataset loads entire tables into memory at initialization for fast access
    during training. This is suitable for datasets that fit in RAM (typically <10GB).

    Attributes:
        file_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to load.
        search_key (str): Column name to use as the dataset key.
        columns (List[str]): List of column names to load as features.
        rename_map (Dict[str, str]): Mapping to rename columns in output.
        in_memory (bool): Whether data is loaded into memory.
        _data (Dict[str, Dict[str, Any]]): In-memory data cache.

    Example:
        >>> # Load reaction-protein pairs
        >>> pairs_dataset = SQLDataset(
        ...     file_path="data/pairs.db",
        ...     table_name="pairs",
        ...     search_key="pair_id",
        ...     columns=["query_id", "target_id"],
        ...     in_memory=True
        ... )
        >>> pair = pairs_dataset["pair_0"]  # {"query_id": "rxn1", "target_id": "prot5"}
        >>>
        >>> # Load reaction SMILES
        >>> reactions_dataset = SQLDataset(
        ...     file_path="data/reactions.db",
        ...     table_name="reactions",
        ...     search_key="reaction_id",
        ...     columns=["reaction_smiles"],
        ...     in_memory=True
        ... )
        >>> rxn = reactions_dataset["rxn1"]  # {"reaction_smiles": "CCO.O>>CC=O"}
    """

    def __init__(
        self,
        file_path: str,
        table_name: str,
        search_key: str,
        columns: Optional[Union[str, Sequence[str]]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        in_memory: bool = True,
        transforms: Optional[Callable[[str, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the SQLDataset.

        Args:
            file_path: Path to the SQLite database file. Must exist.
            table_name: Name of the table in the database.
            search_key: Column name to use as the key for accessing data.
                This column should contain unique values.
            columns: Column names to load. If None, loads all columns except
                the search_key. Can be a single string or sequence of strings.
                Defaults to None.
            rename_map: Optional dictionary to rename columns in the output.
                Example: {"reaction_smiles": "smiles"}. Defaults to None.
            in_memory: Whether to load all data into memory at initialization.
                Recommended for fast training. Defaults to True.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            FileNotFoundError: If the database file doesn't exist.
            sqlite3.DatabaseError: If the table doesn't exist.
            sqlite3.DatabaseError: If the search_key column doesn't exist.
            ValueError: If columns don't exist in the table.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Database file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.table_name = table_name
        self.search_key = search_key
        self.rename_map = rename_map or {}
        self.in_memory = in_memory

        # Connect to database
        self.connection = sqlite3.connect(self.file_path)
        self.connection.row_factory = sqlite3.Row  # Access columns by name

        # Verify table exists
        self._verify_table()

        # Get and verify columns
        if columns is None:
            # Load all columns except search_key
            self.columns = self._get_all_columns()
        elif isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = list(columns)

        self._verify_columns()

        # Get keys and load data if in_memory
        keys = self._load_keys()

        if self.in_memory:
            self._data = self._load_all_data()
        else:
            self._data = None

        # Initialize base dataset
        super().__init__(keys=keys, transforms=transforms, **kwargs)

    def _verify_table(self) -> None:
        """Verify that the table exists in the database."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,)
        )
        result = cursor.fetchone()

        if result is None:
            # Get available tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            raise sqlite3.DatabaseError(
                f"Table '{self.table_name}' not found in database. " f"Available tables: {tables}"
            )
        cursor.close()

    def _get_all_columns(self) -> List[str]:
        """Get all column names from the table, excluding search_key."""
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns = [row[1] for row in cursor.fetchall() if row[1] != self.search_key]
        cursor.close()
        return columns

    def _verify_columns(self) -> None:
        """Verify that all specified columns exist in the table."""
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        table_columns = {row[1] for row in cursor.fetchall()}
        cursor.close()

        # Check search_key exists
        if self.search_key not in table_columns:
            raise sqlite3.DatabaseError(
                f"Search key column '{self.search_key}' not found in table. "
                f"Available columns: {sorted(table_columns)}"
            )

        # Check all requested columns exist
        missing_cols = set(self.columns) - table_columns
        if missing_cols:
            raise ValueError(
                f"Columns {missing_cols} not found in table '{self.table_name}'. "
                f"Available columns: {sorted(table_columns)}"
            )

    def _load_keys(self) -> List[str]:
        """Load all unique keys from the search_key column."""
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT DISTINCT {self.search_key} FROM {self.table_name} "
            f"ORDER BY {self.search_key}"
        )
        keys = [row[0] for row in cursor.fetchall()]
        cursor.close()

        if len(keys) == 0:
            raise ValueError(f"No data found in table '{self.table_name}'")

        return keys

    def _load_all_data(self) -> Dict[str, Dict[str, Any]]:
        """Load all data into memory as a dictionary."""
        cursor = self.connection.cursor()

        # Build column list for query
        cols_to_select = [self.search_key] + self.columns
        cols_str = ", ".join(cols_to_select)

        cursor.execute(f"SELECT {cols_str} FROM {self.table_name}")

        data: Dict[str, Dict[str, Any]] = {}
        for row in cursor.fetchall():
            key = row[self.search_key]

            # Build result dict with column values
            row_data = {}
            for col in self.columns:
                # Apply rename if specified
                output_name = self.rename_map.get(col, col)
                row_data[output_name] = row[col]

            # Handle multiple rows with same key (shouldn't happen if key is unique)
            if key in data:
                # For now, just keep the first occurrence
                continue

            data[key] = row_data

        cursor.close()
        return data

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Get a data sample by key.

        Args:
            key: The value from the search_key column.

        Returns:
            Dictionary mapping column names (or renamed names) to values.

        Raises:
            KeyError: If the key is not found in the dataset.
        """
        if self.in_memory:
            if self._data is None:
                raise RuntimeError("Data not loaded in memory")

            if key not in self._data:
                raise KeyError(f"Key '{key}' not found in dataset")

            sample = self._data[key]
        else:
            # Load from database on-the-fly
            cursor = self.connection.cursor()

            cols_str = ", ".join(self.columns)
            cursor.execute(
                f"SELECT {cols_str} FROM {self.table_name} WHERE {self.search_key} = ?", (key,)
            )
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                raise KeyError(f"Key '{key}' not found in table '{self.table_name}'")

            # Build result dict
            sample = {}
            for col in self.columns:
                output_name = self.rename_map.get(col, col)
                sample[output_name] = row[col]

        return self._apply_transforms(key, sample)

    def __del__(self):
        """Close database connection when dataset is destroyed."""
        if hasattr(self, "connection") and self.connection is not None:
            self.connection.close()
