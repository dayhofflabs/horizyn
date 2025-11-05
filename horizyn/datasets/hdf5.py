"""
Dataset for loading embeddings from HDF5 files.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional

import h5py
import torch

from horizyn.datasets.base import BaseDataset


class EmbedDataset(BaseDataset[str]):
    """
    Dataset for loading pre-computed embeddings from HDF5 files.

    This dataset loads vector embeddings (e.g., protein T5 embeddings) from HDF5
    files with support for in-memory caching. HDF5 files are expected to have:
        - 'ids': dataset of string identifiers (shape: [N])
        - 'vectors': dataset of embedding vectors (shape: [N, D])

    Where N is the number of embeddings and D is the embedding dimension.

    For training efficiency, the entire embedding matrix can be loaded into
    memory at initialization. This is recommended for datasets that fit in RAM
    (typically <16GB for embedding matrices).

    Attributes:
        file_path (str): Path to the HDF5 file.
        in_memory (bool): Whether embeddings are loaded into memory.
        dtype (torch.dtype): Data type for returned tensors.
        num_vecs (int): Number of vectors in the dataset.
        vec_dim (int): Dimension of each vector.
        file (h5py.File): Open HDF5 file handle (if not in_memory).
        data (torch.Tensor): In-memory tensor of all embeddings (if in_memory).

    Example:
        >>> # Load protein T5 embeddings
        >>> protein_embeds = EmbedDataset(
        ...     file_path="data/swissprot_t5.h5",
        ...     in_memory=True,
        ...     dtype=torch.float32
        ... )
        >>> embedding = protein_embeds["P12345"]  # Returns tensor of shape [1024]
        >>> print(f"Dataset has {len(protein_embeds)} proteins")
        >>> print(f"Embedding dim: {protein_embeds.vec_dim}")
    """

    def __init__(
        self,
        file_path: str,
        in_memory: bool = True,
        dtype: torch.dtype = torch.float32,
        transforms: Optional[Callable[[str, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the EmbedDataset.

        Args:
            file_path: Path to the HDF5 file. Must exist and contain 'ids' and
                'vectors' datasets.
            in_memory: Whether to load all embeddings into memory at initialization.
                Recommended for training (faster access). Defaults to True.
            dtype: PyTorch data type for returned tensors. Defaults to torch.float32.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist.
            KeyError: If 'ids' or 'vectors' datasets are missing from the file.
            ValueError: If 'ids' and 'vectors' have mismatched lengths.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        self.file_path = str(file_path_obj)
        self.in_memory = in_memory
        self.dtype = dtype

        # Open HDF5 file
        self.file = h5py.File(self.file_path, "r")

        # Verify required datasets exist
        if "ids" not in self.file:
            available = list(self.file.keys())
            self.file.close()
            raise KeyError(
                f"Required dataset 'ids' not found in HDF5 file. "
                f"Available datasets: {available}"
            )

        if "vectors" not in self.file:
            available = list(self.file.keys())
            self.file.close()
            raise KeyError(
                f"Required dataset 'vectors' not found in HDF5 file. "
                f"Available datasets: {available}"
            )

        # Get shapes
        self.num_vecs, self.vec_dim = self.file["vectors"].shape
        num_ids = len(self.file["ids"])

        if self.num_vecs != num_ids:
            self.file.close()
            raise ValueError(
                f"Mismatch between number of ids ({num_ids}) and " f"vectors ({self.num_vecs})"
            )

        # Load ids (always load these, they're small)
        # Decode bytes to strings if necessary
        ids_data = self.file["ids"][:]
        if ids_data.dtype.kind == "S" or ids_data.dtype.kind == "O":
            # Byte strings or object array
            keys = [
                id_val.decode("utf-8") if isinstance(id_val, bytes) else str(id_val)
                for id_val in ids_data
            ]
        else:
            keys = [str(id_val) for id_val in ids_data]

        # Load vectors into memory if requested
        if self.in_memory:
            # Load all vectors at once
            self.data = torch.from_numpy(self.file["vectors"][:]).to(dtype=self.dtype)
            # Close file since we don't need it anymore
            self.file.close()
            self.file = None
        else:
            self.data = None

        # Initialize base dataset with key_to_idx mapping enabled
        super().__init__(keys=keys, use_key_to_idx=True, transforms=transforms, **kwargs)

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Get an embedding vector by key.

        Args:
            key: The identifier from the 'ids' dataset.

        Returns:
            Tensor of shape [vec_dim] containing the embedding.

        Raises:
            KeyError: If the key is not found in the dataset.
        """
        # Get the index for this key
        idx = self._get_idx(key)

        if self.in_memory:
            if self.data is None:
                raise RuntimeError("Data not initialized")
            vector = self.data[idx]
        else:
            if self.file is None:
                raise RuntimeError("HDF5 file is closed")
            # Load from disk on-the-fly
            # torch.from_numpy avoids copying and shares memory with numpy array
            vector = torch.from_numpy(self.file["vectors"][idx]).to(dtype=self.dtype)

        return self._apply_transforms(key, vector)

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, "file") and self.file is not None:
            self.file.close()
