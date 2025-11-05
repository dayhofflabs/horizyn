"""
Lightning DataModule for Horizyn contrastive learning.

This module loads pre-split datasets and creates dataloaders for training
and validation. All data is loaded into memory at setup time.
"""

from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from horizyn.datasets.collection import MergeDataset, TupleDataset
from horizyn.datasets.fingerprints import (
    DRFPFingerprintDataset,
    RDKitPlusFingerprintDataset,
)
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.sql import SQLDataset
from horizyn.datasets.transform import ConcatTensorTransform
from horizyn.utils import default_collate


class HorizynDataModule(pl.LightningDataModule):
    """
    DataModule for Horizyn SOTA contrastive learning.

    Loads pre-split training and validation data, generates fingerprints for
    reactions, and creates dataloaders. All data is loaded into memory at
    setup time for fast training.

    SOTA Configuration:
        - Reactions: RDKit+ (1024-dim) + DRFP (1024-dim) concatenated
        - Proteins: T5 embeddings (1024-dim) pre-computed
        - Training batch size: 16384
        - Retrieval batch size: 128

    Args:
        train_pairs_path: Path to training pairs SQLite file.
        val_pairs_path: Path to validation pairs SQLite file.
        reactions_path: Path to reactions SQLite file.
        proteins_path: Path to protein embeddings HDF5 file.
        train_batch_size: Batch size for training. Defaults to 16384.
        retrieval_batch_size: Batch size for retrieval metrics. Defaults to 128.
        num_workers: Number of dataloader workers. Defaults to 4.
        pin_memory: Whether to pin memory. Defaults to False.
        rdkit_fp_dim: Dimension of RDKit+ fingerprints. Defaults to 1024.
        drfp_dim: Dimension of DRFP fingerprints. Defaults to 1024.
        standardize: Whether to standardize SMILES. Defaults to True.

    Example:
        >>> dm = HorizynDataModule(
        ...     train_pairs_path="data/train_pairs.db",
        ...     val_pairs_path="data/val_pairs.db",
        ...     reactions_path="data/reactions.db",
        ...     proteins_path="data/proteins_t5.h5",
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
        >>> val_loaders = dm.val_dataloader()
    """

    def __init__(
        self,
        train_pairs_path: str,
        val_pairs_path: str,
        reactions_path: str,
        proteins_path: str,
        train_batch_size: int = 16384,
        retrieval_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        rdkit_fp_dim: int = 1024,
        drfp_dim: int = 1024,
        standardize: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store paths
        self.train_pairs_path = Path(train_pairs_path)
        self.val_pairs_path = Path(val_pairs_path)
        self.reactions_path = Path(reactions_path)
        self.proteins_path = Path(proteins_path)

        # Batch sizes
        self.train_batch_size = train_batch_size
        self.retrieval_batch_size = retrieval_batch_size

        # DataLoader settings
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Fingerprint settings
        self.rdkit_fp_dim = rdkit_fp_dim
        self.drfp_dim = drfp_dim
        self.standardize = standardize

        # Dataset containers (populated in setup)
        self._train_data = None
        self._val_data = None
        self._val_query_data = None
        self._query_data = None  # Shared query dataset (reactions with fingerprints)
        self._target_data = None  # Shared target dataset (protein embeddings)

    def setup(self, stage: Optional[str] = None):
        """
        Load all datasets into memory.

        This loads pairs, reactions, and proteins, generates fingerprints,
        and caches everything in memory for fast training.

        Args:
            stage: Stage ('fit', 'validate', 'test', or 'predict').
        """
        if stage in (None, "fit", "validate"):
            self._setup_training_data()
            self._setup_validation_data()

    def _setup_training_data(self):
        """Setup training dataset with fingerprints and embeddings."""
        print("Setting up training data...")

        # Load training pairs
        train_pairs = SQLDataset(
            file_path=str(self.train_pairs_path),
            table_name="pairs",
            search_key="pair_id",
            columns=["query_id", "target_id"],
            in_memory=True,
        )
        print(f"  Loaded {len(train_pairs)} training pairs")

        # Load query data (reactions with fingerprints) - shared with validation
        self._query_data = self._create_query_dataset()
        print(f"  Loaded {len(self._query_data)} reactions")

        # Load target data (protein embeddings) - shared with validation
        self._target_data = self._create_target_dataset()
        print(f"  Loaded {len(self._target_data)} proteins")

        # Merge pairs with query and target data
        self._train_data = TupleDataset(
            tuple_dataset=train_pairs,
            key_name_to_dataset={
                "query": self._query_data,
                "target": self._target_data,
            },
            rename_map={
                "query": "query_vec",
                "target": "target_vec",
            },
        )

        print(f"Training dataset ready: {len(self._train_data)} samples")

    def _setup_validation_data(self):
        """Setup validation dataset for retrieval metrics."""
        print("Setting up validation data...")

        # Load validation pairs
        val_pairs = SQLDataset(
            file_path=str(self.val_pairs_path),
            table_name="pairs",
            search_key="pair_id",
            columns=["query_id", "target_id"],
            in_memory=True,
        )
        print(f"  Loaded {len(val_pairs)} validation pairs")

        # Reuse query and target datasets from training (already cached in memory)
        if self._query_data is None or self._target_data is None:
            raise RuntimeError(
                "Training data must be setup before validation data. "
                "Query and target datasets are shared between train and validation."
            )

        # Validation data for loss computation
        self._val_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query": self._query_data,
                "target": self._target_data,
            },
            rename_map={
                "query": "query_vec",
                "target": "target_vec",
            },
        )

        # Store datasets for retrieval metrics
        self._val_query_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query": self._query_data,
            },
            rename_map={
                "query": "query_vec",
            },
        )

        print(f"Validation dataset ready: {len(self._val_data)} samples")

    def _create_query_dataset(self):
        """
        Create query dataset with RDKit+ and DRFP fingerprints.

        Returns:
            Dataset that returns concatenated 2048-dim fingerprints.
        """
        # Load reaction SMILES
        reactions = SQLDataset(
            file_path=str(self.reactions_path),
            table_name="reactions",
            search_key="reaction_id",
            columns=["reaction_smiles"],
            in_memory=True,
        )

        # Generate RDKit+ fingerprints (1024-dim)
        rdkit_fp = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.rdkit_fp_dim,
            mol_fp_type="morgan",
            rxn_fp_type="struct",
            use_chirality=True,
            standardize=self.standardize,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )

        # Generate DRFP fingerprints (1024-dim)
        drfp_fp = DRFPFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.drfp_dim,
            radius=3,
            rings=True,
            standardize=self.standardize,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )

        # Merge fingerprints
        merged = MergeDataset(
            datasets={"rdkit": rdkit_fp, "drfp": drfp_fp},
            add_prefix=False,
        )

        # Concatenate into single vector (2048-dim)
        merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))

        return merged

    def _create_target_dataset(self):
        """
        Create target dataset with pre-computed T5 embeddings.

        Returns:
            Dataset that returns 1024-dim T5 embeddings.
        """
        return EmbedDataset(
            file_path=str(self.proteins_path),
            in_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.

        Returns:
            DataLoader with training data.
        """
        if self._train_data is None:
            raise RuntimeError("Training data not setup. Call setup() first.")

        return DataLoader(
            self._train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=default_collate,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """
        Create validation dataloaders.

        Returns list of dataloaders:
            1. Validation loss dataloader
            2. Target lookup table dataloader
            3. Query retrieval metrics dataloader

        Returns:
            List of DataLoaders for validation.
        """
        if self._val_data is None or self._target_data is None:
            raise RuntimeError("Validation data not setup. Call setup() first.")

        dataloaders = []

        # 1. Validation loss dataloader
        dataloaders.append(
            DataLoader(
                self._val_data,
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=default_collate,
            )
        )

        # 2. Target lookup table (for retrieval metrics)
        dataloaders.append(
            DataLoader(
                self._target_data,
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=default_collate,
            )
        )

        # 3. Query retrieval metrics
        dataloaders.append(
            DataLoader(
                self._val_query_data,
                batch_size=self.retrieval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=default_collate,
            )
        )

        return dataloaders
