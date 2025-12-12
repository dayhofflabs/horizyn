"""
Lightning DataModule for Horizyn contrastive learning.

This module loads pre-split datasets and creates dataloaders for training
and validation. All data is loaded into memory at setup time.
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset, TupleDataset
from horizyn.datasets.csv import CSVDataset
from horizyn.datasets.fingerprints import (
    DRFPFingerprintDataset,
    RDKitPlusFingerprintDataset,
)
from horizyn.datasets.hdf5 import EmbedDataset
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
        train_pairs_path: Path to training pairs CSV file.
        val_pairs_path: Path to validation pairs CSV file.
        train_reactions_path: Path to training reactions CSV file.
        val_reactions_path: Path to validation reactions CSV file.
        protein_embeds_path: Path to protein embeddings HDF5 file.
        train_batch_size: Batch size for training. Defaults to 16384.
        retrieval_batch_size: Batch size for retrieval metrics. Defaults to 128.
        num_workers: Number of dataloader workers. Defaults to 4.
        pin_memory: Whether to pin memory. Defaults to False.
        rdkit_fp_dim: Dimension of RDKit+ fingerprints. Defaults to 1024.
        drfp_dim: Dimension of DRFP fingerprints. Defaults to 1024.
        standardize_reactions: Whether to standardize reactions. Defaults to True.
        standardize_hypervalent: Whether to standardize hypervalent atoms. Defaults to True.
        standardize_remove_hs: Whether to remove explicit hydrogen atoms. Defaults to True.
        standardize_kekulize: Whether to kekulize aromatic compounds. Defaults to False.
        standardize_uncharge: Whether to uncharge molecules. Defaults to True.
        standardize_metals: Whether to standardize metals. Defaults to True.

    Example:
        >>> dm = HorizynDataModule(
        ...     train_pairs_path="data/sota/train_pairs.csv",
        ...     val_pairs_path="data/sota/val_pairs.csv",
        ...     train_reactions_path="data/sota/train_rxns.csv",
        ...     val_reactions_path="data/sota/val_rxns.csv",
        ...     protein_embeds_path="data/sota/protein_embeds.h5",
        ... )
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
        >>> val_loaders = dm.val_dataloader()
    """

    def __init__(
        self,
        train_pairs_path: str,
        val_pairs_path: str,
        train_reactions_path: str,
        val_reactions_path: str,
        protein_embeds_path: str,
        train_batch_size: int = 16384,
        retrieval_batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
        rdkit_fp_dim: int = 1024,
        drfp_dim: int = 1024,
        standardize_reactions: bool = True,
        standardize_hypervalent: bool = True,
        standardize_remove_hs: bool = True,
        standardize_kekulize: bool = False,
        standardize_uncharge: bool = True,
        standardize_metals: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store paths
        self.train_pairs_path = Path(train_pairs_path)
        self.val_pairs_path = Path(val_pairs_path)
        self.train_reactions_path = Path(train_reactions_path)
        self.val_reactions_path = Path(val_reactions_path)
        self.protein_embeds_path = Path(protein_embeds_path)

        # Batch sizes
        self.train_batch_size = train_batch_size
        self.retrieval_batch_size = retrieval_batch_size

        # DataLoader settings
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Fingerprint settings
        self.rdkit_fp_dim = rdkit_fp_dim
        self.drfp_dim = drfp_dim
        self.standardize_reactions = standardize_reactions
        self.standardize_hypervalent = standardize_hypervalent
        self.standardize_remove_hs = standardize_remove_hs
        self.standardize_kekulize = standardize_kekulize
        self.standardize_uncharge = standardize_uncharge
        self.standardize_metals = standardize_metals

        # Dataset containers (populated in setup)
        self._train_data = None
        self._val_data = None
        self._val_query_data = None
        self._train_query_data = None  # Training query dataset (train reactions)
        self._val_query_data_raw = None  # Validation query dataset (val reactions)
        self._target_data = None  # Protein embeddings
        self._screening_target_data = None  # Full screening set

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

    def _augment_pairs_bidirectional(self, pairs: BaseDataset) -> BaseDataset:
        """
        Augment pairs with bidirectional reaction variants.

        For each pair (reaction_id, protein_id):
        - Forward: (reaction_id_f, protein_id)
        - Backward: (reaction_id_r, protein_id)

        Args:
            pairs: Dataset with query_id (reaction_id) and target_id (protein_id).

        Returns:
            Augmented dataset with both forward and backward pairs.
        """
        augmented_keys = []
        augmented_data = []

        for pair_key in pairs.keys:
            pair_data = pairs[pair_key]
            query_id = pair_data["query_id"]
            target_id = pair_data["target_id"]

            # Forward direction
            augmented_keys.append(f"{pair_key}_f")
            augmented_data.append(
                {
                    "query_id": f"{query_id}_f",
                    "target_id": target_id,
                }
            )

            # Backward direction
            augmented_keys.append(f"{pair_key}_r")
            augmented_data.append(
                {
                    "query_id": f"{query_id}_r",
                    "target_id": target_id,
                }
            )

        return BaseDataset(keys=augmented_keys, array_data=augmented_data)

    def _setup_training_data(self):
        """Setup training dataset with fingerprints and embeddings."""
        print("Setting up training data...")

        # Load training pairs
        train_pairs = CSVDataset(
            file_path=str(self.train_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        original_pair_count = len(train_pairs)
        print(f"  Loaded {original_pair_count} training pairs")

        # Augment pairs with bidirectional reactions (forward + backward)
        train_pairs = self._augment_pairs_bidirectional(train_pairs)
        print(f"  Augmented to {len(train_pairs)} bidirectional pairs")

        # Load query data (train reactions with fingerprints)
        self._train_query_data = self._create_query_dataset(self.train_reactions_path)
        print(f"  Loaded {len(self._train_query_data)} train reactions")

        # Load target data (protein embeddings)
        self._target_data = self._create_target_dataset()
        print(f"  Loaded {len(self._target_data)} proteins")

        # Merge pairs with query and target data
        self._train_data = TupleDataset(
            tuple_dataset=train_pairs,
            key_name_to_dataset={
                "query_id": self._train_query_data,
                "target_id": self._target_data,
            },
            rename_map={
                "query_id": "query_vec",
                "target_id": "target_vec",
            },
        )

        print(f"Training dataset ready: {len(self._train_data)} samples")

    def _setup_validation_data(self):
        """Setup validation dataset for retrieval metrics."""
        print("Setting up validation data...")

        # Load validation pairs
        val_pairs = CSVDataset(
            file_path=str(self.val_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        original_val_pair_count = len(val_pairs)
        print(f"  Loaded {original_val_pair_count} validation pairs")

        # Augment validation pairs with bidirectional reactions (forward + backward)
        val_pairs = self._augment_pairs_bidirectional(val_pairs)
        print(f"  Augmented to {len(val_pairs)} bidirectional pairs")

        # Load validation query data (val reactions with fingerprints)
        self._val_query_data_raw = self._create_query_dataset(self.val_reactions_path)
        print(f"  Loaded {len(self._val_query_data_raw)} val reactions")

        # Ensure target data is loaded
        if self._target_data is None:
            raise RuntimeError(
                "Training data must be setup before validation data. "
                "Target dataset must be loaded first."
            )

        # Load full protein embeddings for screening
        full_protein_dataset = EmbedDataset(
            file_path=str(self.protein_embeds_path),
            in_memory=True,
        )
        self._screening_target_data = full_protein_dataset

        # Collect protein stats
        train_pairs_for_stats = CSVDataset(
            file_path=str(self.train_pairs_path),
            key_column="pr_id",
            columns=["reaction_id", "protein_id"],
            rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
        )
        train_protein_ids = set(
            train_pairs_for_stats[k]["target_id"] for k in train_pairs_for_stats.keys
        )
        val_protein_ids = set(val_pairs[k]["target_id"] for k in val_pairs.keys)
        screening_protein_ids = set(full_protein_dataset.keys)

        print(f"  Training proteins: {len(train_protein_ids)}")
        print(f"  Validation proteins: {len(val_protein_ids)}")
        print(f"  Screening set (full): {len(screening_protein_ids)} proteins")
        print(f"  Overlap (train ∩ val): {len(train_protein_ids & val_protein_ids)}")
        print(f"  Val-only proteins: {len(val_protein_ids - train_protein_ids)}")

        # Validation data for loss computation
        self._val_data = TupleDataset(
            tuple_dataset=val_pairs,
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
                "target_id": self._target_data,
            },
            rename_map={
                "query_id": "query_vec",
                "target_id": "target_vec",
            },
        )

        # Group pairs by query_id for multi-label retrieval metrics
        query_to_targets = defaultdict(list)
        for pair_key in val_pairs.keys:
            pair = val_pairs[pair_key]
            query_id = pair["query_id"]
            target_id = pair["target_id"]
            query_to_targets[query_id].append(target_id)

        # Get unique query IDs (sorted for determinism)
        unique_query_ids = sorted(query_to_targets.keys())

        # Create retrieval dataset: maps query_id -> list of target_ids
        retrieval_array_data = [query_to_targets[qid] for qid in unique_query_ids]

        # Store query-to-targets mapping
        self._query_to_targets = query_to_targets

        # Create dataset for retrieval queries (unique queries only)
        self._val_query_data = TupleDataset(
            tuple_dataset=BaseDataset(
                keys=unique_query_ids,
                array_data=[{"query_id": qid} for qid in unique_query_ids],
            ),
            key_name_to_dataset={
                "query_id": self._val_query_data_raw,
            },
            rename_map={
                "query_id": "query_vec",
            },
        )

        # Store target list dataset for metrics computation
        self._val_retrieval_targets = BaseDataset(
            keys=unique_query_ids,
            array_data=retrieval_array_data,
        )

        print(f"Validation dataset ready: {len(self._val_data)} samples")
        print(f"  Unique validation queries: {len(unique_query_ids)}")
        print(f"  Avg targets per query: {len(val_pairs) / len(unique_query_ids):.2f}")

    def _augment_reactions_bidirectional(self, reactions: BaseDataset) -> BaseDataset:
        """
        Augment reactions with bidirectional variants (forward and reverse).

        For each reaction with key 'rxn_id' and SMILES 'A>>B':
        - Forward: key='rxn_id_f', smiles='A>>B'
        - Backward: key='rxn_id_r', smiles='B>>A'

        Args:
            reactions: Dataset with reaction_smiles.

        Returns:
            Augmented dataset with both forward and backward reactions.
        """
        augmented_keys = []
        augmented_data = []

        for rxn_id in reactions.keys:
            rxn_data = reactions[rxn_id]
            smiles = rxn_data["reaction_smiles"]

            # Forward direction
            augmented_keys.append(f"{rxn_id}_f")
            augmented_data.append({"reaction_smiles": smiles})

            # Backward direction (reverse reactants and products)
            if ">>" in smiles:
                parts = smiles.split(">>")
                if len(parts) == 2:
                    reversed_smiles = f"{parts[1]}>>{parts[0]}"
                    augmented_keys.append(f"{rxn_id}_r")
                    augmented_data.append({"reaction_smiles": reversed_smiles})
                else:
                    # Malformed SMILES - skip backward
                    print(f"Warning: Malformed reaction SMILES for {rxn_id}: {smiles}")
            else:
                # No '>>' separator - skip backward
                print(f"Warning: No '>>' separator in reaction SMILES for {rxn_id}: {smiles}")

        return BaseDataset(keys=augmented_keys, array_data=augmented_data)

    def _create_query_dataset(self, reactions_path: Path):
        """
        Create query dataset with RDKit+ and DRFP fingerprints.

        Reactions are augmented bidirectionally (forward + backward) to double
        the training data and ensure reversible reactions are learned properly.

        Args:
            reactions_path: Path to reactions CSV file.

        Returns:
            Dataset that returns concatenated 2048-dim fingerprints.
        """
        # Load reaction SMILES
        reactions = CSVDataset(
            file_path=str(reactions_path),
            key_column="reaction_id",
            columns=["reaction_smiles"],
        )

        # Augment with bidirectional reactions (forward + backward)
        reactions = self._augment_reactions_bidirectional(reactions)
        print(f"  Augmented to {len(reactions)} bidirectional reactions")

        # Generate RDKit+ fingerprints (1024-dim)
        rdkit_fp = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.rdkit_fp_dim,
            mol_fp_type="morgan",
            rxn_fp_type="struct",
            use_chirality=True,
            standardize=self.standardize_reactions,
            standardize_hypervalent=self.standardize_hypervalent,
            standardize_remove_hs=self.standardize_remove_hs,
            standardize_kekulize=self.standardize_kekulize,
            standardize_uncharge=self.standardize_uncharge,
            standardize_metals=self.standardize_metals,
        )

        # Generate DRFP fingerprints (1024-dim)
        drfp_fp = DRFPFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=self.drfp_dim,
            radius=3,
            rings=True,
            standardize=self.standardize_reactions,
            standardize_hypervalent=self.standardize_hypervalent,
            standardize_remove_hs=self.standardize_remove_hs,
            standardize_kekulize=self.standardize_kekulize,
            standardize_uncharge=self.standardize_uncharge,
            standardize_metals=self.standardize_metals,
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
            file_path=str(self.protein_embeds_path),
            in_memory=True,
        )

    @property
    def train_data(self):
        """Access to training dataset (read-only)."""
        return self._train_data

    @property
    def val_data(self):
        """Access to validation dataset (read-only)."""
        return self._val_data

    @property
    def val_query_data(self):
        """Access to validation query dataset (read-only)."""
        return self._val_query_data

    @property
    def val_retrieval_pairs(self):
        """Access to validation query dataset (alias for compatibility)."""
        return self._val_query_data

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
            2. Target lookup table dataloader (FULL screening set)
            3. Query retrieval metrics dataloader

        Returns:
            List of DataLoaders for validation.
        """
        if self._val_data is None or self._screening_target_data is None:
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

        # 2. Target lookup table (FULL screening set for retrieval metrics)
        dataloaders.append(
            DataLoader(
                self._screening_target_data,
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
