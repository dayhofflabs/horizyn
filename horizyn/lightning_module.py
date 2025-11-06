"""
Lightning module for Horizyn contrastive learning.

This module implements the training and validation loop for the dual-encoder
contrastive learning model using Multi-Label Noise Contrastive Estimation (MLNCE).
"""

from typing import Any, Dict, List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from horizyn.losses import FullBatchMLNCELoss
from horizyn.metrics import create_retrieval_metrics
from horizyn.model import DualContrastiveModel


class HorizynLitModule(pl.LightningModule):
    """
    Lightning module for Horizyn SOTA contrastive learning.

    This module orchestrates training and validation for the dual-encoder
    contrastive model. It computes MLNCE loss during training, and evaluates
    retrieval metrics during validation.

    SOTA Configuration:
        - Model: DualContrastiveModel with MLP encoders
        - Loss: FullBatchMLNCELoss with beta=10.0
        - Optimizer: AdamW with lr=1e-4
        - Distance: Cosine distance

    Args:
        query_encoder_dims: Dimensions for query (reaction) encoder MLP.
        target_encoder_dims: Dimensions for target (protein) encoder MLP.
        embedding_dim: Output embedding dimension (default: 512).
        beta: Inverse temperature for MLNCE loss (default: 10.0).
        learn_beta: Whether to learn beta parameter (default: False).
        learning_rate: Learning rate for AdamW optimizer (default: 1e-4).
        weight_decay: Weight decay for AdamW optimizer (default: 0.01).
        metric_ks: List of k values for top-k metrics (default: [1, 5, 10, 50]).

    Example:
        >>> lit_module = HorizynLitModule(
        ...     query_encoder_dims=[2048, 4096, 512],
        ...     target_encoder_dims=[1024, 4096, 512],
        ...     embedding_dim=512,
        ...     beta=10.0,
        ...     learn_beta=False,
        ...     learning_rate=1e-4,
        ... )
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(lit_module, datamodule=data_module)
    """

    def __init__(
        self,
        query_encoder_dims: List[int],
        target_encoder_dims: List[int],
        embedding_dim: int = 512,
        beta: float = 10.0,
        learn_beta: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        metric_ks: List[int] = [1, 5, 10, 50],
        pos_score: bool = False,
        neg_score: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate and map encoder dims to MLP kwargs
        if len(query_encoder_dims) < 2 or len(target_encoder_dims) < 2:
            raise ValueError(
                "Encoder dims must include at least [input_dim, output_dim]. "
                f"Got query_encoder_dims={query_encoder_dims}, target_encoder_dims={target_encoder_dims}"
            )
        # Fail fast if the provided dims' output differs from embedding_dim
        if query_encoder_dims[-1] != embedding_dim:
            raise ValueError(
                f"query_encoder_dims final element must equal embedding_dim. "
                f"Got query_encoder_dims[-1]={query_encoder_dims[-1]} vs embedding_dim={embedding_dim}"
            )
        if target_encoder_dims[-1] != embedding_dim:
            raise ValueError(
                f"target_encoder_dims final element must equal embedding_dim. "
                f"Got target_encoder_dims[-1]={target_encoder_dims[-1]} vs embedding_dim={embedding_dim}"
            )

        # Build encoder kwargs from dims: [in, h1, ..., hN, out]
        # widths correspond to hidden layers only (exclude input and output dims)
        query_hidden_widths = query_encoder_dims[1:-1]
        target_hidden_widths = target_encoder_dims[1:-1]

        query_encoder_kwargs = {
            "input_dim": query_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": max(len(query_hidden_widths), 0),
            "widths": query_hidden_widths if len(query_hidden_widths) > 0 else [],
            "normalise_output": True,
        }

        target_encoder_kwargs = {
            "input_dim": target_encoder_dims[0],
            "output_dim": embedding_dim,
            "num_layers": max(len(target_hidden_widths), 0),
            "widths": target_hidden_widths if len(target_hidden_widths) > 0 else [],
            "normalise_output": True,
        }

        # Model
        self.model = DualContrastiveModel(
            query_encoder_kwargs=query_encoder_kwargs,
            target_encoder_kwargs=target_encoder_kwargs,
        )

        # Loss function
        self.loss_fn = FullBatchMLNCELoss(
            beta=beta,
            learn_beta=learn_beta,
        )

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Metrics
        self.metric_functionals = create_retrieval_metrics(
            top_k=metric_ks, include_mrr=True, pos_score=pos_score, neg_score=neg_score
        )

        # Target lookup table for validation retrieval metrics
        self.target_lookup_table = None
        self.num_targets = None

    def forward(
        self, query_vec: torch.Tensor, target_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dual encoder model.

        Args:
            query_vec: Query input vectors (e.g., reaction fingerprints).
            target_vec: Target input vectors (e.g., protein embeddings).

        Returns:
            Tuple of (query_embeddings, target_embeddings).
        """
        return self.model(query_vec, target_vec)

    def _compute_cosine_distances(
        self, query_vecs: torch.Tensor, target_vecs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine distances between query and target embeddings.

        Args:
            query_vecs: Query embeddings of shape (num_queries, embedding_dim).
            target_vecs: Target embeddings of shape (num_targets, embedding_dim).

        Returns:
            Distance matrix of shape (num_queries, num_targets).
        """
        # Cosine distance = 1 - cosine_similarity
        # Both inputs are already L2-normalized by the model
        return 1.0 - torch.matmul(query_vecs, target_vecs.T)

    def _deduplicate_inputs(
        self, vecs: torch.Tensor, ids: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deduplicate input vectors based on unique IDs.

        This is used to avoid redundant forward passes when the same
        query or target appears multiple times in a batch.

        Args:
            vecs: Input vectors of shape (batch_size, vec_dim).
            ids: List of IDs corresponding to each vector.

        Returns:
            Tuple of (deduplicated_vectors, inverse_indices).
            - deduplicated_vectors: Unique vectors.
            - inverse_indices: Indices to reconstruct original batch.
        """
        _, unique_idx, unique_inverse = np.unique(ids, return_index=True, return_inverse=True)

        # Convert to tensors
        unique_idx_tensor = torch.from_numpy(unique_idx).to(dtype=torch.long, device=self.device)
        unique_inverse_tensor = torch.from_numpy(unique_inverse).to(
            dtype=torch.long, device=self.device
        )

        # Get unique vectors
        unique_vecs = vecs[unique_idx_tensor]

        return unique_vecs, unique_inverse_tensor

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step: compute embeddings and MLNCE loss.

        Args:
            batch: Batch dict containing:
                - query_vec: Reaction fingerprints (batch_size, query_dim)
                - target_vec: Protein embeddings (batch_size, target_dim)
                - query_id: List of query IDs
                - target_id: List of target IDs
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        query_vecs = batch["query_vec"]
        target_vecs = batch["target_vec"]
        query_ids = batch["query_id"]
        target_ids = batch["target_id"]

        batch_size = len(query_ids)

        # Deduplicate inputs (same query/target may appear multiple times)
        unique_query_vecs, unique_inverse_query_ids = self._deduplicate_inputs(
            query_vecs, query_ids
        )
        unique_target_vecs, unique_inverse_target_ids = self._deduplicate_inputs(
            target_vecs, target_ids
        )

        # Encode
        query_embeds, target_embeds = self.model(unique_query_vecs, unique_target_vecs)

        # Compute distances
        dists = self._compute_cosine_distances(query_embeds, target_embeds)

        # Compute loss
        loss = self.loss_fn(dists, unique_inverse_query_ids, unique_inverse_target_ids)

        # Log
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        if self.loss_fn.learn_beta:
            self.log(
                "train/beta",
                self.loss_fn.beta,
                on_epoch=True,
                on_step=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return loss

    def on_validation_epoch_start(self):
        """Initialize target lookup table at the start of validation."""
        # Get datamodule to determine number of targets
        datamodule = self.trainer.datamodule

        # Preallocate target lookup table using FULL screening set (train + val proteins)
        # CRITICAL: Must use _screening_target_data, not _target_data
        self.num_targets = len(datamodule._screening_target_data)
        vec_dim = self.model.target_encoder.output_dim

        self.target_lookup_table = torch.zeros(
            (self.num_targets, vec_dim), dtype=torch.float32, device=self.device
        )

        # Create mapping from target IDs to lookup table indices
        self.target_id_to_idx = {
            target_id: idx for idx, target_id in enumerate(datamodule._screening_target_data.keys)
        }

    def _update_target_lookup_table(self, batch: Dict[str, Any], target_embeds: torch.Tensor):
        """
        Update the target lookup table with encoded target embeddings.

        Args:
            batch: Batch dict containing target_lookup_row_idx.
            target_embeds: Encoded target embeddings.
        """
        row_indices = batch["target_lookup_row_idx"]  # CPU tensor

        # Gather across DDP ranks
        world_size = self.trainer.world_size

        gathered_embeds = self.all_gather(target_embeds)
        if world_size == 1 and gathered_embeds.dim() == target_embeds.dim():
            gathered_embeds = gathered_embeds.unsqueeze(0)

        row_indices_device = row_indices.to(self.device)
        gathered_row_indices = self.all_gather(row_indices_device)
        if world_size == 1 and gathered_row_indices.dim() == row_indices_device.dim():
            gathered_row_indices = gathered_row_indices.unsqueeze(0)

        # Update lookup table from all ranks
        for rank in range(world_size):
            embeds_from_rank = gathered_embeds[rank].float()
            indices_from_rank = gathered_row_indices[rank]

            if indices_from_rank.numel() > 0:
                self.target_lookup_table[indices_from_rank] = embeds_from_rank

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor | None:
        """
        Validation step: compute loss or retrieval metrics.

        The datamodule returns 3 dataloaders:
            0. Validation loss dataloader
            1. Target lookup table dataloader
            2. Query retrieval metrics dataloader

        Args:
            batch: Batch dict (contents vary by dataloader_idx).
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader (0, 1, or 2).

        Returns:
            Loss value for dataloader 0, None otherwise.
        """
        if dataloader_idx == 0:
            # Compute validation loss
            return self._validation_loss_step(batch, batch_idx)

        elif dataloader_idx == 1:
            # Build target lookup table
            self._validation_lookup_step(batch, batch_idx)
            return None

        else:
            # Compute retrieval metrics
            self._validation_retrieval_step(batch, batch_idx)
            return None

    def _validation_loss_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Compute validation loss.

        Args:
            batch: Batch dict containing query_vec, target_vec, query_id, target_id.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        query_vecs = batch["query_vec"]
        target_vecs = batch["target_vec"]
        query_ids = batch["query_id"]
        target_ids = batch["target_id"]

        batch_size = len(query_ids)

        # Deduplicate inputs
        unique_query_vecs, unique_inverse_query_ids = self._deduplicate_inputs(
            query_vecs, query_ids
        )
        unique_target_vecs, unique_inverse_target_ids = self._deduplicate_inputs(
            target_vecs, target_ids
        )

        # Encode
        query_embeds, target_embeds = self.model(unique_query_vecs, unique_target_vecs)

        # Compute distances
        dists = self._compute_cosine_distances(query_embeds, target_embeds)

        # Compute loss
        loss = self.loss_fn(dists, unique_inverse_query_ids, unique_inverse_target_ids)

        # Log
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss

    def _validation_lookup_step(self, batch: torch.Tensor | Dict[str, Any], batch_idx: int):
        """
        Build target lookup table by encoding all target embeddings.

        Args:
            batch: Either a tensor (target vectors) or batch dict containing
                   target_vec and target_lookup_row_idx.
            batch_idx: Index of the batch.
        """
        # Get datamodule for batch size info
        datamodule = self.trainer.datamodule

        # Handle both tensor and dict inputs
        if isinstance(batch, torch.Tensor):
            # Batch is just the target vectors from EmbedDataset
            target_vecs = batch
            # Compute row indices based on batch_idx and batch_size
            batch_size = target_vecs.shape[0]
            # Use train_batch_size since val loader 1 uses that
            start_idx = batch_idx * datamodule.train_batch_size
            row_indices = torch.arange(start_idx, start_idx + batch_size, dtype=torch.long)
            # Create a dict for _update_target_lookup_table
            batch_dict = {"target_lookup_row_idx": row_indices}
        else:
            # Batch is a dict (for backward compatibility with tests)
            target_vecs = batch["target_vec"]
            batch_dict = batch

        # Encode targets
        target_embeds = self.model.target_encoder(target_vecs)

        # Update lookup table
        self._update_target_lookup_table(batch_dict, target_embeds)

        # Synchronize across DDP ranks
        self.trainer.strategy.barrier()

    def _validation_retrieval_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Compute retrieval metrics using the target lookup table.

        Each query can have multiple valid targets (multi-label retrieval).
        Metrics check if ANY valid target appears in top-K.

        Args:
            batch: Batch dict containing:
                - query_vec: Query vectors (batch_size, query_dim)
                - query_id: List of query IDs (used to get target lists)
            batch_idx: Index of the batch.
        """
        query_vecs = batch["query_vec"]
        query_ids = batch["query_id"]

        # Encode queries
        query_embeds = self.model.query_encoder(query_vecs)

        # Compute distances to all targets in lookup table
        dists = self._compute_cosine_distances(query_embeds, self.target_lookup_table)

        # Convert distances to scores (higher is better)
        scores = -dists

        # Get datamodule to access target lists
        datamodule = self.trainer.datamodule

        # Compute metrics for each query
        batch_size = query_embeds.shape[0]

        for metric_name, metric_func in self.metric_functionals.items():
            metric_values = []
            for idx in range(batch_size):
                # Get query ID and its list of valid target IDs
                query_id = query_ids[idx]

                # Get all valid target IDs for this query from the retrieval dataset
                valid_target_ids = datamodule._val_retrieval_targets[query_id]

                # Convert target IDs to lookup table indices
                target_indices = []
                for target_id in valid_target_ids:
                    if target_id in self.target_id_to_idx:
                        target_indices.append(self.target_id_to_idx[target_id])

                # Create tensor of target indices (padded with -1 for metric functions)
                if len(target_indices) == 0:
                    # No valid targets found - skip this query
                    continue

                target_idx_tensor = torch.tensor(
                    target_indices, dtype=torch.long, device=self.device
                )

                # Compute metric (metrics handle multiple targets via torch.isin)
                metric_value = metric_func(scores[idx], target_idx_tensor)
                metric_values.append(metric_value)

            # Log mean metric
            if len(metric_values) > 0:
                metric_tensor = torch.stack(metric_values)
                self.log(
                    f"val/{metric_name}",
                    metric_tensor.mean(),
                    on_epoch=True,
                    on_step=False,
                    prog_bar=(metric_name == "top_1_hit_rate"),  # Show Top-1 in progress bar
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        """
        Configure AdamW optimizer.

        Returns:
            Optimizer instance.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
