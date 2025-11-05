"""
Loss functions for contrastive learning.

This module implements Multi-Label Noise Contrastive Estimation (MLNCE) and related
contrastive loss functions for dual-encoder architectures.
"""

from typing import Optional

import torch
from torch import nn


class FullBatchNCELoss(nn.Module):
    """
    Base class for Noise Contrastive Estimation (NCE) losses that utilize an entire
    batch to construct negative examples.

    This loss function supports a learnable inverse temperature parameter (beta) that
    controls the scale of the distance metric. Beta is stored in log space for
    numerical stability and can be constrained to a specified range during training.

    The loss operates on pairwise distances between query and target embeddings,
    where each query-target pair can be labeled as positive (matching) or negative
    (non-matching). The full batch of targets serves as a pool of negatives for
    each query.

    Attributes:
        beta_init (float): Initial value for the beta parameter.
        learn_beta (bool): Whether beta is a learnable parameter.
        beta_min (float): Minimum allowed value for beta (when learned).
        beta_max (float): Maximum allowed value for beta (when learned).
        logbeta (nn.Parameter): The beta parameter in log space.
    """

    def __init__(
        self,
        beta: float = 1.0,
        learn_beta: bool = False,
        beta_min: float = -float("inf"),
        beta_max: float = float("inf"),
        *args,
        **kwargs,
    ):
        """
        Initialize the FullBatchNCELoss.

        Args:
            beta (float, optional): Initial inverse temperature parameter that controls
                the scale of distances. Higher values make the loss more sensitive to
                small distance differences. Defaults to 1.0.
            learn_beta (bool, optional): Whether to learn beta during training.
                Defaults to False.
            beta_min (float, optional): Minimum value for beta when learning. Only
                enforced if learn_beta is True. Defaults to -inf (no constraint).
            beta_max (float, optional): Maximum value for beta when learning. Only
                enforced if learn_beta is True. Defaults to inf (no constraint).
            *args: Additional positional arguments passed to nn.Module.
            **kwargs: Additional keyword arguments passed to nn.Module.
        """
        super().__init__(*args, **kwargs)

        self.beta_init = beta
        self.learn_beta = learn_beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        self._setup_beta(self.beta_init, self.learn_beta)

    def _setup_beta(self, beta_init: float, learn_beta: bool) -> None:
        """
        Set up the beta parameter in log space.

        Beta is stored as log(beta) for numerical stability, particularly when
        learning beta. This prevents beta from becoming negative and improves
        gradient behavior.

        Args:
            beta_init (float): Initial value for beta.
            learn_beta (bool): Whether to learn beta during training.
        """
        self.logbeta = nn.Parameter(torch.log(torch.tensor(beta_init)), requires_grad=learn_beta)

        if learn_beta:
            self.register_forward_pre_hook(self._clip_beta)

    @property
    def beta(self) -> torch.Tensor:
        """
        Get the beta parameter in linear space.

        Returns:
            torch.Tensor: The inverse temperature parameter (scalar).
        """
        return torch.exp(self.logbeta)

    def _clip_beta(self, module: nn.Module, input: tuple) -> None:
        """
        Clip beta to the specified range [beta_min, beta_max].

        This method is registered as a forward pre-hook when learn_beta is True,
        ensuring beta stays within the specified bounds during training.

        Args:
            module (nn.Module): The current module (self). Required for hook signature.
            input (tuple): Input to the forward pass. Required for hook signature.
        """
        beta = self.beta.data
        beta = torch.clamp(beta, self.beta_min, self.beta_max)
        self.logbeta.data = torch.log(beta)

    def forward(
        self, dists: torch.Tensor, query_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the NCE loss for a batch of data.

        This method must be implemented by subclasses to define the specific loss
        computation strategy.

        Args:
            dists (torch.Tensor): Pairwise distances of shape (num_queries, num_targets).
                Each entry dists[i, j] is the distance between query i and target j.
                For cosine similarity, this is typically 1 - cosine_similarity.
            query_idx (torch.Tensor): Indices of queries in positive pairs, shape (num_pairs,).
                Each entry is a row index into the dists tensor.
            target_idx (torch.Tensor): Indices of targets in positive pairs, shape (num_pairs,).
                Each entry is a column index into the dists tensor.

        Returns:
            torch.Tensor: Scalar loss value for the batch.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class FullBatchMLNCELoss(FullBatchNCELoss):
    """
    Multi-Label Noise Contrastive Estimation (MLNCE) loss.

    MLNCE is a contrastive loss function that handles the case where each query can
    have multiple correct targets (multi-label), unlike standard InfoNCE which assumes
    one-to-one query-target relationships.

    The loss maximizes the likelihood of positive pairs relative to all possible pairs
    in the batch. Specifically, it treats the problem as learning a probability
    distribution over all query-target pairs, where positive pairs should have high
    probability.

    Mathematical Formulation:
        For a batch with Q queries and T targets, let:
        - d_ij be the distance between query i and target j
        - P be the set of positive (query, target) pairs
        - β (beta) be the inverse temperature parameter

        The MLNCE loss is:
            L = (β / |P|) * Σ_{(i,j) ∈ P} d_ij + log(Σ_i Σ_j exp(-β * d_ij))

        where:
        - The first term encourages small distances for positive pairs
        - The second term (partition function) normalizes over all pairs

    Key Properties:
        - Handles multiple positive targets per query naturally
        - Symmetric with respect to queries and targets
        - Scales to large batch sizes (full-batch contrastive learning)
        - Temperature parameter β controls the concentration of the distribution

    Usage Example:
        >>> loss_fn = FullBatchMLNCELoss(beta=10.0, learn_beta=False)
        >>> # Compute distances (e.g., 1 - cosine_similarity for normalized embeddings)
        >>> dists = 1 - torch.mm(query_embeds, target_embeds.t())  # (Q, T)
        >>> # Define positive pairs
        >>> query_idx = torch.tensor([0, 0, 1, 2])  # Query 0 has 2 targets
        >>> target_idx = torch.tensor([0, 3, 1, 2])
        >>> loss = loss_fn(dists, query_idx, target_idx)
    """

    def forward(
        self, dists: torch.Tensor, query_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the MLNCE loss for a batch of data.

        Args:
            dists (torch.Tensor): Pairwise distances of shape (num_queries, num_targets).
                For normalized embeddings with cosine similarity, use:
                dists = 1 - torch.mm(query_embeds, target_embeds.t())
            query_idx (torch.Tensor): Indices of queries in positive pairs, shape (num_pairs,).
                Each entry is a row index into the dists tensor.
            target_idx (torch.Tensor): Indices of targets in positive pairs, shape (num_pairs,).
                Each entry is a column index into the dists tensor.

        Returns:
            torch.Tensor: Scalar loss value for the batch.

        Note:
            The loss is computed as:
            1. Extract distances of positive pairs: pos_dists = dists[query_idx, target_idx]
            2. Compute global partition function: logZ = logsumexp(-beta * dists)
            3. Return: mean(beta * pos_dists) + logZ
        """
        # Extract the distances of the positive pairs (num_pairs,)
        pos_dists = dists[query_idx, target_idx]

        # Compute the partition function over all pairs in the batch
        # logZ = log(Σ_i Σ_j exp(-β * d_ij))
        logZ = torch.logsumexp(-self.beta * dists, dim=(0, 1))

        # MLNCE loss: mean distance of positives + global partition function
        return self.beta * pos_dists.mean() + logZ
