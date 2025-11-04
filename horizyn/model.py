"""
Neural network model architectures for Horizyn.

This module contains the base model classes and MLP implementation used in
the Horizyn contrastive learning model.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for all models in Horizyn.

    Provides a structured way to organize model layers into pre-processing,
    main body, and post-processing stages, along with optional output heads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the base model.

        Args:
            *args: Variable length argument list (should be empty).
            **kwargs: Arbitrary keyword arguments (should be empty).

        Raises:
            ValueError: If extra arguments are provided.
        """
        super(BaseModel, self).__init__()
        if args or kwargs:
            error_msg = (
                f"Extra unused arguments provided to BaseModel: args={args}, kwargs={kwargs}"
            )
            raise ValueError(error_msg)

        # Define the pre-nn layers (preprocessing)
        self.pre_nn_layers = nn.ModuleList()
        # Define the main body of nn layers
        self.main_nn = nn.ModuleList()
        # Define the post-nn layers (postprocessing)
        self.post_nn_layers = nn.ModuleList()
        # Optional output heads for multi-task learning
        self.output_heads = nn.ModuleDict()

    @property
    def model_body(self) -> nn.ModuleList:
        """
        Get the main body of the model (all layers excluding output heads).

        Returns:
            ModuleList containing all pre-processing, main, and post-processing layers.
        """
        return nn.ModuleList([*self.pre_nn_layers, *self.main_nn, *self.post_nn_layers])

    @property
    def layers(self) -> nn.ModuleList:
        """
        Get all layers in the model including output heads.

        Returns:
            ModuleList containing all model layers.
        """
        return nn.ModuleList(
            [
                *self.pre_nn_layers,
                *self.main_nn,
                *self.post_nn_layers,
                *self.output_heads.values(),
            ]
        )

    @property
    def num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor, or dict of outputs if output heads are defined.
        """
        for layer in self.model_body:
            x = layer(x)
        # Handle multiple output heads if present
        if len(self.output_heads) > 0:
            return {key: head(x) for key, head in self.output_heads.items()}
        return x


class NormalizeLayer(nn.Module):
    """
    Normalization layer for L2 normalization of tensors.

    This layer normalizes input tensors along a specified dimension using the
    L2 norm (Euclidean distance). Commonly used to normalize embeddings in
    contrastive learning.
    """

    def __init__(self, p: float = 2, dim: int = -1, eps: float = 1e-12):
        """
        Initialize the NormalizeLayer.

        Args:
            p: The p-norm to use for normalization (default: 2 for L2 norm).
            dim: The dimension along which to compute the norm (default: -1, last dimension).
            eps: Small value for numerical stability (default: 1e-12).
        """
        super(NormalizeLayer, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x: Input tensor to be normalized.

        Returns:
            Normalized tensor with unit norm along the specified dimension.
        """
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """
        Return string representation of layer parameters for printing.

        Returns:
            String describing layer configuration.
        """
        return f"p={self.p}, dim={self.dim}, eps={self.eps}"


class MLP(BaseModel):
    """
    Multi-Layer Perceptron (MLP) neural network.

    Implements a flexible MLP architecture with customizable layers, activation
    functions, layer normalization, dropout, and optional output normalization.
    This is the primary encoder architecture used in the Horizyn SOTA model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 1,
        widths: int | list[int] = 32,
        activations: nn.Module | list[nn.Module] = nn.ReLU(),
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalise_output: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output features.
            num_layers: Number of hidden layers (default: 1).
            widths: Width(s) of hidden layers. If int, all layers have same width.
                If list, each element specifies width of corresponding layer.
            activations: Activation function(s). If single Module, used for all layers.
                If list, each element specifies activation for corresponding layer.
            use_layer_norm: Whether to apply layer normalization after each hidden layer.
            dropout: Dropout probability (0.0 means no dropout).
            bias: Whether to include bias terms in linear layers.
            normalise_output: Whether to L2-normalize the final output.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Example:
            >>> # SOTA reaction encoder: 2048 → 4096 → 512
            >>> mlp = MLP(
            ...     input_dim=2048,
            ...     output_dim=512,
            ...     num_layers=1,
            ...     widths=4096,
            ...     normalise_output=True
            ... )
        """
        super(MLP, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build the main neural network
        self._build_network(num_layers, widths, activations, use_layer_norm, dropout, bias)

        # Add output normalization if requested
        if normalise_output:
            self.post_nn_layers.append(NormalizeLayer(p=2, dim=-1))

    def _build_network(
        self,
        num_layers: int,
        widths: int | list[int],
        activations: nn.Module | list[nn.Module],
        use_layer_norm: bool,
        dropout: float,
        bias: bool,
    ):
        """
        Build the main neural network structure.

        Constructs the layers of the MLP based on the provided parameters,
        including linear layers, activations, layer normalization, and dropout.

        Args:
            num_layers: Number of hidden layers.
            widths: Width(s) of hidden layers.
            activations: Activation function(s) to use.
            use_layer_norm: Whether to use layer normalization.
            dropout: Dropout probability.
            bias: Whether to include bias in linear layers.
        """
        # Ensure widths is a list
        if isinstance(widths, int):
            widths = [widths] * num_layers
        else:
            num_layers = len(widths)

        # Ensure activations is a list
        if not isinstance(activations, list):
            activations = [activations] * num_layers

        assert (
            len(activations) == num_layers
        ), "Number of activations must match number of hidden layers"

        prev_dim = self.input_dim

        # Construct hidden layers
        for width, activation in zip(widths, activations):
            self.main_nn.append(nn.Linear(prev_dim, width, bias=bias))
            self.main_nn.append(activation)
            if use_layer_norm:
                self.main_nn.append(nn.LayerNorm(width))
            if dropout > 0:
                self.main_nn.append(nn.Dropout(dropout))
            prev_dim = width

        # Add output layer
        self.main_nn.append(nn.Linear(prev_dim, self.output_dim, bias=bias))

