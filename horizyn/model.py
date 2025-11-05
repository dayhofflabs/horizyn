"""
Neural network model architectures for Horizyn.

This module contains the base model classes and MLP implementation used in
the Horizyn contrastive learning model.
"""

import copy
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

        # Validate core hyperparameters early (fail fast)
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if isinstance(widths, int):
            if widths <= 0 and num_layers > 0:
                raise ValueError("widths must be a positive integer when num_layers > 0")
        else:
            if len(widths) == 0 and num_layers > 0:
                raise ValueError("widths list must be non-empty when num_layers > 0")
            if any(w <= 0 for w in widths):
                raise ValueError("all hidden layer widths must be positive integers")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0]")

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
    ) -> None:
        """
        Build the main neural network structure.

        Constructs the layers of the MLP based on the provided parameters,
        including linear layers, activations, layer normalization, and dropout.

        Notes:
            - If `widths` is a list, its length defines the number of hidden layers
              and overrides `num_layers`.
            - If `activations` is provided as a single nn.Module instance, a deep copy
              of that instance is used per hidden layer to avoid reusing the same
              module object across layers.

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
            # Use deep copies so each layer gets its own module instance
            activations = [copy.deepcopy(activations) for _ in range(num_layers)]
        if len(activations) != num_layers:
            raise ValueError("Number of activations must match number of hidden layers")
        if any(not isinstance(act, nn.Module) for act in activations):
            raise ValueError("All activations must be instances of nn.Module")

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


class DualContrastiveModel(BaseModel):
    """
    Dual encoder contrastive learning model for Horizyn.

    This model uses separate encoders for query (reaction) and target (protein)
    inputs, producing normalized embeddings for contrastive learning. This is the
    core architecture of the Horizyn SOTA model.

    The model enforces that both encoders output normalized embeddings by checking
    for a NormalizeLayer as the final layer in each encoder.
    """

    def __init__(
        self,
        query_encoder_kwargs: dict[str, Any],
        target_encoder_kwargs: dict[str, Any],
        query_encoder: type[BaseModel] = MLP,
        target_encoder: type[BaseModel] = MLP,
        enforce_normalisation: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the DualContrastiveModel.

        Args:
            query_encoder_kwargs: Keyword arguments for query encoder (reactions).
            target_encoder_kwargs: Keyword arguments for target encoder (proteins).
            query_encoder: Query encoder class (default: MLP).
            target_encoder: Target encoder class (default: MLP).
            enforce_normalisation: Whether to enforce that encoders have normalized outputs.
            *args: Additional arguments (must be empty).
            **kwargs: Additional keyword arguments (must be empty).

        Raises:
            ValueError: If enforce_normalisation is True and encoders don't have
                NormalizeLayer as final layer.

        Example:
            >>> # SOTA configuration
            >>> model = DualContrastiveModel(
            ...     query_encoder_kwargs={
            ...         "input_dim": 2048,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ...     target_encoder_kwargs={
            ...         "input_dim": 1024,
            ...         "output_dim": 512,
            ...         "num_layers": 1,
            ...         "widths": 4096,
            ...         "normalise_output": True,
            ...     },
            ... )
        """
        super().__init__(*args, **kwargs)
        self.query_encoder = query_encoder(**query_encoder_kwargs)
        self.target_encoder = target_encoder(**target_encoder_kwargs)

        # Validate that encoders have normalized outputs
        if enforce_normalisation:
            if not isinstance(self.query_encoder, BaseModel):
                raise ValueError("query_encoder must be a BaseModel instance")
            if not isinstance(self.target_encoder, BaseModel):
                raise ValueError("target_encoder must be a BaseModel instance")

            # Check that query encoder has NormalizeLayer as last layer
            if len(self.query_encoder.layers) == 0 or not isinstance(
                self.query_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "query_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in query_encoder_kwargs."
                )

            # Check that target encoder has NormalizeLayer as last layer
            if len(self.target_encoder.layers) == 0 or not isinstance(
                self.target_encoder.layers[-1], NormalizeLayer
            ):
                raise ValueError(
                    "target_encoder must have a NormalizeLayer as its last layer. "
                    "Set normalise_output=True in target_encoder_kwargs."
                )

    def forward(
        self, query_inputs: dict | torch.Tensor, target_inputs: dict | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dual contrastive model.

        Encodes both query and target inputs through their respective encoders.
        Supports both tensor and dict inputs for flexibility.

        Args:
            query_inputs: Input for query encoder (reactions).
                Can be a tensor or dict of tensors.
            target_inputs: Input for target encoder (proteins).
                Can be a tensor or dict of tensors.

        Returns:
            Tuple of (query_embeddings, target_embeddings), both normalized.

        Raises:
            ValueError: If query and target embeddings have different dimensions.

        Example:
            >>> query_fps = torch.randn(16, 2048)  # Reaction fingerprints
            >>> target_embs = torch.randn(16, 1024)  # Protein T5 embeddings
            >>> query_out, target_out = model(query_fps, target_embs)
            >>> query_out.shape, target_out.shape
            (torch.Size([16, 512]), torch.Size([16, 512]))
        """
        # Encode query inputs
        query = (
            self.query_encoder(**query_inputs)
            if isinstance(query_inputs, dict)
            else self.query_encoder(query_inputs)
        )

        # Encode target inputs
        target = (
            self.target_encoder(**target_inputs)
            if isinstance(target_inputs, dict)
            else self.target_encoder(target_inputs)
        )

        # Validate output dimensions match
        if query.shape[1] != target.shape[1]:
            raise ValueError(
                f"Query and target encoder output dimension mismatch: "
                f"query={query.shape[1]}, target={target.shape[1]}"
            )

        return query, target
