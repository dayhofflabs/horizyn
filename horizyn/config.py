"""
Configuration system for Horizyn.

This module provides a simple configuration system for loading and validating
YAML config files. It supports dot-notation access and command-line overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DotDict(dict):
    """A dictionary that supports dot notation access.

    Example:
        >>> config = DotDict({'model': {'layers': 3}})
        >>> config.model.layers  # Returns 3
    """

    def __init__(self, *args, **kwargs):
        """Initialize DotDict and recursively convert nested dicts."""
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        arg[k] = DotDict(v)
                kwargs.update(arg)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                kwargs[k] = DotDict(v)
        super().__init__(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation."""
        if isinstance(name, str):
            self[name] = value
        else:
            raise TypeError(f"attribute name must be string, not '{type(name).__name__}'")

    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> DotDict:
    """
    Load a YAML configuration file and apply overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dictionary of config overrides (supports dot notation keys).
        validate: Whether to validate the config structure.

    Returns:
        Loaded and validated configuration as a DotDict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.

    Example:
        >>> config = load_config('configs/sota.yaml')
        >>> config = load_config('configs/sota.yaml', {'training.max_epochs': 50})
    """
    # Load YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure the path is correct relative to the working directory."
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Convert to DotDict
    config = DotDict(config_dict)

    # Apply overrides
    if overrides:
        config = apply_overrides(config, overrides)

    # Validate config structure
    if validate:
        validate_config(config)

    return config


def apply_overrides(config: DotDict, overrides: Dict[str, Any]) -> DotDict:
    """
    Apply command-line overrides to config.

    Supports dot notation for nested keys:
        {'training.max_epochs': 50} -> config.training.max_epochs = 50

    Args:
        config: Base configuration.
        overrides: Dictionary of overrides with dot notation keys.

    Returns:
        Updated configuration.
    """
    for key, value in overrides.items():
        keys = key.split(".")
        current = config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = DotDict()
            elif not isinstance(current[k], (dict, DotDict)):
                raise ValueError(
                    f"Cannot override '{key}': '{k}' is not a dict (got {type(current[k]).__name__})"
                )
            current = current[k]

        # Set the final value
        current[keys[-1]] = value

    return config


def validate_config(config: DotDict) -> None:
    """
    Validate the configuration structure.

    Ensures all required sections and parameters are present with correct types.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If validation fails with a helpful error message.
    """
    # Check required top-level sections
    required_sections = ["data", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}'\n"
                f"Config must have sections: {required_sections}"
            )

    # Validate data section
    required_data_keys = [
        "train_pairs_path",
        "test_pairs_path",
        "train_reactions_path",
        "test_reactions_path",
        "protein_embeds_path",
    ]
    for key in required_data_keys:
        if key not in config.data:
            raise ValueError(
                f"Missing required data config parameter: 'data.{key}'\n"
                f"Required data parameters: {required_data_keys}"
            )

    # Validate model section
    required_model_keys = ["query_encoder_dims", "target_encoder_dims", "embedding_dim"]
    for key in required_model_keys:
        if key not in config.model:
            raise ValueError(
                f"Missing required model config parameter: 'model.{key}'\n"
                f"Required model parameters: {required_model_keys}"
            )

    # Validate training section
    if "max_epochs" not in config.training:
        raise ValueError("Missing required training parameter: 'training.max_epochs'")

    # Type validation
    if not isinstance(config.training.max_epochs, int):
        raise ValueError(
            f"'training.max_epochs' must be an integer, got {type(config.training.max_epochs).__name__}"
        )

    if not isinstance(config.model.query_encoder_dims, list):
        raise ValueError(
            f"'model.query_encoder_dims' must be a list, got {type(config.model.query_encoder_dims).__name__}"
        )

    if not isinstance(config.model.target_encoder_dims, list):
        raise ValueError(
            f"'model.target_encoder_dims' must be a list, got {type(config.model.target_encoder_dims).__name__}"
        )

    if not isinstance(config.model.embedding_dim, int):
        raise ValueError(
            f"'model.embedding_dim' must be an integer, got {type(config.model.embedding_dim).__name__}"
        )


def parse_overrides(args: list[str]) -> Dict[str, Any]:
    """
    Parse command-line overrides in the format --key=value or --key value.

    Args:
        args: List of command-line arguments.

    Returns:
        Dictionary of parsed overrides.

    Example:
        >>> parse_overrides(['--training.max_epochs=50', '--training.learning_rate', '1e-3'])
        {'training.max_epochs': 50, 'training.learning_rate': 0.001}
    """
    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # Remove leading dashes
            arg = arg[2:]

            # Check for = format
            if "=" in arg:
                key, value = arg.split("=", 1)
                overrides[key] = _parse_value(value)
                i += 1
            else:
                # Check for space-separated format
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    key = arg
                    value = args[i + 1]
                    overrides[key] = _parse_value(value)
                    i += 2
                else:
                    # Boolean flag (no value provided)
                    overrides[arg] = True
                    i += 1
        else:
            i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """
    Parse a string value to the appropriate Python type.

    Tries to parse as int, float, bool, or keeps as string.

    Args:
        value: String value to parse.

    Returns:
        Parsed value with appropriate type.
    """
    # Try boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Keep as string
    return value
