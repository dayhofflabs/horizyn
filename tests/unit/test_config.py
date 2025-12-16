"""
Unit tests for the configuration system.
"""

from pathlib import Path

import pytest

from horizyn.config import (
    DotDict,
    _parse_value,
    apply_overrides,
    load_config,
    parse_overrides,
    validate_config,
)


class TestDotDict:
    """Tests for the DotDict class."""

    def test_initialization_empty(self):
        """Test DotDict initialization with no arguments."""
        config = DotDict()
        assert len(config) == 0
        assert isinstance(config, dict)

    def test_initialization_from_dict(self):
        """Test DotDict initialization from a regular dict."""
        data = {"key1": "value1", "key2": {"nested": "value2"}}
        config = DotDict(data)
        assert config["key1"] == "value1"
        assert config["key2"]["nested"] == "value2"
        assert isinstance(config["key2"], DotDict)

    def test_dot_notation_access(self):
        """Test accessing values using dot notation."""
        config = DotDict({"model": {"layers": 3, "dim": 512}})
        assert config.model.layers == 3
        assert config.model.dim == 512

    def test_dot_notation_set(self):
        """Test setting values using dot notation."""
        config = DotDict()
        config.model = {"layers": 3}
        assert config["model"]["layers"] == 3

    def test_nested_dotdict(self):
        """Test that nested dicts are converted to DotDict."""
        config = DotDict({"level1": {"level2": {"level3": "value"}}})
        assert isinstance(config.level1, DotDict)
        assert isinstance(config.level1.level2, DotDict)
        assert config.level1.level2.level3 == "value"

    def test_get_method(self):
        """Test get method with default values."""
        config = DotDict({"key1": "value1"})
        assert config.get("key1") == "value1"
        assert config.get("missing_key", "default") == "default"
        assert config.get("missing_key") is None

    def test_attribute_error(self):
        """Test that accessing missing attributes raises AttributeError."""
        config = DotDict({"key1": "value1"})
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            _ = config.missing


class TestParseValue:
    """Tests for the _parse_value helper function."""

    def test_parse_boolean_true(self):
        """Test parsing boolean true values."""
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True
        assert _parse_value("1") is True

    def test_parse_boolean_false(self):
        """Test parsing boolean false values."""
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False
        assert _parse_value("0") is False

    def test_parse_integer(self):
        """Test parsing integer values."""
        assert _parse_value("42") == 42
        assert _parse_value("-10") == -10
        assert _parse_value("0") is False  # Special case: "0" is parsed as False first

    def test_parse_float(self):
        """Test parsing float values."""
        assert _parse_value("3.14") == 3.14
        assert _parse_value("1e-4") == 1e-4
        assert _parse_value("-0.5") == -0.5

    def test_parse_string(self):
        """Test that non-numeric strings remain strings."""
        assert _parse_value("hello") == "hello"
        assert _parse_value("data/path.db") == "data/path.db"


class TestParseOverrides:
    """Tests for the parse_overrides function."""

    def test_parse_equals_format(self):
        """Test parsing overrides with = format."""
        overrides = parse_overrides(["--key1=value1", "--key2=42"])
        assert overrides["key1"] == "value1"
        assert overrides["key2"] == 42

    def test_parse_space_format(self):
        """Test parsing overrides with space-separated format."""
        overrides = parse_overrides(["--key1", "value1", "--key2", "42"])
        assert overrides["key1"] == "value1"
        assert overrides["key2"] == 42

    def test_parse_boolean_flag(self):
        """Test parsing boolean flags (no value)."""
        overrides = parse_overrides(["--flag1", "--flag2"])
        assert overrides["flag1"] is True
        assert overrides["flag2"] is True

    def test_parse_mixed_formats(self):
        """Test parsing mixed override formats."""
        overrides = parse_overrides(["--key1=value1", "--key2", "42", "--flag", "--key3=3.14"])
        assert overrides["key1"] == "value1"
        assert overrides["key2"] == 42
        assert overrides["flag"] is True
        assert overrides["key3"] == 3.14

    def test_parse_nested_keys(self):
        """Test parsing overrides with dot notation keys."""
        overrides = parse_overrides(["--training.max_epochs=50"])
        assert overrides["training.max_epochs"] == 50


class TestApplyOverrides:
    """Tests for the apply_overrides function."""

    def test_apply_simple_override(self):
        """Test applying a simple override."""
        config = DotDict({"key1": "old_value"})
        overrides = {"key1": "new_value"}
        result = apply_overrides(config, overrides)
        assert result.key1 == "new_value"

    def test_apply_nested_override(self):
        """Test applying nested overrides with dot notation."""
        config = DotDict({"training": {"max_epochs": 100}})
        overrides = {"training.max_epochs": 50}
        result = apply_overrides(config, overrides)
        assert result.training.max_epochs == 50

    def test_apply_deep_nested_override(self):
        """Test applying deeply nested overrides."""
        config = DotDict({"level1": {"level2": {"level3": "old"}}})
        overrides = {"level1.level2.level3": "new"}
        result = apply_overrides(config, overrides)
        assert result.level1.level2.level3 == "new"

    def test_apply_creates_missing_keys(self):
        """Test that overrides create missing nested keys."""
        config = DotDict({})
        overrides = {"new.nested.key": "value"}
        result = apply_overrides(config, overrides)
        assert result.new.nested.key == "value"

    def test_apply_invalid_override_type(self):
        """Test that overriding a non-dict value raises an error."""
        config = DotDict({"key": "string_value"})
        overrides = {"key.nested": "value"}
        with pytest.raises(ValueError, match="is not a dict"):
            apply_overrides(config, overrides)


class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_validate_complete_config(self):
        """Test validation passes with a complete config."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                "model": {
                    "query_encoder_dims": [2048, 4096, 512],
                    "target_encoder_dims": [1024, 4096, 512],
                    "embedding_dim": 512,
                },
                "training": {"max_epochs": 100},
            }
        )
        # Should not raise any errors
        validate_config(config)

    def test_validate_missing_section(self):
        """Test validation fails with missing top-level section."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                # Missing 'model' section
                "training": {"max_epochs": 100},
            }
        )
        with pytest.raises(ValueError, match="Missing required config section: 'model'"):
            validate_config(config)

    def test_validate_missing_data_key(self):
        """Test validation fails with missing data parameter."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train.db",
                    # Missing other required paths
                },
                "model": {
                    "query_encoder_dims": [2048, 4096, 512],
                    "target_encoder_dims": [1024, 4096, 512],
                    "embedding_dim": 512,
                },
                "training": {"max_epochs": 100},
            }
        )
        with pytest.raises(ValueError, match="Missing required data config parameter"):
            validate_config(config)

    def test_validate_missing_model_key(self):
        """Test validation fails with missing model parameter."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                "model": {
                    "query_encoder_dims": [2048, 4096, 512],
                    # Missing target_encoder_dims
                    "embedding_dim": 512,
                },
                "training": {"max_epochs": 100},
            }
        )
        with pytest.raises(ValueError, match="Missing required model config parameter"):
            validate_config(config)

    def test_validate_missing_training_key(self):
        """Test validation fails with missing training parameter."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                "model": {
                    "query_encoder_dims": [2048, 4096, 512],
                    "target_encoder_dims": [1024, 4096, 512],
                    "embedding_dim": 512,
                },
                "training": {},  # Missing max_epochs
            }
        )
        with pytest.raises(ValueError, match="Missing required training parameter"):
            validate_config(config)

    def test_validate_wrong_type_max_epochs(self):
        """Test validation fails with wrong type for max_epochs."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                "model": {
                    "query_encoder_dims": [2048, 4096, 512],
                    "target_encoder_dims": [1024, 4096, 512],
                    "embedding_dim": 512,
                },
                "training": {"max_epochs": "100"},  # Wrong type (string)
            }
        )
        with pytest.raises(ValueError, match="must be an integer"):
            validate_config(config)

    def test_validate_wrong_type_encoder_dims(self):
        """Test validation fails with wrong type for encoder dims."""
        config = DotDict(
            {
                "data": {
                    "train_pairs_path": "data/train_pairs.csv",
                    "test_pairs_path": "data/test_pairs.csv",
                    "train_reactions_path": "data/train_rxns.csv",
                    "test_reactions_path": "data/test_rxns.csv",
                    "protein_embeds_path": "data/protein_embeds.h5",
                },
                "model": {
                    "query_encoder_dims": "2048,4096,512",  # Wrong type (string)
                    "target_encoder_dims": [1024, 4096, 512],
                    "embedding_dim": 512,
                },
                "training": {"max_epochs": 100},
            }
        )
        with pytest.raises(ValueError, match="must be a list"):
            validate_config(config)


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_sota_config(self):
        """Test loading the SOTA config file."""
        config_path = Path("configs/sota.yaml")
        config = load_config(str(config_path))

        # Check top-level sections exist
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert "logging" in config

        # Check specific values
        assert config.seed == 42
        assert config.training.max_epochs == 100
        assert config.model.embedding_dim == 512
        assert config.data.train_batch_size == 16384

    def test_load_config_missing_file(self):
        """Test loading a non-existent config file raises an error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent.yaml")

    def test_load_config_with_overrides(self):
        """Test loading config with overrides applied."""
        config_path = Path("configs/sota.yaml")
        overrides = {
            "training.max_epochs": 50,
            "data.train_batch_size": 1024,
        }
        config = load_config(str(config_path), overrides=overrides)

        # Check overrides were applied
        assert config.training.max_epochs == 50
        assert config.data.train_batch_size == 1024

        # Check other values unchanged
        assert config.seed == 42
        assert config.model.embedding_dim == 512

    def test_load_config_without_validation(self):
        """Test loading config without validation."""
        config_path = Path("configs/sota.yaml")
        # Should not raise even if config is incomplete (validation skipped)
        config = load_config(str(config_path), validate=False)
        assert isinstance(config, DotDict)

    def test_load_empty_config_file(self, tmp_path):
        """Test loading an empty config file raises an error."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
            load_config(str(empty_config))

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises an error."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("key: value\n  bad indentation")

        with pytest.raises(Exception):  # YAML parsing error
            load_config(str(invalid_config))

    def test_dot_notation_access_loaded_config(self):
        """Test that loaded config supports dot notation access."""
        config_path = Path("configs/sota.yaml")
        config = load_config(str(config_path))

        # Should be able to access using dot notation
        assert config.data.train_pairs_path == "data/sota/train_pairs.csv"
        assert config.model.query_encoder_dims == [2048, 4096, 512]
        assert config.training.learning_rate == 1e-4


class TestIntegration:
    """Integration tests for the config system."""

    def test_end_to_end_config_loading(self):
        """Test complete config loading workflow."""
        # Simulate command-line arguments
        cli_args = ["--training.max_epochs=50", "--seed", "123"]

        # Parse overrides
        overrides = parse_overrides(cli_args)
        assert overrides["training.max_epochs"] == 50
        assert overrides["seed"] == 123

        # Load config with overrides
        config = load_config("configs/sota.yaml", overrides=overrides)

        # Verify overrides applied
        assert config.training.max_epochs == 50
        assert config.seed == 123

        # Verify validation passed
        assert config.data.train_pairs_path == "data/sota/train_pairs.csv"

    def test_config_modification_after_loading(self):
        """Test that config can be modified after loading."""
        config = load_config("configs/sota.yaml")

        # Modify config
        config.training.max_epochs = 200
        config.new_section = DotDict({"new_key": "new_value"})

        # Verify modifications
        assert config.training.max_epochs == 200
        assert config.new_section.new_key == "new_value"
