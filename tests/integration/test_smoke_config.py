"""Smoke tests for configuration and error handling."""

import pytest


pytestmark = pytest.mark.integration


class TestSmokeConfigAndErrors:
    """Smoke tests for configuration and error handling."""

    def test_config_override_from_command_line(self):
        """Test that command-line overrides work correctly."""
        from horizyn.config import apply_overrides, load_config

        # Load base config
        config = load_config("configs/nano.yaml")
        original_epochs = config.training.max_epochs

        # Apply override
        overrides = {"training.max_epochs": 999}
        apply_overrides(config, overrides)

        assert config.training.max_epochs == 999
        assert config.training.max_epochs != original_epochs

    def test_missing_data_files_produce_clear_errors(self):
        """Test that missing data files produce helpful error messages."""
        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule

        config = load_config("configs/nano.yaml")

        # Override to point to non-existent files
        config.data.reactions_path = "data/nonexistent/reactions.db"

        data_module = HorizynDataModule(**config.data)

        # Should raise clear error when trying to setup
        with pytest.raises(Exception) as exc_info:
            data_module.setup("fit")

        # Error message should mention the missing file
        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg or "not found" in error_msg.lower()

