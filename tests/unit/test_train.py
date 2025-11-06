"""
Unit tests for the train.py entry point.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the main function (we'll mock most of its dependencies)
# Note: We can't directly import train.main() without triggering imports,
# so we'll test the components instead


class TestTrainScript:
    """Tests for the train.py script."""

    def test_train_script_exists(self):
        """Test that train.py exists and is executable."""
        train_script = Path("train.py")
        assert train_script.exists(), "train.py not found"
        assert train_script.is_file(), "train.py is not a file"

    def test_train_script_has_shebang(self):
        """Test that train.py has proper shebang."""
        train_script = Path("train.py")
        with open(train_script, "r") as f:
            first_line = f.readline()
        assert first_line.startswith("#!/usr/bin/env python"), "Missing or incorrect shebang"

    def test_train_script_has_docstring(self):
        """Test that train.py has a docstring."""
        train_script = Path("train.py")
        with open(train_script, "r") as f:
            content = f.read()
        assert '"""' in content, "Missing docstring"
        assert "Usage:" in content, "Missing usage documentation"

    def test_train_script_imports(self):
        """Test that train.py imports correctly."""
        import train  # noqa: F401

        # Should not raise ImportError

    @patch("sys.argv", ["train.py"])
    def test_train_no_config_argument(self):
        """Test that missing --config argument produces error."""
        import train

        with pytest.raises(SystemExit) as exc_info:
            train.main()

        # argparse exits with code 2 for argument errors
        assert exc_info.value.code == 2

    def test_train_help_message(self):
        """Test that --help produces usage information."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--config" in result.stdout
        assert "--seed" in result.stdout
        assert "--resume" in result.stdout
