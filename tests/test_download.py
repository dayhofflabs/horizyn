"""
Unit tests for the scripts/download_data.py script.
"""

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestDownloadScript:
    """Tests for the download_data.py script."""

    def test_download_script_exists(self):
        """Test that download_data.py exists."""
        download_script = Path("scripts/download_data.py")
        assert download_script.exists(), "scripts/download_data.py not found"
        assert download_script.is_file(), "scripts/download_data.py is not a file"

    def test_download_script_has_shebang(self):
        """Test that download_data.py has proper shebang."""
        download_script = Path("scripts/download_data.py")
        with open(download_script, "r") as f:
            first_line = f.readline()
        assert first_line.startswith("#!/usr/bin/env python"), "Missing or incorrect shebang"

    def test_download_script_has_docstring(self):
        """Test that download_data.py has a docstring."""
        download_script = Path("scripts/download_data.py")
        with open(download_script, "r") as f:
            content = f.read()
        assert '"""' in content, "Missing docstring"
        assert "Usage:" in content, "Missing usage documentation"

    def test_download_script_imports(self):
        """Test that download_data.py can be imported."""
        # Add scripts/ to path
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data  # noqa: F401

            # Should not raise ImportError
        finally:
            sys.path.pop(0)

    def test_dataset_config_structure(self):
        """Test that DATASET_CONFIG has required fields."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            config = download_data.DATASET_CONFIG

            # Check required fields
            assert "name" in config
            assert "version" in config
            assert "url" in config
            assert "size_gb" in config
            assert "checksum" in config
            assert "files" in config

            # Check files list
            assert isinstance(config["files"], list)
            assert len(config["files"]) > 0
            assert "train_pairs.db" in config["files"]
            assert "val_pairs.db" in config["files"]
            assert "reactions.db" in config["files"]
            assert "proteins_t5.h5" in config["files"]

        finally:
            sys.path.pop(0)

    def test_verify_checksum_sha256(self, tmp_path):
        """Test checksum verification with SHA256."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Create test file
            test_file = tmp_path / "test.txt"
            test_content = b"Hello, World!"
            test_file.write_bytes(test_content)

            # Compute expected hash
            expected_hash = hashlib.sha256(test_content).hexdigest()
            checksum = f"sha256:{expected_hash}"

            # Verify
            result = download_data.verify_checksum(test_file, checksum)
            assert result is True

        finally:
            sys.path.pop(0)

    def test_verify_checksum_mismatch(self, tmp_path):
        """Test checksum verification with wrong hash."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Create test file
            test_file = tmp_path / "test.txt"
            test_file.write_bytes(b"Hello, World!")

            # Use wrong hash
            wrong_checksum = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

            # Verify
            result = download_data.verify_checksum(test_file, wrong_checksum)
            assert result is False

        finally:
            sys.path.pop(0)

    def test_verify_checksum_placeholder(self, tmp_path):
        """Test that placeholder checksum is skipped."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Create test file
            test_file = tmp_path / "test.txt"
            test_file.write_bytes(b"Hello, World!")

            # Use placeholder checksum
            placeholder_checksum = "sha256:XXXXX"

            # Verify (should skip and return True)
            result = download_data.verify_checksum(test_file, placeholder_checksum)
            assert result is True

        finally:
            sys.path.pop(0)

    def test_verify_dataset_files_all_present(self, tmp_path):
        """Test dataset file verification when all files present."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Create test files
            expected_files = ["file1.db", "file2.db", "file3.h5"]
            for filename in expected_files:
                (tmp_path / filename).write_text("test content")

            # Verify
            result = download_data.verify_dataset_files(tmp_path, expected_files)
            assert result is True

        finally:
            sys.path.pop(0)

    def test_verify_dataset_files_missing(self, tmp_path):
        """Test dataset file verification when files missing."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Create only some files
            expected_files = ["file1.db", "file2.db", "file3.h5"]
            (tmp_path / "file1.db").write_text("test content")
            # file2.db and file3.h5 are missing

            # Verify
            result = download_data.verify_dataset_files(tmp_path, expected_files)
            assert result is False

        finally:
            sys.path.pop(0)

    @patch("sys.argv", ["download_data.py", "--help"])
    def test_download_help_message(self):
        """Test that --help produces usage information."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download_data.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--output_dir" in result.stdout
        assert "--skip_checksum" in result.stdout
        assert "--force" in result.stdout

    @patch("sys.argv", ["download_data.py", "--output_dir", "data/"])
    def test_download_with_placeholder_url(self):
        """Test that placeholder URL produces helpful error."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            # Should exit with error for placeholder URL
            with pytest.raises(SystemExit) as exc_info:
                download_data.main()

            assert exc_info.value.code == 1

        finally:
            sys.path.pop(0)

    def test_download_script_has_correct_expected_files(self):
        """Test that expected files match SOTA config requirements."""
        import sys
        from pathlib import Path

        scripts_dir = Path("scripts")
        sys.path.insert(0, str(scripts_dir))

        try:
            import download_data

            expected_files = download_data.DATASET_CONFIG["files"]

            # These files are required by HorizynDataModule
            required_files = [
                "train_pairs.db",
                "val_pairs.db",
                "reactions.db",
                "proteins_t5.h5",
            ]

            for required_file in required_files:
                assert required_file in expected_files, f"Missing required file: {required_file}"

        finally:
            sys.path.pop(0)



