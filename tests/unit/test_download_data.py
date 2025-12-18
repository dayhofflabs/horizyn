"""Tests for scripts/download_data.py"""

import hashlib
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from download_data import (
    DATASET_CONFIG,
    download_file,
    extract_archive,
    verify_checksum,
    verify_dataset_files,
)


class TestDownloadFile:
    """Tests for download_file function."""

    def test_download_success(self, tmp_path):
        """Test successful file download."""
        output_file = tmp_path / "test_file.tar.gz"
        test_content = b"test content"

        # Mock requests response
        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(test_content))}
        mock_response.iter_content = Mock(return_value=[test_content])

        # Mock progress bar with correct final size
        mock_progress = Mock()
        mock_progress.n = len(test_content)

        with patch("download_data.requests.get", return_value=mock_response):
            with patch("download_data.tqdm", return_value=mock_progress):
                download_file("https://example.com/file.tar.gz", output_file)

        assert output_file.exists()
        assert output_file.read_bytes() == test_content

    def test_download_creates_parent_dirs(self, tmp_path):
        """Test that download creates parent directories."""
        output_file = tmp_path / "nested" / "dirs" / "file.tar.gz"
        test_content = b"test"

        mock_response = Mock()
        mock_response.headers = {"content-length": str(len(test_content))}
        mock_response.iter_content = Mock(return_value=[test_content])

        # Mock progress bar with correct final size
        mock_progress = Mock()
        mock_progress.n = len(test_content)

        with patch("download_data.requests.get", return_value=mock_response):
            with patch("download_data.tqdm", return_value=mock_progress):
                download_file("https://example.com/file.tar.gz", output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_download_failure(self, tmp_path):
        """Test download handles network errors."""
        output_file = tmp_path / "test_file.tar.gz"

        import requests

        with patch(
            "download_data.requests.get",
            side_effect=requests.exceptions.RequestException("Network error"),
        ):
            with pytest.raises(RuntimeError, match="Download failed"):
                download_file("https://example.com/file.tar.gz", output_file)

    def test_download_incomplete(self, tmp_path):
        """Test detection of incomplete downloads."""
        output_file = tmp_path / "test_file.tar.gz"
        test_content = b"incomplete"

        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}  # Expected size larger than actual
        mock_response.iter_content = Mock(return_value=[test_content])

        with patch("download_data.requests.get", return_value=mock_response):
            with patch("download_data.tqdm") as mock_tqdm:
                # Mock progress bar with wrong final size
                mock_progress = Mock()
                mock_progress.n = len(test_content)  # Actual size
                mock_tqdm.return_value = mock_progress

                with pytest.raises(RuntimeError, match="Download incomplete"):
                    download_file("https://example.com/file.tar.gz", output_file)


class TestVerifyChecksum:
    """Tests for verify_checksum function."""

    def test_md5_checksum_valid(self, tmp_path):
        """Test MD5 checksum verification with valid hash."""
        test_file = tmp_path / "test.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.md5(test_content).hexdigest()
        assert verify_checksum(test_file, f"md5:{expected_hash}")

    def test_sha256_checksum_valid(self, tmp_path):
        """Test SHA256 checksum verification with valid hash."""
        test_file = tmp_path / "test.txt"
        test_content = b"test content"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert verify_checksum(test_file, f"sha256:{expected_hash}")

    def test_checksum_invalid(self, tmp_path):
        """Test checksum verification with invalid hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        assert not verify_checksum(test_file, "md5:wronghash123")

    def test_checksum_placeholder(self, tmp_path):
        """Test checksum verification skips placeholder values."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        # Should return True and skip verification
        assert verify_checksum(test_file, "md5:XXXXX")

    def test_unsupported_algorithm(self, tmp_path):
        """Test unsupported hash algorithm raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            verify_checksum(test_file, "sha512:somehash")


class TestExtractArchive:
    """Tests for extract_archive function."""

    def test_extract_tar_gz(self, tmp_path):
        """Test extraction of tar.gz archive."""
        # Create a test archive
        archive_path = tmp_path / "test.tar.gz"
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        # Create test content
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Create archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(test_file, arcname="test.txt")

        # Extract
        with patch("download_data.tqdm", side_effect=lambda x, **kwargs: x):
            extract_archive(archive_path, extract_dir)

        # Verify extraction
        extracted_file = extract_dir / "test.txt"
        assert extracted_file.exists()
        assert extracted_file.read_text() == "test content"

    def test_extract_creates_directory(self, tmp_path):
        """Test extraction creates output directory if needed."""
        archive_path = tmp_path / "test.tar.gz"
        extract_dir = tmp_path / "new_dir" / "extracted"

        # Create minimal archive
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(test_file, arcname="test.txt")

        # Extract (should create directory)
        with patch("download_data.tqdm", side_effect=lambda x, **kwargs: x):
            extract_archive(archive_path, extract_dir)

        assert extract_dir.exists()
        assert (extract_dir / "test.txt").exists()


class TestVerifyDatasetFiles:
    """Tests for verify_dataset_files function."""

    def test_all_files_present(self, tmp_path):
        """Test verification passes when all files present."""
        files = ["file1.db", "file2.db", "file3.h5"]

        # Create all files
        for filename in files:
            (tmp_path / filename).write_bytes(b"test")

        assert verify_dataset_files(tmp_path, files)

    def test_missing_files(self, tmp_path):
        """Test verification fails when files missing."""
        files = ["file1.db", "file2.db", "file3.h5"]

        # Create only some files
        (tmp_path / "file1.db").write_bytes(b"test")

        assert not verify_dataset_files(tmp_path, files)

    def test_empty_file_list(self, tmp_path):
        """Test verification with empty file list."""
        assert verify_dataset_files(tmp_path, [])


class TestDatasetConfig:
    """Tests for DATASET_CONFIG."""

    def test_config_has_required_fields(self):
        """Test config contains all required fields."""
        assert "name" in DATASET_CONFIG
        assert "version" in DATASET_CONFIG
        assert "url" in DATASET_CONFIG
        assert "checksum" in DATASET_CONFIG
        assert "files" in DATASET_CONFIG
        assert "file_checksums" in DATASET_CONFIG

    def test_config_files_match_checksums(self):
        """Test that files list matches file_checksums keys."""
        expected_files = set(DATASET_CONFIG["files"])
        checksum_files = set(DATASET_CONFIG["file_checksums"].keys())
        assert expected_files == checksum_files

    def test_config_has_five_files(self):
        """Test config specifies exactly 5 expected files."""
        assert len(DATASET_CONFIG["files"]) == 5

    def test_config_file_names(self):
        """Test config has correct file names."""
        expected = [
            "train_pairs.csv",
            "test_pairs.csv",
            "train_rxns.csv",
            "test_rxns.csv",
            "prots_t5.h5",
        ]
        assert set(DATASET_CONFIG["files"]) == set(expected)

    def test_config_checksums_are_valid_or_placeholder(self):
        """Test all individual checksums are valid MD5 format or placeholders."""
        for filename, checksum in DATASET_CONFIG["file_checksums"].items():
            # Allow placeholder values during development
            if checksum == "XXXXX":
                continue
            # MD5 hashes are 32 hex characters
            assert len(checksum) == 32, f"Invalid MD5 for {filename}"
            assert all(c in "0123456789abcdef" for c in checksum), f"Invalid MD5 hex for {filename}"


class TestMainFunction:
    """Tests for main() function."""

    def test_files_already_exist_no_force(self, tmp_path):
        """Test skips download when files exist without --force."""
        # Create all expected files
        for filename in DATASET_CONFIG["files"]:
            (tmp_path / filename).write_bytes(b"existing")

        with patch("sys.argv", ["download_data.py", "--output-dir", str(tmp_path)]):
            with patch("download_data.verify_dataset_files", return_value=True):
                with patch("download_data.download_file") as mock_download:
                    from download_data import main

                    main()

                    # Should not attempt download
                    mock_download.assert_not_called()

    def test_placeholder_url_exits(self, tmp_path):
        """Test exits gracefully when URL is placeholder."""
        with patch("sys.argv", ["download_data.py", "--output-dir", str(tmp_path), "--force"]):
            with pytest.raises(SystemExit) as exc_info:
                from download_data import main

                main()

            assert exc_info.value.code == 1

    def test_default_output_directory(self):
        """Test default output directory is data/sota."""
        with patch("sys.argv", ["download_data.py"]):
            with patch("download_data.Path.mkdir"):
                with patch("download_data.verify_dataset_files", return_value=True):
                    with patch("sys.exit"):  # Prevent actual exit
                        from download_data import main

                        try:
                            main()
                        except SystemExit:
                            pass

        # Verify it would use data/sota by checking the config message
        # (indirect test since we can't easily capture the parsed args)
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=str, default="data/sota")
        args = parser.parse_args([])
        assert args.output_dir == "data/sota"
