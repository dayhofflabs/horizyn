#!/usr/bin/env python3
"""
Download Horizyn SOTA Dataset

Downloads the pre-split SwissProt dataset for reproducing the paper results.

Usage:
    python scripts/download_data.py --output-dir data/swissprot

Requirements:
    - ~2 GB free disk space for download
    - ~16 GB RAM during training (all data loaded into memory)

Dataset Contents:
    - train_pairs.db: 257,733 training reaction-protein pairs (13.8 MB)
    - val_pairs.db: 36,433 validation pairs (2.25 MB)
    - reactions.db: 15,969 reaction SMILES from Rhea v131 (5.38 MB)
    - proteins_t5_embeddings.h5: 216,132 ProtT5-XL embeddings from SwissProt v2023_05 (904 MB)

Total uncompressed: ~930 MB

Note:
    All data will be loaded entirely into memory during training.
    Make sure you have sufficient RAM (~16 GB recommended).
"""

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Dict

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install requests tqdm")
    sys.exit(1)


# Dataset configuration
DATASET_CONFIG = {
    "name": "horizyn_sota_swissprot",
    "version": "v1.0",
    "url": "https://zenodo.org/record/XXXXX/files/horizyn_sota_swissprot_v1.tar.gz",
    # TODO: Replace with actual Zenodo URL after dataset upload
    "size_gb": 1.0,  # ~930 MB uncompressed
    "checksum": "md5:XXXXX",  # TODO: Replace with actual checksum after packaging
    "files": [
        "train_pairs.db",
        "val_pairs.db",
        "reactions.db",
        "proteins_t5_embeddings.h5",
    ],
    # Individual file checksums (MD5 from DVC, for verification)
    "file_checksums": {
        "train_pairs.db": "0cf73b3ad6588fbef901a8fd40114709",
        "val_pairs.db": "3f47f3eeb1a8c20afb63c14c73891401",
        "reactions.db": "168cb64ef90972d43738258681e4a634",
        "proteins_t5_embeddings.h5": "282cf3f6e7a502d98ece793d366e75e9",
    },
}


def download_file(url: str, output_path: Path, expected_size: float | None = None) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from.
        output_path: Local path to save file.
        expected_size: Expected file size in GB (for display).
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")

    total_size = int(response.headers.get("content-length", 0))

    # Create progress bar
    progress_bar = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=output_path.name,
    )

    # Download with progress updates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            progress_bar.update(size)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Download incomplete")

    print(f"✓ Downloaded: {output_path.name}\n")


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify file checksum.

    Args:
        file_path: Path to file to verify.
        expected_checksum: Expected checksum in format "algorithm:hash".

    Returns:
        True if checksum matches, False otherwise.
    """
    # Check for placeholder before splitting
    if "XXXXX" in expected_checksum:
        print("Warning: Checksum verification skipped (placeholder value)")
        return True

    algorithm, expected_hash = expected_checksum.split(":", 1)

    print(f"Verifying {algorithm} checksum...")

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Compute hash
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    actual_hash = hasher.hexdigest()

    if actual_hash == expected_hash:
        print(f"✓ Checksum verified: {actual_hash}\n")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected_hash}")
        print(f"  Got:      {actual_hash}\n")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """
    Extract tar.gz archive.

    Args:
        archive_path: Path to archive file.
        output_dir: Directory to extract to.
    """
    print(f"Extracting: {archive_path.name}")

    with tarfile.open(archive_path, "r:gz") as tar:
        # Get list of members
        members = tar.getmembers()

        # Extract with progress bar
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=output_dir)

    print(f"✓ Extracted to: {output_dir}\n")


def verify_dataset_files(output_dir: Path, expected_files: list) -> bool:
    """
    Verify all expected dataset files are present.

    Args:
        output_dir: Directory containing extracted files.
        expected_files: List of expected filenames.

    Returns:
        True if all files present, False otherwise.
    """
    print("Verifying dataset files...")

    all_present = True
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (missing)")
            all_present = False

    print()
    return all_present


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(
        description="Download Horizyn SOTA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/swissprot",
        help="Output directory for dataset (default: data/swissprot)",
    )
    parser.add_argument(
        "--skip_checksum",
        action="store_true",
        help="Skip checksum verification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files exist",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print dataset info
    print("=" * 80)
    print("HORIZYN SOTA DATASET DOWNLOAD")
    print("=" * 80)
    print(f"Dataset: {DATASET_CONFIG['name']} ({DATASET_CONFIG['version']})")
    print(f"Size: ~{DATASET_CONFIG['size_gb']:.1f} GB")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80 + "\n")

    # Check if dataset already exists
    if not args.force:
        if all((output_dir / f).exists() for f in DATASET_CONFIG["files"]):
            print("Dataset files already exist.")
            print("Use --force to re-download.\n")
            if verify_dataset_files(output_dir, DATASET_CONFIG["files"]):
                print("✓ All dataset files present and ready for training!\n")
                print("To train the model, run:")
                print("    python train.py --config configs/sota.yaml")
                return
            else:
                print("Some files are missing. Re-downloading...\n")

    # Download archive
    archive_path = output_dir / f"{DATASET_CONFIG['name']}.tar.gz"

    if DATASET_CONFIG["url"].startswith("https://zenodo.org/record/XXXXX"):
        print("=" * 80)
        print("ERROR: Dataset URL not configured")
        print("=" * 80)
        print("\nThe dataset URL is a placeholder and needs to be updated.")
        print("This will be configured after the dataset is uploaded to Zenodo.")
        print("\nFor now, you can:")
        print("1. Create mock data for testing (see tests/)")
        print("2. Wait for the dataset to be published")
        print("\nExpected files in data/:")
        for filename in DATASET_CONFIG["files"]:
            print(f"  - {filename}")
        sys.exit(1)

    try:
        # Download
        download_file(
            DATASET_CONFIG["url"],
            archive_path,
            DATASET_CONFIG["size_gb"],
        )

        # Verify checksum
        if not args.skip_checksum:
            if not verify_checksum(archive_path, DATASET_CONFIG["checksum"]):
                print("Error: Checksum verification failed!")
                print("The downloaded file may be corrupted.")
                print("Try downloading again or use --skip_checksum to bypass.")
                sys.exit(1)

        # Extract
        extract_archive(archive_path, output_dir)

        # Verify all files present
        if not verify_dataset_files(output_dir, DATASET_CONFIG["files"]):
            print("Error: Some dataset files are missing after extraction!")
            sys.exit(1)

        # Clean up archive
        print(f"Cleaning up: {archive_path.name}")
        archive_path.unlink()
        print(f"✓ Removed archive\n")

        # Success
        print("=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print("✓ All dataset files ready for training!\n")
        print("Dataset size breakdown:")
        total_size_mb = 0
        for filename in DATASET_CONFIG["files"]:
            file_path = output_dir / filename
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  - {filename}: {size_mb:.1f} MB")
        print(f"\nTotal: {total_size_mb:.1f} MB (~{total_size_mb/1024:.1f} GB)")
        print("\nTo train the model, run:")
        print("    python train.py --config configs/sota.yaml")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
