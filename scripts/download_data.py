#!/usr/bin/env python3
"""
Download Horizyn SOTA Dataset

Downloads the SOTA dataset for reproducing the paper results.

Usage:
    python scripts/download_data.py --output-dir data/sota

Requirements:
    - ~2 GB free disk space for download
    - ~8 GB RAM during training (all data loaded into memory)

Dataset Contents:
    - train_pairs.csv: 257,733 training reaction-protein pairs
    - test_pairs.csv: 33,996 test pairs
    - train_rxns.csv: 10,785 training reactions from Rhea
    - test_rxns.csv: 1,012 test reactions from Rhea
    - prots_t5.h5: 216,132 ProtT5-XL embeddings (~900 MB)
        - 192,769 proteins in training pairs
        - 32,100 proteins in test pairs
        - 1,223 negative examples (not in any pairs)
    - prots.fasta: 216,132 protein sequences (~85 MB, optional reference)

Total uncompressed: ~1 GB

Zenodo:
    DOI: 10.5281/zenodo.17957034
    Record: https://zenodo.org/records/17957034

Note:
    All data will be loaded entirely into memory during training.
    Make sure you have sufficient RAM (~8 GB recommended).
"""

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Dict

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install requests tqdm")
    sys.exit(1)


# Zenodo record: https://zenodo.org/records/17957034
ZENODO_RECORD_ID = 17957034
ZENODO_API_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Dataset configuration: files hosted individually on Zenodo (no single archive).
# Each entry: Zenodo key -> (output filename, md5 from Zenodo).
# .gz files are downloaded and decompressed to the output filename.
DATASET_FILES = {
    "train_pairs.csv": ("train_pairs.csv", "d77c894783a2d3552b90b26eb253633b"),
    "test_pairs.csv": ("test_pairs.csv", "cdfab924e78d86b35adfcd7c01700974"),
    "train_rxns.csv": ("train_rxns.csv", "7b0335ac694e4afee87e7a0a970f56e4"),
    "test_rxns.csv": ("test_rxns.csv", "a45305ba22d4077d7a3f07d5f5d93ff5"),
    "prots_t5.h5.gz": ("prots_t5.h5", "eaf845701188e52e50abab1a239c0d34"),
    "prots.fasta.gz": ("prots.fasta", "2df315e48ff32764180e7e5feefe8b93"),
}

# MD5 checksums of the final decompressed output files (for post-download verification).
DECOMPRESSED_CHECKSUMS = {
    "prots_t5.h5": "282cf3f6e7a502d98ece793d366e75e9",
    "prots.fasta": "b0946eddf6d3b89047ac6b1b1ca374ae",
}

DATASET_CONFIG = {
    "name": "horizyn_sota",
    "version": "v1.0",
    "doi": "10.5281/zenodo.17957034",
    "size_gb": 1.0,  # ~1 GB uncompressed
    "files": [out_name for _, (out_name, _) in DATASET_FILES.items()],
    "file_checksums": {
        out_name: md5
        for _, (out_name, md5) in DATASET_FILES.items()
        if not out_name.endswith((".h5", ".fasta"))
    }
    | DECOMPRESSED_CHECKSUMS,
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


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file to out_path and remove the .gz file."""
    print(f"Decompressing: {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print(f"✓ Decompressed: {out_path.name}\n")


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


def verify_file_checksums(output_dir: Path, file_checksums: Dict[str, str]) -> bool:
    """
    Verify checksums of individual dataset files.

    Args:
        output_dir: Directory containing extracted files.
        file_checksums: Dict mapping filenames to expected MD5 checksums.

    Returns:
        True if all checksums match, False otherwise.
    """
    print("Verifying file checksums...")

    all_valid = True
    for filename, expected_hash in file_checksums.items():
        file_path = output_dir / filename

        if not file_path.exists():
            print(f"  ✗ {filename} (missing)")
            all_valid = False
            continue

        # Compute MD5 hash
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        actual_hash = hasher.hexdigest()

        if actual_hash == expected_hash:
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (checksum mismatch)")
            print(f"    Expected: {expected_hash}")
            print(f"    Got:      {actual_hash}")
            all_valid = False

    print()
    return all_valid


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
        default="data/sota",
        help="Output directory for dataset (default: data/sota)",
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

    try:
        for zenodo_key, (out_name, expected_md5) in DATASET_FILES.items():
            out_path = output_dir / out_name
            if not args.force and out_path.exists():
                print(f"Skipping (exists): {out_name}\n")
                continue

            url = f"{ZENODO_API_BASE}/files/{zenodo_key}/content"
            is_gz = zenodo_key.endswith(".gz")
            if is_gz:
                download_path = output_dir / zenodo_key
            else:
                download_path = out_path

            download_file(url, download_path, None)

            if not args.skip_checksum:
                if not verify_checksum(download_path, f"md5:{expected_md5}"):
                    print("Error: Checksum verification failed!")
                    print("The downloaded file may be corrupted.")
                    sys.exit(1)

            if is_gz:
                decompress_gz(download_path, out_path)

        # Verify all files present
        if not verify_dataset_files(output_dir, DATASET_CONFIG["files"]):
            print("Error: Some dataset files are missing!")
            sys.exit(1)

        # Verify individual file checksums
        if not verify_file_checksums(output_dir, DATASET_CONFIG["file_checksums"]):
            print("Error: File checksum verification failed!")
            print("One or more files may be corrupted.")
            sys.exit(1)

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
