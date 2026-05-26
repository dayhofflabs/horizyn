#!/usr/bin/env python3
"""
Download Horizyn Pre-trained Checkpoint

Downloads the official pre-trained Horizyn v1 checkpoint from Zenodo.

Usage:
    python scripts/download_checkpoint.py
    python scripts/download_checkpoint.py --output-dir checkpoints

The checkpoint can be used directly for evaluation:
    python scripts/evaluate.py --checkpoint checkpoints/horizyn-v1.ckpt

Zenodo:
    DOI: 10.5281/zenodo.20348783
    Record: https://zenodo.org/records/20348783
"""

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install requests tqdm")
    sys.exit(1)

ZENODO_RECORD_ID = 20348783
ZENODO_API_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

CHECKPOINT_ZENODO_KEY = "horizyn_v1_0_dev.ckpt"
CHECKPOINT_LOCAL_NAME = "horizyn-v1.ckpt"
CHECKPOINT_MD5 = "5b1f938f8b0a82fbe91892a3b4e2bf2c"
CHECKPOINT_SIZE_MB = 201


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")

    total_size = int(response.headers.get("content-length", 0))

    progress_bar = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=output_path.name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            progress_bar.update(size)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Download incomplete")

    print(f"✓ Downloaded: {output_path.name}\n")


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    print("Verifying md5 checksum...")

    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    actual = hasher.hexdigest()

    if actual == expected_md5:
        print(f"✓ Checksum verified: {actual}\n")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected_md5}")
        print(f"  Got:      {actual}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Horizyn pre-trained checkpoint from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory (default: checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / CHECKPOINT_LOCAL_NAME

    print("=" * 70)
    print("HORIZYN CHECKPOINT DOWNLOAD")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_LOCAL_NAME}")
    print(f"Size: ~{CHECKPOINT_SIZE_MB} MB")
    print(f"Output: {output_path}")
    print("=" * 70 + "\n")

    if output_path.exists() and not args.force:
        print(f"Checkpoint already exists: {output_path}")
        print("Use --force to re-download.\n")
        if verify_checksum(output_path, CHECKPOINT_MD5):
            print("✓ Checkpoint ready for evaluation!\n")
            print("To evaluate, run:")
            print(f"    python scripts/evaluate.py --checkpoint {output_path}")
            return
        else:
            print("Existing file is corrupted. Re-downloading...\n")

    url = f"{ZENODO_API_BASE}/files/{CHECKPOINT_ZENODO_KEY}/content"
    download_file(url, output_path)

    if not verify_checksum(output_path, CHECKPOINT_MD5):
        print("Error: Checksum verification failed!")
        print("The downloaded file may be corrupted.")
        sys.exit(1)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"✓ Checkpoint ready: {output_path} ({size_mb:.1f} MB)\n")
    print("To evaluate, run:")
    print(f"    python scripts/evaluate.py --checkpoint {output_path}")


if __name__ == "__main__":
    main()
