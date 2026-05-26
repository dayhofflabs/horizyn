#!/usr/bin/env python3
"""
Download Horizyn Pre-trained Checkpoints

Downloads the official Horizyn v1 checkpoints from Zenodo:
  - horizyn_v1_0_dev.ckpt  (paper-faithful, for evaluation)
  - horizyn_v1_0_inf.ckpt  (full training data, for prediction)

Usage:
    python scripts/download_checkpoint.py
    python scripts/download_checkpoint.py --only dev
    python scripts/download_checkpoint.py --only inf
    python scripts/download_checkpoint.py --output-dir checkpoints

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

CHECKPOINTS = {
    "dev": {
        "zenodo_key": "horizyn_v1_0_dev.ckpt",
        "local_name": "horizyn_v1_0_dev.ckpt",
        "md5": "5b1f938f8b0a82fbe91892a3b4e2bf2c",
        "size_mb": 201,
        "description": "Development (paper-faithful, train-split only)",
    },
    "inf": {
        "zenodo_key": "horizyn_v1_0_inf.ckpt",
        "local_name": "horizyn_v1_0_inf.ckpt",
        "md5": "cf6775b775287462099ae0681485a6bc",
        "size_mb": 201,
        "description": "Inference (full training data, for prediction)",
    },
}


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


def download_one(ckpt_info: dict, output_dir: Path, force: bool) -> bool:
    """Download and verify a single checkpoint. Returns True on success."""
    output_path = output_dir / ckpt_info["local_name"]

    print(f"  {ckpt_info['local_name']}  ({ckpt_info['description']})")
    print(f"  Size: ~{ckpt_info['size_mb']} MB")
    print(f"  Output: {output_path}\n")

    if output_path.exists() and not force:
        print(f"Already exists: {output_path}")
        print("Use --force to re-download.\n")
        if verify_checksum(output_path, ckpt_info["md5"]):
            return True
        print("Existing file is corrupted. Re-downloading...\n")

    url = f"{ZENODO_API_BASE}/files/{ckpt_info['zenodo_key']}/content"
    download_file(url, output_path)

    if not verify_checksum(output_path, ckpt_info["md5"]):
        print("Error: Checksum verification failed!")
        print("The downloaded file may be corrupted.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Horizyn pre-trained checkpoints from Zenodo",
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
        "--only",
        choices=["dev", "inf"],
        default=None,
        help="Download only one checkpoint (default: both)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    targets = [args.only] if args.only else ["dev", "inf"]

    print("=" * 70)
    print("HORIZYN CHECKPOINT DOWNLOAD")
    print("=" * 70)
    print(f"Record: https://zenodo.org/records/{ZENODO_RECORD_ID}")
    print(f"Output directory: {output_dir}/")
    print("=" * 70 + "\n")

    failed = []
    for key in targets:
        info = CHECKPOINTS[key]
        print("-" * 70)
        if not download_one(info, output_dir, args.force):
            failed.append(key)

    print("=" * 70)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)

    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    print("To evaluate (uses dev checkpoint, matches paper metrics):")
    print("    python scripts/evaluate.py")
    print()
    print("To predict enzymes for a reaction (uses inf checkpoint):")
    print('    python scripts/predict.py "REACTANTS>>PRODUCTS" --top-k 10')


if __name__ == "__main__":
    main()
