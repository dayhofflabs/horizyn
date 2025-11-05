#!/usr/bin/env python3
"""
Prepare horizyn data files by splitting and renaming.

This script handles two scenarios:

1. **Nanodata** (integration tests): Splits a single pairs file by reaction_id
   into train/val splits and renames all files to the expected format.

2. **Swissprot** (full training): Renames pre-sliced files (cain/abel/eve) to
   the expected format without splitting.

Usage for nanodata:
    python data/provenance/split_pairs.py nanodata \\
        --data-dir data/nanodata \\
        --val-fraction 0.2 \\
        --seed 42

Usage for swissprot:
    python data/provenance/split_pairs.py swissprot \\
        --data-dir data/swissprot

Both commands will rename files in-place to match the expected format:
    - reactions.db (reactions table)
    - proteins_t5_embeddings.h5 (embeddings)
    - train_pairs.db (training pairs)
    - val_pairs.db (validation pairs)
"""

import argparse
import random
import shutil
import sqlite3
from pathlib import Path
from typing import List, Set


def get_unique_reactions(conn: sqlite3.Connection) -> List[str]:
    """Get all unique reaction IDs from the pairs table.

    Args:
        conn: SQLite database connection

    Returns:
        List of unique reaction IDs
    """
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT reaction_id FROM protein_to_reaction")
    return [row[0] for row in cursor.fetchall()]


def split_reactions(
    reaction_ids: List[str],
    val_fraction: float,
    seed: int,
) -> tuple[Set[str], Set[str]]:
    """Split reaction IDs into train and validation sets.

    Args:
        reaction_ids: List of all reaction IDs
        val_fraction: Fraction of reactions to use for validation (0-1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_reaction_ids, val_reaction_ids)
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    # Shuffle with seed
    random.seed(seed)
    shuffled_reactions = reaction_ids.copy()
    random.shuffle(shuffled_reactions)

    # Split
    n_val = int(len(shuffled_reactions) * val_fraction)
    if n_val == 0:
        n_val = 1  # Ensure at least 1 validation reaction
    val_reactions = set(shuffled_reactions[:n_val])
    train_reactions = set(shuffled_reactions[n_val:])

    return train_reactions, val_reactions


def write_split_pairs(
    input_conn: sqlite3.Connection,
    output_path: Path,
    reaction_ids: Set[str],
    split_name: str,
) -> int:
    """Write pairs for a specific set of reactions to a new database.

    Input schema (from DVC metadata):
        table_name="protein_to_reaction"
        columns: pr_id (INTEGER), reaction_id (TEXT), protein_id (TEXT), db_source (TEXT)

    Output schema (standardized):
        table_name="pairs"
        columns: pair_id (INTEGER), query_id (TEXT), target_id (TEXT)

    Args:
        input_conn: Connection to input database
        output_path: Path to output database
        reaction_ids: Set of reaction IDs to include
        split_name: Name of split (for logging)

    Returns:
        Number of pairs written
    """
    # Create output database
    output_conn = sqlite3.connect(output_path)
    output_cursor = output_conn.cursor()

    # Create table with standardized schema
    output_cursor.execute(
        """
        CREATE TABLE pairs (
            pair_id INTEGER PRIMARY KEY,
            query_id TEXT NOT NULL,
            target_id TEXT NOT NULL
        )
    """
    )

    # Create indices for efficient lookups
    output_cursor.execute("CREATE INDEX idx_query_id ON pairs(query_id)")
    output_cursor.execute("CREATE INDEX idx_target_id ON pairs(target_id)")

    # Read pairs from input and write matching ones to output
    input_cursor = input_conn.cursor()
    input_cursor.execute(
        """
        SELECT pr_id, reaction_id, protein_id
        FROM protein_to_reaction
    """
    )

    n_written = 0
    batch = []
    batch_size = 10000

    for row in input_cursor:
        pair_id, reaction_id, protein_id = row
        if reaction_id in reaction_ids:
            batch.append((pair_id, reaction_id, protein_id))
            n_written += 1

            if len(batch) >= batch_size:
                output_cursor.executemany(
                    "INSERT INTO pairs (pair_id, query_id, target_id) VALUES (?, ?, ?)", batch
                )
                batch = []

    # Write remaining batch
    if batch:
        output_cursor.executemany(
            "INSERT INTO pairs (pair_id, query_id, target_id) VALUES (?, ?, ?)", batch
        )

    output_conn.commit()
    output_conn.close()

    print(f"  Wrote {n_written:,} pairs to {output_path.name}")
    return n_written


def handle_nanodata(data_dir: Path, val_fraction: float, seed: int, dry_run: bool):
    """Handle nanodata: split pairs and rename files.

    Args:
        data_dir: Path to nanodata directory
        val_fraction: Fraction for validation split
        seed: Random seed
        dry_run: Whether to perform a dry run
    """
    print("=" * 80)
    print("Processing NANODATA (integration tests)")
    print("=" * 80)

    # Expected input files
    pairs_input = data_dir / "pairs.db"
    rxns_input = data_dir / "rxns.db"
    embeds_input = data_dir / "prot_embeds.h5"

    # Expected output files
    train_pairs_output = data_dir / "train_pairs.db"
    val_pairs_output = data_dir / "val_pairs.db"
    rxns_output = data_dir / "reactions.db"
    embeds_output = data_dir / "proteins_t5_embeddings.h5"

    # Validate inputs
    for path in [pairs_input, rxns_input, embeds_input]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"\nInput files:")
    print(f"  {pairs_input.name} ({pairs_input.stat().st_size / 1024:.1f} KB)")
    print(f"  {rxns_input.name} ({rxns_input.stat().st_size / 1024:.1f} KB)")
    print(f"  {embeds_input.name} ({embeds_input.stat().st_size / 1024:.1f} KB)")

    # Step 1: Split pairs
    print("\n" + "-" * 80)
    print("Step 1: Splitting pairs by reaction")
    print("-" * 80)

    conn = sqlite3.connect(pairs_input)
    reaction_ids = get_unique_reactions(conn)
    print(f"Found {len(reaction_ids):,} unique reactions in table 'protein_to_reaction'")

    train_reactions, val_reactions = split_reactions(reaction_ids, val_fraction, seed)
    print(f"Split: {len(train_reactions):,} train, {len(val_reactions):,} val")

    if not dry_run:
        n_train = write_split_pairs(conn, train_pairs_output, train_reactions, "train")
        n_val = write_split_pairs(conn, val_pairs_output, val_reactions, "val")
        print(f"\nTotal pairs: {n_train + n_val:,}")
        print(f"  Train: {n_train:,} ({n_train / (n_train + n_val):.1%})")
        print(f"  Val:   {n_val:,} ({n_val / (n_train + n_val):.1%})")
    else:
        print("\n[Would write split pairs to:]")
        print(f"  {train_pairs_output.name}")
        print(f"  {val_pairs_output.name}")

    conn.close()

    # Step 2: Rename other files
    print("\n" + "-" * 80)
    print("Step 2: Renaming files")
    print("-" * 80)

    renames = [
        (rxns_input, rxns_output, "reactions database"),
        (embeds_input, embeds_output, "protein embeddings"),
    ]

    for src, dst, description in renames:
        if not dry_run:
            if dst.exists():
                print(f"  Removing existing {dst.name}")
                dst.unlink()
            print(f"  Renaming {src.name} -> {dst.name} ({description})")
            shutil.move(src, dst)
        else:
            print(f"[Would rename {src.name} -> {dst.name}]")

    # Step 3: Clean up original pairs file
    if not dry_run:
        print(f"\n  Removing original {pairs_input.name}")
        pairs_input.unlink()
    else:
        print(f"\n[Would remove {pairs_input.name}]")

    print("\n" + "=" * 80)
    print("Nanodata processing complete!")
    print("=" * 80)

    if not dry_run:
        print("\nOutput files:")
        for path in [train_pairs_output, val_pairs_output, rxns_output, embeds_output]:
            size_kb = path.stat().st_size / 1024
            print(f"  {path.name} ({size_kb:.1f} KB)")


def handle_swissprot(data_dir: Path, dry_run: bool):
    """Handle swissprot: rename pre-sliced files.

    Args:
        data_dir: Path to swissprot directory
        dry_run: Whether to perform a dry run
    """
    print("=" * 80)
    print("Processing SWISSPROT (full training data)")
    print("=" * 80)

    # Expected input files (pre-sliced)
    train_pairs_input = data_dir / "cain_pairs_swissprot.db"
    val_pairs_input = data_dir / "abel_pairs_swissprot.db"
    rxns_input = data_dir / "eve_rxns.db"
    embeds_input = data_dir / "eve_swissprots_t5.h5"

    # Expected output files
    train_pairs_output = data_dir / "train_pairs.db"
    val_pairs_output = data_dir / "val_pairs.db"
    rxns_output = data_dir / "reactions.db"
    embeds_output = data_dir / "proteins_t5_embeddings.h5"

    # Validate inputs
    for path in [train_pairs_input, val_pairs_input, rxns_input, embeds_input]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"\nInput files:")
    print(f"  {train_pairs_input.name} ({train_pairs_input.stat().st_size / 1024**2:.1f} MB)")
    print(f"  {val_pairs_input.name} ({val_pairs_input.stat().st_size / 1024**2:.1f} MB)")
    print(f"  {rxns_input.name} ({rxns_input.stat().st_size / 1024**2:.1f} MB)")
    print(f"  {embeds_input.name} ({embeds_input.stat().st_size / 1024**2:.1f} MB)")

    print("\n" + "-" * 80)
    print("Renaming pre-sliced files")
    print("-" * 80)

    renames = [
        (train_pairs_input, train_pairs_output, "training pairs (cain)"),
        (val_pairs_input, val_pairs_output, "validation pairs (abel)"),
        (rxns_input, rxns_output, "reactions database (eve)"),
        (embeds_input, embeds_output, "protein embeddings (eve)"),
    ]

    for src, dst, description in renames:
        if not dry_run:
            if dst.exists():
                print(f"  Removing existing {dst.name}")
                dst.unlink()
            print(f"  Renaming {src.name} -> {dst.name} ({description})")
            shutil.move(src, dst)
        else:
            print(f"[Would rename {src.name} -> {dst.name}]")

    print("\n" + "=" * 80)
    print("Swissprot processing complete!")
    print("=" * 80)

    if not dry_run:
        print("\nOutput files:")
        for path in [train_pairs_output, val_pairs_output, rxns_output, embeds_output]:
            size_mb = path.stat().st_size / 1024**2
            print(f"  {path.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare horizyn data files by splitting and renaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Data preparation mode")

    # Nanodata subcommand
    nano_parser = subparsers.add_parser(
        "nanodata",
        help="Process nanodata (split pairs + rename files)",
    )
    nano_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/nanodata"),
        help="Path to nanodata directory (default: data/nanodata)",
    )
    nano_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of reactions for validation (default: 0.2)",
    )
    nano_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    nano_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it",
    )

    # Swissprot subcommand
    swiss_parser = subparsers.add_parser(
        "swissprot",
        help="Process swissprot (rename pre-sliced files)",
    )
    swiss_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/swissprot"),
        help="Path to swissprot directory (default: data/swissprot)",
    )
    swiss_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]\n")

    try:
        if args.mode == "nanodata":
            handle_nanodata(args.data_dir, args.val_fraction, args.seed, args.dry_run)
        elif args.mode == "swissprot":
            handle_swissprot(args.data_dir, args.dry_run)

        if not args.dry_run:
            print("\nYou can now run training with:")
            print(f"  python train.py --config configs/sota.yaml")
        else:
            print("\nRe-run without --dry-run to actually perform the operations.")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
