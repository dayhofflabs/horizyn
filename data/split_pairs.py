#!/usr/bin/env python3
"""
Split reaction-protein pairs into train/val splits and rename files.

This script performs two operations:
1. Splits pairs by reaction_id (preventing data leakage where the same reaction
   appears in both train and val)
2. Renames files to match the expected format from download_data.py

Usage:
    python data/split_pairs.py \\
        --input-pairs data/cain_pairs_swissprot.db \\
        --input-reactions data/eve_rxns.db \\
        --input-proteins data/cain_swissprots_t5.h5 \\
        --output-dir data/ \\
        --val-fraction 0.1 \\
        --seed 42

The script will create:
    - train_pairs.db (pairs table)
    - val_pairs.db (pairs table)
    - reactions.db (renamed from input-reactions)
    - proteins_t5_embeddings.h5 (renamed from input-proteins)
"""

import argparse
import random
import shutil
import sqlite3
from pathlib import Path
from typing import List, Set, Tuple


def get_unique_reactions(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get all unique reaction IDs from the pairs table.

    Args:
        conn: SQLite database connection
        table_name: Name of the table containing pairs

    Returns:
        List of unique reaction IDs
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT reaction_id FROM {table_name}")
    return [row[0] for row in cursor.fetchall()]


def split_reactions(
    reaction_ids: List[str],
    val_fraction: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
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

    # Create table with same schema as input
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
            # Map to expected schema: pair_id, query_id (reaction), target_id (protein)
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


def main():
    parser = argparse.ArgumentParser(
        description="Split pairs by reaction and rename files for horizyn training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-pairs",
        type=Path,
        required=True,
        help="Path to input pairs database (e.g., cain_pairs_swissprot.db)",
    )
    parser.add_argument(
        "--input-reactions",
        type=Path,
        required=True,
        help="Path to input reactions database (e.g., eve_rxns.db)",
    )
    parser.add_argument(
        "--input-proteins",
        type=Path,
        required=True,
        help="Path to input proteins HDF5 file (e.g., cain_swissprots_t5.h5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write output files (default: data/)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of reactions to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_pairs.exists():
        raise FileNotFoundError(f"Input pairs file not found: {args.input_pairs}")
    if not args.input_reactions.exists():
        raise FileNotFoundError(f"Input reactions file not found: {args.input_reactions}")
    if not args.input_proteins.exists():
        raise FileNotFoundError(f"Input proteins file not found: {args.input_proteins}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    train_pairs_path = args.output_dir / "train_pairs.db"
    val_pairs_path = args.output_dir / "val_pairs.db"
    reactions_path = args.output_dir / "reactions.db"
    proteins_path = args.output_dir / "proteins_t5_embeddings.h5"

    print("=" * 80)
    print("Horizyn Data Splitting and Renaming")
    print("=" * 80)
    print(f"\nInput files:")
    print(f"  Pairs:     {args.input_pairs}")
    print(f"  Reactions: {args.input_reactions}")
    print(f"  Proteins:  {args.input_proteins}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Validation fraction: {args.val_fraction:.1%}")
    print(f"Random seed: {args.seed}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")

    # Step 1: Load and split reactions
    print("\n" + "-" * 80)
    print("Step 1: Loading pairs and splitting by reaction")
    print("-" * 80)

    conn = sqlite3.connect(args.input_pairs)
    reaction_ids = get_unique_reactions(conn, "protein_to_reaction")
    print(f"Found {len(reaction_ids):,} unique reactions")

    train_reactions, val_reactions = split_reactions(
        reaction_ids,
        args.val_fraction,
        args.seed,
    )
    print(f"Split: {len(train_reactions):,} train, {len(val_reactions):,} val")

    # Step 2: Write split pairs
    if not args.dry_run:
        print("\n" + "-" * 80)
        print("Step 2: Writing split pairs databases")
        print("-" * 80)

        n_train = write_split_pairs(conn, train_pairs_path, train_reactions, "train")
        n_val = write_split_pairs(conn, val_pairs_path, val_reactions, "val")

        print(f"\nTotal pairs: {n_train + n_val:,}")
        print(f"  Train: {n_train:,} ({n_train / (n_train + n_val):.1%})")
        print(f"  Val:   {n_val:,} ({n_val / (n_train + n_val):.1%})")
    else:
        print("\n[Would write split pairs to:]")
        print(f"  {train_pairs_path}")
        print(f"  {val_pairs_path}")

    conn.close()

    # Step 3: Rename/copy other files
    print("\n" + "-" * 80)
    print("Step 3: Renaming/copying other files")
    print("-" * 80)

    if not args.dry_run:
        # Copy reactions database
        if reactions_path.exists():
            print(f"  Removing existing {reactions_path.name}")
            reactions_path.unlink()
        print(f"  Copying {args.input_reactions.name} -> {reactions_path.name}")
        shutil.copy2(args.input_reactions, reactions_path)

        # Copy proteins HDF5
        if proteins_path.exists():
            print(f"  Removing existing {proteins_path.name}")
            proteins_path.unlink()
        print(f"  Copying {args.input_proteins.name} -> {proteins_path.name}")
        shutil.copy2(args.input_proteins, proteins_path)
    else:
        print(f"[Would copy {args.input_reactions.name} -> {reactions_path.name}]")
        print(f"[Would copy {args.input_proteins.name} -> {proteins_path.name}]")

    # Summary
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)

    if not args.dry_run:
        print("\nOutput files:")
        print(f"  {train_pairs_path} ({train_pairs_path.stat().st_size / 1024**2:.1f} MB)")
        print(f"  {val_pairs_path} ({val_pairs_path.stat().st_size / 1024**2:.1f} MB)")
        print(f"  {reactions_path} ({reactions_path.stat().st_size / 1024**2:.1f} MB)")
        print(f"  {proteins_path} ({proteins_path.stat().st_size / 1024**2:.1f} MB)")
        print("\nYou can now run training with:")
        print(f"  python train.py --config configs/sota.yaml")
    else:
        print("\nRe-run without --dry-run to actually perform the split.")


if __name__ == "__main__":
    main()
