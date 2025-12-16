#!/usr/bin/env python3
"""
Horizyn Model Evaluation Script

Evaluates a trained Horizyn checkpoint and computes metrics matching the paper table.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_20251214.ckpt

    # Use custom config
    python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --config configs/sota.yaml

    # Output as JSON
    python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --output results.json

Metrics computed:
    - Top-1, Top-10, Top-100, Top-1000 Hit Rates
    - R-precision
    - Average Precision (Avg. precision)

Example:
    # Evaluate the SOTA model
    python scripts/evaluate.py --checkpoint checkpoints/best_20251214/best_20251214.ckpt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.lightning_module import HorizynLitModule
from horizyn.metrics import average_precision, r_precision, top_k_hit_rate


def compute_cosine_distances(
    query_embeds: torch.Tensor, target_embeds: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine distances between query and target embeddings.

    Args:
        query_embeds: Query embeddings of shape (num_queries, embedding_dim).
        target_embeds: Target embeddings of shape (num_targets, embedding_dim).

    Returns:
        Distance matrix of shape (num_queries, num_targets).
    """
    # Cosine distance = 1 - cosine_similarity
    # Both inputs should already be L2-normalized
    return 1.0 - torch.matmul(query_embeds, target_embeds.T)


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "configs/sota.yaml",
    device: str = "cuda",
    batch_size: int = 128,
) -> dict:
    """
    Evaluate a checkpoint and compute all paper metrics.

    Args:
        checkpoint_path: Path to the checkpoint file.
        config_path: Path to the config file.
        device: Device to use for evaluation.
        batch_size: Batch size for encoding.

    Returns:
        Dictionary containing all computed metrics.
    """
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Load model from checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = HorizynLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)

    # Setup validation data directly (faster than full data module setup)
    print("Setting up validation data...")

    # Import required dataset classes
    from horizyn.datasets.base import BaseDataset
    from horizyn.datasets.collection import MergeDataset, TupleDataset
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints import (
        DRFPFingerprintDataset,
        RDKitPlusFingerprintDataset,
    )
    from horizyn.datasets.hdf5 import EmbedDataset
    from horizyn.datasets.transform import ConcatTensorTransform

    # Load validation pairs
    val_pairs = CSVDataset(
        file_path=config.data.test_pairs_path,
        key_column="pr_id",
        columns=["reaction_id", "protein_id"],
        rename_map={"reaction_id": "query_id", "protein_id": "target_id"},
    )
    print(f"  Loaded {len(val_pairs)} validation pairs")

    # Augment pairs with bidirectional reactions
    augmented_keys = []
    augmented_data = []
    for pair_key in val_pairs.keys:
        pair_data = val_pairs[pair_key]
        query_id = pair_data["query_id"]
        target_id = pair_data["target_id"]
        # Forward
        augmented_keys.append(f"{pair_key}_f")
        augmented_data.append({"query_id": f"{query_id}_f", "target_id": target_id})
        # Backward
        augmented_keys.append(f"{pair_key}_r")
        augmented_data.append({"query_id": f"{query_id}_r", "target_id": target_id})

    val_pairs = BaseDataset(keys=augmented_keys, array_data=augmented_data)
    print(f"  Augmented to {len(val_pairs)} bidirectional pairs")

    # Load test reactions and create fingerprints
    reactions = CSVDataset(
        file_path=config.data.test_reactions_path,
        key_column="reaction_id",
        columns=["reaction_smiles"],
    )

    # Augment reactions bidirectionally
    rxn_augmented_keys = []
    rxn_augmented_data = []
    for rxn_id in reactions.keys:
        rxn_data = reactions[rxn_id]
        smiles = rxn_data["reaction_smiles"]
        # Forward
        rxn_augmented_keys.append(f"{rxn_id}_f")
        rxn_augmented_data.append({"reaction_smiles": smiles})
        # Backward
        if ">>" in smiles:
            parts = smiles.split(">>")
            if len(parts) == 2:
                reversed_smiles = f"{parts[1]}>>{parts[0]}"
                rxn_augmented_keys.append(f"{rxn_id}_r")
                rxn_augmented_data.append({"reaction_smiles": reversed_smiles})

    reactions = BaseDataset(keys=rxn_augmented_keys, array_data=rxn_augmented_data)
    print(f"  Loaded {len(reactions)} bidirectional test reactions")

    # Generate fingerprints
    print("  Generating RDKit+ fingerprints...")
    rdkit_fp = RDKitPlusFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("rdkit_fp_dim", 1024),
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
    )

    print("  Generating DRFP fingerprints...")
    drfp_fp = DRFPFingerprintDataset(
        reaction_dataset=reactions,
        vec_dim=config.data.get("drfp_dim", 1024),
        radius=3,
        rings=True,
        standardize=config.data.get("standardize_reactions", True),
        standardize_hypervalent=config.data.get("standardize_hypervalent", True),
        standardize_remove_hs=config.data.get("standardize_remove_hs", True),
        standardize_kekulize=config.data.get("standardize_kekulize", False),
        standardize_uncharge=config.data.get("standardize_uncharge", True),
        standardize_metals=config.data.get("standardize_metals", True),
    )

    # Merge and concatenate fingerprints
    merged_fp = MergeDataset(
        datasets={"rdkit": rdkit_fp, "drfp": drfp_fp},
        add_prefix=False,
    )
    merged_fp.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))

    # Load protein embeddings (full screening set)
    print("  Loading protein embeddings...")
    protein_embeds = EmbedDataset(
        file_path=config.data.protein_embeds_path,
        in_memory=True,
    )

    # Use protein embeddings as screening set
    num_targets = len(protein_embeds)
    print(f"Screening set size: {num_targets} proteins")

    # Create target ID to index mapping
    target_id_to_idx = {target_id: idx for idx, target_id in enumerate(protein_embeds.keys)}

    # Encode all targets
    print("Encoding all target proteins...")
    target_embeds_tensor = torch.zeros(
        num_targets, model.model.target_encoder.output_dim, device=device
    )

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_targets, batch_size), desc="Encoding targets"):
            batch_end = min(batch_start + batch_size, num_targets)
            batch_keys = protein_embeds.keys[batch_start:batch_end]

            # Get target vectors
            target_vecs = torch.stack([protein_embeds[k] for k in batch_keys]).to(device)

            # Encode
            batch_embeds = model.model.target_encoder(target_vecs)
            target_embeds_tensor[batch_start:batch_end] = batch_embeds

    # Group pairs by query_id for multi-label retrieval
    query_to_targets = defaultdict(list)
    for pair_key in val_pairs.keys:
        pair = val_pairs[pair_key]
        query_id = pair["query_id"]
        target_id = pair["target_id"]
        query_to_targets[query_id].append(target_id)

    # Get unique query IDs
    unique_query_ids = sorted(query_to_targets.keys())
    num_queries = len(unique_query_ids)
    print(f"Number of unique validation queries: {num_queries}")

    # Compute metrics for each query
    print("Computing metrics...")

    metric_results = defaultdict(list)

    with torch.no_grad():
        for query_idx in tqdm(range(num_queries), desc="Evaluating queries"):
            query_id = unique_query_ids[query_idx]

            # Get query fingerprint (ConcatTensorTransform returns a tensor)
            query_fp_tensor: torch.Tensor = merged_fp[query_id]
            query_fp = query_fp_tensor.unsqueeze(0).to(device)

            # Encode query
            query_embed = model.model.query_encoder(query_fp)

            # Compute distances to all targets
            dists = compute_cosine_distances(query_embed, target_embeds_tensor)

            # Convert to scores (higher is better)
            scores = -dists.squeeze(0)

            # Get valid target IDs for this query
            valid_target_ids = query_to_targets[query_id]

            # Convert target IDs to indices
            target_indices = []
            for target_id in valid_target_ids:
                if target_id in target_id_to_idx:
                    target_indices.append(target_id_to_idx[target_id])

            if len(target_indices) == 0:
                # Skip queries with no valid targets in screening set
                continue

            target_idx = torch.tensor(target_indices, dtype=torch.long, device=device)

            # Compute metrics
            metric_results["top_1"].append(top_k_hit_rate(scores, target_idx, k=1).item())
            metric_results["top_10"].append(top_k_hit_rate(scores, target_idx, k=10).item())
            metric_results["top_100"].append(top_k_hit_rate(scores, target_idx, k=100).item())
            metric_results["top_1000"].append(top_k_hit_rate(scores, target_idx, k=1000).item())
            metric_results["r_precision"].append(r_precision(scores, target_idx).item())
            metric_results["avg_precision"].append(average_precision(scores, target_idx).item())

    # Compute mean metrics
    results = {}
    for metric_name, values in metric_results.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    # Add metadata
    results["num_queries"] = len(metric_results["top_1"])
    results["num_targets"] = num_targets
    results["checkpoint"] = checkpoint_path
    results["config"] = config_path

    return results


def format_results_table(results: dict) -> str:
    """
    Format results as a table matching the paper format.

    Args:
        results: Dictionary of computed metrics.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("HORIZYN EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Checkpoint: {results.get('checkpoint', 'N/A')}")
    lines.append(f"Config: {results.get('config', 'N/A')}")
    lines.append(f"Queries evaluated: {results.get('num_queries', 'N/A')}")
    lines.append(f"Screening set size: {results.get('num_targets', 'N/A')}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RETRIEVAL METRICS (Paper Table Format)")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Metric':<20} {'Value':<15} {'Value (%)':<15}")
    lines.append("-" * 50)

    # Main paper metrics
    metrics_order = [
        ("Top-1 HR", "top_1"),
        ("Top-10 HR", "top_10"),
        ("Top-100 HR", "top_100"),
        ("Top-1000 HR", "top_1000"),
        ("R-precision", "r_precision"),
        ("Avg. precision", "avg_precision"),
    ]

    for display_name, key in metrics_order:
        value = results.get(key, 0.0)
        lines.append(f"{display_name:<20} {value:.4f}         {value * 100:.1f}%")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Horizyn checkpoint and compute paper metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sota.yaml",
        help="Path to config file (default: configs/sota.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 128)",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    print(f"\nUsing device: {args.device}")

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        batch_size=args.batch_size,
    )

    # Print results
    print("\n" + format_results_table(results))

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
