#!/usr/bin/env python3
"""
Horizyn Prediction Script

Query the Horizyn model with a reaction SMILES to find matching enzymes.

Usage:
    python scripts/predict.py "CC(=O)O>>CC(=O)OC" --top-k 10

    # Use a custom checkpoint
    python scripts/predict.py "CC(=O)O>>CC(=O)OC" --checkpoint checkpoints/my_model.ckpt

    # Output as JSON
    python scripts/predict.py "CC(=O)O>>CC(=O)OC" --output results.json

    # Bidirectional mode (scores both forward and reverse reaction)
    python scripts/predict.py "CC(=O)O>>CC(=O)OC" --bidirectional
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.config import load_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset
from horizyn.datasets.fingerprints import DRFPFingerprintDataset, RDKitPlusFingerprintDataset
from horizyn.datasets.hdf5 import EmbedDataset
from horizyn.datasets.transform import ConcatTensorTransform
from horizyn.lightning_module import HorizynLitModule


def build_reaction_fingerprint(
    reaction_smiles: str,
    config,
    bidirectional: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build concatenated RDKit+ + DRFP fingerprint for a reaction SMILES.

    Returns:
        Tuple of (fingerprint_tensor, labels) where fingerprint_tensor has shape
        (N, 2048) with N=1 or N=2 if bidirectional.
    """
    keys = ["query_f"]
    data = [{"reaction_smiles": reaction_smiles}]

    if bidirectional and ">>" in reaction_smiles:
        parts = reaction_smiles.split(">>")
        if len(parts) == 2:
            keys.append("query_r")
            data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})

    reactions = BaseDataset(keys=keys, array_data=data)

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

    merged = MergeDataset(
        datasets={"rdkit": rdkit_fp, "drfp": drfp_fp},
        add_prefix=False,
    )
    merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))

    fps = torch.stack([merged[k] for k in keys])
    return fps, keys


def predict(
    reaction_smiles: str,
    checkpoint_path: str = "checkpoints/horizyn-v1.ckpt",
    config_path: str = "configs/sota.yaml",
    device: str = "cuda",
    top_k: int = 10,
    batch_size: int = 512,
    bidirectional: bool = False,
) -> dict:
    """
    Query the model with a reaction SMILES and return top-K protein matches.
    """
    config = load_config(config_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = HorizynLitModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    print("Generating reaction fingerprints...")
    query_fps, query_labels = build_reaction_fingerprint(
        reaction_smiles, config, bidirectional=bidirectional
    )
    query_fps = query_fps.to(device)

    print("Loading protein embeddings...")
    protein_embeds = EmbedDataset(
        file_path=config.data.protein_embeds_path,
        in_memory=True,
    )
    num_targets = len(protein_embeds)
    print(f"  {num_targets} proteins in screening set")

    print("Encoding proteins...")
    target_dim = model.model.target_encoder.output_dim
    target_embeds = torch.zeros(num_targets, target_dim, device=device)

    with torch.no_grad():
        for start in tqdm(range(0, num_targets, batch_size), desc="Encoding"):
            end = min(start + batch_size, num_targets)
            batch_keys = protein_embeds.keys[start:end]
            vecs = torch.stack([protein_embeds[k] for k in batch_keys]).to(device)
            target_embeds[start:end] = model.model.target_encoder(vecs)

    print("Encoding reaction and ranking proteins...")
    with torch.no_grad():
        query_embeds = model.model.query_encoder(query_fps)

        # Average embeddings if bidirectional
        if query_embeds.shape[0] > 1:
            query_embed = query_embeds.mean(dim=0, keepdim=True)
        else:
            query_embed = query_embeds

        scores = torch.matmul(query_embed, target_embeds.T).squeeze(0)

    top_scores, top_indices = torch.topk(scores, k=min(top_k, num_targets))

    results = []
    for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({
            "protein_id": protein_embeds.keys[idx],
            "score": round(score, 6),
            "rank": len(results) + 1,
        })

    return {
        "reaction_smiles": reaction_smiles,
        "bidirectional": bidirectional,
        "num_proteins_screened": num_targets,
        "top_k": top_k,
        "results": results,
    }


def format_results(results: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("HORIZYN PREDICTION RESULTS")
    lines.append("=" * 70)
    lines.append(f"Reaction: {results['reaction_smiles']}")
    lines.append(f"Bidirectional: {results['bidirectional']}")
    lines.append(f"Proteins screened: {results['num_proteins_screened']}")
    lines.append("")
    lines.append(f"{'Rank':<6} {'Protein ID':<20} {'Score':<12}")
    lines.append("-" * 38)

    for hit in results["results"]:
        lines.append(f"{hit['rank']:<6} {hit['protein_id']:<20} {hit['score']:<12.6f}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Query Horizyn with a reaction SMILES to find matching enzymes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "reaction_smiles",
        type=str,
        help="Reaction SMILES string (e.g. 'reactants>>products')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/horizyn-v1.ckpt",
        help="Path to checkpoint (default: checkpoints/horizyn-v1.ckpt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sota.yaml",
        help="Path to config file (default: configs/sota.yaml)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top matches to return (default: 10)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Score both forward and reverse reaction directions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Download it with: python scripts/download_checkpoint.py")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    results = predict(
        reaction_smiles=args.reaction_smiles,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        top_k=args.top_k,
        bidirectional=args.bidirectional,
    )

    print("\n" + format_results(results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
