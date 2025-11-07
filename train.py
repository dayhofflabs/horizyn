#!/usr/bin/env python3
"""
Horizyn Model Training Script

Trains the Horizyn contrastive learning model for enzyme-reaction matching.

Usage:
    python train.py --config configs/sota.yaml

    # Override config values
    python train.py --config configs/sota.yaml --training.max_epochs 50

    # Set random seed
    python train.py --config configs/sota.yaml --seed 123

Requirements:
    - Data must be downloaded first (see scripts/download_data.py)
    - Requires ~16GB RAM (all data loaded into memory)
    - Requires single GPU with 16GB+ VRAM

Example:
    # Train SOTA model for 100 epochs
    python train.py --config configs/sota.yaml

    # Train with custom batch size
    python train.py --config configs/sota.yaml --data.train_batch_size 8192
"""

import argparse
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch

from horizyn.config import load_config, parse_overrides
from horizyn.data_module import HorizynDataModule
from horizyn.lightning_module import HorizynLitModule


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Horizyn contrastive learning model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/sota.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config.seed if provided)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Parse known args and capture remaining for overrides
    args, unknown = parser.parse_known_args()

    # Parse config overrides from remaining arguments
    overrides = parse_overrides(unknown)

    # Apply seed override if provided
    if args.seed is not None:
        overrides["seed"] = args.seed

    # Load configuration
    print(f"Loading config from: {args.config}")
    try:
        config = load_config(args.config, overrides=overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you're running from the project root directory.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Config validation failed")
        print(f"{e}")
        sys.exit(1)

    # Print configuration summary
    print("\n" + "=" * 80)
    print("HORIZYN TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Seed: {config.seed}")
    print(f"Max Epochs: {config.training.max_epochs}")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"Retrieval Batch Size: {config.data.retrieval_batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Weight Decay: {config.training.weight_decay}")
    print(f"Model: {config.model.name}")
    print(f"Query Encoder: {config.model.query_encoder_dims}")
    print(f"Target Encoder: {config.model.target_encoder_dims}")
    print(f"Embedding Dim: {config.model.embedding_dim}")
    print(f"Loss: {config.training.loss.name} (beta={config.training.loss.beta})")
    print(f"Log Dir: {config.logging.log_dir}")
    print(f"Checkpoint Dir: {config.logging.checkpoint_dir}")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"Set random seed to: {seed}\n")

    # Check for GPU availability
    if not torch.cuda.is_available():
        print("Warning: No GPU detected. Training will be very slow on CPU.")
        print("Consider using a machine with a CUDA-capable GPU.\n")

    # Setup data module
    print("Initializing data module...")
    try:
        data_module = HorizynDataModule(
            train_pairs_path=config.data.train_pairs_path,
            val_pairs_path=config.data.val_pairs_path,
            reactions_path=config.data.reactions_path,
            proteins_path=config.data.proteins_path,
            train_batch_size=config.data.train_batch_size,
            retrieval_batch_size=config.data.retrieval_batch_size,
            rdkit_fp_dim=config.data.get("rdkit_fp_dim", 1024),
            drfp_dim=config.data.get("drfp_dim", 1024),
            num_workers=config.data.get("num_workers", 0),
            pin_memory=config.data.get("pin_memory", False),
            standardize=config.data.get("standardize_reactions", True),
        )
    except FileNotFoundError as e:
        print(f"\nError: Data file not found")
        print(f"{e}")
        print("\nPlease download the dataset first:")
        print("    python scripts/download_data.py --output_dir data/")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing data module: {e}")
        sys.exit(1)

    print("Data module initialized.\n")

    # Setup model
    print("Initializing model...")
    model = HorizynLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.get("learn_beta", False),
        metric_ks=config.training.metrics.get("top_k", [1, 10, 100, 1000]),
        pos_score=config.training.metrics.get("pos_score", False),
        neg_score=config.training.metrics.get("neg_score", False),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup logging
    logger = pl.loggers.CSVLogger(
        save_dir=config.logging.log_dir,
        name="horizyn_training",
    )

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.logging.checkpoint_dir,
        filename="horizyn-{epoch:02d}",
        every_n_epochs=config.logging.get("save_every_n_epochs", 10),
        save_last=True,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    # Setup trainer
    print("Setting up Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.logging.get("log_every_n_steps", 1),
        check_val_every_n_epoch=config.training.get("check_val_every_n_epoch", 10),
        enable_progress_bar=config.training.get("enable_progress_bar", True),
        deterministic=True,  # For reproducibility
        # Single GPU training (DDP not supported in simplified version)
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="auto",
    )

    print(f"Trainer configured for {config.training.max_epochs} epochs\n")

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    try:
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=args.resume,  # Resume from checkpoint if provided
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print(f"Logs saved to: {config.logging.log_dir}")
    print("\nTo resume training, use:")
    print(
        f"    python train.py --config {args.config} --resume {checkpoint_callback.last_model_path}"
    )


if __name__ == "__main__":
    main()
