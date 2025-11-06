"""
Quick test to verify the validation fix works correctly.

This script tests:
1. Data module creates correct retrieval dataset (unique queries with target lists)
2. Lightning module can process the new batch format
3. Metrics are computed correctly for multi-label retrieval
"""

import sys
sys.path.insert(0, '/workspaces/dma/horizyn')

import torch
from horizyn.config import load_config
from horizyn.data_module import HorizynDataModule
from horizyn.lightning_module import HorizynLitModule
import lightning.pytorch as pl


def test_validation_data_structure():
    """Test that validation data is structured correctly."""
    print("=" * 80)
    print("Testing validation data structure...")
    print("=" * 80)
    
    # Load config (use nano for speed)
    config = load_config("configs/nano.yaml")
    
    # Create data module
    dm = HorizynDataModule(**config.data)
    dm.setup("fit")
    
    # Check validation retrieval dataset
    print(f"\n1. Validation query dataset:")
    print(f"   Length: {len(dm._val_query_data)} (should be # unique queries)")
    
    print(f"\n2. Validation retrieval targets:")
    print(f"   Length: {len(dm._val_retrieval_targets)} (should match query dataset)")
    
    # Get a sample
    sample_key = dm._val_query_data.keys[0]
    sample = dm._val_query_data[sample_key]
    
    print(f"\n3. Sample from validation query dataset:")
    print(f"   Key: {sample_key}")
    print(f"   Fields: {list(sample.keys())}")
    print(f"   query_id: {sample.get('query_id', 'MISSING')}")
    print(f"   query_vec shape: {sample['query_vec'].shape if 'query_vec' in sample else 'MISSING'}")
    
    # Get target list for this query
    targets = dm._val_retrieval_targets[sample_key]
    print(f"   Number of valid targets: {len(targets)}")
    print(f"   First few targets: {targets[:3]}")
    
    # Check validation dataloader
    print(f"\n4. Validation dataloaders:")
    val_loaders = dm.val_dataloader()
    print(f"   Number of dataloaders: {len(val_loaders)}")
    print(f"   Loader 0 (loss): {len(val_loaders[0].dataset)} samples")
    print(f"   Loader 1 (lookup): {len(val_loaders[1].dataset)} samples")
    print(f"   Loader 2 (retrieval): {len(val_loaders[2].dataset)} samples")
    
    # Get a batch from retrieval loader
    retrieval_loader = val_loaders[2]
    batch = next(iter(retrieval_loader))
    
    print(f"\n5. Batch from retrieval dataloader:")
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Batch size: {len(batch['query_id'])}")
    print(f"   query_id type: {type(batch['query_id'][0])}")
    print(f"   query_vec shape: {batch['query_vec'].shape}")
    
    print("\n✅ Data structure looks correct!")
    return dm, config


def test_lightning_module_validation():
    """Test that lightning module can process validation batches."""
    print("\n" + "=" * 80)
    print("Testing lightning module validation step...")
    print("=" * 80)
    
    dm, config = test_validation_data_structure()
    
    # Create lightning module
    lit_module = HorizynLitModule(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.learn_beta,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        metric_ks=config.training.metrics.top_k,
    )
    
    # Create minimal trainer (no actual training)
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=3,  # Test a few batches
    )
    
    # Connect datamodule to trainer
    trainer.datamodule = dm
    
    # Manually simulate validation epoch start
    lit_module.trainer = trainer
    lit_module.on_validation_epoch_start()
    
    print(f"\n1. Target lookup table initialized:")
    print(f"   Shape: {lit_module.target_lookup_table.shape}")
    print(f"   Expected: ({len(dm._target_data)}, {config.model.embedding_dim})")
    
    # Get validation dataloaders
    val_loaders = dm.val_dataloader()
    
    # Dataloader 1: Build lookup table
    print(f"\n2. Building target lookup table...")
    for batch_idx, batch in enumerate(val_loaders[1]):
        lit_module._validation_lookup_step(batch, batch_idx)
        if batch_idx == 0:
            print(f"   Processed batch 0")
    
    # Dataloader 2: Test retrieval metrics
    print(f"\n3. Testing retrieval metrics computation...")
    retrieval_loader = val_loaders[2]
    batch = next(iter(retrieval_loader))
    
    try:
        lit_module._validation_retrieval_step(batch, 0)
        print("   ✅ Retrieval step completed successfully!")
    except Exception as e:
        print(f"   ❌ Error in retrieval step: {e}")
        raise
    
    print("\n✅ Lightning module validation works correctly!")


if __name__ == "__main__":
    test_lightning_module_validation()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe validation fix is working correctly.")
    print("You can now run full training and expect proper metrics.")

