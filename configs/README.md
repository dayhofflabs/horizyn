# Horizyn Configuration Files

This directory contains YAML configuration files for training the Horizyn model.

## Available Configurations

### `sota.yaml` - Production SOTA Model

The state-of-the-art configuration used in the Horizyn publication.

**Dataset**: SOTA (~1 GB)
- ~11,000 reactions from Rhea
- ~200,000 proteins with ProtT5 embeddings
- ~257,000 training pairs, ~34,000 test pairs

**Model Architecture**:
- **Query Encoder**: RDKit+ (2048-bit) + DRFP (2048-bit) → MLP [4096, 512]
- **Target Encoder**: T5 embeddings (1024-dim) → MLP [4096, 512]
- **Loss**: MLNCE with β=10.0 (fixed)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)

**Training Settings**:
- Batch size: 16,384 (large batch for contrastive learning)
- Epochs: 100
- Validation: Every epoch with retrieval metrics (Top-1, Top-10, Top-100, Top-1000)

**Hardware Requirements**:
- RAM: 16GB minimum
- GPU: 16GB+ VRAM (T4, A10G, V100, or better)
- Training time: ~8-12 hours on T4 GPU

**Usage**:
```bash
python train.py --config configs/sota.yaml
```

### `nano.yaml` - Integration Test Configuration

A minimal configuration for fast integration testing.

**Dataset**: Nanodata (~80 KB, included in repository)
- 12 reactions from Rhea
- 11 proteins with T5 embeddings
- ~10 training pairs, ~2 test pairs

**Model Architecture**: Same as SOTA (scaled down data only)

**Training Settings**:
- Batch size: 16 (small for tiny dataset)
- Epochs: 10
- Validation: Every epoch

**Purpose**: 
- Verify the training pipeline works end-to-end
- Run integration tests quickly (< 1 minute on CPU)
- Debug model architecture and data loading
- Not intended for scientific evaluation

**Usage**:
```bash
python train.py --config configs/nano.yaml
```

## Configuration Structure

All configuration files follow this structure:

```yaml
# Data configuration
data:
  train_pairs_path: "path/to/train_pairs.csv"       # Training pairs CSV
  test_pairs_path: "path/to/test_pairs.csv"         # Test pairs CSV
  train_reactions_path: "path/to/train_rxns.csv"    # Training reactions CSV
  test_reactions_path: "path/to/test_rxns.csv"      # Test reactions CSV
  protein_embeds_path: "path/to/prots_t5.h5"        # HDF5 with T5 embeddings
  
  # Reaction fingerprint dimensions
  rdkit_fp_dim: 1024                                 # RDKit+ fingerprint dimension
  drfp_dim: 1024                                     # DRFP fingerprint dimension

# Model architecture
model:
  architecture: "DualContrastiveModel"
  query_encoder:
    architecture: "MLP"
    layer_widths: [4096, 512]
    normalize: true
  target_encoder:
    architecture: "MLP"
    layer_widths: [4096, 512]
    normalize: true

# Optimization
optimizer:
  name: "adamw"
  lr: 1.0e-4
  weight_decay: 1.0e-5

# Loss function
loss:
  architecture: "FullBatchMLNCELoss"
  beta: 10.0         # Temperature parameter
  learn_beta: false  # Fix temperature (don't learn it)

# Learning rate scheduler (optional)
scheduler:
  name: "linear_warmup_cosine_annealing"
  warmup_steps: 1000
  max_steps: 100000

# Training configuration
training:
  max_epochs: 100
  batch_size: 16384
  val_batch_size: 128
  dedup_pairs: true                    # Remove duplicate pairs in batch
  log_retrieval_metrics: true          # Compute Top-K metrics
  limit_train_batches: null            # null = use all data
  limit_val_batches: null
  gradient_clip_val: 1.0               # Clip gradients for stability
  accumulate_grad_batches: 1           # Gradient accumulation

# Logging and checkpointing
logging:
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
  save_top_k: 3                        # Keep top 3 checkpoints
  save_last: true                      # Always save most recent
  monitor: "val_retrieval_queries/top_1"  # Metric to track
  mode: "max"                          # Maximize this metric
  every_n_epochs: 1                    # Checkpoint frequency

# Random seed
seed: 42
```

## Command-Line Overrides

You can override any configuration value from the command line using dot notation:

```bash
# Override single values
python train.py --config configs/sota.yaml --training.max_epochs 50

# Override nested values
python train.py --config configs/sota.yaml --optimizer.lr 5e-5

# Override multiple values
python train.py --config configs/sota.yaml \
    --training.max_epochs 50 \
    --training.batch_size 8192 \
    --optimizer.lr 5e-5
```

Both `--key=value` and `--key value` syntaxes are supported.

## Creating Custom Configurations

To create a new configuration:

1. **Copy an existing config**:
   ```bash
   cp configs/sota.yaml configs/my_experiment.yaml
   ```

2. **Update data paths** (if using different data):
   ```yaml
   data:
     train_pairs_path: "data/my_dataset/train_pairs.csv"
     test_pairs_path: "data/my_dataset/test_pairs.csv"
     train_reactions_path: "data/my_dataset/train_rxns.csv"
     test_reactions_path: "data/my_dataset/test_rxns.csv"
     protein_embeds_path: "data/my_dataset/prots_t5.h5"
   ```

3. **Adjust hyperparameters** as needed:
   ```yaml
   training:
     batch_size: 8192      # Reduce if GPU memory limited
     max_epochs: 200       # Train longer
   
   optimizer:
     lr: 5.0e-5            # Try different learning rate
   ```

4. **Run training**:
   ```bash
   python train.py --config configs/my_experiment.yaml
   ```

## Hyperparameter Recommendations

### Batch Size
- **Large batches** (16K-32K): Better contrastive learning, needs more GPU memory
- **Small batches** (2K-8K): Works on smaller GPUs, may need more epochs
- Rule of thumb: Use largest batch that fits in memory

### Learning Rate
- Default: `1e-4` works well with AdamW and large batches
- If using smaller batches, try `5e-5` or `3e-5`
- Use warmup (1000-5000 steps) to stabilize early training

### Temperature (Beta)
- Default: `β=10.0` (fixed) works well for most cases
- Higher β: More emphasis on hard negatives
- Lower β: Softer similarities, more forgiving
- Can set `learn_beta: true` to learn it during training

### Model Size
- Default: 4096-dim hidden layer works well for SOTA
- Smaller models: Try [2048, 512] or [1024, 512]
- Larger models: Try [8192, 512] or [4096, 1024, 512]

## Validation Metrics

During training, the following metrics are logged:

**Training**:
- `train_loss`: MLNCE loss value
- `learning_rate`: Current learning rate (if scheduler used)

**Validation - Retrieval (Queries)**:
- `val_retrieval_queries/top_1`: Top-1 accuracy (query → target)
- `val_retrieval_queries/top_10`: Top-10 accuracy
- `val_retrieval_queries/top_100`: Top-100 accuracy
- `val_retrieval_queries/top_1000`: Top-1000 accuracy
- `val_retrieval_queries/positive_score`: Average score for positive pairs
- `val_retrieval_queries/negative_score`: Average score for negative pairs

**Validation - Retrieval (Targets)**:
- `val_retrieval_targets/top_1`: Top-1 accuracy (target → query)
- `val_retrieval_targets/top_10`: Top-10 accuracy
- (similar structure to queries)

**Validation - Retrieval (Pairs)**:
- Metrics for specific validation pair evaluation

All metrics are logged to CSV files in the `logs/` directory.

## Common Issues

### Out of Memory Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `training.batch_size` (try 8192, 4096, or 2048)
- Reduce `training.val_batch_size` (try 64 or 32)
- Use gradient accumulation: `training.accumulate_grad_batches: 2`

### Slow Training

**Problem**: Training takes too long

**Solutions**:
- Increase batch size if GPU has room
- Reduce `training.max_epochs`
- Use `training.limit_train_batches` for testing (e.g., 100)
- Ensure data is loaded in memory (`in_memory: true` in data configs)

### Poor Convergence

**Problem**: Loss not decreasing or validation metrics not improving

**Solutions**:
- Check learning rate (try 5e-5 or 3e-5 if too high)
- Increase batch size for better contrastive signal
- Adjust temperature β (try 5.0 or 15.0)
- Enable learning rate warmup
- Train for more epochs

### Data Loading Errors

**Problem**: `FileNotFoundError` or data loading failures

**Solutions**:
- Verify all data paths in config point to existing files
- Run `python scripts/download_data.py` if SOTA data missing
- Check that nanodata files exist for testing
- Ensure file permissions allow reading

## Further Information

- **Model Architecture**: See main README.md
- **Data Formats**: See data/README.md
- **Training Script**: See train.py --help
- **Testing**: Run `pytest tests/test_integration.py` to verify setup

