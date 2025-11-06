# Horizyn User Manual

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Configuration System](#configuration-system)
7. [Usage Guide](#usage-guide)
8. [Implementation Details](#implementation-details)
9. [Testing](#testing)

---

## Overview

Horizyn is a contrastive learning framework for matching enzymatic reactions with their catalyzing proteins. The system uses a dual-encoder architecture that learns a shared embedding space where reactions and their associated proteins are positioned close together.

### Key Features

- **Dual Encoder Architecture**: Separate encoders for reactions (chemical fingerprints) and proteins (pre-computed embeddings)
- **Multi-Label NCE Loss**: Handles the many-to-many relationship between reactions and proteins
- **Full Batch Training**: All training data loaded into memory for efficient contrastive learning
- **Pre-split Datasets**: Eliminates variability from data splitting
- **Single GPU**: Optimized for single-GPU training without distributed computing complexity

### Design Philosophy

Horizyn contains the code required to train the State-of-the-Art (SOTA) model from [the paper](https://www.biorxiv.org/content/10.1101/2025.08.21.671639v1). Every component directly contributes to training this single configuration. The codebase prioritizes:

- **Clarity**: Self-documenting code with comprehensive docstrings
- **Reproducibility**: Fixed data splits and deterministic training
- **Simplicity**: No extensibility abstractions or unused features
- **Performance**: All data in memory for fast training

### Recent Bug Fixes

Three critical bugs were identified and fixed that significantly improve training quality and validation metrics:

1. **Bidirectional Reactions**: Reactions are now trained in both forward and backward directions (doubled training data)
2. **Full Screening Set**: Validation now uses ALL proteins from training AND validation sets (fixed ~68% coverage gap)
3. **Correct Fingerprint Radius**: Morgan fingerprints now use radius=3 (ECFP6) instead of radius=2 (matches API/hatchery)

These fixes are expected to improve top-1 hit rates from ~1% to 30-40% as reported in the paper.

---

## Architecture

### System Components

```
horizyn/
├── horizyn/                    # Main package
│   ├── model.py               # Neural network architectures
│   ├── lightning_module.py    # PyTorch Lightning training logic
│   ├── data_module.py         # Data loading orchestration
│   ├── config.py              # Configuration management
│   ├── losses.py              # Loss functions (MLNCE)
│   ├── metrics.py             # Retrieval metrics
│   ├── datasets/              # Dataset classes
│   │   ├── base.py           # Base dataset abstractions
│   │   ├── collection.py     # Dataset composition utilities
│   │   ├── sql.py            # SQLite dataset loader
│   │   ├── hdf5.py           # HDF5 embedding loader
│   │   ├── transform.py      # Data transformations
│   │   └── fingerprints/     # Chemical fingerprint generation
│   │       ├── base.py       # Fingerprint base class
│   │       ├── rdkit_plus.py # RDKit structural fingerprints
│   │       └── drfp.py       # Differential reaction fingerprints
│   ├── chemistry/             # Chemistry utilities
│   │   └── standardizer.py   # SMILES standardization
│   └── utils/                 # Utility functions
│       ├── cache.py          # In-memory caching
│       └── collate.py        # Batch collation
├── configs/                   # Training configurations
│   ├── sota.yaml             # SOTA configuration
│   └── nano.yaml             # Small test configuration
├── scripts/                   # Helper scripts
│   └── download_data.py      # Dataset download
├── train.py                   # Main training entry point
└── tests/                     # Test suite
```

### Data Flow

The training pipeline follows this data flow:

```
1. Configuration Loading (config.py)
   ↓
2. Data Module Setup (data_module.py)
   ├── Load training pairs (SQLDataset)
   ├── Load validation pairs (SQLDataset)
   ├── Load reactions (SQLDataset)
   ├── Generate RDKit+ fingerprints (RDKitPlusFingerprintDataset)
   ├── Generate DRFP fingerprints (DRFPFingerprintDataset)
   ├── Concatenate fingerprints (MergeDataset + ConcatTensorTransform)
   └── Load protein embeddings (EmbedDataset)
   ↓
3. Lightning Module (lightning_module.py)
   ├── Initialize model (DualContrastiveModel)
   ├── Initialize loss (FullBatchMLNCELoss)
   └── Initialize metrics (create_retrieval_metrics)
   ↓
4. Training Loop
   ├── Forward pass through dual encoders
   ├── Compute MLNCE loss
   ├── Backward pass and optimizer step
   └── Validation with retrieval metrics
   ↓
5. Checkpointing and Logging
```

---

## Component Details

### Model Architecture (`horizyn/model.py`)

The model module contains three main classes:

#### 1. BaseModel

A foundational class providing structure for organizing layers into preprocessing, main body, and postprocessing stages. All Horizyn models inherit from this.

**Key Features**:
- Organized layer structure (pre_nn_layers, main_nn, post_nn_layers)
- Optional output heads for multi-task learning
- Parameter counting utilities
- Device management helpers

#### 2. NormalizeLayer

A simple layer that L2-normalizes its input, crucial for contrastive learning.

```python
output = F.normalize(input, p=2, dim=-1)
```

#### 3. MLP (Multi-Layer Perceptron)

The core encoder architecture used for both reactions and proteins.

**Configuration**:
- `layer_widths`: List of layer dimensions (e.g., [2048, 4096, 512])
- `use_batch_norm`: Apply batch normalization after each layer
- `use_layer_norm`: Apply layer normalization instead of batch norm
- `dropout`: Dropout probability between layers
- `activation`: Activation function (default: ReLU)
- `normalize_output`: L2-normalize final output (required for contrastive learning)

**SOTA Configuration**:
- Query encoder: [2048, 4096, 512] (reaction fingerprints → embeddings)
- Target encoder: [1024, 4096, 512] (protein embeddings → embeddings)
- No batch norm, no dropout, ReLU activation, normalized output

#### 4. DualContrastiveModel

The top-level model combining two encoders into a dual-encoder architecture.

**Components**:
- `query_encoder`: Processes reaction fingerprints
- `target_encoder`: Processes protein embeddings
- Both produce 512-dimensional normalized embeddings

**Forward Pass**:
```python
query_embeddings = model.query_encoder(reaction_fingerprints)
target_embeddings = model.target_encoder(protein_embeddings)
# Both are L2-normalized for cosine similarity computation
```

---

### Loss Functions (`horizyn/losses.py`)

#### FullBatchMLNCELoss (Multi-Label Noise Contrastive Estimation)

The core loss function that handles many-to-many relationships between reactions and proteins.

**Key Concepts**:

1. **Contrastive Learning**: Pulls positive pairs close together while pushing negative pairs apart
2. **Multi-Label Support**: Each query can have multiple valid targets (and vice versa)
3. **Symmetric Loss**: Computes both query→target and target→query losses
4. **Temperature Scaling**: Controls the concentration of the distribution (beta parameter)

**How It Works**:

Given a batch of query embeddings Q and target embeddings T:

1. Compute pairwise cosine distances: `D = 1 - (Q @ T^T)`
2. Apply temperature scaling: `scaled_distances = beta * D`
3. For each query, identify all positive targets in the batch
4. Compute log-sum-exp over all targets, weighted by positive labels
5. Compute symmetric loss (query→target and target→query)

**Parameters**:
- `beta`: Temperature parameter (SOTA uses 10.0, fixed)
- `learn_beta`: Whether to learn beta during training (SOTA uses False)
- `beta_min`, `beta_max`: Constraints on beta if learned

**Key Design Decision**: Uses cosine distance (1 - cosine similarity) rather than raw similarities, which provides more stable gradients.

---

### Metrics (`horizyn/metrics.py`)

The metrics module provides retrieval evaluation during validation.

#### Top-K Hit Rate

Measures the percentage of queries where at least one correct target appears in the top K retrievals.

```python
top_k_hit_rate(distances, targets, k=10)
# Returns: fraction of queries with at least one positive in top-K
```

#### Mean Reciprocal Rank (MRR)

Computes the average of 1/rank for the first positive target.

```python
MRR = mean(1 / rank_of_first_positive)
```

#### Positive/Negative Score Metrics

Track the distribution of scores for positive and negative pairs:
- `positive_score`: Mean distance for true positive pairs
- `negative_score`: Mean distance for negative pairs

**Validation Process**:

The validation loop uses three dataloaders for multi-label retrieval:

1. **Loss Dataloader**: Computes validation loss on held-out pairs (includes both forward and backward)
2. **Lookup Table Dataloader**: Loads ALL target embeddings (full screening set: train + val proteins)
3. **Retrieval Dataloader**: Queries against the lookup table (unique queries only, both directions)

**Multi-Label Retrieval**: Each reaction (query) can be catalyzed by multiple proteins (targets). For example, SwissProt has ~73K validation pairs (after bidirectional augmentation) representing ~2.3K unique reaction-directions with valid proteins per reaction direction.

**Critical**: The screening set must include ALL proteins from both training and validation to ensure validation queries can find their target proteins. This was a critical bug that has been fixed.

For each unique query:
- Retrieve the list of ALL valid target IDs for this query
- Compute distances to all targets in the lookup table
- Rank targets by distance (ascending)
- Check if ANY of the valid targets appear in top-K
- Compute Top-K hit rates (K = 1, 10, 100, 1000)
- Compute MRR, positive scores, negative scores

This is fundamentally different from single-label retrieval where each query has exactly one correct answer.

---

### Data Module (`horizyn/data_module.py`)

The `HorizynDataModule` orchestrates all data loading and preprocessing.

#### Initialization Phase

When `setup("fit")` is called:

1. **Load Training Data**:
   - Training pairs from SQLite (reaction_id, protein_id)
   - Reactions from SQLite (reaction_id, SMILES)
   - **Bidirectional Augmentation**: Each reaction is duplicated as forward (_f) and backward (_r) variants
   - Protein embeddings from HDF5 (protein_id → 1024-dim T5 embedding)

2. **Generate Fingerprints**:
   - RDKit+ structural fingerprints (1024-dim, radius=3) from reaction SMILES
   - DRFP differential fingerprints (1024-dim, radius=3) from reaction SMILES
   - Concatenate to produce 2048-dim reaction fingerprints
   - Generated for both forward and backward reactions

3. **Create Training Dataset**:
   - Merge fingerprints with protein embeddings based on pairs
   - All data loaded into memory (~15GB for SwissProt)
   - Pairs are duplicated for forward and backward reactions

4. **Setup Validation Data**:
   - Load validation pairs from SQLite
   - **Bidirectional Augmentation**: Duplicate validation pairs for forward and backward reactions
   - Group pairs by query_id (reaction) for multi-label retrieval
   - Create unique query dataset (one entry per reaction direction)
   - Store mapping from each query to its list of valid target IDs
   - **Full Screening Set**: Load ALL proteins from both training AND validation sets
   - Reuse reaction fingerprints and protein embeddings (no recomputation)

#### Key Design: Memory vs Speed Tradeoff

Horizyn loads all data into memory for maximum training speed. This requires:
- ~15GB RAM for full SwissProt dataset
- But eliminates I/O bottlenecks during training
- All fingerprints computed once, cached forever

**Data Reuse Optimization**: The same reaction fingerprints and protein embeddings are shared between training and validation datasets, avoiding redundant computation.

---

### Dataset Classes (`horizyn/datasets/`)

The dataset system is built on composable abstractions:

#### Base Classes (`base.py`)

**BaseDataset**: Foundation for all datasets

- **Key-based access**: `dataset[key]` where key is a unique identifier (string)
- **Index-based access**: `dataset[0]` for iteration compatibility
- **Transform support**: Apply functions to data on access
- **Lazy or eager loading**: Subclasses choose their strategy

**Design Decision**: All database keys are converted to strings during loading to avoid ambiguity with integer indices. This creates clean separation:
- Integers are always indices (0, 1, 2, ...)
- Strings are always keys ("rxn_123", "prot_456", ...)

**WrapperDataset**: Applies transforms to existing datasets

```python
transformed = WrapperDataset(original, transform=my_transform)
```

#### Collection Classes (`collection.py`)

**MergeDataset**: Combines multiple datasets by key intersection

```python
merged = MergeDataset(dataset1, dataset2, dataset3)
# Returns: {key: {**data1, **data2, **data3}} for all common keys
```

Used to combine reaction fingerprints with protein embeddings.

**TupleDataset**: Pairs data from multiple datasets

```python
paired = TupleDataset(
    queries=reactions,
    targets=proteins,
    pairs=[(rxn_id, prot_id), ...],
    rename_map={"queries": "reaction", "targets": "protein"}
)
```

**Key Feature**: `skip_missing=True` (default) filters out pairs referencing non-existent keys with warnings, allowing training to proceed with imperfect data.

#### SQL Datasets (`sql.py`)

**SQLDataset**: Loads data from SQLite databases

```python
dataset = SQLDataset(
    db_path="reactions.db",
    table="reaction",
    id_column="reaction_id",
    columns=["smiles"],
    rename_map={"smiles": "reaction"},
    in_memory=True  # Load all data into RAM
)
```

**Features**:
- Column selection and renaming
- In-memory or on-the-fly loading
- Automatic string key conversion
- Missing data handling

#### HDF5 Datasets (`hdf5.py`)

**EmbedDataset**: Loads embeddings from HDF5 files

HDF5 files must contain two datasets: `ids` (string identifiers) and `vectors` (embedding matrix).

```python
dataset = EmbedDataset(
    file_path="data/swissprot/proteins_t5_embeddings.h5",
    in_memory=True,  # Load all embeddings into RAM
    dtype=torch.float32
)
```

**Features**:
- Handles string or byte ID formats
- Memory-efficient HDF5 indexing when not in_memory
- Pre-loading for fast access during training

#### Fingerprint Datasets (`datasets/fingerprints/`)

**BaseFingerprintDataset** (`base.py`): Foundation for fingerprint generation

- Integrates with `Standardizer` for consistent SMILES processing
- Caches fingerprints in memory after first computation
- Wraps existing datasets (e.g., SQLDataset with SMILES)

**RDKitPlusFingerprintDataset** (`rdkit_plus.py`): Structural fingerprints

Generates RDKit fingerprints for reactions using two approaches:

1. **Structural ("struct")**: Concatenate product and reactant fingerprints
   ```
   fp = concat(product_fp, reactant_fp)
   ```

2. **Difference ("diff")**: Compute product - reactant difference
   ```
   fp = product_fp - reactant_fp
   ```

**SOTA uses "struct" mode** with Morgan fingerprints (radius=3, 1024 bits, ECFP6).

**Supported fingerprint types**:
- Morgan (circular fingerprints)
- AtomPair (atom pair fingerprints)
- TopologicalTorsion (torsion fingerprints)
- RDKit (Daylight-like fingerprints)
- Pattern (pattern fingerprints)

**DRFPFingerprintDataset** (`drfp.py`): Differential reaction fingerprints

Generates DRFP fingerprints that encode reaction transformations directly.

```python
drfp_fp = DrfpEncoder.encode(
    rxn_smiles,
    radius=3,
    rings=True
)
```

**SOTA uses 1024-bit DRFP** with radius=3 and rings=True.

---

### Chemistry Utilities (`horizyn/chemistry/`)

#### Standardizer (`standardizer.py`)

The `Standardizer` class normalizes SMILES strings for consistent fingerprint generation.

**Standardization Steps**:

1. **Hypervalent atoms**: Fix unusual valence states
2. **Hydrogen removal**: Remove explicit hydrogens
3. **Kekulization**: Convert aromatic forms to explicit bonds
4. **Uncharging**: Neutralize molecules where appropriate
5. **Metal standardization**: Handle metal-containing compounds

**Configuration**:

```python
standardizer = Standardizer(
    standardize_hypervalent=True,    # Fix hypervalent atoms
    standardize_remove_hs=True,      # Remove explicit hydrogens
    standardize_kekulize=False,      # Keep aromatic representation
    standardize_uncharge=True,       # Neutralize molecules
    standardize_metals=True          # Standardize metals
)
```

**SOTA Configuration** (all parameters explicitly set):
- `standardize_hypervalent=True` - Fix double bonds in hypervalent compounds
- `standardize_remove_hs=True` - Remove explicit hydrogen atoms
- `standardize_kekulize=False` - Keep aromatic representation (don't kekulize)
- `standardize_uncharge=True` - Neutralize molecules by protonation/deprotonation
- `standardize_metals=True` - Disconnect bonds between metals and N, O, F atoms

**Reaction Standardization**: When `reactions=True`, the standardizer processes both sides of the reaction SMILES separately.

**Design Note**: This code handles edge cases in chemical data that would otherwise cause fingerprint generation to fail or produce inconsistent results. All five standardization parameters are fully configurable via the config files.

---

### Lightning Module (`horizyn/lightning_module.py`)

The `HorizynLitModule` implements the PyTorch Lightning training loop.

#### Training Step

For each batch:

1. **Extract Data**:
   ```python
   query_fps = batch["reaction"]     # Reaction fingerprints
   target_embeds = batch["protein"]  # Protein embeddings
   pair_ids = batch["pr_id"]         # Pair IDs
   ```

2. **Deduplicate**: Remove duplicate pairs within the batch (important for MLNCE)

3. **Forward Pass**:
   ```python
   query_embeds = model.query_encoder(query_fps)
   target_embeds = model.target_encoder(target_embeds)
   ```

4. **Compute Loss**:
   ```python
   loss = loss_fn(
       query_embeds=query_embeds,
       target_embeds=target_embeds,
       query_ids=query_ids,
       target_ids=target_ids
   )
   ```

5. **Log Metrics**: Training loss logged every step

#### Validation Step

Validation uses three separate dataloaders:

**Dataloader 0: Loss Computation**
```python
val_loss = loss_fn(query_embeds, target_embeds, query_ids, target_ids)
```

**Dataloader 1: Lookup Table Building**

Builds a complete lookup table of all target embeddings:
```python
lookup_table = {target_id: embedding for all targets}
```

**Dataloader 2: Retrieval Metrics (Multi-Label)**

The retrieval dataloader iterates over **unique queries only** (not all pairs).

For each query batch:
1. Look up the list of valid target IDs for this query (from `_val_retrieval_targets`)
2. Compute distances from query embeddings to all targets in lookup table
3. Rank targets by distance (ascending)
4. Check if ANY of the valid targets appear in top-K
5. Compute Top-K hit rates, MRR, score distributions

**Example**: If query "rxn_123" has 3 valid proteins ["prot_A", "prot_B", "prot_C"], the top-1 metric checks if ANY of these 3 proteins ranks #1 out of all 34,187 proteins.

**Why Three Dataloaders?**

This design separates three distinct validation tasks:
1. **Loss computation** - Uses pairs (36,433 pairs) to measure generalization
2. **Lookup table** - Loads all target embeddings once for efficiency
3. **Retrieval metrics** - Uses unique queries (1,147 queries) for multi-label evaluation

Each has different batching requirements and data access patterns. The retrieval dataloader groups pairs by query to correctly handle the many-to-many relationship between reactions and proteins.

#### Optimizer Configuration

SOTA uses AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

No learning rate scheduler in the SOTA configuration.

---

### Configuration System (`horizyn/config.py`)

The configuration system loads YAML files and applies command-line overrides.

#### Config Structure

Configs are organized into four sections:

**1. Logging**:
```yaml
logging:
  log_dir: logs
  checkpoint_dir: checkpoints
  save_every_n_epochs: 10
```

**2. Data**:
```yaml
data:
  train_pairs_path: data/swissprot/train_pairs.db
  val_pairs_path: data/swissprot/val_pairs.db
  reactions_path: data/swissprot/reactions.db
  proteins_path: data/swissprot/proteins_t5_embeddings.h5
  train_batch_size: 16384
  retrieval_batch_size: 128
  
  # Reaction standardization (all parameters explicit)
  standardize_reactions: true
  standardize_hypervalent: true
  standardize_remove_hs: true
  standardize_kekulize: false
  standardize_uncharge: true
  standardize_metals: true
```

**3. Model**:
```yaml
model:
  name: DualContrastiveModel
  query_encoder_dims: [2048, 4096, 512]
  target_encoder_dims: [1024, 4096, 512]
  embedding_dim: 512
```

**4. Training**:
```yaml
training:
  max_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.01
  loss:
    name: FullBatchMLNCELoss
    beta: 10.0
    learn_beta: false
```

#### DotDict Access

Configs are loaded as `DotDict` objects supporting both dict and attribute access:

```python
config = load_config("configs/sota.yaml")
config.data.train_batch_size  # Attribute access
config["data"]["train_batch_size"]  # Dict access
```

#### Command-Line Overrides

Override any config value from the command line:

```bash
python train.py --config configs/sota.yaml \
    --training.max_epochs 50 \
    --data.train_batch_size 8192 \
    --seed 123
```

Override syntax supports:
- `--key=value` (equals sign)
- `--key value` (space-separated)
- Nested keys with dots: `--section.subsection.key value`

---

## Data Pipeline

### Data Format

Horizyn expects four files:

#### 1. Training Pairs (`train_pairs.db`)

SQLite database with table `protein_to_reaction`:

| Column | Type | Description |
|--------|------|-------------|
| pr_id | INTEGER | Unique pair ID |
| reaction_id | TEXT | Reaction identifier |
| protein_id | TEXT | Protein identifier |

#### 2. Validation Pairs (`val_pairs.db`)

Same schema as training pairs, disjoint set for validation.

#### 3. Reactions (`reactions.db`)

SQLite database with table `reaction`:

| Column | Type | Description |
|--------|------|-------------|
| reaction_id | TEXT | Unique reaction ID |
| smiles | TEXT | Reaction SMILES string |

Format: `reactants>>products` (e.g., `CC(=O)O>>CO.CC(=O)`)

#### 4. Protein Embeddings (`proteins_t5_embeddings.h5`)

HDF5 file with two datasets:

- `ids`: Array of protein IDs (strings or bytes)
- `embeddings`: Array of shape (N, 1024) with T5 embeddings

**Key Requirement**: IDs in pairs files must match IDs in reactions.db and proteins.h5.

### Memory Requirements

For SwissProt dataset (with bidirectional augmentation):
- **Training pairs**: ~200K pairs (doubled) → ~2 MB
- **Validation pairs**: ~40K pairs (doubled) → ~400 KB
- **Reactions**: ~100K reactions (doubled: forward + backward) → ~100 MB (SMILES strings)
- **Protein embeddings**: ~500K proteins × 1024 floats → ~2 GB
- **RDKit+ fingerprints**: ~100K reactions × 1024 bits → ~12 MB (cached, bidirectional)
- **DRFP fingerprints**: ~100K reactions × 1024 bits → ~12 MB (cached, bidirectional)

**Total**: ~2.2 GB base + ~10 GB working memory during training = **~15 GB RAM**

Note: Bidirectional augmentation doubles the reaction count but has minimal impact on memory since fingerprints are computed on-demand and cached.

### Data Standardization

**SMILES Standardization**: All reactions are standardized before fingerprint generation using five explicit parameters:
- `standardize_hypervalent=True` - Fix double bonds in hypervalent compounds
- `standardize_remove_hs=True` - Remove explicit hydrogen atoms
- `standardize_kekulize=False` - Keep aromatic representation (don't kekulize)
- `standardize_uncharge=True` - Neutralize molecules by protonation/deprotonation
- `standardize_metals=True` - Disconnect bonds between metals and N, O, F atoms

These settings ensure consistent fingerprint generation and match the configuration used in hatchery and the API.

**Schema Consistency**: The SwissProt dataset defines the canonical schema:
- Table name: `protein_to_reaction` (not `pairs`)
- Columns: `pr_id`, `reaction_id`, `protein_id`

Other datasets use `rename_map` to adapt their schema to this standard without modifying the actual database files.

---

## Training Pipeline

### Full Training Workflow

#### 1. Setup

```bash
# Install dependencies
uv sync

# Download data
python scripts/download_data.py --output_dir data/
```

#### 2. Training

```bash
python train.py --config configs/sota.yaml
```

**What Happens**:

1. **Config Loading** (< 1 second)
   - Parse YAML config
   - Apply command-line overrides
   - Set random seeds

2. **Data Module Setup** (1-2 minutes)
   - Load training pairs into memory
   - Load validation pairs into memory
   - Load all reactions into memory
   - Load all protein embeddings into memory
   - Generate and cache all RDKit+ fingerprints
   - Generate and cache all DRFP fingerprints
   - Create training and validation datasets

3. **Model Initialization** (< 1 second)
   - Initialize dual encoders (MLP networks)
   - Move to GPU
   - Initialize loss function
   - Initialize optimizer

4. **Training Loop** (hours, depending on epochs)
   - For each epoch:
     - Iterate through training batches
     - Forward pass through dual encoders
     - Compute MLNCE loss
     - Backward pass and optimizer step
     - Log training loss
   - Every N epochs:
     - Run validation
     - Compute validation loss
     - Compute retrieval metrics
     - Save checkpoint

5. **Checkpointing**
   - Save model weights every 10 epochs (configurable)
   - Save final checkpoint at end
   - Checkpoints include full model state for resuming

#### 3. Monitoring

Training progress is logged to:
- **Terminal**: Progress bars, loss values, metrics
- **CSV logs**: `lightning_logs/version_X/metrics.csv`
- **Checkpoints**: `checkpoints/epoch=X-step=Y.ckpt`

### Resuming Training

To resume from a checkpoint:

```bash
python train.py --config configs/sota.yaml --resume checkpoints/last.ckpt
```

The system automatically:
- Loads model weights
- Restores optimizer state
- Continues from the saved epoch

---

## Configuration System

### SOTA Configuration Explained

The SOTA config (`configs/sota.yaml`) reproduces the paper result:

**Data Configuration**:
- **Batch size**: 16384 (large for full-batch MLNCE)
- **Fingerprints**: RDKit+ (1024) + DRFP (1024) = 2048 total
- **Standardization**: hypervalent=True, remove_hs=True, kekulize=False, uncharge=True, metals=True

**Model Configuration**:
- **Query encoder**: 2048 → 4096 → 512 (MLP)
- **Target encoder**: 1024 → 4096 → 512 (MLP)
- **Output**: 512-dim L2-normalized embeddings

**Training Configuration**:
- **Epochs**: 100
- **Optimizer**: AdamW with lr=1e-4, weight_decay=0.01
- **Loss**: MLNCE with beta=10.0 (fixed)
- **Validation**: Every 10 epochs

**Why These Choices?**

- **Large batch size**: MLNCE benefits from many negative examples per batch
- **Single hidden layer**: Simple architecture avoids overfitting
- **Fixed beta**: Temperature found through ablation studies
- **No learning rate schedule**: Training is stable with constant lr

### Nano Configuration

For testing, `configs/nano.yaml` uses a tiny dataset:
- 10 training pairs, 5 validation pairs
- ~5 reactions, ~8 proteins
- Completes in seconds

**Use for**:
- Smoke testing after code changes
- Debugging training loop
- CI/CD integration tests

---

## Usage Guide

### Basic Training

Train the SOTA model:

```bash
python train.py --config configs/sota.yaml
```

### Custom Training

Override batch size:

```bash
python train.py --config configs/sota.yaml --data.train_batch_size 8192
```

Train for fewer epochs:

```bash
python train.py --config configs/sota.yaml --training.max_epochs 50
```

Change learning rate:

```bash
python train.py --config configs/sota.yaml --training.learning_rate 0.0005
```

### Using Your Own Data

1. **Prepare your data** in the required format (see Data Format section)

2. **Create a custom config**:

```yaml
data:
  train_pairs_path: path/to/your/train_pairs.db
  val_pairs_path: path/to/your/val_pairs.db
  reactions_path: path/to/your/reactions.db
  proteins_path: path/to/your/proteins.h5
```

3. **Train**:

```bash
python train.py --config your_config.yaml
```

### Debugging

Enable detailed logging:

```bash
python train.py --config configs/nano.yaml --log_every_n_steps 1
```

Use nano config for quick testing:

```bash
python train.py --config configs/nano.yaml --training.max_epochs 2
```

---

## Implementation Details

### Key Design Decisions

**String Keys**: All database keys are converted to strings during loading to avoid ambiguity with integer indices used by PyTorch DataLoader. This creates clean separation where integers are always indices and strings are always keys.

**Skip Missing Pairs**: `TupleDataset` filters out pairs referencing non-existent keys by default with warnings. This allows training to proceed when reactions or proteins are filtered from the dataset for quality reasons.

**Multi-Label Retrieval**: Validation pairs are grouped by query_id before metrics computation. This correctly handles the many-to-many relationship where each reaction can be catalyzed by multiple proteins. Metrics check if ANY valid target appears in top-K, not just one specific target. This is critical for accurate evaluation.

**Three Validation Dataloaders**: Validation uses three separate dataloaders for distinct tasks:
1. Computing loss on all pairs (36,433 pairs in SwissProt)
2. Building a lookup table of all target embeddings (34,187 proteins)
3. Evaluating retrieval metrics on unique queries (1,147 reactions)

Each has different data formats and batching requirements. The retrieval dataloader uses unique queries with associated target lists to support multi-label evaluation.

**Memory-First Design**: All data loads into memory and fingerprints compute once at initialization. This eliminates I/O bottlenecks for maximum training speed at the cost of requiring ~15GB RAM. The large batch size (16384) benefits from memory-resident data enabling fast random access during shuffling.

**Bidirectional Reactions**: All reactions are trained and evaluated in both forward (reactants→products) and backward (products→reactants) directions. This doubles the training data and ensures the model learns reversible reactions correctly. Reaction keys are suffixed with `_f` (forward) or `_r` (reverse).

**Full Screening Set**: The validation lookup table contains ALL proteins from both training and validation sets (~500K proteins). This ensures validation queries can retrieve their target proteins even if those proteins only appear in validation pairs. Previously, this was a critical bug where ~68% of validation queries had 0% hit rate because their targets weren't in the lookup table.

**Cosine Distance**: The loss function uses cosine distance (1 - cosine similarity) rather than raw similarity scores, providing intuitive semantics where lower values indicate more similar pairs and stable gradients from normalized embeddings.

**Full Standardization**: All five standardization parameters are explicitly configured in the SOTA config (hypervalent, remove_hs, kekulize, uncharge, metals) to ensure reproducibility and match hatchery/API behavior.

### Performance Optimizations

**Batch Size**: SOTA uses 16384 because MLNCE requires many negative examples per query and memory allows it (2048-dim fingerprints × 16384 samples = 128 MB per batch).

**Data Loading Workers**: SOTA uses 0 workers since all data is in memory with no I/O to parallelize, avoiding worker process overhead.

**Fingerprint Caching**: Fingerprints compute once per dataset instance on first access and cache in memory as a dict with no serialization overhead.

---

## Testing

### Development Workflow

When developing or debugging, follow this recommended workflow for fast feedback:

#### 1. Quick Iteration: Run Training Directly

Run training with limited batches to catch errors immediately:

```bash
# Fast smoke test with nanodata (~30 seconds)
python train.py --config configs/nano.yaml \
    --training.max_epochs 2 \
    --training.limit_train_batches 10 \
    --training.limit_val_batches 3
```

**Why this is better than tests for development**:
- ✅ See progress bars, loss values, and metrics in real-time
- ✅ Get immediate feedback with full stack traces
- ✅ Test exactly how users will run your code
- ✅ Fast iteration (< 1 minute with nanodata)

For SwissProt testing without waiting for full training:

```bash
# Test with full data but limited batches (~2 minutes)
python train.py --config configs/sota.yaml \
    --training.max_epochs 1 \
    --training.limit_train_batches 50 \
    --training.limit_val_batches 10
```

#### 2. Verification: Run Fast Tests

After confirming training works, run automated tests for verification:

```bash
# Run all tests (unit + integration, ~1 minute)
pytest
```

This runs 420 tests including:
- 398 unit tests (datasets, models, losses, metrics, config, etc.)
- 22 integration tests (smoke tests with nanodata + SwissProt data validation)

#### 3. Final Validation: Run SwissProt Tests (Optional)

Only run SwissProt tests when you need to validate with production data:

```bash
# Download SwissProt data first
python scripts/download_data.py

# Run SwissProt validation tests (< 5 seconds)
pytest tests/integration/test_swissprot.py -v
```

These tests verify:
- Config points to correct SwissProt files
- All data files exist and are not empty
- Database schemas match expected structure
- Dataset sizes are in expected ranges

**When to run SwissProt tests**:
- After downloading or updating SwissProt data
- Before full SOTA training runs
- When debugging data-related issues

#### 4. See Test Output in Real-Time

If you want to see progress during integration tests, use `-s` to disable output capture:

```bash
pytest tests/integration/test_smoke_nanodata.py::TestSmokeNanodata::test_smoke_training_pipeline -v -s
```

This shows Lightning's progress bars and training output during the test.

### Test Organization

The test suite is organized into unit tests and integration tests:

**Unit Tests** (`tests/unit/`, 398 tests, < 1 minute total): Fast, isolated tests of individual components (model, losses, metrics, datasets, fingerprints, config, etc.). Provides 95% code coverage across all modules.

**Integration Tests** (`tests/integration/`, 22 tests, < 30 seconds total): End-to-end tests using the included nanodata dataset (12 reactions, 11 proteins). Tests the full pipeline without downloading SwissProt data:
- Config and error handling (`test_smoke_config.py`)
- Core pipeline (`test_smoke_nanodata.py`)
- Training dynamics (`test_smoke_training.py`)
- Validation metrics and multi-label retrieval (`test_smoke_validation.py`)
- Edge cases and robustness (`test_smoke_robustness.py`)
- SwissProt data validation (`test_swissprot.py`)

All tests run by default. SwissProt tests verify data integrity (schemas, sizes) without expensive operations like fingerprint computation or training.

### Running Tests

Run all tests (unit + integration):
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=horizyn --cov-report=html
```

Run only unit tests:
```bash
pytest tests/unit/ -v
```

Run only integration tests:
```bash
pytest tests/integration/ -v
```

Run specific test file:
```bash
pytest tests/unit/test_model.py -v
```

Run specific test:
```bash
pytest tests/unit/test_model.py::TestMLP::test_forward -v
```

Run SwissProt data validation tests (requires SwissProt data):
```bash
# Download data first
python scripts/download_data.py

# Run SwissProt tests
pytest tests/integration/test_swissprot.py -v
```

### Quick Smoke Test

Test the full training pipeline without downloading SwissProt:

```bash
# Using pytest (fastest, ~10 seconds)
pytest tests/integration/test_smoke_nanodata.py -v

# Or run training directly with nanodata (~30 seconds)
python train.py --config configs/nano.yaml \
    --training.max_epochs 2 \
    --training.limit_train_batches 10
```

The nanodata files are included in the repository at `data/nanodata/`.

### Testing Validation Metrics

The validation metrics implementation includes specific tests for multi-label retrieval:

```bash
# Test that pairs are grouped by query for multi-label retrieval
pytest tests/integration/test_smoke_validation.py::TestSmokeValidationMetrics::test_validation_query_grouping -v

# Test that metrics are computed correctly (not impossibly low)
pytest tests/integration/test_smoke_validation.py::TestSmokeValidationMetrics::test_validation_metrics_not_impossibly_low -v
```

These tests verify that the validation pipeline correctly handles the many-to-many relationship between reactions and proteins, ensuring metrics check if ANY valid target appears in top-K (not just one specific target).

### Code Quality

Format code:
```bash
black horizyn/ tests/
isort horizyn/ tests/
```

Check linting:
```bash
flake8 horizyn/ tests/
```

