# Horizyn Data Directory

This directory contains the datasets required for training and evaluating the Horizyn model. The data is organized into subdirectories for different use cases.

## Directory Structure

```
data/
├── nanodata/          # Small test dataset for integration tests
└── sota/              # Full SOTA dataset for production training
```

## Expected Files

Each dataset directory should contain five standardized files:

### 1. `train_rxns.csv`
CSV file containing training reactions with SMILES strings.

**Columns:**
```
rs_id,reaction_id,reaction_smiles
0,Rh_10008,*N[C@@H](CS)C(*)=O...>>...
```

**Usage:** Loaded by `CSVDataset` to retrieve reaction SMILES for fingerprint generation during training. The `reaction_id` column is used as the key.

### 2. `val_rxns.csv`
CSV file containing validation reactions (same format as `train_rxns.csv`).

**Usage:** Loaded separately for validation to ensure no reaction leakage between train and val.

### 3. `train_pairs.csv`
CSV file containing training reaction-protein pairs.

**Columns:**
```
pr_id,reaction_id,protein_id
0,Rh_10008,P12345
```

**Usage:** Loaded by `HorizynDataModule` to define positive training pairs.

### 4. `val_pairs.csv`
CSV file containing validation reaction-protein pairs (same format as `train_pairs.csv`).

**Usage:** Loaded by `HorizynDataModule` to evaluate model performance during training.

### 5. `protein_embeds.h5`
HDF5 file containing pre-computed protein embeddings from the ProtT5 model.

**Structure:**
```
/ids      # Dataset of protein IDs (strings)
/vectors  # Dataset of embeddings (float32, shape: [N, 1024])
```

**Usage:** Loaded by `EmbedDataset` to retrieve target protein embeddings.

### 6. `prots.fasta` (optional)
FASTA file containing protein sequences corresponding to the embeddings in the HDF5 file. This is provided for reference and is not used during training.

## Datasets

### Nanodata (Integration Tests)

A minimal dataset included in this repository for integration testing:
- **Size:** ~80 KB total
- **Train Reactions:** 9 from Rhea database
- **Val Reactions:** 2 from Rhea database
- **Proteins:** 11 from UniProt with ProtT5 embeddings
- **Train Pairs:** 10 pairs
- **Val Pairs:** 2 pairs
- **Created:** July 2024

**Purpose:** Fast integration tests to verify the training pipeline works end-to-end without errors. Not intended for scientific evaluation.

**Usage:**

```bash
python train.py --config configs/nano.yaml
```

### SOTA (Production Training)

The full dataset used in the Horizyn publication:
- **Size:** ~1 GB total
- **Train Reactions:** 10,786 from Rhea
- **Val Reactions:** 1,013 from Rhea
- **Proteins:** ~200k from UniProt with ProtT5 embeddings
- **Train Pairs:** 257,734 pairs
- **Val Pairs:** 33,997 pairs
- **Embeddings:** ProtT5-XL (Rostlab/prot_t5_xl_half_uniref50-enc)

**Purpose:** Training the SOTA model for publication results.

**Training:**

```bash
python train.py --config configs/sota.yaml
```

## Data Splitting Strategy

**Critical:** Pairs are split **by reaction ID** to prevent data leakage. All pairs for a given reaction must be in either the training set OR the validation set, never both. This ensures the model is evaluated on truly unseen reactions.

The separate `train_rxns.csv` and `val_rxns.csv` files enforce this split at the data level.

## Creating Custom Datasets

If you want to use Horizyn with your own data:

1. **Prepare reactions**: Create `train_rxns.csv` and `val_rxns.csv` with your reaction SMILES
2. **Prepare proteins**: Create `protein_embeds.h5` with ProtT5 embeddings for all proteins
3. **Define pairs**: Create `train_pairs.csv` and `val_pairs.csv` with positive examples
4. **Split by reaction**: Ensure all pairs for a reaction go to train OR val, not both

## Configuration

Update your config YAML to point to your dataset:

```yaml
data:
  train_pairs_path: "data/my_dataset/train_pairs.csv"
  val_pairs_path: "data/my_dataset/val_pairs.csv"
  train_reactions_path: "data/my_dataset/train_rxns.csv"
  val_reactions_path: "data/my_dataset/val_rxns.csv"
  protein_embeds_path: "data/my_dataset/protein_embeds.h5"
```

## Storage Requirements

- **Nanodata:** ~80 KB (minimal test data, included in git)
- **SOTA:** ~1 GB uncompressed

The embeddings file is the largest component (~900 MB for SOTA).

## Notes

- All data is loaded into memory at the start of training for maximum throughput
- Nanodata files are tracked in git for easy integration testing
- SOTA data should be downloaded separately due to size
