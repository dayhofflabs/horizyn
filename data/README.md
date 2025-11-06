# Horizyn Data Directory

This directory contains the datasets required for training and evaluating the Horizyn model. The data is organized into subdirectories for different use cases.

## Directory Structure

```
data/
├── nanodata/          # Small test dataset for integration tests
└── swissprot/         # Full SwissProt dataset for production training
```

## Expected Files

Each dataset directory (nanodata, swissprot) should contain four standardized files:

### 1. `reactions.db`
SQLite database containing reaction information with SMILES strings.

**Schema:**
```sql
CREATE TABLE reaction (
    rs_id INTEGER PRIMARY KEY,
    reaction_id TEXT,
    reaction_smiles TEXT
);
```

**Usage:** Loaded by `SQLDataset` to retrieve reaction SMILES for fingerprint generation. The `reaction_id` column is used as the search key.

### 2. `proteins_t5_embeddings.h5`
HDF5 file containing pre-computed protein embeddings from the ProtT5 model.

**Structure:**
```
/ids      # Dataset of protein IDs (strings)
/vectors  # Dataset of embeddings (float32, shape: [N, 1024])
```

**Usage:** Loaded by `EmbedDataset` to retrieve target protein embeddings.

### 3. `train_pairs.db`
SQLite database containing training reaction-protein pairs.

**Schema:**
```sql
CREATE TABLE protein_to_reaction (
    pr_id INTEGER PRIMARY KEY,
    reaction_id TEXT,
    protein_id TEXT
);
```

**Usage:** Loaded by `HorizynDataModule` to define positive training pairs. The `pr_id` column is used as the search key, and `reaction_id`/`protein_id` are mapped to `query_id`/`target_id` internally.

### 4. `val_pairs.db`
SQLite database containing validation reaction-protein pairs (same schema as `train_pairs.db`).

**Schema:**
```sql
CREATE TABLE protein_to_reaction (
    pr_id INTEGER PRIMARY KEY,
    reaction_id TEXT,
    protein_id TEXT
    -- Note: swissprot adds 'db_source TEXT' column
);
```

**Usage:** Loaded by `HorizynDataModule` to evaluate model performance during training.

## Datasets

### Nanodata (Integration Tests)

A minimal dataset included in this repository for integration testing:
- **Size:** ~80 KB total (16 KB pairs, 12 KB reactions, 50 KB embeddings)
- **Reactions:** 12 from Rhea database
- **Proteins:** 11 from UniProt with ProtT5 embeddings
- **Pairs:** 12 total (split into ~10 train, ~2 val)
- **Created:** July 2024

**Purpose:** Fast integration tests to verify the training pipeline works end-to-end without errors. Not intended for scientific evaluation. Designed with intentional edge cases:
- At least one protein appears in multiple reactions
- At least one protein in pairs is missing from embeddings (tests error handling)
- At least one protein in embeddings is unused (tests filtering)

**Usage:** The nanodata files are already in the correct format and committed to the repository. Run integration tests with:

```bash
python train.py --config configs/nano.yaml
```

### SwissProt (Production Training)

The full dataset used in the Horizyn publication:
- **Size:** ~930 MB total (14 MB train pairs, 2 MB val pairs, 5 MB reactions, 904 MB embeddings)
- **Reactions:** 15,969 from Rhea v131 (714 duplicate SMILES with different IDs)
  - Training: 10,785 reactions (68%)
  - Validation: 1,147 reactions (7%)
  - Remaining: 4,037 reactions not in train/val pairs
- **Proteins:** 216,132 from SwissProt v2023_05 with ProtT5 embeddings
  - Training: 192,769 unique proteins used
  - Validation: 34,187 unique proteins used
- **Pairs:** 294,166 total reaction-protein associations
  - Training: 257,733 pairs (88%)
  - Validation: 36,433 pairs (12%)
- **Embeddings:** ProtT5-XL (Rostlab/prot_t5_xl_half_uniref50-enc), sequences >5k embedded in chunks
- **Created:** February-August 2024
- **Source:** Rhea-UniProt

**Purpose:** Training the SOTA model for publication results. This is the benchmark dataset combining Rhea biochemical reactions with experimentally validated enzyme annotations from SwissProt.

**Setup:** Download the pre-processed dataset from Zenodo:

```bash
python scripts/download_data.py
```

This will download and extract all four required files to `data/swissprot/`.

**Training:** Run with the SOTA configuration:

```bash
python train.py --config configs/sota.yaml
```

## Data Splitting Strategy

**Critical:** Pairs are split **by reaction ID** to prevent data leakage. All pairs for a given reaction must be in either the training set OR the validation set, never both. This ensures the model is evaluated on truly unseen reactions.

## Downloading the Data

The full SwissProt dataset will be available on Zenodo after publication. To download:

```bash
python scripts/download_data.py --output-dir data/swissprot
```

This will download and extract ~15GB of compressed data.

## Creating Custom Datasets

If you want to use Horizyn with your own data:

1. **Prepare reactions**: Create `reactions.db` with your reaction SMILES
2. **Prepare proteins**: Create `proteins_t5_embeddings.h5` with ProtT5 embeddings for everything in the pairs files
3. **Define pairs**: Create `train_pairs.db` and `val_pairs.db` with positive examples
4. **Split by reaction**: Ensure all pairs for a reaction go to train OR val, not both

## Configuration

Update `configs/sota.yaml` to point to your dataset:

```yaml
data:
  reactions_db: "data/my_dataset/reactions.db"
  proteins_h5: "data/my_dataset/proteins_t5_embeddings.h5"
  train_pairs_db: "data/my_dataset/train_pairs.db"
  val_pairs_db: "data/my_dataset/val_pairs.db"
```

## Storage Requirements

- **Nanodata:** 80 KB (minimal test data, included in git)
- **SwissProt:** 930 MB uncompressed (downloaded from Zenodo)

The embeddings file (`proteins_t5_embeddings.h5`) is the largest component at 904 MB for SwissProt.

## Notes

- All data is loaded into memory at the start of training for maximum throughput
- Nanodata files are tracked in git for easy integration testing
- SwissProt data should be downloaded separately due to size (~1.8 GB)

