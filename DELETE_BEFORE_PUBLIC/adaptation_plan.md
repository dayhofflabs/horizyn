# Adaptation Plan: SwissProt → SOTA Dataset Format

This document outlines the changes needed to adapt horizyn from the old SwissProt SQLite format to the new SOTA CSV format.

## Goals

1. **Drop SQLite support entirely** - Use CSV for all tabular data
2. **Support separate train/val reaction files** - Cleaner data organization
3. **Minimal code** - No backwards compatibility, just the final functionality

## Current vs Target Data Format

### Current (SwissProt - SQLite)
```
data/swissprot/
├── train_pairs.db          # SQLite: pr_id, reaction_id, protein_id
├── val_pairs.db            # SQLite: pr_id, reaction_id, protein_id
├── reactions.db            # SQLite: rs_id, reaction_id, reaction_smiles (ALL reactions)
└── proteins_t5_embeddings.h5
```

### Target (SOTA - CSV)
```
data/sota/
├── train_pairs.csv         # CSV: pr_id, reaction_id, protein_id
├── val_pairs.csv           # CSV: pr_id, reaction_id, protein_id
├── train_rxns.csv          # CSV: rs_id, reaction_id, reaction_smiles (train only)
├── val_rxns.csv            # CSV: rs_id, reaction_id, reaction_smiles (val only)
└── prots_t5.h5             # HDF5: same format, different name
```

## Files to Modify

### 1. `horizyn/datasets/sql.py` → `horizyn/datasets/csv.py`

**Action**: Replace `SQLDataset` with `CSVDataset`

The new class should:
- Load CSV files using pandas
- Keep the same interface: `keys`, `__getitem__`, `columns`, `rename_map`
- Support `in_memory=True` (always, since CSVs are loaded entirely anyway)

```python
class CSVDataset(BaseDataset[str]):
    """Dataset for loading data from CSV files."""
    
    def __init__(
        self,
        file_path: str,
        key_column: str,              # Was: search_key
        columns: list[str] | None,    # Columns to load (None = all except key)
        rename_map: dict | None,      # Column renaming
        ...
    ):
```

### 2. `horizyn/datasets/__init__.py`

**Action**: Export `CSVDataset` instead of `SQLDataset`

### 3. `horizyn/data_module.py`

**Action**: Update `HorizynDataModule` to:
- Accept separate `train_reactions_path` and `val_reactions_path`
- Use `CSVDataset` instead of `SQLDataset`
- Load train reactions for training, val reactions for validation

Key changes to `__init__`:
```python
def __init__(
    self,
    train_pairs_path: str,
    val_pairs_path: str,
    train_reactions_path: str,    # NEW: separate train reactions
    val_reactions_path: str,      # NEW: separate val reactions
    proteins_path: str,
    ...
):
```

Key changes to `_create_query_dataset()`:
- Split into `_create_train_query_dataset()` and `_create_val_query_dataset()`
- Each loads its respective reactions file

### 4. `horizyn/config.py`

**Action**: Update `validate_config()` to require new data keys:
```python
required_data_keys = [
    "train_pairs_path",
    "val_pairs_path",
    "train_reactions_path",   # NEW
    "val_reactions_path",     # NEW
    "proteins_path",
]
```

Remove the old `reactions_path` key.

### 5. `configs/sota.yaml`

**Action**: Update data paths:
```yaml
data:
  train_pairs_path: data/sota/train_pairs.csv
  val_pairs_path: data/sota/val_pairs.csv
  train_reactions_path: data/sota/train_rxns.csv
  val_reactions_path: data/sota/val_rxns.csv
  proteins_path: data/sota/prots_t5.h5
```

### 6. `configs/nano.yaml`

**Action**: Update to use CSV format with separate reaction paths.

## Files to Delete

- `horizyn/datasets/sql.py` - No longer needed

## Files to Update (Documentation)

- `data/README.md` - Document new CSV format, mention FASTA file corresponds to protein embeddings
- `configs/README.md` - Document new config keys (separate train/val reaction paths)
- `horizyn/README.md` - Update data format section

## Implementation Order

1. **Create `CSVDataset`** in `horizyn/datasets/csv.py`
2. **Update `HorizynDataModule`** to use CSV and separate reaction files
3. **Update config validation** in `horizyn/config.py`
4. **Update `configs/sota.yaml`** with new paths
5. **Convert nanodata** from SQLite to CSV format
6. **Update `configs/nano.yaml`** with new paths
7. **Delete `sql.py`** and update imports
8. **Update tests** to use CSV fixtures
9. **Update documentation** (data/README.md, configs/README.md)

## Testing Strategy

1. Unit test `CSVDataset` in isolation
2. Run updated unit tests after each change
3. Smoke test with nanodata: `python train.py --config configs/nano.yaml --training.max_epochs 1`
4. Smoke test with SOTA data: `python train.py --config configs/sota.yaml --training.max_epochs 1`
5. Verify train/val reaction separation (no leakage)

## Decisions

1. **Nanodata**: Keep and convert to CSV format
2. **Column naming**: Keep `rs_id`/`pr_id` as-is
3. **FASTA file**: Keep and document that it corresponds to the protein embeddings

## Additional Tasks

### Convert Nanodata to CSV

Create CSV versions of the nanodata SQLite files:
- `data/nanodata/train_pairs.db` → `data/nanodata/train_pairs.csv`
- `data/nanodata/val_pairs.db` → `data/nanodata/val_pairs.csv`
- `data/nanodata/reactions.db` → split into `train_rxns.csv` + `val_rxns.csv`

Need to determine which reactions go to train vs val based on which pairs reference them.

### Update configs/nano.yaml

Point to the new CSV files with separate reaction paths.

### Update Tests

All tests using SQLite fixtures need updating:
- `tests/unit/test_sql_hdf5_datasets.py` → rename to `test_csv_hdf5_datasets.py`
- `tests/unit/test_datasets.py` - update fixtures
- `tests/unit/test_data_module.py` - update fixtures
- `tests/integration/test_swissprot.py` → update or remove
- Any other tests referencing `.db` files

## Estimated Effort

- `CSVDataset`: ~50 lines (simpler than SQLDataset)
- `data_module.py` changes: ~30 lines modified
- Config changes: ~10 lines
- YAML changes: ~5 lines
- Test updates: ~50 lines
- Documentation: ~100 lines

Total: ~250 lines of changes, mostly simplification
