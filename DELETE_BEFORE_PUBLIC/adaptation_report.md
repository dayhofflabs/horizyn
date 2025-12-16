# Adaptation Report: SwissProt → SOTA Dataset Migration

This document summarizes all changes made to adapt the horizyn codebase from the old SwissProt SQLite format to the new SOTA CSV format.

## Summary

The migration successfully converted the horizyn data pipeline from SQLite-based storage (`.db` files) to CSV-based storage (`.csv` files), with explicit separation of training and validation reaction files. All tests pass after the migration.

---

## 1. New Dataset Class: `CSVDataset`

**File Created**: `horizyn/datasets/csv.py`

Replaced `SQLDataset` with a new `CSVDataset` class that:
- Uses Python's built-in `csv` module (no pandas dependency)
- Loads CSV data into memory for fast access
- Maintains the same interface as the old `SQLDataset`
- Supports key-based and integer-based access
- Supports column renaming via `rename_map`

```python
class CSVDataset(BaseDataset[str]):
    def __init__(
        self,
        file_path: str,
        key_column: str,           # Column to use as keys
        columns: Sequence[str] | str | None = None,
        rename_map: dict[str, str] | None = None,
        ...
    )
```

---

## 2. Data Module Updates

**File Modified**: `horizyn/data_module.py`

### Parameter Changes

| Old Parameter | New Parameter(s) |
|---------------|------------------|
| `reactions_path` | `train_reactions_path`, `test_reactions_path` |
| `proteins_path` | `protein_embeds_path` |

### Key Changes
- Now accepts separate paths for training and test reactions
- Uses `CSVDataset` instead of `SQLDataset` for all tabular data
- `_create_query_dataset()` now takes `reactions_path` as an argument
- Training uses `train_reactions_path`, test uses `test_reactions_path`

---

## 3. Configuration Updates

### `horizyn/config.py`

Updated `validate_config()` to require new data keys:
```python
required_data_keys = [
    "train_pairs_path",
    "test_pairs_path",
    "train_reactions_path",    # NEW
    "test_reactions_path",      # NEW
    "protein_embeds_path",     # RENAMED from proteins_path
]
```

### `configs/sota.yaml`

Updated data paths to use CSV format and SOTA directory:
```yaml
data:
  train_pairs_path: data/sota/train_pairs.csv
  test_pairs_path: data/sota/test_pairs.csv
  train_reactions_path: data/sota/train_rxns.csv
  test_reactions_path: data/sota/test_rxns.csv
  protein_embeds_path: data/sota/prots_t5.h5
```

### `configs/nano.yaml`

Updated to use the new CSV-based nanodata:
```yaml
data:
  train_pairs_path: data/nanodata/train_pairs.csv
  test_pairs_path: data/nanodata/test_pairs.csv
  train_reactions_path: data/nanodata/train_rxns.csv
  test_reactions_path: data/nanodata/test_rxns.csv
  protein_embeds_path: data/nanodata/prots_t5.h5
```

---

## 4. Data File Changes

### Nanodata Conversion

Converted SQLite files to CSV format:

| Old File | New File(s) |
|----------|-------------|
| `train_pairs.db` | `train_pairs.csv` |
| `test_pairs.db` | `test_pairs.csv` |
| `reactions.db` | `train_rxns.csv`, `test_rxns.csv` |
| `proteins_t5_embeddings.h5` | `prots_t5.h5` |

Old SQLite files moved to `DELETE_BEFORE_PUBLIC/nanodata_old/`.

### SOTA Data

Protein embeddings file:
- `prots_t5.h5` (consistent naming with SOTA dataset convention)

---

## 5. Module Exports

**File Modified**: `horizyn/datasets/__init__.py`

- Removed `SQLDataset` export
- Added `CSVDataset` export

---

## 6. Train Script Update

**File Modified**: `train.py`

Updated `HorizynDataModule` instantiation to pass new parameters:
```python
data_module = HorizynDataModule(
    train_reactions_path=config.data.train_reactions_path,
    test_reactions_path=config.data.test_reactions_path,
    protein_embeds_path=config.data.protein_embeds_path,
    ...
)
```

---

## 7. Documentation Updates

### `data/README.md`

- Updated to document CSV format for pairs and reactions
- Documented separate `train_rxns.csv` and `test_rxns.csv`
- Updated protein embeddings filename to `prots_t5.h5`
- Added documentation for optional `prots.fasta` file

### `configs/README.md`

- Updated configuration structure section
- Documented new `train_reactions_path`, `test_reactions_path`, `protein_embeds_path` keys

---

## 8. Test Updates

### Unit Tests

**`tests/unit/test_data_module.py`**
- Rewrote fixtures to create CSV files instead of SQLite
- Updated all `HorizynDataModule` instantiations with new parameter names
- Fixed references to internal attributes (`_query_data` → `_train_query_data`)

**`tests/unit/test_config.py`**
- Updated expected config values to use new paths and file extensions
- Changed assertions from `.db` to `.csv` file extensions
- Updated paths from `swissprot` to `sota`

**`tests/unit/test_standardization_config.py`**
- Rewrote fixtures to create CSV files
- Updated parameter names for `HorizynDataModule`

**`tests/unit/test_csv_hdf5_datasets.py`** (previously `test_sql_hdf5_datasets.py`)
- Already updated in prior work to test `CSVDataset`

### Integration Tests

**`tests/integration/test_smoke_config.py`**
- Updated to use `train_reactions_path` instead of `reactions_path`

**`tests/integration/test_smoke_nanodata.py`**
- Updated expected file names from `.db` to `.csv`
- Updated config path assertions

**`tests/integration/test_smoke_validation.py`**
- Updated to use `protein_embeds_path` instead of `proteins_path`

**`tests/integration/test_swissprot.py`**
- Updated to test SOTA dataset format (CSV files)
- Changed fixture to check for SOTA data directory
- Updated schema checks from SQLite to CSV

---

## 9. Files Deleted/Moved

| Action | File |
|--------|------|
| Deleted | `horizyn/datasets/sql.py` |
| Moved | `data/nanodata/*.db` → `DELETE_BEFORE_PUBLIC/nanodata_old/` |

---

## 10. Docstring Updates

Updated docstring examples in fingerprint datasets to reference `CSVDataset`:

- `horizyn/datasets/fingerprints/base.py`
- `horizyn/datasets/fingerprints/rdkit_plus.py`
- `horizyn/datasets/fingerprints/drfp.py`

---

## Test Results

All 452 tests pass after the migration:
- 427 passed (prior passing)
- 25 previously failing tests now pass

The migration is complete and the codebase now uses the SOTA CSV format exclusively.

---

## Breaking Changes

This migration introduces breaking changes. Users must:

1. **Update configuration files** to use new parameter names
2. **Convert any custom datasets** from SQLite to CSV format
3. **Split reactions** into separate training and validation files

---

## File Summary

| Category | Files Changed |
|----------|---------------|
| New files | 1 (`csv.py`) |
| Modified source | 5 (`data_module.py`, `config.py`, `train.py`, `__init__.py`, fingerprint docs) |
| Modified configs | 2 (`sota.yaml`, `nano.yaml`) |
| Modified docs | 2 (`data/README.md`, `configs/README.md`) |
| Modified tests | 7 (unit and integration tests) |
| Data files renamed | 2 (protein embeddings for nanodata and sota) |
| Data files created | 4 (CSV files for nanodata) |
| Deleted | 1 (`sql.py`) |
| Moved to archive | 4 (old SQLite nanodata files) |
