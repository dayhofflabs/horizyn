# Critical Validation Bugs - Part 2

## Summary

Two additional critical bugs found after fixing the per-pair vs per-query metrics bug:

1. **Bidirectional Reactions**: Missing reaction direction augmentation (forward + backward)
2. **Incomplete Screening Set**: Lookup table only contains training proteins, missing ~23K validation-only proteins

Both bugs make the reported metrics artificially low and incomparable to hatchery results.

---

## Bug 1: Missing Bidirectional Reactions

### The Issue

Hatchery trains and evaluates reactions in **both directions**:
- Forward: `reactants>>products` (key: `Rh_12345_f`)
- Backward: `products>>reactants` (key: `Rh_12345_r`)

This doubles the training data and ensures reversible reactions are learned bidirectionally.

### Current State

Horizyn only uses reactions as-is from the database (forward direction only).

### Impact

- **Training**: Only ~10,785 reactions instead of ~21,570 (missing 50% of training examples)
- **Validation**: Only evaluates forward direction, missing backward matches
- **Metrics**: Artificially low because model never saw reverse reactions

### Evidence from Hatchery

```yaml
# configs/horizyn/health_check.yaml line 77
apply_reaction_directions: True
```

```python
# src/data_modules/contrastive_data_module.py lines 2670-2708
def _augment_reaction_directions(self, ...):
    for key in query_embed_data.keys:
        smiles_f = query_embed_data[key]["reaction_smiles"]
        if reaction_dir[key] == "both" or reaction_dir[key] == "forward":
            keys.append(f"{key}_f")
            array_data.append(smiles_f)
        if reaction_dir[key] == "both" or reaction_dir[key] == "backward":
            smiles_r = ">>".join(reversed(smiles_f.split(">>")))  # Reverse
            keys.append(f"{key}_r")
            array_data.append(smiles_r)
```

### Fix Required

1. **Data Module**: Augment reactions during `_create_query_dataset()`:
   - For each reaction, create both `_f` (forward) and `_r` (backward) versions
   - Reverse SMILES by splitting on `>>` and swapping sides
   
2. **Pairs**: Update pair loading to include direction suffixes:
   - Load pairs as normal
   - Duplicate each pair for forward and backward
   - Append `_f` or `_r` to reaction_id

3. **Validation**: Score both directions and aggregate
   - Each unique reaction evaluates in both directions
   - Metrics computed over both forward and backward retrievals

---

## Bug 2: Incomplete Screening Set

### The Issue

The validation lookup table should contain **ALL proteins from both training AND validation sets**. Currently it only contains training proteins.

### Current State

```python
# data_module.py _setup_validation_data()
# Reuses self._target_data from training (only training proteins)
```

**Statistics**:
- Training-only proteins: ~181,945
- Validation-only proteins: ~23,363 ❌ **MISSING FROM LOOKUP**
- Shared proteins: ~10,824
- Total should be: ~216,132

### Impact

For the ~23,363 validation queries whose targets are validation-only proteins:
- **Hit rate: 0%** (their valid targets aren't even in the screening set!)
- This severely depresses all metrics

For example, if a reaction should retrieve `protein_X` but `protein_X` is only in validation pairs (not training), it will never rank high because it's not in the lookup table at all.

### Evidence

```bash
$ python -c "... check protein overlap ..."
Training proteins: 192769
Validation proteins: 34187
Union (should be screening set): 216132
Overlap: 10824  # Only these are currently retrievable!
```

### Fix Required

1. **Load ALL proteins** (not just training):
```python
# data_module.py
def _setup_validation_data(self):
    # Load validation pairs
    val_pairs = SQLDataset(...)
    
    # Get ALL unique protein IDs from both train and val
    all_protein_ids = set()
    
    # From training pairs
    train_pairs_data = SQLDataset(file_path=self.train_pairs_path, ...)
    for pair_key in train_pairs_data.keys:
        all_protein_ids.add(train_pairs_data[pair_key]['protein_id'])
    
    # From validation pairs  
    for pair_key in val_pairs.keys:
        all_protein_ids.add(val_pairs[pair_key]['protein_id'])
    
    # Load protein embeddings for ALL proteins (not just training)
    all_protein_ids = sorted(all_protein_ids)
    full_target_dataset = EmbedDataset(
        file_path=str(self.proteins_path),
        in_memory=True,
    )
    
    # Filter to only proteins we need
    # (But in practice, just load all ~500K proteins - they're needed)
```

2. **Update lookup table size** in `lightning_module.py`:
```python
def on_validation_epoch_start(self):
    # Should use ALL proteins, not just training
    datamodule = self.trainer.datamodule
    
    # Create full screening set (train + val proteins)
    all_target_ids = set(datamodule._target_data.keys) | set(datamodule._val_target_data.keys)
    
    self.num_targets = len(all_target_ids)
    # ... build lookup table for ALL targets
```

---

## Combined Impact

These bugs compound:

1. **Missing bidirectional**: Model never learns reverse reactions → poor performance
2. **Incomplete screening**: ~68% of validation queries (those with val-only proteins) automatically get 0% hit rate
3. **Per-pair metrics** (already fixed): Even when targets are in screening set, metrics were computed wrong

**Result**: Reported ~1% top-1 is the product of all three bugs. The true model performance is likely **much higher** after all fixes.

---

## Priority

**CRITICAL - Fix before any training runs**

These are data pipeline bugs, not model issues. The model has been training on incomplete/incorrect data.

## Next Steps

1. Implement bidirectional reaction augmentation
2. Fix screening set to include all train+val proteins  
3. Re-run training from scratch with corrected data
4. Expect top-1 hit rates of 30-40% (as in paper) after all fixes

---

## Files to Modify

1. `horizyn/data_module.py` - Add reaction direction augmentation and full screening set
2. `horizyn/lightning_module.py` - Update lookup table to use full screening set
3. Add tests to verify:
   - Reactions are doubled (forward + backward)
   - Screening set contains all train+val proteins
   - Validation queries can retrieve their targets

