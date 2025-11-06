# Critical Training & Validation Bugs

## Summary

Three critical bugs found that explain why metrics are ~1% instead of the expected 30-40%:

1. **Bidirectional Reactions**: Missing reaction direction augmentation (forward + backward)
   - Impact: Training on only 50% of available examples
   
2. **Incomplete Screening Set**: Lookup table only contains training proteins, missing ~23K validation-only proteins
   - Impact: 68% of validation queries have 0% hit rate (their targets aren't in the lookup table)
   
3. **Wrong Fingerprint Radius**: Training code uses `radius=2` instead of `radius=3` (API is correct)
   - Impact: Models trained with horizyn are incompatible with the API
   - Source: Horizyn distilled from hatchery but got this parameter wrong

All three bugs must be fixed together before retraining. The combination explains the severe metric degradation.

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

### Bug 1 & 2: Bidirectional Reactions + Screening Set
1. `horizyn/data_module.py` - Add reaction direction augmentation and full screening set
2. `horizyn/lightning_module.py` - Update lookup table to use full screening set
3. Add tests to verify:
   - Reactions are doubled (forward + backward)
   - Screening set contains all train+val proteins
   - Validation queries can retrieve their targets

### Bug 3: Morgan Fingerprint Radius
1. `horizyn/horizyn/datasets/fingerprints/rdkit_plus.py` - Change `radius=2` to `radius=3` (line 150)
2. `horizyn/tests/unit/test_fingerprints.py` - **Reviewed: No changes needed** (tests only check shapes/properties, not specific values)
3. `horizyn/tests/unit/test_datasets.py` - Check for any fingerprint value assertions
4. `horizyn/tests/integration/test_smoke_*.py` - Check for any fingerprint validation
5. Add cross-validation test: Generate same SMILES in horizyn and API, verify identical fingerprints

---

## Bug 3: Training Code Uses Wrong Morgan Fingerprint Radius

### The Issue

The **Horizyn training code uses `radius=2`** for Morgan fingerprints, but the **API (which is based on the original hatchery codebase) correctly uses `radius=3`**. Since the horizyn repo is a distillation of the older codebase, it inherited the wrong parameter.

### Current State

**Training Code** (`horizyn/horizyn/datasets/fingerprints/rdkit_plus.py` line 150):
```python
# Morgan fingerprints with radius=2 (ECFP4) ❌ WRONG
self._fp_gen = generator_func(
    radius=2,  # Should be 3!
    fpSize=self.fp_size,
    includeChirality=self.use_chirality,
)
```

**API Code** (`horizyn-api/src/horizyn/fingerprints/horizyn1.py` line 31-39):
```python
MORGAN_CONFIG = MorganFingerprintConfig(
    fp_size=512,
    radius=3,  # ✅ CORRECT (from original hatchery)
    count_simulation=False,
    include_chirality=True,
    ...
)
```

### Impact

Morgan fingerprint radius determines the size of the atom neighborhood considered when generating fingerprints:
- **radius=2** captures atoms up to 2 bonds away (ECFP4 / circular diameter 4)
- **radius=3** captures atoms up to 3 bonds away (ECFP6 / circular diameter 6)

**These produce completely different fingerprints**, meaning:
1. The training code generates fingerprints that don't match the original model
2. Any model trained with the current horizyn code will be incompatible with the API
3. The existing checkpoint (`isp7e77b`) was trained with radius=3 (from hatchery)
4. Retrieval quality will be severely degraded with wrong fingerprints

### Evidence

**Training configuration (WRONG):**
- File: `horizyn/horizyn/datasets/fingerprints/rdkit_plus.py`
- Line 150: `radius=2` (hardcoded, incorrect)
- Used for all current horizyn training runs

**API configuration (CORRECT):**
- File: `horizyn-api/src/horizyn/fingerprints/horizyn1.py`
- Line 33: `radius=3` in `MORGAN_CONFIG`
- Based on original hatchery codebase (source of truth)

**Hatchery original (CORRECT):**
- File: `hatchery/src/datasets/fingerprint_data.py` line 646
- Class: `RDKitplusFingerprintDataset`
- Default parameter: `radius: int = 3`
- All hatchery configs (health_check.yaml, paper configs, etc.) use this default
- The existing checkpoint `isp7e77b` was trained with radius=3 from hatchery
- Horizyn was distilled from hatchery but incorrectly hardcoded radius=2

**Other settings match:**
- Standardization: Both use `hypervalent=True, remove_hs=True, kekulize=False, uncharge=True, metals=True`
- DRFP: Both use `radius=3, rings=True, vec_dim=1024`
- Fingerprint type: Both use Morgan ("struct" mode for training, equivalent in API)
- Chirality: Both use `include_chirality=True`

### Fix Required

**1. Update training code to match API (hatchery source):**

```python
# horizyn/horizyn/datasets/fingerprints/rdkit_plus.py line 149-153
if self.mol_fp_type == "morgan":
    # Morgan fingerprints (ECFP-like)
    self._fp_gen = generator_func(
        radius=3,  # ✅ Changed from 2 to match hatchery/API
        fpSize=self.fp_size,
        includeChirality=self.use_chirality,
    )
```

**2. Update unit tests that may hardcode radius=2 expectations:**

Files to check:
- `horizyn/tests/unit/test_fingerprints.py` - May have tests expecting radius=2 fingerprints
- `horizyn/tests/unit/test_datasets.py` - May test RDKitPlusFingerprintDataset with radius=2
- `horizyn/tests/integration/test_smoke_*.py` - Integration tests may validate fingerprint outputs

Search for:
- Tests that validate specific fingerprint values
- Tests that compare fingerprint outputs against expected values
- Mock or fixture fingerprints that assume radius=2

**3. Verification steps:**

After fixing:
1. Run all unit tests and update any that fail due to changed fingerprints
2. Generate fingerprints from same SMILES in both training code and API
3. Verify they produce identical outputs
4. Re-train model from scratch with corrected fingerprints
5. Verify model checkpoint is compatible with API

### Validation

After fixing, verify:
1. Training code generates identical fingerprints to API for same SMILES
2. Unit tests pass with updated fingerprint expectations
3. Model trained with fixed code is compatible with API
4. Retrieval metrics improve to expected levels (30-40% top-1) after fixing all three bugs

---

## Priority

**CRITICAL - All three bugs must be fixed together**

These bugs interact:
1. **Bug 1 (Bidirectional)**: Missing 50% of training examples
2. **Bug 2 (Screening set)**: Missing 68% of validation targets
3. **Bug 3 (Fingerprints)**: API produces wrong fingerprints for the trained model

The combination explains why reported metrics are ~1% instead of the expected 30-40%.

