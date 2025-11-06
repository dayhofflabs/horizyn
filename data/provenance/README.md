# Data Provenance

This directory contains DVC metadata files for tracking data provenance and versions. These files are used internally during development and will be **deleted before publication**.

## Purpose

DVC (Data Version Control) allows us to:
- Track large data files without storing them in git
- Version datasets across experiments
- Share data between team members via remote storage
- Ensure reproducibility by pinning exact data versions

## File Mapping

This table maps internal filenames (tracked by DVC) to the standardized names used in the published `horizyn` repository:

### Nanodata (Integration Tests)

| Internal Filename | Published Filename | Size | Count | Description |
|------------------|-------------------|------|-------|-------------|
| `pairs.db` | `train_pairs.db` + `val_pairs.db` | 16 KB | 12 pairs | Split by reaction_id (~80% train) |
| `rxns.db` | `reactions.db` | 12 KB | 12 reactions | Renamed, no modification |
| `prot_embeds.h5` | `proteins_t5_embeddings.h5` | 50 KB | 11 proteins | Renamed, no modification |

**Origin:** Subset of Rhea reactions and UniProt proteins from the warehouse `data/elixir/` pipeline  
**Created:** July 12, 2024  
**Authors:** Josh & Daniel  
**MD5 Hashes:**
- pairs.db: `65bf8babd37437a447fbebc7b354c5dc`
- rxns.db: `9ca1c4749d2ccaca9a1b60152f777b7f`
- prot_embeds.h5: `d38e72e71aed9bbd785fb712779d63a5`

### SwissProt (Full Dataset)

| Internal Filename | Published Filename | Size | Count | Description |
|------------------|-------------------|------|-------|-------------|
| `cain_pairs_swissprot.db` | `train_pairs.db` | 13.8 MB | 257,733 pairs | Training pairs: 192,769 proteins × 10,785 reactions |
| `abel_pairs_swissprot.db` | `val_pairs.db` | 2.25 MB | 36,433 pairs | Validation pairs: 34,187 proteins × 1,147 reactions |
| `eve_rxns.db` | `reactions.db` | 5.38 MB | 15,969 reactions | Rhea v131 reactions (includes 714 duplicate SMILES) |
| `eve_swissprots_t5.h5` | `proteins_t5_embeddings.h5` | 904 MB | 216,132 proteins | SwissProt v2023_05 ProtT5-XL embeddings |

**Origin:** EVE benchmark (Rhea v131 + UniProt 2023_05, SwissProt subset)  
**Created:** February-August 2024  
**Authors:** Daniel Martin-Alarcon, Joana (reactions)  
**Embedding Model:** ProtT5-XL (Rostlab/prot_t5_xl_half_uniref50-enc)  
**MD5 Hashes:**
- cain_pairs_swissprot.db: `0cf73b3ad6588fbef901a8fd40114709`
- abel_pairs_swissprot.db: `3f47f3eeb1a8c20afb63c14c73891401`
- eve_rxns.db: `168cb64ef90972d43738258681e4a634`
- eve_swissprots_t5.h5: `282cf3f6e7a502d98ece793d366e75e9`

## Internal Naming Convention

The internal names follow the laboratory's naming convention:
- **EVE**: Enzyme-substrate reaction benchmark dataset (queries)
- **Cain**: Training split of EVE reactions paired with SwissProt proteins
- **Abel**: Validation/test split of EVE reactions paired with SwissProt proteins
- **SwissProt**: Target protein database

These names are meaningful in the context of the larger `hatchery` repository but are abstracted away in the simplified `horizyn` repository for clarity.

## DVC Metadata Files

Each `.dvc` file contains:
- MD5 hash of the data file
- Size in bytes
- Remote storage path (if configured)
- DVC version metadata

Example structure:
```yaml
outs:
- md5: a1b2c3d4...
  size: 14680064
  path: cain_pairs_swissprot.db
```

## Data Preparation Script

The `split_pairs.py` script in this directory handles data preparation:

```bash
# Split and rename nanodata files
python data/provenance/split_pairs.py nanodata --data-dir data/nanodata

# Rename pre-sliced swissprot files  
python data/provenance/split_pairs.py swissprot --data-dir data/swissprot
```

This script:
- **Nanodata:** Splits `pairs.db` by reaction_id into train/val, renames all files
- **SwissProt:** Renames cain/abel/eve files to standardized names

## Preparing Data for Publication

Before publishing the `horizyn` repository:

1. **Run standardization scripts:**
   ```bash
   python data/provenance/split_pairs.py nanodata --data-dir data/nanodata
   python data/provenance/split_pairs.py swissprot --data-dir data/swissprot
   ```

2. **Upload to Zenodo:**
   - Package `data/swissprot/` contents (after renaming)
   - Generate checksums
   - Update `scripts/download_data.py` with Zenodo URL and checksums

3. **Delete this directory:**
   ```bash
   rm -rf data/provenance/
   ```

4. **Update `.gitignore`:**
   - Keep `data/nanodata/*` tracked in git (small files)
   - Ignore `data/swissprot/*` (downloaded via Zenodo)

## Internal Use Only

This directory and its contents should **never** be included in the public release. It exists solely for internal version tracking during development.

