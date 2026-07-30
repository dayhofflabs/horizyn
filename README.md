# Horizyn: Contrastive Learning for Enzyme-Reaction Matching

```ascii
    __  __           _                  
   / / / /___  _____(_)___  __  ______  
  / /_/ / __ \/ ___/ /_  / / / / / __ \ 
 / __  / /_/ / /  / / / /_/ /_/ / / / / 
/_/ /_/\____/_/  /_/ /___/\__, /_/ /_/  
                         /____/                               
```

Official implementation of the [Horizyn model](https://www.pnas.org/doi/10.1073/pnas.2520070123) for matching reactions and enzymes.

Two ways to use Horizyn:

- **This repository** — train the model, reproduce the paper's evaluation, and run predictions locally against ~216K bundled protein embeddings. Requires a GPU.
- **The hosted Horizyn API** — search 6.33M enzymes with annotations and filtering, no GPU required. Free API keys are issued by email; see the **[Horizyn API Guide](horizyn-api-guide.md)**.

## Overview

Horizyn is a dual-encoder contrastive learning model that learns to match enzymatic reactions with their catalyzing proteins. The model uses:

- **Reaction Encoder**: Concatenated RDKit+ (structural) and DRFP fingerprints → MLP
- **Protein Encoder**: Pre-computed T5 embeddings → MLP
- **Loss**: Maximum Likelihood Noise Contrastive Estimation (MLNCE)
- **Embeddings**: 512-dimensional normalized outputs for both encoders

## Quick Start

### Installation

Install dependencies with UV (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Download Dataset

Download the training dataset and protein embeddings (~1GB). This provides the ~216K pre-computed ProtT5-XL protein embeddings needed by both evaluation and prediction:

```bash
uv run python scripts/download_training_data.py
```

### Download Pre-trained Checkpoints

Download both official checkpoints (~201MB each, ~402MB total):

```bash
uv run python scripts/download_checkpoint.py
```

This downloads two checkpoints:
- **`horizyn_v1_0_dev.ckpt`** — trained on the train split only (paper-faithful); use for evaluation
- **`horizyn_v1_0_inf.ckpt`** — trained on full data; use for prediction

To download only one: `uv run python scripts/download_checkpoint.py --only dev`

### Evaluate the Model

Evaluate the dev checkpoint on the test set (requires both the dataset and dev checkpoint above):

```bash
uv run python scripts/evaluate.py
```

The evaluation script computes retrieval metrics (Top-K hit rates, MRR) on the held-out test set. Expected: top-1 ≈ 32.4%.

### Query with a Reaction

Find the most likely catalyzing enzymes for a reaction SMILES, using the inference checkpoint against the bundled ~216K protein embeddings (requires both the dataset and inf checkpoint above):

```bash
# Example: ADP + H2O -> AMP + phosphate
uv run python scripts/predict.py "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](OP(=O)([O-])[O-])[C@H]1O.[H]O[H]>>NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])O" --top-k 10
```

Use `--bidirectional` to score both forward and reverse reaction directions (averaged):

```bash
uv run python scripts/predict.py "SMILES>>SMILES" --bidirectional --top-k 20
```

### Train the Model

Train the SOTA model from scratch (requires ~16GB RAM, single GPU with 16GB+ VRAM):

```bash
uv run python train.py --config configs/sota.yaml
```

### Run Tests

```bash
uv run pytest
```

## Query the Hosted API

If you want to screen a reaction against far more enzymes than the bundled set —
6.33M proteins, with names, organisms, EC numbers, cofactors, literature, and
filtering — use the hosted Horizyn API instead. It needs no GPU and no local
data download.

Getting a key takes one email round-trip:

```bash
# 1. Request a verification code
curl -X POST https://api.horizyn1.dayhofflabs.com/keys/request \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'

# 2. Confirm with the code from your inbox to receive your key
curl -X POST https://api.horizyn1.dayhofflabs.com/keys/confirm \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "code": "123456", "name": "my-project"}'
```

Then query a reaction. Use balanced reactions — the example below is GabT
transamination of 2-oxoglutarate using 6-aminohexanoate:

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/query/reaction \
  -H "Authorization: Bearer $HORIZYN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "[NH3+]CCCCCC([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[O-]C(=O)CCCCC=O.[NH3+][C@@H](CCC([O-])=O)C([O-])=O"}'
```

Full documentation — every endpoint, request options, filtering, clustering,
rate limits, and Python examples — is in the
**[Horizyn API Guide](horizyn-api-guide.md)**. Interactive OpenAPI docs are at
[api.horizyn1.dayhofflabs.com/docs](https://api.horizyn1.dayhofflabs.com/docs).

## Hardware Requirements

- **RAM**: 8GB minimum (4GB for data loaded entirely in memory)
- **GPU**: Single NVIDIA GPU with 16GB+ VRAM (e.g., T4, A10G, V100)
- **Disk**: 20GB free space for dataset and checkpoints
- **Platform**: Linux x86_64 with CUDA 12.1

## Project Structure

```
horizyn/
├── horizyn/                    # Main package
│   ├── model.py               # DualContrastiveModel, MLP
│   ├── lightning_module.py    # Training loop logic
│   ├── data_module.py         # Data loading orchestration
│   ├── config.py              # Configuration management
│   ├── losses.py              # MLNCE loss function
│   ├── metrics.py             # Retrieval metrics
│   ├── datasets/              # Dataset classes
│   │   ├── base.py           # Base dataset abstractions
│   │   ├── collection.py     # Dataset composition utilities
│   │   ├── csv.py            # CSV dataset loader
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
│   ├── download_training_data.py  # Training data download
│   ├── download_checkpoint.py     # Pre-trained checkpoint download
│   ├── predict.py                 # Query model with a reaction SMILES
│   └── evaluate.py                # Model evaluation
├── train.py                   # Main training entry point
└── tests/                     # Test suite
```

## Documentation

- **[Horizyn API Guide](horizyn-api-guide.md)** — using the hosted API: get a key, call every endpoint, filter and cluster results
- **[Horizyn User Manual](horizyn-user-manual.md)** — in-depth reference for this codebase: architecture, data pipeline, training, configuration, and testing

## Model Architecture

The Horizyn model uses a dual-encoder architecture:

- **Query Encoder** (Reactions): 2048-dim fingerprints → 4096-dim hidden → 512-dim embedding
- **Target Encoder** (Proteins): 1024-dim T5 embeddings → 4096-dim hidden → 512-dim embedding
- **Loss Function**: MLNCE with temperature parameter (β=10.0)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{horizyn2026,
  title = {Dual-encoder contrastive learning accelerates enzyme discovery},
  author = {Rocks, Jason W. and Truong, Dat P. and Rappoport, Dmitrij and Maddrell-Mander, Sam and Martin-Alarcon, Daniel A. and Lee, Toni and Crossan, Steve and Goldford, Joshua E.},
  journal = {Proc. Natl. Acad. Sci. U.S.A.},
  volume = {123},
  number = {12},
  pages = {e2520070123},
  year = {2026},
  doi = {10.1073/pnas.2520070123},
}
```

## License

This code is licensed under **PolyForm Noncommercial License 1.0.0**.

- ✅ **Noncommercial use**: Free to use and modify for noncommercial purposes
- ✅ **Research and education**: Permitted for academic, research, and educational purposes
- ❌ **Commercial use**: Prohibited without separate commercial licensing
- 📧 **Commercial inquiries**: [info@dayhofflabs.com](mailto:info@dayhofflabs.com)

See [LICENSE](LICENSE) for full terms or visit [https://polyformproject.org/licenses/noncommercial/1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)

## Contributing

This repository is maintained by Dayhoff Labs. For questions or issues, please open a GitHub issue.
