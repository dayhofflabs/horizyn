# Horizyn: Contrastive Learning for Enzyme-Reaction Matching

```ascii
    __  __           _                  
   / / / /___  _____(_)___  __  ______  
  / /_/ / __ \/ ___/ /_  / / / / / __ \ 
 / __  / /_/ / /  / / / /_/ /_/ / / / / 
/_/ /_/\____/_/  /_/ /___/\__, /_/ /_/  
                         /____/                               
```

Official implementation of the [Horizyn  model](https://www.pnas.org/doi/10.1073/pnas.2520070123) for matching reactions and enzymes. Register [here](https://horizyn.dayhofflabs.com/) to query Horizyn interactively or via the API.

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

Download the SOTA training dataset (~1GB):

```bash
uv run python scripts/download_training_data.py
```

### Download Pre-trained Checkpoint

Download the official pre-trained checkpoint (~201MB):

```bash
uv run python scripts/download_checkpoint.py
```

### Evaluate the Model

Evaluate the pre-trained checkpoint on the test set:

```bash
uv run python scripts/evaluate.py
```

The evaluation script computes retrieval metrics (Top-K hit rates, MRR) on the held-out test set. Checkpoints are also saved during training to the `checkpoints/` directory.

### Query with a Reaction

Find the most likely catalyzing enzymes for a reaction SMILES, searching the bundled set of ~216K proteins with pre-computed ProtT5-XL embeddings:

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
