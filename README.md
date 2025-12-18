# Horizyn: Contrastive Learning for Enzyme-Reaction Matching

```ascii
    __  __           _                  
   / / / /___  _____(_)___  __  ______  
  / /_/ / __ \/ ___/ /_  / / / / / __ \ 
 / __  / /_/ / /  / / / /_/ /_/ / / / / 
/_/ /_/\____/_/  /_/ /___/\__, /_/ /_/  
                         /____/                               
```

Official implementation of the Horizyn SOTA model for contrastive learning between enzymatic reactions and proteins.

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

Download the SOTA dataset (~1GB):

```bash
python scripts/download_data.py
```

### Train the Model

Train the SOTA model (requires ~16GB RAM, single GPU with 16GB+ VRAM):

```bash
python train.py --config configs/sota.yaml
```

### Evaluate the Model

Evaluate a trained model checkpoint on the test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/epoch=99-step=XXXXX.ckpt
```

The evaluation script computes retrieval metrics (Top-K hit rates, MRR) on the held-out test set. Checkpoints are saved during training to the `checkpoints/` directory.

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
│   └── download_data.py      # Dataset download
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
@article{horizyn2025,
  title = {Dual-encoder contrastive learning accelerates enzyme discovery},
  author = {Rocks, Jason W. and Truong, Dat P. and Rappoport, Dmitrij and Maddrell-Mander, Sam and Martin-Alarcon, Daniel A. and Lee, Toni and Crossan, Steve and Goldford, Joshua E.},
  journal = {bioRxiv}
  year = {2025},
  doi = {10.1101/2025.08.21.671639},
}
```

## License

This code is licensed under **PolyForm Noncommercial License 1.0.0**.

- ✅ **Noncommercial use**: Free to use and modify for noncommercial purposes
- ✅ **Research and education**: Permitted for academic, research, and educational purposes
- ❌ **Commercial use**: Prohibited without separate commercial licensing
- 📧 **Commercial inquiries**: contact@dayhofflabs.com

See [LICENSE](LICENSE) for full terms or visit https://polyformproject.org/licenses/noncommercial/1.0.0

## Contributing

This repository is maintained by Dayhoff Labs. For questions or issues, please open a GitHub issue.
