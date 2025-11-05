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

> **Note**: This repository is currently under development. Full functionality will be available upon publication of the accompanying research paper.

## Overview

Horizyn is a dual-encoder contrastive learning model that learns to match enzymatic reactions with their catalyzing proteins. The model uses:

- **Reaction Encoder**: Concatenated RDKit+ (structural) and DRFP fingerprints → MLP
- **Protein Encoder**: Pre-computed T5 embeddings → MLP
- **Loss**: Multi-Label Noise Contrastive Estimation (MLNCE)
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

Download the pre-split SwissProt dataset (~15GB):

```bash
python scripts/download_data.py --output_dir data/
```

### Train the Model

Train the SOTA model (requires ~16GB RAM, single GPU with 16GB+ VRAM):

```bash
python train.py --config configs/sota.yaml
```

## Hardware Requirements

- **RAM**: 16GB minimum (15GB for data loaded entirely in memory)
- **GPU**: Single NVIDIA GPU with 16GB+ VRAM (e.g., T4, A10G, V100)
- **Disk**: 20GB free space for dataset and checkpoints
- **Platform**: Linux x86_64 with CUDA 12.1

## Project Structure

```
horizyn/
├── horizyn/              # Main package
│   ├── model.py         # DualContrastiveModel, MLP
│   ├── losses.py        # MLNCE loss
│   ├── metrics.py       # Retrieval metrics
│   ├── data_module.py   # Lightning DataModule
│   ├── lightning_module.py  # Training loop
│   ├── config.py        # Configuration system
│   ├── datasets/        # Dataset classes
│   ├── chemistry/       # RDKit utilities
│   └── utils/           # Utilities
├── configs/
│   └── sota.yaml        # SOTA configuration
├── data/                # Dataset directory
├── scripts/             # Helper scripts
├── tests/               # Test suite
├── train.py             # Training entry point
└── pyproject.toml       # Dependencies
```

## Model Architecture

The Horizyn model uses a dual-encoder architecture:

- **Query Encoder** (Reactions): 2048-dim fingerprints → 4096-dim hidden → 512-dim embedding
- **Target Encoder** (Proteins): 1024-dim T5 embeddings → 4096-dim hidden → 512-dim embedding
- **Loss Function**: Multi-Label NCE with temperature parameter (β=10.0)

## Development

Run tests:

```bash
pytest tests/ -v
```

Format code:

```bash
black horizyn/ tests/
isort horizyn/ tests/
```

Check linting:

```bash
flake8 horizyn/ tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{horizyn2025,
  title={Horizyn: Contrastive Learning for Enzyme-Reaction Matching},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This code is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0).

- ✅ **Academic and research use**: Free to use and modify
- ❌ **Commercial use**: Prohibited without separate licensing
- 📤 **Sharing**: Derivatives must use the same license
- 📧 **Commercial inquiries**: contact@dayhofflabs.com

See [LICENSE](LICENSE) for full terms.

## Contributing

This repository is maintained by Dayhoff Labs. For questions or issues, please open a GitHub issue.

## Acknowledgments

- RDKit for molecular fingerprinting
- DRFP for differential reaction fingerprints
- PyTorch Lightning for training infrastructure
