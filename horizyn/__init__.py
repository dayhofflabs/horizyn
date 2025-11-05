"""
Horizyn: Contrastive Learning for Enzyme-Reaction Matching

Official implementation of the Horizyn SOTA model for matching enzymatic
reactions with their catalyzing proteins using contrastive learning.

License: CC BY-NC-SA 4.0
Copyright (c) 2025 Dayhoff Labs
"""

__version__ = "0.1.0"
__author__ = "Dayhoff Labs"
__license__ = "CC BY-NC-SA 4.0"

from horizyn.config import DotDict, load_config, parse_overrides
from horizyn.data_module import HorizynDataModule
from horizyn.lightning_module import HorizynLitModule
from horizyn.losses import FullBatchMLNCELoss, FullBatchNCELoss
from horizyn.metrics import create_retrieval_metrics
from horizyn.model import DualContrastiveModel, MLP

# Package exports
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "DotDict",
    "load_config",
    "parse_overrides",
    "HorizynDataModule",
    "HorizynLitModule",
    "DualContrastiveModel",
    "MLP",
    "FullBatchNCELoss",
    "FullBatchMLNCELoss",
    "create_retrieval_metrics",
]

