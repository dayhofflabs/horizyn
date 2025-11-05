"""Pytest configuration and shared fixtures for Horizyn tests."""

import pytest
import torch
import numpy as np


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (runs by default)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test requiring full SwissProt data (slow, skipped by default)"
    )


@pytest.fixture
def device():
    """Get the device to use for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def sample_embeddings(device):
    """Generate sample embeddings for testing."""
    batch_size = 8
    embed_dim = 512
    query_embeds = torch.randn(batch_size, embed_dim, device=device)
    target_embeds = torch.randn(batch_size, embed_dim, device=device)
    # Normalize
    query_embeds = torch.nn.functional.normalize(query_embeds, dim=1)
    target_embeds = torch.nn.functional.normalize(target_embeds, dim=1)
    return query_embeds, target_embeds


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing chemistry utilities."""
    return {
        "simple_molecule": "CCO",  # Ethanol
        "aromatic": "c1ccccc1",  # Benzene
        "charged": "[NH4+]",  # Ammonium
        "reaction": "CCO.O=O>>CC(=O)O.[H][H]",  # Ethanol oxidation (simplified)
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": {
            "query_encoder": "MLP",
            "target_encoder": "MLP",
            "query_encoder_kwargs": {
                "input_dim": 2048,
                "num_layers": 2,
                "widths": 4096,
                "output_dim": 512,
                "normalise_output": True,
            },
            "target_encoder_kwargs": {
                "input_dim": 1024,
                "num_layers": 2,
                "widths": 4096,
                "output_dim": 512,
                "normalise_output": True,
            },
        },
        "training": {
            "optimiser": "adamw",
            "optimiser_kwargs": {"lr": 1e-4, "weight_decay": 1e-2},
            "loss_fn": "FullBatchMLNCELoss",
            "loss_fn_kwargs": {
                "beta": 10.0,
                "learn_beta": False,
                "beta_min": 0.01,
                "beta_max": 100.0,
            },
            "metric_kwargs": {"top_k": [1, 10, 100, 1000]},
        },
        "data": {
            "train_batch_size": 16,
            "val_batch_size": 8,
            "num_workers": 0,
        },
    }

