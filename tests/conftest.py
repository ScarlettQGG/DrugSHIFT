"""Shared fixtures for the DrugSHIFT test suite.

Small, deterministic, CPU-only tensors so the whole suite runs in seconds and
never needs real data or a GPU.
"""
import numpy as np
import pytest
import torch

MODALITIES = ["EPIC", "APMS", "Image", "Sequence"]
MOD_DIMS = {"EPIC": 12, "APMS": 16, "Image": 16, "Sequence": 20}
BATCH = 24


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def mod_dims():
    return dict(MOD_DIMS)


@pytest.fixture
def inputs():
    return {m: torch.randn(BATCH, d) for m, d in MOD_DIMS.items()}


@pytest.fixture
def masks():
    """All modalities present for every sample (float 1.0)."""
    return {m: torch.ones(BATCH) for m in MODALITIES}
