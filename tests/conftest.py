"""
Shared pytest configuration and fixtures for Triton-Augment tests.

This file contains common setup, fixtures, and utilities used across all test files.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# Skip all tests if CUDA or torchvision not available
def pytest_collection_modifyitems(config, items):
    """Add skip markers to all tests if requirements not met."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_tv = pytest.mark.skip(reason="torchvision not available")
    
    for item in items:
        if not torch.cuda.is_available():
            item.add_marker(skip_cuda)
        if not TORCHVISION_AVAILABLE:
            item.add_marker(skip_tv)


@pytest.fixture
def device():
    """Return CUDA device."""
    return torch.device('cuda')


@pytest.fixture
def imagenet_mean_std():
    """Return ImageNet normalization parameters."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std

