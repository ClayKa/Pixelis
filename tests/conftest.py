"""
Shared pytest fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Any

import pytest
import multiprocessing
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_wandb(mocker):
    """
    Auto-used fixture to completely mock the wandb library for all tests
    that might import it. Prevents any real network calls.

    This works by patching 'wandb' in the specific modules where it is imported and used.
    """
    # A list of all modules where 'import wandb' might occur.
    # We will attempt to patch 'wandb' in each of these locations.
    modules_to_patch = [
        'core.reproducibility.artifact_manager.wandb',
        'scripts.train.wandb',       # Assuming a unified train.py
        'scripts.train_rft.wandb',   # Assuming a specific rft script
        'scripts.train_sft.wandb',   # Assuming a specific sft script
        # Add any other module paths here if they also import wandb
    ]

    for module_path in modules_to_patch:
        try:
            # For each potential location, patch 'wandb' with a MagicMock
            mocker.patch(module_path, MagicMock())
        except (ModuleNotFoundError, AttributeError):
            # This is expected and safe. It just means the test currently being
            # run doesn't involve a module that imports wandb from that path.
            # For example, when testing test_voting.py, it won't find 'scripts.train.wandb'.
            pass

@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_start_method():
    """
    Set the multiprocessing start method to 'spawn' for all tests.

    This is crucial to prevent deadlocks on Linux/macOS when using libraries
    that have their own internal thread pools (like numpy with BLAS/LAPACK)
    in a forked process, especially within a pytest environment.
    'spawn' creates a clean new process, avoiding state inheritance issues.
    """
    # We only need to do this if the default is 'fork'
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config() -> dict:
    """Create a mock configuration dictionary."""
    return {
        "model": {
            "model_name": "test_model",
            "use_lora": True,
            "lora_r": 8,
        },
        "training": {
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
        },
        "experiment": {
            "experiment_name": "test_experiment",
            "use_wandb": False,
            "track_artifacts": True,
        },
        "system": {
            "device": "cpu",
            "num_gpus": 0,
        },
    }


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set environment variables for testing."""
    os.environ["PIXELIS_OFFLINE_MODE"] = "true"
    os.environ["PIXELIS_TEST_MODE"] = "true"
    yield
    # Cleanup
    os.environ.pop("PIXELIS_TEST_MODE", None)


@pytest.fixture
def mock_artifact_data() -> dict:
    """Create mock artifact data for testing."""
    return {
        "name": "test_artifact",
        "version": "v1",
        "type": "model",
        "metadata": {
            "created_at": "2024-01-01T00:00:00",
            "tags": ["test", "mock"],
        },
        "content": {"data": "mock_data"},
    }


@pytest.fixture
def mock_experience() -> dict:
    """Create mock experience data for TTRL testing."""
    return {
        "id": "exp_001",
        "input": "test input",
        "output": "test output",
        "reward": 0.8,
        "confidence": 0.9,
        "timestamp": "2024-01-01T00:00:00",
    }


@pytest.fixture
def cleanup_artifacts():
    """Clean up test artifacts after tests."""
    yield
    # Clean up any test artifacts created
    test_dirs = ["./test_runs", "./test_artifacts", "./test_checkpoints"]
    for dir_path in test_dirs:
        if Path(dir_path).exists():
            import shutil
            shutil.rmtree(dir_path)


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "reproducibility: marks reproducibility system tests"
    )


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and environment."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)