"""
Shared pytest fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Any

import pytest

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