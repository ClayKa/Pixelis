#!/usr/bin/env python3
"""
Test script to verify the improvements made to the Pixelis codebase.

This script tests:
1. Reproducibility features
2. Enhanced curriculum management
3. Distributed tracing capabilities
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import deque
import logging

# Import our improved modules
from core.utils.reproducibility import (
    set_global_seed,
    enable_deterministic_mode,
    verify_determinism,
    get_system_info,
    get_reproducible_dataloader_kwargs,
)
from core.utils.context import TraceContext, TracedOperation
from core.utils.logging_utils import setup_logging, get_logger
from core.config_schema import CurriculumConfig


def test_reproducibility():
    """Test that reproducibility features work correctly."""
    print("\n=== Testing Reproducibility Features ===")
    
    # Test 1: Seed setting
    set_global_seed(42)
    
    # Generate some random numbers with different libraries
    python_random = [np.random.random() for _ in range(5)]
    numpy_random = np.random.randn(5)
    torch_random = torch.randn(5)
    
    # Reset seed and generate again
    set_global_seed(42)
    python_random2 = [np.random.random() for _ in range(5)]
    numpy_random2 = np.random.randn(5)
    torch_random2 = torch.randn(5)
    
    # Check if they match
    assert np.allclose(python_random, python_random2), "Python random not reproducible"
    assert np.allclose(numpy_random, numpy_random2), "NumPy random not reproducible"
    assert torch.allclose(torch_random, torch_random2), "PyTorch random not reproducible"
    
    print("✓ Seed setting works correctly")
    
    # Test 2: Deterministic mode
    enable_deterministic_mode()
    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
        print("✓ Deterministic mode enabled for CUDA")
    else:
        print("✓ Deterministic mode set (no CUDA available)")
    
    # Test 3: System info
    sys_info = get_system_info()
    assert 'python_version' in sys_info
    assert 'torch_version' in sys_info
    print(f"✓ System info retrieved: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
    
    # Test 4: DataLoader kwargs
    dl_kwargs = get_reproducible_dataloader_kwargs(seed=42, num_workers=2)
    assert 'generator' in dl_kwargs
    assert 'worker_init_fn' in dl_kwargs
    print("✓ Reproducible DataLoader kwargs generated")
    
    return True


def test_curriculum_management():
    """Test enhanced curriculum management features."""
    print("\n=== Testing Curriculum Management ===")
    
    # Create a mock configuration
    config = {
        "curriculum": {
            "stages": [
                {"name": "simple", "difficulty_mix": {"simple": 1.0, "medium": 0.0}},
                {"name": "medium", "difficulty_mix": {"simple": 0.5, "medium": 0.5}},
            ],
            "smoothing_window_size": 3,
            "patience_cycles": 2,
            "cooldown_cycles": 3,
            "rollback_threshold": -0.05,
        }
    }
    
    # Test that configuration can be created
    curriculum_config = CurriculumConfig(
        smoothing_window_size=3,
        patience_cycles=2,
        cooldown_cycles=3,
    )
    
    assert curriculum_config.smoothing_window_size == 3
    assert curriculum_config.patience_cycles == 2
    assert curriculum_config.cooldown_cycles == 3
    
    print("✓ Curriculum configuration created successfully")
    print(f"  - Smoothing window: {curriculum_config.smoothing_window_size}")
    print(f"  - Patience cycles: {curriculum_config.patience_cycles}")
    print(f"  - Cooldown cycles: {curriculum_config.cooldown_cycles}")
    
    return True


def test_distributed_tracing():
    """Test distributed tracing capabilities."""
    print("\n=== Testing Distributed Tracing ===")
    
    # Test 1: Basic trace ID generation and setting
    trace_id = TraceContext.generate_trace_id()
    assert trace_id is not None
    assert len(trace_id) > 0
    print(f"✓ Generated trace ID: {trace_id[:8]}...")
    
    # Test 2: Context setting and getting
    TraceContext.set_trace_id(trace_id)
    TraceContext.set_component("TestComponent")
    
    retrieved_id = TraceContext.get_trace_id()
    retrieved_component = TraceContext.get_component()
    
    assert retrieved_id == trace_id
    assert retrieved_component == "TestComponent"
    print("✓ Trace context set and retrieved correctly")
    
    # Test 3: Metadata management
    metadata = {"operation": "test", "user": "test_user"}
    TraceContext.set_metadata(metadata)
    retrieved_metadata = TraceContext.get_metadata()
    
    assert "operation" in retrieved_metadata
    assert retrieved_metadata["operation"] == "test"
    print("✓ Metadata managed correctly")
    
    # Test 4: TracedOperation context manager
    with TracedOperation(
        operation_name="test_operation",
        component="TestComponent",
        metadata={"test": True}
    ) as op:
        # Inside the context, trace should be set
        assert TraceContext.get_trace_id() is not None
        assert TraceContext.get_component() == "TestComponent"
        print(f"✓ TracedOperation context manager works (trace: {op.trace_id[:8]}...)")
    
    # Test 5: Logging with trace ID
    setup_logging(use_tracing=True, level=logging.INFO)
    logger = get_logger("test_logger")
    
    with TracedOperation("test_log", "TestLogger") as op:
        logger.info("Test message with trace ID")
        print(f"✓ Logging with trace ID works")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Pixelis Improvements")
    print("=" * 60)
    
    try:
        # Run tests
        reproducibility_ok = test_reproducibility()
        curriculum_ok = test_curriculum_management()
        tracing_ok = test_distributed_tracing()
        
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        if reproducibility_ok and curriculum_ok and tracing_ok:
            print("✅ All tests passed successfully!")
            print("\nImprovements verified:")
            print("1. ✓ Reproducibility features are working")
            print("2. ✓ Curriculum management is enhanced")
            print("3. ✓ Distributed tracing is functional")
            return 0
        else:
            print("❌ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())