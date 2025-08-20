"""
Reproducibility utility module for ensuring deterministic behavior across all experiments.

This module provides functions to set global seeds for all relevant libraries and
enforce deterministic behavior in PyTorch and cuDNN.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_global_seed(seed: int) -> None:
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: The seed value to use for all random number generators.
    """
    # Set Python's hash seed for consistent hash ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set Python's built-in random module
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # If using CUDA, set seeds for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Important for multi-GPU setups
        
    # Set the seed for torch's default RNG state
    torch.set_rng_state(torch.manual_seed(seed).get_state())


def enable_deterministic_mode() -> None:
    """
    Enforces deterministic behavior in PyTorch/cuDNN.
    
    Note: This can have a performance cost, but is essential for reproducibility.
    """
    if torch.cuda.is_available():
        # Ensure deterministic behavior (reproducible results)
        torch.backends.cudnn.deterministic = True
        # Disable cuDNN auto-tuner (which can introduce randomness)
        torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms in PyTorch (for newer versions)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            # Some operations might not have deterministic implementations
            print(f"Warning: Could not enable fully deterministic algorithms: {e}")
            print("Falling back to partial determinism (cudnn only)")


def seed_worker(worker_id: int) -> None:
    """
    Initializes each DataLoader worker with a unique but predictable seed.
    
    This is essential for reproducible data loading and augmentation.
    Should be passed to DataLoader's worker_init_fn parameter.
    
    Args:
        worker_id: The ID of the DataLoader worker process.
    """
    # The worker seed is derived from the main process's initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_reproducible_dataloader_kwargs(seed: int, num_workers: int = 0) -> dict:
    """
    Returns kwargs for DataLoader that ensure reproducible behavior.
    
    Args:
        seed: The base seed to use for the generator.
        num_workers: Number of worker processes for data loading.
    
    Returns:
        Dictionary of kwargs to pass to DataLoader constructor.
    """
    # Create a generator object and seed it for reproducible shuffling/sampling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    kwargs = {
        'generator': generator,
        'worker_init_fn': seed_worker if num_workers > 0 else None,
    }
    
    return kwargs


def verify_determinism(run_function, *args, num_runs: int = 2, seed: int = 42, **kwargs) -> bool:
    """
    Utility function to verify that a function produces deterministic results.
    
    Args:
        run_function: The function to test for determinism.
        *args: Positional arguments to pass to the function.
        num_runs: Number of times to run the function.
        seed: The seed to use for each run.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        True if all runs produce identical results, False otherwise.
    """
    results = []
    
    for _ in range(num_runs):
        set_global_seed(seed)
        enable_deterministic_mode()
        result = run_function(*args, **kwargs)
        results.append(result)
    
    # Check if all results are identical
    for i in range(1, len(results)):
        if not torch.equal(results[0], results[i]) if isinstance(results[0], torch.Tensor) else results[0] != results[i]:
            return False
    
    return True


def get_system_info() -> dict:
    """
    Collects system information relevant for reproducibility.
    
    Returns:
        Dictionary containing system and library version information.
    """
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'numpy_version': np.__version__,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info