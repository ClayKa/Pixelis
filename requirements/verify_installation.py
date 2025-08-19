#!/usr/bin/env python3
"""
Verify the Pixelis environment installation.
"""

import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:30} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30} Not installed")
        return False
    except Exception as e:
        print(f"⚠ {package_name:30} Import error: {str(e)[:50]}")
        return False

def main():
    print("=" * 60)
    print("Pixelis Environment Verification")
    print("=" * 60)
    print()
    
    # Core Python info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Essential packages
    print("Essential Packages:")
    print("-" * 40)
    essential = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('accelerate', 'accelerate'),
        ('datasets', 'datasets'),
        ('peft', 'peft'),
        ('safetensors', 'safetensors'),
        ('huggingface-hub', 'huggingface_hub'),
        ('einops', 'einops'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
    ]
    
    essential_ok = all(check_package(name, imp) for name, imp in essential)
    print()
    
    # Training packages
    print("Training Packages:")
    print("-" * 40)
    training = [
        ('deepspeed', 'deepspeed'),
        ('ray', 'ray'),
        ('wandb', 'wandb'),
        ('tensorboard', 'tensorboard'),
        ('trl', 'trl'),
    ]
    
    for name, imp in training:
        check_package(name, imp)
    print()
    
    # Optional packages
    print("Optional/System-specific Packages:")
    print("-" * 40)
    optional = [
        ('vllm', 'vllm'),
        ('flash-attn', 'flash_attn'),
        ('xformers', 'xformers'),
        ('triton', 'triton'),
        ('liger-kernel', 'liger_kernel'),
    ]
    
    for name, imp in optional:
        check_package(name, imp)
    print()
    
    # Model-specific packages
    print("Model-specific Packages:")
    print("-" * 40)
    model_specific = [
        ('qwen-vl-utils', 'qwen_vl_utils'),
        ('sentencepiece', 'sentencepiece'),
        ('tiktoken', 'tiktoken'),
        ('tokenizers', 'tokenizers'),
    ]
    
    for name, imp in model_specific:
        check_package(name, imp)
    print()
    
    # Check PyTorch device availability
    print("PyTorch Device Configuration:")
    print("-" * 40)
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print(f"MPS built: {torch.backends.mps.is_built()}")
    except Exception as e:
        print(f"Error checking PyTorch devices: {e}")
    print()
    
    # Summary
    print("=" * 60)
    if essential_ok:
        print("✓ Essential packages are installed correctly!")
        print("  The environment is ready for basic operations.")
    else:
        print("✗ Some essential packages are missing.")
        print("  Please run install_dependencies.sh to complete setup.")
    
    print()
    print("Note: Optional packages may fail on certain systems")
    print("(e.g., vLLM on Mac, Flash Attention on CPU-only systems).")
    print("This is expected and won't affect core functionality.")
    print("=" * 60)

if __name__ == "__main__":
    main()