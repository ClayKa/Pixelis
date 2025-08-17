"""
Core Models Module

Contains model-related utilities and PEFT integration for the Pixelis project.
"""

from .peft_model import (
    DynamicLoRAConfig,
    PEFTModelFactory,
    create_model_with_dynamic_lora
)

__all__ = [
    'DynamicLoRAConfig',
    'PEFTModelFactory', 
    'create_model_with_dynamic_lora'
]