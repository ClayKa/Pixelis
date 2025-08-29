"""
Data Generation Module for Pixelis CoTA synthesis.

This module provides the task generator registry and imports for all specialized
generators used in Chain-of-Thought-Action data synthesis.
"""

from .base_generator import BaseTaskGenerator
from .geometric_comparison import GeometricComparisonTaskGenerator
from .targeted_ocr import TargetedOCRTaskGenerator
from .spatiotemporal import SpatioTemporalTaskGenerator
from .zoom_in import ZoomInTaskGenerator
from .select_frame import SelectFrameTaskGenerator
from .detail_perception import DetailPerceptionTaskGenerator
from .trajectory_augmenter import TrajectoryAugmenter

# Task Generator Registry
# Maps generator class names (as strings in config) to actual class objects
TASK_GENERATOR_REGISTRY = {
    # Primary generators for new visual operations
    "GeometricComparisonTaskGenerator": GeometricComparisonTaskGenerator,
    "TargetedOCRTaskGenerator": TargetedOCRTaskGenerator,
    "SpatioTemporalTaskGenerator": SpatioTemporalTaskGenerator,
    "ZoomInTaskGenerator": ZoomInTaskGenerator,
    "SelectFrameTaskGenerator": SelectFrameTaskGenerator,
    "DetailPerceptionTaskGenerator": DetailPerceptionTaskGenerator,
    
    # Aliases for backward compatibility and config flexibility
    "geometric_comparison": GeometricComparisonTaskGenerator,
    "targeted_ocr": TargetedOCRTaskGenerator,
    "spatiotemporal": SpatioTemporalTaskGenerator,
    "zoom_in": ZoomInTaskGenerator,
    "select_frame": SelectFrameTaskGenerator,
    "detail_perception": DetailPerceptionTaskGenerator,
}

# Export all generators and utilities
__all__ = [
    'BaseTaskGenerator',
    'GeometricComparisonTaskGenerator',
    'TargetedOCRTaskGenerator',
    'SpatioTemporalTaskGenerator',
    'ZoomInTaskGenerator',
    'SelectFrameTaskGenerator',
    'DetailPerceptionTaskGenerator',
    'TrajectoryAugmenter',
    'TASK_GENERATOR_REGISTRY',
]

def get_generator_class(name: str):
    """
    Get a task generator class by name.
    
    Args:
        name: The name of the generator class (from config)
        
    Returns:
        The generator class object
        
    Raises:
        ValueError: If the generator name is not found in registry
    """
    if name not in TASK_GENERATOR_REGISTRY:
        available = ', '.join(TASK_GENERATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown task generator: '{name}'. "
            f"Available generators: {available}"
        )
    
    return TASK_GENERATOR_REGISTRY[name]