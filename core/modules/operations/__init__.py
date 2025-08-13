"""
Visual Operations Module

This module contains all visual operations that can be executed through
the Visual Operation Registry. Each operation is self-contained and
automatically registers itself with the global registry upon import.
"""

# Import base operation class from registry
from ..operation_registry import BaseOperation

# Import all operations to trigger registration
from .segment_object import SegmentObjectOperation
from .read_text import ReadTextOperation
from .get_properties import GetPropertiesOperation
from .track_object import TrackObjectOperation
from .zoom_in import ZoomInOperation

# List of all available operations
__all__ = [
    'BaseOperation',
    'SegmentObjectOperation',
    'ReadTextOperation',
    'GetPropertiesOperation',
    'TrackObjectOperation',
    'ZoomInOperation',
]

# Operation categories for organization
OPERATION_CATEGORIES = {
    'segmentation': ['SEGMENT_OBJECT_AT'],
    'text_extraction': ['READ_TEXT'],
    'analysis': ['GET_PROPERTIES'],
    'tracking': ['TRACK_OBJECT'],
    'transformation': ['ZOOM_IN'],
}

def list_operations_by_category():
    """
    List all operations organized by category.
    
    Returns:
        Dictionary mapping categories to operation names
    """
    return OPERATION_CATEGORIES