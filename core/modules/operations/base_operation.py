"""
Base Operation Module

Defines the abstract base class for all visual operations.
This ensures a consistent interface across all operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseOperation(ABC):
    """
    Abstract base class for all visual operations.
    
    All specific operations must inherit from this class and implement
    the run method to ensure a consistent interface.
    """
    
    def __init__(self):
        """Initialize the base operation."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the operation.
        
        Args:
            **kwargs: Operation-specific arguments
            
        Returns:
            Operation result (format depends on specific operation)
        """
        pass
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input arguments for the operation.
        
        Override in subclasses to provide specific validation.
        
        Args:
            **kwargs: Operation-specific arguments
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return True
    
    def preprocess(self, **kwargs) -> Dict[str, Any]:
        """
        Preprocess inputs before execution.
        
        Override in subclasses if preprocessing is needed.
        
        Args:
            **kwargs: Raw input arguments
            
        Returns:
            Preprocessed arguments
        """
        return kwargs
    
    def postprocess(self, result: Any) -> Any:
        """
        Postprocess the operation result.
        
        Override in subclasses if postprocessing is needed.
        
        Args:
            result: Raw operation result
            
        Returns:
            Postprocessed result
        """
        return result
    
    def get_required_params(self) -> List[str]:
        """
        Get list of required parameters for this operation.
        
        Override in subclasses to specify required parameters.
        
        Returns:
            List of required parameter names
        """
        return []
    
    def get_optional_params(self) -> Dict[str, Any]:
        """
        Get dictionary of optional parameters with their default values.
        
        Override in subclasses to specify optional parameters.
        
        Returns:
            Dictionary mapping parameter names to default values
        """
        return {}
    
    def __repr__(self) -> str:
        """String representation of the operation."""
        return f"{self.__class__.__name__}()"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        doc = self.__doc__ or "No description available"
        # Get first line of docstring
        description = doc.strip().split('\n')[0]
        return f"{self.__class__.__name__}: {description}"