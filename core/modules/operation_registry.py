"""
Visual Operation Registry Module

Central, decoupled system for managing and executing all visual operations (tools).
Implements a singleton pattern to ensure a single global registry instance.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod

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
    
    def __repr__(self) -> str:
        """String representation of the operation."""
        return f"{self.__class__.__name__}()"


class VisualOperationRegistry:
    """
    Singleton registry for managing visual operations.
    
    This class maintains a central registry of all available visual operations
    and provides methods to register new operations and execute them by name.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Ensure singleton pattern - only one instance exists.
        """
        if cls._instance is None:
            cls._instance = super(VisualOperationRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the registry.
        """
        # Avoid re-initialization for singleton
        if self._initialized:
            return
            
        self._operations: Dict[str, Type[BaseOperation]] = {}
        self._operation_metadata: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        
        logger.info("Visual Operation Registry initialized")
    
    def register(
        self,
        operation_name: str,
        operation_class: Type[BaseOperation],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new visual operation.
        
        Args:
            operation_name: Name identifier for the operation (e.g., 'SEGMENT_OBJECT_AT')
            operation_class: Class implementing the operation (must inherit from BaseOperation)
            metadata: Optional metadata about the operation (description, parameters, etc.)
        
        Raises:
            ValueError: If operation_name already exists or operation_class is invalid
        """
        # Validate inputs
        if not operation_name:
            raise ValueError("Operation name cannot be empty")
        
        if operation_name in self._operations:
            raise ValueError(f"Operation '{operation_name}' is already registered")
        
        if not issubclass(operation_class, BaseOperation):
            raise ValueError(
                f"Operation class {operation_class.__name__} must inherit from BaseOperation"
            )
        
        # Register the operation
        self._operations[operation_name] = operation_class
        
        # Store metadata if provided
        if metadata:
            self._operation_metadata[operation_name] = metadata
        else:
            self._operation_metadata[operation_name] = {
                'class_name': operation_class.__name__,
                'module': operation_class.__module__,
                'doc': operation_class.__doc__
            }
        
        logger.debug(f"Registered operation: {operation_name} -> {operation_class.__name__}")
    
    def unregister(self, operation_name: str) -> bool:
        """
        Unregister a visual operation.
        
        Args:
            operation_name: Name of the operation to unregister
            
        Returns:
            True if successfully unregistered, False if operation not found
        """
        if operation_name in self._operations:
            del self._operations[operation_name]
            del self._operation_metadata[operation_name]
            logger.debug(f"Unregistered operation: {operation_name}")
            return True
        return False
    
    def execute(
        self,
        operation_name: str,
        **kwargs
    ) -> Any:
        """
        Execute a registered visual operation.
        
        Args:
            operation_name: Name of the operation to execute
            **kwargs: Arguments to pass to the operation's run method
            
        Returns:
            Result from the operation execution
            
        Raises:
            NotImplementedError: If operation is not registered
            Exception: Any exception raised by the operation itself
        """
        # Check if operation exists
        if operation_name not in self._operations:
            available_ops = ', '.join(self._operations.keys())
            raise NotImplementedError(
                f"Operation '{operation_name}' is not registered. "
                f"Available operations: {available_ops}"
            )
        
        # Get the operation class
        operation_class = self._operations[operation_name]
        
        try:
            # Instantiate the operation
            operation_instance = operation_class()
            
            # Validate inputs if validation is implemented
            if not operation_instance.validate_inputs(**kwargs):
                raise ValueError(f"Invalid inputs for operation '{operation_name}'")
            
            # Execute the operation
            logger.debug(f"Executing operation: {operation_name}")
            result = operation_instance.run(**kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing operation '{operation_name}': {e}")
            raise
    
    def list_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered operations with their metadata.
        
        Returns:
            Dictionary mapping operation names to their metadata
        """
        return dict(self._operation_metadata)
    
    def has_operation(self, operation_name: str) -> bool:
        """
        Check if an operation is registered.
        
        Args:
            operation_name: Name of the operation to check
            
        Returns:
            True if operation is registered, False otherwise
        """
        return operation_name in self._operations
    
    def get_operation_class(
        self,
        operation_name: str
    ) -> Optional[Type[BaseOperation]]:
        """
        Get the class for a registered operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation class if registered, None otherwise
        """
        return self._operations.get(operation_name)
    
    def get_operation_metadata(
        self,
        operation_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a registered operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation metadata if registered, None otherwise
        """
        return self._operation_metadata.get(operation_name)
    
    def clear(self) -> None:
        """
        Clear all registered operations.
        
        Use with caution - this removes all registered operations.
        """
        self._operations.clear()
        self._operation_metadata.clear()
        logger.warning("All operations have been cleared from the registry")
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"VisualOperationRegistry("
            f"operations={list(self._operations.keys())}, "
            f"count={len(self._operations)})"
        )


# Create the global singleton instance
registry = VisualOperationRegistry()


# Decorator for auto-registration (optional convenience feature)
def register_operation(
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to automatically register an operation class.
    
    Usage:
        @register_operation("SEGMENT_OBJECT_AT")
        class SegmentObjectOperation(BaseOperation):
            ...
    
    Args:
        name: Name to register the operation under
        metadata: Optional metadata for the operation
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseOperation]) -> Type[BaseOperation]:
        registry.register(name, cls, metadata)
        return cls
    return decorator