"""
Context management utilities for distributed tracing.

This module provides context variables for maintaining trace IDs across
asynchronous operations and distributed components, enabling end-to-end
request tracing for enhanced debuggability.
"""

from contextvars import ContextVar
from typing import Optional, Dict, Any
import uuid
from datetime import datetime


# The context variable will hold a string trace_id, or None if not set
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

# Additional context variables for enriched tracing
request_metadata_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "request_metadata", default=None
)

# Context for tracking the originating component
component_var: ContextVar[Optional[str]] = ContextVar("component", default=None)


class TraceContext:
    """
    Manager for distributed trace context.
    
    This class provides utilities for managing trace IDs and associated
    metadata throughout the lifecycle of a request.
    """
    
    @staticmethod
    def generate_trace_id() -> str:
        """
        Generate a new unique trace ID.
        
        Returns:
            A unique trace ID string.
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def set_trace_id(trace_id: Optional[str] = None) -> str:
        """
        Set the trace ID for the current context.
        
        Args:
            trace_id: The trace ID to set. If None, generates a new one.
            
        Returns:
            The trace ID that was set.
        """
        if trace_id is None:
            trace_id = TraceContext.generate_trace_id()
        
        token = trace_id_var.set(trace_id)
        return trace_id
    
    @staticmethod
    def get_trace_id() -> Optional[str]:
        """
        Get the current trace ID from context.
        
        Returns:
            The current trace ID or None if not set.
        """
        return trace_id_var.get()
    
    @staticmethod
    def set_metadata(metadata: Dict[str, Any]) -> None:
        """
        Set request metadata for the current context.
        
        Args:
            metadata: Dictionary of metadata to associate with the trace.
        """
        current = request_metadata_var.get()
        if current is None:
            current = {}
        
        # Merge with existing metadata
        current.update(metadata)
        request_metadata_var.set(current)
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """
        Get the current request metadata.
        
        Returns:
            Dictionary of metadata or empty dict if not set.
        """
        metadata = request_metadata_var.get()
        return metadata if metadata is not None else {}
    
    @staticmethod
    def set_component(component: str) -> None:
        """
        Set the component name for the current context.
        
        Args:
            component: Name of the component handling the request.
        """
        component_var.set(component)
    
    @staticmethod
    def get_component() -> Optional[str]:
        """
        Get the current component name.
        
        Returns:
            Component name or None if not set.
        """
        return component_var.get()
    
    @staticmethod
    def create_child_context(parent_trace_id: Optional[str] = None) -> str:
        """
        Create a child trace context.
        
        Args:
            parent_trace_id: Parent trace ID. If None, uses current context.
            
        Returns:
            New trace ID for the child context.
        """
        if parent_trace_id is None:
            parent_trace_id = TraceContext.get_trace_id()
        
        # Create child trace ID that includes parent reference
        if parent_trace_id:
            child_id = f"{parent_trace_id}:{uuid.uuid4().hex[:8]}"
        else:
            child_id = TraceContext.generate_trace_id()
        
        TraceContext.set_trace_id(child_id)
        
        # Set parent reference in metadata
        TraceContext.set_metadata({"parent_trace_id": parent_trace_id})
        
        return child_id
    
    @staticmethod
    def to_dict() -> Dict[str, Any]:
        """
        Export current context as a dictionary.
        
        Returns:
            Dictionary containing all context information.
        """
        return {
            "trace_id": TraceContext.get_trace_id(),
            "component": TraceContext.get_component(),
            "metadata": TraceContext.get_metadata(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    @staticmethod
    def from_dict(context_dict: Dict[str, Any]) -> None:
        """
        Import context from a dictionary.
        
        Args:
            context_dict: Dictionary containing context information.
        """
        if "trace_id" in context_dict:
            TraceContext.set_trace_id(context_dict["trace_id"])
        
        if "component" in context_dict:
            TraceContext.set_component(context_dict["component"])
        
        if "metadata" in context_dict:
            TraceContext.set_metadata(context_dict["metadata"])


class TracedOperation:
    """
    Context manager for traced operations.
    
    Automatically sets up and tears down trace context for an operation.
    """
    
    def __init__(
        self,
        operation_name: str,
        component: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a traced operation.
        
        Args:
            operation_name: Name of the operation being traced.
            component: Component performing the operation.
            trace_id: Optional trace ID. Generated if not provided.
            metadata: Optional metadata to include.
        """
        self.operation_name = operation_name
        self.component = component
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.metadata["operation"] = operation_name
        
        # Store tokens for cleanup
        self._trace_token = None
        self._component_token = None
        self._metadata_token = None
        
    def __enter__(self):
        """Set up trace context."""
        # Set trace ID
        if self.trace_id is None:
            self.trace_id = TraceContext.generate_trace_id()
        self._trace_token = trace_id_var.set(self.trace_id)
        
        # Set component
        self._component_token = component_var.set(self.component)
        
        # Set metadata
        self.metadata["start_time"] = datetime.utcnow().isoformat()
        self._metadata_token = request_metadata_var.set(self.metadata)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up trace context."""
        # Add end time to metadata
        if self._metadata_token:
            metadata = request_metadata_var.get()
            if metadata:
                metadata["end_time"] = datetime.utcnow().isoformat()
                if exc_type is not None:
                    metadata["error"] = str(exc_val)
                    metadata["error_type"] = exc_type.__name__
        
        # Reset context variables
        if self._trace_token:
            trace_id_var.reset(self._trace_token)
        if self._component_token:
            component_var.reset(self._component_token)
        if self._metadata_token:
            request_metadata_var.reset(self._metadata_token)
        
        return False  # Don't suppress exceptions