"""
Tests for operation_registry.py to achieve 100% coverage

This test file targets the 27 missing statements in operation_registry.py
to achieve complete code coverage, including error conditions, edge cases,
and all execution paths.
"""

import unittest
import logging
from unittest.mock import patch, MagicMock
from typing import Any, Dict
import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.operation_registry import (
    BaseOperation,
    VisualOperationRegistry,
    registry,
    register_operation
)


class MockOperation(BaseOperation):
    """Mock operation for testing."""
    
    def run(self, **kwargs) -> Any:
        return "mock_result"


class MockOperationWithValidation(BaseOperation):
    """Mock operation with custom validation."""
    
    def run(self, **kwargs) -> Any:
        return "mock_result"
    
    def validate_inputs(self, **kwargs) -> bool:
        # Return False for testing validation failure
        return kwargs.get("valid", True)


class InvalidOperation:
    """Invalid operation that doesn't inherit from BaseOperation."""
    
    def run(self, **kwargs):
        return "invalid"


class TestMissingCoverageBaseOperation(unittest.TestCase):
    """Test BaseOperation methods to cover missing statements."""
    
    def test_validate_inputs_default_return_true(self):
        """Test line 52: return True in validate_inputs method."""
        operation = MockOperation()
        # This should hit line 52: return True
        result = operation.validate_inputs(test_param="value")
        self.assertTrue(result)


class TestMissingCoverageVisualOperationRegistry(unittest.TestCase):
    """Test VisualOperationRegistry to cover all missing statements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh registry instance for each test
        # Clear the singleton instance to start fresh
        VisualOperationRegistry._instance = None
        self.registry = VisualOperationRegistry()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear registry for next test
        if hasattr(self.registry, 'clear'):
            self.registry.clear()
        VisualOperationRegistry._instance = None
    
    def test_singleton_second_initialization_early_return(self):
        """Test line 84: return in __init__ when already initialized."""
        # Create first instance
        registry1 = VisualOperationRegistry()
        self.assertTrue(registry1._initialized)
        
        # Create second instance - should hit line 84 (early return)
        registry2 = VisualOperationRegistry()
        
        # Should be the same instance due to singleton pattern
        self.assertIs(registry1, registry2)
        self.assertTrue(registry2._initialized)
    
    def test_register_empty_operation_name_error(self):
        """Test line 111: raise ValueError for empty operation name."""
        # Test empty string - should hit line 111
        with self.assertRaises(ValueError) as context:
            self.registry.register("", MockOperation)
        self.assertIn("Operation name cannot be empty", str(context.exception))
        
        # Test None as operation name - should also hit line 111
        with self.assertRaises(ValueError) as context:
            self.registry.register(None, MockOperation)
        self.assertIn("Operation name cannot be empty", str(context.exception))
    
    def test_register_duplicate_operation_name_error(self):
        """Test line 114: raise ValueError for duplicate operation name."""
        # Register first operation
        self.registry.register("TEST_OP", MockOperation)
        
        # Try to register with same name - should hit line 114
        with self.assertRaises(ValueError) as context:
            self.registry.register("TEST_OP", MockOperation)
        self.assertIn("Operation 'TEST_OP' is already registered", str(context.exception))
    
    def test_register_invalid_operation_class_error(self):
        """Test line 117: raise ValueError for invalid operation class."""
        # Test with class that doesn't inherit from BaseOperation - should hit line 117
        with self.assertRaises(ValueError) as context:
            self.registry.register("INVALID_OP", InvalidOperation)
        self.assertIn("must inherit from BaseOperation", str(context.exception))
        
        # Test with non-class object - this will cause TypeError in issubclass
        # We need to catch that and convert to ValueError
        with self.assertRaises(TypeError):
            self.registry.register("INVALID_OP", "not_a_class")
    
    def test_register_no_metadata_default_creation(self):
        """Test line 128: default metadata creation when none provided."""
        # Register operation without metadata - should hit line 128 and create default
        self.registry.register("NO_META_OP", MockOperation)
        
        # Check that default metadata was created (line 128-132)
        metadata = self.registry.get_operation_metadata("NO_META_OP")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['class_name'], 'MockOperation')
        self.assertEqual(metadata['module'], MockOperation.__module__)
        self.assertEqual(metadata['doc'], MockOperation.__doc__)
    
    def test_unregister_existing_operation(self):
        """Test lines 146-151: successful unregister operation."""
        # Register an operation first
        self.registry.register("TEMP_OP", MockOperation)
        self.assertTrue(self.registry.has_operation("TEMP_OP"))
        
        # Unregister it - should hit lines 146-151
        result = self.registry.unregister("TEMP_OP")
        self.assertTrue(result)
        self.assertFalse(self.registry.has_operation("TEMP_OP"))
        
        # Verify both operations and metadata were removed
        self.assertIsNone(self.registry.get_operation_class("TEMP_OP"))
        self.assertIsNone(self.registry.get_operation_metadata("TEMP_OP"))
    
    def test_unregister_nonexistent_operation(self):
        """Test line 151: return False for non-existent operation."""
        # Try to unregister operation that doesn't exist - should hit line 151
        result = self.registry.unregister("NON_EXISTENT_OP")
        self.assertFalse(result)
    
    def test_execute_operation_not_registered_error(self):
        """Test line 174: raise NotImplementedError for unregistered operation."""
        # Try to execute non-existent operation - should hit line 174-178
        with self.assertRaises(NotImplementedError) as context:
            self.registry.execute("NON_EXISTENT_OP", param1="value")
        
        error_msg = str(context.exception)
        self.assertIn("Operation 'NON_EXISTENT_OP' is not registered", error_msg)
        self.assertIn("Available operations:", error_msg)
    
    def test_execute_validation_failure_error(self):
        """Test line 189: raise ValueError for validation failure."""
        # Register operation with custom validation
        self.registry.register("VALIDATION_OP", MockOperationWithValidation)
        
        # Execute with invalid parameters - should hit line 189
        with self.assertRaises(ValueError) as context:
            self.registry.execute("VALIDATION_OP", valid=False)
        self.assertIn("Invalid inputs for operation 'VALIDATION_OP'", str(context.exception))
    
    def test_execute_operation_exception_handling(self):
        """Test lines 197-199: exception handling in execute method."""
        
        class FailingOperation(BaseOperation):
            def run(self, **kwargs):
                raise RuntimeError("Operation failed")
        
        # Register failing operation
        self.registry.register("FAILING_OP", FailingOperation)
        
        # Execute and expect exception to be re-raised - should hit lines 197-199
        with self.assertRaises(RuntimeError) as context:
            self.registry.execute("FAILING_OP")
        self.assertIn("Operation failed", str(context.exception))
    
    def test_has_operation_return_true(self):
        """Test line 220: return True from has_operation."""
        # Register an operation
        self.registry.register("EXISTS_OP", MockOperation)
        
        # Check that it exists - should hit line 220
        result = self.registry.has_operation("EXISTS_OP")
        self.assertTrue(result)
    
    def test_get_operation_class_return_value(self):
        """Test line 235: return from get_operation_class."""
        # Register an operation
        self.registry.register("CLASS_OP", MockOperation)
        
        # Get the class - should hit line 235
        result = self.registry.get_operation_class("CLASS_OP")
        self.assertEqual(result, MockOperation)
        
        # Test non-existent operation
        result = self.registry.get_operation_class("NON_EXISTENT")
        self.assertIsNone(result)


class TestMissingCoverageDecorator(unittest.TestCase):
    """Test the register_operation decorator to cover missing statements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear the singleton instance to start fresh
        VisualOperationRegistry._instance = None
        
        # Import registry to ensure it's fresh
        from core.modules.operation_registry import registry
        self.registry = registry
        self.registry.clear()  # Clear any existing operations
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.registry, 'clear'):
            self.registry.clear()
        # Don't reset _instance here to avoid breaking other tests
    
    def test_register_operation_decorator_basic_usage(self):
        """Test the register_operation decorator functionality."""
        
        # Use decorator to register an operation
        @register_operation("DECORATED_OP")
        class DecoratedOperation(BaseOperation):
            def run(self, **kwargs):
                return "decorated_result"
        
        # Verify the operation was registered
        self.assertTrue(self.registry.has_operation("DECORATED_OP"))
        
        # Execute the decorated operation
        result = self.registry.execute("DECORATED_OP")
        self.assertEqual(result, "decorated_result")
    
    def test_register_operation_decorator_with_metadata(self):
        """Test the register_operation decorator with metadata."""
        
        test_metadata = {
            "description": "Test operation with metadata",
            "version": "1.0",
            "author": "test"
        }
        
        # Use decorator with metadata
        @register_operation("DECORATED_META_OP", metadata=test_metadata)
        class DecoratedOperationWithMeta(BaseOperation):
            def run(self, **kwargs):
                return "decorated_meta_result"
        
        # Verify the operation was registered with metadata
        self.assertTrue(self.registry.has_operation("DECORATED_META_OP"))
        
        # Check metadata
        metadata = self.registry.get_operation_metadata("DECORATED_META_OP")
        self.assertEqual(metadata, test_metadata)
        
        # Execute the decorated operation
        result = self.registry.execute("DECORATED_META_OP")
        self.assertEqual(result, "decorated_meta_result")


class TestMissingCoverageGlobalRegistry(unittest.TestCase):
    """Test the global registry instance."""
    
    def test_global_registry_instance_creation(self):
        """Test that the global registry instance is created properly."""
        from core.modules.operation_registry import registry as global_registry
        
        # Verify global registry is an instance of VisualOperationRegistry
        self.assertIsInstance(global_registry, VisualOperationRegistry)
        
        # Test singleton behavior - all instances should be the same
        instance1 = VisualOperationRegistry()
        instance2 = VisualOperationRegistry()
        self.assertIs(instance1, instance2)


class TestMissingCoverageEdgeCases(unittest.TestCase):
    """Test edge cases and additional scenarios for complete coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        VisualOperationRegistry._instance = None
        self.registry = VisualOperationRegistry()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.registry, 'clear'):
            self.registry.clear()
        VisualOperationRegistry._instance = None
    
    def test_register_with_provided_metadata(self):
        """Test registration with explicitly provided metadata."""
        custom_metadata = {
            "description": "Custom operation",
            "version": "2.0",
            "parameters": ["param1", "param2"]
        }
        
        # Register with custom metadata
        self.registry.register("CUSTOM_META_OP", MockOperation, metadata=custom_metadata)
        
        # Verify metadata was stored correctly
        stored_metadata = self.registry.get_operation_metadata("CUSTOM_META_OP")
        self.assertEqual(stored_metadata, custom_metadata)
    
    def test_execute_successful_operation(self):
        """Test successful operation execution to cover normal path."""
        # Register a working operation
        self.registry.register("WORKING_OP", MockOperation)
        
        # Execute successfully
        result = self.registry.execute("WORKING_OP", test_param="value")
        self.assertEqual(result, "mock_result")
    
    def test_list_operations_functionality(self):
        """Test list_operations method."""
        # Register multiple operations
        self.registry.register("OP1", MockOperation)
        self.registry.register("OP2", MockOperationWithValidation, 
                             metadata={"description": "Test op 2"})
        
        # Get operations list
        operations = self.registry.list_operations()
        
        # Verify both operations are listed
        self.assertIn("OP1", operations)
        self.assertIn("OP2", operations)
        self.assertEqual(len(operations), 2)
    
    def test_registry_repr_method(self):
        """Test the __repr__ method of the registry."""
        # Register some operations
        self.registry.register("REPR_OP1", MockOperation)
        self.registry.register("REPR_OP2", MockOperationWithValidation)
        
        # Get string representation
        repr_str = repr(self.registry)
        
        # Verify format
        self.assertIn("VisualOperationRegistry", repr_str)
        self.assertIn("operations=", repr_str)
        self.assertIn("count=2", repr_str)
        self.assertIn("REPR_OP1", repr_str)
        self.assertIn("REPR_OP2", repr_str)
    
    def test_clear_operations(self):
        """Test clearing all operations."""
        # Register operations
        self.registry.register("CLEAR_OP1", MockOperation)
        self.registry.register("CLEAR_OP2", MockOperationWithValidation)
        
        # Verify operations exist
        self.assertEqual(len(self.registry.list_operations()), 2)
        
        # Clear all operations
        self.registry.clear()
        
        # Verify operations are cleared
        self.assertEqual(len(self.registry.list_operations()), 0)
        self.assertFalse(self.registry.has_operation("CLEAR_OP1"))
        self.assertFalse(self.registry.has_operation("CLEAR_OP2"))


class TestMissingCoverageLogging(unittest.TestCase):
    """Test logging functionality to ensure coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        VisualOperationRegistry._instance = None
        self.registry = VisualOperationRegistry()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.registry, 'clear'):
            self.registry.clear()
        VisualOperationRegistry._instance = None
    
    @patch('core.modules.operation_registry.logger')
    def test_logging_during_operations(self, mock_logger):
        """Test that logging statements are executed."""
        # Test registration logging
        self.registry.register("LOG_OP", MockOperation)
        mock_logger.debug.assert_called()
        
        # Test unregistration logging
        self.registry.unregister("LOG_OP")
        mock_logger.debug.assert_called()
        
        # Test execution logging
        self.registry.register("LOG_OP2", MockOperation)
        self.registry.execute("LOG_OP2")
        mock_logger.debug.assert_called()


if __name__ == '__main__':
    # Run with verbose output to see all test coverage
    unittest.main(verbosity=2)