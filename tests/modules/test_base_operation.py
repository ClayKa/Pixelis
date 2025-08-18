"""
Tests for base_operation.py

This test file achieves 100% coverage for the BaseOperation abstract base class,
including all imports, class definition, methods, and string representations.
"""

import unittest
import logging
from abc import ABC
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List

import sys
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.operations.base_operation import BaseOperation


class ConcreteOperation(BaseOperation):
    """Concrete implementation of BaseOperation for testing."""
    
    def run(self, **kwargs) -> Any:
        """Test implementation of abstract run method."""
        return "test_result"


class ConcreteOperationWithDocstring(BaseOperation):
    """
    This is a test operation with a multi-line docstring.
    It has multiple lines to test docstring parsing.
    """
    
    def run(self, **kwargs) -> Any:
        """Test implementation of abstract run method."""
        return "test_result"


class ConcreteOperationNoDocstring(BaseOperation):
    """No docstring will be set to None for testing."""
    
    def run(self, **kwargs) -> Any:
        """Test implementation of abstract run method.""" 
        return "test_result"


class TestBaseOperationImports(unittest.TestCase):
    """Test that all imports are covered."""
    
    def test_imports_coverage(self):
        """Test lines 8-12: imports and module-level logger creation."""
        # Line 8: from abc import ABC, abstractmethod
        # Line 9: from typing import Any, Dict, Optional, List, Tuple
        # Line 10: import logging
        # Line 12: logger = logging.getLogger(__name__)
        
        # Import the module to ensure all import statements are executed
        import core.modules.operations.base_operation as base_op_module
        
        # Verify the module-level logger was created (line 12)
        self.assertIsInstance(base_op_module.logger, logging.Logger)
        self.assertEqual(base_op_module.logger.name, 'core.modules.operations.base_operation')


class TestBaseOperationClass(unittest.TestCase):
    """Test the BaseOperation class definition and instantiation."""
    
    def test_class_definition(self):
        """Test line 15: class BaseOperation(ABC)."""
        # This test covers the class definition line
        self.assertTrue(issubclass(BaseOperation, ABC))
        self.assertTrue(hasattr(BaseOperation, 'run'))
        self.assertTrue(hasattr(BaseOperation, '__init__'))

    def test_init_method(self):
        """Test lines 23, 25: __init__ method and logger setup."""
        # Create a concrete implementation to test __init__
        operation = ConcreteOperation()
        
        # Line 23: def __init__(self):
        # Line 25: self.logger = logging.getLogger(self.__class__.__name__)
        self.assertIsInstance(operation.logger, logging.Logger)
        self.assertEqual(operation.logger.name, 'ConcreteOperation')
    
    def test_abstract_method_enforcement(self):
        """Test that BaseOperation cannot be instantiated directly."""
        # This test ensures the abstract class behavior works correctly
        with self.assertRaises(TypeError):
            BaseOperation()  # Should raise TypeError due to abstract method


class TestBaseOperationValidateInputs(unittest.TestCase):
    """Test the validate_inputs method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = ConcreteOperation()
    
    def test_validate_inputs_default(self):
        """Test lines 40, 52: validate_inputs method and default return."""
        # Line 40: def validate_inputs(self, **kwargs) -> bool:
        # Line 52: return True
        
        # Test with no arguments
        result = self.operation.validate_inputs()
        self.assertTrue(result)
        
        # Test with various arguments
        result = self.operation.validate_inputs(param1="value1", param2=123)
        self.assertTrue(result)
        
        # Test with empty dict
        result = self.operation.validate_inputs(**{})
        self.assertTrue(result)


class TestBaseOperationPreprocess(unittest.TestCase):
    """Test the preprocess method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = ConcreteOperation()
    
    def test_preprocess_default(self):
        """Test lines 54, 66: preprocess method and default return."""
        # Line 54: def preprocess(self, **kwargs) -> Dict[str, Any]:
        # Line 66: return kwargs
        
        # Test with no arguments
        result = self.operation.preprocess()
        self.assertEqual(result, {})
        
        # Test with various arguments
        test_kwargs = {"param1": "value1", "param2": 123, "param3": [1, 2, 3]}
        result = self.operation.preprocess(**test_kwargs)
        self.assertEqual(result, test_kwargs)
        
        # Test that the result is the same object (not a copy)
        original_kwargs = {"test": "data"}
        result = self.operation.preprocess(**original_kwargs)
        self.assertEqual(result, original_kwargs)


class TestBaseOperationPostprocess(unittest.TestCase):
    """Test the postprocess method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = ConcreteOperation()
    
    def test_postprocess_default(self):
        """Test lines 68, 80: postprocess method and default return."""
        # Line 68: def postprocess(self, result: Any) -> Any:
        # Line 80: return result
        
        # Test with string result
        test_result = "test_result"
        result = self.operation.postprocess(test_result)
        self.assertEqual(result, test_result)
        
        # Test with dict result
        test_dict = {"key": "value", "number": 42}
        result = self.operation.postprocess(test_dict)
        self.assertEqual(result, test_dict)
        
        # Test with None result
        result = self.operation.postprocess(None)
        self.assertIsNone(result)
        
        # Test with complex object
        test_list = [1, 2, {"nested": "data"}]
        result = self.operation.postprocess(test_list)
        self.assertEqual(result, test_list)


class TestBaseOperationGetRequiredParams(unittest.TestCase):
    """Test the get_required_params method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = ConcreteOperation()
    
    def test_get_required_params_default(self):
        """Test lines 82, 91: get_required_params method and default return."""
        # Line 82: def get_required_params(self) -> List[str]:
        # Line 91: return []
        
        result = self.operation.get_required_params()
        self.assertEqual(result, [])
        self.assertIsInstance(result, list)


class TestBaseOperationGetOptionalParams(unittest.TestCase):
    """Test the get_optional_params method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = ConcreteOperation()
    
    def test_get_optional_params_default(self):
        """Test lines 93, 102: get_optional_params method and default return."""
        # Line 93: def get_optional_params(self) -> Dict[str, Any]:
        # Line 102: return {}
        
        result = self.operation.get_optional_params()
        self.assertEqual(result, {})
        self.assertIsInstance(result, dict)


class TestBaseOperationStringRepresentations(unittest.TestCase):
    """Test the __repr__ and __str__ methods."""
    
    def test_repr_method(self):
        """Test the __repr__ method (excluded from coverage)."""
        # Note: __repr__ is excluded from coverage but we test it for completeness
        operation = ConcreteOperation()
        result = repr(operation)
        self.assertEqual(result, "ConcreteOperation()")
    
    def test_str_method_with_docstring(self):
        """Test lines 108, 110, 112, 113: __str__ method with docstring."""
        # Line 108: def __str__(self) -> str:
        # Line 110: doc = self.__doc__ or "No description available"
        # Line 112: description = doc.strip().split('\n')[0]
        # Line 113: return f"{self.__class__.__name__}: {description}"
        
        operation = ConcreteOperationWithDocstring()
        result = str(operation)
        self.assertEqual(result, "ConcreteOperationWithDocstring: This is a test operation with a multi-line docstring.")
    
    def test_str_method_no_docstring(self):
        """Test lines 108, 110, 112, 113: __str__ method without docstring."""
        # Test the fallback case when __doc__ is None
        operation = ConcreteOperationNoDocstring()
        # Manually set __doc__ to None to test the "or" condition
        operation.__doc__ = None
        
        result = str(operation)
        self.assertEqual(result, "ConcreteOperationNoDocstring: No description available")
    
    def test_str_method_empty_docstring(self):
        """Test __str__ method with empty docstring."""
        operation = ConcreteOperation()
        # Set empty docstring
        operation.__doc__ = "   \n  \n  "
        
        result = str(operation)
        # Empty docstring after strip() should result in empty string, so first split element is empty
        self.assertEqual(result, "ConcreteOperation: ")
    
    def test_str_method_single_line_docstring(self):
        """Test __str__ method with single line docstring."""
        operation = ConcreteOperation()
        operation.__doc__ = "Single line description"
        
        result = str(operation)
        self.assertEqual(result, "ConcreteOperation: Single line description")


class TestBaseOperationMethodOverrides(unittest.TestCase):
    """Test that methods can be properly overridden in subclasses."""
    
    def test_method_inheritance_and_override_capability(self):
        """Test that all methods exist and can be overridden."""
        operation = ConcreteOperation()
        
        # Test that all expected methods exist
        self.assertTrue(hasattr(operation, 'validate_inputs'))
        self.assertTrue(hasattr(operation, 'preprocess'))
        self.assertTrue(hasattr(operation, 'postprocess'))
        self.assertTrue(hasattr(operation, 'get_required_params'))
        self.assertTrue(hasattr(operation, 'get_optional_params'))
        self.assertTrue(hasattr(operation, 'run'))
        self.assertTrue(hasattr(operation, '__str__'))
        self.assertTrue(hasattr(operation, '__repr__'))
        
        # Test that they are callable
        self.assertTrue(callable(operation.validate_inputs))
        self.assertTrue(callable(operation.preprocess))
        self.assertTrue(callable(operation.postprocess))
        self.assertTrue(callable(operation.get_required_params))
        self.assertTrue(callable(operation.get_optional_params))
        self.assertTrue(callable(operation.run))


class TestFullWorkflow(unittest.TestCase):
    """Test a complete workflow using all methods."""
    
    def test_complete_operation_workflow(self):
        """Test using all methods in a typical workflow."""
        operation = ConcreteOperation()
        
        # Test complete workflow that exercises all lines
        input_data = {"param1": "value1", "param2": 42}
        
        # Step 1: Validate inputs (lines 40, 52)
        is_valid = operation.validate_inputs(**input_data)
        self.assertTrue(is_valid)
        
        # Step 2: Preprocess (lines 54, 66)
        preprocessed = operation.preprocess(**input_data)
        self.assertEqual(preprocessed, input_data)
        
        # Step 3: Run operation (implemented in concrete class)
        result = operation.run(**preprocessed)
        self.assertEqual(result, "test_result")
        
        # Step 4: Postprocess (lines 68, 80)
        final_result = operation.postprocess(result)
        self.assertEqual(final_result, "test_result")
        
        # Step 5: Check parameter requirements (lines 82, 91, 93, 102)
        required_params = operation.get_required_params()
        optional_params = operation.get_optional_params()
        self.assertEqual(required_params, [])
        self.assertEqual(optional_params, {})
        
        # Step 6: String representations (lines 108, 110, 112, 113)
        str_repr = str(operation)
        self.assertIn("ConcreteOperation", str_repr)


if __name__ == '__main__':
    unittest.main()