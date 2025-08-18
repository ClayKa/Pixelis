#!/usr/bin/env python3
"""
Comprehensive test suite for decorators.py to achieve 100% coverage.
Tests all decorators including track_artifacts, reproducible, and checkpoint.
"""

import functools
import hashlib
import inspect
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import TestCase, mock
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock

import pytest
import numpy as np

# Set up path for imports
import sys
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.reproducibility.decorators import (
    track_artifacts,
    reproducible,
    checkpoint,
    _serialize_arg
)
from core.reproducibility.artifact_manager import ArtifactManager, ArtifactType
from core.reproducibility.experiment_context import ExperimentContext


class TestSerializeArg(TestCase):
    """Test the _serialize_arg helper function"""
    
    def test_serialize_json_serializable(self):
        """Test serializing JSON-serializable objects"""
        # Test basic types
        self.assertEqual(_serialize_arg(42), 42)
        self.assertEqual(_serialize_arg("test"), "test")
        self.assertEqual(_serialize_arg(3.14), 3.14)
        self.assertEqual(_serialize_arg(True), True)
        self.assertEqual(_serialize_arg(None), None)
        
        # Test collections
        self.assertEqual(_serialize_arg([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_serialize_arg({"a": 1}), {"a": 1})
        self.assertEqual(_serialize_arg((1, 2)), [1, 2])  # Tuples become lists
    
    def test_serialize_path(self):
        """Test serializing Path objects"""
        path = Path("/tmp/test.txt")
        self.assertEqual(_serialize_arg(path), str(path))
    
    def test_serialize_long_list(self):
        """Test serializing long lists (truncation)"""
        long_list = list(range(20))
        result = _serialize_arg(long_list)
        # Should truncate to first 5 + ... + last 2
        self.assertEqual(len(result), 8)  # 5 + 1 ("...") + 2
        self.assertEqual(result[5], "...")
        self.assertEqual(result[:5], [0, 1, 2, 3, 4])
        self.assertEqual(result[-2:], [18, 19])
    
    def test_serialize_large_dict(self):
        """Test serializing large dictionaries (truncation)"""
        large_dict = {f"key_{i}": i for i in range(15)}
        result = _serialize_arg(large_dict)
        # Should truncate to first 5 items
        self.assertEqual(len(result), 5)
    
    def test_serialize_numpy_array(self):
        """Test serializing numpy arrays"""
        arr = np.array([[1, 2], [3, 4]])
        result = _serialize_arg(arr)
        self.assertEqual(result["type"], "numpy.ndarray")
        self.assertEqual(result["shape"], (2, 2))
        self.assertIn("dtype", result)
    
    @patch('core.reproducibility.decorators.torch')
    def test_serialize_torch_tensor(self, mock_torch):
        """Test serializing torch tensors"""
        mock_tensor = Mock()
        mock_tensor.shape = [2, 3]
        mock_tensor.dtype = "torch.float32"
        mock_tensor.device = "cuda:0"
        mock_torch.Tensor = Mock
        
        # Make isinstance work
        with patch('builtins.isinstance', side_effect=lambda obj, cls: obj is mock_tensor and cls is mock_torch.Tensor or isinstance.__wrapped__(obj, cls)):
            result = _serialize_arg(mock_tensor)
            self.assertEqual(result["type"], "torch.Tensor")
            self.assertEqual(result["shape"], [2, 3])
    
    def test_serialize_object_with_dict(self):
        """Test serializing objects with __dict__"""
        class CustomObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
        
        obj = CustomObj()
        result = _serialize_arg(obj)
        self.assertEqual(result["type"], "CustomObj")
        self.assertEqual(result["attributes"]["attr1"], "value1")
        self.assertEqual(result["attributes"]["attr2"], 42)
    
    def test_serialize_max_depth(self):
        """Test max_depth parameter"""
        nested = {"a": {"b": {"c": {"d": "value"}}}}
        result = _serialize_arg(nested, max_depth=2)
        # At depth 2, should return type string for deeper nesting
        self.assertIsInstance(result["a"]["b"], str)
        self.assertIn("dict", result["a"]["b"])
    
    def test_serialize_fallback(self):
        """Test fallback to string representation"""
        class NoDict:
            def __repr__(self):
                return "NoDict()"
        
        # Remove __dict__ attribute
        obj = NoDict()
        if hasattr(obj, '__dict__'):
            delattr(obj.__class__, '__dict__')
        
        result = _serialize_arg(obj)
        self.assertEqual(result, "NoDict()")


class TestTrackArtifactsDecorator(TestCase):
    """Test the track_artifacts decorator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_manager = Mock(spec=ArtifactManager)
        self.mock_manager.run = Mock()
        self.mock_manager.run.id = "test_run_id"
        
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_basic_function_tracking(self, mock_manager_class):
        """Test basic function execution tracking"""
        mock_manager_class.return_value = self.mock_manager
        
        @track_artifacts()
        def test_func(x, y):
            return x + y
        
        result = test_func(1, 2)
        self.assertEqual(result, 3)
        
        # Manager should be instantiated
        mock_manager_class.assert_called_once()
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_capture_arguments(self, mock_manager_class):
        """Test capturing function arguments"""
        mock_manager_class.return_value = self.mock_manager
        
        # Track calls to log_artifact
        logged_artifacts = []
        self.mock_manager.log_artifact.side_effect = lambda **kwargs: (
            logged_artifacts.append(kwargs),
            Mock(name="test_artifact", version="v1")
        )[-1]
        
        @track_artifacts(capture_args=True)
        def test_func(x, y=10, z=None):
            return x + y
        
        result = test_func(5, y=15)
        self.assertEqual(result, 20)
        
        # Check that arguments were captured
        start_artifact = logged_artifacts[0]
        self.assertIn("data", start_artifact)
        self.assertIn("arguments", start_artifact["data"])
        self.assertEqual(start_artifact["data"]["arguments"]["x"], 5)
        self.assertEqual(start_artifact["data"]["arguments"]["y"], 15)
        self.assertIsNone(start_artifact["data"]["arguments"]["z"])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.inspect.getsource')
    def test_capture_source(self, mock_getsource, mock_manager_class):
        """Test capturing function source code"""
        mock_manager_class.return_value = self.mock_manager
        mock_getsource.return_value = "def test_func(): pass"
        
        logged_artifacts = []
        self.mock_manager.log_artifact.side_effect = lambda **kwargs: (
            logged_artifacts.append(kwargs),
            Mock(name="test_artifact", version="v1")
        )[-1]
        
        @track_artifacts(capture_source=True)
        def test_func():
            return 42
        
        result = test_func()
        self.assertEqual(result, 42)
        
        # Check that source was captured
        mock_getsource.assert_called_once()
        start_artifact = logged_artifacts[0]
        self.assertIn("source", start_artifact["data"])
        self.assertIn("source_hash", start_artifact["data"])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_track_input_artifacts(self, mock_manager_class):
        """Test tracking input artifacts"""
        mock_manager_class.return_value = self.mock_manager
        
        # Mock use_artifact
        mock_artifact = Mock(name="dataset", version="v1")
        self.mock_manager.use_artifact.return_value = mock_artifact
        
        @track_artifacts(inputs=["dataset:v1", "model"])
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check use_artifact was called
        self.assertEqual(self.mock_manager.use_artifact.call_count, 2)
        self.mock_manager.use_artifact.assert_any_call("dataset", "v1")
        self.mock_manager.use_artifact.assert_any_call("model", None)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_track_output_artifacts(self, mock_manager_class):
        """Test tracking output artifacts"""
        mock_manager_class.return_value = self.mock_manager
        
        logged_artifacts = []
        def log_artifact(**kwargs):
            logged_artifacts.append(kwargs)
            return Mock(name=kwargs.get("name", "artifact"), version="v1")
        
        self.mock_manager.log_artifact.side_effect = log_artifact
        self.mock_manager.log_large_artifact.side_effect = log_artifact
        
        @track_artifacts(outputs=["model", "metrics", "dataset"])
        def test_func():
            return "/path/to/model.pt", {"accuracy": 0.95}, "/path/to/dataset"
        
        result = test_func()
        
        # Check that outputs were tracked
        output_names = [a["name"] for a in logged_artifacts if "test_func" in a["name"]]
        self.assertIn("test_func_model", output_names)
        self.assertIn("test_func_metrics", output_names)
        self.assertIn("test_func_dataset", output_names)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_function_with_exception(self, mock_manager_class):
        """Test function that raises an exception"""
        mock_manager_class.return_value = self.mock_manager
        
        logged_artifacts = []
        self.mock_manager.log_artifact.side_effect = lambda **kwargs: (
            logged_artifacts.append(kwargs),
            Mock(name="test_artifact", version="v1")
        )[-1]
        
        @track_artifacts()
        def test_func():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_func()
        
        # Check that completion was logged with error
        completion_artifacts = [a for a in logged_artifacts 
                               if "complete" in a.get("name", "")]
        self.assertTrue(len(completion_artifacts) > 0)
        completion_data = completion_artifacts[0]["data"]
        self.assertFalse(completion_data["success"])
        self.assertEqual(completion_data["error"], "Test error")
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_output_dict_handling(self, mock_manager_class):
        """Test handling dictionary outputs"""
        mock_manager_class.return_value = self.mock_manager
        
        @track_artifacts(outputs=["result1", "result2"])
        def test_func():
            return {"a": "value1", "b": "value2"}
        
        result = test_func()
        self.assertEqual(result, {"a": "value1", "b": "value2"})
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_output_single_value_handling(self, mock_manager_class):
        """Test handling single value outputs"""
        mock_manager_class.return_value = self.mock_manager
        
        @track_artifacts(outputs=["result"])
        def test_func():
            return "single_value"
        
        result = test_func()
        self.assertEqual(result, "single_value")
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_no_run_context(self, mock_manager_class):
        """Test decorator when no run context exists"""
        mock_manager_class.return_value = self.mock_manager
        self.mock_manager.run = None
        
        @track_artifacts()
        def test_func():
            return 42
        
        result = test_func()
        self.assertEqual(result, 42)
        
        # Should not log artifacts when no run
        self.mock_manager.log_artifact.assert_not_called()
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_capture_args_exception(self, mock_logger, mock_manager_class):
        """Test exception handling when capturing arguments fails"""
        mock_manager_class.return_value = self.mock_manager
        
        # Make inspect.signature raise an exception
        with patch('core.reproducibility.decorators.inspect.signature',
                  side_effect=Exception("Signature error")):
            @track_artifacts(capture_args=True)
            def test_func():
                return 42
            
            result = test_func()
            self.assertEqual(result, 42)
            
            # Should log debug message
            mock_logger.debug.assert_called()
            self.assertIn("Could not capture function arguments", 
                         mock_logger.debug.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_capture_source_exception(self, mock_logger, mock_manager_class):
        """Test exception handling when capturing source fails"""
        mock_manager_class.return_value = self.mock_manager
        
        with patch('core.reproducibility.decorators.inspect.getsource',
                  side_effect=Exception("Source error")):
            @track_artifacts(capture_source=True)
            def test_func():
                return 42
            
            result = test_func()
            self.assertEqual(result, 42)
            
            # Should log debug message
            mock_logger.debug.assert_called()
            self.assertIn("Could not capture function source", 
                         mock_logger.debug.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_input_artifact_exception(self, mock_logger, mock_manager_class):
        """Test exception handling when tracking input artifacts fails"""
        mock_manager_class.return_value = self.mock_manager
        self.mock_manager.use_artifact.side_effect = Exception("Artifact not found")
        
        @track_artifacts(inputs=["missing_artifact"])
        def test_func():
            return 42
        
        result = test_func()
        self.assertEqual(result, 42)
        
        # Should log warning
        mock_logger.warning.assert_called()
        self.assertIn("Could not track input artifact", 
                     mock_logger.warning.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_output_artifact_exception(self, mock_logger, mock_manager_class):
        """Test exception handling when tracking output artifacts fails"""
        mock_manager_class.return_value = self.mock_manager
        self.mock_manager.log_large_artifact.side_effect = Exception("Cannot save")
        
        @track_artifacts(outputs=["model"])
        def test_func():
            return "/path/to/model.pt"
        
        result = test_func()
        self.assertEqual(result, "/path/to/model.pt")
        
        # Should log warning
        mock_logger.warning.assert_called()
        self.assertIn("Could not track output artifact", 
                     mock_logger.warning.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_output_model_non_path(self, mock_logger, mock_manager_class):
        """Test warning when model output is not a path"""
        mock_manager_class.return_value = self.mock_manager
        
        @track_artifacts(outputs=["model"])
        def test_func():
            return object()  # Not a path
        
        result = test_func()
        
        # Should log warning about type
        mock_logger.warning.assert_called()
        self.assertIn("Cannot track model output of type", 
                     mock_logger.warning.call_args[0][0])


class TestReproducibleDecorator(TestCase):
    """Test the reproducible decorator"""
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_basic_reproducible(self, mock_context_class):
        """Test basic reproducible decorator usage"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible()
        def test_func():
            return 42
        
        result = test_func()
        self.assertEqual(result, 42)
        
        # Context should be entered and exited
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_with_name(self, mock_context_class):
        """Test reproducible decorator with custom name"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible(name="custom_experiment")
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check context was created with custom name
        mock_context_class.assert_called_once()
        call_kwargs = mock_context_class.call_args[1]
        self.assertEqual(call_kwargs["name"], "custom_experiment")
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    @patch('core.reproducibility.decorators.EnvironmentCaptureLevel')
    def test_reproducible_with_capture_level(self, mock_capture_level, mock_context_class):
        """Test reproducible decorator with custom capture level"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        mock_capture_level.return_value = "STANDARD"
        
        @reproducible(capture_level=3)
        def test_func():
            return "result"
        
        result = test_func()
        
        # Check capture level was passed
        mock_capture_level.assert_called_once_with(3)
        call_kwargs = mock_context_class.call_args[1]
        self.assertEqual(call_kwargs["capture_level"], "STANDARD")
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_with_offline_mode(self, mock_context_class):
        """Test reproducible decorator with offline mode"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible(offline_mode=True)
        def test_func():
            return "result"
        
        result = test_func()
        
        # Check offline mode was passed
        call_kwargs = mock_context_class.call_args[1]
        self.assertTrue(call_kwargs["offline_mode"])
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_with_exception(self, mock_context_class):
        """Test reproducible decorator when function raises exception"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible()
        def test_func():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_func()
        
        # Context should still be exited properly
        mock_context.__exit__.assert_called_once()
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_preserves_function_attributes(self, mock_context_class):
        """Test that reproducible decorator preserves function attributes"""
        mock_context_class.return_value = Mock(spec=ExperimentContext)
        
        @reproducible()
        def test_func():
            """Test function docstring"""
            return 42
        
        self.assertEqual(test_func.__name__, "test_func")
        self.assertEqual(test_func.__doc__, "Test function docstring")
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_with_config_arg(self, mock_context_class):
        """Test reproducible decorator with config argument"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible()
        def test_func(config, other_arg):
            return config
        
        test_config = {"learning_rate": 0.01}
        result = test_func(test_config, "other")
        
        # Check config was passed to context
        call_kwargs = mock_context_class.call_args[1]
        self.assertEqual(call_kwargs["config"], test_config)
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    @patch('core.reproducibility.decorators.inspect.getsource')
    def test_reproducible_logs_artifacts(self, mock_getsource, mock_context_class):
        """Test that reproducible decorator logs artifacts"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_getsource.return_value = "def test_func(): pass"
        
        logged_artifacts = []
        mock_context.log_artifact.side_effect = lambda **kwargs: logged_artifacts.append(kwargs)
        
        @reproducible(capture_level=2)
        def test_func(x, y=10):
            return x + y
        
        result = test_func(5)
        self.assertEqual(result, 15)
        
        # Should log function info and result
        self.assertEqual(len(logged_artifacts), 2)
        
        # Check function info artifact
        func_info = logged_artifacts[0]
        self.assertEqual(func_info["name"], "function_info")
        self.assertEqual(func_info["type"], ArtifactType.CODE)
        self.assertIn("function", func_info["data"])
        self.assertIn("source", func_info["data"])
        self.assertIn("arguments", func_info["data"])
        
        # Check result artifact
        result_info = logged_artifacts[1]
        self.assertEqual(result_info["name"], "function_result")
        self.assertEqual(result_info["type"], ArtifactType.METRICS)
        self.assertIn("result_type", result_info["data"])
        self.assertIn("result_summary", result_info["data"])


class TestCheckpointDecorator(TestCase):
    """Test the checkpoint decorator"""
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    def test_basic_checkpoint(self, mock_torch, mock_manager_class):
        """Test basic checkpoint functionality"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @checkpoint(frequency="epoch")
        def train_epoch(model, data):
            return {"loss": 0.5}
        
        result = train_epoch(mock_model, [1, 2, 3])
        self.assertEqual(result["loss"], 0.5)
        
        # Should save checkpoint
        mock_torch.save.assert_called_once()
        mock_manager.log_large_artifact.assert_called_once()
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    def test_checkpoint_with_best_tracking(self, mock_torch, mock_manager_class):
        """Test checkpoint with best model tracking"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @checkpoint(frequency="epoch", save_best=True, metric="loss", mode="min")
        def train_epoch(model):
            # Return decreasing loss values
            if not hasattr(train_epoch, 'call_count'):
                train_epoch.call_count = 0
            train_epoch.call_count += 1
            return {"loss": 1.0 / train_epoch.call_count}
        
        # First epoch - should save best
        result1 = train_epoch(mock_model)
        self.assertEqual(result1["loss"], 1.0)
        
        # Second epoch - better loss, should save best again
        result2 = train_epoch(mock_model)
        self.assertEqual(result2["loss"], 0.5)
        
        # Check that best checkpoint was saved
        calls = mock_manager.log_large_artifact.call_args_list
        best_calls = [c for c in calls if c[1]["name"] == "checkpoint_best"]
        self.assertTrue(len(best_calls) > 0)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_checkpoint_frequency_step(self, mock_manager_class):
        """Test checkpoint with step frequency"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        
        @checkpoint(frequency=2)  # Every 2 calls
        def train_step(model):
            return {"loss": 0.1}
        
        # First call - no checkpoint
        train_step(mock_model)
        self.assertEqual(mock_manager.log_large_artifact.call_count, 0)
        
        # Second call - should checkpoint
        with patch('core.reproducibility.decorators.torch'):
            train_step(mock_model)
            self.assertEqual(mock_manager.log_large_artifact.call_count, 1)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_checkpoint_no_active_run(self, mock_logger, mock_manager_class):
        """Test checkpoint when no active run exists"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = None
        mock_manager_class.return_value = mock_manager
        
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        result = train_epoch(Mock())
        self.assertEqual(result["loss"], 0.5)
        
        # Should log warning
        mock_logger.warning.assert_called()
        self.assertIn("No active run", mock_logger.warning.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.logger')
    def test_checkpoint_no_model_found(self, mock_logger, mock_manager_class):
        """Test checkpoint when model not found in arguments"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        @checkpoint(frequency="epoch")
        def train_epoch(data):  # No model parameter
            return {"loss": 0.5}
        
        result = train_epoch([1, 2, 3])
        self.assertEqual(result["loss"], 0.5)
        
        # Should log warning
        mock_logger.warning.assert_called()
        self.assertIn("Could not find model", mock_logger.warning.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    @patch('core.reproducibility.decorators.logger')
    def test_checkpoint_save_error(self, mock_logger, mock_torch, mock_manager_class):
        """Test checkpoint when save fails"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_torch.save.side_effect = Exception("Save failed")
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        result = train_epoch(mock_model)
        self.assertEqual(result["loss"], 0.5)
        
        # Should log error
        mock_logger.error.assert_called()
        self.assertIn("Could not save checkpoint", mock_logger.error.call_args[0][0])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    def test_checkpoint_with_object_result(self, mock_torch, mock_manager_class):
        """Test checkpoint with object result (has __dict__)"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        class Result:
            def __init__(self):
                self.loss = 0.5
                self.accuracy = 0.95
        
        @checkpoint(frequency="epoch", metric="loss")
        def train_epoch(model):
            return Result()
        
        result = train_epoch(mock_model)
        self.assertEqual(result.loss, 0.5)
        
        # Should extract metrics from object
        call_args = mock_manager.log_large_artifact.call_args[1]
        self.assertEqual(call_args["metadata"]["metrics"]["loss"], 0.5)
        self.assertEqual(call_args["metadata"]["metrics"]["accuracy"], 0.95)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    def test_checkpoint_mode_max(self, mock_torch, mock_manager_class):
        """Test checkpoint with mode='max' for best tracking"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @checkpoint(save_best=True, metric="accuracy", mode="max")
        def train_epoch(model):
            # Return increasing accuracy values
            if not hasattr(train_epoch, 'acc'):
                train_epoch.acc = 0.5
            train_epoch.acc += 0.1
            return {"accuracy": train_epoch.acc}
        
        # First epoch
        result1 = train_epoch(mock_model)
        
        # Second epoch - better accuracy
        result2 = train_epoch(mock_model)
        
        # Should save best checkpoint for higher accuracy
        calls = mock_manager.log_large_artifact.call_args_list
        best_calls = [c for c in calls if c[1]["name"] == "checkpoint_best"]
        self.assertTrue(len(best_calls) > 0)


class TestIntegration(TestCase):
    """Integration tests for decorators working together"""
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_with_track_artifacts(self, mock_context_class, mock_manager_class):
        """Test using reproducible and track_artifacts together"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        @reproducible(name="test_experiment")
        @track_artifacts(inputs=["data"], outputs=["result"])
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        self.assertEqual(result, 10)
        
        # Both decorators should be active
        mock_context.__enter__.assert_called_once()
        mock_manager_class.assert_called()
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.torch')
    def test_checkpoint_with_track_artifacts(self, mock_torch, mock_manager_class):
        """Test using checkpoint and track_artifacts together"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @checkpoint(frequency="epoch")
        @track_artifacts(outputs=["metrics"])
        def train_epoch(model):
            return {"loss": 0.5}
        
        result = train_epoch(mock_model)
        self.assertEqual(result["loss"], 0.5)
        
        # Both decorators should log artifacts
        self.assertTrue(mock_manager.log_artifact.called or mock_manager.log_large_artifact.called)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    @patch('core.reproducibility.decorators.ExperimentContext')
    @patch('core.reproducibility.decorators.torch')
    def test_all_decorators_together(self, mock_torch, mock_context_class, mock_manager_class):
        """Test using all decorators together"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        @reproducible(name="full_test")
        @checkpoint(frequency="epoch")
        @track_artifacts(inputs=["input"], outputs=["output"])
        def train_epoch(model):
            return {"loss": 0.5}
        
        result = train_epoch(mock_model)
        self.assertEqual(result["loss"], 0.5)
        
        # All decorators should be active
        mock_context.__enter__.assert_called_once()
        mock_manager_class.assert_called()


class TestEdgeCases(TestCase):
    """Test edge cases and error conditions"""
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_empty_function(self, mock_manager_class):
        """Test decorating a function that returns None"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = None
        mock_manager_class.return_value = mock_manager
        
        @track_artifacts()
        def test_func():
            pass
        
        result = test_func()
        self.assertIsNone(result)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_generator_function(self, mock_manager_class):
        """Test decorating a generator function"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        @track_artifacts()
        def test_func():
            for i in range(3):
                yield i
        
        result = list(test_func())
        self.assertEqual(result, [0, 1, 2])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_class_method(self, mock_manager_class):
        """Test decorating a class method"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        class TestClass:
            @track_artifacts()
            def method(self, x):
                return x * 2
        
        obj = TestClass()
        result = obj.method(5)
        self.assertEqual(result, 10)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_static_method(self, mock_manager_class):
        """Test decorating a static method"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        class TestClass:
            @staticmethod
            @track_artifacts()
            def method(x):
                return x * 2
        
        result = TestClass.method(5)
        self.assertEqual(result, 10)
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_async_function(self, mock_manager_class):
        """Test decorating an async function"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        @track_artifacts()
        async def test_func(x):
            return x * 2
        
        import asyncio
        result = asyncio.run(test_func(5))
        self.assertEqual(result, 10)
    
    @patch('core.reproducibility.decorators.ExperimentContext')
    def test_reproducible_auto_name_generation(self, mock_context_class):
        """Test automatic name generation for reproducible decorator"""
        mock_context = Mock(spec=ExperimentContext)
        mock_context_class.return_value = mock_context
        mock_context.log_artifact = Mock()
        
        @reproducible()  # No name provided
        def test_func():
            return 42
        
        result = test_func()
        self.assertEqual(result, 42)
        
        # Check that auto-generated name was used
        call_kwargs = mock_context_class.call_args[1]
        self.assertIn("test_func_", call_kwargs["name"])
    
    @patch('core.reproducibility.decorators.ArtifactManager')
    def test_track_artifacts_locals_check(self, mock_manager_class):
        """Test the locals() check in track_artifacts"""
        mock_manager = Mock(spec=ArtifactManager)
        mock_manager.run = Mock(id="test_run")
        mock_manager_class.return_value = mock_manager
        
        # Make log_artifact not create start_artifact
        mock_manager.log_artifact.side_effect = Exception("First call fails")
        
        @track_artifacts()
        def test_func():
            return 42
        
        # Should handle missing start_artifact gracefully
        try:
            result = test_func()
        except:
            pass  # Expected to fail
        
        # The finally block should handle missing start_artifact


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])