"""
Complete Test Coverage for decorators.py

This test file ensures 100% coverage of all lines, branches, and edge cases
in the core/reproducibility/decorators.py module.
"""

import unittest
import tempfile
import shutil
import time
import json
import inspect
import hashlib
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict
import sys
import os

sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.reproducibility.decorators import (
    track_artifacts,
    reproducible,
    checkpoint,
    _serialize_arg
)
from core.reproducibility.artifact_manager import ArtifactManager, ArtifactType
from core.reproducibility.experiment_context import ExperimentContext


class TestSerializeArg(unittest.TestCase):
    """Test the _serialize_arg helper function."""
    
    def test_serialize_basic_types(self):
        """Test serialization of basic types."""
        # Test string
        self.assertEqual(_serialize_arg("test"), "test")
        
        # Test int
        self.assertEqual(_serialize_arg(42), 42)
        
        # Test float
        self.assertEqual(_serialize_arg(3.14), 3.14)
        
        # Test bool
        self.assertEqual(_serialize_arg(True), True)
        
        # Test None
        self.assertIsNone(_serialize_arg(None))
    
    def test_serialize_path(self):
        """Test serialization of Path objects."""
        path = Path("/test/path")
        self.assertEqual(_serialize_arg(path), "/test/path")
    
    def test_serialize_list_small(self):
        """Test serialization of small lists."""
        lst = [1, 2, 3]
        result = _serialize_arg(lst)
        self.assertEqual(result, [1, 2, 3])
    
    def test_serialize_list_large(self):
        """Test serialization of large lists (truncation)."""
        lst = list(range(20))
        result = _serialize_arg(lst)
        # Should truncate: first 5 + "..." + last 2
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0:5], [0, 1, 2, 3, 4])
        self.assertEqual(result[5], "...")
        self.assertEqual(result[6:8], [18, 19])
    
    def test_serialize_tuple(self):
        """Test serialization of tuples."""
        tpl = (1, "test", 3.14)
        result = _serialize_arg(tpl)
        self.assertEqual(result, [1, "test", 3.14])
    
    def test_serialize_dict_small(self):
        """Test serialization of small dictionaries."""
        dct = {"a": 1, "b": 2}
        result = _serialize_arg(dct)
        self.assertEqual(result, {"a": 1, "b": 2})
    
    def test_serialize_dict_large(self):
        """Test serialization of large dictionaries (truncation)."""
        dct = {f"key_{i}": i for i in range(20)}
        result = _serialize_arg(dct)
        # Should truncate to first 5 items
        self.assertEqual(len(result), 5)
    
    def test_serialize_nested_structures(self):
        """Test serialization with nested structures."""
        nested = {
            "list": [1, 2, {"inner": "value"}],
            "dict": {"a": 1, "b": [3, 4]},
        }
        result = _serialize_arg(nested)
        self.assertIsInstance(result, dict)
        self.assertIn("list", result)
        self.assertIn("dict", result)
    
    def test_serialize_max_depth(self):
        """Test max depth limitation."""
        deep = {"level1": {"level2": {"level3": {"level4": "value"}}}}
        result = _serialize_arg(deep, max_depth=2)
        # At depth 2, level3 should be stringified
        self.assertEqual(result["level1"]["level2"], str(type({"level3": {"level4": "value"}})))
    
    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        try:
            import numpy as np
            arr = np.array([1, 2, 3])
            result = _serialize_arg(arr)
            self.assertEqual(result["type"], "numpy.ndarray")
            self.assertEqual(result["shape"], (3,))
            self.assertEqual(result["dtype"], "int64")
        except ImportError:
            self.skipTest("NumPy not installed")
    
    def test_serialize_torch_tensor(self):
        """Test serialization of torch tensors."""
        try:
            import torch
            tensor = torch.tensor([1, 2, 3])
            result = _serialize_arg(tensor)
            self.assertEqual(result["type"], "torch.Tensor")
            self.assertEqual(result["shape"], [3])
            self.assertIn("dtype", result)
            self.assertIn("device", result)
        except ImportError:
            self.skipTest("PyTorch not installed")
    
    def test_serialize_object_with_dict(self):
        """Test serialization of objects with __dict__."""
        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
        
        obj = TestObject()
        result = _serialize_arg(obj)
        self.assertEqual(result["type"], "TestObject")
        self.assertEqual(result["attributes"]["attr1"], "value1")
        self.assertEqual(result["attributes"]["attr2"], 42)
    
    def test_serialize_class(self):
        """Test serialization of class objects."""
        class TestClass:
            pass
        
        # Classes should fall back to string representation
        result = _serialize_arg(TestClass)
        self.assertIsInstance(result, str)
    
    def test_serialize_fallback(self):
        """Test fallback to string representation."""
        # Object without __dict__
        obj = object()
        result = _serialize_arg(obj)
        self.assertIsInstance(result, str)
        self.assertIn("object", result)


class TestTrackArtifacts(unittest.TestCase):
    """Test the track_artifacts decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager_patcher = patch('core.reproducibility.decorators.ArtifactManager')
        self.mock_manager_class = self.manager_patcher.start()
        self.mock_manager = MagicMock()
        self.mock_manager_class.return_value = self.mock_manager
        self.mock_manager.run = MagicMock()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.manager_patcher.stop()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_basic_tracking(self):
        """Test basic artifact tracking."""
        @track_artifacts()
        def test_func(x, y):
            return x + y
        
        result = test_func(1, 2)
        self.assertEqual(result, 3)
        
        # Check that ArtifactManager was instantiated
        self.mock_manager_class.assert_called()
    
    def test_capture_args(self):
        """Test capturing function arguments."""
        @track_artifacts(capture_args=True)
        def test_func(x, y=10):
            return x + y
        
        result = test_func(5)
        self.assertEqual(result, 15)
    
    def test_capture_source(self):
        """Test capturing function source code."""
        @track_artifacts(capture_source=True)
        def test_func():
            return "test"
        
        result = test_func()
        self.assertEqual(result, "test")
    
    def test_capture_args_exception(self):
        """Test exception handling in argument capture."""
        with patch('inspect.signature', side_effect=Exception("Test error")):
            @track_artifacts(capture_args=True)
            def test_func(x):
                return x * 2
            
            with patch('core.reproducibility.decorators.logger') as mock_logger:
                result = test_func(5)
                self.assertEqual(result, 10)
                mock_logger.debug.assert_called()
    
    def test_capture_source_exception(self):
        """Test exception handling in source capture."""
        with patch('inspect.getsource', side_effect=Exception("Test error")):
            @track_artifacts(capture_source=True)
            def test_func():
                return "test"
            
            with patch('core.reproducibility.decorators.logger') as mock_logger:
                result = test_func()
                self.assertEqual(result, "test")
                mock_logger.debug.assert_called()
    
    def test_track_input_artifacts(self):
        """Test tracking input artifacts."""
        self.mock_manager.use_artifact.return_value = Mock(name="dataset", version="v1")
        
        @track_artifacts(inputs=["dataset:v1", "model"])
        def test_func():
            return "result"
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = test_func()
            self.assertEqual(result, "result")
            
            # Check use_artifact was called
            self.assertEqual(self.mock_manager.use_artifact.call_count, 2)
            mock_logger.info.assert_called()
    
    def test_track_input_artifacts_exception(self):
        """Test exception handling in input artifact tracking."""
        self.mock_manager.use_artifact.side_effect = Exception("Artifact not found")
        
        @track_artifacts(inputs=["missing:v1"])
        def test_func():
            return "result"
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = test_func()
            self.assertEqual(result, "result")
            mock_logger.warning.assert_called()
    
    def test_function_execution_success(self):
        """Test tracking successful function execution."""
        self.mock_manager.run = True
        mock_artifact = Mock(name="start", version="v1")
        self.mock_manager.log_artifact.return_value = mock_artifact
        
        @track_artifacts()
        def test_func():
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
        
        # Check artifacts were logged
        self.mock_manager.log_artifact.assert_called()
    
    def test_function_execution_failure(self):
        """Test tracking failed function execution."""
        self.mock_manager.run = True
        
        @track_artifacts()
        def test_func():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_func()
        
        # Check completion artifact was still logged
        self.mock_manager.log_artifact.assert_called()
    
    def test_track_output_model(self):
        """Test tracking model output artifacts."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["model"])
        def test_func():
            return Path("/path/to/model.pt")
        
        result = test_func()
        
        # Check model artifact was logged
        self.mock_manager.log_large_artifact.assert_called()
    
    def test_track_output_model_invalid(self):
        """Test tracking invalid model output."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["model"])
        def test_func():
            return {"model": "data"}  # Not a path
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = test_func()
            mock_logger.warning.assert_called()
    
    def test_track_output_metrics(self):
        """Test tracking metrics output."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["metrics"])
        def test_func():
            return {"loss": 0.5, "accuracy": 0.95}
        
        result = test_func()
        
        # Check metrics artifact was logged
        self.mock_manager.log_artifact.assert_called()
    
    def test_track_output_dataset_path(self):
        """Test tracking dataset output as path."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["dataset"])
        def test_func():
            return Path("/path/to/dataset.csv")
        
        result = test_func()
        
        # Check dataset artifact was logged
        self.mock_manager.log_large_artifact.assert_called()
    
    def test_track_output_dataset_data(self):
        """Test tracking dataset output as data."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["dataset"])
        def test_func():
            return [1, 2, 3, 4, 5]
        
        result = test_func()
        
        # Check dataset artifact was logged
        self.mock_manager.log_artifact.assert_called()
    
    def test_track_output_generic(self):
        """Test tracking generic output."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["custom"])
        def test_func():
            return {"custom": "data"}
        
        result = test_func()
        
        # Check generic artifact was logged
        self.mock_manager.log_artifact.assert_called()
    
    def test_track_multiple_outputs_tuple(self):
        """Test tracking multiple outputs as tuple."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["model", "metrics"])
        def test_func():
            return Path("/model.pt"), {"loss": 0.5}
        
        result = test_func()
        
        # Check both artifacts were logged
        self.mock_manager.log_large_artifact.assert_called()
        self.mock_manager.log_artifact.assert_called()
    
    def test_track_multiple_outputs_dict(self):
        """Test tracking multiple outputs as dict."""
        self.mock_manager.run = True
        
        @track_artifacts(outputs=["metrics", "dataset"])
        def test_func():
            return {"metrics": {"loss": 0.5}, "dataset": [1, 2, 3]}
        
        result = test_func()
        
        # Check artifacts were logged
        self.assertEqual(self.mock_manager.log_artifact.call_count, 2)
    
    def test_track_output_exception(self):
        """Test exception handling in output tracking."""
        self.mock_manager.run = True
        self.mock_manager.log_artifact.side_effect = Exception("Log error")
        
        @track_artifacts(outputs=["metrics"])
        def test_func():
            return {"loss": 0.5}
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = test_func()
            self.assertEqual(result, {"loss": 0.5})
            mock_logger.warning.assert_called()
    
    def test_no_active_run(self):
        """Test behavior when no active run exists."""
        self.mock_manager.run = None
        
        @track_artifacts(inputs=["dataset"], outputs=["model"])
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")


class TestReproducible(unittest.TestCase):
    """Test the reproducible decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.context_patcher = patch('core.reproducibility.decorators.ExperimentContext')
        self.mock_context_class = self.context_patcher.start()
        self.mock_context = MagicMock()
        self.mock_context_class.return_value.__enter__ = Mock(return_value=self.mock_context)
        self.mock_context_class.return_value.__exit__ = Mock(return_value=None)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.context_patcher.stop()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_basic_reproducible(self):
        """Test basic reproducible decorator."""
        @reproducible()
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check ExperimentContext was created
        self.mock_context_class.assert_called()
    
    def test_reproducible_with_name(self):
        """Test reproducible with custom name."""
        @reproducible(name="test_experiment")
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check name was passed
        call_args = self.mock_context_class.call_args
        self.assertEqual(call_args.kwargs["name"], "test_experiment")
    
    def test_reproducible_auto_name(self):
        """Test reproducible with auto-generated name."""
        @reproducible()
        def test_func():
            return "result"
        
        with patch('time.time', return_value=1234567890):
            result = test_func()
            self.assertEqual(result, "result")
            
            # Check auto-generated name
            call_args = self.mock_context_class.call_args
            self.assertIn("test_func", call_args.kwargs["name"])
    
    def test_reproducible_with_config(self):
        """Test reproducible extracting config from arguments."""
        @reproducible()
        def test_func(config, x=10):
            return config["value"] + x
        
        test_config = {"value": 5}
        result = test_func(test_config)
        self.assertEqual(result, 15)
        
        # Check config was passed to context
        call_args = self.mock_context_class.call_args
        self.assertEqual(call_args.kwargs["config"], test_config)
    
    def test_reproducible_with_alt_config_names(self):
        """Test reproducible with alternative config parameter names."""
        @reproducible()
        def test_func(cfg, x=10):
            return cfg["value"] + x
        
        test_config = {"value": 5}
        result = test_func(test_config)
        self.assertEqual(result, 15)
        
        # Check config was found
        call_args = self.mock_context_class.call_args
        self.assertEqual(call_args.kwargs["config"], test_config)
    
    def test_reproducible_capture_levels(self):
        """Test different capture levels."""
        @reproducible(capture_level=3)
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check capture level
        call_args = self.mock_context_class.call_args
        self.assertIsNotNone(call_args.kwargs["capture_level"])
    
    def test_reproducible_offline_mode(self):
        """Test offline mode setting."""
        @reproducible(offline_mode=True)
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check offline mode
        call_args = self.mock_context_class.call_args
        self.assertTrue(call_args.kwargs["offline_mode"])
    
    def test_reproducible_logging(self):
        """Test artifact logging within reproducible."""
        @reproducible(capture_level=2)
        def test_func(x, y):
            return x + y
        
        result = test_func(1, 2)
        self.assertEqual(result, 3)
        
        # Check artifacts were logged
        self.assertEqual(self.mock_context.log_artifact.call_count, 2)
    
    def test_reproducible_source_capture(self):
        """Test source code capture with high capture level."""
        @reproducible(capture_level=2)
        def test_func():
            return "result"
        
        with patch('inspect.getsource', return_value="def test_func():\n    return 'result'"):
            result = test_func()
            self.assertEqual(result, "result")
            
            # Check source was captured
            call_args = self.mock_context.log_artifact.call_args_list[0]
            self.assertIn("source", call_args.kwargs["data"])
    
    def test_reproducible_no_source_capture(self):
        """Test no source capture with low capture level."""
        @reproducible(capture_level=1)
        def test_func():
            return "result"
        
        result = test_func()
        self.assertEqual(result, "result")
        
        # Check source was not captured
        call_args = self.mock_context.log_artifact.call_args_list[0]
        self.assertIsNone(call_args.kwargs["data"]["source"])


class TestCheckpoint(unittest.TestCase):
    """Test the checkpoint decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager_patcher = patch('core.reproducibility.decorators.ArtifactManager')
        self.mock_manager_class = self.manager_patcher.start()
        self.mock_manager = MagicMock()
        self.mock_manager_class.return_value = self.mock_manager
        self.mock_manager.run = MagicMock()
        
        # Create checkpoints directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.manager_patcher.stop()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
    
    def test_checkpoint_epoch_frequency(self):
        """Test checkpoint with epoch frequency."""
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            result = train_epoch(mock_model)
            self.assertEqual(result["loss"], 0.5)
            
            # Check checkpoint was saved
            mock_torch.save.assert_called()
    
    def test_checkpoint_step_frequency(self):
        """Test checkpoint with step frequency."""
        @checkpoint(frequency="step")
        def train_step(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            result = train_step(mock_model)
            self.assertEqual(result["loss"], 0.5)
            
            # Check checkpoint was saved
            mock_torch.save.assert_called()
    
    def test_checkpoint_integer_frequency(self):
        """Test checkpoint with integer frequency."""
        @checkpoint(frequency=3)
        def train_step(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            # Call multiple times
            for i in range(5):
                result = train_step(mock_model)
            
            # Check checkpoint was saved on 3rd call
            self.assertEqual(mock_torch.save.call_count, 1)
    
    def test_checkpoint_no_active_run(self):
        """Test checkpoint with no active run."""
        self.mock_manager.run = None
        
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = train_epoch(mock_model)
            self.assertEqual(result["loss"], 0.5)
            mock_logger.warning.assert_called_with("No active run, skipping checkpoint")
    
    def test_checkpoint_no_model_found(self):
        """Test checkpoint when model not found in arguments."""
        @checkpoint(frequency="epoch")
        def train_epoch(data):  # No model parameter
            return {"loss": 0.5}
        
        with patch('core.reproducibility.decorators.logger') as mock_logger:
            result = train_epoch("data")
            self.assertEqual(result["loss"], 0.5)
            mock_logger.warning.assert_called_with("Could not find model in function arguments")
    
    def test_checkpoint_find_model_alt_names(self):
        """Test finding model with alternative parameter names."""
        @checkpoint(frequency="epoch")
        def train_epoch(net):  # Alternative name
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            result = train_epoch(mock_model)
            mock_torch.save.assert_called()
    
    def test_checkpoint_metrics_from_dict(self):
        """Test extracting metrics from dict result."""
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5, "accuracy": 0.95}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch'):
            result = train_epoch(mock_model)
            self.assertEqual(result["loss"], 0.5)
            self.assertEqual(result["accuracy"], 0.95)
    
    def test_checkpoint_metrics_from_object(self):
        """Test extracting metrics from object with __dict__."""
        class Result:
            def __init__(self):
                self.loss = 0.5
                self.accuracy = 0.95
        
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return Result()
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch'):
            result = train_epoch(mock_model)
            self.assertEqual(result.loss, 0.5)
    
    def test_checkpoint_save_exception(self):
        """Test exception handling during checkpoint save."""
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            mock_torch.save.side_effect = Exception("Save error")
            
            with patch('core.reproducibility.decorators.logger') as mock_logger:
                result = train_epoch(mock_model)
                self.assertEqual(result["loss"], 0.5)
                mock_logger.error.assert_called()
    
    def test_checkpoint_save_best_min(self):
        """Test saving best checkpoint with min mode."""
        @checkpoint(frequency="epoch", save_best=True, metric="loss", mode="min")
        def train_epoch(model):
            return {"loss": train_epoch.loss_value}
        
        train_epoch.loss_value = 1.0
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            # First call with high loss
            train_epoch.loss_value = 1.0
            result1 = train_epoch(mock_model)
            
            # Second call with lower loss (better)
            train_epoch.loss_value = 0.5
            result2 = train_epoch(mock_model)
            
            # Check best checkpoint was saved
            save_calls = mock_torch.save.call_args_list
            self.assertTrue(any("best" in str(call) for call in save_calls))
    
    def test_checkpoint_save_best_max(self):
        """Test saving best checkpoint with max mode."""
        @checkpoint(frequency="epoch", save_best=True, metric="accuracy", mode="max")
        def train_epoch(model):
            return {"accuracy": train_epoch.acc_value}
        
        train_epoch.acc_value = 0.5
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            # First call with low accuracy
            train_epoch.acc_value = 0.5
            result1 = train_epoch(mock_model)
            
            # Second call with higher accuracy (better)
            train_epoch.acc_value = 0.95
            result2 = train_epoch(mock_model)
            
            # Check best checkpoint was saved
            save_calls = mock_torch.save.call_args_list
            self.assertTrue(any("best" in str(call) for call in save_calls))
    
    def test_checkpoint_save_best_missing_metric(self):
        """Test save_best when metric is missing from result."""
        @checkpoint(frequency="epoch", save_best=True, metric="val_loss", mode="min")
        def train_epoch(model):
            return {"loss": 0.5}  # Missing val_loss
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            result = train_epoch(mock_model)
            
            # Regular checkpoint saved, but not best
            save_calls = mock_torch.save.call_args_list
            self.assertFalse(any("best" in str(call) for call in save_calls))
    
    def test_checkpoint_save_best_exception(self):
        """Test exception handling when saving best checkpoint."""
        @checkpoint(frequency="epoch", save_best=True, metric="loss", mode="min")
        def train_epoch(model):
            return {"loss": 0.1}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch('core.reproducibility.decorators.torch') as mock_torch:
            # First save succeeds, second (best) fails
            mock_torch.save.side_effect = [None, Exception("Save error")]
            
            with patch('core.reproducibility.decorators.logger') as mock_logger:
                result = train_epoch(mock_model)
                self.assertEqual(result["loss"], 0.1)
                # Check error was logged
                error_calls = [call for call in mock_logger.error.call_args_list 
                              if "best checkpoint" in str(call)]
                self.assertTrue(len(error_calls) > 0)
    
    def test_checkpoint_import_torch_error(self):
        """Test handling of torch import error."""
        @checkpoint(frequency="epoch")
        def train_epoch(model):
            return {"loss": 0.5}
        
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weights": "data"}
        
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError("No torch")):
                with patch('core.reproducibility.decorators.logger') as mock_logger:
                    result = train_epoch(mock_model)
                    self.assertEqual(result["loss"], 0.5)
                    mock_logger.error.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for decorators."""
    
    def test_combined_decorators(self):
        """Test combining multiple decorators."""
        with patch('core.reproducibility.decorators.ArtifactManager') as mock_manager_class:
            with patch('core.reproducibility.decorators.ExperimentContext') as mock_context_class:
                mock_manager = MagicMock()
                mock_manager.run = True
                mock_manager_class.return_value = mock_manager
                
                mock_context = MagicMock()
                mock_context_class.return_value.__enter__ = Mock(return_value=mock_context)
                mock_context_class.return_value.__exit__ = Mock(return_value=None)
                
                @reproducible(name="test_exp")
                @track_artifacts(outputs=["metrics"])
                def train():
                    return {"loss": 0.5}
                
                result = train()
                self.assertEqual(result["loss"], 0.5)
                
                # Check both decorators worked
                mock_manager_class.assert_called()
                mock_context_class.assert_called()
    
    def test_nested_serialization(self):
        """Test deeply nested structure serialization."""
        deep_structure = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": "value"
                        }
                    }
                }
            }
        }
        
        result = _serialize_arg(deep_structure, max_depth=3)
        # At depth 3, level4 should be stringified
        self.assertIsInstance(result["level1"]["level2"]["level3"], str)
    
    def test_serialize_without_numpy(self):
        """Test serialization when numpy is not available."""
        with patch.dict('sys.modules', {'numpy': None}):
            # Should not raise error, just skip numpy handling
            result = _serialize_arg([1, 2, 3])
            self.assertEqual(result, [1, 2, 3])
    
    def test_serialize_without_torch(self):
        """Test serialization when torch is not available."""
        with patch.dict('sys.modules', {'torch': None}):
            # Should not raise error, just skip torch handling
            result = _serialize_arg({"data": "value"})
            self.assertEqual(result, {"data": "value"})


if __name__ == "__main__":
    unittest.main()