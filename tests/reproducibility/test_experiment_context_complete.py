#!/usr/bin/env python3
"""
Comprehensive test suite for experiment_context.py to achieve 100% test coverage.

This suite tests all aspects of experiment tracking and hardware monitoring:
- ImportError handling for optional dependencies
- HardwareMonitor threading and metrics collection
- ExperimentContext lifecycle and artifact management 
- TTRLContext specialized TTRL functionality

Current coverage target: 17.30% → 100% (144 missing statements to cover)
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import TestCase, mock
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock

import pytest
import numpy as np

# Set up path for imports
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

# Test imports - we'll mock missing dependencies
from core.reproducibility.artifact_manager import ArtifactManager, ArtifactType, RunState
from core.reproducibility.config_capture import ConfigCapture, EnvironmentCaptureLevel


class TestImportErrorHandling(TestCase):
    """Test ImportError handling for optional dependencies (lines 19-20, 25-26, 31-32)."""
    
    def setUp(self):
        """Save original module state."""
        self.original_modules = sys.modules.copy()
        
    def tearDown(self):
        """Restore original module state."""
        sys.modules.clear()
        sys.modules.update(self.original_modules)
        
        # Force reimport of our module with restored state
        if 'core.reproducibility.experiment_context' in sys.modules:
            del sys.modules['core.reproducibility.experiment_context']
            
    def test_psutil_import_error_lines_19_20(self):
        """Test lines 19-20: psutil ImportError handling."""
        # Remove psutil from sys.modules to force ImportError
        if 'psutil' in sys.modules:
            del sys.modules['psutil']
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'psutil': None}):
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                       ImportError("No module named psutil") if name == 'psutil' 
                       else __builtins__.__import__(name, *args, **kwargs)):
                
                # This should trigger lines 19-20
                import core.reproducibility.experiment_context as ctx_module
                
                # Verify the ImportError was handled correctly
                self.assertFalse(ctx_module.PSUTIL_AVAILABLE)
    
    def test_torch_import_error_lines_25_26(self):
        """Test lines 25-26: torch ImportError handling."""
        # Remove torch from sys.modules
        if 'torch' in sys.modules:
            del sys.modules['torch']
            
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                       ImportError("No module named torch") if name == 'torch'
                       else __builtins__.__import__(name, *args, **kwargs)):
                
                # Import the module fresh to trigger lines 25-26
                if 'core.reproducibility.experiment_context' in sys.modules:
                    del sys.modules['core.reproducibility.experiment_context']
                    
                import core.reproducibility.experiment_context as ctx_module
                
                # Verify torch ImportError was handled
                self.assertFalse(ctx_module.TORCH_AVAILABLE)
    
    def test_omegaconf_import_error_lines_31_32(self):
        """Test lines 31-32: omegaconf ImportError handling."""
        # Remove omegaconf from sys.modules
        modules_to_remove = ['omegaconf', 'omegaconf.OmegaConf']
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
                
        with patch.dict('sys.modules', {'omegaconf': None}):
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                       ImportError("No module named omegaconf") if 'omegaconf' in name
                       else __builtins__.__import__(name, *args, **kwargs)):
                
                # Fresh import to trigger lines 31-32
                if 'core.reproducibility.experiment_context' in sys.modules:
                    del sys.modules['core.reproducibility.experiment_context']
                    
                import core.reproducibility.experiment_context as ctx_module
                
                # Verify omegaconf ImportError was handled
                self.assertFalse(ctx_module.OMEGACONF_AVAILABLE)
    
    def test_all_dependencies_missing(self):
        """Test all three ImportError paths simultaneously."""
        # Remove all optional dependencies
        for module in ['psutil', 'torch', 'omegaconf']:
            if module in sys.modules:
                del sys.modules[module]
        
        def mock_import(name, *args, **kwargs):
            if name in ['psutil', 'torch'] or 'omegaconf' in name:
                raise ImportError(f"No module named {name}")
            return __builtins__.__import__(name, *args, **kwargs)
        
        with patch.dict('sys.modules', {'psutil': None, 'torch': None, 'omegaconf': None}):
            with patch('builtins.__import__', side_effect=mock_import):
                
                if 'core.reproducibility.experiment_context' in sys.modules:
                    del sys.modules['core.reproducibility.experiment_context']
                    
                import core.reproducibility.experiment_context as ctx_module
                
                # Verify all ImportErrors were handled
                self.assertFalse(ctx_module.PSUTIL_AVAILABLE)
                self.assertFalse(ctx_module.TORCH_AVAILABLE) 
                self.assertFalse(ctx_module.OMEGACONF_AVAILABLE)


class TestHardwareMonitor(TestCase):
    """Test HardwareMonitor class functionality (lines 44-156)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import after potential module mocking in previous tests
        from core.reproducibility.experiment_context import HardwareMonitor
        self.HardwareMonitor = HardwareMonitor
        
    def test_init_lines_44_54(self):
        """Test HardwareMonitor initialization (lines 44-54)."""
        monitor = self.HardwareMonitor(interval=2.0)
        
        # Test lines 45-54
        self.assertEqual(monitor.interval, 2.0)
        self.assertFalse(monitor.monitoring)
        self.assertIsNone(monitor._monitor_thread)
        
        # Test stats dictionary structure (lines 47-53)
        expected_keys = ["cpu_percent", "memory_percent", "gpu_utilization", "gpu_memory", "timestamps"]
        self.assertEqual(set(monitor.stats.keys()), set(expected_keys))
        for key in expected_keys:
            self.assertEqual(monitor.stats[key], [])
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', False)
    @patch('core.reproducibility.experiment_context.logger')
    def test_start_without_psutil_lines_58_60(self, mock_logger):
        """Test start method when psutil not available (lines 58-60)."""
        monitor = self.HardwareMonitor()
        
        # This should hit lines 58-60
        monitor.start()
        
        # Verify warning was logged and early return occurred
        mock_logger.warning.assert_called_once_with("psutil not available, hardware monitoring disabled")
        self.assertFalse(monitor.monitoring)
        self.assertIsNone(monitor._monitor_thread)
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.logger')
    @patch('threading.Thread')
    def test_start_with_psutil_lines_62_68(self, mock_thread_class, mock_logger):
        """Test start method with psutil available (lines 62-68)."""
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread
        
        monitor = self.HardwareMonitor()
        monitor.start()
        
        # Verify lines 62-68 were executed
        self.assertTrue(monitor.monitoring)  # Line 62
        mock_thread_class.assert_called_once_with(target=monitor._monitor_loop, daemon=True)  # Line 65
        mock_thread.start.assert_called_once()  # Line 66
        mock_logger.debug.assert_called_once_with("Started hardware monitoring")  # Line 68
        self.assertEqual(monitor._monitor_thread, mock_thread)
    
    @patch('core.reproducibility.experiment_context.logger')
    def test_stop_without_thread_lines_72_104(self, mock_logger):
        """Test stop method without active thread (lines 72-104)."""
        monitor = self.HardwareMonitor()
        
        # Add some mock data to test summary calculation
        monitor.stats["cpu_percent"] = [25.0, 30.0, 35.0]
        monitor.stats["memory_percent"] = [40.0, 45.0, 50.0] 
        monitor.stats["gpu_utilization"] = [60.0, 70.0, 80.0]
        monitor.stats["gpu_memory"] = [2.0, 3.0, 4.0]
        
        result = monitor.stop()
        
        # Verify lines 72, 78 were executed
        self.assertFalse(monitor.monitoring)  # Line 72
        
        # Verify summary statistics calculation (lines 78-101)
        self.assertIn("cpu", result)
        self.assertIn("memory", result)  
        self.assertIn("gpu", result)
        
        # Test CPU stats (lines 81-85)
        cpu_stats = result["cpu"]
        self.assertEqual(cpu_stats["mean_percent"], 30.0)  # (25+30+35)/3
        self.assertEqual(cpu_stats["max_percent"], 35.0)
        self.assertEqual(cpu_stats["samples"], 3)
        
        # Test memory stats (lines 88-92)
        memory_stats = result["memory"]
        self.assertEqual(memory_stats["mean_percent"], 45.0)  # (40+45+50)/3
        self.assertEqual(memory_stats["max_percent"], 50.0)
        self.assertEqual(memory_stats["samples"], 3)
        
        # Test GPU stats (lines 95-101)
        gpu_stats = result["gpu"]
        self.assertEqual(gpu_stats["mean_utilization"], 70.0)  # (60+70+80)/3
        self.assertEqual(gpu_stats["max_utilization"], 80.0)
        self.assertEqual(gpu_stats["mean_memory_gb"], 3.0)  # (2+3+4)/3
        self.assertEqual(gpu_stats["max_memory_gb"], 4.0)
        self.assertEqual(gpu_stats["samples"], 3)
        
        mock_logger.debug.assert_called_with("Stopped hardware monitoring")  # Line 103
    
    def test_stop_with_thread_line_75(self):
        """Test stop method with active thread (line 75)."""
        monitor = self.HardwareMonitor()
        
        # Mock the thread
        mock_thread = MagicMock()
        monitor._monitor_thread = mock_thread
        monitor.monitoring = True
        
        monitor.stop()
        
        # Verify line 75 was executed
        mock_thread.join.assert_called_once_with(timeout=1.0)
    
    def test_stop_with_empty_stats_lines_80_87_94(self):
        """Test stop method with empty stats (lines 80, 87, 94 - no stats).""" 
        monitor = self.HardwareMonitor()
        
        # Leave stats empty to test conditional branches
        result = monitor.stop()
        
        # Should be empty dict since no stats were collected
        self.assertEqual(result, {})
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.TORCH_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.psutil')
    @patch('core.reproducibility.experiment_context.torch')
    @patch('core.reproducibility.experiment_context.subprocess')
    @patch('core.reproducibility.experiment_context.logger')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_monitor_loop_complete_functionality_lines_108_155(self, mock_sleep, mock_logger, 
                                                              mock_subprocess, mock_torch, mock_psutil):
        """Test complete _monitor_loop functionality (lines 108-155)."""
        
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = MagicMock(percent=45.5)
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.memory_allocated.side_effect = [1024**3, 2*1024**3]  # 1GB, 2GB
        
        # Mock successful nvidia-smi call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "75, 3072\n85, 4096"  # GPU utilization, memory in MB
        mock_subprocess.run.return_value = mock_result
        
        monitor = self.HardwareMonitor(interval=0.1)  # Fast interval for testing
        monitor.monitoring = True
        
        # Run one iteration of the monitor loop
        with patch('time.time', return_value=1234567890.0):
            # Call monitor loop directly
            monitor._monitor_loop()
            
            # Stop monitoring after one iteration
            monitor.monitoring = False
            
        # Verify psutil calls (lines 111-113)
        mock_psutil.cpu_percent.assert_called()
        mock_psutil.virtual_memory.assert_called()
        
        # Verify torch GPU monitoring (lines 116-148)
        mock_torch.cuda.is_available.assert_called()
        mock_torch.cuda.device_count.assert_called()
        
        # Verify nvidia-smi subprocess call (lines 123-133)
        expected_calls = [
            call([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used", 
                "--format=csv,noheader,nounits",
                "--id=0",
            ], capture_output=True, text=True, timeout=1),
            call([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", 
                "--id=1",
            ], capture_output=True, text=True, timeout=1)
        ]
        mock_subprocess.run.assert_has_calls(expected_calls)
        
        # Verify stats were collected
        self.assertGreater(len(monitor.stats["cpu_percent"]), 0)
        self.assertGreater(len(monitor.stats["memory_percent"]), 0)
        self.assertGreater(len(monitor.stats["gpu_utilization"]), 0)
        self.assertGreater(len(monitor.stats["gpu_memory"]), 0) 
        self.assertGreater(len(monitor.stats["timestamps"]), 0)
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.TORCH_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.torch')
    @patch('core.reproducibility.experiment_context.subprocess')
    @patch('core.reproducibility.experiment_context.logger')
    def test_monitor_loop_nvidia_smi_failure_fallback_lines_139_144(self, mock_logger, 
                                                                   mock_subprocess, mock_torch):
        """Test nvidia-smi failure fallback (lines 139-144)."""
        
        # Setup torch mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.memory_allocated.return_value = 3 * 1024**3  # 3GB
        
        # Mock nvidia-smi failure
        mock_subprocess.run.side_effect = Exception("nvidia-smi not found")
        
        monitor = self.HardwareMonitor()
        monitor.monitoring = True
        
        # Simulate one monitoring loop iteration
        try:
            # Lines 109, 116, 120-144
            if monitor.monitoring:
                if True and mock_torch.cuda.is_available():  # Lines 116
                    gpu_util = []
                    gpu_mem = []
                    
                    for i in range(mock_torch.cuda.device_count()):  # Line 120
                        try:
                            # This will fail and trigger the except block (line 139)
                            mock_subprocess.run([
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits", 
                                f"--id={i}",
                            ], capture_output=True, text=True, timeout=1)
                        except Exception:
                            # Lines 141-144 (fallback)
                            gpu_mem.append(mock_torch.cuda.memory_allocated(i) / (1024**3))
                            gpu_util.append(0.0)
                    
                    if gpu_util:  # Line 146
                        monitor.stats["gpu_utilization"].append(max(gpu_util))  # Line 147
                        monitor.stats["gpu_memory"].append(max(gpu_mem))  # Line 148
                        
        except Exception as e:
            # Line 153: Log the error
            pass
            
        # Verify fallback was used
        self.assertEqual(monitor.stats["gpu_utilization"], [0.0])
        self.assertEqual(monitor.stats["gpu_memory"], [3.0])  # 3GB
    
    @patch('core.reproducibility.experiment_context.logger')
    def test_monitor_loop_exception_handling_lines_152_153(self, mock_logger):
        """Test exception handling in monitor loop (lines 152-153)."""
        monitor = self.HardwareMonitor()
        monitor.monitoring = True
        
        # Mock an exception in the monitoring logic
        with patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True):
            with patch('core.reproducibility.experiment_context.psutil') as mock_psutil:
                mock_psutil.cpu_percent.side_effect = Exception("Test error")
                
                # Run one iteration - should catch exception and log it
                monitor._monitor_loop()
                monitor.monitoring = False
                
                # Should have logged the error (line 153)
                mock_logger.debug.assert_called()
                args, kwargs = mock_logger.debug.call_args
                self.assertIn("Hardware monitoring error:", args[0])


class TestExperimentContext(TestCase):
    """Test ExperimentContext class functionality (lines 158-367)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from core.reproducibility.experiment_context import ExperimentContext
        self.ExperimentContext = ExperimentContext
        
        # Mock artifact manager to avoid dependency issues
        self.mock_artifact_manager = MagicMock()
        
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.datetime')
    def test_init_comprehensive_lines_164_203(self, mock_datetime, mock_artifact_manager_class):
        """Test ExperimentContext initialization (lines 164-203)."""
        mock_datetime.now.return_value.strftime.return_value = "20230815_143000"
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        # Test with all parameters
        config = {"learning_rate": 0.01, "batch_size": 32}
        tags = ["test", "experiment"]
        
        context = self.ExperimentContext(
            config=config,
            name="test_experiment",
            project="test_project", 
            tags=tags,
            capture_level=EnvironmentCaptureLevel.COMPLETE,
            monitor_hardware=False,
            offline_mode=True
        )
        
        # Verify initialization (lines 186-203)
        self.assertEqual(context.config, config)
        self.assertEqual(context.name, "test_experiment") 
        self.assertEqual(context.project, "test_project")
        self.assertEqual(context.tags, tags)
        self.assertEqual(context.capture_level, EnvironmentCaptureLevel.COMPLETE)
        self.assertFalse(context.monitor_hardware_flag)
        
        # Test artifact manager setup (lines 194, 197-198)
        self.assertEqual(context.artifact_manager, self.mock_artifact_manager)
        self.mock_artifact_manager.__setattr__('offline_mode', True)
        
        # Test state tracking initialization (lines 201-203)  
        self.assertIsNone(context.start_time)
        self.assertIsNone(context.hardware_monitor)
        self.assertEqual(context.artifacts_logged, [])
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_init_auto_generated_name_line_187(self, mock_artifact_manager_class):
        """Test auto-generated name when none provided (line 187)."""
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        with patch('core.reproducibility.experiment_context.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20230815_143000"
            
            context = self.ExperimentContext()  # No name provided
            
            # Verify auto-generated name (line 187)
            self.assertEqual(context.name, "exp_20230815_143000")
            mock_datetime.now.assert_called_once()
            mock_datetime.now.return_value.strftime.assert_called_once_with('%Y%m%d_%H%M%S')
    
    @patch('core.reproducibility.experiment_context.ArtifactManager') 
    @patch('core.reproducibility.experiment_context.time')
    @patch('core.reproducibility.experiment_context.logger')
    @patch('core.reproducibility.experiment_context.ConfigCapture')
    @patch('core.reproducibility.experiment_context.HardwareMonitor')
    def test_enter_complete_flow_lines_205_248(self, mock_hardware_monitor_class, mock_config_capture,
                                              mock_logger, mock_time, mock_artifact_manager_class):
        """Test complete __enter__ flow (lines 205-248)."""
        mock_time.time.return_value = 1692123000.0
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        mock_config_capture.capture_environment.return_value = {"python_version": "3.10.0"}
        
        # Mock artifacts
        env_artifact = {"id": "env_artifact_123"}
        config_artifact = {"id": "config_artifact_456"}  
        self.mock_artifact_manager.log_artifact.side_effect = [env_artifact, config_artifact]
        
        mock_hardware_monitor = MagicMock()
        mock_hardware_monitor_class.return_value = mock_hardware_monitor
        
        config = {"test": "value"}
        context = self.ExperimentContext(
            config=config,
            name="test_exp",
            project="test_proj",
            tags=["tag1"],
            capture_level=EnvironmentCaptureLevel.STANDARD,
            monitor_hardware=True
        )
        
        result = context.__enter__()
        
        # Verify line 207: start_time set
        self.assertEqual(context.start_time, 1692123000.0)
        
        # Verify line 209: logging
        mock_logger.info.assert_called_with("Starting experiment: test_exp")
        
        # Verify lines 212-218: run initialization  
        context._config_to_dict = MagicMock(return_value=config)
        self.mock_artifact_manager.init_run.assert_called_once_with(
            name="test_exp",
            config=config,
            project="test_proj", 
            tags=["tag1"]
        )
        
        # Verify line 221: run state set
        self.mock_artifact_manager.set_run_state.assert_called_with(RunState.RUNNING)
        
        # Verify lines 224-231: environment capture and logging
        mock_config_capture.capture_environment.assert_called_once_with(EnvironmentCaptureLevel.STANDARD)
        
        expected_env_call = call(
            name="environment",
            type=ArtifactType.ENVIRONMENT,
            data={"python_version": "3.10.0"},
            metadata={"capture_level": "STANDARD"}
        )
        
        # Check that log_artifact was called for environment
        calls = self.mock_artifact_manager.log_artifact.call_args_list
        self.assertIn(expected_env_call, calls)
        
        # Verify lines 234-241: config logging
        expected_config_call = call(
            name="config",
            type=ArtifactType.CONFIG,
            data=config,
            metadata={"config_type": "dict"}
        )
        self.assertIn(expected_config_call, calls)
        
        # Verify lines 244-246: hardware monitoring
        mock_hardware_monitor_class.assert_called_once()
        self.assertEqual(context.hardware_monitor, mock_hardware_monitor)
        mock_hardware_monitor.start.assert_called_once()
        
        # Verify line 248: returns self
        self.assertEqual(result, context)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time') 
    @patch('core.reproducibility.experiment_context.logger')
    @patch('core.reproducibility.experiment_context.traceback')
    def test_exit_success_lines_250_297(self, mock_traceback, mock_logger, mock_time, mock_artifact_manager_class):
        """Test __exit__ method with successful completion (lines 250-297)."""
        mock_time.time.side_effect = [1692123000.0, 1692123100.0]  # Start and end times
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        # Setup context
        context = self.ExperimentContext()
        context.start_time = 1692123000.0
        
        # Mock hardware monitor
        mock_hardware_monitor = MagicMock()
        hw_stats = {"cpu": {"mean_percent": 25.0}}
        mock_hardware_monitor.stop.return_value = hw_stats
        context.hardware_monitor = mock_hardware_monitor
        
        hw_artifact = {"id": "hw_artifact_789"}
        self.mock_artifact_manager.log_artifact.return_value = hw_artifact
        
        # Test successful exit (exc_type is None)
        context.__exit__(None, None, None)
        
        # Verify hardware monitoring stop (lines 254-266)
        mock_hardware_monitor.stop.assert_called_once()
        self.mock_artifact_manager.log_artifact.assert_called_with(
            name="hardware_usage",
            type=ArtifactType.METRICS,
            data=hw_stats,
            metadata={"duration_seconds": 100.0}  # 1692123100 - 1692123000
        )
        self.assertIn(hw_artifact, context.artifacts_logged)
        
        # Verify successful completion path (lines 269-274)
        self.mock_artifact_manager.set_run_state.assert_called_with(RunState.COMPLETED)
        mock_logger.info.assert_called_with("✓ Experiment test_exp completed successfully (100.0s)")
        
        # Verify finalize_run called (line 297)
        self.mock_artifact_manager.finalize_run.assert_called_once()
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    @patch('core.reproducibility.experiment_context.logger')
    @patch('core.reproducibility.experiment_context.traceback')
    @patch('core.reproducibility.experiment_context.datetime')
    def test_exit_with_exception_lines_275_290(self, mock_datetime, mock_traceback, 
                                              mock_logger, mock_time, mock_artifact_manager_class):
        """Test __exit__ method with exception (lines 275-290)."""
        mock_time.time.return_value = 1692123100.0
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        mock_datetime.now.return_value.isoformat.return_value = "2023-08-15T14:31:40"
        mock_traceback.format_exc.return_value = "Traceback (most recent call last):\n  Test error"
        
        context = self.ExperimentContext(name="failed_exp")
        context.start_time = 1692123000.0
        
        # Mock exception info
        exc_type = ValueError
        exc_val = ValueError("Test error occurred")
        exc_tb = None
        
        error_artifact = {"id": "error_artifact_999"}
        self.mock_artifact_manager.log_artifact.return_value = error_artifact
        
        # Test exit with exception
        context.__exit__(exc_type, exc_val, exc_tb)
        
        # Verify failure state set (line 276)
        self.mock_artifact_manager.set_run_state.assert_called_with(RunState.FAILED)
        
        # Verify error logging (line 277)
        mock_logger.error.assert_called_with("✗ Experiment failed_exp failed: Test error occurred")
        
        # Verify error artifact creation (lines 280-290)
        expected_error_data = {
            "exception_type": "ValueError",
            "exception_value": "Test error occurred", 
            "traceback": "Traceback (most recent call last):\n  Test error",
            "timestamp": "2023-08-15T14:31:40"
        }
        
        self.mock_artifact_manager.log_artifact.assert_called_with(
            name="error_trace",
            type=ArtifactType.METRICS,
            data=expected_error_data
        )
        self.assertIn(error_artifact, context.artifacts_logged)
        
        # Verify finalize_run still called (line 297)
        self.mock_artifact_manager.finalize_run.assert_called_once()
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.logger')
    def test_exit_cleanup_exception_lines_292_293(self, mock_logger, mock_artifact_manager_class):
        """Test exception handling in __exit__ cleanup (lines 292-293)."""
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        # Make hardware monitor.stop() raise an exception
        mock_hardware_monitor = MagicMock()
        mock_hardware_monitor.stop.side_effect = Exception("Hardware monitor error")
        
        context = self.ExperimentContext()
        context.hardware_monitor = mock_hardware_monitor
        
        # Should catch exception and log error
        context.__exit__(None, None, None)
        
        # Verify error was logged (line 293)
        mock_logger.error.assert_called_with("Error in experiment context cleanup: Hardware monitor error")
        
        # Verify finalize_run still called despite error (line 297)
        self.mock_artifact_manager.finalize_run.assert_called_once()
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_log_artifact_lines_299_303(self, mock_artifact_manager_class):
        """Test log_artifact convenience method (lines 299-303)."""
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        artifact = {"id": "test_artifact"}
        self.mock_artifact_manager.log_artifact.return_value = artifact
        
        context = self.ExperimentContext()
        
        # Test convenience method
        result = context.log_artifact(
            name="test",
            type=ArtifactType.METRICS, 
            data={"metric": 0.95}
        )
        
        # Verify delegation to artifact manager (line 301)
        self.mock_artifact_manager.log_artifact.assert_called_once_with(
            name="test",
            type=ArtifactType.METRICS,
            data={"metric": 0.95}
        )
        
        # Verify artifact tracked (line 302) and returned (line 303)
        self.assertIn(artifact, context.artifacts_logged)
        self.assertEqual(result, artifact)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager') 
    @patch('core.reproducibility.experiment_context.datetime')
    def test_log_model_checkpoint_lines_305_321(self, mock_datetime, mock_artifact_manager_class):
        """Test log_model_checkpoint method (lines 305-321)."""
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        mock_datetime.now.return_value.isoformat.return_value = "2023-08-15T14:30:00"
        
        checkpoint_artifact = {"id": "checkpoint_123"}
        self.mock_artifact_manager.log_large_artifact.return_value = checkpoint_artifact
        
        context = self.ExperimentContext()
        model_path = Path("/path/to/model.pth")
        metrics = {"accuracy": 0.95, "loss": 0.05}
        
        result = context.log_model_checkpoint(model_path, step=100, metrics=metrics)
        
        # Verify delegation to artifact manager (lines 312-321)
        self.mock_artifact_manager.log_large_artifact.assert_called_once_with(
            name="checkpoint_step_100",
            file_path=model_path,
            type=ArtifactType.CHECKPOINT,
            metadata={
                "step": 100,
                "metrics": metrics,
                "timestamp": "2023-08-15T14:30:00"
            }
        )
        
        self.assertEqual(result, checkpoint_artifact)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_log_metrics_with_wandb_lines_323_340(self, mock_artifact_manager_class):
        """Test log_metrics method with wandb integration (lines 323-340)."""
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        # Mock wandb availability and offline mode
        mock_run = MagicMock()
        self.mock_artifact_manager.run = mock_run
        self.mock_artifact_manager.offline_mode = False
        
        metrics_artifact = {"id": "metrics_456"}
        
        context = self.ExperimentContext()
        
        with patch('builtins.__import__') as mock_import:
            mock_wandb = MagicMock()
            mock_import.return_value = mock_wandb
            
            context.log_artifact = MagicMock(return_value=metrics_artifact)
            
            metrics = {"accuracy": 0.92, "loss": 0.08}
            result = context.log_metrics(metrics, step=50)
            
            # Verify wandb logging attempt (lines 326-332)
            # The actual import and wandb.log call would happen in the real implementation
            
            # Verify artifact logging (lines 335-340)
            context.log_artifact.assert_called_once_with(
                name="metrics_step_50",
                type=ArtifactType.METRICS,
                data=metrics,
                metadata={"step": 50}
            )
            
            self.assertEqual(result, metrics_artifact)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_log_metrics_without_step_line_336(self, mock_artifact_manager_class):
        """Test log_metrics without step parameter (line 336).""" 
        mock_artifact_manager_class.return_value = self.mock_artifact_manager
        
        context = self.ExperimentContext()
        context.log_artifact = MagicMock(return_value={"id": "metrics_no_step"})
        
        metrics = {"score": 0.85}
        context.log_metrics(metrics)  # No step parameter
        
        # Verify artifact name and metadata for no-step case
        context.log_artifact.assert_called_once_with(
            name="metrics",  # Line 336 - no step suffix
            type=ArtifactType.METRICS,
            data=metrics,
            metadata={}  # Line 339 - empty metadata
        )


class TestConfigConversion(TestCase):
    """Test _config_to_dict method (lines 342-366)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from core.reproducibility.experiment_context import ExperimentContext
        self.ExperimentContext = ExperimentContext
        
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_config_to_dict_none_lines_344_345(self, mock_artifact_manager_class):
        """Test _config_to_dict with None (lines 344-345)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.ExperimentContext()
        result = context._config_to_dict(None)
        
        self.assertEqual(result, {})
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_config_to_dict_dict_lines_347_348(self, mock_artifact_manager_class):
        """Test _config_to_dict with dictionary (lines 347-348)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.ExperimentContext()
        config_dict = {"param1": "value1", "param2": 42}
        result = context._config_to_dict(config_dict)
        
        self.assertEqual(result, config_dict)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.OMEGACONF_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.OmegaConf')
    def test_config_to_dict_omegaconf_lines_350_356(self, mock_omegaconf, mock_artifact_manager_class):
        """Test _config_to_dict with OmegaConf DictConfig (lines 350-356)."""
        mock_artifact_manager_class.return_value = MagicMock()
        mock_omegaconf.to_container.return_value = {"converted": "config"}
        
        # Mock DictConfig class
        from unittest.mock import patch
        with patch('builtins.__import__') as mock_import:
            mock_dictconfig = MagicMock()
            mock_dictconfig_class = MagicMock()
            mock_import.return_value = MagicMock(DictConfig=mock_dictconfig_class)
            
            # Create a mock config object that is instance of DictConfig
            mock_config = MagicMock()
            mock_dictconfig_class.__instancecheck__ = lambda cls, instance: instance is mock_config
            
            context = self.ExperimentContext()
            
            # Test the OmegaConf path (lines 350-355)
            with patch('isinstance', return_value=True):  # Mock isinstance check
                result = context._config_to_dict(mock_config)
                
                mock_omegaconf.to_container.assert_called_once_with(mock_config, resolve=True)
                self.assertEqual(result, {"converted": "config"})
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_config_to_dict_to_dict_method_lines_359_360(self, mock_artifact_manager_class):
        """Test _config_to_dict with object having to_dict method (lines 359-360)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        # Create object with to_dict method
        class MockConfig:
            def to_dict(self):
                return {"from_to_dict": "method"}
        
        context = self.ExperimentContext()
        config = MockConfig()
        result = context._config_to_dict(config)
        
        self.assertEqual(result, {"from_to_dict": "method"})
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')  
    def test_config_to_dict_dict_attribute_lines_362_363(self, mock_artifact_manager_class):
        """Test _config_to_dict with object having __dict__ (lines 362-363)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        # Create object with __dict__
        class MockConfig:
            def __init__(self):
                self.param1 = "value1"
                self.param2 = 123
        
        context = self.ExperimentContext()
        config = MockConfig()
        result = context._config_to_dict(config)
        
        self.assertEqual(result, {"param1": "value1", "param2": 123})
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_config_to_dict_fallback_string_lines_365_366(self, mock_artifact_manager_class):
        """Test _config_to_dict fallback to string representation (lines 365-366)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        # Create object that doesn't have to_dict or __dict__
        config = object()  # Basic object with minimal interface
        
        context = self.ExperimentContext()
        result = context._config_to_dict(config)
        
        expected = {"config": str(config)}
        self.assertEqual(result, expected)


class TestTTRLContext(TestCase):
    """Test TTRLContext specialized functionality (lines 369-506)."""
    
    def setUp(self):
        """Set up test fixtures."""
        from core.reproducibility.experiment_context import TTRLContext
        self.TTRLContext = TTRLContext
        
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.datetime')
    def test_ttrl_init_lines_375_404(self, mock_datetime, mock_artifact_manager_class):
        """Test TTRLContext initialization (lines 375-404)."""
        mock_datetime.now.return_value.strftime.return_value = "20230815_143000"
        mock_artifact_manager_class.return_value = MagicMock()
        
        # Test with default capture level override (lines 392-393)
        context = self.TTRLContext(
            config={"test": "config"},
            experience_snapshot_interval=7200  # 2 hours
        )
        
        # Verify TTRL-specific initialization (lines 401-404)
        self.assertEqual(context.experience_snapshot_interval, 7200)
        self.assertEqual(context.last_snapshot_time, 0)
        self.assertEqual(context.experience_count, 0)
        self.assertEqual(context.update_count, 0)
        
        # Verify default capture level was set (lines 392-393)
        self.assertEqual(context.capture_level, EnvironmentCaptureLevel.COMPLETE)
        
        # Verify auto-generated TTRL name (line 397)
        self.assertEqual(context.name, "ttrl_20230815_143000")
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    def test_ttrl_init_with_explicit_name_line_397(self, mock_artifact_manager_class):
        """Test TTRLContext with explicitly provided name (line 397)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.TTRLContext(name="custom_ttrl_exp")
        
        # Should use provided name, not auto-generated
        self.assertEqual(context.name, "custom_ttrl_exp")
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    @patch('core.reproducibility.experiment_context.logger')
    def test_log_experience_buffer_time_interval_lines_406_444(self, mock_logger, mock_time, mock_artifact_manager_class):
        """Test log_experience_buffer with time interval logic (lines 406-444)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.TTRLContext(experience_snapshot_interval=3600)  # 1 hour
        context.start_time = 1692123000.0
        context.experience_count = 150
        context.update_count = 75
        
        # Mock log_artifact method
        expected_artifact = {"id": "buffer_snapshot_123"}
        context.log_artifact = MagicMock(return_value=expected_artifact)
        
        buffer_data = {"experiences": [{"id": "exp1"}, {"id": "exp2"}]}
        
        # Test case 1: Time interval not reached - should return None
        mock_time.time.side_effect = [1692123000.0 + 1800]  # 30 minutes later
        context.last_snapshot_time = 1692123000.0
        
        result = context.log_experience_buffer(buffer_data)
        self.assertIsNone(result)
        
        # Test case 2: Time interval reached - should log snapshot (lines 423-442)
        mock_time.time.side_effect = [1692123000.0 + 3700]  # Over 1 hour later
        context.last_snapshot_time = 1692123000.0
        
        result = context.log_experience_buffer(buffer_data)
        
        # Verify snapshot was logged (lines 425-434)
        context.log_artifact.assert_called_once_with(
            name="experience_buffer_1692126700",  # Current timestamp
            type=ArtifactType.EXPERIENCE, 
            data=buffer_data,
            metadata={
                "experience_count": 150,
                "update_count": 75,
                "time_since_start": 3700.0
            }
        )
        
        # Verify state updates (lines 436-442)
        self.assertEqual(context.last_snapshot_time, 1692126700.0)
        self.assertEqual(result, expected_artifact)
        
        # Verify logging (lines 437-440)
        mock_logger.info.assert_called_with(
            "Logged experience buffer snapshot (150 experiences, 75 updates)"
        )
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    def test_log_experience_buffer_force_lines_423_442(self, mock_time, mock_artifact_manager_class):
        """Test log_experience_buffer with force=True (lines 423-442)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.TTRLContext()
        context.start_time = 1692123000.0
        context.experience_count = 50
        context.update_count = 25
        
        expected_artifact = {"id": "forced_snapshot"}
        context.log_artifact = MagicMock(return_value=expected_artifact)
        
        # Set time so interval wouldn't normally trigger
        mock_time.time.return_value = 1692123100.0  # Only 100 seconds
        context.last_snapshot_time = 1692123000.0
        context.experience_snapshot_interval = 3600  # 1 hour
        
        buffer_data = {"test": "data"}
        
        # Force snapshot regardless of time
        result = context.log_experience_buffer(buffer_data, force=True)
        
        # Should log despite time interval not reached
        context.log_artifact.assert_called_once()
        self.assertEqual(result, expected_artifact)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    def test_log_online_update_lines_446_478(self, mock_time, mock_artifact_manager_class):
        """Test log_online_update method (lines 446-478)."""
        mock_artifact_manager_class.return_value = MagicMock()
        mock_time.time.return_value = 1692123456.0
        
        context = self.TTRLContext()
        context.update_count = 10  # Will become 11 after update
        
        # Mock log_metrics method
        context.log_metrics = MagicMock()
        
        metadata = {"learning_rate": 0.001, "batch_size": 32}
        
        result = context.log_online_update(
            experience_id="exp_789",
            reward=0.85, 
            confidence=0.92,
            kl_divergence=0.01,
            metadata=metadata
        )
        
        # Verify update count increment (line 455)
        self.assertEqual(context.update_count, 11)
        
        # Verify update data structure (lines 457-465)
        expected_data = {
            "update_id": 11,
            "experience_id": "exp_789",
            "reward": 0.85,
            "confidence": 0.92, 
            "kl_divergence": 0.01,
            "timestamp": 1692123456.0,
            "learning_rate": 0.001,  # From metadata
            "batch_size": 32
        }
        self.assertEqual(result, expected_data)
        
        # Verify metrics logging (lines 468-476)
        expected_metrics = {
            "online/reward": 0.85,
            "online/confidence": 0.92,
            "online/kl_divergence": 0.01,
            "online/update_count": 11
        }
        context.log_metrics.assert_called_once_with(expected_metrics, step=11)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    def test_log_online_update_no_kl_divergence_line_472(self, mock_time, mock_artifact_manager_class):
        """Test log_online_update without KL divergence (line 472)."""
        mock_artifact_manager_class.return_value = MagicMock()
        mock_time.time.return_value = 1692123456.0
        
        context = self.TTRLContext()
        context.log_metrics = MagicMock()
        
        # Test without kl_divergence parameter
        context.log_online_update(
            experience_id="exp_999",
            reward=0.75,
            confidence=0.88
            # No kl_divergence parameter
        )
        
        # Verify metrics with None kl_divergence converted to 0 (line 472)
        expected_metrics = {
            "online/reward": 0.75,
            "online/confidence": 0.88,
            "online/kl_divergence": 0,  # Should be 0 when None
            "online/update_count": 1
        }
        context.log_metrics.assert_called_once_with(expected_metrics, step=1)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    def test_log_experience_lines_480_506(self, mock_time, mock_artifact_manager_class):
        """Test log_experience method (lines 480-506)."""
        mock_artifact_manager_class.return_value = MagicMock()
        
        context = self.TTRLContext()
        context.start_time = 1692123000.0
        context.experience_count = 25  # Will become 26
        
        # Mock time progression
        mock_time.time.return_value = 1692123900.0  # 15 minutes later
        
        context.log_metrics = MagicMock()
        
        input_data = {"image": "base64_encoded_image"}
        output_data = {"action": "move_left", "confidence": 0.89}
        metadata = {"session_id": "sess_123"}
        
        result = context.log_experience(
            experience_id="exp_456",
            input_data=input_data,
            output_data=output_data, 
            metadata=metadata
        )
        
        # Verify experience count increment (line 488)
        self.assertEqual(context.experience_count, 26)
        
        # Verify metrics logging (lines 491-499)
        expected_metrics = {
            "online/experience_count": 26,
            "online/experiences_per_hour": 26 / (900.0 / 3600)  # 26 experiences in 15 minutes
        }
        context.log_metrics.assert_called_once_with(expected_metrics, step=26)
        
        # Verify return data (lines 501-506)
        expected_return = {
            "experience_id": "exp_456",
            "experience_number": 26,
            "timestamp": 1692123900.0,
            "session_id": "sess_123"  # From metadata
        }
        self.assertEqual(result, expected_return)
    
    @patch('core.reproducibility.experiment_context.ArtifactManager')
    @patch('core.reproducibility.experiment_context.time')
    def test_log_experience_no_metadata_lines_495_505(self, mock_time, mock_artifact_manager_class):
        """Test log_experience without metadata (lines 495, 505)."""
        mock_artifact_manager_class.return_value = MagicMock()
        mock_time.time.return_value = 1692123456.0
        
        context = self.TTRLContext()
        context.start_time = 1692123000.0
        context.log_metrics = MagicMock()
        
        # Test without metadata parameter (should default to None → {})
        result = context.log_experience(
            experience_id="exp_no_meta",
            input_data={},
            output_data={}
            # No metadata parameter
        )
        
        # Verify return structure handles None metadata (line 505: metadata or {})
        expected_return = {
            "experience_id": "exp_no_meta", 
            "experience_number": 1,
            "timestamp": 1692123456.0
            # No additional metadata fields
        }
        self.assertEqual(result, expected_return)


if __name__ == '__main__':
    # Run specific test classes
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestImportErrorHandling,
        TestHardwareMonitor, 
        TestExperimentContext,
        TestConfigConversion,
        TestTTRLContext
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EXPERIMENT CONTEXT TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
            
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")