#!/usr/bin/env python3
"""
Simplified test suite for experiment_context.py to achieve 100% coverage.
Focus on core functionality without complex import manipulation.
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

# Set up path for imports
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.reproducibility.experiment_context import (
    HardwareMonitor, ExperimentContext, TTRLContext, 
    PSUTIL_AVAILABLE, TORCH_AVAILABLE, OMEGACONF_AVAILABLE
)
from core.reproducibility.artifact_manager import ArtifactManager, ArtifactType, RunState
from core.reproducibility.config_capture import ConfigCapture, EnvironmentCaptureLevel


class TestHardwareMonitor(TestCase):
    """Test HardwareMonitor functionality."""
    
    def test_hardware_monitor_init(self):
        """Test HardwareMonitor initialization."""
        monitor = HardwareMonitor(interval=2.0)
        
        self.assertEqual(monitor.interval, 2.0)
        self.assertFalse(monitor.monitoring)
        self.assertIsNotNone(monitor.stats)
        self.assertIsNone(monitor._monitor_thread)
        
        # Test default stats structure
        expected_keys = ["cpu_percent", "memory_percent", "gpu_utilization", "gpu_memory", "timestamps"]
        for key in expected_keys:
            self.assertIn(key, monitor.stats)
            self.assertEqual(monitor.stats[key], [])
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', False)
    def test_start_without_psutil(self):
        """Test start() when psutil is not available."""
        monitor = HardwareMonitor()
        
        with patch('core.reproducibility.experiment_context.logger') as mock_logger:
            monitor.start()
            
            # Should log warning and not start monitoring
            mock_logger.warning.assert_called_once_with(
                "psutil not available, hardware monitoring disabled"
            )
            self.assertFalse(monitor.monitoring)
            self.assertIsNone(monitor._monitor_thread)
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True)
    @patch('threading.Thread')
    def test_start_with_psutil(self, mock_thread):
        """Test start() when psutil is available."""
        monitor = HardwareMonitor()
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        with patch('core.reproducibility.experiment_context.logger') as mock_logger:
            monitor.start()
            
            self.assertTrue(monitor.monitoring)
            mock_thread.assert_called_once_with(target=monitor._monitor_loop, daemon=True)
            mock_thread_instance.start.assert_called_once()
            mock_logger.debug.assert_called_once_with("Started hardware monitoring")
    
    def test_stop_with_empty_stats(self):
        """Test stop() with empty stats."""
        monitor = HardwareMonitor()
        
        with patch('core.reproducibility.experiment_context.logger') as mock_logger:
            summary = monitor.stop()
            
            # Should return empty summary since no stats collected
            self.assertEqual(summary, {})
            self.assertFalse(monitor.monitoring)
            mock_logger.debug.assert_called_once_with("Stopped hardware monitoring")
    
    def test_stop_with_thread_join(self):
        """Test stop() with thread cleanup."""
        monitor = HardwareMonitor()
        mock_thread = Mock()
        monitor._monitor_thread = mock_thread
        monitor.monitoring = True
        
        with patch('core.reproducibility.experiment_context.logger'):
            monitor.stop()
            
            self.assertFalse(monitor.monitoring)
            mock_thread.join.assert_called_once_with(timeout=1.0)
    
    def test_stop_with_cpu_stats(self):
        """Test stop() with CPU statistics."""
        monitor = HardwareMonitor()
        monitor.stats["cpu_percent"] = [50.0, 60.0, 40.0]
        
        summary = monitor.stop()
        
        self.assertIn("cpu", summary)
        self.assertEqual(summary["cpu"]["mean_percent"], 50.0)
        self.assertEqual(summary["cpu"]["max_percent"], 60.0)
        self.assertEqual(summary["cpu"]["samples"], 3)
    
    def test_stop_with_memory_stats(self):
        """Test stop() with memory statistics."""
        monitor = HardwareMonitor()
        monitor.stats["memory_percent"] = [70.0, 80.0, 75.0]
        
        summary = monitor.stop()
        
        self.assertIn("memory", summary)
        self.assertEqual(summary["memory"]["mean_percent"], 75.0)
        self.assertEqual(summary["memory"]["max_percent"], 80.0)
        self.assertEqual(summary["memory"]["samples"], 3)
    
    def test_stop_with_gpu_stats(self):
        """Test stop() with GPU statistics."""
        monitor = HardwareMonitor()
        monitor.stats["gpu_utilization"] = [90.0, 95.0, 85.0]
        monitor.stats["gpu_memory"] = [8.0, 9.0, 7.5]
        
        summary = monitor.stop()
        
        self.assertIn("gpu", summary)
        self.assertEqual(summary["gpu"]["mean_utilization"], 90.0)
        self.assertEqual(summary["gpu"]["max_utilization"], 95.0)
        self.assertEqual(summary["gpu"]["mean_memory_gb"], 8.16666666666666)
        self.assertEqual(summary["gpu"]["max_memory_gb"], 9.0)
        self.assertEqual(summary["gpu"]["samples"], 3)
    
    @patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', True)
    @patch('core.reproducibility.experiment_context.TORCH_AVAILABLE', False)
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('time.sleep')
    def test_monitor_loop_psutil_only(self, mock_sleep, mock_virtual_memory, mock_cpu_percent):
        """Test _monitor_loop with psutil only (no torch)."""
        monitor = HardwareMonitor()
        monitor.monitoring = True
        
        # Mock psutil calls
        mock_cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_virtual_memory.return_value = mock_memory
        
        # Mock sleep to stop the loop after one iteration
        def side_effect(*args):
            monitor.monitoring = False
            
        mock_sleep.side_effect = side_effect
        
        monitor._monitor_loop()
        
        # Verify stats were collected
        self.assertEqual(monitor.stats["cpu_percent"], [50.0])
        self.assertEqual(monitor.stats["memory_percent"], [70.0])
        self.assertEqual(len(monitor.stats["timestamps"]), 1)
    
    @patch('core.reproducibility.experiment_context.TORCH_AVAILABLE', True)
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('subprocess.run')
    @patch('time.sleep')
    def test_monitor_loop_with_gpu_nvidia_smi(self, mock_sleep, mock_subprocess, mock_device_count, mock_cuda_available):
        """Test _monitor_loop with GPU monitoring via nvidia-smi."""
        monitor = HardwareMonitor()
        monitor.monitoring = True
        
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock successful nvidia-smi call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "90, 8192"  # 90% util, 8192 MB memory
        mock_subprocess.return_value = mock_result
        
        # Stop after one iteration
        def side_effect(*args):
            monitor.monitoring = False
            
        mock_sleep.side_effect = side_effect
        
        with patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', False):
            monitor._monitor_loop()
        
        # Verify nvidia-smi was called correctly
        expected_cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used", 
            "--format=csv,noheader,nounits",
            "--id=0"
        ]
        mock_subprocess.assert_called_with(
            expected_cmd,
            capture_output=True,
            text=True,
            timeout=1
        )
        
        # Verify stats were collected
        self.assertEqual(monitor.stats["gpu_utilization"], [90.0])
        self.assertEqual(monitor.stats["gpu_memory"], [8.0])  # Converted from MB to GB
    
    @patch('core.reproducibility.experiment_context.TORCH_AVAILABLE', True)
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.memory_allocated')
    @patch('subprocess.run')
    @patch('time.sleep')
    def test_monitor_loop_gpu_fallback(self, mock_sleep, mock_subprocess, mock_memory_allocated, mock_device_count, mock_cuda_available):
        """Test _monitor_loop GPU fallback when nvidia-smi fails."""
        monitor = HardwareMonitor()
        monitor.monitoring = True
        
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock failed nvidia-smi call
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        
        # Mock torch memory
        mock_memory_allocated.return_value = 8 * (1024**3)  # 8 GB in bytes
        
        # Stop after one iteration
        def side_effect(*args):
            monitor.monitoring = False
            
        mock_sleep.side_effect = side_effect
        
        with patch('core.reproducibility.experiment_context.PSUTIL_AVAILABLE', False):
            monitor._monitor_loop()
        
        # Should fallback to torch memory stats
        self.assertEqual(monitor.stats["gpu_utilization"], [0.0])  # Can't get utilization from torch
        self.assertEqual(monitor.stats["gpu_memory"], [8.0])  # From torch memory_allocated
    
    @patch('time.sleep')
    def test_monitor_loop_exception_handling(self, mock_sleep):
        """Test _monitor_loop handles exceptions gracefully."""
        monitor = HardwareMonitor()
        monitor.monitoring = True
        
        # Make first sleep raise exception, second sleep stops monitoring
        sleep_call_count = 0
        def side_effect(*args):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count == 1:
                raise Exception("Test exception")
            else:
                monitor.monitoring = False
                
        mock_sleep.side_effect = side_effect
        
        with patch('core.reproducibility.experiment_context.logger') as mock_logger:
            monitor._monitor_loop()
            
            # Should log the exception but continue running
            mock_logger.debug.assert_called_with("Hardware monitoring error: Test exception")


class TestExperimentContext(TestCase):
    """Test ExperimentContext functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_basic(self):
        """Test basic ExperimentContext initialization."""
        ctx = ExperimentContext(
            config={"test": "value"},
            name="test_experiment",
            project="test_project",
            tags=["test", "experiment"],
            capture_level=EnvironmentCaptureLevel.MINIMAL,
            monitor_hardware=False,
            offline_mode=True
        )
        
        self.assertEqual(ctx.config, {"test": "value"})
        self.assertEqual(ctx.name, "test_experiment")
        self.assertEqual(ctx.project, "test_project")
        self.assertEqual(ctx.tags, ["test", "experiment"])
        self.assertEqual(ctx.capture_level, EnvironmentCaptureLevel.MINIMAL)
        self.assertFalse(ctx.monitor_hardware_flag)
        self.assertIsInstance(ctx.artifact_manager, ArtifactManager)
        self.assertTrue(ctx.artifact_manager.offline_mode)
    
    def test_init_with_auto_name(self):
        """Test initialization with auto-generated name."""
        with patch('core.reproducibility.experiment_context.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240815_143022"
            
            ctx = ExperimentContext()
            
            self.assertEqual(ctx.name, "exp_20240815_143022")
            self.assertEqual(ctx.tags, [])
    
    @patch('core.reproducibility.experiment_context.ConfigCapture')
    @patch.object(ArtifactManager, 'init_run')
    @patch.object(ArtifactManager, 'set_run_state')
    @patch.object(ArtifactManager, 'log_artifact')
    def test_enter_context(self, mock_log_artifact, mock_set_run_state, mock_init_run, mock_config_capture):
        """Test entering the context manager."""
        # Mock environment capture
        env_data = {"python_version": "3.10.0"}
        mock_config_capture.capture_environment.return_value = env_data
        
        # Mock artifact logging
        env_artifact = Mock()
        config_artifact = Mock()
        mock_log_artifact.side_effect = [env_artifact, config_artifact]
        
        ctx = ExperimentContext(
            config={"param": "value"},
            name="test_exp",
            project="test_project",
            tags=["test"],
            capture_level=EnvironmentCaptureLevel.STANDARD,
            monitor_hardware=False
        )
        
        with patch('time.time', return_value=1234567890):
            with patch('core.reproducibility.experiment_context.logger') as mock_logger:
                result = ctx.__enter__()
        
        # Verify return value
        self.assertEqual(result, ctx)
        
        # Verify initialization
        self.assertEqual(ctx.start_time, 1234567890)
        mock_logger.info.assert_called_once_with("Starting experiment: test_exp")
        
        # Verify artifact manager calls
        mock_init_run.assert_called_once_with(
            name="test_exp",
            config={"param": "value"},
            project="test_project",
            tags=["test"]
        )
        
        mock_set_run_state.assert_called_once_with(RunState.RUNNING)
        
        # Verify environment capture
        mock_config_capture.capture_environment.assert_called_once_with(EnvironmentCaptureLevel.STANDARD)
        
        # Verify artifact logging
        expected_calls = [
            call(
                name="environment",
                type=ArtifactType.ENVIRONMENT,
                data=env_data,
                metadata={"capture_level": "STANDARD"}
            ),
            call(
                name="config", 
                type=ArtifactType.CONFIG,
                data={"param": "value"},
                metadata={"config_type": "dict"}
            )
        ]
        mock_log_artifact.assert_has_calls(expected_calls)
        
        # Verify artifacts were tracked
        self.assertEqual(ctx.artifacts_logged, [env_artifact, config_artifact])
    
    @patch('core.reproducibility.experiment_context.HardwareMonitor')
    def test_enter_with_hardware_monitoring(self, mock_hardware_monitor_class):
        """Test entering context with hardware monitoring enabled."""
        mock_monitor = Mock()
        mock_hardware_monitor_class.return_value = mock_monitor
        
        ctx = ExperimentContext(monitor_hardware=True)
        
        with patch.object(ctx.artifact_manager, 'init_run'), \
             patch.object(ctx.artifact_manager, 'set_run_state'), \
             patch.object(ctx.artifact_manager, 'log_artifact'):
            ctx.__enter__()
        
        # Verify hardware monitor was created and started
        mock_hardware_monitor_class.assert_called_once()
        mock_monitor.start.assert_called_once()
        self.assertEqual(ctx.hardware_monitor, mock_monitor)


class TestConfigConversion(TestCase):
    """Test configuration conversion methods."""
    
    def test_config_to_dict_none(self):
        """Test _config_to_dict with None input."""
        ctx = ExperimentContext()
        result = ctx._config_to_dict(None)
        self.assertEqual(result, {})
    
    def test_config_to_dict_already_dict(self):
        """Test _config_to_dict with dict input.""" 
        ctx = ExperimentContext()
        config = {"key": "value", "nested": {"inner": 123}}
        result = ctx._config_to_dict(config)
        self.assertEqual(result, config)
    
    @patch('core.reproducibility.experiment_context.OMEGACONF_AVAILABLE', True)
    @patch('omegaconf.OmegaConf.to_container')
    def test_config_to_dict_omega_conf(self, mock_to_container):
        """Test _config_to_dict with OmegaConf DictConfig."""
        mock_to_container.return_value = {"converted": "config"}
        
        # Mock DictConfig class
        with patch('omegaconf.DictConfig') as mock_dict_config:
            mock_config = Mock()
            mock_dict_config.__instancecheck__ = lambda cls, obj: obj is mock_config
            
            ctx = ExperimentContext()
            result = ctx._config_to_dict(mock_config)
            
            mock_to_container.assert_called_once_with(mock_config, resolve=True)
            self.assertEqual(result, {"converted": "config"})
    
    def test_config_to_dict_to_dict_method(self):
        """Test _config_to_dict with object having to_dict method."""
        class ConfigObject:
            def to_dict(self):
                return {"from": "to_dict"}
        
        ctx = ExperimentContext()
        config_obj = ConfigObject()
        result = ctx._config_to_dict(config_obj)
        self.assertEqual(result, {"from": "to_dict"})
    
    def test_config_to_dict_dict_attribute(self):
        """Test _config_to_dict with object having __dict__ attribute."""
        class SimpleConfig:
            def __init__(self):
                self.param1 = "value1"
                self.param2 = 42
        
        ctx = ExperimentContext()
        config_obj = SimpleConfig()
        result = ctx._config_to_dict(config_obj)
        expected = {"param1": "value1", "param2": 42}
        self.assertEqual(result, expected)
    
    def test_config_to_dict_fallback_string(self):
        """Test _config_to_dict fallback to string representation."""
        ctx = ExperimentContext()
        config_obj = object()  # Object with no special methods
        result = ctx._config_to_dict(config_obj)
        
        self.assertIsInstance(result, dict)
        self.assertIn("config", result)
        self.assertTrue(result["config"].startswith("<object object at "))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])