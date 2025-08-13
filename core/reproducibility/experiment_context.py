"""
Experiment context managers for automatic state capture and tracking.
"""

import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from .artifact_manager import ArtifactManager, ArtifactType, RunState
from .config_capture import ConfigCapture, EnvironmentCaptureLevel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class HardwareMonitor:
    """Monitor hardware usage during experiments."""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.monitoring = False
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_utilization": [],
            "gpu_memory": [],
            "timestamps": [],
        }
        self._monitor_thread = None
    
    def start(self):
        """Start monitoring in background thread."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, hardware monitoring disabled")
            return
        
        self.monitoring = True
        
        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.debug("Started hardware monitoring")
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected stats."""
        self.monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        # Calculate summary statistics
        summary = {}
        
        if self.stats["cpu_percent"]:
            summary["cpu"] = {
                "mean_percent": sum(self.stats["cpu_percent"]) / len(self.stats["cpu_percent"]),
                "max_percent": max(self.stats["cpu_percent"]),
                "samples": len(self.stats["cpu_percent"]),
            }
        
        if self.stats["memory_percent"]:
            summary["memory"] = {
                "mean_percent": sum(self.stats["memory_percent"]) / len(self.stats["memory_percent"]),
                "max_percent": max(self.stats["memory_percent"]),
                "samples": len(self.stats["memory_percent"]),
            }
        
        if self.stats["gpu_utilization"]:
            summary["gpu"] = {
                "mean_utilization": sum(self.stats["gpu_utilization"]) / len(self.stats["gpu_utilization"]),
                "max_utilization": max(self.stats["gpu_utilization"]),
                "mean_memory_gb": sum(self.stats["gpu_memory"]) / len(self.stats["gpu_memory"]),
                "max_memory_gb": max(self.stats["gpu_memory"]),
                "samples": len(self.stats["gpu_utilization"]),
            }
        
        logger.debug("Stopped hardware monitoring")
        return summary
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory
                if PSUTIL_AVAILABLE:
                    self.stats["cpu_percent"].append(psutil.cpu_percent())
                    self.stats["memory_percent"].append(psutil.virtual_memory().percent)
                
                # GPU if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_util = []
                    gpu_mem = []
                    
                    for i in range(torch.cuda.device_count()):
                        # Get utilization via nvidia-smi if available
                        try:
                            result = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits",
                                    f"--id={i}",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            
                            if result.returncode == 0:
                                parts = result.stdout.strip().split(",")
                                gpu_util.append(float(parts[0]))
                                gpu_mem.append(float(parts[1]) / 1024)  # Convert to GB
                        except Exception:
                            # Fallback to torch memory stats
                            gpu_mem.append(
                                torch.cuda.memory_allocated(i) / (1024**3)
                            )
                            gpu_util.append(0.0)  # Can't get utilization from torch
                    
                    if gpu_util:
                        self.stats["gpu_utilization"].append(max(gpu_util))
                        self.stats["gpu_memory"].append(max(gpu_mem))
                
                self.stats["timestamps"].append(time.time())
                
            except Exception as e:
                logger.debug(f"Hardware monitoring error: {e}")
            
            time.sleep(self.interval)


class ExperimentContext:
    """
    Context manager for comprehensive experiment tracking.
    Automatically captures configuration, environment, and artifacts.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        name: Optional[str] = None,
        project: str = "pixelis",
        tags: Optional[List[str]] = None,
        capture_level: EnvironmentCaptureLevel = EnvironmentCaptureLevel.STANDARD,
        monitor_hardware: bool = True,
        offline_mode: Optional[bool] = None,
    ):
        """
        Initialize experiment context.
        
        Args:
            config: Experiment configuration (Hydra config or dict)
            name: Experiment name (auto-generated if not provided)
            project: WandB project name
            tags: Experiment tags
            capture_level: Level of environment capture detail
            monitor_hardware: Whether to monitor hardware usage
            offline_mode: Force offline mode (None = auto-detect)
        """
        self.config = config
        self.name = name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.project = project
        self.tags = tags or []
        self.capture_level = capture_level
        self.monitor_hardware_flag = monitor_hardware
        
        # Get artifact manager singleton
        self.artifact_manager = ArtifactManager()
        
        # Set offline mode if specified
        if offline_mode is not None:
            self.artifact_manager.offline_mode = offline_mode
        
        # State tracking
        self.start_time = None
        self.hardware_monitor = None
        self.artifacts_logged = []
    
    def __enter__(self):
        """Enter context and capture initial state."""
        self.start_time = time.time()
        
        logger.info(f"Starting experiment: {self.name}")
        
        # Initialize run
        config_dict = self._config_to_dict(self.config)
        self.artifact_manager.init_run(
            name=self.name,
            config=config_dict,
            project=self.project,
            tags=self.tags,
        )
        
        # Set run state
        self.artifact_manager.set_run_state(RunState.RUNNING)
        
        # Capture and log environment
        env_data = ConfigCapture.capture_environment(self.capture_level)
        env_artifact = self.artifact_manager.log_artifact(
            name="environment",
            type=ArtifactType.ENVIRONMENT,
            data=env_data,
            metadata={"capture_level": self.capture_level.name},
        )
        self.artifacts_logged.append(env_artifact)
        
        # Log configuration
        if self.config is not None:
            config_artifact = self.artifact_manager.log_artifact(
                name="config",
                type=ArtifactType.CONFIG,
                data=config_dict,
                metadata={"config_type": type(self.config).__name__},
            )
            self.artifacts_logged.append(config_artifact)
        
        # Start hardware monitoring
        if self.monitor_hardware_flag:
            self.hardware_monitor = HardwareMonitor()
            self.hardware_monitor.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and finalize tracking."""
        try:
            # Stop hardware monitoring
            if self.hardware_monitor:
                hardware_stats = self.hardware_monitor.stop()
                
                # Log hardware usage
                hw_artifact = self.artifact_manager.log_artifact(
                    name="hardware_usage",
                    type=ArtifactType.METRICS,
                    data=hardware_stats,
                    metadata={
                        "duration_seconds": time.time() - self.start_time,
                    },
                )
                self.artifacts_logged.append(hw_artifact)
            
            # Update run state based on exit status
            if exc_type is None:
                self.artifact_manager.set_run_state(RunState.COMPLETED)
                logger.info(
                    f"✓ Experiment {self.name} completed successfully "
                    f"({time.time() - self.start_time:.1f}s)"
                )
            else:
                self.artifact_manager.set_run_state(RunState.FAILED)
                logger.error(f"✗ Experiment {self.name} failed: {exc_val}")
                
                # Log error details
                error_artifact = self.artifact_manager.log_artifact(
                    name="error_trace",
                    type=ArtifactType.METRICS,
                    data={
                        "exception_type": str(exc_type.__name__) if exc_type else None,
                        "exception_value": str(exc_val),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                self.artifacts_logged.append(error_artifact)
        
        except Exception as e:
            logger.error(f"Error in experiment context cleanup: {e}")
        
        finally:
            # Always finalize run
            self.artifact_manager.finalize_run()
    
    def log_artifact(self, *args, **kwargs):
        """Convenience method to log artifact within context."""
        artifact = self.artifact_manager.log_artifact(*args, **kwargs)
        self.artifacts_logged.append(artifact)
        return artifact
    
    def log_model_checkpoint(
        self,
        model_path: Path,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log model checkpoint with metadata."""
        return self.artifact_manager.log_large_artifact(
            name=f"checkpoint_step_{step}",
            file_path=model_path,
            type=ArtifactType.CHECKPOINT,
            metadata={
                "step": step,
                "metrics": metrics or {},
                "timestamp": datetime.now().isoformat(),
            },
        )
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to tracking system."""
        # Log to WandB if available
        if hasattr(self.artifact_manager, "run") and self.artifact_manager.run:
            if not self.artifact_manager.offline_mode:
                try:
                    import wandb
                    wandb.log(metrics, step=step)
                except ImportError:
                    pass
        
        # Also save as artifact for reproducibility
        return self.log_artifact(
            name=f"metrics_step_{step}" if step else "metrics",
            type=ArtifactType.METRICS,
            data=metrics,
            metadata={"step": step} if step else {},
        )
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert various config formats to dictionary."""
        if config is None:
            return {}
        
        if isinstance(config, dict):
            return config
        
        if OMEGACONF_AVAILABLE:
            try:
                from omegaconf import DictConfig
                if isinstance(config, DictConfig):
                    return OmegaConf.to_container(config, resolve=True)
            except Exception:
                pass
        
        # Try to convert to dict
        if hasattr(config, "to_dict"):
            return config.to_dict()
        
        if hasattr(config, "__dict__"):
            return config.__dict__
        
        # Fallback to string representation
        return {"config": str(config)}


class TTRLContext(ExperimentContext):
    """
    Specialized context for Test-Time Reinforcement Learning experiments.
    Adds support for experience buffer snapshots and online metrics.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        name: Optional[str] = None,
        experience_snapshot_interval: float = 3600,  # 1 hour
        **kwargs,
    ):
        """
        Initialize TTRL context.
        
        Args:
            config: Experiment configuration
            name: Experiment name
            experience_snapshot_interval: Seconds between experience buffer snapshots
            **kwargs: Additional arguments for ExperimentContext
        """
        # Default to complete environment capture for TTRL
        if "capture_level" not in kwargs:
            kwargs["capture_level"] = EnvironmentCaptureLevel.COMPLETE
        
        super().__init__(
            config=config,
            name=name or f"ttrl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            **kwargs,
        )
        
        self.experience_snapshot_interval = experience_snapshot_interval
        self.last_snapshot_time = 0
        self.experience_count = 0
        self.update_count = 0
    
    def log_experience_buffer(
        self,
        buffer_data: Any,
        force: bool = False,
    ) -> Optional[Any]:
        """
        Log experience buffer snapshot if interval has passed.
        
        Args:
            buffer_data: Experience buffer data to snapshot
            force: Force snapshot regardless of interval
        
        Returns:
            Artifact metadata if snapshot was taken, None otherwise
        """
        current_time = time.time()
        
        if force or (current_time - self.last_snapshot_time) >= self.experience_snapshot_interval:
            # Log experience buffer snapshot
            snapshot_artifact = self.log_artifact(
                name=f"experience_buffer_{int(current_time)}",
                type=ArtifactType.EXPERIENCE,
                data=buffer_data,
                metadata={
                    "experience_count": self.experience_count,
                    "update_count": self.update_count,
                    "time_since_start": current_time - self.start_time,
                },
            )
            
            self.last_snapshot_time = current_time
            logger.info(
                f"Logged experience buffer snapshot "
                f"({self.experience_count} experiences, {self.update_count} updates)"
            )
            
            return snapshot_artifact
        
        return None
    
    def log_online_update(
        self,
        experience_id: str,
        reward: float,
        confidence: float,
        kl_divergence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log online learning update."""
        self.update_count += 1
        
        update_data = {
            "update_id": self.update_count,
            "experience_id": experience_id,
            "reward": reward,
            "confidence": confidence,
            "kl_divergence": kl_divergence,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        
        # Log to tracking system
        self.log_metrics(
            {
                "online/reward": reward,
                "online/confidence": confidence,
                "online/kl_divergence": kl_divergence or 0,
                "online/update_count": self.update_count,
            },
            step=self.update_count,
        )
        
        return update_data
    
    def log_experience(
        self,
        experience_id: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log individual experience."""
        self.experience_count += 1
        
        # Log experience metrics
        self.log_metrics(
            {
                "online/experience_count": self.experience_count,
                "online/experiences_per_hour": (
                    self.experience_count / ((time.time() - self.start_time) / 3600)
                ),
            },
            step=self.experience_count,
        )
        
        return {
            "experience_id": experience_id,
            "experience_number": self.experience_count,
            "timestamp": time.time(),
            **(metadata or {}),
        }