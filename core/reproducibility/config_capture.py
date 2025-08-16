"""
Configuration and environment capture utilities for reproducibility.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class EnvironmentCaptureLevel(IntEnum):
    """Levels of environment capture detail."""
    BASIC = 1      # pip, python, git commit
    STANDARD = 2   # + git diff, system info
    COMPLETE = 3   # + Docker info if available


class ConfigCapture:
    """Utilities for capturing configuration and environment state."""
    
    @staticmethod
    def capture_environment(
        level: EnvironmentCaptureLevel = EnvironmentCaptureLevel.STANDARD
    ) -> Dict[str, Any]:
        """
        Capture environment information at specified detail level.
        
        Args:
            level: Level of detail to capture
        
        Returns:
            Dictionary containing environment information
        """
        env_data = {}
        
        # Level 1: Basic capture (always included)
        env_data["capture_level"] = level.name
        env_data["timestamp"] = os.environ.get("EPOCHREALTIME", str(os.times()))
        
        # Python environment
        env_data["python"] = {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 paths
        }
        
        # Package versions
        env_data["packages"] = ConfigCapture._capture_packages()
        
        # Git information
        env_data["git"] = ConfigCapture._capture_git_info()
        
        # Warn if git is dirty
        if env_data["git"].get("is_dirty", False):
            logger.warning(
                "⚠️ Git working directory has uncommitted changes! "
                "This may affect reproducibility."
            )
        
        # Level 2: Standard capture
        if level >= EnvironmentCaptureLevel.STANDARD:
            # Add git diff if dirty
            if env_data["git"].get("is_dirty", False):
                env_data["git"]["diff"] = ConfigCapture._get_git_diff()
            
            # System information
            env_data["system"] = ConfigCapture._capture_system_info()
            
            # CUDA information if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                env_data["cuda"] = ConfigCapture._capture_cuda_info()
        
        # Level 3: Complete capture with Docker
        if level >= EnvironmentCaptureLevel.COMPLETE:
            docker_info = ConfigCapture._capture_docker_info()
            if docker_info:
                env_data["docker"] = docker_info
                logger.info("✓ Docker environment captured for maximum reproducibility")
            
            # Environment variables (filtered for security)
            env_data["env_vars"] = ConfigCapture._capture_env_vars()
        
        return env_data
    
    @staticmethod
    def _capture_packages() -> Dict[str, Any]:
        """Capture installed package versions."""
        packages = {}
        
        # Pip packages
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                packages["pip"] = result.stdout.strip().split("\n")[:50]  # First 50
        except Exception as e:
            logger.debug(f"Could not capture pip packages: {e}")
            packages["pip"] = []
        
        # Conda packages (if in conda environment)
        if "CONDA_DEFAULT_ENV" in os.environ:
            try:
                result = subprocess.run(
                    ["conda", "list", "--export"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    packages["conda"] = result.stdout.strip().split("\n")[:50]
                    packages["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
            except Exception as e:
                logger.debug(f"Could not capture conda packages: {e}")
        
        # Key library versions
        key_libs = {}
        
        # PyTorch
        if TORCH_AVAILABLE:
            key_libs["torch"] = torch.__version__
            if hasattr(torch, "version"):
                if hasattr(torch.version, "cuda"):
                    key_libs["torch_cuda"] = torch.version.cuda
                if hasattr(torch.version, "cudnn"):
                    key_libs["torch_cudnn"] = str(torch.backends.cudnn.version())
        
        # Transformers
        try:
            import transformers
            key_libs["transformers"] = transformers.__version__
        except ImportError:
            pass
        
        # PEFT
        try:
            import peft
            key_libs["peft"] = peft.__version__
        except ImportError:
            pass
        
        # WandB
        try:
            import wandb
            key_libs["wandb"] = wandb.__version__
        except ImportError:
            pass
        
        packages["key_libraries"] = key_libs
        
        return packages
    
    @staticmethod
    def _capture_git_info() -> Dict[str, Any]:
        """Capture git repository information."""
        git_info = {}
        
        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["commit"] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["remote"] = result.stdout.strip()
            
            # Check if working directory is clean
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_info["is_dirty"] = bool(result.stdout.strip())
            
            # Get last commit message
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_info["last_commit_message"] = result.stdout.strip()[:200]
            
        except Exception as e:
            logger.debug(f"Could not capture git info: {e}")
            git_info["error"] = str(e)
        
        return git_info
    
    @staticmethod
    def _get_git_diff() -> str:
        """Get git diff for uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                diff = result.stdout
                # Limit diff size
                if len(diff) > 10000:
                    diff = diff[:10000] + "\n... (truncated)"
                return diff
        except Exception as e:
            logger.debug(f"Could not get git diff: {e}")
        
        return ""
    
    @staticmethod
    def _capture_system_info() -> Dict[str, Any]:
        """Capture system information."""
        system_info = {
            "platform": platform.platform(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "python_build": platform.python_build(),
            "python_compiler": platform.python_compiler(),
        }
        
        # CPU information
        system_info["cpu"] = {
            "count": os.cpu_count(),
            "architecture": platform.machine(),
        }
        
        # Memory information
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                system_info["memory"] = {
                    "total_gb": vm.total / (1024**3),
                    "available_gb": vm.available / (1024**3),
                    "percent_used": vm.percent,
                }
                
                # Disk information
                disk = psutil.disk_usage("/")
                system_info["disk"] = {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_used": disk.percent,
                }
            except Exception as e:
                logger.debug(f"Could not capture system metrics: {e}")
        
        # OS-specific information
        if platform.system() == "Linux":
            try:
                # Get Linux distribution info
                result = subprocess.run(
                    ["lsb_release", "-a"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    system_info["linux_distribution"] = result.stdout
            except Exception:
                pass
        
        return system_info
    
    @staticmethod
    def _capture_cuda_info() -> Dict[str, Any]:
        """Capture CUDA and GPU information."""
        cuda_info = {}
        
        if not TORCH_AVAILABLE:
            return cuda_info
        
        try:
            cuda_info["version"] = torch.version.cuda
            cuda_info["cudnn_version"] = torch.backends.cudnn.version()
            cuda_info["cudnn_enabled"] = torch.backends.cudnn.enabled
            cuda_info["device_count"] = torch.cuda.device_count()
            
            # Per-device information
            devices = []
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                }
                
                # Current memory usage
                if torch.cuda.is_available():
                    device_info["allocated_memory_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                    device_info["cached_memory_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
                
                devices.append(device_info)
            
            cuda_info["devices"] = devices
            
            # NVIDIA driver version
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    cuda_info["driver_version"] = result.stdout.strip()
            except Exception:
                pass
            
        except Exception as e:
            logger.debug(f"Could not capture CUDA info: {e}")
            cuda_info["error"] = str(e)
        
        return cuda_info
    
    @staticmethod
    def _capture_docker_info() -> Optional[Dict[str, Any]]:
        """Capture Docker container information if running in Docker."""
        docker_info = {}
        
        # Check if running in Docker
        if not os.path.exists("/.dockerenv") and not os.path.exists("/proc/1/cgroup"):
            return None
        
        try:
            # Get container ID
            with open("/proc/self/cgroup", "r") as f:
                for line in f:
                    if "docker" in line:
                        container_id = line.split("/")[-1].strip()
                        docker_info["container_id"] = container_id[:12]
                        break
            
            # Get Docker version
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                docker_info["docker_version"] = result.stdout.strip()
            
            # Try to get image information using Docker SDK
            if DOCKER_AVAILABLE:
                try:
                    client = docker.from_env()
                    
                    # Get container info
                    if "container_id" in docker_info:
                        container = client.containers.get(docker_info["container_id"])
                        docker_info["image"] = container.image.tags[0] if container.image.tags else container.image.id
                        docker_info["status"] = container.status
                except Exception as e:
                    logger.debug(f"Could not get Docker container info: {e}")
            
            # Check for Dockerfile
            dockerfile_path = Path("Dockerfile")
            if dockerfile_path.exists():
                # Hash Dockerfile for versioning
                with open(dockerfile_path, "rb") as f:
                    dockerfile_hash = hashlib.sha256(f.read()).hexdigest()
                docker_info["dockerfile_hash"] = dockerfile_hash[:16]
            
        except Exception as e:
            logger.debug(f"Could not capture Docker info: {e}")
            docker_info["error"] = str(e)
        
        return docker_info if docker_info else None
    
    @staticmethod
    def _capture_env_vars() -> Dict[str, str]:
        """Capture relevant environment variables (filtered for security)."""
        # Whitelist of safe environment variable patterns
        safe_patterns = [
            "CUDA",
            "TORCH",
            "PYTHON",
            "CONDA",
            "PATH",
            "LD_LIBRARY_PATH",
            "PIXELIS",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "HF_",
            "TRANSFORMERS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NCCL",
            "RANK",
            "WORLD_SIZE",
            "MASTER",
            "NODE",
        ]
        
        # Blacklist of patterns to exclude (for security)
        blacklist_patterns = [
            "KEY",
            "SECRET",
            "TOKEN",
            "PASSWORD",
            "CREDENTIAL",
            "AUTH",
            "API",
        ]
        
        env_vars = {}
        
        for key, value in os.environ.items():
            # Check if key matches safe patterns
            is_safe = any(pattern in key.upper() for pattern in safe_patterns)
            
            # Check if key contains blacklisted patterns
            is_blacklisted = any(pattern in key.upper() for pattern in blacklist_patterns)
            
            if is_safe and not is_blacklisted:
                # Truncate long values
                if len(value) > 200:
                    value = value[:200] + "... (truncated)"
                env_vars[key] = value
        
        return env_vars
    
    @staticmethod
    def validate_environment(
        required_packages: Optional[List[str]] = None,
        min_gpu_memory_gb: float = 0,
        min_cuda_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate that environment meets requirements.
        
        Args:
            required_packages: List of required package names
            min_gpu_memory_gb: Minimum GPU memory required
            min_cuda_version: Minimum CUDA version required
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }
        
        # Check required packages
        if required_packages:
            installed = []
            try:
                result = subprocess.run(
                    ["pip", "freeze"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    installed = [line.split("==")[0].lower() for line in result.stdout.strip().split("\n")]
            except Exception:
                validation["warnings"].append("Could not check installed packages")
            
            for package in required_packages:
                if package.lower() not in installed:
                    validation["errors"].append(f"Required package not found: {package}")
                    validation["is_valid"] = False
        
        # Check GPU memory
        if min_gpu_memory_gb > 0:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    if total_memory < min_gpu_memory_gb:
                        validation["warnings"].append(
                            f"GPU {i} has {total_memory:.1f}GB memory, "
                            f"minimum recommended is {min_gpu_memory_gb}GB"
                        )
            else:
                validation["errors"].append("No CUDA GPUs available")
                validation["is_valid"] = False
        
        # Check CUDA version
        if min_cuda_version and TORCH_AVAILABLE:
            try:
                current_version = torch.version.cuda
                if current_version < min_cuda_version:
                    validation["warnings"].append(
                        f"CUDA version {current_version} is below minimum {min_cuda_version}"
                    )
            except Exception:
                validation["warnings"].append("Could not check CUDA version")
        
        return validation