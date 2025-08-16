"""
Artifact Manager for comprehensive experiment tracking and versioning.
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    import torch.distributed as dist
    TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    TORCH_DISTRIBUTED_AVAILABLE = False

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RunState(Enum):
    """Experiment run states for atomic tracking."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ArtifactType(Enum):
    """Standard artifact types for consistent naming."""
    DATASET = "dataset"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    CONFIG = "config"
    METRICS = "metrics"
    EVALUATION = "evaluation"
    ENVIRONMENT = "environment"
    CODE = "code"
    EXPERIENCE = "experience"


@dataclass
class ArtifactMetadata:
    """Metadata for tracked artifacts."""
    name: str
    type: ArtifactType
    version: str
    hash: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: str = None
    run_id: Optional[str] = None
    parent_artifacts: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.parent_artifacts is None:
            self.parent_artifacts = []
        if self.metadata is None:
            self.metadata = {}


class StorageBackend:
    """Abstract base class for artifact storage backends."""
    
    def exists(self, content_hash: str) -> bool:
        raise NotImplementedError
    
    def upload(self, file_path: Path, content_hash: str) -> str:
        raise NotImplementedError
    
    def download(self, content_hash: str, target_path: Path) -> Path:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage for offline mode."""
    
    def __init__(self, cache_dir: Union[str, Path] = "./artifact_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.content_dir = self.cache_dir / "content"
        self.content_dir.mkdir(exist_ok=True)
    
    def exists(self, content_hash: str) -> bool:
        return (self.content_dir / content_hash).exists()
    
    def upload(self, file_path: Path, content_hash: str) -> str:
        """Copy file to content-addressable storage."""
        target = self.content_dir / content_hash
        if not target.exists():
            import shutil
            shutil.copy2(file_path, target)
            logger.debug(f"Stored artifact content: {content_hash[:8]}...")
        return str(target)
    
    def download(self, content_hash: str, target_path: Path) -> Path:
        """Retrieve file from content-addressable storage."""
        source = self.content_dir / content_hash
        if not source.exists():
            raise FileNotFoundError(f"Artifact content not found: {content_hash}")
        
        import shutil
        shutil.copy2(source, target_path)
        return target_path


class WandBStorageBackend(StorageBackend):
    """WandB artifact storage backend."""
    
    def __init__(self):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        self.run = None
    
    def set_run(self, run):
        """Set the current WandB run."""
        self.run = run
    
    def exists(self, content_hash: str) -> bool:
        """Check if artifact exists in WandB."""
        # For WandB, we typically always upload and let it dedupe
        return False
    
    def upload(self, file_path: Path, content_hash: str) -> str:
        """Upload file as WandB artifact."""
        if self.run is None:
            raise RuntimeError("WandB run not initialized")
        
        artifact = wandb.Artifact(
            name=f"file_{content_hash[:16]}",
            type="binary",
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)
        return artifact.name
    
    def download(self, content_hash: str, target_path: Path) -> Path:
        """Download artifact from WandB."""
        if self.run is None:
            raise RuntimeError("WandB run not initialized")
        
        artifact = self.run.use_artifact(f"file_{content_hash[:16]}:latest")
        artifact_dir = artifact.download()
        
        # Move to target location
        import shutil
        source_file = Path(artifact_dir) / file_path.name
        shutil.move(str(source_file), str(target_path))
        return target_path


class ArtifactManager:
    """
    Singleton manager for all artifact tracking and versioning.
    Thread-safe and distributed-training aware.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.run = None
            self.run_state = None
            self.run_id = None
            self.artifact_cache = {}
            self.lineage_graph = {}
            
            # Offline mode detection
            self.offline_mode = os.environ.get("PIXELIS_OFFLINE_MODE", "false").lower() == "true"
            
            # Distributed training detection
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
            
            if TORCH_DISTRIBUTED_AVAILABLE:
                try:
                    if dist.is_initialized():
                        self.is_distributed = True
                        self.rank = dist.get_rank()
                        self.world_size = dist.get_world_size()
                        logger.info(f"Distributed mode: rank {self.rank}/{self.world_size}")
                except Exception:
                    pass
            
            # Initialize storage backend
            self.storage_backend = self._init_storage_backend()
            
            # Local metadata cache for offline mode
            self.local_metadata_cache = []
            
    def _init_storage_backend(self) -> StorageBackend:
        """Initialize appropriate storage backend."""
        if self.offline_mode or not WANDB_AVAILABLE:
            logger.info("Using local storage backend for artifacts")
            return LocalStorageBackend()
        else:
            logger.info("Using WandB storage backend for artifacts")
            return WandBStorageBackend()
    
    def init_run(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        project: str = "pixelis",
        tags: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Initialize a new experimental run."""
        with self._lock:
            # Only rank 0 initializes run in distributed mode
            if self.is_distributed and self.rank != 0:
                logger.debug(f"Rank {self.rank} skipping run initialization")
                return None
            
            self.run_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self.offline_mode or not WANDB_AVAILABLE:
                # Offline mode: create local run directory
                run_dir = Path("./runs") / self.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # Save initial config
                if config:
                    with open(run_dir / "config.json", "w") as f:
                        json.dump(config, f, indent=2)
                
                logger.info(f"Started offline run: {self.run_id}")
                self.run = {"id": self.run_id, "dir": str(run_dir)}
                
            else:
                # Online mode: initialize WandB
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    tags=tags or [],
                )
                self.run_id = self.run.id
                
                # Set WandB run for storage backend
                if isinstance(self.storage_backend, WandBStorageBackend):
                    self.storage_backend.set_run(self.run)
                
                logger.info(f"Started WandB run: {self.run_id}")
            
            self.run_state = RunState.RUNNING
            return self.run
    
    def log_artifact(
        self,
        name: str,
        type: Union[str, ArtifactType],
        data: Optional[Any] = None,
        file_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_artifacts: Optional[List[str]] = None,
    ) -> ArtifactMetadata:
        """Log an artifact with automatic versioning and lineage tracking."""
        with self._lock:
            # Convert type to enum if string
            if isinstance(type, str):
                type = ArtifactType[type.upper()]
            
            # Generate version
            version = self._generate_version(name, type)
            
            # Create metadata object
            artifact_meta = ArtifactMetadata(
                name=name,
                type=type,
                version=version,
                run_id=self.run_id,
                parent_artifacts=parent_artifacts or [],
                metadata=metadata or {},
            )
            
            # Handle file artifacts
            if file_path:
                file_path = Path(file_path)
                if file_path.exists():
                    # Compute hash
                    artifact_meta.hash = self._compute_file_hash(file_path)
                    artifact_meta.size_bytes = file_path.stat().st_size
                    
                    # Upload if doesn't exist
                    if not self.storage_backend.exists(artifact_meta.hash):
                        self.storage_backend.upload(file_path, artifact_meta.hash)
                    
                    # Log file metadata
                    logger.info(
                        f"Logged file artifact: {name} v{version} "
                        f"({artifact_meta.size_bytes / (1024**2):.2f} MB)"
                    )
            
            # Handle data artifacts
            elif data is not None:
                # Serialize data
                serialized = self._serialize_data(data)
                artifact_meta.hash = hashlib.sha256(serialized.encode()).hexdigest()
                artifact_meta.size_bytes = len(serialized)
                
                # Save to storage
                self._save_data_artifact(artifact_meta, serialized)
                
                logger.info(f"Logged data artifact: {name} v{version}")
            
            # Update lineage graph
            self._update_lineage(artifact_meta)
            
            # Cache metadata
            self.artifact_cache[f"{name}:{version}"] = artifact_meta
            
            # Log to WandB if available
            if self.run and not self.offline_mode and WANDB_AVAILABLE:
                wandb.log({
                    f"artifact/{name}/version": version,
                    f"artifact/{name}/type": type.value,
                    f"artifact/{name}/hash": artifact_meta.hash[:8] if artifact_meta.hash else None,
                })
            
            return artifact_meta
    
    def log_large_artifact(
        self,
        name: str,
        file_path: Union[str, Path],
        type: Union[str, ArtifactType] = ArtifactType.MODEL,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 8192,
    ) -> ArtifactMetadata:
        """Log large file artifacts with streaming and content-addressable storage."""
        with self._lock:
            # Only rank 0 logs in distributed mode
            if self.is_distributed and self.rank != 0:
                return None
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {file_path}")
            
            # Stream-compute hash to avoid memory issues
            file_hash = self._compute_file_hash(file_path, chunk_size)
            
            logger.info(
                f"Processing large artifact: {name} "
                f"({file_path.stat().st_size / (1024**3):.2f} GB)"
            )
            
            # Check if content already exists
            if not self.storage_backend.exists(file_hash):
                logger.info(f"Uploading new content: {file_hash[:16]}...")
                self.storage_backend.upload(file_path, file_hash)
            else:
                logger.info(f"Content already exists: {file_hash[:16]}...")
            
            # Log metadata artifact
            return self.log_artifact(
                name=name,
                type=type,
                metadata={
                    "file_path": str(file_path),
                    "content_hash": file_hash,
                    "size_gb": file_path.stat().st_size / (1024**3),
                    **(metadata or {}),
                }
            )
    
    def use_artifact(
        self,
        name: str,
        version: Optional[str] = None,
        type: Optional[Union[str, ArtifactType]] = None,
    ) -> ArtifactMetadata:
        """Retrieve and use a previously logged artifact."""
        # Construct artifact key
        if version:
            key = f"{name}:{version}"
        else:
            # Find latest version
            matching_keys = [
                k for k in self.artifact_cache.keys()
                if k.startswith(f"{name}:")
            ]
            if not matching_keys:
                raise ValueError(f"No artifact found with name: {name}")
            key = sorted(matching_keys)[-1]  # Get latest version
        
        if key in self.artifact_cache:
            artifact = self.artifact_cache[key]
            logger.info(f"Using cached artifact: {key}")
            return artifact
        
        # Try to load from storage
        if self.offline_mode:
            artifact = self._load_offline_artifact(name, version)
        else:
            artifact = self._load_wandb_artifact(name, version, type)
        
        # Cache for future use
        self.artifact_cache[key] = artifact
        return artifact
    
    def set_run_state(self, state: RunState):
        """Update run state for atomic tracking."""
        with self._lock:
            self.run_state = state
            
            if self.run:
                if self.offline_mode:
                    # Save state to local file
                    run_dir = Path(self.run["dir"])
                    with open(run_dir / "state.txt", "w") as f:
                        f.write(state.value)
                elif WANDB_AVAILABLE:
                    self.run.summary["run_state"] = state.value
            
            logger.info(f"Run state updated: {state.value}")
    
    def finalize_run(self):
        """Finalize the current run and save all metadata."""
        with self._lock:
            if self.run is None:
                return
            
            # Only rank 0 finalizes in distributed mode
            if self.is_distributed and self.rank != 0:
                return
            
            # Save final metadata
            if self.offline_mode:
                self._save_offline_metadata()
            
            # Finish WandB run
            if not self.offline_mode and WANDB_AVAILABLE:
                wandb.finish()
            
            logger.info(f"Finalized run: {self.run_id}")
            
            # Reset state
            self.run = None
            self.run_id = None
            self.run_state = None
    
    def _compute_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of file using streaming."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_version(self, name: str, type: ArtifactType) -> str:
        """Generate next version number for artifact."""
        # Find existing versions
        existing = [
            k.split(":")[-1] for k in self.artifact_cache.keys()
            if k.startswith(f"{name}:")
        ]
        
        if not existing:
            return "v1"
        
        # Extract version numbers
        versions = []
        for v in existing:
            if v.startswith("v") and v[1:].isdigit():
                versions.append(int(v[1:]))
        
        if versions:
            next_version = max(versions) + 1
        else:
            next_version = 1
        
        return f"v{next_version}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data to JSON string."""
        import json
        from pathlib import Path
        import numpy as np
        
        def json_encoder(obj):
            if isinstance(obj, (Path, type)):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(data, default=json_encoder, indent=2)
    
    def _save_data_artifact(self, metadata: ArtifactMetadata, data: str):
        """Save data artifact to storage."""
        if self.offline_mode:
            # Save to local cache
            cache_dir = Path("./artifact_cache/data")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = cache_dir / f"{metadata.name}_{metadata.version}.json"
            with open(file_path, "w") as f:
                f.write(data)
        
        elif WANDB_AVAILABLE and self.run:
            # Log to WandB
            artifact = wandb.Artifact(
                name=f"{metadata.name}",
                type=metadata.type.value,
                metadata=asdict(metadata),
            )
            
            # Save data to temp file and add to artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(data)
                temp_path = f.name
            
            artifact.add_file(temp_path)
            self.run.log_artifact(artifact)
            
            # Clean up temp file
            os.unlink(temp_path)
    
    def _update_lineage(self, artifact: ArtifactMetadata):
        """Update artifact lineage graph."""
        key = f"{artifact.name}:{artifact.version}"
        
        # Add to graph
        self.lineage_graph[key] = {
            "artifact": artifact,
            "parents": artifact.parent_artifacts,
            "children": [],
        }
        
        # Update parent references
        for parent_key in artifact.parent_artifacts:
            if parent_key in self.lineage_graph:
                self.lineage_graph[parent_key]["children"].append(key)
    
    def _save_offline_metadata(self):
        """Save all metadata for offline run."""
        run_dir = Path(self.run["dir"])
        
        # Save artifact metadata
        metadata_file = run_dir / "artifacts.json"
        artifacts_data = {
            key: asdict(meta) for key, meta in self.artifact_cache.items()
        }
        
        with open(metadata_file, "w") as f:
            json.dump(artifacts_data, f, indent=2)
        
        # Save lineage graph
        lineage_file = run_dir / "lineage.json"
        lineage_data = {
            key: {
                "parents": node["parents"],
                "children": node["children"],
            }
            for key, node in self.lineage_graph.items()
        }
        
        with open(lineage_file, "w") as f:
            json.dump(lineage_data, f, indent=2)
        
        logger.info(f"Saved offline metadata to: {run_dir}")
    
    def _load_offline_artifact(
        self,
        name: str,
        version: Optional[str] = None
    ) -> ArtifactMetadata:
        """Load artifact from offline storage."""
        # Search in runs directory
        runs_dir = Path("./runs")
        if not runs_dir.exists():
            raise FileNotFoundError("No offline runs found")
        
        # Search all runs for artifact
        for run_dir in runs_dir.iterdir():
            metadata_file = run_dir / "artifacts.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    artifacts = json.load(f)
                
                # Search for matching artifact
                for key, meta_dict in artifacts.items():
                    if key.startswith(f"{name}:"):
                        if version is None or key.endswith(f":{version}"):
                            return ArtifactMetadata(**meta_dict)
        
        raise FileNotFoundError(f"Artifact not found: {name}:{version or 'latest'}")
    
    def _load_wandb_artifact(
        self,
        name: str,
        version: Optional[str] = None,
        type: Optional[Union[str, ArtifactType]] = None,
    ) -> ArtifactMetadata:
        """Load artifact from WandB."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not available")
        
        # Construct artifact path
        artifact_path = f"{name}:{version or 'latest'}"
        
        # Use artifact
        artifact = self.run.use_artifact(artifact_path)
        
        # Convert to metadata
        return ArtifactMetadata(
            name=name,
            type=ArtifactType[artifact.type.upper()],
            version=artifact.version,
            metadata=artifact.metadata,
        )
    
    def get_lineage(self, artifact_key: str) -> Dict[str, Any]:
        """Get full lineage tree for an artifact."""
        if artifact_key not in self.lineage_graph:
            return {}
        
        def build_tree(key: str, visited: set = None):
            if visited is None:
                visited = set()
            
            if key in visited:
                return {"circular_reference": key}
            
            visited.add(key)
            
            node = self.lineage_graph.get(key, {})
            tree = {
                "artifact": key,
                "parents": [],
                "children": [],
            }
            
            # Recursively build parent tree
            for parent_key in node.get("parents", []):
                tree["parents"].append(build_tree(parent_key, visited.copy()))
            
            # Recursively build child tree
            for child_key in node.get("children", []):
                tree["children"].append(build_tree(child_key, visited.copy()))
            
            return tree
        
        return build_tree(artifact_key)


# Global singleton instance
artifact_manager = ArtifactManager()