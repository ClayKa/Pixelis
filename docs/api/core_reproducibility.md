# core.reproducibility

## Classes

### class `ArtifactManager`

Singleton manager for all artifact tracking and versioning.
Thread-safe and distributed-training aware.

#### Methods

##### `__init__(self)`

Initialize self.  See help(type(self)) for accurate signature.

##### `__new__(cls)`

Create and return a new object.  See help(type) for accurate signature.

##### `finalize_run(self)`

Finalize the current run and save all metadata.

##### `get_lineage(self, artifact_key: str) -> Dict[str, Any]`

Get full lineage tree for an artifact.

##### `init_run(self, name: str, config: Optional[Dict[str, Any]] = None, project: str = 'pixelis', tags: Optional[List[str]] = None) -> Optional[Any]`

Initialize a new experimental run.

##### `log_artifact(self, name: str, type: Union[str, core.reproducibility.artifact_manager.ArtifactType], data: Optional[Any] = None, file_path: Union[str, pathlib.Path, NoneType] = None, metadata: Optional[Dict[str, Any]] = None, parent_artifacts: Optional[List[str]] = None) -> core.reproducibility.artifact_manager.ArtifactMetadata`

Log an artifact with automatic versioning and lineage tracking.

##### `log_large_artifact(self, name: str, file_path: Union[str, pathlib.Path], type: Union[str, core.reproducibility.artifact_manager.ArtifactType] = <ArtifactType.MODEL: 'model'>, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 8192) -> core.reproducibility.artifact_manager.ArtifactMetadata`

Log large file artifacts with streaming and content-addressable storage.

##### `set_run_state(self, state: core.reproducibility.artifact_manager.RunState)`

Update run state for atomic tracking.

##### `use_artifact(self, name: str, version: Optional[str] = None, type: Union[str, core.reproducibility.artifact_manager.ArtifactType, NoneType] = None) -> core.reproducibility.artifact_manager.ArtifactMetadata`

Retrieve and use a previously logged artifact.

---

### class `ArtifactMetadata`

```python
ArtifactMetadata(name: str, type: core.reproducibility.artifact_manager.ArtifactType, version: str, hash: Optional[str] = None, size_bytes: Optional[int] = None, created_at: str = None, run_id: Optional[str] = None, parent_artifacts: List[str] = None, metadata: Dict[str, Any] = None) -> None
```

Metadata for tracked artifacts.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, name: str, type: core.reproducibility.artifact_manager.ArtifactType, version: str, hash: Optional[str] = None, size_bytes: Optional[int] = None, created_at: str = None, run_id: Optional[str] = None, parent_artifacts: List[str] = None, metadata: Dict[str, Any] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

##### `__repr__(self)`

Return repr(self).

---

### class `ArtifactType`

```python
ArtifactType(*args, **kwds)
```

Standard artifact types for consistent naming.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

---

### class `ConfigCapture`

Utilities for capturing configuration and environment state.

#### Methods

##### `capture_environment(level: core.reproducibility.config_capture.EnvironmentCaptureLevel = <EnvironmentCaptureLevel.STANDARD: 2>) -> Dict[str, Any]`

Capture environment information at specified detail level.

Args:
    level: Level of detail to capture

Returns:
    Dictionary containing environment information

##### `validate_environment(required_packages: Optional[List[str]] = None, min_gpu_memory_gb: float = 0, min_cuda_version: Optional[str] = None) -> Dict[str, Any]`

Validate that environment meets requirements.

Args:
    required_packages: List of required package names
    min_gpu_memory_gb: Minimum GPU memory required
    min_cuda_version: Minimum CUDA version required

Returns:
    Dictionary with validation results

---

### class `EnvironmentCaptureLevel`

```python
EnvironmentCaptureLevel(*args, **kwds)
```

Levels of environment capture detail.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__dir__(self)`

Returns public methods and other interesting attributes.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__init__(self, *args, **kwds)`

Initialize self.  See help(type(self)) for accurate signature.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

##### `__new__(cls, value)`

Create and return a new object.  See help(type) for accurate signature.

##### `__reduce_ex__(self, proto)`

Helper for pickle.

##### `__repr__(self)`

Return repr(self).

---

### class `ExperimentContext`

```python
ExperimentContext(config: Optional[Any] = None, name: Optional[str] = None, project: str = 'pixelis', tags: Optional[List[str]] = None, capture_level: core.reproducibility.config_capture.EnvironmentCaptureLevel = <EnvironmentCaptureLevel.STANDARD: 2>, monitor_hardware: bool = True, offline_mode: Optional[bool] = None)
```

Context manager for comprehensive experiment tracking.
Automatically captures configuration, environment, and artifacts.

#### Methods

##### `__enter__(self)`

Enter context and capture initial state.

##### `__exit__(self, exc_type, exc_val, exc_tb)`

Exit context and finalize tracking.

##### `__init__(self, config: Optional[Any] = None, name: Optional[str] = None, project: str = 'pixelis', tags: Optional[List[str]] = None, capture_level: core.reproducibility.config_capture.EnvironmentCaptureLevel = <EnvironmentCaptureLevel.STANDARD: 2>, monitor_hardware: bool = True, offline_mode: Optional[bool] = None)`

Initialize experiment context.

Args:
    config: Experiment configuration (Hydra config or dict)
    name: Experiment name (auto-generated if not provided)
    project: WandB project name
    tags: Experiment tags
    capture_level: Level of environment capture detail
    monitor_hardware: Whether to monitor hardware usage
    offline_mode: Force offline mode (None = auto-detect)

##### `log_artifact(self, *args, **kwargs)`

Convenience method to log artifact within context.

##### `log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None)`

Log metrics to tracking system.

##### `log_model_checkpoint(self, model_path: pathlib.Path, step: int, metrics: Optional[Dict[str, float]] = None)`

Log model checkpoint with metadata.

---

### class `HardwareMonitor`

```python
HardwareMonitor(interval: float = 5.0)
```

Monitor hardware usage during experiments.

#### Methods

##### `__init__(self, interval: float = 5.0)`

Initialize self.  See help(type(self)) for accurate signature.

##### `start(self)`

Start monitoring in background thread.

##### `stop(self) -> Dict[str, Any]`

Stop monitoring and return collected stats.

---

### class `LineageNode`

```python
LineageNode(artifact_id: str, artifact_name: str, artifact_type: str, version: str, created_at: str, run_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> None
```

Node in the lineage graph representing an artifact.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, artifact_id: str, artifact_name: str, artifact_type: str, version: str, created_at: str, run_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

##### `__repr__(self)`

Return repr(self).

---

### class `LineageTracker`

Track and visualize artifact lineage and dependencies.

#### Methods

##### `__init__(self)`

Initialize lineage tracker.

##### `add_artifact(self, artifact_id: str, name: str, artifact_type: str, version: str, run_id: Optional[str] = None, parent_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> core.reproducibility.lineage_tracker.LineageNode`

Add an artifact to the lineage graph.

Args:
    artifact_id: Unique identifier for the artifact
    name: Artifact name
    artifact_type: Type of artifact
    version: Artifact version
    run_id: Run that created this artifact
    parent_ids: List of parent artifact IDs
    metadata: Additional metadata

Returns:
    Created LineageNode

##### `detect_cycles(self) -> List[List[str]]`

Detect cycles in the lineage graph.

Returns:
    List of cycles (each cycle is a list of artifact IDs)

##### `export_to_dot(self, output_path: Optional[pathlib.Path] = None) -> str`

Export lineage graph to DOT format for visualization.

Args:
    output_path: Optional path to save DOT file

Returns:
    DOT format string

##### `export_to_mermaid(self) -> str`

Export lineage graph to Mermaid format for Markdown embedding.

Returns:
    Mermaid format string

##### `get_ancestors(self, artifact_id: str, max_depth: Optional[int] = None) -> Set[str]`

Get all ancestor artifacts (transitively).

Args:
    artifact_id: Starting artifact ID
    max_depth: Maximum depth to traverse

Returns:
    Set of ancestor artifact IDs

##### `get_descendants(self, artifact_id: str, max_depth: Optional[int] = None) -> Set[str]`

Get all descendant artifacts (transitively).

Args:
    artifact_id: Starting artifact ID
    max_depth: Maximum depth to traverse

Returns:
    Set of descendant artifact IDs

##### `get_lineage_path(self, from_id: str, to_id: str) -> Optional[List[str]]`

Find path between two artifacts if one exists.

Args:
    from_id: Starting artifact ID
    to_id: Target artifact ID

Returns:
    List of artifact IDs forming the path, or None if no path exists

##### `get_run_lineage(self, run_id: str) -> Dict[str, Any]`

Get complete lineage for all artifacts in a run.

Args:
    run_id: Run identifier

Returns:
    Dictionary containing lineage information

##### `load(self, path: pathlib.Path)`

Load lineage graph from JSON file.

Args:
    path: Path to load file

##### `save(self, path: pathlib.Path)`

Save lineage graph to JSON file.

Args:
    path: Path to save file

##### `validate_lineage(self) -> Dict[str, Any]`

Validate the lineage graph for issues.

Returns:
    Dictionary containing validation results

---

### class `LocalStorageBackend`

```python
LocalStorageBackend(cache_dir: Union[str, pathlib.Path] = './artifact_cache')
```

Local filesystem storage for offline mode.

#### Methods

##### `__init__(self, cache_dir: Union[str, pathlib.Path] = './artifact_cache')`

Initialize self.  See help(type(self)) for accurate signature.

##### `download(self, content_hash: str, target_path: pathlib.Path) -> pathlib.Path`

Retrieve file from content-addressable storage.

##### `exists(self, content_hash: str) -> bool`

##### `upload(self, file_path: pathlib.Path, content_hash: str) -> str`

Copy file to content-addressable storage.

---

### class `RunState`

```python
RunState(*args, **kwds)
```

Experiment run states for atomic tracking.

#### Methods

##### `__contains__(value)`

Return True if `value` is in `cls`.

`value` is in `cls` if:
1) `value` is a member of `cls`, or
2) `value` is the value of one of the `cls`'s members.

##### `__getitem__(name)`

Return the member matching `name`.

##### `__iter__()`

Return members in definition order.

##### `__len__()`

Return the number of members (no aliases)

---

### class `StorageBackend`

Abstract base class for artifact storage backends.

#### Methods

##### `download(self, content_hash: str, target_path: pathlib.Path) -> pathlib.Path`

##### `exists(self, content_hash: str) -> bool`

##### `upload(self, file_path: pathlib.Path, content_hash: str) -> str`

---

### class `TTRLContext`

```python
TTRLContext(config: Optional[Any] = None, name: Optional[str] = None, experience_snapshot_interval: float = 3600, **kwargs)
```

Specialized context for Test-Time Reinforcement Learning experiments.
Adds support for experience buffer snapshots and online metrics.

#### Methods

##### `__enter__(self)`

Enter context and capture initial state.

##### `__exit__(self, exc_type, exc_val, exc_tb)`

Exit context and finalize tracking.

##### `__init__(self, config: Optional[Any] = None, name: Optional[str] = None, experience_snapshot_interval: float = 3600, **kwargs)`

Initialize TTRL context.

Args:
    config: Experiment configuration
    name: Experiment name
    experience_snapshot_interval: Seconds between experience buffer snapshots
    **kwargs: Additional arguments for ExperimentContext

##### `log_artifact(self, *args, **kwargs)`

Convenience method to log artifact within context.

##### `log_experience(self, experience_id: str, input_data: Any, output_data: Any, metadata: Optional[Dict[str, Any]] = None)`

Log individual experience.

##### `log_experience_buffer(self, buffer_data: Any, force: bool = False) -> Optional[Any]`

Log experience buffer snapshot if interval has passed.

Args:
    buffer_data: Experience buffer data to snapshot
    force: Force snapshot regardless of interval

Returns:
    Artifact metadata if snapshot was taken, None otherwise

##### `log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None)`

Log metrics to tracking system.

##### `log_model_checkpoint(self, model_path: pathlib.Path, step: int, metrics: Optional[Dict[str, float]] = None)`

Log model checkpoint with metadata.

##### `log_online_update(self, experience_id: str, reward: float, confidence: float, kl_divergence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None)`

Log online learning update.

---

### class `WandBStorageBackend`

WandB artifact storage backend.

#### Methods

##### `__init__(self)`

Initialize self.  See help(type(self)) for accurate signature.

##### `download(self, content_hash: str, target_path: pathlib.Path) -> pathlib.Path`

Download artifact from WandB.

##### `exists(self, content_hash: str) -> bool`

Check if artifact exists in WandB.

##### `set_run(self, run)`

Set the current WandB run.

##### `upload(self, file_path: pathlib.Path, content_hash: str) -> str`

Upload file as WandB artifact.

---

## Functions

### `checkpoint(frequency: Union[int, str] = 'epoch', save_best: bool = True, metric: str = 'loss', mode: str = 'min') -> Callable`

Decorator to automatically checkpoint model during training.

Args:
    frequency: Checkpoint frequency ("epoch", "step", or integer)
    save_best: Whether to save best checkpoint
    metric: Metric to monitor for best checkpoint
    mode: "min" or "max" for metric comparison

Example:
    @checkpoint(frequency="epoch", save_best=True, metric="val_loss", mode="min")
    def train_epoch(model, data_loader, optimizer):
        # Training logic
        return {"loss": loss, "val_loss": val_loss}

---

### `reproducible(name: Optional[str] = None, capture_level: int = 2, offline_mode: Optional[bool] = None) -> Callable`

Decorator to make a function fully reproducible with automatic context management.

Args:
    name: Experiment name (auto-generated if not provided)
    capture_level: Environment capture level (1=basic, 2=standard, 3=complete)
    offline_mode: Force offline mode (None = auto-detect)

Example:
    @reproducible(name="train_experiment")
    def train_model(config):
        # Training logic
        return model

---

### `track_artifacts(inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None, capture_args: bool = True, capture_source: bool = False) -> Callable`

Decorator to automatically track function inputs and outputs as artifacts.

Args:
    inputs: List of input artifact names to track
    outputs: List of output artifact types to track
    capture_args: Whether to capture function arguments
    capture_source: Whether to capture function source code

Example:
    @track_artifacts(inputs=["dataset:v1"], outputs=["model", "metrics"])
    def train_model(config):
        # Training logic
        return model, metrics

---

