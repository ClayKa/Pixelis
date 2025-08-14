# core.engine

## Classes

### class `InferenceEngine`

```python
InferenceEngine(model, experience_buffer, voting_module, reward_orchestrator, config: Dict[str, Any])
```

Core inference engine for the Pixelis framework.

Handles:
- Model inference and prediction
- k-NN retrieval from experience buffer
- Temporal ensemble voting
- Confidence gating for learning updates
- Communication with the update worker
- Shared memory management for tensor transfer
- Watchdog for resource cleanup

#### Methods

##### `__init__(self, model, experience_buffer, voting_module, reward_orchestrator, config: Dict[str, Any])`

Initialize the Inference Engine.

Args:
    model: The main model for inference
    experience_buffer: Experience buffer for k-NN retrieval
    voting_module: Module for temporal ensemble voting
    reward_orchestrator: Module for reward calculation
    config: Configuration dictionary

##### `infer_and_adapt(self, input_data: Dict[str, Any]) -> Tuple[Any, float, Dict[str, Any]]`

Main inference and adaptation loop with cold start bootstrapping.

Orchestrates the complete online evolution loop:
1. Model inference
2. Experience buffer retrieval (if available)
3. Temporal ensemble voting (if sufficient experiences)
4. Confidence-gated learning updates
5. Monitoring and health tracking

Args:
    input_data: Input data containing image features and question
    
Returns:
    Tuple of (prediction, confidence_score, metadata)

##### `process_human_review_decision(self, task_id: str, approved: bool, reviewer_notes: Optional[str] = None)`

Process a human reviewer's decision on a task.

Args:
    task_id: ID of the reviewed task
    approved: Whether the task was approved
    reviewer_notes: Optional notes from the reviewer

##### `record_model_update(self)`

Record that a model update was performed.

##### `run(self)`

Main run loop for the inference engine.
Processes requests from the request queue.

##### `shutdown(self)`

Gracefully shutdown the inference engine.

##### `start_monitoring(self)`

Start the health monitoring thread.

##### `start_update_worker(self)`

Start the update worker process with proper synchronization.

##### `start_watchdog(self)`

Start the watchdog thread for monitoring shared memory and worker health.

---

### class `KLConfig`

```python
KLConfig(beta_update_mode: str = 'auto', initial_beta: float = 0.01, target_kl: float = 0.05, kl_tolerance: float = 0.01, beta_increase_factor: float = 1.2, beta_decrease_factor: float = 1.2, min_beta: float = 0.0001, max_beta: float = 1.0) -> None
```

Configuration for KL divergence penalty.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, beta_update_mode: str = 'auto', initial_beta: float = 0.01, target_kl: float = 0.05, kl_tolerance: float = 0.01, beta_increase_factor: float = 1.2, beta_decrease_factor: float = 1.2, min_beta: float = 0.0001, max_beta: float = 1.0) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate KL configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `SharedMemoryInfo`

```python
SharedMemoryInfo(name: str, shape: Tuple[int, ...], dtype: torch.dtype, created_at: datetime.datetime, size_bytes: int) -> None
```

Information about a shared memory segment.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, created_at: datetime.datetime, size_bytes: int) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `age_seconds(self) -> float`

Get age of the shared memory segment in seconds.

---

### class `SharedMemoryManager`

```python
SharedMemoryManager(timeout_seconds: float = 60.0)
```

Manages shared memory segments for tensor transfer between processes.

Implements a robust lifecycle management system with:
- Creation and tracking of shared memory segments
- Watchdog-based cleanup of stale segments
- Confirmation-based cleanup for normal operation

#### Methods

##### `__init__(self, timeout_seconds: float = 60.0)`

Initialize the shared memory manager.

Args:
    timeout_seconds: Timeout for stale segment cleanup

##### `cleanup_stale_segments(self, worker_alive: bool = True) -> List[str]`

Clean up stale shared memory segments.

Args:
    worker_alive: Whether the worker process is alive
    
Returns:
    List of cleaned segment names

##### `create_shared_tensor(self, tensor: torch.Tensor) -> core.engine.inference_engine.SharedMemoryInfo`

Create a shared memory segment for a tensor.

Args:
    tensor: Tensor to share
    
Returns:
    SharedMemoryInfo with metadata about the shared segment

##### `get_status(self) -> Dict[str, Any]`

Get status of the shared memory manager.

Returns:
    Status dictionary

##### `mark_cleaned(self, shm_name: str)`

Mark a shared memory segment as cleaned up.

Args:
    shm_name: Name of the cleaned segment

##### `reconstruct_tensor(self, shm_info: core.engine.inference_engine.SharedMemoryInfo) -> torch.Tensor`

Reconstruct a tensor from shared memory info.

NOTE: This method is intended to show the interface.
In production, PyTorch's built-in shared memory through
tensor.share_memory_() should be used with proper IPC handle passing.

Args:
    shm_info: Shared memory metadata
    
Returns:
    Reconstructed tensor

---

### class `SharedMemoryReconstructor`

Handles reconstruction of tensors from shared memory in the worker process.

#### Methods

##### `reconstruct_tensor_from_info(shm_info: Dict[str, Any]) -> torch.Tensor`

Reconstruct a tensor from shared memory info.

Args:
    shm_info: Shared memory metadata dictionary
    
Returns:
    Reconstructed tensor

---

### class `UpdateWorker`

```python
UpdateWorker(model: torch.nn.modules.module.Module, update_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x106dbdfd0>>, cleanup_confirmation_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x106dbdfd0>>, config: Dict[str, Any], reward_orchestrator: Optional[core.modules.reward_shaping.RewardOrchestrator] = None, model_save_path: Optional[str] = None)
```

Model update worker that processes learning updates asynchronously.

Implements a conservative update strategy with multiple safety mechanisms
to ensure stable online learning without catastrophic forgetting.

#### Methods

##### `__init__(self, model: torch.nn.modules.module.Module, update_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x106dbdfd0>>, cleanup_confirmation_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x106dbdfd0>>, config: Dict[str, Any], reward_orchestrator: Optional[core.modules.reward_shaping.RewardOrchestrator] = None, model_save_path: Optional[str] = None)`

Initialize the update worker.

Args:
    model: The model to update
    update_queue: Queue for receiving update tasks
    cleanup_confirmation_queue: Queue for sending cleanup confirmations
    config: Configuration dictionary
    reward_orchestrator: Optional reward orchestrator instance
    model_save_path: Path for saving model checkpoints

##### `get_ema_model(self)`

Get the EMA model for inference.

##### `get_statistics(self) -> Dict[str, Any]`

Get current statistics.

##### `run(self)`

Main worker loop that processes update tasks.

##### `shutdown(self)`

Gracefully shutdown the update worker.

---

