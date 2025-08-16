# core

## Classes

### class `Action`

```python
Action(type: core.data_structures.ActionType, operation: str, arguments: Dict[str, Any] = <factory>, result: Optional[Any] = None, confidence: float = 1.0, timestamp: Optional[datetime.datetime] = None) -> None
```

Represents a single action in a reasoning trajectory.

Attributes:
    type: Type of action (visual operation, reasoning, etc.)
    operation: Name of the operation (e.g., 'SEGMENT_OBJECT_AT')
    arguments: Arguments passed to the operation
    result: Result from the operation
    confidence: Confidence score for this action
    timestamp: When the action was taken

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, type: core.data_structures.ActionType, operation: str, arguments: Dict[str, Any] = <factory>, result: Optional[Any] = None, confidence: float = 1.0, timestamp: Optional[datetime.datetime] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate action data after initialization.

##### `__repr__(self)`

Return repr(self).

##### `from_dict(data: Dict[str, Any]) -> 'Action'`

Create action from dictionary.

##### `to_dict(self) -> Dict[str, Any]`

Convert action to dictionary.

---

### class `ActionType`

```python
ActionType(*args, **kwds)
```

Types of actions in a trajectory.

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

### class `DataConfig`

```python
DataConfig(train_data_path: Optional[str] = None, eval_data_path: Optional[str] = None, test_data_path: Optional[str] = None, max_seq_length: int = 2048, preprocessing_num_workers: int = 4, overwrite_cache: bool = False, use_augmentation: bool = True, augmentation_prob: float = 0.3, use_curriculum: bool = True, curriculum_strategy: str = 'difficulty', difficulty_bins: List[str] = <factory>, sampling_strategy: str = 'uniform', sample_weights: Optional[Dict[str, float]] = None, data_format: str = 'json', text_column: str = 'text', label_column: str = 'label', image_column: str = 'image') -> None
```

Configuration for data processing.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, train_data_path: Optional[str] = None, eval_data_path: Optional[str] = None, test_data_path: Optional[str] = None, max_seq_length: int = 2048, preprocessing_num_workers: int = 4, overwrite_cache: bool = False, use_augmentation: bool = True, augmentation_prob: float = 0.3, use_curriculum: bool = True, curriculum_strategy: str = 'difficulty', difficulty_bins: List[str] = <factory>, sampling_strategy: str = 'uniform', sample_weights: Optional[Dict[str, float]] = None, data_format: str = 'json', text_column: str = 'text', label_column: str = 'label', image_column: str = 'image') -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate data configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `Experience`

```python
Experience(experience_id: str, image_features: Union[torch.Tensor, numpy.ndarray], question_text: str, trajectory: core.data_structures.Trajectory, model_confidence: float, timestamp: datetime.datetime = <factory>, status: core.data_structures.ExperienceStatus = <ExperienceStatus.PENDING: 'pending'>, embeddings: Optional[Dict[str, torch.Tensor]] = None, priority: float = 1.0, retrieval_count: int = 0, success_count: int = 0) -> None
```

Represents an experience in the experience buffer.

Attributes:
    experience_id: Unique identifier for the experience
    image_features: Visual features of the image
    question_text: The question/prompt text
    trajectory: The reasoning trajectory taken
    model_confidence: Model's confidence in the answer
    timestamp: When the experience was created
    status: Current status of the experience
    embeddings: Cached embeddings for similarity search
    priority: Priority score for sampling
    retrieval_count: Number of times retrieved from buffer
    success_count: Number of successful uses in voting

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, experience_id: str, image_features: Union[torch.Tensor, numpy.ndarray], question_text: str, trajectory: core.data_structures.Trajectory, model_confidence: float, timestamp: datetime.datetime = <factory>, status: core.data_structures.ExperienceStatus = <ExperienceStatus.PENDING: 'pending'>, embeddings: Optional[Dict[str, torch.Tensor]] = None, priority: float = 1.0, retrieval_count: int = 0, success_count: int = 0) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate experience data after initialization.

##### `__repr__(self)`

Return repr(self).

##### `from_dict(data: Dict[str, Any]) -> 'Experience'`

Create experience from dictionary.

##### `get_embedding(self, embedding_type: str = 'combined') -> Optional[torch.Tensor]`

Get cached embedding of specified type.

Args:
    embedding_type: Type of embedding ('visual', 'text', 'combined')
    
Returns:
    Embedding tensor if available

##### `get_input_ids(self) -> torch.Tensor`

Get tokenized input IDs for the question.
Placeholder - actual implementation would use tokenizer.

##### `get_labels(self) -> torch.Tensor`

Get labels for training.
Placeholder - actual implementation would process trajectory.

##### `set_embedding(self, embedding: torch.Tensor, embedding_type: str = 'combined')`

Set cached embedding.

Args:
    embedding: Embedding tensor
    embedding_type: Type of embedding

##### `to_dict(self) -> Dict[str, Any]`

Convert experience to dictionary for serialization.

##### `update_priority(self, uncertainty: float, reward: float, decay_factor: float = 0.95)`

Update the priority score based on uncertainty and reward.

Args:
    uncertainty: Current uncertainty (1 - confidence)
    reward: Reward received
    decay_factor: Decay factor for aging

##### `update_usage_stats(self, was_successful: bool)`

Update usage statistics after retrieval.

Args:
    was_successful: Whether the experience led to a successful prediction

#### Properties

##### `success_rate`

Calculate success rate from counts.

---

### class `ExperienceStatus`

```python
ExperienceStatus(*args, **kwds)
```

Status of an experience in the buffer.

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

### class `ExperimentConfig`

```python
ExperimentConfig(experiment_name: str = 'pixelis_experiment', run_name: Optional[str] = None, tags: List[str] = <factory>, use_wandb: bool = True, wandb_project: str = 'pixelis', wandb_entity: Optional[str] = None, wandb_mode: str = 'online', track_artifacts: bool = True, save_code: bool = True, save_config: bool = True, save_environment: bool = True, num_seeds: int = 3, seeds: List[int] = <factory>, ablation_mode: bool = False, ablation_components: List[str] = <factory>) -> None
```

Configuration for experiment tracking.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, experiment_name: str = 'pixelis_experiment', run_name: Optional[str] = None, tags: List[str] = <factory>, use_wandb: bool = True, wandb_project: str = 'pixelis', wandb_entity: Optional[str] = None, wandb_mode: str = 'online', track_artifacts: bool = True, save_code: bool = True, save_config: bool = True, save_environment: bool = True, num_seeds: int = 3, seeds: List[int] = <factory>, ablation_mode: bool = False, ablation_components: List[str] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate experiment configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `ModelConfig`

```python
ModelConfig(model_name: str = 'Qwen/Qwen2.5-VL-7B', model_type: str = 'qwen2_vl', load_in_8bit: bool = False, load_in_4bit: bool = False, device_map: str = 'auto', torch_dtype: str = 'float16', max_length: int = 4096, max_pixels: int = 4014080, min_pixels: int = 401408, image_resolution: int = 448, use_lora: bool = True, lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.1, lora_target_modules: List[str] = <factory>, use_flash_attention: bool = True, gradient_checkpointing: bool = True, base_model_path: Optional[str] = None, adapter_path: Optional[str] = None) -> None
```

Configuration for model architecture and loading.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B', model_type: str = 'qwen2_vl', load_in_8bit: bool = False, load_in_4bit: bool = False, device_map: str = 'auto', torch_dtype: str = 'float16', max_length: int = 4096, max_pixels: int = 4014080, min_pixels: int = 401408, image_resolution: int = 448, use_lora: bool = True, lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.1, lora_target_modules: List[str] = <factory>, use_flash_attention: bool = True, gradient_checkpointing: bool = True, base_model_path: Optional[str] = None, adapter_path: Optional[str] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate model configuration after initialization.

##### `__repr__(self)`

Return repr(self).

---

### class `OnlineConfig`

```python
OnlineConfig(confidence_threshold: float = 0.7, min_confidence_for_update: float = 0.5, min_learning_rate: float = 1e-06, max_learning_rate: float = 0.0001, lr_adaptation_strategy: str = 'proportional', kl_weight: float = 0.01, max_kl_divergence: float = 0.05, use_ema: bool = True, ema_decay: float = 0.999, ema_update_freq: int = 1, buffer_size: int = 10000, k_neighbors: int = 5, similarity_metric: str = 'cosine', faiss_backend: str = 'gpu', faiss_n_probes: int = 10, faiss_use_gpu_fallback: bool = True, persistence_backend: str = 'file', persistence_path: str = './experience_buffer', enable_persistence: bool = True, snapshot_interval: int = 100, max_snapshots: int = 3, visual_weight: float = 0.7, text_weight: float = 0.3, voting_strategy: core.config_schema.VotingStrategy = <VotingStrategy.WEIGHTED: 'weighted'>, min_votes_required: int = 3, update_queue_size: int = 100, update_batch_size: int = 1, update_frequency: int = 1, gradient_clip_norm: float = 1.0, max_updates_per_minute: int = 60, hil_mode_enabled: bool = False, hil_review_percentage: float = 0.02, hil_interface_host: str = '127.0.0.1', hil_interface_port: int = 7860, hil_auto_approve_timeout: Optional[int] = None) -> None
```

Configuration for online learning.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, confidence_threshold: float = 0.7, min_confidence_for_update: float = 0.5, min_learning_rate: float = 1e-06, max_learning_rate: float = 0.0001, lr_adaptation_strategy: str = 'proportional', kl_weight: float = 0.01, max_kl_divergence: float = 0.05, use_ema: bool = True, ema_decay: float = 0.999, ema_update_freq: int = 1, buffer_size: int = 10000, k_neighbors: int = 5, similarity_metric: str = 'cosine', faiss_backend: str = 'gpu', faiss_n_probes: int = 10, faiss_use_gpu_fallback: bool = True, persistence_backend: str = 'file', persistence_path: str = './experience_buffer', enable_persistence: bool = True, snapshot_interval: int = 100, max_snapshots: int = 3, visual_weight: float = 0.7, text_weight: float = 0.3, voting_strategy: core.config_schema.VotingStrategy = <VotingStrategy.WEIGHTED: 'weighted'>, min_votes_required: int = 3, update_queue_size: int = 100, update_batch_size: int = 1, update_frequency: int = 1, gradient_clip_norm: float = 1.0, max_updates_per_minute: int = 60, hil_mode_enabled: bool = False, hil_review_percentage: float = 0.02, hil_interface_host: str = '127.0.0.1', hil_interface_port: int = 7860, hil_auto_approve_timeout: Optional[int] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate online configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `PixelisConfig`

```python
PixelisConfig(model: core.config_schema.ModelConfig = <factory>, training: core.config_schema.TrainingConfig = <factory>, reward: core.config_schema.RewardConfig = <factory>, online: core.config_schema.OnlineConfig = <factory>, data: core.config_schema.DataConfig = <factory>, experiment: core.config_schema.ExperimentConfig = <factory>, system: core.config_schema.SystemConfig = <factory>) -> None
```

Main configuration class combining all sub-configurations.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, model: core.config_schema.ModelConfig = <factory>, training: core.config_schema.TrainingConfig = <factory>, reward: core.config_schema.RewardConfig = <factory>, online: core.config_schema.OnlineConfig = <factory>, data: core.config_schema.DataConfig = <factory>, experiment: core.config_schema.ExperimentConfig = <factory>, system: core.config_schema.SystemConfig = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `from_dict(config_dict: Dict[str, Any]) -> 'PixelisConfig'`

Create configuration from dictionary.

##### `to_dict(self) -> Dict[str, Any]`

Convert configuration to dictionary.

##### `validate(self)`

Validate the entire configuration.

---

### class `RewardComponents`

```python
RewardComponents(task_reward: float, curiosity_reward: float = 0.0, coherence_reward: float = 0.0, tool_penalty: float = 0.0, total_reward: float = 0.0, metadata: Dict[str, Any] = <factory>) -> None
```

Multi-component reward structure.

Attributes:
    task_reward: Reward for task completion
    curiosity_reward: Reward from curiosity module
    coherence_reward: Reward for trajectory coherence
    tool_penalty: Penalty for tool misuse
    total_reward: Combined total reward
    metadata: Additional reward metadata

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, task_reward: float, curiosity_reward: float = 0.0, coherence_reward: float = 0.0, tool_penalty: float = 0.0, total_reward: float = 0.0, metadata: Dict[str, Any] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Calculate total reward after initialization.

##### `__repr__(self)`

Return repr(self).

##### `calculate_total(self, weights: Optional[Dict[str, float]] = None)`

Calculate total reward with optional custom weights.

Args:
    weights: Dictionary of component weights

##### `normalize(self, method: str = 'zscore')`

Normalize reward components.

Args:
    method: Normalization method ('zscore', 'minmax', 'clip')

##### `to_dict(self) -> Dict[str, Any]`

Convert reward components to dictionary.

##### `to_tensor(self) -> torch.Tensor`

Convert reward components to tensor.

---

### class `RewardConfig`

```python
RewardConfig(task_reward_weight: float = 1.0, curiosity_reward_weight: float = 0.3, coherence_reward_weight: float = 0.2, curiosity_beta: float = 0.2, curiosity_eta: float = 0.5, intrinsic_reward_scale: float = 0.1, coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, trajectory_min_length: int = 2, tool_misuse_penalty: float = -0.1, excessive_tool_use_threshold: int = 10, excessive_tool_use_penalty: float = -0.2, normalize_rewards: bool = True, reward_clip_value: float = 10.0, use_curriculum: bool = True, curriculum_stages: List[Dict[str, Any]] = <factory>) -> None
```

Configuration for reward calculation.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, task_reward_weight: float = 1.0, curiosity_reward_weight: float = 0.3, coherence_reward_weight: float = 0.2, curiosity_beta: float = 0.2, curiosity_eta: float = 0.5, intrinsic_reward_scale: float = 0.1, coherence_threshold: float = 0.7, repetition_penalty: float = 0.5, trajectory_min_length: int = 2, tool_misuse_penalty: float = -0.1, excessive_tool_use_threshold: int = 10, excessive_tool_use_penalty: float = -0.2, normalize_rewards: bool = True, reward_clip_value: float = 10.0, use_curriculum: bool = True, curriculum_stages: List[Dict[str, Any]] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate reward configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `RewardType`

```python
RewardType(*args, **kwds)
```

Enumeration of reward types.

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

### class `SystemConfig`

```python
SystemConfig(device: str = 'cuda', num_gpus: int = 1, gpu_ids: Optional[List[int]] = None, max_memory_mb: Optional[int] = None, empty_cache_freq: int = 100, num_workers: int = 4, dataloader_num_workers: int = 4, cache_dir: str = './cache', temp_dir: str = './tmp', log_dir: str = './logs', log_level: str = 'INFO', log_to_file: bool = True, log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', debug_mode: bool = False, profile: bool = False, detect_anomaly: bool = False) -> None
```

Configuration for system-level settings.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, device: str = 'cuda', num_gpus: int = 1, gpu_ids: Optional[List[int]] = None, max_memory_mb: Optional[int] = None, empty_cache_freq: int = 100, num_workers: int = 4, dataloader_num_workers: int = 4, cache_dir: str = './cache', temp_dir: str = './tmp', log_dir: str = './logs', log_level: str = 'INFO', log_to_file: bool = True, log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', debug_mode: bool = False, profile: bool = False, detect_anomaly: bool = False) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate system configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `TrainingConfig`

```python
TrainingConfig(mode: core.config_schema.TrainingMode = <TrainingMode.SFT: 'sft'>, num_epochs: int = 3, batch_size: int = 4, gradient_accumulation_steps: int = 4, learning_rate: float = 5e-05, warmup_steps: int = 500, weight_decay: float = 0.01, max_grad_norm: float = 1.0, optimizer: str = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, scheduler: str = 'cosine', num_cycles: float = 0.5, eval_steps: int = 500, eval_batch_size: int = 8, evaluation_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 3, logging_steps: int = 10, logging_first_step: bool = True, report_to: List[str] = <factory>, output_dir: str = './outputs', save_strategy: str = 'steps', resume_from_checkpoint: Optional[str] = None, fp16: bool = False, bf16: bool = True, tf32: bool = True, ddp_find_unused_parameters: bool = False, fsdp: Optional[str] = None, deepspeed: Optional[str] = None, seed: int = 42) -> None
```

Configuration for training parameters.

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, mode: core.config_schema.TrainingMode = <TrainingMode.SFT: 'sft'>, num_epochs: int = 3, batch_size: int = 4, gradient_accumulation_steps: int = 4, learning_rate: float = 5e-05, warmup_steps: int = 500, weight_decay: float = 0.01, max_grad_norm: float = 1.0, optimizer: str = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, scheduler: str = 'cosine', num_cycles: float = 0.5, eval_steps: int = 500, eval_batch_size: int = 8, evaluation_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 3, logging_steps: int = 10, logging_first_step: bool = True, report_to: List[str] = <factory>, output_dir: str = './outputs', save_strategy: str = 'steps', resume_from_checkpoint: Optional[str] = None, fp16: bool = False, bf16: bool = True, tf32: bool = True, ddp_find_unused_parameters: bool = False, fsdp: Optional[str] = None, deepspeed: Optional[str] = None, seed: int = 42) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate training configuration.

##### `__repr__(self)`

Return repr(self).

---

### class `TrainingMode`

```python
TrainingMode(*args, **kwds)
```

Enumeration of training modes.

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

### class `Trajectory`

```python
Trajectory(actions: List[core.data_structures.Action] = <factory>, final_answer: Optional[Any] = None, total_reward: float = 0.0, metadata: Dict[str, Any] = <factory>) -> None
```

Represents a complete reasoning trajectory.

Attributes:
    actions: List of actions taken
    final_answer: The final answer produced
    total_reward: Total reward received
    metadata: Additional trajectory metadata

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, actions: List[core.data_structures.Action] = <factory>, final_answer: Optional[Any] = None, total_reward: float = 0.0, metadata: Dict[str, Any] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__repr__(self)`

Return repr(self).

##### `add_action(self, action: core.data_structures.Action)`

Add an action to the trajectory.

##### `from_dict(data: Dict[str, Any]) -> 'Trajectory'`

Create trajectory from dictionary.

##### `get_tool_usage_count(self) -> Dict[str, int]`

Get count of each tool used in trajectory.

##### `get_trajectory_length(self) -> int`

Get the length of the trajectory.

##### `has_repetitions(self, threshold: int = 2) -> bool`

Check if trajectory has repetitive actions.

##### `to_dict(self) -> Dict[str, Any]`

Convert trajectory to dictionary.

---

### class `UpdateTask`

```python
UpdateTask(task_id: str, experience: core.data_structures.Experience, reward_tensor: torch.Tensor, learning_rate: float, original_logits: Optional[torch.Tensor] = None, metadata: Dict[str, Any] = <factory>, created_at: datetime.datetime = <factory>, processed_at: Optional[datetime.datetime] = None) -> None
```

Represents a task for the update worker.

Attributes:
    task_id: Unique identifier for the task
    experience: The experience to learn from
    reward_tensor: Multi-component reward tensor
    learning_rate: Adaptive learning rate for this update
    original_logits: Original model logits for KL calculation
    metadata: Additional task metadata
    created_at: When the task was created
    processed_at: When the task was processed

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, task_id: str, experience: core.data_structures.Experience, reward_tensor: torch.Tensor, learning_rate: float, original_logits: Optional[torch.Tensor] = None, metadata: Dict[str, Any] = <factory>, created_at: datetime.datetime = <factory>, processed_at: Optional[datetime.datetime] = None) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate update task data after initialization.

##### `__repr__(self)`

Return repr(self).

##### `get_processing_time(self) -> Optional[float]`

Get processing time in seconds.

Returns:
    Processing time if task is processed, None otherwise

##### `mark_processed(self)`

Mark the task as processed.

##### `to_dict(self) -> Dict[str, Any]`

Convert update task to dictionary.

---

### class `VotingResult`

```python
VotingResult(final_answer: Any, confidence: float, provenance: Dict[str, Any] = <factory>) -> None
```

Result from the temporal ensemble voting module.

Attributes:
    final_answer: The consensus answer
    confidence: Confidence score for the answer
    provenance: Detailed provenance information including audit trail

#### Methods

##### `__eq__(self, other)`

Return self==value.

##### `__init__(self, final_answer: Any, confidence: float, provenance: Dict[str, Any] = <factory>) -> None`

Initialize self.  See help(type(self)) for accurate signature.

##### `__post_init__(self)`

Validate voting result data and ensure required provenance fields.

##### `__repr__(self)`

Return repr(self).

##### `get_consensus_strength(self) -> float`

Calculate the strength of consensus.

Returns:
    Strength score (0 = no consensus, 1 = perfect consensus)

##### `get_vote_distribution(self) -> Dict[str, float]`

Get distribution of votes from provenance.

Returns:
    Dictionary mapping answers to their counts

##### `to_dict(self) -> Dict[str, Any]`

Convert voting result to dictionary.

---

### class `VotingStrategy`

```python
VotingStrategy(*args, **kwds)
```

Enumeration of voting strategies.

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

## Functions

### `generate_json_schema()`

Generate JSON schemas for all data structures.

This can be used for validation when saving/loading data.

---

### `validate_trajectory(trajectory: Union[core.data_structures.Trajectory, Dict, List]) -> core.data_structures.Trajectory`

Validate and convert input to Trajectory object.

Args:
    trajectory: Input trajectory in various formats
    
Returns:
    Validated Trajectory object

---

