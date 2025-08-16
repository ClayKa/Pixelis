# core.models

## Classes

### class `DynamicLoRAConfig`

```python
DynamicLoRAConfig(config_path: str = 'configs/lora_rank_config.json')
```

Manages dynamic LoRA configuration based on SVD analysis

#### Methods

##### `__init__(self, config_path: str = 'configs/lora_rank_config.json')`

Initialize with configuration file path.

Args:
    config_path: Path to the LoRA rank configuration JSON file

##### `create_lora_config(self, task_type: peft.utils.peft_types.TaskType = <TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode: bool = False, **kwargs) -> peft.tuners.lora.config.LoraConfig`

Create a LoraConfig object with dynamic ranks.

Args:
    task_type: Type of task (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)
    inference_mode: Whether to create config for inference
    **kwargs: Additional arguments to override defaults
    
Returns:
    LoraConfig object with dynamic rank allocation

##### `get_compression_ratio(self) -> float`

Get overall compression ratio

##### `get_layer_ranks(self) -> Dict[str, int]`

Get layer-specific rank configuration

##### `get_metadata(self) -> Dict`

Get analysis metadata

---

### class `PEFTModelFactory`

Factory class for creating PEFT models with dynamic LoRA configuration

#### Methods

##### `create_peft_model_from_config(base_model: Union[str, transformers.modeling_utils.PreTrainedModel], rank_config_path: str = 'configs/lora_rank_config.json', model_type: str = 'auto', load_in_8bit: bool = False, load_in_4bit: bool = False, device_map: str = 'auto', torch_dtype: torch.dtype = torch.float16, gradient_checkpointing: bool = True, **peft_kwargs) -> tuple[transformers.modeling_utils.PreTrainedModel, typing.Any]`

Create a PEFT model with dynamic LoRA configuration from SVD analysis.

Args:
    base_model: Model name/path or pre-loaded model
    rank_config_path: Path to LoRA rank configuration JSON
    model_type: Type of model ("qwen2_vl", "qwen3", or "auto")
    load_in_8bit: Whether to load model in 8-bit precision
    load_in_4bit: Whether to load model in 4-bit precision
    device_map: Device map for model parallelism
    torch_dtype: Data type for model weights
    gradient_checkpointing: Whether to enable gradient checkpointing
    **peft_kwargs: Additional arguments for LoraConfig
    
Returns:
    Tuple of (peft_model, tokenizer/processor)

##### `load_peft_checkpoint(base_model_name_or_path: str, peft_model_path: str, model_type: str = 'auto', device_map: str = 'auto', torch_dtype: torch.dtype = torch.float16, **kwargs) -> tuple[peft.peft_model.PeftModel, typing.Any]`

Load a saved PEFT model checkpoint.

Args:
    base_model_name_or_path: Base model name or path
    peft_model_path: Path to saved PEFT adapter weights
    model_type: Type of model
    device_map: Device map for model parallelism
    torch_dtype: Data type for model weights
    **kwargs: Additional arguments for model loading
    
Returns:
    Tuple of (peft_model, tokenizer/processor)

---

## Functions

### `create_model_with_dynamic_lora(model_name: str = 'Qwen/Qwen2.5-VL-7B', rank_config_path: str = 'configs/lora_rank_config.json', **kwargs) -> tuple[peft.peft_model.PeftModel, typing.Any]`

Convenience function to create a model with dynamic LoRA configuration.

Args:
    model_name: Name or path of the base model
    rank_config_path: Path to LoRA rank configuration
    **kwargs: Additional arguments for model creation
    
Returns:
    Tuple of (peft_model, tokenizer/processor)

---

