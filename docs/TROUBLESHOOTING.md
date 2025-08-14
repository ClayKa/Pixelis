# Pixelis Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [CUDA and GPU Issues](#cuda-and-gpu-issues)
3. [Training Issues](#training-issues)
4. [Configuration Issues](#configuration-issues)
5. [Environment Setup Issues](#environment-setup-issues)
6. [Model Loading Issues](#model-loading-issues)
7. [Online Learning System Issues](#online-learning-system-issues)
8. [Performance Issues](#performance-issues)
9. [Debugging Tools](#debugging-tools)

---

## Common Issues

### Issue: "ImportError: No module named 'core'"

**Symptoms**: Python cannot find the core modules when running scripts.

**Solution**:
```bash
# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Pixelis"

# Or install in development mode
pip install -e .
```

### Issue: "FileNotFoundError: configs/main.yaml"

**Symptoms**: Hydra cannot find configuration files.

**Solution**:
```bash
# Run scripts from project root
cd /path/to/Pixelis
python scripts/train.py

# Or specify config path explicitly
python scripts/train.py --config-path /absolute/path/to/configs
```

### Issue: "WandB API Key Not Found"

**Symptoms**: WandB initialization fails.

**Solution**:
```bash
# Set API key via environment variable
export WANDB_API_KEY="your_api_key_here"

# Or login interactively
wandb login

# Or disable wandb for testing
export WANDB_MODE=offline
```

---

## CUDA and GPU Issues

### Issue: "CUDA out of memory"

**Symptoms**: Training or inference fails with OOM error.

**Solutions**:

1. **Reduce batch size**:
```yaml
# In config file
training:
  batch_size: 2  # Reduce from default
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

2. **Enable gradient checkpointing**:
```python
model.gradient_checkpointing_enable()
```

3. **Use mixed precision training**:
```yaml
training:
  fp16: true
  # or for newer GPUs
  bf16: true
```

4. **Clear GPU cache**:
```python
import torch
torch.cuda.empty_cache()
```

5. **Monitor GPU memory**:
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check current usage in Python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Issue: "CUDA version mismatch"

**Symptoms**: PyTorch doesn't detect GPU or throws CUDA errors.

**Solution**:
```bash
# Check CUDA version
nvidia-smi  # System CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "RuntimeError: NCCL error"

**Symptoms**: Multi-GPU training fails with NCCL communication errors.

**Solution**:
```bash
# Set NCCL debug level
export NCCL_DEBUG=INFO

# Try different NCCL backend
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Use Gloo backend for debugging
python scripts/train.py --distributed-backend gloo
```

---

## Training Issues

### Issue: "Training loss is NaN"

**Symptoms**: Loss becomes NaN after a few steps.

**Solutions**:

1. **Check learning rate**:
```yaml
training:
  learning_rate: 1e-5  # Reduce from default
  warmup_ratio: 0.1  # Add warmup
```

2. **Enable gradient clipping**:
```yaml
training:
  max_grad_norm: 1.0
```

3. **Check for bad data**:
```python
# Add data validation
def validate_batch(batch):
    assert not torch.isnan(batch['input_ids']).any()
    assert not torch.isinf(batch['labels']).any()
    return batch
```

4. **Use loss scaling for mixed precision**:
```python
from torch.cuda.amp import GradScaler
scaler = GradScaler()
```

### Issue: "Training is extremely slow"

**Symptoms**: Training takes much longer than expected.

**Solutions**:

1. **Profile the training loop**:
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Run one training step
    train_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. **Enable Flash Attention**:
```python
model.config.use_flash_attention_2 = True
```

3. **Optimize data loading**:
```python
dataloader = DataLoader(
    dataset,
    num_workers=4,  # Increase workers
    pin_memory=True,  # Pin memory for faster transfer
    prefetch_factor=2  # Prefetch batches
)
```

4. **Check I/O bottlenecks**:
```bash
# Monitor disk I/O
iostat -x 1

# Use faster storage for datasets
export DATA_DIR=/path/to/ssd/data
```

### Issue: "Validation performance drops during training"

**Symptoms**: Model overfits or validation metrics degrade.

**Solutions**:

1. **Add regularization**:
```yaml
training:
  weight_decay: 0.01
  dropout: 0.1
```

2. **Implement early stopping**:
```python
early_stopping = EarlyStopping(patience=3, min_delta=0.001)
if early_stopping(val_loss):
    print("Early stopping triggered")
    break
```

3. **Use curriculum learning**:
```yaml
curriculum:
  enabled: true
  stages: ["easy", "medium", "hard"]
  transition_steps: [1000, 2000]
```

---

## Configuration Issues

### Issue: "Config validation error"

**Symptoms**: Hydra or OmegaConf throws validation errors.

**Solution**:
```python
# Debug config structure
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(cfg))

# Validate against schema
from core.config_schema import Config
try:
    validated_config = Config(**cfg)
except Exception as e:
    print(f"Validation error: {e}")
```

### Issue: "Override not working"

**Symptoms**: Command-line overrides don't take effect.

**Solution**:
```bash
# Correct override syntax
python scripts/train.py training.learning_rate=1e-5 training.batch_size=4

# Check final config
python scripts/train.py --cfg job --resolve
```

---

## Environment Setup Issues

### Issue: "Conflicting package versions"

**Symptoms**: Import errors or unexpected behavior.

**Solution**:
```bash
# Create fresh environment
conda create -n pixelis-debug python=3.10
conda activate pixelis-debug

# Install with strict versions
pip install -r requirements.txt --no-deps
pip check  # Verify no conflicts

# Export working environment
conda env export > environment_working.yml
```

### Issue: "FAISS installation fails"

**Symptoms**: Cannot install or import faiss-gpu.

**Solution**:
```bash
# Install via conda (recommended)
conda install -c pytorch faiss-gpu

# Or CPU version as fallback
pip install faiss-cpu

# Verify installation
python -c "import faiss; print(faiss.__version__)"
```

---

## Model Loading Issues

### Issue: "Model weights shape mismatch"

**Symptoms**: Error when loading pretrained weights.

**Solution**:
```python
# Load with strict=False
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Check mismatched keys
missing, unexpected = model.load_state_dict(
    checkpoint['state_dict'], strict=False
)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
```

### Issue: "LoRA adapter not found"

**Symptoms**: Cannot load LoRA weights.

**Solution**:
```python
# Check adapter config
from peft import PeftModel
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=True
)

# Verify adapter is loaded
print(model.peft_config)
```

### Issue: "Tokenizer mismatch"

**Symptoms**: Tokenization produces different results than expected.

**Solution**:
```python
# Ensure same tokenizer version
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    revision="main",  # Specify exact revision
    trust_remote_code=True
)

# Verify special tokens
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
```

---

## Online Learning System Issues

### Issue: "Experience buffer not responding"

**Symptoms**: k-NN searches timeout or fail.

**Solution**:
```python
# Check buffer status
buffer_status = experience_buffer.get_status()
print(f"Buffer size: {buffer_status['size']}")
print(f"Index status: {buffer_status['index_ready']}")

# Rebuild index if corrupted
experience_buffer.rebuild_index()

# Clear and restart buffer
experience_buffer.clear()
experience_buffer.initialize()
```

### Issue: "Update worker crashes"

**Symptoms**: Model doesn't update despite high confidence scores.

**Solution**:
```python
# Check worker status
if not update_worker.is_alive():
    print("Update worker died, restarting...")
    update_worker.restart()

# Monitor update queue
print(f"Queue size: {update_queue.qsize()}")
print(f"Updates processed: {update_worker.updates_processed}")

# Check for errors in worker log
tail -f logs/update_worker.log
```

### Issue: "IPC queue full"

**Symptoms**: Inter-process communication fails.

**Solution**:
```python
# Increase queue size
update_queue = mp.Queue(maxsize=1000)  # Increase from default

# Add timeout handling
try:
    update_queue.put(task, timeout=5.0)
except queue.Full:
    logger.warning("Update queue full, dropping task")
```

### Issue: "Confidence scores always low"

**Symptoms**: System never triggers updates.

**Solution**:
```yaml
# Adjust confidence threshold
online:
  confidence_threshold: 0.6  # Lower from default 0.7
  
# Check voting module
voting_result = voting_module.vote(neighbors)
print(f"Confidence: {voting_result.confidence}")
print(f"Agreement: {voting_result.agreement_score}")
```

---

## Performance Issues

### Issue: "Inference latency too high"

**Symptoms**: End-to-end latency exceeds requirements.

**Solution**:

1. **Profile inference pipeline**:
```python
from scripts.profile_bottlenecks import profile_inference
bottlenecks = profile_inference(model, sample_input)
print(bottlenecks.summary())
```

2. **Enable optimizations**:
```python
# Compile model
model = torch.compile(model, mode="reduce-overhead")

# Use INT8 quantization
from torch.ao.quantization import quantize_dynamic
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

3. **Optimize k-NN search**:
```python
# Use approximate search
index = faiss.IndexIVFFlat(index_flat, n_clusters)
index.nprobe = 10  # Trade accuracy for speed
```

### Issue: "Memory leak during long runs"

**Symptoms**: Memory usage grows continuously.

**Solution**:

1. **Monitor memory usage**:
```python
import tracemalloc
tracemalloc.start()

# ... run for a while ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

2. **Fix common leaks**:
```python
# Clear gradient buffers
optimizer.zero_grad(set_to_none=True)

# Delete large tensors explicitly
del large_tensor
torch.cuda.empty_cache()

# Limit experience buffer size
experience_buffer.max_size = 100000
```

3. **Use memory profiler**:
```bash
python -m memory_profiler scripts/train.py
```

---

## Debugging Tools

### Useful Commands

```bash
# System monitoring
htop  # CPU/Memory usage
nvidia-smi -l 1  # GPU monitoring
iotop  # Disk I/O
netstat -an  # Network connections

# Python debugging
python -m pdb scripts/train.py  # Interactive debugger
python -m cProfile scripts/train.py  # CPU profiling
python -m torch.utils.bottleneck scripts/train.py  # PyTorch bottleneck analysis

# Log analysis
tail -f logs/*.log  # Monitor all logs
grep ERROR logs/*.log  # Find errors
journalctl -u pixelis  # System service logs
```

### Debug Configuration

```yaml
# configs/debug.yaml
debug:
  enabled: true
  breakpoint_on_nan: true
  log_level: DEBUG
  save_intermediate_outputs: true
  profile_memory: true
  trace_calls: true
```

### Interactive Debugging

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use IPython for better interface
from IPython import embed; embed()

# Remote debugging with debugpy
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### Logging Best Practices

```python
import logging
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add context to logs
logger = logging.getLogger(__name__)
logger.info("Starting training", extra={
    'epoch': epoch,
    'batch_size': batch_size,
    'learning_rate': lr
})
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: https://github.com/yourusername/pixelis/issues
2. **Search discussions**: https://github.com/yourusername/pixelis/discussions
3. **Create detailed bug report** with:
   - Environment details (`python env_info.py`)
   - Full error traceback
   - Minimal reproduction code
   - Config files used
   - System specifications

### Diagnostic Script

Save and run this script to collect diagnostic information:

```python
# save as diagnose.py
import sys
import torch
import platform
import subprocess

print("=== System Information ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n=== Package Versions ===")
packages = ["transformers", "peft", "wandb", "hydra-core", "faiss-gpu", "accelerate"]
for package in packages:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith("Version:"):
                print(f"{package}: {line.split()[1]}")
                break
    except:
        print(f"{package}: Not installed")

print("\n=== Environment Variables ===")
import os
env_vars = ["CUDA_VISIBLE_DEVICES", "PYTHONPATH", "WANDB_API_KEY", "HF_HOME"]
for var in env_vars:
    value = os.environ.get(var, "Not set")
    if var == "WANDB_API_KEY" and value != "Not set":
        value = "***" + value[-4:]  # Hide API key
    print(f"{var}: {value}")
```

Run with: `python diagnose.py > diagnostic_report.txt`