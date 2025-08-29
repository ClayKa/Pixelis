# CoTA Data Generation System - Quick Start Guide

## Overview
The CoTA (Chain-of-Thought-Action) data generation system is a two-stage pipeline for creating high-quality training data for the Pixelis vision-language model.

## Prerequisites

### 1. Environment Setup
```bash
# Create and activate conda environment
conda create -n pixelis python=3.10
conda activate pixelis

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
Set up your LLM API key (OpenRouter recommended):
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or add to your `.env` file:
```
OPENROUTER_API_KEY=your-api-key-here
```

## Quick Start

### Stage 1: Generate Specialized Datasets

```bash
# Basic usage
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/specialized \
    --verbose

# Dry run to test configuration
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/specialized \
    --dry-run

# Generate specific tasks only
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/specialized \
    --tasks geometric_reasoning_task targeted_ocr_task

# Resume from checkpoint (automatic)
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/specialized \
    --resume
```

### Stage 2: Fuse and Validate Datasets

```bash
# Create final SFT and RFT datasets
python scripts/2_fuse_and_validate_dataset.py \
    --fusion-manifest configs/data_fusion_manifest.yaml \
    --input-dir data_outputs/specialized \
    --output-dir data_outputs/final \
    --verbose

# Dry run to preview
python scripts/2_fuse_and_validate_dataset.py \
    --fusion-manifest configs/data_fusion_manifest.yaml \
    --input-dir data_outputs/specialized \
    --output-dir data_outputs/final \
    --dry-run
```

## Configuration

### Data Generation Manifest (`configs/data_generation_manifest.yaml`)

```yaml
# Global settings
global_config:
  checkpoint_every_n_samples: 100
  api_profiles:
    generator_api:
      base_url: "https://openrouter.ai/api/v1"
      model: "meta-llama/llama-3.2-90b-vision-instruct"

# Task definitions
tasks:
  geometric_reasoning_task:
    name: "geometric_reasoning"
    generator_class: "GeometricComparisonTaskGenerator"
    prompt_template: "prompts/geometric_reasoning.md"
    num_samples: 35000
    generator_params:
      temperature: 0.7
      max_tokens: 4096
      max_retries: 3
    data_loaders:
      - coco2017_train
      - lvis_v1_train
      - part_imagenet_train
```

### Data Fusion Manifest (`configs/data_fusion_manifest.yaml`)

```yaml
# Trajectory augmentation settings
trajectory_augmentation:
  enabled: true
  use_llm_for_correction: true
  correction_thoughts_template: "prompts/correction_template.md"

# Trajectory proportions per task
trajectory_proportions:
  geometric_tasks:
    golden_ratio: 0.6
    trap_ratio: 0.2
    self_correction_ratio: 0.2
  ocr_tasks:
    golden_ratio: 0.5
    trap_ratio: 0.25
    self_correction_ratio: 0.25

# Dataset split
dataset_split:
  sft_ratio: 0.7  # 70% for SFT
  rft_ratio: 0.3  # 30% for RFT

# Validation rules
validation_rules:
  min_samples_per_task: 1000
  max_imbalance_ratio: 10.0
  min_diversity_score: 0.5
  quality_thresholds:
    min_trajectory_length: 2
    max_trajectory_length: 50
```

## Development Mode (No API Key)

The system supports mock mode for development without API access:

```bash
# Will use mock data generation
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/test \
    --num-samples 10 \
    --verbose
```

## Monitoring Progress

### Real-time Progress
- Progress bars show generation status
- Verbose mode provides detailed logging
- Checkpoints saved automatically

### Check Generation Status
```bash
# View checkpoint files
ls -la data_outputs/specialized/checkpoints/

# Count generated samples
wc -l data_outputs/specialized/*.jsonl

# View generation logs
tail -f logs/generation.log
```

### Analyze Results
```bash
# View fusion report
cat data_outputs/final/fusion_report.json | python -m json.tool

# Check dataset statistics
cat data_outputs/final/dataset_metadata.json | python -m json.tool

# Sample generated data
head -n 1 data_outputs/final/pixelis_sft_dataset.jsonl | python -m json.tool
```

## Custom Task Generation

### 1. Create Your Task Generator

```python
# core/data_generation/my_custom_task.py
from .base_generator import BaseTaskGenerator

class MyCustomTaskGenerator(BaseTaskGenerator):
    def _build_context_placeholders(self) -> Dict[str, str]:
        # Your task-specific logic here
        return {
            'context_field_1': 'value1',
            'context_field_2': 'value2',
            # ...
        }
```

### 2. Register the Generator

```python
# core/data_generation/__init__.py
from .my_custom_task import MyCustomTaskGenerator

TASK_GENERATOR_REGISTRY = {
    # ... existing generators ...
    "MyCustomTaskGenerator": MyCustomTaskGenerator,
}
```

### 3. Create Prompt Template

```markdown
<!-- prompts/my_custom_task.md -->
Generate a CoTA trajectory for the following task:

Context: {context_field_1}
Details: {context_field_2}

Requirements:
- Use appropriate visual operations
- Provide step-by-step reasoning
- Output valid JSON format
```

### 4. Add to Manifest

```yaml
# configs/data_generation_manifest.yaml
tasks:
  my_custom_task:
    name: "my_custom"
    generator_class: "MyCustomTaskGenerator"
    prompt_template: "prompts/my_custom_task.md"
    num_samples: 10000
    data_loaders:
      - required_loader_1
      - required_loader_2
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'core'**
   ```bash
   # Run from project root
   cd /path/to/Pixelis
   python scripts/1_generate_specialized_datasets.py ...
   ```

2. **No API Key Warning**
   - System will use mock mode automatically
   - Set `OPENROUTER_API_KEY` environment variable for real generation

3. **Checkpoint Recovery**
   - Checkpoints are saved in `output_dir/checkpoints/`
   - Script automatically resumes from last checkpoint
   - Delete checkpoint file to start fresh

4. **Memory Issues**
   ```bash
   # Reduce batch size in manifest
   generator_params:
     batch_size: 50  # Lower value
   ```

5. **Rate Limiting**
   ```yaml
   # Add delay between API calls
   generator_params:
     retry_delay: 5.0  # seconds
   ```

## Production Deployment

### Large-Scale Generation

```bash
# Use screen or tmux for long-running jobs
screen -S cota_generation

# Run with production settings
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir /data/pixelis/cota_v1 \
    --verbose 2>&1 | tee logs/generation_$(date +%Y%m%d_%H%M%S).log

# Detach: Ctrl+A, D
# Reattach: screen -r cota_generation
```

### Parallel Generation

```bash
# Generate different tasks in parallel
python scripts/1_generate_specialized_datasets.py --tasks geometric_reasoning_task &
python scripts/1_generate_specialized_datasets.py --tasks targeted_ocr_task &
python scripts/1_generate_specialized_datasets.py --tasks spatiotemporal_task &
wait
```

### Quality Monitoring

```python
# scripts/monitor_quality.py
import json
from pathlib import Path

def analyze_quality(dataset_path):
    with open(dataset_path, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    # Analyze trajectory lengths
    lengths = [len(s['trajectory']) for s in samples]
    print(f"Avg trajectory length: {sum(lengths)/len(lengths):.2f}")
    
    # Check diversity
    unique_actions = set()
    for s in samples:
        for step in s['trajectory']:
            unique_actions.add(step.get('action'))
    print(f"Unique actions used: {unique_actions}")

analyze_quality('data_outputs/final/pixelis_sft_dataset.jsonl')
```

## Next Steps

1. **Verify Installation**
   ```bash
   python -c "from core.data_generation import TASK_GENERATOR_REGISTRY; print('✅ System ready!')"
   ```

2. **Run Small Test**
   ```bash
   python scripts/1_generate_specialized_datasets.py \
       --manifest configs/data_generation_manifest.yaml \
       --output-dir data_outputs/test \
       --num-samples 10 \
       --dry-run
   ```

3. **Start Generation**
   - Begin with small batches (100-1000 samples)
   - Monitor quality and adjust parameters
   - Scale up to full dataset

4. **Integration with Training**
   - Use generated datasets with training scripts
   - See `reference/Pixel-Reasoner/instruction_tuning/sft.sh`
   - See `reference/Reason-RFT/scripts/train/`

## Support

- Check logs in `logs/` directory
- Review validation errors in `data_outputs/final/validation_errors.txt`
- Examine fusion report in `data_outputs/final/fusion_report.json`

For issues, check:
1. API key configuration
2. Data loader availability
3. Prompt template paths
4. Memory and disk space
5. Network connectivity for API calls