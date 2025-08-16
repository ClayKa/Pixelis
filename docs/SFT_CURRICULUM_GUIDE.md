# SFT with Curriculum Learning - Implementation Guide

## Overview

This document describes the implementation of Supervised Fine-Tuning (SFT) with enhanced curriculum learning for the Pixelis project. The system progressively trains the model on increasingly difficult examples, with automatic advancement and rollback mechanisms based on performance.

## Architecture

### Core Components

1. **CurriculumDataset** (`scripts/train_sft.py`)
   - Manages progressive difficulty introduction
   - Starts with simple examples, gradually adds medium and hard samples
   - Supports dynamic reweighting of difficulty categories

2. **CurriculumManager** (`scripts/train_sft.py`)
   - Tracks curriculum state and progression
   - Makes advancement/rollback decisions based on performance
   - Manages cooldown periods and advancement intervals

3. **CurriculumCallback** (`scripts/train_sft.py`)
   - Integrates with HuggingFace Trainer
   - Monitors evaluation metrics
   - Triggers curriculum advancement checks
   - Logs curriculum state to WandB

4. **Unified Training Script** (`scripts/train.py`)
   - Entry point for all training modes (SFT, RFT, TTRL)
   - Handles configuration loading
   - Manages reproducibility and artifact tracking

## Configuration

### Training Parameters (`configs/training_params.yaml`)

```yaml
curriculum:
  enabled: true
  
  # Difficulty stages
  stages:
    - name: "simple"
      difficulty_mix:
        simple: 1.0
        medium: 0.0
        hard: 0.0
    - name: "balanced"
      difficulty_mix:
        simple: 0.4
        medium: 0.4
        hard: 0.2
    - name: "full"
      difficulty_mix:
        simple: 0.2
        medium: 0.3
        hard: 0.5
  
  # Advancement settings
  advancement_interval: 500  # Steps between checks
  min_performance_for_advance: 0.6
  
  # Rollback settings
  rollback_enabled: true
  rollback_threshold: -0.05  # Max performance drop
  rollback_cooldown: 1000  # Steps before retry
  rollback_factor: 2.0  # Interval multiplier after rollback
```

### Model Architecture (`configs/model_arch.yaml`)

```yaml
model:
  model_name: "Qwen/Qwen2.5-VL-7B"
  gradient_checkpointing: true
  use_lora: true
  lora_r: 32  # Dynamically determined by SVD
  lora_alpha: 64
  lora_dropout: 0.1
```

## Data Preparation

### 1. Generate CoTA Data

```bash
python scripts/generate_cota_data.py \
    --output data/raw/cota_dataset.json \
    --num_samples 10000 \
    --include_negatives \
    --include_traps
```

### 2. Preprocess with Difficulty Scoring

```bash
python scripts/preprocess_data.py \
    --input data/raw/cota_dataset.json \
    --output data/processed/curriculum \
    --split-by-category
```

This creates three files:
- `cota_simple.json`: Easy examples (short trajectories, basic operations)
- `cota_medium.json`: Moderate difficulty (medium length, some complexity)
- `cota_hard.json`: Challenging examples (long trajectories, self-corrections)

## Training Process

### 1. Launch SFT Training

```bash
# Basic launch
bash scripts/launch_sft_training.sh

# With specific options
bash scripts/launch_sft_training.sh \
    --exp-name "sft_curriculum_v1" \
    --offline  # Run without WandB
```

Or directly with Python:

```bash
python scripts/train.py \
    --mode sft \
    --config configs/training_params.yaml \
    --exp-name "sft_curriculum_experiment"
```

### 2. Resume from Checkpoint

```bash
python scripts/train.py \
    --mode sft \
    --config configs/training_params.yaml \
    --resume outputs/sft/checkpoint-1000
```

## Curriculum Progression Logic

### Advancement Triggers

The system attempts curriculum advancement when:
1. `steps_since_advance >= advancement_interval`
2. Not in rollback cooldown period
3. Not at the final stage

### Performance Evaluation

When advancement is triggered:
1. Record current validation performance (`perf_before`)
2. Advance to next difficulty stage
3. Immediately re-evaluate on new data mix (`perf_after`)
4. Compare performance delta

### Decision Making

**Keep Advancement if:**
- Performance drop < `rollback_threshold` (default: -0.05)
- Average performance > `min_performance_for_advance` (default: 0.6)

**Rollback if:**
- Performance drop >= `rollback_threshold`
- Average performance < `min_performance_for_advance`

### Rollback Behavior

When rollback occurs:
1. Revert to previous difficulty weights
2. Enter cooldown period (`rollback_cooldown` steps)
3. Increase next advancement interval by `rollback_factor`
4. Log event to WandB for analysis

## Monitoring and Logging

### WandB Metrics

The system logs comprehensive metrics to WandB:

```python
wandb.log({
    "curriculum/stage": current_stage_index,
    "curriculum/stage_name": stage_name,
    "curriculum/event": "advance" | "rollback",
    "curriculum/performance_drop": delta,
    "curriculum/rollback_count": total_rollbacks,
})
```

### Console Output

```
✅ Advanced curriculum at step 2000. Performance: 0.7234 (delta: -0.0156)
🔙 Rolled back curriculum at step 4000. Performance drop: -0.0823
```

### Final Statistics

At training completion:
```
========================================
Final Curriculum Statistics:
  Final stage: full
  Total rollbacks: 2
  Final weights: {'simple': 0.2, 'medium': 0.3, 'hard': 0.5}
  Total samples used: 15000
========================================
```

## Key Features

### 1. Dynamic Difficulty Adjustment
- Automatic progression through difficulty stages
- Performance-based advancement decisions
- Safety rollback mechanism

### 2. Memory Efficient Training
- Gradient checkpointing enabled by default
- LoRA for parameter-efficient fine-tuning
- Dynamic batch size based on available VRAM

### 3. Robust Error Handling
- Graceful handling of missing data files
- Automatic creation of minimal test data
- Checkpoint recovery support

### 4. Comprehensive Tracking
- Full reproducibility with artifact management
- Detailed curriculum state logging
- Integration with WandB for visualization

## Testing

### Unit Tests

```bash
python tests/test_sft_curriculum.py
```

Tests cover:
- CurriculumDataset functionality
- CurriculumManager logic
- CurriculumCallback integration
- End-to-end integration

### Validation Checklist

- [x] Data loading with different difficulty levels
- [x] Curriculum advancement logic
- [x] Rollback mechanism
- [x] Performance tracking
- [x] WandB logging
- [x] Checkpoint saving/loading
- [x] Gradient checkpointing
- [x] LoRA configuration

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in `training_params.yaml`
   - Ensure `gradient_checkpointing: true`
   - Reduce `max_length` for sequences

2. **Curriculum Not Advancing**
   - Check `min_performance_for_advance` threshold
   - Verify evaluation metrics are being computed
   - Ensure `advancement_interval` is reasonable

3. **Frequent Rollbacks**
   - Increase `rollback_threshold` (less sensitive)
   - Adjust `min_performance_for_advance`
   - Check data quality in harder categories

4. **Missing Data Files**
   - Run `generate_cota_data.py` first
   - Then run `preprocess_data.py`
   - Or use `launch_sft_training.sh` which handles this

## Performance Considerations

### Recommended Settings

For Qwen2.5-VL-7B on single GPU:
- Batch size: 4
- Gradient accumulation: 4 (effective batch size: 16)
- LoRA rank: 32 (or use SVD analysis)
- Max sequence length: 4096
- Gradient checkpointing: Enabled

### Expected Training Time

- 10K samples: ~4-6 hours on A100
- 100K samples: ~40-60 hours on A100
- Curriculum overhead: ~10% additional time for re-evaluations

## Future Enhancements

1. **Multi-GPU Support**
   - Distributed training with FSDP
   - Gradient accumulation across devices

2. **Advanced Curriculum Strategies**
   - Self-paced learning
   - Competence-based progression
   - Multi-dimensional difficulty scoring

3. **Online Curriculum Adaptation**
   - Real-time difficulty adjustment
   - Personalized learning paths
   - Active learning integration

## References

- [Curriculum Learning (Bengio et al., 2009)](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
- [Self-Paced Learning (Kumar et al., 2010)](https://papers.nips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)
- [HuggingFace Trainer Documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)