# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pixelis is a novel vision-language agent designed to reason directly within the pixel space of images and videos. The project builds upon three foundational works:
- **Pixel-Reasoner**: Provides pixel-space reasoning capabilities with visual operations
- **Reason-RFT**: Reinforcement Fine-Tuning framework for visual reasoning
- **TTRL (verl)**: Test-Time Representation Learning engine for continuous online evolution
- 
## Development Context
See @reference/ROADMAP.md for current status and next steps
Important!!!: When finish one task or one round, or even one phase in the ROADMAP.md, replace the ⚪ with the ✅.

## Repository Structure

```
Pixelis/
├── reference/               # Reference implementations from source projects
│   ├── Pixel-Reasoner/     # Base pixel-space reasoning implementation
│   │   ├── curiosity_driven_rl/  # RL training with curiosity rewards
│   │   ├── instruction_tuning/   # SFT implementation
│   │   └── onestep_evaluation/   # Evaluation scripts
│   ├── Reason-RFT/         # Reinforcement fine-tuning framework
│   │   ├── train/          # Training implementations (SFT & RL stages)
│   │   ├── eval/           # Evaluation scripts
│   │   └── scripts/        # Training execution scripts
│   └── TTRL/verl/          # Online learning framework
│       ├── trainer/        # Training orchestration
│       ├── workers/        # Distributed workers (FSDP, Megatron)
│       └── models/         # Model implementations
├── tasks/                  # Development roadmap tasks
└── ROADMAP.md             # Complete project roadmap
```

## Key Technologies & Frameworks

- **Base Models**: Qwen2.5-VL (7B), Qwen3 (8B)
- **Training Frameworks**: 
  - OpenRLHF (curiosity-driven RL)
  - LlamaFactory (SFT)
  - verl (TTRL/online learning)
- **Distributed Training**: Ray, FSDP, Megatron-LM
- **Inference**: vLLM, SGLang
- **Optimization**: GRPO (Group Relative Policy Optimization)

## Common Development Commands

### Environment Setup
```bash
# Create conda environments for different components
conda create -n pixelis-sft python=3.10
conda create -n pixelis-rl python=3.10
conda create -n pixelis-ttrl python=3.10

# Install dependencies (component-specific)
pip install -r reference/Pixel-Reasoner/curiosity_driven_rl/requirements.txt  # For RL
pip install -r reference/Reason-RFT/requirements_sft.txt                      # For SFT
pip install -r reference/TTRL/verl/requirements.txt                          # For TTRL
```

### Training Commands

#### Supervised Fine-Tuning (SFT)
```bash
# Pixel-Reasoner SFT
cd reference/Pixel-Reasoner/instruction_tuning
bash sft.sh

# Reason-RFT SFT (with curriculum learning)
cd reference/Reason-RFT
conda activate reasonrft_sft
bash scripts/train/cot_sft/resume_finetune_qwen2vl_7b_task1_cot_sft.sh
```

#### Reinforcement Learning Training
```bash
# Pixel-Reasoner RL (curiosity-driven)
cd reference/Pixel-Reasoner/curiosity_driven_rl
bash scripts/train_vlm_single.sh  # Single node
bash scripts/train_vlm_multi.sh   # Multi-node

# Reason-RFT RL (GRPO-based)
cd reference/Reason-RFT
conda activate reasonrft_rl
bash scripts/train/reason_rft/stage_rl/resume_finetune_qwen2vl_7b_task1_stage2_rl.sh
```

#### TTRL/Online Learning
```bash
cd reference/TTRL/verl
# PPO training
bash examples/ppo_trainer/run_qwen2-7b.sh
# GRPO training  
bash examples/grpo_trainer/run_qwen2-7b.sh
```

### Evaluation Commands
```bash
# Pixel-Reasoner evaluation
cd reference/Pixel-Reasoner/curiosity_driven_rl
bash scripts/eval_vlm_new.sh

# Reason-RFT evaluation
cd reference/Reason-RFT
bash scripts/eval/open_source_models/single_gpu_eval/eval_by_vllm_task1_reason_rft_single_gpu.sh
```

### Testing
```bash
# verl tests
cd reference/TTRL/verl
pytest tests/
```

## High-Level Architecture

### Training Pipeline
1. **Phase 1: Offline Training**
   - **SFT Stage**: Supervised fine-tuning with Chain-of-Thought-Action (CoTA) data
   - **RFT Stage**: Reinforcement fine-tuning with dual reward system:
     - Curiosity-driven reward (R_curiosity) for exploration
     - Trajectory coherence reward (R_coherence) for logical reasoning

2. **Phase 2: Online Evolution (TTRL)**
   - Asynchronous architecture for continuous learning
   - Experience buffer with k-NN retrieval
   - Confidence-gated conservative updates

### Key Components

#### Visual Operations Registry
- Central system for managing pixel-space operations
- Pluggable operations: `ZOOM_IN`, `SEGMENT_OBJECT_AT`, `GET_PROPERTIES`, `READ_TEXT`, `TRACK_OBJECT`

#### Reward System
- Multi-component reward orchestration
- Task reward + Curiosity reward + Coherence reward
- Normalization and curriculum-based introduction

#### Online Learning Engine
- Inference engine + Update worker (asynchronous)
- Experience buffer with hybrid k-NN index
- Three-tiered safety system for stable updates

## Important Configuration Notes

### Critical Environment Variables
```bash
# For Pixel-Reasoner RL
export MAX_PIXELS=4014080  # Maximum image resolution
export MIN_PIXELS=401408   # Minimum image resolution
export temperature=1.0      # Sampling temperature
export algo=group          # GRPO algorithm

# For verl/TTRL
export HF_ENDPOINT=https://hf-mirror.com  # For users in mainland China
```

### Model Configuration
- Always verify the base model path before training
- LoRA configuration is dynamically determined via SVD analysis
- Gradient checkpointing is enabled by default to reduce VRAM usage

### Common Issues & Solutions

1. **Context Length Exceeded**: Reduce `MAX_PIXELS` or limit max images
2. **vLLM Version Mismatch**: Reinstall transformers from git source
3. **logp_bsz Must Be 1**: Required for correct logprob computation
4. **Ray Environment Variables**: Ensure proper propagation across nodes

## Development Workflow

The project follows a structured development approach as outlined in `ROADMAP.md`:

1. **Planning**: Define goals and requirements
2. **Understanding**: Analyze the problem deeply
3. **Problem Solving**: Design solutions and architectures
4. **Coding**: Implement with clean, maintainable code
5. **Functional Review**: Test functionality
6. **Code Quality Review**: Ensure code quality and standards
7. **Commit**: Create meaningful commit messages
8. **Push**: Synchronize with GitHub

## Key Datasets

- **Training**: SA1B, FineWeb, STARQA, PartImageNet, MathVista, Ego4D
- **Evaluation**: MM-Vet, MMMU, ViRL39K, V*Bench, TallyQA-Complex, InfographicsVQA

## Performance Optimization

- Flash Attention 2 enabled by default
- Sequence packing for efficient batching
- Dynamic batching in vLLM
- INT8 quantization support
- Gradient accumulation for large batch sizes