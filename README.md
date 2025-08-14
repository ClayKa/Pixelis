# Pixelis: An Evolving Agent for Pixel-Space Reasoning

## Overview

Pixelis is a novel vision-language agent designed to reason directly within the pixel space of images and videos. This project combines three cutting-edge ML frameworks to create a continuously evolving visual intelligence system.

## Key Features

- **Pixel-Space Reasoning**: Direct interaction with visual data through operations like ZOOM_IN, SEGMENT_OBJECT_AT, READ_TEXT, and TRACK_OBJECT
- **Dual Reward System**: Curiosity-driven exploration + trajectory coherence for logical reasoning
- **Online Evolution**: Continuous learning and adaptation through Test-Time Representation Learning (TTRL)
- **Multi-Model Support**: Built for Qwen2.5-VL (7B) and Qwen3 (8B) base models

## Architecture

The project integrates three major components:

1. **Pixel-Reasoner**: Provides core pixel-space reasoning capabilities
2. **Reason-RFT**: Implements reinforcement fine-tuning with GRPO
3. **TTRL/verl**: Enables online learning and continuous evolution

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pixelis/pixelis.git
cd Pixelis

# Create and activate conda environment
conda env create -f environment_setup.yml
conda activate pixelis

# Install dependencies
./install_dependencies.sh

# Verify installation
python verify_installation.py
```

### 🚀 Reproducibility Kit - Run in 15 Minutes!

We provide a complete reproducibility kit that allows you to verify our core results on consumer-grade hardware (e.g., RTX 4090) in just 15 minutes:

```bash
# Step 1: Download the reproducibility kit
wget https://github.com/pixelis/releases/pixelis_reproducibility_kit_v1.0.0.tar.gz
tar -xzf pixelis_reproducibility_kit_v1.0.0.tar.gz
cd reproducibility_kit

# Step 2: Run the quickstart script (15 minutes on RTX 4090)
bash quickstart.sh
```

This will:
1. ✅ Set up a minimal environment
2. ✅ Run SFT training on tiny dataset (100 samples, 5 minutes)
3. ✅ Run RFT training with reward shaping (200 samples, 10 minutes)
4. ✅ Evaluate pre-trained minimal adapters
5. ✅ Generate comparison plots showing RFT improvements
6. ✅ Launch interactive demo (optional)

**Expected Results on Tiny Dataset:**
- SFT Baseline: ~65% accuracy
- RFT-Full: ~72% accuracy (+7% improvement)
- Clear evidence of curiosity-driven exploration
- Improved trajectory coherence

### Full Training Pipeline

For complete reproduction with full datasets:

```bash
# 1. Supervised Fine-Tuning (SFT)
python scripts/train.py --mode sft --config configs/experiments/pixelis_sft.yaml

# 2. Reinforcement Fine-Tuning (RFT)
python scripts/train.py --mode rft --config configs/experiments/pixelis_rft.yaml

# 3. Online Adaptation (TTRL)
python scripts/run_online_simulation.py --config configs/experiments/pixelis_online.yaml

# 4. Evaluation
python scripts/evaluate.py --config configs/experiments/evaluation.yaml
```

### Basic Usage

For detailed usage instructions, refer to:
- Training workflows: See `reference/ROADMAP.md`
- Model configuration: See `CLAUDE.md`
- Setup details: See `SETUP.md`
- Troubleshooting: See `docs/TROUBLESHOOTING.md`

## Project Structure

```
Pixelis/
├── reference/          # Source implementations
│   ├── Pixel-Reasoner/ # Visual reasoning framework
│   ├── Reason-RFT/     # Reinforcement fine-tuning
│   └── TTRL/verl/      # Online learning engine
├── tasks/              # Development roadmap
├── requirements.txt    # Merged dependencies
└── CLAUDE.md          # AI assistant guidance
```

## Training Pipeline

### Phase 1: Offline Training
- Supervised Fine-Tuning (SFT) with Chain-of-Thought-Action data
- Reinforcement Fine-Tuning (RFT) with dual reward system

### Phase 2: Online Evolution
- Asynchronous inference and learning
- Experience buffer with k-NN retrieval
- Conservative, confidence-gated updates

## Key Technologies

- **Base Models**: Qwen2.5-VL, Qwen3
- **Training**: PyTorch, DeepSpeed, Ray, vLLM
- **Optimization**: GRPO, Flash Attention, LoRA
- **Infrastructure**: HuggingFace, Weights & Biases

## Documentation

- **Setup Guide**: [`SETUP.md`](SETUP.md)
- **Development Roadmap**: [`reference/ROADMAP.md`](reference/ROADMAP.md)
- **AI Assistant Guide**: [`CLAUDE.md`](CLAUDE.md)
- **Architecture Overview**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Benchmarks & Results**: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- **Troubleshooting Guide**: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- **Security & Privacy**: [`docs/SECURITY_AND_PRIVACY.md`](docs/SECURITY_AND_PRIVACY.md)
- **Computational Budget**: [`COMPUTE_BUDGET.md`](COMPUTE_BUDGET.md)
- **Task Details**: `tasks/Phase*.md`

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 32GB+ RAM recommended
- 80GB+ disk space for models and data

## Status

✅ **Phase 0 Complete**: Project Initialization and Setup (6 rounds)
✅ **Phase 1 Complete**: Offline Training - SFT and RFT (4 rounds)  
✅ **Phase 2 Complete**: Online Training - TTRL Evolution (6 rounds)
✅ **Phase 3 Complete**: Experiments, Evaluation, and Analysis (5 rounds)

## License

This project integrates multiple open-source components. Please refer to individual LICENSE files in the reference implementations.

## Acknowledgments

Built upon:
- [Pixel-Reasoner](https://github.com/tiger-ai-lab/pixel-reasoner) by TIGER-Lab
- [Reason-RFT](https://github.com/tanhuajie/Reason-RFT) 
- [TTRL/verl](https://github.com/volcengine/verl) by Volcano Engine

---
*For detailed development instructions, see `reference/ROADMAP.md`*