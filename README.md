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
git clone <repository-url>
cd Pixelis

# Create and activate conda environment
conda env create -f environment_setup.yml
conda activate pixelis

# Install dependencies
./install_dependencies.sh

# Verify installation
python verify_installation.py
```

### Basic Usage

For detailed usage instructions, refer to:
- Training workflows: See `reference/ROADMAP.md`
- Model configuration: See `CLAUDE.md`
- Setup details: See `SETUP.md`

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

- **Setup Guide**: `SETUP.md`
- **Development Roadmap**: `reference/ROADMAP.md`
- **AI Assistant Guide**: `CLAUDE.md`
- **Task Details**: `tasks/Phase*.md`

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 32GB+ RAM recommended
- 80GB+ disk space for models and data

## Status

✅ **Phase 0 Round 1 Complete**: Environment setup and dependency management

## License

This project integrates multiple open-source components. Please refer to individual LICENSE files in the reference implementations.

## Acknowledgments

Built upon:
- [Pixel-Reasoner](https://github.com/tiger-ai-lab/pixel-reasoner) by TIGER-Lab
- [Reason-RFT](https://github.com/tanhuajie/Reason-RFT) 
- [TTRL/verl](https://github.com/volcengine/verl) by Volcano Engine

---
*For detailed development instructions, see `reference/ROADMAP.md`*