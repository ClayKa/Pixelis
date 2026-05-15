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
conda env create -f environment.yml
conda activate pixelis

# Install dependencies
./install_dependencies.sh

# Verify installation
python requirements/verify_installation.py
```

### Smoke Demo

The repository includes a small smoke/demo workflow for validating that the
local environment and project wiring are usable. It uses generated toy data and
mock adapters; it is not a benchmark result.

```bash
bash quickstart.sh
```

### Full Training Pipeline

For complete reproduction with full datasets:

```bash
# 1. Supervised Fine-Tuning (SFT)
python scripts/train.py --mode sft --config configs/training_params.yaml --offline

# 2. Reinforcement Fine-Tuning (RFT)
python scripts/train.py --mode rft --config configs/rft_config.yaml --offline

# 3. Test-Time Reinforcement Learning (TTRL)
python scripts/train.py --mode ttrl --config configs/ttrl_config.yaml --offline

# 4. Evaluation requires concrete local model and dataset paths
python scripts/evaluate.py \
  --model local_model \
  --model-path checkpoints/model.pt \
  --dataset local_eval \
  --dataset-path data/eval.json \
  --benchmark custom \
  --offline
```

For mock-system validation only, use:

```bash
python scripts/run_online_simulation.py --config configs/training_params.yaml
```

Add `--start-update-worker` only when the host allows PyTorch multiprocessing
shared memory; the default mock smoke path keeps the worker disabled.

### Basic Usage

For detailed usage instructions, refer to:
- Training workflows: See `reference/ROADMAP.md`
- Model configuration: See `CLAUDE.md`
- Environment setup: See `environment.yml` and `requirements/verify_installation.py`
- Troubleshooting: See `docs/TROUBLESHOOTING.md`

## Real TTRL Backend

`scripts/train.py --mode ttrl` is wired through `core/engine/ttrl_trainer.py`.
It loads a real local model, consumes a JSON/JSONL online request stream, runs
confidence-gated inference through `InferenceEngine`, and applies asynchronous
updates through `UpdateWorker`.

Before running, set these fields in `configs/ttrl_config.yaml`:

```yaml
ttrl:
  request_path: "/absolute/path/to/ttrl_requests.jsonl"
  model_loader: "your_package.ttrl_loader:load_model"
  output_dir: "./outputs/ttrl"
```

The preferred loader signature is:

```python
def load_model(config):
    return model, tokenizer, processor
```

You can also set `ttrl.model_path` to a local HuggingFace checkpoint and choose
`ttrl.auto_model_class`, for example `Qwen2_5_VLForConditionalGeneration` when
your installed `transformers` version exposes that class. Remote downloads are
off by default; set `ttrl.allow_remote_download: true` only in a networked
training environment.

Each request record should carry the actual training tensors or enough text for
the configured loader/tokenizer to build them:

```json
{
  "request_id": "sample-0001",
  "question": "What text is on the sign?",
  "input_ids": [[151644, 8948, 374, 389, 279, 4146, 30]],
  "attention_mask": [[1, 1, 1, 1, 1, 1, 1]],
  "labels": [[-100, -100, -100, -100, -100, 8251, 30]],
  "embedding": [0.01, 0.02, 0.03]
}
```

For Qwen2.5-VL/Qwen3 deployments, keep the model-specific preprocessing inside
`model_loader`: load the tokenizer/processor exactly as your TTRL/verl stack
does, convert images/video frames into model inputs, and emit the fields above.
If you restore the original `reference/TTRL/verl` checkout, the same loader
boundary is the right place to wrap verl workers, Ray/FSDP configuration, or a
custom GRPO update implementation without hard-coding those choices into
Pixelis.

## Evaluation Backends

Real evaluators are intentionally not fixed to one benchmark implementation.
For production evaluation, add a benchmark adapter that:

- loads the concrete dataset from `--dataset-path`;
- loads the concrete model from `--model-path`;
- converts model outputs into the metric schema expected by `scripts/evaluate.py`;
- fails when required artifacts are missing.

`--allow-mock-metrics` is reserved for deterministic CI smoke tests. Do not use
it for reported results.

## Production vs Mock/Demo

Production paths:

- `scripts/train.py --mode sft|rft|ttrl`
- `scripts/evaluate.py` without `--allow-mock-metrics`
- `scripts/1_generate_specialized_datasets.py` with real datasource paths and
  API-backed generation
- `scripts/2_fuse_and_validate_dataset.py`

Mock/demo paths:

- `quickstart.sh`
- `scripts/quick_start.sh` option 1
- `scripts/run_online_simulation.py`
- `scripts/launch_demo.py` and `scripts/launch_public_demo.py`
- `scripts/simulate_benchmark.py` and reproducibility demo scripts

## Project Structure

```
Pixelis/
├── reference/          # Source implementations
│   ├── Pixel-Reasoner/ # Visual reasoning framework
│   ├── Reason-RFT/     # Reinforcement fine-tuning
│   └── TTRL/verl/      # Online learning engine
├── configs/            # Training, data generation, and experiment configs
│   └── ttrl_config.yaml # Real TTRL backend configuration template
├── core/engine/         # Inference, update worker, and TTRL backend wiring
├── docs/               # Architecture, reproducibility, and phase summaries
├── tasks/              # Development roadmap
├── tests/              # Unit and integration tests
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

- **Environment**: [`environment.yml`](environment.yml)
- **Development Roadmap**: [`reference/ROADMAP.md`](reference/ROADMAP.md)
- **AI Assistant Guide**: [`CLAUDE.md`](CLAUDE.md)
- **Architecture Overview**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Benchmarks & Results**: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- **Troubleshooting Guide**: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- **Security & Privacy**: [`docs/SECURITY_AND_PRIVACY.md`](docs/SECURITY_AND_PRIVACY.md)
- **Computational Budget**: [`docs/COMPUTE_BUDGET.md`](docs/COMPUTE_BUDGET.md)
- **Task Details**: `tasks/Phase*.md`
- **Historical Notes**: [`docs/archive/`](docs/archive/)

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 32GB+ RAM recommended
- 80GB+ disk space for models and data

## Status

The roadmap documents completed implementation phases, while this repository
still separates production paths from smoke/demo paths:

- SFT/RFT entrypoints are wired through `scripts/train.py`.
- TTRL production training is wired through `core/engine/ttrl_trainer.py` and fails fast only when the required local model or request stream is missing.
- Evaluation is fail-fast unless a concrete evaluator backend and local model/dataset paths are supplied.
- FAISS-heavy tests are gated behind `PIXELIS_RUN_FAISS_TESTS=1` because some local FAISS builds abort in native code.

## License

This project integrates multiple open-source components. Please refer to individual LICENSE files in the reference implementations.

## Acknowledgments

Built upon:
- [Pixel-Reasoner](https://github.com/tiger-ai-lab/pixel-reasoner) by TIGER-Lab
- [Reason-RFT](https://github.com/tanhuajie/Reason-RFT) 
- [TTRL/verl](https://github.com/volcengine/verl) by Volcano Engine

---
*For detailed development instructions, see `reference/ROADMAP.md`*
