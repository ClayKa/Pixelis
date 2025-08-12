# Pixelis Project Setup Documentation

## Phase 0 Round 1: Environment and Codebase Setup ✅

This document describes the completed setup process for the Pixelis project, which integrates three major ML frameworks:
- **Pixel-Reasoner**: Visual reasoning with pixel-space operations
- **Reason-RFT**: Reinforcement Fine-Tuning framework
- **TTRL (verl)**: Test-Time Representation Learning engine

## Completed Tasks

### ✅ Task 1: Initialize Git Repository
- Created new git repository at `/Users/clayka7/Documents/Pixelis`
- Added comprehensive `.gitignore` file for Python/ML projects
- Repository structure organized with reference implementations

### ✅ Task 2: Clone and Integrate Source Repositories
- All three source repositories are available in `reference/` directory:
  - `reference/Pixel-Reasoner/`: Base pixel-space reasoning implementation
  - `reference/Reason-RFT/`: Reinforcement fine-tuning framework
  - `reference/TTRL/verl/`: Online learning framework

### ✅ Task 3: Analyze and Merge Dependencies
- Created `merge_dependencies.py` script to intelligently merge dependencies
- Analyzed 253 unique packages across 5 requirements files
- Resolved 31 version conflicts automatically
- Generated unified `requirements.txt` with categorized dependencies

#### Key Dependency Versions:
- Python: 3.10
- PyTorch: 2.6.0
- Transformers: 4.50.0 (git version)
- Accelerate: 1.6.0
- DeepSpeed: 0.16.3
- Ray: 2.42.0
- vLLM: 0.7.3 (GPU only)

### ✅ Task 4: Create Conda Environment
- Created `pixelis` conda environment with Python 3.10
- Location: `/opt/anaconda3/envs/pixelis`

### ✅ Task 5: Install Dependencies
- Created `install_dependencies.sh` for systematic installation
- Script handles platform-specific requirements (CUDA vs MPS)
- Includes fallback for optional packages

### ✅ Task 6: Export Environment for Reproducibility
- Generated `environment.yml` for exact environment reproduction
- Created `environment_setup.yml` with comprehensive dependency list
- Includes environment variables for model configuration

## Project Structure

```
Pixelis/
├── .git/                    # Git repository
├── .gitignore              # Comprehensive gitignore
├── CLAUDE.md               # AI assistant guidance
├── SETUP.md                # This file
├── environment.yml         # Exported conda environment
├── environment_setup.yml   # Clean environment specification
├── requirements.txt        # Merged Python dependencies
├── install_dependencies.sh # Installation script
├── verify_installation.py  # Verification script
├── merge_dependencies.py   # Dependency merger tool
├── reference/              # Source implementations
│   ├── Pixel-Reasoner/
│   ├── Reason-RFT/
│   └── TTRL/
└── tasks/                  # Development roadmap tasks
```

## Installation Instructions

### Quick Setup
```bash
# 1. Create and activate environment
conda env create -f environment_setup.yml
conda activate pixelis

# 2. Run installation script
./install_dependencies.sh

# 3. Verify installation
python verify_installation.py
```

### Manual Setup
```bash
# 1. Create conda environment
conda create -n pixelis python=3.10 -y
conda activate pixelis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install special packages
pip install git+https://github.com/huggingface/trl.git@main
pip install git+https://github.com/huggingface/transformers.git@main
```

## Platform-Specific Notes

### Apple Silicon (M1/M2)
- PyTorch uses MPS backend instead of CUDA
- Flash Attention, vLLM not supported
- DeepSpeed may have limited functionality
- Core training still works with MPS acceleration

### CUDA Systems
- Requires CUDA 11.8 or higher
- Full support for all acceleration libraries
- vLLM and Flash Attention available

## Environment Variables

Key environment variables for model training:
```bash
export MAX_PIXELS=4014080  # Maximum image resolution
export MIN_PIXELS=401408   # Minimum image resolution
export CUDA_HOME=/usr/local/cuda  # CUDA installation path
```

## Verification

Run the verification script to check your installation:
```bash
python verify_installation.py
```

This will check:
- Essential packages (PyTorch, Transformers, etc.)
- Training infrastructure (Ray, DeepSpeed, etc.)
- Optional acceleration libraries
- Device configuration (CUDA/MPS)

## Known Issues

1. **Flash Attention**: Only works on CUDA-enabled GPUs
2. **vLLM**: Requires CUDA, not compatible with CPU/MPS
3. **DeepSpeed**: Full features only on Linux with CUDA
4. **xformers**: May require building from source on some systems

## Next Steps

With Phase 0 Round 1 complete, the project is ready for:
- Phase 0 Round 2: Establish directory structure
- Phase 0 Round 3: Modify model architecture
- Phase 1: Begin offline training implementation

## Troubleshooting

### Import Errors
If packages fail to import, try:
```bash
pip install --upgrade <package_name>
# or
pip install --force-reinstall <package_name>
```

### CUDA Issues
Ensure CUDA toolkit matches PyTorch version:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### Memory Issues
For large models, use gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```

## Contact

For issues or questions about the setup, refer to:
- Project roadmap: `reference/ROADMAP.md`
- Development tasks: `tasks/Phase0Round*.md`
- AI assistance: `CLAUDE.md`

---
*Setup completed on: 2025-08-12*
*Environment: macOS, Apple Silicon*