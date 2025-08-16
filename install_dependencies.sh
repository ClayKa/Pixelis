#!/bin/bash

# Pixelis Environment Setup Script
# This script installs all dependencies for the Pixelis project

set -e  # Exit on error

echo "=========================================="
echo "Pixelis Dependency Installation"
echo "=========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "pixelis" ]]; then
    echo "Please activate the pixelis conda environment first:"
    echo "  conda activate pixelis"
    exit 1
fi

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Step 1: Upgrade pip and essential tools
echo ""
echo "Step 1: Upgrading pip and essential tools..."
pip install --upgrade pip setuptools wheel

# Step 2: Install PyTorch and related packages
echo ""
echo "Step 2: Installing PyTorch ecosystem..."
# For Mac M1/M2, use MPS backend
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Detected Apple Silicon, installing PyTorch with MPS support..."
    pip install torch==2.6.0 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.6.0 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Step 3: Install key ML packages first (to avoid conflicts)
echo ""
echo "Step 3: Installing core ML packages..."
pip install transformers==4.50.0
pip install accelerate==1.6.0
pip install datasets==3.5.1
pip install safetensors==0.5.3
pip install peft==0.15.2
pip install huggingface-hub==0.30.2

# Step 4: Install Flash Attention (if supported)
echo ""
echo "Step 4: Attempting to install Flash Attention..."
if [[ $(uname -m) != 'arm64' ]]; then
    pip install flash-attn==2.7.4.post1 --no-build-isolation || echo "Flash Attention installation failed, continuing..."
else
    echo "Skipping Flash Attention on Apple Silicon"
fi

# Step 5: Install DeepSpeed
echo ""
echo "Step 5: Installing DeepSpeed..."
pip install deepspeed==0.16.3 || echo "DeepSpeed installation failed, continuing..."

# Step 6: Install vLLM (if supported)
echo ""
echo "Step 6: Installing vLLM..."
if [[ $(uname -m) != 'arm64' ]]; then
    pip install vllm==0.7.3 || echo "vLLM installation failed, continuing..."
else
    echo "Skipping vLLM on Apple Silicon"
fi

# Step 7: Install Ray
echo ""
echo "Step 7: Installing Ray..."
pip install "ray[default]==2.42.0"

# Step 8: Install TRL from git
echo ""
echo "Step 8: Installing TRL..."
pip install git+https://github.com/huggingface/trl.git@main

# Step 9: Install transformers from git (special version)
echo ""
echo "Step 9: Installing transformers from git..."
pip install --force-reinstall git+https://github.com/huggingface/transformers.git@main

# Step 10: Install remaining packages from requirements
echo ""
echo "Step 10: Installing remaining packages..."

# Create a filtered requirements file without the already installed packages
cat > requirements_filtered.txt << 'EOF'
# Remaining packages to install
einops==0.8.1
qwen-vl-utils==0.0.11
qwen_vl_utils==0.0.11
wandb==0.19.5
tensorboard
opencv-python-headless==4.11.0.86
pillow==11.2.1
numpy==2.3.1
pandas==2.3.0
scipy==1.15.1
matplotlib==3.10.0
tqdm==4.67.1
regex==2024.11.6
PyYAML==6.0.2
jsonlines
flask
loguru
sentencepiece==0.2.0
tiktoken==0.8.0
tokenizers==0.21.0
sympy==1.13.1
hydra-core==1.3.2
omegaconf==2.3.0
mlflow==3.1.1
gradio==5.12.0
pytest==8.4.1
black
isort
flake8
ruff==0.9.3
Jinja2==3.1.5
filelock==3.18.0
psutil==6.1.0
rich==13.9.4
packaging==25.0
typing_extensions==4.12.2
EOF

pip install -r requirements_filtered.txt

# Step 11: Install model-specific utilities
echo ""
echo "Step 11: Installing model-specific utilities..."
pip install liger-kernel==0.5.10 || echo "Liger kernel installation failed, continuing..."
pip install xformers==0.0.28.post3 || echo "xformers installation failed, continuing..."
pip install triton==3.1.0 || echo "Triton installation failed, continuing..."

# Step 12: Verify installation
echo ""
echo "Step 12: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Note: Some packages may have failed to install due to system compatibility."
echo "This is expected on Apple Silicon or systems without CUDA."
echo "The core functionality should still work."
echo ""
echo "To verify the environment, run:"
echo "  python verify_installation.py"