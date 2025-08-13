#!/bin/bash
# Micro-benchmark execution script for Pixelis project
# This script runs the computational cost micro-benchmark with appropriate configurations

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pixelis Micro-Benchmark Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
MODEL_NAME="${MODEL_NAME:-qwen/Qwen2.5-VL-7B}"
USE_DUMMY="${USE_DUMMY:-false}"
PRECISION="${PRECISION:-bf16}"
NUM_STEPS="${NUM_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
LR="${LR:-1e-5}"
USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-32}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-true}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
UPDATE_BUDGET="${UPDATE_BUDGET:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dummy)
            USE_DUMMY="true"
            shift
            ;;
        --no-lora)
            USE_LORA="false"
            shift
            ;;
        --no-checkpoint)
            GRAD_CHECKPOINT="false"
            shift
            ;;
        --no-update)
            UPDATE_BUDGET="false"
            shift
            ;;
        --steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dummy              Use dummy model (no download required)"
            echo "  --no-lora            Disable LoRA"
            echo "  --no-checkpoint      Disable gradient checkpointing"
            echo "  --no-update          Don't update COMPUTE_BUDGET.md"
            echo "  --steps NUM          Number of benchmark steps (default: 100)"
            echo "  --batch-size SIZE    Batch size per GPU (default: 1)"
            echo "  --gpus NUM           Number of GPUs (default: 1)"
            echo "  --model NAME         Model name/path (default: qwen/Qwen2.5-VL-7B)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${YELLOW}⚠ CUDA not detected, using CPU (results will not be representative)${NC}"
fi

# Check Python environment
echo ""
echo "Python environment:"
echo "  Python: $(python --version 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'Not installed')"
echo "  Transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1 || echo 'Not installed')"
echo "  PEFT: $(python -c 'import peft; print(peft.__version__)' 2>&1 || echo 'Not installed')"

# Build command
CMD="python scripts/micro_benchmark.py"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --num_steps $NUM_STEPS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRAD_ACCUM"
CMD="$CMD --sequence_length $SEQ_LENGTH"
CMD="$CMD --learning_rate $LR"
CMD="$CMD --num_gpus $NUM_GPUS"
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --output_dir $OUTPUT_DIR"

if [ "$USE_DUMMY" = "true" ]; then
    CMD="$CMD --use_dummy_model"
fi

if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora --lora_rank $LORA_RANK"
fi

if [ "$GRAD_CHECKPOINT" = "true" ]; then
    CMD="$CMD --gradient_checkpointing"
fi

if [ "$UPDATE_BUDGET" = "true" ]; then
    CMD="$CMD --update_budget"
fi

# Print configuration
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL_NAME"
echo "  Use Dummy: $USE_DUMMY"
echo "  Precision: $PRECISION"
echo "  Steps: $NUM_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Learning Rate: $LR"
echo "  Use LoRA: $USE_LORA (rank: $LORA_RANK)"
echo "  Gradient Checkpointing: $GRAD_CHECKPOINT"
echo "  GPUs: $NUM_GPUS"
echo "  Update Budget: $UPDATE_BUDGET"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
echo -e "${GREEN}Starting benchmark...${NC}"
echo "Command: $CMD"
echo ""

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

# Run the benchmark
if $CMD; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully!${NC}"
    
    # Show latest results
    LATEST_REPORT=$(ls -t "$OUTPUT_DIR"/benchmark_*.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        echo ""
        echo -e "${YELLOW}Latest results saved to:${NC}"
        echo "  Report: $LATEST_REPORT"
        echo "  JSON: ${LATEST_REPORT%.txt}.json"
    fi
else
    echo ""
    echo -e "${RED}✗ Benchmark failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete${NC}"
echo -e "${GREEN}========================================${NC}"