#!/bin/bash

# SVD Analysis Workflow Script
# This script runs the complete workflow for dynamic LoRA rank determination
# 1. Preliminary fine-tuning on a small subset of data
# 2. SVD analysis on weight deltas
# 3. Generation of optimized LoRA configuration

set -e  # Exit on error

# Configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-7B"}
DATA_PATH=${DATA_PATH:-"data/cota_train.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"saved_models/preliminary_finetune"}
SUBSET_RATIO=${SUBSET_RATIO:-0.01}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}

# SVD Analysis parameters
SVD_THRESHOLD=${SVD_THRESHOLD:-0.9}
MIN_RANK=${MIN_RANK:-4}
MAX_RANK=${MAX_RANK:-128}
SMOOTHING_FACTOR=${SMOOTHING_FACTOR:-0.8}

echo "=========================================="
echo "PIXELIS SVD ANALYSIS WORKFLOW"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Run preliminary fine-tuning
echo "Step 1: Running preliminary fine-tuning..."
echo "------------------------------------------"

python scripts/preliminary_finetune.py \
    --model-name "$MODEL_NAME" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --subset-ratio $SUBSET_RATIO \
    --max-samples $MAX_SAMPLES \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRADIENT_ACCUMULATION \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS

# Check if fine-tuning was successful
if [ ! -d "$OUTPUT_DIR/pretrained" ] || [ ! -d "$OUTPUT_DIR/finetuned" ]; then
    echo "Error: Fine-tuning did not complete successfully."
    echo "Expected directories not found:"
    echo "  - $OUTPUT_DIR/pretrained"
    echo "  - $OUTPUT_DIR/finetuned"
    exit 1
fi

echo ""
echo "Fine-tuning completed successfully!"
echo ""

# Step 2: Run SVD analysis
echo "Step 2: Running SVD analysis..."
echo "------------------------------------------"

python scripts/analyze_lora_ranks.py \
    --pretrained "$OUTPUT_DIR/pretrained" \
    --finetuned "$OUTPUT_DIR/finetuned" \
    --output-dir "analysis_outputs/svd" \
    --svd-threshold $SVD_THRESHOLD \
    --min-rank $MIN_RANK \
    --max-rank $MAX_RANK \
    --smoothing-factor $SMOOTHING_FACTOR

# Check if SVD analysis was successful
if [ ! -f "configs/lora_rank_config.json" ]; then
    echo "Error: SVD analysis did not complete successfully."
    echo "LoRA configuration file not found: configs/lora_rank_config.json"
    exit 1
fi

echo ""
echo "=========================================="
echo "SVD ANALYSIS WORKFLOW COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Pretrained checkpoint: $OUTPUT_DIR/pretrained"
echo "  - Finetuned checkpoint: $OUTPUT_DIR/finetuned"
echo "  - LoRA configuration: configs/lora_rank_config.json"
echo "  - Analysis outputs: analysis_outputs/svd/"
echo ""
echo "Next steps:"
echo "  1. Review the generated LoRA configuration"
echo "  2. Use the configuration for efficient fine-tuning"
echo "  3. Run tests: python -m pytest tests/modules/test_model_init.py"
echo ""