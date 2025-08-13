#!/bin/bash

#############################################################################
# Launch Script for SFT Training with Curriculum Learning
#
# This script:
# 1. Prepares the environment
# 2. Generates or processes training data
# 3. Launches SFT training with curriculum learning
#############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Pixelis SFT Training Launcher      ${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
PROCESSED_DATA_DIR="${DATA_DIR}/processed/curriculum"
RAW_DATA_DIR="${DATA_DIR}/raw"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sft"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "${PROCESSED_DATA_DIR}"
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Check if we have processed curriculum data
if [ ! -f "${PROCESSED_DATA_DIR}/cota_simple.json" ]; then
    echo -e "${YELLOW}No processed curriculum data found.${NC}"
    
    # Check if we have raw CoTA data
    if [ ! -f "${RAW_DATA_DIR}/cota_dataset.json" ]; then
        echo -e "${YELLOW}Generating sample CoTA data for testing...${NC}"
        
        # First check if generate_cota_data.py exists
        if [ -f "${PROJECT_ROOT}/scripts/generate_cota_data.py" ]; then
            python "${PROJECT_ROOT}/scripts/generate_cota_data.py" \
                --output "${RAW_DATA_DIR}/cota_dataset.json" \
                --num_samples 100 \
                --include_negatives \
                --include_traps \
                --debug
        else
            echo -e "${RED}Error: generate_cota_data.py not found!${NC}"
            echo -e "${YELLOW}Creating minimal test data...${NC}"
            
            # Create minimal test data
            cat > "${RAW_DATA_DIR}/cota_dataset.json" << 'EOF'
{
  "dataset_name": "cota_test",
  "version": "1.0",
  "samples": [
    {
      "id": "test_001",
      "question": "What is the main object in this image?",
      "image_path": "test_image_001.jpg",
      "trajectory": [
        {
          "type": "visual_operation",
          "operation": "SEGMENT_OBJECT_AT",
          "arguments": {"x": 256, "y": 256},
          "result": "Found object: cat"
        },
        {
          "type": "reasoning",
          "operation": "THINK",
          "arguments": {},
          "result": "The segmented object appears to be a cat based on its features"
        }
      ],
      "answer": "The main object is a cat",
      "label": "positive"
    },
    {
      "id": "test_002",
      "question": "How many people are in the image?",
      "image_path": "test_image_002.jpg",
      "trajectory": [
        {
          "type": "visual_operation",
          "operation": "SEGMENT_OBJECT_AT",
          "arguments": {"x": 100, "y": 150},
          "result": "Found object: person"
        },
        {
          "type": "visual_operation",
          "operation": "SEGMENT_OBJECT_AT",
          "arguments": {"x": 300, "y": 150},
          "result": "Found object: person"
        },
        {
          "type": "reasoning",
          "operation": "THINK",
          "arguments": {},
          "result": "I found 2 people in different locations"
        }
      ],
      "answer": "There are 2 people in the image",
      "label": "positive"
    }
  ]
}
EOF
        fi
    fi
    
    # Process the data with curriculum scoring
    echo -e "${YELLOW}Processing data with difficulty scoring...${NC}"
    
    if [ -f "${PROJECT_ROOT}/scripts/preprocess_data.py" ]; then
        python "${PROJECT_ROOT}/scripts/preprocess_data.py" \
            --input "${RAW_DATA_DIR}/cota_dataset.json" \
            --output "${PROCESSED_DATA_DIR}" \
            --split-by-category \
            --verbose
    else
        echo -e "${RED}Warning: preprocess_data.py not found!${NC}"
        echo -e "${YELLOW}Creating minimal processed data...${NC}"
        
        # Create minimal processed files
        for difficulty in simple medium hard; do
            cat > "${PROCESSED_DATA_DIR}/cota_${difficulty}.json" << EOF
{
  "difficulty": "${difficulty}",
  "samples": [
    {
      "id": "test_${difficulty}_001",
      "question": "Test question for ${difficulty}",
      "trajectory": [
        {"operation": "THINK", "result": "Processing ${difficulty} sample"}
      ],
      "answer": "Test answer for ${difficulty}",
      "difficulty": "${difficulty}",
      "difficulty_score": 0.5
    }
  ]
}
EOF
        done
    fi
    
    echo -e "${GREEN}✓ Data preparation complete${NC}"
else
    echo -e "${GREEN}✓ Processed curriculum data found${NC}"
fi

# Training configuration options
TRAINING_MODE="sft"
CONFIG_FILE="${PROJECT_ROOT}/configs/training_params.yaml"
EXPERIMENT_NAME="sft_curriculum_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="true"
            shift
            ;;
        --offline)
            OFFLINE_MODE="true"
            shift
            ;;
        --exp-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Build the training command
echo -e "${YELLOW}Preparing training command...${NC}"

TRAIN_CMD="python ${PROJECT_ROOT}/scripts/train.py"
TRAIN_CMD="${TRAIN_CMD} --mode ${TRAINING_MODE}"
TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG_FILE}"
TRAIN_CMD="${TRAIN_CMD} --exp-name ${EXPERIMENT_NAME}"

# Add optional flags
if [ ! -z "${RESUME_FROM}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME_FROM}"
    echo -e "${YELLOW}Resuming from checkpoint: ${RESUME_FROM}${NC}"
fi

if [ "${OFFLINE_MODE}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --offline"
    echo -e "${YELLOW}Running in offline mode (no WandB)${NC}"
fi

if [ "${DEBUG_MODE}" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=""  # Use CPU for debugging
    echo -e "${YELLOW}Debug mode: Using CPU only${NC}"
fi

# Display configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Configuration:${NC}"
echo -e "  Mode: ${TRAINING_MODE}"
echo -e "  Config: ${CONFIG_FILE}"
echo -e "  Data: ${PROCESSED_DATA_DIR}"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  Experiment: ${EXPERIMENT_NAME}"
echo -e "${GREEN}========================================${NC}"

# Launch training
echo -e "${YELLOW}Launching SFT training with curriculum learning...${NC}"
echo -e "${YELLOW}Command: ${TRAIN_CMD}${NC}"
echo ""

# Execute the training command
eval ${TRAIN_CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}  Output directory: ${OUTPUT_DIR}${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Training failed!${NC}"
    echo -e "${RED}  Check logs for details${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi