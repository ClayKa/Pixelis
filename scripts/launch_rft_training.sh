#!/bin/bash

#############################################################################
# Launch Script for RFT (Reinforcement Fine-Tuning) Training
#
# This script:
# 1. Validates prerequisites (SFT model must exist)
# 2. Prepares the environment for RL training
# 3. Launches RFT training with GRPO and multi-component rewards
# 4. Monitors training with comprehensive metrics
#############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}    Pixelis RFT Training Launcher      ${NC}"
echo -e "${MAGENTA}     (GRPO + Multi-Component Rewards)  ${NC}"
echo -e "${MAGENTA}========================================${NC}"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
SFT_MODEL_DIR="${CHECKPOINT_DIR}/sft_curriculum_final"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/rft"
MONITOR_DIR="${OUTPUT_DIR}/monitor"
CONFIG_FILE="${PROJECT_ROOT}/configs/rft_config.yaml"

# Training parameters
EXPERIMENT_NAME="rft_grpo_$(date +%Y%m%d_%H%M%S)"
USE_WANDB="true"
DEBUG_MODE="false"
MULTI_GPU="false"
RESUME_FROM=""
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft-model)
            SFT_MODEL_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="true"
            shift
            ;;
        --offline)
            USE_WANDB="false"
            shift
            ;;
        --multi-gpu)
            MULTI_GPU="true"
            NUM_GPUS="$2"
            shift 2
            ;;
        --exp-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --monitor)
            ENABLE_MONITOR="true"
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MONITOR_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/trajectories"
mkdir -p "${OUTPUT_DIR}/logs"

# Validate prerequisites
echo -e "${YELLOW}Validating prerequisites...${NC}"

# Check if SFT model exists
if [ ! -d "${SFT_MODEL_DIR}" ] && [ -z "${RESUME_FROM}" ]; then
    echo -e "${RED}Error: SFT model not found at ${SFT_MODEL_DIR}${NC}"
    echo -e "${RED}Please run SFT training first or specify --sft-model${NC}"
    exit 1
fi

# Check if configuration file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}Config file not found. Creating default RFT config...${NC}"
    cat > "${CONFIG_FILE}" << 'EOF'
# RFT Training Configuration with Performance-Triggered Curriculum

training:
  mode: "rft"
  num_epochs: 1
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  warmup_steps: 100
  max_grad_norm: 0.5
  
  # GRPO specific
  grpo:
    enabled: true
    group_size: 4
    advantage_normalization: true
    clip_ratio: 0.2
    value_coefficient: 0.5
    entropy_coefficient: 0.01
    
  # PPO parameters
  ppo:
    num_rollouts: 4
    chunk_size: 128
    mini_batch_size: 32
    optimization_epochs: 4
    
  # Evaluation
  eval_steps: 100
  save_steps: 500
  logging_steps: 10
  
  # Checkpointing
  save_total_limit: 5
  save_at_curriculum_boundaries: true
  
# Performance-Triggered Reward Curriculum
reward_curriculum:
  enabled: true
  
  stages:
    - name: "Phase1_Learn_Goal"
      weights:
        task_reward: 1.0
        curiosity_reward: 0.0
        coherence_reward: 0.0
      exit_conditions:
        - metric: "success_rate_ma100"  # Moving average over 100 steps
          threshold: 0.70
          comparison: "greater"
        - metric: "steps_completed"
          threshold: 10000
          comparison: "greater"  # Fallback: advance after 10k steps
          
    - name: "Phase2_Learn_Coherence"
      weights:
        task_reward: 1.0
        curiosity_reward: 0.0
        coherence_reward: 0.1
      exit_conditions:
        - metric: "coherence_improvement_slope"
          threshold: 0.001
          comparison: "less"  # Advance when improvement plateaus
          patience: 5  # Check over 5 evaluation cycles
        - metric: "steps_completed"
          threshold: 20000
          comparison: "greater"
          
    - name: "Phase3_Full_Rewards"
      weights:
        task_reward: 1.0
        curiosity_reward: 0.05
        coherence_reward: 0.1
      # Final stage - no exit conditions
      
  # Monitoring settings
  metrics_window: 100  # Steps for moving average
  evaluation_frequency: 100  # Steps between curriculum checks
  
# Comprehensive Monitoring
monitoring:
  # Metrics to track
  track_metrics:
    - "reward_breakdown"  # R_final, R_curiosity, R_coherence
    - "kl_divergence"
    - "grpo_filtering_rate"
    - "trajectory_length"
    - "tool_usage_frequency"
    - "rapr"  # Rate of Pixel-space Reasoning
    
  # Moving averages
  moving_averages:
    - metric: "success_rate"
      window: 100
      name: "success_rate_ma100"
    - metric: "total_reward"
      window: 500
      name: "reward_ma500"
    - metric: "coherence_score"
      window: 100
      name: "coherence_ma100"
      
  # Real-time export for dashboard
  export_to_json: true
  json_export_path: "${MONITOR_DIR}/metrics.json"
  export_frequency: 10  # Steps
  
  # WandB configuration
  wandb:
    project: "pixelis-rft"
    entity: null  # Set your WandB entity
    tags:
      - "rft"
      - "grpo"
      - "curriculum"
    
# Model configuration
model:
  base_model_path: null  # Will be set from SFT checkpoint
  use_lora: true
  lora_config_path: "configs/lora_config.json"
  gradient_checkpointing: true
  
# Data configuration  
data:
  train_data_path: "data/processed/cota_train.json"
  eval_data_path: "data/processed/cota_eval.json"
  max_sequence_length: 2048
  
# Hardware configuration
hardware:
  mixed_precision: "bf16"
  tf32: true
  compile_model: false  # torch.compile
  use_flash_attention: true
EOF
fi

# Set environment variables
echo -e "${YELLOW}Setting environment variables...${NC}"
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"
export WANDB_PROJECT="pixelis-rft"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}"

if [ "${DEBUG_MODE}" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=""  # Use CPU for debugging
    export WANDB_MODE="disabled"
    echo -e "${YELLOW}Debug mode: Using CPU only, WandB disabled${NC}"
fi

if [ "${USE_WANDB}" = "false" ]; then
    export WANDB_MODE="disabled"
    echo -e "${YELLOW}Running in offline mode (WandB disabled)${NC}"
fi

if [ "${MULTI_GPU}" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    echo -e "${YELLOW}Multi-GPU mode: Using ${NUM_GPUS} GPUs${NC}"
fi

# Build the training command
echo -e "${YELLOW}Preparing training command...${NC}"

if [ "${MULTI_GPU}" = "true" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    # Use torchrun for distributed training
    TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500"
else
    TRAIN_CMD="python"
fi

TRAIN_CMD="${TRAIN_CMD} ${PROJECT_ROOT}/scripts/train_rft.py"
TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG_FILE}"
TRAIN_CMD="${TRAIN_CMD} --sft_model_path ${SFT_MODEL_DIR}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --exp_name ${EXPERIMENT_NAME}"

# Add optional flags
if [ ! -z "${RESUME_FROM}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume_from_checkpoint ${RESUME_FROM}"
    echo -e "${YELLOW}Resuming from checkpoint: ${RESUME_FROM}${NC}"
fi

# Launch monitor in background if requested
if [ "${ENABLE_MONITOR}" = "true" ]; then
    echo -e "${YELLOW}Launching interactive monitor in background...${NC}"
    
    # Check if monitor dependencies are installed
    if python -c "import gradio" 2>/dev/null; then
        nohup python "${PROJECT_ROOT}/scripts/launch_monitor.py" \
            --metrics_path "${MONITOR_DIR}/metrics.json" \
            --port 7860 \
            > "${OUTPUT_DIR}/logs/monitor.log" 2>&1 &
        
        MONITOR_PID=$!
        echo -e "${GREEN}✓ Monitor launched (PID: ${MONITOR_PID})${NC}"
        echo -e "${GREEN}  Access at: http://localhost:7860${NC}"
        
        # Save PID for cleanup
        echo ${MONITOR_PID} > "${OUTPUT_DIR}/.monitor.pid"
    else
        echo -e "${YELLOW}Warning: Gradio not installed. Skipping monitor.${NC}"
        echo -e "${YELLOW}Install with: pip install gradio${NC}"
    fi
fi

# Display configuration summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RFT Training Configuration:${NC}"
echo -e "  Mode: RFT with GRPO"
echo -e "  Config: ${CONFIG_FILE}"
echo -e "  SFT Model: ${SFT_MODEL_DIR}"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  Experiment: ${EXPERIMENT_NAME}"
echo -e "  Multi-GPU: ${MULTI_GPU} (${NUM_GPUS} GPUs)"
echo -e "  WandB: ${USE_WANDB}"
echo -e "${BLUE}========================================${NC}"

# Launch training
echo -e "${GREEN}Launching RFT training with GRPO...${NC}"
echo -e "${YELLOW}Command: ${TRAIN_CMD}${NC}"
echo ""

# Create a training log file
LOG_FILE="${OUTPUT_DIR}/logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# Execute the training command with logging
set +e  # Don't exit on error for training command
${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT_CODE=$?
set -e

# Cleanup monitor if running
if [ -f "${OUTPUT_DIR}/.monitor.pid" ]; then
    MONITOR_PID=$(cat "${OUTPUT_DIR}/.monitor.pid")
    if ps -p ${MONITOR_PID} > /dev/null; then
        echo -e "${YELLOW}Stopping monitor (PID: ${MONITOR_PID})...${NC}"
        kill ${MONITOR_PID} 2>/dev/null || true
    fi
    rm -f "${OUTPUT_DIR}/.monitor.pid"
fi

# Check exit status
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ RFT Training completed successfully!${NC}"
    echo -e "${GREEN}  Output directory: ${OUTPUT_DIR}${NC}"
    echo -e "${GREEN}  Checkpoints: ${OUTPUT_DIR}/checkpoints${NC}"
    echo -e "${GREEN}  Trajectories: ${OUTPUT_DIR}/trajectories${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Run post-training analysis if available
    if [ -f "${PROJECT_ROOT}/scripts/analyze_trajectories.py" ]; then
        echo -e "${YELLOW}Running trajectory analysis...${NC}"
        python "${PROJECT_ROOT}/scripts/analyze_trajectories.py" \
            --model_path "${OUTPUT_DIR}/checkpoints/final" \
            --output_dir "${OUTPUT_DIR}/analysis" \
            --num_samples 10
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Training failed with exit code: ${TRAIN_EXIT_CODE}${NC}"
    echo -e "${RED}  Check log file: ${LOG_FILE}${NC}"
    echo -e "${RED}========================================${NC}"
    exit ${TRAIN_EXIT_CODE}
fi