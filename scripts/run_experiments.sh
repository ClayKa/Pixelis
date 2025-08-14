#!/bin/bash
# Multi-seed experimental run script for Pixelis
# This script automates running experiments with multiple random seeds for statistical validity

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run multi-seed experiments for Pixelis models

OPTIONS:
    -c, --config CONFIG_FILE    Path to configuration file (required)
    -m, --mode MODE            Training mode: sft or rft (required)
    -e, --exp-name NAME        Experiment name (required)
    -s, --seeds SEEDS          Comma-separated list of seeds (default: 42,84,126)
    -g, --gpus GPUS           Number of GPUs to use (default: 1)
    -n, --num-nodes NODES      Number of nodes for distributed training (default: 1)
    -w, --wandb-project NAME   WandB project name (default: pixelis-experiments)
    -t, --wandb-tags TAGS      Comma-separated WandB tags
    -d, --dry-run             Print commands without executing
    -r, --resume              Resume from checkpoint if available
    -o, --output-dir DIR      Base output directory (default: outputs/experiments)
    -h, --help                Show this help message

EXAMPLES:
    # Run SFT with default seeds
    $0 -c configs/sft_config.yaml -m sft -e baseline_sft

    # Run RFT with custom seeds
    $0 -c configs/rft_config.yaml -m rft -e pixelis_rft -s 10,20,30

    # Distributed training on 2 nodes with 8 GPUs each
    $0 -c configs/distributed.yaml -m rft -e large_scale -g 8 -n 2

    # Dry run to preview commands
    $0 -c configs/test.yaml -m sft -e test_run --dry-run
EOF
}

# Default values
SEEDS="42,84,126"
GPUS=1
NUM_NODES=1
WANDB_PROJECT="pixelis-experiments"
WANDB_TAGS=""
DRY_RUN=false
RESUME=false
OUTPUT_DIR="outputs/experiments"
CONFIG_FILE=""
MODE=""
EXP_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        -s|--seeds)
            SEEDS="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -n|--num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        -w|--wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -t|--wandb-tags)
            WANDB_TAGS="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_FILE" ]]; then
    print_error "Configuration file is required (-c/--config)"
    usage
    exit 1
fi

if [[ -z "$MODE" ]]; then
    print_error "Training mode is required (-m/--mode)"
    usage
    exit 1
fi

if [[ "$MODE" != "sft" && "$MODE" != "rft" ]]; then
    print_error "Mode must be either 'sft' or 'rft'"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    print_error "Experiment name is required (-e/--exp-name)"
    usage
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Convert comma-separated seeds to array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Create experiment registry entry
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="${EXP_NAME}_${TIMESTAMP}"
REGISTRY_FILE="experiments/registry.json"

# Create experiments directory if it doesn't exist
mkdir -p experiments
mkdir -p "$OUTPUT_DIR"

# Initialize registry file if it doesn't exist
if [[ ! -f "$REGISTRY_FILE" ]]; then
    echo "[]" > "$REGISTRY_FILE"
fi

# Function to add experiment to registry
add_to_registry() {
    local exp_id=$1
    local exp_name=$2
    local seeds=$3
    local mode=$4
    local config=$5
    
    python3 << EOF
import json
import os
from datetime import datetime

registry_file = "$REGISTRY_FILE"
with open(registry_file, 'r') as f:
    registry = json.load(f)

entry = {
    "experiment_id": "$exp_id",
    "name": "$exp_name",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "mode": "$mode",
    "config": "$config",
    "seeds": $seeds,
    "wandb_project": "$WANDB_PROJECT",
    "wandb_runs": [],
    "status": "running",
    "output_dir": "$OUTPUT_DIR/$exp_id"
}

registry.append(entry)

with open(registry_file, 'w') as f:
    json.dump(registry, f, indent=2)

print(f"Registered experiment: {entry['experiment_id']}")
EOF
}

# Function to update registry status
update_registry_status() {
    local exp_id=$1
    local status=$2
    local seed=$3
    local run_id=$4
    
    python3 << EOF
import json

registry_file = "$REGISTRY_FILE"
with open(registry_file, 'r') as f:
    registry = json.load(f)

for entry in registry:
    if entry["experiment_id"] == "$exp_id":
        if "$status" == "completed":
            entry["status"] = "completed"
        elif "$status" == "failed":
            entry["status"] = "failed"
        
        if "$run_id":
            entry["wandb_runs"].append({
                "seed": $seed,
                "run_id": "$run_id"
            })
        break

with open(registry_file, 'w') as f:
    json.dump(registry, f, indent=2)
EOF
}

# Register experiment
print_info "Registering experiment: $EXP_ID"
if [[ "$DRY_RUN" == false ]]; then
    add_to_registry "$EXP_ID" "$EXP_NAME" "[${SEEDS}]" "$MODE" "$CONFIG_FILE"
fi

# Print experiment configuration
print_info "="*60
print_info "Experiment Configuration"
print_info "="*60
print_info "Experiment ID: $EXP_ID"
print_info "Experiment Name: $EXP_NAME"
print_info "Mode: $MODE"
print_info "Config: $CONFIG_FILE"
print_info "Seeds: ${SEED_ARRAY[@]}"
print_info "GPUs: $GPUS"
print_info "Nodes: $NUM_NODES"
print_info "WandB Project: $WANDB_PROJECT"
print_info "Output Directory: $OUTPUT_DIR/$EXP_ID"
print_info "="*60

# Function to run single seed experiment
run_single_seed() {
    local seed=$1
    local seed_idx=$2
    local total_seeds=$3
    
    print_info ""
    print_info "="*60
    print_info "Running Seed $seed_idx/$total_seeds: $seed"
    print_info "="*60
    
    # Create seed-specific output directory
    SEED_OUTPUT_DIR="$OUTPUT_DIR/$EXP_ID/seed_$seed"
    mkdir -p "$SEED_OUTPUT_DIR"
    
    # Build the training command
    CMD="python scripts/train.py"
    CMD="$CMD --config $CONFIG_FILE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --seed $seed"
    CMD="$CMD --output_dir $SEED_OUTPUT_DIR"
    CMD="$CMD --wandb_project $WANDB_PROJECT"
    CMD="$CMD --wandb_run_name ${EXP_NAME}_seed${seed}"
    CMD="$CMD --experiment_id $EXP_ID"
    
    # Add tags
    if [[ -n "$WANDB_TAGS" ]]; then
        CMD="$CMD --wandb_tags $WANDB_TAGS,multi_seed,seed_$seed"
    else
        CMD="$CMD --wandb_tags multi_seed,seed_$seed"
    fi
    
    # Add resume flag if specified
    if [[ "$RESUME" == true ]]; then
        CMD="$CMD --resume"
    fi
    
    # Handle distributed training
    if [[ $GPUS -gt 1 || $NUM_NODES -gt 1 ]]; then
        DISTRIBUTED_CMD="torchrun"
        DISTRIBUTED_CMD="$DISTRIBUTED_CMD --nproc_per_node=$GPUS"
        
        if [[ $NUM_NODES -gt 1 ]]; then
            DISTRIBUTED_CMD="$DISTRIBUTED_CMD --nnodes=$NUM_NODES"
            DISTRIBUTED_CMD="$DISTRIBUTED_CMD --node_rank=\$NODE_RANK"
            DISTRIBUTED_CMD="$DISTRIBUTED_CMD --master_addr=\$MASTER_ADDR"
            DISTRIBUTED_CMD="$DISTRIBUTED_CMD --master_port=\$MASTER_PORT"
        fi
        
        CMD="$DISTRIBUTED_CMD $CMD"
    fi
    
    # Log command to file
    echo "$CMD" > "$SEED_OUTPUT_DIR/command.txt"
    
    print_info "Command: $CMD"
    
    if [[ "$DRY_RUN" == true ]]; then
        print_warning "DRY RUN - Command not executed"
    else
        # Run the training
        print_info "Starting training..."
        
        # Capture output and error
        LOG_FILE="$SEED_OUTPUT_DIR/training.log"
        ERROR_FILE="$SEED_OUTPUT_DIR/error.log"
        
        if $CMD > >(tee "$LOG_FILE") 2> >(tee "$ERROR_FILE" >&2); then
            print_success "Seed $seed completed successfully"
            
            # Extract WandB run ID from log if available
            WANDB_RUN_ID=$(grep -oP 'wandb: Run ID: \K[^\s]+' "$LOG_FILE" || echo "")
            if [[ -n "$WANDB_RUN_ID" ]]; then
                update_registry_status "$EXP_ID" "running" "$seed" "$WANDB_RUN_ID"
            fi
        else
            print_error "Seed $seed failed! Check $ERROR_FILE for details"
            update_registry_status "$EXP_ID" "failed" "$seed" ""
            
            # Ask whether to continue with other seeds
            read -p "Continue with remaining seeds? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Main execution loop
TOTAL_SEEDS=${#SEED_ARRAY[@]}
print_info "Starting multi-seed experiment with $TOTAL_SEEDS seeds"

for i in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$i]}"
    SEED_IDX=$((i + 1))
    
    run_single_seed "$SEED" "$SEED_IDX" "$TOTAL_SEEDS"
done

# Mark experiment as completed
if [[ "$DRY_RUN" == false ]]; then
    update_registry_status "$EXP_ID" "completed" "" ""
fi

print_success ""
print_success "="*60
print_success "All seeds completed!"
print_success "Experiment ID: $EXP_ID"
print_success "Results directory: $OUTPUT_DIR/$EXP_ID"
print_success "="*60

# Generate summary
if [[ "$DRY_RUN" == false ]]; then
    print_info "Generating experiment summary..."
    
    python3 << EOF
import json
import os
from pathlib import Path

exp_dir = Path("$OUTPUT_DIR/$EXP_ID")
summary = {
    "experiment_id": "$EXP_ID",
    "experiment_name": "$EXP_NAME",
    "mode": "$MODE",
    "config": "$CONFIG_FILE",
    "seeds": $SEEDS,
    "seed_outputs": {}
}

for seed in $SEEDS:
    seed_dir = exp_dir / f"seed_{seed}"
    if seed_dir.exists():
        summary["seed_outputs"][seed] = {
            "directory": str(seed_dir),
            "log_file": str(seed_dir / "training.log") if (seed_dir / "training.log").exists() else None,
            "checkpoints": [str(f) for f in seed_dir.glob("checkpoint-*")] if seed_dir.exists() else []
        }

summary_file = exp_dir / "experiment_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_file}")
EOF
fi

print_info "To analyze results, run:"
print_info "  python scripts/analyze_results.py --experiment_id $EXP_ID"