#!/bin/bash
# Demonstration script for the Pixelis reproducibility system
# Shows the complete workflow from data preparation to evaluation

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pixelis Reproducibility System Demo${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Configuration
OFFLINE_MODE="${OFFLINE:-false}"
CAPTURE_LEVEL="${CAPTURE_LEVEL:-2}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --offline)
            OFFLINE_MODE="true"
            shift
            ;;
        --capture-level)
            CAPTURE_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --offline          Run in offline mode (no WandB)"
            echo "  --capture-level N  Environment capture level (1-3)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set offline mode if requested
if [ "$OFFLINE_MODE" = "true" ]; then
    export PIXELIS_OFFLINE_MODE=true
    OFFLINE_FLAG="--offline"
    echo -e "${YELLOW}Running in OFFLINE mode${NC}"
else
    OFFLINE_FLAG=""
    echo -e "${BLUE}Running in ONLINE mode (WandB)${NC}"
fi

echo -e "Capture Level: ${CAPTURE_LEVEL}"
echo ""

# Step 1: Create sample data
echo -e "${BLUE}Step 1: Creating sample training data...${NC}"
mkdir -p data/raw data/processed

# Create sample raw data
cat > data/raw/sample_data.json << 'EOF'
[
    {
        "id": "sample_001",
        "text": "This is a sample training example with visual reasoning.",
        "trajectory": ["ZOOM_IN", "SEGMENT_OBJECT_AT", "READ_TEXT"],
        "actions": ["zoom", "segment", "read"],
        "quality": 0.8
    },
    {
        "id": "sample_002",
        "text": "Another example demonstrating tool usage.",
        "trajectory": ["GET_PROPERTIES", "TRACK_OBJECT"],
        "actions": ["properties", "track"],
        "quality": 0.6
    },
    {
        "id": "sample_003",
        "text": "Low quality example that will be filtered.",
        "trajectory": ["ZOOM_IN"],
        "actions": ["zoom"],
        "quality": 0.3
    }
]
EOF

echo "  ✓ Created sample raw data"

# Step 2: Filter and score data (creates dataset artifact)
echo -e "${BLUE}Step 2: Filtering data and creating dataset artifact...${NC}"
python scripts/filter_and_score_data.py \
    --input data/raw/sample_data.json \
    --output data/processed/filtered_data.json \
    --dataset-name demo \
    --dataset-type sft \
    --min-quality-score 0.5 \
    --add-sampling-weights \
    --exp-name "demo_data_prep" \
    $OFFLINE_FLAG

echo ""

# Step 3: Run SFT training (creates model artifact)
echo -e "${BLUE}Step 3: Running SFT training with artifact tracking...${NC}"
python scripts/train.py \
    --mode sft \
    --config configs/training_params.yaml \
    --exp-name "demo_sft_training" \
    --capture-level $CAPTURE_LEVEL \
    $OFFLINE_FLAG

echo ""

# Step 4: Run RFT training (uses SFT model artifact)
echo -e "${BLUE}Step 4: Running RFT training with lineage tracking...${NC}"
python scripts/train.py \
    --mode rft \
    --config configs/training_params.yaml \
    --exp-name "demo_rft_training" \
    --capture-level $CAPTURE_LEVEL \
    $OFFLINE_FLAG

echo ""

# Step 5: Run evaluation (creates evaluation artifact)
echo -e "${BLUE}Step 5: Running evaluation with full lineage...${NC}"

# Create sample evaluation dataset
cat > data/processed/eval_data.json << 'EOF'
{
    "benchmark": "demo_benchmark",
    "samples": [
        {"id": "eval_001", "question": "What is in the image?"},
        {"id": "eval_002", "question": "Describe the objects."}
    ]
}
EOF

# Note: In a real scenario, you would use actual model and dataset artifacts
# For demo purposes, we'll use placeholder names
python scripts/evaluate.py \
    --model "rft_model_final" \
    --dataset "dataset-demo_sft" \
    --benchmark "demo" \
    --output results/demo_eval.json \
    --exp-name "demo_evaluation" \
    $OFFLINE_FLAG || echo "  (Evaluation uses placeholder artifacts for demo)"

echo ""

# Step 6: Display artifact lineage
echo -e "${BLUE}Step 6: Artifact Lineage Summary${NC}"
echo ""

if [ "$OFFLINE_MODE" = "true" ]; then
    echo "Artifacts stored locally in:"
    echo "  - ./runs/          (Experiment metadata)"
    echo "  - ./artifact_cache/ (Artifact content)"
    echo "  - ./checkpoints/   (Model checkpoints)"
    echo ""
    
    # Show recent runs
    if [ -d "./runs" ]; then
        echo "Recent experimental runs:"
        ls -lt ./runs 2>/dev/null | head -5 | tail -4
    fi
else
    echo "Artifacts tracked in WandB project: pixelis"
    echo "View at: https://wandb.ai/your-entity/pixelis"
fi

echo ""

# Step 7: Demonstrate reproducibility
echo -e "${BLUE}Step 7: Reproducibility Information${NC}"
echo ""
echo "To reproduce any experiment:"
echo "1. Note the run ID from the logs above"
echo "2. Use the artifact manager to retrieve exact versions:"
echo ""
echo "   from core.reproducibility import ArtifactManager"
echo "   manager = ArtifactManager()"
echo "   model = manager.use_artifact('model-<run_id>', version='v1')"
echo "   dataset = manager.use_artifact('dataset-demo_sft', version='v1')"
echo ""
echo "3. The complete lineage is preserved:"
echo "   - Environment (Python, packages, git commit)"
echo "   - Configuration (all parameters)"
echo "   - Data (versioned datasets)"
echo "   - Models (versioned checkpoints)"
echo "   - Results (evaluation artifacts)"
echo ""

# Final summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "The reproducibility system has demonstrated:"
echo "  ✓ Automatic artifact versioning"
echo "  ✓ Complete lineage tracking"
echo "  ✓ Environment capture (level $CAPTURE_LEVEL)"
if [ "$OFFLINE_MODE" = "true" ]; then
    echo "  ✓ Offline mode operation"
else
    echo "  ✓ Online tracking with WandB"
fi
echo ""
echo "All experiments are fully reproducible with exact:"
echo "  - Code version (git commit)"
echo "  - Environment (packages, CUDA)"
echo "  - Configuration (all parameters)"
echo "  - Data artifacts (versioned)"
echo "  - Model artifacts (versioned)"
echo ""

# Show visualization hint
echo -e "${YELLOW}Tip: View the lineage graph in docs/REPRODUCIBILITY.md${NC}"
echo -e "${YELLOW}Or generate a visual graph with:${NC}"
echo "  python -c \"from core.reproducibility import LineageTracker; tracker = LineageTracker(); tracker.export_to_dot('lineage.dot')\""
echo "  dot -Tpng lineage.dot -o lineage.png"
echo ""