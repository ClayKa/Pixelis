#!/bin/bash

# ============================================================================
# CoTA Data Generation and Filtering Pipeline
# ============================================================================
# This script demonstrates the complete data synthesis pipeline for Pixelis
# It generates CoTA training data and applies comprehensive quality filtering
# ============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
CONFIG_FILE="$PROJECT_ROOT/configs/cota_generation.yaml"

# Create necessary directories
mkdir -p "$DATA_DIR/raw"
mkdir -p "$DATA_DIR/filtered"
mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   CoTA Data Pipeline Execution${NC}"
echo -e "${GREEN}========================================${NC}"

# ============================================================================
# Step 0: Create Example Annotations (for demo purposes)
# ============================================================================

echo -e "\n${YELLOW}Step 0: Creating example annotations...${NC}"

cat > "$DATA_DIR/annotations/example_annotations.json" << 'EOF'
[
  {
    "image_path": "/data/images/example_001.jpg",
    "source_dataset": "COCO",
    "original_id": "COCO_train2017_000000000001",
    "annotations": [
      {
        "category": "person",
        "bbox": [100, 100, 50, 100]
      },
      {
        "category": "car",
        "bbox": [200, 150, 100, 80]
      },
      {
        "category": "dog",
        "bbox": [50, 200, 40, 30]
      }
    ],
    "text_annotations": [
      {
        "text": "STOP",
        "bbox": [300, 50, 60, 30],
        "description": "stop sign"
      }
    ]
  },
  {
    "image_path": "/data/images/example_002.jpg",
    "source_dataset": "MathVista",
    "original_id": "MathVista_001",
    "annotations": [
      {
        "category": "circle",
        "bbox": [150, 150, 100, 100]
      },
      {
        "category": "square",
        "bbox": [300, 150, 80, 80]
      }
    ],
    "text_annotations": [
      {
        "text": "Area = πr²",
        "bbox": [150, 260, 100, 20],
        "description": "formula text"
      }
    ]
  }
]
EOF

echo -e "${GREEN}✓ Example annotations created${NC}"

# ============================================================================
# Step 1: Generate CoTA Data
# ============================================================================

echo -e "\n${YELLOW}Step 1: Generating CoTA training data...${NC}"

# Set generation parameters
NUM_SAMPLES=${1:-1000}  # Default to 1000 samples, or use first argument
OUTPUT_FILE="$DATA_DIR/raw/cota_dataset_$(date +%Y%m%d_%H%M%S).json"

echo "Generating $NUM_SAMPLES samples..."

python "$SCRIPT_DIR/generate_cota_data.py" \
  --annotations "$DATA_DIR/annotations/example_annotations.json" \
  --num-samples "$NUM_SAMPLES" \
  --output "$OUTPUT_FILE" \
  --config "$CONFIG_FILE" \
  --seed 42 \
  || { echo -e "${RED}Data generation failed!${NC}"; exit 1; }

echo -e "${GREEN}✓ Generated $NUM_SAMPLES samples${NC}"
echo "Output saved to: $OUTPUT_FILE"

# ============================================================================
# Step 2: Filter and Score Data
# ============================================================================

echo -e "\n${YELLOW}Step 2: Filtering and scoring data...${NC}"

FILTERED_OUTPUT="$DATA_DIR/filtered/cota_filtered_$(date +%Y%m%d_%H%M%S).json"

python "$SCRIPT_DIR/filter_and_score_data.py" \
  --input "$OUTPUT_FILE" \
  --output "$FILTERED_OUTPUT" \
  --dataset-name "cota_v1" \
  --dataset-type "sft" \
  --quality-threshold 4.0 \
  --min-trajectory-length 2 \
  --max-trajectory-length 20 \
  --enable-hard-negative-mining \
  --trap-sample-weight 1.5 \
  --self-correction-weight 1.2 \
  --stratify-by-difficulty \
  --min-samples-per-category 10 \
  --seed 42 \
  || { echo -e "${RED}Data filtering failed!${NC}"; exit 1; }

echo -e "${GREEN}✓ Data filtered and scored${NC}"
echo "Filtered output: $FILTERED_OUTPUT"
echo "Quality report: ${FILTERED_OUTPUT%.json}.report.txt"

# ============================================================================
# Step 3: Generate Summary Statistics
# ============================================================================

echo -e "\n${YELLOW}Step 3: Generating summary statistics...${NC}"

python -c "
import json
import sys
from pathlib import Path

# Load filtered data
filtered_path = '$FILTERED_OUTPUT'
with open(filtered_path, 'r') as f:
    data = json.load(f)

samples = data.get('samples', [])
metadata = data.get('metadata', {})
stats = metadata.get('statistics', {})

print('='*60)
print('PIPELINE SUMMARY')
print('='*60)
print(f'Total samples generated: {stats.get(\"total_samples\", 0)}')
print(f'After heuristic filtering: {stats.get(\"heuristic_filtered\", 0)}')
print(f'After quality filtering: {stats.get(\"quality_filtered\", 0)}')
print(f'Final samples: {stats.get(\"final_samples\", 0)}')

if stats.get('final_samples', 0) > 0:
    retention = (stats['final_samples'] / stats['total_samples']) * 100
    print(f'Retention rate: {retention:.1f}%')

# Sample type distribution
if 'distribution' in stats:
    dist = stats['distribution']
    if 'sample_types' in dist:
        print('\nSample Type Distribution:')
        for sample_type, count in dist['sample_types'].items():
            print(f'  {sample_type}: {count}')
    
    if 'task_types' in dist:
        print('\nTask Type Distribution:')
        for task_type, count in dist['task_types'].items():
            print(f'  {task_type}: {count}')

# Warnings
if stats.get('warnings'):
    print('\n⚠️  Warnings:')
    for warning in stats['warnings'][:5]:
        print(f'  - {warning}')

print('='*60)
"

# ============================================================================
# Step 4: Validate Output
# ============================================================================

echo -e "\n${YELLOW}Step 4: Validating output files...${NC}"

# Check if files exist and have content
if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}✓ Raw data file valid${NC}"
else
    echo -e "${RED}✗ Raw data file missing or empty${NC}"
    exit 1
fi

if [ -f "$FILTERED_OUTPUT" ] && [ -s "$FILTERED_OUTPUT" ]; then
    echo -e "${GREEN}✓ Filtered data file valid${NC}"
else
    echo -e "${RED}✗ Filtered data file missing or empty${NC}"
    exit 1
fi

REPORT_FILE="${FILTERED_OUTPUT%.json}.report.txt"
if [ -f "$REPORT_FILE" ] && [ -s "$REPORT_FILE" ]; then
    echo -e "${GREEN}✓ Quality report generated${NC}"
else
    echo -e "${RED}✗ Quality report missing${NC}"
    exit 1
fi

# ============================================================================
# Completion
# ============================================================================

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Pipeline Execution Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\nOutput files:"
echo "  • Raw data: $OUTPUT_FILE"
echo "  • Filtered data: $FILTERED_OUTPUT"
echo "  • Quality report: $REPORT_FILE"

echo -e "\nNext steps:"
echo "  1. Review the quality report for any warnings"
echo "  2. Inspect sample data for quality verification"
echo "  3. Use filtered data for SFT training (Phase 1 Round 2)"

# Optional: Display first few samples
echo -e "\n${YELLOW}Sample data preview:${NC}"
python -c "
import json
import sys

with open('$FILTERED_OUTPUT', 'r') as f:
    data = json.load(f)
    samples = data.get('samples', [])
    
if samples:
    sample = samples[0]
    print('First sample:')
    print(f'  ID: {sample.get(\"sample_id\", \"N/A\")}')
    print(f'  Type: {sample.get(\"task_type\", \"N/A\")}')
    print(f'  Question: {sample.get(\"question\", \"N/A\")[:100]}...')
    print(f'  Trajectory length: {len(sample.get(\"trajectory\", []))}')
    print(f'  Answer: {sample.get(\"answer\", \"N/A\")}')
"

exit 0