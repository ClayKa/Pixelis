# CoTA Data Synthesis Pipeline

## Overview
This directory contains the complete Chain-of-Thought-Action (CoTA) data synthesis pipeline for the Pixelis project. The pipeline generates high-quality training data for vision-language models with pixel-space reasoning capabilities.

## Pipeline Components

### 1. Data Generation (`generate_cota_data.py`)
Synthesizes structured CoTA training samples with diverse trajectories and sample types.

**Key Features**:
- Multiple task types (counting, comparison, text extraction, etc.)
- Advanced negative samples (perceptual and logical traps)
- Self-correction trajectories
- Full provenance tracking
- Structured JSON output

**Usage**:
```bash
python generate_cota_data.py \
  --annotations path/to/annotations.json \
  --num-samples 10000 \
  --output data/cota_raw.json \
  --config configs/cota_generation.yaml \
  --seed 42
```

### 2. Data Filtering & Scoring (`filter_and_score_data.py`)
Applies comprehensive quality control through multi-stage filtering and scoring.

**Pipeline Stages**:
1. Heuristic filtering (syntax, structure, completeness)
2. Model-based quality scoring with consistency checks
3. Distribution analysis and balance monitoring
4. Hard-negative mining with weighted sampling
5. Report generation

**Usage**:
```bash
python filter_and_score_data.py \
  --input data/cota_raw.json \
  --output data/cota_filtered.json \
  --quality-threshold 4.0 \
  --enable-hard-negative-mining \
  --trap-sample-weight 1.5 \
  --stratify-by-difficulty
```

### 3. End-to-End Pipeline (`run_cota_pipeline.sh`)
Automated script that runs the complete pipeline from annotation to filtered dataset.

**Usage**:
```bash
# Generate 1000 samples (default)
./run_cota_pipeline.sh

# Generate custom number of samples
./run_cota_pipeline.sh 5000
```

## Data Format

### Input: Image Annotations
```json
{
  "image_path": "path/to/image.jpg",
  "source_dataset": "COCO",
  "original_id": "COCO_001",
  "annotations": [
    {
      "category": "person",
      "bbox": [x, y, width, height]
    }
  ],
  "text_annotations": [
    {
      "text": "STOP",
      "bbox": [x, y, width, height],
      "description": "stop sign"
    }
  ]
}
```

### Output: CoTA Sample
```json
{
  "sample_id": "cota_abc123",
  "task_type": "object_counting",
  "sample_type": "positive",
  "question": "How many people are in the image?",
  "image_path": "path/to/image.jpg",
  "trajectory": [
    {
      "action": "THINK",
      "thought": "I need to count all people...",
      "parameters": {}
    },
    {
      "action": "SEGMENT_OBJECT_AT",
      "parameters": {"coordinates": [150, 200]},
      "result": "Found person"
    }
  ],
  "answer": "3",
  "ground_truth": "3",
  "provenance": {
    "source_dataset": "COCO",
    "original_sample_id": "COCO_001",
    "synthesis_timestamp": "2025-01-13T10:00:00",
    "synthesis_version": "1.0.0"
  },
  "sampling_weight": 1.0,
  "difficulty": "medium"
}
```

## Configuration

Configuration is managed through `configs/cota_generation.yaml`:

```yaml
generation:
  temperature_range: [0.3, 0.7, 1.0]
  trap_sample_ratio: 0.2
  self_correction_ratio: 0.1
  
filtering:
  quality:
    quality_threshold: 4.0
    consistency_threshold: 1.0
  hard_negative:
    trap_sample_weight: 1.5
```

## Sample Types

| Type | Description | Weight |
|------|-------------|--------|
| **Positive** | Correct reasoning and answer | 1.0 |
| **Outcome-Negative** | Wrong final answer | 1.0 |
| **Trap-Perceptual** | Subtle perception errors | 1.5 |
| **Trap-Logical** | Flawed reasoning | 1.5 |
| **Self-Correction** | Error recovery trajectories | 1.2 |

## Task Types

1. **Object Counting**: Count specific objects in images
2. **Geometric Comparison**: Compare sizes/properties of objects
3. **Text Extraction**: Read and extract text from images
4. **Spatial Reasoning**: Understand spatial relationships
5. **Temporal Tracking**: Track objects across video frames
6. **Attribute Recognition**: Identify object attributes
7. **Relationship Detection**: Detect relationships between objects

## Quality Metrics

### Filtering Statistics
- **Heuristic Pass Rate**: Expected 85-95%
- **Quality Pass Rate**: Expected 70-80%
- **Final Retention**: Expected 60-75%

### Distribution Targets
- **Sample Types**: 60% positive, 20% trap, 10% self-correction
- **Trajectory Length**: Mean 5-8 steps, max 20
- **Action Coverage**: All operations used in >5% of samples

## Output Files

After running the pipeline, you'll find:

```
data/
├── raw/
│   └── cota_dataset_TIMESTAMP.json      # Raw generated data
├── filtered/
│   ├── cota_filtered_TIMESTAMP.json     # Filtered high-quality data
│   └── cota_filtered_TIMESTAMP.report.txt  # Quality report
└── reports/
    └── pipeline_summary_TIMESTAMP.html  # Visual summary (if enabled)
```

## Quality Report

The quality report includes:
- Filtering statistics
- Sample distribution analysis
- Task type coverage
- Trajectory length statistics
- Warnings about under-represented categories
- Sample errors (if any)

## Integration with Training

Use the filtered dataset with weighted sampling in training:

```python
from torch.utils.data import WeightedRandomSampler

# Load filtered data
with open('data/cota_filtered.json', 'r') as f:
    data = json.load(f)
    
samples = data['samples']
weights = [s['sampling_weight'] for s in samples]

# Create weighted sampler for training
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(samples),
    replacement=True
)
```

## Troubleshooting

### Common Issues

1. **Low retention rate (<50%)**
   - Lower quality threshold
   - Check annotation quality
   - Increase trajectory length bounds

2. **Distribution warnings**
   - Generate more samples
   - Adjust sample type ratios
   - Check annotation diversity

3. **Memory issues with large datasets**
   - Enable streaming mode
   - Process in batches
   - Use JSONL format

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python generate_cota_data.py --debug ...
```

## API Integration

For production use with GPT-4 scoring:

1. Set API key:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. Update config:
   ```yaml
   filtering:
     quality:
       judge_model: "gpt-4"
       use_api: true
   ```

## Performance Optimization

### Parallel Processing
```bash
# Use multiple workers
python filter_and_score_data.py \
  --num-workers 8 \
  --batch-size 500 ...
```

### Caching
Enable caching for repeated runs:
```bash
export CACHE_DIR="/path/to/cache"
python generate_cota_data.py --use-cache ...
```

## Contributing

When adding new features:
1. Update task types in `generate_cota_data.py`
2. Add validation rules in `filter_and_score_data.py`
3. Update configuration schema
4. Add tests in `tests/`
5. Update this documentation

## License

Part of the Pixelis project. See main LICENSE file.

---

For questions or issues, please refer to the main project documentation or create an issue on GitHub.