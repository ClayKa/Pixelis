# Case Study Generation Script

## Overview
`generate_case_studies.py` creates comprehensive qualitative analysis of reasoning trajectories for the Pixelis project.

## Purpose
Generate publication-quality visualizations that highlight:
- Curiosity-driven exploration differences
- Self-correction behaviors
- Tool usage pattern variations
- Critical reasoning path divergences
- Model performance variations

## Usage
```bash
python scripts/generate_case_studies.py \
    --base_model /path/to/base/model \
    --online_model /path/to/online/model \
    --test_dataset /path/to/test_dataset.json \
    --num_studies 5 \
    --strategy diverse
```

## Arguments
- `--base_model`: Path to base model checkpoint
- `--online_model`: Path to online model checkpoint
- `--test_dataset`: Path to test dataset JSON
- `--num_studies`: Number of case studies to generate (default: 5)
- `--strategy`: Case study selection strategy 
  - `diverse`: Balanced selection
  - `curiosity`: Focus on curiosity-driven exploration
  - `tool_usage`: Emphasize tool usage variations

## Outputs
1. `case_studies.html`: Interactive visualization
2. `case_studies_summary.md`: Markdown analysis
3. `case_studies/`: Directory with PNG visualizations
4. Wandb logging of case study metrics

## Requirements
- Python 3.10+
- PyTorch
- Plotly
- Matplotlib
- Numpy