# Experimental Protocol Guide

This directory contains the experimental framework for conducting rigorous, reproducible experiments in the Pixelis project.

## Quick Start

### Running a Multi-Seed Experiment

#### Using Bash Script
```bash
# Basic SFT experiment with default seeds
./scripts/run_experiments.sh -c configs/sft_config.yaml -m sft -e my_sft_exp

# RFT experiment with custom seeds
./scripts/run_experiments.sh -c configs/rft_config.yaml -m rft -e my_rft_exp -s 10,20,30

# Distributed training
./scripts/run_experiments.sh -c configs/distributed.yaml -m rft -e distributed_exp -g 4 -n 2
```

#### Using Python Script
```bash
# Basic experiment
python scripts/run_multi_seed_experiment.py \
    -c configs/sft_config.yaml \
    -m sft \
    -e my_experiment \
    -s 42 84 126

# Parallel seed execution
python scripts/run_multi_seed_experiment.py \
    -c configs/rft_config.yaml \
    -m rft \
    -e parallel_exp \
    --parallel \
    --max-workers 3

# From YAML configuration
python scripts/run_multi_seed_experiment.py \
    --from-yaml configs/experiments/example_sft_experiment.yaml
```

### Analyzing Results

```bash
# Analyze single experiment
python scripts/analyze_results.py --experiment_id my_exp_20240115_120000

# Compare multiple experiments
python scripts/analyze_results.py \
    --compare baseline_exp improved_exp \
    --metrics accuracy f1_score loss \
    --output_format markdown

# Generate detailed report
python scripts/analyze_results.py \
    --experiment_id my_exp_20240115_120000 \
    --report \
    --output_file results/my_exp_report.txt

# Plot metric comparison
python scripts/analyze_results.py \
    --compare exp1 exp2 exp3 \
    --plot_metric accuracy \
    --save_plot plots/accuracy_comparison.png
```

## Directory Structure

```
experiments/
├── README.md                  # This file
├── registry.json              # Central experiment registry
├── exp_20240115_120000/      # Example experiment output
│   ├── seed_42/              # Seed-specific outputs
│   │   ├── training.log
│   │   ├── error.log
│   │   ├── command.txt
│   │   ├── checkpoints/
│   │   └── metrics.json
│   ├── seed_84/
│   ├── seed_126/
│   └── experiment_summary.json
└── negative_results/          # Failed experiments documentation
    └── failed_exp_analysis.md
```

## Experimental Protocol Requirements

### 1. Multi-Seed Runs (Mandatory)

All experiments must be run with **at least 3 different random seeds**:
- Standard seeds: 42, 84, 126
- Additional seeds for critical experiments: 168, 210

### 2. Aggregated Reporting (Mandatory)

All metrics must be reported as **mean ± standard deviation**:
- Example: `Accuracy: 84.3 ± 0.5`
- Use `analyze_results.py` to automatically generate formatted tables

### 3. Statistical Significance Testing (Required for Claims)

When claiming improvements:
- Use paired t-test for matched samples
- Report p-values with significance markers:
  - `*` : p < 0.05
  - `**` : p < 0.01
  - `***` : p < 0.001

### 4. Reproducibility Checklist

Before publishing results:
- [ ] Ran with at least 3 seeds
- [ ] Calculated mean ± std
- [ ] Performed significance tests
- [ ] Logged to WandB
- [ ] Saved all artifacts
- [ ] Created reproducibility script

## Configuration Examples

### SFT Experiment
```yaml
experiment_name: "baseline_sft"
mode: "sft"
config_file: "configs/sft_config.yaml"
seeds: [42, 84, 126]
wandb_project: "pixelis-experiments"
wandb_tags: ["sft", "baseline"]
```

### RFT Experiment
```yaml
experiment_name: "improved_rft"
mode: "rft"
config_file: "configs/rft_config.yaml"
seeds: [42, 84, 126]
num_gpus: 2
parallel_seeds: false
```

## Registry Management

The `registry.json` file tracks all experiments:

```json
{
  "experiment_id": "exp_20240115_120000",
  "name": "baseline_sft",
  "date": "2024-01-15T12:00:00",
  "mode": "sft",
  "seeds": [42, 84, 126],
  "status": "completed",
  "wandb_runs": [...]
}
```

### Registry Operations

```python
# List all experiments
cat experiments/registry.json | jq '.[].experiment_id'

# Find completed experiments
cat experiments/registry.json | jq '.[] | select(.status=="completed")'

# Get experiment details
python -c "
import json
with open('experiments/registry.json') as f:
    registry = json.load(f)
    for exp in registry:
        print(f'{exp['experiment_id']}: {exp['status']}')
"
```

## Output Formats

### Markdown Table
```markdown
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Baseline | 80.3 ± 0.5 | 78.1 ± 0.6 |
| Improved | 84.3 ± 0.4*** | 82.5 ± 0.5*** |
```

### LaTeX Table
```latex
\begin{table}[h]
\centering
\caption{Multi-seed Experimental Results}
\begin{tabular}{lrr}
\toprule
Model & Accuracy & F1-Score \\
\midrule
Baseline & 80.3 ± 0.5 & 78.1 ± 0.6 \\
Improved & 84.3 ± 0.4*** & 82.5 ± 0.5*** \\
\bottomrule
\end{tabular}
\end{table}
```

## Best Practices

### 1. Experiment Naming
- Use descriptive names: `qwen25vl_sft_cota_v2`
- Include model, method, and dataset info
- Avoid spaces and special characters

### 2. Seed Selection
- Use consistent seeds across comparable experiments
- Document any deviation from standard seeds
- Consider seed sensitivity analysis for critical results

### 3. Metric Selection
- Report all primary metrics
- Include both task-specific and general metrics
- Document metric calculation methods

### 4. Error Handling
- Document all failed experiments
- Analyze failure patterns
- Include negative results in reports

### 5. Resource Tracking
- Log GPU hours and memory usage
- Report training time per seed
- Track inference latency

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Run seeds sequentially instead of parallel

2. **Inconsistent Results**
   - Verify all seeds use same config
   - Check for non-deterministic operations
   - Ensure data loading is deterministic

3. **WandB Sync Issues**
   - Set `WANDB_MODE=offline` for offline logging
   - Use `wandb sync` to upload later
   - Check API key configuration

4. **Statistical Test Failures**
   - Ensure same number of seeds
   - Check for NaN or infinite values
   - Verify metric calculation consistency

## Advanced Features

### Custom Statistical Tests
```python
from scripts.analyze_results import ResultAnalyzer

analyzer = ResultAnalyzer()
# Custom bootstrap test with more iterations
stat, p_value = analyzer._bootstrap_test(
    values_a=[0.85, 0.84, 0.86],
    values_b=[0.80, 0.79, 0.81],
    n_bootstrap=100000
)
```

### Parallel Seed Execution
```python
config = ExperimentConfig(
    experiment_name="parallel_test",
    mode="sft",
    config_file="config.yaml",
    parallel_seeds=True,
    max_workers=4
)
```

### Custom Metrics
```python
# Add custom metric calculation
def calculate_custom_metric(predictions, targets):
    return custom_score

# Include in results
result.metrics["custom_metric"] = calculate_custom_metric(preds, targets)
```

## Compliance Verification

Run the compliance check:
```bash
python tests/test_experimental_protocol.py
```

This verifies:
- Multi-seed requirement
- Aggregated reporting format
- Statistical testing implementation
- Registry management
- Reproducibility features

## References

- [EXPERIMENTAL_PROTOCOL.md](../docs/EXPERIMENTAL_PROTOCOL.md) - Full protocol specification
- [Statistical Methods in ML](https://arxiv.org/abs/xxx) - Best practices paper
- [Reproducibility Checklist](https://reproducibility-checklist.org)

## Support

For issues or questions:
1. Check this README
2. Review the protocol document
3. Run the test suite
4. Open an issue on GitHub