# Experimental Protocol for Pixelis Project

## Overview

This document establishes the rigorous experimental standards that must be followed for all experiments conducted in the Pixelis project. These protocols ensure scientific validity, statistical significance, and reproducibility of all reported results.

## Core Principles

1. **Reproducibility**: Every experiment must be fully reproducible with documented configurations, environments, and random seeds.
2. **Statistical Validity**: All comparative claims must be supported by appropriate statistical testing.
3. **Transparency**: All experimental conditions, including failures and negative results, must be documented.
4. **Consistency**: All experiments must follow the same evaluation protocols and metrics.

## 1. Mandatory Requirements

### 1.1 Multi-Seed Experimental Runs

**Requirement**: Every key model configuration must be trained and evaluated with a minimum of **three (3) different random seeds**.

**Implementation**:
- Seeds must be predefined and documented (standard seeds: 42, 84, 126)
- Each seed affects:
  - Model weight initialization
  - Data shuffling order
  - Dropout and other stochastic operations
  - Sampling in RL trajectories

**Exceptions**:
- Preliminary experiments for hyperparameter search may use single seeds
- Debugging runs are exempt from multi-seed requirements

### 1.2 Aggregated Result Reporting

**Format**: All metrics must be reported as `mean ± standard deviation`

**Example**:
```
Accuracy: 84.3 ± 0.5
F1-Score: 0.821 ± 0.012
Loss: 0.342 ± 0.008
```

**Requirements**:
- Minimum 3 runs for standard deviation calculation
- Round to appropriate significant figures (typically 1-2 decimal places)
- Report both mean and std for all primary metrics

### 1.3 Statistical Significance Testing

**When Required**:
- Comparing hero model vs. baseline
- Claiming improvement over prior work
- Ablation studies showing component importance

**Methods**:
- **Paired t-test**: For normally distributed metrics with matched samples
- **Wilcoxon signed-rank test**: For non-parametric comparisons
- **Bootstrap test**: For complex metrics or small sample sizes
- **Bonferroni correction**: When multiple comparisons are made

**Significance Levels**:
- p < 0.05: Statistically significant (*)
- p < 0.01: Highly significant (**)
- p < 0.001: Very highly significant (***)

## 2. Experimental Configuration Management

### 2.1 Configuration Versioning

All experimental configurations must be:
- Version controlled in `configs/experiments/`
- Tagged with unique experiment IDs
- Linked to WandB artifacts

### 2.2 Environment Documentation

Each experiment must document:
- Python version and virtual environment
- CUDA/cuDNN versions
- All package versions (via `requirements.txt` snapshot)
- Hardware specifications (GPU model, RAM, etc.)

### 2.3 Data Versioning

- Training data must be versioned and checksummed
- Evaluation datasets must remain frozen throughout all experiments
- Any data preprocessing must be deterministic

## 3. Experiment Tracking

### 3.1 WandB Integration

**Required Logging**:
- All hyperparameters
- Training curves (loss, metrics) at regular intervals
- Validation metrics after each epoch
- Final test metrics
- System metrics (GPU utilization, memory usage)
- Random seeds used
- Git commit hash

### 3.2 Artifact Management

**Required Artifacts**:
- Model checkpoints (best and final)
- Configuration files used
- Training logs
- Evaluation outputs
- Generated trajectories (for RL models)

## 4. Evaluation Protocol

### 4.1 Standard Benchmarks

**Core Evaluation Suite**:
- MM-Vet
- MMMU
- ViRL39K
- Custom Capabilities Benchmark (Phase 3, Round 1, Task 2)

**Metrics to Report**:
- Task Success Rate
- Tool Usage Accuracy
- Reasoning Coherence Score
- Efficiency Metrics (steps to solution, inference time)

### 4.2 Evaluation Conditions

- Models must be evaluated in the same inference mode (temperature, sampling)
- Batch size must be consistent across comparisons
- No cherry-picking of results - report all configured benchmarks

### 4.3 Error Analysis

For each experiment, maintain:
- Confusion matrices where applicable
- Error case analysis (categorized by error type)
- Qualitative examples of successes and failures

## 5. Ablation Study Requirements

### 5.1 Component Ablations

When claiming a component's importance:
- Train with and without the component
- Keep all other factors identical
- Run multi-seed experiments for both conditions
- Report statistical significance of the difference

### 5.2 Hyperparameter Sensitivity

- Document the hyperparameter search space
- Report sensitivity analysis for critical hyperparameters
- Justify the final chosen values

## 6. Human Evaluation Protocol

### 6.1 Inter-Annotator Agreement

- Minimum 3 annotators per sample
- Report Fleiss' Kappa or similar agreement metric
- Agreement threshold: κ > 0.6 for substantial agreement

### 6.2 Blind Evaluation

- Annotators must not know which model generated which output
- Randomize presentation order
- Include attention checks

## 7. Compute Resource Reporting

### 7.1 Training Resources

Report for each experiment:
- Total GPU hours
- Peak memory usage
- Number of GPUs used
- Wall-clock training time

### 7.2 Inference Resources

Report:
- Inference latency (ms per sample)
- Throughput (samples per second)
- Memory footprint
- Model size (parameters and disk space)

## 8. Negative Results Policy

### 8.1 Documentation

- Failed experiments must be documented in `experiments/negative_results/`
- Include hypothesis, method, results, and lessons learned

### 8.2 Reporting

- Do not hide negative results
- Discuss why certain approaches failed
- Use negative results to motivate design decisions

## 9. Reproducibility Checklist

Before declaring an experiment complete:

- [ ] Ran with at least 3 different seeds
- [ ] Calculated mean ± std for all metrics
- [ ] Performed statistical significance tests where applicable
- [ ] Logged all runs to WandB with proper tags
- [ ] Saved all configurations and artifacts
- [ ] Documented any deviations from protocol
- [ ] Created reproducibility script
- [ ] Verified results can be reproduced on different machine

## 10. Compliance and Exceptions

### 10.1 Mandatory Compliance

These protocols are mandatory for:
- Results reported in publications
- Claims of improvement over baselines
- Final model comparisons

### 10.2 Exception Process

Exceptions require:
- Written justification in experiment log
- Approval from project lead
- Clear notation in any reported results

## 11. Standard Operating Procedures

### 11.1 Starting a New Experiment

1. Create experiment configuration in `configs/experiments/`
2. Register experiment in `experiments/registry.json`
3. Initialize WandB run with proper project and tags
4. Run multi-seed training script
5. Analyze results with standard analysis script
6. Document findings in experiment log

### 11.2 Comparing Models

1. Ensure both models trained with same data version
2. Run evaluation on same benchmarks
3. Use same inference parameters
4. Apply statistical tests
5. Create comparison table with significance markers

### 11.3 Reporting Results

1. Use standard LaTeX table format
2. Include mean ± std for all metrics
3. Mark statistical significance
4. Provide experiment IDs for traceability
5. Link to WandB runs in footnotes

## Appendix A: Statistical Testing Code Examples

```python
from scipy import stats
import numpy as np

def paired_t_test(model_a_results, model_b_results, alpha=0.05):
    """
    Perform paired t-test between two models' results.
    
    Args:
        model_a_results: List of metrics from model A (one per seed)
        model_b_results: List of metrics from model B (one per seed)
        alpha: Significance level
    
    Returns:
        t_statistic, p_value, is_significant
    """
    t_stat, p_value = stats.ttest_rel(model_a_results, model_b_results)
    return t_stat, p_value, p_value < alpha

def bootstrap_test(model_a_results, model_b_results, n_bootstrap=10000):
    """
    Perform bootstrap significance test.
    """
    differences = np.array(model_a_results) - np.array(model_b_results)
    mean_diff = np.mean(differences)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(differences, size=len(differences), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    p_value = np.sum(np.array(bootstrap_means) <= 0) / n_bootstrap
    return mean_diff, p_value
```

## Appendix B: Experiment Registry Schema

```json
{
  "experiment_id": "exp_20240115_001",
  "name": "Baseline vs Pixelis-Online Comparison",
  "date": "2024-01-15",
  "seeds": [42, 84, 126],
  "models": ["pixel-reasoner-baseline", "pixelis-online"],
  "wandb_project": "pixelis-experiments",
  "wandb_runs": ["run_id_1", "run_id_2", "run_id_3"],
  "status": "completed",
  "results_file": "results/exp_20240115_001.json"
}
```

## Version History

- v1.0 (2024-01-15): Initial protocol establishment
- v1.1 (TBD): Updates based on experimental learnings

---

**Protocol Owner**: Pixelis Project Team  
**Last Updated**: 2024-01-15  
**Review Schedule**: Quarterly or as needed