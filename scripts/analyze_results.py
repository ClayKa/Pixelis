#!/usr/bin/env python3
"""
Multi-seed experiment analysis script for Pixelis
Handles aggregated result reporting and statistical significance testing
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Try to import wandb for fetching results
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Will only analyze local results.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class ExperimentResult:
    """Container for experiment results from a single seed"""
    experiment_id: str
    seed: int
    metrics: Dict[str, float]
    wandb_run_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    training_time: Optional[float] = None
    peak_memory_gb: Optional[float] = None


@dataclass 
class AggregatedResult:
    """Container for aggregated results across seeds"""
    experiment_id: str
    experiment_name: str
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    metrics_raw: Dict[str, List[float]]
    num_seeds: int
    seeds: List[int]
    
    def format_metric(self, metric_name: str, precision: int = 3) -> str:
        """Format a metric as mean ± std"""
        if metric_name not in self.metrics_mean:
            return "N/A"
        
        mean = self.metrics_mean[metric_name]
        std = self.metrics_std[metric_name]
        
        if precision == 0:
            return f"{mean:.0f} ± {std:.0f}"
        else:
            format_str = f"{{:.{precision}f}} ± {{:.{precision}f}}"
            return format_str.format(mean, std)


class ResultAnalyzer:
    """Main class for analyzing multi-seed experimental results"""
    
    def __init__(self, output_dir: str = "outputs/experiments"):
        self.output_dir = Path(output_dir)
        self.registry_file = Path("experiments/registry.json")
        self.results_cache: Dict[str, AggregatedResult] = {}
        
    def load_registry(self) -> List[Dict]:
        """Load experiment registry"""
        if not self.registry_file.exists():
            logger.warning(f"Registry file not found: {self.registry_file}")
            return []
        
        with open(self.registry_file, 'r') as f:
            return json.load(f)
    
    def find_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Find experiment in registry by ID"""
        registry = self.load_registry()
        for entry in registry:
            if entry["experiment_id"] == experiment_id:
                return entry
        return None
    
    def load_local_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Load results from local output directories"""
        experiment = self.find_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        results = []
        exp_dir = Path(experiment.get("output_dir", f"{self.output_dir}/{experiment_id}"))
        
        for seed in experiment["seeds"]:
            seed_dir = exp_dir / f"seed_{seed}"
            if not seed_dir.exists():
                logger.warning(f"Seed directory not found: {seed_dir}")
                continue
            
            # Try to load metrics from various sources
            metrics = {}
            
            # Check for evaluation results file
            eval_file = seed_dir / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    metrics.update(eval_data.get("metrics", {}))
            
            # Check for final metrics file
            metrics_file = seed_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics.update(json.load(f))
            
            # Parse training log for metrics
            log_file = seed_dir / "training.log"
            if log_file.exists():
                parsed_metrics = self._parse_log_metrics(log_file)
                metrics.update(parsed_metrics)
            
            if metrics:
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    seed=seed,
                    metrics=metrics
                )
                results.append(result)
            else:
                logger.warning(f"No metrics found for seed {seed}")
        
        return results
    
    def _parse_log_metrics(self, log_file: Path) -> Dict[str, float]:
        """Parse metrics from training log file"""
        metrics = {}
        
        # Common patterns to look for in logs
        patterns = {
            r"final_accuracy[:\s]+([0-9.]+)": "accuracy",
            r"final_f1[:\s]+([0-9.]+)": "f1_score",
            r"final_loss[:\s]+([0-9.]+)": "loss",
            r"success_rate[:\s]+([0-9.]+)": "success_rate",
            r"reward[:\s]+([0-9.-]+)": "reward",
        }
        
        try:
            import re
            with open(log_file, 'r') as f:
                content = f.read()
                for pattern, metric_name in patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Take the last occurrence
                        metrics[metric_name] = float(matches[-1])
        except Exception as e:
            logger.warning(f"Error parsing log file {log_file}: {e}")
        
        return metrics
    
    def load_wandb_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Load results from WandB"""
        if not WANDB_AVAILABLE:
            logger.warning("WandB not available, skipping WandB results")
            return []
        
        experiment = self.find_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        results = []
        api = wandb.Api()
        
        for run_info in experiment.get("wandb_runs", []):
            try:
                run = api.run(f"{experiment['wandb_project']}/{run_info['run_id']}")
                
                # Get final metrics
                metrics = {}
                for key, value in run.summary.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
                
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    seed=run_info["seed"],
                    metrics=metrics,
                    wandb_run_id=run_info["run_id"],
                    training_time=run.summary.get("_runtime"),
                    peak_memory_gb=run.summary.get("peak_memory_gb")
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error loading WandB run {run_info['run_id']}: {e}")
        
        return results
    
    def aggregate_results(self, results: List[ExperimentResult]) -> AggregatedResult:
        """Aggregate results across seeds"""
        if not results:
            raise ValueError("No results to aggregate")
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Aggregate each metric
        metrics_raw = {}
        for metric in all_metrics:
            values = []
            for result in results:
                if metric in result.metrics:
                    values.append(result.metrics[metric])
            
            if values:
                metrics_raw[metric] = values
        
        # Calculate statistics
        metrics_mean = {}
        metrics_std = {}
        
        for metric, values in metrics_raw.items():
            if len(values) >= 2:
                metrics_mean[metric] = np.mean(values)
                metrics_std[metric] = np.std(values, ddof=1)  # Sample std
            elif len(values) == 1:
                metrics_mean[metric] = values[0]
                metrics_std[metric] = 0.0
        
        return AggregatedResult(
            experiment_id=results[0].experiment_id,
            experiment_name=results[0].experiment_id.rsplit('_', 1)[0],
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            metrics_raw=metrics_raw,
            num_seeds=len(results),
            seeds=[r.seed for r in results]
        )
    
    def perform_statistical_test(
        self,
        results_a: AggregatedResult,
        results_b: AggregatedResult,
        metric: str,
        test_type: str = "paired_t"
    ) -> Tuple[float, float, bool]:
        """
        Perform statistical significance test between two experiments
        
        Args:
            results_a: Results from experiment A
            results_b: Results from experiment B
            metric: Metric to compare
            test_type: Type of test ('paired_t', 'wilcoxon', 'bootstrap')
        
        Returns:
            test_statistic, p_value, is_significant
        """
        if metric not in results_a.metrics_raw or metric not in results_b.metrics_raw:
            raise ValueError(f"Metric {metric} not found in both experiments")
        
        values_a = results_a.metrics_raw[metric]
        values_b = results_b.metrics_raw[metric]
        
        if len(values_a) != len(values_b):
            logger.warning(f"Different number of seeds: {len(values_a)} vs {len(values_b)}")
            # Use unpaired test
            test_type = "unpaired_t"
        
        alpha = 0.05
        
        if test_type == "paired_t":
            statistic, p_value = stats.ttest_rel(values_a, values_b)
        elif test_type == "unpaired_t":
            statistic, p_value = stats.ttest_ind(values_a, values_b)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(values_a, values_b)
        elif test_type == "bootstrap":
            statistic, p_value = self._bootstrap_test(values_a, values_b)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        is_significant = p_value < alpha
        
        return statistic, p_value, is_significant
    
    def _bootstrap_test(
        self,
        values_a: List[float],
        values_b: List[float],
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Perform bootstrap significance test"""
        values_a = np.array(values_a)
        values_b = np.array(values_b)
        
        # If different lengths, pad with mean
        if len(values_a) != len(values_b):
            max_len = max(len(values_a), len(values_b))
            if len(values_a) < max_len:
                values_a = np.pad(values_a, (0, max_len - len(values_a)), 
                                 mode='constant', constant_values=np.mean(values_a))
            if len(values_b) < max_len:
                values_b = np.pad(values_b, (0, max_len - len(values_b)),
                                 mode='constant', constant_values=np.mean(values_b))
        
        differences = values_a - values_b
        mean_diff = np.mean(differences)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(differences, size=len(differences), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Two-tailed p-value
        if mean_diff > 0:
            p_value = 2 * np.sum(bootstrap_means <= 0) / n_bootstrap
        else:
            p_value = 2 * np.sum(bootstrap_means >= 0) / n_bootstrap
        
        return mean_diff, p_value
    
    def generate_comparison_table(
        self,
        experiments: List[str],
        metrics: List[str],
        output_format: str = "latex",
        baseline_idx: int = 0
    ) -> str:
        """
        Generate comparison table for multiple experiments
        
        Args:
            experiments: List of experiment IDs
            metrics: List of metrics to include
            output_format: 'latex', 'markdown', or 'html'
            baseline_idx: Index of baseline experiment for significance testing
        
        Returns:
            Formatted table string
        """
        # Load and aggregate results for each experiment
        all_results = []
        for exp_id in experiments:
            results = self.load_local_results(exp_id)
            if not results:
                results = self.load_wandb_results(exp_id)
            
            if results:
                aggregated = self.aggregate_results(results)
                all_results.append(aggregated)
            else:
                logger.warning(f"No results found for {exp_id}")
        
        if not all_results:
            raise ValueError("No results to compare")
        
        # Build table data
        headers = ["Model"] + metrics
        rows = []
        
        baseline_result = all_results[baseline_idx] if baseline_idx < len(all_results) else None
        
        for i, result in enumerate(all_results):
            row = [result.experiment_name]
            
            for metric in metrics:
                if metric in result.metrics_mean:
                    # Format as mean ± std
                    formatted = result.format_metric(metric)
                    
                    # Add significance markers if not baseline
                    if baseline_result and i != baseline_idx and metric in baseline_result.metrics_raw:
                        try:
                            _, p_value, _ = self.perform_statistical_test(
                                result, baseline_result, metric
                            )
                            
                            if p_value < 0.001:
                                formatted += "***"
                            elif p_value < 0.01:
                                formatted += "**"
                            elif p_value < 0.05:
                                formatted += "*"
                        except Exception as e:
                            logger.warning(f"Statistical test failed: {e}")
                    
                    row.append(formatted)
                else:
                    row.append("N/A")
            
            rows.append(row)
        
        # Format table
        if output_format == "latex":
            table = self._format_latex_table(headers, rows)
        elif output_format == "markdown":
            table = tabulate(rows, headers=headers, tablefmt="github")
        elif output_format == "html":
            table = tabulate(rows, headers=headers, tablefmt="html")
        else:
            table = tabulate(rows, headers=headers, tablefmt="grid")
        
        return table
    
    def _format_latex_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format table as LaTeX"""
        lines = []
        
        # Begin table
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Multi-seed Experimental Results}")
        
        # Column specification
        col_spec = "l" + "r" * (len(headers) - 1)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # Headers
        lines.append(" & ".join(headers) + " \\\\")
        lines.append("\\midrule")
        
        # Data rows
        for row in rows:
            # Escape underscores
            escaped_row = [cell.replace("_", "\\_") for cell in row]
            lines.append(" & ".join(escaped_row) + " \\\\")
        
        # End table
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\label{tab:results}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def plot_metric_comparison(
        self,
        experiments: List[str],
        metric: str,
        save_path: Optional[str] = None
    ):
        """Create bar plot comparing a metric across experiments"""
        # Load results
        all_results = []
        for exp_id in experiments:
            results = self.load_local_results(exp_id)
            if not results:
                results = self.load_wandb_results(exp_id)
            
            if results:
                aggregated = self.aggregate_results(results)
                all_results.append(aggregated)
        
        if not all_results:
            logger.warning("No results to plot")
            return
        
        # Prepare data
        names = [r.experiment_name for r in all_results]
        means = []
        stds = []
        
        for result in all_results:
            if metric in result.metrics_mean:
                means.append(result.metrics_mean[metric])
                stds.append(result.metrics_std[metric])
            else:
                means.append(0)
                stds.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(names))
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(
        self,
        experiment_id: str,
        output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive report for an experiment"""
        # Load results
        results = self.load_local_results(experiment_id)
        if not results:
            results = self.load_wandb_results(experiment_id)
        
        if not results:
            raise ValueError(f"No results found for {experiment_id}")
        
        aggregated = self.aggregate_results(results)
        
        # Build report
        lines = []
        lines.append("=" * 60)
        lines.append(f"Experiment Report: {aggregated.experiment_name}")
        lines.append("=" * 60)
        lines.append("")
        
        lines.append(f"Experiment ID: {aggregated.experiment_id}")
        lines.append(f"Number of Seeds: {aggregated.num_seeds}")
        lines.append(f"Seeds: {aggregated.seeds}")
        lines.append("")
        
        lines.append("Metrics (mean ± std):")
        lines.append("-" * 40)
        
        for metric in sorted(aggregated.metrics_mean.keys()):
            formatted = aggregated.format_metric(metric)
            lines.append(f"  {metric}: {formatted}")
        
        lines.append("")
        lines.append("Raw Values by Seed:")
        lines.append("-" * 40)
        
        for metric in sorted(aggregated.metrics_raw.keys()):
            values = aggregated.metrics_raw[metric]
            values_str = ", ".join([f"{v:.4f}" for v in values])
            lines.append(f"  {metric}: [{values_str}]")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze multi-seed experimental results"
    )
    
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Experiment ID to analyze"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="List of experiment IDs to compare"
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "f1_score", "loss"],
        help="Metrics to include in comparison"
    )
    
    parser.add_argument(
        "--output_format",
        choices=["latex", "markdown", "html", "text"],
        default="markdown",
        help="Output format for tables"
    )
    
    parser.add_argument(
        "--plot_metric",
        type=str,
        help="Metric to plot in comparison"
    )
    
    parser.add_argument(
        "--save_plot",
        type=str,
        help="Path to save plot"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed report"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for results"
    )
    
    parser.add_argument(
        "--test_type",
        choices=["paired_t", "wilcoxon", "bootstrap"],
        default="paired_t",
        help="Statistical test type"
    )
    
    parser.add_argument(
        "--baseline_idx",
        type=int,
        default=0,
        help="Index of baseline experiment for significance testing"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultAnalyzer()
    
    # Handle different modes
    if args.compare:
        # Comparison mode
        logger.info(f"Comparing experiments: {args.compare}")
        
        table = analyzer.generate_comparison_table(
            args.compare,
            args.metrics,
            args.output_format,
            args.baseline_idx
        )
        
        print("\n" + table + "\n")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(table)
            logger.info(f"Table saved to {args.output_file}")
        
        # Plot if requested
        if args.plot_metric:
            analyzer.plot_metric_comparison(
                args.compare,
                args.plot_metric,
                args.save_plot
            )
    
    elif args.experiment_id:
        # Single experiment analysis
        logger.info(f"Analyzing experiment: {args.experiment_id}")
        
        if args.report:
            report = analyzer.generate_report(
                args.experiment_id,
                args.output_file
            )
            print("\n" + report + "\n")
        else:
            # Load and display results
            results = analyzer.load_local_results(args.experiment_id)
            if not results:
                results = analyzer.load_wandb_results(args.experiment_id)
            
            if results:
                aggregated = analyzer.aggregate_results(results)
                
                print(f"\nResults for {aggregated.experiment_name}:")
                print(f"Seeds: {aggregated.seeds}")
                print(f"Metrics (mean ± std):")
                
                for metric in sorted(aggregated.metrics_mean.keys()):
                    formatted = aggregated.format_metric(metric)
                    print(f"  {metric}: {formatted}")
            else:
                logger.error("No results found")
    
    else:
        # List all experiments
        registry = analyzer.load_registry()
        
        if registry:
            print("\nAvailable experiments:")
            for entry in registry:
                status = entry.get("status", "unknown")
                print(f"  - {entry['experiment_id']} ({status})")
        else:
            print("No experiments found")


if __name__ == "__main__":
    main()