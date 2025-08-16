#!/usr/bin/env python3
"""
Pixelis: Comprehensive Wandb Metrics Analysis Script

This script performs in-depth analysis of wandb logs from Pixelis experiments,
generating insights into the learning dynamics, reward signals, and model performance.

Key Features:
- Fetch and analyze metrics from multiple experiment runs
- Generate statistical insights with multi-seed analysis
- Create visualizations that tell the learning story
- Support correlation analysis and phase transition tracking
"""

import os
import typing as t
import numpy as np
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from omegaconf import OmegaConf

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
config = OmegaConf.load(CONFIG_PATH)

# Mapping of experimental configurations
EXPERIMENT_CONFIGS = {
    'sft_only': {'name': 'Supervised Fine-Tuning', 'color': 'blue'},
    'rft_base': {'name': 'Base Reinforcement Fine-Tuning', 'color': 'green'},
    'rft_full': {'name': 'Full Reinforcement Fine-Tuning', 'color': 'red'},
    'online': {'name': 'Online Adaptation Model', 'color': 'purple'}
}

def fetch_wandb_metrics(
    project: str = 'pixelis',
    entity: t.Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch metrics from Wandb for all runs in the Pixelis project.
    
    Args:
        project: Wandb project name
        entity: Optional Wandb entity/team name
    
    Returns:
        Pandas DataFrame with consolidated metrics
    """
    wandb.login()
    api = wandb.Api()
    
    # Fetch runs dynamically based on project configuration
    runs = api.runs(
        path=f"{entity or config.experiment.wandb_entity}/{project}",
        filters={
            "config.experiment.ablation_mode": False
        }
    )
    
    # Collect metrics across all runs
    all_metrics = []
    for run in runs:
        run_metrics = {
            'run_id': run.id,
            'config': run.config,
            'metrics': run.history(keys=[
                'r_coherence', 'r_curiosity', 'r_final', 
                'task_performance', 'kl_divergence',
                'tool_usage_freq'
            ])
        }
        all_metrics.append(run_metrics)
    
    return pd.DataFrame(all_metrics)

def compute_multi_seed_statistics(metrics_df: pd.DataFrame) -> dict:
    """
    Compute multi-seed statistics for key metrics.
    
    Args:
        metrics_df: DataFrame with run metrics
    
    Returns:
        Dictionary of statistical insights
    """
    stats_summary = {}
    metrics_to_analyze = [
        'r_coherence', 'r_curiosity', 'r_final', 
        'task_performance', 'kl_divergence'
    ]
    
    for metric in metrics_to_analyze:
        # Group metrics by configuration
        grouped_metrics = metrics_df.groupby('config.experiment.run_name')[metric]
        
        # Compute statistics
        stats_summary[metric] = {
            'mean': grouped_metrics.mean(),
            'std': grouped_metrics.std(),
            'confidence_interval': grouped_metrics.apply(
                lambda x: stats.t.interval(
                    alpha=0.95, 
                    df=len(x)-1, 
                    loc=np.mean(x), 
                    scale=stats.sem(x)
                )
            )
        }
    
    return stats_summary

def plot_reward_dynamics(metrics_df: pd.DataFrame):
    """
    Generate time-series plots of reward components across different model configurations.
    
    Args:
        metrics_df: DataFrame with run metrics
    """
    plt.figure(figsize=(15, 10))
    
    reward_components = ['r_coherence', 'r_curiosity', 'r_final']
    for i, component in enumerate(reward_components, 1):
        plt.subplot(len(reward_components), 1, i)
        
        for config_name, config_info in EXPERIMENT_CONFIGS.items():
            subset = metrics_df[metrics_df['config.experiment.run_name'] == config_name]
            plt.plot(
                subset['metrics'][component], 
                label=config_info['name'], 
                color=config_info['color']
            )
        
        plt.title(f'{component.replace("_", " ").title()} Dynamics')
        plt.ylabel('Reward Value')
        plt.legend()
    
    plt.xlabel('Training Steps')
    plt.tight_layout()
    plt.savefig('reward_dynamics.png')
    plt.close()

def plot_correlation_heatmap(metrics_df: pd.DataFrame):
    """
    Generate a correlation heatmap of different metrics.
    
    Args:
        metrics_df: DataFrame with run metrics
    """
    # Combine metrics from all runs
    combined_metrics = pd.concat([
        run['metrics'] for run in metrics_df['metrics']
    ])
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = combined_metrics.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        square=True
    )
    plt.title('Metric Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def main():
    """
    Main execution function for metrics analysis.
    """
    # Fetch metrics
    metrics_df = fetch_wandb_metrics(
        project=config.experiment.wandb_project,
        entity=config.experiment.wandb_entity
    )
    
    # Compute multi-seed statistics
    stats_summary = compute_multi_seed_statistics(metrics_df)
    
    # Generate visualizations
    plot_reward_dynamics(metrics_df)
    plot_correlation_heatmap(metrics_df)
    
    # Export statistical summary
    with open('metrics_summary.txt', 'w') as f:
        for metric, stats in stats_summary.items():
            f.write(f"{metric.title()} Statistics:\n")
            f.write(f"Mean: {stats['mean']}\n")
            f.write(f"Standard Deviation: {stats['std']}\n")
            f.write(f"95% Confidence Interval: {stats['confidence_interval']}\n\n")

if __name__ == '__main__':
    main()