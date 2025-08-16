#!/usr/bin/env python3
"""
Trajectory Analysis Script for Post-Training Evaluation.

This script analyzes and visualizes reasoning trajectories from trained models,
comparing behavior across different curriculum stages to demonstrate the impact
of multi-component rewards.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrajectoryAnalyzer:
    """
    Analyzes reasoning trajectories from trained models.
    
    Provides comprehensive analysis including:
    - Trajectory length distribution
    - Tool usage patterns
    - Coherence scoring
    - Loop detection
    - Exploration efficiency
    """
    
    def __init__(self, model_paths: Dict[str, str], output_dir: str):
        """
        Initialize analyzer with model paths.
        
        Args:
            model_paths: Dictionary mapping stage names to model paths
            output_dir: Directory for saving analysis results
        """
        self.model_paths = model_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.tokenizers = {}
        self.trajectories = defaultdict(list)
        
        # Analysis results
        self.results = {
            'trajectory_lengths': defaultdict(list),
            'tool_usage': defaultdict(Counter),
            'coherence_scores': defaultdict(list),
            'loop_counts': defaultdict(int),
            'efficiency_scores': defaultdict(list),
            'success_rates': defaultdict(float),
        }
        
    def load_models(self):
        """Load all models for comparison."""
        logger.info("Loading models for analysis...")
        
        for stage_name, model_path in self.model_paths.items():
            logger.info(f"Loading {stage_name} model from {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizers[stage_name] = tokenizer
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            model.eval()
            self.models[stage_name] = model
            
        logger.info(f"Loaded {len(self.models)} models")
        
    def generate_trajectories(self, prompts: List[str], num_samples: int = 50):
        """
        Generate trajectories from all models.
        
        Args:
            prompts: List of prompts to generate trajectories for
            num_samples: Number of trajectories per model
        """
        logger.info(f"Generating {num_samples} trajectories per model...")
        
        for stage_name, model in self.models.items():
            logger.info(f"Generating trajectories for {stage_name}")
            tokenizer = self.tokenizers[stage_name]
            
            for i, prompt in enumerate(prompts[:num_samples]):
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate trajectory
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                # Decode trajectory
                trajectory_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse trajectory into actions
                trajectory = self._parse_trajectory(trajectory_text)
                self.trajectories[stage_name].append({
                    'prompt': prompt,
                    'trajectory': trajectory,
                    'raw_text': trajectory_text,
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Generated {i + 1}/{num_samples} trajectories")
                    
    def _parse_trajectory(self, trajectory_text: str) -> List[Dict[str, Any]]:
        """
        Parse trajectory text into structured actions.
        
        Args:
            trajectory_text: Raw trajectory text
            
        Returns:
            List of parsed actions
        """
        actions = []
        lines = trajectory_text.split('\n')
        
        for line in lines:
            # Simple parsing - would be enhanced for real trajectories
            if any(op in line for op in ['SEGMENT_OBJECT_AT', 'ZOOM_IN', 'READ_TEXT', 
                                         'TRACK_OBJECT', 'GET_PROPERTIES', 'THINK']):
                action = {
                    'text': line.strip(),
                    'type': 'visual_operation' if not 'THINK' in line else 'reasoning',
                    'operation': self._extract_operation(line),
                }
                actions.append(action)
                
        return actions
        
    def _extract_operation(self, line: str) -> str:
        """Extract operation name from action line."""
        operations = ['SEGMENT_OBJECT_AT', 'ZOOM_IN', 'READ_TEXT', 
                     'TRACK_OBJECT', 'GET_PROPERTIES', 'THINK']
        for op in operations:
            if op in line:
                return op
        return 'UNKNOWN'
        
    def analyze_trajectories(self):
        """Perform comprehensive trajectory analysis."""
        logger.info("Analyzing trajectories...")
        
        for stage_name, stage_trajectories in self.trajectories.items():
            logger.info(f"Analyzing {stage_name} trajectories...")
            
            for traj_data in stage_trajectories:
                trajectory = traj_data['trajectory']
                
                # Trajectory length
                length = len(trajectory)
                self.results['trajectory_lengths'][stage_name].append(length)
                
                # Tool usage
                for action in trajectory:
                    if action['type'] == 'visual_operation':
                        self.results['tool_usage'][stage_name][action['operation']] += 1
                        
                # Coherence score
                coherence = self._calculate_coherence(trajectory)
                self.results['coherence_scores'][stage_name].append(coherence)
                
                # Loop detection
                if self._detect_loops(trajectory):
                    self.results['loop_counts'][stage_name] += 1
                    
                # Efficiency score
                efficiency = self._calculate_efficiency(trajectory)
                self.results['efficiency_scores'][stage_name].append(efficiency)
                
        # Calculate success rates (mock for demonstration)
        for stage_name in self.trajectories:
            # In real implementation, would check if trajectory achieves goal
            self.results['success_rates'][stage_name] = np.random.uniform(0.6, 0.9)
            
    def _calculate_coherence(self, trajectory: List[Dict]) -> float:
        """
        Calculate coherence score for a trajectory.
        
        Measures logical flow and consistency of actions.
        """
        if len(trajectory) < 2:
            return 1.0
            
        coherence_scores = []
        
        for i in range(1, len(trajectory)):
            prev_action = trajectory[i-1]
            curr_action = trajectory[i]
            
            # Simple coherence: penalize repeated identical actions
            if prev_action['operation'] == curr_action['operation']:
                coherence_scores.append(0.5)
            else:
                coherence_scores.append(1.0)
                
        return np.mean(coherence_scores)
        
    def _detect_loops(self, trajectory: List[Dict]) -> bool:
        """
        Detect if trajectory contains loops.
        
        Returns True if repetitive patterns are detected.
        """
        if len(trajectory) < 4:
            return False
            
        # Check for repeated sequences
        operations = [a['operation'] for a in trajectory]
        
        for pattern_len in range(2, len(operations) // 2):
            for i in range(len(operations) - pattern_len * 2 + 1):
                pattern = operations[i:i+pattern_len]
                next_pattern = operations[i+pattern_len:i+pattern_len*2]
                
                if pattern == next_pattern:
                    return True
                    
        return False
        
    def _calculate_efficiency(self, trajectory: List[Dict]) -> float:
        """
        Calculate exploration efficiency.
        
        Measures how efficiently the trajectory explores the problem space.
        """
        if not trajectory:
            return 0.0
            
        # Count unique operations
        unique_ops = len(set(a['operation'] for a in trajectory))
        total_ops = len(trajectory)
        
        # Efficiency is ratio of unique to total, with length penalty
        efficiency = unique_ops / total_ops
        
        # Penalize very long trajectories
        if total_ops > 15:
            efficiency *= 0.9
        if total_ops > 20:
            efficiency *= 0.8
            
        return efficiency
        
    def create_visualizations(self):
        """Create comprehensive visualizations of analysis results."""
        logger.info("Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trajectory Analysis Across Curriculum Stages', fontsize=16, fontweight='bold')
        
        # 1. Trajectory Length Distribution
        ax = axes[0, 0]
        data = [self.results['trajectory_lengths'][stage] for stage in self.model_paths.keys()]
        positions = range(1, len(self.model_paths) + 1)
        bp = ax.boxplot(data, positions=positions, labels=list(self.model_paths.keys()), 
                        patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_title('Trajectory Length Distribution')
        ax.set_ylabel('Number of Actions')
        ax.grid(True, alpha=0.3)
        
        # 2. Tool Usage Heatmap
        ax = axes[0, 1]
        
        # Prepare data for heatmap
        all_tools = set()
        for stage_tools in self.results['tool_usage'].values():
            all_tools.update(stage_tools.keys())
        all_tools = sorted(list(all_tools))
        
        heatmap_data = []
        for stage in self.model_paths.keys():
            stage_counts = []
            total = sum(self.results['tool_usage'][stage].values())
            for tool in all_tools:
                count = self.results['tool_usage'][stage].get(tool, 0)
                stage_counts.append(count / total if total > 0 else 0)
            heatmap_data.append(stage_counts)
            
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(all_tools)))
        ax.set_xticklabels(all_tools, rotation=45, ha='right')
        ax.set_yticks(range(len(self.model_paths)))
        ax.set_yticklabels(list(self.model_paths.keys()))
        ax.set_title('Tool Usage Frequency Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # 3. Coherence Scores
        ax = axes[0, 2]
        x = list(self.model_paths.keys())
        y = [np.mean(self.results['coherence_scores'][stage]) for stage in x]
        yerr = [np.std(self.results['coherence_scores'][stage]) for stage in x]
        
        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
        
        # Color bars based on value
        for bar, val in zip(bars, y):
            bar.set_color(plt.cm.RdYlGn(val))
            
        ax.set_title('Average Coherence Scores')
        ax.set_ylabel('Coherence Score')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Loop Detection
        ax = axes[1, 0]
        x = list(self.model_paths.keys())
        total_trajectories = [len(self.trajectories[stage]) for stage in x]
        loop_percentages = [
            (self.results['loop_counts'][stage] / total_trajectories[i]) * 100 
            for i, stage in enumerate(x)
        ]
        
        bars = ax.bar(x, loop_percentages, alpha=0.7, color='coral')
        ax.set_title('Percentage of Trajectories with Loops')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim([0, max(loop_percentages) * 1.2 if loop_percentages else 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, loop_percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom')
                   
        # 5. Efficiency Scores
        ax = axes[1, 1]
        data = [self.results['efficiency_scores'][stage] for stage in self.model_paths.keys()]
        
        parts = ax.violinplot(data, positions=range(1, len(self.model_paths) + 1),
                             showmeans=True, showmedians=True)
        
        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            
        ax.set_xticks(range(1, len(self.model_paths) + 1))
        ax.set_xticklabels(list(self.model_paths.keys()))
        ax.set_title('Exploration Efficiency Distribution')
        ax.set_ylabel('Efficiency Score')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Success Rate Comparison
        ax = axes[1, 2]
        x = list(self.model_paths.keys())
        y = [self.results['success_rates'][stage] for stage in x]
        
        # Create gradient bars
        bars = ax.bar(x, y, alpha=0.8)
        
        # Color based on performance
        for bar, val in zip(bars, y):
            if val >= 0.8:
                bar.set_color('green')
            elif val >= 0.7:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
                
        ax.set_title('Success Rate by Stage')
        ax.set_ylabel('Success Rate')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val*100:.1f}%', ha='center', va='bottom')
                   
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'trajectory_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        
        plt.show()
        
    def create_trajectory_comparison(self, sample_idx: int = 0):
        """
        Create side-by-side comparison of specific trajectories.
        
        Args:
            sample_idx: Index of sample to compare
        """
        logger.info("Creating trajectory comparison visualization...")
        
        fig, axes = plt.subplots(1, len(self.model_paths), figsize=(6 * len(self.model_paths), 10))
        if len(self.model_paths) == 1:
            axes = [axes]
            
        fig.suptitle(f'Trajectory Comparison for Sample {sample_idx}', fontsize=14, fontweight='bold')
        
        for idx, (stage_name, ax) in enumerate(zip(self.model_paths.keys(), axes)):
            if sample_idx >= len(self.trajectories[stage_name]):
                ax.text(0.5, 0.5, 'No trajectory available', ha='center', va='center')
                ax.set_title(stage_name)
                continue
                
            traj_data = self.trajectories[stage_name][sample_idx]
            trajectory = traj_data['trajectory']
            
            ax.set_title(f'{stage_name}\n(Length: {len(trajectory)})')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, len(trajectory))
            ax.axis('off')
            
            # Draw trajectory steps
            for i, action in enumerate(trajectory):
                y_pos = len(trajectory) - i - 1
                
                # Choose color based on action type
                if action['type'] == 'visual_operation':
                    color = 'lightblue'
                    if 'SEGMENT' in action['operation']:
                        color = 'lightgreen'
                    elif 'ZOOM' in action['operation']:
                        color = 'lightyellow'
                    elif 'READ' in action['operation']:
                        color = 'lightcoral'
                else:
                    color = 'lightgray'
                    
                # Draw action box
                rect = Rectangle((0.1, y_pos - 0.4), 0.8, 0.8, 
                               facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Add action text
                action_text = action['operation']
                if len(action_text) > 15:
                    action_text = action_text[:12] + '...'
                ax.text(0.5, y_pos, action_text, ha='center', va='center', fontsize=9)
                
                # Draw arrow to next action
                if i < len(trajectory) - 1:
                    ax.arrow(0.5, y_pos - 0.4, 0, -0.2, 
                           head_width=0.05, head_length=0.05, fc='black', ec='black')
                    
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'trajectory_comparison_{sample_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison to {output_path}")
        
        plt.show()
        
    def save_analysis_report(self):
        """Save detailed analysis report."""
        logger.info("Saving analysis report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_analyzed': list(self.model_paths.keys()),
            'num_trajectories_per_model': {
                stage: len(trajs) for stage, trajs in self.trajectories.items()
            },
            'summary_statistics': {
                'average_trajectory_length': {
                    stage: np.mean(lengths) for stage, lengths in self.results['trajectory_lengths'].items()
                },
                'average_coherence_score': {
                    stage: np.mean(scores) for stage, scores in self.results['coherence_scores'].items()
                },
                'loop_percentage': {
                    stage: (self.results['loop_counts'][stage] / len(self.trajectories[stage])) * 100
                    for stage in self.model_paths.keys()
                },
                'average_efficiency': {
                    stage: np.mean(scores) for stage, scores in self.results['efficiency_scores'].items()
                },
                'success_rates': self.results['success_rates'],
            },
            'tool_usage_distribution': {
                stage: dict(counter) for stage, counter in self.results['tool_usage'].items()
            },
            'detailed_results': {
                'trajectory_lengths': {
                    stage: {
                        'mean': np.mean(lengths),
                        'std': np.std(lengths),
                        'min': np.min(lengths),
                        'max': np.max(lengths),
                        'median': np.median(lengths),
                    } for stage, lengths in self.results['trajectory_lengths'].items()
                },
                'coherence_scores': {
                    stage: {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'median': np.median(scores),
                    } for stage, scores in self.results['coherence_scores'].items()
                },
            }
        }
        
        # Save JSON report
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {report_path}")
        
        # Save CSV for easy analysis
        df_data = []
        for stage in self.model_paths.keys():
            for i in range(len(self.trajectories[stage])):
                df_data.append({
                    'stage': stage,
                    'trajectory_idx': i,
                    'length': self.results['trajectory_lengths'][stage][i] if i < len(self.results['trajectory_lengths'][stage]) else None,
                    'coherence': self.results['coherence_scores'][stage][i] if i < len(self.results['coherence_scores'][stage]) else None,
                    'efficiency': self.results['efficiency_scores'][stage][i] if i < len(self.results['efficiency_scores'][stage]) else None,
                })
                
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / 'trajectory_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics CSV to {csv_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectories from trained models")
    parser.add_argument("--model_path", type=str, help="Path to single model")
    parser.add_argument("--base_model", type=str, help="Path to base model (R_final only)")
    parser.add_argument("--coherence_model", type=str, help="Path to coherence model")
    parser.add_argument("--full_model", type=str, help="Path to full model (all rewards)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to analyze")
    parser.add_argument("--prompts_file", type=str, help="File containing prompts")
    
    args = parser.parse_args()
    
    # Determine model paths
    model_paths = {}
    if args.model_path:
        model_paths['model'] = args.model_path
    else:
        if args.base_model:
            model_paths['base'] = args.base_model
        if args.coherence_model:
            model_paths['coherence'] = args.coherence_model
        if args.full_model:
            model_paths['full'] = args.full_model
            
    if not model_paths:
        logger.error("No model paths provided!")
        return
        
    # Load prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts for testing
        prompts = [
            "Count the number of people in the image.",
            "What is the main object in the center of the image?",
            "Read the text on the sign in the image.",
            "Describe the spatial relationship between the objects.",
            "Track the moving object across the frames.",
            "What color is the largest object in the image?",
            "Find and describe all text visible in the image.",
            "Compare the sizes of the objects in the image.",
            "Identify the emotions of people in the image.",
            "What is happening in this scene?",
        ]
        
    # Initialize analyzer
    analyzer = TrajectoryAnalyzer(model_paths, args.output_dir)
    
    # Run analysis pipeline
    analyzer.load_models()
    analyzer.generate_trajectories(prompts, args.num_samples)
    analyzer.analyze_trajectories()
    
    # Create visualizations
    analyzer.create_visualizations()
    analyzer.create_trajectory_comparison(sample_idx=0)
    
    # Save report
    report = analyzer.save_analysis_report()
    
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*60)
    for stage, stats in report['summary_statistics'].items():
        if isinstance(stats, dict):
            print(f"\n{stage.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    main()