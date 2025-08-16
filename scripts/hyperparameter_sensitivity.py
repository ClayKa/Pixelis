#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis for Pixelis models.
This script systematically varies critical hyperparameters and measures their impact on performance.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm
import yaml
import pandas as pd
import argparse
from scipy import interpolate
from scipy.optimize import minimize
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.engine.inference_engine import InferenceEngine
from core.modules.reward_shaping_enhanced import RewardOrchestrator
from core.models.peft_model import PEFTModel
from core.utils.logging_utils import setup_logging


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter experiment."""
    experiment_id: str
    task_weight: float
    curiosity_weight: float
    coherence_weight: float
    confidence_threshold: float
    learning_rate: float
    buffer_capacity: int
    knn_neighbors: int
    update_frequency: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SensitivityResult:
    """Results from a single hyperparameter configuration."""
    config: HyperparameterConfig
    final_accuracy: float
    convergence_speed: float  # Episodes to reach 90% of peak
    stability_score: float  # Inverse of performance variance
    sample_efficiency: float
    update_rejection_rate: float
    mean_confidence: float
    performance_trajectory: List[float]
    training_time: float
    
    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "final_accuracy": self.final_accuracy,
            "convergence_speed": self.convergence_speed,
            "stability_score": self.stability_score,
            "sample_efficiency": self.sample_efficiency,
            "update_rejection_rate": self.update_rejection_rate,
            "mean_confidence": self.mean_confidence,
            "performance_trajectory": self.performance_trajectory,
            "training_time": self.training_time
        }


@dataclass
class SensitivityAnalysis:
    """Complete sensitivity analysis results."""
    parameter_impacts: Dict[str, float]  # Parameter name -> impact score
    optimal_config: HyperparameterConfig
    robust_range: Dict[str, Tuple[float, float]]  # Parameter -> (min, max) robust range
    interaction_effects: Dict[Tuple[str, str], float]  # Parameter pairs -> interaction strength
    pareto_frontier: List[HyperparameterConfig]  # Pareto-optimal configurations
    
    def to_dict(self) -> Dict:
        return {
            "parameter_impacts": self.parameter_impacts,
            "optimal_config": self.optimal_config.to_dict(),
            "robust_range": self.robust_range,
            "interaction_effects": {str(k): v for k, v in self.interaction_effects.items()},
            "pareto_frontier": [c.to_dict() for c in self.pareto_frontier]
        }


class HyperparameterSensitivityAnalyzer:
    """Analyze sensitivity of model performance to hyperparameters."""
    
    def __init__(
        self,
        base_config_path: str,
        device: str = "cuda",
        wandb_project: str = "pixelis-hyperparameter-analysis",
        num_workers: int = 4
    ):
        self.base_config_path = base_config_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        
        # Setup logging
        self.logger = setup_logging("hyperparameter_sensitivity")
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=f"hyperparam_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "base_config": base_config_path,
                "device": str(self.device)
            }
        )
        
        # Load base configuration
        self._load_base_config()
        
        # Define parameter search spaces
        self._define_search_spaces()
        
        # Results storage
        self.results = []
    
    def _load_base_config(self):
        """Load base configuration from file."""
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Extract default values
        self.default_params = HyperparameterConfig(
            experiment_id="default",
            task_weight=self.base_config.get('task_weight', 1.0),
            curiosity_weight=self.base_config.get('curiosity_weight', 0.1),
            coherence_weight=self.base_config.get('coherence_weight', 0.1),
            confidence_threshold=self.base_config.get('confidence_threshold', 0.7),
            learning_rate=self.base_config.get('learning_rate', 1e-4),
            buffer_capacity=self.base_config.get('buffer_capacity', 10000),
            knn_neighbors=self.base_config.get('knn_neighbors', 10),
            update_frequency=self.base_config.get('update_frequency', 100)
        )
    
    def _define_search_spaces(self):
        """Define hyperparameter search spaces."""
        self.search_spaces = {
            'task_weight': [0.5, 0.75, 1.0, 1.25, 1.5],
            'curiosity_weight': [0.01, 0.05, 0.1, 0.2, 0.3],
            'coherence_weight': [0.01, 0.05, 0.1, 0.2, 0.3],
            'confidence_threshold': [0.5, 0.6, 0.7, 0.8, 0.9],
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'buffer_capacity': [1000, 5000, 10000, 20000, 50000],
            'knn_neighbors': [5, 10, 20, 30, 50],
            'update_frequency': [10, 50, 100, 200, 500]
        }
        
        # Define which parameters to focus on (for computational efficiency)
        self.primary_params = ['curiosity_weight', 'coherence_weight', 'confidence_threshold']
        self.secondary_params = ['learning_rate', 'buffer_capacity', 'knn_neighbors']
    
    def generate_experiment_configs(
        self,
        param_names: List[str],
        mode: str = 'grid'
    ) -> List[HyperparameterConfig]:
        """Generate experiment configurations."""
        configs = []
        
        if mode == 'grid':
            # Grid search over specified parameters
            param_values = [self.search_spaces[p] for p in param_names]
            
            for values in itertools.product(*param_values):
                config_dict = self.default_params.to_dict()
                
                for param, value in zip(param_names, values):
                    config_dict[param] = value
                
                config_dict['experiment_id'] = f"grid_{'_'.join([f'{p}={v}' for p, v in zip(param_names, values)])}"
                
                configs.append(HyperparameterConfig(**config_dict))
        
        elif mode == 'random':
            # Random search
            num_samples = 100
            
            for i in range(num_samples):
                config_dict = self.default_params.to_dict()
                
                for param in param_names:
                    config_dict[param] = np.random.choice(self.search_spaces[param])
                
                config_dict['experiment_id'] = f"random_{i:04d}"
                
                configs.append(HyperparameterConfig(**config_dict))
        
        elif mode == 'latin_hypercube':
            # Latin hypercube sampling for better coverage
            from scipy.stats import qmc
            
            num_samples = 50
            sampler = qmc.LatinHypercube(d=len(param_names))
            samples = sampler.random(n=num_samples)
            
            for i, sample in enumerate(samples):
                config_dict = self.default_params.to_dict()
                
                for j, param in enumerate(param_names):
                    # Scale sample to parameter range
                    param_range = self.search_spaces[param]
                    idx = int(sample[j] * len(param_range))
                    idx = min(idx, len(param_range) - 1)
                    config_dict[param] = param_range[idx]
                
                config_dict['experiment_id'] = f"lhs_{i:04d}"
                
                configs.append(HyperparameterConfig(**config_dict))
        
        return configs
    
    def evaluate_configuration(
        self,
        config: HyperparameterConfig,
        num_episodes: int = 100
    ) -> SensitivityResult:
        """Evaluate a single hyperparameter configuration."""
        self.logger.info(f"Evaluating configuration: {config.experiment_id}")
        
        start_time = datetime.now()
        
        # Create temporary config file
        temp_config = self.base_config.copy()
        temp_config.update({
            'task_weight': config.task_weight,
            'curiosity_weight': config.curiosity_weight,
            'coherence_weight': config.coherence_weight,
            'confidence_threshold': config.confidence_threshold,
            'learning_rate': config.learning_rate,
            'buffer_capacity': config.buffer_capacity,
            'knn_neighbors': config.knn_neighbors,
            'update_frequency': config.update_frequency
        })
        
        # Initialize model with configuration
        model = self._initialize_model(temp_config)
        
        # Run evaluation episodes
        performance_trajectory = []
        confidence_scores = []
        update_rejections = 0
        total_updates = 0
        
        for episode in tqdm(range(num_episodes), desc=f"Config {config.experiment_id}"):
            # Create synthetic evaluation task
            task = self._create_evaluation_task()
            
            # Evaluate model
            result = model.infer_and_adapt(
                input_data=task,
                allow_update=True
            )
            
            # Track metrics
            performance = self._evaluate_performance(result, task)
            performance_trajectory.append(performance)
            confidence_scores.append(result.confidence)
            
            if result.confidence < config.confidence_threshold:
                update_rejections += 1
            total_updates += 1
        
        # Calculate metrics
        final_accuracy = np.mean(performance_trajectory[-20:])  # Last 20% of episodes
        
        # Convergence speed
        peak_performance = max(performance_trajectory)
        target_performance = 0.9 * peak_performance
        convergence_speed = num_episodes  # Default to max
        
        for i, perf in enumerate(performance_trajectory):
            if perf >= target_performance:
                convergence_speed = i
                break
        
        # Stability score (inverse of variance)
        stability_score = 1.0 / (np.var(performance_trajectory[-20:]) + 1e-6)
        
        # Sample efficiency
        sample_efficiency = final_accuracy / (convergence_speed + 1)
        
        # Update rejection rate
        update_rejection_rate = update_rejections / total_updates if total_updates > 0 else 0
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return SensitivityResult(
            config=config,
            final_accuracy=final_accuracy,
            convergence_speed=convergence_speed,
            stability_score=stability_score,
            sample_efficiency=sample_efficiency,
            update_rejection_rate=update_rejection_rate,
            mean_confidence=np.mean(confidence_scores),
            performance_trajectory=performance_trajectory,
            training_time=training_time
        )
    
    def _initialize_model(self, config: Dict) -> InferenceEngine:
        """Initialize model with given configuration."""
        return InferenceEngine(
            model_path=config['model']['path'],
            device=self.device,
            buffer_capacity=config['buffer_capacity'],
            confidence_threshold=config['confidence_threshold'],
            learning_rate=config['learning_rate'],
            knn_neighbors=config['knn_neighbors']
        )
    
    def _create_evaluation_task(self) -> Dict:
        """Create synthetic evaluation task."""
        return {
            'image': torch.randn(3, 224, 224, device=self.device),
            'text': f"Task {np.random.randint(1000)}",
            'ground_truth': torch.randint(0, 10, (1,), device=self.device)
        }
    
    def _evaluate_performance(self, result: Any, task: Dict) -> float:
        """Evaluate model performance on task."""
        # Simplified evaluation - customize based on actual task
        if hasattr(result, 'prediction'):
            # Check if prediction matches ground truth
            return float(result.prediction == task['ground_truth'].item())
        return np.random.random()  # Placeholder
    
    def analyze_single_parameter(
        self,
        param_name: str,
        fixed_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze sensitivity to a single parameter."""
        self.logger.info(f"Analyzing sensitivity to {param_name}")
        
        # Fix other parameters to default
        if fixed_params is None:
            fixed_params = self.default_params.to_dict()
        
        # Generate configs varying only this parameter
        configs = []
        for value in self.search_spaces[param_name]:
            config_dict = fixed_params.copy()
            config_dict[param_name] = value
            config_dict['experiment_id'] = f"{param_name}={value}"
            configs.append(HyperparameterConfig(**config_dict))
        
        # Evaluate each configuration
        results = []
        for config in configs:
            result = self.evaluate_configuration(config)
            results.append(result)
            
            # Log to wandb
            wandb.log({
                f"{param_name}/value": getattr(config, param_name),
                f"{param_name}/accuracy": result.final_accuracy,
                f"{param_name}/convergence": result.convergence_speed,
                f"{param_name}/stability": result.stability_score
            })
        
        # Analyze results
        param_values = [getattr(c, param_name) for c in configs]
        accuracies = [r.final_accuracy for r in results]
        
        # Calculate impact score (variance of performance)
        impact_score = np.var(accuracies)
        
        # Find robust range (where performance > 90% of peak)
        peak_accuracy = max(accuracies)
        threshold = 0.9 * peak_accuracy
        robust_indices = [i for i, acc in enumerate(accuracies) if acc >= threshold]
        
        if robust_indices:
            robust_range = (
                param_values[min(robust_indices)],
                param_values[max(robust_indices)]
            )
        else:
            robust_range = (param_values[0], param_values[-1])
        
        return {
            'parameter': param_name,
            'values': param_values,
            'accuracies': accuracies,
            'impact_score': impact_score,
            'robust_range': robust_range,
            'results': results
        }
    
    def analyze_parameter_interactions(
        self,
        param_pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], float]:
        """Analyze interactions between parameter pairs."""
        self.logger.info("Analyzing parameter interactions")
        
        interaction_effects = {}
        
        for param1, param2 in param_pairs:
            self.logger.info(f"Analyzing interaction: {param1} x {param2}")
            
            # Create 2D grid of configurations
            configs = []
            for v1 in self.search_spaces[param1][:3]:  # Use subset for efficiency
                for v2 in self.search_spaces[param2][:3]:
                    config_dict = self.default_params.to_dict()
                    config_dict[param1] = v1
                    config_dict[param2] = v2
                    config_dict['experiment_id'] = f"{param1}={v1}_{param2}={v2}"
                    configs.append(HyperparameterConfig(**config_dict))
            
            # Evaluate configurations
            results = []
            for config in configs:
                result = self.evaluate_configuration(config, num_episodes=50)  # Fewer episodes for speed
                results.append(result)
            
            # Calculate interaction effect using ANOVA-style analysis
            # Reshape results into 2D grid
            n1 = len(self.search_spaces[param1][:3])
            n2 = len(self.search_spaces[param2][:3])
            
            accuracy_grid = np.array([r.final_accuracy for r in results]).reshape(n1, n2)
            
            # Calculate main effects
            main_effect_1 = np.var(np.mean(accuracy_grid, axis=1))
            main_effect_2 = np.var(np.mean(accuracy_grid, axis=0))
            
            # Calculate total variance
            total_variance = np.var(accuracy_grid)
            
            # Interaction effect is variance not explained by main effects
            interaction_effect = max(0, total_variance - main_effect_1 - main_effect_2)
            
            interaction_effects[(param1, param2)] = interaction_effect
            
            # Create interaction plot
            self._plot_interaction(param1, param2, accuracy_grid)
        
        return interaction_effects
    
    def _plot_interaction(self, param1: str, param2: str, accuracy_grid: np.ndarray):
        """Plot parameter interaction heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            accuracy_grid,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=self.search_spaces[param2][:3],
            yticklabels=self.search_spaces[param1][:3],
            ax=ax
        )
        
        ax.set_xlabel(param2)
        ax.set_ylabel(param1)
        ax.set_title(f'Interaction: {param1} x {param2}')
        
        # Save plot
        plot_path = Path(f"results/interaction_{param1}_{param2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({f"interaction_plot_{param1}_{param2}": wandb.Image(str(plot_path))})
    
    def find_optimal_configuration(
        self,
        objective: str = 'accuracy',
        constraints: Optional[Dict] = None
    ) -> HyperparameterConfig:
        """Find optimal hyperparameter configuration."""
        self.logger.info(f"Finding optimal configuration for objective: {objective}")
        
        # Use Bayesian optimization for efficiency
        from scipy.optimize import differential_evolution
        
        # Define objective function
        def objective_function(params):
            config = HyperparameterConfig(
                experiment_id=f"opt_{datetime.now().strftime('%H%M%S')}",
                task_weight=1.0,  # Fixed
                curiosity_weight=params[0],
                coherence_weight=params[1],
                confidence_threshold=params[2],
                learning_rate=params[3],
                buffer_capacity=int(params[4]),
                knn_neighbors=int(params[5]),
                update_frequency=int(params[6])
            )
            
            result = self.evaluate_configuration(config, num_episodes=50)
            
            if objective == 'accuracy':
                return -result.final_accuracy  # Minimize negative accuracy
            elif objective == 'efficiency':
                return -result.sample_efficiency
            elif objective == 'stability':
                return -result.stability_score
            else:
                # Multi-objective: weighted sum
                return -(0.5 * result.final_accuracy + 0.3 * result.sample_efficiency + 0.2 * result.stability_score)
        
        # Define bounds for each parameter
        bounds = [
            (0.01, 0.3),  # curiosity_weight
            (0.01, 0.3),  # coherence_weight
            (0.5, 0.9),   # confidence_threshold
            (1e-5, 1e-3), # learning_rate
            (1000, 50000),  # buffer_capacity
            (5, 50),      # knn_neighbors
            (10, 500)     # update_frequency
        ]
        
        # Apply constraints if specified
        if constraints:
            # Modify bounds based on constraints
            pass
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            workers=self.num_workers,
            disp=True
        )
        
        # Create optimal configuration
        optimal_config = HyperparameterConfig(
            experiment_id="optimal",
            task_weight=1.0,
            curiosity_weight=result.x[0],
            coherence_weight=result.x[1],
            confidence_threshold=result.x[2],
            learning_rate=result.x[3],
            buffer_capacity=int(result.x[4]),
            knn_neighbors=int(result.x[5]),
            update_frequency=int(result.x[6])
        )
        
        self.logger.info(f"Found optimal configuration: {optimal_config}")
        
        return optimal_config
    
    def compute_pareto_frontier(self, results: List[SensitivityResult]) -> List[HyperparameterConfig]:
        """Compute Pareto frontier for multi-objective optimization."""
        self.logger.info("Computing Pareto frontier")
        
        # Extract objectives
        objectives = np.array([
            [r.final_accuracy, r.sample_efficiency, r.stability_score]
            for r in results
        ])
        
        # Find Pareto-optimal points
        pareto_indices = []
        
        for i, obj_i in enumerate(objectives):
            is_dominated = False
            
            for j, obj_j in enumerate(objectives):
                if i != j:
                    # Check if j dominates i (better in all objectives)
                    if all(obj_j >= obj_i) and any(obj_j > obj_i):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        # Extract Pareto-optimal configurations
        pareto_configs = [results[i].config for i in pareto_indices]
        
        self.logger.info(f"Found {len(pareto_configs)} Pareto-optimal configurations")
        
        return pareto_configs
    
    def run_complete_analysis(self) -> SensitivityAnalysis:
        """Run complete hyperparameter sensitivity analysis."""
        self.logger.info("Starting complete hyperparameter sensitivity analysis")
        
        all_results = []
        
        # 1. Single parameter sensitivity
        self.logger.info("=== Analyzing Single Parameter Sensitivity ===")
        parameter_impacts = {}
        robust_ranges = {}
        
        for param in self.primary_params:
            analysis = self.analyze_single_parameter(param)
            parameter_impacts[param] = analysis['impact_score']
            robust_ranges[param] = analysis['robust_range']
            all_results.extend(analysis['results'])
        
        # 2. Parameter interactions
        self.logger.info("=== Analyzing Parameter Interactions ===")
        param_pairs = list(itertools.combinations(self.primary_params, 2))
        interaction_effects = self.analyze_parameter_interactions(param_pairs)
        
        # 3. Find optimal configuration
        self.logger.info("=== Finding Optimal Configuration ===")
        optimal_config = self.find_optimal_configuration(objective='accuracy')
        
        # Evaluate optimal configuration more thoroughly
        optimal_result = self.evaluate_configuration(optimal_config, num_episodes=200)
        all_results.append(optimal_result)
        
        # 4. Compute Pareto frontier
        self.logger.info("=== Computing Pareto Frontier ===")
        pareto_configs = self.compute_pareto_frontier(all_results)
        
        # Create analysis object
        analysis = SensitivityAnalysis(
            parameter_impacts=parameter_impacts,
            optimal_config=optimal_config,
            robust_range=robust_ranges,
            interaction_effects=interaction_effects,
            pareto_frontier=pareto_configs
        )
        
        # Generate report
        self._generate_report(analysis, all_results)
        
        return analysis
    
    def _generate_report(self, analysis: SensitivityAnalysis, results: List[SensitivityResult]):
        """Generate comprehensive analysis report."""
        report_path = Path(f"results/hyperparameter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        report_data = {
            'analysis': analysis.to_dict(),
            'all_results': [r.to_dict() for r in results]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create visualizations
        self._create_summary_plots(analysis, results)
        
        # Print summary
        print("\n" + "="*60)
        print("HYPERPARAMETER SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)
        
        print("\nParameter Impact Scores (higher = more sensitive):")
        for param, impact in sorted(analysis.parameter_impacts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {impact:.4f}")
        
        print("\nRobust Operating Ranges:")
        for param, (min_val, max_val) in analysis.robust_range.items():
            print(f"  {param}: [{min_val:.3f}, {max_val:.3f}]")
        
        print(f"\nOptimal Configuration:")
        for key, value in analysis.optimal_config.to_dict().items():
            if key != 'experiment_id':
                print(f"  {key}: {value}")
        
        print(f"\nNumber of Pareto-optimal configurations: {len(analysis.pareto_frontier)}")
        
        print(f"\nDetailed report saved to: {report_path}")
    
    def _create_summary_plots(self, analysis: SensitivityAnalysis, results: List[SensitivityResult]):
        """Create summary visualizations."""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Parameter impact bar chart
        ax1 = plt.subplot(2, 3, 1)
        params = list(analysis.parameter_impacts.keys())
        impacts = list(analysis.parameter_impacts.values())
        
        ax1.bar(params, impacts, color='steelblue')
        ax1.set_xlabel('Parameter')
        ax1.set_ylabel('Impact Score')
        ax1.set_title('Parameter Sensitivity')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Robust ranges
        ax2 = plt.subplot(2, 3, 2)
        for i, (param, (min_val, max_val)) in enumerate(analysis.robust_range.items()):
            ax2.barh(i, max_val - min_val, left=min_val, height=0.5, 
                    label=param, alpha=0.7)
            ax2.text(min_val, i, f'{min_val:.2f}', va='center')
            ax2.text(max_val, i, f'{max_val:.2f}', va='center', ha='right')
        
        ax2.set_yticks(range(len(analysis.robust_range)))
        ax2.set_yticklabels(list(analysis.robust_range.keys()))
        ax2.set_xlabel('Parameter Value')
        ax2.set_title('Robust Operating Ranges')
        ax2.grid(True, alpha=0.3)
        
        # 3. Interaction effects heatmap
        ax3 = plt.subplot(2, 3, 3)
        if analysis.interaction_effects:
            # Create matrix for heatmap
            params = list(set([p for pair in analysis.interaction_effects.keys() for p in pair]))
            n = len(params)
            interaction_matrix = np.zeros((n, n))
            
            for (p1, p2), effect in analysis.interaction_effects.items():
                i = params.index(p1)
                j = params.index(p2)
                interaction_matrix[i, j] = effect
                interaction_matrix[j, i] = effect
            
            sns.heatmap(interaction_matrix, annot=True, fmt='.4f', 
                       xticklabels=params, yticklabels=params,
                       cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Parameter Interaction Effects')
        
        # 4. Performance trajectory of optimal config
        ax4 = plt.subplot(2, 3, 4)
        optimal_result = next((r for r in results if r.config.experiment_id == "optimal"), None)
        if optimal_result:
            ax4.plot(optimal_result.performance_trajectory, linewidth=2, color='green')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Performance')
            ax4.set_title('Optimal Configuration Performance')
            ax4.grid(True, alpha=0.3)
        
        # 5. Pareto frontier (2D projection)
        ax5 = plt.subplot(2, 3, 5)
        if results:
            accuracies = [r.final_accuracy for r in results]
            efficiencies = [r.sample_efficiency for r in results]
            
            # Mark Pareto-optimal points
            pareto_ids = [c.experiment_id for c in analysis.pareto_frontier]
            pareto_results = [r for r in results if r.config.experiment_id in pareto_ids]
            
            ax5.scatter(accuracies, efficiencies, alpha=0.5, label='All configs')
            
            if pareto_results:
                pareto_acc = [r.final_accuracy for r in pareto_results]
                pareto_eff = [r.sample_efficiency for r in pareto_results]
                ax5.scatter(pareto_acc, pareto_eff, color='red', s=100, 
                          marker='*', label='Pareto optimal')
            
            ax5.set_xlabel('Final Accuracy')
            ax5.set_ylabel('Sample Efficiency')
            ax5.set_title('Pareto Frontier (2D Projection)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 3D Pareto frontier
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        if results:
            accuracies = [r.final_accuracy for r in results]
            efficiencies = [r.sample_efficiency for r in results]
            stabilities = [r.stability_score for r in results]
            
            ax6.scatter(accuracies, efficiencies, stabilities, alpha=0.5)
            
            if pareto_results:
                pareto_acc = [r.final_accuracy for r in pareto_results]
                pareto_eff = [r.sample_efficiency for r in pareto_results]
                pareto_stab = [r.stability_score for r in pareto_results]
                ax6.scatter(pareto_acc, pareto_eff, pareto_stab, 
                          color='red', s=100, marker='*')
            
            ax6.set_xlabel('Accuracy')
            ax6.set_ylabel('Efficiency')
            ax6.set_zlabel('Stability')
            ax6.set_title('3D Pareto Frontier')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(f"results/hyperparameter_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({"hyperparameter_analysis_plots": wandb.Image(str(plot_path))})


def main():
    """Main function to run hyperparameter sensitivity analysis."""
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/pixelis_rft_full.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "interaction", "optimization", "full"],
        default="full",
        help="Analysis mode"
    )
    parser.add_argument(
        "--param",
        type=str,
        help="Parameter to analyze (for single mode)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameter_analysis",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = HyperparameterSensitivityAnalyzer(
        base_config_path=args.config,
        num_workers=args.num_workers
    )
    
    # Run analysis based on mode
    if args.mode == "single":
        if not args.param:
            print("Error: --param required for single mode")
            return
        
        result = analyzer.analyze_single_parameter(args.param)
        print(f"Impact score for {args.param}: {result['impact_score']:.4f}")
        print(f"Robust range: {result['robust_range']}")
    
    elif args.mode == "interaction":
        param_pairs = list(itertools.combinations(analyzer.primary_params, 2))
        effects = analyzer.analyze_parameter_interactions(param_pairs)
        
        print("Interaction effects:")
        for pair, effect in effects.items():
            print(f"  {pair}: {effect:.4f}")
    
    elif args.mode == "optimization":
        optimal = analyzer.find_optimal_configuration()
        print(f"Optimal configuration: {optimal}")
    
    else:  # full
        analysis = analyzer.run_complete_analysis()
        print("\n=== Hyperparameter Sensitivity Analysis Complete ===")
        print(f"Results saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()