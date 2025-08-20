#!/usr/bin/env python3
"""
Test Continual Learning and Domain Adaptation for Pixelis models.
This script evaluates the online model's ability to adapt to new domains
and resist catastrophic forgetting.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import argparse

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.data_structures import Experience
from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.models.peft_model import PEFTModelFactory, DynamicLoRAConfig
from core.utils.logging_utils import setup_logging


@dataclass
class DomainTestResult:
    """Results from testing a specific domain."""
    domain_name: str
    initial_accuracy: float
    adaptation_curve: List[float]
    final_accuracy: float
    adaptation_speed: float  # Samples needed to reach 90% of peak performance
    peak_performance: float
    time_to_adapt: float  # Wall-clock time
    confidence_scores: List[float]
    update_rates: List[float]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ForgettingTestResult:
    """Results from catastrophic forgetting test."""
    task_sequence: List[str]
    initial_performances: Dict[str, float]
    final_performances: Dict[str, float]
    forgetting_rates: Dict[str, float]  # (initial - final) / initial
    retention_scores: Dict[str, float]  # final / initial
    performance_timeline: Dict[str, List[float]]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DomainAdaptationTester:
    """Tests the model's ability to adapt to new domains."""
    
    def __init__(
        self,
        model_config: str,
        domains: List[str],
        device: str = "cuda",
        wandb_project: str = "pixelis-continual-learning"
    ):
        self.model_config = model_config
        self.domains = domains
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []
        
        # Setup logging
        self.logger = setup_logging("continual_learning_test")
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=f"continual_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_config": model_config,
                "domains": domains,
                "test_type": "continual_learning"
            }
        )
        
        # Load model and inference engine
        self._load_model()
    
    def _load_model(self):
        """Load the Pixelis-Online model with inference engine."""
        self.logger.info(f"Loading model from {self.model_config}")
        
        # Load configuration
        with open(self.model_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize inference engine (includes experience buffer and voting)
        self.inference_engine = InferenceEngine(
            model_path=config['model']['path'],
            device=self.device,
            buffer_capacity=config.get('buffer_capacity', 10000),
            confidence_threshold=config.get('confidence_threshold', 0.7)
        )
        
        # For comparison, also load static model
        self.static_model = PEFTModelFactory.create_model_with_lora(
            model_name=config['model']['path'],
            lora_config=DynamicLoRAConfig(config['model'].get('lora_config')) if config['model'].get('lora_config') else None,
            device_map=str(self.device)
        )[0]  # Get just the model, not the tokenizer
        self.static_model.eval()
    
    def load_domain_data(self, domain: str) -> DataLoader:
        """Load data for a specific domain."""
        domain_data_path = Path(f"data/domains/{domain}")
        
        # Create synthetic domain-specific data if it doesn't exist
        if not domain_data_path.exists():
            self.logger.info(f"Creating synthetic data for domain: {domain}")
            self._create_synthetic_domain_data(domain, domain_data_path)
        
        # Load the data
        dataset = DomainDataset(domain_data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        return dataloader
    
    def _create_synthetic_domain_data(self, domain: str, path: Path):
        """Create synthetic domain-specific test data."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Domain-specific task templates
        domain_templates = {
            "medical_imaging": {
                "tasks": ["segment tumor", "measure lesion", "track blood flow"],
                "visual_ops": ["SEGMENT_OBJECT_AT", "GET_PROPERTIES", "TRACK_OBJECT"],
                "difficulty": 0.8
            },
            "autonomous_driving": {
                "tasks": ["track pedestrians", "read traffic signs", "segment road"],
                "visual_ops": ["TRACK_OBJECT", "READ_TEXT", "SEGMENT_OBJECT_AT"],
                "difficulty": 0.7
            },
            "document_analysis": {
                "tasks": ["extract table data", "read handwriting", "segment diagrams"],
                "visual_ops": ["READ_TEXT", "SEGMENT_OBJECT_AT", "GET_PROPERTIES"],
                "difficulty": 0.6
            },
            "satellite_imagery": {
                "tasks": ["track deforestation", "segment water bodies", "measure urban growth"],
                "visual_ops": ["TRACK_OBJECT", "SEGMENT_OBJECT_AT", "GET_PROPERTIES"],
                "difficulty": 0.9
            },
            "robotics": {
                "tasks": ["grasp object", "navigate obstacles", "track target"],
                "visual_ops": ["SEGMENT_OBJECT_AT", "GET_PROPERTIES", "TRACK_OBJECT"],
                "difficulty": 0.75
            }
        }
        
        template = domain_templates.get(domain, domain_templates["medical_imaging"])
        
        # Generate 100 test samples for the domain
        samples = []
        for i in range(100):
            task_idx = i % len(template["tasks"])
            sample = {
                "id": f"{domain}_{i:04d}",
                "task": template["tasks"][task_idx],
                "visual_operation": template["visual_ops"][task_idx],
                "difficulty": template["difficulty"] + np.random.normal(0, 0.1),
                "image_path": f"synthetic_{domain}_{i:04d}.jpg",
                "ground_truth": self._generate_ground_truth(template["visual_ops"][task_idx])
            }
            samples.append(sample)
        
        # Save samples
        with open(path / "test_samples.json", 'w') as f:
            json.dump(samples, f, indent=2)
    
    def _generate_ground_truth(self, operation: str) -> Any:
        """Generate ground truth for a visual operation."""
        if operation == "SEGMENT_OBJECT_AT":
            return {"mask": np.random.rand(224, 224).tolist()}
        elif operation == "READ_TEXT":
            return {"text": f"Sample text {np.random.randint(1000, 9999)}"}
        elif operation == "TRACK_OBJECT":
            return {"trajectory": [(x, np.random.randint(0, 224)) for x in range(0, 224, 10)]}
        elif operation == "GET_PROPERTIES":
            return {"properties": {"color": "red", "size": "medium", "shape": "square"}}
        else:
            return {}
    
    def test_adaptation_speed(self, domain: str) -> DomainTestResult:
        """Test how quickly the model adapts to a new domain."""
        self.logger.info(f"Testing adaptation speed for domain: {domain}")
        
        dataloader = self.load_domain_data(domain)
        
        # Track performance over time
        adaptation_curve = []
        confidence_scores = []
        update_rates = []
        
        # Initial performance (no adaptation)
        initial_accuracy = self._evaluate_on_subset(dataloader, use_online=False, n_samples=10)
        
        # Start adaptation
        start_time = time.time()
        samples_processed = 0
        window_size = 10
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Adapting to {domain}")):
            # Process through online system
            result = self.inference_engine.infer_and_adapt(
                input_data=batch,
                allow_update=True
            )
            
            # Track metrics
            confidence_scores.append(result.confidence)
            update_rates.append(1.0 if result.update_triggered else 0.0)
            
            # Evaluate performance every window_size samples
            if (batch_idx + 1) % window_size == 0:
                current_accuracy = self._evaluate_on_subset(
                    dataloader, 
                    use_online=True, 
                    n_samples=window_size
                )
                adaptation_curve.append(current_accuracy)
                
                # Log to wandb
                wandb.log({
                    f"{domain}/accuracy": current_accuracy,
                    f"{domain}/confidence": np.mean(confidence_scores[-window_size:]),
                    f"{domain}/update_rate": np.mean(update_rates[-window_size:]),
                    "samples_processed": samples_processed
                })
            
            samples_processed += 1
        
        # Final evaluation
        final_accuracy = self._evaluate_on_subset(dataloader, use_online=True, n_samples=20)
        adaptation_time = time.time() - start_time
        
        # Calculate adaptation speed (samples to 90% peak)
        peak_performance = max(adaptation_curve) if adaptation_curve else initial_accuracy
        target_performance = initial_accuracy + 0.9 * (peak_performance - initial_accuracy)
        
        adaptation_speed = len(dataloader)  # Default to full dataset
        for i, acc in enumerate(adaptation_curve):
            if acc >= target_performance:
                adaptation_speed = (i + 1) * window_size
                break
        
        result = DomainTestResult(
            domain_name=domain,
            initial_accuracy=initial_accuracy,
            adaptation_curve=adaptation_curve,
            final_accuracy=final_accuracy,
            adaptation_speed=adaptation_speed,
            peak_performance=peak_performance,
            time_to_adapt=adaptation_time,
            confidence_scores=confidence_scores,
            update_rates=update_rates
        )
        
        self.logger.info(f"Domain {domain} - Initial: {initial_accuracy:.3f}, Final: {final_accuracy:.3f}, Speed: {adaptation_speed} samples")
        
        return result
    
    def _evaluate_on_subset(self, dataloader: DataLoader, use_online: bool, n_samples: int) -> float:
        """Evaluate model accuracy on a subset of data."""
        correct = 0
        total = 0
        
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            
            if use_online:
                # Use online model with adaptation
                result = self.inference_engine.infer_and_adapt(
                    input_data=batch,
                    allow_update=False  # Don't update during evaluation
                )
                prediction = result.prediction
            else:
                # Use static model
                with torch.no_grad():
                    prediction = self.static_model(batch['input'])
            
            # Simple accuracy check (customize based on task)
            if self._check_prediction(prediction, batch['ground_truth']):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _check_prediction(self, prediction: Any, ground_truth: Any) -> bool:
        """Check if prediction matches ground truth."""
        # Simplified check - customize based on task type
        if isinstance(ground_truth, dict):
            if 'text' in ground_truth:
                # Text comparison
                return prediction.get('text', '') == ground_truth['text']
            elif 'mask' in ground_truth:
                # Segmentation IoU
                return self._calculate_iou(prediction.get('mask'), ground_truth['mask']) > 0.5
        return False
    
    def _calculate_iou(self, pred_mask: Any, gt_mask: Any) -> float:
        """Calculate Intersection over Union for segmentation masks."""
        if pred_mask is None or gt_mask is None:
            return 0.0
        
        # Convert to numpy arrays if needed
        pred = np.array(pred_mask)
        gt = np.array(gt_mask)
        
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def test_catastrophic_forgetting(self, task_sequence: List[str]) -> ForgettingTestResult:
        """Test resistance to catastrophic forgetting."""
        self.logger.info(f"Testing catastrophic forgetting with sequence: {task_sequence}")
        
        initial_performances = {}
        performance_timeline = defaultdict(list)
        
        # Evaluate initial performance on all tasks
        for task in task_sequence:
            dataloader = self.load_domain_data(task)
            initial_performances[task] = self._evaluate_on_subset(
                dataloader, use_online=True, n_samples=20
            )
            performance_timeline[task].append(initial_performances[task])
            self.logger.info(f"Initial performance on {task}: {initial_performances[task]:.3f}")
        
        # Sequential learning
        for current_task_idx, current_task in enumerate(task_sequence):
            self.logger.info(f"Learning task {current_task_idx + 1}/{len(task_sequence)}: {current_task}")
            
            # Train on current task
            dataloader = self.load_domain_data(current_task)
            for batch in tqdm(dataloader, desc=f"Learning {current_task}"):
                self.inference_engine.infer_and_adapt(
                    input_data=batch,
                    allow_update=True
                )
            
            # Evaluate on all previous tasks
            for task in task_sequence[:current_task_idx + 1]:
                test_dataloader = self.load_domain_data(task)
                performance = self._evaluate_on_subset(
                    test_dataloader, use_online=True, n_samples=20
                )
                performance_timeline[task].append(performance)
                
                wandb.log({
                    f"forgetting/{task}_performance": performance,
                    "current_task": current_task,
                    "task_index": current_task_idx
                })
        
        # Final evaluation on all tasks
        final_performances = {}
        for task in task_sequence:
            dataloader = self.load_domain_data(task)
            final_performances[task] = self._evaluate_on_subset(
                dataloader, use_online=True, n_samples=20
            )
            self.logger.info(f"Final performance on {task}: {final_performances[task]:.3f}")
        
        # Calculate forgetting metrics
        forgetting_rates = {}
        retention_scores = {}
        
        for task in task_sequence:
            initial = initial_performances[task]
            final = final_performances[task]
            
            if initial > 0:
                forgetting_rates[task] = max(0, (initial - final) / initial)
                retention_scores[task] = final / initial
            else:
                forgetting_rates[task] = 0.0
                retention_scores[task] = 1.0
        
        result = ForgettingTestResult(
            task_sequence=task_sequence,
            initial_performances=initial_performances,
            final_performances=final_performances,
            forgetting_rates=forgetting_rates,
            retention_scores=retention_scores,
            performance_timeline=dict(performance_timeline)
        )
        
        # Log summary metrics
        avg_forgetting = np.mean(list(forgetting_rates.values()))
        avg_retention = np.mean(list(retention_scores.values()))
        
        wandb.log({
            "forgetting/average_forgetting_rate": avg_forgetting,
            "forgetting/average_retention_score": avg_retention
        })
        
        self.logger.info(f"Average forgetting rate: {avg_forgetting:.3f}")
        self.logger.info(f"Average retention score: {avg_retention:.3f}")
        
        return result
    
    def compare_with_static_model(self, domain: str) -> Dict[str, Any]:
        """Compare online model adaptation with static model performance."""
        self.logger.info(f"Comparing online vs static model on domain: {domain}")
        
        dataloader = self.load_domain_data(domain)
        
        # Static model performance
        static_performance = []
        for i in range(0, 100, 10):
            acc = self._evaluate_on_subset(dataloader, use_online=False, n_samples=10)
            static_performance.append(acc)
        
        # Online model adaptation
        online_result = self.test_adaptation_speed(domain)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_points = list(range(10, 101, 10))
        ax.plot(x_points, static_performance, 'r--', label='Static Model', linewidth=2)
        ax.plot(x_points[:len(online_result.adaptation_curve)], 
                online_result.adaptation_curve, 'b-', label='Online Model', linewidth=2)
        
        ax.set_xlabel('Samples Processed')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Adaptation Comparison: {domain}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = Path(f"results/adaptation_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({f"adaptation_plot_{domain}": wandb.Image(str(plot_path))})
        
        comparison = {
            "domain": domain,
            "static_final_accuracy": static_performance[-1] if static_performance else 0.0,
            "online_final_accuracy": online_result.final_accuracy,
            "improvement": online_result.final_accuracy - (static_performance[-1] if static_performance else 0.0),
            "adaptation_speed": online_result.adaptation_speed
        }
        
        return comparison
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete continual learning test suite."""
        self.logger.info("Starting full continual learning test suite")
        
        all_results = {
            "adaptation_tests": [],
            "forgetting_test": None,
            "comparisons": []
        }
        
        # Test 1: Domain adaptation for each domain
        for domain in self.domains:
            result = self.test_adaptation_speed(domain)
            all_results["adaptation_tests"].append(result.to_dict())
            
            # Compare with static model
            comparison = self.compare_with_static_model(domain)
            all_results["comparisons"].append(comparison)
        
        # Test 2: Catastrophic forgetting
        forgetting_result = self.test_catastrophic_forgetting(self.domains)
        all_results["forgetting_test"] = forgetting_result.to_dict()
        
        # Generate final report
        self._generate_report(all_results)
        
        return all_results
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report."""
        report_path = Path(f"results/continual_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary visualization
        self._create_summary_plots(results)
        
        # Log final summary to wandb
        summary = {
            "avg_adaptation_speed": np.mean([r["adaptation_speed"] for r in results["adaptation_tests"]]),
            "avg_improvement_over_static": np.mean([c["improvement"] for c in results["comparisons"]]),
            "avg_retention_score": np.mean(list(results["forgetting_test"]["retention_scores"].values()))
        }
        
        wandb.log({"summary": summary})
        
        self.logger.info(f"Report saved to {report_path}")
        self.logger.info(f"Summary: {summary}")
    
    def _create_summary_plots(self, results: Dict[str, Any]):
        """Create summary visualizations."""
        # Plot 1: Adaptation speeds across domains
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Adaptation speed comparison
        domains = [r["domain_name"] for r in results["adaptation_tests"]]
        speeds = [r["adaptation_speed"] for r in results["adaptation_tests"]]
        
        axes[0, 0].bar(domains, speeds, color='skyblue')
        axes[0, 0].set_xlabel('Domain')
        axes[0, 0].set_ylabel('Samples to 90% Peak')
        axes[0, 0].set_title('Adaptation Speed by Domain')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Final accuracy comparison
        static_accs = [c["static_final_accuracy"] for c in results["comparisons"]]
        online_accs = [c["online_final_accuracy"] for c in results["comparisons"]]
        
        x = np.arange(len(domains))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, static_accs, width, label='Static', color='coral')
        axes[0, 1].bar(x + width/2, online_accs, width, label='Online', color='lightgreen')
        axes[0, 1].set_xlabel('Domain')
        axes[0, 1].set_ylabel('Final Accuracy')
        axes[0, 1].set_title('Static vs Online Model Performance')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(domains, rotation=45)
        axes[0, 1].legend()
        
        # Forgetting rates
        if results["forgetting_test"]:
            tasks = results["forgetting_test"]["task_sequence"]
            retention = list(results["forgetting_test"]["retention_scores"].values())
            
            axes[1, 0].bar(tasks, retention, color='purple', alpha=0.7)
            axes[1, 0].set_xlabel('Task')
            axes[1, 0].set_ylabel('Retention Score')
            axes[1, 0].set_title('Task Retention After Sequential Learning')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='90% retention')
            axes[1, 0].legend()
        
        # Performance timeline
        if results["forgetting_test"]:
            timeline = results["forgetting_test"]["performance_timeline"]
            for task, performances in timeline.items():
                axes[1, 1].plot(performances, marker='o', label=task)
            
            axes[1, 1].set_xlabel('Learning Step')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Performance Evolution During Sequential Learning')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(f"results/continual_learning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({"summary_plots": wandb.Image(str(plot_path))})


class DomainDataset(Dataset):
    """Dataset for domain-specific data."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
        # Load samples
        with open(data_path / "test_samples.json", 'r') as f:
            self.samples = json.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create synthetic input (in real scenario, load actual images)
        return {
            "input": {
                "image": torch.randn(3, 224, 224),  # Placeholder image
                "text": sample["task"],
                "operation": sample["visual_operation"]
            },
            "ground_truth": sample["ground_truth"],
            "id": sample["id"]
        }


def main():
    """Main function to run continual learning tests."""
    parser = argparse.ArgumentParser(description="Test continual learning capabilities")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/experiments/pixelis_online.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["medical_imaging", "autonomous_driving", "document_analysis", 
                 "satellite_imagery", "robotics"],
        help="List of domains to test"
    )
    parser.add_argument(
        "--test-type",
        choices=["adaptation", "forgetting", "both"],
        default="both",
        help="Type of test to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/continual_learning",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tester
    tester = DomainAdaptationTester(
        model_config=args.model_config,
        domains=args.domains
    )
    
    # Run tests based on type
    if args.test_type == "adaptation":
        results = []
        for domain in args.domains:
            result = tester.test_adaptation_speed(domain)
            results.append(result.to_dict())
            comparison = tester.compare_with_static_model(domain)
            print(f"Domain {domain}: Improvement = {comparison['improvement']:.3f}")
    
    elif args.test_type == "forgetting":
        result = tester.test_catastrophic_forgetting(args.domains)
        print(f"Average retention score: {np.mean(list(result.retention_scores.values())):.3f}")
    
    else:  # both
        results = tester.run_full_test_suite()
        print("\n=== Continual Learning Test Complete ===")
        print(f"Results saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()