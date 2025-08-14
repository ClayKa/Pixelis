#!/usr/bin/env python3
"""
Test Robustness to Noisy Data for Pixelis models.
This script evaluates how the model handles corrupted, noisy, or adversarial inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
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
import cv2
from PIL import Image
import torchvision.transforms as transforms
from scipy import ndimage
import random

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.models.peft_model import PEFTModel
from core.utils.logging_utils import setup_logging


@dataclass
class NoiseTestResult:
    """Results from testing with noisy data."""
    noise_type: str
    noise_level: float
    baseline_accuracy: float
    noisy_accuracy: float
    accuracy_drop: float
    confidence_scores: List[float]
    update_trigger_rate: float
    rejected_samples: int
    total_samples: int
    false_positive_rate: float  # Model wrongly accepts noisy data
    false_negative_rate: float  # Model wrongly rejects clean data
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RobustnessMetrics:
    """Overall robustness metrics."""
    noise_resilience_score: float  # 0-1, higher is better
    confidence_calibration_error: float  # Expected calibration error
    rejection_precision: float  # Precision of noise detection
    rejection_recall: float  # Recall of noise detection
    average_confidence_drop: float
    robustness_per_noise_type: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NoiseGenerator:
    """Generate various types of noise for robustness testing."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def add_gaussian_noise(self, image: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Gaussian noise to image."""
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        return torch.clamp(noisy_image, -3, 3)  # Clamp to reasonable range for normalized images
    
    def add_salt_pepper_noise(self, image: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add salt and pepper noise."""
        mask = torch.rand_like(image)
        noisy_image = image.clone()
        
        # Salt (white pixels)
        salt_mask = mask < (noise_level / 2)
        noisy_image[salt_mask] = 3.0  # Max value for normalized image
        
        # Pepper (black pixels)
        pepper_mask = (mask >= (noise_level / 2)) & (mask < noise_level)
        noisy_image[pepper_mask] = -3.0  # Min value for normalized image
        
        return noisy_image
    
    def add_motion_blur(self, image: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        """Add motion blur to image."""
        # Convert to numpy for cv2 processing
        image_np = image.cpu().numpy()
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply blur to each channel
        blurred = np.zeros_like(image_np)
        for c in range(image_np.shape[0]):
            blurred[c] = ndimage.convolve(image_np[c], kernel, mode='constant')
        
        return torch.from_numpy(blurred).to(self.device)
    
    def add_occlusion(self, image: torch.Tensor, occlusion_ratio: float) -> torch.Tensor:
        """Add random occlusion patches."""
        noisy_image = image.clone()
        h, w = image.shape[-2:]
        
        # Calculate occlusion size
        occlusion_size = int(min(h, w) * occlusion_ratio)
        
        # Random position for occlusion
        x = np.random.randint(0, w - occlusion_size)
        y = np.random.randint(0, h - occlusion_size)
        
        # Apply occlusion (gray patch)
        noisy_image[:, y:y+occlusion_size, x:x+occlusion_size] = 0
        
        return noisy_image
    
    def add_adversarial_perturbation(
        self, 
        image: torch.Tensor, 
        model: nn.Module,
        epsilon: float = 0.1,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Add adversarial perturbation using PGD attack."""
        image = image.clone().detach().requires_grad_(True)
        original_image = image.clone()
        
        # Simple PGD attack
        for _ in range(num_steps):
            # Forward pass
            output = model(image.unsqueeze(0))
            
            # Calculate loss (maximize entropy for confusion)
            if isinstance(output, dict) and 'logits' in output:
                logits = output['logits']
            else:
                logits = output
            
            loss = -F.softmax(logits, dim=-1).max()
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update image
            sign_grad = image.grad.sign()
            image = image + epsilon / num_steps * sign_grad
            
            # Project back to epsilon ball
            image = torch.clamp(image, original_image - epsilon, original_image + epsilon)
            image = torch.clamp(image, -3, 3)  # Ensure valid range
            image = image.detach().requires_grad_(True)
        
        return image.detach()
    
    def corrupt_text(self, text: str, corruption_level: float) -> str:
        """Corrupt text input with typos and character swaps."""
        if not text:
            return text
        
        corrupted = list(text)
        num_corruptions = int(len(text) * corruption_level)
        
        for _ in range(num_corruptions):
            corruption_type = np.random.choice(['swap', 'delete', 'insert', 'replace'])
            pos = np.random.randint(0, len(corrupted))
            
            if corruption_type == 'swap' and pos > 0:
                # Swap with previous character
                corrupted[pos], corrupted[pos-1] = corrupted[pos-1], corrupted[pos]
            elif corruption_type == 'delete' and len(corrupted) > 1:
                # Delete character
                del corrupted[pos]
            elif corruption_type == 'insert':
                # Insert random character
                corrupted.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif corruption_type == 'replace':
                # Replace with random character
                corrupted[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
        
        return ''.join(corrupted)
    
    def add_compression_artifacts(self, image: torch.Tensor, quality: int = 10) -> torch.Tensor:
        """Add JPEG compression artifacts."""
        # Convert to PIL Image
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        # Convert to PIL and compress
        pil_image = Image.fromarray(image_np)
        
        # Save with low quality and reload
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        compressed_tensor = self.transform(compressed_image)
        
        return compressed_tensor.to(self.device)


class RobustnessT

(self, dataloader: DataLoader, use_online: bool, n_samples: int) -> float:
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