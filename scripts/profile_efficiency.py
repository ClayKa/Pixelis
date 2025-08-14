#!/usr/bin/env python3
"""
Profile Efficiency and Latency for Pixelis models.
This script uses torch.profiler to measure the performance of key components.
"""

import torch
import torch.nn as nn
import torch.profiler
import numpy as np
import json
import time
import psutil
import GPUtil
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import pandas as pd
import argparse
import tracemalloc
from memory_profiler import profile as memory_profile
import cProfile
import pstats
import io

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.modules.dynamics_model import DynamicsModel
from core.modules.reward_shaping_enhanced import RewardOrchestrator
from core.models.peft_model import PEFTModel
from core.utils.logging_utils import setup_logging


@dataclass
class ComponentLatency:
    """Latency metrics for a specific component."""
    component_name: str
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    num_calls: int
    total_time_ms: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    cpu_memory_mb: float
    gpu_memory_mb: float
    peak_cpu_memory_mb: float
    peak_gpu_memory_mb: float
    cpu_memory_delta_mb: float
    gpu_memory_delta_mb: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProfileResults:
    """Complete profiling results."""
    component_latencies: Dict[str, ComponentLatency]
    memory_metrics: MemoryMetrics
    throughput_fps: float
    total_inference_time_ms: float
    overhead_breakdown: Dict[str, float]  # Percentage of time spent in each component
    flops_summary: Dict[str, int]
    trace_file: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "component_latencies": {k: v.to_dict() for k, v in self.component_latencies.items()},
            "memory_metrics": self.memory_metrics.to_dict(),
            "throughput_fps": self.throughput_fps,
            "total_inference_time_ms": self.total_inference_time_ms,
            "overhead_breakdown": self.overhead_breakdown,
            "flops_summary": self.flops_summary,
            "trace_file": self.trace_file
        }


class EfficiencyProfiler:
    """Profile the efficiency and latency of Pixelis models."""
    
    def __init__(
        self,
        model_config: str,
        device: str = "cuda",
        wandb_project: str = "pixelis-efficiency",
        profile_memory: bool = True,
        profile_cuda: bool = True
    ):
        self.model_config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.profile_memory = profile_memory
        self.profile_cuda = profile_cuda and torch.cuda.is_available()
        
        # Setup logging
        self.logger = setup_logging("efficiency_profiler")
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=f"efficiency_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_config": model_config,
                "device": str(self.device),
                "profile_memory": profile_memory,
                "profile_cuda": self.profile_cuda
            }
        )
        
        # Component timing data
        self.component_times = defaultdict(list)
        
        # Load model and components
        self._load_model()
    
    def _load_model(self):
        """Load the Pixelis-Online model and its components."""
        self.logger.info(f"Loading model from {self.model_config}")
        
        # Load configuration
        with open(self.model_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(
            model_path=config['model']['path'],
            device=self.device,
            buffer_capacity=config.get('buffer_capacity', 10000),
            confidence_threshold=config.get('confidence_threshold', 0.7)
        )
        
        # Initialize individual components for targeted profiling
        self.experience_buffer = self.inference_engine.experience_buffer
        self.dynamics_model = DynamicsModel(
            state_dim=config.get('dynamics_state_dim', 768),
            action_dim=config.get('dynamics_action_dim', 256),
            device=self.device
        )
        self.reward_orchestrator = RewardOrchestrator(
            task_weight=config.get('task_weight', 1.0),
            curiosity_weight=config.get('curiosity_weight', 0.1),
            coherence_weight=config.get('coherence_weight', 0.1),
            device=self.device
        )
    
    def create_test_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Create a synthetic test batch for profiling."""
        return {
            "image": torch.randn(batch_size, 3, 224, 224, device=self.device),
            "text": torch.randint(0, 1000, (batch_size, 128), device=self.device),
            "embeddings": torch.randn(batch_size, 768, device=self.device)
        }
    
    def profile_component_latency(
        self,
        component_fn: callable,
        component_name: str,
        inputs: Any,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> ComponentLatency:
        """Profile the latency of a specific component."""
        self.logger.info(f"Profiling {component_name}...")
        
        # Warmup
        for _ in range(num_warmup):
            _ = component_fn(*inputs) if isinstance(inputs, tuple) else component_fn(inputs)
        
        # Synchronize CUDA
        if self.profile_cuda:
            torch.cuda.synchronize()
        
        # Collect timing data
        latencies = []
        
        for _ in tqdm(range(num_iterations), desc=f"Profiling {component_name}"):
            start_time = time.perf_counter()
            
            if self.profile_cuda:
                torch.cuda.synchronize()
            
            _ = component_fn(*inputs) if isinstance(inputs, tuple) else component_fn(inputs)
            
            if self.profile_cuda:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        return ComponentLatency(
            component_name=component_name,
            mean_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p90_latency_ms=np.percentile(latencies, 90),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            std_latency_ms=np.std(latencies),
            num_calls=num_iterations,
            total_time_ms=np.sum(latencies)
        )
    
    def profile_knn_search(self, num_queries: int = 100) -> ComponentLatency:
        """Profile k-NN search in the experience buffer."""
        self.logger.info("Profiling k-NN search...")
        
        # Populate buffer with some experiences
        for _ in range(1000):
            exp = self._create_synthetic_experience()
            self.experience_buffer.add(exp)
        
        # Create query embeddings
        query_embeddings = [torch.randn(768, device=self.device) for _ in range(num_queries)]
        
        def knn_search_fn(embedding):
            return self.experience_buffer.search_index(embedding, k=10)
        
        # Profile k-NN search
        latencies = []
        
        for embedding in tqdm(query_embeddings, desc="Profiling k-NN"):
            start_time = time.perf_counter()
            
            if self.profile_cuda:
                torch.cuda.synchronize()
            
            _ = knn_search_fn(embedding)
            
            if self.profile_cuda:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        latencies = np.array(latencies)
        
        return ComponentLatency(
            component_name="knn_search",
            mean_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p90_latency_ms=np.percentile(latencies, 90),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            std_latency_ms=np.std(latencies),
            num_calls=num_queries,
            total_time_ms=np.sum(latencies)
        )
    
    def profile_curiosity_reward(self, batch_size: int = 1) -> ComponentLatency:
        """Profile curiosity reward calculation."""
        # Create test inputs
        state = torch.randn(batch_size, 768, device=self.device)
        action = torch.randn(batch_size, 256, device=self.device)
        next_state = torch.randn(batch_size, 768, device=self.device)
        
        def curiosity_fn(s, a, ns):
            with torch.no_grad():
                return self.dynamics_model.compute_intrinsic_reward(s, a, ns)
        
        return self.profile_component_latency(
            curiosity_fn,
            "curiosity_reward",
            (state, action, next_state),
            num_iterations=100
        )
    
    def profile_coherence_reward(self, batch_size: int = 1) -> ComponentLatency:
        """Profile coherence reward calculation."""
        # Create test trajectory
        trajectory = [torch.randn(768, device=self.device) for _ in range(10)]
        
        def coherence_fn(traj):
            similarities = []
            for i in range(len(traj) - 1):
                sim = torch.cosine_similarity(traj[i].unsqueeze(0), traj[i+1].unsqueeze(0))
                similarities.append(sim)
            return torch.mean(torch.stack(similarities))
        
        return self.profile_component_latency(
            coherence_fn,
            "coherence_reward",
            trajectory,
            num_iterations=100
        )
    
    def profile_full_inference(self, num_samples: int = 50) -> Tuple[ComponentLatency, Dict[str, float]]:
        """Profile the complete inference and adaptation pipeline."""
        self.logger.info("Profiling full inference pipeline...")
        
        # Create test samples
        test_samples = [self.create_test_batch() for _ in range(num_samples)]
        
        # Track component breakdown
        component_times = defaultdict(list)
        total_times = []
        
        for sample in tqdm(test_samples, desc="Full inference profiling"):
            start_total = time.perf_counter()
            
            # Model inference
            start = time.perf_counter()
            with torch.no_grad():
                model_output = self.inference_engine.model(sample)
            if self.profile_cuda:
                torch.cuda.synchronize()
            component_times["model_inference"].append((time.perf_counter() - start) * 1000)
            
            # k-NN search
            start = time.perf_counter()
            neighbors = self.experience_buffer.search_index(sample["embeddings"], k=10)
            if self.profile_cuda:
                torch.cuda.synchronize()
            component_times["knn_search"].append((time.perf_counter() - start) * 1000)
            
            # Voting
            start = time.perf_counter()
            # Simulate voting (simplified)
            confidence = torch.rand(1).item()
            if self.profile_cuda:
                torch.cuda.synchronize()
            component_times["voting"].append((time.perf_counter() - start) * 1000)
            
            # Reward calculation (if confidence > threshold)
            if confidence > 0.7:
                start = time.perf_counter()
                # Simulate reward calculation
                reward = torch.randn(3, device=self.device)  # task, curiosity, coherence
                if self.profile_cuda:
                    torch.cuda.synchronize()
                component_times["reward_calculation"].append((time.perf_counter() - start) * 1000)
            
            total_times.append((time.perf_counter() - start_total) * 1000)
        
        # Calculate overhead breakdown
        total_mean = np.mean(total_times)
        overhead_breakdown = {}
        
        for component, times in component_times.items():
            mean_time = np.mean(times) if times else 0
            overhead_breakdown[component] = (mean_time / total_mean * 100) if total_mean > 0 else 0
        
        # Create latency object for full pipeline
        total_times = np.array(total_times)
        full_latency = ComponentLatency(
            component_name="full_inference",
            mean_latency_ms=np.mean(total_times),
            p50_latency_ms=np.percentile(total_times, 50),
            p90_latency_ms=np.percentile(total_times, 90),
            p95_latency_ms=np.percentile(total_times, 95),
            p99_latency_ms=np.percentile(total_times, 99),
            min_latency_ms=np.min(total_times),
            max_latency_ms=np.max(total_times),
            std_latency_ms=np.std(total_times),
            num_calls=num_samples,
            total_time_ms=np.sum(total_times)
        )
        
        return full_latency, overhead_breakdown
    
    def profile_memory_usage(self) -> MemoryMetrics:
        """Profile memory usage of the model and components."""
        self.logger.info("Profiling memory usage...")
        
        # CPU memory
        process = psutil.Process()
        cpu_memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory
        gpu_memory_start = 0
        if self.profile_cuda:
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                gpu_memory_start = gpu.memoryUsed
        
        # Track memory during inference
        tracemalloc.start()
        
        # Run some inferences
        peak_cpu = cpu_memory_start
        peak_gpu = gpu_memory_start
        
        for _ in range(10):
            batch = self.create_test_batch(batch_size=4)
            with torch.no_grad():
                _ = self.inference_engine.model(batch)
            
            # Update peak memory
            current_cpu = process.memory_info().rss / 1024 / 1024
            peak_cpu = max(peak_cpu, current_cpu)
            
            if self.profile_cuda and gpu:
                gpu = GPUtil.getGPUs()[0]
                peak_gpu = max(peak_gpu, gpu.memoryUsed)
        
        # Final memory
        cpu_memory_end = process.memory_info().rss / 1024 / 1024
        gpu_memory_end = 0
        if self.profile_cuda:
            gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
            if gpu:
                gpu_memory_end = gpu.memoryUsed
        
        tracemalloc.stop()
        
        return MemoryMetrics(
            cpu_memory_mb=cpu_memory_end,
            gpu_memory_mb=gpu_memory_end,
            peak_cpu_memory_mb=peak_cpu,
            peak_gpu_memory_mb=peak_gpu,
            cpu_memory_delta_mb=cpu_memory_end - cpu_memory_start,
            gpu_memory_delta_mb=gpu_memory_end - gpu_memory_start
        )
    
    def profile_with_torch_profiler(self, num_iterations: int = 20) -> str:
        """Use PyTorch profiler for detailed analysis."""
        self.logger.info("Running PyTorch profiler...")
        
        trace_path = Path(f"results/profiler_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.profile_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True,
            with_flops=True
        ) as prof:
            with torch.profiler.record_function("full_inference"):
                for _ in range(num_iterations):
                    batch = self.create_test_batch()
                    
                    with torch.profiler.record_function("model_forward"):
                        with torch.no_grad():
                            output = self.inference_engine.model(batch)
                    
                    with torch.profiler.record_function("knn_search"):
                        neighbors = self.experience_buffer.search_index(
                            batch["embeddings"], k=10
                        )
                    
                    with torch.profiler.record_function("reward_calculation"):
                        # Simulate reward calculation
                        state = torch.randn(1, 768, device=self.device)
                        action = torch.randn(1, 256, device=self.device)
                        next_state = torch.randn(1, 768, device=self.device)
                        reward = self.dynamics_model.compute_intrinsic_reward(
                            state, action, next_state
                        )
        
        # Export trace
        prof.export_chrome_trace(str(trace_path))
        
        # Print summary
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        
        # Extract FLOPs information
        flops_summary = {}
        for event in prof.key_averages():
            if event.flops > 0:
                flops_summary[event.key] = event.flops
        
        return str(trace_path), flops_summary
    
    def calculate_throughput(self, batch_size: int = 8, num_batches: int = 100) -> float:
        """Calculate model throughput in FPS."""
        self.logger.info(f"Calculating throughput with batch size {batch_size}...")
        
        # Warmup
        for _ in range(10):
            batch = self.create_test_batch(batch_size)
            with torch.no_grad():
                _ = self.inference_engine.model(batch)
        
        if self.profile_cuda:
            torch.cuda.synchronize()
        
        # Measure throughput
        start_time = time.perf_counter()
        
        for _ in tqdm(range(num_batches), desc="Throughput measurement"):
            batch = self.create_test_batch(batch_size)
            with torch.no_grad():
                _ = self.inference_engine.model(batch)
        
        if self.profile_cuda:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_samples = batch_size * num_batches
        
        throughput_fps = total_samples / total_time
        
        self.logger.info(f"Throughput: {throughput_fps:.2f} FPS")
        
        return throughput_fps
    
    def _create_synthetic_experience(self) -> Any:
        """Create a synthetic experience for buffer population."""
        return {
            "state": torch.randn(768, device=self.device),
            "action": torch.randn(256, device=self.device),
            "reward": torch.randn(3, device=self.device),
            "embedding": torch.randn(768, device=self.device),
            "confidence": torch.rand(1).item()
        }
    
    def run_complete_profile(self) -> ProfileResults:
        """Run complete efficiency profiling suite."""
        self.logger.info("Starting complete efficiency profile...")
        
        results = {
            "component_latencies": {}
        }
        
        # 1. Profile individual components
        self.logger.info("=== Profiling Individual Components ===")
        
        # k-NN search
        knn_latency = self.profile_knn_search(num_queries=100)
        results["component_latencies"]["knn_search"] = knn_latency
        
        # Curiosity reward
        curiosity_latency = self.profile_curiosity_reward()
        results["component_latencies"]["curiosity_reward"] = curiosity_latency
        
        # Coherence reward
        coherence_latency = self.profile_coherence_reward()
        results["component_latencies"]["coherence_reward"] = coherence_latency
        
        # 2. Profile full inference pipeline
        self.logger.info("=== Profiling Full Inference Pipeline ===")
        full_latency, overhead_breakdown = self.profile_full_inference(num_samples=50)
        results["component_latencies"]["full_inference"] = full_latency
        results["overhead_breakdown"] = overhead_breakdown
        
        # 3. Memory profiling
        self.logger.info("=== Profiling Memory Usage ===")
        memory_metrics = self.profile_memory_usage()
        results["memory_metrics"] = memory_metrics
        
        # 4. Throughput measurement
        self.logger.info("=== Measuring Throughput ===")
        throughput = self.calculate_throughput(batch_size=8, num_batches=100)
        results["throughput_fps"] = throughput
        
        # 5. PyTorch profiler
        self.logger.info("=== Running PyTorch Profiler ===")
        trace_file, flops_summary = self.profile_with_torch_profiler(num_iterations=20)
        results["trace_file"] = trace_file
        results["flops_summary"] = flops_summary
        
        # Calculate total inference time
        results["total_inference_time_ms"] = full_latency.mean_latency_ms
        
        # Create ProfileResults object
        profile_results = ProfileResults(
            component_latencies=results["component_latencies"],
            memory_metrics=memory_metrics,
            throughput_fps=throughput,
            total_inference_time_ms=results["total_inference_time_ms"],
            overhead_breakdown=overhead_breakdown,
            flops_summary=flops_summary,
            trace_file=trace_file
        )
        
        # Generate report
        self._generate_report(profile_results)
        
        # Log to wandb
        self._log_to_wandb(profile_results)
        
        return profile_results
    
    def _generate_report(self, results: ProfileResults):
        """Generate comprehensive profiling report."""
        report_path = Path(f"results/efficiency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Create visualizations
        self._create_visualizations(results)
        
        # Print summary
        print("\n" + "="*60)
        print("EFFICIENCY PROFILE SUMMARY")
        print("="*60)
        
        # Component latencies
        print("\nComponent Latencies (P99):")
        for name, latency in results.component_latencies.items():
            print(f"  {name}: {latency.p99_latency_ms:.2f}ms")
        
        # Memory usage
        print(f"\nMemory Usage:")
        print(f"  CPU: {results.memory_metrics.cpu_memory_mb:.1f} MB")
        print(f"  GPU: {results.memory_metrics.gpu_memory_mb:.1f} MB")
        print(f"  Peak CPU: {results.memory_metrics.peak_cpu_memory_mb:.1f} MB")
        print(f"  Peak GPU: {results.memory_metrics.peak_gpu_memory_mb:.1f} MB")
        
        # Throughput
        print(f"\nThroughput: {results.throughput_fps:.2f} FPS")
        
        # Overhead breakdown
        print(f"\nOverhead Breakdown:")
        for component, percentage in results.overhead_breakdown.items():
            print(f"  {component}: {percentage:.1f}%")
        
        print(f"\nDetailed report saved to: {report_path}")
        print(f"Profiler trace saved to: {results.trace_file}")
    
    def _create_visualizations(self, results: ProfileResults):
        """Create visualizations of profiling results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Component latency comparison
        components = list(results.component_latencies.keys())
        p99_latencies = [results.component_latencies[c].p99_latency_ms for c in components]
        
        axes[0, 0].bar(components, p99_latencies, color='skyblue')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('P99 Latency (ms)')
        axes[0, 0].set_title('Component Latencies (P99)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Latency distribution for full inference
        full_latency = results.component_latencies["full_inference"]
        latency_data = [
            full_latency.p50_latency_ms,
            full_latency.p90_latency_ms,
            full_latency.p95_latency_ms,
            full_latency.p99_latency_ms
        ]
        percentiles = ['P50', 'P90', 'P95', 'P99']
        
        axes[0, 1].plot(percentiles, latency_data, marker='o', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Percentile')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Full Inference Latency Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Overhead breakdown pie chart
        if results.overhead_breakdown:
            labels = list(results.overhead_breakdown.keys())
            sizes = list(results.overhead_breakdown.values())
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
            
            axes[0, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Component Overhead Breakdown')
        
        # 4. Memory usage comparison
        memory_labels = ['CPU', 'GPU', 'Peak CPU', 'Peak GPU']
        memory_values = [
            results.memory_metrics.cpu_memory_mb,
            results.memory_metrics.gpu_memory_mb,
            results.memory_metrics.peak_cpu_memory_mb,
            results.memory_metrics.peak_gpu_memory_mb
        ]
        
        axes[1, 0].bar(memory_labels, memory_values, color=['blue', 'green', 'lightblue', 'lightgreen'])
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_title('Memory Usage')
        
        # 5. Component mean vs std latency
        mean_latencies = [results.component_latencies[c].mean_latency_ms for c in components]
        std_latencies = [results.component_latencies[c].std_latency_ms for c in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, mean_latencies, width, label='Mean', color='coral')
        axes[1, 1].bar(x + width/2, std_latencies, width, label='Std Dev', color='lightcoral')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Mean vs Standard Deviation')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(components, rotation=45)
        axes[1, 1].legend()
        
        # 6. Throughput and efficiency metrics
        metrics_data = {
            'Throughput\n(FPS)': results.throughput_fps,
            'Total Inference\n(ms)': results.total_inference_time_ms,
            'k-NN Search\n(ms)': results.component_latencies["knn_search"].mean_latency_ms
        }
        
        axes[1, 2].bar(metrics_data.keys(), metrics_data.values(), color='purple', alpha=0.7)
        axes[1, 2].set_title('Key Performance Metrics')
        axes[1, 2].set_ylabel('Value')
        
        # Add values on bars
        for i, (k, v) in enumerate(metrics_data.items()):
            axes[1, 2].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(f"results/efficiency_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({"efficiency_plots": wandb.Image(str(plot_path))})
    
    def _log_to_wandb(self, results: ProfileResults):
        """Log profiling results to wandb."""
        # Log component latencies
        for name, latency in results.component_latencies.items():
            wandb.log({
                f"latency/{name}/mean": latency.mean_latency_ms,
                f"latency/{name}/p50": latency.p50_latency_ms,
                f"latency/{name}/p90": latency.p90_latency_ms,
                f"latency/{name}/p95": latency.p95_latency_ms,
                f"latency/{name}/p99": latency.p99_latency_ms
            })
        
        # Log memory metrics
        wandb.log({
            "memory/cpu_mb": results.memory_metrics.cpu_memory_mb,
            "memory/gpu_mb": results.memory_metrics.gpu_memory_mb,
            "memory/peak_cpu_mb": results.memory_metrics.peak_cpu_memory_mb,
            "memory/peak_gpu_mb": results.memory_metrics.peak_gpu_memory_mb
        })
        
        # Log throughput
        wandb.log({"throughput_fps": results.throughput_fps})
        
        # Log overhead breakdown
        for component, percentage in results.overhead_breakdown.items():
            wandb.log({f"overhead/{component}": percentage})
        
        # Log summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["P99 Latency (ms)", results.component_latencies["full_inference"].p99_latency_ms],
                ["Throughput (FPS)", results.throughput_fps],
                ["Peak GPU Memory (MB)", results.memory_metrics.peak_gpu_memory_mb],
                ["Peak CPU Memory (MB)", results.memory_metrics.peak_cpu_memory_mb]
            ]
        )
        wandb.log({"summary_table": summary_table})


def main():
    """Main function to run efficiency profiling."""
    parser = argparse.ArgumentParser(description="Profile efficiency and latency")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/experiments/pixelis_online.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        default=True,
        help="Profile memory usage"
    )
    parser.add_argument(
        "--profile-cuda",
        action="store_true",
        default=True,
        help="Profile CUDA operations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/efficiency",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize profiler
    profiler = EfficiencyProfiler(
        model_config=args.model_config,
        profile_memory=args.profile_memory,
        profile_cuda=args.profile_cuda
    )
    
    # Run complete profile
    results = profiler.run_complete_profile()
    
    print("\n=== Efficiency Profiling Complete ===")
    print(f"Results saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()