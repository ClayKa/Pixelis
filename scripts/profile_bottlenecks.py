#!/usr/bin/env python3
"""
Comprehensive Bottleneck Profiling for Pixelis Inference Pipeline

This script performs detailed profiling to identify performance bottlenecks
in the inference pipeline, focusing on:
1. Base model forward pass
2. k-NN search operations
3. Dynamics model (curiosity reward)
4. Visual operations
5. Memory transfer operations
"""

import torch
import torch.profiler
import torch.nn as nn
import numpy as np
import json
import time
import psutil
import GPUtil
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
import argparse
import tracemalloc
import cProfile
import pstats
import io
import warnings
import functools
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import core modules
from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.modules.dynamics_model import DynamicsModel
from core.modules.reward_shaping_enhanced import RewardOrchestrator
from core.models.peft_model import PEFTModel
from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class BottleneckProfile:
    """Detailed profile of a potential bottleneck."""
    component_name: str
    total_time_ms: float
    percentage_of_total: float
    calls_count: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    cuda_memory_mb: float
    flops: Optional[int] = None
    is_bottleneck: bool = False
    bottleneck_severity: str = "none"  # none, low, medium, high, critical
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetailedProfileResults:
    """Complete profiling results with bottleneck analysis."""
    bottlenecks: List[BottleneckProfile]
    total_inference_time_ms: float
    throughput_samples_per_sec: float
    memory_peak_mb: float
    gpu_utilization_percent: float
    critical_path: List[str]  # Components on the critical path
    optimization_potential: Dict[str, float]  # Potential speedup from optimizing each component
    profile_metadata: Dict[str, Any]
    trace_files: Dict[str, str]
    
    def to_dict(self) -> Dict:
        return {
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "total_inference_time_ms": self.total_inference_time_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "critical_path": self.critical_path,
            "optimization_potential": self.optimization_potential,
            "profile_metadata": self.profile_metadata,
            "trace_files": self.trace_files
        }


class AdvancedBottleneckProfiler:
    """
    Advanced profiler for identifying and analyzing performance bottlenecks
    in the Pixelis inference pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        warmup_iterations: int = 5,
        profile_iterations: int = 50,
        output_dir: Path = Path("profiling_results")
    ):
        """
        Initialize the advanced profiler.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run profiling on
            warmup_iterations: Number of warmup iterations
            profile_iterations: Number of iterations to profile
            output_dir: Directory to save profiling results
        """
        self.model_path = model_path
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing storage
        self.component_timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict] = []
        
        # Components to profile
        self.components_to_profile = [
            "model_forward",
            "knn_search",
            "dynamics_model",
            "visual_operations",
            "memory_transfer",
            "reward_computation",
            "ensemble_voting",
            "preprocessing",
            "postprocessing"
        ]
        
        # Initialize components (mock if needed)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize or mock the components to be profiled."""
        # Mock components for demonstration
        # In production, these would be actual model instances
        
        # Base model (mock with a small transformer)
        if self.model_path and self.model_path.exists():
            # Load actual model
            logger.info(f"Loading model from {self.model_path}")
            # self.model = load_model(self.model_path)
        else:
            # Create mock model for profiling
            logger.info("Creating mock model for profiling")
            from transformers import AutoModel
            try:
                self.model = AutoModel.from_pretrained(
                    "bert-base-uncased",
                    torch_dtype=torch.float16
                ).to(self.device)
            except:
                # Fallback to simple mock
                self.model = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.ReLU(),
                    nn.Linear(768, 768)
                ).to(self.device)
        
        # Experience buffer with k-NN
        self.experience_buffer = self._create_mock_experience_buffer()
        
        # Dynamics model for curiosity
        self.dynamics_model = self._create_mock_dynamics_model()
        
        # Visual operations registry
        self.visual_ops = self._create_mock_visual_ops()
    
    def _create_mock_experience_buffer(self) -> Any:
        """Create a mock experience buffer with k-NN search."""
        class MockExperienceBuffer:
            def __init__(self, device):
                self.device = device
                # Create mock FAISS index
                import faiss
                self.dimension = 768
                self.index = faiss.IndexFlatL2(self.dimension)
                # Add some mock vectors
                mock_vectors = np.random.randn(1000, self.dimension).astype('float32')
                self.index.add(mock_vectors)
            
            def search_index(self, query: torch.Tensor, k: int = 10):
                """Mock k-NN search."""
                if query.is_cuda:
                    query = query.cpu().numpy()
                else:
                    query = query.numpy()
                
                if len(query.shape) == 1:
                    query = query.reshape(1, -1)
                
                distances, indices = self.index.search(query.astype('float32'), k)
                return torch.tensor(distances), torch.tensor(indices)
        
        return MockExperienceBuffer(self.device)
    
    def _create_mock_dynamics_model(self) -> nn.Module:
        """Create a mock dynamics model for curiosity reward."""
        class MockDynamicsModel(nn.Module):
            def __init__(self, state_dim: int = 768, action_dim: int = 128):
                super().__init__()
                self.forward_model = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, state_dim)
                )
                self.inverse_model = nn.Sequential(
                    nn.Linear(state_dim * 2, 512),
                    nn.ReLU(),
                    nn.Linear(512, action_dim)
                )
            
            def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
                # Forward model prediction
                state_action = torch.cat([state, action], dim=-1)
                predicted_next_state = self.forward_model(state_action)
                
                # Inverse model prediction
                state_pair = torch.cat([state, next_state], dim=-1)
                predicted_action = self.inverse_model(state_pair)
                
                return predicted_next_state, predicted_action
        
        return MockDynamicsModel().to(self.device)
    
    def _create_mock_visual_ops(self) -> Dict[str, Callable]:
        """Create mock visual operations."""
        def mock_segment(image: torch.Tensor, x: int, y: int) -> torch.Tensor:
            """Mock segmentation operation."""
            # Simulate some computation
            result = torch.nn.functional.conv2d(
                image.unsqueeze(0) if len(image.shape) == 3 else image,
                torch.randn(1, 3, 3, 3).to(image.device),
                padding=1
            )
            return result.squeeze(0)
        
        def mock_ocr(image: torch.Tensor, bbox: Tuple[int, int, int, int]) -> str:
            """Mock OCR operation."""
            # Simulate text extraction
            time.sleep(0.01)  # Simulate OCR latency
            return "mock_text"
        
        return {
            "SEGMENT_OBJECT_AT": mock_segment,
            "READ_TEXT": mock_ocr,
            "ZOOM_IN": lambda img, x, y, z: img,
            "GET_PROPERTIES": lambda img, obj: {"size": "medium", "color": "red"}
        }
    
    @contextmanager
    def profile_component(self, component_name: str):
        """Context manager for profiling a specific component."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        # Memory snapshot before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = 0
        
        try:
            yield
        finally:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            # Record timing
            elapsed_ms = (end_time - start_time) * 1000
            self.component_timings[component_name].append(elapsed_ms)
            
            # Memory snapshot after
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                mem_delta = mem_after - mem_before
                self.memory_snapshots.append({
                    "component": component_name,
                    "memory_delta_mb": mem_delta / 1024 / 1024,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                })
    
    def profile_model_forward(self, batch_size: int = 1) -> None:
        """Profile the base model forward pass."""
        # Create mock input
        if hasattr(self.model, 'config'):
            # Transformer model
            input_ids = torch.randint(0, 1000, (batch_size, 128)).to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            with self.profile_component("model_forward"):
                with torch.no_grad():
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # Simple model
            input_tensor = torch.randn(batch_size, 768).to(self.device)
            
            with self.profile_component("model_forward"):
                with torch.no_grad():
                    _ = self.model(input_tensor)
    
    def profile_knn_search(self, batch_size: int = 1, k: int = 10) -> None:
        """Profile k-NN search operations."""
        query = torch.randn(batch_size, 768).to(self.device)
        
        with self.profile_component("knn_search"):
            _ = self.experience_buffer.search_index(query, k=k)
    
    def profile_dynamics_model(self, batch_size: int = 1) -> None:
        """Profile dynamics model for curiosity reward."""
        state = torch.randn(batch_size, 768).to(self.device)
        action = torch.randn(batch_size, 128).to(self.device)
        next_state = torch.randn(batch_size, 768).to(self.device)
        
        with self.profile_component("dynamics_model"):
            with torch.no_grad():
                _ = self.dynamics_model(state, action, next_state)
    
    def profile_visual_operations(self, num_ops: int = 3) -> None:
        """Profile visual operations."""
        image = torch.randn(3, 224, 224).to(self.device)
        
        with self.profile_component("visual_operations"):
            for _ in range(num_ops):
                # Randomly select and execute an operation
                op_name = np.random.choice(list(self.visual_ops.keys()))
                op_func = self.visual_ops[op_name]
                
                if op_name == "SEGMENT_OBJECT_AT":
                    _ = op_func(image, 100, 100)
                elif op_name == "READ_TEXT":
                    _ = op_func(image, (50, 50, 150, 150))
                else:
                    _ = op_func(image, 0, 0, 0)
    
    def profile_memory_transfer(self, tensor_size: Tuple[int, ...] = (1, 768)) -> None:
        """Profile memory transfer operations."""
        # CPU to GPU transfer
        cpu_tensor = torch.randn(*tensor_size)
        
        with self.profile_component("memory_transfer"):
            gpu_tensor = cpu_tensor.to(self.device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            _ = gpu_tensor.cpu()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    def run_comprehensive_profiling(self) -> DetailedProfileResults:
        """Run comprehensive profiling of all components."""
        logger.info("Starting comprehensive profiling...")
        
        # Warmup
        logger.info(f"Running {self.warmup_iterations} warmup iterations...")
        for _ in tqdm(range(self.warmup_iterations), desc="Warmup"):
            self.profile_model_forward()
            self.profile_knn_search()
            self.profile_dynamics_model()
            self.profile_visual_operations()
            self.profile_memory_transfer()
        
        # Clear warmup timings
        self.component_timings.clear()
        self.memory_snapshots.clear()
        
        # Profile with torch.profiler
        logger.info(f"Running {self.profile_iterations} profiling iterations...")
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        trace_file = str(self.output_dir / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_file)
        ) as prof:
            for i in tqdm(range(self.profile_iterations), desc="Profiling"):
                # Profile each component
                self.profile_model_forward()
                prof.step()
                
                self.profile_knn_search()
                prof.step()
                
                self.profile_dynamics_model()
                prof.step()
                
                self.profile_visual_operations()
                prof.step()
                
                self.profile_memory_transfer()
                prof.step()
                
                # Additional components
                with self.profile_component("preprocessing"):
                    # Simulate preprocessing
                    dummy = torch.randn(1, 3, 224, 224).to(self.device)
                    dummy = torch.nn.functional.interpolate(dummy, size=(256, 256))
                prof.step()
                
                with self.profile_component("postprocessing"):
                    # Simulate postprocessing
                    dummy = torch.randn(1, 1000).to(self.device)
                    dummy = torch.nn.functional.softmax(dummy, dim=-1)
                prof.step()
        
        # Analyze results
        results = self._analyze_profiling_results(prof, trace_file)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        # Save detailed report
        self._save_detailed_report(results)
        
        return results
    
    def _analyze_profiling_results(
        self,
        prof: torch.profiler.profile,
        trace_file: str
    ) -> DetailedProfileResults:
        """Analyze profiling results and identify bottlenecks."""
        bottlenecks = []
        
        # Calculate statistics for each component
        total_time = 0
        for component_name, timings in self.component_timings.items():
            if not timings:
                continue
            
            timings_array = np.array(timings)
            total_time += np.sum(timings_array)
        
        # Analyze each component
        for component_name, timings in self.component_timings.items():
            if not timings:
                continue
            
            timings_array = np.array(timings)
            
            # Calculate statistics
            total_component_time = np.sum(timings_array)
            percentage = (total_component_time / total_time) * 100 if total_time > 0 else 0
            
            # Memory statistics
            component_memory = [
                m["memory_delta_mb"] for m in self.memory_snapshots
                if m["component"] == component_name
            ]
            avg_memory = np.mean(component_memory) if component_memory else 0
            
            # Determine if it's a bottleneck
            is_bottleneck = percentage > 20  # More than 20% of total time
            severity = self._determine_bottleneck_severity(percentage)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                component_name, percentage, timings_array
            )
            
            profile = BottleneckProfile(
                component_name=component_name,
                total_time_ms=total_component_time,
                percentage_of_total=percentage,
                calls_count=len(timings),
                avg_time_ms=np.mean(timings_array),
                min_time_ms=np.min(timings_array),
                max_time_ms=np.max(timings_array),
                std_time_ms=np.std(timings_array),
                p50_ms=np.percentile(timings_array, 50),
                p90_ms=np.percentile(timings_array, 90),
                p95_ms=np.percentile(timings_array, 95),
                p99_ms=np.percentile(timings_array, 99),
                memory_allocated_mb=avg_memory,
                memory_reserved_mb=0,  # Would need more detailed tracking
                cuda_memory_mb=avg_memory if torch.cuda.is_available() else 0,
                is_bottleneck=is_bottleneck,
                bottleneck_severity=severity,
                optimization_suggestions=suggestions
            )
            
            bottlenecks.append(profile)
        
        # Sort by percentage (descending)
        bottlenecks.sort(key=lambda x: x.percentage_of_total, reverse=True)
        
        # Identify critical path
        critical_path = [b.component_name for b in bottlenecks if b.is_bottleneck]
        
        # Calculate optimization potential
        optimization_potential = {}
        for bottleneck in bottlenecks:
            if bottleneck.is_bottleneck:
                # Estimate potential speedup based on common optimizations
                if "model_forward" in bottleneck.component_name:
                    potential = 0.5  # 50% speedup possible with quantization/compilation
                elif "knn_search" in bottleneck.component_name:
                    potential = 0.3  # 30% speedup with approximate search
                elif "dynamics_model" in bottleneck.component_name:
                    potential = 0.4  # 40% speedup with model optimization
                else:
                    potential = 0.2  # 20% general optimization potential
                
                optimization_potential[bottleneck.component_name] = potential
        
        # Get GPU utilization
        gpu_util = 0
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = gpus[0].load * 100
            except:
                pass
        
        # Calculate throughput
        avg_iteration_time = total_time / self.profile_iterations if self.profile_iterations > 0 else 1
        throughput = 1000 / avg_iteration_time if avg_iteration_time > 0 else 0
        
        # Peak memory
        peak_memory = max([m["peak_memory_mb"] for m in self.memory_snapshots]) if self.memory_snapshots else 0
        
        return DetailedProfileResults(
            bottlenecks=bottlenecks,
            total_inference_time_ms=total_time,
            throughput_samples_per_sec=throughput,
            memory_peak_mb=peak_memory,
            gpu_utilization_percent=gpu_util,
            critical_path=critical_path,
            optimization_potential=optimization_potential,
            profile_metadata={
                "device": self.device,
                "warmup_iterations": self.warmup_iterations,
                "profile_iterations": self.profile_iterations,
                "timestamp": datetime.now().isoformat()
            },
            trace_files={"tensorboard": trace_file}
        )
    
    def _determine_bottleneck_severity(self, percentage: float) -> str:
        """Determine the severity of a bottleneck."""
        if percentage >= 50:
            return "critical"
        elif percentage >= 30:
            return "high"
        elif percentage >= 20:
            return "medium"
        elif percentage >= 10:
            return "low"
        else:
            return "none"
    
    def _generate_optimization_suggestions(
        self,
        component_name: str,
        percentage: float,
        timings: np.ndarray
    ) -> List[str]:
        """Generate optimization suggestions for a component."""
        suggestions = []
        
        if "model_forward" in component_name:
            suggestions.extend([
                "Apply torch.compile() for graph optimization",
                "Use INT8 quantization to reduce compute",
                "Enable Flash Attention 2 for transformer models",
                "Consider model distillation for smaller model",
                "Use mixed precision (fp16/bf16) training"
            ])
        
        elif "knn_search" in component_name:
            suggestions.extend([
                "Use approximate nearest neighbor search (e.g., HNSW)",
                "Reduce embedding dimensionality with PCA",
                "Implement hierarchical search with clustering",
                "Cache frequently accessed neighbors",
                "Use GPU-accelerated FAISS indices"
            ])
        
        elif "dynamics_model" in component_name:
            suggestions.extend([
                "Apply torch.compile() to dynamics model",
                "Reduce model size with pruning",
                "Use cached predictions for similar states",
                "Batch multiple predictions together",
                "Consider simpler architecture"
            ])
        
        elif "visual_operations" in component_name:
            suggestions.extend([
                "Batch multiple operations when possible",
                "Cache operation results",
                "Use lower resolution for initial detection",
                "Implement early stopping for iterative operations",
                "Parallelize independent operations"
            ])
        
        elif "memory_transfer" in component_name:
            suggestions.extend([
                "Use pinned memory for CPU-GPU transfers",
                "Implement double buffering",
                "Reduce transfer frequency with batching",
                "Keep tensors on GPU when possible",
                "Use unified memory on supported hardware"
            ])
        
        # High variance suggests optimization opportunity
        if timings.std() / timings.mean() > 0.3:
            suggestions.append("High variance suggests unstable performance - investigate causes")
        
        return suggestions
    
    def _generate_visualizations(self, results: DetailedProfileResults) -> None:
        """Generate visualization plots for the profiling results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Pixelis Inference Pipeline Profiling Results", fontsize=16)
        
        # 1. Component time breakdown (pie chart)
        ax = axes[0, 0]
        components = [b.component_name for b in results.bottlenecks]
        percentages = [b.percentage_of_total for b in results.bottlenecks]
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        
        wedges, texts, autotexts = ax.pie(
            percentages,
            labels=components,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title("Time Breakdown by Component")
        
        # 2. Latency distribution (box plot)
        ax = axes[0, 1]
        latency_data = []
        labels = []
        for component_name, timings in self.component_timings.items():
            if timings:
                latency_data.append(timings)
                labels.append(component_name.replace("_", "\n"))
        
        if latency_data:
            bp = ax.boxplot(latency_data, labels=labels)
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Latency Distribution")
            ax.tick_params(axis='x', rotation=45)
        
        # 3. Bottleneck severity (bar chart)
        ax = axes[0, 2]
        bottleneck_names = [b.component_name for b in results.bottlenecks if b.is_bottleneck]
        bottleneck_times = [b.total_time_ms for b in results.bottlenecks if b.is_bottleneck]
        
        if bottleneck_names:
            bars = ax.bar(range(len(bottleneck_names)), bottleneck_times)
            ax.set_xticks(range(len(bottleneck_names)))
            ax.set_xticklabels(bottleneck_names, rotation=45, ha='right')
            ax.set_ylabel("Total Time (ms)")
            ax.set_title("Identified Bottlenecks")
            
            # Color bars by severity
            severity_colors = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow",
                "low": "green"
            }
            for i, b in enumerate([b for b in results.bottlenecks if b.is_bottleneck]):
                bars[i].set_color(severity_colors.get(b.bottleneck_severity, "blue"))
        
        # 4. Memory usage
        ax = axes[1, 0]
        if self.memory_snapshots:
            components = list(set([m["component"] for m in self.memory_snapshots]))
            avg_memory = []
            for comp in components:
                comp_memory = [m["memory_delta_mb"] for m in self.memory_snapshots if m["component"] == comp]
                avg_memory.append(np.mean(comp_memory) if comp_memory else 0)
            
            ax.bar(range(len(components)), avg_memory)
            ax.set_xticks(range(len(components)))
            ax.set_xticklabels(components, rotation=45, ha='right')
            ax.set_ylabel("Memory (MB)")
            ax.set_title("Average Memory Usage by Component")
        
        # 5. Optimization potential
        ax = axes[1, 1]
        if results.optimization_potential:
            components = list(results.optimization_potential.keys())
            potential = list(results.optimization_potential.values())
            potential_percent = [p * 100 for p in potential]
            
            ax.barh(range(len(components)), potential_percent)
            ax.set_yticks(range(len(components)))
            ax.set_yticklabels(components)
            ax.set_xlabel("Potential Speedup (%)")
            ax.set_title("Optimization Potential")
        
        # 6. Timeline view
        ax = axes[1, 2]
        timeline_data = []
        current_time = 0
        for component_name in self.components_to_profile[:5]:  # Top 5 components
            if component_name in self.component_timings:
                avg_time = np.mean(self.component_timings[component_name])
                timeline_data.append((component_name, current_time, avg_time))
                current_time += avg_time
        
        if timeline_data:
            for i, (name, start, duration) in enumerate(timeline_data):
                ax.barh(i, duration, left=start, height=0.5)
                ax.text(start + duration/2, i, name, ha='center', va='center')
            
            ax.set_xlabel("Time (ms)")
            ax.set_title("Execution Timeline")
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f"profiling_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        
        plt.close()
    
    def _save_detailed_report(self, results: DetailedProfileResults) -> None:
        """Save detailed profiling report."""
        # JSON report
        json_path = self.output_dir / f"profiling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")
        
        # Human-readable text report
        text_path = self.output_dir / f"profiling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PIXELIS INFERENCE PIPELINE PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {results.profile_metadata['timestamp']}\n")
            f.write(f"Device: {results.profile_metadata['device']}\n")
            f.write(f"Total Inference Time: {results.total_inference_time_ms:.2f} ms\n")
            f.write(f"Throughput: {results.throughput_samples_per_sec:.2f} samples/sec\n")
            f.write(f"Peak Memory: {results.memory_peak_mb:.2f} MB\n")
            f.write(f"GPU Utilization: {results.gpu_utilization_percent:.1f}%\n\n")
            
            f.write("IDENTIFIED BOTTLENECKS\n")
            f.write("-" * 40 + "\n")
            for bottleneck in results.bottlenecks:
                if bottleneck.is_bottleneck:
                    f.write(f"\n{bottleneck.component_name.upper()}:\n")
                    f.write(f"  Severity: {bottleneck.bottleneck_severity}\n")
                    f.write(f"  Time: {bottleneck.total_time_ms:.2f} ms ({bottleneck.percentage_of_total:.1f}%)\n")
                    f.write(f"  Avg Latency: {bottleneck.avg_time_ms:.2f} ms\n")
                    f.write(f"  P99 Latency: {bottleneck.p99_ms:.2f} ms\n")
                    f.write(f"  Memory: {bottleneck.memory_allocated_mb:.2f} MB\n")
                    f.write("  Optimization Suggestions:\n")
                    for suggestion in bottleneck.optimization_suggestions:
                        f.write(f"    • {suggestion}\n")
            
            f.write("\nCRITICAL PATH\n")
            f.write("-" * 40 + "\n")
            f.write(" -> ".join(results.critical_path) + "\n")
            
            f.write("\nOPTIMIZATION POTENTIAL\n")
            f.write("-" * 40 + "\n")
            for component, potential in results.optimization_potential.items():
                f.write(f"  {component}: {potential*100:.0f}% potential speedup\n")
        
        logger.info(f"Saved text report to {text_path}")


def main():
    """Main entry point for bottleneck profiling."""
    parser = argparse.ArgumentParser(description="Profile Pixelis inference pipeline bottlenecks")
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run profiling on"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=50,
        help="Number of iterations to profile"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = AdvancedBottleneckProfiler(
        model_path=Path(args.model_path) if args.model_path else None,
        device=args.device,
        warmup_iterations=args.warmup_iterations,
        profile_iterations=args.profile_iterations,
        output_dir=Path(args.output_dir)
    )
    
    # Run profiling
    results = profiler.run_comprehensive_profiling()
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"Total Inference Time: {results.total_inference_time_ms:.2f} ms")
    print(f"Throughput: {results.throughput_samples_per_sec:.2f} samples/sec")
    print(f"Critical Path: {' -> '.join(results.critical_path)}")
    
    print("\nTop Bottlenecks:")
    for i, bottleneck in enumerate(results.bottlenecks[:3]):
        if bottleneck.is_bottleneck:
            print(f"  {i+1}. {bottleneck.component_name}: {bottleneck.percentage_of_total:.1f}% ({bottleneck.bottleneck_severity})")
    
    print(f"\nFull report saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())