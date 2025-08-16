#!/usr/bin/env python3
"""
Apply Standard Optimizations to Pixelis Inference Pipeline

This script implements state-of-the-art optimizations:
1. torch.compile() for graph optimization
2. INT8 quantization for reduced compute
3. Flash Attention 2 for transformer acceleration
4. Mixed precision inference
5. Memory optimization techniques
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.cuda.amp import autocast
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import logging
from contextlib import contextmanager
import functools

warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimizations."""
    enable_compile: bool = True
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune
    compile_backend: str = "inductor"  # inductor, cudagraphs, onnxrt
    enable_quantization: bool = True
    quantization_dtype: str = "qint8"  # qint8, quint8
    enable_flash_attention: bool = True
    flash_attention_version: int = 2
    enable_mixed_precision: bool = True
    mixed_precision_dtype: torch.dtype = torch.float16  # float16 or bfloat16
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False
    enable_memory_efficient_attention: bool = True
    optimize_for_inference: bool = True
    batch_size: int = 1
    sequence_length: int = 2048
    
    def to_dict(self) -> Dict:
        return {k: str(v) if isinstance(v, torch.dtype) else v 
                for k, v in asdict(self).items()}


@dataclass
class OptimizationResults:
    """Results from applying optimizations."""
    original_latency_ms: float
    optimized_latency_ms: float
    speedup_factor: float
    original_memory_mb: float
    optimized_memory_mb: float
    memory_reduction_percent: float
    original_accuracy: Optional[float] = None
    optimized_accuracy: Optional[float] = None
    accuracy_drop: Optional[float] = None
    optimizations_applied: List[str] = None
    compilation_time_s: Optional[float] = None
    quantization_time_s: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelOptimizer:
    """
    Applies state-of-the-art optimizations to PyTorch models
    for maximum inference performance.
    """
    
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize the model optimizer.
        
        Args:
            config: Optimization configuration
            device: Device to run optimizations on
        """
        self.config = config or OptimizationConfig()
        self.device = device
        self.optimizations_applied = []
        
        # Check available optimizations
        self._check_available_optimizations()
    
    def _check_available_optimizations(self):
        """Check which optimizations are available on the current system."""
        # Check for CUDA
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, disabling GPU optimizations")
            self.config.enable_flash_attention = False
            self.device = "cpu"
        
        # Check for torch.compile support (requires PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            self.config.enable_compile = False
        
        # Check for Flash Attention
        try:
            import flash_attn
            logger.info(f"Flash Attention available: {flash_attn.__version__}")
        except ImportError:
            logger.warning("Flash Attention not installed")
            self.config.enable_flash_attention = False
        
        # Check for mixed precision support
        if self.device == "cpu":
            self.config.enable_mixed_precision = False
        elif not torch.cuda.is_available():
            self.config.enable_mixed_precision = False
        elif torch.cuda.get_device_capability()[0] < 7:
            # Mixed precision requires compute capability 7.0+
            logger.warning("GPU does not support mixed precision (requires compute capability 7.0+)")
            self.config.enable_mixed_precision = False
    
    def apply_torch_compile(
        self,
        model: nn.Module,
        example_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        Apply torch.compile optimization to the model.
        
        Args:
            model: Model to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Compiled model
        """
        if not self.config.enable_compile or not hasattr(torch, 'compile'):
            logger.info("Skipping torch.compile optimization")
            return model
        
        logger.info(f"Applying torch.compile with mode={self.config.compile_mode}, backend={self.config.compile_backend}")
        
        start_time = time.time()
        
        # Configure compilation options
        compile_kwargs = {
            "mode": self.config.compile_mode,
            "backend": self.config.compile_backend,
            "fullgraph": False,  # Allow graph breaks for flexibility
            "dynamic": True,  # Support dynamic shapes
        }
        
        # Additional options for specific backends
        if self.config.compile_backend == "inductor":
            compile_kwargs["options"] = {
                "triton.cudagraphs": True,
                "triton.cudagraph_trees": True,
                "max_autotune": self.config.compile_mode == "max-autotune",
                "coordinate_descent_tuning": True,
                "epilogue_fusion": True,
                "shape_padding": True,
            }
        
        try:
            # Compile the model
            compiled_model = torch.compile(model, **compile_kwargs)
            
            # Warm up the compiled model if example inputs provided
            if example_inputs:
                logger.info("Warming up compiled model...")
                with torch.no_grad():
                    for _ in range(3):
                        _ = compiled_model(**example_inputs)
            
            compilation_time = time.time() - start_time
            logger.info(f"Model compiled successfully in {compilation_time:.2f}s")
            self.optimizations_applied.append("torch_compile")
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            logger.info("Returning original model")
            return model
    
    def apply_quantization(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None
    ) -> nn.Module:
        """
        Apply INT8 quantization to the model.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration (for PTQ)
            
        Returns:
            Quantized model
        """
        if not self.config.enable_quantization:
            logger.info("Skipping quantization")
            return model
        
        logger.info(f"Applying {self.config.quantization_dtype} quantization")
        
        start_time = time.time()
        
        try:
            # Move model to CPU for quantization
            model_cpu = model.cpu()
            
            # Configure quantization
            if self.config.quantization_dtype == "qint8":
                qconfig = quantization.get_default_qconfig('fbgemm')
            else:
                qconfig = quantization.get_default_qconfig('qnnpack')
            
            # Prepare model for quantization
            model_cpu.qconfig = qconfig
            quantization.prepare(model_cpu, inplace=True)
            
            # Calibrate with data if provided
            if calibration_data:
                logger.info("Calibrating quantized model...")
                model_cpu.eval()
                with torch.no_grad():
                    for data in calibration_data[:100]:  # Use first 100 samples
                        if isinstance(data, dict):
                            _ = model_cpu(**data)
                        else:
                            _ = model_cpu(data)
            
            # Convert to quantized model
            quantized_model = quantization.convert(model_cpu, inplace=False)
            
            # Move back to original device
            if self.device != "cpu":
                # Note: Quantized models typically run on CPU
                # For GPU, we'd use different quantization methods
                logger.info("Quantized models run on CPU; keeping on CPU")
            
            quantization_time = time.time() - start_time
            logger.info(f"Model quantized successfully in {quantization_time:.2f}s")
            self.optimizations_applied.append("int8_quantization")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            logger.info("Returning original model")
            return model
    
    def apply_flash_attention(self, model: nn.Module) -> nn.Module:
        """
        Enable Flash Attention 2 for transformer models.
        
        Args:
            model: Transformer model
            
        Returns:
            Model with Flash Attention enabled
        """
        if not self.config.enable_flash_attention:
            logger.info("Skipping Flash Attention")
            return model
        
        logger.info(f"Enabling Flash Attention {self.config.flash_attention_version}")
        
        try:
            # Check if model is from transformers library
            if hasattr(model, 'config'):
                # Transformers model
                if hasattr(model.config, 'attn_implementation'):
                    model.config.attn_implementation = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled via config")
                elif hasattr(model.config, '_flash_attn_2_enabled'):
                    model.config._flash_attn_2_enabled = True
                    logger.info("Flash Attention 2 enabled via flag")
                else:
                    # Try to patch attention modules
                    self._patch_attention_modules(model)
            else:
                # Custom model - try to patch attention modules
                self._patch_attention_modules(model)
            
            self.optimizations_applied.append("flash_attention")
            return model
            
        except Exception as e:
            logger.error(f"Failed to enable Flash Attention: {e}")
            return model
    
    def _patch_attention_modules(self, model: nn.Module):
        """
        Patch attention modules to use Flash Attention.
        
        Args:
            model: Model to patch
        """
        try:
            from flash_attn import flash_attn_func
            
            def flash_attention_forward(self, query, key, value, attn_mask=None):
                """Flash Attention forward pass."""
                # Reshape for flash attention
                batch_size, seq_len, num_heads, head_dim = query.shape
                
                # Flash attention expects (batch, seq, heads, dim)
                q = query.transpose(1, 2)
                k = key.transpose(1, 2)
                v = value.transpose(1, 2)
                
                # Apply flash attention
                output = flash_attn_func(q, k, v, causal=True)
                
                # Reshape back
                output = output.transpose(1, 2)
                return output
            
            # Patch MultiheadAttention modules
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    # Replace forward method
                    module.forward = functools.partial(flash_attention_forward, module)
                    logger.info(f"Patched {name} with Flash Attention")
                    
        except ImportError:
            logger.warning("Flash Attention not available for patching")
    
    def apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """
        Configure model for mixed precision inference.
        
        Args:
            model: Model to configure
            
        Returns:
            Model configured for mixed precision
        """
        if not self.config.enable_mixed_precision:
            logger.info("Skipping mixed precision")
            return model
        
        dtype_str = "float16" if self.config.mixed_precision_dtype == torch.float16 else "bfloat16"
        logger.info(f"Configuring mixed precision with {dtype_str}")
        
        # Convert model to half precision
        if self.device != "cpu":
            model = model.to(dtype=self.config.mixed_precision_dtype)
            logger.info(f"Model converted to {dtype_str}")
            self.optimizations_applied.append(f"mixed_precision_{dtype_str}")
        else:
            logger.warning("Mixed precision not supported on CPU")
        
        return model
    
    def apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimization techniques.
        
        Args:
            model: Model to optimize
            
        Returns:
            Memory-optimized model
        """
        logger.info("Applying memory optimizations")
        
        # Enable gradient checkpointing if available
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
                self.optimizations_applied.append("gradient_checkpointing")
        
        # Enable memory efficient attention
        if self.config.enable_memory_efficient_attention:
            if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
                logger.info("Memory efficient attention enabled")
                self.optimizations_applied.append("memory_efficient_attention")
        
        # CPU offload for large models
        if self.config.enable_cpu_offload:
            try:
                from accelerate import cpu_offload_with_hook
                model, hook = cpu_offload_with_hook(model)
                logger.info("CPU offload enabled")
                self.optimizations_applied.append("cpu_offload")
            except ImportError:
                logger.warning("Accelerate not available for CPU offload")
        
        # Set model to eval mode and disable gradient computation
        if self.config.optimize_for_inference:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Model set to inference mode")
        
        return model
    
    def optimize_model(
        self,
        model: nn.Module,
        example_inputs: Optional[Dict[str, torch.Tensor]] = None,
        calibration_data: Optional[List[torch.Tensor]] = None
    ) -> Tuple[nn.Module, OptimizationResults]:
        """
        Apply all configured optimizations to the model.
        
        Args:
            model: Model to optimize
            example_inputs: Example inputs for compilation
            calibration_data: Data for quantization calibration
            
        Returns:
            Tuple of (optimized_model, results)
        """
        logger.info("Starting model optimization pipeline")
        
        # Benchmark original model
        original_latency, original_memory = self._benchmark_model(
            model, example_inputs
        )
        
        # Apply optimizations in sequence
        optimized_model = model
        
        # 1. Memory optimizations (should be first)
        optimized_model = self.apply_memory_optimizations(optimized_model)
        
        # 2. Flash Attention (before compilation)
        optimized_model = self.apply_flash_attention(optimized_model)
        
        # 3. Mixed precision
        optimized_model = self.apply_mixed_precision(optimized_model)
        
        # 4. Quantization (may not work with all optimizations)
        if self.device == "cpu" or not self.config.enable_compile:
            # Quantization typically works better on CPU or without compilation
            optimized_model = self.apply_quantization(
                optimized_model, calibration_data
            )
        
        # 5. Compilation (should be last)
        optimized_model = self.apply_torch_compile(
            optimized_model, example_inputs
        )
        
        # Benchmark optimized model
        optimized_latency, optimized_memory = self._benchmark_model(
            optimized_model, example_inputs
        )
        
        # Calculate results
        results = OptimizationResults(
            original_latency_ms=original_latency,
            optimized_latency_ms=optimized_latency,
            speedup_factor=original_latency / optimized_latency if optimized_latency > 0 else 0,
            original_memory_mb=original_memory,
            optimized_memory_mb=optimized_memory,
            memory_reduction_percent=((original_memory - optimized_memory) / original_memory * 100) if original_memory > 0 else 0,
            optimizations_applied=self.optimizations_applied.copy()
        )
        
        logger.info(f"Optimization complete. Speedup: {results.speedup_factor:.2f}x, Memory reduction: {results.memory_reduction_percent:.1f}%")
        
        return optimized_model, results
    
    def _benchmark_model(
        self,
        model: nn.Module,
        example_inputs: Optional[Dict[str, torch.Tensor]] = None,
        num_iterations: int = 100
    ) -> Tuple[float, float]:
        """
        Benchmark model latency and memory usage.
        
        Args:
            model: Model to benchmark
            example_inputs: Example inputs
            num_iterations: Number of iterations for timing
            
        Returns:
            Tuple of (latency_ms, memory_mb)
        """
        if example_inputs is None:
            # Create dummy inputs
            example_inputs = self._create_dummy_inputs()
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    _ = model(**example_inputs)
                except:
                    # Try as single tensor input
                    _ = model(example_inputs['input_ids'] if 'input_ids' in example_inputs else example_inputs)
        
        # Measure latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                try:
                    _ = model(**example_inputs)
                except:
                    _ = model(example_inputs['input_ids'] if 'input_ids' in example_inputs else example_inputs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) / num_iterations * 1000
        
        # Measure memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        
        return latency_ms, memory_mb
    
    def _create_dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for benchmarking."""
        return {
            'input_ids': torch.randint(
                0, 30000,
                (self.config.batch_size, self.config.sequence_length),
                device=self.device
            ),
            'attention_mask': torch.ones(
                (self.config.batch_size, self.config.sequence_length),
                device=self.device
            )
        }


class DynamicsModelOptimizer(ModelOptimizer):
    """Specialized optimizer for the dynamics model used in curiosity reward."""
    
    def optimize_dynamics_model(
        self,
        dynamics_model: nn.Module,
        state_dim: int = 768,
        action_dim: int = 128
    ) -> Tuple[nn.Module, OptimizationResults]:
        """
        Optimize the dynamics model specifically.
        
        Args:
            dynamics_model: Dynamics model to optimize
            state_dim: State dimension
            action_dim: Action dimension
            
        Returns:
            Tuple of (optimized_model, results)
        """
        logger.info("Optimizing dynamics model for curiosity reward")
        
        # Create example inputs for dynamics model
        example_inputs = {
            'state': torch.randn(1, state_dim, device=self.device),
            'action': torch.randn(1, action_dim, device=self.device),
            'next_state': torch.randn(1, state_dim, device=self.device)
        }
        
        # Apply optimizations
        return self.optimize_model(dynamics_model, example_inputs)


class KNNSearchOptimizer:
    """Optimizer for k-NN search operations."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize k-NN optimizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
    def optimize_faiss_index(
        self,
        index: Any,
        use_approximate: bool = True,
        nprobe: int = 32
    ) -> Any:
        """
        Optimize FAISS index for faster search.
        
        Args:
            index: FAISS index
            use_approximate: Use approximate search
            nprobe: Number of clusters to search (for IVF indices)
            
        Returns:
            Optimized index
        """
        import faiss
        
        logger.info("Optimizing FAISS index")
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except:
                logger.warning("Failed to move FAISS to GPU")
        
        # Configure for approximate search
        if use_approximate and hasattr(index, 'nprobe'):
            index.nprobe = nprobe
            logger.info(f"Set nprobe to {nprobe} for approximate search")
        
        return index


def create_optimization_report(results: OptimizationResults, output_path: Path):
    """
    Create a detailed optimization report.
    
    Args:
        results: Optimization results
        output_path: Path to save report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results.to_dict(),
        "summary": {
            "speedup": f"{results.speedup_factor:.2f}x",
            "memory_savings": f"{results.memory_reduction_percent:.1f}%",
            "latency_improvement": f"{results.original_latency_ms - results.optimized_latency_ms:.2f}ms",
            "optimizations_count": len(results.optimizations_applied or [])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Optimization report saved to {output_path}")


def main():
    """Main entry point for applying optimizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply optimizations to Pixelis models")
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--optimization-config",
        type=str,
        help="Path to optimization config JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimized_models",
        help="Directory to save optimized models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run optimizations on"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.optimization_config:
        with open(args.optimization_config, 'r') as f:
            config_dict = json.load(f)
            config = OptimizationConfig(**config_dict)
    else:
        config = OptimizationConfig()
    
    # Create optimizer
    optimizer = ModelOptimizer(config=config, device=args.device)
    
    # Create mock model for demonstration
    logger.info("Creating mock model for optimization demonstration")
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 1000)
    ).to(args.device)
    
    # Optimize model
    optimized_model, results = optimizer.optimize_model(model)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized model
    model_path = output_dir / "optimized_model.pt"
    torch.save(optimized_model.state_dict(), model_path)
    logger.info(f"Optimized model saved to {model_path}")
    
    # Save report
    report_path = output_dir / "optimization_report.json"
    create_optimization_report(results, report_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Speedup: {results.speedup_factor:.2f}x")
    print(f"Memory reduction: {results.memory_reduction_percent:.1f}%")
    print(f"Original latency: {results.original_latency_ms:.2f}ms")
    print(f"Optimized latency: {results.optimized_latency_ms:.2f}ms")
    print(f"Optimizations applied: {', '.join(results.optimizations_applied or [])}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())