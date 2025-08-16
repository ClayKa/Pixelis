#!/usr/bin/env python3
"""
Micro-benchmark script for computational cost estimation.
Runs a small training loop to measure actual performance metrics.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.profiler as profiler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class BenchmarkMetrics:
    """Container for benchmark measurements."""
    
    # Timing metrics
    total_time_seconds: float
    avg_step_time_seconds: float
    min_step_time_seconds: float
    max_step_time_seconds: float
    data_loading_time_ratio: float
    forward_time_ratio: float
    backward_time_ratio: float
    optimizer_time_ratio: float
    
    # Memory metrics
    peak_vram_gb: float
    avg_vram_gb: float
    model_size_gb: float
    gradient_size_gb: float
    optimizer_state_size_gb: float
    activation_memory_gb: float
    
    # Throughput metrics
    samples_per_second: float
    tokens_per_second: float
    tflops: float
    
    # Configuration
    num_gpus: int
    gpu_model: str
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    num_steps: int
    model_name: str
    precision: str
    
    # System info
    cuda_version: str
    pytorch_version: str
    transformers_version: str
    
    def to_json(self, filepath: Path):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        report = []
        report.append("=" * 60)
        report.append("MICRO-BENCHMARK RESULTS")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        report.append("Configuration:")
        report.append(f"  Model: {self.model_name}")
        report.append(f"  GPUs: {self.num_gpus} x {self.gpu_model}")
        report.append(f"  Batch Size: {self.batch_size}")
        report.append(f"  Sequence Length: {self.sequence_length}")
        report.append(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        report.append(f"  Precision: {self.precision}")
        report.append("")
        
        report.append("Timing Metrics:")
        report.append(f"  Total Time: {self.total_time_seconds:.2f} seconds")
        report.append(f"  Avg Step Time: {self.avg_step_time_seconds:.3f} seconds")
        report.append(f"  Step Time Range: [{self.min_step_time_seconds:.3f}, {self.max_step_time_seconds:.3f}] seconds")
        report.append(f"  Data Loading: {self.data_loading_time_ratio:.1%} of time")
        report.append(f"  Forward Pass: {self.forward_time_ratio:.1%} of time")
        report.append(f"  Backward Pass: {self.backward_time_ratio:.1%} of time")
        report.append(f"  Optimizer Step: {self.optimizer_time_ratio:.1%} of time")
        report.append("")
        
        report.append("Memory Metrics:")
        report.append(f"  Peak VRAM: {self.peak_vram_gb:.2f} GB")
        report.append(f"  Average VRAM: {self.avg_vram_gb:.2f} GB")
        report.append(f"  Model Size: {self.model_size_gb:.2f} GB")
        report.append(f"  Gradient Size: {self.gradient_size_gb:.2f} GB")
        report.append(f"  Optimizer State: {self.optimizer_state_size_gb:.2f} GB")
        report.append(f"  Activation Memory: {self.activation_memory_gb:.2f} GB")
        report.append("")
        
        report.append("Throughput Metrics:")
        report.append(f"  Samples/Second: {self.samples_per_second:.2f}")
        report.append(f"  Tokens/Second: {self.tokens_per_second:.0f}")
        report.append(f"  TFLOPS: {self.tflops:.2f}")
        report.append("")
        
        report.append("System Info:")
        report.append(f"  CUDA Version: {self.cuda_version}")
        report.append(f"  PyTorch Version: {self.pytorch_version}")
        report.append(f"  Transformers Version: {self.transformers_version}")
        report.append("")
        
        report.append("Extrapolated Estimates (Single GPU):")
        hours_per_epoch_10k = (10000 * self.avg_step_time_seconds) / 3600
        report.append(f"  10,000 steps: {hours_per_epoch_10k:.1f} hours")
        report.append(f"  100,000 steps: {hours_per_epoch_10k * 10:.1f} hours")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class DummyDataset(Dataset):
    """Dummy dataset for benchmarking."""
    
    def __init__(self, size: int, seq_length: int, vocab_size: int):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def get_gpu_memory_stats() -> Tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return current, peak
    return 0.0, 0.0


def profile_model_memory(model: nn.Module) -> Dict[str, float]:
    """Profile model memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate sizes in GB (assuming fp16/bf16)
    model_size = (total_params * 2) / 1024**3  # 2 bytes per param
    gradient_size = (trainable_params * 2) / 1024**3
    optimizer_size = gradient_size * 2  # Adam has 2 momentum buffers
    
    return {
        "model_size_gb": model_size,
        "gradient_size_gb": gradient_size,
        "optimizer_state_size_gb": optimizer_size,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
    }


def run_benchmark(args: argparse.Namespace) -> BenchmarkMetrics:
    """Run the benchmark and collect metrics."""
    
    print(f"Starting micro-benchmark with {args.num_steps} steps...")
    
    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    
    # Initialize model
    print(f"Loading model: {args.model_name}...")
    if args.use_dummy_model:
        # Create a dummy transformer model for testing
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("gpt2")
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto" if args.num_gpus > 1 else device,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        vocab_size = len(tokenizer)
    
    # Apply LoRA if specified
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Move model to device
    if args.num_gpus == 1 and not args.use_dummy_model:
        model = model.to(device)
    
    # Get model memory profile
    memory_profile = profile_model_memory(model)
    
    # Create dummy dataset and dataloader
    dataset = DummyDataset(
        size=args.num_steps * args.batch_size * args.gradient_accumulation_steps * 2,
        seq_length=args.sequence_length,
        vocab_size=vocab_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=args.num_steps,
    )
    
    # Metrics collection
    step_times = []
    memory_usage = []
    data_loading_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Training loop
    model.train()
    data_iter = iter(dataloader)
    total_start_time = time.time()
    
    for step in range(args.num_steps):
        step_start_time = time.time()
        
        # Data loading
        data_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        if device.type == "cuda":
            batch = {k: v.to(device) for k, v in batch.items()}
        data_loading_times.append(time.time() - data_start)
        
        # Forward pass
        forward_start = time.time()
        with torch.cuda.amp.autocast(dtype=dtype) if device.type == "cuda" else torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
        forward_times.append(time.time() - forward_start)
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_times.append(time.time() - backward_start)
        
        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer_start = time.time()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_times.append(time.time() - optimizer_start)
        else:
            optimizer_times.append(0.0)
        
        # Record metrics
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        current_mem, peak_mem = get_gpu_memory_stats()
        memory_usage.append(current_mem)
        
        # Progress reporting
        if (step + 1) % 10 == 0:
            avg_step_time = sum(step_times[-10:]) / len(step_times[-10:])
            print(f"Step {step + 1}/{args.num_steps} | "
                  f"Time: {avg_step_time:.3f}s | "
                  f"Memory: {current_mem:.2f}GB | "
                  f"Loss: {loss.item():.4f}")
    
    total_time = time.time() - total_start_time
    
    # Calculate metrics
    avg_step_time = sum(step_times) / len(step_times)
    total_data_time = sum(data_loading_times)
    total_forward_time = sum(forward_times)
    total_backward_time = sum(backward_times)
    total_optimizer_time = sum(optimizer_times)
    
    # Get GPU info
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    else:
        gpu_model = "CPU"
        cuda_version = "N/A"
    
    # Calculate throughput
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    samples_per_second = (args.num_steps * effective_batch_size) / total_time
    tokens_per_second = samples_per_second * args.sequence_length
    
    # Estimate TFLOPS (rough approximation for transformer models)
    # Formula: 2 * params * seqlen * batch_size * 3 (for forward, backward, and gradient)
    if "total_params_millions" in memory_profile:
        flops_per_step = (2 * memory_profile["total_params_millions"] * 1e6 * 
                         args.sequence_length * effective_batch_size * 3)
        tflops = (flops_per_step * args.num_steps) / (total_time * 1e12)
    else:
        tflops = 0.0
    
    # Get peak memory after training
    _, peak_vram = get_gpu_memory_stats()
    
    # Calculate activation memory (peak - model - gradient - optimizer)
    activation_memory = peak_vram - (
        memory_profile["model_size_gb"] +
        memory_profile["gradient_size_gb"] +
        memory_profile["optimizer_state_size_gb"]
    )
    
    # Create metrics object
    metrics = BenchmarkMetrics(
        total_time_seconds=total_time,
        avg_step_time_seconds=avg_step_time,
        min_step_time_seconds=min(step_times),
        max_step_time_seconds=max(step_times),
        data_loading_time_ratio=total_data_time / total_time,
        forward_time_ratio=total_forward_time / total_time,
        backward_time_ratio=total_backward_time / total_time,
        optimizer_time_ratio=total_optimizer_time / total_time,
        peak_vram_gb=peak_vram,
        avg_vram_gb=sum(memory_usage) / len(memory_usage),
        model_size_gb=memory_profile["model_size_gb"],
        gradient_size_gb=memory_profile["gradient_size_gb"],
        optimizer_state_size_gb=memory_profile["optimizer_state_size_gb"],
        activation_memory_gb=max(0, activation_memory),
        samples_per_second=samples_per_second,
        tokens_per_second=tokens_per_second,
        tflops=tflops,
        num_gpus=args.num_gpus,
        gpu_model=gpu_model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps=args.num_steps,
        model_name=args.model_name,
        precision=args.precision,
        cuda_version=cuda_version,
        pytorch_version=torch.__version__,
        transformers_version="4.36.0",  # Update based on actual version
    )
    
    return metrics


def update_compute_budget(metrics: BenchmarkMetrics, budget_file: Path):
    """Update the COMPUTE_BUDGET.md file with benchmark results."""
    
    print("\nUpdating COMPUTE_BUDGET.md with benchmark results...")
    
    # Read existing file
    with open(budget_file, 'r') as f:
        content = f.read()
    
    # Find the micro-benchmark section
    marker = "## Micro-Benchmark Validation"
    if marker not in content:
        print("Warning: Could not find micro-benchmark section in COMPUTE_BUDGET.md")
        return
    
    # Generate update content
    update = []
    update.append("")
    update.append(f"**Status**: COMPLETED ({datetime.now().date()})")
    update.append("")
    update.append("### Measured Performance Metrics")
    update.append("")
    update.append("#### Hardware Configuration")
    update.append(f"- **GPU Model**: {metrics.gpu_model}")
    update.append(f"- **Number of GPUs**: {metrics.num_gpus}")
    update.append(f"- **Precision**: {metrics.precision}")
    update.append("")
    update.append("#### Measured Timings")
    update.append(f"- **Average Step Time**: {metrics.avg_step_time_seconds:.3f} seconds")
    update.append(f"- **Data Loading**: {metrics.data_loading_time_ratio:.1%} of total time")
    update.append(f"- **Forward Pass**: {metrics.forward_time_ratio:.1%} of total time")
    update.append(f"- **Backward Pass**: {metrics.backward_time_ratio:.1%} of total time")
    update.append(f"- **Optimizer Step**: {metrics.optimizer_time_ratio:.1%} of total time")
    update.append("")
    update.append("#### Measured Memory Usage")
    update.append(f"- **Peak VRAM**: {metrics.peak_vram_gb:.2f} GB")
    update.append(f"- **Average VRAM**: {metrics.avg_vram_gb:.2f} GB")
    update.append(f"- **Model Size**: {metrics.model_size_gb:.2f} GB")
    update.append(f"- **Gradient Memory**: {metrics.gradient_size_gb:.2f} GB")
    update.append(f"- **Optimizer State**: {metrics.optimizer_state_size_gb:.2f} GB")
    update.append(f"- **Activation Memory**: {metrics.activation_memory_gb:.2f} GB")
    update.append("")
    update.append("#### Throughput Metrics")
    update.append(f"- **Samples/Second**: {metrics.samples_per_second:.2f}")
    update.append(f"- **Tokens/Second**: {metrics.tokens_per_second:.0f}")
    update.append("")
    update.append("### Revised Projections Based on Measurements")
    update.append("")
    
    # Calculate revised estimates
    hours_per_10k_steps = (10000 * metrics.avg_step_time_seconds) / 3600
    
    update.append("#### Updated SFT Estimates")
    update.append(f"- **10,000 steps**: {hours_per_10k_steps:.1f} hours per GPU")
    update.append(f"- **3 epochs (30,000 steps)**: {hours_per_10k_steps * 3:.1f} hours per GPU")
    update.append(f"- **8 GPU scaling efficiency (estimated 85%)**: {(hours_per_10k_steps * 3) / (8 * 0.85):.1f} hours")
    update.append("")
    
    update.append("#### Updated RFT Estimates")
    # RFT is approximately 5-8x more expensive due to generation
    rft_multiplier = 6.5  # Mid-point estimate
    update.append(f"- **Generation overhead multiplier**: {rft_multiplier}x")
    update.append(f"- **5,000 RL steps**: {(5000 * metrics.avg_step_time_seconds * rft_multiplier) / 3600:.1f} hours per GPU")
    update.append(f"- **8 GPU scaling**: {(5000 * metrics.avg_step_time_seconds * rft_multiplier) / (3600 * 8 * 0.85):.1f} hours")
    update.append("")
    
    update.append("#### I/O Analysis")
    if metrics.data_loading_time_ratio > 0.1:
        update.append(f"⚠️ **I/O Bottleneck Detected**: Data loading takes {metrics.data_loading_time_ratio:.1%} of total time")
        update.append("  - Consider increasing number of data loader workers")
        update.append("  - Ensure data is on fast SSD storage")
        update.append("  - Consider data prefetching strategies")
    else:
        update.append(f"✓ **No I/O Bottleneck**: Data loading is only {metrics.data_loading_time_ratio:.1%} of total time")
    update.append("")
    
    # Join the update content
    update_text = "\n".join(update)
    
    # Replace the pending section
    before_marker = content[:content.index(marker)]
    after_marker = content[content.index(marker):]
    
    # Find the end of the micro-benchmark section (next ## or end of file)
    import re
    next_section = re.search(r'\n##[^#]', after_marker[1:])
    if next_section:
        section_end = next_section.start() + 1
        after_section = after_marker[section_end:]
        new_content = before_marker + marker + update_text + "\n" + after_section
    else:
        new_content = before_marker + marker + update_text
    
    # Write updated content
    with open(budget_file, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated {budget_file}")


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmark for computational cost estimation")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="qwen/Qwen2.5-VL-7B",
                       help="Model name or path")
    parser.add_argument("--use_dummy_model", action="store_true",
                       help="Use dummy model for testing without downloading")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16",
                       help="Training precision")
    
    # Training configuration
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--sequence_length", type=int, default=2048,
                       help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    
    # Optimization configuration
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    
    # Hardware configuration
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--update_budget", action="store_true",
                       help="Update COMPUTE_BUDGET.md with results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run benchmark
    try:
        metrics = run_benchmark(args)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON metrics
        json_file = output_dir / f"benchmark_{timestamp}.json"
        metrics.to_json(json_file)
        print(f"\n✓ Saved metrics to {json_file}")
        
        # Generate and save report
        report = metrics.generate_report()
        report_file = output_dir / f"benchmark_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Saved report to {report_file}")
        
        # Print report to console
        print("\n" + report)
        
        # Update COMPUTE_BUDGET.md if requested
        if args.update_budget:
            budget_file = Path(__file__).parent.parent / "COMPUTE_BUDGET.md"
            if budget_file.exists():
                update_compute_budget(metrics, budget_file)
            else:
                print(f"Warning: {budget_file} not found")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())