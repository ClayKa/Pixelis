#!/usr/bin/env python3
"""
Python-based multi-seed experiment runner for Pixelis
Provides programmatic control over multi-seed experiments
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a multi-seed experiment"""
    experiment_name: str
    mode: str  # 'sft' or 'rft'
    config_file: str
    seeds: List[int] = field(default_factory=lambda: [42, 84, 126])
    output_dir: str = "outputs/experiments"
    wandb_project: str = "pixelis-experiments"
    wandb_tags: List[str] = field(default_factory=list)
    num_gpus: int = 1
    num_nodes: int = 1
    parallel_seeds: bool = False
    max_workers: int = 1
    resume_from_checkpoint: bool = False
    dry_run: bool = False
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        if self.mode not in ['sft', 'rft']:
            raise ValueError(f"Mode must be 'sft' or 'rft', got {self.mode}")
        
        if not Path(self.config_file).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        if self.parallel_seeds and self.max_workers > len(self.seeds):
            self.max_workers = len(self.seeds)


@dataclass
class SeedResult:
    """Result from a single seed run"""
    seed: int
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    output_dir: Optional[str] = None
    wandb_run_id: Optional[str] = None
    error_message: Optional[str] = None
    training_time: Optional[float] = None
    peak_memory_gb: Optional[float] = None


class ExperimentRunner:
    """Manages multi-seed experiment execution"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.registry_file = Path("experiments/registry.json")
        self.results: List[SeedResult] = []
        
        # Create necessary directories
        Path("experiments").mkdir(exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.experiment_name}_{timestamp}"
    
    def _register_experiment(self):
        """Register experiment in registry"""
        # Load existing registry
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
        
        # Add new entry
        entry = {
            "experiment_id": self.experiment_id,
            "name": self.config.experiment_name,
            "date": datetime.now().isoformat(),
            "mode": self.config.mode,
            "config": self.config.config_file,
            "seeds": self.config.seeds,
            "wandb_project": self.config.wandb_project,
            "wandb_runs": [],
            "status": "running",
            "output_dir": f"{self.config.output_dir}/{self.experiment_id}"
        }
        
        registry.append(entry)
        
        # Save registry
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Registered experiment: {self.experiment_id}")
    
    def _update_registry(self, status: str, seed_result: Optional[SeedResult] = None):
        """Update experiment status in registry"""
        if not self.registry_file.exists():
            return
        
        with open(self.registry_file, 'r') as f:
            registry = json.load(f)
        
        for entry in registry:
            if entry["experiment_id"] == self.experiment_id:
                entry["status"] = status
                
                if seed_result and seed_result.wandb_run_id:
                    entry["wandb_runs"].append({
                        "seed": seed_result.seed,
                        "run_id": seed_result.wandb_run_id
                    })
                break
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _build_training_command(self, seed: int, seed_output_dir: str) -> List[str]:
        """Build training command for a single seed"""
        cmd = ["python", "scripts/train.py"]
        
        # Add required arguments
        cmd.extend(["--config", self.config.config_file])
        cmd.extend(["--mode", self.config.mode])
        cmd.extend(["--seed", str(seed)])
        cmd.extend(["--output_dir", seed_output_dir])
        
        # Add WandB configuration
        if WANDB_AVAILABLE:
            cmd.extend(["--wandb_project", self.config.wandb_project])
            cmd.extend(["--wandb_run_name", f"{self.config.experiment_name}_seed{seed}"])
            
            # Add tags
            tags = self.config.wandb_tags + ["multi_seed", f"seed_{seed}"]
            cmd.extend(["--wandb_tags", ",".join(tags)])
        
        # Add experiment ID
        cmd.extend(["--experiment_id", self.experiment_id])
        
        # Add resume flag if specified
        if self.config.resume_from_checkpoint:
            cmd.append("--resume")
        
        # Handle distributed training
        if self.config.num_gpus > 1 or self.config.num_nodes > 1:
            distributed_cmd = [
                "torchrun",
                f"--nproc_per_node={self.config.num_gpus}"
            ]
            
            if self.config.num_nodes > 1:
                distributed_cmd.extend([
                    f"--nnodes={self.config.num_nodes}",
                    "--node_rank=${NODE_RANK}",
                    "--master_addr=${MASTER_ADDR}",
                    "--master_port=${MASTER_PORT}"
                ])
            
            cmd = distributed_cmd + cmd
        
        return cmd
    
    def _run_single_seed(self, seed: int) -> SeedResult:
        """Run training for a single seed"""
        logger.info(f"Starting seed {seed}")
        
        # Create output directory
        seed_output_dir = f"{self.config.output_dir}/{self.experiment_id}/seed_{seed}"
        Path(seed_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = self._build_training_command(seed, seed_output_dir)
        
        # Save command for reference
        with open(f"{seed_output_dir}/command.txt", 'w') as f:
            f.write(" ".join(cmd))
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        if self.config.dry_run:
            logger.info("DRY RUN - Command not executed")
            return SeedResult(seed=seed, success=True)
        
        # Set up environment
        env = os.environ.copy()
        env.update(self.config.env_vars)
        
        # Run training
        start_time = time.time()
        result = SeedResult(seed=seed, success=False, output_dir=seed_output_dir)
        
        try:
            # Run command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                universal_newlines=True
            )
            
            # Stream output to files and console
            log_file = open(f"{seed_output_dir}/training.log", 'w')
            error_file = open(f"{seed_output_dir}/error.log", 'w')
            
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                
                # Parse for WandB run ID
                if "wandb: Run ID:" in line:
                    parts = line.split("wandb: Run ID:")
                    if len(parts) > 1:
                        result.wandb_run_id = parts[1].strip()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Capture any remaining stderr
            stderr = process.stderr.read()
            if stderr:
                error_file.write(stderr)
            
            log_file.close()
            error_file.close()
            
            # Check success
            if return_code == 0:
                result.success = True
                result.training_time = time.time() - start_time
                
                # Load metrics if available
                result.metrics = self._load_seed_metrics(seed_output_dir)
                
                logger.info(f"Seed {seed} completed successfully in {result.training_time:.1f}s")
            else:
                result.success = False
                result.error_message = f"Process exited with code {return_code}"
                logger.error(f"Seed {seed} failed with return code {return_code}")
        
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Seed {seed} failed with exception: {e}")
            traceback.print_exc()
        
        return result
    
    def _load_seed_metrics(self, seed_output_dir: str) -> Dict[str, float]:
        """Load metrics from seed output directory"""
        metrics = {}
        
        # Try to load from standard locations
        locations = [
            "evaluation_results.json",
            "final_metrics.json",
            "metrics.json"
        ]
        
        for location in locations:
            metrics_file = Path(seed_output_dir) / location
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            # Extract numeric metrics
                            for key, value in data.items():
                                if isinstance(value, (int, float)):
                                    metrics[key] = float(value)
                                elif isinstance(value, dict) and "metrics" in value:
                                    # Handle nested metrics
                                    for k, v in value["metrics"].items():
                                        if isinstance(v, (int, float)):
                                            metrics[k] = float(v)
                except Exception as e:
                    logger.warning(f"Error loading metrics from {metrics_file}: {e}")
        
        return metrics
    
    def _run_parallel_seeds(self) -> List[SeedResult]:
        """Run multiple seeds in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all seeds
            future_to_seed = {
                executor.submit(self._run_single_seed, seed): seed
                for seed in self.config.seeds
            }
            
            # Process completed seeds
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update registry
                    self._update_registry("running", result)
                    
                except Exception as e:
                    logger.error(f"Seed {seed} failed with exception: {e}")
                    results.append(
                        SeedResult(seed=seed, success=False, error_message=str(e))
                    )
        
        return results
    
    def _run_sequential_seeds(self) -> List[SeedResult]:
        """Run seeds sequentially"""
        results = []
        
        for i, seed in enumerate(self.config.seeds, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running seed {i}/{len(self.config.seeds)}: {seed}")
            logger.info(f"{'='*60}")
            
            result = self._run_single_seed(seed)
            results.append(result)
            
            # Update registry
            self._update_registry("running", result)
            
            # Ask whether to continue if failed
            if not result.success and not self.config.dry_run:
                if i < len(self.config.seeds):
                    response = input(f"Seed {seed} failed. Continue with remaining seeds? (y/n): ")
                    if response.lower() != 'y':
                        logger.info("Stopping experiment")
                        break
        
        return results
    
    def run(self) -> Tuple[bool, List[SeedResult]]:
        """Run the multi-seed experiment"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Multi-Seed Experiment")
        logger.info(f"{'='*60}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Experiment Name: {self.config.experiment_name}")
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Config: {self.config.config_file}")
        logger.info(f"Seeds: {self.config.seeds}")
        logger.info(f"Output Directory: {self.config.output_dir}/{self.experiment_id}")
        logger.info(f"{'='*60}\n")
        
        # Register experiment
        if not self.config.dry_run:
            self._register_experiment()
        
        # Run seeds
        if self.config.parallel_seeds:
            logger.info(f"Running {len(self.config.seeds)} seeds in parallel (max workers: {self.config.max_workers})")
            self.results = self._run_parallel_seeds()
        else:
            logger.info(f"Running {len(self.config.seeds)} seeds sequentially")
            self.results = self._run_sequential_seeds()
        
        # Determine overall success
        all_success = all(r.success for r in self.results)
        
        # Update final status
        if not self.config.dry_run:
            status = "completed" if all_success else "failed"
            self._update_registry(status)
        
        # Generate summary
        self._generate_summary()
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Successful seeds: {sum(1 for r in self.results if r.success)}/{len(self.results)}")
        
        for result in self.results:
            status = "✓" if result.success else "✗"
            logger.info(f"  Seed {result.seed}: {status}")
            if result.error_message:
                logger.info(f"    Error: {result.error_message}")
        
        logger.info(f"{'='*60}\n")
        
        return all_success, self.results
    
    def _generate_summary(self):
        """Generate experiment summary file"""
        if self.config.dry_run:
            return
        
        summary_file = Path(self.config.output_dir) / self.experiment_id / "experiment_summary.json"
        
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.config.experiment_name,
            "mode": self.config.mode,
            "config": self.config.config_file,
            "seeds": self.config.seeds,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for result in self.results:
            summary["results"].append({
                "seed": result.seed,
                "success": result.success,
                "metrics": result.metrics,
                "output_dir": result.output_dir,
                "wandb_run_id": result.wandb_run_id,
                "error_message": result.error_message,
                "training_time": result.training_time
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")


def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run multi-seed experiments for Pixelis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "-m", "--mode",
        required=True,
        choices=["sft", "rft"],
        help="Training mode"
    )
    
    parser.add_argument(
        "-e", "--exp-name",
        required=True,
        help="Experiment name"
    )
    
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        nargs="+",
        default=[42, 84, 126],
        help="Random seeds to use"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="outputs/experiments",
        help="Base output directory"
    )
    
    parser.add_argument(
        "-w", "--wandb-project",
        default="pixelis-experiments",
        help="WandB project name"
    )
    
    parser.add_argument(
        "-t", "--wandb-tags",
        nargs="+",
        default=[],
        help="WandB tags"
    )
    
    parser.add_argument(
        "-g", "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node"
    )
    
    parser.add_argument(
        "-n", "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run seeds in parallel"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum parallel workers"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--env",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="Environment variables to set"
    )
    
    parser.add_argument(
        "--from-yaml",
        help="Load experiment config from YAML file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.from_yaml:
        # Load from YAML file
        yaml_config = load_yaml_config(args.from_yaml)
        config = ExperimentConfig(**yaml_config)
    else:
        # Build from command line arguments
        env_vars = {}
        if args.env:
            for key, value in args.env:
                env_vars[key] = value
        
        config = ExperimentConfig(
            experiment_name=args.exp_name,
            mode=args.mode,
            config_file=args.config,
            seeds=args.seeds,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_tags=args.wandb_tags,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            parallel_seeds=args.parallel,
            max_workers=args.max_workers,
            resume_from_checkpoint=args.resume,
            dry_run=args.dry_run,
            env_vars=env_vars
        )
    
    # Run experiment
    runner = ExperimentRunner(config)
    success, results = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()