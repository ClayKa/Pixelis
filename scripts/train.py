#!/usr/bin/env python3
"""
Main training script for Pixelis with comprehensive reproducibility tracking.
Supports SFT, RFT, and online TTRL modes.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.reproducibility import (
    ArtifactManager,
    ArtifactType,
    ExperimentContext,
    TTRLContext,
    EnvironmentCaptureLevel,
    track_artifacts,
    reproducible,
)
from core.config_schema import PixelisConfig as TrainingConfig
from core.utils.logging_utils import setup_logging, get_logger

# Import SFT training module
try:
    from train_sft import (
        CurriculumDataset,
        CurriculumManager,
        CurriculumCallback,
        load_model_with_lora,
        run_sft_training,
    )
except ImportError:
    # If running from different directory
    from scripts.train_sft import (
        CurriculumDataset,
        CurriculumManager,
        CurriculumCallback,
        load_model_with_lora,
        run_sft_training,
    )

# Setup logging
setup_logging()
logger = get_logger(__name__)


@track_artifacts(inputs=["dataset"], outputs=["model", "metrics"])
def run_sft(config: Dict[str, Any], artifact_manager: ArtifactManager):
    """
    Run Supervised Fine-Tuning (SFT) with curriculum learning and artifact tracking.
    
    Args:
        config: Training configuration dictionary
        artifact_manager: Artifact manager instance
    
    Returns:
        Tuple of (model_path, metrics)
    """
    logger.info("Starting SFT training with curriculum learning...")
    
    # Load configuration from files if needed
    if isinstance(config, TrainingConfig):
        # Convert to dictionary if it's a dataclass
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    else:
        config_dict = config
    
    # Ensure we have both training and model configs
    if "model" not in config_dict:
        # Load model configuration
        model_config_path = Path("configs/model_arch.yaml")
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
                config_dict.update(model_config)
    
    # Load curriculum configuration if not present
    if "curriculum" not in config_dict:
        training_params_path = Path("configs/training_params.yaml")
        if training_params_path.exists():
            with open(training_params_path, 'r') as f:
                training_params = yaml.safe_load(f)
                config_dict.update(training_params)
    
    # Initialize wandb if configured
    import wandb
    if wandb.run is None and "wandb" in config_dict.get("training", {}).get("report_to", []):
        wandb.init(
            project="pixelis-sft",
            name=f"sft_curriculum_{wandb.util.generate_id()}",
            config=config_dict,
            tags=["sft", "curriculum"],
        )
    
    # Load model and tokenizer with LoRA
    logger.info("Loading model with LoRA configuration...")
    model, tokenizer = load_model_with_lora(config_dict)
    
    # Log model configuration
    if artifact_manager:
        artifact_manager.log_artifact(
            name="sft_model_config",
            type=ArtifactType.CONFIG,
            data={
                "model_name": config_dict.get("model", {}).get("model_name"),
                "lora_config": config_dict.get("model", {}).get("lora_target_modules"),
                "gradient_checkpointing": config_dict.get("model", {}).get("gradient_checkpointing"),
            },
        )
    
    # Create curriculum dataset
    curriculum_config = config_dict.get("curriculum", {})
    data_path = curriculum_config.get("data_path", "data/processed/curriculum")
    
    logger.info(f"Loading curriculum dataset from {data_path}")
    train_dataset = CurriculumDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config_dict.get("model", {}).get("max_length", 4096),
        use_split_files=curriculum_config.get("use_split_files", True),
        initial_stage="simple"
    )
    
    # Log dataset statistics
    dataset_stats = train_dataset.get_statistics()
    logger.info(f"Dataset statistics: {dataset_stats}")
    
    if artifact_manager:
        artifact_manager.log_artifact(
            name="sft_dataset_stats",
            type=ArtifactType.METRICS,
            data=dataset_stats,
        )
    
    # Set output directory
    output_dir = config_dict.get("training", {}).get("output_dir", "./outputs/sft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for resume checkpoint
    resume_from = config_dict.get("training", {}).get("resume_from_checkpoint")
    
    # Run training with curriculum learning
    logger.info("Starting curriculum-based SFT training...")
    model, metrics = run_sft_training(
        config=config_dict,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,  # Could load a separate eval dataset here
        output_dir=output_dir,
        resume_from_checkpoint=resume_from
    )
    
    # Save final model path
    model_path = Path(output_dir) / "pytorch_model.bin"
    
    # Log final model artifact
    if artifact_manager:
        model_artifact = artifact_manager.log_large_artifact(
            name="sft_model_final",
            file_path=model_path,
            type=ArtifactType.MODEL,
            metadata={
                "training_config": config_dict,
                "final_metrics": metrics,
                "curriculum_stats": train_dataset.get_statistics(),
            },
        )
        logger.info(f"✓ Model artifact logged: {model_artifact.name}:{model_artifact.version}")
    
    # Save training summary
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": config_dict,
            "metrics": metrics,
            "dataset_stats": dataset_stats,
            "model_path": str(model_path),
        }, f, indent=2)
    
    logger.info(f"✓ SFT training complete. Model saved to {model_path}")
    logger.info(f"  Final metrics: {metrics}")
    
    return str(model_path), metrics


@track_artifacts(inputs=["model", "dataset"], outputs=["model", "metrics"])
def run_rft(config: Dict[str, Any], artifact_manager: ArtifactManager):
    """
    Run Reinforcement Fine-Tuning (RFT) with GRPO and artifact tracking.
    
    Args:
        config: Training configuration dictionary
        artifact_manager: Artifact manager instance
    
    Returns:
        Tuple of (model_path, metrics)
    """
    logger.info("Starting RFT training with GRPO...")
    
    # Import RFT training module
    try:
        from train_rft import run_rft_training
    except ImportError:
        from scripts.train_rft import run_rft_training
    
    # Load configuration from files if needed
    if isinstance(config, TrainingConfig):
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    else:
        config_dict = config
    
    # Ensure we have all necessary configs
    if "reward" not in config_dict:
        training_params_path = Path("configs/training_params.yaml")
        if training_params_path.exists():
            with open(training_params_path, 'r') as f:
                training_params = yaml.safe_load(f)
                config_dict.update(training_params)
    
    # Initialize wandb if configured
    import wandb
    if wandb.run is None and "wandb" in config_dict.get("training", {}).get("report_to", []):
        wandb.init(
            project="pixelis-rft",
            name=f"rft_grpo_{wandb.util.generate_id()}",
            config=config_dict,
            tags=["rft", "grpo"],
        )
    
    # Get SFT model path
    sft_model_path = config_dict.get("sft_model_path", "outputs/sft/final")
    if not Path(sft_model_path).exists():
        # Try to find the most recent SFT checkpoint
        sft_output_dir = Path("outputs/sft")
        if sft_output_dir.exists():
            checkpoints = list(sft_output_dir.glob("checkpoint-*"))
            if checkpoints:
                sft_model_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                logger.info(f"Using SFT checkpoint: {sft_model_path}")
            else:
                logger.warning("No SFT checkpoints found, will use base model")
                sft_model_path = None
    
    # Log configuration
    if artifact_manager:
        artifact_manager.log_artifact(
            name="rft_config",
            type=ArtifactType.CONFIG,
            data={
                "sft_model": sft_model_path,
                "reward_config": config_dict.get("reward", {}),
                "grpo_config": config_dict.get("grpo", {}),
            },
        )
    
    # Set output directory
    output_dir = config_dict.get("training", {}).get("output_dir", "./outputs/rft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run RFT training
    logger.info("Starting GRPO-powered RFT training...")
    final_model_path = run_rft_training(
        config=config_dict,
        sft_model_path=sft_model_path or "",
        output_dir=output_dir,
        resume_from_checkpoint=config_dict.get("training", {}).get("resume_from_checkpoint")
    )
    
    # Collect final metrics
    metrics = {
        "final_model_path": final_model_path,
        "training_complete": True,
        "grpo_enabled": True,
        "reward_components": ["task", "curiosity", "coherence", "penalty"],
    }
    
    # Log final model artifact
    if artifact_manager and Path(final_model_path).exists():
        model_artifact = artifact_manager.log_large_artifact(
            name="rft_model_final",
            file_path=Path(final_model_path),
            type=ArtifactType.MODEL,
            metadata={
                "training_config": config_dict,
                "base_model": sft_model_path,
                "final_metrics": metrics,
                "grpo_config": config_dict.get("grpo", {}),
            },
        )
        logger.info(f"✓ Model artifact logged: {model_artifact.name}:{model_artifact.version}")
    
    # Save training summary
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "config": config_dict,
            "metrics": metrics,
            "model_path": str(final_model_path),
            "sft_base": sft_model_path,
        }, f, indent=2)
    
    logger.info(f"✓ RFT training complete. Model saved to {final_model_path}")
    
    return str(final_model_path), metrics


def run_ttrl(config: TrainingConfig, artifact_manager: ArtifactManager):
    """
    Run Test-Time Reinforcement Learning (TTRL) with specialized context.
    
    Args:
        config: Training configuration
        artifact_manager: Artifact manager instance
    
    Returns:
        Tuple of (model_path, metrics)
    """
    logger.info("Starting TTRL online learning...")
    
    # Use specialized TTRL context
    with TTRLContext(
        config=config,
        name="ttrl_online",
        experience_snapshot_interval=config.online.snapshot_interval,
        capture_level=EnvironmentCaptureLevel.COMPLETE,
    ) as ctx:
        
        # Load base model
        model_artifact = ctx.artifact_manager.use_artifact(
            name=config.model.artifact_name or "rft_model_final",
            version=config.model.artifact_version or "latest",
        )
        logger.info(f"Using model: {model_artifact.name}:{model_artifact.version}")
        
        # TODO: Implement actual TTRL online learning logic
        # This is a placeholder implementation
        
        import time
        import random
        
        # Simulate online learning
        experience_buffer = []
        metrics = {}
        
        for step in range(config.online.num_steps):
            # Simulate experience
            time.sleep(0.05)
            
            experience = {
                "step": step + 1,
                "input": f"query_{step}",
                "output": f"response_{step}",
                "reward": random.uniform(-1, 1),
                "confidence": random.uniform(0.5, 1.0),
            }
            
            experience_buffer.append(experience)
            
            # Log experience
            ctx.log_experience(
                experience_id=f"exp_{step}",
                input_data=experience["input"],
                output_data=experience["output"],
                metadata={
                    "reward": experience["reward"],
                    "confidence": experience["confidence"],
                },
            )
            
            # Periodically log experience buffer
            if (step + 1) % 100 == 0:
                ctx.log_experience_buffer(experience_buffer)
                logger.info(f"Step {step + 1}: Buffer size={len(experience_buffer)}")
            
            # Simulate online update
            if experience["confidence"] > 0.85:
                ctx.log_online_update(
                    experience_id=f"exp_{step}",
                    reward=experience["reward"],
                    confidence=experience["confidence"],
                    kl_divergence=random.uniform(0.01, 0.05),
                )
            
            metrics[f"step_{step + 1}"] = experience
        
        # Save final model
        model_path = Path("checkpoints") / "ttrl_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, "w") as f:
            f.write("# Placeholder TTRL model file\n")
        
        # Log final model
        final_model = ctx.log_artifact(
            name="ttrl_model_final",
            type=ArtifactType.MODEL,
            file_path=model_path,
            metadata={
                "training_config": config.to_dict(),
                "base_model": f"{model_artifact.name}:{model_artifact.version}",
                "total_experiences": len(experience_buffer),
                "total_updates": ctx.update_count,
            },
        )
        
        logger.info(
            f"✓ TTRL online learning complete. "
            f"Model saved: {final_model.name}:{final_model.version}"
        )
        
        return str(model_path), metrics


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Pixelis Training Script")
    
    # Training mode
    parser.add_argument(
        "--mode",
        choices=["sft", "rft", "ttrl"],
        required=True,
        help="Training mode: sft (supervised), rft (reinforcement), ttrl (online)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_params.yaml",
        help="Path to training configuration file",
    )
    
    # Experiment settings
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="pixelis",
        help="WandB project name",
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Experiment tags",
    )
    
    # Environment capture
    parser.add_argument(
        "--capture-level",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Environment capture level: 1=basic, 2=standard, 3=complete",
    )
    
    # Offline mode
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no WandB)",
    )
    
    # Hardware monitoring
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable hardware monitoring",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    
    # Load the actual configuration from YAML files
    config_dict = {}
    
    # Load training parameters
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            training_params = yaml.safe_load(f)
            config_dict.update(training_params)
    
    # Load model architecture config
    model_config_path = Path("configs/model_arch.yaml")
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            config_dict.update(model_config)
    
    # For compatibility with reproducibility framework, create a TrainingConfig instance
    # but use the dictionary for actual training
    config = TrainingConfig()
    
    # Set offline mode
    if args.offline:
        os.environ["PIXELIS_OFFLINE_MODE"] = "true"
    
    # Create experiment context
    capture_level = EnvironmentCaptureLevel(args.capture_level)
    
    with ExperimentContext(
        config=config,
        name=args.exp_name or f"{args.mode}_experiment",
        project=args.project,
        tags=args.tags or [args.mode],
        capture_level=capture_level,
        monitor_hardware=not args.no_monitor,
        offline_mode=args.offline,
    ) as ctx:
        
        # Log training mode
        ctx.log_artifact(
            name="training_mode",
            type=ArtifactType.CONFIG,
            data={"mode": args.mode},
        )
        
        # Run appropriate training mode
        if args.mode == "sft":
            # Pass dictionary config for SFT
            model_path, metrics = run_sft(config_dict, ctx.artifact_manager)
        elif args.mode == "rft":
            # Pass dictionary config for RFT
            model_path, metrics = run_rft(config_dict, ctx.artifact_manager)
        elif args.mode == "ttrl":
            model_path, metrics = run_ttrl(config, ctx.artifact_manager)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Log final results
        ctx.log_artifact(
            name="training_results",
            type=ArtifactType.METRICS,
            data={
                "mode": args.mode,
                "model_path": str(model_path),
                "final_metrics": metrics,
            },
        )
        
        logger.info(f"✓ Training complete in {args.mode} mode")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Artifacts logged: {len(ctx.artifacts_logged)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())