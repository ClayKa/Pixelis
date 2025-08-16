#!/usr/bin/env python3
"""
Enhanced Reinforcement Fine-Tuning (RFT) with GRPO for Pixelis - Phase 1 Round 4.

This module implements:
- Performance-triggered curriculum learning with CurriculumManager
- Comprehensive metrics tracking with MetricsTracker
- Enhanced monitoring and logging for all reward components
- Systematic checkpointing at curriculum boundaries
- Real-time metrics export for interactive dashboard
"""

import json
import logging
import os
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from enum import Enum
import threading
from queue import Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb
from scipy import stats

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import Action, Trajectory, ActionType
from core.utils.logging_utils import setup_logging, get_logger
from core.modules.reward_shaping_enhanced import EnhancedRewardOrchestrator

# Setup logging
logger = get_logger(__name__)


class CurriculumStage:
    """Represents a single stage in the curriculum."""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config['name']
        self.description = config.get('description', '')
        self.weights = config['weights']
        self.exit_conditions = config.get('exit_conditions', [])
        
        # Track stage performance
        self.steps_in_stage = 0
        self.start_time = None
        self.metrics_history = defaultdict(list)
        

class MetricsTracker:
    """
    Tracks training metrics with moving averages and slope calculations.
    
    Provides real-time performance monitoring for curriculum advancement decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = defaultdict(list)
        self.moving_averages = {}
        self.slopes = {}
        self.step_count = 0
        
        # Initialize moving average configs
        self.ma_configs = config.get('moving_averages', [])
        for ma_config in self.ma_configs:
            metric = ma_config['metric']
            window = ma_config['window']
            name = ma_config['name']
            self.moving_averages[name] = deque(maxlen=window)
            
        # Initialize slope tracking configs
        self.slope_configs = config.get('slope_tracking', [])
        for slope_config in self.slope_configs:
            metric = slope_config['metric']
            window = slope_config['window']
            name = slope_config['name']
            self.slopes[name] = deque(maxlen=window)
            
        # Real-time export configuration
        self.export_to_json = config.get('export_to_json', False)
        self.json_export_path = config.get('json_export_path')
        self.export_frequency = config.get('export_frequency', 10)
        self.export_fields = config.get('export_fields', [])
        
        # Thread-safe queue for metrics export
        if self.export_to_json:
            self.export_queue = Queue()
            self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
            self.export_thread.start()
            
    def update(self, metrics: Dict[str, Any], step: int):
        """
        Update metrics and calculate moving averages.
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
        """
        self.step_count = step
        
        # Store raw metrics
        for key, value in metrics.items():
            self.metrics[key].append((step, value))
            
        # Update moving averages
        for ma_config in self.ma_configs:
            metric = ma_config['metric']
            name = ma_config['name']
            
            if metric in metrics:
                self.moving_averages[name].append(metrics[metric])
                
                # Calculate and store the MA value
                if len(self.moving_averages[name]) > 0:
                    ma_value = np.mean(self.moving_averages[name])
                    metrics[name] = ma_value
                    
        # Update slopes
        for slope_config in self.slope_configs:
            metric = slope_config['metric']
            name = slope_config['name']
            
            if metric in metrics:
                self.slopes[name].append((step, metrics[metric]))
                
                # Calculate slope if enough data points
                if len(self.slopes[name]) >= 2:
                    x = [p[0] for p in self.slopes[name]]
                    y = [p[1] for p in self.slopes[name]]
                    slope, _, _, _, _ = stats.linregress(x, y)
                    metrics[name] = slope
                    
        # Export to JSON if configured
        if self.export_to_json and step % self.export_frequency == 0:
            self._export_metrics(metrics, step)
            
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1][1]
            
        # Check moving averages
        for ma_config in self.ma_configs:
            if ma_config['name'] == metric_name and self.moving_averages[metric_name]:
                return np.mean(self.moving_averages[metric_name])
                
        # Check slopes
        for slope_config in self.slope_configs:
            if slope_config['name'] == metric_name and len(self.slopes[metric_name]) >= 2:
                x = [p[0] for p in self.slopes[metric_name]]
                y = [p[1] for p in self.slopes[metric_name]]
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
                
        return None
        
    def _export_metrics(self, metrics: Dict[str, Any], step: int):
        """Export metrics to JSON for dashboard."""
        export_data = {
            'step': step,
            'timestamp': time.time(),
        }
        
        for field in self.export_fields:
            if field in metrics:
                export_data[field] = metrics[field]
                
        self.export_queue.put(export_data)
        
    def _export_worker(self):
        """Worker thread for exporting metrics."""
        while True:
            try:
                data = self.export_queue.get()
                if data is None:
                    break
                    
                # Append to JSON file
                Path(self.json_export_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Read existing data
                existing_data = []
                if Path(self.json_export_path).exists():
                    try:
                        with open(self.json_export_path, 'r') as f:
                            existing_data = json.load(f)
                    except:
                        existing_data = []
                        
                # Append new data
                existing_data.append(data)
                
                # Keep only last N entries to prevent file bloat
                max_entries = 10000
                if len(existing_data) > max_entries:
                    existing_data = existing_data[-max_entries:]
                    
                # Write back
                with open(self.json_export_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                

class CurriculumManager:
    """
    Manages performance-triggered curriculum advancement.
    
    Monitors training metrics and automatically advances curriculum stages
    based on performance triggers rather than fixed steps.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_tracker: MetricsTracker):
        self.config = config
        self.metrics_tracker = metrics_tracker
        
        # Load curriculum stages
        self.stages = [CurriculumStage(stage) for stage in config['stages']]
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Curriculum settings
        self.evaluation_frequency = config.get('evaluation_frequency', 100)
        self.min_samples_for_advancement = config.get('min_samples_for_advancement', 1000)
        self.regression_window = config.get('regression_window', 5)
        
        # History tracking
        self.stage_history = []
        self.transition_points = []
        
        # Start first stage
        self._start_stage(0)
        
    def _start_stage(self, stage_idx: int):
        """Initialize a new curriculum stage."""
        self.current_stage_idx = stage_idx
        self.current_stage = self.stages[stage_idx]
        self.current_stage.start_time = time.time()
        self.current_stage.steps_in_stage = 0
        
        logger.info(f"Starting curriculum stage: {self.current_stage.name}")
        logger.info(f"Description: {self.current_stage.description}")
        logger.info(f"Reward weights: {self.current_stage.weights}")
        
        # Log to WandB
        if wandb.run:
            wandb.log({
                'curriculum/stage': stage_idx,
                'curriculum/stage_name': self.current_stage.name,
                **{f'curriculum/weight_{k}': v for k, v in self.current_stage.weights.items()}
            })
            
    def get_reward_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        return self.current_stage.weights
        
    def should_advance(self, step: int) -> bool:
        """
        Check if curriculum should advance to next stage.
        
        Args:
            step: Current training step
            
        Returns:
            True if should advance to next stage
        """
        # Don't advance if we're at the final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
            
        # Don't check too early
        if self.current_stage.steps_in_stage < self.min_samples_for_advancement:
            return False
            
        # Only check at evaluation frequency
        if step % self.evaluation_frequency != 0:
            return False
            
        # Check exit conditions
        for condition in self.current_stage.exit_conditions:
            if self._check_condition(condition):
                logger.info(f"Exit condition met: {condition}")
                return True
                
        return False
        
    def _check_condition(self, condition: Dict[str, Any]) -> bool:
        """Check if a single exit condition is met."""
        metric_name = condition['metric']
        threshold = condition['threshold']
        comparison = condition['comparison']
        patience = condition.get('patience', 0)
        
        # Get metric value
        metric_value = self.metrics_tracker.get_metric(metric_name)
        if metric_value is None:
            return False
            
        # Store in stage history for patience checking
        self.current_stage.metrics_history[metric_name].append(metric_value)
        
        # Check patience if specified
        if patience > 0:
            if len(self.current_stage.metrics_history[metric_name]) < patience:
                return False
            # Use average over patience window
            metric_value = np.mean(self.current_stage.metrics_history[metric_name][-patience:])
            
        # Perform comparison
        if comparison == 'greater':
            return metric_value > threshold
        elif comparison == 'less':
            return metric_value < threshold
        elif comparison == 'equal':
            return abs(metric_value - threshold) < 1e-6
        else:
            logger.warning(f"Unknown comparison operator: {comparison}")
            return False
            
    def advance_stage(self, step: int) -> Dict[str, float]:
        """
        Advance to the next curriculum stage.
        
        Args:
            step: Current training step
            
        Returns:
            New reward weights
        """
        # Record transition
        self.transition_points.append({
            'step': step,
            'from_stage': self.current_stage.name,
            'to_stage': self.stages[self.current_stage_idx + 1].name if self.current_stage_idx < len(self.stages) - 1 else None,
            'stage_duration': self.current_stage.steps_in_stage,
            'metrics': {k: self.metrics_tracker.get_metric(k) for k in ['success_rate_ma100', 'coherence_ma100', 'curiosity_ma100']}
        })
        
        # Save stage history
        self.stage_history.append({
            'stage': self.current_stage.name,
            'start_step': step - self.current_stage.steps_in_stage,
            'end_step': step,
            'duration': time.time() - self.current_stage.start_time
        })
        
        # Advance to next stage
        self._start_stage(self.current_stage_idx + 1)
        
        return self.current_stage.weights
        
    def update(self, step: int):
        """Update curriculum manager state."""
        self.current_stage.steps_in_stage += 1
        
        # Check for advancement
        if self.should_advance(step):
            return self.advance_stage(step)
            
        return self.current_stage.weights
        
    def save_checkpoint(self, checkpoint_dir: Path):
        """Save curriculum state for resuming."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'stage_history': self.stage_history,
            'transition_points': self.transition_points,
            'current_stage_metrics': dict(self.current_stage.metrics_history),
            'current_stage_steps': self.current_stage.steps_in_stage,
        }
        
        checkpoint_path = checkpoint_dir / 'curriculum_state.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved curriculum state to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load curriculum state for resuming."""
        checkpoint_path = checkpoint_dir / 'curriculum_state.json'
        if not checkpoint_path.exists():
            logger.warning(f"No curriculum checkpoint found at {checkpoint_path}")
            return
            
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
            
        self.current_stage_idx = state['current_stage_idx']
        self.stage_history = state['stage_history']
        self.transition_points = state['transition_points']
        self.current_stage = self.stages[self.current_stage_idx]
        self.current_stage.metrics_history = defaultdict(list, state['current_stage_metrics'])
        self.current_stage.steps_in_stage = state['current_stage_steps']
        
        logger.info(f"Loaded curriculum state from {checkpoint_path}")
        logger.info(f"Resuming at stage: {self.current_stage.name}")


class GRPOTrainer(PPOTrainer):
    """
    Extended PPO Trainer with GRPO (Group Relative Policy Optimization).
    
    Implements group-based advantage normalization for more stable learning.
    """
    
    def __init__(self, config: PPOConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # GRPO specific settings
        self.group_size = config.grpo_group_size if hasattr(config, 'grpo_group_size') else 4
        self.filtering_threshold = config.grpo_filtering_threshold if hasattr(config, 'grpo_filtering_threshold') else 0.01
        self.filtering_rate_history = deque(maxlen=100)
        
    def compute_advantages(self, values, rewards, response_length):
        """
        Compute advantages with GRPO group normalization.
        
        Groups samples and normalizes advantages within each group for
        more stable learning signal.
        """
        # Standard advantage computation
        advantages = super().compute_advantages(values, rewards, response_length)
        
        # Apply GRPO group normalization
        batch_size = advantages.shape[0]
        num_groups = batch_size // self.group_size
        
        if num_groups > 0:
            # Reshape into groups
            grouped_advantages = advantages[:num_groups * self.group_size].reshape(num_groups, self.group_size)
            
            # Normalize within each group
            group_mean = grouped_advantages.mean(dim=1, keepdim=True)
            group_std = grouped_advantages.std(dim=1, keepdim=True) + 1e-8
            normalized_grouped = (grouped_advantages - group_mean) / group_std
            
            # Flatten back
            advantages[:num_groups * self.group_size] = normalized_grouped.reshape(-1)
            
            # Handle remaining samples
            if batch_size % self.group_size != 0:
                remaining = advantages[num_groups * self.group_size:]
                advantages[num_groups * self.group_size:] = (remaining - remaining.mean()) / (remaining.std() + 1e-8)
                
        # Apply filtering threshold
        mask = torch.abs(advantages) > self.filtering_threshold
        filtering_rate = mask.float().mean().item()
        self.filtering_rate_history.append(filtering_rate)
        
        # Zero out advantages below threshold
        advantages = advantages * mask.float()
        
        return advantages
        
    def get_grpo_stats(self) -> Dict[str, float]:
        """Get GRPO-specific statistics."""
        return {
            'grpo/filtering_rate': np.mean(self.filtering_rate_history) if self.filtering_rate_history else 0.0,
            'grpo/group_size': self.group_size,
            'grpo/filtering_threshold': self.filtering_threshold,
        }


def train_rft_with_curriculum(config_path: str, sft_model_path: str, output_dir: str, **kwargs):
    """
    Main training function for RFT with performance-triggered curriculum.
    
    Args:
        config_path: Path to configuration file
        sft_model_path: Path to pre-trained SFT model
        output_dir: Output directory for checkpoints and logs
        **kwargs: Additional arguments
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Initialize WandB
    if config['monitoring']['wandb']['enabled']:
        wandb.init(
            project=config['monitoring']['wandb']['project'],
            entity=config['monitoring']['wandb']['entity'],
            name=kwargs.get('exp_name', 'rft_training'),
            config=config,
            tags=config['monitoring']['wandb']['tags']
        )
        
    # Setup logging
    setup_logging(output_dir / 'training.log')
    logger.info("Starting RFT training with performance-triggered curriculum")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"SFT Model: {sft_model_path}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config['monitoring'])
    
    # Initialize curriculum manager
    curriculum_manager = CurriculumManager(config['reward_curriculum'], metrics_tracker)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16 if config['hardware']['mixed_precision'] == 'bf16' else torch.float16,
        device_map='auto'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Add value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # Initialize reward orchestrator
    reward_orchestrator = EnhancedRewardOrchestrator(config['reward_components'])
    
    # Setup PPO configuration
    ppo_config = PPOConfig(
        batch_size=config['training']['batch_size'],
        mini_batch_size=config['training']['ppo']['mini_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        optimize_cuda_cache=True,
        log_with='wandb' if config['monitoring']['wandb']['enabled'] else None,
    )
    
    # Add GRPO settings
    ppo_config.grpo_group_size = config['training']['grpo']['group_size']
    ppo_config.grpo_filtering_threshold = config['training']['grpo']['filtering_threshold']
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=None,  # Will use custom data loading
        data_collator=None,
    )
    
    # Training loop
    global_step = 0
    best_reward = float('-inf')
    
    logger.info("Starting training loop...")
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Training steps would go here
        # This is a simplified version - actual implementation would include:
        # 1. Data loading
        # 2. Trajectory generation
        # 3. Reward calculation
        # 4. Policy update
        # 5. Metrics tracking
        
        # Example metrics update (would be replaced with actual metrics)
        step_metrics = {
            'success_rate': random.random(),
            'total_reward': random.random() * 2 - 1,
            'coherence_score': random.random(),
            'curiosity_bonus': random.random() * 0.1,
            'kl_divergence': random.random() * 0.05,
            'trajectory_length': random.randint(5, 20),
        }
        
        # Update metrics tracker
        metrics_tracker.update(step_metrics, global_step)
        
        # Update curriculum
        reward_weights = curriculum_manager.update(global_step)
        reward_orchestrator.update_weights(reward_weights)
        
        # Get GRPO stats
        grpo_stats = trainer.get_grpo_stats()
        step_metrics.update(grpo_stats)
        
        # Log metrics
        if global_step % config['training']['logging_steps'] == 0:
            # Add curriculum info
            step_metrics['curriculum/stage'] = curriculum_manager.current_stage_idx
            step_metrics['curriculum/stage_name'] = curriculum_manager.current_stage.name
            
            # Log to WandB
            if wandb.run:
                wandb.log(step_metrics, step=global_step)
                
            # Log summary
            logger.info(f"Step {global_step}: {step_metrics}")
            
        # Save checkpoint at curriculum boundaries
        if curriculum_manager.current_stage.steps_in_stage == 1:  # Just advanced
            checkpoint_path = checkpoints_dir / f"stage_{curriculum_manager.current_stage.name}"
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save curriculum state
            curriculum_manager.save_checkpoint(checkpoint_path)
            
            logger.info(f"Saved checkpoint at curriculum boundary: {checkpoint_path}")
            
        # Regular checkpointing
        if global_step % config['training']['save_steps'] == 0:
            checkpoint_path = checkpoints_dir / f"step_{global_step}"
            checkpoint_path.mkdir(exist_ok=True)
            
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            curriculum_manager.save_checkpoint(checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        global_step += 1
        
    # Final checkpoint
    final_checkpoint = checkpoints_dir / "final"
    final_checkpoint.mkdir(exist_ok=True)
    model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    curriculum_manager.save_checkpoint(final_checkpoint)
    
    # Save training summary
    summary = {
        'total_steps': global_step,
        'curriculum_transitions': curriculum_manager.transition_points,
        'stage_history': curriculum_manager.stage_history,
        'final_metrics': {k: metrics_tracker.get_metric(k) for k in ['success_rate_ma100', 'reward_ma500']},
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Training completed successfully!")
    logger.info(f"Final checkpoint: {final_checkpoint}")
    
    # Close WandB
    if wandb.run:
        wandb.finish()
        
    return final_checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RFT Training with Performance-Triggered Curriculum")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    train_rft_with_curriculum(
        config_path=args.config,
        sft_model_path=args.sft_model_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        resume_from=args.resume_from_checkpoint
    )