#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) module with curriculum learning for Pixelis.

This module implements:
- CurriculumDataset for progressive difficulty management
- CurriculumManager for curriculum state management
- Custom TrainerCallback for automatic advancement and rollback
- Complete SFT training pipeline with LoRA and gradient checkpointing
"""

import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import Action, Trajectory, ActionType
from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
logger = get_logger(__name__)


class CurriculumDataset(Dataset):
    """
    Dataset class that manages curriculum learning with progressive difficulty.
    
    This dataset starts with simple examples and progressively adds more difficult
    samples based on the curriculum stage.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 4096,
        use_split_files: bool = True,
        initial_stage: str = "simple"
    ):
        """
        Initialize the curriculum dataset.
        
        Args:
            data_path: Path to processed curriculum data
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            use_split_files: Whether to use separate files for each difficulty
            initial_stage: Initial curriculum stage
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_split_files = use_split_files
        
        # Load data by difficulty level
        self.data_by_difficulty = {
            "simple": [],
            "medium": [],
            "hard": []
        }
        
        self._load_data()
        
        # Current active data pool
        self.current_stage = initial_stage
        self.active_data = []
        self.difficulty_weights = {"simple": 1.0, "medium": 0.0, "hard": 0.0}
        
        # Initialize with simple data
        self._update_active_data()
        
        logger.info(f"Initialized CurriculumDataset with {len(self.active_data)} samples")
    
    def _load_data(self):
        """Load data from files."""
        if self.use_split_files:
            # Load from separate difficulty files
            for difficulty in ["simple", "medium", "hard"]:
                file_path = self.data_path / f"cota_{difficulty}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.data_by_difficulty[difficulty] = data["samples"]
                    logger.info(f"Loaded {len(self.data_by_difficulty[difficulty])} {difficulty} samples")
        else:
            # Load from single file with difficulty field
            file_path = self.data_path / "cota_processed.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for sample in data["samples"]:
                        difficulty = sample.get("difficulty", "medium")
                        self.data_by_difficulty[difficulty].append(sample)
                
                for difficulty, samples in self.data_by_difficulty.items():
                    logger.info(f"Loaded {len(samples)} {difficulty} samples")
    
    def _update_active_data(self):
        """Update the active data pool based on current difficulty weights."""
        self.active_data = []
        
        for difficulty, weight in self.difficulty_weights.items():
            if weight > 0:
                samples = self.data_by_difficulty[difficulty]
                num_samples = int(len(samples) * weight)
                if num_samples > 0:
                    selected = random.sample(samples, min(num_samples, len(samples)))
                    self.active_data.extend(selected)
        
        # Shuffle the combined data
        random.shuffle(self.active_data)
        
        logger.info(
            f"Updated active data pool: {len(self.active_data)} samples "
            f"(weights: {self.difficulty_weights})"
        )
    
    def advance_curriculum(self, new_weights: Dict[str, float]):
        """
        Advance to a new curriculum stage with different difficulty weights.
        
        Args:
            new_weights: Dictionary mapping difficulty to weight (0-1)
        """
        old_weights = self.difficulty_weights.copy()
        self.difficulty_weights = new_weights
        self._update_active_data()
        
        logger.info(f"Advanced curriculum: {old_weights} -> {new_weights}")
        
        return old_weights
    
    def rollback_curriculum(self, previous_weights: Dict[str, float]):
        """
        Rollback to previous curriculum weights.
        
        Args:
            previous_weights: Previous difficulty weights to restore
        """
        self.difficulty_weights = previous_weights
        self._update_active_data()
        
        logger.info(f"Rolled back curriculum to: {previous_weights}")
    
    def __len__(self):
        return len(self.active_data)
    
    def __getitem__(self, idx):
        """Get a single training sample."""
        sample = self.active_data[idx]
        
        # Format the sample for training
        text = self._format_sample(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),  # For language modeling
        }
    
    def _format_sample(self, sample: Dict[str, Any]) -> str:
        """
        Format a sample into training text.
        
        Args:
            sample: Raw sample data
            
        Returns:
            Formatted text for training
        """
        # Extract components
        question = sample.get("question", "")
        trajectory = sample.get("trajectory", [])
        answer = sample.get("answer", "")
        
        # Build formatted text
        parts = [f"Question: {question}\n"]
        
        for i, action in enumerate(trajectory):
            if isinstance(action, dict):
                operation = action.get("operation", "")
                arguments = action.get("arguments", {})
                result = action.get("result", "")
                
                parts.append(f"Step {i+1}: {operation}")
                if arguments:
                    parts.append(f"  Arguments: {json.dumps(arguments)}")
                if result:
                    parts.append(f"  Result: {result}")
        
        parts.append(f"\nAnswer: {answer}")
        
        return "\n".join(parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_samples": len(self.active_data),
            "difficulty_weights": self.difficulty_weights,
            "samples_by_difficulty": {
                k: len(v) for k, v in self.data_by_difficulty.items()
            },
            "current_stage": self.current_stage
        }


@dataclass
class CurriculumManager:
    """
    Manages the curriculum learning state and progression logic.
    
    This class tracks performance history, makes advancement decisions,
    and handles automatic rollback when performance drops.
    Enhanced with metric smoothing and patience mechanisms for stability.
    """
    
    config: Dict[str, Any]
    performance_history: deque = field(default_factory=lambda: deque(maxlen=10))
    current_stage_idx: int = 0
    steps_since_advance: int = 0
    steps_since_rollback: int = 0
    rollback_count: int = 0
    advancement_interval: int = 500
    
    def __post_init__(self):
        """Initialize manager settings from config."""
        curriculum_config = self.config.get("curriculum", {})
        self.stages = curriculum_config.get("stages", [])
        self.rollback_threshold = curriculum_config.get("rollback_threshold", -0.05)
        self.rollback_cooldown = curriculum_config.get("rollback_cooldown", 1000)
        self.rollback_factor = curriculum_config.get("rollback_factor", 2.0)
        self.min_performance = curriculum_config.get("min_performance_for_advance", 0.6)
        self.performance_window = curriculum_config.get("performance_window", 3)
        self.base_advancement_interval = curriculum_config.get("advancement_interval", 500)
        self.advancement_interval = self.base_advancement_interval
        
        # Enhanced settings for stability
        self.smoothing_window_size = curriculum_config.get("smoothing_window_size", 3)
        self.patience_cycles = curriculum_config.get("patience_cycles", 2)
        self.cooldown_cycles = curriculum_config.get("cooldown_cycles", 3)
        
        # Initialize enhanced tracking
        self.metric_history = {
            "loss": deque(maxlen=self.smoothing_window_size),
            "accuracy": deque(maxlen=self.smoothing_window_size),
            "perplexity": deque(maxlen=self.smoothing_window_size),
        }
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.advancement_counter = 0
    
    def _get_smoothed_metric(self, metric_name: str) -> Optional[float]:
        """
        Get smoothed value for a metric using moving average.
        
        Args:
            metric_name: Name of the metric to smooth
            
        Returns:
            Smoothed metric value or None if insufficient data
        """
        history = self.metric_history.get(metric_name, deque())
        if not history or len(history) < self.smoothing_window_size:
            return None  # Not enough data to compute smoothed value
        return np.mean(list(history))
    
    def _update_metric_history(self, metrics: Dict[str, float]):
        """
        Update the metric history with new values.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        for name, value in metrics.items():
            if name in self.metric_history:
                self.metric_history[name].append(value)
    
    def _check_advancement_criteria(self) -> bool:
        """
        Check if advancement criteria are met using smoothed metrics.
        
        Returns:
            True if criteria are met, False otherwise
        """
        # Get current stage's exit criteria
        stage = self.current_stage
        exit_criteria = stage.get("exit_criteria", {})
        
        if not exit_criteria:
            # Default criteria: check accuracy
            smoothed_accuracy = self._get_smoothed_metric("accuracy")
            if smoothed_accuracy is None:
                return False
            return smoothed_accuracy >= self.min_performance
        
        # Check all specified criteria
        for metric_name, threshold in exit_criteria.items():
            smoothed_value = self._get_smoothed_metric(metric_name)
            if smoothed_value is None:
                return False
            
            # For loss, lower is better
            if metric_name == "loss":
                if smoothed_value > threshold:
                    return False
            else:
                # For accuracy and other metrics, higher is better
                if smoothed_value < threshold:
                    return False
        
        return True
    
    @property
    def current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage configuration."""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]  # Return last stage if out of bounds
    
    @property
    def current_weights(self) -> Dict[str, float]:
        """Get current difficulty weights."""
        return self.current_stage.get("difficulty_mix", {
            "simple": 1.0, "medium": 0.0, "hard": 0.0
        })
    
    def should_attempt_advance(self, global_step: int) -> bool:
        """
        Check if we should attempt to advance the curriculum.
        Enhanced with cooldown mechanism for stability.
        
        Args:
            global_step: Current training step
            
        Returns:
            Whether to attempt advancement
        """
        # Check if we're in cooldown period (either after advancement or rollback)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Check if we're in cooldown after rollback
        if self.steps_since_rollback < self.rollback_cooldown:
            return False
        
        # Check if enough steps have passed since last advance
        if self.steps_since_advance < self.advancement_interval:
            return False
        
        # Check if we've reached the final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        # Check if current step is within stage bounds
        stage = self.current_stage
        max_steps = stage.get("max_steps")
        if max_steps and global_step >= max_steps:
            return True
        
        # Check if advancement criteria are met using smoothed metrics
        if self._check_advancement_criteria():
            self.patience_counter += 1
            
            # Need sustained performance for patience_cycles
            if self.patience_counter >= self.patience_cycles:
                self.patience_counter = 0
                return True
        else:
            self.patience_counter = 0  # Reset if criteria not met
        
        return False
    
    def decide_and_update(
        self,
        perf_before: float,
        perf_after: float,
        dataset: CurriculumDataset,
        global_step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Decide whether to keep advancement or rollback based on performance.
        Enhanced with metric smoothing for stability.
        
        Args:
            perf_before: Performance before advancement
            perf_after: Performance after advancement
            dataset: The curriculum dataset
            global_step: Current training step
            metrics: Optional dictionary of current metrics
            
        Returns:
            Tuple of (advanced, message)
        """
        # Update metric history if provided
        if metrics:
            self._update_metric_history(metrics)
        
        # Record performance
        self.performance_history.append(perf_after)
        
        # Calculate performance drop
        delta = perf_after - perf_before
        
        # Check for catastrophic drop
        if delta < self.rollback_threshold:
            # Rollback
            self.rollback(dataset)
            message = (
                f"🔙 Rolled back curriculum at step {global_step}. "
                f"Performance drop: {delta:.4f} "
                f"(before: {perf_before:.4f}, after: {perf_after:.4f})"
            )
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    "curriculum/event": "rollback",
                    "curriculum/performance_drop": delta,
                    "curriculum/rollback_count": self.rollback_count,
                }, step=global_step)
            
            return False, message
        
        # Check if performance is good enough to keep advancement
        avg_performance = np.mean(list(self.performance_history)[-self.performance_window:])
        if avg_performance < self.min_performance:
            # Rollback due to low average performance
            self.rollback(dataset)
            message = (
                f"🔙 Rolled back curriculum at step {global_step}. "
                f"Average performance too low: {avg_performance:.4f}"
            )
            
            if wandb.run:
                wandb.log({
                    "curriculum/event": "rollback_low_avg",
                    "curriculum/avg_performance": avg_performance,
                }, step=global_step)
            
            return False, message
        
        # Keep advancement
        message = (
            f"✅ Advanced curriculum at step {global_step}. "
            f"Performance: {perf_after:.4f} (delta: {delta:+.4f})"
        )
        
        if wandb.run:
            wandb.log({
                "curriculum/event": "advance_success",
                "curriculum/performance": perf_after,
                "curriculum/stage": self.current_stage_idx,
            }, step=global_step)
        
        return True, message
    
    def advance(self, dataset: CurriculumDataset) -> Dict[str, float]:
        """
        Advance to the next curriculum stage.
        Enhanced with cooldown mechanism.
        
        Args:
            dataset: The curriculum dataset to update
            
        Returns:
            New difficulty weights
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.steps_since_advance = 0
            self.advancement_counter += 1
            
            # Start cooldown period after advancement
            self.cooldown_counter = self.cooldown_cycles
            
            new_weights = self.current_weights
            old_weights = dataset.advance_curriculum(new_weights)
            
            logger.info(
                f"Advanced to stage {self.current_stage_idx}: "
                f"{self.current_stage['name']}. "
                f"Cooldown for {self.cooldown_cycles} cycles."
            )
            
            return new_weights
        
        return self.current_weights
    
    def rollback(self, dataset: CurriculumDataset):
        """
        Rollback to the previous curriculum stage.
        Enhanced with cooldown mechanism.
        
        Args:
            dataset: The curriculum dataset to update
        """
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self.rollback_count += 1
            self.steps_since_rollback = 0
            
            # Start cooldown period after rollback
            self.cooldown_counter = self.cooldown_cycles
            
            # Increase advancement interval after rollback
            self.advancement_interval = int(
                self.advancement_interval * self.rollback_factor
            )
            
            # Restore previous weights
            previous_weights = self.current_weights
            dataset.rollback_curriculum(previous_weights)
            
            logger.warning(
                f"Rolled back to stage {self.current_stage_idx}: "
                f"{self.current_stage['name']}. "
                f"Next advancement in {self.advancement_interval} steps. "
                f"Cooldown for {self.cooldown_cycles} cycles."
            )
    
    def update_step_counters(self):
        """Update internal step counters."""
        self.steps_since_advance += 1
        self.steps_since_rollback += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current curriculum status with enhanced tracking."""
        return {
            "stage_idx": self.current_stage_idx,
            "stage_name": self.current_stage.get("name", "unknown"),
            "weights": self.current_weights,
            "rollback_count": self.rollback_count,
            "advancement_count": self.advancement_counter,
            "steps_since_advance": self.steps_since_advance,
            "steps_since_rollback": self.steps_since_rollback,
            "advancement_interval": self.advancement_interval,
            "performance_history": list(self.performance_history),
            "patience_counter": self.patience_counter,
            "cooldown_counter": self.cooldown_counter,
            "smoothed_metrics": {
                name: self._get_smoothed_metric(name)
                for name in self.metric_history.keys()
            },
        }


class CurriculumCallback(TrainerCallback):
    """
    Custom callback for managing curriculum learning during training.
    
    This callback handles automatic curriculum advancement and rollback
    based on validation performance.
    """
    
    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        curriculum_dataset: CurriculumDataset,
        trainer: Optional[Trainer] = None
    ):
        """
        Initialize the curriculum callback.
        
        Args:
            curriculum_manager: Manager for curriculum state
            curriculum_dataset: Dataset with curriculum support
            trainer: Optional trainer instance
        """
        self.manager = curriculum_manager
        self.dataset = curriculum_dataset
        self.trainer = trainer
        self.last_eval_metrics = {}
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Update step counters at the end of each training step."""
        self.manager.update_step_counters()
        
        # Log curriculum stage periodically
        if state.global_step % args.logging_steps == 0:
            stage_idx = self.manager.current_stage_idx
            
            if wandb.run:
                wandb.log({
                    "curriculum/stage": stage_idx,
                    "curriculum/stage_name": self.manager.current_stage.get("name", "unknown"),
                }, step=state.global_step)
        
        return control
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs
    ):
        """
        Handle curriculum advancement/rollback on evaluation.
        
        This is called after each evaluation phase.
        """
        # Store current metrics
        self.last_eval_metrics = metrics
        current_performance = metrics.get("eval_loss", float('inf'))
        
        # Convert loss to performance score (lower loss = higher performance)
        if current_performance != float('inf'):
            current_performance = 1.0 / (1.0 + current_performance)
        else:
            current_performance = 0.0
        
        # Check if we should attempt advancement
        if self.manager.should_attempt_advance(state.global_step):
            logger.info(f"Attempting curriculum advancement at step {state.global_step}")
            
            # Store performance before advancement
            perf_before = current_performance
            
            # Advance curriculum
            old_weights = self.dataset.difficulty_weights.copy()
            new_weights = self.manager.advance(self.dataset)
            
            # Force immediate re-evaluation if trainer is available
            if self.trainer:
                logger.info("Running immediate re-evaluation on new curriculum mix")
                
                # Create evaluation dataloader with new mix
                eval_dataloader = self.trainer.get_eval_dataloader()
                
                # Run evaluation
                eval_metrics = self.trainer.evaluate(eval_dataset=eval_dataloader)
                
                # Get performance after advancement
                perf_after_loss = eval_metrics.get("eval_loss", float('inf'))
                perf_after = 1.0 / (1.0 + perf_after_loss) if perf_after_loss != float('inf') else 0.0
            else:
                # If no trainer, use a heuristic (not ideal)
                perf_after = perf_before * 0.95  # Assume slight drop
            
            # Let manager decide whether to keep or rollback
            advanced, message = self.manager.decide_and_update(
                perf_before=perf_before,
                perf_after=perf_after,
                dataset=self.dataset,
                global_step=state.global_step
            )
            
            logger.info(message)
            
            # Log comprehensive curriculum state
            if wandb.run:
                status = self.manager.get_status()
                wandb.log({
                    "curriculum/status": status,
                    "curriculum/advanced": advanced,
                }, step=state.global_step)
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log final curriculum statistics at the end of training."""
        final_status = self.manager.get_status()
        dataset_stats = self.dataset.get_statistics()
        
        logger.info("=" * 50)
        logger.info("Final Curriculum Statistics:")
        logger.info(f"  Final stage: {final_status['stage_name']}")
        logger.info(f"  Total rollbacks: {final_status['rollback_count']}")
        logger.info(f"  Final weights: {final_status['weights']}")
        logger.info(f"  Total samples used: {dataset_stats['total_samples']}")
        logger.info("=" * 50)
        
        if wandb.run:
            wandb.log({
                "curriculum/final_status": final_status,
                "curriculum/final_dataset_stats": dataset_stats,
            })
        
        return control


def load_model_with_lora(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load the base model and apply LoRA configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config.get("model", {})
    model_name = model_config.get("model_name", "Qwen/Qwen2.5-VL-7B")
    
    logger.info(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if model_config.get("torch_dtype") == "float16" else torch.bfloat16,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing
    if model_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Load LoRA configuration
    lora_config_path = Path("configs/lora_rank_config.json")
    if lora_config_path.exists():
        with open(lora_config_path, 'r') as f:
            lora_ranks = json.load(f)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config.get("lora_r", 32),
            lora_alpha=model_config.get("lora_alpha", 64),
            lora_dropout=model_config.get("lora_dropout", 0.1),
            target_modules=list(lora_ranks.get("layer_ranks", {}).keys()),
            modules_to_save=None,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        logger.info("Applied LoRA configuration to model")
    else:
        logger.warning("LoRA configuration file not found, using full fine-tuning")
    
    return model, tokenizer


def run_sft_training(
    config: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: CurriculumDataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./outputs",
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run the SFT training with curriculum learning.
    
    Args:
        config: Training configuration
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset with curriculum support
        eval_dataset: Optional evaluation dataset
        output_dir: Output directory for checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    training_config = config.get("training", {})
    curriculum_config = config.get("curriculum", {})
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 500),
        logging_steps=training_config.get("logging_steps", 10),
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=training_config.get("bf16", True),
        tf32=training_config.get("tf32", True),
        gradient_checkpointing=True,
        report_to=training_config.get("report_to", ["wandb"]),
        run_name=f"sft_curriculum_{wandb.util.generate_id()}" if wandb.run else None,
        logging_first_step=True,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize curriculum manager
    curriculum_manager = CurriculumManager(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset or train_dataset,  # Use train dataset if no eval provided
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Add curriculum callback
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        curriculum_dataset=train_dataset,
        trainer=trainer
    )
    trainer.add_callback(curriculum_callback)
    
    # Check for checkpoint
    last_checkpoint = None
    if resume_from_checkpoint:
        last_checkpoint = resume_from_checkpoint
    elif output_dir and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info(f"Found checkpoint: {last_checkpoint}")
    
    # Start training
    logger.info("Starting SFT training with curriculum learning")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate final model
    if eval_dataset:
        logger.info("Running final evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        metrics.update(eval_metrics)
    
    # Save final curriculum state
    final_status = curriculum_manager.get_status()
    status_path = Path(output_dir) / "curriculum_final_status.json"
    with open(status_path, 'w') as f:
        json.dump(final_status, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    
    return model, metrics


def main():
    """Standalone SFT training script."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="SFT Training with Curriculum Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_params.yaml",
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/curriculum",
        help="Path to processed curriculum data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/sft",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model configuration
    with open("configs/model_arch.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
        config.update(model_config)
    
    # Initialize wandb
    if "wandb" in config.get("training", {}).get("report_to", []):
        wandb.init(
            project="pixelis-sft",
            name=f"sft_curriculum_{wandb.util.generate_id()}",
            config=config,
            tags=["sft", "curriculum"],
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_with_lora(config)
    
    # Create curriculum dataset
    train_dataset = CurriculumDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.get("model", {}).get("max_length", 4096),
        use_split_files=config.get("curriculum", {}).get("use_split_files", True),
        initial_stage="simple"
    )
    
    # Run training
    model, metrics = run_sft_training(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume
    )
    
    # Log final metrics
    logger.info("Final training metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Close wandb
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()