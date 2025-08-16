"""
Update Worker Module

Implements the asynchronous model update worker with a three-tiered safety system:
1. Behavioral Guardrail: KL divergence penalty with dynamic beta adjustment
2. Magnitude Guardrail: Gradient clipping to prevent exploding gradients
3. Temporal Guardrail: EMA smoothing with atomic synchronization

This worker processes update tasks from the queue and safely updates the model
while maintaining stability and preventing catastrophic forgetting.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import time
import json
from datetime import datetime
from pathlib import Path
import threading
import tempfile
import shutil
import signal
import sys

from ..data_structures import UpdateTask, Experience, RewardComponents
from ..modules.reward_shaping import RewardOrchestrator
from ..modules.audit import AuditLogger, AuditEventType, AuditResult, audit_log

logger = logging.getLogger(__name__)


@dataclass
class KLConfig:
    """Configuration for KL divergence penalty."""
    beta_update_mode: str = "auto"  # "fixed" or "auto"
    initial_beta: float = 0.01
    target_kl: float = 0.05
    kl_tolerance: float = 0.01
    beta_increase_factor: float = 1.2
    beta_decrease_factor: float = 1.2
    min_beta: float = 1e-4
    max_beta: float = 1.0
    
    def __post_init__(self):
        """Validate KL configuration."""
        if self.beta_update_mode not in ["fixed", "auto"]:
            raise ValueError("beta_update_mode must be 'fixed' or 'auto'")
        
        if self.target_kl <= 0:
            raise ValueError("target_kl must be positive")
        
        if self.min_beta > self.max_beta:
            raise ValueError("min_beta must be <= max_beta")


class SharedMemoryReconstructor:
    """
    Handles reconstruction of tensors from shared memory in the worker process.
    """
    
    @staticmethod
    def reconstruct_tensor_from_info(shm_info: Dict[str, Any]) -> torch.Tensor:
        """
        Reconstruct a tensor from shared memory info.
        
        Args:
            shm_info: Shared memory metadata dictionary
            
        Returns:
            Reconstructed tensor
        """
        # Create a tensor with the specified shape and dtype
        # In production, this would connect to the actual shared memory segment
        shape = tuple(shm_info['shape'])
        dtype = shm_info.get('dtype', torch.float32)
        
        # For PyTorch shared memory, we would typically:
        # 1. Get the shared memory handle from the name
        # 2. Create a storage view of the shared memory
        # 3. Create a tensor from the storage
        
        logger.debug(f"Reconstructing tensor from shared memory: {shm_info.get('name', 'unknown')}")
        
        # Placeholder implementation - in production would access actual shared memory
        # This shows the interface structure
        tensor = torch.zeros(shape, dtype=dtype)
        
        return tensor


class UpdateWorker:
    """
    Model update worker that processes learning updates asynchronously.
    
    Implements a conservative update strategy with multiple safety mechanisms
    to ensure stable online learning without catastrophic forgetting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        update_queue: Queue,
        cleanup_confirmation_queue: Queue,
        config: Dict[str, Any],
        reward_orchestrator: Optional[RewardOrchestrator] = None,
        model_save_path: Optional[str] = None
    ):
        """
        Initialize the update worker.
        
        Args:
            model: The model to update
            update_queue: Queue for receiving update tasks
            cleanup_confirmation_queue: Queue for sending cleanup confirmations
            config: Configuration dictionary
            reward_orchestrator: Optional reward orchestrator instance
            model_save_path: Path for saving model checkpoints
        """
        self.model = model
        self.update_queue = update_queue
        self.cleanup_confirmation_queue = cleanup_confirmation_queue
        self.config = config
        self.reward_orchestrator = reward_orchestrator
        
        # Create save directory
        if model_save_path:
            self.model_save_path = Path(model_save_path)
        else:
            self.model_save_path = Path("./checkpoints/online_updates")
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # KL divergence configuration
        self.kl_config = KLConfig(
            beta_update_mode=config.get('kl_beta_update_mode', 'auto'),
            initial_beta=config.get('kl_initial_beta', 0.01),
            target_kl=config.get('kl_target_kl', 0.05),
            kl_tolerance=config.get('kl_tolerance', 0.01),
            beta_increase_factor=config.get('kl_beta_increase_factor', 1.2),
            beta_decrease_factor=config.get('kl_beta_decrease_factor', 1.2),
            min_beta=config.get('kl_min_beta', 1e-4),
            max_beta=config.get('kl_max_beta', 1.0)
        )
        
        # Current beta value for KL penalty
        self.current_beta = self.kl_config.initial_beta
        
        # KL divergence tracking
        self.kl_history = []
        self.kl_window_size = config.get('kl_window_size', 100)
        
        # Gradient clipping
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        
        # EMA model for smoothing
        self.use_ema = config.get('use_ema', True)
        self.ema_decay = config.get('ema_decay', 0.999)
        
        if self.use_ema:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
        
        # Synchronization settings
        self.sync_frequency = config.get('sync_frequency', 100)  # Updates between syncs
        self.updates_since_sync = 0
        self.model_version = 0
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Shared memory reconstructor
        self.shm_reconstructor = SharedMemoryReconstructor()
        
        # Statistics tracking
        self.stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'mean_kl': 0.0,
            'mean_gradient_norm': 0.0,
            'current_beta': self.current_beta
        }
        
        # Logging
        self.update_log_path = self.model_save_path / "update_audit.log"
        self.contribution_log_path = self.model_save_path / "update_contribution.jsonl"
        
        # Initialize audit logger
        self.audit_logger = AuditLogger(
            audit_dir=str(self.model_save_path / "audit"),
            retention_days=config.get('audit_retention_days', 365),
            enable_async=True
        )
        
        # Initialize logs
        self._init_logs()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Update worker initialized (PID: {os.getpid()}) with KL config: {self.kl_config}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.shutdown()
        sys.exit(0)
    
    def _create_ema_model(self) -> nn.Module:
        """
        Create an EMA copy of the model.
        
        Returns:
            EMA model copy
        """
        import copy
        
        try:
            # Create a deep copy of the model
            ema_model = copy.deepcopy(self.model)
            
            # Disable gradients for EMA model
            ema_model.eval()
            for param in ema_model.parameters():
                param.requires_grad = False
            
            return ema_model
        
        except Exception as e:
            logger.warning(f"Could not create EMA model: {e}")
            return None
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for model updates.
        
        Returns:
            Optimizer instance
        """
        # Filter only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            logger.warning("No trainable parameters found!")
            return None
        
        optimizer_type = self.config.get('optimizer', 'adamw')
        learning_rate = self.config.get('base_learning_rate', 1e-5)
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=learning_rate,
                momentum=0.9
            )
        
        logger.info(f"Created {optimizer_type} optimizer for {len(trainable_params)} parameters")
        return optimizer
    
    def _init_logs(self):
        """Initialize log files."""
        # Create audit log header
        with open(self.update_log_path, 'w') as f:
            f.write(f"# Update Audit Log - Started {datetime.now().isoformat()}\n")
            f.write(f"# KL Configuration: {json.dumps(self.kl_config.__dict__, indent=2)}\n")
            f.write("-" * 80 + "\n")
        
        # Create contribution log (JSONL format)
        with open(self.contribution_log_path, 'w') as f:
            # Write metadata as first line
            metadata = {
                'type': 'metadata',
                'timestamp': datetime.now().isoformat(),
                'kl_config': self.kl_config.__dict__,
                'config': self.config
            }
            f.write(json.dumps(metadata) + "\n")
    
    def run(self):
        """
        Main worker loop that processes update tasks.
        """
        logger.info("Update worker started")
        
        while True:
            try:
                # Get update task from queue (blocking with timeout)
                task = self.update_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    logger.info("Update worker received shutdown signal")
                    break
                
                # Process the update
                self._process_update(task)
                
                # Periodic EMA sync
                if self.use_ema and self.updates_since_sync >= self.sync_frequency:
                    self._save_ema_snapshot()
                    self.updates_since_sync = 0
                
            except mp.queues.Empty:
                # No tasks available - expected
                continue
            except Exception as e:
                logger.error(f"Error in update worker: {e}", exc_info=True)
                self.stats['failed_updates'] += 1
        
        # Final cleanup
        self.shutdown()
        logger.info("Update worker stopped")
    
    def _process_update(self, task: UpdateTask):
        """
        Process a single update task with all safety mechanisms.
        
        Implements the three-tiered safety system:
        1. Behavioral Guardrail: KL divergence penalty
        2. Magnitude Guardrail: Gradient clipping
        3. Temporal Guardrail: EMA smoothing
        
        Args:
            task: Update task containing experience and reward
        """
        start_time = time.time()
        shm_name = None
        
        try:
            # Extract data from task
            experience = task.experience
            reward_tensor = task.reward_tensor
            learning_rate = task.learning_rate
            original_logits = task.original_logits
            
            # Handle shared memory if present
            if hasattr(experience, 'metadata') and 'shm_info' in experience.metadata:
                shm_info = experience.metadata['shm_info']
                shm_name = shm_info.get('name') if isinstance(shm_info, dict) else getattr(shm_info, 'name', None)
                
                # Reconstruct tensor from shared memory
                experience.image_features = self.shm_reconstructor.reconstruct_tensor_from_info(
                    shm_info if isinstance(shm_info, dict) else {
                        'name': getattr(shm_info, 'name', 'unknown'),
                        'shape': getattr(shm_info, 'shape', (1, 512)),
                        'dtype': getattr(shm_info, 'dtype', torch.float32)
                    }
                )
                logger.debug(f"Reconstructed tensor from shared memory: {shm_name}")
            
            # Skip update if no optimizer
            if self.optimizer is None:
                logger.warning("No optimizer available, skipping update")
                return
            
            # Prepare inputs
            input_ids = experience.get_input_ids()
            labels = experience.get_labels()
            
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            # Forward pass
            if hasattr(experience, 'image_features') and experience.image_features is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    images=experience.image_features,
                    labels=labels
                )
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            # Get RL loss
            rl_loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            current_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Weight by reward if provided
            if reward_tensor is not None:
                if isinstance(reward_tensor, torch.Tensor):
                    rl_loss = rl_loss * reward_tensor.mean()
                else:
                    rl_loss = rl_loss * reward_tensor
            
            # Calculate KL divergence penalty (Behavioral Guardrail)
            kl_penalty, kl_div = self._calculate_kl_penalty(current_logits, original_logits)
            
            # Total loss with KL penalty
            total_loss = rl_loss + self.current_beta * kl_penalty
            
            # Check KL constraint
            if kl_div.item() > self.kl_config.target_kl * 2:
                logger.warning(
                    f"KL divergence {kl_div.item():.4f} exceeds 2x target "
                    f"({self.kl_config.target_kl * 2:.4f}), skipping update"
                )
                self.stats['failed_updates'] += 1
                
                # Log failed update to audit trail
                self.audit_logger.log(
                    event_type=AuditEventType.MODEL_UPDATE,
                    actor=f"update_worker_{os.getpid()}",
                    action="apply_online_update",
                    resource=f"model_v{self.model_version}",
                    result=AuditResult.BLOCKED,
                    metadata={
                        'task_id': task.task_id,
                        'reason': 'kl_divergence_exceeded',
                        'kl_divergence': kl_div.item(),
                        'kl_limit': self.kl_config.target_kl * 2
                    }
                )
                return
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (Magnitude Guardrail)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update EMA model (Temporal Guardrail)
            if self.use_ema:
                self._update_ema_model()
            
            # Update KL tracking
            self._update_kl_tracking(kl_div.item())
            
            # Auto-adjust beta if enabled
            if self.kl_config.beta_update_mode == "auto":
                self._adjust_beta()
            
            # Update statistics
            self.stats['total_updates'] += 1
            self.stats['successful_updates'] += 1
            self.stats['mean_gradient_norm'] = (
                0.9 * self.stats['mean_gradient_norm'] + 0.1 * grad_norm.item()
            )
            self.updates_since_sync += 1
            
            # Log update details
            self._log_update(
                task=task,
                rl_loss=rl_loss.item(),
                kl_div=kl_div.item(),
                total_loss=total_loss.item(),
                grad_norm=grad_norm.item(),
                duration=time.time() - start_time
            )
            
            # Mark task as processed
            task.mark_processed()
            
            logger.debug(
                f"Processed update {task.task_id}: "
                f"loss={total_loss.item():.4f}, kl={kl_div.item():.4f}, "
                f"grad_norm={grad_norm.item():.4f}, beta={self.current_beta:.6f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to process update {task.task_id}: {e}", exc_info=True)
            self.stats['failed_updates'] += 1
        
        finally:
            # Send cleanup confirmation if using shared memory
            if shm_name and self.cleanup_confirmation_queue is not None:
                try:
                    self.cleanup_confirmation_queue.put(shm_name, timeout=1.0)
                    logger.debug(f"Sent cleanup confirmation for {shm_name}")
                except:
                    logger.warning(f"Could not send cleanup confirmation for {shm_name}")
    
    def _calculate_kl_penalty(
        self,
        current_logits: torch.Tensor,
        original_logits: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate KL divergence penalty between current and original policy.
        
        Uses forward KL: KL(π_old || π_new) to prevent the new policy
        from deviating too much from the original.
        
        Mathematical formulation (documented in ARCHITECTURE.md):
        KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        
        Where:
        - P is the original policy (π_old)
        - Q is the current policy (π_new)
        
        Args:
            current_logits: Logits from current policy
            original_logits: Logits from original policy
            
        Returns:
            Tuple of (kl_penalty, kl_divergence)
        """
        if original_logits is None:
            # No original logits, return zero penalty
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # Convert logits to probabilities
        current_probs = F.softmax(current_logits, dim=-1)
        original_probs = F.softmax(original_logits, dim=-1)
        
        # Calculate forward KL divergence: KL(π_old || π_new)
        # This penalizes the new policy for deviating from the old
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-8),
            original_probs,
            reduction='batchmean'
        )
        
        # KL penalty is the divergence itself
        kl_penalty = kl_div
        
        return kl_penalty, kl_div
    
    def _update_ema_model(self):
        """
        Update EMA model parameters using exponential moving average.
        
        EMA formula: θ_ema = α * θ_ema + (1 - α) * θ_model
        Where α is the decay factor (typically 0.999)
        """
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data,
                    alpha=1 - self.ema_decay
                )
    
    def _update_kl_tracking(self, kl_value: float):
        """
        Update KL divergence tracking for auto-adjustment.
        
        Args:
            kl_value: Current KL divergence value
        """
        self.kl_history.append(kl_value)
        
        # Maintain window size
        if len(self.kl_history) > self.kl_window_size:
            self.kl_history.pop(0)
        
        # Update mean KL
        if self.kl_history:
            self.stats['mean_kl'] = np.mean(self.kl_history)
    
    def _adjust_beta(self):
        """
        Automatically adjust beta based on mean KL divergence.
        
        Implements adaptive KL penalty adjustment:
        - Increases beta if KL is too high (policy changing too fast)
        - Decreases beta if KL is too low (learning may be stalled)
        """
        if len(self.kl_history) < 10:
            # Not enough samples yet
            return
        
        mean_kl = self.stats['mean_kl']
        target_kl = self.kl_config.target_kl
        tolerance = self.kl_config.kl_tolerance
        
        if mean_kl > target_kl + tolerance:
            # KL too high - increase beta to constrain more
            old_beta = self.current_beta
            new_beta = self.current_beta * self.kl_config.beta_increase_factor
            self.current_beta = min(new_beta, self.kl_config.max_beta)
            logger.debug(
                f"Increased beta: {self.current_beta:.6f} "
                f"(mean_kl={mean_kl:.4f} > {target_kl + tolerance:.4f})"
            )
            
            # Log beta adjustment to audit trail
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGE,
                actor=f"update_worker_{os.getpid()}",
                action="adjust_kl_beta",
                resource="kl_penalty_coefficient",
                result=AuditResult.SUCCESS,
                metadata={
                    'old_beta': old_beta,
                    'new_beta': self.current_beta,
                    'reason': 'kl_too_high',
                    'mean_kl': mean_kl,
                    'target_kl': target_kl
                }
            )
        
        elif mean_kl < target_kl - tolerance:
            # KL too low - decrease beta to allow more learning
            old_beta = self.current_beta
            new_beta = self.current_beta / self.kl_config.beta_decrease_factor
            self.current_beta = max(new_beta, self.kl_config.min_beta)
            logger.debug(
                f"Decreased beta: {self.current_beta:.6f} "
                f"(mean_kl={mean_kl:.4f} < {target_kl - tolerance:.4f})"
            )
            
            # Log beta adjustment to audit trail
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGE,
                actor=f"update_worker_{os.getpid()}",
                action="adjust_kl_beta",
                resource="kl_penalty_coefficient",
                result=AuditResult.SUCCESS,
                metadata={
                    'old_beta': old_beta,
                    'new_beta': self.current_beta,
                    'reason': 'kl_too_low',
                    'mean_kl': mean_kl,
                    'target_kl': target_kl
                }
            )
        
        # Update stats
        self.stats['current_beta'] = self.current_beta
    
    def _save_ema_snapshot(self):
        """
        Save EMA model snapshot with atomic versioning.
        
        Implements the versioned, atomic write protocol to prevent
        race conditions during model synchronization.
        """
        if self.ema_model is None:
            return
        
        try:
            # Increment version
            self.model_version += 1
            version_str = f"v{self.model_version}"
            
            # Create temporary file
            temp_path = self.model_save_path / f"ema_model_snapshot.{version_str}.pt.tmp"
            final_path = self.model_save_path / f"ema_model_snapshot.{version_str}.pt"
            
            # Save to temporary file
            torch.save({
                'model_state_dict': self.ema_model.state_dict(),
                'version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'kl_config': self.kl_config.__dict__,
                'current_beta': self.current_beta
            }, temp_path)
            
            # Atomic rename to final path
            os.rename(temp_path, final_path)
            
            # Update pointer file atomically
            pointer_temp = self.model_save_path / "latest_model_version.txt.tmp"
            pointer_final = self.model_save_path / "latest_model_version.txt"
            
            with open(pointer_temp, 'w') as f:
                f.write(f"ema_model_snapshot.{version_str}.pt\n")
                f.write(f"version: {self.model_version}\n")
                f.write(f"timestamp: {datetime.now().isoformat()}\n")
            
            # Atomic rename of pointer file
            os.rename(pointer_temp, pointer_final)
            
            logger.info(f"Saved EMA snapshot version {self.model_version}")
            
            # Clean up old snapshots (keep last 3)
            self._cleanup_old_snapshots()
            
        except Exception as e:
            logger.error(f"Failed to save EMA snapshot: {e}")
    
    def _cleanup_old_snapshots(self, keep_last: int = 3):
        """
        Clean up old model snapshots, keeping only the most recent ones.
        
        Args:
            keep_last: Number of snapshots to keep
        """
        try:
            # Find all snapshot files
            snapshot_files = list(self.model_save_path.glob("ema_model_snapshot.v*.pt"))
            
            # Sort by version number
            snapshot_files.sort(key=lambda f: int(f.stem.split('.v')[1].split('.')[0]))
            
            # Remove old files
            if len(snapshot_files) > keep_last:
                for old_file in snapshot_files[:-keep_last]:
                    old_file.unlink()
                    logger.debug(f"Removed old snapshot: {old_file.name}")
        
        except Exception as e:
            logger.warning(f"Failed to cleanup old snapshots: {e}")
    
    def _log_update(
        self,
        task: UpdateTask,
        rl_loss: float,
        kl_div: float,
        total_loss: float,
        grad_norm: float,
        duration: float
    ):
        """
        Log update details for audit and analysis.
        
        Uses comprehensive audit logging with cryptographic hash chain
        for tamper-proof audit trail.
        
        Args:
            task: Update task
            rl_loss: RL loss value
            kl_div: KL divergence value
            total_loss: Total loss value
            grad_norm: Gradient norm
            duration: Processing duration
        """
        # Create comprehensive metadata
        audit_metadata = {
            'task_id': task.task_id,
            'experience_id': task.experience.experience_id,
            'model_confidence': task.experience.model_confidence,
            'learning_rate': task.learning_rate,
            'losses': {
                'rl_loss': rl_loss,
                'kl_divergence': kl_div,
                'total_loss': total_loss
            },
            'gradients': {
                'norm': grad_norm,
                'clipped': grad_norm > self.gradient_clip_norm
            },
            'kl_control': {
                'current_beta': self.current_beta,
                'mean_kl': self.stats['mean_kl'],
                'target_kl': self.kl_config.target_kl
            },
            'reward_components': {
                'task_reward': float(task.reward_tensor[0]) if isinstance(task.reward_tensor, torch.Tensor) and len(task.reward_tensor) > 0 else 0.0,
                'curiosity_reward': float(task.reward_tensor[1]) if isinstance(task.reward_tensor, torch.Tensor) and len(task.reward_tensor) > 1 else 0.0,
                'coherence_reward': float(task.reward_tensor[2]) if isinstance(task.reward_tensor, torch.Tensor) and len(task.reward_tensor) > 2 else 0.0,
                'tool_penalty': float(task.reward_tensor[3]) if isinstance(task.reward_tensor, torch.Tensor) and len(task.reward_tensor) > 3 else 0.0,
                'total_reward': float(task.reward_tensor[-1]) if isinstance(task.reward_tensor, torch.Tensor) and len(task.reward_tensor) > 0 else 0.0
            },
            'duration_seconds': duration,
            'update_number': self.stats['total_updates'],
            'model_version': self.model_version
        }
        
        # Log to audit trail with cryptographic hash chain
        self.audit_logger.log(
            event_type=AuditEventType.MODEL_UPDATE,
            actor=f"update_worker_{os.getpid()}",
            action="apply_online_update",
            resource=f"model_v{self.model_version}",
            result=AuditResult.SUCCESS,
            metadata=audit_metadata
        )
        
        # Also maintain legacy logs for backward compatibility
        # Simple text audit log
        audit_entry = (
            f"{datetime.now().isoformat()} | "
            f"Task: {task.task_id} | "
            f"Loss: {total_loss:.4f} | "
            f"KL: {kl_div:.4f} | "
            f"Beta: {self.current_beta:.6f} | "
            f"GradNorm: {grad_norm:.4f} | "
            f"LR: {task.learning_rate:.6f} | "
            f"Duration: {duration:.3f}s\n"
        )
        
        with open(self.update_log_path, 'a') as f:
            f.write(audit_entry)
        
        # Contribution log entry (detailed JSONL)
        contribution_entry = {
            'type': 'update',
            'timestamp': datetime.now().isoformat(),
            'task_id': task.task_id,
            'experience_id': task.experience.experience_id,
            'reward_tensor': task.reward_tensor.tolist() if isinstance(task.reward_tensor, torch.Tensor) else task.reward_tensor,
            'learning_rate': task.learning_rate,
            'losses': {
                'rl_loss': rl_loss,
                'kl_divergence': kl_div,
                'total_loss': total_loss
            },
            'gradients': {
                'norm': grad_norm,
                'clipped': grad_norm > self.gradient_clip_norm
            },
            'kl_control': {
                'current_beta': self.current_beta,
                'mean_kl': self.stats['mean_kl'],
                'target_kl': self.kl_config.target_kl
            },
            'duration_seconds': duration,
            'update_number': self.stats['total_updates']
        }
        
        with open(self.contribution_log_path, 'a') as f:
            f.write(json.dumps(contribution_entry) + "\n")
    
    def get_ema_model(self):
        """Get the EMA model for inference."""
        return self.ema_model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'total_updates': self.stats['total_updates'],
            'successful_updates': self.stats['successful_updates'],
            'failed_updates': self.stats['failed_updates'],
            'mean_kl': self.stats['mean_kl'],
            'current_beta': self.current_beta,
            'mean_gradient_norm': self.stats['mean_gradient_norm'],
            'model_version': self.model_version,
            'updates_since_sync': self.updates_since_sync
        }
    
    def shutdown(self):
        """Gracefully shutdown the update worker."""
        logger.info("Shutting down Update Worker")
        
        try:
            # Save final EMA snapshot
            if self.use_ema:
                self._save_ema_snapshot()
            
            # Save final statistics
            stats_path = self.model_save_path / "final_stats.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'kl_config': self.kl_config.__dict__,
                    'final_beta': self.current_beta,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            # Log shutdown to audit trail
            self.audit_logger.log(
                event_type=AuditEventType.SYSTEM_ERROR,
                actor=f"update_worker_{os.getpid()}",
                action="shutdown",
                resource="update_worker",
                result=AuditResult.SUCCESS,
                metadata={
                    'total_updates': self.stats['total_updates'],
                    'successful_updates': self.stats['successful_updates'],
                    'failed_updates': self.stats['failed_updates'],
                    'final_beta': self.current_beta,
                    'model_version': self.model_version
                }
            )
            
            # Verify audit log integrity before shutdown
            verification_result = self.audit_logger.verify_integrity()
            if not verification_result['valid']:
                logger.error(f"Audit log integrity check failed: {verification_result['errors']}")
            else:
                logger.info(f"Audit log integrity verified: {verification_result['total_entries']} entries")
            
            # Shutdown audit logger
            self.audit_logger.shutdown()
            
            logger.info(f"Update worker cleanup complete. Total updates: {self.stats['total_updates']}")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")