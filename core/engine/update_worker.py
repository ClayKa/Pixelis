"""
Update Worker Module

Handles asynchronous model parameter updates in a separate background process.
Implements a conservative update strategy with multiple safety mechanisms.
"""

import logging
import torch
import torch.nn.functional as F
from torch.multiprocessing import Queue
from typing import Dict, Any, Optional
import time
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class UpdateWorker:
    """
    Background worker for handling model updates.
    
    Implements a three-tiered safety system:
    1. KL-divergence penalty (Behavioral Guardrail)
    2. Gradient clipping (Magnitude Guardrail)  
    3. EMA smoothing (Temporal Guardrail)
    """
    
    def __init__(
        self,
        model,
        update_queue: Queue,
        config: Dict[str, Any]
    ):
        """
        Initialize the Update Worker.
        
        Args:
            model: The model to update
            update_queue: Queue for receiving update tasks
            config: Configuration dictionary
        """
        self.model = model
        self.update_queue = update_queue
        self.config = config
        
        # Safety parameters
        self.kl_weight = config.get('kl_weight', 0.01)
        self.max_kl = config.get('max_kl', 0.05)
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)
        self.ema_decay = config.get('ema_decay', 0.999)
        
        # EMA model for temporal smoothing
        self.ema_model = self._create_ema_model()
        
        # Metrics tracking
        self.update_history = deque(maxlen=100)
        self.total_updates = 0
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        logger.info("Update Worker initialized")
    
    def _create_ema_model(self):
        """
        Create an Exponential Moving Average (EMA) copy of the model.
        
        Returns:
            EMA model copy
        """
        ema_model = type(self.model)(self.model.config)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _create_optimizer(self):
        """
        Create the optimizer for model updates.
        
        Returns:
            Optimizer instance
        """
        # Only optimize LoRA parameters if using PEFT
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
        
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.get('base_learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        return optimizer
    
    def run(self):
        """
        Main loop for the update worker.
        
        Continuously processes update tasks from the queue.
        """
        logger.info("Update Worker started")
        
        while True:
            try:
                # Get update task from queue (blocking)
                update_task = self.update_queue.get(timeout=1.0)
                
                if update_task is None:  # Shutdown signal
                    break
                
                # Process the update
                self._process_update(update_task)
                
            except Exception as e:
                logger.error(f"Error in update worker: {e}")
                continue
        
        logger.info("Update Worker stopped")
    
    def _process_update(self, update_task):
        """
        Process a single update task.
        
        Args:
            update_task: UpdateTask dataclass containing update information
        """
        start_time = time.time()
        
        # Extract components
        experience = update_task.experience
        reward = update_task.reward_tensor
        learning_rate = update_task.learning_rate
        original_logits = update_task.original_logits
        
        # Step 1: Calculate KL divergence penalty
        kl_loss = self._calculate_kl_loss(experience, original_logits)
        
        # Step 2: Calculate main loss
        main_loss = self._calculate_main_loss(experience, reward)
        
        # Step 3: Combine losses
        total_loss = main_loss + self.kl_weight * kl_loss
        
        # Step 4: Check KL constraint
        if kl_loss.item() > self.max_kl:
            logger.warning(f"KL divergence {kl_loss.item():.4f} exceeds max {self.max_kl}, skipping update")
            return
        
        # Step 5: Perform gradient update
        self.optimizer.zero_grad()
        
        # Scale learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        total_loss.backward()
        
        # Step 6: Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip_norm
        )
        
        # Step 7: Update model
        self.optimizer.step()
        
        # Step 8: Update EMA model
        self._update_ema_model()
        
        # Step 9: Log metrics
        update_time = time.time() - start_time
        self._log_update_metrics(
            kl_loss.item(),
            main_loss.item(),
            learning_rate,
            update_time
        )
        
        self.total_updates += 1
    
    def _calculate_kl_loss(
        self,
        experience,
        original_logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate KL divergence between current and original policy.
        
        Args:
            experience: Experience object
            original_logits: Original model logits
            
        Returns:
            KL divergence loss
        """
        if original_logits is None:
            return torch.tensor(0.0)
        
        # Get current logits
        with torch.no_grad():
            current_output = self.model(
                input_ids=experience.get_input_ids(),
                images=experience.image_features
            )
            current_logits = current_output.logits
        
        # Calculate KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            F.softmax(original_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl_loss
    
    def _calculate_main_loss(
        self,
        experience,
        reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the main training loss.
        
        Args:
            experience: Experience object
            reward: Reward tensor
            
        Returns:
            Main loss
        """
        # This is a simplified placeholder
        # In practice, this would implement the full RL loss
        # (e.g., PPO loss with reward-weighted log probabilities)
        
        output = self.model(
            input_ids=experience.get_input_ids(),
            images=experience.image_features,
            labels=experience.get_labels()
        )
        
        # Weight the loss by reward
        loss = output.loss * reward.mean()
        
        return loss
    
    def _update_ema_model(self):
        """
        Update the EMA model parameters.
        """
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data,
                    alpha=1 - self.ema_decay
                )
    
    def _log_update_metrics(
        self,
        kl_loss: float,
        main_loss: float,
        learning_rate: float,
        update_time: float
    ):
        """
        Log metrics for the update.
        
        Args:
            kl_loss: KL divergence loss value
            main_loss: Main loss value
            learning_rate: Learning rate used
            update_time: Time taken for update
        """
        metrics = {
            'kl_loss': kl_loss,
            'main_loss': main_loss,
            'learning_rate': learning_rate,
            'update_time': update_time,
            'total_updates': self.total_updates
        }
        
        self.update_history.append(metrics)
        
        if self.total_updates % 10 == 0:
            # Log summary every 10 updates
            recent_kl = np.mean([m['kl_loss'] for m in self.update_history])
            recent_loss = np.mean([m['main_loss'] for m in self.update_history])
            logger.info(
                f"Update {self.total_updates}: "
                f"Avg KL={recent_kl:.4f}, "
                f"Avg Loss={recent_loss:.4f}"
            )
    
    def get_ema_model(self):
        """
        Get the EMA model for inference.
        
        Returns:
            EMA model
        """
        return self.ema_model
    
    def shutdown(self):
        """
        Gracefully shutdown the update worker.
        """
        logger.info("Shutting down Update Worker")
        # Save final checkpoint if needed
        self._save_checkpoint()
    
    def _save_checkpoint(self):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'config': self.config
        }
        
        checkpoint_path = f"saved_models/update_worker_checkpoint_{self.total_updates}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")