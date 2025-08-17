#!/usr/bin/env python3
"""
Reinforcement Fine-Tuning (RFT) with GRPO for Pixelis.

This module implements:
- GRPO (Group Relative Policy Optimization) training
- Multi-component reward system with curiosity and coherence
- Tool misuse penalty system
- Comprehensive logging and monitoring
"""

import json
import logging
import os
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import Action, Trajectory, ActionType
from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
logger = get_logger(__name__)


class LightweightDynamicsModel(nn.Module):
    """
    Lightweight dynamics model with LoRA for curiosity reward.
    
    This model predicts next state given current state and action,
    using LoRA adapters to reduce parameter count.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Base forward model
        self.base_forward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)
        
        # LoRA adapters for the forward model
        self.lora_down = nn.Linear(state_dim + action_dim, lora_rank, bias=False).to(device)
        self.lora_up = nn.Linear(lora_rank, state_dim, bias=False).to(device)
        self.lora_scale = lora_alpha / lora_rank
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state with LoRA adaptation.
        
        Args:
            state: Current state embedding
            action: Action embedding
            
        Returns:
            Predicted next state
        """
        # Get the target device from the model itself
        target_device = next(self.base_forward.parameters()).device
        
        # Move all input tensors to the target device
        state = state.to(target_device)
        action = action.to(target_device)
        
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=-1)
        
        # Base forward pass
        base_output = self.base_forward(state_action)
        
        # LoRA forward pass
        lora_output = self.lora_up(self.lora_down(state_action)) * self.lora_scale
        
        # Combine base and LoRA outputs
        return base_output + lora_output


class EnhancedCuriosityModule(nn.Module):
    """
    Enhanced curiosity module with efficient dynamics model and caching.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        beta: float = 0.2,
        eta: float = 0.5,
        cache_size: int = 1000,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.beta = beta
        self.eta = eta
        self.device = device
        
        # Lightweight dynamics model with LoRA
        self.dynamics_model = LightweightDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Inverse model for action prediction
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(device)
        
        # LRU cache for embeddings
        self.cache = {}
        self.cache_keys = deque(maxlen=cache_size)
        
    def compute_curiosity_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute curiosity reward based on prediction error.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Tuple of (curiosity_reward, metrics)
        """
        # Ensure all tensors are on the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        
        # Check cache
        cache_key = (state.cpu().numpy().tobytes(), action.cpu().numpy().tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Predict next state
        predicted_next = self.dynamics_model(state, action)
        
        # Predict action from state pair (for inverse model loss)
        state_pair = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_model(state_pair)
        
        # Calculate losses
        forward_loss = F.mse_loss(predicted_next, next_state.detach())
        inverse_loss = F.mse_loss(predicted_action, action.detach())
        
        # Curiosity reward is prediction error
        with torch.no_grad():
            prediction_error = F.mse_loss(
                predicted_next,
                next_state,
                reduction='none'
            ).mean(dim=-1)
            curiosity_reward = self.eta * prediction_error
        
        # Update cache
        result = (curiosity_reward, {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'prediction_error': prediction_error.mean().item()
        })
        
        # Add to cache with LRU eviction
        if cache_key not in self.cache:
            # If cache is at max capacity, remove oldest item
            if len(self.cache) >= self.cache_keys.maxlen:
                # The deque will automatically remove the oldest key when we append
                # But we need to manually remove it from the cache dict
                if len(self.cache_keys) == self.cache_keys.maxlen:
                    oldest_key = self.cache_keys[0]
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        self.cache_keys.append(cache_key)
        
        return result


class EnhancedCoherenceAnalyzer:
    """
    Enhanced trajectory coherence analyzer with detailed analysis.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        repetition_penalty: float = 0.5,
        sequence_bonus: float = 0.2
    ):
        self.coherence_threshold = coherence_threshold
        self.repetition_penalty = repetition_penalty
        self.sequence_bonus = sequence_bonus
        
    def compute_coherence_reward(
        self,
        trajectory: List[Dict[str, Any]],
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute coherence reward for trajectory.
        
        Args:
            trajectory: List of action dictionaries
            embeddings: Optional state embeddings
            
        Returns:
            Tuple of (coherence_reward, metrics)
        """
        if len(trajectory) < 2:
            return torch.tensor(0.0), {'coherence_score': 0.0}
        
        rewards = []
        metrics = {
            'repetitions': 0,
            'good_sequences': 0,
            'avg_similarity': 0.0
        }
        
        # Check for repetitions
        for i in range(1, len(trajectory)):
            if trajectory[i] == trajectory[i-1]:
                rewards.append(-self.repetition_penalty)
                metrics['repetitions'] += 1
            
            # Check for good sequences
            if i < len(trajectory) - 1:
                # SEGMENT -> GET_PROPERTIES is good
                if (trajectory[i-1].get('operation') == 'SEGMENT_OBJECT_AT' and
                    trajectory[i].get('operation') == 'GET_PROPERTIES'):
                    rewards.append(self.sequence_bonus)
                    metrics['good_sequences'] += 1
                
                # ZOOM_IN -> READ_TEXT is good
                if (trajectory[i-1].get('operation') == 'ZOOM_IN' and
                    trajectory[i].get('operation') == 'READ_TEXT'):
                    rewards.append(self.sequence_bonus)
                    metrics['good_sequences'] += 1
        
        # Compute embedding similarity if provided
        if embeddings and len(embeddings) > 1:
            similarities = []
            for i in range(1, len(embeddings)):
                sim = F.cosine_similarity(
                    embeddings[i-1].unsqueeze(0),
                    embeddings[i].unsqueeze(0)
                ).item()
                similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            metrics['avg_similarity'] = avg_sim
            
            # Reward moderate similarity (not too similar, not too different)
            if 0.3 < avg_sim < 0.7:
                rewards.append(0.2)
            elif avg_sim > 0.9:  # Too similar
                rewards.append(-0.1)
        
        total_reward = np.mean(rewards) if rewards else 0.0
        return torch.tensor(total_reward), metrics


class ToolMisusePenaltyCalculator:
    """
    Calculate penalties for incorrect tool usage.
    """
    
    def __init__(self, penalty_weight: float = 0.1):
        self.penalty_weight = penalty_weight
        
        # Define tool constraints
        self.tool_constraints = {
            'TRACK_OBJECT': {'requires': 'video', 'prerequisite': None},
            'GET_PROPERTIES': {'requires': 'mask', 'prerequisite': 'SEGMENT_OBJECT_AT'},
            'READ_TEXT': {'requires': 'image', 'prerequisite': None},
            'ZOOM_IN': {'requires': 'coordinates', 'prerequisite': None},
            'SEGMENT_OBJECT_AT': {'requires': 'coordinates', 'prerequisite': None}
        }
    
    def calculate_penalties(
        self,
        trajectory: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, int]]:
        """
        Calculate tool misuse penalties.
        
        Args:
            trajectory: List of actions
            context: Context information (e.g., input type)
            
        Returns:
            Tuple of (total_penalty, violation_counts)
        """
        total_penalty = 0.0
        violations = defaultdict(int)
        
        # Track what has been segmented
        segmented_objects = set()
        
        for i, action in enumerate(trajectory):
            operation = action.get('operation', '')
            
            # Check if operation exists
            if operation not in self.tool_constraints:
                continue
            
            constraints = self.tool_constraints[operation]
            
            # Check prerequisites
            if constraints['prerequisite']:
                prerequisite_met = False
                for j in range(i):
                    if trajectory[j].get('operation') == constraints['prerequisite']:
                        prerequisite_met = True
                        break
                
                if not prerequisite_met:
                    total_penalty -= self.penalty_weight
                    violations[f'missing_prerequisite_{operation}'] += 1
            
            # Check specific violations
            if operation == 'TRACK_OBJECT' and context.get('input_type') != 'video':
                total_penalty -= self.penalty_weight * 2  # Severe violation
                violations['track_on_static_image'] += 1
            
            # Check coordinate bounds
            if operation in ['SEGMENT_OBJECT_AT', 'ZOOM_IN']:
                coords = action.get('arguments', {})
                x, y = coords.get('x', 0), coords.get('y', 0)
                
                # Assuming image dimensions
                if not (0 <= x <= 1024 and 0 <= y <= 1024):
                    total_penalty -= self.penalty_weight
                    violations['out_of_bounds_coordinates'] += 1
            
            # Track segmented objects
            if operation == 'SEGMENT_OBJECT_AT':
                result = action.get('result', {})
                if 'object_id' in result:
                    segmented_objects.add(result['object_id'])
        
        return total_penalty, dict(violations)


class EnhancedRewardOrchestrator:
    """
    Enhanced central reward orchestrator with normalization and curriculum.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Component weights
        self.task_weight = config.get('task_reward_weight', 1.0)
        self.curiosity_weight = config.get('curiosity_reward_weight', 0.3)
        self.coherence_weight = config.get('coherence_reward_weight', 0.2)
        
        # Get device from config or default to cuda if available
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize modules
        self.curiosity_module = EnhancedCuriosityModule(
            beta=config.get('curiosity_beta', 0.2),
            eta=config.get('curiosity_eta', 0.5),
            device=self.device
        )
        
        self.coherence_analyzer = EnhancedCoherenceAnalyzer(
            coherence_threshold=config.get('coherence_threshold', 0.7),
            repetition_penalty=config.get('repetition_penalty', 0.5)
        )
        
        self.penalty_calculator = ToolMisusePenaltyCalculator(
            penalty_weight=config.get('tool_misuse_penalty', 0.1)
        )
        
        # Running statistics for normalization
        self.running_stats = {
            'task': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'curiosity': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'coherence': {'mean': 0.0, 'std': 1.0, 'count': 0}
        }
        
        # Curriculum stage
        self.curriculum_step = 0
        self.curriculum_stages = config.get('curriculum_stages', [])
        
    def calculate_total_reward(
        self,
        trajectory: List[Dict[str, Any]],
        final_answer: Any,
        ground_truth: Any,
        state_embeddings: Optional[List[torch.Tensor]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate total reward with all components.
        
        Args:
            trajectory: Action trajectory
            final_answer: Model's answer
            ground_truth: Correct answer
            state_embeddings: State embeddings for curiosity
            context: Additional context
            
        Returns:
            Dictionary with reward components and total
        """
        # Task reward
        task_reward = 1.0 if final_answer == ground_truth else 0.0
        
        # Curiosity reward
        curiosity_reward = 0.0
        curiosity_metrics = {}
        if state_embeddings and len(state_embeddings) > 1:
            curiosity_rewards = []
            for i in range(len(state_embeddings) - 1):
                # Create action embedding
                action_embedding = self._create_action_embedding(
                    trajectory[i] if i < len(trajectory) else {}
                )
                
                reward, metrics = self.curiosity_module.compute_curiosity_reward(
                    state_embeddings[i],
                    action_embedding,
                    state_embeddings[i + 1]
                )
                curiosity_rewards.append(reward.item())
                curiosity_metrics.update(metrics)
            
            curiosity_reward = np.mean(curiosity_rewards)
        
        # Coherence reward
        coherence_reward_tensor, coherence_metrics = self.coherence_analyzer.compute_coherence_reward(
            trajectory,
            state_embeddings
        )
        coherence_reward = coherence_reward_tensor.item()
        
        # Tool penalties
        tool_penalty, violations = self.penalty_calculator.calculate_penalties(
            trajectory,
            context or {}
        )
        
        # Update running statistics
        self._update_running_stats('task', task_reward)
        self._update_running_stats('curiosity', curiosity_reward)
        self._update_running_stats('coherence', coherence_reward)
        
        # Normalize rewards
        task_norm = self._normalize_reward('task', task_reward)
        curiosity_norm = self._normalize_reward('curiosity', curiosity_reward)
        coherence_norm = self._normalize_reward('coherence', coherence_reward)
        
        # Get curriculum weights
        weights = self._get_curriculum_weights()
        
        # Calculate total reward
        total_reward = (
            weights['task'] * task_norm +
            weights['curiosity'] * curiosity_norm +
            weights['coherence'] * coherence_norm +
            tool_penalty
        )
        
        # Clip total reward
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return {
            'total': total_reward,
            'components': {
                'task': {'raw': task_reward, 'normalized': task_norm},
                'curiosity': {'raw': curiosity_reward, 'normalized': curiosity_norm},
                'coherence': {'raw': coherence_reward, 'normalized': coherence_norm},
                'penalty': tool_penalty
            },
            'metrics': {
                'curiosity': curiosity_metrics,
                'coherence': coherence_metrics,
                'violations': violations
            },
            'weights': weights
        }
    
    def _create_action_embedding(self, action: Dict[str, Any]) -> torch.Tensor:
        """Create action embedding from action dictionary."""
        # Simple embedding - in practice would use proper encoder
        embedding = torch.randn(128).cuda()
        
        # Add structure based on operation
        operation = action.get('operation', '')
        if operation == 'SEGMENT_OBJECT_AT':
            embedding[0] = 1.0
        elif operation == 'READ_TEXT':
            embedding[1] = 1.0
        elif operation == 'TRACK_OBJECT':
            embedding[2] = 1.0
        
        return embedding
    
    def _update_running_stats(self, component: str, value: float):
        """Update running statistics for normalization."""
        stats = self.running_stats[component]
        stats['count'] += 1
        
        # Update mean
        delta = value - stats['mean']
        stats['mean'] += delta / stats['count']
        
        # Update variance (Welford's algorithm)
        if stats['count'] > 1:
            delta2 = value - stats['mean']
            variance = ((stats['count'] - 1) * stats['std']**2 + delta * delta2) / stats['count']
            stats['std'] = np.sqrt(variance)
    
    def _normalize_reward(self, component: str, value: float) -> float:
        """Normalize reward using running statistics."""
        stats = self.running_stats[component]
        if stats['count'] < 10:  # Not enough data
            return value
        
        return (value - stats['mean']) / (stats['std'] + 1e-8)
    
    def _get_curriculum_weights(self) -> Dict[str, float]:
        """Get curriculum-based weights."""
        if not self.curriculum_stages:
            return {
                'task': self.task_weight,
                'curiosity': self.curiosity_weight,
                'coherence': self.coherence_weight
            }
        
        # Find current stage
        for stage in self.curriculum_stages:
            if self.curriculum_step >= stage.get('step', 0):
                current_weights = stage.get('weights', {})
            else:
                break
        
        return {
            'task': current_weights.get('task', self.task_weight),
            'curiosity': current_weights.get('curiosity', self.curiosity_weight),
            'coherence': current_weights.get('coherence', self.coherence_weight)
        }
    
    def update_curriculum_step(self, step: int):
        """Update curriculum step."""
        self.curriculum_step = step


class GRPOTrainer(PPOTrainer):
    """
    Custom GRPO trainer extending PPOTrainer.
    
    GRPO (Group Relative Policy Optimization) addresses vanishing advantages
    by using selective sample replay and group-based advantage estimation.
    """
    
    def __init__(self, *args, grpo_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GRPO specific configuration
        self.grpo_config = grpo_config or {}
        self.group_size = self.grpo_config.get('group_size', 4)
        self.replay_buffer_size = self.grpo_config.get('replay_buffer_size', 100)
        self.replay_ratio = self.grpo_config.get('replay_ratio', 0.5)
        
        # Replay buffer for high-advantage samples
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
    def compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages with GRPO group normalization.
        
        Args:
            values: Value estimates
            rewards: Rewards
            mask: Mask for valid positions
            
        Returns:
            Normalized advantages
        """
        # Standard advantage computation
        advantages = super().compute_advantages(values, rewards, mask)
        
        # Group-based normalization to prevent vanishing
        batch_size = advantages.size(0)
        num_groups = batch_size // self.group_size
        
        if num_groups > 0:
            # Reshape for group processing
            grouped = advantages[:num_groups * self.group_size].view(num_groups, self.group_size)
            
            # Normalize within groups
            group_mean = grouped.mean(dim=1, keepdim=True)
            group_std = grouped.std(dim=1, keepdim=True) + 1e-8
            normalized = (grouped - group_mean) / group_std
            
            # Flatten back
            advantages[:num_groups * self.group_size] = normalized.view(-1)
        
        return advantages
    
    def step(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
        rewards: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None
    ):
        """
        GRPO training step with selective replay.
        
        Args:
            queries: Input queries
            responses: Generated responses
            rewards: Computed rewards
            masks: Optional attention masks
        """
        # Store high-advantage samples in replay buffer
        with torch.no_grad():
            # Compute advantages for selection
            values = self.model.forward_value(
                torch.cat(queries),
                torch.cat(responses)
            )
            advantages = self.compute_advantages(
                values,
                torch.cat(rewards),
                torch.cat(masks) if masks else None
            )
            
            # Select high-advantage samples
            threshold = torch.quantile(advantages, 0.7)
            high_advantage_idx = (advantages > threshold).nonzero().squeeze()
            
            for idx in high_advantage_idx:
                if idx < len(queries):
                    self.replay_buffer.append({
                        'query': queries[idx],
                        'response': responses[idx],
                        'reward': rewards[idx],
                        'mask': masks[idx] if masks else None
                    })
        
        # Mix current batch with replay samples
        if len(self.replay_buffer) > 0:
            num_replay = int(len(queries) * self.replay_ratio)
            replay_samples = random.sample(
                self.replay_buffer,
                min(num_replay, len(self.replay_buffer))
            )
            
            # Add replay samples to batch
            for sample in replay_samples:
                queries.append(sample['query'])
                responses.append(sample['response'])
                rewards.append(sample['reward'])
                if masks and sample['mask'] is not None:
                    masks.append(sample['mask'])
        
        # Run standard PPO step with augmented batch
        return super().step(queries, responses, rewards, masks)


def create_rft_dataset(data_path: str, tokenizer: Any) -> Dataset:
    """
    Create dataset for RFT training.
    
    Args:
        data_path: Path to training data
        tokenizer: Tokenizer
        
    Returns:
        Dataset for RFT
    """
    class RFTDataset(Dataset):
        def __init__(self, data_path: str, tokenizer: Any):
            self.data_path = Path(data_path)
            self.tokenizer = tokenizer
            
            # Load data
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            
            self.samples = self.data.get('samples', [])
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Extract question
            question = sample.get('question', '')
            
            # Tokenize
            encoding = self.tokenizer(
                question,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'question': question,
                'answer': sample.get('answer', ''),
                'trajectory': sample.get('trajectory', [])
            }
    
    return RFTDataset(data_path, tokenizer)


def run_rft_training(
    config: Dict[str, Any],
    sft_model_path: str,
    output_dir: str = "./outputs/rft",
    resume_from_checkpoint: Optional[str] = None
):
    """
    Run RFT training with GRPO.
    
    Args:
        config: Training configuration
        sft_model_path: Path to SFT-trained model
        output_dir: Output directory
        resume_from_checkpoint: Optional checkpoint path
    """
    logger.info("Initializing RFT training with GRPO...")
    
    # Load model and tokenizer
    model_config = config.get('model', {})
    model_name = model_config.get('model_name', 'Qwen/Qwen2.5-VL-7B')
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Load SFT LoRA adapters
    if Path(sft_model_path).exists():
        logger.info(f"Loading SFT adapters from {sft_model_path}")
        model = PeftModel.from_pretrained(base_model, sft_model_path)
    else:
        logger.warning("SFT model not found, using base model")
        model = base_model
    
    # Enable gradient checkpointing
    if model_config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
    
    # Wrap model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        batch_size=config.get('training', {}).get('batch_size', 4),
        mini_batch_size=config.get('training', {}).get('mini_batch_size', 1),
        gradient_accumulation_steps=config.get('training', {}).get('gradient_accumulation_steps', 4),
        learning_rate=config.get('training', {}).get('learning_rate', 1e-5),
        optimize_cuda_cache=True,
        log_with='wandb' if 'wandb' in config.get('training', {}).get('report_to', []) else None,
        project_kwargs={
            'project': 'pixelis-rft',
            'name': f"rft_grpo_{wandb.util.generate_id()}" if wandb.run else None
        }
    )
    
    # Create GRPO trainer
    grpo_config = {
        'group_size': config.get('grpo', {}).get('group_size', 4),
        'replay_buffer_size': config.get('grpo', {}).get('replay_buffer_size', 100),
        'replay_ratio': config.get('grpo', {}).get('replay_ratio', 0.5)
    }
    
    trainer = GRPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        grpo_config=grpo_config
    )
    
    # Create reward orchestrator
    reward_orchestrator = EnhancedRewardOrchestrator(config.get('reward', {}))
    
    # Load dataset
    dataset = create_rft_dataset(
        config.get('curriculum', {}).get('data_path', 'data/processed/curriculum'),
        tokenizer
    )
    
    dataloader = DataLoader(dataset, batch_size=ppo_config.batch_size, shuffle=True)
    
    # Generation configuration
    generation_config = GenerationConfig(
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Training loop
    logger.info("Starting GRPO training loop...")
    
    global_step = 0
    action_distribution = defaultdict(int)
    
    for epoch in range(config.get('training', {}).get('num_epochs', 3)):
        logger.info(f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            query_tensors = batch['input_ids'].cuda()
            
            # Generate responses
            with torch.no_grad():
                response_tensors = trainer.model.generate(
                    query_tensors,
                    generation_config=generation_config
                )
            
            # Decode responses
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Calculate rewards
            rewards = []
            batch_metrics = defaultdict(list)
            
            for i, response in enumerate(responses):
                # Parse trajectory from response
                trajectory = parse_trajectory(response)
                
                # Update action distribution
                for action in trajectory:
                    operation = action.get('operation', 'unknown')
                    action_distribution[operation] += 1
                
                # Get ground truth
                ground_truth = batch['answer'][i]
                
                # Calculate reward
                reward_dict = reward_orchestrator.calculate_total_reward(
                    trajectory=trajectory,
                    final_answer=extract_answer(response),
                    ground_truth=ground_truth,
                    context={'input_type': 'image'}  # Simplified
                )
                
                rewards.append(torch.tensor(reward_dict['total']).cuda())
                
                # Collect metrics
                for key, value in reward_dict['components'].items():
                    batch_metrics[f'reward/{key}/raw'].append(value.get('raw', value))
                    if 'normalized' in value:
                        batch_metrics[f'reward/{key}/normalized'].append(value['normalized'])
            
            # Update curriculum step
            reward_orchestrator.update_curriculum_step(global_step)
            
            # PPO step
            stats = trainer.step(
                [query_tensors[i:i+1] for i in range(len(query_tensors))],
                [response_tensors[i:i+1] for i in range(len(response_tensors))],
                rewards
            )
            
            # Log metrics
            if global_step % 10 == 0:
                log_dict = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'ppo/kl': stats.get('objective/kl', 0),
                    'ppo/entropy': stats.get('objective/entropy', 0),
                    'ppo/policy_loss': stats.get('policy/loss', 0),
                    'ppo/value_loss': stats.get('value/loss', 0),
                }
                
                # Add reward metrics
                for key, values in batch_metrics.items():
                    log_dict[key] = np.mean(values)
                
                # Add action distribution
                total_actions = sum(action_distribution.values())
                if total_actions > 0:
                    for action, count in action_distribution.items():
                        log_dict[f'actions/{action}'] = count / total_actions
                
                # Log to wandb
                if wandb.run:
                    wandb.log(log_dict, step=global_step)
                
                # Console logging
                logger.info(
                    f"Step {global_step}: "
                    f"KL={log_dict.get('ppo/kl', 0):.4f}, "
                    f"Reward={np.mean(rewards):.4f}"
                )
            
            global_step += 1
            
            # Save checkpoint
            if global_step % 1000 == 0:
                checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(final_dir)
    
    logger.info(f"✓ RFT training complete. Model saved to {final_dir}")
    
    return str(final_dir)


def parse_trajectory(response: str) -> List[Dict[str, Any]]:
    """Parse trajectory from model response."""
    # Simplified parsing - in practice would be more robust
    trajectory = []
    
    # Look for action patterns
    lines = response.split('\n')
    for line in lines:
        if any(op in line for op in ['SEGMENT_OBJECT_AT', 'READ_TEXT', 'TRACK_OBJECT', 'ZOOM_IN', 'GET_PROPERTIES']):
            # Extract operation
            for op in ['SEGMENT_OBJECT_AT', 'READ_TEXT', 'TRACK_OBJECT', 'ZOOM_IN', 'GET_PROPERTIES']:
                if op in line:
                    trajectory.append({
                        'operation': op,
                        'arguments': {},  # Would parse arguments
                        'result': {}  # Would parse result
                    })
                    break
    
    return trajectory


def extract_answer(response: str) -> str:
    """Extract final answer from response."""
    # Look for answer tags
    import re
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to last line
    lines = response.strip().split('\n')
    return lines[-1] if lines else ''


def main():
    """Standalone RFT training script."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="RFT Training with GRPO")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_params.yaml',
        help='Path to training configuration'
    )
    parser.add_argument(
        '--sft-model',
        type=str,
        default='outputs/sft/final',
        help='Path to SFT-trained model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/rft',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    if 'wandb' in config.get('training', {}).get('report_to', []):
        wandb.init(
            project='pixelis-rft',
            name=f"rft_grpo_{wandb.util.generate_id()}",
            config=config,
            tags=['rft', 'grpo']
        )
    
    # Run training
    run_rft_training(
        config=config,
        sft_model_path=args.sft_model,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume
    )
    
    # Close wandb
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()