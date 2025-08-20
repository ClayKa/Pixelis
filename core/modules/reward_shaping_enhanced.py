"""
Enhanced Reward Shaping Module for RFT

Implements performance-aware curiosity rewards, trajectory coherence analysis,
tool misuse penalties, and reward normalization with curriculum support.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for all reward components."""
    task_reward: float
    curiosity_reward: float
    coherence_reward: float
    tool_penalty: float
    total_reward: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task': self.task_reward,
            'curiosity': self.curiosity_reward,
            'coherence': self.coherence_reward,
            'penalty': self.tool_penalty,
            'total': self.total_reward,
            'metadata': self.metadata
        }


class LoRADynamicsModel(nn.Module):
    """
    Lightweight dynamics model with LoRA adapters for efficient curiosity computation.
    
    This model uses low-rank adaptation to reduce parameters while maintaining
    expressiveness for next-state prediction.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.lora_scale = lora_alpha / lora_rank
        
        # Base forward dynamics network (frozen during training)
        self.base_layers = nn.ModuleList([
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        ])
        
        # LoRA adapters for each layer
        self.lora_adapters = nn.ModuleList()
        input_dims = [state_dim + action_dim, hidden_dim, hidden_dim]
        output_dims = [hidden_dim, hidden_dim, state_dim]
        
        for in_dim, out_dim in zip(input_dims, output_dims):
            adapter = nn.ModuleDict({
                'down': nn.Linear(in_dim, lora_rank, bias=False),
                'up': nn.Linear(lora_rank, out_dim, bias=False)
            })
            # Initialize LoRA weights
            nn.init.kaiming_uniform_(adapter['down'].weight, a=np.sqrt(5))
            nn.init.zeros_(adapter['up'].weight)
            self.lora_adapters.append(adapter)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        self.to(device)
        
        # Freeze base layers
        for param in self.base_layers.parameters():
            param.requires_grad = False
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state embedding [batch_size, state_dim]
            action: Action embedding [batch_size, action_dim]
            
        Returns:
            Predicted next state [batch_size, state_dim]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward through layers with LoRA adaptation
        for i, (base_layer, lora_adapter) in enumerate(zip(self.base_layers, self.lora_adapters)):
            # Base forward pass
            base_out = base_layer(x)
            
            # LoRA forward pass
            lora_out = lora_adapter['up'](lora_adapter['down'](x)) * self.lora_scale
            
            # Combine base and LoRA
            x = base_out + lora_out
            
            # Apply activation and dropout (except last layer)
            if i < len(self.base_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.lora_adapters.parameters())


class PerformanceAwareCuriosityModule(nn.Module):
    """
    Performance-aware curiosity module with efficient caching and LoRA dynamics.
    
    Implements intrinsic curiosity reward based on prediction error while
    maintaining computational efficiency through caching and lightweight models.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        beta: float = 0.2,  # Weight for forward model loss
        eta: float = 0.5,   # Intrinsic reward scaling
        cache_size: int = 1000,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.beta = beta
        self.eta = eta
        self.device = device
        
        # Lightweight dynamics model with LoRA
        self.dynamics_model = LoRADynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lora_rank=8,
            lora_alpha=16,
            device=device
        )
        
        # Inverse model for action prediction (lightweight)
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # Feature encoder for state embeddings
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim)
        ).to(device)
        
        # LRU cache for computed curiosity rewards
        self.cache = {}
        self.cache_keys = deque(maxlen=cache_size)
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Curiosity module initialized with {self.dynamics_model.get_num_trainable_params()} trainable params")
    
    def compute_curiosity_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        return_losses: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute curiosity reward based on prediction error.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action taken [batch_size, action_dim]
            next_state: Resulting state [batch_size, state_dim]
            return_losses: Whether to compute and return losses
            
        Returns:
            Tuple of (curiosity_reward, metrics_dict)
        """
        # Create cache key
        cache_key = self._create_cache_key(state, action)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Forward dynamics: predict next state
        predicted_next = self.dynamics_model(state_feat, action)
        
        # Calculate prediction error (curiosity reward)
        with torch.no_grad():
            prediction_error = F.mse_loss(
                predicted_next,
                next_state_feat,
                reduction='none'
            ).mean(dim=-1)
            curiosity_reward = self.eta * prediction_error
        
        metrics = {
            'prediction_error': prediction_error.mean().item(),
            'curiosity_reward': curiosity_reward.mean().item(),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
        }
        
        if return_losses:
            # Inverse dynamics: predict action from state pair
            state_pair = torch.cat([state_feat, next_state_feat], dim=-1)
            predicted_action = self.inverse_model(state_pair)
            
            # Calculate losses for training
            forward_loss = F.mse_loss(predicted_next, next_state_feat.detach())
            inverse_loss = F.mse_loss(predicted_action, action.detach())
            total_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss
            
            metrics.update({
                'forward_loss': forward_loss.item(),
                'inverse_loss': inverse_loss.item(),
                'total_loss': total_loss.item()
            })
        
        # Update cache
        result = (curiosity_reward, metrics)
        self._update_cache(cache_key, result)
        
        return result
    
    def _create_cache_key(self, state: torch.Tensor, action: torch.Tensor) -> str:
        """Create cache key from state and action tensors."""
        # Use first few dimensions for efficiency
        state_summary = state.flatten()[:32].cpu().numpy()
        action_summary = action.flatten()[:32].cpu().numpy()
        key_array = np.concatenate([state_summary, action_summary])
        # FIX: Convert bytes to hex string for proper cache key
        return key_array.tobytes().hex()
    
    def _update_cache(self, key: str, value: Tuple):
        """Update LRU cache."""
        if key not in self.cache:
            if len(self.cache_keys) >= self.cache_keys.maxlen:
                # Remove oldest entry
                oldest_key = self.cache_keys[0]
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            self.cache_keys.append(key)
        self.cache[key] = value


class EnhancedTrajectoryCoherenceAnalyzer:
    """
    Enhanced analyzer for trajectory coherence with detailed pattern recognition.
    
    Rewards logical action sequences and penalizes repetitive or illogical patterns.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        repetition_penalty: float = 0.5,
        sequence_bonus: float = 0.2,
        contradiction_penalty: float = 0.3
    ):
        self.coherence_threshold = coherence_threshold
        self.repetition_penalty = repetition_penalty
        self.sequence_bonus = sequence_bonus
        self.contradiction_penalty = contradiction_penalty
        
        # Define good action sequences
        self.good_sequences = [
            ('SEGMENT_OBJECT_AT', 'GET_PROPERTIES'),
            ('ZOOM_IN', 'READ_TEXT'),
            ('SEGMENT_OBJECT_AT', 'TRACK_OBJECT'),
            ('READ_TEXT', 'THINK'),
            ('GET_PROPERTIES', 'THINK')
        ]
        
        # Define bad patterns
        self.bad_patterns = [
            ('TRACK_OBJECT', 'TRACK_OBJECT'),  # Redundant tracking
            ('ZOOM_IN', 'ZOOM_IN'),  # Excessive zooming
            ('GET_PROPERTIES', 'SEGMENT_OBJECT_AT')  # Illogical order
        ]
    
    def compute_coherence_reward(
        self,
        trajectory: List[Dict[str, Any]],
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute coherence reward for trajectory.
        
        Args:
            trajectory: List of actions in trajectory
            embeddings: Optional state embeddings for semantic analysis
            
        Returns:
            Tuple of (coherence_reward, metrics)
        """
        if len(trajectory) < 2:
            return 0.0, {'length': len(trajectory)}
        
        rewards = []
        metrics = {
            'repetitions': 0,
            'good_sequences': 0,
            'bad_patterns': 0,
            'contradictions': 0,
            'avg_similarity': 0.0
        }
        
        # Analyze action patterns
        for i in range(1, len(trajectory)):
            prev_action = trajectory[i-1]
            curr_action = trajectory[i]
            
            prev_op = prev_action.get('operation', '')
            curr_op = curr_action.get('operation', '')
            
            # Check for immediate repetition
            if prev_op == curr_op and prev_action.get('arguments') == curr_action.get('arguments'):
                rewards.append(-self.repetition_penalty)
                metrics['repetitions'] += 1
            
            # Check for good sequences
            if (prev_op, curr_op) in self.good_sequences:
                rewards.append(self.sequence_bonus)
                metrics['good_sequences'] += 1
            
            # Check for bad patterns
            if (prev_op, curr_op) in self.bad_patterns:
                rewards.append(-self.contradiction_penalty)
                metrics['bad_patterns'] += 1
        
        # Analyze semantic coherence if embeddings provided
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
            
            # Reward moderate similarity
            if 0.3 < avg_sim < 0.7:
                rewards.append(0.3)  # Good coherence
            elif avg_sim > 0.9:
                rewards.append(-0.2)  # Too similar (stuck)
            elif avg_sim < 0.1:
                rewards.append(-0.3)  # Too different (random)
        
        # Check for logical contradictions
        contradictions = self._check_contradictions(trajectory)
        if contradictions > 0:
            rewards.append(-self.contradiction_penalty * contradictions)
            metrics['contradictions'] = contradictions
        
        # Calculate final reward
        total_reward = np.mean(rewards) if rewards else 0.0
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return total_reward, metrics
    
    def _check_contradictions(self, trajectory: List[Dict[str, Any]]) -> int:
        """Check for logical contradictions in trajectory."""
        contradictions = 0
        
        # Track what has been done
        segmented_objects = set()
        zoomed_regions = set()
        read_texts = set()
        
        for action in trajectory:
            operation = action.get('operation', '')
            args = action.get('arguments', {})
            
            if operation == 'SEGMENT_OBJECT_AT':
                coord = (args.get('x', 0), args.get('y', 0))
                if coord in segmented_objects:
                    contradictions += 1  # Re-segmenting same location
                segmented_objects.add(coord)
            
            elif operation == 'GET_PROPERTIES':
                # Check if object was segmented
                if not segmented_objects:
                    contradictions += 1  # Getting properties without segmentation
            
            elif operation == 'TRACK_OBJECT':
                # Check if object was segmented
                if not segmented_objects:
                    contradictions += 1  # Tracking without segmentation
        
        return contradictions


class ToolMisusePenaltySystem:
    """
    System for calculating penalties for incorrect tool usage.
    
    Enforces proper tool sequencing and parameter validation.
    """
    
    def __init__(
        self,
        base_penalty: float = 0.1,
        severe_penalty_multiplier: float = 2.0
    ):
        self.base_penalty = base_penalty
        self.severe_penalty_multiplier = severe_penalty_multiplier
        
        # Define tool constraints
        self.tool_constraints = {
            'TRACK_OBJECT': {
                'requires_input': 'video',
                'prerequisite': 'SEGMENT_OBJECT_AT',
                'max_uses_per_trajectory': 5
            },
            'GET_PROPERTIES': {
                'requires_input': 'segmented_object',
                'prerequisite': 'SEGMENT_OBJECT_AT',
                'max_uses_per_trajectory': 10
            },
            'READ_TEXT': {
                'requires_input': 'image_region',
                'prerequisite': None,
                'max_uses_per_trajectory': 15
            },
            'ZOOM_IN': {
                'requires_input': 'coordinates',
                'prerequisite': None,
                'max_uses_per_trajectory': 3
            },
            'SEGMENT_OBJECT_AT': {
                'requires_input': 'coordinates',
                'prerequisite': None,
                'max_uses_per_trajectory': 10
            }
        }
    
    def calculate_penalties(
        self,
        trajectory: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, int]]:
        """
        Calculate total penalties for tool misuse.
        
        Args:
            trajectory: Action trajectory
            context: Context information (input type, etc.)
            
        Returns:
            Tuple of (total_penalty, violation_counts)
        """
        total_penalty = 0.0
        violations = defaultdict(int)
        
        # Track tool usage
        tool_usage_counts = defaultdict(int)
        prerequisites_met = defaultdict(bool)
        
        for i, action in enumerate(trajectory):
            operation = action.get('operation', '')
            
            if operation not in self.tool_constraints:
                continue
            
            constraints = self.tool_constraints[operation]
            tool_usage_counts[operation] += 1
            
            # Check usage limit
            if tool_usage_counts[operation] > constraints['max_uses_per_trajectory']:
                total_penalty -= self.base_penalty
                violations[f'{operation}_overuse'] += 1
            
            # Check prerequisites
            if constraints['prerequisite'] and not prerequisites_met[constraints['prerequisite']]:
                total_penalty -= self.base_penalty
                violations[f'{operation}_missing_prerequisite'] += 1
            
            # Check specific constraints
            if operation == 'TRACK_OBJECT':
                if context.get('input_type') != 'video':
                    # Severe violation: tracking on static image
                    total_penalty -= self.base_penalty * self.severe_penalty_multiplier
                    violations['track_on_static_image'] += 1
            
            elif operation == 'GET_PROPERTIES':
                if 'SEGMENT_OBJECT_AT' not in prerequisites_met:
                    total_penalty -= self.base_penalty
                    violations['properties_without_segmentation'] += 1
            
            # Check parameter validity
            penalty, param_violations = self._check_parameters(action)
            total_penalty += penalty
            for key, count in param_violations.items():
                violations[key] += count
            
            # Mark operation as done
            prerequisites_met[operation] = True
        
        # Check for missing essential operations
        if 'answer' not in [a.get('operation', '').lower() for a in trajectory]:
            total_penalty -= self.base_penalty
            violations['missing_answer'] += 1
        
        return total_penalty, dict(violations)
    
    def _check_parameters(self, action: Dict[str, Any]) -> Tuple[float, Dict[str, int]]:
        """Check parameter validity for an action."""
        penalty = 0.0
        violations = defaultdict(int)
        
        operation = action.get('operation', '')
        args = action.get('arguments', {})
        
        if operation in ['SEGMENT_OBJECT_AT', 'ZOOM_IN']:
            # Check coordinate bounds
            x = args.get('x', 0)
            y = args.get('y', 0)
            
            # Assuming normalized coordinates [0, 1]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                penalty -= self.base_penalty * 0.5
                violations['out_of_bounds_coordinates'] += 1
        
        return penalty, violations


class NormalizedRewardOrchestrator:
    """
    Central orchestrator with reward normalization and curriculum support.
    
    Combines all reward components with proper scaling and curriculum-based weighting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base weights
        self.base_weights = {
            'task': config.get('task_reward_weight', 1.0),
            'curiosity': config.get('curiosity_reward_weight', 0.3),
            'coherence': config.get('coherence_reward_weight', 0.2)
        }
        
        # Initialize component modules
        self.curiosity_module = PerformanceAwareCuriosityModule(
            beta=config.get('curiosity_beta', 0.2),
            eta=config.get('curiosity_eta', 0.5),
            cache_size=config.get('curiosity_cache_size', 1000)
        )
        
        self.coherence_analyzer = EnhancedTrajectoryCoherenceAnalyzer(
            coherence_threshold=config.get('coherence_threshold', 0.7),
            repetition_penalty=config.get('repetition_penalty', 0.5),
            sequence_bonus=config.get('sequence_bonus', 0.2)
        )
        
        self.penalty_system = ToolMisusePenaltySystem(
            base_penalty=abs(config.get('tool_misuse_penalty', 0.1))
        )
        
        # Normalization settings
        self.normalize = config.get('normalize_rewards', True)
        self.clip_value = config.get('reward_clip_value', 10.0)
        
        # Running statistics for normalization
        self.running_stats = {
            'task': RunningStats(),
            'curiosity': RunningStats(),
            'coherence': RunningStats()
        }
        
        # Curriculum settings
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_stages = config.get('curriculum_stages', [])
        self.current_step = 0
        
        logger.info("Reward orchestrator initialized with normalization and curriculum support")
    
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
            final_answer: Model's final answer
            ground_truth: Ground truth answer
            state_embeddings: Optional state embeddings
            context: Additional context
            
        Returns:
            Dictionary with all reward information
        """
        # Task reward (simple exact match for now)
        task_reward = 1.0 if str(final_answer).strip() == str(ground_truth).strip() else 0.0
        
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
                
                # Compute curiosity for this step
                reward, metrics = self.curiosity_module.compute_curiosity_reward(
                    state_embeddings[i].unsqueeze(0) if state_embeddings[i].dim() == 1 else state_embeddings[i],
                    action_embedding.unsqueeze(0) if action_embedding.dim() == 1 else action_embedding,
                    state_embeddings[i+1].unsqueeze(0) if state_embeddings[i+1].dim() == 1 else state_embeddings[i+1]
                )
                
                curiosity_rewards.append(reward.mean().item())
                curiosity_metrics.update(metrics)
            
            curiosity_reward = np.mean(curiosity_rewards) if curiosity_rewards else 0.0
        
        # Coherence reward
        coherence_reward, coherence_metrics = self.coherence_analyzer.compute_coherence_reward(
            trajectory,
            state_embeddings
        )
        
        # Tool misuse penalties
        tool_penalty, violations = self.penalty_system.calculate_penalties(
            trajectory,
            context or {}
        )
        
        # Update running statistics
        self.running_stats['task'].update(task_reward)
        self.running_stats['curiosity'].update(curiosity_reward)
        self.running_stats['coherence'].update(coherence_reward)
        
        # Normalize if enabled
        if self.normalize:
            task_norm = self.running_stats['task'].normalize(task_reward)
            curiosity_norm = self.running_stats['curiosity'].normalize(curiosity_reward)
            coherence_norm = self.running_stats['coherence'].normalize(coherence_reward)
        else:
            task_norm = task_reward
            curiosity_norm = curiosity_reward
            coherence_norm = coherence_reward
        
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
        total_reward = np.clip(total_reward, -self.clip_value, self.clip_value)
        
        # Create comprehensive result
        result = {
            'total': total_reward,
            'components': {
                'task': {'raw': task_reward, 'normalized': task_norm, 'weight': weights['task']},
                'curiosity': {'raw': curiosity_reward, 'normalized': curiosity_norm, 'weight': weights['curiosity']},
                'coherence': {'raw': coherence_reward, 'normalized': coherence_norm, 'weight': weights['coherence']},
                'penalty': tool_penalty
            },
            'metrics': {
                'curiosity': curiosity_metrics,
                'coherence': coherence_metrics,
                'violations': violations
            },
            'statistics': {
                'task_mean': self.running_stats['task'].mean,
                'task_std': self.running_stats['task'].std,
                'curiosity_mean': self.running_stats['curiosity'].mean,
                'curiosity_std': self.running_stats['curiosity'].std,
                'coherence_mean': self.running_stats['coherence'].mean,
                'coherence_std': self.running_stats['coherence'].std
            },
            'curriculum': {
                'step': self.current_step,
                'weights': weights
            }
        }
        
        return result
    
    def _create_action_embedding(self, action: Dict[str, Any]) -> torch.Tensor:
        """Create action embedding from action dictionary."""
        # Create a structured embedding based on action type
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding = torch.zeros(128).to(device)
        
        # FIX: Support both 'operation' and 'action' keys for compatibility
        operation = action.get('operation', action.get('action', ''))
        
        # One-hot encode operation type (first 10 dimensions)
        operations = ['SEGMENT_OBJECT_AT', 'READ_TEXT', 'TRACK_OBJECT', 'ZOOM_IN', 'GET_PROPERTIES', 'THINK']
        if operation in operations:
            embedding[operations.index(operation)] = 1.0
        
        # Encode arguments (next dimensions)
        # FIX: Support both 'arguments' and direct 'coordinates'
        args = action.get('arguments', {})
        coordinates = action.get('coordinates', [])
        
        if 'x' in args and 'y' in args:
            embedding[10] = args['x']
            embedding[11] = args['y']
        elif len(coordinates) >= 2:
            embedding[10] = coordinates[0] / 1000.0  # Normalize coordinates
            embedding[11] = coordinates[1] / 1000.0
        
        # Add some random noise for diversity
        embedding[20:] = torch.randn(108).to(device) * 0.1
        
        return embedding
    
    def _get_curriculum_weights(self) -> Dict[str, float]:
        """Get curriculum-based weights for current step."""
        if not self.use_curriculum or not self.curriculum_stages:
            return self.base_weights
        
        # Find applicable stage
        current_weights = self.base_weights.copy()
        
        for stage in self.curriculum_stages:
            if self.current_step >= stage.get('step', 0):
                stage_weights = stage.get('weights', {})
                current_weights.update({
                    'task': stage_weights.get('task', current_weights['task']),
                    'curiosity': stage_weights.get('curiosity', current_weights['curiosity']),
                    'coherence': stage_weights.get('coherence', current_weights['coherence'])
                })
        
        return current_weights
    
    def update_step(self, step: int):
        """Update current training step for curriculum."""
        self.current_step = step


class RunningStats:
    """Helper class for maintaining running statistics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size  # FIX: Added window_size attribute
        self.window = deque(maxlen=window_size)
        self.values = []  # FIX: Added values attribute for test compatibility
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
    
    def update(self, value: float):
        """Update statistics with new value."""
        self.window.append(value)
        self.values.append(value)  # FIX: Also append to values list
        self.count += 1
        
        if len(self.window) > 1:
            self.mean = np.mean(self.window)
            self.std = np.std(self.window) + 1e-8
    
    def normalize(self, value: float) -> float:
        """Normalize value using running statistics."""
        if self.count < 10:  # Not enough data
            return value
        
        return (value - self.mean) / self.std