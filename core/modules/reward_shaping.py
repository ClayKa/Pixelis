"""
Reward Shaping Module

Implements multi-component reward calculation including task rewards,
curiosity rewards, coherence rewards, and tool usage penalties.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from ..data_structures import Trajectory, Action, ActionType, RewardComponents

logger = logging.getLogger(__name__)


class CuriosityRewardModule(nn.Module):
    """
    Implements curiosity-driven reward based on prediction error.
    
    Uses a lightweight dynamics model to predict next state and
    rewards based on prediction error (encouraging exploration).
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        beta: float = 0.2,
        eta: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize curiosity module.
        
        Args:
            state_dim: Dimension of state embeddings
            action_dim: Dimension of action embeddings
            hidden_dim: Hidden layer dimension
            beta: Weight for forward model loss
            eta: Scaling factor for intrinsic reward
            device: Device for computation
        """
        super().__init__()
        
        self.beta = beta
        self.eta = eta
        self.device = device
        
        # Forward dynamics model: predicts next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)
        
        # Inverse model: predicts action given current and next state
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # Feature encoder for states
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(device)
        
        # Initialize with small weights for stability
        for module in [self.forward_model, self.inverse_model, self.feature_encoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate curiosity reward.
        
        Args:
            state: Current state embedding
            action: Action embedding
            next_state: Next state embedding
            
        Returns:
            Tuple of (curiosity_reward, loss_dict)
        """
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Forward model: predict next state
        state_action = torch.cat([state_feat, action], dim=-1)
        predicted_next_state = self.forward_model(state_action)
        
        # Inverse model: predict action
        state_pair = torch.cat([state_feat, next_state_feat], dim=-1)
        predicted_action = self.inverse_model(state_pair)
        
        # Calculate losses
        forward_loss = F.mse_loss(predicted_next_state, next_state_feat.detach())
        inverse_loss = F.mse_loss(predicted_action, action.detach())
        
        # Intrinsic reward is the prediction error (encourages exploration)
        with torch.no_grad():
            prediction_error = F.mse_loss(
                predicted_next_state,
                next_state_feat,
                reduction='none'
            ).mean(dim=-1)
            curiosity_reward = self.eta * prediction_error
        
        # Combined loss for training
        total_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss
        
        loss_dict = {
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_loss': total_loss
        }
        
        return curiosity_reward, loss_dict
    
    def calculate_trajectory_curiosity(
        self,
        trajectory: Trajectory,
        state_embeddings: List[torch.Tensor]
    ) -> float:
        """
        Calculate curiosity reward for entire trajectory.
        
        Args:
            trajectory: Reasoning trajectory
            state_embeddings: List of state embeddings
            
        Returns:
            Average curiosity reward
        """
        if len(state_embeddings) < 2:
            return 0.0
        
        curiosity_rewards = []
        
        for i in range(len(state_embeddings) - 1):
            # Get state transition
            state = state_embeddings[i].to(self.device)
            next_state = state_embeddings[i + 1].to(self.device)
            
            # Create action embedding (simplified)
            if i < len(trajectory.actions):
                action = self._encode_action(trajectory.actions[i])
            else:
                action = torch.zeros(128).to(self.device)
            
            # Calculate curiosity
            with torch.no_grad():
                reward, _ = self.forward(
                    state.unsqueeze(0),
                    action.unsqueeze(0),
                    next_state.unsqueeze(0)
                )
                curiosity_rewards.append(reward.item())
        
        return np.mean(curiosity_rewards) if curiosity_rewards else 0.0
    
    def _encode_action(self, action: Action) -> torch.Tensor:
        """
        Encode action to embedding.
        
        Args:
            action: Action object
            
        Returns:
            Action embedding
        """
        # Simplified action encoding
        # In practice, this would use a proper encoder
        embedding = torch.randn(128)
        
        # Add some structure based on action type
        if action.type == ActionType.VISUAL_OPERATION:
            embedding[0] = 1.0
        elif action.type == ActionType.REASONING:
            embedding[1] = 1.0
        elif action.type == ActionType.ANSWER:
            embedding[2] = 1.0
        
        return embedding


class TrajectoryCoherenceAnalyzer:
    """
    Analyzes trajectory coherence and logical flow.
    
    Rewards coherent reasoning and penalizes repetitive or
    contradictory actions.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        repetition_penalty: float = 0.5,
        min_trajectory_length: int = 2
    ):
        """
        Initialize coherence analyzer.
        
        Args:
            coherence_threshold: Minimum coherence score
            repetition_penalty: Penalty for repetitive actions
            min_trajectory_length: Minimum length for analysis
        """
        self.coherence_threshold = coherence_threshold
        self.repetition_penalty = repetition_penalty
        self.min_trajectory_length = min_trajectory_length
    
    def calculate_coherence_reward(
        self,
        trajectory: Trajectory,
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> float:
        """
        Calculate coherence reward for trajectory.
        
        Args:
            trajectory: Reasoning trajectory
            embeddings: Optional embeddings for semantic similarity
            
        Returns:
            Coherence reward value
        """
        if trajectory.get_trajectory_length() < self.min_trajectory_length:
            return 0.0
        
        rewards = []
        
        # Check for repetitions
        if trajectory.has_repetitions():
            rewards.append(-self.repetition_penalty)
        
        # Check logical flow
        flow_score = self._analyze_logical_flow(trajectory)
        rewards.append(flow_score)
        
        # Check semantic coherence if embeddings provided
        if embeddings and len(embeddings) > 1:
            coherence_score = self._calculate_semantic_coherence(embeddings)
            rewards.append(coherence_score)
        
        # Check tool usage pattern
        tool_pattern_score = self._analyze_tool_usage_pattern(trajectory)
        rewards.append(tool_pattern_score)
        
        return np.mean(rewards) if rewards else 0.0
    
    def _analyze_logical_flow(self, trajectory: Trajectory) -> float:
        """
        Analyze logical flow of actions.
        
        Args:
            trajectory: Reasoning trajectory
            
        Returns:
            Flow score
        """
        score = 0.0
        actions = trajectory.actions
        
        # Check for logical progression
        for i in range(1, len(actions)):
            prev_action = actions[i-1]
            curr_action = actions[i]
            
            # Penalize immediate repetition of same operation
            if (prev_action.type == curr_action.type and
                prev_action.operation == curr_action.operation):
                score -= 0.2
            
            # Reward diverse action types
            if prev_action.type != curr_action.type:
                score += 0.1
            
            # Reward reasoning after visual operations
            if (prev_action.type == ActionType.VISUAL_OPERATION and
                curr_action.type == ActionType.REASONING):
                score += 0.2
        
        # Normalize by trajectory length
        if len(actions) > 1:
            score /= (len(actions) - 1)
        
        return np.clip(score, -1.0, 1.0)
    
    def _calculate_semantic_coherence(
        self,
        embeddings: List[torch.Tensor]
    ) -> float:
        """
        Calculate semantic coherence using embeddings.
        
        Args:
            embeddings: List of state embeddings
            
        Returns:
            Coherence score
        """
        if len(embeddings) < 2:
            return 0.0
        
        similarities = []
        
        for i in range(1, len(embeddings)):
            # Calculate cosine similarity
            prev_emb = embeddings[i-1]
            curr_emb = embeddings[i]
            
            if isinstance(prev_emb, torch.Tensor):
                prev_emb = prev_emb.detach().cpu().numpy()
            if isinstance(curr_emb, torch.Tensor):
                curr_emb = curr_emb.detach().cpu().numpy()
            
            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb) + 1e-8
            )
            similarities.append(similarity)
        
        # Reward moderate similarity (not too high, not too low)
        avg_similarity = np.mean(similarities)
        
        if avg_similarity < 0.3:
            # Too dissimilar - lacks coherence
            return -0.5
        elif avg_similarity > 0.9:
            # Too similar - lacks progression
            return -0.3
        else:
            # Good coherence
            return (avg_similarity - 0.3) / 0.6  # Normalize to [0, 1]
    
    def _analyze_tool_usage_pattern(self, trajectory: Trajectory) -> float:
        """
        Analyze pattern of tool usage.
        
        Args:
            trajectory: Reasoning trajectory
            
        Returns:
            Pattern score
        """
        tool_counts = trajectory.get_tool_usage_count()
        
        if not tool_counts:
            return 0.0
        
        score = 0.0
        
        # Reward diverse tool usage
        num_unique_tools = len(tool_counts)
        score += min(num_unique_tools * 0.1, 0.5)
        
        # Penalize excessive use of single tool
        max_usage = max(tool_counts.values())
        if max_usage > 5:
            score -= (max_usage - 5) * 0.1
        
        # Check for logical tool sequencing
        actions = trajectory.actions
        for i in range(1, len(actions)):
            if actions[i-1].type == ActionType.VISUAL_OPERATION:
                # Common good patterns
                prev_op = actions[i-1].operation
                curr_op = actions[i].operation if i < len(actions) else None
                
                # SEGMENT -> GET_PROPERTIES is logical
                if prev_op == "SEGMENT_OBJECT_AT" and curr_op == "GET_PROPERTIES":
                    score += 0.2
                
                # ZOOM_IN -> READ_TEXT is logical
                if prev_op == "ZOOM_IN" and curr_op == "READ_TEXT":
                    score += 0.2
        
        return np.clip(score, -1.0, 1.0)


class RewardOrchestrator:
    """
    Central reward orchestrator that combines all reward components.
    
    Manages task rewards, curiosity rewards, coherence rewards,
    and tool usage penalties with normalization and curriculum.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward orchestrator.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        
        # Component weights
        self.task_weight = config.get('task_reward_weight', 1.0)
        self.curiosity_weight = config.get('curiosity_reward_weight', 0.3)
        self.coherence_weight = config.get('coherence_reward_weight', 0.2)
        
        # Tool usage penalties
        self.tool_misuse_penalty = config.get('tool_misuse_penalty', -0.1)
        self.excessive_tool_threshold = config.get('excessive_tool_use_threshold', 10)
        self.excessive_tool_penalty = config.get('excessive_tool_use_penalty', -0.2)
        
        # Normalization
        self.normalize = config.get('normalize_rewards', True)
        self.clip_value = config.get('reward_clip_value', 10.0)
        
        # Curriculum
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_stages = config.get('curriculum_stages', [])
        self.current_step = 0
        
        # Initialize sub-modules
        self.curiosity_module = CuriosityRewardModule(
            beta=config.get('curiosity_beta', 0.2),
            eta=config.get('curiosity_eta', 0.5)
        )
        
        self.coherence_analyzer = TrajectoryCoherenceAnalyzer(
            coherence_threshold=config.get('coherence_threshold', 0.7),
            repetition_penalty=config.get('repetition_penalty', 0.5)
        )
        
        # Running statistics for normalization
        self.reward_stats = {
            'task': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'curiosity': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'coherence': {'mean': 0.0, 'std': 1.0, 'count': 0}
        }
        
        logger.info("Reward orchestrator initialized")
    
    def calculate_reward(
        self,
        trajectory: Trajectory,
        final_answer: Any,
        ground_truth: Any,
        state_embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all reward components.
        
        Args:
            trajectory: Reasoning trajectory
            final_answer: Model's final answer
            ground_truth: Ground truth answer (or pseudo-label)
            state_embeddings: Optional state embeddings
            
        Returns:
            Dictionary with reward components and total
        """
        # Calculate individual components
        task_reward = self._calculate_task_reward(final_answer, ground_truth)
        
        curiosity_reward = self.curiosity_module.calculate_trajectory_curiosity(
            trajectory,
            state_embeddings or []
        )
        
        coherence_reward = self.coherence_analyzer.calculate_coherence_reward(
            trajectory,
            state_embeddings
        )
        
        tool_penalty = self._calculate_tool_penalty(trajectory)
        
        # Update statistics
        self._update_statistics('task', task_reward)
        self._update_statistics('curiosity', curiosity_reward)
        self._update_statistics('coherence', coherence_reward)
        
        # Normalize if enabled
        if self.normalize:
            task_reward = self._normalize_reward('task', task_reward)
            curiosity_reward = self._normalize_reward('curiosity', curiosity_reward)
            coherence_reward = self._normalize_reward('coherence', coherence_reward)
        
        # Apply curriculum weights
        weights = self._get_curriculum_weights()
        
        # Calculate total reward
        total_reward = (
            weights['task'] * task_reward +
            weights['curiosity'] * curiosity_reward +
            weights['coherence'] * coherence_reward +
            tool_penalty  # Penalty is always applied
        )
        
        # Clip total reward
        total_reward = np.clip(total_reward, -self.clip_value, self.clip_value)
        
        # Create reward components object
        components = RewardComponents(
            task_reward=task_reward,
            curiosity_reward=curiosity_reward,
            coherence_reward=coherence_reward,
            tool_penalty=tool_penalty,
            total_reward=total_reward,
            metadata={
                'weights': weights,
                'normalized': self.normalize,
                'curriculum_step': self.current_step
            }
        )
        
        return components.to_dict()
    
    def _calculate_task_reward(
        self,
        final_answer: Any,
        ground_truth: Any
    ) -> float:
        """
        Calculate task completion reward.
        
        Args:
            final_answer: Model's answer
            ground_truth: Ground truth
            
        Returns:
            Task reward
        """
        # Simple exact match for now
        # In practice, this would be task-specific
        if final_answer == ground_truth:
            return 1.0
        
        # Partial credit based on similarity
        if isinstance(final_answer, str) and isinstance(ground_truth, str):
            # Simple character-level similarity
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, final_answer, ground_truth).ratio()
            return similarity * 0.5  # Partial credit
        
        return 0.0
    
    def _calculate_tool_penalty(self, trajectory: Trajectory) -> float:
        """
        Calculate penalty for tool misuse.
        
        Args:
            trajectory: Reasoning trajectory
            
        Returns:
            Tool usage penalty
        """
        penalty = 0.0
        
        # Check for excessive tool usage
        tool_counts = trajectory.get_tool_usage_count()
        total_tools = sum(tool_counts.values())
        
        if total_tools > self.excessive_tool_threshold:
            penalty += self.excessive_tool_penalty * (total_tools - self.excessive_tool_threshold)
        
        # Check for repetitive tool usage
        if trajectory.has_repetitions():
            penalty += self.tool_misuse_penalty
        
        # Check for invalid tool sequences
        actions = trajectory.actions
        for i in range(1, len(actions)):
            if actions[i-1].type == ActionType.VISUAL_OPERATION:
                # Penalize certain bad patterns
                prev_op = actions[i-1].operation
                curr_op = actions[i].operation if i < len(actions) else None
                
                # Repetitive zooming
                if prev_op == "ZOOM_IN" and curr_op == "ZOOM_IN":
                    penalty += self.tool_misuse_penalty * 0.5
        
        return penalty
    
    def _get_curriculum_weights(self) -> Dict[str, float]:
        """
        Get current curriculum weights.
        
        Returns:
            Dictionary of component weights
        """
        if not self.use_curriculum:
            return {
                'task': self.task_weight,
                'curiosity': self.curiosity_weight,
                'coherence': self.coherence_weight
            }
        
        # Find appropriate stage
        weights = {
            'task': self.task_weight,
            'curiosity': 0.0,
            'coherence': 0.0
        }
        
        for stage in self.curriculum_stages:
            if self.current_step >= stage.get('step', 0):
                stage_weights = stage.get('weights', {})
                weights['task'] = stage_weights.get('task', weights['task'])
                weights['curiosity'] = stage_weights.get('curiosity', weights['curiosity'])
                weights['coherence'] = stage_weights.get('coherence', weights['coherence'])
        
        return weights
    
    def _update_statistics(self, component: str, value: float):
        """
        Update running statistics for normalization.
        
        Args:
            component: Reward component name
            value: Reward value
        """
        stats = self.reward_stats[component]
        stats['count'] += 1
        
        # Update mean (exponential moving average)
        alpha = 0.01
        stats['mean'] = (1 - alpha) * stats['mean'] + alpha * value
        
        # Update std (exponential moving average)
        variance = (value - stats['mean']) ** 2
        stats['std'] = np.sqrt((1 - alpha) * stats['std']**2 + alpha * variance)
        
        # Ensure minimum std
        stats['std'] = max(stats['std'], 0.1)
    
    def _normalize_reward(self, component: str, value: float) -> float:
        """
        Normalize reward using z-score normalization.
        
        Args:
            component: Reward component name
            value: Raw reward value
            
        Returns:
            Normalized reward
        """
        stats = self.reward_stats[component]
        
        if stats['count'] < 10:
            # Not enough samples for reliable normalization
            return value
        
        # Z-score normalization
        normalized = (value - stats['mean']) / stats['std']
        
        # Clip to reasonable range
        return np.clip(normalized, -3.0, 3.0)
    
    def step(self):
        """Increment curriculum step."""
        self.current_step += 1
    
    def reset_statistics(self):
        """Reset reward statistics."""
        for component in self.reward_stats:
            self.reward_stats[component] = {
                'mean': 0.0,
                'std': 1.0,
                'count': 0
            }