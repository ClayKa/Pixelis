"""
Dynamics Model Module

Defines the auxiliary dynamics model required by the curiosity reward module.
This model predicts state transitions for exploration-based learning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DynamicsModel(nn.Module):
    """
    Lightweight dynamics model for state prediction.
    
    Used by the curiosity module to predict next states and
    generate intrinsic rewards based on prediction errors.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize dynamics model.
        
        Args:
            state_dim: Dimension of state representations
            action_dim: Dimension of action representations
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            device: Device for computation
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.device = device
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined processor
        combined_dim = hidden_dim + hidden_dim // 2
        
        layers = []
        for i in range(num_layers):
            in_dim = combined_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.processor = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, state_dim)
        
        # Residual gate (learnable)
        if use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(device)
        
        logger.info(f"Dynamics model initialized with {self.count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Predicted next state [batch_size, state_dim]
        """
        # Encode state and action
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        
        # Combine representations
        combined = torch.cat([state_encoded, action_encoded], dim=-1)
        
        # Process through layers
        processed = self.processor(combined)
        
        # Project to state space
        state_delta = self.output_projection(processed)
        
        # Apply residual connection if enabled
        if self.use_residual:
            next_state = state + self.residual_gate * state_delta
        else:
            next_state = state_delta
        
        return next_state
    
    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        action_sequence: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Predict a sequence of states given initial state and actions.
        
        Args:
            initial_state: Initial state tensor
            action_sequence: List of action tensors
            
        Returns:
            List of predicted states
        """
        states = [initial_state]
        current_state = initial_state
        
        for action in action_sequence:
            with torch.no_grad():
                next_state = self.forward(current_state, action)
                states.append(next_state)
                current_state = next_state
        
        return states
    
    def compute_prediction_error(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute prediction error for curiosity reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state
            reduction: How to reduce error ('mean', 'sum', 'none')
            
        Returns:
            Prediction error
        """
        predicted_next = self.forward(state, action)
        
        if reduction == 'none':
            error = F.mse_loss(predicted_next, next_state, reduction='none')
        elif reduction == 'sum':
            error = F.mse_loss(predicted_next, next_state, reduction='sum')
        else:  # mean
            error = F.mse_loss(predicted_next, next_state, reduction='mean')
        
        return error
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InverseModel(nn.Module):
    """
    Inverse dynamics model that predicts actions from state transitions.
    
    Used to learn meaningful action representations and ensure
    the state encoder captures action-relevant information.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        Initialize inverse model.
        
        Args:
            state_dim: Dimension of state representations
            action_dim: Dimension of action representations
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            device: Device for computation
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Process concatenated states
        layers = []
        for i in range(num_layers):
            in_dim = state_dim * 2 if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.processor = nn.Sequential(*layers)
        
        # Output projection to action space
        self.action_projection = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict action that caused state transition.
        
        Args:
            state: Current state [batch_size, state_dim]
            next_state: Next state [batch_size, state_dim]
            
        Returns:
            Predicted action [batch_size, action_dim]
        """
        # Concatenate states
        state_pair = torch.cat([state, next_state], dim=-1)
        
        # Process through network
        processed = self.processor(state_pair)
        
        # Project to action space
        action = self.action_projection(processed)
        
        return action


class StateEncoder(nn.Module):
    """
    Encoder for state representations.
    
    Transforms raw states into meaningful representations
    for dynamics modeling and curiosity calculation.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize state encoder.
        
        Args:
            input_dim: Input state dimension
            output_dim: Output representation dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            use_layer_norm: Whether to use layer normalization
            device: Device for computation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.device = device
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:  # Not last layer
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
        # Output normalization
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(output_dim)
        else:
            self.output_norm = nn.Identity()
        
        self.to(device)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to representation.
        
        Args:
            state: Input state tensor
            
        Returns:
            Encoded representation
        """
        encoded = self.encoder(state)
        normalized = self.output_norm(encoded)
        return normalized


class ActionEncoder(nn.Module):
    """
    Encoder for action representations.
    
    Transforms discrete or continuous actions into
    dense representations for dynamics modeling.
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        continuous_dim: Optional[int] = None,
        output_dim: int = 128,
        hidden_dim: int = 256,
        device: str = "cuda"
    ):
        """
        Initialize action encoder.
        
        Args:
            vocab_size: Size of action vocabulary (for discrete actions)
            continuous_dim: Dimension of continuous actions
            output_dim: Output representation dimension
            hidden_dim: Hidden layer dimension
            device: Device for computation
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.continuous_dim = continuous_dim
        self.output_dim = output_dim
        self.device = device
        
        if vocab_size is not None:
            # Discrete actions - use embedding
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif continuous_dim is not None:
            # Continuous actions - use MLP
            self.processor = nn.Sequential(
                nn.Linear(continuous_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError("Must specify either vocab_size or continuous_dim")
        
        self.to(device)
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Encode action to representation.
        
        Args:
            action: Action tensor (indices for discrete, values for continuous)
            
        Returns:
            Encoded action representation
        """
        if self.vocab_size is not None:
            # Discrete action
            embedded = self.embedding(action.long())
            encoded = self.processor(embedded)
        else:
            # Continuous action
            encoded = self.processor(action)
        
        return encoded


class CuriosityDynamicsModel(nn.Module):
    """
    Complete dynamics model for curiosity-driven learning.
    
    Combines forward model, inverse model, and encoders
    for computing intrinsic rewards.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 128,
        encoded_dim: int = 256,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize complete curiosity dynamics model.
        
        Args:
            state_dim: Raw state dimension
            action_dim: Raw action dimension
            encoded_dim: Encoded representation dimension
            config: Additional configuration
        """
        super().__init__()
        
        config = config or {}
        device = config.get('device', 'cuda')
        
        # State encoder
        self.state_encoder = StateEncoder(
            input_dim=state_dim,
            output_dim=encoded_dim,
            hidden_dim=config.get('encoder_hidden_dim', 512),
            device=device
        )
        
        # Action encoder
        self.action_encoder = ActionEncoder(
            continuous_dim=action_dim,
            output_dim=encoded_dim // 2,
            hidden_dim=config.get('action_hidden_dim', 256),
            device=device
        )
        
        # Forward dynamics model
        self.forward_model = DynamicsModel(
            state_dim=encoded_dim,
            action_dim=encoded_dim // 2,
            hidden_dim=config.get('dynamics_hidden_dim', 256),
            device=device
        )
        
        # Inverse model
        self.inverse_model = InverseModel(
            state_dim=encoded_dim,
            action_dim=encoded_dim // 2,
            hidden_dim=config.get('inverse_hidden_dim', 256),
            device=device
        )
        
        # Loss weights
        self.forward_weight = config.get('forward_weight', 0.2)
        self.inverse_weight = 1.0 - self.forward_weight
        
        # Intrinsic reward scaling
        self.reward_scale = config.get('intrinsic_reward_scale', 0.1)
        
        self.device = device
    
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute intrinsic curiosity reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Tuple of (intrinsic_reward, loss_dict)
        """
        # Encode states
        state_encoded = self.state_encoder(state)
        next_state_encoded = self.state_encoder(next_state)
        
        # Encode action
        action_encoded = self.action_encoder(action)
        
        # Forward prediction
        predicted_next = self.forward_model(state_encoded, action_encoded)
        forward_error = F.mse_loss(predicted_next, next_state_encoded.detach())
        
        # Inverse prediction
        predicted_action = self.inverse_model(state_encoded, next_state_encoded)
        inverse_error = F.mse_loss(predicted_action, action_encoded.detach())
        
        # Intrinsic reward is the prediction error
        with torch.no_grad():
            reward = self.reward_scale * forward_error
        
        # Combined loss for training
        total_loss = (
            self.forward_weight * forward_error +
            self.inverse_weight * inverse_error
        )
        
        loss_dict = {
            'forward_error': forward_error,
            'inverse_error': inverse_error,
            'total_loss': total_loss
        }
        
        return reward, loss_dict