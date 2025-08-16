"""
Core Data Structures Module

Defines the core data structures used throughout the Pixelis system with
strict type validation using Python dataclasses. These ensure data consistency
and type safety across all modules and processes.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import torch
import numpy as np
import json
from enum import Enum


class ExperienceStatus(Enum):
    """Status of an experience in the buffer."""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ActionType(Enum):
    """Types of actions in a trajectory."""
    VISUAL_OPERATION = "visual_operation"
    REASONING = "reasoning"
    ANSWER = "answer"
    CORRECTION = "correction"


@dataclass
class Action:
    """
    Represents a single action in a reasoning trajectory.
    
    Attributes:
        type: Type of action (visual operation, reasoning, etc.)
        operation: Name of the operation (e.g., 'SEGMENT_OBJECT_AT')
        arguments: Arguments passed to the operation
        result: Result from the operation
        confidence: Confidence score for this action
        timestamp: When the action was taken
    """
    type: ActionType
    operation: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate action data after initialization."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "type": self.type.value,
            "operation": self.operation,
            "arguments": self.arguments,
            "result": str(self.result) if self.result is not None else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create action from dictionary."""
        return cls(
            type=ActionType(data["type"]),
            operation=data["operation"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            confidence=data.get("confidence", 1.0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )


@dataclass
class Trajectory:
    """
    Represents a complete reasoning trajectory.
    
    Attributes:
        actions: List of actions taken
        final_answer: The final answer produced
        total_reward: Total reward received
        metadata: Additional trajectory metadata
    """
    actions: List[Action] = field(default_factory=list)
    final_answer: Optional[Any] = None
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: Action):
        """Add an action to the trajectory."""
        self.actions.append(action)
    
    def get_tool_usage_count(self) -> Dict[str, int]:
        """Get count of each tool used in trajectory."""
        tool_counts = {}
        for action in self.actions:
            if action.type == ActionType.VISUAL_OPERATION:
                tool_counts[action.operation] = tool_counts.get(action.operation, 0) + 1
        return tool_counts
    
    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory."""
        return len(self.actions)
    
    def has_repetitions(self, threshold: int = 2) -> bool:
        """Check if trajectory has repetitive actions."""
        recent_ops = []
        for action in self.actions:
            if action.type == ActionType.VISUAL_OPERATION:
                recent_ops.append(action.operation)
                if len(recent_ops) > threshold:
                    recent_ops.pop(0)
                if len(recent_ops) == threshold and len(set(recent_ops)) == 1:
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "actions": [action.to_dict() for action in self.actions],
            "final_answer": self.final_answer,
            "total_reward": self.total_reward,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create trajectory from dictionary."""
        return cls(
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            final_answer=data.get("final_answer"),
            total_reward=data.get("total_reward", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class Experience:
    """
    Represents an experience in the experience buffer.
    
    Attributes:
        experience_id: Unique identifier for the experience
        image_features: Visual features of the image
        question_text: The question/prompt text
        trajectory: The reasoning trajectory taken
        model_confidence: Model's confidence in the answer
        timestamp: When the experience was created
        status: Current status of the experience
        embeddings: Cached embeddings for similarity search
        priority: Priority score for sampling
        retrieval_count: Number of times retrieved from buffer
        success_count: Number of successful uses in voting
    """
    experience_id: str
    image_features: Union[torch.Tensor, np.ndarray]
    question_text: str
    trajectory: Trajectory
    model_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    status: ExperienceStatus = ExperienceStatus.PENDING
    embeddings: Optional[Dict[str, torch.Tensor]] = None
    priority: float = 1.0
    retrieval_count: int = 0
    success_count: int = 0
    
    def __post_init__(self):
        """Validate experience data after initialization."""
        if self.model_confidence < 0 or self.model_confidence > 1:
            raise ValueError(f"Model confidence must be between 0 and 1, got {self.model_confidence}")
        
        if self.priority < 0:
            raise ValueError(f"Priority must be non-negative, got {self.priority}")
        
        # Generate ID if not provided
        if not self.experience_id:
            import uuid
            self.experience_id = str(uuid.uuid4())
    
    def update_priority(self, uncertainty: float, reward: float, decay_factor: float = 0.95):
        """
        Update the priority score based on uncertainty and reward.
        
        Args:
            uncertainty: Current uncertainty (1 - confidence)
            reward: Reward received
            decay_factor: Decay factor for aging
        """
        # Priority based on uncertainty and reward
        base_priority = uncertainty * 0.5 + abs(reward) * 0.5
        
        # Apply decay based on age
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        age_decay = decay_factor ** age_hours
        
        self.priority = base_priority * age_decay
    
    def update_usage_stats(self, was_successful: bool):
        """
        Update usage statistics after retrieval.
        
        Args:
            was_successful: Whether the experience led to a successful prediction
        """
        self.retrieval_count += 1
        if was_successful:
            self.success_count += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate from counts."""
        if self.retrieval_count == 0:
            return 0.0
        return self.success_count / self.retrieval_count
    
    def get_embedding(self, embedding_type: str = "combined") -> Optional[torch.Tensor]:
        """
        Get cached embedding of specified type.
        
        Args:
            embedding_type: Type of embedding ('visual', 'text', 'combined')
            
        Returns:
            Embedding tensor if available
        """
        if self.embeddings is None:
            return None
        return self.embeddings.get(embedding_type)
    
    def set_embedding(self, embedding: torch.Tensor, embedding_type: str = "combined"):
        """
        Set cached embedding.
        
        Args:
            embedding: Embedding tensor
            embedding_type: Type of embedding
        """
        if self.embeddings is None:
            self.embeddings = {}
        self.embeddings[embedding_type] = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary for serialization."""
        return {
            "experience_id": self.experience_id,
            "question_text": self.question_text,
            "trajectory": self.trajectory.to_dict(),
            "model_confidence": self.model_confidence,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "priority": self.priority,
            "retrieval_count": self.retrieval_count,
            "success_count": self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create experience from dictionary."""
        # Note: image_features and embeddings need special handling
        return cls(
            experience_id=data["experience_id"],
            image_features=None,  # Must be loaded separately
            question_text=data["question_text"],
            trajectory=Trajectory.from_dict(data["trajectory"]),
            model_confidence=data["model_confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=ExperienceStatus(data["status"]),
            embeddings=None,  # Must be loaded separately
            priority=data.get("priority", 1.0),
            retrieval_count=data.get("retrieval_count", 0),
            success_count=data.get("success_count", 0)
        )
    
    def get_input_ids(self) -> torch.Tensor:
        """
        Get tokenized input IDs for the question.
        Placeholder - actual implementation would use tokenizer.
        """
        # This would be implemented with actual tokenizer
        return torch.tensor([0])  # Placeholder
    
    def get_labels(self) -> torch.Tensor:
        """
        Get labels for training.
        Placeholder - actual implementation would process trajectory.
        """
        # This would be implemented based on trajectory
        return torch.tensor([0])  # Placeholder


@dataclass
class UpdateTask:
    """
    Represents a task for the update worker.
    
    Attributes:
        task_id: Unique identifier for the task
        experience: The experience to learn from
        reward_tensor: Multi-component reward tensor
        learning_rate: Adaptive learning rate for this update
        original_logits: Original model logits for KL calculation
        metadata: Additional task metadata
        created_at: When the task was created
        processed_at: When the task was processed
    """
    task_id: str
    experience: Experience
    reward_tensor: torch.Tensor
    learning_rate: float
    original_logits: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate update task data after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        
        # Generate ID if not provided
        if not self.task_id:
            import uuid
            self.task_id = str(uuid.uuid4())
    
    def mark_processed(self):
        """Mark the task as processed."""
        self.processed_at = datetime.now()
    
    def get_processing_time(self) -> Optional[float]:
        """
        Get processing time in seconds.
        
        Returns:
            Processing time if task is processed, None otherwise
        """
        if self.processed_at is None:
            return None
        return (self.processed_at - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert update task to dictionary."""
        return {
            "task_id": self.task_id,
            "experience_id": self.experience.experience_id,
            "learning_rate": self.learning_rate,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


@dataclass
class VotingResult:
    """
    Result from the temporal ensemble voting module.
    
    Attributes:
        final_answer: The consensus answer
        confidence: Confidence score for the answer
        provenance: Detailed provenance information including audit trail
    """
    final_answer: Any
    confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate voting result data and ensure required provenance fields."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Ensure required provenance fields exist
        required_fields = ['model_self_answer', 'retrieved_neighbors_count', 
                          'neighbor_answers', 'voting_strategy']
        for field in required_fields:
            if field not in self.provenance:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Missing required provenance field: {field}")
                self.provenance[field] = None
    
    def get_vote_distribution(self) -> Dict[str, float]:
        """
        Get distribution of votes from provenance.
        
        Returns:
            Dictionary mapping answers to their counts
        """
        distribution = {}
        neighbor_answers = self.provenance.get('neighbor_answers', [])
        
        # Count model's own answer
        model_answer = self.provenance.get('model_self_answer')
        if model_answer is not None:
            distribution[str(model_answer)] = distribution.get(str(model_answer), 0) + 1
        
        # Count neighbor answers
        for neighbor in neighbor_answers:
            answer = str(neighbor.get('answer', 'unknown'))
            distribution[answer] = distribution.get(answer, 0) + 1
        
        return distribution
    
    def get_consensus_strength(self) -> float:
        """
        Calculate the strength of consensus.
        
        Returns:
            Strength score (0 = no consensus, 1 = perfect consensus)
        """
        distribution = self.get_vote_distribution()
        if not distribution:
            return 0.0
        
        total_votes = sum(distribution.values())
        if total_votes == 0:
            return 0.0
        
        # Calculate strength as proportion of votes for winning answer
        final_answer_votes = distribution.get(str(self.final_answer), 0)
        return final_answer_votes / total_votes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voting result to dictionary."""
        return {
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "consensus_strength": self.get_consensus_strength()
        }


@dataclass
class RewardComponents:
    """
    Multi-component reward structure.
    
    Attributes:
        task_reward: Reward for task completion
        curiosity_reward: Reward from curiosity module
        coherence_reward: Reward for trajectory coherence
        tool_penalty: Penalty for tool misuse
        total_reward: Combined total reward
        metadata: Additional reward metadata
    """
    task_reward: float
    curiosity_reward: float = 0.0
    coherence_reward: float = 0.0
    tool_penalty: float = 0.0
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate total reward after initialization."""
        self.calculate_total()
    
    def calculate_total(self, weights: Optional[Dict[str, float]] = None):
        """
        Calculate total reward with optional custom weights.
        
        Args:
            weights: Dictionary of component weights
        """
        if weights is None:
            # Default equal weighting
            weights = {
                "task": 1.0,
                "curiosity": 1.0,
                "coherence": 1.0,
                "penalty": 1.0
            }
        
        self.total_reward = (
            weights.get("task", 1.0) * self.task_reward +
            weights.get("curiosity", 1.0) * self.curiosity_reward +
            weights.get("coherence", 1.0) * self.coherence_reward +
            weights.get("penalty", 1.0) * self.tool_penalty
        )
    
    def normalize(self, method: str = "zscore"):
        """
        Normalize reward components.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'clip')
        """
        if method == "clip":
            # Clip to [-10, 10] range
            self.task_reward = np.clip(self.task_reward, -10, 10)
            self.curiosity_reward = np.clip(self.curiosity_reward, -10, 10)
            self.coherence_reward = np.clip(self.coherence_reward, -10, 10)
            self.tool_penalty = np.clip(self.tool_penalty, -10, 0)
        
        self.calculate_total()
    
    def to_tensor(self) -> torch.Tensor:
        """Convert reward components to tensor."""
        return torch.tensor([
            self.task_reward,
            self.curiosity_reward,
            self.coherence_reward,
            self.tool_penalty,
            self.total_reward
        ], dtype=torch.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reward components to dictionary."""
        return {
            "task_reward": self.task_reward,
            "curiosity_reward": self.curiosity_reward,
            "coherence_reward": self.coherence_reward,
            "tool_penalty": self.tool_penalty,
            "total_reward": self.total_reward,
            "metadata": self.metadata
        }


# Utility functions for data structure validation and conversion

def validate_trajectory(trajectory: Union[Trajectory, Dict, List]) -> Trajectory:
    """
    Validate and convert input to Trajectory object.
    
    Args:
        trajectory: Input trajectory in various formats
        
    Returns:
        Validated Trajectory object
    """
    if isinstance(trajectory, Trajectory):
        return trajectory
    elif isinstance(trajectory, dict):
        return Trajectory.from_dict(trajectory)
    elif isinstance(trajectory, list):
        # Assume list of actions
        actions = []
        for item in trajectory:
            if isinstance(item, Action):
                actions.append(item)
            elif isinstance(item, dict):
                actions.append(Action.from_dict(item))
            else:
                raise ValueError(f"Invalid action type: {type(item)}")
        return Trajectory(actions=actions)
    else:
        raise ValueError(f"Invalid trajectory type: {type(trajectory)}")


def generate_json_schema():
    """
    Generate JSON schemas for all data structures.
    
    This can be used for validation when saving/loading data.
    """
    schemas = {}
    
    # Action schema
    schemas["Action"] = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": [e.value for e in ActionType]},
            "operation": {"type": "string"},
            "arguments": {"type": "object"},
            "result": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "timestamp": {"type": ["string", "null"]}
        },
        "required": ["type", "operation"]
    }
    
    # Trajectory schema
    schemas["Trajectory"] = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"$ref": "#/definitions/Action"}
            },
            "final_answer": {"type": ["string", "number", "boolean", "null"]},
            "total_reward": {"type": "number"},
            "metadata": {"type": "object"}
        },
        "required": ["actions"]
    }
    
    # Experience schema
    schemas["Experience"] = {
        "type": "object",
        "properties": {
            "experience_id": {"type": "string"},
            "question_text": {"type": "string"},
            "trajectory": {"$ref": "#/definitions/Trajectory"},
            "model_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "timestamp": {"type": "string"},
            "status": {"type": "string", "enum": [e.value for e in ExperienceStatus]},
            "priority": {"type": "number", "minimum": 0},
            "usage_count": {"type": "integer", "minimum": 0},
            "success_rate": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["experience_id", "question_text", "trajectory", "model_confidence"]
    }
    
    return schemas


# Export all data structures
__all__ = [
    'ExperienceStatus',
    'ActionType',
    'Action',
    'Trajectory',
    'Experience',
    'UpdateTask',
    'VotingResult',
    'RewardComponents',
    'validate_trajectory',
    'generate_json_schema'
]