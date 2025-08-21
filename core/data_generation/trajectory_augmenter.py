"""
Trajectory Augmenter Module for Self-Correction Data Generation
================================================================
This module implements the logic for augmenting golden trajectories with
self-correction behavior, teaching models to identify and recover from errors.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Represents a reasoning trajectory with actions and observations."""
    task_id: str
    question: str
    actions: List[Dict[str, Any]]
    final_answer: str
    trajectory_type: str = "golden"  # golden, trap, self_correction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistractorAction:
    """Represents an incorrect action to be inserted for self-correction."""
    action_type: str
    parameters: Dict[str, Any]
    observation: str
    error_type: str  # e.g., "wrong_coordinates", "incorrect_object", "invalid_operation"


class TrajectoryAugmenter:
    """
    Augments golden trajectories with self-correction behavior.
    
    This class takes correct trajectories and intentionally introduces errors
    followed by corrective thoughts, teaching the model to recover from mistakes.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the TrajectoryAugmenter.
        
        Args:
            llm_client: Optional LLM client for generating corrective thoughts.
                       If None, will use templated responses.
        """
        self.llm_client = llm_client
        self.distractor_templates = self._initialize_distractor_templates()
        self.correction_templates = self._initialize_correction_templates()
        
    def _initialize_distractor_templates(self) -> Dict[str, List[DistractorAction]]:
        """Initialize common distractor action templates."""
        return {
            "SEGMENT_OBJECT_AT": [
                DistractorAction(
                    action_type="SEGMENT_OBJECT_AT",
                    parameters={"x": -1, "y": -1},  # Invalid coordinates
                    observation="Error: Coordinates out of bounds",
                    error_type="invalid_coordinates"
                ),
                DistractorAction(
                    action_type="SEGMENT_OBJECT_AT",
                    parameters={"x": 100, "y": 100},  # Wrong location
                    observation="Found: background region with no distinct objects",
                    error_type="wrong_location"
                ),
            ],
            "READ_TEXT": [
                DistractorAction(
                    action_type="READ_TEXT",
                    parameters={"region": [0, 0, 10, 10]},  # Too small region
                    observation="No text detected in the specified region",
                    error_type="wrong_region"
                ),
            ],
            "ZOOM_IN": [
                DistractorAction(
                    action_type="ZOOM_IN",
                    parameters={"scale": 0.5},  # Zoom out instead of in
                    observation="Image became smaller, details are less visible",
                    error_type="wrong_parameter"
                ),
            ],
            "TRACK_OBJECT": [
                DistractorAction(
                    action_type="TRACK_OBJECT",
                    parameters={"object_id": -1, "frames": [0, 5]},
                    observation="Error: Invalid object ID",
                    error_type="invalid_id"
                ),
            ],
        }
        
    def _initialize_correction_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for corrective thoughts based on error types."""
        return {
            "invalid_coordinates": [
                "That resulted in an error. Let me try with valid coordinates within the image bounds.",
                "The coordinates were out of range. I need to specify a location within the image.",
            ],
            "wrong_location": [
                "That doesn't seem to be the right area. Let me try a different location.",
                "No relevant objects found there. I should look elsewhere in the image.",
            ],
            "wrong_region": [
                "No text was found in that region. I need to specify a larger or different area.",
                "The region was too small or incorrect. Let me adjust the boundaries.",
            ],
            "wrong_parameter": [
                "That parameter had the opposite effect. Let me use the correct value.",
                "The action didn't work as intended. I need to adjust the parameters.",
            ],
            "invalid_id": [
                "That object ID is invalid. Let me use a valid identifier.",
                "The specified ID doesn't exist. I need to reference a valid object.",
            ],
        }
        
    def augment_trajectory(
        self, 
        trajectory: Trajectory,
        distractor_action: Optional[DistractorAction] = None
    ) -> Trajectory:
        """
        Augment a golden trajectory with self-correction behavior.
        
        Args:
            trajectory: The original golden trajectory
            distractor_action: Optional specific distractor to use
            
        Returns:
            New trajectory with self-correction behavior
        """
        if trajectory.trajectory_type != "golden":
            logger.warning(f"Expected golden trajectory, got {trajectory.trajectory_type}")
            return trajectory
            
        # Select or generate a distractor action
        if distractor_action is None:
            distractor_action = self._select_distractor(trajectory)
            
        if distractor_action is None:
            logger.warning("Could not generate distractor for trajectory")
            return trajectory
            
        # Generate corrective thought
        corrective_thought = self._generate_corrective_thought(
            distractor_action,
            trajectory.actions[0] if trajectory.actions else None
        )
        
        # Build augmented action sequence
        augmented_actions = []
        
        # 1. Add the distractor action
        augmented_actions.append({
            "action": distractor_action.action_type,
            "parameters": distractor_action.parameters,
            "observation": distractor_action.observation,
        })
        
        # 2. Add the corrective thought
        augmented_actions.append({
            "thought": corrective_thought,
            "type": "self_correction"
        })
        
        # 3. Add the original trajectory actions
        augmented_actions.extend(trajectory.actions)
        
        # Create new augmented trajectory
        augmented_trajectory = Trajectory(
            task_id=f"{trajectory.task_id}_sc",
            question=trajectory.question,
            actions=augmented_actions,
            final_answer=trajectory.final_answer,
            trajectory_type="self_correction",
            metadata={
                **trajectory.metadata,
                "original_trajectory_id": trajectory.task_id,
                "distractor_type": distractor_action.error_type,
                "augmentation_method": "self_correction"
            }
        )
        
        return augmented_trajectory
        
    def _select_distractor(self, trajectory: Trajectory) -> Optional[DistractorAction]:
        """
        Select an appropriate distractor action based on the trajectory.
        
        Args:
            trajectory: The trajectory to analyze
            
        Returns:
            A distractor action or None if no appropriate distractor found
        """
        # Analyze first action in the trajectory
        if not trajectory.actions:
            # For empty trajectories, return a default distractor
            default_distractors = self.distractor_templates.get('SEGMENT_OBJECT_AT', [])
            return random.choice(default_distractors) if default_distractors else None
            
        first_action = trajectory.actions[0]
        
        # If it's a dict with 'action' key
        if isinstance(first_action, dict) and 'action' in first_action:
            action_type = first_action['action']
            
            # Check if we have distractors for this action type
            if action_type in self.distractor_templates:
                distractors = self.distractor_templates[action_type]
                return random.choice(distractors) if distractors else None
                
        return None
        
    def _generate_corrective_thought(
        self,
        distractor: DistractorAction,
        next_action: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a corrective thought for the given distractor.
        
        Args:
            distractor: The distractor action that was performed
            next_action: The next correct action in the trajectory
            
        Returns:
            A corrective thought string
        """
        # If LLM client is available, use it for more natural responses
        if self.llm_client is not None:
            return self._generate_with_llm(distractor, next_action)
            
        # Otherwise use templates
        if distractor.error_type in self.correction_templates:
            templates = self.correction_templates[distractor.error_type]
            return random.choice(templates)
            
        # Fallback generic correction
        return "That didn't work as expected. Let me try a different approach."
        
    def _generate_with_llm(
        self,
        distractor: DistractorAction,
        next_action: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate corrective thought using LLM.
        
        Args:
            distractor: The distractor action
            next_action: The next correct action
            
        Returns:
            LLM-generated corrective thought
        """
        prompt = f"""You are an AI assistant analyzing a reasoning trace. An incorrect action was just performed, leading to an unhelpful observation. Generate a brief, natural "thought" that acknowledges this mistake and states the intention to try a different approach. The thought should be concise and serve as a bridge to the next, correct action.

Incorrect Action: {distractor.action_type}
Parameters: {json.dumps(distractor.parameters)}
Observation: {distractor.observation}
"""
        
        if next_action:
            prompt += f"\nNext Correct Action: {next_action.get('action', 'Unknown')}\n"
            
        prompt += "\nGenerate only the corrective thought text. Example: 'That doesn't seem right, the object I found is not what I was looking for. I will try a different location.'"
        
        # Check if LLM client is available and has generate method
        if self.llm_client and hasattr(self.llm_client, 'generate'):
            return self.llm_client.generate(prompt)
        else:
            # Fallback to template response
            return "That approach didn't yield the expected results. Let me reconsider and try a different strategy."
        
    def batch_augment(
        self,
        trajectories: List[Trajectory],
        augmentation_ratio: float = 0.2
    ) -> Tuple[List[Trajectory], Dict[str, int]]:
        """
        Augment a batch of trajectories with self-correction samples.
        
        Args:
            trajectories: List of golden trajectories
            augmentation_ratio: Fraction of trajectories to augment
            
        Returns:
            Tuple of (all trajectories including augmented, statistics dict)
        """
        golden_trajectories = [t for t in trajectories if t.trajectory_type == "golden"]
        num_to_augment = int(len(golden_trajectories) * augmentation_ratio)
        
        # Randomly select trajectories to augment
        to_augment = random.sample(golden_trajectories, min(num_to_augment, len(golden_trajectories)))
        
        augmented = []
        stats = {
            "total_input": len(trajectories),
            "golden_count": len(golden_trajectories),
            "augmented_count": 0,
            "failed_augmentation": 0
        }
        
        for trajectory in to_augment:
            try:
                aug_trajectory = self.augment_trajectory(trajectory)
                if aug_trajectory.trajectory_type == "self_correction":
                    augmented.append(aug_trajectory)
                    stats["augmented_count"] += 1
                else:
                    stats["failed_augmentation"] += 1
            except Exception as e:
                logger.error(f"Failed to augment trajectory {trajectory.task_id}: {e}")
                stats["failed_augmentation"] += 1
                
        # Combine original and augmented trajectories
        result = trajectories + augmented
        
        logger.info(f"Augmentation complete: {stats}")
        return result, stats


def load_trajectories_from_file(filepath: Path) -> List[Trajectory]:
    """Load trajectories from a JSON file."""
    trajectories = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            trajectory = Trajectory(
                task_id=data.get('task_id', ''),
                question=data.get('question', ''),
                actions=data.get('actions', []),
                final_answer=data.get('final_answer', ''),
                trajectory_type=data.get('trajectory_type', 'golden'),
                metadata=data.get('metadata', {})
            )
            trajectories.append(trajectory)
            
    return trajectories


def save_trajectories_to_file(trajectories: List[Trajectory], filepath: Path):
    """Save trajectories to a JSON file."""
    with open(filepath, 'w') as f:
        for trajectory in trajectories:
            data = {
                'task_id': trajectory.task_id,
                'question': trajectory.question,
                'actions': trajectory.actions,
                'final_answer': trajectory.final_answer,
                'trajectory_type': trajectory.trajectory_type,
                'metadata': trajectory.metadata
            }
            f.write(json.dumps(data) + '\n')