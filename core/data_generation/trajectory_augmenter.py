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
from tqdm import tqdm
import traceback

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
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        """
        Initialize the TrajectoryAugmenter.
        
        Args:
            config: Configuration dictionary with proportions and settings
            llm_client: Optional LLM client for generating corrective thoughts.
                       If None, will use templated responses.
        """
        self.config = config
        
        # Extract proportions from config
        proportions_config = config.get('proportions', {})
        
        # Handle new trap_samples structure
        trap_config = proportions_config.get('trap_samples', {})
        if isinstance(trap_config, dict) and 'total_proportion' in trap_config:
            trap_proportion = trap_config['total_proportion']
        else:
            trap_proportion = proportions_config.get('trap', 0.2)
        
        self.proportions = {
            'golden': proportions_config.get('golden_positive', 0.6),
            'trap': trap_proportion,
            'self_correction': proportions_config.get('self_correction', 0.2)
        }
        
        # Store trap sub-types configuration
        self.trap_samples_config = trap_config if isinstance(trap_config, dict) else {}
        
        self.llm_client = llm_client
        self.distractor_templates = self._initialize_distractor_templates()
        self.correction_templates = self._initialize_correction_templates()
        self.augmentation_stats = {
            'total_processed': 0,
            'golden_kept': 0,
            'traps_created': 0,
            'self_corrections_created': 0,
            'augmentation_failures': 0,
            'distractor_generation_failures': 0
        }
        
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
                DistractorAction(
                    action_type="SEGMENT_OBJECT_AT",
                    parameters={"x": 50, "y": 50},  # Ambiguous location
                    observation="Multiple overlapping objects detected, segmentation unclear",
                    error_type="ambiguous_location"
                ),
            ],
            "READ_TEXT": [
                DistractorAction(
                    action_type="READ_TEXT",
                    parameters={"region": [0, 0, 10, 10]},  # Too small region
                    observation="No text detected in the specified region",
                    error_type="wrong_region"
                ),
                DistractorAction(
                    action_type="READ_TEXT",
                    parameters={"region": [0, 0, 100, 100]},  # Wrong area
                    observation="Detected text: [unreadable/blurry]",
                    error_type="unclear_text"
                ),
            ],
            "ZOOM_IN": [
                DistractorAction(
                    action_type="ZOOM_IN",
                    parameters={"scale": 0.5},  # Zoom out instead of in
                    observation="Image became smaller, details are less visible",
                    error_type="wrong_parameter"
                ),
                DistractorAction(
                    action_type="ZOOM_IN",
                    parameters={"scale": 10.0},  # Too much zoom
                    observation="Image is too pixelated to identify objects",
                    error_type="excessive_zoom"
                ),
            ],
            "ZOOM-IN": [  # Alternative naming
                DistractorAction(
                    action_type="ZOOM-IN",
                    parameters={"bbox": [0, 0, 50, 50]},  # Wrong region
                    observation="Zoomed into empty background area",
                    error_type="wrong_region"
                ),
            ],
            "SELECT_FRAME": [
                DistractorAction(
                    action_type="SELECT_FRAME",
                    parameters={"frame_number": -1},
                    observation="Error: Invalid frame number",
                    error_type="invalid_frame"
                ),
                DistractorAction(
                    action_type="SELECT_FRAME",
                    parameters={"start_time_sec": 0, "end_time_sec": 0.1},
                    observation="Selected frames show motion blur, content unclear",
                    error_type="wrong_temporal_window"
                ),
            ],
            "SELECT-FRAME": [  # Alternative naming
                DistractorAction(
                    action_type="SELECT-FRAME",
                    parameters={"timestamp": -1},
                    observation="Error: Timestamp out of video bounds",
                    error_type="invalid_timestamp"
                ),
            ],
            "TRACK_OBJECT": [
                DistractorAction(
                    action_type="TRACK_OBJECT",
                    parameters={"object_id": -1, "frames": [0, 5]},
                    observation="Error: Invalid object ID",
                    error_type="invalid_id"
                ),
                DistractorAction(
                    action_type="TRACK_OBJECT",
                    parameters={"object_id": 999, "frames": [0, 10]},
                    observation="Object lost after frame 2, tracking failed",
                    error_type="tracking_lost"
                ),
            ],
            "TRACK-OBJECT": [  # Alternative naming
                DistractorAction(
                    action_type="TRACK-OBJECT",
                    parameters={"mask": "invalid_mask_data"},
                    observation="Error: Invalid mask format",
                    error_type="invalid_mask"
                ),
            ],
            "GET_PROPERTIES": [
                DistractorAction(
                    action_type="GET_PROPERTIES",
                    parameters={"object_id": None},
                    observation="Error: No object specified",
                    error_type="missing_object"
                ),
            ],
            "GET-PROPERTIES": [  # Alternative naming
                DistractorAction(
                    action_type="GET-PROPERTIES",
                    parameters={"region": []},
                    observation="Error: Invalid region specification",
                    error_type="invalid_region"
                ),
            ],
        }
        
    def _initialize_correction_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for corrective thoughts based on error types."""
        return {
            "invalid_coordinates": [
                "That resulted in an error. Let me try with valid coordinates within the image bounds.",
                "The coordinates were out of range. I need to specify a location within the image.",
                "Invalid coordinates. I'll adjust to stay within the image boundaries.",
            ],
            "wrong_location": [
                "That doesn't seem to be the right area. Let me try a different location.",
                "No relevant objects found there. I should look elsewhere in the image.",
                "I'm looking at the wrong part of the image. Let me refocus on the target area.",
            ],
            "ambiguous_location": [
                "Multiple objects are overlapping here. I need to be more precise with my selection.",
                "The segmentation is unclear due to overlapping objects. Let me try a clearer area.",
            ],
            "wrong_region": [
                "No text was found in that region. I need to specify a larger or different area.",
                "The region was too small or incorrect. Let me adjust the boundaries.",
                "I missed the text area. Let me expand the region to capture the text.",
            ],
            "unclear_text": [
                "The text is blurry or unreadable. I need to zoom in or select a clearer region.",
                "Can't make out the text clearly. Let me try a different approach.",
            ],
            "wrong_parameter": [
                "That parameter had the opposite effect. Let me use the correct value.",
                "The action didn't work as intended. I need to adjust the parameters.",
                "Wrong parameter value. Let me correct it.",
            ],
            "excessive_zoom": [
                "I zoomed in too much and lost detail. Let me use a more moderate zoom level.",
                "The image is too pixelated. I need to reduce the zoom factor.",
            ],
            "invalid_id": [
                "That object ID is invalid. Let me use a valid identifier.",
                "The specified ID doesn't exist. I need to reference a valid object.",
            ],
            "invalid_frame": [
                "Invalid frame number. I need to select a frame within the video bounds.",
                "That frame doesn't exist. Let me choose a valid frame.",
            ],
            "wrong_temporal_window": [
                "The selected time window doesn't show the relevant content. Let me adjust the timing.",
                "Wrong temporal selection. I need to find the correct moment in the video.",
                "The frames are blurry or don't contain what I'm looking for. Let me try a different time range.",
            ],
            "invalid_timestamp": [
                "The timestamp is out of bounds. I need to stay within the video duration.",
                "Invalid timestamp. Let me select a valid time point.",
            ],
            "tracking_lost": [
                "Lost track of the object. I need to reinitialize tracking with a better starting point.",
                "Tracking failed. Let me try with a clearer initial selection.",
            ],
            "invalid_mask": [
                "The mask format is invalid. I need to provide a proper segmentation mask.",
                "Invalid mask data. Let me generate a correct mask.",
            ],
            "missing_object": [
                "No object was specified. I need to identify the target object first.",
                "Missing object reference. Let me select an object to analyze.",
            ],
            "invalid_region": [
                "Invalid region specification. I need to provide proper coordinates.",
                "The region format is incorrect. Let me specify it correctly.",
            ],
            "analysis_error": [
                "The analysis failed to extract properties. Let me try a different approach.",
                "Couldn't determine the object properties. I need to segment it better first.",
            ],
            "wrong_object": [
                "I selected the wrong object. Let me identify the correct one.",
                "That's not the target object. I need to find the right one.",
            ],
            "wrong_frame": [
                "This frame doesn't show what I expected. Let me select a different one.",
                "Wrong frame selected. I need to find the correct moment.",
            ],
        }
        
        
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
        
    def augment_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """
        Augment a single trajectory with self-correction behavior.
        This is a convenience method for single-trajectory augmentation.
        
        Args:
            trajectory: The original golden trajectory
            
        Returns:
            New trajectory with self-correction behavior
        """
        # Use the internal self-correction processing method
        result = self._process_as_self_correction([trajectory])
        return result[0] if result else trajectory
    
    def _augment_perceptual_near_miss(self, golden_sample: Trajectory) -> Trajectory:
        """
        Create a trap sample where the visual action is subtly incorrect.
        
        This creates "near-miss" samples where actions are slightly perturbed,
        teaching the model to pay close attention to precise geometric details.
        
        Args:
            golden_sample: The original correct trajectory
            
        Returns:
            New trajectory with perceptual near-miss trap
        """
        # Create a deep copy of the actions
        new_actions = []
        for step in golden_sample.actions:
            if isinstance(step, dict):
                new_actions.append(step.copy())
            else:
                new_actions.append(step)
        
        # Find the first action step
        action_found = False
        for i, step in enumerate(new_actions):
            if isinstance(step, dict) and step.get('type') == 'action':
                action_found = True
                original_action = step
                
                # Apply perturbation based on action type
                if 'parameters' in original_action:
                    params = original_action['parameters'].copy()
                    
                    # Handle different parameter types
                    if 'bbox' in params and isinstance(params['bbox'], list) and len(params['bbox']) >= 4:
                        # Perturb bounding box coordinates
                        bbox = params['bbox'].copy()
                        coord_to_perturb = random.randint(0, 3)
                        
                        # Calculate perturbation (5-10% of typical image dimension)
                        # Assuming typical image width/height around 640-1024 pixels
                        perturbation_amount = random.uniform(30, 60) * random.choice([-1, 1])
                        
                        bbox[coord_to_perturb] = bbox[coord_to_perturb] + perturbation_amount
                        params['bbox'] = bbox
                        
                        logger.debug(f"Perturbed bbox coordinate {coord_to_perturb}: {original_action['parameters']['bbox']} -> {bbox}")
                        
                    elif 'point' in params and isinstance(params['point'], list) and len(params['point']) >= 2:
                        # Perturb point coordinates
                        point = params['point'].copy()
                        coord_to_perturb = random.randint(0, 1)
                        perturbation_amount = random.uniform(20, 40) * random.choice([-1, 1])
                        
                        point[coord_to_perturb] = point[coord_to_perturb] + perturbation_amount
                        params['point'] = point
                        
                        logger.debug(f"Perturbed point coordinate {coord_to_perturb}: {original_action['parameters']['point']} -> {point}")
                        
                    elif 'coordinates' in params and isinstance(params['coordinates'], list):
                        # Handle coordinates parameter (alternative to point)
                        coords = params['coordinates'].copy()
                        if len(coords) >= 2:
                            coord_to_perturb = random.randint(0, 1)
                            perturbation_amount = random.uniform(20, 40) * random.choice([-1, 1])
                            coords[coord_to_perturb] = coords[coord_to_perturb] + perturbation_amount
                            params['coordinates'] = coords
                    
                    # Update the action with perturbed parameters
                    new_actions[i] = {
                        **original_action,
                        'parameters': params
                    }
                
                break
        
        if not action_found:
            logger.warning(f"No action found in trajectory {golden_sample.task_id} for perceptual near-miss")
            return golden_sample
        
        # Create a subtly incorrect final answer
        # This makes the trap more effective as the model must detect the subtle error
        incorrect_answer = self._generate_incorrect_answer(golden_sample.final_answer)
        
        # Create the new trap trajectory
        trap_trajectory = Trajectory(
            task_id=f"{golden_sample.task_id}_pnm_{random.randint(0,99)}",
            question=golden_sample.question,
            actions=new_actions,
            final_answer=incorrect_answer,
            trajectory_type="trap",
            metadata={
                **golden_sample.metadata,
                'original_trajectory_id': golden_sample.task_id,
                'augmentation_method': 'perceptual_near_miss',
                'trap_type': 'perceptual_near_miss',
                'provenance': {
                    'trap_type': 'perceptual_near_miss',
                    'original_answer': golden_sample.final_answer
                }
            }
        )
        
        return trap_trajectory
    
    def _augment_logical_fallacy(self, golden_sample: Trajectory) -> Trajectory:
        """
        Create a trap sample where visual perception is correct but reasoning contains a logical flaw.
        
        This teaches the model to be a critical thinker and validate reasoning against evidence.
        
        Args:
            golden_sample: The original correct trajectory
            
        Returns:
            New trajectory with logical fallacy trap
        """
        # Create a deep copy of the actions
        new_actions = []
        for step in golden_sample.actions:
            if isinstance(step, dict):
                new_actions.append(step.copy())
            else:
                new_actions.append(step)
        
        # Find the last thought in the trajectory
        last_thought_index = -1
        last_thought_content = ""
        
        for i in range(len(new_actions) - 1, -1, -1):
            step = new_actions[i]
            if isinstance(step, dict) and step.get('type') == 'thought':
                last_thought_index = i
                last_thought_content = step.get('content', '')
                break
        
        if last_thought_index == -1 or not last_thought_content:
            logger.warning(f"No thought found in trajectory {golden_sample.task_id} for logical fallacy")
            return golden_sample
        
        # Generate flawed reasoning
        if self.llm_client and hasattr(self.llm_client, 'generate'):
            # Use LLM to generate sophisticated logical fallacy
            fallacy_prompt = f"""You are an expert in creating educational trap examples. I have a question and a correct reasoning step. Your task is to rewrite the reasoning to contain a subtle but definite logical fallacy, while still arriving at an INCORRECT conclusion.

Original Question: {golden_sample.question}
Correct Final Thought: {last_thought_content}
Correct Answer: {golden_sample.final_answer}

Create a flawed reasoning that:
1. Uses the same observations/data points
2. Contains a logical error (e.g., reversed comparison, faulty causation, incorrect inference)
3. Arrives at a plausible but incorrect conclusion

Output ONLY the flawed reasoning text, nothing else."""
            
            try:
                flawed_reasoning = self.llm_client.generate(fallacy_prompt)
            except:
                # Fallback to template-based flawed reasoning
                flawed_reasoning = self._generate_template_fallacy(last_thought_content, golden_sample.final_answer)
        else:
            # Use template-based approach
            flawed_reasoning = self._generate_template_fallacy(last_thought_content, golden_sample.final_answer)
        
        # Update the last thought with flawed reasoning
        new_actions[last_thought_index] = {
            'type': 'thought',
            'content': flawed_reasoning
        }
        
        # Generate an incorrect final answer based on the flawed reasoning
        incorrect_answer = self._generate_incorrect_answer(golden_sample.final_answer)
        
        # Create the new trap trajectory
        trap_trajectory = Trajectory(
            task_id=f"{golden_sample.task_id}_lf_{random.randint(0,99)}",
            question=golden_sample.question,
            actions=new_actions,
            final_answer=incorrect_answer,
            trajectory_type="trap",
            metadata={
                **golden_sample.metadata,
                'original_trajectory_id': golden_sample.task_id,
                'augmentation_method': 'logical_fallacy',
                'trap_type': 'logical_fallacy',
                'provenance': {
                    'trap_type': 'logical_fallacy',
                    'original_answer': golden_sample.final_answer,
                    'original_reasoning': last_thought_content
                }
            }
        )
        
        return trap_trajectory
    
    def _generate_template_fallacy(self, original_thought: str, original_answer: str) -> str:
        """Generate a template-based logical fallacy."""
        fallacy_templates = [
            # Reversed comparison
            lambda t, a: t.replace("larger", "smaller").replace("bigger", "smaller").replace("more", "less").replace("greater", "lesser"),
            # Incorrect inference
            lambda t, a: f"Based on the observations, since both values are present, they must be equal. Therefore, neither is the answer.",
            # Faulty causation
            lambda t, a: f"I see the data points. However, the first mentioned item is always the correct answer by convention. So the answer must be the first one.",
            # Misinterpretation
            lambda t, a: t.replace("therefore", "however, this actually means the opposite, so"),
            # Circular reasoning
            lambda t, a: f"The observation confirms what we expected. Since we expected it, it must be correct. The answer is clearly the first option.",
        ]
        
        # Apply a random fallacy template
        fallacy_func = random.choice(fallacy_templates)
        return fallacy_func(original_thought, original_answer)
    
    def _generate_incorrect_answer(self, correct_answer: str) -> str:
        """Generate a plausible but incorrect answer."""
        # Simple strategies for generating incorrect answers
        if "yes" in correct_answer.lower():
            return correct_answer.replace("yes", "no").replace("Yes", "No")
        elif "no" in correct_answer.lower():
            return correct_answer.replace("no", "yes").replace("No", "Yes")
        elif any(word in correct_answer.lower() for word in ["larger", "bigger", "greater"]):
            return correct_answer.replace("larger", "smaller").replace("bigger", "smaller").replace("greater", "lesser")
        elif any(word in correct_answer.lower() for word in ["smaller", "lesser", "fewer"]):
            return correct_answer.replace("smaller", "larger").replace("lesser", "greater").replace("fewer", "more")
        elif "left" in correct_answer.lower():
            return correct_answer.replace("left", "right").replace("Left", "Right")
        elif "right" in correct_answer.lower():
            return correct_answer.replace("right", "left").replace("Right", "Left")
        else:
            # For other cases, prepend "not" or "incorrect"
            if correct_answer.startswith("The"):
                return f"The opposite of what was stated. {correct_answer}"
            else:
                return f"Not {correct_answer}"
    
    def augment_as_trap(self, trajectory: Trajectory) -> Trajectory:
        """
        Converts a golden trajectory into a process-negative "trap" sample
        by introducing a subtle flaw.

        Args:
            trajectory: The original golden trajectory.

        Returns:
            A new trajectory with a process-level error.
        """
        if trajectory.trajectory_type != "golden":
            logger.warning(f"Trap augmentation expects a golden trajectory, got {trajectory.trajectory_type}.")
            return trajectory

        # Create a deep copy to avoid modifying the original actions list
        new_actions = []
        for step in trajectory.actions:
            if isinstance(step, dict):
                new_actions.append(step.copy())
            else:
                new_actions.append(step)
        
        # --- Define Trap Strategies ---
        # Randomly select a trap strategy
        trap_strategies = ["missing_action", "wrong_tool", "bad_parameter", "perceptual_error"]
        trap_type = random.choice(trap_strategies)
        
        # --- Apply Trap Logic Based on Strategy ---
        has_action = False
        action_index = -1
        
        for i, step in enumerate(new_actions):
            if isinstance(step, dict) and step.get('type') == 'action':
                has_action = True
                action_index = i
                break
        
        if has_action and action_index >= 0:
            if trap_type == "missing_action":
                # Strategy 1: Remove the action step (keeping only thoughts)
                # This simulates the model forgetting to perform the actual action
                removed_action = new_actions.pop(action_index)
                logger.debug(f"Created trap by removing action: {removed_action.get('name', 'unknown')}")
                
            elif trap_type == "wrong_tool":
                # Strategy 2: Replace action with wrong but plausible tool
                original_action = new_actions[action_index]
                wrong_action = self._generate_wrong_tool(original_action)
                new_actions[action_index] = wrong_action
                logger.debug(f"Created trap by using wrong tool: {original_action.get('name')} -> {wrong_action.get('name')}")
                
            elif trap_type == "bad_parameter":
                # Strategy 3: Keep the right tool but use invalid parameters
                original_action = new_actions[action_index]
                bad_action = self._generate_bad_parameters(original_action)
                new_actions[action_index] = bad_action
                logger.debug(f"Created trap with bad parameters for: {original_action.get('name')}")
                
            else:  # perceptual_error
                # Introduce perceptual error in the final thought
                for i in range(len(new_actions) - 1, -1, -1):
                    step = new_actions[i]
                    if isinstance(step, dict) and step.get('type') == 'thought':
                        original_thought = step.get('content', '')
                        flawed_thought = self._introduce_perceptual_error(original_thought)
                        step['content'] = flawed_thought
                        logger.debug(f"Applied perceptual trap: '{original_thought[:50]}...' -> '{flawed_thought[:50]}...'")
                        break
        else:
            # Fallback: Introduce perceptual error in the final thought
            trap_type = "perceptual_error"
            for i in range(len(new_actions) - 1, -1, -1):
                step = new_actions[i]
                if isinstance(step, dict) and step.get('type') == 'thought':
                    original_thought = step.get('content', '')
                    flawed_thought = self._introduce_perceptual_error(original_thought)
                    step['content'] = flawed_thought
                    logger.debug(f"Applied perceptual trap: '{original_thought[:50]}...' -> '{flawed_thought[:50]}...'")
                    break
        
        # Create the new trap trajectory object
        trap_trajectory = Trajectory(
            task_id=f"{trajectory.task_id}_trap_{random.randint(0,9)}",
            question=trajectory.question,
            actions=new_actions,  # The trajectory now contains the flaw
            final_answer=trajectory.final_answer,  # Answer might still seem plausible
            trajectory_type="trap",
            metadata={
                **trajectory.metadata,
                "original_trajectory_id": trajectory.task_id,
                "augmentation_method": "trap_generation",
                "trap_type": trap_type
            }
        )
        
        return trap_trajectory

    def _generate_wrong_tool(self, original_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a wrong but plausible tool for the given action.
        
        Args:
            original_action: The original correct action
            
        Returns:
            A new action with wrong tool but similar intent
        """
        action_name = original_action.get('name', '').upper().replace('_', '-')
        
        # Map of plausible wrong tool substitutions
        wrong_tool_map = {
            'ZOOM-IN': 'READ-TEXT',  # Try to read text instead of zooming
            'READ-TEXT': 'ZOOM-IN',  # Try to zoom instead of reading
            'SEGMENT_OBJECT_AT': 'GET_PROPERTIES',  # Try to get properties without segmenting
            'SEGMENT-OBJECT-AT': 'GET-PROPERTIES',
            'GET_PROPERTIES': 'SEGMENT_OBJECT_AT',  # Try to segment when should get properties
            'GET-PROPERTIES': 'SEGMENT-OBJECT-AT',
            'SELECT-FRAME': 'TRACK-OBJECT',  # Try to track instead of selecting frame
            'TRACK-OBJECT': 'SELECT-FRAME',  # Try to select frame instead of tracking
        }
        
        wrong_tool = wrong_tool_map.get(action_name, 'READ-TEXT')  # Default to READ-TEXT
        
        # Create new action with wrong tool
        wrong_action = original_action.copy()
        wrong_action['name'] = wrong_tool
        wrong_action['parameters'] = original_action.get('parameters', {})
        
        # Add error observation
        wrong_action['observation'] = f"Error: {wrong_tool} is not the appropriate operation for this task"
        
        return wrong_action
    
    def _generate_bad_parameters(self, original_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate bad parameters for the given action.
        
        Args:
            original_action: The original correct action
            
        Returns:
            A new action with same tool but invalid parameters
        """
        action_name = original_action.get('name', '').upper().replace('_', '-')
        bad_action = original_action.copy()
        
        # Generate bad parameters based on action type
        if 'ZOOM' in action_name:
            # Bad bbox: zero area or invalid coordinates
            bad_action['parameters'] = {
                'bbox': random.choice([
                    [0, 0, 0, 0],  # Zero area
                    [-100, -100, -50, -50],  # Negative coordinates
                    [1000000, 1000000, 1000001, 1000001],  # Out of bounds
                ])
            }
            bad_action['observation'] = "Error: Invalid bounding box coordinates"
            
        elif 'SEGMENT' in action_name:
            # Bad point: out of bounds or invalid
            bad_action['parameters'] = {
                'point': random.choice([
                    [-1, -1],  # Negative coordinates
                    [999999, 999999],  # Way out of bounds
                    [None, None],  # Null values
                ])
            }
            bad_action['observation'] = "Error: Invalid point coordinates for segmentation"
            
        elif 'READ' in action_name:
            # Bad region: too small or invalid
            bad_action['parameters'] = {
                'bbox': [0, 0, 1, 1]  # Too small to read anything
            }
            bad_action['observation'] = "Error: Region too small to contain readable text"
            
        elif 'TRACK' in action_name:
            # Bad frame numbers
            bad_action['parameters'] = {
                'object_id': 'invalid_id',
                'start_frame': -1,
                'end_frame': -10
            }
            bad_action['observation'] = "Error: Invalid frame range for tracking"
            
        elif 'SELECT' in action_name and 'FRAME' in action_name:
            # Bad frame number
            bad_action['parameters'] = {
                'frame_num': -999
            }
            bad_action['observation'] = "Error: Frame number out of valid range"
            
        else:
            # Generic bad parameters
            bad_action['parameters'] = {}
            bad_action['observation'] = "Error: Required parameters missing"
        
        return bad_action

    def _introduce_perceptual_error(self, thought: str) -> str:
        """
        A helper method to introduce subtle factual errors into a thought string.
        
        Args:
            thought: The original thought string.
            
        Returns:
            The thought string with a subtle error.
        """
        # Simple, rule-based replacements for common attributes
        replacements = {
            "red": "blue", "blue": "green", "green": "red",
            "left": "right", "right": "left",
            "on": "under", "under": "on",
            "small": "large", "large": "small",
            "visible": "hidden", "hidden": "visible",
            "clear": "blurry", "blurry": "clear",
            "scratches": "smooth surface", "smooth": "scratched",
            "bird": "butterfly", "warning sign": "information sign"
        }
        
        thought_lower = thought.lower()
        for old, new in replacements.items():
            if old in thought_lower:
                # Perform case-insensitive replacement
                import re
                pattern = re.compile(re.escape(old), re.IGNORECASE)
                return pattern.sub(new, thought, count=1)
        
        # Fallback if no replaceable term is found
        # Just add a subtle misinterpretation
        if "observation is:" in thought:
            return thought.replace("observation is:", "observation seems to be:")
        return thought

    def process(self, golden_samples: List[Dict]) -> List[Dict]:
        """
        The main public method to augment a list of golden samples.
        
        Args:
            golden_samples: List of golden trajectory dictionaries
            
        Returns:
            List of augmented trajectories including golden, trap, and self-correction samples
        """
        logger.info(f"Processing {len(golden_samples)} golden samples for augmentation")
        
        # Convert dict samples to Trajectory objects if needed
        trajectories = []
        for sample in golden_samples:
            if isinstance(sample, dict):
                trajectory = Trajectory(
                    task_id=sample.get('task_id', ''),
                    question=sample.get('question', ''),
                    actions=sample.get('trajectory', sample.get('actions', [])),
                    final_answer=sample.get('final_answer', ''),
                    trajectory_type='golden',
                    metadata=sample.get('metadata', {})
                )
            else:
                trajectory = sample
            trajectories.append(trajectory)
        
        # Shuffle for random distribution
        random.shuffle(trajectories)
        
        # Calculate split sizes based on proportions
        total = len(trajectories)
        num_golden = int(total * self.proportions.get('golden', 0.5))
        num_trap = int(total * self.proportions.get('trap', 0.25))
        num_self_correction = total - num_golden - num_trap  # Remainder goes to self-correction
        
        # Split samples into groups
        samples_for_golden = trajectories[:num_golden]
        samples_for_trap = trajectories[num_golden:num_golden + num_trap]
        samples_for_self_correction = trajectories[num_golden + num_trap:]
        
        # Process each group
        final_augmented_list = []
        final_augmented_list.extend(self._process_as_golden(samples_for_golden))
        final_augmented_list.extend(self._process_as_trap(samples_for_trap))
        final_augmented_list.extend(self._process_as_self_correction(samples_for_self_correction))
        
        # Shuffle final results
        random.shuffle(final_augmented_list)
        
        # Log statistics
        self._log_statistics()
        
        # Convert back to dict format for compatibility
        result = []
        for traj in final_augmented_list:
            result.append({
                'task_id': traj.task_id,
                'question': traj.question,
                'trajectory': traj.actions,
                'final_answer': traj.final_answer,
                'trajectory_type': traj.trajectory_type,
                'metadata': traj.metadata
            })
        
        return result
    
    # --- Private Helper Methods for Each Augmentation Type ---
    
    def _process_as_golden(self, samples: List[Trajectory]) -> List[Trajectory]:
        """Simply adds 'trajectory_type': 'golden' to each sample."""
        logger.info(f"Processing {len(samples)} samples as golden")
        for sample in samples:
            sample.trajectory_type = 'golden'
            sample.metadata['augmentation_method'] = 'none'
            self.augmentation_stats['golden_kept'] += 1
        return samples
    
    def _process_as_trap(self, samples: List[Trajectory]) -> List[Trajectory]:
        """Process samples as trap trajectories with different sub-types."""
        logger.info(f"Processing {len(samples)} samples as traps")
        
        # Get trap sub-type proportions from config
        sub_types = self.trap_samples_config.get('sub_types', [
            {'name': 'process_negative', 'proportion': 0.5},
            {'name': 'perceptual_near_miss', 'proportion': 0.25},
            {'name': 'logical_fallacy', 'proportion': 0.25}
        ])
        
        # Calculate number of samples for each sub-type
        total = len(samples)
        sub_type_counts = {}
        remaining = total
        
        for i, sub_type in enumerate(sub_types):
            if i == len(sub_types) - 1:
                # Last sub-type gets remainder
                count = remaining
            else:
                count = int(total * sub_type['proportion'])
                remaining -= count
            sub_type_counts[sub_type['name']] = count
        
        # Split samples for each sub-type
        augmented = []
        sample_idx = 0
        
        for sub_type_name, count in sub_type_counts.items():
            sub_samples = samples[sample_idx:sample_idx + count]
            sample_idx += count
            
            logger.info(f"Creating {count} {sub_type_name} trap samples")
            
            for sample in tqdm(sub_samples, desc=f"Creating {sub_type_name} traps"):
                try:
                    # Select appropriate augmentation method
                    if sub_type_name == 'perceptual_near_miss':
                        trap_trajectory = self._augment_perceptual_near_miss(sample)
                    elif sub_type_name == 'logical_fallacy':
                        trap_trajectory = self._augment_logical_fallacy(sample)
                    else:  # process_negative or default
                        trap_trajectory = self.augment_as_trap(sample)
                    
                    augmented.append(trap_trajectory)
                    self.augmentation_stats['traps_created'] += 1
                    
                    # Track sub-type statistics
                    stat_key = f'{sub_type_name}_created'
                    if stat_key not in self.augmentation_stats:
                        self.augmentation_stats[stat_key] = 0
                    self.augmentation_stats[stat_key] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to create {sub_type_name} trap for {sample.task_id}: {e}")
                    # Keep as golden on failure
                    sample.trajectory_type = 'golden'
                    sample.metadata['augmentation_failure'] = f'{sub_type_name}_trap_creation_failed: {str(e)}'
                    augmented.append(sample)
                    self.augmentation_stats['augmentation_failures'] += 1
        
        return augmented
    
    def _process_as_self_correction(self, samples: List[Trajectory]) -> List[Trajectory]:
        """Process samples as self-correction trajectories with robust error handling."""
        augmented_samples = []
        
        for sample in tqdm(samples, desc="Augmenting Self-Correction"):
            logger.info(f"--- Attempting Self-Correction for task_id: {sample.task_id} ---")
            
            try:
                # 1. Generate a plausible "distractor" action. This is the most likely failure point.
                distractor_action = self._generate_distractor_action(sample)
                
                if distractor_action is None:
                    # This is a GRACEFUL failure. It means no valid distractor could be
                    # created for this specific sample. We log it and keep the original.
                    logger.warning(f"Could not generate a valid distractor for {sample.task_id}. Keeping as golden.")
                    sample.trajectory_type = 'golden'
                    sample.metadata['augmentation_failure'] = 'distractor_generation_failed'
                    augmented_samples.append(sample)
                    self.augmentation_stats['distractor_generation_failures'] += 1
                    continue
                
                logger.info(f"✓ Successfully generated distractor action for {sample.task_id}")
                
                # 2. Generate a corrective thought
                correction_thought = self._generate_corrective_thought(
                    distractor_action,
                    sample.actions[0] if sample.actions else None
                )
                
                # 3. Construct the new, augmented trajectory
                new_trajectory = [
                    {
                        "type": "action",
                        "name": distractor_action.action_type,
                        "parameters": distractor_action.parameters,
                        "observation": distractor_action.observation
                    },
                    {
                        "type": "thought",
                        "content": correction_thought
                    }
                ] + sample.actions
                
                # 4. Create the new sample
                new_sample = Trajectory(
                    task_id=f"{sample.task_id}_sc",
                    question=sample.question,
                    actions=new_trajectory,
                    final_answer=sample.final_answer,
                    trajectory_type='self_correction',
                    metadata={
                        **sample.metadata,
                        'original_trajectory_id': sample.task_id,
                        'augmentation_method': 'self_correction',
                        'distractor_type': distractor_action.error_type
                    }
                )
                
                augmented_samples.append(new_sample)
                self.augmentation_stats['self_corrections_created'] += 1
                
            except Exception as e:
                # This catches unexpected programming errors.
                logger.error(f"FATAL error during self-correction for {sample.task_id}: {e}", exc_info=True)
                logger.error(f"Traceback: {traceback.format_exc()}")
                # We still keep the original golden sample to not lose data.
                sample.trajectory_type = 'golden'
                sample.metadata['augmentation_failure'] = str(e)
                augmented_samples.append(sample)
                self.augmentation_stats['augmentation_failures'] += 1
        
        self.augmentation_stats['total_processed'] += len(samples)
        return augmented_samples
    
    def _generate_distractor_action(self, original_sample: Trajectory) -> Optional[DistractorAction]:
        """
        Intelligently generates a plausible but incorrect "distractor" action
        based on the original sample's trajectory.
        Returns None if no valid distractor can be generated.
        """
        # Check if actions is None or empty
        if not original_sample.actions:
            logger.debug(f"No actions in trajectory {original_sample.task_id}")
            return None
        
        # Find the first real action in the golden trajectory
        original_action = None
        for step in original_sample.actions:
            if isinstance(step, dict):
                if step.get('type') == 'action' or 'action' in step or 'name' in step:
                    original_action = step
                    break
        
        if not original_action:
            logger.debug(f"No action found in trajectory {original_sample.task_id}")
            return None  # Cannot create a distractor if there's no original action to corrupt
        
        # Extract action details
        action_name = original_action.get('name') or original_action.get('action')
        params = original_action.get('parameters', {})
        
        logger.debug(f"Creating distractor for action: {action_name} with params: {params}")
        
        # --- Strategy 1: Corrupt BBox for spatial actions ---
        if action_name in ['ZOOM-IN', 'ZOOM_IN', 'SEGMENT_OBJECT_AT', 'READ-TEXT', 'READ_TEXT']:
            if 'bbox' in params:
                original_bbox = params['bbox']
                # Create a shifted bbox (offset by 20-30% of image dimensions)
                new_bbox = self._corrupt_bbox(original_bbox)
                return DistractorAction(
                    action_type=action_name,
                    parameters={"bbox": new_bbox},
                    observation="No relevant objects found in the specified region",
                    error_type="wrong_location"
                )
            elif 'x' in params and 'y' in params:
                # Corrupt point coordinates
                new_x = max(0, params['x'] + random.randint(-100, 100))
                new_y = max(0, params['y'] + random.randint(-100, 100))
                return DistractorAction(
                    action_type=action_name,
                    parameters={"x": new_x, "y": new_y},
                    observation="Found: background region with no distinct objects",
                    error_type="wrong_location"
                )
            elif 'region' in params:
                # Corrupt region coordinates
                original_region = params['region']
                new_region = self._corrupt_region(original_region)
                return DistractorAction(
                    action_type=action_name,
                    parameters={"region": new_region},
                    observation="No text detected in the specified region",
                    error_type="wrong_region"
                )
        
        # --- Strategy 2: Corrupt Frame Selection for temporal actions ---
        if action_name in ['SELECT-FRAME', 'SELECT_FRAME']:
            if 'start_time_sec' in params:
                # Select a different time window
                new_start = max(0, params.get('start_time_sec', 0) + random.uniform(-5, 5))
                new_end = new_start + random.uniform(1, 3)
                return DistractorAction(
                    action_type=action_name,
                    parameters={"start_time_sec": new_start, "end_time_sec": new_end},
                    observation="Selected frames do not contain the target event",
                    error_type="wrong_temporal_window"
                )
            elif 'frame_number' in params:
                # Select wrong frame
                new_frame = max(0, params['frame_number'] + random.randint(-10, 10))
                return DistractorAction(
                    action_type=action_name,
                    parameters={"frame_number": new_frame},
                    observation="Frame does not show the expected content",
                    error_type="wrong_frame"
                )
        
        # --- Strategy 3: Corrupt Object Selection for tracking actions ---
        if action_name in ['TRACK_OBJECT', 'TRACK-OBJECT']:
            # Try to select a different object ID or wrong initial mask
            if 'object_id' in params:
                # Use a different object ID
                wrong_id = params['object_id'] + random.randint(1, 5)
                return DistractorAction(
                    action_type=action_name,
                    parameters={**params, "object_id": wrong_id},
                    observation="Tracking failed: Object lost after 2 frames",
                    error_type="wrong_object"
                )
            elif 'initial_mask' in params:
                # Slightly corrupt the initial mask
                return DistractorAction(
                    action_type=action_name,
                    parameters={**params, "initial_mask": "corrupted_mask_data"},
                    observation="Tracking failed: Invalid initial mask",
                    error_type="invalid_mask"
                )
        
        # --- Strategy 4: Generic parameter corruption ---
        if action_name in ['GET_PROPERTIES', 'GET-PROPERTIES']:
            # Return wrong properties
            return DistractorAction(
                action_type=action_name,
                parameters=params,
                observation="Properties: color=unknown, size=undefined, position=unclear",
                error_type="analysis_error"
            )
        
        # --- Fallback: Use template if available ---
        if action_name in self.distractor_templates:
            template_distractors = self.distractor_templates[action_name]
            if template_distractors:
                return random.choice(template_distractors)
        
        logger.debug(f"No distractor strategy found for action: {action_name}")
        return None
    
    def _corrupt_bbox(self, bbox: List[float]) -> List[float]:
        """Corrupt a bounding box by shifting it."""
        if len(bbox) != 4:
            return bbox
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Shift by 20-40% of dimensions
        shift_x = random.uniform(0.2, 0.4) * width * random.choice([-1, 1])
        shift_y = random.uniform(0.2, 0.4) * height * random.choice([-1, 1])
        
        new_bbox = [
            max(0, x1 + shift_x),
            max(0, y1 + shift_y),
            max(0, x2 + shift_x),
            max(0, y2 + shift_y)
        ]
        
        return new_bbox
    
    def _corrupt_region(self, region: List[float]) -> List[float]:
        """Corrupt a region specification."""
        if len(region) != 4:
            return region
        
        # Similar to bbox corruption
        return self._corrupt_bbox(region)
    
    def _log_statistics(self):
        """Log augmentation statistics."""
        logger.info("="*50)
        logger.info("Augmentation Statistics:")
        for key, value in self.augmentation_stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*50)


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