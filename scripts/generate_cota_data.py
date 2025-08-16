#!/usr/bin/env python3
"""
Generate CoTA (Chain-of-Thought-Action) Data for Pixelis

This script synthesizes structured training data with Chain-of-Thought-Action trajectories
for training vision-language models with pixel-space reasoning capabilities.
"""

import json
import random
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path to import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import Action, Trajectory, ActionType
from core.modules.operation_registry import VisualOperationRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes and Enums
# ============================================================================

class SampleType(Enum):
    """Types of synthesized samples"""
    POSITIVE = "positive"
    OUTCOME_NEGATIVE = "outcome_negative"
    TRAP_PERCEPTUAL = "trap_perceptual"
    TRAP_LOGICAL = "trap_logical"
    SELF_CORRECTION = "self_correction"

class TaskType(Enum):
    """Types of visual reasoning tasks"""
    OBJECT_COUNTING = "object_counting"
    GEOMETRIC_COMPARISON = "geometric_comparison"
    TEXT_EXTRACTION = "text_extraction"
    SPATIAL_REASONING = "spatial_reasoning"
    TEMPORAL_TRACKING = "temporal_tracking"
    ATTRIBUTE_RECOGNITION = "attribute_recognition"
    RELATIONSHIP_DETECTION = "relationship_detection"

@dataclass
class CoTASample:
    """Represents a single CoTA training sample"""
    sample_id: str
    task_type: TaskType
    sample_type: SampleType
    question: str
    image_path: str
    trajectory: List[Dict[str, Any]]
    answer: str
    ground_truth: str
    provenance: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling_weight: float = 1.0
    difficulty_score: Optional[float] = None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type.value,
            "sample_type": self.sample_type.value,
            "question": self.question,
            "image_path": self.image_path,
            "trajectory": self.trajectory,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "sampling_weight": self.sampling_weight,
            "difficulty_score": self.difficulty_score
        }

# ============================================================================
# Prompt Templates for Diversity
# ============================================================================

PROMPT_TEMPLATES = {
    TaskType.OBJECT_COUNTING: [
        "How many {object_type} are visible in the image?",
        "Count the number of {object_type} shown in this picture.",
        "What is the total count of {object_type} in the scene?",
        "Can you tell me how many {object_type} appear in this image?",
        "Please count all the {object_type} you can see.",
    ],
    TaskType.GEOMETRIC_COMPARISON: [
        "Which object is larger: the one at coordinate ({x1}, {y1}) or the one at ({x2}, {y2})?",
        "Compare the size of objects at positions ({x1}, {y1}) and ({x2}, {y2}). Which is bigger?",
        "Between the items located at ({x1}, {y1}) and ({x2}, {y2}), which has greater area?",
        "What object is smaller: the one at ({x1}, {y1}) or at ({x2}, {y2})?",
        "Analyze the relative sizes of objects at coordinates ({x1}, {y1}) and ({x2}, {y2}).",
    ],
    TaskType.TEXT_EXTRACTION: [
        "What text is written in the {region_description}?",
        "Can you read the text from the {region_description}?",
        "Extract the text content from the {region_description}.",
        "What does it say in the {region_description}?",
        "Please identify the text in the {region_description} of the image.",
    ],
    TaskType.SPATIAL_REASONING: [
        "What is the spatial relationship between the {object1} and the {object2}?",
        "Where is the {object1} located relative to the {object2}?",
        "Describe the position of {object1} with respect to {object2}.",
        "Is the {object1} above, below, left, or right of the {object2}?",
        "How are the {object1} and {object2} positioned relative to each other?",
    ],
    TaskType.TEMPORAL_TRACKING: [
        "Did the {tracked_object} ever enter the {region_description}?",
        "Track the {tracked_object} and determine if it crossed into the {region_description}.",
        "Has the {tracked_object} moved through the {region_description} at any point?",
        "Monitor whether the {tracked_object} passes through the {region_description}.",
        "Did the {tracked_object} remain outside the {region_description} throughout?",
    ],
}

# ============================================================================
# Data Synthesis Engine
# ============================================================================

class CoTADataGenerator:
    """Main engine for generating CoTA training data"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data generator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.registry = VisualOperationRegistry()
        self.synthesis_version = "1.0.0"
        self.samples_generated = 0
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "temperature_range": [0.3, 0.7, 1.0],
            "trap_sample_ratio": 0.2,
            "self_correction_ratio": 0.1,
            "max_trajectory_length": 10,
            "min_trajectory_length": 2,
            "diversity_penalty": 0.5,
            "quality_threshold": 4.0,
            "trap_sample_weight": 1.5,
            "output_format": "json"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def generate_sample_id(self, task_type: TaskType, sample_type: SampleType) -> str:
        """Generate unique sample ID"""
        timestamp = datetime.now().isoformat()
        content = f"{task_type.value}_{sample_type.value}_{timestamp}_{self.samples_generated}"
        hash_id = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"cota_{hash_id}"
    
    def create_provenance(self, source_dataset: str, original_id: str, 
                         synthesis_method: str) -> Dict[str, Any]:
        """Create provenance metadata for a sample"""
        return {
            "source_dataset": source_dataset,
            "original_sample_id": original_id,
            "synthesis_timestamp": datetime.now().isoformat(),
            "synthesis_version": self.synthesis_version,
            "synthesis_method": synthesis_method
        }
    
    def select_temperature(self) -> float:
        """Select temperature for diversity"""
        return random.choice(self.config["temperature_range"])
    
    def select_prompt_template(self, task_type: TaskType, **kwargs) -> str:
        """Select and format a prompt template"""
        templates = PROMPT_TEMPLATES.get(task_type, [])
        if not templates:
            raise ValueError(f"No templates available for task type: {task_type}")
        
        template = random.choice(templates)
        return template.format(**kwargs)
    
    # ========================================================================
    # Task-Specific Generators
    # ========================================================================
    
    def generate_object_counting_sample(self, image_data: Dict[str, Any],
                                       sample_type: SampleType) -> CoTASample:
        """Generate object counting task sample"""
        sample_id = self.generate_sample_id(TaskType.OBJECT_COUNTING, sample_type)
        
        # Extract object annotations from image data
        objects = image_data.get("annotations", [])
        object_types = {}
        for obj in objects:
            obj_type = obj.get("category", "object")
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Select a random object type to count
        if not object_types:
            raise ValueError("No objects found in image annotations")
        
        target_type = random.choice(list(object_types.keys()))
        true_count = object_types[target_type]
        
        # Generate question
        question = self.select_prompt_template(
            TaskType.OBJECT_COUNTING,
            object_type=target_type
        )
        
        # Generate trajectory
        trajectory = self._generate_counting_trajectory(
            target_type, objects, sample_type, true_count
        )
        
        # Determine answer based on sample type
        if sample_type == SampleType.POSITIVE:
            answer = str(true_count)
        elif sample_type == SampleType.OUTCOME_NEGATIVE:
            # Wrong final answer
            answer = str(true_count + random.choice([-2, -1, 1, 2]))
        elif sample_type == SampleType.TRAP_PERCEPTUAL:
            # Miscount due to perceptual error
            answer = str(true_count - 1)
        else:
            answer = str(true_count)
        
        return CoTASample(
            sample_id=sample_id,
            task_type=TaskType.OBJECT_COUNTING,
            sample_type=sample_type,
            question=question,
            image_path=image_data["image_path"],
            trajectory=trajectory,
            answer=answer,
            ground_truth=str(true_count),
            provenance=self.create_provenance(
                image_data["source_dataset"],
                image_data["original_id"],
                "object_counting_synthesis"
            ),
            sampling_weight=self.config["trap_sample_weight"] if "trap" in sample_type.value else 1.0
        )
    
    def generate_geometric_comparison_sample(self, image_data: Dict[str, Any],
                                            sample_type: SampleType) -> CoTASample:
        """Generate geometric comparison task sample"""
        sample_id = self.generate_sample_id(TaskType.GEOMETRIC_COMPARISON, sample_type)
        
        # Select two objects from annotations
        objects = image_data.get("annotations", [])
        if len(objects) < 2:
            raise ValueError("Need at least 2 objects for comparison")
        
        obj1, obj2 = random.sample(objects, 2)
        
        # Get coordinates and areas
        coord1 = (obj1["bbox"][0] + obj1["bbox"][2]/2, 
                 obj1["bbox"][1] + obj1["bbox"][3]/2)
        coord2 = (obj2["bbox"][0] + obj2["bbox"][2]/2,
                 obj2["bbox"][1] + obj2["bbox"][3]/2)
        area1 = obj1["bbox"][2] * obj1["bbox"][3]
        area2 = obj2["bbox"][2] * obj2["bbox"][3]
        
        # Generate question
        question = self.select_prompt_template(
            TaskType.GEOMETRIC_COMPARISON,
            x1=int(coord1[0]), y1=int(coord1[1]),
            x2=int(coord2[0]), y2=int(coord2[1])
        )
        
        # Generate trajectory
        trajectory = self._generate_comparison_trajectory(
            coord1, coord2, area1, area2, sample_type
        )
        
        # Determine answer
        true_answer = f"Object at ({int(coord1[0])}, {int(coord1[1])})" if area1 > area2 else f"Object at ({int(coord2[0])}, {int(coord2[1])})"
        
        if sample_type == SampleType.POSITIVE:
            answer = true_answer
        elif sample_type in [SampleType.OUTCOME_NEGATIVE, SampleType.TRAP_LOGICAL]:
            # Wrong comparison result
            answer = f"Object at ({int(coord2[0])}, {int(coord2[1])})" if area1 > area2 else f"Object at ({int(coord1[0])}, {int(coord1[1])})"
        else:
            answer = true_answer
        
        return CoTASample(
            sample_id=sample_id,
            task_type=TaskType.GEOMETRIC_COMPARISON,
            sample_type=sample_type,
            question=question,
            image_path=image_data["image_path"],
            trajectory=trajectory,
            answer=answer,
            ground_truth=true_answer,
            provenance=self.create_provenance(
                image_data["source_dataset"],
                image_data["original_id"],
                "geometric_comparison_synthesis"
            ),
            sampling_weight=self.config["trap_sample_weight"] if "trap" in sample_type.value else 1.0
        )
    
    def generate_text_extraction_sample(self, image_data: Dict[str, Any],
                                       sample_type: SampleType) -> CoTASample:
        """Generate text extraction task sample"""
        sample_id = self.generate_sample_id(TaskType.TEXT_EXTRACTION, sample_type)
        
        # Get text regions from annotations
        text_regions = image_data.get("text_annotations", [])
        if not text_regions:
            raise ValueError("No text regions found in image")
        
        # Select a random text region
        region = random.choice(text_regions)
        bbox = region["bbox"]
        true_text = region["text"]
        region_desc = region.get("description", "highlighted area")
        
        # Generate question
        question = self.select_prompt_template(
            TaskType.TEXT_EXTRACTION,
            region_description=region_desc
        )
        
        # Generate trajectory
        trajectory = self._generate_text_extraction_trajectory(
            bbox, true_text, sample_type
        )
        
        # Determine answer
        if sample_type == SampleType.POSITIVE:
            answer = true_text
        elif sample_type == SampleType.TRAP_PERCEPTUAL:
            # OCR error - modify a few characters
            chars = list(true_text)
            if len(chars) > 2:
                idx = random.randint(0, len(chars)-1)
                chars[idx] = chr(ord(chars[idx]) + random.choice([-1, 1]))
            answer = "".join(chars)
        else:
            answer = true_text
        
        return CoTASample(
            sample_id=sample_id,
            task_type=TaskType.TEXT_EXTRACTION,
            sample_type=sample_type,
            question=question,
            image_path=image_data["image_path"],
            trajectory=trajectory,
            answer=answer,
            ground_truth=true_text,
            provenance=self.create_provenance(
                image_data["source_dataset"],
                image_data["original_id"],
                "text_extraction_synthesis"
            ),
            sampling_weight=self.config["trap_sample_weight"] if "trap" in sample_type.value else 1.0
        )
    
    def generate_self_correction_sample(self, base_sample: CoTASample) -> CoTASample:
        """Generate a self-correction trajectory from a base sample"""
        sample_id = self.generate_sample_id(base_sample.task_type, SampleType.SELF_CORRECTION)
        
        # Clone the base trajectory
        trajectory = base_sample.trajectory.copy()
        
        # Insert an error in early steps
        if len(trajectory) >= 3:
            error_idx = random.randint(1, min(3, len(trajectory)-1))
            
            # Modify the action at error_idx to be incorrect
            original_action = trajectory[error_idx]
            error_action = original_action.copy()
            
            # Add error based on action type
            if error_action["action"] == "SEGMENT_OBJECT_AT":
                # Use wrong coordinates
                error_action["parameters"]["coordinates"] = [
                    error_action["parameters"]["coordinates"][0] + random.randint(-50, 50),
                    error_action["parameters"]["coordinates"][1] + random.randint(-50, 50)
                ]
                error_action["result"] = "Segmented wrong object"
            
            trajectory[error_idx] = error_action
            
            # Insert correction step
            correction_step = {
                "action": "THINK",
                "thought": "That doesn't seem right. The object I found is not what I was looking for. Let me try a different approach.",
                "parameters": {}
            }
            trajectory.insert(error_idx + 1, correction_step)
            
            # Insert corrected action
            trajectory.insert(error_idx + 2, original_action)
        
        return CoTASample(
            sample_id=sample_id,
            task_type=base_sample.task_type,
            sample_type=SampleType.SELF_CORRECTION,
            question=base_sample.question,
            image_path=base_sample.image_path,
            trajectory=trajectory,
            answer=base_sample.ground_truth,  # Self-correction should reach correct answer
            ground_truth=base_sample.ground_truth,
            provenance=self.create_provenance(
                base_sample.provenance["source_dataset"],
                base_sample.provenance["original_sample_id"],
                "self_correction_synthesis"
            ),
            sampling_weight=1.2  # Slightly higher weight for self-correction samples
        )
    
    # ========================================================================
    # Trajectory Generation Helpers
    # ========================================================================
    
    def _generate_counting_trajectory(self, target_type: str, objects: List[Dict],
                                     sample_type: SampleType, true_count: int) -> List[Dict]:
        """Generate a counting trajectory"""
        trajectory = []
        
        # Initial thought
        trajectory.append({
            "action": "THINK",
            "thought": f"I need to count all the {target_type} in this image. Let me examine the scene carefully.",
            "parameters": {}
        })
        
        # Segment objects
        count = 0
        for obj in objects:
            if obj.get("category") == target_type:
                if sample_type == SampleType.TRAP_PERCEPTUAL and count == true_count - 1:
                    # Skip one object for perceptual trap
                    continue
                    
                bbox = obj["bbox"]
                coord = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                
                trajectory.append({
                    "action": "SEGMENT_OBJECT_AT",
                    "parameters": {"coordinates": [int(coord[0]), int(coord[1])]},
                    "result": f"Found {target_type} at position"
                })
                count += 1
        
        # Final reasoning
        trajectory.append({
            "action": "THINK",
            "thought": f"After examining the entire image, I found {count} {target_type}.",
            "parameters": {}
        })
        
        return trajectory
    
    def _generate_comparison_trajectory(self, coord1: Tuple, coord2: Tuple,
                                       area1: float, area2: float,
                                       sample_type: SampleType) -> List[Dict]:
        """Generate a comparison trajectory"""
        trajectory = []
        
        # Initial thought
        trajectory.append({
            "action": "THINK",
            "thought": "I need to compare the sizes of two objects at the given coordinates.",
            "parameters": {}
        })
        
        # Segment first object
        trajectory.append({
            "action": "SEGMENT_OBJECT_AT",
            "parameters": {"coordinates": [int(coord1[0]), int(coord1[1])]},
            "result": "Segmented first object"
        })
        
        # Get properties of first object
        result1 = area1 if sample_type != SampleType.TRAP_PERCEPTUAL else area1 * 0.8
        trajectory.append({
            "action": "GET_PROPERTIES",
            "parameters": {"mask": "mask_1"},
            "result": {"area": result1, "perimeter": result1 * 4}
        })
        
        # Segment second object
        trajectory.append({
            "action": "SEGMENT_OBJECT_AT",
            "parameters": {"coordinates": [int(coord2[0]), int(coord2[1])]},
            "result": "Segmented second object"
        })
        
        # Get properties of second object
        trajectory.append({
            "action": "GET_PROPERTIES",
            "parameters": {"mask": "mask_2"},
            "result": {"area": area2, "perimeter": area2 * 4}
        })
        
        # Compare
        if sample_type == SampleType.TRAP_LOGICAL:
            # Wrong logical deduction
            comparison = "first object is larger" if area1 < area2 else "second object is larger"
        else:
            comparison = "first object is larger" if result1 > area2 else "second object is larger"
            
        trajectory.append({
            "action": "THINK",
            "thought": f"Comparing the areas: {result1:.0f} vs {area2:.0f}, the {comparison}.",
            "parameters": {}
        })
        
        return trajectory
    
    def _generate_text_extraction_trajectory(self, bbox: List[float], true_text: str,
                                            sample_type: SampleType) -> List[Dict]:
        """Generate a text extraction trajectory"""
        trajectory = []
        
        # Initial thought
        trajectory.append({
            "action": "THINK",
            "thought": "I need to read the text from the specified region in the image.",
            "parameters": {}
        })
        
        # Zoom to text region if needed
        if random.random() < 0.5:
            trajectory.append({
                "action": "ZOOM_IN",
                "parameters": {"bbox": bbox},
                "result": "Zoomed to text region"
            })
        
        # Read text
        if sample_type == SampleType.TRAP_PERCEPTUAL:
            # OCR error
            chars = list(true_text)
            if len(chars) > 2:
                idx = random.randint(0, len(chars)-1)
                chars[idx] = chr(ord(chars[idx]) + random.choice([-1, 1]))
            read_text = "".join(chars)
        else:
            read_text = true_text
            
        trajectory.append({
            "action": "READ_TEXT",
            "parameters": {"bbox": bbox},
            "result": read_text
        })
        
        # Final thought
        trajectory.append({
            "action": "THINK", 
            "thought": f"The text in the region reads: '{read_text}'",
            "parameters": {}
        })
        
        return trajectory
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    def validate_sample(self, sample: CoTASample) -> Tuple[bool, List[str]]:
        """Validate a generated sample
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not sample.sample_id:
            errors.append("Missing sample_id")
        if not sample.question:
            errors.append("Missing question")
        if not sample.image_path:
            errors.append("Missing image_path")
        if not sample.trajectory:
            errors.append("Empty trajectory")
        if sample.answer is None:
            errors.append("Missing answer")
        if sample.ground_truth is None:
            errors.append("Missing ground_truth")
        
        # Check trajectory structure
        if sample.trajectory:
            if len(sample.trajectory) < self.config["min_trajectory_length"]:
                errors.append(f"Trajectory too short: {len(sample.trajectory)}")
            if len(sample.trajectory) > self.config["max_trajectory_length"]:
                errors.append(f"Trajectory too long: {len(sample.trajectory)}")
            
            # Validate each action
            valid_actions = {"THINK", "SEGMENT_OBJECT_AT", "GET_PROPERTIES", 
                           "READ_TEXT", "TRACK_OBJECT", "ZOOM_IN"}
            for i, step in enumerate(sample.trajectory):
                if "action" not in step:
                    errors.append(f"Step {i} missing 'action' field")
                elif step["action"] not in valid_actions:
                    errors.append(f"Step {i} has invalid action: {step['action']}")
                if "parameters" not in step:
                    errors.append(f"Step {i} missing 'parameters' field")
        
        # Check provenance
        if not sample.provenance:
            errors.append("Missing provenance")
        else:
            required_prov_fields = ["source_dataset", "original_sample_id", 
                                   "synthesis_timestamp", "synthesis_version"]
            for field in required_prov_fields:
                if field not in sample.provenance:
                    errors.append(f"Missing provenance field: {field}")
        
        return len(errors) == 0, errors
    
    # ========================================================================
    # Main Generation Pipeline
    # ========================================================================
    
    def generate_dataset(self, image_annotations: List[Dict[str, Any]],
                         num_samples: int,
                         output_path: str) -> Dict[str, Any]:
        """Generate a complete CoTA dataset
        
        Args:
            image_annotations: List of image data with annotations
            num_samples: Number of samples to generate
            output_path: Path to save the generated dataset
            
        Returns:
            Statistics about the generated dataset
        """
        samples = []
        stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "sample_types": {},
            "task_types": {},
            "errors": []
        }
        
        # Calculate sample distribution
        num_positive = int(num_samples * (1 - self.config["trap_sample_ratio"] - 
                                         self.config["self_correction_ratio"]))
        num_trap = int(num_samples * self.config["trap_sample_ratio"])
        num_self_correction = int(num_samples * self.config["self_correction_ratio"])
        
        logger.info(f"Generating {num_samples} samples:")
        logger.info(f"  - Positive: {num_positive}")
        logger.info(f"  - Trap: {num_trap}")
        logger.info(f"  - Self-correction: {num_self_correction}")
        
        # Generate positive samples
        for _ in tqdm(range(num_positive), desc="Generating positive samples"):
            try:
                img_data = random.choice(image_annotations)
                task_type = random.choice(list(TaskType))
                
                if task_type == TaskType.OBJECT_COUNTING:
                    sample = self.generate_object_counting_sample(img_data, SampleType.POSITIVE)
                elif task_type == TaskType.GEOMETRIC_COMPARISON:
                    sample = self.generate_geometric_comparison_sample(img_data, SampleType.POSITIVE)
                elif task_type == TaskType.TEXT_EXTRACTION:
                    sample = self.generate_text_extraction_sample(img_data, SampleType.POSITIVE)
                else:
                    continue
                
                is_valid, errors = self.validate_sample(sample)
                if is_valid:
                    samples.append(sample)
                    stats["valid_samples"] += 1
                else:
                    stats["invalid_samples"] += 1
                    stats["errors"].extend(errors)
                    
                self.samples_generated += 1
                
            except Exception as e:
                logger.error(f"Error generating sample: {e}")
                stats["errors"].append(str(e))
        
        # Generate trap samples
        trap_types = [SampleType.TRAP_PERCEPTUAL, SampleType.TRAP_LOGICAL, 
                     SampleType.OUTCOME_NEGATIVE]
        for _ in tqdm(range(num_trap), desc="Generating trap samples"):
            try:
                img_data = random.choice(image_annotations)
                task_type = random.choice(list(TaskType))
                trap_type = random.choice(trap_types)
                
                if task_type == TaskType.OBJECT_COUNTING:
                    sample = self.generate_object_counting_sample(img_data, trap_type)
                elif task_type == TaskType.GEOMETRIC_COMPARISON:
                    sample = self.generate_geometric_comparison_sample(img_data, trap_type)
                elif task_type == TaskType.TEXT_EXTRACTION:
                    sample = self.generate_text_extraction_sample(img_data, trap_type)
                else:
                    continue
                
                is_valid, errors = self.validate_sample(sample)
                if is_valid:
                    samples.append(sample)
                    stats["valid_samples"] += 1
                else:
                    stats["invalid_samples"] += 1
                    stats["errors"].extend(errors)
                    
                self.samples_generated += 1
                
            except Exception as e:
                logger.error(f"Error generating trap sample: {e}")
                stats["errors"].append(str(e))
        
        # Generate self-correction samples
        positive_samples = [s for s in samples if s.sample_type == SampleType.POSITIVE]
        for _ in tqdm(range(min(num_self_correction, len(positive_samples))), 
                     desc="Generating self-correction samples"):
            try:
                base_sample = random.choice(positive_samples)
                sample = self.generate_self_correction_sample(base_sample)
                
                is_valid, errors = self.validate_sample(sample)
                if is_valid:
                    samples.append(sample)
                    stats["valid_samples"] += 1
                else:
                    stats["invalid_samples"] += 1
                    stats["errors"].extend(errors)
                    
                self.samples_generated += 1
                
            except Exception as e:
                logger.error(f"Error generating self-correction sample: {e}")
                stats["errors"].append(str(e))
        
        # Calculate statistics
        stats["total_samples"] = len(samples)
        for sample in samples:
            sample_type = sample.sample_type.value
            task_type = sample.task_type.value
            stats["sample_types"][sample_type] = stats["sample_types"].get(sample_type, 0) + 1
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
        
        # Save dataset
        output_data = {
            "metadata": {
                "synthesis_version": self.synthesis_version,
                "generation_timestamp": datetime.now().isoformat(),
                "config": self.config,
                "statistics": stats
            },
            "samples": [sample.to_json() for sample in samples]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        return stats

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate CoTA training data for Pixelis"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to image annotations file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cota_dataset.json",
        help="Output path for generated dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load image annotations
    logger.info(f"Loading annotations from {args.annotations}")
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)
    
    # Initialize generator
    generator = CoTADataGenerator(args.config)
    
    # Generate dataset
    stats = generator.generate_dataset(
        annotations,
        args.num_samples,
        args.output
    )
    
    # Print final statistics
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Invalid samples: {stats['invalid_samples']}")
    print("\nSample type distribution:")
    for sample_type, count in stats['sample_types'].items():
        print(f"  {sample_type}: {count}")
    print("\nTask type distribution:")
    for task_type, count in stats['task_types'].items():
        print(f"  {task_type}: {count}")
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        print("First 5 errors:")
        for error in stats['errors'][:5]:
            print(f"  - {error}")

if __name__ == "__main__":
    main()