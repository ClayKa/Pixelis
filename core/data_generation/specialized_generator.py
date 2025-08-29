"""
Specialized Task Generator Framework with SOLID Principles
This module provides a comprehensive framework for generating Chain-of-Thought-Action (CoTA) data
following SOLID design principles for maintainability and extensibility.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Protocol, Union
import hashlib
import random
from enum import Enum

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ======================== INTERFACES (Dependency Inversion Principle) ========================

class DataLoader(Protocol):
    """Protocol for data loaders - defines the expected interface"""
    def load_sample(self, index: int) -> Dict[str, Any]:
        """Load a single sample by index"""
        ...
    
    def get_total_samples(self) -> int:
        """Get total number of samples available"""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        ...


class LLMClient(Protocol):
    """Protocol for LLM API clients"""
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Generate text from prompt"""
        ...
    
    def score_response(self, response: str, criteria: Dict[str, Any]) -> float:
        """Score a response based on criteria"""
        ...


class QualityScorer(Protocol):
    """Protocol for quality scoring systems"""
    def score(self, trajectory: Dict[str, Any]) -> float:
        """Score a trajectory's quality"""
        ...


# ======================== VALUE OBJECTS (Single Responsibility Principle) ========================

class TrajectoryType(Enum):
    """Types of trajectories for training"""
    GOLDEN = "golden_positive"
    TRAP = "trap_sample"
    SELF_CORRECTION = "self_correction"


class DifficultyLevel(Enum):
    """Difficulty levels for tasks"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class VisualOperation:
    """Represents a single visual operation in a trajectory"""
    name: str
    parameters: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation": self.name,
            "params": self.parameters,
            "output": self.expected_output,
            "time_ms": self.execution_time_ms
        }


@dataclass
class ThoughtStep:
    """Represents a reasoning step in the chain of thought"""
    content: str
    step_type: str  # "analysis", "planning", "reflection", "correction"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.content,
            "type": self.step_type,
            "confidence": self.confidence
        }


@dataclass
class CoTATrajectory:
    """Complete Chain-of-Thought-Action trajectory"""
    task_id: str
    task_description: str
    thoughts: List[ThoughtStep]
    actions: List[VisualOperation]
    final_answer: str
    trajectory_type: TrajectoryType
    difficulty: DifficultyLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.task_id,
            "task": self.task_description,
            "chain_of_thought": [t.to_dict() for t in self.thoughts],
            "visual_operations": [a.to_dict() for a in self.actions],
            "answer": self.final_answer,
            "type": self.trajectory_type.value,
            "difficulty": self.difficulty.value,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "provenance": self.provenance
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate trajectory completeness and consistency"""
        errors = []
        
        if not self.task_id:
            errors.append("Missing task_id")
        if not self.task_description:
            errors.append("Missing task_description")
        if len(self.thoughts) == 0:
            errors.append("No thought steps present")
        if len(self.actions) == 0:
            errors.append("No visual operations present")
        if not self.final_answer:
            errors.append("Missing final_answer")
        if self.quality_score is not None and not 0 <= self.quality_score <= 1:
            errors.append(f"Invalid quality_score: {self.quality_score}")
        
        return len(errors) == 0, errors


# ======================== CORE ABSTRACTIONS (Open/Closed Principle) ========================

class PromptTemplate(ABC):
    """Abstract base for prompt templates"""
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt with given parameters"""
        pass
    
    @abstractmethod
    def get_variables(self) -> List[str]:
        """Get list of required variables"""
        pass


class SimplePromptTemplate(PromptTemplate):
    """Simple string-based prompt template"""
    
    def __init__(self, template: str):
        self.template = template
        self._variables = self._extract_variables(template)
    
    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def get_variables(self) -> List[str]:
        return self._variables
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template string"""
        import re
        return re.findall(r'\{(\w+)\}', template)


class TrajectoryAugmenter:
    """Handles trajectory augmentation for creating diverse training samples"""
    
    def __init__(self, trap_probability: float = 0.2, correction_probability: float = 0.2):
        self.trap_probability = trap_probability
        self.correction_probability = correction_probability
    
    def augment(self, trajectory: CoTATrajectory) -> CoTATrajectory:
        """Augment a golden trajectory into trap or self-correction variant"""
        rand = random.random()
        
        if rand < self.trap_probability:
            return self._create_trap_trajectory(trajectory)
        elif rand < self.trap_probability + self.correction_probability:
            return self._create_correction_trajectory(trajectory)
        else:
            return trajectory
    
    def _create_trap_trajectory(self, golden: CoTATrajectory) -> CoTATrajectory:
        """Create a trap trajectory with subtle errors"""
        trap = CoTATrajectory(
            task_id=f"{golden.task_id}_trap",
            task_description=golden.task_description,
            thoughts=golden.thoughts.copy(),
            actions=golden.actions.copy(),
            final_answer=golden.final_answer,
            trajectory_type=TrajectoryType.TRAP,
            difficulty=golden.difficulty,
            metadata={**golden.metadata, "original_id": golden.task_id},
            provenance=golden.provenance
        )
        
        # Introduce subtle error in middle of trajectory
        if len(trap.actions) > 2:
            error_idx = len(trap.actions) // 2
            original_action = trap.actions[error_idx]
            
            # Modify parameters slightly
            error_params = original_action.parameters.copy()
            if "coordinates" in error_params:
                # Shift coordinates slightly
                coords = error_params["coordinates"]
                if isinstance(coords, list) and len(coords) >= 2:
                    error_params["coordinates"] = [coords[0] + 50, coords[1] + 50]
            
            trap.actions[error_idx] = VisualOperation(
                name=original_action.name,
                parameters=error_params,
                expected_output={"error": "incorrect_target"}
            )
            
            # Add confused thought
            trap.thoughts.insert(
                error_idx + 1,
                ThoughtStep(
                    content="The result seems unexpected, but I'll continue.",
                    step_type="reflection",
                    confidence=0.6
                )
            )
        
        return trap
    
    def _create_correction_trajectory(self, golden: CoTATrajectory) -> CoTATrajectory:
        """Create a self-correction trajectory"""
        correction = CoTATrajectory(
            task_id=f"{golden.task_id}_correction",
            task_description=golden.task_description,
            thoughts=[],
            actions=[],
            final_answer=golden.final_answer,
            trajectory_type=TrajectoryType.SELF_CORRECTION,
            difficulty=golden.difficulty,
            metadata={**golden.metadata, "original_id": golden.task_id},
            provenance=golden.provenance
        )
        
        # Start with normal execution
        mid_point = len(golden.actions) // 2
        correction.thoughts = golden.thoughts[:mid_point].copy()
        correction.actions = golden.actions[:mid_point].copy()
        
        # Add error and correction
        correction.thoughts.append(
            ThoughtStep(
                content="Wait, I think I made an error. Let me reconsider.",
                step_type="correction",
                confidence=0.8
            )
        )
        
        correction.thoughts.append(
            ThoughtStep(
                content="I need to re-examine the previous step more carefully.",
                step_type="reflection",
                confidence=0.9
            )
        )
        
        # Complete with corrected trajectory
        correction.actions.extend(golden.actions[mid_point:])
        correction.thoughts.extend(golden.thoughts[mid_point:])
        
        return correction


# ======================== TASK GENERATORS (Liskov Substitution Principle) ========================

class SpecializedTaskGenerator(ABC):
    """Abstract base for specialized task generators"""
    
    def __init__(
        self,
        task_name: str,
        data_loaders: Dict[str, DataLoader],
        llm_client: LLMClient,
        quality_scorer: Optional[QualityScorer] = None,
        output_dir: Optional[Path] = None
    ):
        self.task_name = task_name
        self.data_loaders = data_loaders
        self.llm_client = llm_client
        self.quality_scorer = quality_scorer
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_generated": 0,
            "total_failed": 0,
            "quality_distribution": [],
            "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
            "type_distribution": {"golden": 0, "trap": 0, "correction": 0}
        }
    
    @abstractmethod
    def generate_trajectory(
        self,
        sample_data: Dict[str, Any],
        difficulty: DifficultyLevel
    ) -> Optional[CoTATrajectory]:
        """Generate a single trajectory from sample data"""
        pass
    
    @abstractmethod
    def get_prompt_template(self, difficulty: DifficultyLevel) -> PromptTemplate:
        """Get appropriate prompt template for difficulty level"""
        pass
    
    def generate_dataset(
        self,
        target_count: int,
        difficulty_distribution: Dict[DifficultyLevel, float] = None,
        augmentation_config: Optional[Dict[str, float]] = None
    ) -> List[CoTATrajectory]:
        """Generate complete dataset of trajectories"""
        if difficulty_distribution is None:
            difficulty_distribution = {
                DifficultyLevel.EASY: 0.3,
                DifficultyLevel.MEDIUM: 0.5,
                DifficultyLevel.HARD: 0.2
            }
        
        augmenter = None
        if augmentation_config:
            augmenter = TrajectoryAugmenter(**augmentation_config)
        
        trajectories = []
        progress_bar = tqdm(total=target_count, desc=f"Generating {self.task_name}")
        
        while len(trajectories) < target_count:
            # Select difficulty level
            difficulty = self._sample_difficulty(difficulty_distribution)
            
            # Get sample data
            loader = self._select_loader_for_difficulty(difficulty)
            if not loader:
                logger.warning(f"No loader available for {difficulty}")
                continue
            
            try:
                sample_idx = random.randint(0, loader.get_total_samples() - 1)
                sample_data = loader.load_sample(sample_idx)
                
                # Generate trajectory
                trajectory = self.generate_trajectory(sample_data, difficulty)
                
                if trajectory:
                    # Validate
                    is_valid, errors = trajectory.validate()
                    if not is_valid:
                        logger.warning(f"Invalid trajectory: {errors}")
                        self.stats["total_failed"] += 1
                        continue
                    
                    # Score quality
                    if self.quality_scorer:
                        trajectory.quality_score = self.quality_scorer.score(trajectory.to_dict())
                        self.stats["quality_distribution"].append(trajectory.quality_score)
                    
                    # Augment if configured
                    if augmenter:
                        trajectory = augmenter.augment(trajectory)
                    
                    # Update statistics
                    self._update_stats(trajectory)
                    
                    trajectories.append(trajectory)
                    progress_bar.update(1)
                    
                    # Checkpoint periodically
                    if len(trajectories) % 100 == 0:
                        self._save_checkpoint(trajectories)
                
            except Exception as e:
                logger.error(f"Error generating trajectory: {e}")
                self.stats["total_failed"] += 1
        
        progress_bar.close()
        
        # Final save
        self._save_dataset(trajectories)
        self._save_statistics()
        
        return trajectories
    
    def _sample_difficulty(self, distribution: Dict[DifficultyLevel, float]) -> DifficultyLevel:
        """Sample difficulty level based on distribution"""
        levels = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(levels, weights=weights)[0]
    
    def _select_loader_for_difficulty(self, difficulty: DifficultyLevel) -> Optional[DataLoader]:
        """Select appropriate data loader for difficulty level"""
        # Override in subclasses for specific logic
        if self.data_loaders:
            return random.choice(list(self.data_loaders.values()))
        return None
    
    def _update_stats(self, trajectory: CoTATrajectory):
        """Update generation statistics"""
        self.stats["total_generated"] += 1
        self.stats["difficulty_distribution"][trajectory.difficulty.value] += 1
        self.stats["type_distribution"][trajectory.trajectory_type.value] += 1
    
    def _save_checkpoint(self, trajectories: List[CoTATrajectory]):
        """Save checkpoint of current progress"""
        checkpoint_path = self.output_dir / f"{self.task_name}_checkpoint_{len(trajectories)}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(
                [t.to_dict() for t in trajectories],
                f,
                indent=2
            )
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_dataset(self, trajectories: List[CoTATrajectory]):
        """Save final dataset"""
        output_path = self.output_dir / f"{self.task_name}_dataset.json"
        with open(output_path, 'w') as f:
            json.dump(
                {
                    "task": self.task_name,
                    "total_samples": len(trajectories),
                    "trajectories": [t.to_dict() for t in trajectories],
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "statistics": self.stats
                    }
                },
                f,
                indent=2
            )
        logger.info(f"Dataset saved: {output_path}")
    
    def _save_statistics(self):
        """Save generation statistics"""
        stats_path = self.output_dir / f"{self.task_name}_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics saved: {stats_path}")


# ======================== CONCRETE IMPLEMENTATIONS ========================

class EnhancedGeometricReasoningGenerator(SpecializedTaskGenerator):
    """Generator for geometric reasoning tasks using SEGMENT_OBJECT_AT + GET_PROPERTIES"""
    
    def get_prompt_template(self, difficulty: DifficultyLevel) -> PromptTemplate:
        """Get prompt template based on difficulty"""
        if difficulty == DifficultyLevel.EASY:
            template = """
Given an image containing {num_objects} objects, identify and compare their geometric properties.

Task: {task_description}

Generate a chain of thought and visual operations to solve this task.
Format your response as:
1. Initial analysis thought
2. SEGMENT_OBJECT_AT operation for first object
3. GET_PROPERTIES to extract geometric features
4. Repeat for other objects
5. Comparison and final answer

Be precise with coordinates and property extraction.
"""
        elif difficulty == DifficultyLevel.MEDIUM:
            template = """
Complex geometric analysis required for {num_objects} objects with occlusion.

Task: {task_description}
Constraints: {constraints}

Generate detailed reasoning with visual operations handling:
- Partial visibility
- Overlapping objects
- Precise boundary detection
- Multi-property comparison

Include confidence scores for uncertain detections.
"""
        else:  # HARD
            template = """
Advanced geometric reasoning challenge with {num_objects} objects.

Task: {task_description}
Constraints: {constraints}
Special requirements: {special_requirements}

Generate expert-level analysis including:
- Multiple segmentation attempts if needed
- Handling ambiguous boundaries
- Complex property calculations
- Error detection and correction
- Detailed geometric relationships

Use all available visual operations optimally.
"""
        
        return SimplePromptTemplate(template)
    
    def generate_trajectory(
        self,
        sample_data: Dict[str, Any],
        difficulty: DifficultyLevel
    ) -> Optional[CoTATrajectory]:
        """Generate geometric reasoning trajectory"""
        
        # Extract relevant data from sample
        image_info = sample_data.get("image", {})
        objects = sample_data.get("objects", [])
        
        if not objects:
            logger.warning("No objects found in sample data")
            return None
        
        # Prepare prompt
        prompt_template = self.get_prompt_template(difficulty)
        prompt = prompt_template.format(
            num_objects=len(objects),
            task_description=self._generate_task_description(objects, difficulty),
            constraints=self._generate_constraints(difficulty),
            special_requirements=self._generate_special_requirements(difficulty)
        )
        
        # Generate with LLM
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7 if difficulty == DifficultyLevel.EASY else 0.8,
                max_tokens=2048
            )
            
            # Parse response into trajectory
            trajectory = self._parse_response_to_trajectory(
                response,
                sample_data,
                difficulty
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Failed to generate trajectory: {e}")
            return None
    
    def _generate_task_description(self, objects: List[Dict], difficulty: DifficultyLevel) -> str:
        """Generate task description based on objects and difficulty"""
        if difficulty == DifficultyLevel.EASY:
            return f"Compare the sizes of the {objects[0]['category']} and {objects[1]['category']}"
        elif difficulty == DifficultyLevel.MEDIUM:
            return f"Analyze spatial relationships between {len(objects)} objects and identify the most centrally located one"
        else:
            return f"Perform complex geometric analysis of {len(objects)} objects including shape similarity, spatial clustering, and hierarchical relationships"
    
    def _generate_constraints(self, difficulty: DifficultyLevel) -> str:
        """Generate constraints based on difficulty"""
        if difficulty == DifficultyLevel.EASY:
            return "Clear visibility, simple shapes"
        elif difficulty == DifficultyLevel.MEDIUM:
            return "Partial occlusion, varied lighting"
        else:
            return "Heavy occlusion, complex shapes, ambiguous boundaries"
    
    def _generate_special_requirements(self, difficulty: DifficultyLevel) -> str:
        """Generate special requirements for hard tasks"""
        if difficulty == DifficultyLevel.HARD:
            return "Handle edge cases, verify results, provide confidence scores"
        return "Standard precision required"
    
    def _parse_response_to_trajectory(
        self,
        response: str,
        sample_data: Dict[str, Any],
        difficulty: DifficultyLevel
    ) -> CoTATrajectory:
        """Parse LLM response into structured trajectory"""
        
        # This is a simplified parser - in production, use more robust parsing
        lines = response.strip().split('\n')
        thoughts = []
        actions = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("SEGMENT_OBJECT_AT"):
                # Parse visual operation
                # Extract coordinates from line (simplified)
                actions.append(VisualOperation(
                    name="SEGMENT_OBJECT_AT",
                    parameters={"coordinates": [100, 200]},  # Parse from line
                    expected_output={"mask": "binary_mask_data"}
                ))
            elif line.startswith("GET_PROPERTIES"):
                actions.append(VisualOperation(
                    name="GET_PROPERTIES",
                    parameters={"mask": "previous_mask"},
                    expected_output={"area": 5000, "perimeter": 300}
                ))
            elif line and not line.startswith('#'):
                # Treat as thought
                thoughts.append(ThoughtStep(
                    content=line,
                    step_type="analysis" if "analyze" in line.lower() else "planning",
                    confidence=0.9
                ))
        
        # Extract final answer (last non-empty line)
        final_answer = lines[-1] if lines else "No answer generated"
        
        # Generate unique task ID
        task_id = hashlib.md5(
            f"{sample_data.get('id', 'unknown')}_{difficulty.value}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        return CoTATrajectory(
            task_id=task_id,
            task_description=self._generate_task_description(sample_data.get("objects", []), difficulty),
            thoughts=thoughts,
            actions=actions,
            final_answer=final_answer,
            trajectory_type=TrajectoryType.GOLDEN,
            difficulty=difficulty,
            metadata={
                "source_sample_id": sample_data.get("id"),
                "image_size": sample_data.get("image", {}).get("size"),
                "num_objects": len(sample_data.get("objects", []))
            },
            provenance={
                "generator": self.__class__.__name__,
                "llm_model": "configured_model",
                "timestamp": datetime.now().isoformat()
            }
        )


# ======================== FACTORY PATTERN (for creating generators) ========================

class TaskGeneratorFactory:
    """Factory for creating specialized task generators"""
    
    _generators = {}
    
    @classmethod
    def register(cls, task_type: str, generator_class: type):
        """Register a generator class for a task type"""
        cls._generators[task_type] = generator_class
    
    @classmethod
    def create(
        cls,
        task_type: str,
        data_loaders: Dict[str, DataLoader],
        llm_client: LLMClient,
        **kwargs
    ) -> SpecializedTaskGenerator:
        """Create a generator instance"""
        if task_type not in cls._generators:
            raise ValueError(f"Unknown task type: {task_type}")
        
        generator_class = cls._generators[task_type]
        return generator_class(
            task_name=task_type,
            data_loaders=data_loaders,
            llm_client=llm_client,
            **kwargs
        )
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available task types"""
        return list(cls._generators.keys())


# Register the enhanced generator
TaskGeneratorFactory.register("geometric_reasoning", EnhancedGeometricReasoningGenerator)


# ======================== QUALITY SCORING SYSTEM ========================

class ComprehensiveQualityScorer:
    """Comprehensive quality scoring for trajectories"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "completeness": 0.25,
            "coherence": 0.25,
            "efficiency": 0.20,
            "correctness": 0.30
        }
    
    def score(self, trajectory: Dict[str, Any]) -> float:
        """Score a trajectory on multiple dimensions"""
        scores = {
            "completeness": self._score_completeness(trajectory),
            "coherence": self._score_coherence(trajectory),
            "efficiency": self._score_efficiency(trajectory),
            "correctness": self._score_correctness(trajectory)
        }
        
        # Weighted average
        total_score = sum(
            scores[dim] * self.weights[dim]
            for dim in scores
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    def _score_completeness(self, trajectory: Dict[str, Any]) -> float:
        """Score based on trajectory completeness"""
        required_fields = ["id", "task", "chain_of_thought", "visual_operations", "answer"]
        present_fields = sum(1 for field in required_fields if field in trajectory and trajectory[field])
        return present_fields / len(required_fields)
    
    def _score_coherence(self, trajectory: Dict[str, Any]) -> float:
        """Score based on logical coherence"""
        thoughts = trajectory.get("chain_of_thought", [])
        if not thoughts:
            return 0.0
        
        # Check for logical flow
        coherence_score = 1.0
        for i in range(1, len(thoughts)):
            # Simplified coherence check
            if thoughts[i].get("type") == "correction" and thoughts[i-1].get("confidence", 1.0) > 0.8:
                coherence_score -= 0.1  # Correction after high confidence is suspicious
        
        return max(coherence_score, 0.0)
    
    def _score_efficiency(self, trajectory: Dict[str, Any]) -> float:
        """Score based on operation efficiency"""
        operations = trajectory.get("visual_operations", [])
        if not operations:
            return 0.0
        
        # Penalize redundant operations
        seen_ops = set()
        redundancy_penalty = 0
        for op in operations:
            op_key = f"{op.get('operation')}_{str(op.get('params'))}"
            if op_key in seen_ops:
                redundancy_penalty += 0.1
            seen_ops.add(op_key)
        
        return max(1.0 - redundancy_penalty, 0.0)
    
    def _score_correctness(self, trajectory: Dict[str, Any]) -> float:
        """Score based on answer correctness (simplified)"""
        # In production, this would validate against ground truth
        answer = trajectory.get("answer", "")
        if not answer:
            return 0.0
        
        # Basic heuristics
        if len(answer) < 5:
            return 0.5  # Too short
        if len(answer) > 500:
            return 0.7  # Too verbose
        
        return 0.9  # Assume mostly correct for now


# ======================== USAGE EXAMPLE ========================

def example_usage():
    """Example of how to use the specialized generator framework"""
    
    # Mock implementations for demonstration
    class MockDataLoader:
        def load_sample(self, index: int) -> Dict[str, Any]:
            return {
                "id": f"sample_{index}",
                "image": {"size": [640, 480]},
                "objects": [
                    {"category": "cat", "bbox": [100, 100, 200, 200]},
                    {"category": "dog", "bbox": [300, 150, 400, 250]}
                ]
            }
        
        def get_total_samples(self) -> int:
            return 1000
        
        def get_metadata(self) -> Dict[str, Any]:
            return {"dataset": "mock", "version": "1.0"}
    
    class MockLLMClient:
        def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
            return """
            Analyzing the image to compare object sizes.
            SEGMENT_OBJECT_AT [150, 150]
            The segmentation reveals the cat object.
            GET_PROPERTIES
            Extracting geometric properties: area=10000, perimeter=400
            SEGMENT_OBJECT_AT [350, 200]
            The segmentation reveals the dog object.
            GET_PROPERTIES
            Extracting geometric properties: area=15000, perimeter=500
            The dog is 1.5 times larger than the cat.
            """
        
        def score_response(self, response: str, criteria: Dict[str, Any]) -> float:
            return 0.85
    
    # Create instances
    data_loaders = {"mock": MockDataLoader()}
    llm_client = MockLLMClient()
    quality_scorer = ComprehensiveQualityScorer()
    
    # Create generator
    generator = TaskGeneratorFactory.create(
        task_type="geometric_reasoning",
        data_loaders=data_loaders,
        llm_client=llm_client,
        quality_scorer=quality_scorer,
        output_dir=Path("test_outputs")
    )
    
    # Generate dataset
    trajectories = generator.generate_dataset(
        target_count=10,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.4,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.2
        },
        augmentation_config={
            "trap_probability": 0.2,
            "correction_probability": 0.2
        }
    )
    
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Statistics: {generator.stats}")


if __name__ == "__main__":
    # Run example when executed directly
    example_usage()