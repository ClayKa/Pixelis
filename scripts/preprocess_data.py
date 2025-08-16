#!/usr/bin/env python3
"""
Preprocess CoTA Data with Difficulty Scoring for Curriculum Learning

This script loads raw CoTA data, calculates composite difficulty scores,
categorizes samples into difficulty levels, and saves the processed data
for use in curriculum-based supervised fine-tuning.
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from tqdm import tqdm
import yaml
import hashlib

# Add parent directory to path to import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_structures import Action, Trajectory, ActionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration and Constants
# ============================================================================

# Operation complexity weights for difficulty scoring
OPERATION_COMPLEXITY = {
    "TRACK_OBJECT": 3.0,      # Most complex - temporal reasoning
    "SEGMENT_OBJECT_AT": 3.0,  # Complex - spatial localization
    "GET_PROPERTIES": 2.0,     # Medium - analysis
    "ZOOM_IN": 2.0,           # Medium - navigation
    "READ_TEXT": 1.0,         # Simple - extraction
    "THINK": 0.5              # Reasoning step
}

# Task type base difficulty
TASK_DIFFICULTY = {
    "temporal_tracking": 1.0,
    "geometric_comparison": 0.8,
    "spatial_reasoning": 0.7,
    "relationship_detection": 0.6,
    "attribute_recognition": 0.5,
    "object_counting": 0.4,
    "text_extraction": 0.3
}

# Sample type difficulty modifiers
SAMPLE_TYPE_MODIFIERS = {
    "positive": 0.0,
    "outcome_negative": 0.1,
    "trap_perceptual": 0.2,
    "trap_logical": 0.25,
    "self_correction": 0.15
}

# Default configuration
DEFAULT_CONFIG = {
    "difficulty_weights": {
        "trajectory_complexity": 0.30,
        "operation_sophistication": 0.25,
        "reasoning_depth": 0.20,
        "error_patterns": 0.15,
        "task_type": 0.10
    },
    "categorization": {
        "simple_percentile": 33,
        "medium_percentile": 66,
        "min_difficulty": 0.1,
        "max_difficulty": 1.0
    },
    "trajectory_limits": {
        "min_length": 2,
        "max_length": 15,
        "optimal_thinking_ratio_min": 0.2,
        "optimal_thinking_ratio_max": 0.6
    },
    "curriculum": {
        "initial_simple_ratio": 0.8,
        "initial_medium_ratio": 0.2,
        "progression_rate": 0.1
    },
    "output_format": "json",
    "generate_statistics": True,
    "validate_distribution": True
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DifficultyMetrics:
    """Detailed difficulty metrics for a sample"""
    trajectory_length: int
    operation_count: int
    unique_operations: int
    thinking_steps: int
    repetition_count: int
    has_self_correction: bool
    has_backtracking: bool
    operation_sophistication_score: float
    reasoning_depth_ratio: float
    error_pattern_score: float
    task_type_score: float
    composite_score: float
    difficulty_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class ProcessedSample:
    """Processed CoTA sample with difficulty scoring"""
    sample_id: str
    task_type: str
    sample_type: str
    question: str
    image_path: str
    trajectory: List[Dict[str, Any]]
    answer: str
    ground_truth: str
    provenance: Dict[str, Any]
    metadata: Dict[str, Any]
    sampling_weight: float
    difficulty_score: float
    difficulty_category: str
    difficulty_metrics: DifficultyMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "sample_type": self.sample_type,
            "question": self.question,
            "image_path": self.image_path,
            "trajectory": self.trajectory,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "sampling_weight": self.sampling_weight,
            "difficulty_score": self.difficulty_score,
            "difficulty_category": self.difficulty_category,
            "difficulty_metrics": self.difficulty_metrics.to_dict()
        }

# ============================================================================
# Difficulty Scoring Engine
# ============================================================================

class DifficultyScorer:
    """Calculate composite difficulty scores for CoTA samples"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the difficulty scorer
        
        Args:
            config: Configuration dictionary
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.weights = self.config["difficulty_weights"]
        self.limits = self.config["trajectory_limits"]
        self.categorization = self.config["categorization"]
        
        # Statistics tracking
        self.score_distribution = []
        self.category_counts = Counter()
        
    def calculate_trajectory_complexity(self, trajectory: List[Dict]) -> float:
        """Calculate trajectory complexity based on length
        
        Uses log scaling to handle outliers gracefully.
        
        Args:
            trajectory: List of trajectory steps
            
        Returns:
            Normalized complexity score [0, 1]
        """
        length = len(trajectory)
        
        # Handle edge cases
        if length < self.limits["min_length"]:
            return 0.1
        
        # Log scale to handle outliers
        max_length = self.limits["max_length"]
        score = np.log(length + 1) / np.log(max_length + 1)
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def calculate_operation_sophistication(self, trajectory: List[Dict]) -> Tuple[float, int, int]:
        """Calculate operation sophistication score
        
        Args:
            trajectory: List of trajectory steps
            
        Returns:
            Tuple of (sophistication_score, operation_count, unique_count)
        """
        operations = []
        total_complexity = 0.0
        max_possible = 0.0
        
        for step in trajectory:
            action = step.get("action", "")
            if action in OPERATION_COMPLEXITY:
                operations.append(action)
                total_complexity += OPERATION_COMPLEXITY[action]
                max_possible += OPERATION_COMPLEXITY["TRACK_OBJECT"]  # Max complexity
        
        if max_possible == 0:
            return 0.3, 0, 0  # Base score for no operations
        
        sophistication_score = total_complexity / max_possible
        unique_operations = len(set(operations))
        
        # Bonus for operation diversity
        if len(operations) > 0:
            diversity_bonus = unique_operations / len(operations) * 0.2
            sophistication_score = min(sophistication_score + diversity_bonus, 1.0)
        
        return sophistication_score, len(operations), unique_operations
    
    def calculate_reasoning_depth(self, trajectory: List[Dict]) -> Tuple[float, int]:
        """Calculate reasoning depth based on thinking steps
        
        Args:
            trajectory: List of trajectory steps
            
        Returns:
            Tuple of (depth_score, thinking_step_count)
        """
        thinking_steps = sum(1 for step in trajectory if step.get("action") == "THINK")
        total_steps = len(trajectory)
        
        if total_steps == 0:
            return 0.0, 0
        
        ratio = thinking_steps / total_steps
        
        # Optimal ratio is between min and max thresholds
        min_ratio = self.limits["optimal_thinking_ratio_min"]
        max_ratio = self.limits["optimal_thinking_ratio_max"]
        
        if min_ratio <= ratio <= max_ratio:
            # Optimal range
            score = 1.0
        elif ratio < min_ratio:
            # Too few thinking steps
            score = ratio / min_ratio
        else:
            # Too many thinking steps (might indicate confusion)
            score = 1.0 - (ratio - max_ratio) / (1.0 - max_ratio) * 0.3
        
        return score, thinking_steps
    
    def detect_error_patterns(self, trajectory: List[Dict], sample_type: str) -> Tuple[float, bool, bool, int]:
        """Detect error patterns and corrections in trajectory
        
        Args:
            trajectory: List of trajectory steps
            sample_type: Type of sample
            
        Returns:
            Tuple of (error_score, has_correction, has_backtracking, repetition_count)
        """
        error_score = 0.0
        has_correction = sample_type == "self_correction"
        has_backtracking = False
        repetition_count = 0
        
        # Check for self-correction patterns
        for i, step in enumerate(trajectory):
            thought = step.get("thought", "").lower()
            if any(phrase in thought for phrase in 
                   ["doesn't seem right", "let me try", "actually", "wait", "correction"]):
                has_correction = True
                error_score += 0.3
                break
        
        # Check for repetitive operations
        recent_ops = []
        for step in trajectory:
            if step.get("action") in OPERATION_COMPLEXITY:
                recent_ops.append(step["action"])
                
                # Check last 3 operations for repetition
                if len(recent_ops) >= 3:
                    if len(set(recent_ops[-3:])) == 1:
                        repetition_count += 1
                        error_score += 0.2
        
        # Check for backtracking (returning to previous coordinates)
        coordinates_seen = []
        for step in trajectory:
            if step.get("action") == "SEGMENT_OBJECT_AT":
                coords = step.get("parameters", {}).get("coordinates")
                if coords:
                    coords_tuple = tuple(coords)
                    if coords_tuple in coordinates_seen:
                        has_backtracking = True
                        error_score += 0.2
                    coordinates_seen.append(coords_tuple)
        
        # Apply sample type modifier
        if sample_type in SAMPLE_TYPE_MODIFIERS:
            error_score += SAMPLE_TYPE_MODIFIERS[sample_type]
        
        return min(error_score, 1.0), has_correction, has_backtracking, repetition_count
    
    def get_task_difficulty(self, task_type: str) -> float:
        """Get base difficulty for task type
        
        Args:
            task_type: Type of task
            
        Returns:
            Task difficulty score
        """
        return TASK_DIFFICULTY.get(task_type.lower(), 0.5)
    
    def calculate_composite_score(self, sample: Dict[str, Any]) -> DifficultyMetrics:
        """Calculate composite difficulty score for a sample
        
        Args:
            sample: CoTA sample dictionary
            
        Returns:
            DifficultyMetrics object with detailed scoring
        """
        trajectory = sample.get("trajectory", [])
        task_type = sample.get("task_type", "")
        sample_type = sample.get("sample_type", "positive")
        
        # Calculate individual components
        trajectory_complexity = self.calculate_trajectory_complexity(trajectory)
        
        operation_sophistication, op_count, unique_ops = self.calculate_operation_sophistication(trajectory)
        
        reasoning_depth, thinking_steps = self.calculate_reasoning_depth(trajectory)
        
        error_score, has_correction, has_backtracking, repetitions = self.detect_error_patterns(
            trajectory, sample_type
        )
        
        task_difficulty = self.get_task_difficulty(task_type)
        
        # Calculate weighted composite score
        composite_score = (
            self.weights["trajectory_complexity"] * trajectory_complexity +
            self.weights["operation_sophistication"] * operation_sophistication +
            self.weights["reasoning_depth"] * reasoning_depth +
            self.weights["error_patterns"] * error_score +
            self.weights["task_type"] * task_difficulty
        )
        
        # Apply bounds
        composite_score = np.clip(
            composite_score,
            self.categorization["min_difficulty"],
            self.categorization["max_difficulty"]
        )
        
        # Create metrics object
        metrics = DifficultyMetrics(
            trajectory_length=len(trajectory),
            operation_count=op_count,
            unique_operations=unique_ops,
            thinking_steps=thinking_steps,
            repetition_count=repetitions,
            has_self_correction=has_correction,
            has_backtracking=has_backtracking,
            operation_sophistication_score=operation_sophistication,
            reasoning_depth_ratio=reasoning_depth,
            error_pattern_score=error_score,
            task_type_score=task_difficulty,
            composite_score=composite_score,
            difficulty_category=""  # Will be set later
        )
        
        return metrics
    
    def categorize_difficulties(self, scores: List[float]) -> List[str]:
        """Categorize difficulty scores using percentile-based thresholds
        
        Args:
            scores: List of difficulty scores
            
        Returns:
            List of difficulty categories
        """
        if not scores:
            return []
        
        # Calculate percentiles using robust statistics
        scores_array = np.array(scores)
        
        # Remove outliers using IQR
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers for percentile calculation
        filtered_scores = scores_array[
            (scores_array >= lower_bound) & (scores_array <= upper_bound)
        ]
        
        if len(filtered_scores) == 0:
            filtered_scores = scores_array
        
        # Calculate thresholds
        simple_threshold = np.percentile(filtered_scores, self.categorization["simple_percentile"])
        medium_threshold = np.percentile(filtered_scores, self.categorization["medium_percentile"])
        
        # Categorize each score
        categories = []
        for score in scores:
            if score <= simple_threshold:
                category = "simple"
            elif score <= medium_threshold:
                category = "medium"
            else:
                category = "hard"
            categories.append(category)
            self.category_counts[category] += 1
        
        return categories
    
    def generate_statistics(self, processed_samples: List[ProcessedSample]) -> Dict[str, Any]:
        """Generate statistics about the difficulty distribution
        
        Args:
            processed_samples: List of processed samples
            
        Returns:
            Statistics dictionary
        """
        scores = [s.difficulty_score for s in processed_samples]
        
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        
        # Calculate statistics
        stats = {
            "total_samples": len(processed_samples),
            "difficulty_scores": {
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array)),
                "median": float(np.median(scores_array)),
                "q1": float(np.percentile(scores_array, 25)),
                "q3": float(np.percentile(scores_array, 75))
            },
            "category_distribution": dict(self.category_counts),
            "category_percentages": {
                cat: count / len(processed_samples) * 100
                for cat, count in self.category_counts.items()
            },
            "task_type_distribution": {},
            "sample_type_distribution": {},
            "trajectory_length_stats": {},
            "operation_usage": {}
        }
        
        # Task type distribution by difficulty
        task_by_difficulty = defaultdict(lambda: defaultdict(int))
        for sample in processed_samples:
            task_by_difficulty[sample.difficulty_category][sample.task_type] += 1
        stats["task_type_by_difficulty"] = dict(task_by_difficulty)
        
        # Sample type distribution
        sample_types = Counter(s.sample_type for s in processed_samples)
        stats["sample_type_distribution"] = dict(sample_types)
        
        # Trajectory length statistics
        lengths = [s.difficulty_metrics.trajectory_length for s in processed_samples]
        stats["trajectory_length_stats"] = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths))
        }
        
        # Operation usage statistics
        all_operations = []
        for sample in processed_samples:
            for step in sample.trajectory:
                if step.get("action") in OPERATION_COMPLEXITY:
                    all_operations.append(step["action"])
        
        operation_counts = Counter(all_operations)
        stats["operation_usage"] = dict(operation_counts)
        
        return stats

# ============================================================================
# Data Processing Pipeline
# ============================================================================

class CoTAPreprocessor:
    """Main preprocessing pipeline for CoTA data"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scorer = DifficultyScorer(self.config)
        self.processed_samples = []
        self.statistics = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config = DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Deep merge configurations
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
        
        return config
    
    def load_raw_data(self, input_path: str) -> List[Dict[str, Any]]:
        """Load raw CoTA data from JSON file
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            List of raw samples
        """
        logger.info(f"Loading raw data from {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Handle both direct list and structured format
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            raise ValueError("Unexpected data format in input file")
        
        logger.info(f"Loaded {len(samples)} raw samples")
        return samples
    
    def process_sample(self, sample: Dict[str, Any]) -> ProcessedSample:
        """Process a single CoTA sample
        
        Args:
            sample: Raw sample dictionary
            
        Returns:
            ProcessedSample object
        """
        # Calculate difficulty metrics
        metrics = self.scorer.calculate_composite_score(sample)
        
        # Create processed sample
        processed = ProcessedSample(
            sample_id=sample.get("sample_id", ""),
            task_type=sample.get("task_type", ""),
            sample_type=sample.get("sample_type", "positive"),
            question=sample.get("question", ""),
            image_path=sample.get("image_path", ""),
            trajectory=sample.get("trajectory", []),
            answer=sample.get("answer", ""),
            ground_truth=sample.get("ground_truth", ""),
            provenance=sample.get("provenance", {}),
            metadata=sample.get("metadata", {}),
            sampling_weight=sample.get("sampling_weight", 1.0),
            difficulty_score=metrics.composite_score,
            difficulty_category="",  # Will be set after categorization
            difficulty_metrics=metrics
        )
        
        # Track score for distribution
        self.scorer.score_distribution.append(metrics.composite_score)
        
        return processed
    
    def process_dataset(self, input_path: str) -> List[ProcessedSample]:
        """Process entire CoTA dataset
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            List of processed samples
        """
        # Load raw data
        raw_samples = self.load_raw_data(input_path)
        
        # Process each sample
        logger.info("Processing samples and calculating difficulty scores...")
        processed_samples = []
        
        for sample in tqdm(raw_samples, desc="Processing samples"):
            try:
                processed = self.process_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
                continue
        
        # Categorize difficulties based on distribution
        logger.info("Categorizing difficulties based on percentiles...")
        scores = [s.difficulty_score for s in processed_samples]
        categories = self.scorer.categorize_difficulties(scores)
        
        # Assign categories to samples
        for sample, category in zip(processed_samples, categories):
            sample.difficulty_category = category
            sample.difficulty_metrics.difficulty_category = category
        
        self.processed_samples = processed_samples
        
        # Generate statistics
        if self.config["generate_statistics"]:
            self.statistics = self.scorer.generate_statistics(processed_samples)
        
        # Validate distribution
        if self.config["validate_distribution"]:
            self._validate_distribution(processed_samples)
        
        return processed_samples
    
    def _validate_distribution(self, samples: List[ProcessedSample]):
        """Validate the difficulty distribution meets requirements
        
        Args:
            samples: List of processed samples
        """
        category_counts = Counter(s.difficulty_category for s in samples)
        total = len(samples)
        
        # Check minimum percentage per category (25%)
        min_percentage = 0.25
        warnings = []
        
        for category in ["simple", "medium", "hard"]:
            if category in category_counts:
                percentage = category_counts[category] / total
                if percentage < min_percentage:
                    warnings.append(
                        f"Category '{category}' has only {percentage:.1%} of samples "
                        f"(minimum recommended: {min_percentage:.1%})"
                    )
        
        # Check task type diversity per category
        for category in ["simple", "medium", "hard"]:
            category_samples = [s for s in samples if s.difficulty_category == category]
            task_types = set(s.task_type for s in category_samples)
            if len(task_types) < 3:
                warnings.append(
                    f"Category '{category}' has low task type diversity: {len(task_types)} types"
                )
        
        if warnings:
            logger.warning("Distribution validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Distribution validation passed!")
    
    def save_processed_data(self, output_path: str, split_by_category: bool = True):
        """Save processed data to JSON file(s)
        
        Args:
            output_path: Output file path or directory
            split_by_category: Whether to split into separate files by difficulty
        """
        output_path = Path(output_path)
        
        if split_by_category:
            # Create directory if needed
            if output_path.suffix == ".json":
                output_dir = output_path.parent / output_path.stem
            else:
                output_dir = output_path
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Split samples by category
            categories = defaultdict(list)
            for sample in self.processed_samples:
                categories[sample.difficulty_category].append(sample)
            
            # Save each category
            for category, samples in categories.items():
                category_file = output_dir / f"{category}.json"
                
                data = {
                    "metadata": {
                        "preprocessing_timestamp": datetime.now().isoformat(),
                        "difficulty_category": category,
                        "num_samples": len(samples),
                        "config": self.config
                    },
                    "samples": [s.to_dict() for s in samples]
                }
                
                with open(category_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Saved {len(samples)} {category} samples to {category_file}")
            
            # Save combined metadata
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    "preprocessing_timestamp": datetime.now().isoformat(),
                    "total_samples": len(self.processed_samples),
                    "category_files": {
                        category: f"{category}.json"
                        for category in categories.keys()
                    },
                    "statistics": self.statistics,
                    "config": self.config
                }, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_file}")
            
        else:
            # Save all samples to single file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "metadata": {
                    "preprocessing_timestamp": datetime.now().isoformat(),
                    "num_samples": len(self.processed_samples),
                    "statistics": self.statistics,
                    "config": self.config
                },
                "samples": [s.to_dict() for s in self.processed_samples]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.processed_samples)} samples to {output_path}")
    
    def save_statistics_report(self, output_path: str):
        """Save detailed statistics report
        
        Args:
            output_path: Path for statistics report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate readable report
        report = []
        report.append("=" * 80)
        report.append("CoTA Data Preprocessing Statistics Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total samples: {self.statistics['total_samples']}")
        report.append("")
        
        # Difficulty score statistics
        report.append("DIFFICULTY SCORES")
        report.append("-" * 40)
        for key, value in self.statistics["difficulty_scores"].items():
            report.append(f"  {key:10s}: {value:.4f}")
        report.append("")
        
        # Category distribution
        report.append("CATEGORY DISTRIBUTION")
        report.append("-" * 40)
        for category in ["simple", "medium", "hard"]:
            count = self.statistics["category_distribution"].get(category, 0)
            percentage = self.statistics["category_percentages"].get(category, 0)
            report.append(f"  {category:8s}: {count:5d} ({percentage:5.1f}%)")
        report.append("")
        
        # Task type distribution
        report.append("TASK TYPE DISTRIBUTION BY DIFFICULTY")
        report.append("-" * 40)
        for difficulty, tasks in self.statistics.get("task_type_by_difficulty", {}).items():
            report.append(f"  {difficulty}:")
            for task, count in tasks.items():
                report.append(f"    {task}: {count}")
        report.append("")
        
        # Sample type distribution
        report.append("SAMPLE TYPE DISTRIBUTION")
        report.append("-" * 40)
        for sample_type, count in self.statistics.get("sample_type_distribution", {}).items():
            report.append(f"  {sample_type}: {count}")
        report.append("")
        
        # Trajectory length statistics
        report.append("TRAJECTORY LENGTH STATISTICS")
        report.append("-" * 40)
        for key, value in self.statistics.get("trajectory_length_stats", {}).items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.2f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
        
        # Operation usage
        report.append("OPERATION USAGE")
        report.append("-" * 40)
        for op, count in sorted(
            self.statistics.get("operation_usage", {}).items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report.append(f"  {op:20s}: {count:5d}")
        report.append("")
        
        # Configuration used
        report.append("CONFIGURATION")
        report.append("-" * 40)
        report.append(json.dumps(self.config, indent=2))
        
        # Save report
        report_text = "\n".join(report)
        
        # Save as text
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        # Also save as JSON
        json_path = output_path.parent / f"{output_path.stem}_data.json"
        with open(json_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"Saved statistics report to {output_path}")
        logger.info(f"Saved statistics data to {json_path}")

# ============================================================================
# Curriculum Data Loader
# ============================================================================

class CurriculumDataset:
    """Dataset wrapper for curriculum learning"""
    
    def __init__(self, data_dir: str, initial_difficulty: str = "simple"):
        """Initialize curriculum dataset
        
        Args:
            data_dir: Directory containing difficulty-split data
            initial_difficulty: Starting difficulty level
        """
        self.data_dir = Path(data_dir)
        self.current_difficulty = initial_difficulty
        self.difficulties = ["simple", "medium", "hard"]
        self.data = {}
        
        # Load all difficulty levels
        for difficulty in self.difficulties:
            file_path = self.data_dir / f"{difficulty}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.data[difficulty] = data.get("samples", [])
                logger.info(f"Loaded {len(self.data[difficulty])} {difficulty} samples")
            else:
                logger.warning(f"Missing data file: {file_path}")
                self.data[difficulty] = []
    
    def get_curriculum_batch(self, stage: int, batch_size: int) -> List[Dict]:
        """Get a batch of samples for current curriculum stage
        
        Args:
            stage: Current curriculum stage (0-based)
            batch_size: Number of samples to return
            
        Returns:
            List of samples for training
        """
        # Determine mixture based on stage
        if stage == 0:
            # Initial stage: mostly simple
            mixture = {"simple": 0.8, "medium": 0.2, "hard": 0.0}
        elif stage == 1:
            # Early stage: balanced simple/medium
            mixture = {"simple": 0.5, "medium": 0.4, "hard": 0.1}
        elif stage == 2:
            # Middle stage: mostly medium
            mixture = {"simple": 0.3, "medium": 0.5, "hard": 0.2}
        elif stage == 3:
            # Advanced stage: balanced medium/hard
            mixture = {"simple": 0.2, "medium": 0.4, "hard": 0.4}
        else:
            # Final stage: mostly hard
            mixture = {"simple": 0.1, "medium": 0.3, "hard": 0.6}
        
        # Sample from each difficulty
        batch = []
        for difficulty, ratio in mixture.items():
            n_samples = int(batch_size * ratio)
            if n_samples > 0 and self.data[difficulty]:
                samples = np.random.choice(
                    self.data[difficulty],
                    size=min(n_samples, len(self.data[difficulty])),
                    replace=True
                ).tolist()
                batch.extend(samples)
        
        # Shuffle the batch
        np.random.shuffle(batch)
        
        return batch[:batch_size]

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CoTA data with difficulty scoring for curriculum learning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CoTA data JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/curriculum",
        help="Output path for processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--split-by-category",
        action="store_true",
        default=True,
        help="Split output into separate files by difficulty category"
    )
    parser.add_argument(
        "--statistics-report",
        type=str,
        default="data/processed/statistics_report.txt",
        help="Path for statistics report"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize preprocessor
    preprocessor = CoTAPreprocessor(args.config)
    
    # Process dataset
    processed_samples = preprocessor.process_dataset(args.input)
    
    # Save processed data
    preprocessor.save_processed_data(args.output, args.split_by_category)
    
    # Save statistics report
    preprocessor.save_statistics_report(args.statistics_report)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"Total samples processed: {len(processed_samples)}")
    print("\nDifficulty distribution:")
    for category, count in preprocessor.scorer.category_counts.items():
        percentage = count / len(processed_samples) * 100
        print(f"  {category:8s}: {count:5d} ({percentage:5.1f}%)")
    
    print(f"\nProcessed data saved to: {args.output}")
    print(f"Statistics report saved to: {args.statistics_report}")
    
    # Print warnings if any
    if len(processed_samples) < 100:
        print("\nWARNING: Low sample count. Consider generating more data for effective training.")
    
    min_category = min(preprocessor.scorer.category_counts.values())
    if min_category < len(processed_samples) * 0.25:
        print("\nWARNING: Imbalanced distribution detected. Some categories have few samples.")

if __name__ == "__main__":
    main()