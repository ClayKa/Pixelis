#!/usr/bin/env python3
"""
Data filtering and scoring script with artifact management.
Creates versioned dataset artifacts for training with comprehensive quality control.
"""

import argparse
import json
import sys
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.reproducibility import (
    ArtifactManager,
    ArtifactType,
    ExperimentContext,
    EnvironmentCaptureLevel,
    track_artifacts,
)
from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# ============================================================================
# Data Quality Assessment
# ============================================================================

@dataclass
class QualityMetrics:
    """Quality metrics for a data sample"""
    heuristic_score: float
    model_score: Optional[float]
    consistency_score: Optional[float]
    final_score: float
    passed_heuristics: bool
    errors: List[str]

class HeuristicFilter:
    """Rule-based filters for data quality"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize heuristic filter with configuration"""
        self.min_trajectory_length = config.get("min_trajectory_length", 2)
        self.max_trajectory_length = config.get("max_trajectory_length", 20)
        self.min_question_length = config.get("min_question_length", 10)
        self.max_question_length = config.get("max_question_length", 500)
        self.valid_actions = config.get("valid_actions", [
            "THINK", "SEGMENT_OBJECT_AT", "GET_PROPERTIES",
            "READ_TEXT", "TRACK_OBJECT", "ZOOM_IN"
        ])
        self.required_fields = config.get("required_fields", [
            "sample_id", "task_type", "sample_type", "question",
            "trajectory", "answer", "ground_truth", "provenance"
        ])
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Apply heuristic filters to a sample
        
        Returns:
            Tuple of (passed_all_checks, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in sample or sample[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate JSON structure
        try:
            json.dumps(sample)
        except (TypeError, ValueError) as e:
            errors.append(f"Invalid JSON structure: {e}")
        
        # Check trajectory length
        if "trajectory" in sample:
            trajectory = sample["trajectory"]
            if not isinstance(trajectory, list):
                errors.append("Trajectory must be a list")
            elif len(trajectory) < self.min_trajectory_length:
                errors.append(f"Trajectory too short: {len(trajectory)} < {self.min_trajectory_length}")
            elif len(trajectory) > self.max_trajectory_length:
                errors.append(f"Trajectory too long: {len(trajectory)} > {self.max_trajectory_length}")
            
            # Validate action syntax
            for i, action in enumerate(trajectory):
                if not isinstance(action, dict):
                    errors.append(f"Trajectory step {i} is not a dictionary")
                elif "action" not in action:
                    errors.append(f"Trajectory step {i} missing 'action' field")
                elif action["action"] not in self.valid_actions:
                    errors.append(f"Invalid action at step {i}: {action.get('action')}")
                
                # Check for parameters field
                if "parameters" not in action:
                    errors.append(f"Trajectory step {i} missing 'parameters' field")
        
        # Check question length
        if "question" in sample:
            q_length = len(sample["question"])
            if q_length < self.min_question_length:
                errors.append(f"Question too short: {q_length} characters")
            elif q_length > self.max_question_length:
                errors.append(f"Question too long: {q_length} characters")
        
        # Check answer format
        if "answer" in sample and not sample["answer"]:
            errors.append("Answer is empty")
        
        # Check provenance
        if "provenance" in sample:
            prov = sample["provenance"]
            required_prov = ["source_dataset", "original_sample_id", "synthesis_timestamp"]
            for field in required_prov:
                if field not in prov:
                    errors.append(f"Missing provenance field: {field}")
        
        return len(errors) == 0, errors

class ModelBasedScorer:
    """Model-based quality scoring with consistency checks"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model-based scorer"""
        self.config = config
        self.judge_model = config.get("judge_model", "gpt-4")
        self.consistency_threshold = config.get("consistency_threshold", 1.0)
        self.consistency_sample_rate = config.get("consistency_sample_rate", 0.01)
        self.num_consistency_runs = config.get("num_consistency_runs", 3)
        self.scoring_temperature = config.get("scoring_temperature", 0.7)
        
    async def score_sample_async(self, sample: Dict[str, Any], 
                                session: aiohttp.ClientSession) -> float:
        """
        Score a sample using the judge model (async)
        
        Returns:
            Score between 1-5
        """
        # This is a placeholder for actual API calls
        # In production, this would call GPT-4 or similar
        
        # For now, return a simulated score based on sample properties
        score = 3.0
        
        # Bonus for complete trajectories
        if "trajectory" in sample and len(sample["trajectory"]) >= 3:
            score += 0.5
        
        # Bonus for self-correction
        if sample.get("sample_type") == "self_correction":
            score += 0.5
        
        # Penalty for traps (they're harder to judge)
        if "trap" in sample.get("sample_type", ""):
            score -= 0.3
        
        # Add some randomness for simulation
        score += random.uniform(-0.5, 0.5)
        
        return max(1.0, min(5.0, score))
    
    async def check_consistency(self, sample: Dict[str, Any],
                               session: aiohttp.ClientSession) -> Tuple[float, float]:
        """
        Check scoring consistency by running multiple times
        
        Returns:
            Tuple of (mean_score, std_deviation)
        """
        scores = []
        for _ in range(self.num_consistency_runs):
            score = await self.score_sample_async(sample, session)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        return mean_score, std_dev
    
    async def score_dataset_async(self, dataset: List[Dict[str, Any]]) -> List[QualityMetrics]:
        """
        Score entire dataset with consistency checks
        """
        metrics_list = []
        
        async with aiohttp.ClientSession() as session:
            # Determine which samples need consistency checking
            consistency_indices = set(
                random.sample(
                    range(len(dataset)),
                    min(int(len(dataset) * self.consistency_sample_rate), len(dataset))
                )
            )
            
            for i, sample in enumerate(tqdm(dataset, desc="Scoring samples")):
                if i in consistency_indices:
                    # Run consistency check
                    mean_score, std_dev = await self.check_consistency(sample, session)
                    
                    if std_dev > self.consistency_threshold:
                        logger.warning(
                            f"Sample {sample.get('sample_id', i)} has high scoring variance: "
                            f"mean={mean_score:.2f}, std={std_dev:.2f}"
                        )
                    
                    model_score = mean_score
                    consistency_score = 1.0 - (std_dev / 5.0)  # Normalize std to 0-1
                else:
                    # Single scoring run
                    model_score = await self.score_sample_async(sample, session)
                    consistency_score = None
                
                # Store in sample
                sample["_model_score"] = model_score
                sample["_consistency_score"] = consistency_score
                
        return dataset

class DataQualityScorer:
    """Comprehensive data quality scoring and filtering"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize scorer with configuration."""
        self.config = config or {}
        self.heuristic_filter = HeuristicFilter(self.config.get("heuristic_config", {}))
        self.model_scorer = ModelBasedScorer(self.config.get("model_config", {}))
        self.quality_threshold = self.config.get("quality_threshold", 4.0)
        self.min_samples_per_category = self.config.get("min_samples_per_category", 100)
    
    async def process_dataset(self, dataset: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process entire dataset through filtering and scoring pipeline
        
        Returns:
            Tuple of (filtered_dataset, statistics)
        """
        stats = {
            "total_samples": len(dataset),
            "heuristic_filtered": 0,
            "quality_filtered": 0,
            "final_samples": 0,
            "errors": [],
            "distribution": {},
            "warnings": []
        }
        
        # Step 1: Apply heuristic filters
        logger.info("Step 1: Applying heuristic filters...")
        heuristic_passed = []
        for sample in tqdm(dataset, desc="Heuristic filtering"):
            passed, errors = self.heuristic_filter.validate_sample(sample)
            if passed:
                heuristic_passed.append(sample)
            else:
                stats["errors"].extend(errors[:5])  # Keep first 5 errors per sample
        
        stats["heuristic_filtered"] = len(heuristic_passed)
        logger.info(f"Heuristic filtering: {len(heuristic_passed)}/{len(dataset)} passed")
        
        if not heuristic_passed:
            logger.error("No samples passed heuristic filtering!")
            return [], stats
        
        # Step 2: Model-based scoring with consistency checks
        logger.info("Step 2: Model-based quality scoring...")
        scored_dataset = await self.model_scorer.score_dataset_async(heuristic_passed)
        
        # Step 3: Apply quality threshold
        logger.info("Step 3: Applying quality threshold...")
        quality_passed = []
        for sample in scored_dataset:
            model_score = sample.get("_model_score", 0)
            
            # Calculate final score
            if model_score >= self.quality_threshold:
                # Normalize model score to 0-1
                sample["_quality_score"] = model_score / 5.0
                quality_passed.append(sample)
        
        stats["quality_filtered"] = len(quality_passed)
        logger.info(f"Quality filtering: {len(quality_passed)}/{len(heuristic_passed)} passed")
        
        # Step 4: Distribution analysis
        logger.info("Step 4: Analyzing distribution...")
        distribution = self.analyze_distribution(quality_passed)
        stats["distribution"] = distribution
        
        # Step 5: Check minimum sample counts
        logger.info("Step 5: Checking minimum sample requirements...")
        warnings = self.check_minimum_samples(distribution)
        stats["warnings"] = warnings
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        stats["final_samples"] = len(quality_passed)
        
        return quality_passed, stats
    
    def analyze_distribution(self, dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the distribution of samples across different categories
        """
        distribution = {
            "task_types": defaultdict(int),
            "sample_types": defaultdict(int),
            "action_usage": defaultdict(int),
            "trajectory_lengths": [],
            "source_datasets": defaultdict(int)
        }
        
        for sample in dataset:
            # Task type distribution
            task_type = sample.get("task_type", "unknown")
            distribution["task_types"][task_type] += 1
            
            # Sample type distribution
            sample_type = sample.get("sample_type", "unknown")
            distribution["sample_types"][sample_type] += 1
            
            # Action usage distribution
            if "trajectory" in sample:
                trajectory = sample["trajectory"]
                distribution["trajectory_lengths"].append(len(trajectory))
                
                for action in trajectory:
                    action_type = action.get("action", "unknown")
                    distribution["action_usage"][action_type] += 1
            
            # Source dataset distribution
            if "provenance" in sample:
                source = sample["provenance"].get("source_dataset", "unknown")
                distribution["source_datasets"][source] += 1
        
        # Calculate percentages
        total = len(dataset)
        if total > 0:
            for category in ["task_types", "sample_types", "source_datasets"]:
                distribution[f"{category}_percentages"] = {
                    k: (v / total) * 100
                    for k, v in distribution[category].items()
                }
            
            # Action usage percentage (over all actions)
            total_actions = sum(distribution["action_usage"].values())
            if total_actions > 0:
                distribution["action_usage_percentages"] = {
                    k: (v / total_actions) * 100
                    for k, v in distribution["action_usage"].items()
                }
            
            # Trajectory length statistics
            if distribution["trajectory_lengths"]:
                lengths = distribution["trajectory_lengths"]
                distribution["trajectory_stats"] = {
                    "mean": np.mean(lengths),
                    "std": np.std(lengths),
                    "min": min(lengths),
                    "max": max(lengths),
                    "median": np.median(lengths)
                }
        
        return dict(distribution)
    
    def check_minimum_samples(self, distribution: Dict[str, Any]) -> List[str]:
        """
        Check if minimum sample requirements are met for each category
        """
        warnings = []
        
        # Check task types
        for task_type, count in distribution.get("task_types", {}).items():
            if count < self.min_samples_per_category:
                warnings.append(
                    f"Task type '{task_type}' has only {count} samples "
                    f"(minimum: {self.min_samples_per_category})"
                )
        
        # Check sample types
        for sample_type, count in distribution.get("sample_types", {}).items():
            if sample_type in ["trap_perceptual", "trap_logical", "self_correction"]:
                # These are critical for robust training
                min_required = self.min_samples_per_category // 2
                if count < min_required:
                    warnings.append(
                        f"Critical sample type '{sample_type}' has only {count} samples "
                        f"(minimum: {min_required})"
                    )
        
        # Check for missing action types
        expected_actions = ["SEGMENT_OBJECT_AT", "GET_PROPERTIES", "READ_TEXT"]
        for action in expected_actions:
            if action not in distribution.get("action_usage", {}):
                warnings.append(f"No samples use the '{action}' operation")
        
        return warnings


@track_artifacts(outputs=["dataset"])
async def process_and_filter_data(
    input_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    artifact_manager: ArtifactManager,
) -> Path:
    """
    Process and filter data with comprehensive quality control.
    
    Args:
        input_path: Path to input data
        output_path: Path to save filtered data
        config: Processing configuration
        artifact_manager: Artifact manager instance
    
    Returns:
        Path to filtered dataset
    """
    logger.info(f"Processing data from {input_path}")
    
    # Load raw data
    with open(input_path, "r") as f:
        if input_path.suffix == ".json":
            data = json.load(f)
            # Handle both flat list and nested structure
            if isinstance(data, dict) and "samples" in data:
                raw_data = data["samples"]
                metadata = data.get("metadata", {})
            else:
                raw_data = data if isinstance(data, list) else [data]
                metadata = {}
        elif input_path.suffix == ".jsonl":
            raw_data = [json.loads(line) for line in f]
            metadata = {}
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    logger.info(f"Loaded {len(raw_data)} raw samples")
    
    # Initialize scorer with comprehensive config
    quality_config = {
        "heuristic_config": {
            "min_trajectory_length": config.get("min_trajectory_length", 2),
            "max_trajectory_length": config.get("max_trajectory_length", 20),
            "min_question_length": config.get("min_question_length", 10),
            "max_question_length": config.get("max_question_length", 500),
        },
        "model_config": {
            "judge_model": config.get("judge_model", "gpt-4"),
            "consistency_threshold": config.get("consistency_threshold", 1.0),
            "consistency_sample_rate": config.get("consistency_sample_rate", 0.01),
            "num_consistency_runs": config.get("num_consistency_runs", 3),
        },
        "quality_threshold": config.get("quality_threshold", 4.0),
        "min_samples_per_category": config.get("min_samples_per_category", 100),
    }
    
    scorer = DataQualityScorer(quality_config)
    
    # Process dataset through comprehensive pipeline
    filtered_data, stats = await scorer.process_dataset(raw_data)
    
    logger.info(f"Filtered to {len(filtered_data)} samples")
    logger.info(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
    
    # Apply hard-negative mining strategy (Task 9)
    if config.get("enable_hard_negative_mining", True):
        logger.info("Applying hard-negative mining strategy...")
        for sample in filtered_data:
            sample_type = sample.get("sample_type", "")
            
            # Default weight
            weight = 1.0
            
            # Increase weight for trap samples
            if "trap" in sample_type:
                weight = config.get("trap_sample_weight", 1.5)
            # Slightly increase for self-correction
            elif sample_type == "self_correction":
                weight = config.get("self_correction_weight", 1.2)
            
            # Combine with quality score
            quality_score = sample.get("_quality_score", 1.0)
            sample["sampling_weight"] = weight * (quality_score ** 0.5)
    
    # Apply stratification if requested
    if config.get("stratify_by_difficulty", False):
        logger.info("Stratifying by difficulty...")
        for sample in filtered_data:
            score = sample.get("_quality_score", 0.5)
            trajectory_len = len(sample.get("trajectory", []))
            
            # Combined difficulty assessment
            if score < 0.6 or trajectory_len > 15:
                sample["difficulty"] = "hard"
            elif score < 0.8 or trajectory_len > 8:
                sample["difficulty"] = "medium"
            else:
                sample["difficulty"] = "easy"
    
    # Prepare output data
    output_data = {
        "metadata": {
            "original_metadata": metadata,
            "filtering_timestamp": datetime.now().isoformat(),
            "filtering_config": config,
            "statistics": stats,
        },
        "samples": filtered_data
    }
    
    # Save filtered data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        if output_path.suffix == ".json":
            json.dump(output_data, f, indent=2)
        elif output_path.suffix == ".jsonl":
            for sample in filtered_data:
                f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Saved filtered data to {output_path}")
    
    # Generate detailed report
    report_path = output_path.with_suffix(".report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("FILTERING STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"After heuristic filtering: {stats['heuristic_filtered']}\n")
        f.write(f"After quality filtering: {stats['quality_filtered']}\n")
        f.write(f"Final samples: {stats['final_samples']}\n")
        f.write(f"Retention rate: {(stats['final_samples']/stats['total_samples']*100):.1f}%\n\n")
        
        if stats.get("distribution"):
            f.write("DISTRIBUTION ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Task types
            f.write("\nTask Type Distribution:\n")
            for task_type, percentage in stats["distribution"].get("task_types_percentages", {}).items():
                count = stats["distribution"]["task_types"][task_type]
                f.write(f"  {task_type}: {count} samples ({percentage:.1f}%)\n")
            
            # Sample types
            f.write("\nSample Type Distribution:\n")
            for sample_type, percentage in stats["distribution"].get("sample_types_percentages", {}).items():
                count = stats["distribution"]["sample_types"][sample_type]
                f.write(f"  {sample_type}: {count} samples ({percentage:.1f}%)\n")
            
            # Trajectory statistics
            if "trajectory_stats" in stats["distribution"]:
                f.write("\nTrajectory Length Statistics:\n")
                t_stats = stats["distribution"]["trajectory_stats"]
                f.write(f"  Mean: {t_stats['mean']:.1f}\n")
                f.write(f"  Std: {t_stats['std']:.1f}\n")
                f.write(f"  Min: {t_stats['min']}\n")
                f.write(f"  Max: {t_stats['max']}\n")
                f.write(f"  Median: {t_stats['median']:.1f}\n")
        
        if stats.get("warnings"):
            f.write("\nWARNINGS:\n")
            f.write("-" * 40 + "\n")
            for warning in stats["warnings"]:
                f.write(f"⚠️  {warning}\n")
        
        if stats.get("errors"):
            f.write("\nSAMPLE ERRORS (first 20):\n")
            f.write("-" * 40 + "\n")
            for error in stats["errors"][:20]:
                f.write(f"  - {error}\n")
    
    logger.info(f"Generated quality report: {report_path}")
    
    # Log as artifact
    dataset_name = config.get("dataset_name", "filtered_dataset")
    dataset_artifact = artifact_manager.log_artifact(
        name=f"dataset-{dataset_name}",
        type=ArtifactType.DATASET,
        file_path=output_path,
        metadata={
            "source_file": str(input_path),
            "processing_config": config,
            "statistics": stats,
            "num_samples": len(filtered_data),
            "report_path": str(report_path),
        },
    )
    
    logger.info(f"✓ Created dataset artifact: {dataset_artifact.name}:{dataset_artifact.version}")
    
    return output_path


def main():
    """Main entry point for data filtering and scoring."""
    parser = argparse.ArgumentParser(
        description="Comprehensive data filtering and scoring for CoTA datasets"
    )
    
    # Input/output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input dataset (JSON or JSONL)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save filtered dataset",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cota_filtered",
        help="Name for dataset artifact",
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["sft", "rft", "eval"],
        default="sft",
        help="Type of dataset being created",
    )
    
    # Heuristic filtering
    parser.add_argument(
        "--min-trajectory-length",
        type=int,
        default=2,
        help="Minimum trajectory length",
    )
    
    parser.add_argument(
        "--max-trajectory-length",
        type=int,
        default=20,
        help="Maximum trajectory length",
    )
    
    parser.add_argument(
        "--min-question-length",
        type=int,
        default=10,
        help="Minimum question length",
    )
    
    parser.add_argument(
        "--max-question-length",
        type=int,
        default=500,
        help="Maximum question length",
    )
    
    # Quality scoring
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=4.0,
        help="Minimum quality score threshold (1-5 scale)",
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        help="Model to use for quality scoring",
    )
    
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=1.0,
        help="Maximum allowed std deviation for consistency",
    )
    
    parser.add_argument(
        "--consistency-sample-rate",
        type=float,
        default=0.01,
        help="Fraction of samples to check for consistency",
    )
    
    # Distribution requirements
    parser.add_argument(
        "--min-samples-per-category",
        type=int,
        default=100,
        help="Minimum samples required per category",
    )
    
    # Hard-negative mining
    parser.add_argument(
        "--enable-hard-negative-mining",
        action="store_true",
        default=True,
        help="Enable hard-negative mining with weighted sampling",
    )
    
    parser.add_argument(
        "--trap-sample-weight",
        type=float,
        default=1.5,
        help="Sampling weight for trap samples",
    )
    
    parser.add_argument(
        "--self-correction-weight",
        type=float,
        default=1.2,
        help="Sampling weight for self-correction samples",
    )
    
    # Processing options
    parser.add_argument(
        "--stratify-by-difficulty",
        action="store_true",
        help="Add difficulty labels for curriculum learning",
    )
    
    # Experiment tracking
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name",
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare configuration
    config = {
        "dataset_name": f"{args.dataset_name}_{args.dataset_type}",
        "dataset_type": args.dataset_type,
        # Heuristic config
        "min_trajectory_length": args.min_trajectory_length,
        "max_trajectory_length": args.max_trajectory_length,
        "min_question_length": args.min_question_length,
        "max_question_length": args.max_question_length,
        # Quality config
        "quality_threshold": args.quality_threshold,
        "judge_model": args.judge_model,
        "consistency_threshold": args.consistency_threshold,
        "consistency_sample_rate": args.consistency_sample_rate,
        # Distribution config
        "min_samples_per_category": args.min_samples_per_category,
        # Hard-negative mining
        "enable_hard_negative_mining": args.enable_hard_negative_mining,
        "trap_sample_weight": args.trap_sample_weight,
        "self_correction_weight": args.self_correction_weight,
        # Other options
        "stratify_by_difficulty": args.stratify_by_difficulty,
        "seed": args.seed,
    }
    
    # Create experiment context
    exp_name = args.exp_name or f"cota_filtering_{args.dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run async function in sync context
    async def run_filtering():
        with ExperimentContext(
            config=config,
            name=exp_name,
            capture_level=EnvironmentCaptureLevel.STANDARD,
            offline_mode=args.offline,
        ) as ctx:
            
            # Process data
            input_path = Path(args.input)
            output_path = Path(args.output)
            
            filtered_path = await process_and_filter_data(
                input_path=input_path,
                output_path=output_path,
                config=config,
                artifact_manager=ctx.artifact_manager,
            )
            
            # Log summary
            ctx.log_artifact(
                name="filtering_summary",
                type=ArtifactType.METRICS,
                data={
                    "input_file": str(input_path),
                    "output_file": str(filtered_path),
                    "config": config,
                },
            )
            
            logger.info(f"✓ Data filtering complete: {filtered_path}")
            
            return filtered_path
    
    # Run the async function
    try:
        filtered_path = asyncio.run(run_filtering())
        print(f"\n{'='*60}")
        print("DATA FILTERING COMPLETE!")
        print(f"{'='*60}")
        print(f"Filtered dataset saved to: {filtered_path}")
        print(f"Quality report saved to: {filtered_path.with_suffix('.report.txt')}")
        return 0
    except Exception as e:
        logger.error(f"Error during data filtering: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())