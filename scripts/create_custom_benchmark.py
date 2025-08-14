#!/usr/bin/env python3
"""
Create Custom Capabilities Benchmark from synthesized CoTA data.
This benchmark tests capabilities that require new visual operations.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


class CustomBenchmarkCreator:
    """Create challenging evaluation benchmark for new visual operations."""
    
    # Define task categories and required operations
    TASK_CATEGORIES = {
        "precise_segmentation": {
            "required_ops": ["SEGMENT_OBJECT_AT"],
            "optional_ops": ["GET_PROPERTIES", "ZOOM_IN"],
            "sample_count": 500,
            "difficulty": "medium",
            "description": "Tasks requiring precise object segmentation"
        },
        "text_extraction": {
            "required_ops": ["READ_TEXT"],
            "optional_ops": ["ZOOM_IN", "SEGMENT_OBJECT_AT"],
            "sample_count": 500,
            "difficulty": "medium",
            "description": "Tasks requiring text reading from images"
        },
        "object_tracking": {
            "required_ops": ["TRACK_OBJECT"],
            "optional_ops": ["SEGMENT_OBJECT_AT", "SELECT_FRAME"],
            "sample_count": 300,
            "difficulty": "hard",
            "description": "Tasks requiring object tracking across frames"
        },
        "property_analysis": {
            "required_ops": ["GET_PROPERTIES"],
            "optional_ops": ["SEGMENT_OBJECT_AT", "ZOOM_IN"],
            "sample_count": 400,
            "difficulty": "medium",
            "description": "Tasks requiring detailed property extraction"
        },
        "complex_reasoning": {
            "required_ops": ["SEGMENT_OBJECT_AT", "READ_TEXT", "GET_PROPERTIES"],
            "optional_ops": ["ZOOM_IN", "TRACK_OBJECT"],
            "sample_count": 300,
            "difficulty": "hard",
            "description": "Complex tasks requiring multiple operations"
        }
    }
    
    # Define evaluation criteria for each operation
    EVALUATION_CRITERIA = {
        "SEGMENT_OBJECT_AT": {
            "metric": "iou",
            "threshold": 0.7,
            "additional_metrics": ["boundary_f1"]
        },
        "READ_TEXT": {
            "metric": "edit_distance",
            "threshold": 0.2,
            "additional_metrics": ["word_error_rate"]
        },
        "TRACK_OBJECT": {
            "metric": "mota",
            "threshold": 0.6,
            "additional_metrics": ["motp", "id_switches"]
        },
        "GET_PROPERTIES": {
            "metric": "accuracy",
            "threshold": 0.8,
            "additional_metrics": ["attribute_precision"]
        }
    }
    
    def __init__(self, seed: int = 42):
        """Initialize benchmark creator."""
        self.seed = seed
        random.seed(seed)
        self.samples = defaultdict(list)
        self.metadata = {
            "version": "1.0",
            "seed": seed,
            "task_categories": self.TASK_CATEGORIES,
            "evaluation_criteria": self.EVALUATION_CRITERIA
        }
    
    def load_cota_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load synthesized CoTA data."""
        logger.info(f"Loading CoTA data from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def categorize_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """Categorize a sample based on required operations."""
        trajectory = sample.get("trajectory", [])
        if not trajectory:
            return None
        
        # Extract operations used in trajectory
        operations = set()
        for step in trajectory:
            if "action" in step:
                operations.add(step["action"])
        
        # Find matching category
        for category, config in self.TASK_CATEGORIES.items():
            required = set(config["required_ops"])
            if required.issubset(operations):
                # Prioritize complex reasoning if multiple ops present
                if category == "complex_reasoning" and len(required) > 2:
                    return category
                elif category != "complex_reasoning":
                    return category
        
        return None
    
    def filter_high_quality_samples(
        self,
        samples: List[Dict[str, Any]],
        category: str
    ) -> List[Dict[str, Any]]:
        """Filter samples for quality and relevance."""
        filtered = []
        config = self.TASK_CATEGORIES[category]
        
        for sample in samples:
            # Check trajectory completeness
            trajectory = sample.get("trajectory", [])
            if len(trajectory) < 2:
                continue
            
            # Check for required operations
            operations = {step.get("action") for step in trajectory}
            required = set(config["required_ops"])
            
            if not required.issubset(operations):
                continue
            
            # Check difficulty level
            difficulty = sample.get("difficulty", "medium")
            if config["difficulty"] == "hard" and difficulty == "simple":
                continue
            
            # Check for ground truth annotations
            if category == "precise_segmentation":
                if "segmentation_mask" not in sample.get("ground_truth", {}):
                    continue
            elif category == "text_extraction":
                if "extracted_text" not in sample.get("ground_truth", {}):
                    continue
            elif category == "object_tracking":
                if "tracking_annotations" not in sample.get("ground_truth", {}):
                    continue
            
            # Check for trap samples (valuable for testing)
            is_trap = sample.get("is_trap", False)
            if is_trap:
                # Include trap samples but mark them
                sample["evaluation_type"] = "trap"
            else:
                sample["evaluation_type"] = "standard"
            
            filtered.append(sample)
        
        return filtered
    
    def balance_samples(
        self,
        samples: List[Dict[str, Any]],
        target_count: int
    ) -> List[Dict[str, Any]]:
        """Balance samples to reach target count."""
        if len(samples) <= target_count:
            return samples
        
        # Stratified sampling based on difficulty and type
        stratified = defaultdict(list)
        for sample in samples:
            difficulty = sample.get("difficulty", "medium")
            eval_type = sample.get("evaluation_type", "standard")
            key = f"{difficulty}_{eval_type}"
            stratified[key].append(sample)
        
        # Calculate samples per stratum
        strata_count = len(stratified)
        base_count = target_count // strata_count
        remainder = target_count % strata_count
        
        balanced = []
        for i, (key, stratum_samples) in enumerate(stratified.items()):
            count = base_count + (1 if i < remainder else 0)
            if len(stratum_samples) <= count:
                balanced.extend(stratum_samples)
            else:
                balanced.extend(random.sample(stratum_samples, count))
        
        return balanced[:target_count]
    
    def add_evaluation_metadata(
        self,
        sample: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """Add evaluation-specific metadata to sample."""
        sample["benchmark_category"] = category
        sample["evaluation_criteria"] = []
        
        # Add specific evaluation criteria
        config = self.TASK_CATEGORIES[category]
        for op in config["required_ops"]:
            if op in self.EVALUATION_CRITERIA:
                criteria = self.EVALUATION_CRITERIA[op].copy()
                criteria["operation"] = op
                sample["evaluation_criteria"].append(criteria)
        
        # Generate unique ID
        content = json.dumps(sample, sort_keys=True)
        sample["benchmark_id"] = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        # Add human-readable description
        sample["task_description"] = self._generate_task_description(sample, category)
        
        return sample
    
    def _generate_task_description(
        self,
        sample: Dict[str, Any],
        category: str
    ) -> str:
        """Generate human-readable task description."""
        descriptions = {
            "precise_segmentation": (
                "Segment the specified object in the image and provide "
                "precise boundaries. The segmentation mask should achieve "
                "an IoU > 0.7 with the ground truth."
            ),
            "text_extraction": (
                "Extract and read all text visible in the image. The "
                "extracted text should have an edit distance < 0.2 from "
                "the ground truth text."
            ),
            "object_tracking": (
                "Track the specified object across all frames in the video. "
                "The tracking should maintain MOTA > 0.6 and minimize ID switches."
            ),
            "property_analysis": (
                "Analyze the specified object and extract its properties "
                "including color, size, texture, and other attributes. "
                "Property extraction accuracy should exceed 80%."
            ),
            "complex_reasoning": (
                "Complete the complex visual reasoning task using multiple "
                "operations. This requires segmentation, text reading, and "
                "property analysis in the correct sequence."
            )
        }
        
        return descriptions.get(category, "Complete the visual task as specified.")
    
    def create_benchmark(
        self,
        cota_data: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Create the complete custom benchmark."""
        logger.info("Creating custom capabilities benchmark")
        
        # Categorize all samples
        categorized = defaultdict(list)
        uncategorized = []
        
        for sample in cota_data:
            category = self.categorize_sample(sample)
            if category:
                categorized[category].append(sample)
            else:
                uncategorized.append(sample)
        
        logger.info(f"Categorized {len(cota_data) - len(uncategorized)} samples")
        logger.info(f"Uncategorized: {len(uncategorized)} samples")
        
        # Process each category
        benchmark = {
            "metadata": self.metadata,
            "categories": {},
            "statistics": {}
        }
        
        total_samples = 0
        for category, config in self.TASK_CATEGORIES.items():
            logger.info(f"\nProcessing {category}...")
            
            # Get samples for this category
            category_samples = categorized.get(category, [])
            logger.info(f"  Found {len(category_samples)} samples")
            
            # Filter for quality
            filtered = self.filter_high_quality_samples(category_samples, category)
            logger.info(f"  After filtering: {len(filtered)} samples")
            
            # Balance to target count
            target = config["sample_count"]
            balanced = self.balance_samples(filtered, target)
            logger.info(f"  After balancing: {len(balanced)} samples (target: {target})")
            
            # Add evaluation metadata
            processed = []
            for sample in balanced:
                processed.append(self.add_evaluation_metadata(sample, category))
            
            # Store in benchmark
            benchmark["categories"][category] = {
                "config": config,
                "samples": processed,
                "count": len(processed)
            }
            
            total_samples += len(processed)
        
        # Add statistics
        benchmark["statistics"] = {
            "total_samples": total_samples,
            "categories": len(benchmark["categories"]),
            "category_distribution": {
                cat: data["count"] 
                for cat, data in benchmark["categories"].items()
            },
            "difficulty_distribution": self._calculate_difficulty_distribution(benchmark),
            "operation_frequency": self._calculate_operation_frequency(benchmark)
        }
        
        # Save benchmark
        self._save_benchmark(benchmark, output_dir)
        
        return benchmark
    
    def _calculate_difficulty_distribution(
        self,
        benchmark: Dict[str, Any]
    ) -> Dict[str, int]:
        """Calculate difficulty distribution across benchmark."""
        distribution = defaultdict(int)
        
        for category_data in benchmark["categories"].values():
            for sample in category_data["samples"]:
                difficulty = sample.get("difficulty", "medium")
                distribution[difficulty] += 1
        
        return dict(distribution)
    
    def _calculate_operation_frequency(
        self,
        benchmark: Dict[str, Any]
    ) -> Dict[str, int]:
        """Calculate operation frequency across benchmark."""
        frequency = defaultdict(int)
        
        for category_data in benchmark["categories"].values():
            for sample in category_data["samples"]:
                trajectory = sample.get("trajectory", [])
                for step in trajectory:
                    if "action" in step:
                        frequency[step["action"]] += 1
        
        return dict(frequency)
    
    def _save_benchmark(
        self,
        benchmark: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save benchmark to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete benchmark
        full_path = output_dir / "custom_capabilities_benchmark.json"
        with open(full_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        logger.info(f"Saved complete benchmark to {full_path}")
        
        # Save category-specific files for easier loading
        for category, data in benchmark["categories"].items():
            category_path = output_dir / f"benchmark_{category}.json"
            category_data = {
                "metadata": benchmark["metadata"],
                "category": category,
                "config": data["config"],
                "samples": data["samples"],
                "count": data["count"]
            }
            with open(category_path, 'w') as f:
                json.dump(category_data, f, indent=2)
            logger.info(f"Saved {category} benchmark to {category_path}")
        
        # Save statistics
        stats_path = output_dir / "benchmark_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(benchmark["statistics"], f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
        
        # Create a lightweight index file
        index = {
            "version": benchmark["metadata"]["version"],
            "total_samples": benchmark["statistics"]["total_samples"],
            "categories": list(benchmark["categories"].keys()),
            "files": {
                "full": "custom_capabilities_benchmark.json",
                "statistics": "benchmark_statistics.json",
                "categories": {
                    cat: f"benchmark_{cat}.json"
                    for cat in benchmark["categories"].keys()
                }
            }
        }
        
        index_path = output_dir / "benchmark_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved index to {index_path}")
    
    def create_splits(
        self,
        benchmark: Dict[str, Any],
        val_ratio: float = 0.1,
        test_ratio: float = 0.15
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Create train/val/test splits from benchmark."""
        train_samples = []
        val_samples = []
        test_samples = []
        
        for category, data in benchmark["categories"].items():
            samples = data["samples"]
            random.shuffle(samples)
            
            n_samples = len(samples)
            n_test = int(n_samples * test_ratio)
            n_val = int(n_samples * val_ratio)
            n_train = n_samples - n_test - n_val
            
            test_samples.extend(samples[:n_test])
            val_samples.extend(samples[n_test:n_test + n_val])
            train_samples.extend(samples[n_test + n_val:])
        
        # Create split dictionaries
        train_split = {
            "split": "train",
            "samples": train_samples,
            "count": len(train_samples)
        }
        
        val_split = {
            "split": "validation",
            "samples": val_samples,
            "count": len(val_samples)
        }
        
        test_split = {
            "split": "test",
            "samples": test_samples,
            "count": len(test_samples)
        }
        
        logger.info(f"Created splits - Train: {len(train_samples)}, "
                   f"Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return train_split, val_split, test_split


def main():
    """Main entry point for benchmark creation."""
    parser = argparse.ArgumentParser(
        description="Create custom capabilities benchmark from CoTA data"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to synthesized CoTA data file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmarks/custom",
        help="Output directory for benchmark files"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val/test splits"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio"
    )
    
    args = parser.parse_args()
    
    # Create benchmark creator
    creator = CustomBenchmarkCreator(seed=args.seed)
    
    # Load CoTA data
    cota_data = creator.load_cota_data(Path(args.input))
    
    # Create benchmark
    benchmark = creator.create_benchmark(
        cota_data=cota_data,
        output_dir=Path(args.output)
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Custom Capabilities Benchmark Created")
    logger.info("=" * 60)
    logger.info(f"Total samples: {benchmark['statistics']['total_samples']}")
    logger.info(f"Categories: {benchmark['statistics']['categories']}")
    logger.info("\nCategory distribution:")
    for category, count in benchmark["statistics"]["category_distribution"].items():
        logger.info(f"  {category}: {count} samples")
    logger.info("\nDifficulty distribution:")
    for difficulty, count in benchmark["statistics"]["difficulty_distribution"].items():
        logger.info(f"  {difficulty}: {count} samples")
    
    # Create splits if requested
    if args.create_splits:
        logger.info("\nCreating train/val/test splits...")
        train, val, test = creator.create_splits(
            benchmark,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Save splits
        output_dir = Path(args.output)
        for split_data, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            split_path = output_dir / f"benchmark_{split_name}.json"
            with open(split_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"Saved {split_name} split to {split_path}")
    
    logger.info("\n✓ Benchmark creation complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())