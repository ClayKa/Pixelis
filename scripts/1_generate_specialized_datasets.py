#!/usr/bin/env python3
"""
Specialized Dataset Generation Script
======================================
This script generates specialized datasets for training Pixelis, including
self-correction trajectories and various visual reasoning tasks.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import yaml
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation.trajectory_augmenter import (
    TrajectoryAugmenter, 
    Trajectory,
    load_trajectories_from_file,
    save_trajectories_to_file
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpecializedDatasetGenerator:
    """Main class for generating specialized training datasets."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the generator with a configuration file.
        
        Args:
            config_path: Path to the data_generation_manifest.yaml
        """
        self.config = self._load_config(config_path)
        self.augmenter = TrajectoryAugmenter()
        self.output_dir = Path(self.config['global_config']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def generate_task_trajectories(self, task_config: Dict[str, Any]) -> List[Trajectory]:
        """
        Generate trajectories for a specific task.
        
        Args:
            task_config: Configuration for the task
            
        Returns:
            List of generated trajectories
        """
        trajectories = []
        task_generator_class = task_config['task_generator_class']
        target_count = task_config['target_sample_count']
        
        logger.info(f"Generating {target_count} samples for {task_generator_class}")
        
        # Placeholder for actual task generation logic
        # In production, this would instantiate the appropriate generator class
        # and use the source datasets to create trajectories
        
        for i in range(target_count):
            # Create mock trajectory for demonstration
            trajectory = Trajectory(
                task_id=f"{task_generator_class}_{i:05d}",
                question=f"Sample question for {task_generator_class} task {i}",
                actions=[
                    {
                        "action": "SEGMENT_OBJECT_AT",
                        "parameters": {"x": 256, "y": 256},
                        "observation": "Found object: car at coordinates (256, 256)"
                    },
                    {
                        "action": "GET_PROPERTIES",
                        "parameters": {"object_id": 1},
                        "observation": "Object properties: color=red, size=large"
                    }
                ],
                final_answer="The red car is in the center of the image",
                trajectory_type="golden",
                metadata={"source_dataset": task_config.get('source_datasets', [])}
            )
            trajectories.append(trajectory)
            
        return trajectories
        
    def apply_trajectory_augmentation(
        self, 
        trajectories: List[Trajectory]
    ) -> List[Trajectory]:
        """
        Apply trajectory augmentation including self-correction and traps.
        
        Args:
            trajectories: Original golden trajectories
            
        Returns:
            Augmented trajectory list
        """
        aug_config = self.config['trajectory_augmentation']['proportions']
        
        # Calculate counts for each type
        total_count = len(trajectories)
        golden_count = int(total_count * aug_config['golden_positive'])
        trap_count = int(total_count * aug_config['trap_samples'])
        correction_count = int(total_count * aug_config['self_correction'])
        
        logger.info(f"Augmentation plan: {golden_count} golden, {trap_count} traps, {correction_count} corrections")
        
        result_trajectories = []
        
        # Keep golden trajectories
        result_trajectories.extend(trajectories[:golden_count])
        
        # Generate trap samples (process-negative)
        for i in range(trap_count):
            if i < len(trajectories):
                trap = self._create_trap_trajectory(trajectories[i])
                result_trajectories.append(trap)
                
        # Generate self-correction samples
        augmented, stats = self.augmenter.batch_augment(
            trajectories[golden_count:golden_count + correction_count * 2],
            augmentation_ratio=0.5
        )
        
        # Add the self-correction trajectories
        correction_trajectories = [t for t in augmented if t.trajectory_type == "self_correction"]
        result_trajectories.extend(correction_trajectories[:correction_count])
        
        logger.info(f"Final dataset size: {len(result_trajectories)}")
        return result_trajectories
        
    def _create_trap_trajectory(self, golden: Trajectory) -> Trajectory:
        """
        Create a trap (process-negative) trajectory from a golden one.
        
        Args:
            golden: Original golden trajectory
            
        Returns:
            Trap trajectory with subtle errors
        """
        # Create a trap by introducing a logical error in the reasoning
        trap_actions = golden.actions.copy()
        
        # Add a misleading observation or incorrect reasoning step
        if len(trap_actions) > 1:
            trap_actions[1] = {
                "action": trap_actions[1].get("action", "ANALYZE"),
                "parameters": trap_actions[1].get("parameters", {}),
                "observation": "Incorrect observation: The object appears to be something else",
                "error_injected": True
            }
            
        return Trajectory(
            task_id=f"{golden.task_id}_trap",
            question=golden.question,
            actions=trap_actions,
            final_answer="Incorrect conclusion based on flawed reasoning",
            trajectory_type="trap",
            metadata={
                **golden.metadata,
                "original_trajectory_id": golden.task_id,
                "trap_type": "process_negative"
            }
        )
        
    def generate_all_datasets(self):
        """Generate all datasets according to configuration."""
        all_trajectories = []
        
        # Generate trajectories for each enabled task
        for task_name, task_config in self.config['tasks'].items():
            if task_config.get('enabled', True):
                logger.info(f"Processing task: {task_name}")
                task_trajectories = self.generate_task_trajectories(task_config)
                all_trajectories.extend(task_trajectories)
                
        # Apply trajectory augmentation
        logger.info("Applying trajectory augmentation...")
        augmented_trajectories = self.apply_trajectory_augmentation(all_trajectories)
        
        # Save to output files
        self._save_datasets(augmented_trajectories)
        
        # Generate summary statistics
        self._generate_summary(augmented_trajectories)
        
    def _save_datasets(self, trajectories: List[Trajectory]):
        """Save trajectories to appropriate output files."""
        # Save by trajectory type
        by_type = {}
        for trajectory in trajectories:
            traj_type = trajectory.trajectory_type
            if traj_type not in by_type:
                by_type[traj_type] = []
            by_type[traj_type].append(trajectory)
            
        for traj_type, trajs in by_type.items():
            output_file = self.output_dir / f"{traj_type}_trajectories.jsonl"
            save_trajectories_to_file(trajs, output_file)
            logger.info(f"Saved {len(trajs)} {traj_type} trajectories to {output_file}")
            
        # Save combined dataset
        combined_file = self.output_dir / "combined_trajectories.jsonl"
        save_trajectories_to_file(trajectories, combined_file)
        logger.info(f"Saved {len(trajectories)} total trajectories to {combined_file}")
        
    def _generate_summary(self, trajectories: List[Trajectory]):
        """Generate and save summary statistics."""
        summary = {
            "total_trajectories": len(trajectories),
            "by_type": {},
            "by_task": {},
            "metadata": {
                "config_file": str(self.config_path) if hasattr(self, 'config_path') else "unknown",
                "output_dir": str(self.output_dir)
            }
        }
        
        # Count by type
        for trajectory in trajectories:
            traj_type = trajectory.trajectory_type
            summary["by_type"][traj_type] = summary["by_type"].get(traj_type, 0) + 1
            
            # Extract task from task_id
            task_prefix = trajectory.task_id.split('_')[0] if '_' in trajectory.task_id else "unknown"
            summary["by_task"][task_prefix] = summary["by_task"].get(task_prefix, 0) + 1
            
        # Save summary
        summary_file = self.output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"Dataset generation complete: {summary}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate specialized datasets for Pixelis training"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_generation_manifest.yaml"),
        help="Path to the data generation configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually generating data"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create the configuration file first.")
        sys.exit(1)
        
    # Initialize generator
    generator = SpecializedDatasetGenerator(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        generator.output_dir = args.output_dir
        generator.output_dir.mkdir(parents=True, exist_ok=True)
        
    # Run generation
    if args.dry_run:
        logger.info("Dry run mode - not generating actual data")
        logger.info(f"Would generate data using config: {args.config}")
        logger.info(f"Would save to: {generator.output_dir}")
    else:
        generator.generate_all_datasets()
        

if __name__ == "__main__":
    main()