#!/usr/bin/env python3
"""
Stage 2: Fuse and Validate Final Datasets

This script combines specialized datasets from Stage 1, applies trajectory augmentation,
validates data quality, and creates the final SFT and RFT training datasets.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import yaml
import hashlib
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_generation.trajectory_augmenter import TrajectoryAugmenter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetFuser:
    """
    Handles the fusion, augmentation, and validation of CoTA datasets.
    
    This class:
    1. Loads specialized datasets from Stage 1
    2. Applies trajectory augmentation (golden, trap, self-correction)
    3. Validates data quality and balance
    4. Creates final SFT and RFT datasets
    """
    
    def __init__(self, fusion_manifest: Dict[str, Any], verbose: bool = False):
        """
        Initialize the dataset fuser.
        
        Args:
            fusion_manifest: Configuration for data fusion
            verbose: Enable verbose logging
        """
        self.config = fusion_manifest
        self.verbose = verbose
        
        # Initialize statistics tracking
        self.stats = defaultdict(lambda: defaultdict(int))
        self.validation_errors = []
        
        # Initialize trajectory augmenter if configured
        augmenter_config = self.config.get('trajectory_augmentation', {})
        if augmenter_config.get('enabled', True):
            self.augmenter = TrajectoryAugmenter(augmenter_config)
        else:
            self.augmenter = None
            
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def load_specialized_datasets(self, input_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load all specialized datasets from Stage 1 output.
        
        Args:
            input_dir: Directory containing specialized dataset files
            
        Returns:
            Dictionary mapping task names to lists of samples
        """
        datasets = {}
        
        # Find all dataset files
        dataset_files = list(input_dir.glob("*.jsonl"))
        if not dataset_files:
            logger.warning(f"No dataset files found in {input_dir}")
            return datasets
        
        logger.info(f"Found {len(dataset_files)} dataset files to load")
        
        for file_path in tqdm(dataset_files, desc="Loading datasets"):
            task_name = file_path.stem
            samples = []
            
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            sample = json.loads(line.strip())
                            
                            # Validate sample structure
                            if self._validate_sample_structure(sample, task_name, line_num):
                                samples.append(sample)
                                self.stats[task_name]['loaded'] += 1
                            else:
                                self.stats[task_name]['invalid'] += 1
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in {file_path}:{line_num}: {e}")
                            self.stats[task_name]['json_errors'] += 1
                
                datasets[task_name] = samples
                logger.info(f"Loaded {len(samples)} samples from {task_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                self.stats[task_name]['load_failed'] = 1
        
        return datasets
    
    def _validate_sample_structure(self, sample: Dict, task_name: str, line_num: int) -> bool:
        """
        Validate the structure of a loaded sample.
        
        Args:
            sample: The sample to validate
            task_name: Name of the task (for error reporting)
            line_num: Line number (for error reporting)
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['trajectory', 'final_answer']
        for field in required_fields:
            if field not in sample:
                error = f"{task_name}:{line_num} - Missing required field: {field}"
                self.validation_errors.append(error)
                if self.verbose:
                    logger.debug(error)
                return False
        
        # Validate trajectory structure
        if not isinstance(sample['trajectory'], list):
            error = f"{task_name}:{line_num} - Trajectory is not a list"
            self.validation_errors.append(error)
            return False
        
        if len(sample['trajectory']) == 0:
            error = f"{task_name}:{line_num} - Empty trajectory"
            self.validation_errors.append(error)
            return False
        
        # Validate each step in trajectory
        for i, step in enumerate(sample['trajectory']):
            if not isinstance(step, dict):
                error = f"{task_name}:{line_num} - Step {i} is not a dict"
                self.validation_errors.append(error)
                return False
            
            if 'action' not in step:
                error = f"{task_name}:{line_num} - Step {i} missing 'action'"
                self.validation_errors.append(error)
                return False
        
        return True
    
    def apply_trajectory_augmentation(
        self, 
        datasets: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Apply trajectory augmentation to create diverse training samples.
        
        Args:
            datasets: Original datasets
            
        Returns:
            Augmented datasets with golden, trap, and self-correction samples
        """
        if not self.augmenter:
            logger.info("Trajectory augmentation disabled, skipping")
            return datasets
        
        augmented_datasets = {}
        
        for task_name, samples in datasets.items():
            logger.info(f"Augmenting {task_name} dataset...")
            
            # Get task-specific augmentation config
            task_config = self._get_task_augmentation_config(task_name)
            
            augmented_samples = []
            
            # Process samples in batches for efficiency
            batch_size = 100
            for i in tqdm(range(0, len(samples), batch_size), 
                         desc=f"Augmenting {task_name}"):
                batch = samples[i:i+batch_size]
                
                # Generate different trajectory types based on config
                if task_config.get('golden_ratio', 0) > 0:
                    golden_samples = self._generate_golden_samples(
                        batch, task_config['golden_ratio']
                    )
                    augmented_samples.extend(golden_samples)
                    self.stats[task_name]['golden'] += len(golden_samples)
                
                if task_config.get('trap_ratio', 0) > 0:
                    trap_samples = self.augmenter.batch_augment(
                        batch, 
                        augmentation_type='trap',
                        ratio=task_config['trap_ratio']
                    )
                    augmented_samples.extend(trap_samples)
                    self.stats[task_name]['trap'] += len(trap_samples)
                
                if task_config.get('self_correction_ratio', 0) > 0:
                    correction_samples = self.augmenter.batch_augment(
                        batch,
                        augmentation_type='self_correction',
                        ratio=task_config['self_correction_ratio']
                    )
                    augmented_samples.extend(correction_samples)
                    self.stats[task_name]['self_correction'] += len(correction_samples)
                
                # Keep original samples
                augmented_samples.extend(batch)
                self.stats[task_name]['original'] += len(batch)
            
            augmented_datasets[task_name] = augmented_samples
            logger.info(f"Augmented {task_name}: {len(samples)} → {len(augmented_samples)} samples")
        
        return augmented_datasets
    
    def _get_task_augmentation_config(self, task_name: str) -> Dict[str, float]:
        """
        Get augmentation configuration for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Augmentation configuration with ratios
        """
        # Default configuration
        default_config = {
            'golden_ratio': 0.6,
            'trap_ratio': 0.2,
            'self_correction_ratio': 0.2
        }
        
        # Check for task-specific overrides
        trajectory_config = self.config.get('trajectory_proportions', {})
        if task_name in trajectory_config:
            return trajectory_config[task_name]
        
        # Check for pattern-based configuration
        if 'geometric' in task_name.lower():
            return trajectory_config.get('geometric_tasks', default_config)
        elif 'ocr' in task_name.lower() or 'text' in task_name.lower():
            return trajectory_config.get('ocr_tasks', default_config)
        elif 'temporal' in task_name.lower() or 'video' in task_name.lower():
            return trajectory_config.get('temporal_tasks', default_config)
        
        return default_config
    
    def _generate_golden_samples(self, samples: List[Dict], ratio: float) -> List[Dict]:
        """
        Generate golden (perfectly correct) trajectory samples.
        
        Args:
            samples: Original samples
            ratio: Ratio of golden samples to generate
            
        Returns:
            Golden trajectory samples
        """
        num_golden = int(len(samples) * ratio)
        golden_samples = []
        
        for _ in range(num_golden):
            # Select a base sample
            base_sample = random.choice(samples)
            
            # Create a golden variant (clean, optimal trajectory)
            golden = base_sample.copy()
            golden['trajectory_type'] = 'golden'
            golden['quality_score'] = 1.0
            
            # Clean up trajectory to be optimal
            golden['trajectory'] = self._optimize_trajectory(golden['trajectory'])
            
            golden_samples.append(golden)
        
        return golden_samples
    
    def _optimize_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Optimize a trajectory to be more efficient/correct.
        
        Args:
            trajectory: Original trajectory
            
        Returns:
            Optimized trajectory
        """
        # Remove redundant steps
        optimized = []
        seen_actions = set()
        
        for step in trajectory:
            action_key = (step.get('action'), str(step.get('parameters', {})))
            if action_key not in seen_actions:
                optimized.append(step)
                seen_actions.add(action_key)
        
        return optimized
    
    def validate_dataset_quality(self, datasets: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Perform comprehensive validation on the fused datasets.
        
        Args:
            datasets: Fused datasets to validate
            
        Returns:
            Validation report with statistics and issues
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': sum(len(samples) for samples in datasets.values()),
            'task_distribution': {},
            'quality_metrics': {},
            'validation_rules': [],
            'issues': []
        }
        
        # Task distribution analysis
        for task_name, samples in datasets.items():
            report['task_distribution'][task_name] = {
                'count': len(samples),
                'percentage': len(samples) / report['total_samples'] * 100
            }
        
        # Apply validation rules from config
        validation_rules = self.config.get('validation_rules', {})
        
        # 1. Minimum samples per task
        min_samples = validation_rules.get('min_samples_per_task', 1000)
        for task_name, samples in datasets.items():
            if len(samples) < min_samples:
                issue = f"{task_name} has only {len(samples)} samples (minimum: {min_samples})"
                report['issues'].append(issue)
                logger.warning(issue)
        
        # 2. Maximum imbalance ratio
        max_imbalance = validation_rules.get('max_imbalance_ratio', 10.0)
        sample_counts = [len(samples) for samples in datasets.values()]
        if sample_counts:
            imbalance_ratio = max(sample_counts) / min(sample_counts)
            if imbalance_ratio > max_imbalance:
                issue = f"Dataset imbalance ratio {imbalance_ratio:.2f} exceeds maximum {max_imbalance}"
                report['issues'].append(issue)
                logger.warning(issue)
        
        # 3. Trajectory quality checks
        quality_thresholds = validation_rules.get('quality_thresholds', {})
        for task_name, samples in datasets.items():
            quality_scores = []
            
            for sample in samples:
                score = self._calculate_sample_quality(sample, quality_thresholds)
                quality_scores.append(score)
            
            if quality_scores:
                report['quality_metrics'][task_name] = {
                    'mean_quality': np.mean(quality_scores),
                    'std_quality': np.std(quality_scores),
                    'min_quality': np.min(quality_scores),
                    'max_quality': np.max(quality_scores)
                }
        
        # 4. Diversity checks
        for task_name, samples in datasets.items():
            diversity_score = self._calculate_diversity(samples)
            report['quality_metrics'][task_name]['diversity'] = diversity_score
            
            min_diversity = validation_rules.get('min_diversity_score', 0.5)
            if diversity_score < min_diversity:
                issue = f"{task_name} has low diversity score: {diversity_score:.3f}"
                report['issues'].append(issue)
        
        # 5. Check for duplicates
        for task_name, samples in datasets.items():
            duplicates = self._find_duplicates(samples)
            if duplicates > 0:
                issue = f"{task_name} contains {duplicates} duplicate samples"
                report['issues'].append(issue)
                report['quality_metrics'][task_name]['duplicates'] = duplicates
        
        # Summary
        report['validation_passed'] = len(report['issues']) == 0
        report['num_issues'] = len(report['issues'])
        
        return report
    
    def _calculate_sample_quality(self, sample: Dict, thresholds: Dict) -> float:
        """
        Calculate quality score for a single sample.
        
        Args:
            sample: The sample to evaluate
            thresholds: Quality thresholds from config
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Check trajectory length
        trajectory_len = len(sample.get('trajectory', []))
        min_steps = thresholds.get('min_trajectory_length', 1)
        max_steps = thresholds.get('max_trajectory_length', 50)
        
        if trajectory_len < min_steps:
            score *= 0.5
        elif trajectory_len > max_steps:
            score *= 0.8
        
        # Check for required fields
        if 'provenance' not in sample:
            score *= 0.9
        
        # Check for error samples
        if 'error' in sample:
            score *= 0.1
        
        # Check trajectory type distribution
        traj_type = sample.get('trajectory_type', 'original')
        if traj_type == 'golden':
            score *= 1.0
        elif traj_type == 'self_correction':
            score *= 0.95
        elif traj_type == 'trap':
            score *= 0.9
        
        return score
    
    def _calculate_diversity(self, samples: List[Dict]) -> float:
        """
        Calculate diversity score for a set of samples.
        
        Args:
            samples: List of samples
            
        Returns:
            Diversity score between 0 and 1
        """
        if not samples:
            return 0.0
        
        # Calculate diversity based on action sequences
        action_sequences = []
        for sample in samples:
            trajectory = sample.get('trajectory', [])
            actions = tuple(step.get('action', 'unknown') for step in trajectory)
            action_sequences.append(actions)
        
        # Count unique sequences
        unique_sequences = len(set(action_sequences))
        diversity = unique_sequences / len(action_sequences)
        
        return diversity
    
    def _find_duplicates(self, samples: List[Dict]) -> int:
        """
        Find duplicate samples based on content hash.
        
        Args:
            samples: List of samples
            
        Returns:
            Number of duplicate samples
        """
        hashes = set()
        duplicates = 0
        
        for sample in samples:
            # Create hash of essential content
            content = json.dumps({
                'trajectory': sample.get('trajectory', []),
                'final_answer': sample.get('final_answer', '')
            }, sort_keys=True)
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if content_hash in hashes:
                duplicates += 1
            else:
                hashes.add(content_hash)
        
        return duplicates
    
    def create_final_datasets(
        self, 
        datasets: Dict[str, List[Dict]], 
        output_dir: Path
    ) -> Tuple[Path, Path]:
        """
        Create final SFT and RFT datasets from validated data.
        
        Args:
            datasets: Validated and augmented datasets
            output_dir: Directory to save final datasets
            
        Returns:
            Paths to SFT and RFT dataset files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate samples for SFT and RFT
        sft_samples = []
        rft_samples = []
        
        # Distribution configuration
        dataset_split = self.config.get('dataset_split', {})
        sft_ratio = dataset_split.get('sft_ratio', 0.7)
        
        for task_name, samples in datasets.items():
            # Shuffle samples for random split
            shuffled = samples.copy()
            random.shuffle(shuffled)
            
            # Split based on ratio
            split_point = int(len(shuffled) * sft_ratio)
            
            # Add task metadata to each sample
            for sample in shuffled[:split_point]:
                sample['dataset'] = 'sft'
                sample['task'] = task_name
                sft_samples.append(sample)
            
            for sample in shuffled[split_point:]:
                sample['dataset'] = 'rft'
                sample['task'] = task_name
                rft_samples.append(sample)
        
        # Shuffle final datasets
        random.shuffle(sft_samples)
        random.shuffle(rft_samples)
        
        # Save SFT dataset
        sft_path = output_dir / 'pixelis_sft_dataset.jsonl'
        with open(sft_path, 'w') as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Created SFT dataset with {len(sft_samples)} samples: {sft_path}")
        
        # Save RFT dataset
        rft_path = output_dir / 'pixelis_rft_dataset.jsonl'
        with open(rft_path, 'w') as f:
            for sample in rft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Created RFT dataset with {len(rft_samples)} samples: {rft_path}")
        
        # Save dataset metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'sft_samples': len(sft_samples),
            'rft_samples': len(rft_samples),
            'total_samples': len(sft_samples) + len(rft_samples),
            'task_distribution': {
                task: {
                    'sft': sum(1 for s in sft_samples if s.get('task') == task),
                    'rft': sum(1 for s in rft_samples if s.get('task') == task)
                }
                for task in datasets.keys()
            },
            'trajectory_type_distribution': Counter(
                s.get('trajectory_type', 'original') 
                for s in sft_samples + rft_samples
            )
        }
        
        metadata_path = output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved dataset metadata: {metadata_path}")
        
        return sft_path, rft_path
    
    def generate_report(self, output_dir: Path, validation_report: Dict):
        """
        Generate a comprehensive fusion report.
        
        Args:
            output_dir: Directory to save report
            validation_report: Validation results
        """
        report = {
            'fusion_summary': {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'statistics': dict(self.stats),
                'validation': validation_report
            }
        }
        
        # Add detailed statistics
        report['fusion_summary']['detailed_stats'] = {
            'total_samples_processed': sum(
                stats.get('loaded', 0) for stats in self.stats.values()
            ),
            'total_augmented_samples': sum(
                stats.get('golden', 0) + stats.get('trap', 0) + stats.get('self_correction', 0)
                for stats in self.stats.values()
            ),
            'validation_errors_encountered': len(self.validation_errors),
            'tasks_processed': list(self.stats.keys())
        }
        
        # Save report
        report_path = output_dir / 'fusion_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated fusion report: {report_path}")
        
        # Also save validation errors if any
        if self.validation_errors:
            errors_path = output_dir / 'validation_errors.txt'
            with open(errors_path, 'w') as f:
                for error in self.validation_errors:
                    f.write(error + '\n')
            logger.warning(f"Validation errors saved to: {errors_path}")


def main():
    """Main entry point for the fusion and validation script."""
    parser = argparse.ArgumentParser(
        description="Fuse and validate CoTA datasets from Stage 1"
    )
    parser.add_argument(
        '--fusion-manifest',
        type=str,
        required=True,
        help='Path to the fusion manifest YAML file'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing specialized datasets from Stage 1'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save final fused datasets'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without saving outputs'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--skip-augmentation',
        action='store_true',
        help='Skip trajectory augmentation step'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation checks (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Load fusion manifest
    try:
        with open(args.fusion_manifest, 'r') as f:
            fusion_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load fusion manifest: {e}")
        sys.exit(1)
    
    # Override config if flags are set
    if args.skip_augmentation:
        fusion_config['trajectory_augmentation'] = {'enabled': False}
    
    # Initialize fuser
    fuser = DatasetFuser(fusion_config, verbose=args.verbose)
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Stage 2: Dataset Fusion and Validation")
    logger.info("=" * 80)
    
    # Step 1: Load specialized datasets
    logger.info("\n📚 Loading specialized datasets...")
    datasets = fuser.load_specialized_datasets(input_dir)
    
    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(datasets)} task datasets")
    
    # Step 2: Apply trajectory augmentation
    if not args.skip_augmentation:
        logger.info("\n🔄 Applying trajectory augmentation...")
        datasets = fuser.apply_trajectory_augmentation(datasets)
    
    # Step 3: Validate dataset quality
    if not args.skip_validation:
        logger.info("\n✅ Validating dataset quality...")
        validation_report = fuser.validate_dataset_quality(datasets)
        
        if not validation_report['validation_passed']:
            logger.warning(f"Validation found {validation_report['num_issues']} issues")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
            
            if not args.dry_run:
                response = input("\nContinue despite validation issues? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Exiting due to validation issues")
                    sys.exit(1)
    else:
        validation_report = {'skipped': True}
    
    # Step 4: Create final datasets
    if not args.dry_run:
        logger.info("\n💾 Creating final SFT and RFT datasets...")
        sft_path, rft_path = fuser.create_final_datasets(datasets, output_dir)
        
        # Step 5: Generate report
        logger.info("\n📊 Generating fusion report...")
        fuser.generate_report(output_dir, validation_report)
        
        logger.info("\n" + "=" * 80)
        logger.info("✨ Dataset fusion completed successfully!")
        logger.info(f"  SFT Dataset: {sft_path}")
        logger.info(f"  RFT Dataset: {rft_path}")
        logger.info(f"  Reports: {output_dir}")
        logger.info("=" * 80)
    else:
        logger.info("\n🔍 Dry run completed. No outputs saved.")
        logger.info(f"Would have created datasets in: {output_dir}")


if __name__ == "__main__":
    main()