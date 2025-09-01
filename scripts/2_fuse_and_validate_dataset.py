#!/usr/bin/env python3
"""
Stage 2: Fuse and Validate Dataset Script
==========================================
This script fuses pre-augmented specialized datasets from Stage 1
and produces the final training datasets.

This is Stage 2 of the two-stage pipeline:
- Stage 1: Generate specialized datasets with integrated augmentation
- Stage 2 (this script): Fuse pre-augmented datasets and validate
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
log = logging.getLogger(__name__)


class DatasetFusionEngine:
    """
    Fusion Engine for Stage 2 of the CoTA data pipeline.
    Responsible for loading, fusing, and validating pre-augmented datasets.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the fusion engine with Hydra configuration.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.stage1_output_dir = Path(cfg.stage1_output_dir)
        self.final_output_dir = Path(cfg.final_output_dir)
        self.dry_run = cfg.script_args.get('dry_run', False)
        
        # Create output directory
        if not self.dry_run:
            self.final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'loaded': {},
            'fused': {},
            'composition': defaultdict(lambda: defaultdict(int))
        }
    
    def load_specialized_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all specialized datasets from Stage 1.
        
        Returns:
            Dictionary mapping task names to their samples
        """
        all_task_samples = {}
        total_loaded = 0
        
        # Get the fusion recipe from config
        fusion_recipe = self.cfg.sft_dataset_recipe
        
        for source in fusion_recipe.sources:
            task_name = source['name']
            source_file = self.stage1_output_dir / f"{task_name}.jsonl"
            
            if not source_file.is_file():
                log.error(f"FATAL: Input file for task '{task_name}' not found at {source_file}")
                raise FileNotFoundError(f"Required source file missing: {source_file}")
            
            samples = []
            with open(source_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        log.warning(f"Skipping invalid JSON at {source_file}:{line_num}: {e}")
            
            all_task_samples[task_name] = samples
            self.stats['loaded'][task_name] = len(samples)
            total_loaded += len(samples)
            log.info(f"  ✓ Loaded {len(samples)} samples from '{task_name}'")
        
        log.info(f"Total samples loaded: {total_loaded}")
        return all_task_samples
    
    def fuse_datasets(self, all_task_samples: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Fuse datasets according to the recipe proportions.
        
        Args:
            all_task_samples: Dictionary of loaded datasets by task
            
        Returns:
            Fused dataset with proper proportions
        """
        fusion_recipe = self.cfg.sft_dataset_recipe
        target_total = fusion_recipe.target_total_samples
        fused_samples = []
        
        for source in fusion_recipe.sources:
            task_name = source['name']
            proportion = source['proportion']
            num_to_sample = int(target_total * proportion)
            
            available_samples = all_task_samples.get(task_name, [])
            
            if len(available_samples) < num_to_sample:
                log.warning(
                    f"  ⚠️ Not enough samples for '{task_name}'. "
                    f"Required: {num_to_sample}, Available: {len(available_samples)}. Using all."
                )
                num_to_sample = len(available_samples)
            
            # Sample the required number
            sampled = random.sample(available_samples, num_to_sample)
            fused_samples.extend(sampled)
            self.stats['fused'][task_name] = num_to_sample
            
            log.info(f"  → Sampled {num_to_sample} samples from '{task_name}' (proportion: {proportion:.1%})")
        
        log.info(f"Total samples in fusion pool: {len(fused_samples)}")
        return fused_samples
    
    def split_train_val(
        self, 
        dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into training and validation sets.
        
        Args:
            dataset: Full dataset to split
            
        Returns:
            Tuple of (train_set, val_set)
        """
        train_ratio = self.cfg.final_split.train_ratio
        split_index = int(len(dataset) * train_ratio)
        
        # Shuffle before splitting
        random.shuffle(dataset)
        
        train_set = dataset[:split_index]
        val_set = dataset[split_index:]
        
        log.info(f"  • Train set size: {len(train_set)} ({train_ratio:.0%})")
        log.info(f"  • Validation set size: {len(val_set)} ({1-train_ratio:.0%})")
        
        return train_set, val_set
    
    def generate_final_report(self, train_set: List[Dict[str, Any]], val_set: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive report of the final dataset.
        
        Args:
            train_set: Training dataset
            val_set: Validation dataset
            
        Returns:
            Markdown-formatted report string
        """
        # Analyze composition
        train_stats = self._analyze_composition(train_set)
        val_stats = self._analyze_composition(val_set)
        
        report_lines = [
            "# Final Dataset Report",
            f"\nGenerated at: {datetime.now().isoformat()}",
            "\n## Dataset Overview",
            f"- **Total samples**: {len(train_set) + len(val_set)}",
            f"- **Training samples**: {len(train_set)}",
            f"- **Validation samples**: {len(val_set)}",
            f"- **Split ratio**: {self.cfg.final_split.train_ratio:.0%} train / {1-self.cfg.final_split.train_ratio:.0%} val",
            "\n## Training Set Composition"
        ]
        
        # Add trajectory type distribution for train set
        if train_stats['trajectory_types']:
            report_lines.append("\n### Trajectory Types")
            for traj_type, count in sorted(train_stats['trajectory_types'].items()):
                ratio = count / len(train_set) * 100
                report_lines.append(f"- {traj_type}: {count} ({ratio:.1f}%)")
        
        # Add task distribution for train set
        if train_stats['tasks']:
            report_lines.append("\n### Task Distribution")
            for task, count in sorted(train_stats['tasks'].items()):
                ratio = count / len(train_set) * 100
                report_lines.append(f"- {task}: {count} ({ratio:.1f}%)")
        
        report_lines.append("\n## Validation Set Composition")
        
        # Add trajectory type distribution for val set
        if val_stats['trajectory_types']:
            report_lines.append("\n### Trajectory Types")
            for traj_type, count in sorted(val_stats['trajectory_types'].items()):
                ratio = count / len(val_set) * 100
                report_lines.append(f"- {traj_type}: {count} ({ratio:.1f}%)")
        
        # Add task distribution for val set
        if val_stats['tasks']:
            report_lines.append("\n### Task Distribution")
            for task, count in sorted(val_stats['tasks'].items()):
                ratio = count / len(val_set) * 100
                report_lines.append(f"- {task}: {count} ({ratio:.1f}%)")
        
        # Add fusion statistics
        report_lines.append("\n## Fusion Statistics")
        report_lines.append("\n### Loaded from Stage 1")
        for task, count in sorted(self.stats['loaded'].items()):
            report_lines.append(f"- {task}: {count} samples")
        
        report_lines.append("\n### Used in Final Dataset")
        for task, count in sorted(self.stats['fused'].items()):
            report_lines.append(f"- {task}: {count} samples")
        
        # Add quality metrics
        report_lines.append("\n## Quality Metrics")
        report_lines.append(f"- Average trajectory length (train): {train_stats['avg_traj_length']:.1f}")
        report_lines.append(f"- Average trajectory length (val): {val_stats['avg_traj_length']:.1f}")
        report_lines.append(f"- Samples with visual operations (train): {train_stats['has_visual_ops']:.1%}")
        report_lines.append(f"- Samples with visual operations (val): {val_stats['has_visual_ops']:.1%}")
        
        return '\n'.join(report_lines)
    
    def _analyze_composition(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the composition of a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'trajectory_types': Counter(),
            'tasks': Counter(),
            'avg_traj_length': 0,
            'has_visual_ops': 0,
            'total_traj_length': 0
        }
        
        visual_ops = {'ZOOM_IN', 'SEGMENT_OBJECT_AT', 'GET_PROPERTIES', 'READ_TEXT', 
                      'TRACK_OBJECT', 'SELECT_FRAME'}
        
        for sample in dataset:
            # Count trajectory types
            traj_type = sample.get('trajectory_type', 'unknown')
            stats['trajectory_types'][traj_type] += 1
            
            # Count tasks
            if 'provenance' in sample:
                task_name = sample['provenance'].get('task_name', 'unknown')
                stats['tasks'][task_name] += 1
            
            # Analyze trajectory
            if 'trajectory' in sample:
                traj = sample['trajectory']
                if isinstance(traj, list):
                    stats['total_traj_length'] += len(traj)
                    # Check for visual operations
                    has_vis_op = any(
                        any(op in str(step) for op in visual_ops) 
                        for step in traj
                    )
                    if has_vis_op:
                        stats['has_visual_ops'] += 1
        
        # Calculate averages
        if dataset:
            stats['avg_traj_length'] = stats['total_traj_length'] / len(dataset)
            stats['has_visual_ops'] = stats['has_visual_ops'] / len(dataset)
        
        return stats
    
    def run(self):
        """Execute the complete fusion pipeline."""
        log.info("=" * 60)
        log.info("STAGE 2: DATASET FUSION AND VALIDATION")
        log.info("=" * 60)
        
        if self.dry_run:
            log.info("[DRY RUN MODE] Simulating pipeline execution. No files will be written.")
        
        # Step 1: Load specialized datasets
        log.info("\n[Step 1/5] Loading specialized datasets from Stage 1...")
        all_task_samples = self.load_specialized_datasets()
        
        # Step 2: Fuse datasets
        log.info("\n[Step 2/5] Fusing datasets according to recipe...")
        fused_samples = self.fuse_datasets(all_task_samples)
        
        # Step 3: Shuffle
        log.info("\n[Step 3/5] Performing final global shuffle...")
        random.shuffle(fused_samples)
        
        # Step 4: Split train/val
        log.info("\n[Step 4/5] Splitting into train/validation sets...")
        train_set, val_set = self.split_train_val(fused_samples)
        
        # Step 5: Generate report and save
        log.info("\n[Step 5/5] Generating report and saving final datasets...")
        report = self.generate_final_report(train_set, val_set)
        
        if not self.dry_run:
            # Save train set
            train_path = self.final_output_dir / "sft_train.jsonl"
            with open(train_path, 'w') as f:
                for sample in train_set:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            log.info(f"  ✓ Saved train set to {train_path}")
            
            # Save val set
            val_path = self.final_output_dir / "sft_val.jsonl"
            with open(val_path, 'w') as f:
                for sample in val_set:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            log.info(f"  ✓ Saved validation set to {val_path}")
            
            # Save report
            report_path = self.final_output_dir / "final_dataset_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            log.info(f"  ✓ Saved final report to {report_path}")
        else:
            log.info(f"[DRY RUN] Would save train set ({len(train_set)} samples)")
            log.info(f"[DRY RUN] Would save val set ({len(val_set)} samples)")
            log.info("\n--- [DRY RUN] REPORT PREVIEW ---")
            print(report)
        
        log.info("\n" + "=" * 60)
        log.info("STAGE 2 COMPLETE!")
        log.info("=" * 60)
        if not self.dry_run:
            log.info(f"Output directory: {self.final_output_dir}")
            log.info("Ready for training with SFT!")


@hydra.main(version_base=None, config_path="../configs", config_name="data_fusion_manifest")
def main(cfg: DictConfig) -> None:
    """
    Stage 2: Fuses pre-augmented, specialized datasets into the final
    training and validation sets.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    random.seed(seed)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run the fusion engine
    engine = DatasetFusionEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()