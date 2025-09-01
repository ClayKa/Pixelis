#!/usr/bin/env python3
"""
Stage 1: Specialized Dataset Generation Script
==============================================
This script generates task-specific CoTA datasets by orchestrating
specialized task generators. Each generator produces golden trajectories
for its specific visual reasoning capability.

This is Stage 1 of the two-stage pipeline:
- Stage 1 (this script): Generate specialized datasets with integrated augmentation
- Stage 2: Fuse and validate final datasets from pre-augmented sources
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os
import importlib
from dataclasses import asdict
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the TrajectoryAugmenter for integrated augmentation
from core.data_generation.trajectory_augmenter import (
    TrajectoryAugmenter, 
    Trajectory
)

# A logger for clean output
log = logging.getLogger(__name__)


class SpecializedDatasetGenerator:
    """Orchestrates specialized task generators to produce task-specific datasets."""
    
    def __init__(self, config: DictConfig, dry_run: bool = False):
        """
        Initialize the generator with a configuration.
        
        Args:
            config: Hydra configuration object
            dry_run: If True, skip actual generation
        """
        self.config = config
        self.dry_run = dry_run
        self.output_dir = Path(config.global_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create specialized output directory for task-specific datasets
        self.specialized_output_dir = self.output_dir / 'specialized'
        self.specialized_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        self.data_loaders = self._initialize_data_loaders()
        
        # Task generator registry
        self.generator_registry = self._build_generator_registry()
        
        # Initialize the TrajectoryAugmenter for integrated augmentation
        augment_config = config.get('trajectory_augmentation', {})
        if augment_config.get('enabled', True):
            # Prepare config for TrajectoryAugmenter
            augmenter_config = {
                'proportions': OmegaConf.to_container(augment_config.get('proportions', {
                    'golden': 0.60,
                    'trap': 0.20,
                    'self_correction': 0.20
                }))
            }
            self.augmenter = TrajectoryAugmenter(config=augmenter_config)
            self.augment_proportions = augmenter_config['proportions']
            log.info(f"TrajectoryAugmenter initialized with proportions: {self.augment_proportions}")
        else:
            self.augmenter = None
            log.info("Trajectory augmentation is disabled")
    
    def _initialize_data_loaders(self) -> Dict[str, Any]:
        """Initialize ALL data loader instances from datasources in manifest."""
        log.info("Initializing all available dataloaders from manifest...")
        all_loaders = {}
        
        # Get datasource configurations from manifest
        datasources = self.config.get('datasources', {})
        
        for datasource_name, datasource_config in datasources.items():
            try:
                # For now, create mock loaders as a placeholder
                # In production, this would use a factory based on datasource_config['type']
                loader = self._initialize_loader(datasource_name, OmegaConf.to_container(datasource_config))
                all_loaders[datasource_name] = loader
                log.info(f"-> Successfully initialized loader for '{datasource_name}'")
            except Exception as e:
                log.error(f"-> Failed to initialize loader for '{datasource_name}': {e}")
        
        log.info(f"Total initialized loaders: {len(all_loaders)}")
        return all_loaders
    
    def _initialize_loader(self, name: str, config: Dict[str, Any]) -> Any:
        """Initialize a single data loader based on datasource configuration."""
        # For now, just use mock loaders to test the dispatch logic
        # In production, uncomment the code below to use real loaders
        return self._create_mock_loader(name, config)
        
        # # Try to import and use the real loader factory
        # try:
        #     from core.dataloaders.loader_factory import create_dataloader
        #     return create_dataloader(name, config)
        # except (ImportError, ValueError) as e:
        #     log.warning(f"Could not create real loader for '{name}': {e}. Using mock loader.")
        #     return self._create_mock_loader(name, config)
    
    def _create_mock_loader(self, name: str, config: Dict[str, Any]) -> Any:
        """Create a mock data loader for development."""
        class MockLoader:
            def __init__(self, name, config):
                self.name = name
                self.config = config
            
            def sample(self, n=1):
                return [{"image_id": f"{self.name}_{i}", "data": "mock"} for i in range(n)]
            
            def get_item(self, idx):
                return {"image_id": f"{self.name}_{idx}", "data": "mock"}
            
            def __len__(self):
                return 1000  # Mock length
        
        return MockLoader(name, config)
    
    def _build_generator_registry(self) -> Dict[str, type]:
        """Build registry of available task generator classes."""
        registry = {}
        
        # Map of generator names to their module paths
        generator_mappings = {
            'DetailPerceptionTaskGenerator': 'core.data_generation.detail_perception',
            'GeometricComparisonTaskGenerator': 'core.data_generation.geometric_comparison',
            'SpatioTemporalTaskGenerator': 'core.data_generation.spatiotemporal',
            'TargetedOCRTaskGenerator': 'core.data_generation.targeted_ocr',
            'ZoomInTaskGenerator': 'core.data_generation.zoom_in',
            'SelectFrameTaskGenerator': 'core.data_generation.select_frame'
        }
        
        for class_name, module_path in generator_mappings.items():
            try:
                module = importlib.import_module(module_path)
                generator_class = getattr(module, class_name)
                registry[class_name] = generator_class
                log.debug(f"Registered generator: {class_name}")
            except Exception as e:
                log.warning(f"Failed to register {class_name}: {e}")
        
        return registry
    
    def _create_sample_signature(self, sample: Dict) -> tuple:
        """
        Create a unique signature for a sample to detect duplicates.
        
        Args:
            sample: The sample to create a signature for
            
        Returns:
            A tuple that can be used as a set key
        """
        return (
            sample.get('question', ''),
            sample.get('final_answer', ''),
            # Include first trajectory step to be more robust
            sample.get('trajectory', [{}])[0].get('content', '') if sample.get('trajectory') else ''
        )
    
    def _save_checkpoint(self, samples: List[Dict], checkpoint_path: Path):
        """
        Save samples to checkpoint file.
        
        Args:
            samples: List of samples to save
            checkpoint_path: Path to save checkpoint
        """
        try:
            # Write to temporary file first for atomicity
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Atomic rename
            temp_path.replace(checkpoint_path)
            log.debug(f"Saved checkpoint with {len(samples)} samples to {checkpoint_path}")
            
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
    
    def _apply_augmentation(self, golden_samples: List[Dict[str, Any]], task_name: str) -> List[Dict[str, Any]]:
        """
        Apply trajectory augmentation to golden samples.
        
        Args:
            golden_samples: List of generated golden trajectory samples
            task_name: Name of the task (for logging)
            
        Returns:
            List of augmented samples (golden + trap + self-correction)
        """
        if not self.augmenter or not golden_samples:
            return golden_samples
        
        log.info(f"Applying trajectory augmentation for task '{task_name}'...")
        
        # Convert samples to Trajectory objects
        trajectories = []
        for sample in golden_samples:
            # Map 'trajectory' field to 'actions' for the Trajectory dataclass
            actions = sample.get('trajectory', sample.get('actions', []))
            
            trajectory = Trajectory(
                task_id=sample.get('task_id', f"{task_name}_{len(trajectories)}"),
                question=sample.get('question', ''),
                actions=actions,  # Use trajectory field if present, fallback to actions
                final_answer=sample.get('final_answer', ''),
                trajectory_type='golden',
                metadata=sample.get('metadata', {})
            )
            trajectories.append(trajectory)
        
        # Calculate how many of each type to generate
        total_target = len(golden_samples)
        proportions = self.augment_proportions
        
        # Calculate samples needed for each type
        n_golden = int(total_target * proportions.get('golden_positive', 0.60))
        n_trap = int(total_target * proportions.get('trap_samples', 0.20))
        n_self_correction = int(total_target * proportions.get('self_correction', 0.20))
        
        # Adjust to ensure we hit the target
        remainder = total_target - (n_golden + n_trap + n_self_correction)
        n_golden += remainder
        
        augmented_samples = []
        
        # 1. Add golden samples (subset)
        golden_subset = random.sample(trajectories, min(n_golden, len(trajectories)))
        for traj in golden_subset:
            augmented_samples.append({
                'task_id': traj.task_id,
                'question': traj.question,
                'actions': traj.actions,
                'final_answer': traj.final_answer,
                'trajectory_type': 'golden',
                'metadata': traj.metadata
            })
        
        # 2. Generate trap samples (process-negative)
        trap_count = 0
        for i in range(min(n_trap, len(trajectories))):
            source_traj = random.choice(trajectories)
            try:
                # Use the augmenter to create trap trajectory
                trap_traj = self.augmenter.augment_as_trap(source_traj)
                augmented_samples.append({
                    'task_id': trap_traj.task_id,
                    'question': trap_traj.question,
                    'actions': trap_traj.actions,
                    'final_answer': trap_traj.final_answer,
                    'trajectory_type': trap_traj.trajectory_type,
                    'metadata': trap_traj.metadata
                })
                trap_count += 1
            except Exception as e:
                log.warning(f"Failed to create trap sample: {e}")
        
        # 3. Generate self-correction samples
        sc_count = 0
        for i in range(min(n_self_correction, len(trajectories))):
            source_traj = random.choice(trajectories)
            try:
                # Use the augmenter to create self-correction trajectory
                aug_traj = self.augmenter.augment_trajectory(source_traj)
                augmented_samples.append({
                    'task_id': aug_traj.task_id,
                    'question': aug_traj.question,
                    'actions': aug_traj.actions,
                    'final_answer': aug_traj.final_answer,
                    'trajectory_type': aug_traj.trajectory_type,
                    'metadata': aug_traj.metadata
                })
                sc_count += 1
            except Exception as e:
                log.warning(f"Failed to create self-correction sample: {e}")
        
        # Shuffle the augmented samples
        random.shuffle(augmented_samples)
        
        log.info(f"Augmentation complete for '{task_name}': "
                   f"{n_golden} golden, {trap_count} trap, {sc_count} self-correction samples")
        
        return augmented_samples
    
    # DEPRECATED: This method has been replaced by TrajectoryAugmenter.augment_as_trap
    # The old implementation was causing malformed trap samples with missing action steps.
    # Now using the improved augment_as_trap method in TrajectoryAugmenter class.
            
    def generate_task_dataset(self, task_name: str, task_config: DictConfig) -> Optional[Path]:
        """
        Generate dataset for a specific task using its specialized generator.
        
        Args:
            task_name: Name of the task
            task_config: Configuration for the task
            
        Returns:
            Path to the generated dataset file, or None if generation failed
        """
        if not task_config.get('enabled', True):
            log.info(f"Skipping disabled task: {task_name}")
            return None
        
        task_generator_class = task_config.get('task_generator_class')
        if not task_generator_class:
            log.error(f"No generator class specified for task: {task_name}")
            return None
        
        # Get generator class from registry
        generator_class = self.generator_registry.get(task_generator_class)
        if not generator_class:
            log.error(f"Generator class not found in registry: {task_generator_class}")
            return None
        
        try:
            # --- START OF CRITICAL FIX ---
            
            # Step A: Prepare the specific dictionary of loaders for THIS task.
            task_specific_loaders = {}
            source_datasets = task_config.get('source_datasets', [])
            
            # DEBUG: Show raw config
            log.debug(f"For task '{task_name}', found raw source_datasets config: {source_datasets}")
            
            # Extract dataset names, handling both string and dict formats
            source_dataset_names = []
            for dataset_info in source_datasets:
                # Handle both string and dict formats
                if isinstance(dataset_info, str):
                    source_dataset_names.append(dataset_info)
                elif isinstance(dataset_info, (dict, DictConfig)):  # Also handle OmegaConf DictConfig
                    name = dataset_info.get('name')
                    if name:
                        source_dataset_names.append(name)
                    else:
                        log.warning(f"Dataset config missing 'name' field: {dataset_info}")
            
            log.info(f"Task '{task_name}' requires these datasources: {source_dataset_names}")
            
            # Now get the loaders for this specific task
            for source_name in source_dataset_names:
                if source_name in self.data_loaders:
                    task_specific_loaders[source_name] = self.data_loaders[source_name]
                    log.debug(f"  -> Added loader '{source_name}' for task '{task_name}'")
                else:
                    # This is now a configuration error that should be reported
                    log.error(
                        f"Configuration Error: Task '{task_name}' requires datasource '{source_name}', "
                        f"but it is not defined in the `datasources` section or failed to initialize."
                    )
                    # Continue anyway with mock loader for development
                    log.warning(f"Creating mock loader for missing datasource '{source_name}'")
                    task_specific_loaders[source_name] = self._create_mock_loader(source_name, {})
            
            log.info(f"Injecting {len(task_specific_loaders)} loaders into '{task_name}'")
            
            # --- END OF CRITICAL FIX ---
            
            # Prepare configuration for generator
            log.info(f"Preparing configuration for {task_generator_class} task '{task_name}'")
            
            # Convert OmegaConf to dict for compatibility
            task_config_dict = OmegaConf.to_container(task_config)
            
            # Ensure task name is in config for proper logging
            if 'name' not in task_config_dict:
                task_config_dict['name'] = task_name
            
            # Set up output path
            output_file = self.specialized_output_dir / f"{task_name}.jsonl"
            checkpoint_file = self.specialized_output_dir / f"{task_name}_checkpoint.jsonl"
            
            if self.dry_run:
                log.info(f"[DRY RUN] Would generate {task_config.get('target_sample_count', 0)} samples for {task_name}")
                log.info(f"[DRY RUN] Output would be saved to: {output_file}")
                return output_file
            
            # Generate samples with deduplication and regeneration loop
            target_count = task_config.get('target_sample_count', 100)
            log.info(f"Target: {target_count} unique samples for task '{task_name}'...")
            
            # Initialize collections for deduplication
            seen_signatures = set()
            unique_samples = []
            all_generated_samples = []
            
            # Load from checkpoint if exists
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r') as f:
                        for line in f:
                            sample = json.loads(line.strip())
                            all_generated_samples.append(sample)
                            # Add to unique samples set
                            signature = self._create_sample_signature(sample)
                            if signature not in seen_signatures:
                                seen_signatures.add(signature)
                                unique_samples.append(sample)
                    log.info(f"Loaded checkpoint: {len(unique_samples)} unique samples from {len(all_generated_samples)} total")
                except Exception as e:
                    log.error(f"Failed to load checkpoint: {e}. Starting fresh.")
                    all_generated_samples = []
                    unique_samples = []
                    seen_signatures = set()
            
            # Check if we already have enough unique samples
            if len(unique_samples) >= target_count:
                log.info(f"Already have {len(unique_samples)} unique samples, meeting target of {target_count}")
                unique_samples = unique_samples[:target_count]
            else:
                # Maximum attempts to prevent infinite loops
                max_generation_rounds = 5
                generation_round = 0
                
                # Continue generating until we have enough unique samples
                while len(unique_samples) < target_count and generation_round < max_generation_rounds:
                    generation_round += 1
                    
                    # Calculate how many more samples we need
                    samples_needed = target_count - len(unique_samples)
                    
                    # Generate more samples (overshoot by 20% to account for duplicates)
                    samples_to_generate = int(samples_needed * 1.2) + 5 if generation_round > 1 else target_count
                    
                    log.info(f"Generation round {generation_round}: Generating {samples_to_generate} samples "
                            f"(need {samples_needed} more unique samples)...")
                    
                    # Create a fresh generator for this round (stateless)
                    generator = generator_class(
                        loaders=task_specific_loaders,
                        config=task_config_dict,
                        global_config=OmegaConf.to_container(self.config.global_config)
                    )
                    
                    # Generate new batch of samples (stateless call)
                    golden_samples = generator.generate(num_samples=samples_to_generate)
                
                    # Apply augmentation to the new batch
                    augmented_samples = self._apply_augmentation(golden_samples, task_name)
                    all_generated_samples.extend(augmented_samples)
                    
                    # Deduplicate across all samples generated so far
                    log.info("Running deduplication...")
                    new_unique_count = 0
                    for sample in augmented_samples:
                        # Create a signature for this sample
                        signature = self._create_sample_signature(sample)
                        
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            unique_samples.append(sample)
                            new_unique_count += 1
                    
                    log.info(f"Round {generation_round} deduplication: "
                            f"Generated {len(augmented_samples)} samples, "
                            f"{new_unique_count} were unique. "
                            f"Total unique: {len(unique_samples)}/{target_count}")
                    
                    # Save checkpoint after each round
                    self._save_checkpoint(all_generated_samples, checkpoint_file)
                    
                    # If we're not making progress, log a warning
                    if new_unique_count == 0 and len(unique_samples) < target_count:
                        log.warning(f"No new unique samples in round {generation_round}. "
                                  f"May need to adjust generation parameters.")
            
            # Final status
            if len(unique_samples) >= target_count:
                unique_samples = unique_samples[:target_count]  # Trim to exact target
                log.info(f"✓ Successfully generated {target_count} unique samples")
            else:
                log.warning(f"Could only generate {len(unique_samples)}/{target_count} unique samples "
                          f"after {generation_round} rounds")
            
            log.info(f"Final deduplication summary: "
                     f"Total generated: {len(all_generated_samples)}, "
                     f"Unique samples: {len(unique_samples)}")
            
            # Save the unique samples to task-specific file
            with open(output_file, 'w') as f:
                for sample in unique_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            log.info(f"Saved {len(unique_samples)} unique samples for '{task_name}' -> {output_file}")
            
            # Clean up checkpoint if generation completed successfully
            if checkpoint_file.exists() and len(unique_samples) >= target_count:
                checkpoint_file.unlink()
                log.debug(f"Removed checkpoint file: {checkpoint_file}")
            
            return output_file
            
        except Exception as e:
            log.error(f"Failed to generate dataset for task '{task_name}': {e}")
            import traceback
            log.debug(traceback.format_exc())
            return None
        
        
    def generate_all_datasets(self, tasks: Optional[List[str]] = None):
        """
        Generate datasets for all or specified tasks.
        
        Args:
            tasks: Optional list of specific tasks to generate. If None, generate all enabled tasks.
        """
        generated_files = []
        task_stats = {}
        
        # Determine which tasks to process
        tasks_to_process = {}
        for task_name, task_config in self.config.tasks.items():
            if tasks and task_name not in tasks:
                continue
            if task_config.get('enabled', True):
                tasks_to_process[task_name] = task_config
        
        if not tasks_to_process:
            log.warning("No tasks to process. Check configuration or task list.")
            return
        
        log.info(f"Processing {len(tasks_to_process)} tasks...")
        
        # Generate each task's dataset independently
        for task_name, task_config in tasks_to_process.items():
            log.info(f"\n{'='*60}")
            log.info(f"Processing task: {task_name}")
            log.info(f"{'='*60}")
            
            output_file = self.generate_task_dataset(task_name, task_config)
            
            if output_file:
                generated_files.append(output_file)
                
                # Collect statistics
                if output_file.exists() and not self.dry_run:
                    sample_count = sum(1 for _ in open(output_file))
                    task_stats[task_name] = {
                        'samples': sample_count,
                        'file': str(output_file.relative_to(self.output_dir))
                    }
                elif self.dry_run:
                    task_stats[task_name] = {
                        'samples': task_config.get('target_sample_count', 0),
                        'file': str(output_file.relative_to(self.output_dir)),
                        'dry_run': True
                    }
        
        # Generate summary report
        self._generate_summary(task_stats, generated_files)
        
        
    def _generate_summary(self, task_stats: Dict[str, Any], generated_files: List[Path]):
        """Generate and save summary report."""
        summary = {
            "stage": "Stage 1: Specialized Dataset Generation",
            "output_directory": str(self.specialized_output_dir),
            "tasks_processed": len(task_stats),
            "total_samples": sum(s.get('samples', 0) for s in task_stats.values()),
            "tasks": task_stats,
            "generated_files": [str(f.relative_to(self.output_dir)) for f in generated_files],
            "dry_run": self.dry_run,
            "next_stage": {
                "script": "scripts/2_fuse_and_validate_dataset.py",
                "input_dir": str(self.specialized_output_dir),
                "description": "Fuse pre-augmented specialized datasets and validate"
            }
        }
        
        # Save summary
        summary_file = self.specialized_output_dir / "stage1_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        log.info("\n" + "="*60)
        log.info("STAGE 1 GENERATION SUMMARY")
        log.info("="*60)
        log.info(f"Tasks processed: {len(task_stats)}")
        log.info(f"Total samples generated: {summary['total_samples']}")
        log.info(f"Output directory: {self.specialized_output_dir}")
        
        if task_stats:
            log.info("\nTask breakdown:")
            for task_name, stats in task_stats.items():
                log.info(f"  - {task_name}: {stats['samples']} samples")
        
        log.info(f"\nSummary saved to: {summary_file}")
        log.info(f"\nNext: Run Stage 2 with 'python {summary['next_stage']['script']}'")


# The @hydra.main decorator MUST be present.
# It tells Hydra where to find the configuration files.
@hydra.main(version_base=None, config_path="../configs", config_name="data_generation_manifest")
def main(cfg: DictConfig) -> None:
    """
    Main function for Stage 1, now correctly managed by Hydra.
    All configuration is passed via the 'cfg' object.
    """
    
    # [CRITICAL] Configure logging - set to DEBUG for forensic analysis
    # Access script arguments from config first to determine log level
    is_dry_run = cfg.get('dry_run', False)
    tasks = cfg.get('tasks_to_run', None)  # Can be overridden via command line
    verbose = cfg.get('verbose', False)
    
    # Configure logging with appropriate level
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration to ensure DEBUG messages are shown
    )
    
    # Also set all loggers to DEBUG when verbose
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Ensure child loggers also use DEBUG
        for logger_name in ['core.data_generation', '__main__']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    log.info("="*60)
    log.info("STAGE 1: SPECIALIZED DATASET GENERATION")
    log.info("="*60)
    log.info(f"Starting Stage 1 generation. Dry run: {is_dry_run}")
    
    # Initialize generator with Hydra config
    generator = SpecializedDatasetGenerator(cfg, dry_run=is_dry_run)
    
    # Check for output directory override
    output_dir_override = cfg.get('output_dir_override', None)
    if output_dir_override:
        generator.output_dir = Path(output_dir_override)
        # Check if the output_dir already ends with 'specialized'
        if generator.output_dir.name == 'specialized':
            generator.specialized_output_dir = generator.output_dir
        else:
            generator.specialized_output_dir = generator.output_dir / 'specialized'
        generator.specialized_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run generation
    if is_dry_run:
        log.info("[DRY RUN MODE] No API calls will be made")
    
    # Convert tasks list if provided
    task_list = None
    if tasks:
        if isinstance(tasks, str):
            task_list = [tasks]
        elif isinstance(tasks, list):
            task_list = tasks
    
    generator.generate_all_datasets(tasks=task_list)
    
    log.info("Stage 1 script has completed successfully.")


# This block ensures the script is executable and launches the Hydra app.
if __name__ == "__main__":
    main()