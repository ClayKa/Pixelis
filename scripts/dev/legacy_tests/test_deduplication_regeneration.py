#!/usr/bin/env python3
"""
Test script to verify the deduplication and regeneration loop.
This simulates a scenario where duplicates occur and verifies that
the system continues generating until the target number of unique samples is reached.
"""

import sys
import json
import logging
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import needed for exec to work
import importlib.util
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockGeneratorWithDuplicates:
    """Mock generator that intentionally produces duplicates to test deduplication."""
    
    def __init__(self, loaders, config, global_config):
        self.loaders = loaders
        self.config = config
        self.global_config = global_config
        self.task_name = config.get('name', 'test_task')
        self.call_count = 0
        self.unique_questions = [
            "Question A", "Question B", "Question C", "Question D", "Question E",
            "Question F", "Question G", "Question H", "Question I", "Question J"
        ]
    
    def generate(self, num_samples: int, checkpoint_path: Path) -> List[Dict]:
        """Generate samples with intentional duplicates."""
        self.call_count += 1
        samples = []
        
        for i in range(num_samples):
            # Create duplicates: for first call, repeat some questions
            if self.call_count == 1:
                # First batch: 50% duplicates
                if i < num_samples // 2:
                    question = self.unique_questions[i % 5]  # Only use first 5 questions
                else:
                    question = self.unique_questions[i % 5]  # Repeat them
            else:
                # Subsequent batches: use remaining unique questions
                question = self.unique_questions[min(5 + i, 9)]
            
            sample = {
                'question': question,
                'trajectory': [
                    {'type': 'thought', 'content': f'Thinking about {question}'},
                    {'type': 'action', 'name': 'ZOOM-IN', 'parameters': {'bbox': [100, 100, 200, 200]}},
                    {'type': 'thought', 'content': f'Answer for {question}'}
                ],
                'final_answer': f'Answer to {question}',
                'difficulty': 'Easy'
            }
            samples.append(sample)
        
        logger.info(f"Mock generator call #{self.call_count}: Generated {len(samples)} samples")
        return samples

def test_deduplication_with_regeneration():
    """Test that the system continues generating until target unique samples are reached."""
    
    logger.info("=" * 60)
    logger.info("TESTING DEDUPLICATION WITH REGENERATION")
    logger.info("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dedup_test_")
    
    try:
        # Load the SpecializedDatasetGenerator class
        script_path = project_root / 'scripts' / '1_generate_specialized_datasets.py'
        spec = importlib.util.spec_from_file_location("stage1", script_path)
        stage1_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage1_module)
        SpecializedDatasetGenerator = stage1_module.SpecializedDatasetGenerator
        
        # Create test configuration as OmegaConf dict
        config_dict = OmegaConf.create({
            'manifest_version': '1.0',
            'global_config': {
                'output_dir': temp_dir,
                'api_profiles': {
                    'generator_api': {
                        'model': 'mock',
                        'temperature': 0.7
                    }
                },
                'checkpoint_every_n_samples': 100
            },
            'trajectory_augmentation': {
                'enabled': False,
                'proportions': {
                    'golden_positive': 0.6,
                    'trap_samples': 0.2,
                    'self_correction': 0.2
                }
            },
            'datasources': {},
            'tasks': {
                'test_task': {
                    'task_generator_class': 'MockGeneratorWithDuplicates',
                    'prompt_template': 'test.md',
                    'target_sample_count': 10,  # We want 10 unique samples
                    'augmentation': {
                        'enabled': False  # Disable augmentation for this test
                    }
                }
            }
        })
        
        # Initialize SpecializedDatasetGenerator (config_dict is already OmegaConf)
        generator = SpecializedDatasetGenerator(config_dict, dry_run=False)
        generator.specialized_output_dir = Path(temp_dir) / "output"
        generator.specialized_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add our mock generator to the generator's registry
        generator.generator_registry['MockGeneratorWithDuplicates'] = MockGeneratorWithDuplicates
        
        # Or use patch
        with patch.dict(generator.generator_registry, 
                       {'MockGeneratorWithDuplicates': MockGeneratorWithDuplicates}):
            
            # Mock the augmenter to disable augmentation
            # Set augment_proportions first
            generator.augment_proportions = {
                'golden_positive': 1.0,  # All samples are golden
                'trap_samples': 0.0,
                'self_correction': 0.0
            }
            generator.augmenter = None  # Disable augmentation completely
            
            # Generate dataset for test task
            output_file = generator.generate_task_dataset('test_task', config_dict.tasks.test_task)
            
            # Verify the output
            if output_file and output_file.exists():
                with open(output_file, 'r') as f:
                    samples = [json.loads(line) for line in f]
                
                logger.info(f"\nResults:")
                logger.info(f"Output file: {output_file}")
                logger.info(f"Number of samples in file: {len(samples)}")
                
                # Check uniqueness
                unique_questions = set(s['question'] for s in samples)
                logger.info(f"Number of unique questions: {len(unique_questions)}")
                logger.info(f"Unique questions: {sorted(unique_questions)}")
                
                # Verify we have exactly the target number
                target = config_dict.tasks.test_task.target_sample_count
                if len(samples) == target:
                    logger.info(f"\n✅ SUCCESS: Generated exactly {target} unique samples as requested!")
                    
                    # Verify they are actually unique
                    if len(unique_questions) == len(samples):
                        logger.info("✅ All samples are unique!")
                        return True
                    else:
                        logger.error(f"❌ Some samples are not unique!")
                        return False
                else:
                    logger.error(f"\n❌ FAILURE: Expected {target} samples but got {len(samples)}")
                    return False
            else:
                logger.error("❌ Output file was not created")
                return False
                
    finally:
        # Clean up
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")

def main():
    """Main test runner."""
    success = test_deduplication_with_regeneration()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("TEST PASSED ✅")
        sys.exit(0)
    else:
        logger.error("TEST FAILED ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()