#!/usr/bin/env python3
"""
Test script to verify the Generate-Until-Valid loop implementation.
This script tests that the generator continues retrying until it gets
the exact number of valid samples requested.
"""

import sys
import json
import logging
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_generate_until_valid():
    """Test that the generator produces exactly the requested number of valid samples."""
    
    logger.info("=" * 60)
    logger.info("TESTING GENERATE-UNTIL-VALID LOOP")
    logger.info("=" * 60)
    
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp(prefix="cota_test_")
    checkpoint_path = Path(temp_dir) / "test_checkpoint.jsonl"
    
    try:
        # Create test configuration
        config = {
            'name': 'test_detail_perception',
            'prompt_template': 'prompts/detail_perception.md',
            'generator_config': {
                'validation_strictness': 'ultra_lenient'  # Use lenient validation for testing
            }
        }
        
        global_config = {
            'api_profiles': {
                'generator_api': {
                    'model': 'mock',
                    'temperature': 0.7,
                    'max_tokens': 2048
                }
            },
            'checkpoint_every_n_samples': 5
        }
        
        # Initialize generator (no loaders needed for mock mode)
        loaders = {}
        generator = DetailPerceptionTaskGenerator(loaders, config, global_config)
        
        # Request a specific number of samples
        target_samples = 10
        logger.info(f"\nRequesting exactly {target_samples} VALID samples...")
        
        # Generate samples
        samples = generator.generate(target_samples, checkpoint_path)
        
        # Verify we got exactly the requested number
        actual_count = len(samples)
        logger.info(f"\nGeneration complete!")
        logger.info(f"Requested: {target_samples} samples")
        logger.info(f"Received:  {actual_count} samples")
        
        # Print statistics
        stats = generator.generation_stats
        logger.info(f"\nGeneration Statistics:")
        logger.info(f"  Valid samples:   {stats.get('samples_generated', 0)}")
        logger.info(f"  Invalid samples: {stats.get('samples_invalid', 0)}")
        logger.info(f"  Failed attempts: {stats.get('samples_failed', 0)}")
        
        # Verify all samples are valid (have required fields)
        all_valid = True
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                logger.error(f"Sample {i} is not a dictionary!")
                all_valid = False
            elif 'question' not in sample or 'final_answer' not in sample:
                logger.error(f"Sample {i} missing required fields!")
                all_valid = False
        
        # Check success
        if actual_count == target_samples and all_valid:
            logger.info("\n✅ SUCCESS: Generate-Until-Valid loop works correctly!")
            logger.info(f"   Generated exactly {target_samples} valid samples as requested.")
            return True
        else:
            logger.error("\n❌ FAILURE: Generate-Until-Valid loop did not work as expected!")
            if actual_count != target_samples:
                logger.error(f"   Expected {target_samples} samples but got {actual_count}")
            if not all_valid:
                logger.error("   Some samples were invalid")
            return False
            
    finally:
        # Clean up temporary directory
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")

def main():
    """Main test runner."""
    success = test_generate_until_valid()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("TEST PASSED ✅")
        sys.exit(0)
    else:
        logger.error("TEST FAILED ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()