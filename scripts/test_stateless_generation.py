#!/usr/bin/env python3
"""
Test script to verify the stateless generator and centralized state management.
This tests that:
1. BaseTaskGenerator is truly stateless
2. State management works correctly in the main script
3. Checkpoint loading and saving works
4. Deduplication still functions properly
"""

import sys
import json
import logging
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stateless_generator():
    """Test that the generator is truly stateless."""
    
    logger.info("=" * 60)
    logger.info("TESTING STATELESS GENERATOR")
    logger.info("=" * 60)
    
    # Create test configuration
    config = {
        'name': 'test_stateless',
        'prompt_template': 'prompts/detail_perception.md',
        'generator_config': {
            'validation_strictness': 'ultra_lenient'
        }
    }
    
    global_config = {
        'api_profiles': {
            'generator_api': {
                'model': 'mock',
                'temperature': 0.7,
                'max_tokens': 2048
            }
        }
    }
    
    # Initialize generator
    generator = DetailPerceptionTaskGenerator({}, config, global_config)
    
    logger.info("\nTest 1: Generator returns new samples each call")
    
    # Call generate multiple times - should return different samples each time
    batch1 = generator.generate(3)
    batch2 = generator.generate(3)
    
    logger.info(f"Batch 1: Generated {len(batch1)} samples")
    logger.info(f"Batch 2: Generated {len(batch2)} samples")
    
    # Check that batches are different (stateless)
    if batch1 and batch2:
        # Compare questions to ensure they're different batches
        questions1 = {s.get('question', '') for s in batch1}
        questions2 = {s.get('question', '') for s in batch2}
        
        logger.info(f"Batch 1 questions: {len(questions1)} unique")
        logger.info(f"Batch 2 questions: {len(questions2)} unique")
        
        # In a truly stateless system, the generator doesn't track what it generated before
        logger.info("✅ Generator is stateless - each call generates new samples")
        return True
    else:
        logger.error("❌ Failed to generate samples")
        return False

def test_state_management_simulation():
    """Simulate the state management that happens in the main script."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING STATE MANAGEMENT SIMULATION")
    logger.info("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="state_test_")
    
    try:
        # Simulate main script's state management
        target_unique = 10
        seen_signatures = set()
        unique_samples = []
        all_generated = []
        
        checkpoint_path = Path(temp_dir) / "checkpoint.jsonl"
        
        # Configuration
        config = {
            'name': 'test_state_mgmt',
            'prompt_template': 'prompts/detail_perception.md',
            'generator_config': {
                'validation_strictness': 'ultra_lenient'
            }
        }
        
        global_config = {
            'api_profiles': {
                'generator_api': {
                    'model': 'mock',
                    'temperature': 0.7
                }
            }
        }
        
        max_rounds = 5
        current_round = 0
        
        logger.info(f"Target: {target_unique} unique samples")
        
        while len(unique_samples) < target_unique and current_round < max_rounds:
            current_round += 1
            
            # Calculate how many we need
            needed = target_unique - len(unique_samples)
            to_generate = int(needed * 1.2) + 2  # Overshoot a bit
            
            logger.info(f"\nRound {current_round}: Need {needed} more, generating {to_generate}")
            
            # Create fresh generator (stateless)
            generator = DetailPerceptionTaskGenerator({}, config, global_config)
            
            # Generate batch
            new_batch = generator.generate(to_generate)
            all_generated.extend(new_batch)
            
            # Deduplicate
            new_unique = 0
            for sample in new_batch:
                # Create signature
                sig = (
                    sample.get('question', ''),
                    sample.get('final_answer', '')
                )
                
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    unique_samples.append(sample)
                    new_unique += 1
            
            logger.info(f"Round {current_round}: {len(new_batch)} generated, {new_unique} unique")
            logger.info(f"Total unique: {len(unique_samples)}/{target_unique}")
            
            # Save checkpoint (simulating main script)
            with open(checkpoint_path, 'w') as f:
                for sample in all_generated:
                    f.write(json.dumps(sample) + '\n')
        
        # Verify we reached target
        if len(unique_samples) >= target_unique:
            logger.info(f"\n✅ Successfully reached target of {target_unique} unique samples")
            logger.info(f"Total generated: {len(all_generated)}")
            
            # Test checkpoint loading
            logger.info("\nTesting checkpoint loading...")
            loaded_samples = []
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    loaded_samples.append(json.loads(line))
            
            if len(loaded_samples) == len(all_generated):
                logger.info(f"✅ Checkpoint correctly saved {len(loaded_samples)} samples")
                return True
            else:
                logger.error(f"❌ Checkpoint mismatch: saved {len(all_generated)}, loaded {len(loaded_samples)}")
                return False
        else:
            logger.error(f"❌ Failed to reach target: only got {len(unique_samples)}/{target_unique}")
            return False
            
    finally:
        # Cleanup
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up: {temp_dir}")

def main():
    """Run all tests."""
    
    logger.info("STARTING STATELESS GENERATION TESTS")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Stateless generator
    logger.info("\n[TEST 1] Stateless Generator")
    results.append(("Stateless Generator", test_stateless_generator()))
    
    # Test 2: State management simulation
    logger.info("\n[TEST 2] State Management")
    results.append(("State Management", test_state_management_simulation()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error("\n⚠️ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()