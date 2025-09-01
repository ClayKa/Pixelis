#!/usr/bin/env python3
"""
Test script for the enhanced data generation pipeline.
Tests the improvements from improvement.md implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import json
from pathlib import Path

# Import the generators
from core.data_generation.detail_perception import DetailPerceptionTaskGenerator
from core.data_generation.trajectory_augmenter import TrajectoryAugmenter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dynamic_scenario_generation():
    """Test Pillar 2.1: Dynamic Scenario Generation"""
    logger.info("\n" + "="*60)
    logger.info("Testing Dynamic Scenario Generation (Pillar 2.1)")
    logger.info("="*60)
    
    # Create a mock generator
    config = {
        'task_name': 'detail_perception_task',
        'samples_per_run': 5,
        'validation_strictness': 'lenient',
        'prompt_template_path': 'core/data_generation/prompt_templates/detail_perception.md'
    }
    global_config = {
        'data_generation': {
            'prompt_templates_dir': 'core/data_generation/prompt_templates'
        }
    }
    
    generator = DetailPerceptionTaskGenerator({}, config, global_config)
    
    # Test generating diverse observations
    observations = set()
    difficulties = ['Easy', 'Medium', 'Hard']
    
    for _ in range(20):
        for difficulty in difficulties:
            obs = generator._generate_dynamic_observation(difficulty)
            observations.add(obs)
            logger.info(f"{difficulty}: {obs}")
    
    logger.info(f"\nGenerated {len(observations)} unique observations from 60 attempts")
    logger.info(f"Diversity ratio: {len(observations)/60:.2%}")
    
    # Check for nothing found and ambiguous scenarios
    nothing_found_count = sum(1 for obs in observations if 'no ' in obs.lower() or 'absent' in obs.lower() or 'not present' in obs.lower())
    ambiguous_count = sum(1 for obs in observations if 'unclear' in obs.lower() or 'might be' in obs.lower() or 'possibly' in obs.lower())
    
    logger.info(f"Nothing Found scenarios: {nothing_found_count}")
    logger.info(f"Ambiguous scenarios: {ambiguous_count}")
    
    return len(observations) > 40  # Expect high diversity

def test_trap_augmentation():
    """Test expanded trap types (wrong_tool and bad_parameter)"""
    logger.info("\n" + "="*60)
    logger.info("Testing Expanded Trap Types")
    logger.info("="*60)
    
    config = {
        'proportions': {
            'golden': 0.0,
            'trap': 1.0,  # Force all to be traps for testing
            'self_correction': 0.0
        }
    }
    
    augmenter = TrajectoryAugmenter(config)
    
    # Create sample trajectories
    sample_trajectories = [
        {
            'task_id': f'test_{i}',
            'question': f'Test question {i}',
            'trajectory': [
                {'type': 'thought', 'content': 'Initial thought'},
                {'type': 'action', 'name': 'ZOOM-IN', 'parameters': {'bbox': [100, 100, 200, 200]}},
                {'type': 'thought', 'content': 'Final thought'}
            ],
            'final_answer': 'Test answer'
        }
        for i in range(10)
    ]
    
    # Process and check trap types
    augmented = augmenter.process(sample_trajectories)
    
    trap_types = {}
    for sample in augmented:
        if sample.get('trajectory_type') == 'trap':
            trap_type = sample.get('metadata', {}).get('trap_type', 'unknown')
            trap_types[trap_type] = trap_types.get(trap_type, 0) + 1
            
            # Log example of each trap type
            if trap_types[trap_type] == 1:
                logger.info(f"\nExample of {trap_type} trap:")
                logger.info(f"  Question: {sample['question']}")
                logger.info(f"  Trajectory: {json.dumps(sample['trajectory'], indent=2)[:500]}...")
    
    logger.info(f"\nTrap type distribution: {trap_types}")
    
    # Check that we have multiple trap types
    return len(trap_types) >= 3  # Expect at least 3 different trap types

def test_nothing_found_scenarios():
    """Test Pillar 3.1: Nothing Found scenarios"""
    logger.info("\n" + "="*60)
    logger.info("Testing 'Nothing Found' Scenarios (Pillar 3.1)")
    logger.info("="*60)
    
    config = {
        'task_name': 'detail_perception_task',
        'samples_per_run': 5,
        'validation_strictness': 'lenient',
        'prompt_template_path': 'core/data_generation/prompt_templates/detail_perception.md'
    }
    global_config = {
        'data_generation': {
            'prompt_templates_dir': 'core/data_generation/prompt_templates'
        }
    }
    
    generator = DetailPerceptionTaskGenerator({}, config, global_config)
    vocab = generator._get_dynamic_vocabulary()
    
    # Generate several "nothing found" observations
    nothing_found_examples = []
    for _ in range(5):
        obs = generator._generate_nothing_found_observation('Medium', vocab)
        nothing_found_examples.append(obs)
        logger.info(f"  - {obs}")
    
    # Check that they indicate absence
    valid_count = sum(1 for obs in nothing_found_examples 
                     if any(word in obs.lower() for word in ['no ', 'absent', 'unable', 'not ', 'cannot']))
    
    logger.info(f"\nValid 'Nothing Found' observations: {valid_count}/5")
    return valid_count == 5

def test_ambiguity_scenarios():
    """Test Pillar 3.2: Ambiguity scenarios"""
    logger.info("\n" + "="*60)
    logger.info("Testing Ambiguity Scenarios (Pillar 3.2)")
    logger.info("="*60)
    
    config = {
        'task_name': 'detail_perception_task',
        'samples_per_run': 5,
        'validation_strictness': 'lenient'
    }
    
    generator = DetailPerceptionTaskGenerator({}, config, {})
    vocab = generator._get_dynamic_vocabulary()
    
    # Generate several ambiguous observations
    ambiguous_examples = []
    for _ in range(5):
        obs = generator._generate_ambiguous_observation('Hard', vocab)
        ambiguous_examples.append(obs)
        logger.info(f"  - {obs}")
    
    # Check that they express uncertainty
    valid_count = sum(1 for obs in ambiguous_examples 
                     if any(word in obs.lower() for word in ['might', 'unclear', 'possibly', 'uncertain', 'inconclusive']))
    
    logger.info(f"\nValid ambiguous observations: {valid_count}/5")
    return valid_count == 5

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("ENHANCED DATA GENERATION PIPELINE TEST SUITE")
    logger.info("="*80)
    
    results = {
        "Dynamic Scenario Generation": test_dynamic_scenario_generation(),
        "Expanded Trap Types": test_trap_augmentation(),
        "Nothing Found Scenarios": test_nothing_found_scenarios(),
        "Ambiguity Scenarios": test_ambiguity_scenarios()
    }
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The enhanced data generation pipeline is working correctly.")
    else:
        logger.info("\n⚠️ Some tests failed. Please review the output above for details.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())