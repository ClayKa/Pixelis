#!/usr/bin/env python3
"""
Test script for the improved TrajectoryAugmenter module.
This script validates the robustness of the self-correction augmentation pipeline.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generation.trajectory_augmenter import (
    TrajectoryAugmenter,
    Trajectory,
    DistractorAction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_trajectories() -> List[Dict]:
    """Create sample golden trajectories for testing."""
    samples = [
        {
            "task_id": "test_001",
            "question": "What is the color of the car in the image?",
            "trajectory": [
                {
                    "type": "thought",
                    "content": "I need to locate the car in the image first."
                },
                {
                    "type": "action",
                    "name": "SEGMENT_OBJECT_AT",
                    "parameters": {"x": 250, "y": 300},
                    "observation": "Found object: car at coordinates (250, 300)"
                },
                {
                    "type": "action",
                    "name": "GET_PROPERTIES",
                    "parameters": {"object_id": 1},
                    "observation": "Properties: color=red, size=large, type=sedan"
                },
                {
                    "type": "thought",
                    "content": "The car is red in color."
                }
            ],
            "final_answer": "The car is red.",
            "metadata": {"source": "test"}
        },
        {
            "task_id": "test_002",
            "question": "Read the text on the sign.",
            "trajectory": [
                {
                    "type": "action",
                    "name": "ZOOM_IN",
                    "parameters": {"bbox": [100, 100, 300, 200]},
                    "observation": "Zoomed into region containing a sign"
                },
                {
                    "type": "action",
                    "name": "READ_TEXT",
                    "parameters": {"region": [100, 100, 300, 200]},
                    "observation": "Text detected: 'STOP'"
                },
                {
                    "type": "thought",
                    "content": "The sign says 'STOP'."
                }
            ],
            "final_answer": "STOP",
            "metadata": {"source": "test"}
        },
        {
            "task_id": "test_003",
            "question": "Track the person in the video.",
            "trajectory": [
                {
                    "type": "action",
                    "name": "SELECT_FRAME",
                    "parameters": {"frame_number": 0},
                    "observation": "Frame 0 selected"
                },
                {
                    "type": "action",
                    "name": "TRACK_OBJECT",
                    "parameters": {"object_id": 1, "frames": [0, 10]},
                    "observation": "Object tracked successfully across 10 frames"
                }
            ],
            "final_answer": "Person tracked from frame 0 to 10.",
            "metadata": {"source": "test"}
        },
        {
            "task_id": "test_004",
            "question": "What objects are visible?",
            "trajectory": [],  # Empty trajectory to test edge case
            "final_answer": "Multiple objects visible.",
            "metadata": {"source": "test"}
        },
        {
            "task_id": "test_005",
            "question": "Analyze the scene.",
            "trajectory": [
                {
                    "type": "thought",
                    "content": "Let me examine the scene."
                }
                # No action, only thought - testing edge case
            ],
            "final_answer": "Scene analyzed.",
            "metadata": {"source": "test"}
        }
    ]
    
    return samples


def test_augmentation_pipeline():
    """Test the complete augmentation pipeline."""
    logger.info("="*60)
    logger.info("Testing TrajectoryAugmenter Pipeline")
    logger.info("="*60)
    
    # Create configuration
    config = {
        "proportions": {
            "golden": 0.3,
            "trap": 0.3,
            "self_correction": 0.4
        }
    }
    
    # Initialize augmenter
    augmenter = TrajectoryAugmenter(config=config)
    
    # Create sample trajectories
    samples = create_sample_trajectories()
    logger.info(f"Created {len(samples)} sample trajectories")
    
    # Test the main process method
    logger.info("\n" + "="*40)
    logger.info("Testing main process() method")
    logger.info("="*40)
    
    try:
        augmented_results = augmenter.process(samples)
        logger.info(f"✓ Successfully processed {len(augmented_results)} trajectories")
        
        # Analyze results
        type_counts = {}
        for result in augmented_results:
            traj_type = result.get('trajectory_type', 'unknown')
            type_counts[traj_type] = type_counts.get(traj_type, 0) + 1
        
        logger.info("\nTrajectory type distribution:")
        for traj_type, count in type_counts.items():
            logger.info(f"  {traj_type}: {count}")
        
        # Check for augmentation failures
        failures = [r for r in augmented_results 
                   if 'augmentation_failure' in r.get('metadata', {})]
        if failures:
            logger.warning(f"\nFound {len(failures)} augmentation failures:")
            for failure in failures[:3]:  # Show first 3 failures
                logger.warning(f"  - {failure['task_id']}: {failure['metadata']['augmentation_failure']}")
        
    except Exception as e:
        logger.error(f"✗ Process method failed: {e}", exc_info=True)
        return False
    
    # Test individual augmentation methods
    logger.info("\n" + "="*40)
    logger.info("Testing individual augmentation methods")
    logger.info("="*40)
    
    # Test self-correction generation
    test_trajectory = Trajectory(
        task_id="test_sc",
        question="Test question",
        actions=samples[0]["trajectory"],
        final_answer="Test answer",
        trajectory_type="golden",
        metadata={}
    )
    
    logger.info("\n1. Testing distractor generation:")
    distractor = augmenter._generate_distractor_action(test_trajectory)
    if distractor:
        logger.info(f"  ✓ Generated distractor: {distractor.error_type}")
        logger.info(f"    Action: {distractor.action_type}")
        logger.info(f"    Observation: {distractor.observation}")
    else:
        logger.warning("  ✗ Failed to generate distractor")
    
    logger.info("\n2. Testing trap generation:")
    trap_result = augmenter.augment_as_trap(test_trajectory)
    if trap_result.trajectory_type == "trap":
        logger.info(f"  ✓ Generated trap trajectory")
        logger.info(f"    Trap type: {trap_result.metadata.get('trap_type', 'unknown')}")
    else:
        logger.warning("  ✗ Failed to generate trap")
    
    # Test error handling with invalid input
    logger.info("\n" + "="*40)
    logger.info("Testing error handling")
    logger.info("="*40)
    
    invalid_samples = [
        {"task_id": "invalid_1"},  # Missing required fields
        {"task_id": "invalid_2", "trajectory": None},  # None trajectory
        {"task_id": "invalid_3", "trajectory": "not_a_list"},  # Wrong type
    ]
    
    try:
        error_results = augmenter.process(invalid_samples)
        logger.info(f"✓ Handled {len(invalid_samples)} invalid samples gracefully")
        logger.info(f"  Produced {len(error_results)} results")
    except Exception as e:
        logger.error(f"✗ Failed to handle invalid input: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Testing complete!")
    logger.info("="*60)
    
    return True


def test_specific_edge_cases():
    """Test specific edge cases and failure modes."""
    logger.info("\n" + "="*60)
    logger.info("Testing Edge Cases")
    logger.info("="*60)
    
    config = {"proportions": {"golden": 0, "trap": 0, "self_correction": 1.0}}
    augmenter = TrajectoryAugmenter(config=config)
    
    # Edge case 1: Empty trajectory
    empty_traj = [{
        "task_id": "empty",
        "question": "Test",
        "trajectory": [],
        "final_answer": "Test",
        "metadata": {}
    }]
    
    logger.info("\n1. Testing empty trajectory:")
    try:
        result = augmenter.process(empty_traj)
        logger.info(f"  ✓ Handled empty trajectory: {result[0]['trajectory_type']}")
    except Exception as e:
        logger.error(f"  ✗ Failed on empty trajectory: {e}")
    
    # Edge case 2: Only thoughts, no actions
    thought_only = [{
        "task_id": "thoughts",
        "question": "Test",
        "trajectory": [
            {"type": "thought", "content": "Thinking..."},
            {"type": "thought", "content": "More thinking..."}
        ],
        "final_answer": "Test",
        "metadata": {}
    }]
    
    logger.info("\n2. Testing thought-only trajectory:")
    try:
        result = augmenter.process(thought_only)
        logger.info(f"  ✓ Handled thought-only trajectory: {result[0]['trajectory_type']}")
    except Exception as e:
        logger.error(f"  ✗ Failed on thought-only trajectory: {e}")
    
    # Edge case 3: Unknown action types
    unknown_action = [{
        "task_id": "unknown",
        "question": "Test",
        "trajectory": [
            {"type": "action", "name": "UNKNOWN_ACTION", "parameters": {}}
        ],
        "final_answer": "Test",
        "metadata": {}
    }]
    
    logger.info("\n3. Testing unknown action type:")
    try:
        result = augmenter.process(unknown_action)
        logger.info(f"  ✓ Handled unknown action: {result[0]['trajectory_type']}")
        if 'augmentation_failure' in result[0].get('metadata', {}):
            logger.info(f"    Failure reason: {result[0]['metadata']['augmentation_failure']}")
    except Exception as e:
        logger.error(f"  ✗ Failed on unknown action: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Edge case testing complete!")
    logger.info("="*60)


def main():
    """Main test execution."""
    success = True
    
    # Run main pipeline tests
    if not test_augmentation_pipeline():
        success = False
    
    # Run edge case tests
    test_specific_edge_cases()
    
    # Print final summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())