#!/usr/bin/env python3
"""
Test script to verify the TrajectoryAugmenter instantiation fix.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generation.trajectory_augmenter import TrajectoryAugmenter, Trajectory

def test_augmenter_instantiation():
    """Test that TrajectoryAugmenter can be instantiated with config."""
    print("Testing TrajectoryAugmenter instantiation...")
    
    # Test config
    config = {
        'proportions': {
            'golden': 0.60,
            'trap': 0.20,
            'self_correction': 0.20
        }
    }
    
    try:
        augmenter = TrajectoryAugmenter(config=config)
        print("✓ TrajectoryAugmenter instantiated successfully")
        return augmenter
    except Exception as e:
        print(f"✗ Failed to instantiate TrajectoryAugmenter: {e}")
        return None

def test_augment_methods(augmenter):
    """Test that augment methods work."""
    print("\nTesting augmentation methods...")
    
    # Create a test trajectory
    test_trajectory = Trajectory(
        task_id="test_001",
        question="Test question?",
        actions=[
            {"type": "action", "name": "ZOOM_IN", "parameters": {"bbox": [0, 0, 100, 100]}},
            {"type": "thought", "content": "I can see the details"}
        ],
        final_answer="Test answer",
        trajectory_type="golden",
        metadata={}
    )
    
    # Test augment_trajectory (self-correction)
    try:
        sc_result = augmenter.augment_trajectory(test_trajectory)
        print(f"✓ augment_trajectory works - result type: {sc_result.trajectory_type}")
    except Exception as e:
        print(f"✗ augment_trajectory failed: {e}")
    
    # Test augment_as_trap
    try:
        trap_result = augmenter.augment_as_trap(test_trajectory)
        print(f"✓ augment_as_trap works - result type: {trap_result.trajectory_type}")
    except Exception as e:
        print(f"✗ augment_as_trap failed: {e}")
    
    # Test process method with batch
    try:
        batch_input = [
            {
                'task_id': 'test_002',
                'question': 'Batch test?',
                'trajectory': [
                    {"type": "action", "name": "SEGMENT_OBJECT_AT", "parameters": {"x": 50, "y": 50}}
                ],
                'final_answer': 'Batch answer',
                'metadata': {}
            }
        ]
        batch_result = augmenter.process(batch_input)
        print(f"✓ process method works - returned {len(batch_result)} samples")
    except Exception as e:
        print(f"✗ process method failed: {e}")

def main():
    print("="*60)
    print("Testing TrajectoryAugmenter Fix")
    print("="*60)
    
    # Test instantiation
    augmenter = test_augmenter_instantiation()
    
    if augmenter:
        # Test methods
        test_augment_methods(augmenter)
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    main()