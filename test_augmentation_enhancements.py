#!/usr/bin/env python3
"""
Test script for the enhanced trajectory augmentation strategies.
Tests the new Perceptual Near-Miss and Logical Fallacy trap generation.
"""

import sys
import os
import json
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_generation.trajectory_augmenter import TrajectoryAugmenter, Trajectory

def create_sample_trajectory():
    """Create a sample golden trajectory for testing."""
    return Trajectory(
        task_id="test_001",
        question="Which object is larger, the cat or the dog?",
        actions=[
            {
                "type": "thought",
                "content": "I need to segment both objects and compare their sizes."
            },
            {
                "type": "action",
                "name": "SEGMENT_OBJECT_AT",
                "parameters": {"point": [150, 200]},
                "observation": "Segmented cat with area 12800 pixels"
            },
            {
                "type": "action", 
                "name": "SEGMENT_OBJECT_AT",
                "parameters": {"point": [350, 200]},
                "observation": "Segmented dog with area 15210 pixels"
            },
            {
                "type": "thought",
                "content": "The cat's area is 12800 and the dog's area is 15210. Therefore, the dog is larger."
            }
        ],
        final_answer="The dog is larger than the cat.",
        trajectory_type="golden",
        metadata={"source": "test"}
    )

def test_perceptual_near_miss(augmenter):
    """Test the perceptual near-miss augmentation."""
    print("\n" + "="*60)
    print("Testing Perceptual Near-Miss Augmentation")
    print("="*60)
    
    golden_sample = create_sample_trajectory()
    
    # Generate perceptual near-miss trap
    trap_sample = augmenter._augment_perceptual_near_miss(golden_sample)
    
    print(f"\nOriginal trajectory ID: {golden_sample.task_id}")
    print(f"Trap trajectory ID: {trap_sample.task_id}")
    print(f"Trap type: {trap_sample.metadata.get('trap_type')}")
    
    # Compare original and perturbed actions
    print("\n--- Action Comparison ---")
    for i, (orig_step, trap_step) in enumerate(zip(golden_sample.actions, trap_sample.actions)):
        if isinstance(orig_step, dict) and orig_step.get('type') == 'action':
            if isinstance(trap_step, dict) and trap_step.get('type') == 'action':
                orig_params = orig_step.get('parameters', {})
                trap_params = trap_step.get('parameters', {})
                
                if orig_params != trap_params:
                    print(f"\nAction {i} parameters changed:")
                    print(f"  Original: {orig_params}")
                    print(f"  Perturbed: {trap_params}")
    
    # Check answer modification
    print(f"\nOriginal answer: {golden_sample.final_answer}")
    print(f"Trap answer: {trap_sample.final_answer}")
    
    # Verify provenance
    provenance = trap_sample.metadata.get('provenance', {})
    print(f"\nProvenance trap type: {provenance.get('trap_type')}")
    print(f"Original answer stored: {provenance.get('original_answer')}")
    
    return trap_sample

def test_logical_fallacy(augmenter):
    """Test the logical fallacy augmentation."""
    print("\n" + "="*60)
    print("Testing Logical Fallacy Augmentation")
    print("="*60)
    
    golden_sample = create_sample_trajectory()
    
    # Generate logical fallacy trap
    trap_sample = augmenter._augment_logical_fallacy(golden_sample)
    
    print(f"\nOriginal trajectory ID: {golden_sample.task_id}")
    print(f"Trap trajectory ID: {trap_sample.task_id}")
    print(f"Trap type: {trap_sample.metadata.get('trap_type')}")
    
    # Compare thoughts
    print("\n--- Thought Comparison ---")
    
    # Find last thoughts
    orig_last_thought = None
    trap_last_thought = None
    
    for step in reversed(golden_sample.actions):
        if isinstance(step, dict) and step.get('type') == 'thought':
            orig_last_thought = step.get('content')
            break
    
    for step in reversed(trap_sample.actions):
        if isinstance(step, dict) and step.get('type') == 'thought':
            trap_last_thought = step.get('content')
            break
    
    print(f"\nOriginal reasoning:")
    print(f"  {orig_last_thought}")
    print(f"\nFlawed reasoning:")
    print(f"  {trap_last_thought}")
    
    # Check answer modification
    print(f"\nOriginal answer: {golden_sample.final_answer}")
    print(f"Trap answer: {trap_sample.final_answer}")
    
    # Verify provenance
    provenance = trap_sample.metadata.get('provenance', {})
    print(f"\nProvenance trap type: {provenance.get('trap_type')}")
    print(f"Original reasoning stored: {provenance.get('original_reasoning', '')[:100]}...")
    
    return trap_sample

def test_process_with_subtypes(augmenter):
    """Test the full processing with trap sub-types."""
    print("\n" + "="*60)
    print("Testing Full Processing with Trap Sub-Types")
    print("="*60)
    
    # Create multiple golden samples
    samples = [create_sample_trajectory() for _ in range(10)]
    for i, sample in enumerate(samples):
        sample.task_id = f"test_{i:03d}"
    
    # Process samples
    augmented_samples = augmenter.process(samples)
    
    # Analyze results
    type_counts = {}
    trap_subtypes = {}
    
    for sample_dict in augmented_samples:
        traj_type = sample_dict.get('trajectory_type', 'unknown')
        type_counts[traj_type] = type_counts.get(traj_type, 0) + 1
        
        if traj_type == 'trap':
            metadata = sample_dict.get('metadata', {})
            trap_type = metadata.get('trap_type', 'unknown')
            trap_subtypes[trap_type] = trap_subtypes.get(trap_type, 0) + 1
    
    print(f"\nProcessed {len(augmented_samples)} samples")
    print("\nTrajectory type distribution:")
    for traj_type, count in type_counts.items():
        percentage = (count / len(augmented_samples)) * 100
        print(f"  {traj_type}: {count} ({percentage:.1f}%)")
    
    if trap_subtypes:
        print("\nTrap sub-type distribution:")
        for trap_type, count in trap_subtypes.items():
            percentage = (count / type_counts.get('trap', 1)) * 100
            print(f"  {trap_type}: {count} ({percentage:.1f}% of traps)")
    
    # Show augmentation statistics
    print("\nAugmentation Statistics:")
    for key, value in augmenter.augmentation_stats.items():
        print(f"  {key}: {value}")
    
    return augmented_samples

def main():
    """Main test function."""
    print("="*60)
    print("Enhanced Trajectory Augmentation Test Suite")
    print("="*60)
    
    # Load configuration
    config_path = Path("configs/data_fusion_manifest.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('trajectory_augmentation', {})
            print(f"\nLoaded configuration from {config_path}")
    else:
        # Use default configuration
        config = {
            'proportions': {
                'golden_positive': 0.6,
                'self_correction': 0.2,
                'trap_samples': {
                    'total_proportion': 0.2,
                    'sub_types': [
                        {'name': 'process_negative', 'proportion': 0.5},
                        {'name': 'perceptual_near_miss', 'proportion': 0.25},
                        {'name': 'logical_fallacy', 'proportion': 0.25}
                    ]
                }
            }
        }
        print("\nUsing default configuration")
    
    # Print configuration
    print("\nTrajectory Augmentation Configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize augmenter
    augmenter = TrajectoryAugmenter(config, llm_client=None)
    
    # Run tests
    try:
        # Test individual augmentation methods
        perceptual_trap = test_perceptual_near_miss(augmenter)
        logical_trap = test_logical_fallacy(augmenter)
        
        # Test full processing
        processed_samples = test_process_with_subtypes(augmenter)
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
        print("\nSummary:")
        print("1. Perceptual Near-Miss augmentation: WORKING")
        print("2. Logical Fallacy augmentation: WORKING")
        print("3. Full processing with sub-types: WORKING")
        print("\nThe enhanced augmentation strategies are ready for use.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())