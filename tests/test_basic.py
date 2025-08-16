"""
Basic Test Module

Tests basic functionality without external dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime


def test_visual_operations():
    """Test visual operations registry and execution."""
    print("\n=== Testing Visual Operations ===")
    
    from core.modules.operation_registry import registry
    
    # Check registered operations
    operations = registry.list_operations()
    print(f"Registered operations: {list(operations.keys())}")
    
    # Import operations to trigger registration
    from core.modules.operations import (
        SegmentObjectOperation,
        ReadTextOperation,
        GetPropertiesOperation,
        TrackObjectOperation,
        ZoomInOperation
    )
    
    # Check again after import
    operations = registry.list_operations()
    print(f"After import: {list(operations.keys())}")
    
    # Test SEGMENT_OBJECT_AT
    dummy_image = torch.randn(3, 224, 224)
    result = registry.execute(
        "SEGMENT_OBJECT_AT",
        image=dummy_image,
        point=(112, 112)
    )
    print(f"SEGMENT_OBJECT_AT result keys: {result.keys()}")
    assert 'mask' in result
    assert 'bbox' in result
    
    # Test READ_TEXT
    result = registry.execute(
        "READ_TEXT",
        image=dummy_image,
        region=[50, 50, 150, 100]
    )
    print(f"READ_TEXT result keys: {result.keys()}")
    assert 'text' in result
    assert 'lines' in result
    
    print("✓ Visual operations working correctly")


def test_data_structures():
    """Test core data structures."""
    print("\n=== Testing Data Structures ===")
    
    from core.data_structures import (
        Action, ActionType, Trajectory, Experience,
        UpdateTask, VotingResult, RewardComponents
    )
    
    # Create action
    action = Action(
        type=ActionType.VISUAL_OPERATION,
        operation="SEGMENT_OBJECT_AT",
        arguments={"point": (100, 100)},
        confidence=0.95
    )
    print(f"Action created: {action.operation}")
    
    # Create trajectory
    trajectory = Trajectory()
    trajectory.add_action(action)
    trajectory.add_action(Action(
        type=ActionType.REASONING,
        operation="analyze",
        confidence=0.9
    ))
    print(f"Trajectory length: {trajectory.get_trajectory_length()}")
    
    # Create experience
    experience = Experience(
        experience_id="test_exp_1",
        image_features=torch.randn(3, 224, 224),
        question_text="What is in the image?",
        trajectory=trajectory,
        model_confidence=0.85
    )
    print(f"Experience created: {experience.experience_id}")
    
    # Test serialization
    exp_dict = experience.to_dict()
    exp_restored = Experience.from_dict(exp_dict)
    print(f"Experience serialization successful: {exp_restored.experience_id}")
    
    print("✓ Data structures working correctly")


def test_configuration():
    """Test configuration system."""
    print("\n=== Testing Configuration ===")
    
    from core.config_schema import PixelisConfig
    
    # Create default config
    config = PixelisConfig()
    print(f"Model name: {config.model.model_name}")
    print(f"Training mode: {config.training.mode}")
    print(f"Buffer size: {config.online.buffer_size}")
    
    # Validate config
    config.validate()
    print("✓ Configuration validation passed")
    
    # Convert to dict
    config_dict = config.to_dict()
    print(f"Config sections: {list(config_dict.keys())}")
    
    # Restore from dict
    config_restored = PixelisConfig.from_dict(config_dict)
    print(f"Restored model name: {config_restored.model.model_name}")
    
    print("✓ Configuration system working correctly")


def test_voting_module():
    """Test temporal ensemble voting."""
    print("\n=== Testing Voting Module ===")
    
    from core.modules.voting import TemporalEnsembleVoting
    from core.data_structures import Experience, Trajectory
    
    # Create voting module
    voter = TemporalEnsembleVoting(
        strategy="weighted",
        min_votes_required=2
    )
    print(f"Voter created with strategy: {voter.strategy}")
    
    # Create initial prediction
    initial_pred = {
        'answer': 'cat',
        'confidence': 0.8,
        'trajectory': []
    }
    
    # Create neighbor experiences
    neighbors = []
    for i, answer in enumerate(['cat', 'dog', 'cat']):
        traj = Trajectory()
        traj.final_answer = answer
        
        exp = Experience(
            experience_id=f"neighbor_{i}",
            image_features=torch.randn(3, 224, 224),
            question_text="What animal?",
            trajectory=traj,
            model_confidence=0.7 + i * 0.05
        )
        neighbors.append(exp)
    
    # Perform voting
    result = voter.vote(initial_pred, neighbors)
    print(f"Voting result: answer={result.final_answer}, confidence={result.confidence:.3f}")
    print(f"Consensus strength: {result.get_consensus_strength():.3f}")
    
    print("✓ Voting module working correctly")


def test_dynamics_model():
    """Test dynamics model for curiosity."""
    print("\n=== Testing Dynamics Model ===")
    
    from core.modules.dynamics_model import CuriosityDynamicsModel
    
    # Create model
    model = CuriosityDynamicsModel(
        state_dim=768,
        action_dim=128,
        encoded_dim=256,
        config={'device': 'cpu'}  # Use CPU for testing
    )
    print("Dynamics model created")
    
    # Create dummy data
    state = torch.randn(1, 768)
    action = torch.randn(1, 128)
    next_state = torch.randn(1, 768)
    
    # Compute intrinsic reward
    reward, losses = model.compute_intrinsic_reward(state, action, next_state)
    print(f"Intrinsic reward: {reward.item():.4f}")
    print(f"Forward error: {losses['forward_error'].item():.4f}")
    print(f"Inverse error: {losses['inverse_error'].item():.4f}")
    
    print("✓ Dynamics model working correctly")


def run_all_tests():
    """Run all basic tests."""
    print("=" * 50)
    print("PIXELIS BASIC TESTS")
    print("=" * 50)
    
    try:
        test_visual_operations()
        test_data_structures()
        test_configuration()
        test_voting_module()
        test_dynamics_model()
        
        print("\n" + "=" * 50)
        print("ALL BASIC TESTS PASSED ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)