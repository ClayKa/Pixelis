"""
Unit Tests for TrajectoryAugmenter
===================================
Test suite for the trajectory augmentation module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation.trajectory_augmenter import (
    TrajectoryAugmenter,
    Trajectory,
    DistractorAction
)


class TestTrajectoryAugmenter:
    """Test suite for TrajectoryAugmenter."""
    
    @pytest.fixture
    def golden_trajectory(self):
        """Create a mock golden trajectory."""
        return Trajectory(
            task_id="test_001",
            question="What is the largest object in the image?",
            actions=[
                {
                    "action": "SEGMENT_OBJECT_AT",
                    "parameters": {"x": 100, "y": 100},
                    "observation": "Found object: car"
                },
                {
                    "action": "GET_PROPERTIES",
                    "parameters": {"object_id": 1},
                    "observation": "Object properties: size=large"
                }
            ],
            final_answer="The car is the largest object",
            trajectory_type="golden",
            metadata={"source": "test"}
        )
    
    @pytest.fixture
    def augmenter_with_mock_llm(self):
        """Create augmenter with mocked LLM client."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "That doesn't seem right. Let me try a different approach."
        return TrajectoryAugmenter(llm_client=mock_llm)
    
    @pytest.fixture
    def augmenter_without_llm(self):
        """Create augmenter without LLM (uses templates)."""
        return TrajectoryAugmenter(llm_client=None)
    
    def test_self_correction_augmentation_with_templates(self, augmenter_without_llm, golden_trajectory):
        """Test self-correction augmentation using template responses."""
        # Augment the trajectory
        augmented = augmenter_without_llm.augment_trajectory(golden_trajectory)
        
        # Assert trajectory is longer
        assert len(augmented.actions) > len(golden_trajectory.actions)
        
        # Assert first action is a distractor
        first_action = augmented.actions[0]
        assert 'action' in first_action
        assert 'observation' in first_action
        
        # Assert corrective thought is present
        second_item = augmented.actions[1]
        assert 'thought' in second_item
        assert second_item['type'] == 'self_correction'
        
        # Assert original trajectory follows
        original_start_idx = 2
        for i, original_action in enumerate(golden_trajectory.actions):
            assert augmented.actions[original_start_idx + i] == original_action
        
        # Assert metadata is correct
        assert augmented.trajectory_type == "self_correction"
        assert augmented.metadata['original_trajectory_id'] == golden_trajectory.task_id
        assert 'distractor_type' in augmented.metadata
        assert augmented.metadata['augmentation_method'] == 'self_correction'
    
    def test_self_correction_with_mock_llm(self, augmenter_with_mock_llm, golden_trajectory):
        """Test self-correction with mocked LLM response."""
        # Mock the LLM to return a specific correction
        expected_thought = "The coordinates were incorrect. Let me check a different location."
        augmenter_with_mock_llm.llm_client.generate.return_value = expected_thought
        
        # Augment the trajectory
        augmented = augmenter_with_mock_llm.augment_trajectory(golden_trajectory)
        
        # The corrective thought should be from the mock
        thought_action = augmented.actions[1]
        assert thought_action['thought'] == expected_thought
    
    def test_handles_empty_trajectory(self, augmenter_without_llm):
        """Test handling of empty trajectory."""
        empty_trajectory = Trajectory(
            task_id="empty_001",
            question="Test question",
            actions=[],
            final_answer="Test answer",
            trajectory_type="golden"
        )
        
        # Should handle gracefully
        augmented = augmenter_without_llm.augment_trajectory(empty_trajectory)
        
        # Should return original or minimally modified
        assert augmented.task_id == "empty_001_sc"
        assert len(augmented.actions) >= 2  # At least distractor + thought
    
    def test_handles_non_golden_trajectory(self, augmenter_without_llm):
        """Test that non-golden trajectories are not augmented."""
        trap_trajectory = Trajectory(
            task_id="trap_001",
            question="Test question",
            actions=[{"action": "TEST"}],
            final_answer="Test answer",
            trajectory_type="trap"  # Not golden
        )
        
        # Should return original
        result = augmenter_without_llm.augment_trajectory(trap_trajectory)
        assert result == trap_trajectory
    
    def test_batch_augmentation(self, augmenter_without_llm):
        """Test batch augmentation of multiple trajectories."""
        trajectories = [
            Trajectory(
                task_id=f"batch_{i}",
                question=f"Question {i}",
                actions=[
                    {
                        "action": "SEGMENT_OBJECT_AT",
                        "parameters": {"x": i*10, "y": i*10},
                        "observation": f"Found object {i}"
                    }
                ],
                final_answer=f"Answer {i}",
                trajectory_type="golden"
            )
            for i in range(10)
        ]
        
        # Add some non-golden trajectories
        trajectories.append(
            Trajectory(
                task_id="trap_batch",
                question="Trap question",
                actions=[{"action": "TRAP"}],
                final_answer="Trap answer",
                trajectory_type="trap"
            )
        )
        
        # Batch augment with 20% ratio
        augmented_all, stats = augmenter_without_llm.batch_augment(
            trajectories,
            augmentation_ratio=0.2
        )
        
        # Check statistics
        assert stats['total_input'] == 11
        assert stats['golden_count'] == 10
        assert stats['augmented_count'] == 2  # 20% of 10
        
        # Check total count
        assert len(augmented_all) == 13  # 11 original + 2 augmented
        
        # Check that augmented trajectories are present
        self_correction_count = sum(
            1 for t in augmented_all 
            if t.trajectory_type == "self_correction"
        )
        assert self_correction_count == 2
    
    def test_distractor_selection(self, augmenter_without_llm):
        """Test that appropriate distractors are selected."""
        # Create trajectory with specific action
        trajectory = Trajectory(
            task_id="distractor_test",
            question="Test",
            actions=[
                {
                    "action": "READ_TEXT",
                    "parameters": {"region": [10, 10, 100, 100]},
                    "observation": "Text found"
                }
            ],
            final_answer="Answer",
            trajectory_type="golden"
        )
        
        # Augment
        augmented = augmenter_without_llm.augment_trajectory(trajectory)
        
        # First action should be a READ_TEXT distractor
        first_action = augmented.actions[0]
        assert first_action['action'] == "READ_TEXT"
        assert 'No text detected' in first_action['observation'] or 'Error' in first_action['observation']
    
    def test_custom_distractor_action(self, augmenter_without_llm, golden_trajectory):
        """Test augmentation with custom distractor action."""
        custom_distractor = DistractorAction(
            action_type="CUSTOM_ACTION",
            parameters={"test": "param"},
            observation="Custom error occurred",
            error_type="custom_error"
        )
        
        # Augment with custom distractor
        augmented = augmenter_without_llm.augment_trajectory(
            golden_trajectory,
            distractor_action=custom_distractor
        )
        
        # First action should be the custom distractor
        first_action = augmented.actions[0]
        assert first_action['action'] == "CUSTOM_ACTION"
        assert first_action['parameters'] == {"test": "param"}
        assert first_action['observation'] == "Custom error occurred"
        
        # Metadata should reflect custom error type
        assert augmented.metadata['distractor_type'] == "custom_error"
    
    def test_corrective_thought_templates(self, augmenter_without_llm):
        """Test that corrective thoughts use appropriate templates."""
        # Test each error type
        error_types = [
            "invalid_coordinates",
            "wrong_location", 
            "wrong_region",
            "wrong_parameter",
            "invalid_id"
        ]
        
        for error_type in error_types:
            distractor = DistractorAction(
                action_type="TEST_ACTION",
                parameters={},
                observation="Test error",
                error_type=error_type
            )
            
            thought = augmenter_without_llm._generate_corrective_thought(distractor)
            
            # Should return a relevant template
            assert isinstance(thought, str)
            assert len(thought) > 0
            
            # For known error types, should use specific templates
            if error_type in augmenter_without_llm.correction_templates:
                possible_templates = augmenter_without_llm.correction_templates[error_type]
                # The thought should be one of the templates or the fallback
                assert (thought in possible_templates or 
                       thought == "That didn't work as expected. Let me try a different approach.")
    
    def test_augmentation_preserves_final_answer(self, augmenter_without_llm, golden_trajectory):
        """Test that augmentation preserves the final answer."""
        augmented = augmenter_without_llm.augment_trajectory(golden_trajectory)
        
        # Final answer should be preserved
        assert augmented.final_answer == golden_trajectory.final_answer
        
        # Question should be preserved
        assert augmented.question == golden_trajectory.question
    
    def test_augmentation_id_modification(self, augmenter_without_llm, golden_trajectory):
        """Test that augmented trajectory gets modified ID."""
        augmented = augmenter_without_llm.augment_trajectory(golden_trajectory)
        
        # ID should be modified
        assert augmented.task_id == f"{golden_trajectory.task_id}_sc"
        
        # Original ID should be in metadata
        assert augmented.metadata['original_trajectory_id'] == golden_trajectory.task_id


class TestTrajectoryIO:
    """Test trajectory loading and saving functions."""
    
    def test_load_save_round_trip(self, tmp_path):
        """Test that trajectories can be saved and loaded."""
        from core.data_generation.trajectory_augmenter import (
            save_trajectories_to_file,
            load_trajectories_from_file
        )
        
        # Create test trajectories
        trajectories = [
            Trajectory(
                task_id=f"io_test_{i}",
                question=f"Question {i}",
                actions=[{"action": f"ACTION_{i}"}],
                final_answer=f"Answer {i}",
                trajectory_type="golden",
                metadata={"index": i}
            )
            for i in range(3)
        ]
        
        # Save to file
        test_file = tmp_path / "test_trajectories.jsonl"
        save_trajectories_to_file(trajectories, test_file)
        
        # Load from file
        loaded = load_trajectories_from_file(test_file)
        
        # Should match original
        assert len(loaded) == len(trajectories)
        for original, loaded_traj in zip(trajectories, loaded):
            assert loaded_traj.task_id == original.task_id
            assert loaded_traj.question == original.question
            assert loaded_traj.actions == original.actions
            assert loaded_traj.final_answer == original.final_answer
            assert loaded_traj.trajectory_type == original.trajectory_type
            assert loaded_traj.metadata == original.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])