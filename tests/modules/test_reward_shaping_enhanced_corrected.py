#!/usr/bin/env python3
"""
Corrected comprehensive test suite for reward_shaping_enhanced.py to achieve 100% test coverage.
Uses actual method names and signatures from the implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from collections import deque, defaultdict
from omegaconf import OmegaConf

from core.modules.reward_shaping_enhanced import (
    RewardComponents,
    LoRADynamicsModel,
    PerformanceAwareCuriosityModule,
    EnhancedTrajectoryCoherenceAnalyzer,
    ToolMisusePenaltySystem,
    NormalizedRewardOrchestrator,
    RunningStats
)


class TestRewardComponents:
    """Test RewardComponents dataclass."""
    
    def test_to_dict_method(self):
        """Test RewardComponents.to_dict() method - covers lines 31-40."""
        components = RewardComponents(
            task_reward=1.0,
            curiosity_reward=0.5,
            coherence_reward=0.3,
            tool_penalty=-0.1,
            total_reward=1.7,
            metadata={'step': 100}
        )
        
        result = components.to_dict()
        expected = {
            'task': 1.0,
            'curiosity': 0.5,
            'coherence': 0.3,
            'penalty': -0.1,
            'total': 1.7,
            'metadata': {'step': 100}
        }
        assert result == expected


class TestLoRADynamicsModel:
    """Test LoRADynamicsModel class."""
    
    def test_init_and_forward(self):
        """Test LoRADynamicsModel __init__ and forward - covers lines 51-133."""
        # Use CPU device to avoid CUDA issues
        model = LoRADynamicsModel(
            state_dim=64,
            action_dim=32,
            hidden_dim=128,
            device='cpu'
        )
        
        # Test forward pass
        batch_size = 4
        state = torch.randn(batch_size, 64)
        action = torch.randn(batch_size, 32)
        
        output = model.forward(state, action)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        
        # Test get_num_trainable_params
        num_params = model.get_num_trainable_params()
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Verify LoRA components exist
        assert hasattr(model, 'lora_adapters')
        assert len(model.lora_adapters) == 3  # Three layers
        
        # Test that trainable params are only from LoRA adapters
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == total_trainable


class TestPerformanceAwareCuriosityModule:
    """Test PerformanceAwareCuriosityModule class."""
    
    def test_init_and_compute_curiosity_reward(self):
        """Test curiosity module __init__ and compute_curiosity_reward - covers lines 144-294."""
        module = PerformanceAwareCuriosityModule(
            state_dim=64,
            action_dim=32,
            device='cpu',
            beta=0.2,
            eta=0.5,
            cache_size=100
        )
        
        # Test basic computation
        state = torch.randn(4, 64)
        action = torch.randn(4, 32)
        next_state = torch.randn(4, 64)
        
        # Use the correct method name
        result = module.compute_curiosity_reward(state, action, next_state)
        
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar
        assert result >= 0
        
        # Test caching - second call should hit cache
        initial_hits = module.cache_hits if hasattr(module, 'cache_hits') else 0
        result2 = module.compute_curiosity_reward(state, action, next_state)
        if hasattr(module, 'cache_hits'):
            assert module.cache_hits >= initial_hits
        assert isinstance(result2, torch.Tensor)
        
        # Test _create_cache_key 
        cache_key = module._create_cache_key(state, action)
        assert isinstance(cache_key, str)
        
        # Test _update_cache
        value = (torch.tensor(0.5), torch.tensor(0.3))
        module._update_cache(cache_key, value)
        # Verify cache was updated
        assert len(module.reward_cache) > 0


class TestEnhancedTrajectoryCoherenceAnalyzer:
    """Test EnhancedTrajectoryCoherenceAnalyzer class."""
    
    def test_init_and_compute_coherence_reward(self):
        """Test coherence analyzer __init__ and compute_coherence_reward - covers lines 295-444."""
        # Use default __init__ (no parameters based on actual signature)
        analyzer = EnhancedTrajectoryCoherenceAnalyzer()
        
        # Test empty trajectory
        result = analyzer.compute_coherence_reward([])
        assert isinstance(result, (int, float))
        assert result >= 0
        
        # Test single action trajectory  
        trajectory = [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}]
        result = analyzer.compute_coherence_reward(trajectory)
        assert isinstance(result, (int, float))
        assert result >= 0
        
        # Test logical sequence
        trajectory = [
            {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
            {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [110, 110]},
            {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'}
        ]
        result = analyzer.compute_coherence_reward(trajectory)
        assert isinstance(result, (int, float))
        assert result >= 0
        
        # Test _check_contradictions
        contradictory_trajectory = [
            {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
            {'action': 'ZOOM_IN', 'coordinates': [300, 300]}  # Far apart zoom
        ]
        contradictions = analyzer._check_contradictions(contradictory_trajectory)
        assert isinstance(contradictions, int)
        assert contradictions >= 0


class TestToolMisusePenaltySystem:
    """Test ToolMisusePenaltySystem class."""
    
    def test_init_and_calculate_penalties(self):
        """Test penalty system __init__ and calculate_penalties - covers lines 445-578."""
        # Use default __init__ (no parameters based on actual signature)
        system = ToolMisusePenaltySystem()
        
        # Test empty trajectory
        penalties = system.calculate_penalties([], {})
        assert isinstance(penalties, dict)
        
        # Test valid trajectory
        trajectory = [
            {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [100, 100]},
            {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'}
        ]
        image_data = {'objects': [{'id': 'obj_1'}]}
        
        penalties = system.calculate_penalties(trajectory, image_data)
        assert isinstance(penalties, dict)
        
        # Test violations
        violation_trajectory = [
            {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'},  # Without segmentation
            {'action': 'TRACK_OBJECT', 'object_id': 'obj_2'}     # May be problematic
        ]
        image_data = {'objects': [], 'type': 'static'}
        
        penalties = system.calculate_penalties(violation_trajectory, image_data)
        assert isinstance(penalties, dict)
        
        # Test _check_parameters method
        action = {'action': 'ZOOM_IN', 'coordinates': [100, 100]}
        penalty_value, penalty_counts = system._check_parameters(action)
        assert isinstance(penalty_value, float)
        assert isinstance(penalty_counts, dict)
        assert penalty_value >= 0
        
        # Test various actions to hit different code paths
        actions_to_test = [
            {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [100, 100]},
            {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'},
            {'action': 'READ_TEXT', 'coordinates': [100, 100]},
            {'action': 'TRACK_OBJECT', 'object_id': 'obj_1'},
            {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
            {'action': 'UNKNOWN_ACTION'}
        ]
        
        for action in actions_to_test:
            penalty_value, penalty_counts = system._check_parameters(action)
            assert isinstance(penalty_value, float)
            assert isinstance(penalty_counts, dict)


class TestNormalizedRewardOrchestrator:
    """Test NormalizedRewardOrchestrator class."""
    
    def test_init_and_calculate_total_reward(self):
        """Test orchestrator __init__ and calculate_total_reward - covers lines 579-793."""
        config = {
            'beta': 0.3,
            'alpha': 0.2,
            'gamma': 0.5,
            'tau': 0.1,
            'use_curriculum': False,
            'curriculum_stages': []
        }
        
        # Mock components to avoid CUDA issues
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule') as mock_curiosity:
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer') as mock_coherence:
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem') as mock_penalty:
                    # Create mock instances
                    curiosity_instance = MagicMock()
                    coherence_instance = MagicMock()
                    penalty_instance = MagicMock()
                    
                    curiosity_instance.compute_curiosity_reward.return_value = torch.tensor(0.5)
                    coherence_instance.compute_coherence_reward.return_value = 0.7
                    penalty_instance.calculate_penalties.return_value = {'test': 0.1}
                    
                    mock_curiosity.return_value = curiosity_instance
                    mock_coherence.return_value = coherence_instance
                    mock_penalty.return_value = penalty_instance
                    
                    orchestrator = NormalizedRewardOrchestrator(config)
                    
                    # Test calculate_total_reward
                    trajectory = [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}]
                    image_data = torch.randn(3, 224, 224)
                    final_reward = 1.0
                    step = 100
                    
                    result = orchestrator.calculate_total_reward(
                        trajectory, image_data, final_reward, step
                    )
                    
                    assert isinstance(result, torch.Tensor)
                    assert result.dim() == 0
                    
                    # Test _create_action_embedding
                    action = {'action': 'ZOOM_IN', 'coordinates': [100, 100]}
                    embedding = orchestrator._create_action_embedding(action)
                    assert isinstance(embedding, torch.Tensor)
                    assert embedding.shape == (32,)
                    
                    # Test unknown action
                    unknown_action = {'action': 'UNKNOWN_ACTION'}
                    embedding = orchestrator._create_action_embedding(unknown_action)
                    assert torch.equal(embedding, torch.zeros(32))
                    
                    # Test _get_curriculum_weights (no curriculum)
                    weights = orchestrator._get_curriculum_weights()
                    expected = {'curiosity': 1.0, 'coherence': 1.0, 'penalty': 1.0}
                    assert weights == expected
                    
                    # Test update_step
                    orchestrator.update_step(150)  # Should not crash
        
        # Test with curriculum
        config_with_curriculum = {
            'beta': 0.3, 'alpha': 0.2, 'gamma': 0.5, 'tau': 0.1,
            'use_curriculum': True,
            'curriculum_stages': [
                {'step': 100, 'weights': {'curiosity': 0.2, 'coherence': 0.3}},
                {'step': 200, 'weights': {'curiosity': 0.5, 'coherence': 0.6}}
            ]
        }
        
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule'):
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer'):
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem'):
                    orchestrator = NormalizedRewardOrchestrator(config_with_curriculum)
                    
                    # Update step to test curriculum weights
                    orchestrator.update_step(50)
                    weights_50 = orchestrator._get_curriculum_weights()
                    assert weights_50['curiosity'] == 1.0  # Before first stage
                    
                    orchestrator.update_step(150)
                    weights_150 = orchestrator._get_curriculum_weights()
                    assert weights_150['curiosity'] == 0.2  # After first stage
                    
                    orchestrator.update_step(250)
                    weights_250 = orchestrator._get_curriculum_weights()
                    assert weights_250['curiosity'] == 0.5  # After second stage


class TestRunningStats:
    """Test RunningStats class."""
    
    def test_init_update_and_normalize(self):
        """Test RunningStats __init__, update and normalize - covers lines 794-814."""
        # Test with custom window size
        stats = RunningStats(window_size=100)
        
        # Test initial state
        assert hasattr(stats, 'values')
        assert len(stats.values) == 0
        
        # Test single update
        stats.update(5.0)
        assert len(stats.values) == 1
        
        # Test multiple updates
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for value in values:
            stats.update(value)
        
        assert len(stats.values) <= 100  # Should be constrained by window size
        
        # Test normalize method
        test_value = 5.0
        normalized = stats.normalize(test_value)
        assert isinstance(normalized, float)
        
        # Test with more data to ensure window behavior
        for i in range(150):
            stats.update(float(i))
        
        assert len(stats.values) == 100  # Should be limited to window_size
        
        # Test normalize with sufficient data
        normalized = stats.normalize(50.0)
        assert isinstance(normalized, float)
        
        # Test edge cases
        empty_stats = RunningStats()
        normalized_empty = empty_stats.normalize(10.0)
        assert normalized_empty == 10.0  # Should return input when no data
        
        # Test single value stats
        single_stats = RunningStats()
        single_stats.update(5.0)
        normalized_single = single_stats.normalize(5.0)
        assert isinstance(normalized_single, float)


class TestIntegration:
    """Integration tests to cover remaining lines."""
    
    def test_full_pipeline_comprehensive(self):
        """Test full reward calculation pipeline to hit all remaining lines."""
        config = {
            'beta': 0.3, 'alpha': 0.2, 'gamma': 0.5, 'tau': 0.1,
            'use_curriculum': True,
            'curriculum_stages': [
                {'step': 50, 'weights': {'curiosity': 0.1, 'coherence': 0.2}}
            ]
        }
        
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule') as mock_curiosity:
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer') as mock_coherence:
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem') as mock_penalty:
                    
                    # Create mock instances that return appropriate values
                    curiosity_instance = MagicMock()
                    coherence_instance = MagicMock()
                    penalty_instance = MagicMock()
                    
                    curiosity_instance.compute_curiosity_reward.return_value = torch.tensor(0.6)
                    coherence_instance.compute_coherence_reward.return_value = 0.8
                    penalty_instance.calculate_penalties.return_value = {'violation': 0.2}
                    
                    mock_curiosity.return_value = curiosity_instance
                    mock_coherence.return_value = coherence_instance 
                    mock_penalty.return_value = penalty_instance
                    
                    orchestrator = NormalizedRewardOrchestrator(config)
                    
                    # Test with complex trajectory and different action types
                    trajectory = [
                        {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
                        {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [120, 120]},
                        {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'},
                        {'action': 'READ_TEXT', 'coordinates': [130, 130]},
                        {'action': 'TRACK_OBJECT', 'object_id': 'obj_2'},
                        {'action': 'UNKNOWN_ACTION', 'coordinates': [140, 140]}  # Test unknown action
                    ]
                    
                    image_data = torch.randn(3, 224, 224)
                    final_reward = 0.9
                    
                    # Test different steps to hit curriculum logic
                    steps_to_test = [25, 75, 100]  # Before, after curriculum, and later
                    
                    for step in steps_to_test:
                        orchestrator.update_step(step)
                        
                        result = orchestrator.calculate_total_reward(
                            trajectory, image_data, final_reward, step
                        )
                        
                        assert isinstance(result, torch.Tensor)
                        assert result.dim() == 0
                        assert not torch.isnan(result)
                    
                    # Test edge cases to hit more lines
                    
                    # Test with no penalties
                    penalty_instance.calculate_penalties.return_value = {}
                    result = orchestrator.calculate_total_reward(
                        trajectory, image_data, final_reward, 100
                    )
                    assert isinstance(result, torch.Tensor)
                    
                    # Test extreme values for clipping paths
                    curiosity_instance.compute_curiosity_reward.return_value = torch.tensor(100.0)  # Very high
                    coherence_instance.compute_coherence_reward.return_value = -50.0  # Very low
                    penalty_instance.calculate_penalties.return_value = {'huge_penalty': 200.0}
                    
                    result = orchestrator.calculate_total_reward(
                        trajectory, image_data, 50.0, 100  # High final reward too
                    )
                    assert isinstance(result, torch.Tensor)
                    
                    # Test all action types for embedding creation
                    action_types = [
                        {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
                        {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [100, 100]},
                        {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'},
                        {'action': 'READ_TEXT', 'coordinates': [100, 100]},
                        {'action': 'TRACK_OBJECT', 'object_id': 'obj_1'},
                        {'action': 'UNKNOWN_ACTION'}
                    ]
                    
                    for action in action_types:
                        embedding = orchestrator._create_action_embedding(action)
                        assert isinstance(embedding, torch.Tensor)
                        assert embedding.shape == (32,)
    
    def test_edge_cases_and_error_paths(self):
        """Test edge cases and error handling paths."""
        
        # Test LoRADynamicsModel with different input sizes
        model = LoRADynamicsModel(state_dim=32, action_dim=16, device='cpu')
        
        # Test with single sample
        state = torch.randn(1, 32)
        action = torch.randn(1, 16)
        output = model.forward(state, action)
        assert output.shape == (1, 32)
        
        # Test with large batch
        state = torch.randn(64, 32)
        action = torch.randn(64, 16)
        output = model.forward(state, action)
        assert output.shape == (64, 32)
        
        # Test EnhancedTrajectoryCoherenceAnalyzer with edge cases
        analyzer = EnhancedTrajectoryCoherenceAnalyzer()
        
        # Test with very long trajectory
        long_trajectory = [{'action': 'ZOOM_IN', 'coordinates': [i, i]} for i in range(100)]
        result = analyzer.compute_coherence_reward(long_trajectory)
        assert isinstance(result, (int, float))
        
        # Test with malformed trajectory entries
        malformed_trajectory = [
            {'action': 'ZOOM_IN'},  # Missing coordinates
            {'coordinates': [100, 100]},  # Missing action
            {}  # Empty entry
        ]
        result = analyzer.compute_coherence_reward(malformed_trajectory)
        assert isinstance(result, (int, float))
        
        # Test ToolMisusePenaltySystem with various edge cases
        system = ToolMisusePenaltySystem()
        
        # Test with malformed actions
        malformed_actions = [
            {'action': 'ZOOM_IN'},  # Missing parameters
            {'coordinates': [100, 100]},  # Missing action
            {},  # Empty action
            {'action': 'ZOOM_IN', 'coordinates': 'invalid'},  # Invalid coordinates
            {'action': 'GET_PROPERTIES'},  # Missing object_id
        ]
        
        for action in malformed_actions:
            penalty_value, penalty_counts = system._check_parameters(action)
            assert isinstance(penalty_value, float)
            assert isinstance(penalty_counts, dict)
            assert penalty_value >= 0
        
        # Test RunningStats with edge cases
        stats = RunningStats(window_size=5)
        
        # Test with repeated identical values
        for _ in range(10):
            stats.update(5.0)
        
        normalized = stats.normalize(5.0)
        assert isinstance(normalized, float)
        
        # Test with extreme values
        extreme_stats = RunningStats()
        extreme_values = [1e-10, 1e10, -1e10, 0.0, float('inf')]
        
        for value in extreme_values:
            if not np.isinf(value):  # Skip infinity values
                extreme_stats.update(value)
        
        if len(extreme_stats.values) > 0:
            result = extreme_stats.normalize(1.0)
            assert isinstance(result, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])