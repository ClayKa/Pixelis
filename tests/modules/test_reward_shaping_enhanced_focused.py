#!/usr/bin/env python3
"""
Focused test suite for reward_shaping_enhanced.py targeting specific uncovered lines.
Based on the actual implementation structure and focusing on achieving 100% coverage.
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
        """Test RewardComponents.to_dict() method - covers lines 33-40."""
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
        
        # Test that base layers are frozen
        for param in model.base_layers.parameters():
            assert param.requires_grad == False
        
        # Test that LoRA adapters are trainable
        for param in model.lora_adapters.parameters():
            assert param.requires_grad == True
        
        # Test get_num_trainable_params
        num_params = model.get_num_trainable_params()
        assert isinstance(num_params, int)
        assert num_params > 0


class TestPerformanceAwareCuriosityModule:
    """Test PerformanceAwareCuriosityModule class."""
    
    def test_init_and_compute(self):
        """Test curiosity module __init__ and compute_curiosity - covers lines 144-211."""
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
        
        result = module.compute_curiosity(state, action, next_state)
        
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar
        assert result >= 0
        
        # Test caching - second call should hit cache
        initial_hits = module.cache_hits
        result2 = module.compute_curiosity(state, action, next_state)
        assert module.cache_hits == initial_hits + 1
        assert torch.equal(result, result2)
        
        # Test _create_cache_key
        cache_key = module._create_cache_key(state, action, next_state)
        assert isinstance(cache_key, str)
        
        # Test _update_cache
        reward = torch.tensor(0.5)
        module._update_cache(cache_key, reward)
        assert cache_key in module.reward_cache


class TestEnhancedTrajectoryCoherenceAnalyzer:
    """Test EnhancedTrajectoryCoherenceAnalyzer class."""
    
    def test_init_and_compute_coherence(self):
        """Test coherence analyzer __init__ and compute_coherence - covers lines 214-336."""
        analyzer = EnhancedTrajectoryCoherenceAnalyzer(
            temporal_decay=0.9,
            contradiction_penalty=0.5,
            pattern_boost=0.2
        )
        
        # Test empty trajectory
        result = analyzer.compute_coherence([])
        assert result == 0.0
        
        # Test single action trajectory  
        trajectory = [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}]
        result = analyzer.compute_coherence(trajectory)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        
        # Test logical sequence
        trajectory = [
            {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
            {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [110, 110]},
            {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'}
        ]
        result = analyzer.compute_coherence(trajectory)
        assert isinstance(result, float)
        assert result > 0.3  # Should have decent coherence
        
        # Test _detect_logical_patterns
        patterns = analyzer._detect_logical_patterns(trajectory)
        assert isinstance(patterns, list)
        
        # Test _detect_contradictions
        contradictory_trajectory = [
            {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
            {'action': 'ZOOM_IN', 'coordinates': [300, 300]}  # Far apart zoom
        ]
        contradictions = analyzer._detect_contradictions(contradictory_trajectory)
        assert isinstance(contradictions, list)


class TestToolMisusePenaltySystem:
    """Test ToolMisusePenaltySystem class."""
    
    def test_init_and_calculate_penalties(self):
        """Test penalty system __init__ and calculate_penalties - covers lines 338-453."""
        system = ToolMisusePenaltySystem(
            zoom_penalty_weight=0.3,
            segmentation_penalty_weight=0.4,
            text_penalty_weight=0.2,
            properties_penalty_weight=0.5,
            tracking_penalty_weight=0.6
        )
        
        # Test empty trajectory
        penalties = system.calculate_penalties([], {})
        assert penalties == {}
        
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
        
        # Test _check_constraint_violations
        violations = system._check_constraint_violations(violation_trajectory, image_data)
        assert isinstance(violations, list)


class TestNormalizedRewardOrchestrator:
    """Test NormalizedRewardOrchestrator class."""
    
    def test_init_and_calculate_total_reward(self):
        """Test orchestrator __init__ and calculate_total_reward - covers lines 580-814."""
        config = OmegaConf.create({
            'beta': 0.3,
            'alpha': 0.2,
            'gamma': 0.5,
            'tau': 0.1,
            'use_curriculum': False,
            'curriculum_stages': []
        })
        
        # Mock components to avoid CUDA issues
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule') as mock_curiosity:
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer') as mock_coherence:
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem') as mock_penalty:
                    # Create mock instances
                    curiosity_instance = MagicMock()
                    coherence_instance = MagicMock()
                    penalty_instance = MagicMock()
                    
                    curiosity_instance.compute_curiosity.return_value = torch.tensor(0.5)
                    coherence_instance.compute_coherence.return_value = 0.7
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
                    weights = orchestrator._get_curriculum_weights(100)
                    expected = {'curiosity': 1.0, 'coherence': 1.0, 'penalty': 1.0}
                    assert weights == expected
                    
                    # Test update_step
                    orchestrator.update_step(150)  # Should not crash
        
        # Test with curriculum
        config_with_curriculum = OmegaConf.create({
            'beta': 0.3, 'alpha': 0.2, 'gamma': 0.5, 'tau': 0.1,
            'use_curriculum': True,
            'curriculum_stages': [
                {'step': 100, 'weights': {'curiosity': 0.2, 'coherence': 0.3}},
                {'step': 200, 'weights': {'curiosity': 0.5, 'coherence': 0.6}}
            ]
        })
        
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule'):
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer'):
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem'):
                    orchestrator = NormalizedRewardOrchestrator(config_with_curriculum)
                    
                    # Test curriculum weights at different steps
                    weights_50 = orchestrator._get_curriculum_weights(50)
                    assert weights_50['curiosity'] == 1.0  # Before first stage
                    
                    weights_150 = orchestrator._get_curriculum_weights(150)
                    assert weights_150['curiosity'] == 0.2  # After first stage
                    
                    weights_250 = orchestrator._get_curriculum_weights(250)
                    assert weights_250['curiosity'] == 0.5  # After second stage


class TestRunningStats:
    """Test RunningStats class."""
    
    def test_init_and_update(self):
        """Test RunningStats __init__ and update - covers lines 795-807."""
        stats = RunningStats()
        
        # Test initial state
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std == 0.0
        
        # Test single update
        stats.update(5.0)
        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.variance == 0.0
        assert stats.std == 0.0
        
        # Test multiple updates
        values = [1.0, 2.0, 3.0, 4.0]
        for value in values:
            stats.update(value)
        
        assert stats.count == 5  # 1 initial + 4 new
        assert stats.mean > 0
        assert stats.variance >= 0
        assert stats.std >= 0
        
        # Test known statistics
        stats = RunningStats()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for value in values:
            stats.update(value)
        
        expected_mean = 5.0
        assert abs(stats.mean - expected_mean) < 1e-6
        assert stats.variance > 0
        assert stats.std > 0


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_coverage(self):
        """Test full reward calculation pipeline to hit remaining lines."""
        config = OmegaConf.create({
            'beta': 0.3, 'alpha': 0.2, 'gamma': 0.5, 'tau': 0.1,
            'use_curriculum': True,
            'curriculum_stages': [
                {'step': 50, 'weights': {'curiosity': 0.1, 'coherence': 0.2}}
            ]
        })
        
        with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule') as mock_curiosity:
            with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer') as mock_coherence:
                with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem') as mock_penalty:
                    
                    # Create mock instances that return appropriate values
                    curiosity_instance = MagicMock()
                    coherence_instance = MagicMock()
                    penalty_instance = MagicMock()
                    
                    curiosity_instance.compute_curiosity.return_value = torch.tensor(0.6)
                    coherence_instance.compute_coherence.return_value = 0.8
                    penalty_instance.calculate_penalties.return_value = {'violation': 0.2}
                    
                    mock_curiosity.return_value = curiosity_instance
                    mock_coherence.return_value = coherence_instance 
                    mock_penalty.return_value = penalty_instance
                    
                    orchestrator = NormalizedRewardOrchestrator(config)
                    
                    # Set up running stats to test normalization paths
                    orchestrator.running_stats = {
                        'curiosity': RunningStats(),
                        'coherence': RunningStats(),
                        'penalty': RunningStats()
                    }
                    
                    # Add some data to running stats
                    for i in range(10):
                        orchestrator.running_stats['curiosity'].update(0.5 + i * 0.1)
                        orchestrator.running_stats['coherence'].update(0.6 + i * 0.1)
                        orchestrator.running_stats['penalty'].update(0.1 + i * 0.05)
                    
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
                    step = 100  # After curriculum step
                    
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
                        trajectory, image_data, final_reward, step
                    )
                    assert isinstance(result, torch.Tensor)
                    
                    # Test without running stats (different normalization path)
                    orchestrator.running_stats = {}
                    result = orchestrator.calculate_total_reward(
                        trajectory, image_data, final_reward, step
                    )
                    assert isinstance(result, torch.Tensor)
                    
                    # Test extreme values for clipping
                    curiosity_instance.compute_curiosity.return_value = torch.tensor(100.0)  # Very high
                    coherence_instance.compute_coherence.return_value = -50.0  # Very low
                    penalty_instance.calculate_penalties.return_value = {'huge_penalty': 200.0}
                    
                    result = orchestrator.calculate_total_reward(
                        trajectory, image_data, 50.0, step  # High final reward too
                    )
                    assert isinstance(result, torch.Tensor)
                    # Should be clipped
                    assert abs(result.item()) <= 100.0  # Reasonable bound


if __name__ == '__main__':
    pytest.main([__file__, '-v'])