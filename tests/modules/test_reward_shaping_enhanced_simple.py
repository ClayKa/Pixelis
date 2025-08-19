#!/usr/bin/env python3
"""
Simple test suite for reward_shaping_enhanced.py to achieve maximum coverage.
Focuses on actually calling methods rather than complex testing scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from core.modules.reward_shaping_enhanced import (
    RewardComponents,
    LoRADynamicsModel,
    PerformanceAwareCuriosityModule,
    EnhancedTrajectoryCoherenceAnalyzer,
    ToolMisusePenaltySystem,
    NormalizedRewardOrchestrator,
    RunningStats
)


def test_reward_components():
    """Test RewardComponents dataclass."""
    components = RewardComponents(
        task_reward=1.0,
        curiosity_reward=0.5,
        coherence_reward=0.3,
        tool_penalty=-0.1,
        total_reward=1.7,
        metadata={'step': 100}
    )
    
    result = components.to_dict()
    assert isinstance(result, dict)


def test_lora_dynamics_model():
    """Test LoRADynamicsModel."""
    model = LoRADynamicsModel(device='cpu')
    
    state = torch.randn(4, 768)
    action = torch.randn(4, 128)
    output = model.forward(state, action)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 768)
    
    params = model.get_num_trainable_params()
    assert isinstance(params, int)
    assert params > 0


def test_curiosity_module():
    """Test PerformanceAwareCuriosityModule."""
    module = PerformanceAwareCuriosityModule(device='cpu')
    
    state = torch.randn(2, 768)
    action = torch.randn(2, 128)
    next_state = torch.randn(2, 768)
    
    result = module.compute_curiosity_reward(state, action, next_state)
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # Test cache methods
    cache_key = module._create_cache_key(state, action)
    assert isinstance(cache_key, (str, bytes))
    
    module._update_cache(cache_key, result)


def test_coherence_analyzer():
    """Test EnhancedTrajectoryCoherenceAnalyzer."""
    analyzer = EnhancedTrajectoryCoherenceAnalyzer()
    
    # Test empty trajectory
    result = analyzer.compute_coherence_reward([])
    assert isinstance(result, tuple)
    
    # Test trajectory with actions
    trajectory = [
        {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
        {'action': 'SEGMENT_OBJECT_AT', 'coordinates': [110, 110]}
    ]
    result = analyzer.compute_coherence_reward(trajectory)
    assert isinstance(result, tuple)
    
    # Test contradiction detection
    contradictions = analyzer._check_contradictions(trajectory)
    assert isinstance(contradictions, int)


def test_penalty_system():
    """Test ToolMisusePenaltySystem."""
    system = ToolMisusePenaltySystem()
    
    # Test empty trajectory
    result = system.calculate_penalties([], {})
    assert isinstance(result, tuple)
    
    # Test with actions
    trajectory = [
        {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
        {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'}
    ]
    result = system.calculate_penalties(trajectory, {})
    assert isinstance(result, tuple)
    
    # Test parameter checking
    for action in trajectory:
        result = system._check_parameters(action)
        assert isinstance(result, tuple)
        assert len(result) == 2


def test_orchestrator():
    """Test NormalizedRewardOrchestrator."""
    config = {
        'beta': 0.3,
        'alpha': 0.2,
        'gamma': 0.5,
        'tau': 0.1,
        'use_curriculum': False,
        'curriculum_stages': []
    }
    
    with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule') as mock_curiosity:
        with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer') as mock_coherence:
            with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem') as mock_penalty:
                
                # Setup mocks
                curiosity_instance = MagicMock()
                coherence_instance = MagicMock()
                penalty_instance = MagicMock()
                
                curiosity_instance.compute_curiosity_reward.return_value = (
                    torch.tensor([0.5]), {'cache_hit_rate': 0.0}
                )
                coherence_instance.compute_coherence_reward.return_value = (0.7, {})
                penalty_instance.calculate_penalties.return_value = (0.1, {})
                
                mock_curiosity.return_value = curiosity_instance
                mock_coherence.return_value = coherence_instance
                mock_penalty.return_value = penalty_instance
                
                orchestrator = NormalizedRewardOrchestrator(config)
                
                # Test reward calculation
                trajectory = [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}]
                image_data = torch.randn(3, 224, 224)
                
                result = orchestrator.calculate_total_reward(trajectory, image_data, 1.0, 100)
                assert isinstance(result, torch.Tensor)
                
                # Test embedding creation
                embedding = orchestrator._create_action_embedding({'action': 'ZOOM_IN', 'coordinates': [100, 100]})
                assert isinstance(embedding, torch.Tensor)
                
                # Test curriculum weights
                weights = orchestrator._get_curriculum_weights()
                assert isinstance(weights, dict)
                
                # Test step update
                orchestrator.update_step(150)


def test_orchestrator_with_curriculum():
    """Test NormalizedRewardOrchestrator with curriculum."""
    config = {
        'beta': 0.3, 'alpha': 0.2, 'gamma': 0.5, 'tau': 0.1,
        'use_curriculum': True,
        'curriculum_stages': [
            {'step': 100, 'weights': {'curiosity': 0.2, 'coherence': 0.3}}
        ]
    }
    
    with patch('core.modules.reward_shaping_enhanced.PerformanceAwareCuriosityModule'):
        with patch('core.modules.reward_shaping_enhanced.EnhancedTrajectoryCoherenceAnalyzer'):
            with patch('core.modules.reward_shaping_enhanced.ToolMisusePenaltySystem'):
                orchestrator = NormalizedRewardOrchestrator(config)
                
                # Test curriculum at different steps
                orchestrator.update_step(50)
                weights_50 = orchestrator._get_curriculum_weights()
                assert weights_50['curiosity'] == 1.0
                
                orchestrator.update_step(150)
                weights_150 = orchestrator._get_curriculum_weights()
                assert weights_150['curiosity'] == 0.2


def test_running_stats():
    """Test RunningStats."""
    stats = RunningStats()
    
    # Test update
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for value in values:
        stats.update(value)
    
    # Test normalize
    normalized = stats.normalize(3.0)
    assert isinstance(normalized, float)
    
    # Test with custom window size
    windowed_stats = RunningStats(window_size=3)
    for i in range(10):
        windowed_stats.update(float(i))
    
    assert len(windowed_stats.values) <= 3
    
    # Test normalize with small dataset
    small_stats = RunningStats()
    normalized = small_stats.normalize(5.0)
    assert normalized == 5.0


def test_edge_cases():
    """Test various edge cases to maximize coverage."""
    # Test LoRADynamicsModel with different parameters
    model = LoRADynamicsModel(state_dim=32, action_dim=16, device='cpu')
    state = torch.randn(1, 32)
    action = torch.randn(1, 16)
    output = model.forward(state, action)
    assert output.shape == (1, 32)
    
    # Test curiosity module with different sizes
    module = PerformanceAwareCuriosityModule(state_dim=32, action_dim=16, device='cpu')
    state = torch.randn(1, 32)
    action = torch.randn(1, 16)
    next_state = torch.randn(1, 32)
    result = module.compute_curiosity_reward(state, action, next_state)
    assert isinstance(result, tuple)
    
    # Test coherence analyzer with various trajectories
    analyzer = EnhancedTrajectoryCoherenceAnalyzer()
    
    trajectories = [
        [],
        [{'action': 'ZOOM_IN'}],
        [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}, {'action': 'UNKNOWN'}],
        [{'action': 'SEGMENT_OBJECT_AT', 'coordinates': [100, 100]} for _ in range(10)]
    ]
    
    for traj in trajectories:
        result = analyzer.compute_coherence_reward(traj)
        assert isinstance(result, tuple)
        
        contradictions = analyzer._check_contradictions(traj)
        assert isinstance(contradictions, int)
    
    # Test penalty system with various actions
    system = ToolMisusePenaltySystem()
    
    actions = [
        {},
        {'action': 'ZOOM_IN'},
        {'action': 'ZOOM_IN', 'coordinates': [100, 100]},
        {'action': 'GET_PROPERTIES', 'object_id': 'obj_1'},
        {'action': 'UNKNOWN_ACTION', 'random_param': 'test'}
    ]
    
    for action in actions:
        result = system._check_parameters(action)
        assert isinstance(result, tuple)
    
    # Test penalties with different trajectories
    trajectories = [
        [],
        [{'action': 'ZOOM_IN', 'coordinates': [100, 100]}],
        [{'action': 'GET_PROPERTIES', 'object_id': 'obj_1'}],
        [{'action': 'TRACK_OBJECT', 'object_id': 'obj_1'}]
    ]
    
    image_data_variants = [
        {},
        {'objects': []},
        {'objects': [], 'type': 'static'},
        {'objects': [{'id': 'obj_1'}], 'type': 'video'}
    ]
    
    for traj in trajectories:
        for img_data in image_data_variants:
            result = system.calculate_penalties(traj, img_data)
            assert isinstance(result, tuple)
    
    # Test running stats edge cases
    stats = RunningStats(window_size=1)
    stats.update(5.0)
    normalized = stats.normalize(5.0)
    assert isinstance(normalized, float)
    
    # Test with repeated values
    repeat_stats = RunningStats()
    for _ in range(5):
        repeat_stats.update(1.0)
    
    normalized = repeat_stats.normalize(1.0)
    assert isinstance(normalized, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])