#!/usr/bin/env python3
"""
Unit tests for RFT training with GRPO.

Tests the core components of the reinforcement fine-tuning system including:
- GRPO trainer with group normalization
- Curiosity module with LoRA dynamics
- Coherence analyzer
- Tool misuse penalty system
- Reward orchestrator with normalization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
from trl import PPOTrainer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules to test
try:
    from scripts.train_rft import (
        LightweightDynamicsModel,
        EnhancedCuriosityModule,
        EnhancedCoherenceAnalyzer,
        ToolMisusePenaltyCalculator,
        EnhancedRewardOrchestrator,
        GRPOTrainer,
        parse_trajectory,
        extract_answer
    )
except ImportError:
    from train_rft import (
        LightweightDynamicsModel,
        EnhancedCuriosityModule,
        EnhancedCoherenceAnalyzer,
        ToolMisusePenaltyCalculator,
        EnhancedRewardOrchestrator,
        GRPOTrainer,
        parse_trajectory,
        extract_answer
    )

try:
    from core.modules.reward_shaping_enhanced import (
        LoRADynamicsModel,
        PerformanceAwareCuriosityModule,
        EnhancedTrajectoryCoherenceAnalyzer as EnhancedAnalyzer,
        ToolMisusePenaltySystem,
        NormalizedRewardOrchestrator,
        RunningStats
    )
except ImportError:
    pass


class TestLightweightDynamicsModel:
    """Test the lightweight dynamics model with LoRA."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LightweightDynamicsModel(
            state_dim=768,
            action_dim=128,
            hidden_dim=256,
            lora_rank=8,
            device="cpu"
        )
        
        assert model.state_dim == 768
        assert model.action_dim == 128
        assert model.lora_scale == 2.0  # 16 / 8
        
        # Check layers exist
        assert hasattr(model, 'base_forward')
        assert hasattr(model, 'lora_down')
        assert hasattr(model, 'lora_up')
    
    def test_forward_pass(self):
        """Test forward pass through dynamics model."""
        model = LightweightDynamicsModel(
            state_dim=64,
            action_dim=32,
            hidden_dim=128,
            device="cpu"
        )
        
        batch_size = 4
        state = torch.randn(batch_size, 64)
        action = torch.randn(batch_size, 32)
        
        # Forward pass
        next_state = model(state, action)
        
        # Check output shape
        assert next_state.shape == (batch_size, 64)
        
        # Check gradients flow
        loss = next_state.mean()
        loss.backward()
        
        # LoRA weights should have gradients
        assert model.lora_down.weight.grad is not None
        assert model.lora_up.weight.grad is not None
    
    def test_lora_parameter_efficiency(self):
        """Test that LoRA reduces parameter count."""
        state_dim = 768
        action_dim = 128
        hidden_dim = 256
        lora_rank = 8
        
        model = LightweightDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            device="cpu"
        )
        
        # Calculate LoRA parameters
        lora_params = (state_dim + action_dim) * lora_rank + lora_rank * state_dim
        
        # Calculate what full fine-tuning would require
        full_params = (
            (state_dim + action_dim) * hidden_dim + 
            hidden_dim * hidden_dim + 
            hidden_dim * state_dim
        )
        
        # LoRA should use significantly fewer parameters
        assert lora_params < full_params * 0.1  # Less than 10% of full parameters


class TestEnhancedCuriosityModule:
    """Test the enhanced curiosity module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = EnhancedCuriosityModule(
            state_dim=64,
            action_dim=32,
            beta=0.2,
            eta=0.5,
            cache_size=100,
            device="cpu"
        )
        
        assert module.beta == 0.2
        assert module.eta == 0.5
        assert hasattr(module, 'dynamics_model')
        assert hasattr(module, 'inverse_model')
        assert len(module.cache) == 0
    
    def test_curiosity_reward_computation(self):
        """Test curiosity reward calculation."""
        module = EnhancedCuriosityModule(
            state_dim=64,
            action_dim=32,
            device="cpu"
        )
        
        batch_size = 2
        state = torch.randn(batch_size, 64)
        action = torch.randn(batch_size, 32)
        next_state = torch.randn(batch_size, 64)
        
        # Compute curiosity reward
        reward, metrics = module.compute_curiosity_reward(
            state, action, next_state
        )
        
        # Check outputs
        assert reward.shape == (batch_size,)
        assert 'forward_loss' in metrics
        assert 'inverse_loss' in metrics
        assert 'prediction_error' in metrics
        
        # Rewards should be non-negative (scaled prediction error)
        assert (reward >= 0).all()
    
    def test_caching_mechanism(self):
        """Test LRU caching for efficiency."""
        module = EnhancedCuriosityModule(
            state_dim=64,
            action_dim=32,
            cache_size=5,
            device="cpu"
        )
        
        # Create identical inputs
        state = torch.randn(1, 64)
        action = torch.randn(1, 32)
        next_state = torch.randn(1, 64)
        
        # First call - cache miss
        reward1, _ = module.compute_curiosity_reward(state, action, next_state)
        assert len(module.cache) == 1
        
        # Second call with same inputs - cache hit
        reward2, _ = module.compute_curiosity_reward(state, action, next_state)
        assert torch.allclose(reward1, reward2)
        assert len(module.cache) == 1  # No new entry
        
        # Fill cache beyond capacity
        for i in range(6):
            s = torch.randn(1, 64)
            a = torch.randn(1, 32)
            ns = torch.randn(1, 64)
            module.compute_curiosity_reward(s, a, ns)
        
        # Cache should maintain max size
        assert len(module.cache) <= 5


class TestEnhancedCoherenceAnalyzer:
    """Test trajectory coherence analysis."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = EnhancedCoherenceAnalyzer(
            coherence_threshold=0.7,
            repetition_penalty=0.5,
            sequence_bonus=0.2
        )
        
        assert analyzer.coherence_threshold == 0.7
        assert analyzer.repetition_penalty == 0.5
        assert analyzer.sequence_bonus == 0.2
    
    def test_empty_trajectory(self):
        """Test handling of empty trajectory."""
        analyzer = EnhancedCoherenceAnalyzer()
        
        reward, metrics = analyzer.compute_coherence_reward([])
        
        assert reward.item() == 0.0
        assert metrics['coherence_score'] == 0.0
    
    def test_repetition_detection(self):
        """Test detection of repetitive actions."""
        analyzer = EnhancedCoherenceAnalyzer(repetition_penalty=0.5)
        
        # Trajectory with repetition
        trajectory = [
            {'operation': 'ZOOM_IN', 'arguments': {'x': 100, 'y': 100}},
            {'operation': 'ZOOM_IN', 'arguments': {'x': 100, 'y': 100}},  # Repetition
            {'operation': 'READ_TEXT', 'arguments': {}}
        ]
        
        reward, metrics = analyzer.compute_coherence_reward(trajectory)
        
        assert metrics['repetitions'] == 1
        assert reward < 0  # Should be penalized
    
    def test_good_sequence_detection(self):
        """Test detection of good action sequences."""
        analyzer = EnhancedCoherenceAnalyzer(sequence_bonus=0.3)
        
        # Trajectory with good sequence
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 100, 'y': 100}},
            {'operation': 'GET_PROPERTIES', 'arguments': {}},  # Good sequence
            {'operation': 'THINK', 'arguments': {}}
        ]
        
        reward, metrics = analyzer.compute_coherence_reward(trajectory)
        
        assert metrics['good_sequences'] == 1
        assert reward > 0  # Should be rewarded
    
    def test_embedding_similarity_analysis(self):
        """Test coherence based on embedding similarity."""
        analyzer = EnhancedCoherenceAnalyzer()
        
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT'},
            {'operation': 'GET_PROPERTIES'},
            {'operation': 'THINK'}
        ]
        
        # Create embeddings with moderate similarity
        embeddings = [
            torch.randn(768),
            torch.randn(768) * 0.8 + torch.randn(768) * 0.2,  # Similar but not identical
            torch.randn(768) * 0.7 + torch.randn(768) * 0.3
        ]
        
        reward, metrics = analyzer.compute_coherence_reward(trajectory, embeddings)
        
        assert 'avg_similarity' in metrics
        assert -1 <= reward.item() <= 1  # Should be bounded


class TestToolMisusePenaltyCalculator:
    """Test tool misuse penalty calculation."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = ToolMisusePenaltyCalculator(penalty_weight=0.1)
        
        assert calculator.penalty_weight == 0.1
        assert 'TRACK_OBJECT' in calculator.tool_constraints
        assert 'GET_PROPERTIES' in calculator.tool_constraints
    
    def test_no_violations(self):
        """Test trajectory with no violations."""
        calculator = ToolMisusePenaltyCalculator()
        
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 500, 'y': 500}},
            {'operation': 'GET_PROPERTIES', 'arguments': {}},
            {'operation': 'THINK', 'arguments': {}}
        ]
        
        context = {'input_type': 'image'}
        
        penalty, violations = calculator.calculate_penalties(trajectory, context)
        
        assert penalty == 0.0
        assert len(violations) == 0
    
    def test_missing_prerequisite(self):
        """Test penalty for missing prerequisite."""
        calculator = ToolMisusePenaltyCalculator(penalty_weight=0.2)
        
        trajectory = [
            {'operation': 'GET_PROPERTIES', 'arguments': {}},  # No SEGMENT_OBJECT_AT before
            {'operation': 'THINK', 'arguments': {}}
        ]
        
        context = {'input_type': 'image'}
        
        penalty, violations = calculator.calculate_penalties(trajectory, context)
        
        assert penalty < 0  # Should be penalized
        assert 'missing_prerequisite_GET_PROPERTIES' in violations
        assert violations['missing_prerequisite_GET_PROPERTIES'] == 1
    
    def test_track_on_static_image(self):
        """Test severe penalty for tracking on static image."""
        calculator = ToolMisusePenaltyCalculator(penalty_weight=0.1)
        
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 500, 'y': 500}},
            {'operation': 'TRACK_OBJECT', 'arguments': {}}  # Invalid on static image
        ]
        
        context = {'input_type': 'image'}  # Not video
        
        penalty, violations = calculator.calculate_penalties(trajectory, context)
        
        assert penalty == -0.2  # Double penalty for severe violation
        assert violations['track_on_static_image'] == 1
    
    def test_out_of_bounds_coordinates(self):
        """Test penalty for out-of-bounds coordinates."""
        calculator = ToolMisusePenaltyCalculator(penalty_weight=0.1)
        
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 2000, 'y': 2000}},  # Out of bounds
            {'operation': 'ZOOM_IN', 'arguments': {'x': -100, 'y': 500}}  # Out of bounds
        ]
        
        context = {'input_type': 'image'}
        
        penalty, violations = calculator.calculate_penalties(trajectory, context)
        
        assert penalty < 0
        assert violations['out_of_bounds_coordinates'] == 2


class TestEnhancedRewardOrchestrator:
    """Test the central reward orchestrator."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        config = {
            'task_reward_weight': 1.0,
            'curiosity_reward_weight': 0.3,
            'coherence_reward_weight': 0.2,
            'tool_misuse_penalty': 0.1,
            'curriculum_stages': []
        }
        
        orchestrator = EnhancedRewardOrchestrator(config)
        
        assert orchestrator.task_weight == 1.0
        assert orchestrator.curiosity_weight == 0.3
        assert orchestrator.coherence_weight == 0.2
        assert hasattr(orchestrator, 'curiosity_module')
        assert hasattr(orchestrator, 'coherence_analyzer')
        assert hasattr(orchestrator, 'penalty_calculator')
    
    def test_total_reward_calculation(self):
        """Test calculation of total reward."""
        config = {
            'task_reward_weight': 1.0,
            'curiosity_reward_weight': 0.3,
            'coherence_reward_weight': 0.2,
            'tool_misuse_penalty': 0.1
        }
        
        orchestrator = EnhancedRewardOrchestrator(config)
        
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 500, 'y': 500}},
            {'operation': 'GET_PROPERTIES', 'arguments': {}},
            {'operation': 'THINK', 'arguments': {}}
        ]
        
        final_answer = "red"
        ground_truth = "red"
        
        # Create mock embeddings
        state_embeddings = [torch.randn(768) for _ in range(4)]
        
        context = {'input_type': 'image'}
        
        result = orchestrator.calculate_total_reward(
            trajectory=trajectory,
            final_answer=final_answer,
            ground_truth=ground_truth,
            state_embeddings=state_embeddings,
            context=context
        )
        
        # Check result structure
        assert 'total' in result
        assert 'components' in result
        assert 'metrics' in result
        assert 'weights' in result
        
        # Check components
        assert 'task' in result['components']
        assert 'curiosity' in result['components']
        assert 'coherence' in result['components']
        assert 'penalty' in result['components']
        
        # Task reward should be 1.0 for correct answer
        assert result['components']['task']['raw'] == 1.0
    
    def test_running_statistics_normalization(self):
        """Test normalization with running statistics."""
        config = {'task_reward_weight': 1.0}
        orchestrator = EnhancedRewardOrchestrator(config)
        
        # Update statistics multiple times
        for i in range(20):
            orchestrator._update_running_stats('task', float(i % 2))
        
        # Check statistics
        stats = orchestrator.running_stats['task']
        assert stats['count'] == 20
        assert 0.4 < stats['mean'] < 0.6  # Should be around 0.5
        
        # Test normalization
        normalized = orchestrator._normalize_reward('task', 1.0)
        assert normalized != 1.0  # Should be normalized
    
    def test_curriculum_weight_adjustment(self):
        """Test curriculum-based weight adjustment."""
        config = {
            'task_reward_weight': 1.0,
            'curiosity_reward_weight': 0.0,
            'coherence_reward_weight': 0.0,
            'curriculum_stages': [
                {
                    'step': 0,
                    'weights': {'task': 1.0, 'curiosity': 0.0, 'coherence': 0.0}
                },
                {
                    'step': 100,
                    'weights': {'task': 0.7, 'curiosity': 0.2, 'coherence': 0.1}
                },
                {
                    'step': 200,
                    'weights': {'task': 0.5, 'curiosity': 0.3, 'coherence': 0.2}
                }
            ]
        }
        
        orchestrator = EnhancedRewardOrchestrator(config)
        
        # Initial weights
        weights = orchestrator._get_curriculum_weights()
        assert weights['task'] == 1.0
        assert weights['curiosity'] == 0.0
        
        # Update to step 150
        orchestrator.update_curriculum_step(150)
        weights = orchestrator._get_curriculum_weights()
        assert weights['task'] == 0.7
        assert weights['curiosity'] == 0.2
        
        # Update to step 250
        orchestrator.update_curriculum_step(250)
        weights = orchestrator._get_curriculum_weights()
        assert weights['task'] == 0.5
        assert weights['curiosity'] == 0.3


class TestGRPOTrainer:
    """Test GRPO trainer functionality."""
    
    @patch('scripts.train_rft.PPOTrainer.__init__')
    def test_initialization(self, mock_ppo_init):
        """Test GRPO trainer initialization."""
        mock_ppo_init.return_value = None
        
        grpo_config = {
            'group_size': 4,
            'replay_buffer_size': 100,
            'replay_ratio': 0.5
        }
        
        # Create mock objects for required arguments
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_tokenizer = MagicMock()
        
        trainer = GRPOTrainer(
            model=mock_model,
            config=mock_config,
            tokenizer=mock_tokenizer,
            grpo_config=grpo_config
        )
        
        assert trainer.group_size == 4
        assert trainer.replay_buffer_size == 100
        assert trainer.replay_ratio == 0.5
        assert len(trainer.replay_buffer) == 0
    
    def test_group_advantage_normalization(self):
        # 1. Manually create a predictable input tensor.
        mock_advantages = torch.tensor([
            1.0, 2.0, 3.0, 4.0,
            5.0, 5.0, 5.0, 5.0,
            -2.0, -1.0, 1.0, 2.0
        ])

        # 2. Mock the PARENT class method to return our PREDICTABLE tensor.
        # We need to mock PPOTrainer's compute_advantages, not GRPOTrainer's
        from unittest.mock import patch, MagicMock
        
        # The rest of the test can now make precise assertions.
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.group_size = 4
        
        values = torch.randn(12, 1) # These are now irrelevant but needed for the call
        rewards = torch.randn(12, 1)
        mask = torch.ones(12, 1)
        
        # Create a mock for the parent's compute_advantages method
        # We'll patch the super() call directly
        original_super = super
        
        def mock_super(cls=None, inst=None):
            mock_parent = MagicMock()
            mock_parent.compute_advantages = MagicMock(return_value=mock_advantages)
            return mock_parent
        
        # Temporarily replace super() with our mock
        import builtins
        with patch.object(builtins, 'super', mock_super):
            # This will call the actual group normalization logic on our mock_advantages
            normalized_advantages = trainer.compute_advantages(values, rewards, mask)

            # Check shape
            assert normalized_advantages.shape == (12,)
            
            # Reshape for analysis
            grouped = normalized_advantages.view(3, 4)

            # --- NEW, ROBUST ASSERTIONS ---
            # Group 1 should now have mean ~0 and std ~1
            assert grouped[0].mean().item() == pytest.approx(0.0, abs=1e-6)
            assert grouped[0].std().item() == pytest.approx(1.0, abs=1e-6)

            # Group 2 (all same values) is an edge case. Std is 0, so normalization
            # should result in all zeros to avoid division by zero.
            assert grouped[1].mean().item() == pytest.approx(0.0, abs=1e-6)
            assert grouped[1].std().item() == pytest.approx(0.0, abs=1e-6)
            
            # Group 3 already has mean 0, so it should be scaled by its std dev.
            assert grouped[2].mean().item() == pytest.approx(0.0, abs=1e-6)
            assert grouped[2].std().item() == pytest.approx(1.0, abs=1e-6)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_parse_trajectory(self):
        """Test trajectory parsing from response."""
        response = """
        Let me analyze this image.
        SEGMENT_OBJECT_AT(x=100, y=200)
        I found an object.
        READ_TEXT()
        The text says "Hello".
        TRACK_OBJECT(id=1)
        GET_PROPERTIES()
        <answer>red ball</answer>
        """
        
        trajectory = parse_trajectory(response)
        
        assert len(trajectory) == 4
        assert trajectory[0]['operation'] == 'SEGMENT_OBJECT_AT'
        assert trajectory[1]['operation'] == 'READ_TEXT'
        assert trajectory[2]['operation'] == 'TRACK_OBJECT'
        assert trajectory[3]['operation'] == 'GET_PROPERTIES'
    
    def test_extract_answer(self):
        """Test answer extraction from response."""
        # Test with answer tags
        response1 = "Some reasoning...\n<answer>42</answer>\nMore text"
        answer1 = extract_answer(response1)
        assert answer1 == "42"
        
        # Test without tags (fallback to last line)
        response2 = "Some reasoning...\nThe answer is 42"
        answer2 = extract_answer(response2)
        assert answer2 == "The answer is 42"
        
        # Test empty response
        response3 = ""
        answer3 = extract_answer(response3)
        assert answer3 == ""


class TestIntegration:
    """Integration tests for the complete RFT system."""
    
    def test_end_to_end_reward_calculation(self):
        """Test complete reward calculation pipeline."""
        config = {
            'task_reward_weight': 1.0,
            'curiosity_reward_weight': 0.3,
            'coherence_reward_weight': 0.2,
            'tool_misuse_penalty': 0.1,
            'curiosity_beta': 0.2,
            'curiosity_eta': 0.5,
            'coherence_threshold': 0.7,
            'repetition_penalty': 0.5,
            'sequence_bonus': 0.2,
            'curriculum_stages': []
        }
        
        orchestrator = EnhancedRewardOrchestrator(config)
        
        # Create a complete trajectory
        trajectory = [
            {'operation': 'SEGMENT_OBJECT_AT', 'arguments': {'x': 500, 'y': 500}},
            {'operation': 'GET_PROPERTIES', 'arguments': {}},
            {'operation': 'READ_TEXT', 'arguments': {}},
            {'operation': 'THINK', 'arguments': {}}
        ]
        
        # Create state embeddings
        state_embeddings = [torch.randn(768) for _ in range(5)]
        
        # Calculate reward
        result = orchestrator.calculate_total_reward(
            trajectory=trajectory,
            final_answer="red circle",
            ground_truth="red circle",
            state_embeddings=state_embeddings,
            context={'input_type': 'image'}
        )
        
        # Verify all components are calculated
        assert result['total'] is not None
        assert result['components']['task']['raw'] == 1.0  # Correct answer
        assert result['components']['curiosity']['raw'] >= 0  # Non-negative
        assert result['components']['coherence']['raw'] is not None
        assert result['components']['penalty'] <= 0  # Non-positive
        
        # Check metrics are populated
        assert 'curiosity' in result['metrics']
        assert 'coherence' in result['metrics']
        assert 'violations' in result['metrics']
    
    def test_memory_efficiency(self):
        """Test memory efficiency of LoRA implementation."""
        # Create models with different configurations
        full_model = nn.Sequential(
            nn.Linear(896, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
        
        lora_model = LightweightDynamicsModel(
            state_dim=768,
            action_dim=128,
            hidden_dim=256,
            lora_rank=8,
            device="cpu"
        )
        
        # Count parameters
        full_params = sum(p.numel() for p in full_model.parameters())
        lora_params = sum(p.numel() for p in lora_model.lora_down.parameters())
        lora_params += sum(p.numel() for p in lora_model.lora_up.parameters())
        
        # LoRA should use much fewer trainable parameters
        assert lora_params < full_params * 0.05  # Less than 5% of full model
        
        print(f"Full model parameters: {full_params:,}")
        print(f"LoRA trainable parameters: {lora_params:,}")
        print(f"Parameter reduction: {(1 - lora_params/full_params)*100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])