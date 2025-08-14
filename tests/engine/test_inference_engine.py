"""
Unit Tests for Inference Engine

Tests the core inference engine including confidence gating, adaptive learning rate,
human-in-the-loop, and shared memory management.
"""

import unittest
import torch
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import asyncio
import threading
import time
from typing import Dict, Any

from core.engine.inference_engine import (
    InferenceEngine, SharedMemoryManager, SharedMemoryInfo
)
from core.data_structures import VotingResult, Experience, Trajectory
from core.modules.voting import TemporalEnsembleVoting


class TestSharedMemoryManager(unittest.TestCase):
    """Test suite for SharedMemoryManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.shm_manager = SharedMemoryManager(timeout_seconds=5.0)
    
    def test_create_shared_tensor(self):
        """Test creating a shared memory segment for a tensor."""
        # Create a test tensor
        tensor = torch.randn(10, 20)
        
        # Create shared memory
        shm_info = self.shm_manager.create_shared_tensor(tensor)
        
        # Verify metadata
        self.assertIsInstance(shm_info, SharedMemoryInfo)
        self.assertEqual(shm_info.shape, (10, 20))
        self.assertEqual(shm_info.dtype, tensor.dtype)
        self.assertGreater(shm_info.size_bytes, 0)
        
        # Verify tracking
        self.assertIn(shm_info.name, self.shm_manager.pending_shm)
    
    def test_mark_cleaned(self):
        """Test marking a segment as cleaned."""
        tensor = torch.randn(5, 5)
        shm_info = self.shm_manager.create_shared_tensor(tensor)
        
        # Mark as cleaned
        self.shm_manager.mark_cleaned(shm_info.name)
        
        # Verify removed from tracking
        self.assertNotIn(shm_info.name, self.shm_manager.pending_shm)
    
    def test_cleanup_stale_segments(self):
        """Test cleanup of stale segments."""
        # Create segments
        tensor1 = torch.randn(3, 3)
        shm_info1 = self.shm_manager.create_shared_tensor(tensor1)
        
        # Make segment stale by modifying creation time
        self.shm_manager.pending_shm[shm_info1.name].created_at = (
            datetime.now() - datetime.timedelta(seconds=10)
        )
        
        # Clean up stale segments
        cleaned = self.shm_manager.cleanup_stale_segments(worker_alive=True)
        
        # Verify cleanup
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0], shm_info1.name)
        self.assertNotIn(shm_info1.name, self.shm_manager.pending_shm)
    
    def test_get_status(self):
        """Test getting status of shared memory manager."""
        # Create some segments
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(20, 20)
        self.shm_manager.create_shared_tensor(tensor1)
        self.shm_manager.create_shared_tensor(tensor2)
        
        # Get status
        status = self.shm_manager.get_status()
        
        # Verify status
        self.assertEqual(status['pending_segments'], 2)
        self.assertGreater(status['total_bytes'], 0)
        self.assertIn('oldest_segment_age', status)


class TestInferenceEngine(unittest.TestCase):
    """Test suite for InferenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_model = MagicMock()
        self.mock_buffer = MagicMock()
        self.mock_voting = MagicMock()
        self.mock_orchestrator = MagicMock()
        
        # Create config
        self.config = {
            'confidence_threshold': 0.7,
            'min_learning_rate': 1e-6,
            'max_learning_rate': 1e-4,
            'hil_mode_enabled': False,
            'hil_review_percentage': 0.02,
            'k_neighbors': 5,
            'voting_strategy': 'weighted',
            'max_queue_size': 100,
            'shm_timeout': 60.0,
            'watchdog_interval': 5.0
        }
        
        # Create inference engine
        self.engine = InferenceEngine(
            model=self.mock_model,
            experience_buffer=self.mock_buffer,
            voting_module=self.mock_voting,
            reward_orchestrator=self.mock_orchestrator,
            config=self.config
        )
    
    def test_initialization(self):
        """Test inference engine initialization."""
        # Check components
        self.assertEqual(self.engine.model, self.mock_model)
        self.assertEqual(self.engine.experience_buffer, self.mock_buffer)
        self.assertEqual(self.engine.voting_module, self.mock_voting)
        self.assertEqual(self.engine.reward_orchestrator, self.mock_orchestrator)
        
        # Check configuration
        self.assertEqual(self.engine.confidence_threshold, 0.7)
        self.assertEqual(self.engine.min_lr, 1e-6)
        self.assertEqual(self.engine.max_lr, 1e-4)
        self.assertFalse(self.engine.hil_mode_enabled)
        
        # Check queues
        self.assertIsNotNone(self.engine.request_queue)
        self.assertIsNotNone(self.engine.response_queue)
        self.assertIsNotNone(self.engine.update_queue)
        self.assertIsNotNone(self.engine.human_review_queue)
    
    def test_should_trigger_update(self):
        """Test confidence gating mechanism."""
        # Test above threshold
        should_update = self.engine._should_trigger_update(0.8)
        self.assertTrue(should_update)
        
        # Test below threshold
        should_update = self.engine._should_trigger_update(0.6)
        self.assertFalse(should_update)
        
        # Test at threshold
        should_update = self.engine._should_trigger_update(0.7)
        self.assertTrue(should_update)
    
    def test_calculate_adaptive_lr(self):
        """Test adaptive learning rate calculation."""
        # Test with high confidence (low LR)
        lr = self.engine._calculate_adaptive_lr(0.9)
        expected_lr = self.config['max_learning_rate'] * 0.1  # error = 0.1
        self.assertAlmostEqual(lr, expected_lr, places=7)
        
        # Test with low confidence (high LR)
        lr = self.engine._calculate_adaptive_lr(0.1)
        expected_lr = self.config['max_learning_rate'] * 0.9  # error = 0.9
        self.assertAlmostEqual(lr, expected_lr, places=7)
        
        # Test bounds
        # Very high confidence (should hit min_lr)
        lr = self.engine._calculate_adaptive_lr(0.9999)
        self.assertGreaterEqual(lr, self.config['min_learning_rate'])
        
        # Very low confidence (should hit max_lr)
        lr = self.engine._calculate_adaptive_lr(0.0)
        self.assertLessEqual(lr, self.config['max_learning_rate'])
    
    def test_should_request_human_review(self):
        """Test HIL sampling logic."""
        # Test with HIL disabled
        self.engine.hil_mode_enabled = False
        should_review = self.engine._should_request_human_review()
        self.assertFalse(should_review)
        
        # Test with HIL enabled
        self.engine.hil_mode_enabled = True
        
        # Run multiple times to test sampling
        review_count = 0
        num_trials = 1000
        for _ in range(num_trials):
            if self.engine._should_request_human_review():
                review_count += 1
        
        # Should be approximately 2% (with some tolerance)
        expected_rate = self.config['hil_review_percentage']
        actual_rate = review_count / num_trials
        self.assertAlmostEqual(actual_rate, expected_rate, delta=0.01)
    
    @patch('core.engine.inference_engine.asyncio.run')
    def test_infer_and_adapt(self, mock_asyncio_run):
        """Test the main inference and adaptation loop."""
        # Setup mock returns
        self.mock_model.forward = MagicMock(return_value={
            'answer': 'cat',
            'confidence': 0.8,
            'logits': torch.randn(1, 100)
        })
        
        mock_neighbors = [MagicMock() for _ in range(5)]
        self.mock_buffer.search_index = MagicMock(return_value=mock_neighbors)
        
        mock_voting_result = VotingResult(
            final_answer={'answer': 'cat', 'trajectory': []},
            confidence=0.85,
            provenance={'test': 'data'}
        )
        self.mock_voting.vote = MagicMock(return_value=mock_voting_result)
        
        self.mock_orchestrator.calculate_reward = MagicMock(return_value={
            'total_reward': torch.tensor(0.9)
        })
        
        # Create test input
        input_data = {
            'image_features': torch.randn(1, 512),
            'question': 'What animal is this?'
        }
        
        # Configure asyncio mock to run the coroutine
        async def run_coro():
            return await self.engine.infer_and_adapt(input_data)
        
        mock_asyncio_run.return_value = ('cat', 0.85, {'test': 'data'})
        
        # Run inference
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_coro())
        
        # Verify calls
        self.mock_buffer.search_index.assert_called_once()
        self.mock_voting.vote.assert_called_once()
    
    def test_process_human_review_decision(self):
        """Test processing human review decisions."""
        # Test approval
        self.engine.process_human_review_decision(
            task_id="test_task_1",
            approved=True,
            reviewer_notes="Looks good"
        )
        
        # Check stats
        self.assertEqual(self.engine.stats.get('human_approvals', 0), 1)
        
        # Test rejection
        self.engine.process_human_review_decision(
            task_id="test_task_2",
            approved=False,
            reviewer_notes="Incorrect reasoning"
        )
        
        # Check stats
        self.assertEqual(self.engine.stats.get('human_rejections', 0), 1)
    
    def test_watchdog_cleanup(self):
        """Test watchdog cleanup functionality."""
        # Add a cleanup confirmation
        self.engine.cleanup_confirmation_queue.put("test_shm_segment")
        
        # Process cleanup confirmations
        self.engine._process_cleanup_confirmations()
        
        # Should have marked the segment as cleaned
        # (Would need to verify with actual SharedMemoryManager)
        self.assertTrue(self.engine.cleanup_confirmation_queue.empty())
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        # Update various stats
        with self.engine.stats_lock:
            self.engine.stats['total_requests'] = 100
            self.engine.stats['total_updates'] = 50
            self.engine.stats['failed_updates'] = 5
        
        # Verify stats
        with self.engine.stats_lock:
            self.assertEqual(self.engine.stats['total_requests'], 100)
            self.assertEqual(self.engine.stats['total_updates'], 50)
            self.assertEqual(self.engine.stats['failed_updates'], 5)
    
    def test_adaptive_lr_bounds(self):
        """Test that adaptive LR respects bounds across all confidence values."""
        confidences = np.linspace(0, 1, 100)
        
        for confidence in confidences:
            lr = self.engine._calculate_adaptive_lr(float(confidence))
            
            # Check bounds
            self.assertGreaterEqual(lr, self.config['min_learning_rate'])
            self.assertLessEqual(lr, self.config['max_learning_rate'])
            
            # Check monotonicity (higher confidence -> lower LR)
            if confidence > 0:
                prev_lr = self.engine._calculate_adaptive_lr(float(confidence - 0.01))
                self.assertLessEqual(lr, prev_lr + 1e-10)  # Allow for floating point error


class TestIntegration(unittest.TestCase):
    """Integration tests for inference engine components."""
    
    def test_voting_result_provenance(self):
        """Test that VotingResult properly maintains provenance."""
        from core.data_structures import VotingResult
        
        # Create a voting result with full provenance
        result = VotingResult(
            final_answer='cat',
            confidence=0.85,
            provenance={
                'model_self_answer': 'cat',
                'retrieved_neighbors_count': 5,
                'neighbor_answers': [
                    {'answer': 'cat', 'confidence': 0.9},
                    {'answer': 'dog', 'confidence': 0.6}
                ],
                'voting_strategy': 'weighted',
                'votes': [
                    {'answer': 'cat', 'confidence': 0.9},
                    {'answer': 'cat', 'confidence': 0.8}
                ],
                'weights': [0.5, 0.3, 0.2]
            }
        )
        
        # Test required fields are present
        required_fields = [
            'model_self_answer',
            'retrieved_neighbors_count',
            'neighbor_answers',
            'voting_strategy'
        ]
        
        for field in required_fields:
            self.assertIn(field, result.provenance)
        
        # Test methods
        distribution = result.get_vote_distribution()
        self.assertIsInstance(distribution, dict)
        
        strength = result.get_consensus_strength()
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)


if __name__ == '__main__':
    unittest.main()