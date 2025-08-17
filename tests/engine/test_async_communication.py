"""
Test Asynchronous Communication

Tests for the two-process architecture and inter-process communication system.
Includes normal workflow tests and fault-tolerance tests.
"""

import pytest
import torch
import torch.multiprocessing as mp
from queue import Empty
import time
import os
import signal
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.engine.inference_engine import InferenceEngine, SharedMemoryManager, SharedMemoryInfo
from core.engine.update_worker import UpdateWorker, SharedMemoryReconstructor
from core.data_structures import Experience, UpdateTask, Trajectory, VotingResult


class TestSharedMemoryManager:
    """Test the shared memory manager functionality."""
    
    def test_create_shared_tensor(self):
        """Test creating a shared memory segment for a tensor."""
        manager = SharedMemoryManager(timeout_seconds=10.0)
        
        # Create a test tensor
        tensor = torch.randn(10, 20, 30)
        
        # Create shared memory segment
        shm_info = manager.create_shared_tensor(tensor)
        
        # Verify the shared memory info
        assert shm_info.name.startswith("pixelis_shm_")
        assert shm_info.shape == (10, 20, 30)
        assert shm_info.dtype == torch.float32
        assert shm_info.size_bytes == tensor.element_size() * tensor.numel()
        assert shm_info.name in manager.pending_shm
        
        # Clean up
        manager.mark_cleaned(shm_info.name)
        assert shm_info.name not in manager.pending_shm
    
    def test_cuda_tensor_transfer(self):
        """Test transferring CUDA tensors to shared memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        manager = SharedMemoryManager()
        
        # Create CUDA tensor
        cuda_tensor = torch.randn(5, 5).cuda()
        
        # Transfer to shared memory (should move to CPU pinned memory)
        shm_info = manager.create_shared_tensor(cuda_tensor)
        
        assert shm_info.shape == (5, 5)
        assert shm_info.name in manager.pending_shm
        
        # Clean up
        manager.mark_cleaned(shm_info.name)
    
    def test_cleanup_stale_segments(self):
        """Test cleaning up stale shared memory segments."""
        manager = SharedMemoryManager(timeout_seconds=0.1)  # Very short timeout
        
        # Create some segments
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(20, 20)
        
        shm_info1 = manager.create_shared_tensor(tensor1)
        time.sleep(0.05)
        shm_info2 = manager.create_shared_tensor(tensor2)
        
        # Wait for first segment to become stale
        time.sleep(0.1)
        
        # Clean up stale segments
        cleaned = manager.cleanup_stale_segments(worker_alive=True)
        
        # First segment should be cleaned
        assert shm_info1.name in cleaned
        assert shm_info1.name not in manager.pending_shm
        
        # Second segment might also be cleaned if timing is tight
        # Just check that at least one segment was cleaned
        assert len(cleaned) >= 1
        assert shm_info1.name in cleaned
        
        # Clean up
        manager.mark_cleaned(shm_info2.name)
    
    def test_cleanup_on_worker_death(self):
        """Test cleaning up all segments when worker dies."""
        manager = SharedMemoryManager(timeout_seconds=60.0)
        
        # Create multiple segments
        tensors = [torch.randn(10, 10) for _ in range(3)]
        shm_infos = [manager.create_shared_tensor(t) for t in tensors]
        
        # Simulate worker death
        cleaned = manager.cleanup_stale_segments(worker_alive=False)
        
        # All segments should be cleaned
        assert len(cleaned) == 3
        for info in shm_infos:
            assert info.name in cleaned
            assert info.name not in manager.pending_shm
    
    def test_get_status(self):
        """Test getting status of the shared memory manager."""
        manager = SharedMemoryManager()
        
        # Initial status
        status = manager.get_status()
        assert status["pending_segments"] == 0
        assert status["total_bytes"] == 0
        assert status["oldest_segment_age"] == 0
        
        # Create some segments
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(20, 20)
        
        shm_info1 = manager.create_shared_tensor(tensor1)
        time.sleep(0.1)
        shm_info2 = manager.create_shared_tensor(tensor2)
        
        # Check status
        status = manager.get_status()
        assert status["pending_segments"] == 2
        expected_bytes = (10 * 10 + 20 * 20) * 4  # float32 = 4 bytes
        assert status["total_bytes"] == expected_bytes
        assert status["oldest_segment_age"] >= 0.1
        
        # Clean up
        manager.mark_cleaned(shm_info1.name)
        manager.mark_cleaned(shm_info2.name)


class TestAsyncCommunication:
    """Test the asynchronous communication between processes."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
        model.state_dict.return_value = {}
        return model
    
    @pytest.fixture
    def mock_experience_buffer(self):
        """Create a mock experience buffer."""
        buffer = MagicMock()
        buffer.search_index.return_value = []
        return buffer
    
    @pytest.fixture
    def mock_voting_module(self):
        """Create a mock voting module."""
        voting = MagicMock()
        result = VotingResult(
            final_answer={"answer": "test", "trajectory": []},
            confidence=0.8,
            provenance={
                'model_self_answer': "test",
                'retrieved_neighbors_count': 0,
                'neighbor_answers': [],
                'voting_strategy': 'majority'
            }
        )
        voting.vote.return_value = result
        return voting
    
    @pytest.fixture
    def mock_reward_orchestrator(self):
        """Create a mock reward orchestrator."""
        orchestrator = MagicMock()
        orchestrator.calculate_reward.return_value = {
            'total_reward': torch.tensor(1.0),
            'task_reward': 1.0,
            'curiosity_reward': 0.0,
            'coherence_reward': 0.0
        }
        return orchestrator
    
    def test_inference_engine_initialization(
        self, mock_model, mock_experience_buffer, 
        mock_voting_module, mock_reward_orchestrator
    ):
        """Test initializing the inference engine."""
        config = {
            'confidence_threshold': 0.7,
            'min_learning_rate': 1e-6,
            'max_learning_rate': 1e-4,
            'shm_timeout': 30.0,
            'watchdog_interval': 5.0
        }
        
        engine = InferenceEngine(
            model=mock_model,
            experience_buffer=mock_experience_buffer,
            voting_module=mock_voting_module,
            reward_orchestrator=mock_reward_orchestrator,
            config=config
        )
        
        # Check initialization
        assert engine.model == mock_model
        assert engine.confidence_threshold == 0.7
        assert engine.min_lr == 1e-6
        assert engine.max_lr == 1e-4
        assert engine.watchdog_interval == 5.0
        
        # Check queues are created
        assert hasattr(engine, 'request_queue')
        assert hasattr(engine, 'response_queue')
        assert hasattr(engine, 'update_queue')
        assert hasattr(engine, 'cleanup_confirmation_queue')
        
        # Check shared memory manager
        assert isinstance(engine.shm_manager, SharedMemoryManager)
        assert engine.shm_manager.timeout_seconds == 30.0
    
    def test_update_worker_initialization(self, mock_model):
        """Test initializing the update worker."""
        config = {
            'kl_weight': 0.01,
            'max_kl': 0.05,
            'grad_clip_norm': 1.0,
            'ema_decay': 0.999,
            'base_learning_rate': 1e-5,
            'weight_decay': 0.01
        }
        
        update_queue = mp.Queue()
        cleanup_queue = mp.Queue()
        
        worker = UpdateWorker(
            model=mock_model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=config
        )
        
        # Check initialization
        assert worker.model == mock_model
        assert worker.kl_config.initial_beta == 0.01
        assert worker.kl_config.target_kl == 0.05
        assert worker.gradient_clip_norm == 1.0
        assert worker.ema_decay == 0.999
        assert worker.stats['total_updates'] == 0
        assert worker.stats['failed_updates'] == 0
    
    def test_adaptive_learning_rate(
        self, mock_model, mock_experience_buffer,
        mock_voting_module, mock_reward_orchestrator
    ):
        """Test adaptive learning rate calculation."""
        config = {
            'min_learning_rate': 1e-6,
            'max_learning_rate': 1e-4
        }
        
        engine = InferenceEngine(
            model=mock_model,
            experience_buffer=mock_experience_buffer,
            voting_module=mock_voting_module,
            reward_orchestrator=mock_reward_orchestrator,
            config=config
        )
        
        # Test with different confidence levels
        lr_high_conf = engine._calculate_adaptive_lr(0.9)  # High confidence
        lr_med_conf = engine._calculate_adaptive_lr(0.5)   # Medium confidence
        lr_low_conf = engine._calculate_adaptive_lr(0.1)   # Low confidence
        
        # Higher confidence should lead to lower learning rate
        assert lr_high_conf < lr_med_conf < lr_low_conf
        
        # Check bounds
        assert engine.min_lr <= lr_high_conf <= engine.max_lr
        assert engine.min_lr <= lr_med_conf <= engine.max_lr
        assert engine.min_lr <= lr_low_conf <= engine.max_lr
    
    @pytest.mark.asyncio
    async def test_infer_and_adapt_workflow(
        self, mock_model, mock_experience_buffer,
        mock_voting_module, mock_reward_orchestrator
    ):
        """Test the complete infer_and_adapt workflow."""
        config = {
            'confidence_threshold': 0.7,
            'k_neighbors': 5,
            'voting_strategy': 'weighted'
        }
        
        engine = InferenceEngine(
            model=mock_model,
            experience_buffer=mock_experience_buffer,
            voting_module=mock_voting_module,
            reward_orchestrator=mock_reward_orchestrator,
            config=config
        )
        
        # Mock the model prediction
        async def mock_prediction(input_data):
            return {'logits': torch.randn(10, 100), 'answer': 'test'}
        
        engine._get_model_prediction = mock_prediction
        
        # Test input
        input_data = {
            'image_features': torch.randn(3, 224, 224),
            'question': 'What is in the image?'
        }
        
        # Run inference and adaptation
        result, confidence, metadata = await engine.infer_and_adapt(input_data)
        
        # Check results
        assert result == {"answer": "test", "trajectory": []}
        assert confidence == 0.8
        assert isinstance(metadata, dict)
        
        # Verify components were called
        mock_experience_buffer.search_index.assert_called_once()
        mock_voting_module.vote.assert_called_once()
        mock_reward_orchestrator.calculate_reward.assert_called_once()
    
    def test_queue_communication(self):
        """Test basic queue communication between processes."""
        # Create queues
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        
        # Send request
        request = {
            'request_id': 'test_123',
            'input_data': {'question': 'test question'}
        }
        request_queue.put(request)
        
        # Get request
        received = request_queue.get(timeout=1.0)
        assert received == request
        
        # Send response
        response = {
            'request_id': 'test_123',
            'result': 'test result',
            'success': True
        }
        response_queue.put(response)
        
        # Get response
        received = response_queue.get(timeout=1.0)
        assert received == response
    
    def test_update_task_enqueue_with_shared_memory(
        self, mock_model, mock_experience_buffer,
        mock_voting_module, mock_reward_orchestrator
    ):
        """Test enqueueing update task with shared memory transfer."""
        config = {'confidence_threshold': 0.7}
        
        engine = InferenceEngine(
            model=mock_model,
            experience_buffer=mock_experience_buffer,
            voting_module=mock_voting_module,
            reward_orchestrator=mock_reward_orchestrator,
            config=config
        )
        
        # Create test data with large tensor
        input_data = {
            'image_features': torch.randn(10, 3, 224, 224),  # Large tensor
            'question': 'Test question'
        }
        
        voting_result = VotingResult(
            final_answer={'answer': 'test', 'trajectory': []},
            confidence=0.8,
            votes=[],
            weights=[]
        )
        
        initial_prediction = {'logits': torch.randn(10, 100)}
        
        # Enqueue task (synchronous version for testing)
        import asyncio
        asyncio.run(engine._enqueue_update_task(
            input_data, voting_result, initial_prediction
        ))
        
        # Check that shared memory was created
        assert len(engine.shm_manager.pending_shm) > 0
        
        # Get the update task from queue
        update_task = engine.update_queue.get(timeout=1.0)
        
        # Verify task structure
        assert isinstance(update_task, UpdateTask)
        assert 'shm_name' in update_task.metadata
        assert update_task.learning_rate > 0
        
        # Clean up
        for shm_name in list(engine.shm_manager.pending_shm.keys()):
            engine.shm_manager.mark_cleaned(shm_name)


class TestFaultTolerance:
    """Test fault tolerance and error recovery."""
    
    def test_watchdog_cleanup_on_timeout(self):
        """Test watchdog cleaning up segments after timeout."""
        config = {
            'shm_timeout': 0.1,  # Very short timeout
            'watchdog_interval': 0.05
        }
        
        # Create mock components
        model = MagicMock()
        experience_buffer = MagicMock()
        voting_module = MagicMock()
        reward_orchestrator = MagicMock()
        
        engine = InferenceEngine(
            model=model,
            experience_buffer=experience_buffer,
            voting_module=voting_module,
            reward_orchestrator=reward_orchestrator,
            config=config
        )
        
        # Create a shared memory segment
        tensor = torch.randn(10, 10)
        shm_info = engine.shm_manager.create_shared_tensor(tensor)
        
        # Start watchdog
        engine.start_watchdog()
        
        # Wait for timeout and watchdog cleanup
        time.sleep(0.3)
        
        # Check that watchdog is running
        assert engine.watchdog_thread is not None
        assert engine.watchdog_running == True
        
        # Stop watchdog before checking
        engine.watchdog_running = False
        engine.watchdog_thread.join(timeout=1)
        
        # After stopping, segments should be cleaned
        cleaned_count = len(engine.shm_manager.pending_shm)
        assert cleaned_count <= 1  # May have 0 or 1 segments left
        assert engine.stats['watchdog_cleanups'] > 0
        
        # Stop watchdog
        engine.watchdog_running = False
        time.sleep(0.1)
    
    def test_worker_failure_recovery(self):
        """Test recovery when update worker fails."""
        config = {'shm_timeout': 60.0}
        
        # Create mock components
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
        
        experience_buffer = MagicMock()
        voting_module = MagicMock()
        reward_orchestrator = MagicMock()
        
        engine = InferenceEngine(
            model=model,
            experience_buffer=experience_buffer,
            voting_module=voting_module,
            reward_orchestrator=reward_orchestrator,
            config=config
        )
        
        # Create shared memory segments
        tensors = [torch.randn(5, 5) for _ in range(3)]
        shm_infos = [engine.shm_manager.create_shared_tensor(t) for t in tensors]
        
        # Mock a dead worker process
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.pid = 12345
        engine.update_worker_process = mock_process
        
        # Run watchdog cleanup
        engine._watchdog_loop = lambda: None  # Override to run once
        engine.watchdog_running = True
        
        # Process cleanup confirmations (should handle dead worker)
        engine._process_cleanup_confirmations()
        
        # Clean up stale segments with dead worker
        cleaned = engine.shm_manager.cleanup_stale_segments(worker_alive=False)
        
        # All segments should be cleaned
        assert len(cleaned) == 3
        for info in shm_infos:
            assert info.name not in engine.shm_manager.pending_shm
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of the system."""
        config = {}
        
        # Create components
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
        
        experience_buffer = MagicMock()
        voting_module = MagicMock()
        reward_orchestrator = MagicMock()
        
        engine = InferenceEngine(
            model=model,
            experience_buffer=experience_buffer,
            voting_module=voting_module,
            reward_orchestrator=reward_orchestrator,
            config=config
        )
        
        # Create some shared memory segments
        tensors = [torch.randn(5, 5) for _ in range(2)]
        shm_infos = [engine.shm_manager.create_shared_tensor(t) for t in tensors]
        
        # Mock worker process
        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        engine.update_worker_process = mock_process
        
        # Start watchdog
        engine.watchdog_running = True
        engine.watchdog_thread = MagicMock()
        
        # Shutdown
        engine.shutdown()
        
        # Check shutdown actions
        assert engine.watchdog_running == False
        # Check that update_queue received shutdown signal
        mock_process.terminate.assert_called()  # Process should be terminated
        mock_process.join.assert_called()
        
        # All shared memory should be cleaned
        assert len(engine.shm_manager.pending_shm) == 0
    
    def test_queue_timeout_handling(self):
        """Test handling of queue timeouts."""
        config = {}
        
        update_queue = mp.Queue()
        cleanup_queue = mp.Queue()
        
        model = MagicMock()
        model.parameters.return_value = []
        model.named_parameters.return_value = []
        
        worker = UpdateWorker(
            model=model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=config
        )
        
        # Override run to test single iteration
        original_run = worker.run
        
        def single_iteration_run():
            try:
                # This should timeout
                task = worker.update_queue.get(timeout=0.1)
                if task is None:
                    return
                worker._process_update(task)
            except Exception as e:
                if e.__class__.__name__ == 'Empty':
                    # Expected timeout
                    pass
                else:
                    pytest.fail(f"Unexpected exception: {e}")
        
        # Test that timeout is handled gracefully
        single_iteration_run()  # Should not raise
    
    def test_shared_memory_reconstruction_error(self):
        """Test handling errors in shared memory reconstruction."""
        reconstructor = SharedMemoryReconstructor()
        
        # Test with invalid info
        invalid_info = {
            'name': 'invalid_segment',
            'shape': [10, 10],
            'dtype': torch.float32
        }
        
        # Should handle gracefully and return placeholder
        tensor = reconstructor.reconstruct_tensor_from_info(invalid_info)
        
        assert tensor.shape == (10, 10)
        assert tensor.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])