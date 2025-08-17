"""
Test Suite for Update Worker Module

Tests the asynchronous model update worker with the three-tiered safety system:
1. Behavioral Guardrail: KL divergence penalty with dynamic beta adjustment
2. Magnitude Guardrail: Gradient clipping
3. Temporal Guardrail: EMA smoothing with atomic synchronization
"""

import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import numpy as np
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from core.engine.update_worker import UpdateWorker, KLConfig, SharedMemoryReconstructor
from core.data_structures import UpdateTask, Experience, Trajectory, Action, ActionType
from core.modules.reward_shaping import RewardOrchestrator


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, input_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, input_ids=None, images=None, labels=None):
        """Simple forward pass."""
        if images is not None:
            # Use the actual image features if provided
            x = images
        elif input_ids is not None:
            x = torch.randn(1, 512)  # Placeholder with correct size
        else:
            x = torch.randn(1, 512)  # Default with correct size
        
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # Simple loss calculation
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.view(-1))
        else:
            loss = torch.tensor(0.0)
        
        # Return object with loss and logits attributes
        output = MagicMock()
        output.loss = loss
        output.logits = logits
        return output


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture
def update_queues():
    """Create queues for testing."""
    update_queue = Queue(maxsize=10)
    cleanup_queue = Queue(maxsize=10)
    return update_queue, cleanup_queue


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'kl_beta_update_mode': 'auto',
        'kl_initial_beta': 0.01,
        'kl_target_kl': 0.05,
        'kl_tolerance': 0.01,
        'kl_beta_increase_factor': 1.2,
        'kl_beta_decrease_factor': 1.2,
        'kl_min_beta': 1e-4,
        'kl_max_beta': 1.0,
        'gradient_clip_norm': 1.0,
        'use_ema': True,
        'ema_decay': 0.999,
        'sync_frequency': 10,
        'optimizer': 'adamw',
        'base_learning_rate': 1e-5
    }


@pytest.fixture
def update_worker(dummy_model, update_queues, test_config):
    """Create an update worker for testing."""
    update_queue, cleanup_queue = update_queues
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_config['model_save_path'] = tmpdir
        
        worker = UpdateWorker(
            model=dummy_model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=test_config,
            model_save_path=tmpdir
        )
        
        yield worker


@pytest.fixture
def sample_experience():
    """Create a sample experience for testing."""
    # Create a simple trajectory
    action = Action(
        type=ActionType.VISUAL_OPERATION,
        operation="SEGMENT_OBJECT_AT",
        arguments={"x": 100, "y": 200}
    )
    
    trajectory = Trajectory(actions=[action])
    
    return Experience(
        experience_id="test-exp-001",
        image_features=torch.randn(1, 512),
        question_text="What is in this image?",
        trajectory=trajectory,
        model_confidence=0.85
    )


@pytest.fixture
def sample_update_task(sample_experience):
    """Create a sample update task for testing."""
    return UpdateTask(
        task_id="test-task-001",
        experience=sample_experience,
        reward_tensor=torch.tensor(0.8),
        learning_rate=1e-5,
        original_logits=torch.randn(1, 10)
    )


class TestKLConfig:
    """Test KL configuration validation."""
    
    def test_valid_config(self):
        """Test creating a valid KL configuration."""
        config = KLConfig(
            beta_update_mode="auto",
            initial_beta=0.01,
            target_kl=0.05
        )
        assert config.beta_update_mode == "auto"
        assert config.initial_beta == 0.01
        assert config.target_kl == 0.05
    
    def test_invalid_mode(self):
        """Test invalid beta update mode."""
        with pytest.raises(ValueError, match="beta_update_mode must be"):
            KLConfig(beta_update_mode="invalid")
    
    def test_invalid_target_kl(self):
        """Test invalid target KL."""
        with pytest.raises(ValueError, match="target_kl must be positive"):
            KLConfig(target_kl=-0.1)
    
    def test_invalid_beta_bounds(self):
        """Test invalid beta bounds."""
        with pytest.raises(ValueError, match="min_beta must be <= max_beta"):
            KLConfig(min_beta=1.0, max_beta=0.1)


class TestSharedMemoryReconstructor:
    """Test shared memory tensor reconstruction."""
    
    def test_reconstruct_tensor(self):
        """Test reconstructing tensor from shared memory info."""
        shm_info = {
            'name': 'test_shm',
            'shape': [2, 3, 4],
            'dtype': torch.float32
        }
        
        tensor = SharedMemoryReconstructor.reconstruct_tensor_from_info(shm_info)
        
        assert tensor.shape == (2, 3, 4)
        assert tensor.dtype == torch.float32
    
    def test_reconstruct_with_custom_dtype(self):
        """Test reconstruction with custom dtype."""
        shm_info = {
            'name': 'test_shm',
            'shape': [10],
            'dtype': torch.int64
        }
        
        tensor = SharedMemoryReconstructor.reconstruct_tensor_from_info(shm_info)
        
        assert tensor.shape == (10,)
        assert tensor.dtype == torch.int64


class TestUpdateWorker:
    """Test the UpdateWorker class."""
    
    def test_initialization(self, update_worker):
        """Test worker initialization."""
        assert update_worker.model is not None
        assert update_worker.optimizer is not None
        assert update_worker.kl_config.beta_update_mode == "auto"
        assert update_worker.current_beta == 0.01
        assert update_worker.use_ema is True
        assert update_worker.ema_model is not None
    
    def test_create_ema_model(self, update_worker):
        """Test EMA model creation."""
        ema_model = update_worker._create_ema_model()
        
        assert ema_model is not None
        # Check that gradients are disabled
        for param in ema_model.parameters():
            assert param.requires_grad is False
    
    def test_create_optimizer(self, update_worker):
        """Test optimizer creation."""
        optimizer = update_worker._create_optimizer()
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Check learning rate
        for param_group in optimizer.param_groups:
            assert param_group['lr'] == 1e-5
    
    def test_calculate_kl_penalty(self, update_worker):
        """Test KL divergence penalty calculation."""
        current_logits = torch.randn(2, 10)
        original_logits = torch.randn(2, 10)
        
        kl_penalty, kl_div = update_worker._calculate_kl_penalty(
            current_logits, original_logits
        )
        
        assert kl_penalty.item() >= 0  # KL divergence is non-negative
        assert kl_div.item() >= 0
        assert torch.allclose(kl_penalty, kl_div)  # Should be equal
    
    def test_calculate_kl_penalty_no_original(self, update_worker):
        """Test KL penalty with no original logits."""
        current_logits = torch.randn(2, 10)
        
        kl_penalty, kl_div = update_worker._calculate_kl_penalty(
            current_logits, None
        )
        
        assert kl_penalty.item() == 0.0
        assert kl_div.item() == 0.0
    
    def test_update_ema_model(self, update_worker):
        """Test EMA model update."""
        if not update_worker.use_ema:
            pytest.skip("EMA not enabled")
        
        # Get initial EMA parameters
        initial_params = [p.clone() for p in update_worker.ema_model.parameters()]
        
        # Modify main model parameters
        for param in update_worker.model.parameters():
            param.data += 0.1
        
        # Update EMA
        update_worker._update_ema_model()
        
        # Check that EMA parameters changed but not too much
        for initial, ema_param in zip(initial_params, update_worker.ema_model.parameters()):
            assert not torch.allclose(initial, ema_param)  # Should change
            diff = torch.abs(ema_param - initial).mean()
            assert diff < 0.01  # Should change slowly due to decay
    
    def test_update_kl_tracking(self, update_worker):
        """Test KL divergence tracking."""
        # Add some KL values
        for kl in [0.03, 0.04, 0.05, 0.06, 0.07]:
            update_worker._update_kl_tracking(kl)
        
        assert len(update_worker.kl_history) == 5
        assert update_worker.stats['mean_kl'] == pytest.approx(0.05, rel=1e-2)
    
    def test_adjust_beta_increase(self, update_worker):
        """Test beta adjustment when KL is too high."""
        # Set up high KL scenario
        update_worker.kl_history = [0.08] * 20  # High KL values
        update_worker.stats['mean_kl'] = 0.08
        update_worker.current_beta = 0.01
        
        update_worker._adjust_beta()
        
        # Beta should increase
        assert update_worker.current_beta > 0.01
        assert update_worker.current_beta == pytest.approx(0.012, rel=1e-2)
    
    def test_adjust_beta_decrease(self, update_worker):
        """Test beta adjustment when KL is too low."""
        # Set up low KL scenario
        update_worker.kl_history = [0.02] * 20  # Low KL values
        update_worker.stats['mean_kl'] = 0.02
        update_worker.current_beta = 0.01
        
        update_worker._adjust_beta()
        
        # Beta should decrease
        assert update_worker.current_beta < 0.01
        assert update_worker.current_beta == pytest.approx(0.0083, rel=1e-2)
    
    def test_adjust_beta_bounds(self, update_worker):
        """Test beta adjustment respects bounds."""
        # Test max bound
        update_worker.kl_history = [0.5] * 20  # Very high KL
        update_worker.stats['mean_kl'] = 0.5
        update_worker.current_beta = 0.9
        
        update_worker._adjust_beta()
        assert update_worker.current_beta <= update_worker.kl_config.max_beta
        
        # Test min bound
        update_worker.kl_history = [0.001] * 20  # Very low KL
        update_worker.stats['mean_kl'] = 0.001
        update_worker.current_beta = 0.0002
        
        update_worker._adjust_beta()
        assert update_worker.current_beta >= update_worker.kl_config.min_beta
    
    def test_save_ema_snapshot(self, update_worker, tmp_path):
        """Test saving EMA model snapshot."""
        update_worker.model_save_path = tmp_path
        
        # Save snapshot
        update_worker._save_ema_snapshot()
        
        # Check files exist
        snapshot_files = list(tmp_path.glob("ema_model_snapshot.v*.pt"))
        assert len(snapshot_files) == 1
        
        # Check pointer file
        pointer_file = tmp_path / "latest_model_version.txt"
        assert pointer_file.exists()
        
        # Verify pointer content
        with open(pointer_file, 'r') as f:
            content = f.read()
            assert "ema_model_snapshot.v1.pt" in content
    
    def test_cleanup_old_snapshots(self, update_worker, tmp_path):
        """Test cleanup of old model snapshots."""
        update_worker.model_save_path = tmp_path
        
        # Create multiple snapshots
        for i in range(5):
            update_worker.model_version = i + 1
            update_worker._save_ema_snapshot()
        
        # Should keep only last 3
        snapshot_files = list(tmp_path.glob("ema_model_snapshot.v*.pt"))
        assert len(snapshot_files) == 3
        
        # Check that latest versions are kept
        versions = [int(f.stem.split('.v')[1].split('.')[0]) for f in snapshot_files]
        assert sorted(versions) == [3, 4, 5]
    
    @patch('core.engine.update_worker.logger')
    def test_process_update_success(self, mock_logger, update_worker, sample_update_task):
        """Test successful update processing."""
        # Process the update
        update_worker._process_update(sample_update_task)
        
        # Check statistics
        assert update_worker.stats['total_updates'] == 1
        assert update_worker.stats['successful_updates'] == 1
        assert update_worker.stats['failed_updates'] == 0
        
        # Check that task was marked as processed
        assert sample_update_task.processed_at is not None
    
    @patch('core.engine.update_worker.logger')
    def test_process_update_high_kl_skip(self, mock_logger, update_worker, sample_update_task):
        """Test skipping update when KL is too high."""
        # Mock high KL divergence
        with patch.object(update_worker, '_calculate_kl_penalty') as mock_kl:
            mock_kl.return_value = (torch.tensor(0.2), torch.tensor(0.2))  # Very high KL
            
            update_worker._process_update(sample_update_task)
            
            # Should skip the update
            assert update_worker.stats['failed_updates'] == 1
            assert update_worker.stats['successful_updates'] == 0
    
    def test_process_update_with_shared_memory(self, update_worker):
        """Test processing update with shared memory."""
        # Create task with shared memory info
        experience = Experience(
            experience_id="test-shm-exp",
            image_features=None,
            question_text="Test question",
            trajectory=Trajectory(),
            model_confidence=0.9
        )
        
        experience.metadata = {
            'shm_info': {
                'name': 'test_shm_segment',
                'shape': [1, 512],
                'dtype': torch.float32
            }
        }
        
        task = UpdateTask(
            task_id="test-shm-task",
            experience=experience,
            reward_tensor=torch.tensor(0.5),
            learning_rate=1e-5,
            original_logits=torch.randn(1, 10)
        )
        
        # Process the update
        update_worker._process_update(task)
        
        # Should handle shared memory reconstruction
        assert update_worker.stats['total_updates'] == 1
    
    def test_log_update(self, update_worker, sample_update_task, tmp_path):
        """Test update logging."""
        update_worker.model_save_path = tmp_path
        update_worker.update_log_path = tmp_path / "update_audit.log"
        update_worker.contribution_log_path = tmp_path / "update_contribution.jsonl"
        
        # Initialize logs
        update_worker._init_logs()
        
        # Log an update
        update_worker._log_update(
            task=sample_update_task,
            rl_loss=0.5,
            kl_div=0.03,
            total_loss=0.53,
            grad_norm=0.8,
            duration=1.2
        )
        
        # Check audit log
        assert update_worker.update_log_path.exists()
        with open(update_worker.update_log_path, 'r') as f:
            content = f.read()
            assert "test-task-001" in content
            assert "Loss: 0.5300" in content
            assert "KL: 0.0300" in content
        
        # Check contribution log
        assert update_worker.contribution_log_path.exists()
        with open(update_worker.contribution_log_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Metadata + update entry
            
            # Parse update entry
            update_entry = json.loads(lines[1])
            assert update_entry['type'] == 'update'
            assert update_entry['task_id'] == 'test-task-001'
            assert update_entry['losses']['total_loss'] == 0.53
    
    def test_get_statistics(self, update_worker):
        """Test getting worker statistics."""
        # Modify some stats
        update_worker.stats['total_updates'] = 10
        update_worker.stats['successful_updates'] = 8
        update_worker.stats['failed_updates'] = 2
        update_worker.stats['mean_kl'] = 0.045
        
        stats = update_worker.get_statistics()
        
        assert stats['total_updates'] == 10
        assert stats['successful_updates'] == 8
        assert stats['failed_updates'] == 2
        assert stats['mean_kl'] == 0.045
        assert stats['current_beta'] == update_worker.current_beta
    
    def test_shutdown(self, update_worker, tmp_path):
        """Test graceful shutdown."""
        update_worker.model_save_path = tmp_path
        
        # Modify some stats
        update_worker.stats['total_updates'] = 5
        
        # Shutdown
        update_worker.shutdown()
        
        # Check final stats file
        stats_file = tmp_path / "final_stats.json"
        assert stats_file.exists()
        
        with open(stats_file, 'r') as f:
            final_stats = json.load(f)
            assert final_stats['stats']['total_updates'] == 5
            assert 'kl_config' in final_stats
            assert 'final_beta' in final_stats


class TestIntegration:
    """Integration tests for the update worker."""
    
    @pytest.mark.slow
    def test_worker_queue_processing(self, dummy_model, update_queues, test_config):
        """Test worker processing tasks from queue."""
        update_queue, cleanup_queue = update_queues
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['model_save_path'] = tmpdir
            
            # Create worker
            worker = UpdateWorker(
                model=dummy_model,
                update_queue=update_queue,
                cleanup_confirmation_queue=cleanup_queue,
                config=test_config,
                model_save_path=tmpdir
            )
            
            # Create and enqueue tasks
            for i in range(3):
                experience = Experience(
                    experience_id=f"exp-{i}",
                    image_features=torch.randn(1, 512),
                    question_text=f"Question {i}",
                    trajectory=Trajectory(),
                    model_confidence=0.7 + i * 0.1
                )
                
                task = UpdateTask(
                    task_id=f"task-{i}",
                    experience=experience,
                    reward_tensor=torch.tensor(0.5 + i * 0.1),
                    learning_rate=1e-5,
                    original_logits=torch.randn(1, 10)
                )
                
                update_queue.put(task)
            
            # Process tasks
            for _ in range(3):
                task = update_queue.get(timeout=1.0)
                worker._process_update(task)
            
            # Check statistics
            assert worker.stats['total_updates'] == 3
            assert worker.stats['successful_updates'] == 3
    
    @pytest.mark.slow
    def test_ema_synchronization(self, dummy_model, update_queues, test_config):
        """Test EMA model synchronization."""
        update_queue, cleanup_queue = update_queues
        test_config['sync_frequency'] = 2  # Sync every 2 updates
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['model_save_path'] = tmpdir
            
            worker = UpdateWorker(
                model=dummy_model,
                update_queue=update_queue,
                cleanup_confirmation_queue=cleanup_queue,
                config=test_config,
                model_save_path=tmpdir
            )
            
            # Process multiple updates
            for i in range(5):
                experience = Experience(
                    experience_id=f"exp-{i}",
                    image_features=torch.randn(1, 512),
                    question_text=f"Question {i}",
                    trajectory=Trajectory(),
                    model_confidence=0.8
                )
                
                task = UpdateTask(
                    task_id=f"task-{i}",
                    experience=experience,
                    reward_tensor=torch.tensor(0.6),
                    learning_rate=1e-5,
                    original_logits=torch.randn(1, 10)
                )
                
                # Mock the run method behavior
                worker._process_update(task)
                
                # Check for sync
                if worker.updates_since_sync >= worker.sync_frequency:
                    worker._save_ema_snapshot()
                    worker.updates_since_sync = 0
            
            # Check that snapshots were created
            snapshot_files = list(Path(tmpdir).glob("ema_model_snapshot.v*.pt"))
            assert len(snapshot_files) >= 2  # Should have at least 2 snapshots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])