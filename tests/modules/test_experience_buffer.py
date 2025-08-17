"""
Comprehensive Unit Tests for Enhanced Experience Buffer

Tests all functionality including:
- Basic operations (add, get, remove)
- Priority-based sampling
- Hybrid k-NN retrieval
- Persistence and recovery
- Concurrency safety
- FAISS backend configuration
- Value tracking and success rates
"""

import unittest
import tempfile
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Queue
import json

from core.data_structures import Experience, Trajectory, Action, ActionType
from core.config_schema import OnlineConfig
from core.modules.experience_buffer_enhanced import EnhancedExperienceBuffer
from core.modules.persistence_adapter import (
    FilePersistenceAdapter,
    LMDBPersistenceAdapter,
    create_persistence_adapter
)


def top_level_add_experiences(config, start_id, count, result_queue):
    """Worker function to add experiences - moved to module level to avoid pickling issues."""
    # Create buffer in child process to avoid pickling issues
    buffer = EnhancedExperienceBuffer(config)
    
    success_count = 0
    for i in range(count):
        # Create test experience manually since we can't pass test instance
        import uuid
        exp_id = f"exp-{start_id}-{i}"
        
        # Create test trajectory
        from core.data_structures import Action, ActionType, Trajectory, Experience
        import numpy as np
        import torch
        
        action = Action(
            type=ActionType.VISUAL_OPERATION,
            operation="SEGMENT_OBJECT_AT",
            arguments={"x": 100, "y": 200},
            result="test"
        )
        trajectory = Trajectory(
            actions=[action],
            final_answer="test_answer",
            total_reward=1.0
        )
        
        # Create experience
        experience = Experience(
            experience_id=exp_id,
            image_features=np.random.randn(512),
            question_text="What is in the image?",
            trajectory=trajectory,
            model_confidence=0.8
        )
        
        # Add embeddings
        visual_embed = np.random.randn(768)
        text_embed = np.random.randn(768)
        experience.set_embedding(torch.from_numpy(visual_embed), "visual")
        experience.set_embedding(torch.from_numpy(text_embed), "text")
        
        if buffer.add(experience):
            success_count += 1
    
    buffer.shutdown()
    result_queue.put(success_count)


class TestExperienceBuffer(unittest.TestCase):
    """Test suite for enhanced experience buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for persistence
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = OnlineConfig()
        self.config.buffer_size = 100
        self.config.persistence_path = self.temp_dir
        self.config.enable_persistence = True
        self.config.persistence_backend = "file"
        self.config.faiss_backend = "cpu"
        self.config.snapshot_interval = 10
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_experience(
        self,
        exp_id: str = None,
        confidence: float = 0.8,
        reward: float = 1.0
    ) -> Experience:
        """Create a test experience."""
        if exp_id is None:
            import uuid
            exp_id = str(uuid.uuid4())
        
        # Create test trajectory
        action = Action(
            type=ActionType.VISUAL_OPERATION,
            operation="SEGMENT_OBJECT_AT",
            arguments={"x": 100, "y": 200},
            result="object_detected",
            confidence=0.9
        )
        
        trajectory = Trajectory(
            actions=[action],
            final_answer="test_answer",
            total_reward=reward
        )
        
        # Create experience
        experience = Experience(
            experience_id=exp_id,
            image_features=np.random.randn(512),
            question_text="What is in the image?",
            trajectory=trajectory,
            model_confidence=confidence
        )
        
        # Add embeddings
        visual_embed = np.random.randn(768)
        text_embed = np.random.randn(768)
        experience.set_embedding(torch.from_numpy(visual_embed), "visual")
        experience.set_embedding(torch.from_numpy(text_embed), "text")
        
        return experience
    
    def test_basic_add_and_get(self):
        """Test basic add and get operations."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Create and add experience
        exp = self._create_test_experience("test-001")
        self.assertTrue(buffer.add(exp))
        
        # Get experience
        retrieved = buffer.get("test-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.experience_id, "test-001")
        
        # Check buffer size
        self.assertEqual(buffer.size(), 1)
        
        buffer.shutdown()
    
    def test_duplicate_prevention(self):
        """Test that duplicate experiences are rejected."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add experience
        exp1 = self._create_test_experience("test-001")
        self.assertTrue(buffer.add(exp1))
        
        # Try to add duplicate
        exp2 = self._create_test_experience("test-001")
        self.assertFalse(buffer.add(exp2))
        
        # Check size is still 1
        self.assertEqual(buffer.size(), 1)
        
        buffer.shutdown()
    
    def test_priority_calculation(self):
        """Test multi-factor priority calculation."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add experiences with different characteristics
        exp1 = self._create_test_experience("exp-1", confidence=0.9, reward=0.5)
        exp2 = self._create_test_experience("exp-2", confidence=0.5, reward=2.0)
        exp3 = self._create_test_experience("exp-3", confidence=0.7, reward=1.0)
        
        buffer.add(exp1)
        buffer.add(exp2)
        buffer.add(exp3)
        
        # Check priorities
        # exp2 should have higher priority (lower confidence, higher reward)
        self.assertGreater(
            buffer.get("exp-2").priority,
            buffer.get("exp-1").priority
        )
        
        buffer.shutdown()
    
    def test_priority_sampling(self):
        """Test priority-based sampling."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add experiences with different priorities
        for i in range(10):
            confidence = 0.5 + i * 0.05  # Varying confidence
            reward = 2.0 - i * 0.1  # Varying reward
            exp = self._create_test_experience(f"exp-{i}", confidence, reward)
            buffer.add(exp)
        
        # Sample experiences
        sampled = buffer.sample_by_priority(5)
        self.assertEqual(len(sampled), 5)
        
        # Check that all sampled are unique
        sampled_ids = [exp.experience_id for exp in sampled]
        self.assertEqual(len(sampled_ids), len(set(sampled_ids)))
        
        buffer.shutdown()
    
    def test_hybrid_embedding_creation(self):
        """Test hybrid embedding creation."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Create experience with both embeddings
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        # Check that combined embedding was created
        retrieved = buffer.get("test-001")
        combined = retrieved.get_embedding("combined")
        self.assertIsNotNone(combined)
        
        # Check weighting
        visual = retrieved.get_embedding("visual").numpy()
        text = retrieved.get_embedding("text").numpy()
        expected = (
            self.config.visual_weight * visual +
            self.config.text_weight * text
        )
        
        np.testing.assert_array_almost_equal(
            combined.numpy(),
            expected,
            decimal=5
        )
        
        buffer.shutdown()
    
    def test_knn_search(self):
        """Test k-NN search functionality."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add multiple experiences
        experiences = []
        for i in range(20):
            exp = self._create_test_experience(f"exp-{i}")
            buffer.add(exp)
            experiences.append(exp)
        
        # Search for neighbors
        query_exp = self._create_test_experience("query")
        neighbors = buffer.search_index(query_exp, k=5)
        
        self.assertEqual(len(neighbors), 5)
        
        # Check that neighbors are from buffer
        neighbor_ids = [n.experience_id for n in neighbors]
        for nid in neighbor_ids:
            self.assertIn(nid, [f"exp-{i}" for i in range(20)])
        
        buffer.shutdown()
    
    def test_value_tracking(self):
        """Test value tracking with retrieval and success counts."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add experiences
        exp_ids = []
        for i in range(5):
            exp = self._create_test_experience(f"exp-{i}")
            buffer.add(exp)
            exp_ids.append(f"exp-{i}")
        
        # Simulate retrievals
        neighbors = buffer.search_index(
            self._create_test_experience("query"),
            k=3
        )
        
        # Update success for retrieved experiences
        retrieved_ids = [n.experience_id for n in neighbors]
        buffer.update_experience_success(retrieved_ids, was_successful=True)
        
        # Check counts
        for exp_id in retrieved_ids:
            exp = buffer.get(exp_id)
            self.assertEqual(exp.retrieval_count, 1)
            self.assertEqual(exp.success_count, 1)
            self.assertEqual(exp.success_rate, 1.0)
        
        # Simulate failed retrieval
        buffer.update_experience_success(retrieved_ids[:1], was_successful=False)
        
        exp = buffer.get(retrieved_ids[0])
        self.assertEqual(exp.retrieval_count, 1)  # Still 1
        self.assertEqual(exp.success_count, 1)  # Still 1 (already updated)
        
        buffer.shutdown()
    
    def test_persistence_save_and_load(self):
        """Test persistence with save and load."""
        # Create buffer and add experiences
        buffer1 = EnhancedExperienceBuffer(self.config)
        
        exp_ids = []
        for i in range(5):
            exp = self._create_test_experience(f"exp-{i}")
            buffer1.add(exp)
            exp_ids.append(f"exp-{i}")
        
        # Shutdown (should save)
        buffer1.shutdown()
        
        # Create new buffer (should load)
        buffer2 = EnhancedExperienceBuffer(self.config)
        
        # Check that experiences were loaded
        self.assertEqual(buffer2.size(), 5)
        
        for exp_id in exp_ids:
            exp = buffer2.get(exp_id)
            self.assertIsNotNone(exp)
            self.assertEqual(exp.experience_id, exp_id)
        
        buffer2.shutdown()
    
    def test_wal_recovery(self):
        """Test Write-Ahead Log recovery after crash."""
        # Create buffer and add experiences
        buffer1 = EnhancedExperienceBuffer(self.config)
        
        # Add some experiences
        for i in range(3):
            exp = self._create_test_experience(f"exp-{i}")
            buffer1.add(exp)
        
        # Force snapshot
        buffer1._save_snapshot()
        
        # Add more experiences (these will be in WAL)
        for i in range(3, 6):
            exp = self._create_test_experience(f"exp-{i}")
            buffer1.add(exp)
        
        # Simulate crash (no graceful shutdown)
        # Just delete the buffer without shutdown
        del buffer1
        
        # Create new buffer (should recover from snapshot + WAL)
        buffer2 = EnhancedExperienceBuffer(self.config)
        
        # Check all experiences are recovered
        self.assertEqual(buffer2.size(), 6)
        
        for i in range(6):
            exp = buffer2.get(f"exp-{i}")
            self.assertIsNotNone(exp)
        
        buffer2.shutdown()
    
    def test_concurrent_writes(self):
        """Test concurrent write safety."""
        # Create multiple processes
        processes = []
        result_queue = Queue()
        
        for i in range(3):
            p = Process(
                target=top_level_add_experiences,
                args=(self.config, i * 10, 10, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Wait for processes
        for p in processes:
            p.join()
        
        # Check results
        total_added = 0
        while not result_queue.empty():
            total_added += result_queue.get()
        
        # Each process should add 10 experiences, total should be 30
        self.assertEqual(total_added, 30)
    
    def test_faiss_backend_fallback(self):
        """Test FAISS backend fallback from GPU to CPU."""
        # Configure for GPU with fallback
        config = OnlineConfig()
        config.buffer_size = 100
        config.persistence_path = self.temp_dir
        config.enable_persistence = False  # Disable for this test
        config.faiss_backend = "gpu"
        config.faiss_use_gpu_fallback = True
        
        # This should fallback to CPU if GPU not available
        buffer = EnhancedExperienceBuffer(config)
        
        # Add experience and search
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        neighbors = buffer.search_index(exp, k=1)
        self.assertEqual(len(neighbors), 1)
        
        buffer.shutdown()
    
    def test_lmdb_persistence(self):
        """Test LMDB persistence adapter."""
        try:
            import lmdb
        except ImportError:
            self.skipTest("LMDB not installed")
        
        # Configure for LMDB
        config = OnlineConfig()
        config.buffer_size = 100
        config.persistence_path = self.temp_dir
        config.enable_persistence = True
        config.persistence_backend = "lmdb"
        
        # Create buffer with LMDB
        buffer = EnhancedExperienceBuffer(config)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"exp-{i}")
            buffer.add(exp)
        
        buffer.shutdown()
        
        # Create new buffer (should load from LMDB)
        buffer2 = EnhancedExperienceBuffer(config)
        self.assertEqual(buffer2.size(), 5)
        
        buffer2.shutdown()
    
    def test_buffer_overflow(self):
        """Test buffer behavior when full."""
        # Small buffer for testing
        config = OnlineConfig()
        config.buffer_size = 5
        config.persistence_path = self.temp_dir
        config.enable_persistence = False
        
        buffer = EnhancedExperienceBuffer(config)
        
        # Add more than buffer size
        for i in range(10):
            exp = self._create_test_experience(f"exp-{i}")
            buffer.add(exp)
        
        # Buffer should maintain max size
        self.assertEqual(buffer.size(), 5)
        
        # Oldest experiences should be evicted
        self.assertIsNone(buffer.get("exp-0"))
        self.assertIsNotNone(buffer.get("exp-9"))
        
        buffer.shutdown()
    
    def test_statistics(self):
        """Test statistics calculation."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Add experiences with varying characteristics
        for i in range(10):
            exp = self._create_test_experience(
                f"exp-{i}",
                confidence=0.5 + i * 0.05,
                reward=1.0 + i * 0.1
            )
            buffer.add(exp)
        
        # Get statistics
        stats = buffer.get_statistics()
        
        self.assertEqual(stats["size"], 10)
        self.assertEqual(stats["total_additions"], 10)
        self.assertGreater(stats["avg_confidence"], 0)
        self.assertGreater(stats["avg_priority"], 0)
        
        buffer.shutdown()
    
    def test_index_rebuild(self):
        """Test asynchronous index rebuild."""
        # Configure with small snapshot interval
        config = OnlineConfig()
        config.buffer_size = 100
        config.persistence_path = self.temp_dir
        config.enable_persistence = True
        config.snapshot_interval = 5
        
        buffer = EnhancedExperienceBuffer(config)
        
        # Add experiences to trigger rebuild
        for i in range(6):
            exp = self._create_test_experience(f"exp-{i}")
            buffer.add(exp)
            time.sleep(0.1)  # Small delay
        
        # Wait for rebuild
        time.sleep(1.0)
        
        # Index should be rebuilt
        self.assertGreaterEqual(buffer.index.ntotal, 6)
        
        buffer.shutdown()
    
    def test_empty_buffer_operations(self):
        """Test operations on empty buffer."""
        buffer = EnhancedExperienceBuffer(self.config)
        
        # Test get on empty buffer
        self.assertIsNone(buffer.get("non-existent"))
        
        # Test search on empty buffer
        query = self._create_test_experience("query")
        neighbors = buffer.search_index(query, k=5)
        self.assertEqual(len(neighbors), 0)
        
        # Test sampling on empty buffer
        sampled = buffer.sample_by_priority(5)
        self.assertEqual(len(sampled), 0)
        
        # Test statistics on empty buffer
        stats = buffer.get_statistics()
        self.assertEqual(stats["size"], 0)
        
        buffer.shutdown()


class TestPersistenceAdapters(unittest.TestCase):
    """Test suite for persistence adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_file_adapter_basic(self):
        """Test file-based persistence adapter."""
        adapter = FilePersistenceAdapter(self.temp_dir)
        
        # Write experience
        exp_data = {
            "experience_id": "test-001",
            "question_text": "test question",
            "model_confidence": 0.8
        }
        self.assertTrue(adapter.write_experience("test-001", exp_data))
        
        # Write operation
        op_data = {
            "op": "add",
            "experience_id": "test-001"
        }
        self.assertTrue(adapter.write_operation(op_data))
        
        # Read back
        experiences = adapter.read_all_experiences()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]["experience_id"], "test-001")
        
        operations = adapter.read_all_operations()
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["op"], "add")
        
        adapter.close()
    
    def test_file_adapter_snapshot(self):
        """Test file adapter snapshot functionality."""
        adapter = FilePersistenceAdapter(self.temp_dir)
        
        # Save snapshot
        snapshot_data = {
            "experiences": {"test-001": {"data": "test"}},
            "statistics": {"count": 1}
        }
        self.assertTrue(adapter.save_snapshot(snapshot_data))
        
        # Load snapshot
        loaded = adapter.load_snapshot()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["statistics"]["count"], 1)
        
        # Truncate logs
        self.assertTrue(adapter.truncate_logs())
        
        adapter.close()
    
    def test_lmdb_adapter_basic(self):
        """Test LMDB persistence adapter."""
        try:
            adapter = LMDBPersistenceAdapter(self.temp_dir)
        except ImportError:
            self.skipTest("LMDB not installed")
        
        # Write experience
        exp_data = {
            "experience_id": "test-001",
            "question_text": "test question",
            "model_confidence": 0.8
        }
        self.assertTrue(adapter.write_experience("test-001", exp_data))
        
        # Write operations
        for i in range(3):
            op_data = {
                "op": "add",
                "experience_id": f"test-{i:03d}"
            }
            self.assertTrue(adapter.write_operation(op_data))
        
        # Read back
        experiences = adapter.read_all_experiences()
        self.assertEqual(len(experiences), 1)
        
        operations = adapter.read_all_operations()
        self.assertEqual(len(operations), 3)
        
        # Operations should be in order
        for i, op in enumerate(operations):
            self.assertEqual(op["experience_id"], f"test-{i:03d}")
        
        adapter.close()
    
    def test_adapter_factory(self):
        """Test persistence adapter factory."""
        # Create file adapter
        file_adapter = create_persistence_adapter("file", self.temp_dir)
        self.assertIsInstance(file_adapter, FilePersistenceAdapter)
        file_adapter.close()
        
        # Create LMDB adapter
        try:
            lmdb_adapter = create_persistence_adapter("lmdb", self.temp_dir)
            self.assertIsInstance(lmdb_adapter, LMDBPersistenceAdapter)
            lmdb_adapter.close()
        except ImportError:
            pass  # LMDB not installed
        
        # Test invalid adapter type
        with self.assertRaises(ValueError):
            create_persistence_adapter("invalid", self.temp_dir)


if __name__ == "__main__":
    unittest.main()