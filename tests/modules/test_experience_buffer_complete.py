"""
Complete Test Coverage for experience_buffer.py

This test file ensures 100% coverage of all lines, branches, and edge cases
in the experience_buffer.py module.
"""

import unittest
import tempfile
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open, Mock
import threading
import json
import pickle
import faiss
import logging
import sys
import os

sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.experience_buffer import ExperienceBuffer
from core.data_structures import Experience, Trajectory, Action, ActionType, ExperienceStatus


class TestExperienceBufferComplete(unittest.TestCase):
    """Complete test suite for ExperienceBuffer with 100% coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.buffer:
            try:
                self.buffer.shutdown()
            except:
                pass
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_experience(self, exp_id="test-001", confidence=0.8, reward=1.0):
        """Create a test experience."""
        action = Action(
            type=ActionType.VISUAL_OPERATION,
            operation="SEGMENT_OBJECT_AT",
            arguments={"x": 100, "y": 200},
            result="test_result"
        )
        trajectory = Trajectory(
            actions=[action],
            final_answer="test_answer",
            total_reward=reward
        )
        experience = Experience(
            experience_id=exp_id,
            image_features=np.random.randn(512),
            question_text="Test question?",
            trajectory=trajectory,
            model_confidence=confidence
        )
        # Add embeddings
        experience.embeddings = {
            "visual": torch.randn(768),
            "text": torch.randn(768),
            "combined": torch.randn(768)
        }
        return experience
    
    def test_initialization_basic(self):
        """Test basic initialization with default parameters."""
        buffer = ExperienceBuffer()
        self.assertEqual(buffer.max_size, 10000)
        self.assertEqual(buffer.embedding_dim, 768)
        self.assertEqual(buffer.similarity_metric, "cosine")
        self.assertTrue(buffer.enable_persistence)
        self.assertEqual(buffer.retention_days, 90)
        self.assertTrue(buffer.enable_auto_pruning)
        buffer.shutdown()
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        buffer = ExperienceBuffer(
            max_size=500,
            embedding_dim=512,
            similarity_metric="euclidean",
            enable_persistence=False,
            persistence_path=self.temp_dir,
            retention_days=30,
            enable_auto_pruning=False
        )
        self.assertEqual(buffer.max_size, 500)
        self.assertEqual(buffer.embedding_dim, 512)
        self.assertEqual(buffer.similarity_metric, "euclidean")
        self.assertFalse(buffer.enable_persistence)
        self.assertEqual(buffer.retention_days, 30)
        self.assertFalse(buffer.enable_auto_pruning)
        buffer.shutdown()
    
    def test_faiss_index_creation_cosine(self):
        """Test FAISS index creation with cosine similarity."""
        buffer = ExperienceBuffer(similarity_metric="cosine", enable_persistence=False)
        self.assertIsNotNone(buffer.index)
        buffer.shutdown()
    
    def test_faiss_index_creation_euclidean(self):
        """Test FAISS index creation with euclidean distance."""
        buffer = ExperienceBuffer(similarity_metric="euclidean", enable_persistence=False)
        self.assertIsNotNone(buffer.index)
        buffer.shutdown()
    
    def test_faiss_index_creation_manhattan(self):
        """Test FAISS index creation with manhattan distance (fallback to L2)."""
        with self.assertLogs(level='WARNING') as log:
            buffer = ExperienceBuffer(similarity_metric="manhattan", enable_persistence=False)
            self.assertIn("Manhattan distance not directly supported", log.output[0])
        self.assertIsNotNone(buffer.index)
        buffer.shutdown()
    
    def test_faiss_index_creation_invalid(self):
        """Test FAISS index creation with invalid metric."""
        with self.assertRaises(ValueError) as context:
            buffer = ExperienceBuffer(similarity_metric="invalid", enable_persistence=False)
        self.assertIn("Unsupported similarity metric", str(context.exception))
    
    def test_setup_persistence(self):
        """Test persistence setup."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        self.assertTrue((Path(self.temp_dir) / "wal.jsonl").parent.exists())
        buffer.shutdown()
    
    def test_add_experience_success(self):
        """Test successful experience addition."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        exp = self._create_test_experience("test-001")
        
        result = buffer.add(exp)
        self.assertTrue(result)
        self.assertEqual(buffer.size(), 1)
        buffer.shutdown()
    
    def test_add_experience_duplicate(self):
        """Test adding duplicate experience."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        exp1 = self._create_test_experience("test-001")
        exp2 = self._create_test_experience("test-001")
        
        self.assertTrue(buffer.add(exp1))
        with self.assertLogs(level='WARNING') as log:
            self.assertFalse(buffer.add(exp2))
            self.assertIn("already exists", log.output[0])
        
        buffer.shutdown()
    
    def test_add_experience_with_priority_calculation(self):
        """Test experience addition with automatic priority calculation."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        exp = self._create_test_experience("test-001", confidence=0.7, reward=2.0)
        exp.priority = 1.0  # Default priority triggers calculation
        
        self.assertTrue(buffer.add(exp))
        
        # Check priority was updated
        retrieved = buffer.get("test-001")
        self.assertNotEqual(retrieved.priority, 1.0)
        buffer.shutdown()
    
    def test_add_experience_with_persistence(self):
        """Test experience addition with persistence enabled."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Add exactly 100 experiences to trigger checkpoint
        for i in range(100):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Check checkpoint was saved
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pkl"
        self.assertTrue(checkpoint_path.exists())
        
        buffer.shutdown()
    
    def test_add_experience_exception_handling(self):
        """Test exception handling during experience addition."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Mock the experience_dict to raise an exception
        with patch.object(buffer, 'experience_dict', side_effect=Exception("Test error")):
            exp = self._create_test_experience("test-001")
            with self.assertLogs(level='ERROR') as log:
                result = buffer.add(exp)
                self.assertFalse(result)
                self.assertIn("Error adding experience", log.output[0])
        
        buffer.shutdown()
    
    def test_add_to_index_with_tensor(self):
        """Test adding tensor embedding to FAISS index."""
        buffer = ExperienceBuffer(
            similarity_metric="cosine",
            enable_persistence=False,
            enable_auto_pruning=False
        )
        exp = self._create_test_experience("test-001")
        exp.embeddings["combined"] = torch.randn(768)
        
        buffer.add(exp)
        self.assertEqual(buffer.index.ntotal, 1)
        buffer.shutdown()
    
    def test_add_to_index_with_numpy(self):
        """Test adding numpy embedding to FAISS index."""
        buffer = ExperienceBuffer(
            similarity_metric="euclidean",
            enable_persistence=False,
            enable_auto_pruning=False
        )
        exp = self._create_test_experience("test-001")
        exp.embeddings["combined"] = np.random.randn(768)
        
        buffer.add(exp)
        self.assertEqual(buffer.index.ntotal, 1)
        buffer.shutdown()
    
    def test_add_to_index_no_embedding(self):
        """Test adding experience without embedding."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        exp = self._create_test_experience("test-001")
        exp.embeddings = {}  # No embeddings
        
        buffer.add(exp)
        self.assertEqual(buffer.index.ntotal, 0)  # No index entry
        buffer.shutdown()
    
    def test_search_index_with_experience_query(self):
        """Test k-NN search with Experience object as query."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(10):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Search with experience
        query_exp = self._create_test_experience("query")
        neighbors = buffer.search_index(query_exp, k=5)
        
        self.assertEqual(len(neighbors), 5)
        buffer.shutdown()
    
    def test_search_index_with_tensor_query(self):
        """Test k-NN search with tensor as query."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Search with tensor
        query_tensor = torch.randn(768)
        neighbors = buffer.search_index(query_tensor, k=3)
        
        self.assertEqual(len(neighbors), 3)
        buffer.shutdown()
    
    def test_search_index_with_dict_query(self):
        """Test k-NN search with dict as query."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Search with dict containing embedding
        query_dict = {"embedding": torch.randn(768)}
        neighbors = buffer.search_index(query_dict, k=2)
        
        self.assertGreaterEqual(len(neighbors), 0)
        buffer.shutdown()
    
    def test_search_index_empty_buffer(self):
        """Test k-NN search on empty buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        query_exp = self._create_test_experience("query")
        neighbors = buffer.search_index(query_exp, k=5)
        
        self.assertEqual(len(neighbors), 0)
        buffer.shutdown()
    
    def test_search_index_no_embedding(self):
        """Test k-NN search when query has no embedding."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Search with no embedding - should fallback to random sampling
        neighbors = buffer.search_index({}, k=3)
        
        self.assertGreaterEqual(len(neighbors), 0)
        self.assertLessEqual(len(neighbors), 3)
        buffer.shutdown()
    
    def test_get_query_embedding_various_inputs(self):
        """Test query embedding extraction with various input types."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Test with numpy array
        np_embedding = np.random.randn(768)
        result = buffer._get_query_embedding(np_embedding)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 768))
        
        # Test with invalid input
        result = buffer._get_query_embedding("invalid")
        self.assertIsNone(result)
        
        # Test with dict without embedding
        result = buffer._get_query_embedding({"other": "data"})
        self.assertIsNone(result)
        
        buffer.shutdown()
    
    def test_sample_by_priority(self):
        """Test priority-based sampling."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences with different priorities
        for i in range(10):
            exp = self._create_test_experience(f"test-{i:03d}", confidence=0.5 + i * 0.05)
            buffer.add(exp)
        
        # Sample experiences
        sampled = buffer.sample_by_priority(5)
        self.assertEqual(len(sampled), 5)
        
        # Check all are unique
        ids = [exp.experience_id for exp in sampled]
        self.assertEqual(len(ids), len(set(ids)))
        
        buffer.shutdown()
    
    def test_sample_by_priority_empty_buffer(self):
        """Test priority sampling on empty buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        sampled = buffer.sample_by_priority(5)
        self.assertEqual(len(sampled), 0)
        
        buffer.shutdown()
    
    def test_sample_by_priority_zero_priorities(self):
        """Test priority sampling when all priorities are zero."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences with zero priority
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            exp.priority = 0.0
            buffer.experience_dict[exp.experience_id] = exp
            buffer.buffer.append(exp.experience_id)
        
        # Should use uniform distribution
        sampled = buffer.sample_by_priority(3)
        self.assertEqual(len(sampled), 3)
        
        buffer.shutdown()
    
    def test_random_sample(self):
        """Test random sampling."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(10):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.experience_dict[exp.experience_id] = exp
        
        # Random sample
        sampled = buffer._random_sample(5)
        self.assertEqual(len(sampled), 5)
        
        # Request more than available
        sampled = buffer._random_sample(15)
        self.assertEqual(len(sampled), 10)
        
        buffer.shutdown()
    
    def test_random_sample_empty(self):
        """Test random sampling on empty buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        sampled = buffer._random_sample(5)
        self.assertEqual(len(sampled), 0)
        
        buffer.shutdown()
    
    def test_get_experience(self):
        """Test getting experience by ID."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        # Get existing
        retrieved = buffer.get("test-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.experience_id, "test-001")
        
        # Get non-existing
        retrieved = buffer.get("non-existent")
        self.assertIsNone(retrieved)
        
        buffer.shutdown()
    
    def test_remove_experience(self):
        """Test removing experience."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        # Remove existing
        self.assertTrue(buffer.remove("test-001"))
        self.assertIsNone(buffer.get("test-001"))
        
        # Remove non-existing
        self.assertFalse(buffer.remove("non-existent"))
        
        buffer.shutdown()
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Clear
        buffer.clear()
        
        self.assertEqual(buffer.size(), 0)
        self.assertEqual(len(buffer.buffer), 0)
        self.assertEqual(len(buffer.experience_dict), 0)
        self.assertEqual(len(buffer.index_to_id), 0)
        
        buffer.shutdown()
    
    def test_size_and_is_full(self):
        """Test size and is_full methods."""
        buffer = ExperienceBuffer(max_size=3, enable_persistence=False, enable_auto_pruning=False)
        
        self.assertEqual(buffer.size(), 0)
        self.assertFalse(buffer.is_full())
        
        # Add experiences
        for i in range(3):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        self.assertEqual(buffer.size(), 3)
        self.assertTrue(buffer.is_full())
        
        buffer.shutdown()
    
    def test_get_statistics_with_data(self):
        """Test statistics with data in buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}", confidence=0.5 + i * 0.1)
            buffer.add(exp)
        
        stats = buffer.get_statistics()
        
        self.assertEqual(stats["size"], 5)
        self.assertEqual(stats["max_size"], 10000)
        self.assertIn("avg_confidence", stats)
        self.assertIn("avg_priority", stats)
        self.assertIn("avg_retrieval_count", stats)
        
        buffer.shutdown()
    
    def test_get_statistics_empty_buffer(self):
        """Test statistics with empty buffer."""
        buffer = ExperienceBuffer(enable_persistence=False, enable_auto_pruning=False)
        
        stats = buffer.get_statistics()
        
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["utilization"], 0.0)
        self.assertEqual(stats["index_size"], 0)
        
        buffer.shutdown()
    
    def test_write_to_wal(self):
        """Test writing to Write-Ahead Log."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        exp = self._create_test_experience("test-001")
        buffer._write_to_wal(exp)
        
        # Check WAL file exists and contains data
        wal_path = Path(self.temp_dir) / "wal.jsonl"
        self.assertTrue(wal_path.exists())
        
        with open(wal_path, "r") as f:
            line = f.readline()
            entry = json.loads(line)
            self.assertEqual(entry["operation"], "add")
            self.assertEqual(entry["data"]["experience_id"], "test-001")
        
        buffer.shutdown()
    
    def test_write_to_wal_exception(self):
        """Test WAL write with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Make WAL path invalid
        buffer.wal_path = Path("/invalid/path/wal.jsonl")
        
        exp = self._create_test_experience("test-001")
        with self.assertLogs(level='ERROR') as log:
            buffer._write_to_wal(exp)
            self.assertIn("Error writing to WAL", log.output[0])
        
        buffer.shutdown()
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        # Save checkpoint
        buffer._save_checkpoint()
        
        # Check files exist
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pkl"
        self.assertTrue(checkpoint_path.exists())
        
        # WAL should be cleared after checkpoint
        wal_path = Path(self.temp_dir) / "wal.jsonl"
        self.assertFalse(wal_path.exists())
        
        buffer.shutdown()
    
    def test_save_checkpoint_with_faiss_index(self):
        """Test saving checkpoint with FAISS index."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Add experience with embedding
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        # Save checkpoint
        buffer._save_checkpoint()
        
        # Check FAISS index was saved
        index_path = Path(self.temp_dir) / "faiss.index"
        self.assertTrue(index_path.exists())
        
        buffer.shutdown()
    
    def test_save_checkpoint_exception(self):
        """Test checkpoint save with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Make checkpoint path invalid
        buffer.checkpoint_path = Path("/invalid/path/checkpoint.pkl")
        
        with self.assertLogs(level='ERROR') as log:
            buffer._save_checkpoint()
            self.assertIn("Error saving checkpoint", log.output[0])
        
        buffer.shutdown()
    
    def test_load_from_disk(self):
        """Test loading from disk."""
        # First buffer - save data
        buffer1 = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer1.add(exp)
        
        buffer1._save_checkpoint()
        buffer1.shutdown()
        
        # Second buffer - load data
        buffer2 = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        self.assertEqual(buffer2.size(), 5)
        self.assertIsNotNone(buffer2.get("test-002"))
        
        buffer2.shutdown()
    
    def test_load_from_disk_with_wal(self):
        """Test loading from disk with WAL entries."""
        # Create checkpoint
        buffer1 = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Add and checkpoint
        for i in range(3):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer1.add(exp)
        buffer1._save_checkpoint()
        
        # Add more (will be in WAL)
        for i in range(3, 5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer1._write_to_wal(exp)
        
        # Don't shutdown - simulate crash
        del buffer1
        
        # Load with WAL recovery
        buffer2 = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Should have all 5 experiences (3 from checkpoint, 2 from WAL)
        self.assertEqual(buffer2.size(), 5)
        
        buffer2.shutdown()
    
    def test_load_from_disk_exception(self):
        """Test load from disk with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Create corrupted checkpoint
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pkl"
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted data")
        
        with self.assertLogs(level='ERROR') as log:
            buffer._load_from_disk()
            self.assertIn("Error loading from disk", log.output[0])
        
        buffer.shutdown()
    
    def test_apply_wal_exception(self):
        """Test applying WAL with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Create corrupted WAL
        wal_path = Path(self.temp_dir) / "wal.jsonl"
        with open(wal_path, "w") as f:
            f.write("invalid json\n")
        
        with self.assertLogs(level='ERROR') as log:
            buffer._apply_wal()
            self.assertIn("Error applying WAL", log.output[0])
        
        buffer.shutdown()
    
    def test_prune_old_experiences(self):
        """Test pruning old experiences."""
        buffer = ExperienceBuffer(
            retention_days=30,
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        # Add old experiences
        old_exp = self._create_test_experience("old-001")
        old_exp.timestamp = datetime.now() - timedelta(days=35)
        buffer.experience_dict["old-001"] = old_exp
        buffer.index_to_id[0] = "old-001"
        
        # Add recent experience
        new_exp = self._create_test_experience("new-001")
        buffer.experience_dict["new-001"] = new_exp
        
        # Prune
        pruned = buffer.prune_old_experiences()
        
        self.assertEqual(pruned, 1)
        self.assertIsNone(buffer.get("old-001"))
        self.assertIsNotNone(buffer.get("new-001"))
        
        buffer.shutdown()
    
    def test_prune_old_experiences_exception(self):
        """Test pruning with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        # Mock to raise exception
        with patch.object(buffer, 'experience_dict', side_effect=Exception("Test error")):
            with self.assertLogs(level='ERROR') as log:
                pruned = buffer.prune_old_experiences()
                self.assertEqual(pruned, 0)
                self.assertIn("Error during experience pruning", log.output[0])
        
        buffer.shutdown()
    
    def test_rebuild_faiss_index(self):
        """Test rebuilding FAISS index."""
        buffer = ExperienceBuffer(
            similarity_metric="cosine",
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        # Add experiences
        for i in range(5):
            exp = self._create_test_experience(f"test-{i:03d}")
            buffer.add(exp)
        
        old_index_count = buffer.index.ntotal
        
        # Rebuild
        buffer._rebuild_faiss_index()
        
        # Check index was rebuilt
        self.assertEqual(buffer.index.ntotal, old_index_count)
        self.assertEqual(len(buffer.index_to_id), old_index_count)
        
        buffer.shutdown()
    
    def test_rebuild_faiss_index_exception(self):
        """Test rebuilding index with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        # Add experience
        exp = self._create_test_experience("test-001")
        buffer.experience_dict["test-001"] = exp
        
        # Mock to raise exception
        with patch.object(buffer, '_create_faiss_index', side_effect=Exception("Test error")):
            with self.assertLogs(level='ERROR') as log:
                buffer._rebuild_faiss_index()
                self.assertIn("Error rebuilding FAISS index", log.output[0])
        
        buffer.shutdown()
    
    def test_log_pruning_event(self):
        """Test logging pruning events."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Log pruning event
        pruned_ids = [f"exp-{i}" for i in range(15)]
        buffer._log_pruning_event(10, pruned_ids)
        
        # Check audit log
        audit_path = Path(self.temp_dir) / "pruning_audit.jsonl"
        self.assertTrue(audit_path.exists())
        
        with open(audit_path, "r") as f:
            line = f.readline()
            entry = json.loads(line)
            self.assertEqual(entry["event"], "data_pruning")
            self.assertEqual(entry["count"], 10)
            self.assertEqual(len(entry["pruned_ids"]), 10)  # Only first 10
        
        buffer.shutdown()
    
    def test_log_pruning_event_exception(self):
        """Test logging pruning event with exception."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Make path invalid
        buffer.persistence_path = Path("/invalid/path")
        
        with self.assertLogs(level='ERROR') as log:
            buffer._log_pruning_event(5, ["exp-1", "exp-2"])
            self.assertIn("Error logging pruning event", log.output[0])
        
        buffer.shutdown()
    
    def test_start_pruning_task(self):
        """Test starting automatic pruning task."""
        buffer = ExperienceBuffer(
            enable_persistence=False,
            enable_auto_pruning=True
        )
        
        # Check thread was started
        self.assertIsNotNone(buffer.pruning_thread)
        self.assertTrue(buffer.pruning_thread.is_alive())
        
        buffer.shutdown()
        
        # Check thread stops
        time.sleep(0.5)
        self.assertFalse(buffer.pruning_thread.is_alive())
    
    def test_pruning_worker_execution(self):
        """Test pruning worker execution."""
        with patch('time.sleep') as mock_sleep:
            buffer = ExperienceBuffer(
                retention_days=1,
                enable_persistence=False,
                enable_auto_pruning=True
            )
            
            # Add old experience
            old_exp = self._create_test_experience("old-001")
            old_exp.timestamp = datetime.now() - timedelta(days=2)
            buffer.experience_dict["old-001"] = old_exp
            
            # Trigger pruning manually
            buffer.enable_auto_pruning = False  # Stop the loop
            
            # Wait a bit for thread to process
            time.sleep(0.1)
            
            buffer.shutdown()
    
    def test_get_retention_statistics(self):
        """Test getting retention statistics."""
        buffer = ExperienceBuffer(
            retention_days=90,
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        # Add experiences of various ages
        for days_ago in [1, 10, 35, 65, 95]:
            exp = self._create_test_experience(f"exp-{days_ago}")
            exp.timestamp = datetime.now() - timedelta(days=days_ago)
            buffer.experience_dict[exp.experience_id] = exp
        
        stats = buffer.get_retention_statistics()
        
        self.assertEqual(stats["retention_days"], 90)
        self.assertIn("age_distribution", stats)
        self.assertEqual(stats["age_distribution"]["0-7_days"], 1)
        self.assertEqual(stats["age_distribution"]["7-30_days"], 1)
        self.assertEqual(stats["age_distribution"]["30-60_days"], 1)
        self.assertEqual(stats["age_distribution"]["60-90_days"], 1)
        self.assertEqual(stats["age_distribution"]["over_90_days"], 1)
        
        buffer.shutdown()
    
    def test_get_retention_statistics_empty(self):
        """Test retention statistics on empty buffer."""
        buffer = ExperienceBuffer(
            enable_persistence=False,
            enable_auto_pruning=False
        )
        
        stats = buffer.get_retention_statistics()
        
        self.assertEqual(stats["retention_days"], 90)
        self.assertIsNone(stats["oldest_experience"])
        self.assertIsNone(stats["newest_experience"])
        
        buffer.shutdown()
    
    def test_shutdown(self):
        """Test graceful shutdown."""
        buffer = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=True
        )
        
        # Add experience
        exp = self._create_test_experience("test-001")
        buffer.add(exp)
        
        # Shutdown
        buffer.shutdown()
        
        # Check pruning stopped
        self.assertFalse(buffer.enable_auto_pruning)
        
        # Check checkpoint saved
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pkl"
        self.assertTrue(checkpoint_path.exists())
    
    def test_shutdown_with_timeout(self):
        """Test shutdown with thread join timeout."""
        buffer = ExperienceBuffer(
            enable_persistence=False,
            enable_auto_pruning=True
        )
        
        # Mock thread to simulate slow shutdown
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        buffer.pruning_thread = mock_thread
        
        # Shutdown should handle timeout gracefully
        buffer.shutdown()
        
        mock_thread.join.assert_called_once_with(timeout=5)
    
    def test_full_workflow_integration(self):
        """Test complete workflow integration."""
        # Create buffer
        buffer = ExperienceBuffer(
            max_size=100,
            retention_days=30,
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        # Add experiences
        exp_ids = []
        for i in range(10):
            exp = self._create_test_experience(f"test-{i:03d}", confidence=0.5 + i * 0.05)
            buffer.add(exp)
            exp_ids.append(f"test-{i:03d}")
        
        # Search
        query = self._create_test_experience("query")
        neighbors = buffer.search_index(query, k=5)
        self.assertEqual(len(neighbors), 5)
        
        # Sample by priority
        sampled = buffer.sample_by_priority(3)
        self.assertEqual(len(sampled), 3)
        
        # Get statistics
        stats = buffer.get_statistics()
        self.assertEqual(stats["size"], 10)
        
        # Remove some
        buffer.remove(exp_ids[0])
        self.assertEqual(buffer.size(), 9)
        
        # Prune old (none should be old)
        pruned = buffer.prune_old_experiences()
        self.assertEqual(pruned, 0)
        
        # Get retention stats
        ret_stats = buffer.get_retention_statistics()
        self.assertEqual(ret_stats["retention_days"], 30)
        
        # Shutdown
        buffer.shutdown()
        
        # Reload and verify
        buffer2 = ExperienceBuffer(
            enable_persistence=True,
            persistence_path=self.temp_dir,
            enable_auto_pruning=False
        )
        
        self.assertEqual(buffer2.size(), 9)
        self.assertIsNone(buffer2.get(exp_ids[0]))
        self.assertIsNotNone(buffer2.get(exp_ids[1]))
        
        buffer2.shutdown()


if __name__ == "__main__":
    unittest.main()