"""
Simple test file to achieve good coverage for persistence_adapter.py
Focuses on FilePersistenceAdapter which is easier to test without LMDB dependency.
"""

import unittest
import tempfile
import shutil
import os
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.persistence_adapter import (
    PersistenceAdapter,
    FilePersistenceAdapter,
    create_persistence_adapter
)


class TestFilePersistenceAdapter(unittest.TestCase):
    """Test FilePersistenceAdapter comprehensively."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = FilePersistenceAdapter(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        # Check directories were created
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(self.adapter.snapshot_dir.exists())
        
        # Check file paths are set
        self.assertEqual(self.adapter.experience_wal.name, "experience_data.wal")
        self.assertEqual(self.adapter.operations_wal.name, "index_operations.wal")
    
    def test_write_and_read_experience(self):
        """Test writing and reading experiences."""
        # Write an experience
        exp_data = {"user_input": "test", "model_response": "response"}
        result = self.adapter.write_experience("exp1", exp_data)
        self.assertTrue(result)
        
        # Read all experiences
        experiences = self.adapter.read_all_experiences()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0]["user_input"], "test")
    
    def test_write_and_read_operations(self):
        """Test writing and reading operations."""
        # Write operations
        op1 = {"op": "add", "key": "key1"}
        op2 = {"op": "remove", "key": "key2"}
        
        self.assertTrue(self.adapter.write_operation(op1))
        self.assertTrue(self.adapter.write_operation(op2))
        
        # Read all operations - includes timestamps
        operations = self.adapter.read_all_operations()
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0]["op"], "add")
        self.assertEqual(operations[1]["op"], "remove")
        self.assertIn("timestamp", operations[0])
    
    def test_save_and_load_snapshot(self):
        """Test saving and loading snapshots."""
        # Save a snapshot
        snapshot_data = {
            "experiences": [{"exp": 1}, {"exp": 2}],
            "metadata": {"version": "1.0"}
        }
        result = self.adapter.save_snapshot(snapshot_data)
        self.assertTrue(result)
        
        # Load the snapshot
        loaded = self.adapter.load_snapshot()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["metadata"]["version"], "1.0")
        self.assertEqual(len(loaded["experiences"]), 2)
    
    def test_truncate_logs(self):
        """Test truncating WAL logs."""
        # Write some data
        self.adapter.write_experience("exp1", {"data": "test"})
        self.adapter.write_operation({"op": "test"})
        
        # Verify files have content
        self.assertGreater(self.adapter.experience_wal.stat().st_size, 0)
        self.assertGreater(self.adapter.operations_wal.stat().st_size, 0)
        
        # Truncate logs
        result = self.adapter.truncate_logs()
        self.assertTrue(result)
        
        # Verify files are empty
        self.assertEqual(self.adapter.experience_wal.stat().st_size, 0)
        self.assertEqual(self.adapter.operations_wal.stat().st_size, 0)
    
    def test_cleanup_old_snapshots(self):
        """Test cleaning up old snapshots."""
        # Create multiple snapshot files
        for i in range(5):
            snapshot_path = self.adapter.snapshot_dir / f"snapshot_2024010{i}_120000.pkl"
            with open(snapshot_path, 'wb') as f:
                pickle.dump({"data": f"snapshot_{i}"}, f)
        
        # Clean up keeping only 3
        self.adapter._cleanup_old_snapshots(keep=3)
        
        # Should have only 3 snapshots remaining
        remaining = list(self.adapter.snapshot_dir.glob("snapshot_*.pkl"))
        self.assertEqual(len(remaining), 3)
        
        # The oldest should be deleted
        remaining_names = [s.name for s in remaining]
        self.assertNotIn("snapshot_20240100_120000.pkl", remaining_names)
        self.assertNotIn("snapshot_20240101_120000.pkl", remaining_names)
    
    def test_close_method(self):
        """Test close method (no-op for file adapter)."""
        # Should not raise any errors
        self.adapter.close()
    
    def test_read_empty_wal_files(self):
        """Test reading when WAL files don't exist."""
        # Read experiences when file doesn't exist
        experiences = self.adapter.read_all_experiences()
        self.assertEqual(experiences, [])
        
        # Read operations when file doesn't exist
        operations = self.adapter.read_all_operations()
        self.assertEqual(operations, [])
    
    def test_load_snapshot_no_files(self):
        """Test loading snapshot when no snapshots exist."""
        result = self.adapter.load_snapshot()
        self.assertIsNone(result)
    
    def test_multiple_snapshots_keeps_latest(self):
        """Test that loading snapshot returns the latest one."""
        # Create multiple snapshots with different timestamps
        snapshots = [
            ("snapshot_20240101_120000.pkl", {"version": "1"}),
            ("snapshot_20240102_120000.pkl", {"version": "2"}),
            ("snapshot_20240103_120000.pkl", {"version": "3"})
        ]
        
        for filename, data in snapshots:
            path = self.adapter.snapshot_dir / filename
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        # Load should return the latest
        loaded = self.adapter.load_snapshot()
        self.assertEqual(loaded["version"], "3")
    
    def test_concurrent_writes(self):
        """Test that writes are thread-safe with lock."""
        import threading
        import time
        
        results = []
        
        def write_experience(exp_id):
            result = self.adapter.write_experience(exp_id, {"data": exp_id})
            results.append(result)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_experience, args=(f"exp_{i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All writes should succeed
        self.assertTrue(all(results))
        
        # All experiences should be written
        experiences = self.adapter.read_all_experiences()
        self.assertEqual(len(experiences), 5)
    
    def test_atomic_write_with_temp_file(self):
        """Test atomic write creates and cleans up temp files."""
        test_path = Path(self.temp_dir) / "atomic_test.txt"
        test_data = "test content"
        
        # Perform atomic write
        result = self.adapter._atomic_write(test_path, test_data)
        self.assertTrue(result)
        
        # Verify file exists with correct content
        self.assertTrue(test_path.exists())
        with open(test_path, 'r') as f:
            self.assertEqual(f.read(), test_data)
        
        # Verify no temp files remain
        temp_files = list(Path(self.temp_dir).glob("*.tmp"))
        self.assertEqual(len(temp_files), 0)


class TestFactoryFunction(unittest.TestCase):
    """Test the factory function."""
    
    def test_create_file_adapter(self):
        """Test creating file adapter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = create_persistence_adapter("file", temp_dir)
            self.assertIsInstance(adapter, FilePersistenceAdapter)
            adapter.close()
    
    def test_create_unknown_adapter_raises(self):
        """Test that unknown adapter type raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError) as ctx:
                create_persistence_adapter("unknown", temp_dir)
            self.assertIn("Unknown adapter type", str(ctx.exception))


class TestAbstractBase(unittest.TestCase):
    """Test the abstract base class."""
    
    def test_cannot_instantiate_abstract_base(self):
        """Test that abstract base cannot be instantiated."""
        with self.assertRaises(TypeError):
            PersistenceAdapter()
    
    def test_all_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = [
            'write_experience',
            'write_operation',
            'read_all_experiences',
            'read_all_operations',
            'save_snapshot',
            'load_snapshot',
            'truncate_logs',
            'close'
        ]
        
        for method_name in abstract_methods:
            self.assertTrue(hasattr(PersistenceAdapter, method_name))
            method = getattr(PersistenceAdapter, method_name)
            self.assertTrue(hasattr(method, '__isabstractmethod__'))


if __name__ == '__main__':
    unittest.main(verbosity=2)