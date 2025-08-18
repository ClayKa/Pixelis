"""
Tests for persistence_adapter.py to achieve 100% coverage

This test file targets the 66 missing statements in persistence_adapter.py
to achieve complete code coverage, including all error handling, edge cases,
and both FilePersistenceAdapter and LMDBPersistenceAdapter implementations.
"""

import unittest
import tempfile
import shutil
import os
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
from datetime import datetime
import sys
import threading

# Add project root to path
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.persistence_adapter import (
    PersistenceAdapter,
    FilePersistenceAdapter,
    LMDBPersistenceAdapter,
    create_persistence_adapter
)


class TestFilePersistenceAdapterMissingCoverage(unittest.TestCase):
    """Test FilePersistenceAdapter missing coverage lines."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = FilePersistenceAdapter(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_atomic_write_success(self):
        """Test lines 106-121: successful atomic write operation."""
        test_path = Path(self.temp_dir) / "test_file.txt"
        test_data = "test content"
        
        # Test successful atomic write
        result = self.adapter._atomic_write(test_path, test_data)
        
        # Verify success
        self.assertTrue(result)
        self.assertTrue(test_path.exists())
        
        # Verify content
        with open(test_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, test_data)
    
    def test_atomic_write_failure_with_temp_cleanup(self):
        """Test lines 123-127: atomic write failure with temp file cleanup."""
        test_path = Path(self.temp_dir) / "test_file.txt"
        
        # Mock the write operation to fail
        with patch('os.fdopen', side_effect=Exception("Write failed")):
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Test atomic write failure
                result = self.adapter._atomic_write(test_path, "test")
                
                # Verify failure
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                self.assertIn("Atomic write failed", str(mock_logger.error.call_args))
    
    def test_write_experience_exception(self):
        """Test lines 145-147: write_experience exception handling."""
        # Make the WAL file unwritable
        wal_path = self.adapter.experience_wal
        wal_path.touch()
        os.chmod(wal_path, 0o444)  # Read-only
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Try to write experience
                result = self.adapter.write_experience("test_id", {"data": "test"})
                
                # Should fail
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                self.assertIn("Failed to write experience", str(mock_logger.error.call_args))
        finally:
            # Restore permissions for cleanup
            os.chmod(wal_path, 0o644)
    
    def test_write_operation_exception(self):
        """Test lines 163-165: write_operation exception handling."""
        # Make the WAL file unwritable
        wal_path = self.adapter.operations_wal
        wal_path.touch()
        os.chmod(wal_path, 0o444)  # Read-only
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Try to write operation
                result = self.adapter.write_operation({"op": "test"})
                
                # Should fail
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                self.assertIn("Failed to write operation", str(mock_logger.error.call_args))
        finally:
            # Restore permissions
            os.chmod(wal_path, 0o644)
    
    def test_read_all_experiences_exception(self):
        """Test lines 180-181: read_all_experiences exception handling."""
        # Create corrupted WAL file
        wal_path = self.adapter.experience_wal
        with open(wal_path, 'w') as f:
            f.write("invalid json content\n")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            # Try to read experiences
            result = self.adapter.read_all_experiences()
            
            # Should return empty list
            self.assertEqual(result, [])
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("Failed to read experiences", str(mock_logger.error.call_args))
    
    def test_read_all_operations_with_data(self):
        """Test line 190: read_all_operations returning operations."""
        # Write some operations first
        operations = [
            {"op": "add", "id": "1"},
            {"op": "remove", "id": "2"}
        ]
        
        for op in operations:
            self.adapter.write_operation(op)
        
        # Read operations
        result = self.adapter.read_all_operations()
        
        # Should have 2 operations with timestamps added
        self.assertEqual(len(result), 2)
        self.assertIn("timestamp", result[0])
        self.assertEqual(result[0]["op"], "add")
        self.assertEqual(result[1]["op"], "remove")
    
    def test_read_all_operations_exception(self):
        """Test lines 197-198: read_all_operations exception handling."""
        # Create corrupted WAL file
        wal_path = self.adapter.operations_wal
        with open(wal_path, 'w') as f:
            f.write("invalid json\n")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            # Try to read operations
            result = self.adapter.read_all_operations()
            
            # Should return empty list
            self.assertEqual(result, [])
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("Failed to read operations", str(mock_logger.error.call_args))
    
    def test_save_snapshot_with_temp_cleanup(self):
        """Test lines 226-228: save_snapshot exception with temp file cleanup."""
        # Mock pickle.dump to raise exception
        with patch('pickle.dump', side_effect=Exception("Pickle failed")):
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Try to save snapshot
                result = self.adapter.save_snapshot({"test": "data"})
                
                # Should fail
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                self.assertIn("Failed to save snapshot", str(mock_logger.error.call_args))
                
                # Verify no temp files remain
                temp_files = list(self.adapter.snapshot_dir.glob("*.tmp"))
                self.assertEqual(len(temp_files), 0)
    
    def test_load_snapshot_exception(self):
        """Test lines 245-247: load_snapshot exception handling."""
        # Create a corrupted snapshot file
        snapshot_path = self.adapter.snapshot_dir / "snapshot_20240101_120000.pkl"
        with open(snapshot_path, 'wb') as f:
            f.write(b"corrupted data")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            # Try to load snapshot
            result = self.adapter.load_snapshot()
            
            # Should return None
            self.assertIsNone(result)
            
            # Verify error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("Failed to load snapshot", str(mock_logger.error.call_args))
    
    def test_truncate_logs_exception(self):
        """Test lines 261-263: truncate_logs exception handling."""
        # Make the WAL directory read-only
        os.chmod(self.temp_dir, 0o555)
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Try to truncate logs
                result = self.adapter.truncate_logs()
                
                # Should fail
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                self.assertIn("Failed to truncate logs", str(mock_logger.error.call_args))
        finally:
            # Restore permissions
            os.chmod(self.temp_dir, 0o755)
    
    def test_cleanup_old_snapshots(self):
        """Test lines 270-275: cleanup old snapshots with deletion."""
        # Create multiple snapshot files
        for i in range(5):
            snapshot_path = self.adapter.snapshot_dir / f"snapshot_2024010{i}_120000.pkl"
            with open(snapshot_path, 'wb') as f:
                pickle.dump({"data": f"snapshot_{i}"}, f)
        
        # Cleanup keeping only 3
        self.adapter._cleanup_old_snapshots(keep=3)
        
        # Should have only 3 snapshots remaining
        remaining = list(self.adapter.snapshot_dir.glob("snapshot_*.pkl"))
        self.assertEqual(len(remaining), 3)
        
        # The oldest should be deleted
        remaining_names = [s.name for s in remaining]
        self.assertNotIn("snapshot_20240100_120000.pkl", remaining_names)
        self.assertNotIn("snapshot_20240101_120000.pkl", remaining_names)
    
    def test_cleanup_old_snapshots_deletion_failure(self):
        """Test lines 274-275: cleanup failure warning."""
        # Create snapshot files
        for i in range(4):
            snapshot_path = self.adapter.snapshot_dir / f"snapshot_2024010{i}_120000.pkl"
            with open(snapshot_path, 'wb') as f:
                pickle.dump({"data": f"snapshot_{i}"}, f)
        
        # Mock unlink to raise exception for the first file
        original_unlink = Path.unlink
        call_count = [0]
        
        def mock_unlink(self):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("Permission denied")
            original_unlink(self)
        
        with patch.object(Path, 'unlink', mock_unlink):
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                # Try cleanup
                self.adapter._cleanup_old_snapshots(keep=2)
                
                # Should log warning for failed deletion
                mock_logger.warning.assert_called()
                warning_msg = str(mock_logger.warning.call_args)
                self.assertIn("Failed to delete snapshot", warning_msg)


class TestLMDBPersistenceAdapterMissingCoverage(unittest.TestCase):
    """Test LMDBPersistenceAdapter missing coverage lines."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Check if lmdb is available
        try:
            import lmdb
            self.lmdb_available = True
        except ImportError:
            self.lmdb_available = False
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'adapter') and hasattr(self.adapter, 'close'):
            self.adapter.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_lmdb_import_error(self):
        """Test lines 302-305: LMDB import error handling."""
        # Mock lmdb import to fail
        with patch.dict('sys.modules', {'lmdb': None}):
            with self.assertRaises(ImportError) as context:
                adapter = LMDBPersistenceAdapter(self.temp_dir)
            
            # Check error message
            self.assertIn("LMDB persistence requires 'lmdb' package", str(context.exception))
            self.assertIn("pip install lmdb", str(context.exception))
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_get_max_operation_id_with_data(self):
        """Test line 333: get_max_operation_id with existing data."""
        if not self.lmdb_available:
            # Mock LMDB if not available
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_cursor = MagicMock()
                
                # Mock cursor.last() returns True (has data)
                mock_cursor.last.return_value = True
                mock_cursor.key.return_value = (42).to_bytes(8, 'big')
                
                mock_txn.cursor.return_value = mock_cursor
                mock_env.begin.return_value.__enter__.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                
                mock_lmdb.open.return_value = mock_env
                
                # Create adapter
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                # Operation counter should be max_id + 1 = 43
                self.assertEqual(adapter.operation_counter, 43)
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_transaction_context_manager_exception(self):
        """Test lines 344-347: transaction exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                # Test write transaction with exception
                mock_txn.reset_mock()
                with self.assertRaises(Exception):
                    with adapter._transaction(write=True):
                        raise Exception("Test error")
                
                # Verify abort was called
                mock_txn.abort.assert_called_once()
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_write_experience_exception(self):
        """Test lines 360-362: write_experience exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_txn.put.side_effect = Exception("Write failed")
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to write experience
                    result = adapter.write_experience("test_id", {"data": "test"})
                    
                    # Should fail
                    self.assertFalse(result)
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to write experience", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_write_operation_exception(self):
        """Test lines 374-376: write_operation exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_txn.put.side_effect = Exception("Write failed")
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to write operation
                    result = adapter.write_operation({"op": "test"})
                    
                    # Should fail
                    self.assertFalse(result)
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to write operation", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_read_all_experiences_exception(self):
        """Test lines 388-389: read_all_experiences exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.__iter__.side_effect = Exception("Read failed")
                mock_txn.cursor.return_value = mock_cursor
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to read experiences
                    result = adapter.read_all_experiences()
                    
                    # Should return empty list
                    self.assertEqual(result, [])
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to read experiences", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_read_all_operations_exception(self):
        """Test lines 403-404: read_all_operations exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.__iter__.side_effect = Exception("Read failed")
                mock_txn.cursor.return_value = mock_cursor
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to read operations
                    result = adapter.read_all_operations()
                    
                    # Should return empty list
                    self.assertEqual(result, [])
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to read operations", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_load_snapshot_exception(self):
        """Test lines 435-436: load_snapshot exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.last.return_value = True
                mock_cursor.value.return_value = b"corrupted data"
                mock_txn.cursor.return_value = mock_cursor
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to load snapshot
                    result = adapter.load_snapshot()
                    
                    # Should return None
                    self.assertIsNone(result)
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to load snapshot", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_truncate_logs_exception(self):
        """Test lines 452-454: truncate_logs exception handling."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_txn.drop.side_effect = Exception("Drop failed")
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try to truncate logs
                    result = adapter.truncate_logs()
                    
                    # Should fail
                    self.assertFalse(result)
                    
                    # Verify error was logged
                    mock_logger.error.assert_called()
                    self.assertIn("Failed to truncate logs", str(mock_logger.error.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_cleanup_old_snapshots_with_deletion(self):
        """Test lines 467-473: cleanup old snapshots with deletion."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_cursor = MagicMock()
                
                # Mock stat to return count > keep
                mock_txn.stat.return_value = {'entries': 5}
                mock_cursor.first.return_value = True
                mock_cursor.next.return_value = True
                
                mock_txn.cursor.return_value = mock_cursor
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                # Cleanup keeping only 3
                adapter._cleanup_old_snapshots(keep=3)
                
                # Verify cursor operations
                mock_cursor.first.assert_called_once()
                # Should delete 2 (5 - 3)
                self.assertEqual(mock_cursor.delete.call_count, 2)
                self.assertEqual(mock_cursor.next.call_count, 2)
        else:
            self.skipTest("Test requires mocked LMDB")
    
    @unittest.skipUnless('lmdb' in sys.modules or True, "LMDB not installed")
    def test_cleanup_old_snapshots_exception(self):
        """Test line 473: cleanup exception warning."""
        if not self.lmdb_available:
            with patch('core.modules.persistence_adapter.lmdb') as mock_lmdb:
                # Setup mocks
                mock_env = MagicMock()
                mock_txn = MagicMock()
                mock_txn.stat.side_effect = Exception("Stat failed")
                mock_env.begin.return_value = mock_txn
                mock_env.open_db.return_value = MagicMock()
                mock_lmdb.open.return_value = mock_env
                
                adapter = LMDBPersistenceAdapter(self.temp_dir)
                
                with patch('core.modules.persistence_adapter.logger') as mock_logger:
                    # Try cleanup
                    adapter._cleanup_old_snapshots(keep=3)
                    
                    # Should log warning
                    mock_logger.warning.assert_called()
                    self.assertIn("Failed to cleanup snapshots", str(mock_logger.warning.call_args))
        else:
            self.skipTest("Test requires mocked LMDB")


class TestFactoryFunction(unittest.TestCase):
    """Test factory function for creating persistence adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_file_adapter(self):
        """Test creating file-based persistence adapter."""
        adapter = create_persistence_adapter("file", self.temp_dir)
        self.assertIsInstance(adapter, FilePersistenceAdapter)
        adapter.close()
    
    @patch('core.modules.persistence_adapter.lmdb')
    def test_create_lmdb_adapter(self, mock_lmdb):
        """Test creating LMDB-based persistence adapter."""
        # Setup mocks
        mock_env = MagicMock()
        mock_env.open_db.return_value = MagicMock()
        mock_lmdb.open.return_value = mock_env
        
        adapter = create_persistence_adapter("lmdb", self.temp_dir)
        self.assertIsInstance(adapter, LMDBPersistenceAdapter)
        adapter.close()
    
    def test_create_unknown_adapter(self):
        """Test creating adapter with unknown type."""
        with self.assertRaises(ValueError) as context:
            create_persistence_adapter("unknown", self.temp_dir)
        
        self.assertIn("Unknown adapter type: unknown", str(context.exception))


class TestAbstractMethods(unittest.TestCase):
    """Test that abstract methods are properly defined."""
    
    def test_cannot_instantiate_abstract_base(self):
        """Test that PersistenceAdapter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            adapter = PersistenceAdapter()
    
    def test_abstract_methods_defined(self):
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
    # Run with verbose output to see all test coverage
    unittest.main(verbosity=2)