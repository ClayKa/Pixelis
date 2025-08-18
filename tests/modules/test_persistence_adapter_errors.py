"""
Test file to cover error handling paths in persistence_adapter.py
"""

import unittest
import tempfile
import shutil
import os
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add project root to path
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.persistence_adapter import (
    FilePersistenceAdapter,
    LMDBPersistenceAdapter,
    create_persistence_adapter
)


class TestFilePersistenceAdapterErrors(unittest.TestCase):
    """Test error handling in FilePersistenceAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = FilePersistenceAdapter(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_atomic_write_temp_file_cleanup_on_error(self):
        """Test lines 125-127: temp file cleanup on error."""
        test_path = Path(self.temp_dir) / "test.txt"
        
        # Create a mock that tracks temp file path
        temp_files = []
        original_mkstemp = tempfile.mkstemp
        
        def mock_mkstemp(**kwargs):
            fd, path = original_mkstemp(**kwargs)
            temp_files.append(path)
            return fd, path
        
        with patch('tempfile.mkstemp', side_effect=mock_mkstemp):
            with patch('os.fdopen', side_effect=Exception("Write failed")):
                with patch('core.modules.persistence_adapter.logger'):
                    # Try atomic write - should fail
                    result = self.adapter._atomic_write(test_path, "test")
                    self.assertFalse(result)
                    
                    # Verify temp file was cleaned up
                    for temp_path in temp_files:
                        self.assertFalse(os.path.exists(temp_path))
    
    def test_write_experience_error(self):
        """Test lines 145-147: write_experience error handling."""
        # Make WAL directory read-only
        os.chmod(self.temp_dir, 0o555)
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                result = self.adapter.write_experience("test_id", {"data": "test"})
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called()
                self.assertIn("Failed to write experience", str(mock_logger.error.call_args))
        finally:
            os.chmod(self.temp_dir, 0o755)
    
    def test_write_operation_error(self):
        """Test lines 163-165: write_operation error handling."""
        # Make WAL directory read-only
        os.chmod(self.temp_dir, 0o555)
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                result = self.adapter.write_operation({"op": "test"})
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called()
                self.assertIn("Failed to write operation", str(mock_logger.error.call_args))
        finally:
            os.chmod(self.temp_dir, 0o755)
    
    def test_read_experiences_error(self):
        """Test lines 180-181: read_all_experiences error handling."""
        # Create corrupted WAL file
        with open(self.adapter.experience_wal, 'w') as f:
            f.write("invalid json\n")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            result = self.adapter.read_all_experiences()
            self.assertEqual(result, [])
            
            # Verify error was logged
            mock_logger.error.assert_called()
            self.assertIn("Failed to read experiences", str(mock_logger.error.call_args))
    
    def test_read_operations_error(self):
        """Test lines 197-198: read_all_operations error handling."""
        # Create corrupted WAL file
        with open(self.adapter.operations_wal, 'w') as f:
            f.write("invalid json\n")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            result = self.adapter.read_all_operations()
            self.assertEqual(result, [])
            
            # Verify error was logged
            mock_logger.error.assert_called()
            self.assertIn("Failed to read operations", str(mock_logger.error.call_args))
    
    def test_save_snapshot_error_with_cleanup(self):
        """Test lines 225-229: save_snapshot error with temp file cleanup."""
        # Mock pickle.dump to fail
        with patch('pickle.dump', side_effect=Exception("Pickle failed")):
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                result = self.adapter.save_snapshot({"test": "data"})
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called()
                self.assertIn("Failed to save snapshot", str(mock_logger.error.call_args))
                
                # Verify no temp files remain
                temp_files = list(self.adapter.snapshot_dir.glob("*.tmp"))
                self.assertEqual(len(temp_files), 0)
    
    def test_load_snapshot_error(self):
        """Test lines 245-247: load_snapshot error handling."""
        # Create corrupted snapshot file
        snapshot_path = self.adapter.snapshot_dir / "snapshot_20240101_120000.pkl"
        with open(snapshot_path, 'wb') as f:
            f.write(b"corrupted data")
        
        with patch('core.modules.persistence_adapter.logger') as mock_logger:
            result = self.adapter.load_snapshot()
            self.assertIsNone(result)
            
            # Verify error was logged
            mock_logger.error.assert_called()
            self.assertIn("Failed to load snapshot", str(mock_logger.error.call_args))
    
    def test_truncate_logs_error(self):
        """Test lines 261-263: truncate_logs error handling."""
        # Make WAL directory read-only
        os.chmod(self.temp_dir, 0o555)
        
        try:
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                result = self.adapter.truncate_logs()
                self.assertFalse(result)
                
                # Verify error was logged
                mock_logger.error.assert_called()
                self.assertIn("Failed to truncate logs", str(mock_logger.error.call_args))
        finally:
            os.chmod(self.temp_dir, 0o755)
    
    def test_cleanup_snapshots_deletion_warning(self):
        """Test lines 274-275: cleanup snapshot deletion warning."""
        # Create snapshot files
        for i in range(4):
            snapshot_path = self.adapter.snapshot_dir / f"snapshot_2024010{i}_120000.pkl"
            with open(snapshot_path, 'wb') as f:
                pickle.dump({"data": f"snapshot_{i}"}, f)
        
        # Mock unlink to raise exception
        with patch.object(Path, 'unlink', side_effect=OSError("Permission denied")):
            with patch('core.modules.persistence_adapter.logger') as mock_logger:
                self.adapter._cleanup_old_snapshots(keep=2)
                
                # Should log warning for failed deletion
                mock_logger.warning.assert_called()
                self.assertIn("Failed to delete snapshot", str(mock_logger.warning.call_args))


class TestLMDBAdapterImportError(unittest.TestCase):
    """Test LMDB adapter import error."""
    
    def test_lmdb_import_error(self):
        """Test lines 302-305: LMDB import error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock lmdb import to fail
            with patch.dict('sys.modules', {'lmdb': None}):
                with self.assertRaises(ImportError) as ctx:
                    LMDBPersistenceAdapter(temp_dir)
                
                self.assertIn("LMDB persistence requires 'lmdb' package", str(ctx.exception))
                self.assertIn("pip install lmdb", str(ctx.exception))


class TestFactoryLMDBError(unittest.TestCase):
    """Test factory function LMDB error path."""
    
    def test_create_lmdb_import_error(self):
        """Test line 496: LMDB adapter creation in factory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock lmdb to not be available
            with patch.dict('sys.modules', {'lmdb': None}):
                with self.assertRaises(ImportError):
                    create_persistence_adapter("lmdb", temp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)