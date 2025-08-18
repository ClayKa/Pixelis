"""
Complete Test Coverage for audit.py

This test file ensures 100% coverage of all lines, branches, and edge cases
in the core/modules/audit.py module.
"""

import unittest
import tempfile
import shutil
import json
import gzip
import time
import threading
import queue
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, mock_open, call
from typing import Dict, Any
import sys

sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.audit import (
    AuditEventType,
    AuditResult,
    AuditEntry,
    AuditLogger,
    initialize_audit_logger,
    get_audit_logger,
    audit_log
)


class TestAuditEventType(unittest.TestCase):
    """Test AuditEventType enum."""
    
    def test_event_types(self):
        """Test all event type values."""
        self.assertEqual(AuditEventType.MODEL_UPDATE.value, "model_update")
        self.assertEqual(AuditEventType.ACCESS_ATTEMPT.value, "access_attempt")
        self.assertEqual(AuditEventType.CONFIG_CHANGE.value, "config_change")
        self.assertEqual(AuditEventType.DATA_DELETION.value, "data_deletion")
        self.assertEqual(AuditEventType.SECURITY_VIOLATION.value, "security_violation")
        self.assertEqual(AuditEventType.SYSTEM_ERROR.value, "system_error")
        self.assertEqual(AuditEventType.DATA_PRUNING.value, "data_pruning")
        self.assertEqual(AuditEventType.USER_ACTION.value, "user_action")
        self.assertEqual(AuditEventType.AUTHENTICATION.value, "authentication")
        self.assertEqual(AuditEventType.PRIVACY_OPERATION.value, "privacy_operation")


class TestAuditResult(unittest.TestCase):
    """Test AuditResult enum."""
    
    def test_result_types(self):
        """Test all result type values."""
        self.assertEqual(AuditResult.SUCCESS.value, "success")
        self.assertEqual(AuditResult.FAILURE.value, "failure")
        self.assertEqual(AuditResult.PARTIAL.value, "partial")
        self.assertEqual(AuditResult.BLOCKED.value, "blocked")


class TestAuditEntry(unittest.TestCase):
    """Test AuditEntry dataclass."""
    
    def test_default_creation(self):
        """Test creating AuditEntry with defaults."""
        entry = AuditEntry()
        self.assertIsNotNone(entry.timestamp)
        self.assertEqual(entry.event_type, AuditEventType.SYSTEM_ERROR)
        self.assertEqual(entry.actor, "system")
        self.assertEqual(entry.action, "")
        self.assertEqual(entry.resource, "")
        self.assertEqual(entry.result, AuditResult.SUCCESS)
        self.assertEqual(entry.metadata, {})
        self.assertIsNone(entry.hash_previous)
        self.assertIsNone(entry.hash_current)
    
    def test_custom_creation(self):
        """Test creating AuditEntry with custom values."""
        metadata = {"key": "value", "count": 42}
        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.MODEL_UPDATE,
            actor="user123",
            action="update_model",
            resource="/model/v1",
            result=AuditResult.SUCCESS,
            metadata=metadata,
            hash_previous="prev_hash",
            hash_current="curr_hash"
        )
        
        self.assertEqual(entry.timestamp, "2024-01-01T00:00:00")
        self.assertEqual(entry.event_type, AuditEventType.MODEL_UPDATE)
        self.assertEqual(entry.actor, "user123")
        self.assertEqual(entry.action, "update_model")
        self.assertEqual(entry.resource, "/model/v1")
        self.assertEqual(entry.result, AuditResult.SUCCESS)
        self.assertEqual(entry.metadata, metadata)
        self.assertEqual(entry.hash_previous, "prev_hash")
        self.assertEqual(entry.hash_current, "curr_hash")
    
    def test_calculate_hash_no_previous(self):
        """Test hash calculation without previous hash."""
        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS,
            metadata={"ip": "192.168.1.1"}
        )
        
        hash_value = entry.calculate_hash()
        self.assertIsNotNone(hash_value)
        self.assertEqual(len(hash_value), 64)  # SHA-256 hex string length
        
        # Hash should be deterministic
        hash_value2 = entry.calculate_hash()
        self.assertEqual(hash_value, hash_value2)
    
    def test_calculate_hash_with_previous(self):
        """Test hash calculation with previous hash."""
        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS,
            metadata={"ip": "192.168.1.1"}
        )
        
        previous_hash = "a" * 64
        hash_value = entry.calculate_hash(previous_hash)
        self.assertIsNotNone(hash_value)
        
        # Different previous hash should produce different result
        different_prev = "b" * 64
        hash_value2 = entry.calculate_hash(different_prev)
        self.assertNotEqual(hash_value, hash_value2)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = {"key": "value"}
        entry = AuditEntry(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.CONFIG_CHANGE,
            actor="admin",
            action="update_config",
            resource="/config",
            result=AuditResult.SUCCESS,
            metadata=metadata,
            hash_previous="prev",
            hash_current="curr"
        )
        
        data = entry.to_dict()
        
        self.assertEqual(data['timestamp'], "2024-01-01T00:00:00")
        self.assertEqual(data['event_type'], "config_change")
        self.assertEqual(data['actor'], "admin")
        self.assertEqual(data['action'], "update_config")
        self.assertEqual(data['resource'], "/config")
        self.assertEqual(data['result'], "success")
        self.assertEqual(data['metadata'], metadata)
        self.assertEqual(data['hash_previous'], "prev")
        self.assertEqual(data['hash_current'], "curr")
    
    def test_from_dict_complete(self):
        """Test creation from complete dictionary."""
        data = {
            'timestamp': "2024-01-01T00:00:00",
            'event_type': "security_violation",
            'actor': "attacker",
            'action': "unauthorized_access",
            'resource': "/admin",
            'result': "blocked",
            'metadata': {"attempt": 1},
            'hash_previous': "prev",
            'hash_current': "curr"
        }
        
        entry = AuditEntry.from_dict(data)
        
        self.assertEqual(entry.timestamp, "2024-01-01T00:00:00")
        self.assertEqual(entry.event_type, AuditEventType.SECURITY_VIOLATION)
        self.assertEqual(entry.actor, "attacker")
        self.assertEqual(entry.action, "unauthorized_access")
        self.assertEqual(entry.resource, "/admin")
        self.assertEqual(entry.result, AuditResult.BLOCKED)
        self.assertEqual(entry.metadata, {"attempt": 1})
        self.assertEqual(entry.hash_previous, "prev")
        self.assertEqual(entry.hash_current, "curr")
    
    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        data = {
            'event_type': "data_pruning"
        }
        
        entry = AuditEntry.from_dict(data)
        
        self.assertIsNotNone(entry.timestamp)
        self.assertEqual(entry.event_type, AuditEventType.DATA_PRUNING)
        self.assertEqual(entry.actor, "system")
        self.assertEqual(entry.action, "")
        self.assertEqual(entry.resource, "")
        self.assertEqual(entry.result, AuditResult.SUCCESS)
        self.assertEqual(entry.metadata, {})
        self.assertIsNone(entry.hash_previous)
        self.assertIsNone(entry.hash_current)
    
    def test_from_dict_empty(self):
        """Test creation from empty dictionary."""
        entry = AuditEntry.from_dict({})
        
        self.assertIsNotNone(entry.timestamp)
        self.assertEqual(entry.event_type, AuditEventType.SYSTEM_ERROR)
        self.assertEqual(entry.actor, "system")
        self.assertEqual(entry.result, AuditResult.SUCCESS)


class TestAuditLogger(unittest.TestCase):
    """Test AuditLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_dir = Path(self.temp_dir) / "audit"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        logger = AuditLogger(audit_dir=str(self.audit_dir))
        
        self.assertTrue(self.audit_dir.exists())
        self.assertEqual(logger.max_file_size, 100_000_000)
        self.assertEqual(logger.retention_days, 365)
        self.assertTrue(logger.enable_async)
        self.assertFalse(logger.enable_encryption)
        self.assertIsNotNone(logger.log_queue)
        self.assertTrue(logger.worker_thread.is_alive())
        
        logger.shutdown()
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        logger = AuditLogger(
            audit_dir=str(self.audit_dir),
            max_file_size=1000,
            retention_days=30,
            enable_async=False,
            enable_encryption=True
        )
        
        self.assertEqual(logger.max_file_size, 1000)
        self.assertEqual(logger.retention_days, 30)
        self.assertFalse(logger.enable_async)
        self.assertTrue(logger.enable_encryption)
        self.assertIsNone(logger.log_queue)
        
        logger.shutdown()
    
    def test_get_last_hash_no_file(self):
        """Test getting last hash when file doesn't exist."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Delete the current file if it exists
        if logger.current_file_path.exists():
            logger.current_file_path.unlink()
        
        last_hash = logger._get_last_hash()
        self.assertIsNone(last_hash)
        
        logger.shutdown()
    
    def test_get_last_hash_empty_file(self):
        """Test getting last hash from empty file."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Create empty file
        logger.current_file_path.touch()
        
        last_hash = logger._get_last_hash()
        self.assertIsNone(last_hash)
        
        logger.shutdown()
    
    def test_get_last_hash_with_entries(self):
        """Test getting last hash from file with entries."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write some entries
        entries = [
            {'hash_current': 'hash1'},
            {'hash_current': 'hash2'},
            {'hash_current': 'hash3'}
        ]
        
        with open(logger.current_file_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        last_hash = logger._get_last_hash()
        self.assertEqual(last_hash, 'hash3')
        
        logger.shutdown()
    
    def test_get_last_hash_exception(self):
        """Test get_last_hash with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write invalid JSON
        with open(logger.current_file_path, 'w') as f:
            f.write("invalid json\n")
        
        with patch('core.modules.audit.logger') as mock_logger:
            last_hash = logger._get_last_hash()
            self.assertIsNone(last_hash)
            mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_log_success_sync(self):
        """Test successful synchronous logging."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        result = logger.log(
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS,
            metadata={"ip": "192.168.1.1"}
        )
        
        self.assertTrue(result)
        self.assertEqual(logger.stats['total_entries'], 1)
        self.assertEqual(logger.stats['entries_by_type']['user_action'], 1)
        self.assertEqual(logger.stats['entries_by_result']['success'], 1)
        
        # Check file was written
        self.assertTrue(logger.current_file_path.exists())
        
        with open(logger.current_file_path, 'r') as f:
            line = f.readline()
            entry_dict = json.loads(line)
            self.assertEqual(entry_dict['event_type'], 'user_action')
            self.assertEqual(entry_dict['actor'], 'user1')
        
        logger.shutdown()
    
    def test_log_success_async(self):
        """Test successful asynchronous logging."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=True)
        
        result = logger.log(
            event_type=AuditEventType.MODEL_UPDATE,
            actor="system",
            action="update",
            resource="/model",
            result=AuditResult.SUCCESS
        )
        
        self.assertTrue(result)
        
        # Wait for async write
        time.sleep(0.5)
        
        # Check file was written
        self.assertTrue(logger.current_file_path.exists())
        
        logger.shutdown()
    
    def test_log_exception(self):
        """Test logging with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Mock calculate_hash to raise exception
        with patch.object(AuditEntry, 'calculate_hash', side_effect=Exception("Hash error")):
            with patch('core.modules.audit.logger') as mock_logger:
                result = logger.log(
                    event_type=AuditEventType.SYSTEM_ERROR,
                    actor="system",
                    action="error",
                    resource="/error"
                )
                
                self.assertFalse(result)
                mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_write_entry_success(self):
        """Test successful entry writing."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        entry = AuditEntry(
            event_type=AuditEventType.ACCESS_ATTEMPT,
            actor="user2",
            action="read",
            resource="/data"
        )
        entry.hash_current = entry.calculate_hash()
        
        logger._write_entry(entry)
        
        # Check file was written
        self.assertTrue(logger.current_file_path.exists())
        
        with open(logger.current_file_path, 'r') as f:
            line = f.readline()
            entry_dict = json.loads(line)
            self.assertEqual(entry_dict['event_type'], 'access_attempt')
        
        logger.shutdown()
    
    def test_write_entry_with_rotation(self):
        """Test entry writing with file rotation."""
        logger = AuditLogger(
            audit_dir=str(self.audit_dir),
            max_file_size=100,  # Small size to trigger rotation
            enable_async=False
        )
        
        # Write initial data to exceed size
        with open(logger.current_file_path, 'w') as f:
            f.write("x" * 150)
        
        entry = AuditEntry(
            event_type=AuditEventType.CONFIG_CHANGE,
            actor="admin",
            action="update",
            resource="/config"
        )
        
        logger._write_entry(entry)
        
        # Check archives directory was created
        archives_dir = self.audit_dir / "archives"
        self.assertTrue(archives_dir.exists())
        
        # Check for compressed archive
        archives = list(archives_dir.glob("*.gz"))
        self.assertGreater(len(archives), 0)
        
        logger.shutdown()
    
    def test_write_entry_exception(self):
        """Test write entry with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        entry = AuditEntry()
        
        # Make directory read-only to cause write failure
        with patch('builtins.open', side_effect=PermissionError("No write permission")):
            with patch('core.modules.audit.logger') as mock_logger:
                logger._write_entry(entry)
                mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_async_worker_normal(self):
        """Test async worker normal operation."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=True)
        
        # Add entry to queue
        entry = AuditEntry(
            event_type=AuditEventType.AUTHENTICATION,
            actor="user3",
            action="logout",
            resource="/auth"
        )
        logger.log_queue.put(entry)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check file was written
        self.assertTrue(logger.current_file_path.exists())
        
        logger.shutdown()
    
    def test_async_worker_exception(self):
        """Test async worker with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=True)
        
        # Put invalid item in queue
        logger.log_queue.put("invalid_entry")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Worker should continue running despite error
        self.assertTrue(logger.worker_thread.is_alive())
        
        logger.shutdown()
    
    def test_rotate_file_success(self):
        """Test successful file rotation."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Create file with content
        test_content = "test data for rotation"
        with open(logger.current_file_path, 'w') as f:
            f.write(test_content)
        
        original_path = logger.current_file_path
        
        logger._rotate_file()
        
        # Check archive was created
        archives_dir = self.audit_dir / "archives"
        self.assertTrue(archives_dir.exists())
        
        # Check compressed archive exists
        archives = list(archives_dir.glob("*.gz"))
        self.assertEqual(len(archives), 1)
        
        # Check new current file path
        self.assertNotEqual(logger.current_file_path, original_path)
        
        logger.shutdown()
    
    def test_rotate_file_exception(self):
        """Test file rotation with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        with patch('pathlib.Path.rename', side_effect=Exception("Rename error")):
            with patch('core.modules.audit.logger') as mock_logger:
                logger._rotate_file()
                mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_verify_integrity_no_file(self):
        """Test integrity verification when file doesn't exist."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Delete current file
        if logger.current_file_path.exists():
            logger.current_file_path.unlink()
        
        results = logger.verify_integrity()
        
        self.assertFalse(results['valid'])
        self.assertEqual(results['error'], 'File does not exist')
        
        logger.shutdown()
    
    def test_verify_integrity_valid(self):
        """Test integrity verification with valid file."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Log some entries properly
        for i in range(3):
            logger.log(
                event_type=AuditEventType.USER_ACTION,
                actor=f"user{i}",
                action="action",
                resource=f"/resource{i}"
            )
        
        results = logger.verify_integrity()
        
        self.assertTrue(results['valid'])
        self.assertEqual(results['total_entries'], 3)
        self.assertEqual(len(results['errors']), 0)
        
        logger.shutdown()
    
    def test_verify_integrity_invalid_hash(self):
        """Test integrity verification with invalid hash."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write entry with incorrect hash
        entry = AuditEntry(
            event_type=AuditEventType.SECURITY_VIOLATION,
            actor="attacker",
            action="tamper",
            resource="/data"
        )
        entry.hash_current = "invalid_hash"
        
        with open(logger.current_file_path, 'w') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        results = logger.verify_integrity()
        
        self.assertFalse(results['valid'])
        self.assertGreater(len(results['errors']), 0)
        self.assertIn("Hash mismatch", results['errors'][0])
        
        logger.shutdown()
    
    def test_verify_integrity_invalid_chain(self):
        """Test integrity verification with broken hash chain."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write two entries with broken chain
        entry1 = AuditEntry(event_type=AuditEventType.USER_ACTION)
        entry1.hash_previous = None
        entry1.hash_current = entry1.calculate_hash()
        
        entry2 = AuditEntry(event_type=AuditEventType.USER_ACTION)
        entry2.hash_previous = "wrong_previous_hash"
        entry2.hash_current = entry2.calculate_hash("wrong_previous_hash")
        
        with open(logger.current_file_path, 'w') as f:
            f.write(json.dumps(entry1.to_dict()) + '\n')
            f.write(json.dumps(entry2.to_dict()) + '\n')
        
        results = logger.verify_integrity()
        
        self.assertFalse(results['valid'])
        self.assertIn("Previous hash mismatch", str(results['errors']))
        
        logger.shutdown()
    
    def test_verify_integrity_verbose(self):
        """Test integrity verification with verbose output."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write entry with incorrect hash
        entry = AuditEntry()
        entry.hash_current = "wrong_hash"
        
        with open(logger.current_file_path, 'w') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        results = logger.verify_integrity(verbose=True)
        
        self.assertFalse(results['valid'])
        # In verbose mode, errors should contain detailed info
        self.assertTrue(any(isinstance(e, dict) for e in results['errors'] if isinstance(e, dict)))
        
        logger.shutdown()
    
    def test_verify_integrity_json_error(self):
        """Test integrity verification with JSON error."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write invalid JSON
        with open(logger.current_file_path, 'w') as f:
            f.write("invalid json\n")
        
        results = logger.verify_integrity()
        
        self.assertFalse(results['valid'])
        self.assertIn("JSON error", str(results['errors']))
        
        logger.shutdown()
    
    def test_verify_integrity_exception(self):
        """Test integrity verification with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        with patch('builtins.open', side_effect=Exception("Read error")):
            results = logger.verify_integrity()
            
            self.assertFalse(results['valid'])
            self.assertEqual(results['error'], "Read error")
        
        logger.shutdown()
    
    def test_search_basic(self):
        """Test basic search functionality."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Log various entries
        logger.log(
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS
        )
        logger.log(
            event_type=AuditEventType.MODEL_UPDATE,
            actor="system",
            action="update",
            resource="/model",
            result=AuditResult.SUCCESS
        )
        logger.log(
            event_type=AuditEventType.USER_ACTION,
            actor="user2",
            action="logout",
            resource="/auth",
            result=AuditResult.SUCCESS
        )
        
        # Search by event type
        results = logger.search(event_type=AuditEventType.USER_ACTION)
        self.assertEqual(len(results), 2)
        
        # Search by actor
        results = logger.search(actor="user1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].actor, "user1")
        
        # Search by resource (partial match)
        results = logger.search(resource="/auth")
        self.assertEqual(len(results), 2)
        
        logger.shutdown()
    
    def test_search_with_dates(self):
        """Test search with date filters."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Log entries with different timestamps
        now = datetime.now()
        
        entry1 = AuditEntry(
            timestamp=(now - timedelta(days=2)).isoformat(),
            event_type=AuditEventType.USER_ACTION,
            actor="user1"
        )
        entry1.hash_current = entry1.calculate_hash()
        
        entry2 = AuditEntry(
            timestamp=now.isoformat(),
            event_type=AuditEventType.USER_ACTION,
            actor="user2"
        )
        entry2.hash_current = entry2.calculate_hash()
        
        with open(logger.current_file_path, 'w') as f:
            f.write(json.dumps(entry1.to_dict()) + '\n')
            f.write(json.dumps(entry2.to_dict()) + '\n')
        
        # Search with start date
        results = logger.search(start_date=now - timedelta(days=1))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].actor, "user2")
        
        # Search with end date
        results = logger.search(end_date=now - timedelta(days=1))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].actor, "user1")
        
        logger.shutdown()
    
    def test_search_with_limit(self):
        """Test search with result limit."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Log many entries
        for i in range(10):
            logger.log(
                event_type=AuditEventType.USER_ACTION,
                actor=f"user{i}",
                action="action",
                resource="/resource"
            )
        
        # Search with limit
        results = logger.search(limit=5)
        self.assertEqual(len(results), 5)
        
        logger.shutdown()
    
    def test_search_compressed_files(self):
        """Test searching compressed archive files."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Create compressed archive
        archives_dir = self.audit_dir / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        archive_path = archives_dir / "audit_20240101.jsonl.gz"
        
        entry = AuditEntry(
            event_type=AuditEventType.DATA_DELETION,
            actor="system",
            action="cleanup",
            resource="/old_data"
        )
        
        with gzip.open(archive_path, 'wt') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        # Search with date range to include archives
        results = logger.search(
            start_date=datetime.now() - timedelta(days=365),
            event_type=AuditEventType.DATA_DELETION
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "cleanup")
        
        logger.shutdown()
    
    def test_search_exception(self):
        """Test search with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Write invalid JSON to file
        with open(logger.current_file_path, 'w') as f:
            f.write("invalid json\n")
        
        with patch('core.modules.audit.logger') as mock_logger:
            results = logger.search()
            # Should handle error gracefully
            self.assertEqual(len(results), 0)
            mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_get_statistics(self):
        """Test getting statistics."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Log some entries
        logger.log(
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS
        )
        logger.log(
            event_type=AuditEventType.MODEL_UPDATE,
            actor="system",
            action="update",
            resource="/model",
            result=AuditResult.FAILURE
        )
        
        # Verify integrity to update last_verification
        logger.verify_integrity()
        
        stats = logger.get_statistics()
        
        self.assertEqual(stats['total_entries'], 2)
        self.assertEqual(stats['entries_by_type']['user_action'], 1)
        self.assertEqual(stats['entries_by_type']['model_update'], 1)
        self.assertEqual(stats['entries_by_result']['success'], 1)
        self.assertEqual(stats['entries_by_result']['failure'], 1)
        self.assertIsNotNone(stats['last_verification'])
        self.assertIn('current_file', stats)
        self.assertGreater(stats['file_size'], 0)
        
        logger.shutdown()
    
    def test_get_statistics_no_file(self):
        """Test getting statistics when file doesn't exist."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Delete current file
        if logger.current_file_path.exists():
            logger.current_file_path.unlink()
        
        stats = logger.get_statistics()
        
        self.assertEqual(stats['file_size'], 0)
        
        logger.shutdown()
    
    def test_cleanup_old_logs(self):
        """Test cleaning up old audit logs."""
        logger = AuditLogger(
            audit_dir=str(self.audit_dir),
            retention_days=30,
            enable_async=False
        )
        
        # Create old archive files
        archives_dir = self.audit_dir / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        # Create old file (older than retention)
        old_file = archives_dir / "audit_old.jsonl.gz"
        old_file.touch()
        # Set modification time to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(old_file, (old_time, old_time))
        
        # Create recent file (within retention)
        recent_file = archives_dir / "audit_recent.jsonl.gz"
        recent_file.touch()
        
        logger.cleanup_old_logs()
        
        # Old file should be removed
        self.assertFalse(old_file.exists())
        # Recent file should remain
        self.assertTrue(recent_file.exists())
        
        # Check that cleanup was logged
        self.assertGreater(logger.stats['total_entries'], 0)
        
        logger.shutdown()
    
    def test_cleanup_old_logs_no_archives(self):
        """Test cleanup when archives directory doesn't exist."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Ensure archives directory doesn't exist
        archives_dir = self.audit_dir / "archives"
        if archives_dir.exists():
            shutil.rmtree(archives_dir)
        
        # Should not raise exception
        logger.cleanup_old_logs()
        
        logger.shutdown()
    
    def test_cleanup_old_logs_exception(self):
        """Test cleanup with exception."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        # Create archives directory
        archives_dir = self.audit_dir / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        # Create file
        test_file = archives_dir / "audit_test.jsonl.gz"
        test_file.touch()
        
        with patch('pathlib.Path.unlink', side_effect=Exception("Delete error")):
            with patch('core.modules.audit.logger') as mock_logger:
                logger.cleanup_old_logs()
                mock_logger.error.assert_called()
        
        logger.shutdown()
    
    def test_shutdown_async(self):
        """Test shutdown with async logging."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=True)
        
        # Log some entries
        logger.log(
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="action",
            resource="/resource"
        )
        
        logger.shutdown()
        
        # Worker thread should stop
        time.sleep(0.5)
        self.assertFalse(logger.worker_thread.is_alive())
    
    def test_shutdown_sync(self):
        """Test shutdown with sync logging."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=False)
        
        logger.shutdown()
        
        # Should complete without error
        self.assertIsNone(logger.log_queue)
    
    def test_shutdown_worker_timeout(self):
        """Test shutdown when worker doesn't stop in time."""
        logger = AuditLogger(audit_dir=str(self.audit_dir), enable_async=True)
        
        # Mock worker thread to simulate it being stuck
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join.return_value = None
        logger.worker_thread = mock_thread
        
        logger.shutdown()
        
        # Join should be called with timeout
        mock_thread.join.assert_called_once_with(timeout=5)


class TestGlobalFunctions(unittest.TestCase):
    """Test global functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Reset global logger
        import core.modules.audit
        core.modules.audit._audit_logger = None
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown global logger if exists
        import core.modules.audit
        if core.modules.audit._audit_logger:
            core.modules.audit._audit_logger.shutdown()
            core.modules.audit._audit_logger = None
        
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialize_audit_logger(self):
        """Test initializing global audit logger."""
        config = {
            'audit_dir': self.temp_dir,
            'max_file_size': 1000,
            'retention_days': 7,
            'enable_async': False,
            'enable_encryption': True
        }
        
        logger = initialize_audit_logger(config)
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.max_file_size, 1000)
        self.assertEqual(logger.retention_days, 7)
        self.assertFalse(logger.enable_async)
        self.assertTrue(logger.enable_encryption)
        
        # Check global logger was set
        global_logger = get_audit_logger()
        self.assertEqual(logger, global_logger)
    
    def test_initialize_audit_logger_defaults(self):
        """Test initializing with default config."""
        logger = initialize_audit_logger({})
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.audit_dir, Path('./audit'))
        self.assertEqual(logger.max_file_size, 100_000_000)
        self.assertEqual(logger.retention_days, 365)
        self.assertTrue(logger.enable_async)
        self.assertFalse(logger.enable_encryption)
    
    def test_get_audit_logger_none(self):
        """Test getting audit logger when not initialized."""
        logger = get_audit_logger()
        self.assertIsNone(logger)
    
    def test_audit_log_success(self):
        """Test audit_log convenience function."""
        # Initialize logger first
        initialize_audit_logger({'audit_dir': self.temp_dir, 'enable_async': False})
        
        result = audit_log(
            event_type=AuditEventType.USER_ACTION,
            actor="user1",
            action="test",
            resource="/test",
            result=AuditResult.SUCCESS,
            metadata={"key": "value"}
        )
        
        self.assertTrue(result)
    
    def test_audit_log_not_initialized(self):
        """Test audit_log when logger not initialized."""
        with patch('core.modules.audit.logger') as mock_logger:
            result = audit_log(
                event_type=AuditEventType.USER_ACTION,
                actor="user1",
                action="test",
                resource="/test"
            )
            
            self.assertFalse(result)
            mock_logger.warning.assert_called_with("Audit logger not initialized")


class TestIntegration(unittest.TestCase):
    """Integration tests for audit module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete audit workflow."""
        # Initialize logger
        config = {
            'audit_dir': self.temp_dir,
            'retention_days': 30,
            'enable_async': False
        }
        logger = initialize_audit_logger(config)
        
        # Log various events
        audit_log(
            event_type=AuditEventType.AUTHENTICATION,
            actor="user1",
            action="login",
            resource="/auth",
            result=AuditResult.SUCCESS,
            metadata={"ip": "192.168.1.1", "timestamp": time.time()}
        )
        
        audit_log(
            event_type=AuditEventType.MODEL_UPDATE,
            actor="system",
            action="update_weights",
            resource="/model/v1",
            result=AuditResult.SUCCESS,
            metadata={"version": "1.0.1", "size": 1024}
        )
        
        audit_log(
            event_type=AuditEventType.SECURITY_VIOLATION,
            actor="attacker",
            action="unauthorized_access",
            resource="/admin",
            result=AuditResult.BLOCKED,
            metadata={"attempts": 3}
        )
        
        # Verify integrity
        results = logger.verify_integrity()
        self.assertTrue(results['valid'])
        self.assertEqual(results['total_entries'], 3)
        
        # Search events
        security_events = logger.search(event_type=AuditEventType.SECURITY_VIOLATION)
        self.assertEqual(len(security_events), 1)
        self.assertEqual(security_events[0].actor, "attacker")
        
        # Get statistics
        stats = logger.get_statistics()
        self.assertEqual(stats['total_entries'], 3)
        self.assertEqual(stats['entries_by_result']['blocked'], 1)
        
        # Cleanup
        logger.shutdown()
    
    def test_async_logging_performance(self):
        """Test async logging performance."""
        logger = AuditLogger(
            audit_dir=str(self.temp_dir),
            enable_async=True
        )
        
        # Log many entries quickly
        start_time = time.time()
        for i in range(100):
            logger.log(
                event_type=AuditEventType.USER_ACTION,
                actor=f"user{i}",
                action="action",
                resource=f"/resource{i}",
                metadata={"index": i}
            )
        elapsed = time.time() - start_time
        
        # Async should be fast (no disk I/O blocking)
        self.assertLess(elapsed, 1.0)
        
        # Wait for async writes to complete
        time.sleep(2.0)
        
        # Verify all entries were written
        results = logger.search(limit=1000)
        self.assertEqual(len(results), 100)
        
        logger.shutdown()


if __name__ == "__main__":
    unittest.main()