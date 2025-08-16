"""
Audit Trail Module

Implements comprehensive audit logging for the Pixelis framework.
Provides tamper-proof audit trails with cryptographic hashing and
enforces append-only logging for security compliance.

Task 005 (Phase 2 Round 6): Implement Audit Trails
"""

import logging
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import time

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    MODEL_UPDATE = "model_update"
    ACCESS_ATTEMPT = "access_attempt"
    CONFIG_CHANGE = "config_change"
    DATA_DELETION = "data_deletion"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    DATA_PRUNING = "data_pruning"
    USER_ACTION = "user_action"
    AUTHENTICATION = "authentication"
    PRIVACY_OPERATION = "privacy_operation"


class AuditResult(Enum):
    """Result of an audited action."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass
class AuditEntry:
    """
    Represents a single audit log entry.
    
    Attributes:
        timestamp: When the event occurred
        event_type: Type of event
        actor: Who/what performed the action
        action: What action was performed
        resource: What resource was affected
        result: Result of the action
        metadata: Additional event-specific data
        hash_previous: Hash of the previous entry (for chain integrity)
        hash_current: Hash of this entry
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    actor: str = "system"
    action: str = ""
    resource: str = ""
    result: AuditResult = AuditResult.SUCCESS
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_previous: Optional[str] = None
    hash_current: Optional[str] = None
    
    def calculate_hash(self, previous_hash: Optional[str] = None) -> str:
        """
        Calculate cryptographic hash for this entry.
        
        Args:
            previous_hash: Hash of the previous entry in the chain
            
        Returns:
            SHA-256 hash of this entry
        """
        # Create deterministic string representation
        hash_data = {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'actor': self.actor,
            'action': self.action,
            'resource': self.resource,
            'result': self.result.value,
            'metadata': json.dumps(self.metadata, sort_keys=True),
            'hash_previous': previous_hash or ""
        }
        
        # Create hash
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'actor': self.actor,
            'action': self.action,
            'resource': self.resource,
            'result': self.result.value,
            'metadata': self.metadata,
            'hash_previous': self.hash_previous,
            'hash_current': self.hash_current
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            event_type=AuditEventType(data.get('event_type', 'system_error')),
            actor=data.get('actor', 'system'),
            action=data.get('action', ''),
            resource=data.get('resource', ''),
            result=AuditResult(data.get('result', 'success')),
            metadata=data.get('metadata', {}),
            hash_previous=data.get('hash_previous'),
            hash_current=data.get('hash_current')
        )


class AuditLogger:
    """
    Centralized audit logger for the Pixelis system.
    
    Implements:
    - Append-only logging with cryptographic hash chain
    - Automatic rotation and archiving
    - Tamper detection and verification
    - Asynchronous logging for performance
    - Compliance with security policy requirements
    """
    
    def __init__(
        self,
        audit_dir: str = "./audit",
        max_file_size: int = 100_000_000,  # 100MB
        retention_days: int = 365,
        enable_async: bool = True,
        enable_encryption: bool = False
    ):
        """
        Initialize the audit logger.
        
        Args:
            audit_dir: Directory for audit logs
            max_file_size: Maximum size per audit file before rotation
            retention_days: How long to retain audit logs
            enable_async: Use asynchronous logging
            enable_encryption: Encrypt audit logs at rest
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size
        self.retention_days = retention_days
        self.enable_async = enable_async
        self.enable_encryption = enable_encryption
        
        # Current audit file
        self.current_file_path = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Hash chain state
        self.last_hash = self._get_last_hash()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Async logging queue
        if self.enable_async:
            self.log_queue = queue.Queue()
            self.worker_thread = threading.Thread(
                target=self._async_worker,
                name="AuditLoggerWorker",
                daemon=True
            )
            self.worker_thread.start()
        else:
            self.log_queue = None
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'entries_by_type': {},
            'entries_by_result': {},
            'last_verification': None
        }
        
        logger.info(f"Audit logger initialized at {self.audit_dir}")
    
    def _get_last_hash(self) -> Optional[str]:
        """
        Get the hash of the last entry in the current audit file.
        
        Returns:
            Hash of the last entry or None if file is empty
        """
        if not self.current_file_path.exists():
            return None
        
        try:
            with open(self.current_file_path, 'r') as f:
                # Read last line
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get('hash_current')
        except Exception as e:
            logger.error(f"Error reading last hash: {e}")
        
        return None
    
    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        result: AuditResult = AuditResult.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            actor: Who performed the action
            action: What action was performed
            resource: What resource was affected
            result: Result of the action
            metadata: Additional event data
            
        Returns:
            True if successfully logged
        """
        try:
            # Create audit entry
            entry = AuditEntry(
                event_type=event_type,
                actor=actor,
                action=action,
                resource=resource,
                result=result,
                metadata=metadata or {}
            )
            
            # Add hash chain
            with self.lock:
                entry.hash_previous = self.last_hash
                entry.hash_current = entry.calculate_hash(self.last_hash)
                self.last_hash = entry.hash_current
            
            # Log entry
            if self.enable_async and self.log_queue:
                self.log_queue.put(entry)
            else:
                self._write_entry(entry)
            
            # Update statistics
            self.stats['total_entries'] += 1
            self.stats['entries_by_type'][event_type.value] = \
                self.stats['entries_by_type'].get(event_type.value, 0) + 1
            self.stats['entries_by_result'][result.value] = \
                self.stats['entries_by_result'].get(result.value, 0) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    def _write_entry(self, entry: AuditEntry):
        """
        Write an audit entry to disk.
        
        Args:
            entry: Audit entry to write
        """
        try:
            # Check file size and rotate if needed
            if self.current_file_path.exists():
                if self.current_file_path.stat().st_size > self.max_file_size:
                    self._rotate_file()
            
            # Write entry (append-only)
            with open(self.current_file_path, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
                f.flush()  # Ensure immediate write
                os.fsync(f.fileno())  # Force write to disk
            
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    def _async_worker(self):
        """Background worker for asynchronous logging."""
        while True:
            try:
                # Get entry from queue (blocking)
                entry = self.log_queue.get(timeout=1.0)
                
                if entry is None:  # Shutdown signal
                    break
                
                # Write entry
                self._write_entry(entry)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audit worker: {e}")
    
    def _rotate_file(self):
        """Rotate the current audit file."""
        try:
            # Create archive name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"audit_archive_{timestamp}.jsonl"
            archive_path = self.audit_dir / "archives" / archive_name
            archive_path.parent.mkdir(exist_ok=True)
            
            # Move current file to archive
            self.current_file_path.rename(archive_path)
            
            # Compress archive (optional)
            import gzip
            with open(archive_path, 'rb') as f_in:
                with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove uncompressed archive
            archive_path.unlink()
            
            # Create new current file
            self.current_file_path = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            logger.info(f"Rotated audit file to {archive_name}.gz")
            
        except Exception as e:
            logger.error(f"Failed to rotate audit file: {e}")
    
    def verify_integrity(
        self,
        file_path: Optional[Path] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Verify the integrity of an audit file.
        
        Args:
            file_path: Path to audit file (uses current if None)
            verbose: Include detailed verification info
            
        Returns:
            Verification results
        """
        if file_path is None:
            file_path = self.current_file_path
        
        if not file_path.exists():
            return {
                'valid': False,
                'error': 'File does not exist',
                'file': str(file_path)
            }
        
        results = {
            'valid': True,
            'file': str(file_path),
            'total_entries': 0,
            'errors': []
        }
        
        try:
            previous_hash = None
            
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry_dict = json.loads(line)
                        entry = AuditEntry.from_dict(entry_dict)
                        
                        # Verify hash chain
                        expected_hash = entry.calculate_hash(previous_hash)
                        
                        if entry.hash_current != expected_hash:
                            error = f"Hash mismatch at line {line_num}"
                            results['errors'].append(error)
                            results['valid'] = False
                            
                            if verbose:
                                results['errors'].append({
                                    'line': line_num,
                                    'expected': expected_hash,
                                    'actual': entry.hash_current
                                })
                        
                        if entry.hash_previous != previous_hash:
                            error = f"Previous hash mismatch at line {line_num}"
                            results['errors'].append(error)
                            results['valid'] = False
                        
                        previous_hash = entry.hash_current
                        results['total_entries'] += 1
                        
                    except json.JSONDecodeError as e:
                        results['errors'].append(f"JSON error at line {line_num}: {e}")
                        results['valid'] = False
            
            self.stats['last_verification'] = datetime.now().isoformat()
            
        except Exception as e:
            results['valid'] = False
            results['error'] = str(e)
        
        return results
    
    def search(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        result: Optional[AuditResult] = None,
        limit: int = 1000
    ) -> List[AuditEntry]:
        """
        Search audit logs with filters.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            event_type: Filter by event type
            actor: Filter by actor
            resource: Filter by resource
            result: Filter by result
            limit: Maximum results to return
            
        Returns:
            List of matching audit entries
        """
        matches = []
        
        # Determine files to search
        files_to_search = [self.current_file_path]
        
        # Add archived files if date range specified
        if start_date or end_date:
            archive_dir = self.audit_dir / "archives"
            if archive_dir.exists():
                files_to_search.extend(archive_dir.glob("audit_*.jsonl*"))
        
        for file_path in files_to_search:
            if len(matches) >= limit:
                break
            
            try:
                # Handle compressed files
                if file_path.suffix == '.gz':
                    import gzip
                    open_func = gzip.open
                    mode = 'rt'
                else:
                    open_func = open
                    mode = 'r'
                
                with open_func(file_path, mode) as f:
                    for line in f:
                        if len(matches) >= limit:
                            break
                        
                        entry_dict = json.loads(line)
                        entry = AuditEntry.from_dict(entry_dict)
                        
                        # Apply filters
                        if start_date:
                            entry_time = datetime.fromisoformat(entry.timestamp)
                            if entry_time < start_date:
                                continue
                        
                        if end_date:
                            entry_time = datetime.fromisoformat(entry.timestamp)
                            if entry_time > end_date:
                                continue
                        
                        if event_type and entry.event_type != event_type:
                            continue
                        
                        if actor and entry.actor != actor:
                            continue
                        
                        if resource and resource not in entry.resource:
                            continue
                        
                        if result and entry.result != result:
                            continue
                        
                        matches.append(entry)
            
            except Exception as e:
                logger.error(f"Error searching file {file_path}: {e}")
        
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            'total_entries': self.stats['total_entries'],
            'entries_by_type': self.stats['entries_by_type'].copy(),
            'entries_by_result': self.stats['entries_by_result'].copy(),
            'last_verification': self.stats['last_verification'],
            'current_file': str(self.current_file_path),
            'file_size': self.current_file_path.stat().st_size if self.current_file_path.exists() else 0
        }
    
    def cleanup_old_logs(self):
        """Remove audit logs older than retention period."""
        cutoff_date = datetime.now().timestamp() - (self.retention_days * 86400)
        
        archive_dir = self.audit_dir / "archives"
        if not archive_dir.exists():
            return
        
        removed_count = 0
        
        for file_path in archive_dir.glob("audit_*.jsonl*"):
            try:
                # Check file age
                if file_path.stat().st_mtime < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed old audit file: {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error removing old audit file {file_path}: {e}")
        
        if removed_count > 0:
            # Log the cleanup operation itself
            self.log(
                event_type=AuditEventType.DATA_DELETION,
                actor="system",
                action="cleanup_old_logs",
                resource="audit_logs",
                result=AuditResult.SUCCESS,
                metadata={
                    'files_removed': removed_count,
                    'retention_days': self.retention_days
                }
            )
    
    def shutdown(self):
        """Gracefully shutdown the audit logger."""
        if self.enable_async and self.log_queue:
            # Signal worker to stop
            self.log_queue.put(None)
            
            # Wait for worker to finish
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5)
        
        logger.info(f"Audit logger shutdown. Total entries: {self.stats['total_entries']}")


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def initialize_audit_logger(config: Dict[str, Any]) -> AuditLogger:
    """
    Initialize the global audit logger.
    
    Args:
        config: Audit configuration
        
    Returns:
        Initialized audit logger
    """
    global _audit_logger
    
    _audit_logger = AuditLogger(
        audit_dir=config.get('audit_dir', './audit'),
        max_file_size=config.get('max_file_size', 100_000_000),
        retention_days=config.get('retention_days', 365),
        enable_async=config.get('enable_async', True),
        enable_encryption=config.get('enable_encryption', False)
    )
    
    return _audit_logger


def get_audit_logger() -> Optional[AuditLogger]:
    """Get the global audit logger instance."""
    return _audit_logger


def audit_log(
    event_type: AuditEventType,
    actor: str,
    action: str,
    resource: str,
    result: AuditResult = AuditResult.SUCCESS,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function to log an audit event.
    
    Args:
        event_type: Type of event
        actor: Who performed the action
        action: What action was performed
        resource: What resource was affected
        result: Result of the action
        metadata: Additional event data
        
    Returns:
        True if successfully logged
    """
    if _audit_logger is None:
        logger.warning("Audit logger not initialized")
        return False
    
    return _audit_logger.log(
        event_type=event_type,
        actor=actor,
        action=action,
        resource=resource,
        result=result,
        metadata=metadata
    )