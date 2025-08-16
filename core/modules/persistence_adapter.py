"""
Persistence Adapter Interface

Provides pluggable persistence backends for the Experience Buffer.
Supports file-based WAL and embedded KV stores like LMDB or SQLite.
"""

import os
import json
import pickle
import tempfile
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PersistenceAdapter(ABC):
    """Abstract base class for persistence adapters."""
    
    @abstractmethod
    def write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool:
        """Write an experience to persistent storage."""
        pass
    
    @abstractmethod
    def write_operation(self, operation: Dict[str, Any]) -> bool:
        """Write an operation to the operations log."""
        pass
    
    @abstractmethod
    def read_all_experiences(self) -> List[Dict[str, Any]]:
        """Read all experiences from storage."""
        pass
    
    @abstractmethod
    def read_all_operations(self) -> List[Dict[str, Any]]:
        """Read all operations from the operations log."""
        pass
    
    @abstractmethod
    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Save a complete snapshot."""
        pass
    
    @abstractmethod
    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot."""
        pass
    
    @abstractmethod
    def truncate_logs(self) -> bool:
        """Truncate both WAL logs after successful snapshot."""
        pass
    
    @abstractmethod
    def close(self):
        """Close any open resources."""
        pass


class FilePersistenceAdapter(PersistenceAdapter):
    """
    File-based persistence adapter using Write-Ahead Log pattern.
    Uses atomic writes (tmp -> fsync -> rename) for crash safety.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize file-based persistence.
        
        Args:
            base_path: Base directory for persistence files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.experience_wal = self.base_path / "experience_data.wal"
        self.operations_wal = self.base_path / "index_operations.wal"
        self.snapshot_dir = self.base_path / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Thread lock for atomic operations
        self.lock = threading.Lock()
        
        logger.info(f"File persistence adapter initialized at {self.base_path}")
    
    def _atomic_write(self, path: Path, data: str) -> bool:
        """
        Perform atomic write using temp file pattern.
        
        Args:
            path: Target file path
            data: Data to write
            
        Returns:
            True if successful
        """
        try:
            # Write to temporary file
            temp_fd, temp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=path.stem,
                suffix='.tmp'
            )
            
            with os.fdopen(temp_fd, 'w') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            shutil.move(temp_path, str(path))
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return False
    
    def write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool:
        """Write an experience to the WAL."""
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "experience_id": experience_id,
                "data": data
            }
            
            # Append to WAL
            try:
                with open(self.experience_wal, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as e:
                logger.error(f"Failed to write experience: {e}")
                return False
    
    def write_operation(self, operation: Dict[str, Any]) -> bool:
        """Write an operation to the operations log."""
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                **operation
            }
            
            try:
                with open(self.operations_wal, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as e:
                logger.error(f"Failed to write operation: {e}")
                return False
    
    def read_all_experiences(self) -> List[Dict[str, Any]]:
        """Read all experiences from the WAL."""
        experiences = []
        
        if not self.experience_wal.exists():
            return experiences
        
        try:
            with open(self.experience_wal, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        experiences.append(entry["data"])
        except Exception as e:
            logger.error(f"Failed to read experiences: {e}")
        
        return experiences
    
    def read_all_operations(self) -> List[Dict[str, Any]]:
        """Read all operations from the operations log."""
        operations = []
        
        if not self.operations_wal.exists():
            return operations
        
        try:
            with open(self.operations_wal, 'r') as f:
                for line in f:
                    if line.strip():
                        operations.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read operations: {e}")
        
        return operations
    
    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Save a complete snapshot."""
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = self.snapshot_dir / f"snapshot_{timestamp}.pkl"
            temp_path = snapshot_path.with_suffix('.tmp')
            
            try:
                # Write to temporary file
                with open(temp_path, 'wb') as f:
                    pickle.dump(snapshot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                temp_path.rename(snapshot_path)
                
                # Keep only the latest N snapshots
                self._cleanup_old_snapshots(keep=3)
                
                logger.info(f"Snapshot saved: {snapshot_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save snapshot: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                return False
    
    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot."""
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*.pkl"))
        
        if not snapshots:
            return None
        
        latest_snapshot = snapshots[-1]
        
        try:
            with open(latest_snapshot, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded snapshot: {latest_snapshot}")
            return data
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None
    
    def truncate_logs(self) -> bool:
        """Truncate both WAL logs after successful snapshot."""
        with self.lock:
            try:
                # Create empty files (truncate)
                for wal_path in [self.experience_wal, self.operations_wal]:
                    if wal_path.exists():
                        wal_path.unlink()
                    wal_path.touch()
                
                logger.info("WAL logs truncated")
                return True
            except Exception as e:
                logger.error(f"Failed to truncate logs: {e}")
                return False
    
    def _cleanup_old_snapshots(self, keep: int = 3):
        """Keep only the latest N snapshots."""
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*.pkl"))
        
        if len(snapshots) > keep:
            for snapshot in snapshots[:-keep]:
                try:
                    snapshot.unlink()
                    logger.debug(f"Deleted old snapshot: {snapshot}")
                except Exception as e:
                    logger.warning(f"Failed to delete snapshot {snapshot}: {e}")
    
    def close(self):
        """Close any open resources."""
        # File-based adapter doesn't hold persistent connections
        pass


class LMDBPersistenceAdapter(PersistenceAdapter):
    """
    LMDB-based persistence adapter for high-throughput scenarios.
    Provides transactional guarantees and better performance than file-based.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize LMDB-based persistence.
        
        Args:
            base_path: Base directory for LMDB database
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import lmdb
            self.lmdb = lmdb
        except ImportError:
            raise ImportError(
                "LMDB persistence requires 'lmdb' package. "
                "Install with: pip install lmdb"
            )
        
        # Open LMDB environment
        self.env = self.lmdb.open(
            str(self.base_path / "lmdb"),
            map_size=10 * 1024 * 1024 * 1024,  # 10GB
            max_dbs=3,
            sync=True,
            writemap=True
        )
        
        # Create named databases
        self.experiences_db = self.env.open_db(b'experiences')
        self.operations_db = self.env.open_db(b'operations')
        self.snapshots_db = self.env.open_db(b'snapshots')
        
        # Operation counter for ordering
        self.operation_counter = self._get_max_operation_id() + 1
        
        logger.info(f"LMDB persistence adapter initialized at {self.base_path}")
    
    def _get_max_operation_id(self) -> int:
        """Get the maximum operation ID from the database."""
        max_id = 0
        with self.env.begin(db=self.operations_db) as txn:
            cursor = txn.cursor()
            if cursor.last():
                max_id = int.from_bytes(cursor.key(), 'big')
        return max_id
    
    @contextmanager
    def _transaction(self, write: bool = True):
        """Context manager for LMDB transactions."""
        txn = self.env.begin(write=write)
        try:
            yield txn
            if write:
                txn.commit()
        except Exception:
            if write:
                txn.abort()
            raise
        finally:
            if not write:
                txn.abort()
    
    def write_experience(self, experience_id: str, data: Dict[str, Any]) -> bool:
        """Write an experience to LMDB."""
        try:
            with self._transaction() as txn:
                key = experience_id.encode('utf-8')
                value = json.dumps(data).encode('utf-8')
                txn.put(key, value, db=self.experiences_db)
            return True
        except Exception as e:
            logger.error(f"Failed to write experience: {e}")
            return False
    
    def write_operation(self, operation: Dict[str, Any]) -> bool:
        """Write an operation to LMDB."""
        try:
            with self._transaction() as txn:
                # Use incrementing counter as key for ordering
                key = self.operation_counter.to_bytes(8, 'big')
                value = json.dumps(operation).encode('utf-8')
                txn.put(key, value, db=self.operations_db)
                self.operation_counter += 1
            return True
        except Exception as e:
            logger.error(f"Failed to write operation: {e}")
            return False
    
    def read_all_experiences(self) -> List[Dict[str, Any]]:
        """Read all experiences from LMDB."""
        experiences = []
        
        try:
            with self._transaction(write=False) as txn:
                cursor = txn.cursor(db=self.experiences_db)
                for key, value in cursor:
                    data = json.loads(value.decode('utf-8'))
                    experiences.append(data)
        except Exception as e:
            logger.error(f"Failed to read experiences: {e}")
        
        return experiences
    
    def read_all_operations(self) -> List[Dict[str, Any]]:
        """Read all operations from LMDB in order."""
        operations = []
        
        try:
            with self._transaction(write=False) as txn:
                cursor = txn.cursor(db=self.operations_db)
                for key, value in cursor:
                    operation = json.loads(value.decode('utf-8'))
                    operations.append(operation)
        except Exception as e:
            logger.error(f"Failed to read operations: {e}")
        
        return operations
    
    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Save a snapshot to LMDB."""
        try:
            timestamp = datetime.now().isoformat()
            with self._transaction() as txn:
                key = timestamp.encode('utf-8')
                value = pickle.dumps(snapshot_data, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, value, db=self.snapshots_db)
            
            # Cleanup old snapshots
            self._cleanup_old_snapshots(keep=3)
            
            logger.info(f"Snapshot saved: {timestamp}")
            return True
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
    
    def load_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot from LMDB."""
        try:
            with self._transaction(write=False) as txn:
                cursor = txn.cursor(db=self.snapshots_db)
                if cursor.last():
                    value = cursor.value()
                    data = pickle.loads(value)
                    return data
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
        
        return None
    
    def truncate_logs(self) -> bool:
        """Clear the operations log after snapshot."""
        try:
            with self._transaction() as txn:
                # Clear operations database
                txn.drop(self.operations_db, delete=False)
                
                # Reset operation counter
                self.operation_counter = 1
            
            logger.info("Operations log truncated")
            return True
        except Exception as e:
            logger.error(f"Failed to truncate logs: {e}")
            return False
    
    def _cleanup_old_snapshots(self, keep: int = 3):
        """Keep only the latest N snapshots."""
        try:
            with self._transaction() as txn:
                cursor = txn.cursor(db=self.snapshots_db)
                
                # Count snapshots
                count = txn.stat(db=self.snapshots_db)['entries']
                
                if count > keep:
                    # Delete oldest snapshots
                    to_delete = count - keep
                    cursor.first()
                    for _ in range(to_delete):
                        cursor.delete()
                        cursor.next()
        except Exception as e:
            logger.warning(f"Failed to cleanup snapshots: {e}")
    
    def close(self):
        """Close LMDB environment."""
        if hasattr(self, 'env'):
            self.env.close()
            logger.info("LMDB environment closed")


def create_persistence_adapter(adapter_type: str, base_path: str) -> PersistenceAdapter:
    """
    Factory function to create persistence adapters.
    
    Args:
        adapter_type: Type of adapter ('file' or 'lmdb')
        base_path: Base path for persistence
        
    Returns:
        PersistenceAdapter instance
    """
    if adapter_type == "file":
        return FilePersistenceAdapter(base_path)
    elif adapter_type == "lmdb":
        return LMDBPersistenceAdapter(base_path)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")