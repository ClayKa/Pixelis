"""
Experience Buffer Module

Implements an intelligent experience buffer with k-NN retrieval capabilities
for the online learning system. Includes priority-based sampling and
efficient similarity search using FAISS.
"""

import logging
import torch
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import faiss
import pickle
import json
from pathlib import Path
from datetime import datetime
import threading
from ..data_structures import Experience, ExperienceStatus

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Intelligent experience buffer with k-NN retrieval.
    
    Features:
    - Priority-based sampling
    - Hybrid k-NN retrieval (visual + text similarity)
    - Automatic size management with deque
    - Persistence with Write-Ahead Log (WAL)
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        embedding_dim: int = 768,
        similarity_metric: str = "cosine",
        enable_persistence: bool = True,
        persistence_path: str = "./experience_buffer"
    ):
        """
        Initialize the experience buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            embedding_dim: Dimension of embeddings for similarity search
            similarity_metric: Metric for similarity ('cosine', 'euclidean', 'manhattan')
            enable_persistence: Whether to enable persistence
            persistence_path: Path for persistent storage
        """
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path)
        
        # Core data structure - deque for automatic size management
        self.buffer = deque(maxlen=max_size)
        
        # Experience lookup by ID
        self.experience_dict: Dict[str, Experience] = {}
        
        # FAISS index for k-NN search
        self.index = self._create_faiss_index()
        self.index_to_id: Dict[int, str] = {}  # Maps FAISS index to experience ID
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Persistence components
        if self.enable_persistence:
            self._setup_persistence()
        
        # Statistics
        self.total_additions = 0
        self.total_retrievals = 0
        
        logger.info(f"Experience buffer initialized with max_size={max_size}")
    
    def _create_faiss_index(self) -> faiss.Index:
        """
        Create FAISS index based on similarity metric.
        
        Returns:
            FAISS index object
        """
        if self.similarity_metric == "cosine":
            # Use Inner Product for cosine similarity (with L2 normalized vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.similarity_metric == "euclidean":
            # Use L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.similarity_metric == "manhattan":
            # FAISS doesn't support L1 directly, use L2 as approximation
            logger.warning("Manhattan distance not directly supported, using L2")
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        # Wrap in IDMap for tracking
        index = faiss.IndexIDMap(index)
        
        return index
    
    def _setup_persistence(self):
        """Setup persistence with Write-Ahead Log."""
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for different components
        self.wal_path = self.persistence_path / "wal.jsonl"
        self.checkpoint_path = self.persistence_path / "checkpoint.pkl"
        self.index_path = self.persistence_path / "faiss.index"
        
        # Load existing data if available
        self._load_from_disk()
    
    def add(self, experience: Experience) -> bool:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
            
        Returns:
            True if successfully added
        """
        with self.lock:
            try:
                # Check if experience already exists
                if experience.experience_id in self.experience_dict:
                    logger.warning(f"Experience {experience.experience_id} already exists")
                    return False
                
                # Calculate priority if not set
                if experience.priority == 1.0:
                    uncertainty = 1.0 - experience.model_confidence
                    reward = experience.trajectory.total_reward
                    experience.update_priority(uncertainty, reward)
                
                # Add to buffer
                self.buffer.append(experience.experience_id)
                self.experience_dict[experience.experience_id] = experience
                
                # Add to index if embedding is available
                if experience.embeddings and "combined" in experience.embeddings:
                    self._add_to_index(experience)
                
                # Write to WAL if persistence is enabled
                if self.enable_persistence:
                    self._write_to_wal(experience)
                
                self.total_additions += 1
                
                # Trigger checkpoint if needed
                if self.total_additions % 100 == 0 and self.enable_persistence:
                    self._save_checkpoint()
                
                logger.debug(f"Added experience {experience.experience_id} to buffer")
                return True
                
            except Exception as e:
                logger.error(f"Error adding experience: {e}")
                return False
    
    def _add_to_index(self, experience: Experience):
        """
        Add experience embedding to FAISS index.
        
        Args:
            experience: Experience with embedding
        """
        embedding = experience.get_embedding("combined")
        if embedding is None:
            return
        
        # Convert to numpy and ensure correct shape
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.similarity_metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Add to FAISS index with ID
        faiss_id = len(self.index_to_id)
        self.index.add_with_ids(embedding, np.array([faiss_id], dtype=np.int64))
        self.index_to_id[faiss_id] = experience.experience_id
    
    def search_index(
        self,
        query: Union[Experience, Dict[str, Any], torch.Tensor],
        k: int = 5
    ) -> List[Experience]:
        """
        Search for k nearest neighbors using hybrid similarity.
        
        Args:
            query: Query experience, embedding, or features
            k: Number of neighbors to retrieve
            
        Returns:
            List of k nearest experiences
        """
        with self.lock:
            # Extract or compute query embedding
            query_embedding = self._get_query_embedding(query)
            
            if query_embedding is None or self.index.ntotal == 0:
                # Fallback to random sampling if no embedding or empty index
                return self._random_sample(k)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Retrieve experiences
            neighbors = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx in self.index_to_id:
                    exp_id = self.index_to_id[idx]
                    if exp_id in self.experience_dict:
                        experience = self.experience_dict[exp_id]
                        experience.update_usage_stats(was_successful=True)  # Will be updated later
                        neighbors.append(experience)
            
            self.total_retrievals += len(neighbors)
            
            return neighbors
    
    def _get_query_embedding(
        self,
        query: Union[Experience, Dict[str, Any], torch.Tensor]
    ) -> Optional[np.ndarray]:
        """
        Extract or compute query embedding.
        
        Args:
            query: Query in various formats
            
        Returns:
            Query embedding as numpy array
        """
        if isinstance(query, torch.Tensor):
            embedding = query.detach().cpu().numpy()
        elif isinstance(query, np.ndarray):
            embedding = query
        elif isinstance(query, Experience):
            embedding = query.get_embedding("combined")
            if embedding is not None and isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
        elif isinstance(query, dict):
            # Try to extract embedding from dict
            if "embedding" in query:
                embedding = query["embedding"]
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
            else:
                return None
        else:
            return None
        
        # Ensure correct shape
        if embedding is not None and len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if embedding is not None and self.similarity_metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def sample_by_priority(self, n: int = 1) -> List[Experience]:
        """
        Sample experiences based on priority scores.
        
        Args:
            n: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        with self.lock:
            if not self.experience_dict:
                return []
            
            # Get all experiences with their priorities
            experiences = list(self.experience_dict.values())
            priorities = np.array([exp.priority for exp in experiences])
            
            # Normalize priorities to probabilities
            if priorities.sum() > 0:
                probabilities = priorities / priorities.sum()
            else:
                probabilities = np.ones(len(priorities)) / len(priorities)
            
            # Sample without replacement
            n = min(n, len(experiences))
            indices = np.random.choice(
                len(experiences),
                size=n,
                replace=False,
                p=probabilities
            )
            
            return [experiences[i] for i in indices]
    
    def _random_sample(self, n: int) -> List[Experience]:
        """
        Random sample from buffer.
        
        Args:
            n: Number of samples
            
        Returns:
            List of sampled experiences
        """
        if not self.experience_dict:
            return []
        
        experiences = list(self.experience_dict.values())
        n = min(n, len(experiences))
        indices = np.random.choice(len(experiences), size=n, replace=False)
        
        return [experiences[i] for i in indices]
    
    def get(self, experience_id: str) -> Optional[Experience]:
        """
        Get experience by ID.
        
        Args:
            experience_id: Experience identifier
            
        Returns:
            Experience if found, None otherwise
        """
        with self.lock:
            return self.experience_dict.get(experience_id)
    
    def remove(self, experience_id: str) -> bool:
        """
        Remove an experience from the buffer.
        
        Args:
            experience_id: Experience identifier
            
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if experience_id not in self.experience_dict:
                return False
            
            # Remove from dict
            del self.experience_dict[experience_id]
            
            # Note: Removal from deque is expensive, skip for performance
            # The deque will naturally evict old IDs when full
            
            # TODO: Remove from FAISS index (requires rebuilding)
            
            return True
    
    def clear(self):
        """Clear all experiences from the buffer."""
        with self.lock:
            self.buffer.clear()
            self.experience_dict.clear()
            self.index = self._create_faiss_index()
            self.index_to_id.clear()
            
            logger.info("Experience buffer cleared")
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.experience_dict)
    
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self.buffer) >= self.max_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            if self.experience_dict:
                confidences = [exp.model_confidence for exp in self.experience_dict.values()]
                priorities = [exp.priority for exp in self.experience_dict.values()]
                usage_counts = [exp.usage_count for exp in self.experience_dict.values()]
                success_rates = [exp.success_rate for exp in self.experience_dict.values()]
                
                stats = {
                    "size": self.size(),
                    "max_size": self.max_size,
                    "utilization": self.size() / self.max_size,
                    "total_additions": self.total_additions,
                    "total_retrievals": self.total_retrievals,
                    "avg_confidence": np.mean(confidences),
                    "avg_priority": np.mean(priorities),
                    "avg_usage_count": np.mean(usage_counts),
                    "avg_success_rate": np.mean(success_rates),
                    "index_size": self.index.ntotal
                }
            else:
                stats = {
                    "size": 0,
                    "max_size": self.max_size,
                    "utilization": 0.0,
                    "total_additions": self.total_additions,
                    "total_retrievals": self.total_retrievals,
                    "index_size": 0
                }
            
            return stats
    
    # Persistence methods
    
    def _write_to_wal(self, experience: Experience):
        """
        Write experience to Write-Ahead Log.
        
        Args:
            experience: Experience to persist
        """
        try:
            with open(self.wal_path, "a") as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "operation": "add",
                    "data": experience.to_dict()
                }
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to WAL: {e}")
    
    def _save_checkpoint(self):
        """Save full checkpoint to disk."""
        try:
            checkpoint = {
                "buffer": list(self.buffer),
                "experiences": {
                    exp_id: exp.to_dict()
                    for exp_id, exp in self.experience_dict.items()
                },
                "index_to_id": self.index_to_id,
                "statistics": {
                    "total_additions": self.total_additions,
                    "total_retrievals": self.total_retrievals
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save checkpoint
            temp_path = self.checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                pickle.dump(checkpoint, f)
            
            # Atomic rename
            temp_path.rename(self.checkpoint_path)
            
            # Save FAISS index
            if self.index.ntotal > 0:
                faiss.write_index(self.index, str(self.index_path))
            
            # Clear WAL after successful checkpoint
            if self.wal_path.exists():
                self.wal_path.unlink()
            
            logger.debug("Checkpoint saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _load_from_disk(self):
        """Load buffer from disk."""
        try:
            # Load checkpoint if exists
            if self.checkpoint_path.exists():
                with open(self.checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)
                
                # Restore buffer
                self.buffer = deque(checkpoint["buffer"], maxlen=self.max_size)
                
                # Restore experiences (without image features)
                for exp_id, exp_dict in checkpoint["experiences"].items():
                    exp = Experience.from_dict(exp_dict)
                    self.experience_dict[exp_id] = exp
                
                # Restore index mapping
                self.index_to_id = checkpoint["index_to_id"]
                
                # Restore statistics
                self.total_additions = checkpoint["statistics"]["total_additions"]
                self.total_retrievals = checkpoint["statistics"]["total_retrievals"]
                
                # Load FAISS index
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                
                logger.info(f"Loaded {len(self.experience_dict)} experiences from checkpoint")
            
            # Apply WAL if exists
            if self.wal_path.exists():
                self._apply_wal()
            
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
    
    def _apply_wal(self):
        """Apply Write-Ahead Log entries."""
        try:
            with open(self.wal_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["operation"] == "add":
                        exp = Experience.from_dict(entry["data"])
                        self.experience_dict[exp.experience_id] = exp
                        self.buffer.append(exp.experience_id)
            
            logger.info("Applied WAL entries")
            
        except Exception as e:
            logger.error(f"Error applying WAL: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the buffer."""
        if self.enable_persistence:
            self._save_checkpoint()
        
        logger.info("Experience buffer shutdown complete")