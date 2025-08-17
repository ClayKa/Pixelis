"""
Inference Engine Module

Manages the primary predict-and-adapt online learning loop for the Pixelis framework.
This module orchestrates the main inference process, temporal ensemble voting,
confidence-based gating, and communication with the update worker process.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from dataclasses import dataclass
import time
import os
import signal
from datetime import datetime, timedelta
import threading
import numpy as np

logger = logging.getLogger(__name__)

# Import alerting and monitoring modules
from ..modules.alerter import Alerter, HealthMonitor, AlertSeverity
# Import privacy and security modules
from ..modules.privacy import DataAnonymizer, PrivacyConfig

# Shared memory management classes
@dataclass
class SharedMemoryInfo:
    """Information about a shared memory segment."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    created_at: datetime
    size_bytes: int
    
    def age_seconds(self) -> float:
        """Get age of the shared memory segment in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class SharedMemoryManager:
    """
    Manages shared memory segments for tensor transfer between processes.
    
    Implements a robust lifecycle management system with:
    - Creation and tracking of shared memory segments
    - Watchdog-based cleanup of stale segments
    - Confirmation-based cleanup for normal operation
    """
    
    def __init__(self, timeout_seconds: float = 60.0):
        """
        Initialize the shared memory manager.
        
        Args:
            timeout_seconds: Timeout for stale segment cleanup
        """
        self.pending_shm: Dict[str, SharedMemoryInfo] = {}
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()
        self._shared_memory_cache: Dict[str, torch.Storage] = {}
        
    def create_shared_tensor(self, tensor: torch.Tensor) -> SharedMemoryInfo:
        """
        Create a shared memory segment for a tensor.
        
        Args:
            tensor: Tensor to share
            
        Returns:
            SharedMemoryInfo with metadata about the shared segment
        """
        # Move tensor to CPU pinned memory for efficient transfer
        if tensor.is_cuda:
            tensor = tensor.to('cpu', non_blocking=True).pin_memory()
        elif not tensor.is_pinned:
            tensor = tensor.pin_memory()
        
        # Generate unique name for shared memory segment
        import uuid
        shm_name = f"pixelis_shm_{uuid.uuid4().hex}"
        
        # Create shared memory storage
        storage = tensor.storage()._share_memory_()
        
        # Cache the storage for cleanup
        self._shared_memory_cache[shm_name] = storage
        
        # Create metadata
        shm_info = SharedMemoryInfo(
            name=shm_name,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            created_at=datetime.now(),
            size_bytes=tensor.element_size() * tensor.numel()
        )
        
        # Track the segment
        with self.lock:
            self.pending_shm[shm_name] = shm_info
        
        logger.debug(f"Created shared memory segment: {shm_name} ({shm_info.size_bytes} bytes)")
        
        return shm_info
    
    def reconstruct_tensor(self, shm_info: SharedMemoryInfo) -> torch.Tensor:
        """
        Reconstruct a tensor from shared memory info.
        
        NOTE: This method is intended to show the interface.
        In production, PyTorch's built-in shared memory through
        tensor.share_memory_() should be used with proper IPC handle passing.
        
        Args:
            shm_info: Shared memory metadata
            
        Returns:
            Reconstructed tensor
        """
        # Implementation note: In production, use torch's built-in shared memory:
        # 1. Parent process: tensor.share_memory_() 
        # 2. Pass tensor storage handle through queue
        # 3. Child process: Reconstruct from handle
        
        # For now, return a placeholder tensor with correct shape/dtype
        logger.warning(f"Using placeholder tensor reconstruction for {shm_info.name}")
        tensor = torch.zeros(shm_info.shape, dtype=shm_info.dtype)
        return tensor
    
    def mark_cleaned(self, shm_name: str):
        """
        Mark a shared memory segment as cleaned up.
        
        Args:
            shm_name: Name of the cleaned segment
        """
        with self.lock:
            if shm_name in self.pending_shm:
                del self.pending_shm[shm_name]
                logger.debug(f"Marked shared memory segment as cleaned: {shm_name}")
            
            # Clean up cached storage
            if shm_name in self._shared_memory_cache:
                del self._shared_memory_cache[shm_name]
    
    def cleanup_stale_segments(self, worker_alive: bool = True) -> List[str]:
        """
        Clean up stale shared memory segments.
        
        Args:
            worker_alive: Whether the worker process is alive
            
        Returns:
            List of cleaned segment names
        """
        cleaned = []
        current_time = datetime.now()
        
        with self.lock:
            stale_segments = []
            
            for shm_name, shm_info in list(self.pending_shm.items()):
                age = shm_info.age_seconds()
                logger.debug(f"[Watchdog] Checking segment {shm_name}, age: {age:.2f}s, timeout: {self.timeout_seconds}s")
                
                # Clean if: timeout exceeded OR worker is dead
                if age > self.timeout_seconds or not worker_alive:
                    stale_segments.append((shm_name, shm_info))
            
            for shm_name, shm_info in stale_segments:
                if not worker_alive:
                    logger.warning(f"[Watchdog] Worker dead, cleaning up segment: {shm_name}")
                else:
                    logger.warning(
                        f"[Watchdog] Segment {shm_name} exceeded timeout "
                        f"({shm_info.age_seconds():.1f}s > {self.timeout_seconds}s), cleaning up"
                    )
                
                # Unlink the shared memory segment
                self._unlink_segment(shm_name)
                del self.pending_shm[shm_name]
                cleaned.append(shm_name)
        
        return cleaned
    
    def _unlink_segment(self, shm_name: str):
        """
        Unlink a shared memory segment.
        
        Args:
            shm_name: Name of segment to unlink
        """
        # Clean up cached storage
        if shm_name in self._shared_memory_cache:
            try:
                # In PyTorch, shared memory cleanup happens automatically
                # when the storage object is deleted
                del self._shared_memory_cache[shm_name]
            except Exception as e:
                logger.error(f"Error unlinking segment {shm_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the shared memory manager.
        
        Returns:
            Status dictionary
        """
        with self.lock:
            total_bytes = sum(info.size_bytes for info in self.pending_shm.values())
            oldest_age = max(
                (info.age_seconds() for info in self.pending_shm.values()),
                default=0
            )
            
            return {
                "pending_segments": len(self.pending_shm),
                "total_bytes": total_bytes,
                "oldest_segment_age": oldest_age
            }


class InferenceEngine:
    """
    Core inference engine for the Pixelis framework.
    
    Handles:
    - Model inference and prediction
    - k-NN retrieval from experience buffer
    - Temporal ensemble voting
    - Confidence gating for learning updates
    - Communication with the update worker
    - Shared memory management for tensor transfer
    - Watchdog for resource cleanup
    """
    
    def __init__(
        self,
        model,
        experience_buffer,
        voting_module,
        reward_orchestrator,
        config: Dict[str, Any]
    ):
        """
        Initialize the Inference Engine.
        
        Args:
            model: The main model for inference
            experience_buffer: Experience buffer for k-NN retrieval
            voting_module: Module for temporal ensemble voting
            reward_orchestrator: Module for reward calculation
            config: Configuration dictionary
        """
        self.model = model
        self.experience_buffer = experience_buffer
        self.voting_module = voting_module
        self.reward_orchestrator = reward_orchestrator
        self.config = config
        
        # Initialize IPC queues with size limits
        max_queue_size = config.get('max_queue_size', 1000)
        self.request_queue: Queue = mp.Queue(maxsize=max_queue_size)
        self.response_queue: Queue = mp.Queue(maxsize=max_queue_size)
        self.update_queue: Queue = mp.Queue(maxsize=max_queue_size)
        self.cleanup_confirmation_queue: Queue = mp.Queue(maxsize=max_queue_size * 2)
        
        # Human-in-the-Loop (HIL) queue for expert review
        self.human_review_queue: Queue = mp.Queue(maxsize=max_queue_size)
        
        # Shared memory manager
        self.shm_manager = SharedMemoryManager(
            timeout_seconds=config.get('shm_timeout', 60.0)
        )
        
        # Update worker process
        self.update_worker_process: Optional[Process] = None
        
        # Confidence threshold for triggering updates
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Learning rate bounds
        self.min_lr = config.get('min_learning_rate', 1e-6)
        self.max_lr = config.get('max_learning_rate', 1e-4)
        
        # Human-in-the-Loop (HIL) configuration
        self.hil_mode_enabled = config.get('hil_mode_enabled', False)
        self.hil_review_percentage = config.get('hil_review_percentage', 0.02)  # 2% by default
        self.hil_review_counter = 0
        
        # Watchdog settings
        self.watchdog_interval = config.get('watchdog_interval', 5.0)
        self.watchdog_thread: Optional[threading.Thread] = None
        self.watchdog_running = False
        
        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'total_requests': 0,
            'total_updates': 0,
            'watchdog_cleanups': 0,
            'failed_updates': 0,
            'faiss_failures': 0,
            'critical_failures': 0
        }
        
        # Task 003: Initialize alerter and health monitor
        self.alerter = Alerter(config)
        self.health_monitor = HealthMonitor(self.alerter)
        
        # Monitoring thread
        self.monitoring_interval = config.get('monitoring_interval', 10.0)
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_running = False
        
        # Task 002 (Phase 2 Round 6): Initialize privacy protection
        privacy_config = PrivacyConfig(
            enable_pii_redaction=config.get('enable_pii_redaction', True),
            enable_image_metadata_stripping=config.get('enable_metadata_stripping', True),
            enable_differential_privacy=config.get('enable_differential_privacy', False),
            log_redaction_stats=config.get('log_privacy_stats', True)
        )
        self.data_anonymizer = DataAnonymizer(privacy_config)
        
        # Read-only mode for public demonstrator (Task 003 Phase 2 Round 6)
        self.read_only_mode = config.get('read_only_mode', False)
        if self.read_only_mode:
            logger.warning("Inference Engine initialized in READ-ONLY MODE - no learning or updates will occur")
        
        logger.info("Inference Engine initialized with monitoring, alerting, and privacy protection")
    
    async def infer_and_adapt(
        self,
        input_data: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Main inference and adaptation loop with cold start bootstrapping.
        
        Orchestrates the complete online evolution loop:
        1. Model inference
        2. Experience buffer retrieval (if available)
        3. Temporal ensemble voting (if sufficient experiences)
        4. Confidence-gated learning updates
        5. Monitoring and health tracking
        
        Args:
            input_data: Input data containing image features and question
            
        Returns:
            Tuple of (prediction, confidence_score, metadata)
        """
        start_time = time.time()
        metadata = {
            'inference_path': 'unknown',
            'buffer_size': len(self.experience_buffer) if hasattr(self.experience_buffer, '__len__') else 0,
            'cold_start_active': False
        }
        
        try:
            # Step 1: Get initial model prediction
            with torch.no_grad():
                initial_prediction = await self._get_model_prediction(input_data)
            
            # Task 002: Cold Start Bootstrapping Strategy
            # Check if we're in cold start mode
            cold_start_threshold = self.config.get('cold_start_threshold', 100)
            buffer_size = len(self.experience_buffer) if hasattr(self.experience_buffer, '__len__') else 0
            
            if buffer_size < cold_start_threshold:
                # Conservative mode: bypass ensemble voting, build memory
                logger.info(
                    f"Cold start mode active: buffer size {buffer_size} < threshold {cold_start_threshold}"
                )
                metadata['cold_start_active'] = True
                metadata['inference_path'] = 'cold_start_direct'
                
                # Add experience to buffer with high priority for rapid memory building
                await self._add_bootstrap_experience(input_data, initial_prediction)
                
                # Return direct model prediction without ensemble or updates
                confidence = initial_prediction.get('confidence', 0.5) if isinstance(initial_prediction, dict) else 0.5
                
                # Log cold start metrics
                self._log_inference_metrics({
                    'mode': 'cold_start',
                    'buffer_size': buffer_size,
                    'confidence': confidence,
                    'inference_time': time.time() - start_time
                })
                
                # Construct the result dictionary explicitly
                if isinstance(initial_prediction, dict):
                    result_dict = {
                        "answer": initial_prediction.get('answer', ''),
                        "trajectory": initial_prediction.get('trajectory', [])
                    }
                else:
                    # Handle case where initial_prediction is not a dict
                    result_dict = {
                        "answer": initial_prediction,
                        "trajectory": []
                    }
                
                return (
                    result_dict,
                    confidence,
                    metadata
                )
            
            # Normal mode: Full ensemble voting and adaptation
            metadata['inference_path'] = 'ensemble'
            
            # Step 2: Retrieve k-NN neighbors from experience buffer
            neighbors = []
            try:
                neighbors = self.experience_buffer.search_index(
                    input_data,
                    k=self.config.get('k_neighbors', 5)
                )
                metadata['neighbors_retrieved'] = len(neighbors)
            except Exception as e:
                logger.warning(f"k-NN retrieval failed: {e}")
                metadata['knn_failure'] = str(e)
                # Track failure rate for monitoring
                with self.stats_lock:
                    self.stats['faiss_failures'] = self.stats.get('faiss_failures', 0) + 1
            
            # Step 3: Apply temporal ensemble voting
            voting_result = None
            if neighbors:
                try:
                    voting_result = self.voting_module.vote(
                        initial_prediction,
                        neighbors,
                        strategy=self.config.get('voting_strategy', 'weighted')
                    )
                    metadata['voting_strategy'] = self.config.get('voting_strategy', 'weighted')
                    metadata['voting_confidence'] = voting_result.confidence
                except Exception as e:
                    logger.error(f"Voting failed: {e}")
                    metadata['voting_failure'] = str(e)
            
            # Fallback if voting failed
            if voting_result is None:
                from types import SimpleNamespace
                # Create properly formatted answer dictionary for consistency
                if isinstance(initial_prediction, dict):
                    answer_dict = {
                        "answer": initial_prediction.get('answer', ''),
                        "trajectory": initial_prediction.get('trajectory', [])
                    }
                    confidence = initial_prediction.get('confidence', 0.5)
                else:
                    answer_dict = {
                        "answer": initial_prediction,
                        "trajectory": []
                    }
                    confidence = 0.5
                
                voting_result = SimpleNamespace(
                    final_answer=answer_dict,
                    confidence=confidence,
                    provenance={'source': 'direct_model', 'fallback': True}
                )
            
            # Task 003 (Phase 2 Round 6): Check for read-only mode
            if self.read_only_mode:
                logger.debug("Read-only mode active - skipping all updates and learning")
                metadata['read_only'] = True
                metadata['update_path'] = 'disabled'
                # Return response without any updates or storage
                return (
                    voting_result.final_answer,
                    voting_result.confidence,
                    {**voting_result.provenance, **metadata}
                )
            
            # Step 4: Check confidence and potentially trigger update
            update_triggered = False
            if self._should_trigger_update(voting_result.confidence):
                # Determine if this update should go through human review
                if self._should_request_human_review():
                    await self._enqueue_human_review_task(
                        input_data,
                        voting_result,
                        initial_prediction
                    )
                    metadata['update_path'] = 'human_review'
                else:
                    await self._enqueue_update_task(
                        input_data,
                        voting_result,
                        initial_prediction
                    )
                    metadata['update_path'] = 'automatic'
                update_triggered = True
            
            # Step 5: Add to experience buffer for future use (only if not read-only)
            if not self.read_only_mode:
                await self._add_to_experience_buffer(
                    input_data,
                    voting_result,
                    initial_prediction
                )
            
            # Log comprehensive metrics
            self._log_inference_metrics({
                'mode': 'ensemble',
                'buffer_size': buffer_size,
                'confidence': voting_result.confidence,
                'update_triggered': update_triggered,
                'neighbors_used': len(neighbors),
                'inference_time': time.time() - start_time,
                'queue_sizes': self._get_queue_sizes()
            })
            
            # Update metadata with timing
            metadata['inference_time_ms'] = (time.time() - start_time) * 1000
            
            return (
                voting_result.final_answer,
                voting_result.confidence,
                {**voting_result.provenance, **metadata}
            )
            
        except Exception as e:
            logger.error(f"Critical error in infer_and_adapt: {e}", exc_info=True)
            # Track critical failures
            with self.stats_lock:
                self.stats['critical_failures'] = self.stats.get('critical_failures', 0) + 1
            
            # Return fallback response
            return (
                None,
                0.0,
                {'error': str(e), 'inference_path': 'error', **metadata}
            )
    
    async def _get_model_prediction(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get prediction from the model.
        
        Args:
            input_data: Input data for the model
            
        Returns:
            Model prediction dictionary
        """
        # Implementation will depend on the specific model interface
        # This is a placeholder
        return self.model.forward(input_data)
    
    def _should_trigger_update(self, confidence: float) -> bool:
        """
        Determine if a learning update should be triggered.
        
        Implements confidence gating mechanism:
        - Only trigger updates when confidence exceeds threshold
        - This ensures we learn from high-quality pseudo-labels
        
        Args:
            confidence: Confidence score from voting (0 to 1)
            
        Returns:
            Boolean indicating whether to trigger update
        """
        # Confidence gate: only learn from high-confidence predictions
        should_update = confidence >= self.confidence_threshold
        
        if should_update:
            logger.debug(
                f"Confidence gate PASSED: {confidence:.3f} >= {self.confidence_threshold:.3f}, "
                "triggering learning update"
            )
        else:
            logger.debug(
                f"Confidence gate FAILED: {confidence:.3f} < {self.confidence_threshold:.3f}, "
                "skipping learning update"
            )
        
        return should_update
    
    def _calculate_adaptive_lr(self, confidence: float) -> float:
        """
        Calculate adaptive learning rate based on confidence.
        
        Implements proportional and bounded learning rate strategy:
        - Learning rate is proportional to error (1 - confidence)
        - Higher confidence → lower learning rate (conservative updates)
        - Lower confidence → higher learning rate (larger corrections)
        - Bounded within [min_lr, max_lr] for stability
        
        Formula: lr = lr_base * (1.0 - confidence)
        Where lr_base is the maximum learning rate from config
        
        Args:
            confidence: Confidence score (0 to 1)
            
        Returns:
            Adaptive learning rate, bounded within safe range
        """
        # Calculate proportional learning rate
        # When confidence is high (near 1.0), error is low, so LR is low
        # When confidence is low (near 0.0), error is high, so LR is high
        error = 1.0 - confidence
        lr_proportional = self.max_lr * error
        
        # Apply safety bounds to prevent instability
        # Clip to [min_lr, max_lr] range
        lr_bounded = np.clip(lr_proportional, self.min_lr, self.max_lr)
        
        logger.debug(
            f"Adaptive LR calculation: confidence={confidence:.3f}, "
            f"error={error:.3f}, lr_prop={lr_proportional:.6f}, "
            f"lr_final={lr_bounded:.6f} (bounds: [{self.min_lr:.6f}, {self.max_lr:.6f}])"
        )
        
        return lr_bounded
    
    async def _enqueue_update_task(
        self,
        input_data: Dict[str, Any],
        voting_result: Any,
        initial_prediction: Dict[str, Any]
    ):
        """
        Create and enqueue an update task for the update worker.
        Uses shared memory for efficient tensor transfer.
        
        Args:
            input_data: Original input data
            voting_result: Result from voting module
            initial_prediction: Initial model prediction
        """
        from ..data_structures import Trajectory, validate_trajectory
        
        # Extract or create trajectory from voting result
        trajectory = None
        if hasattr(voting_result, 'final_answer'):
            if isinstance(voting_result.final_answer, dict):
                trajectory_data = voting_result.final_answer.get('trajectory', [])
                if trajectory_data:
                    trajectory = validate_trajectory(trajectory_data)
            elif hasattr(voting_result.final_answer, 'trajectory'):
                trajectory = voting_result.final_answer.trajectory
        
        # Create empty trajectory if none exists
        if trajectory is None:
            trajectory = Trajectory()
        
        # Task 001 & 002: Calculate reward using the orchestrator with pseudo-labels
        # Using the consensus answer as a high-quality pseudo-label for R_final
        state_embeddings = None
        if hasattr(initial_prediction, 'embeddings'):
            state_embeddings = initial_prediction.embeddings
        
        reward_dict = self.reward_orchestrator.calculate_reward(
            trajectory=trajectory,
            final_answer=voting_result.final_answer,
            ground_truth=voting_result.final_answer,  # Using consensus as pseudo-label
            state_embeddings=state_embeddings
        )
        
        # Extract total reward tensor
        if isinstance(reward_dict, dict):
            total_reward = reward_dict.get('total_reward', 0.0)
            reward_tensor = torch.tensor(total_reward, dtype=torch.float32)
        else:
            reward_tensor = torch.tensor(0.0, dtype=torch.float32)
        
        # Calculate adaptive learning rate
        learning_rate = self._calculate_adaptive_lr(voting_result.confidence)
        
        # Task 003: Structure and Enqueue the Update Task
        from ..data_structures import UpdateTask, Experience
        
        # Handle image features with shared memory
        image_features = input_data.get('image_features')
        shm_info = None
        
        if image_features is not None and isinstance(image_features, torch.Tensor):
            # Transfer large tensors via shared memory
            shm_info = self.shm_manager.create_shared_tensor(image_features)
            
            # Store only metadata in the experience
            experience = Experience(
                experience_id="",  # Will be auto-generated
                image_features=None,  # Will be reconstructed from shared memory
                question_text=input_data.get('question', ""),
                trajectory=trajectory,  # Use the validated trajectory
                model_confidence=voting_result.confidence
            )
            
            # Add shared memory info to metadata
            experience.metadata = {'shm_info': shm_info}
        else:
            # Small data can go directly through the queue
            experience = Experience(
                experience_id="",
                image_features=image_features,
                question_text=input_data.get('question', ""),
                trajectory=trajectory,  # Use the validated trajectory
                model_confidence=voting_result.confidence
            )
        
        # Extract original logits for KL divergence calculation
        original_logits = None
        if isinstance(initial_prediction, dict):
            original_logits = initial_prediction.get('logits')
        elif hasattr(initial_prediction, 'logits'):
            original_logits = initial_prediction.logits
        
        # Create UpdateTask with all required fields including original_logits
        update_task = UpdateTask(
            task_id="",  # Will be auto-generated
            experience=experience,
            reward_tensor=reward_tensor,  # Multi-component reward tensor
            learning_rate=learning_rate,  # Adaptive learning rate
            original_logits=original_logits  # Essential for KL divergence calculation
        )
        
        # Add shared memory name to task metadata if using shared memory
        if shm_info:
            update_task.metadata['shm_name'] = shm_info.name
        
        # Enqueue task with thread-safe statistics
        try:
            self.update_queue.put(update_task, timeout=1.0)
            with self.stats_lock:
                self.stats['total_updates'] += 1
            logger.debug(f"Enqueued update task {update_task.task_id} with LR={learning_rate:.6f}")
        except Exception as e:
            logger.error(f"Failed to enqueue update task: {e}")
            with self.stats_lock:
                self.stats['failed_updates'] += 1
            # Clean up shared memory if enqueue failed
            if shm_info:
                self.shm_manager.mark_cleaned(shm_info.name)
    
    def start_update_worker(self):
        """
        Start the update worker process with proper synchronization.
        """
        if self.update_worker_process is not None and self.update_worker_process.is_alive():
            logger.warning("Update worker already running")
            return
        
        # Import here to avoid circular dependency
        from .update_worker import UpdateWorker
        
        # Create synchronization event
        worker_ready = mp.Event()
        
        # Create worker process
        self.update_worker_process = mp.Process(
            target=self._run_update_worker,
            args=(self.model, self.update_queue, self.cleanup_confirmation_queue, 
                  self.config, worker_ready),
            daemon=False
        )
        
        self.update_worker_process.start()
        logger.info(f"Started update worker process (PID: {self.update_worker_process.pid})")
        
        # Wait for worker to be ready (with timeout)
        if worker_ready.wait(timeout=10.0):
            logger.info("Update worker is ready")
        else:
            logger.warning("Update worker readiness timeout - continuing anyway")
        
        # Start watchdog
        self.start_watchdog()
    
    @staticmethod
    def _run_update_worker(model, update_queue, cleanup_queue, config, ready_event=None):
        """
        Static method to run the update worker in a separate process.
        
        Args:
            model: Model to update
            update_queue: Queue for receiving updates
            cleanup_queue: Queue for sending cleanup confirmations
            config: Configuration dictionary
            ready_event: Optional event to signal readiness
        """
        from .update_worker import UpdateWorker
        from ..modules.reward_shaping import RewardOrchestrator
        
        # Initialize reward orchestrator for the worker (optional)
        reward_orchestrator = RewardOrchestrator(config) if config.get('use_reward_orchestrator', False) else None
        
        worker = UpdateWorker(
            model=model,
            update_queue=update_queue,
            cleanup_confirmation_queue=cleanup_queue,
            config=config,
            reward_orchestrator=reward_orchestrator
        )
        
        # Signal that worker is ready
        if ready_event:
            ready_event.set()
        
        worker.run()
    
    def start_watchdog(self):
        """
        Start the watchdog thread for monitoring shared memory and worker health.
        """
        if self.watchdog_thread is not None and self.watchdog_thread.is_alive():
            logger.warning("Watchdog already running")
            return
        
        self.watchdog_running = True
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True
        )
        self.watchdog_thread.start()
        logger.info("Started watchdog thread")
        
        # Also start monitoring thread
        self.start_monitoring()
    
    def _watchdog_loop(self):
        """
        Main watchdog loop that runs in a separate thread.
        
        Task 006: Implements supervisor functionality to automatically
        restart the update worker process if it fails.
        """
        consecutive_restart_failures = 0
        max_restart_failures = 3
        restart_cooldown = 5.0  # Seconds to wait before restart
        
        logger.debug(f"[Watchdog] Starting watchdog loop with interval {self.watchdog_interval}s")
        
        while self.watchdog_running:
            try:
                # Process cleanup confirmations
                self._process_cleanup_confirmations()
                
                # Check worker health (Task 006: Action 1)
                # If worker was never started, consider it as "alive" for cleanup purposes
                # This allows testing timeout-based cleanup without a worker
                if self.update_worker_process is None:
                    worker_alive = True  # No worker started, don't clean based on worker status
                else:
                    worker_alive = self.update_worker_process.is_alive()
                
                # Task 006: Action 2 - Implement restart mechanism
                if not worker_alive and self.update_worker_process is not None:
                    # Worker has died unexpectedly
                    logger.critical(
                        f"[Supervisor] Update worker process has terminated unexpectedly "
                        f"(PID: {self.update_worker_process.pid}). Attempting to restart..."
                    )
                    
                    # Send alert
                    self.alerter.send_alert(
                        severity=AlertSeverity.CRITICAL,
                        component="supervisor",
                        message="Update worker process died - initiating restart",
                        details={
                            'old_pid': self.update_worker_process.pid,
                            'restart_attempt': consecutive_restart_failures + 1
                        }
                    )
                    
                    # Clean up stale segments from dead worker
                    cleaned = self.shm_manager.cleanup_stale_segments(worker_alive=False)
                    if cleaned:
                        logger.info(f"[Supervisor] Cleaned {len(cleaned)} segments from dead worker")
                    
                    # Wait for cleanup to complete
                    time.sleep(restart_cooldown)
                    
                    # Attempt to restart worker
                    try:
                        # Clear the old process reference
                        self.update_worker_process = None
                        
                        # Start a new worker process
                        self.start_update_worker()
                        
                        # Track successful restart
                        with self.stats_lock:
                            self.stats['worker_restarts'] = self.stats.get('worker_restarts', 0) + 1
                        
                        # Reset failure counter on successful restart
                        consecutive_restart_failures = 0
                        
                        logger.info(
                            f"[Supervisor] Successfully restarted update worker "
                            f"(new PID: {self.update_worker_process.pid})"
                        )
                        
                        # Send recovery alert
                        self.alerter.send_alert(
                            severity=AlertSeverity.INFO,
                            component="supervisor",
                            message="Update worker successfully restarted",
                            details={'new_pid': self.update_worker_process.pid}
                        )
                        
                        # Update monitoring metrics
                        self.health_monitor.record_update()
                        
                    except Exception as e:
                        consecutive_restart_failures += 1
                        logger.error(
                            f"[Supervisor] Failed to restart worker (attempt {consecutive_restart_failures}): {e}"
                        )
                        
                        # Check if we've exceeded max restart attempts
                        if consecutive_restart_failures >= max_restart_failures:
                            logger.critical(
                                f"[Supervisor] Max restart attempts ({max_restart_failures}) exceeded. "
                                "Worker supervision suspended."
                            )
                            
                            # Send emergency alert
                            self.alerter.send_alert(
                                severity=AlertSeverity.EMERGENCY,
                                component="supervisor",
                                message="Worker restart failed - manual intervention required",
                                details={
                                    'consecutive_failures': consecutive_restart_failures,
                                    'error': str(e)
                                }
                            )
                            
                            # Suspend further restart attempts
                            self.update_worker_process = None
                            break
                        
                        # Exponential backoff for next retry
                        time.sleep(restart_cooldown * (2 ** consecutive_restart_failures))
                
                elif worker_alive:
                    # Worker is healthy, reset failure counter
                    if consecutive_restart_failures > 0:
                        logger.info("[Supervisor] Worker is healthy, resetting failure counter")
                        consecutive_restart_failures = 0
                    
                    # Clean up any stale segments
                    logger.debug(f"[Watchdog] Checking for stale segments. Pending: {len(self.shm_manager.pending_shm)}")
                    cleaned = self.shm_manager.cleanup_stale_segments(worker_alive)
                    if cleaned:
                        with self.stats_lock:
                            self.stats['watchdog_cleanups'] += len(cleaned)
                        logger.info(f"[Watchdog] Cleaned {len(cleaned)} stale segments: {cleaned}")
                    else:
                        logger.debug("[Watchdog] No stale segments found")
                
                # Log status periodically
                if self.stats['total_requests'] % 100 == 0 and self.stats['total_requests'] > 0:
                    self._log_status()
                
                # Task 006: Action 3 - Track restart metrics for chaos test validation
                with self.stats_lock:
                    # Update worker health status
                    self.stats['worker_alive'] = worker_alive
                    self.stats['consecutive_restart_failures'] = consecutive_restart_failures
                
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}", exc_info=True)
            
            # Sleep before next iteration
            time.sleep(self.watchdog_interval)
    
    def _process_cleanup_confirmations(self):
        """
        Process cleanup confirmations from the update worker.
        """
        import queue
        while True:
            try:
                # Non-blocking get from cleanup queue
                shm_name = self.cleanup_confirmation_queue.get_nowait()
                self.shm_manager.mark_cleaned(shm_name)
                logger.debug(f"Received cleanup confirmation for {shm_name}")
            except queue.Empty:
                # Queue is empty - expected condition
                break
            except Exception as e:
                logger.error(f"Unexpected error processing cleanup confirmation: {e}")
                break
    
    def _log_status(self):
        """
        Log current status of the inference engine.
        """
        shm_status = self.shm_manager.get_status()
        logger.info(
            f"InferenceEngine Status - "
            f"Requests: {self.stats['total_requests']}, "
            f"Updates: {self.stats['total_updates']}, "
            f"Failed: {self.stats['failed_updates']}, "
            f"Watchdog cleanups: {self.stats['watchdog_cleanups']}, "
            f"Pending SHM: {shm_status['pending_segments']}, "
            f"SHM bytes: {shm_status['total_bytes']}"
        )
    
    def run(self):
        """
        Main run loop for the inference engine.
        Processes requests from the request queue.
        """
        logger.info("Inference Engine main loop started")
        
        # Start the update worker
        self.start_update_worker()
        
        while True:
            try:
                # Get request from queue (blocking with timeout)
                request = self.request_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    break
                
                with self.stats_lock:
                    self.stats['total_requests'] += 1
                
                # Process the request
                asyncio.run(self._process_request(request))
                
            except Exception as e:
                if "Empty" not in str(e):  # Ignore empty queue timeouts
                    logger.error(f"Error processing request: {e}")
        
        logger.info("Inference Engine main loop stopped")
    
    async def _process_request(self, request: Dict[str, Any]):
        """
        Process a single inference request.
        
        Args:
            request: Request dictionary containing input data and request ID
        """
        request_id = request.get('request_id', 'unknown')
        input_data = request.get('input_data', {})
        
        try:
            # Run inference and adaptation
            result, confidence, metadata = await self.infer_and_adapt(input_data)
            
            # Create response
            response = {
                'request_id': request_id,
                'result': result,
                'confidence': confidence,
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in request {request_id}: {e}")
            response = {
                'request_id': request_id,
                'error': str(e),
                'success': False
            }
        
        # Send response
        self.response_queue.put(response)
    
    def shutdown(self):
        """
        Gracefully shutdown the inference engine.
        """
        logger.info("Shutting down Inference Engine")
        
        # Stop watchdog
        self.watchdog_running = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5.0)
        
        # Send shutdown signal to update worker
        if self.update_queue:
            self.update_queue.put(None)
        
        # Wait for update worker to finish
        if self.update_worker_process:
            self.update_worker_process.join(timeout=10.0)
            if self.update_worker_process.is_alive():
                logger.warning("Update worker didn't stop gracefully, terminating")
                self.update_worker_process.terminate()
                self.update_worker_process.join(timeout=5.0)
        
        # Final cleanup of any remaining shared memory
        cleaned = self.shm_manager.cleanup_stale_segments(worker_alive=False)
        if cleaned:
            logger.info(f"Final cleanup: {len(cleaned)} segments")
        
        # Log final statistics
        self._log_status()
        
        logger.info("Inference Engine shutdown complete")
    
    def _should_request_human_review(self) -> bool:
        """
        Determine if the current update should be sent for human review.
        
        Implements sampling strategy for HIL mode:
        - Only active when HIL mode is enabled
        - Samples a percentage of updates for review
        
        Returns:
            Boolean indicating whether to request human review
        """
        if not self.hil_mode_enabled:
            return False
        
        # Increment counter
        self.hil_review_counter += 1
        
        # Sample based on percentage
        import random
        should_review = random.random() < self.hil_review_percentage
        
        if should_review:
            logger.info(
                f"HIL mode: Sampling update #{self.hil_review_counter} for human review "
                f"(rate: {self.hil_review_percentage:.1%})"
            )
        
        return should_review
    
    async def _enqueue_human_review_task(
        self,
        input_data: Dict[str, Any],
        voting_result: Any,
        initial_prediction: Dict[str, Any]
    ):
        """
        Create and enqueue a task for human review.
        
        Similar to _enqueue_update_task but routes to human_review_queue.
        
        Args:
            input_data: Original input data
            voting_result: Result from voting module
            initial_prediction: Initial model prediction
        """
        from ..data_structures import Trajectory, validate_trajectory, UpdateTask, Experience
        
        # Extract or create trajectory from voting result
        trajectory = None
        if hasattr(voting_result, 'final_answer'):
            if isinstance(voting_result.final_answer, dict):
                trajectory_data = voting_result.final_answer.get('trajectory', [])
                if trajectory_data:
                    trajectory = validate_trajectory(trajectory_data)
            elif hasattr(voting_result.final_answer, 'trajectory'):
                trajectory = voting_result.final_answer.trajectory
        
        # Create empty trajectory if none exists
        if trajectory is None:
            trajectory = Trajectory()
        
        # Calculate reward using the orchestrator with pseudo-labels
        state_embeddings = None
        if hasattr(initial_prediction, 'embeddings'):
            state_embeddings = initial_prediction.embeddings
        
        reward_dict = self.reward_orchestrator.calculate_reward(
            trajectory=trajectory,
            final_answer=voting_result.final_answer,
            ground_truth=voting_result.final_answer,  # Using consensus as pseudo-label
            state_embeddings=state_embeddings
        )
        
        # Extract total reward tensor
        if isinstance(reward_dict, dict):
            total_reward = reward_dict.get('total_reward', 0.0)
            reward_tensor = torch.tensor(total_reward, dtype=torch.float32)
        else:
            reward_tensor = torch.tensor(0.0, dtype=torch.float32)
        
        # Calculate adaptive learning rate
        learning_rate = self._calculate_adaptive_lr(voting_result.confidence)
        
        # Create review task
        experience = Experience(
            experience_id="",  # Will be auto-generated
            image_features=input_data.get('image_features'),
            question_text=input_data.get('question', ""),
            trajectory=trajectory,  # Use the validated trajectory
            model_confidence=voting_result.confidence
        )
        
        # Extract original logits for KL divergence calculation
        original_logits = None
        if isinstance(initial_prediction, dict):
            original_logits = initial_prediction.get('logits')
        elif hasattr(initial_prediction, 'logits'):
            original_logits = initial_prediction.logits
        
        review_task = UpdateTask(
            task_id="",  # Will be auto-generated
            experience=experience,
            reward_tensor=reward_tensor,  # Multi-component reward tensor
            learning_rate=learning_rate,  # Adaptive learning rate
            original_logits=original_logits,  # Essential for KL divergence calculation
            metadata={
                'review_type': 'human',
                'voting_provenance': voting_result.provenance,
                'initial_prediction': initial_prediction,
                'reward_components': reward_dict  # Include full reward breakdown
            }
        )
        
        # Enqueue for human review
        try:
            self.human_review_queue.put(review_task, timeout=1.0)
            with self.stats_lock:
                self.stats['human_review_requests'] = self.stats.get('human_review_requests', 0) + 1
            logger.info(
                f"Enqueued task {review_task.task_id} for human review "
                f"(confidence: {voting_result.confidence:.3f}, LR: {learning_rate:.6f})"
            )
        except Exception as e:
            logger.error(f"Failed to enqueue human review task: {e}")
            with self.stats_lock:
                self.stats['failed_human_reviews'] = self.stats.get('failed_human_reviews', 0) + 1
    
    def process_human_review_decision(
        self,
        task_id: str,
        approved: bool,
        reviewer_notes: Optional[str] = None
    ):
        """
        Process a human reviewer's decision on a task.
        
        Args:
            task_id: ID of the reviewed task
            approved: Whether the task was approved
            reviewer_notes: Optional notes from the reviewer
        """
        # This method would be called by the human review interface
        # to process decisions
        
        if approved:
            logger.info(f"Human review APPROVED task {task_id}: {reviewer_notes or 'No notes'}")
            # Move to regular update queue
            # (Implementation would retrieve task from storage and enqueue)
            with self.stats_lock:
                self.stats['human_approvals'] = self.stats.get('human_approvals', 0) + 1
        else:
            logger.info(f"Human review REJECTED task {task_id}: {reviewer_notes or 'No notes'}")
            # Discard the task
            with self.stats_lock:
                self.stats['human_rejections'] = self.stats.get('human_rejections', 0) + 1
    
    async def _add_bootstrap_experience(
        self,
        input_data: Dict[str, Any],
        initial_prediction: Dict[str, Any]
    ):
        """
        Add experience to buffer during cold start with high priority.
        
        During cold start, we rapidly build memory by adding all experiences
        with high initial priority to bootstrap the system.
        
        Args:
            input_data: Original input data
            initial_prediction: Model's initial prediction
        """
        from ..data_structures import Experience, Trajectory
        
        try:
            # Task 002 (Phase 2 Round 6): Anonymize text data before storage
            anonymized_question = input_data.get('question', "")
            if anonymized_question:
                anonymized_question, _ = self.data_anonymizer.pii_redactor.redact_text(anonymized_question)
            
            # Create trajectory from prediction if available
            trajectory = Trajectory()
            if isinstance(initial_prediction, dict) and 'trajectory' in initial_prediction:
                trajectory = initial_prediction['trajectory']
            
            # Create experience with anonymized data
            experience = Experience(
                experience_id="",  # Will be auto-generated
                image_features=input_data.get('image_features'),  # Already processed, no PII
                question_text=anonymized_question,  # Redacted text
                trajectory=trajectory,
                model_confidence=initial_prediction.get('confidence', 0.5) if isinstance(initial_prediction, dict) else 0.5,
                metadata={
                    'cold_start': True,
                    'timestamp': datetime.now().isoformat(),
                    '_anonymized': True
                }
            )
            
            # Add to buffer with high initial priority (for rapid memory building)
            self.experience_buffer.add(
                experience,
                initial_priority=0.9  # High priority for bootstrap experiences
            )
            
            logger.debug("Added anonymized bootstrap experience to buffer")
            
        except Exception as e:
            logger.error(f"Failed to add bootstrap experience: {e}")
    
    async def _add_to_experience_buffer(
        self,
        input_data: Dict[str, Any],
        voting_result: Any,
        initial_prediction: Dict[str, Any]
    ):
        """
        Add experience to buffer for future retrieval.
        
        Args:
            input_data: Original input data
            voting_result: Result from voting module
            initial_prediction: Initial model prediction
        """
        from ..data_structures import Experience, Trajectory
        
        try:
            # Task 002 (Phase 2 Round 6): Anonymize all text data before storage
            anonymized_question = input_data.get('question', "")
            if anonymized_question:
                anonymized_question, redaction_stats = self.data_anonymizer.pii_redactor.redact_text(anonymized_question)
                
                # Log if PII was found and redacted
                if redaction_stats:
                    logger.info(f"Redacted PII from question before storage: {redaction_stats}")
            
            # Extract trajectory
            trajectory = Trajectory()
            if hasattr(voting_result, 'final_answer'):
                if isinstance(voting_result.final_answer, dict) and 'trajectory' in voting_result.final_answer:
                    trajectory = voting_result.final_answer['trajectory']
            
            # Strip any image metadata if present
            image_features = input_data.get('image_features')
            # Note: Image features are typically tensors, not raw images, so metadata stripping may not apply
            # But we ensure no raw images with EXIF data are stored
            
            # Create experience with anonymized data
            experience = Experience(
                experience_id="",  # Will be auto-generated
                image_features=image_features,  # Tensor data, no PII
                question_text=anonymized_question,  # Redacted text
                trajectory=trajectory,
                model_confidence=voting_result.confidence,
                metadata={
                    'voting_confidence': voting_result.confidence,
                    'timestamp': datetime.now().isoformat(),
                    '_anonymized': True,
                    '_pii_redacted': bool(redaction_stats) if 'redaction_stats' in locals() else False
                }
            )
            
            # Calculate priority based on confidence and uncertainty
            # Higher uncertainty = higher priority for learning
            uncertainty = 1.0 - voting_result.confidence
            priority = uncertainty * 0.5 + 0.25  # Range: 0.25 to 0.75
            
            # Add to buffer
            self.experience_buffer.add(experience, initial_priority=priority)
            
            logger.debug(f"Added anonymized experience to buffer with priority {priority:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to add experience to buffer: {e}")
    
    def _log_inference_metrics(self, metrics: Dict[str, Any]):
        """
        Log comprehensive inference metrics for monitoring.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        try:
            # Update internal statistics
            with self.stats_lock:
                # Update running averages
                if 'confidence' in metrics:
                    current_avg = self.stats.get('avg_confidence', 0.5)
                    self.stats['avg_confidence'] = 0.95 * current_avg + 0.05 * metrics['confidence']
                
                if 'inference_time' in metrics:
                    current_avg = self.stats.get('avg_inference_time', 0.0)
                    self.stats['avg_inference_time'] = 0.95 * current_avg + 0.05 * metrics['inference_time']
                
                # Track mode distribution
                mode = metrics.get('mode', 'unknown')
                mode_key = f'mode_{mode}_count'
                self.stats[mode_key] = self.stats.get(mode_key, 0) + 1
            
            # Log to external monitoring system (e.g., wandb)
            if self.config.get('enable_wandb_logging', False):
                try:
                    import wandb
                    wandb.log(metrics)
                except ImportError:
                    pass
            
            # Log critical metrics at INFO level for visibility
            if metrics.get('mode') == 'cold_start':
                logger.info(f"Cold start inference: buffer_size={metrics.get('buffer_size', 0)}")
            
            # Check for anomalies
            if metrics.get('inference_time', 0) > 1.0:  # More than 1 second
                logger.warning(f"Slow inference detected: {metrics.get('inference_time', 0):.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to log inference metrics: {e}")
    
    def _get_queue_sizes(self) -> Dict[str, int]:
        """
        Get current sizes of all queues for monitoring.
        
        Returns:
            Dictionary mapping queue names to their current sizes
        """
        sizes = {}
        
        try:
            # Check each queue's approximate size
            # Note: qsize() may raise NotImplementedError on some platforms
            if hasattr(self.request_queue, 'qsize'):
                try:
                    sizes['request_queue'] = self.request_queue.qsize()
                except NotImplementedError:
                    sizes['request_queue'] = -1
            
            if hasattr(self.response_queue, 'qsize'):
                try:
                    sizes['response_queue'] = self.response_queue.qsize()
                except NotImplementedError:
                    sizes['response_queue'] = -1
            
            if hasattr(self.update_queue, 'qsize'):
                try:
                    sizes['update_queue'] = self.update_queue.qsize()
                except NotImplementedError:
                    sizes['update_queue'] = -1
            
            if hasattr(self.human_review_queue, 'qsize'):
                try:
                    sizes['human_review_queue'] = self.human_review_queue.qsize()
                except NotImplementedError:
                    sizes['human_review_queue'] = -1
            
        except Exception as e:
            logger.debug(f"Could not get queue sizes: {e}")
        
        return sizes
    
    def start_monitoring(self):
        """
        Start the health monitoring thread.
        """
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.monitoring_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started monitoring thread")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop that tracks health indicators and sends alerts.
        """
        import psutil
        
        while self.monitoring_running:
            try:
                # Collect health metrics
                metrics = {}
                
                # Calculate update rate (updates per minute)
                with self.stats_lock:
                    # Track update rate - simplified for now
                    # In production, would track time-based rate
                    metrics['update_rate'] = self.stats.get('update_rate', 0.0)
                    
                    # Calculate FAISS failure rate
                    total_faiss = self.stats.get('faiss_attempts', 0)
                    if total_faiss > 0:
                        metrics['faiss_failure_rate'] = self.stats.get('faiss_failures', 0) / total_faiss
                    else:
                        metrics['faiss_failure_rate'] = 0.0
                
                # Get queue sizes
                queue_sizes = self._get_queue_sizes()
                metrics['queue_sizes'] = queue_sizes
                
                # Check for growing queues
                update_queue_size = queue_sizes.get('update_queue', 0)
                if update_queue_size > 0:
                    metrics['queue_size'] = update_queue_size
                
                # Get memory usage
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_percent = process.memory_percent()
                    metrics['memory_usage_ratio'] = memory_percent / 100.0
                    metrics['memory_rss_mb'] = memory_info.rss / (1024 * 1024)
                except Exception as e:
                    logger.debug(f"Could not get memory info: {e}")
                
                # Get mean KL divergence from worker if available
                if hasattr(self, 'update_worker_process') and self.update_worker_process:
                    # This would be retrieved from shared state or worker stats
                    # For now, use a placeholder
                    metrics['mean_kl_divergence'] = self.stats.get('mean_kl_divergence', 0.0)
                
                # Calculate inference latency percentiles
                if hasattr(self, 'inference_times'):
                    if len(self.inference_times) > 0:
                        metrics['inference_latency_p99'] = np.percentile(self.inference_times, 99)
                        metrics['inference_latency_p95'] = np.percentile(self.inference_times, 95)
                        metrics['inference_latency_p50'] = np.percentile(self.inference_times, 50)
                
                # Update health monitor
                self.health_monitor.update_metrics(metrics, component="inference_engine")
                
                # Log to wandb if enabled
                if self.config.get('enable_wandb_logging', False):
                    try:
                        import wandb
                        wandb.log({
                            'health/update_rate': metrics.get('update_rate', 0),
                            'health/faiss_failure_rate': metrics.get('faiss_failure_rate', 0),
                            'health/mean_kl': metrics.get('mean_kl_divergence', 0),
                            'health/memory_usage_ratio': metrics.get('memory_usage_ratio', 0),
                            'health/update_queue_size': update_queue_size
                        })
                    except ImportError:
                        pass
                
                # Check for critical conditions
                self._check_critical_conditions(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            
            # Sleep before next iteration
            time.sleep(self.monitoring_interval)
    
    def _check_critical_conditions(self, metrics: Dict[str, Any]):
        """
        Check for critical system conditions that require immediate alerts.
        
        Args:
            metrics: Current health metrics
        """
        # Check if update worker is dead
        if self.update_worker_process and not self.update_worker_process.is_alive():
            self.alerter.send_alert(
                severity=AlertSeverity.CRITICAL,
                component="inference_engine",
                message="Update worker process is dead",
                details={'pid': self.update_worker_process.pid if self.update_worker_process else None}
            )
        
        # Check for sustained high queue size
        update_queue_size = metrics.get('queue_sizes', {}).get('update_queue', 0)
        if update_queue_size > self.config.get('max_queue_size', 1000) * 0.95:
            self.alerter.send_alert(
                severity=AlertSeverity.EMERGENCY,
                component="inference_engine",
                message="Update queue near capacity - system may deadlock",
                details={'queue_size': update_queue_size, 'max_size': self.config.get('max_queue_size', 1000)}
            )
        
        # Check for memory pressure
        if metrics.get('memory_usage_ratio', 0) > 0.95:
            self.alerter.send_alert(
                severity=AlertSeverity.EMERGENCY,
                component="inference_engine",
                message="Critical memory pressure detected",
                details={'memory_usage': f"{metrics.get('memory_usage_ratio', 0) * 100:.1f}%"}
            )
    
    def record_model_update(self):
        """Record that a model update was performed."""
        self.health_monitor.record_update()
        with self.stats_lock:
            # Simple rate tracking - in production would use time windows
            self.stats['update_rate'] = self.stats.get('update_rate', 0) + 0.1