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
            'failed_updates': 0
        }
        
        logger.info("Inference Engine initialized")
    
    async def infer_and_adapt(
        self,
        input_data: Dict[str, Any]
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Main inference and adaptation loop.
        
        Performs inference, retrieves similar experiences, applies voting,
        and optionally triggers a learning update based on confidence.
        
        Args:
            input_data: Input data containing image features and question
            
        Returns:
            Tuple of (prediction, confidence_score, metadata)
        """
        # Step 1: Get initial model prediction
        with torch.no_grad():
            initial_prediction = await self._get_model_prediction(input_data)
        
        # Step 2: Retrieve k-NN neighbors from experience buffer
        neighbors = self.experience_buffer.search_index(
            input_data,
            k=self.config.get('k_neighbors', 5)
        )
        
        # Step 3: Apply temporal ensemble voting
        voting_result = self.voting_module.vote(
            initial_prediction,
            neighbors,
            strategy=self.config.get('voting_strategy', 'weighted')
        )
        
        # Step 4: Check confidence and potentially trigger update
        if self._should_trigger_update(voting_result.confidence):
            # Determine if this update should go through human review
            if self._should_request_human_review():
                await self._enqueue_human_review_task(
                    input_data,
                    voting_result,
                    initial_prediction
                )
            else:
                await self._enqueue_update_task(
                    input_data,
                    voting_result,
                    initial_prediction
                )
        
        return (
            voting_result.final_answer,
            voting_result.confidence,
            voting_result.provenance
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
    
    def _watchdog_loop(self):
        """
        Main watchdog loop that runs in a separate thread.
        """
        while self.watchdog_running:
            try:
                # Process cleanup confirmations
                self._process_cleanup_confirmations()
                
                # Check worker health
                worker_alive = (
                    self.update_worker_process is not None and 
                    self.update_worker_process.is_alive()
                )
                
                # Clean up stale segments
                cleaned = self.shm_manager.cleanup_stale_segments(worker_alive)
                if cleaned:
                    with self.stats_lock:
                        self.stats['watchdog_cleanups'] += len(cleaned)
                    logger.info(f"Watchdog cleaned {len(cleaned)} stale segments")
                
                # Log status periodically
                if self.stats['total_requests'] % 100 == 0 and self.stats['total_requests'] > 0:
                    self._log_status()
                
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")
            
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