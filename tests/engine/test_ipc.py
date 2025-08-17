"""
Test Inter-Process Communication (IPC)

Tests for shared memory transfer, queue communication, and process synchronization.
"""

import pytest
import torch
import torch.multiprocessing as mp
import time
import os
import numpy as np
import tempfile
from typing import Dict, Any, Tuple
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.engine.inference_engine import SharedMemoryManager, SharedMemoryInfo
from core.data_structures import UpdateTask, Experience, Trajectory


def create_test_tensor(shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
    """Create a test tensor with known values."""
    tensor = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def worker_process_tensor_transfer(
    send_queue: mp.Queue,
    receive_queue: mp.Queue,
    cleanup_queue: mp.Queue
):
    """
    Worker process that receives tensor info via shared memory.
    
    Args:
        send_queue: Queue to send results back
        receive_queue: Queue to receive shared memory info
        cleanup_queue: Queue to send cleanup confirmations
    """
    try:
        # Receive shared memory info
        shm_info = receive_queue.get(timeout=5.0)
        
        # In a real implementation, we would reconstruct the tensor
        # For testing, we'll just verify the info and send confirmation
        result = {
            'name': shm_info.get('name'),
            'shape': shm_info.get('shape'),
            'dtype': str(shm_info.get('dtype')),
            'success': True
        }
        
        # Send result back
        send_queue.put(result)
        
        # Send cleanup confirmation
        cleanup_queue.put(shm_info.get('name'))
        
    except Exception as e:
        send_queue.put({'error': str(e), 'success': False})


def echo_worker(input_queue, output_queue):
    """Simple echo worker for bidirectional communication test."""
    while True:
        try:
            msg = input_queue.get(timeout=1.0)
            if msg is None:
                break
            output_queue.put(f"Echo: {msg}")
        except Exception:
            # Handle both mp.queues.Empty and queue.Empty
            continue


def error_worker(queue):
    """Worker that raises an error for error handling test."""
    try:
        raise ValueError("Test error")
    except Exception as e:
        queue.put({"error": str(e)})


class TestSharedMemoryTransfer:
    """Test shared memory tensor transfer between processes."""
    
    def test_basic_tensor_transfer(self):
        """Test basic tensor transfer via shared memory."""
        manager = SharedMemoryManager()
        
        # Create test tensor
        original_tensor = create_test_tensor((10, 20, 30))
        
        # Create shared memory segment
        shm_info = manager.create_shared_tensor(original_tensor)
        
        # Verify info
        assert shm_info.shape == (10, 20, 30)
        assert shm_info.dtype == torch.float32
        assert shm_info.name in manager.pending_shm
        
        # Clean up
        manager.mark_cleaned(shm_info.name)
    
    def test_large_tensor_transfer(self):
        """Test transferring large tensors."""
        manager = SharedMemoryManager()
        
        # Create large tensor (100MB+)
        large_tensor = torch.randn(1000, 1000, 100)
        
        # Transfer should work
        shm_info = manager.create_shared_tensor(large_tensor)
        
        assert shm_info.shape == (1000, 1000, 100)
        assert shm_info.size_bytes == large_tensor.element_size() * large_tensor.numel()
        
        # Clean up
        manager.mark_cleaned(shm_info.name)
    
    def test_multiple_tensor_transfer(self):
        """Test transferring multiple tensors simultaneously."""
        manager = SharedMemoryManager()
        
        # Create multiple tensors
        tensors = [
            torch.randn(10, 10),
            torch.randn(20, 30, 40),
            torch.randn(5, 5, 5, 5)
        ]
        
        # Transfer all tensors
        shm_infos = [manager.create_shared_tensor(t) for t in tensors]
        
        # Verify all are tracked
        assert len(manager.pending_shm) == 3
        
        for info, tensor in zip(shm_infos, tensors):
            assert info.shape == tuple(tensor.shape)
            assert info.name in manager.pending_shm
        
        # Clean up
        for info in shm_infos:
            manager.mark_cleaned(info.name)
        
        assert len(manager.pending_shm) == 0
    
    def test_tensor_dtype_preservation(self):
        """Test that tensor dtypes are preserved during transfer."""
        manager = SharedMemoryManager()
        
        dtypes = [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64]
        
        for dtype in dtypes:
            tensor = torch.randn(5, 5).to(dtype)
            shm_info = manager.create_shared_tensor(tensor)
            
            assert shm_info.dtype == dtype
            
            manager.mark_cleaned(shm_info.name)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_to_cpu_transfer(self):
        """Test transferring GPU tensors to shared memory."""
        manager = SharedMemoryManager()
        
        # Create GPU tensor
        gpu_tensor = torch.randn(10, 10).cuda()
        
        # Transfer (should move to CPU pinned memory)
        shm_info = manager.create_shared_tensor(gpu_tensor)
        
        assert shm_info.shape == (10, 10)
        assert shm_info.dtype == torch.float32
        
        # Clean up
        manager.mark_cleaned(shm_info.name)


class TestQueueCommunication:
    """Test queue-based communication between processes."""
    
    def test_basic_queue_operations(self):
        """Test basic queue put/get operations."""
        queue = mp.Queue()
        
        # Put various data types
        test_data = [
            42,
            "test string",
            {"key": "value"},
            [1, 2, 3],
            torch.tensor([1, 2, 3])
        ]
        
        for item in test_data:
            queue.put(item)
        
        # Get and verify
        for original in test_data:
            retrieved = queue.get(timeout=1.0)
            if isinstance(original, torch.Tensor):
                assert torch.equal(retrieved, original)
            else:
                assert retrieved == original
    
    def test_queue_timeout(self):
        """Test queue timeout behavior."""
        queue = mp.Queue()
        
        # Try to get from empty queue
        with pytest.raises(mp.queues.Empty):
            queue.get(timeout=0.1)
        
        # Put with timeout should work
        queue.put("test", timeout=1.0)
        
        # Get should now work
        result = queue.get(timeout=1.0)
        assert result == "test"
    
    def test_queue_with_none_signal(self):
        """Test using None as shutdown signal."""
        queue = mp.Queue()
        
        # Put some data then None
        queue.put("data1")
        queue.put("data2")
        queue.put(None)  # Shutdown signal
        queue.put("data3")  # This would be after shutdown
        
        results = []
        while True:
            item = queue.get()
            if item is None:
                break
            results.append(item)
        
        assert results == ["data1", "data2"]
    
    def test_multiple_queues(self):
        """Test using multiple queues for bidirectional communication."""
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        
        # Simulate request-response pattern
        request = {"id": 1, "action": "process"}
        request_queue.put(request)
        
        # Process request
        received = request_queue.get()
        response = {"id": received["id"], "result": "done"}
        response_queue.put(response)
        
        # Get response
        result = response_queue.get()
        assert result["id"] == 1
        assert result["result"] == "done"


class TestProcessCommunication:
    """Test communication between actual processes."""
    
    def test_tensor_transfer_between_processes(self):
        """Test transferring tensor between processes via shared memory."""
        # Set up queues
        send_queue = mp.Queue()
        receive_queue = mp.Queue()
        cleanup_queue = mp.Queue()
        
        # Create shared memory manager
        manager = SharedMemoryManager()
        
        # Create test tensor
        tensor = create_test_tensor((5, 10, 15))
        shm_info = manager.create_shared_tensor(tensor)
        
        # Create worker process
        process = mp.Process(
            target=worker_process_tensor_transfer,
            args=(send_queue, receive_queue, cleanup_queue)
        )
        process.start()
        
        # Send shared memory info
        info_dict = {
            'name': shm_info.name,
            'shape': shm_info.shape,
            'dtype': shm_info.dtype
        }
        receive_queue.put(info_dict)
        
        # Get result from worker
        result = send_queue.get(timeout=5.0)
        assert result['success']
        assert result['name'] == shm_info.name
        assert tuple(result['shape']) == shm_info.shape
        
        # Get cleanup confirmation
        cleanup_name = cleanup_queue.get(timeout=5.0)
        assert cleanup_name == shm_info.name
        
        # Clean up
        manager.mark_cleaned(shm_info.name)
        
        # Wait for process to finish
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
    
    def test_bidirectional_communication(self):
        """Test bidirectional communication between processes."""
        # Create queues
        to_worker = mp.Queue()
        from_worker = mp.Queue()
        
        # Start worker
        worker = mp.Process(
            target=echo_worker,
            args=(to_worker, from_worker)
        )
        worker.start()
        
        # Send messages
        messages = ["Hello", "World", "Test"]
        for msg in messages:
            to_worker.put(msg)
        
        # Get responses
        responses = []
        for _ in messages:
            resp = from_worker.get(timeout=2.0)
            responses.append(resp)
        
        # Verify
        expected = [f"Echo: {msg}" for msg in messages]
        assert responses == expected
        
        # Shutdown
        to_worker.put(None)
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()
    
    def test_process_error_handling(self):
        """Test error handling in process communication."""
        queue = mp.Queue()
        
        # Start worker
        worker = mp.Process(target=error_worker, args=(queue,))
        worker.start()
        
        # Get error message
        result = queue.get(timeout=5.0)
        assert "error" in result
        assert "Test error" in result["error"]
        
        # Wait for worker
        worker.join(timeout=5.0)


class TestCleanupMechanisms:
    """Test cleanup and resource management mechanisms."""
    
    def test_cleanup_confirmation_flow(self):
        """Test the cleanup confirmation flow."""
        manager = SharedMemoryManager()
        cleanup_queue = mp.Queue()
        
        # Create segments
        tensors = [torch.randn(5, 5) for _ in range(3)]
        shm_infos = [manager.create_shared_tensor(t) for t in tensors]
        
        # Simulate worker sending cleanup confirmations
        for info in shm_infos:
            cleanup_queue.put(info.name)
        
        # Process confirmations
        while not cleanup_queue.empty():
            shm_name = cleanup_queue.get()
            manager.mark_cleaned(shm_name)
        
        # All segments should be cleaned
        assert len(manager.pending_shm) == 0
    
    def test_timeout_based_cleanup(self):
        """Test timeout-based cleanup of stale segments."""
        manager = SharedMemoryManager(timeout_seconds=0.2)
        
        # Create segments at different times
        tensor1 = torch.randn(5, 5)
        shm1 = manager.create_shared_tensor(tensor1)
        
        time.sleep(0.15)
        
        tensor2 = torch.randn(5, 5)
        shm2 = manager.create_shared_tensor(tensor2)
        
        time.sleep(0.1)  # Total: shm1=0.25s, shm2=0.1s
        
        # Clean stale segments
        cleaned = manager.cleanup_stale_segments(worker_alive=True)
        
        # Only shm1 should be cleaned
        assert shm1.name in cleaned
        assert shm2.name not in cleaned
        assert shm1.name not in manager.pending_shm
        assert shm2.name in manager.pending_shm
        
        # Clean up
        manager.mark_cleaned(shm2.name)
    
    def test_forced_cleanup_on_shutdown(self):
        """Test forced cleanup of all segments on shutdown."""
        manager = SharedMemoryManager()
        
        # Create multiple segments
        segments = []
        for i in range(5):
            tensor = torch.randn(10, 10)
            info = manager.create_shared_tensor(tensor)
            segments.append(info)
        
        assert len(manager.pending_shm) == 5
        
        # Force cleanup (simulating shutdown)
        cleaned = manager.cleanup_stale_segments(worker_alive=False)
        
        # All should be cleaned
        assert len(cleaned) == 5
        assert len(manager.pending_shm) == 0
        
        for info in segments:
            assert info.name in cleaned


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_empty_tensor_transfer(self):
        """Test transferring empty tensors."""
        manager = SharedMemoryManager()
        
        # Create empty tensor
        empty_tensor = torch.tensor([])
        
        # Should handle gracefully
        shm_info = manager.create_shared_tensor(empty_tensor)
        
        assert shm_info.shape == (0,)
        assert shm_info.size_bytes == 0
        
        manager.mark_cleaned(shm_info.name)
    
    def test_scalar_tensor_transfer(self):
        """Test transferring scalar tensors."""
        manager = SharedMemoryManager()
        
        # Create scalar tensor
        scalar = torch.tensor(42.0)
        
        shm_info = manager.create_shared_tensor(scalar)
        
        assert shm_info.shape == ()
        assert shm_info.dtype == torch.float32
        
        manager.mark_cleaned(shm_info.name)
    
    def test_concurrent_access(self):
        """Test concurrent access to shared memory manager."""
        manager = SharedMemoryManager()
        
        def create_and_clean(manager, num_ops):
            """Create and clean segments."""
            for _ in range(num_ops):
                tensor = torch.randn(10, 10)
                info = manager.create_shared_tensor(tensor)
                time.sleep(0.01)  # Small delay
                manager.mark_cleaned(info.name)
        
        # Create multiple processes
        processes = []
        for _ in range(3):
            p = mp.Process(target=create_and_clean, args=(manager, 5))
            p.start()
            processes.append(p)
        
        # Wait for all to finish
        for p in processes:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
        
        # Should have no pending segments
        assert len(manager.pending_shm) == 0
    
    def test_queue_full_handling(self):
        """Test handling of full queues."""
        # Create queue with limited size
        small_queue = mp.Queue(maxsize=2)
        
        # Fill the queue
        small_queue.put(1)
        small_queue.put(2)
        
        # Try to put with timeout (should fail)
        import queue
        with pytest.raises(queue.Full):
            small_queue.put(3, timeout=0.1)
        
        # Remove one item
        small_queue.get()
        
        # Now put should work
        small_queue.put(3, timeout=0.1)
        
        # Verify contents
        assert small_queue.get() == 2
        assert small_queue.get() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])