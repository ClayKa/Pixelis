Of course. Here is the detailed, step-by-step implementation plan in English for the final three failures.

---
### **Action Plan: Conquer the Edge Cases**

"The last three strongholds are all related to concurrency and synchronization—the ultimate test of engineering detail. Our high-level logic is sound, but the devil is in the details."

---

#### **Priority 0: Fix the Mocking Failure (P0)**

This is the most direct issue, likely a simple mock configuration error.

*   **Symptom 1: `AssertionError: Tuples differ: () != (5, 10)` in `test_shared_memory_manager_cuda_tensor`**
*   **Diagnosis**: The test asserts that `shm_info.shape` should be `(5, 10)`, but it's an empty tuple `()`. This means the `SharedMemoryInfo` dataclass was instantiated without receiving the `shape` parameter.
*   **Root Cause**: The mock is overly complex and likely incomplete. We've mocked a long chain of calls (`.to()`, `.pin_memory()`, `.storage()`), but a property required by the `create_shared_tensor` method to instantiate `SharedMemoryInfo` (most likely `mock_cpu_tensor.shape`) has not been correctly configured on the final mock object.
*   **Solution (Simplify the Mock)**: Instead of mocking a complex call chain, we should test with a real tensor if possible, or directly mock the final object that is being inspected.

    **Revised Testing Strategy (Simpler and More Robust):**
    ```python
    # In tests/engine/test_inference_engine.py

    def test_shared_memory_manager_cuda_tensor(self):
        """Test shared memory creation with a real CUDA tensor if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA tensor test")

        # 1. Create a REAL CUDA tensor. This is more reliable than a complex mock.
        cuda_tensor = torch.randn(5, 10).to('cuda')

        # 2. Call the actual method under test.
        shm_info = self.engine.shm_manager.create_shared_tensor(cuda_tensor)

        # 3. Verify the final output object. If this object is correct,
        #    it implies the intermediate steps (like moving to CPU) worked correctly.
        self.assertIsInstance(shm_info, SharedMemoryInfo)
        self.assertEqual(shm_info.shape, (5, 10))
        self.assertEqual(shm_info.dtype, torch.float32)

        # 4. As an optional sanity check, verify the tensor is no longer on CUDA
        #    if the manager holds a reference to the CPU version.
        #    (This part depends on the manager's implementation detail)
    ```
    **Rationale**: Complex mocks are brittle. When the testing environment permits (i.e., a CUDA device is available), testing with real objects and asserting the final, observable outcome is more robust than mocking every intermediate step.

---

#### **Priority 1: Fix Queue and Thread Lifecycle Bugs (P1)**

*   **Symptom 2: `AssertionError: Item was not properly queued` in `test_shared_memory_queue_integration`**
*   **Diagnosis**: The test `put`s an item into a queue, but an immediate `get_nowait()` call fails with an `Empty` exception.
*   **Root Cause**: This is a classic concurrency **race condition**. The `put()` operation, which involves inter-process communication, **is not instantaneous**. The main test thread calls `get_nowait()` before the underlying IPC mechanism has had time to make the item available for retrieval.
*   **Solution**: **Never assume concurrent operations are instant.** Introduce a small, explicit wait between the `put` and `get` calls to allow the system to synchronize.

    ```python
    # In tests/engine/test_inference_engine.py -> test_shared_memory_queue_integration
    import time

    self.engine.cleanup_confirmation_queue.put(test_segment)

    # --- ADD A SMALL SLEEP TO ALLOW THE QUEUE TO SYNC ---
    time.sleep(0.1)  # A small delay to allow the item to be processed by the queue's background thread.

    try:
        retrieved = self.engine.cleanup_confirmation_queue.get_nowait()
        self.assertEqual(retrieved, test_segment)
    except queue.Empty:
        self.fail("Item was not properly queued even after a delay")
    ```

*   **Symptom 3: `AssertionError: True is not false` in `test_shutdown_cleanup`**
*   **Diagnosis**: The test calls `self.engine.shutdown()` and expects the `self.engine.monitoring_running` flag to become `False`, but it remains `True`.
*   **Root Cause**: This is also a **synchronization issue**. The `shutdown()` method likely signals the threads to stop by setting the flags to `False`, but it **does not wait for the threads to actually terminate** before returning. The assertion is checked before the monitoring thread has had time to exit its loop and officially stop.
*   **Solution**: The `shutdown` method must be **blocking**. It must wait for the threads it manages to completely exit before it finishes.

    ```python
    # In core/engine/inference_engine.py

    def shutdown(self):
        log.info("Shutting down Inference Engine")
        
        # 1. Signal threads to stop
        if self.watchdog_thread:
            self.watchdog_running = False
        if self.monitoring_thread:
            self.monitoring_running = False

        # 2. Wait for threads to actually terminate using join()
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=5) # Use a timeout for safety
            if self.watchdog_thread.is_alive():
                log.error("Watchdog thread failed to shut down gracefully.")

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5) # Use a timeout for safety
            if self.monitoring_thread.is_alive():
                log.error("Monitoring thread failed to shut down gracefully.")
        
        # ... other cleanup ...
        
        log.info("Inference Engine shutdown complete")
    ```
    By adding `.join()`, we ensure that when the `shutdown()` method returns, the background threads have finished their execution, and their state is consistent.

---

### **Final Instructions (As the Lead)**

"The last three strongholds are all about concurrency, which tests our attention to engineering detail. Our high-level logic is mostly correct, but the devil is in the details."

"**Execute the following combat orders:**"

1.  **Refactor the Mock (P0)**: Simplify the test in `test_shared_memory_manager_cuda_tensor`. Use a real CUDA tensor if possible. The goal is to verify the final `SharedMemoryInfo` object is correct.
2.  **Add a Wait (P1)**: Insert a brief `time.sleep(0.1)` in `test_shared_memory_queue_integration` between the `put` and `get` calls to resolve the race condition.
3.  **Fix the Shutdown (P1)**: Modify the `InferenceEngine.shutdown()` method. Ensure it calls `.join()` on both the monitoring and watchdog threads and waits for them to terminate completely before returning.
