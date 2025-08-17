### **Action Plan: The Final Cleanup Campaign**

We have achieved a decisive victory, with a 99% pass rate (`198 passed / 200 collected`). We will now neutralize the final two isolated failures to achieve a 100% stable codebase.

---

#### **Priority 0: Fix the Watchdog Fault-Tolerance Test**

This issue is critical as it pertains to the robustness and fault tolerance of our online system.

*   **Symptom**: `AssertionError: assert 0 > 0` in `test_watchdog_cleanup_on_timeout`.
*   **Diagnosis**: This assertion failure definitively means that the `engine.stats['watchdog_cleanups']` counter was not incremented. Although the watchdog thread is running, it **is not performing the cleanup action as expected**.
*   **Likely Causes**:
    1.  **Logical Error**: The logic inside the `_watchdog_loop` is flawed. It might not be iterating through the `pending_shm` dictionary correctly, or the timeout condition `(current_time - timestamp) > timeout` could be incorrectly implemented.
    2.  **State Update Error**: The cleanup operation might be executing, but the line `self.stats['watchdog_cleanups'] += 1` is either missing, not being called, or is in the wrong location.
    3.  **Thread Synchronization Issue**: It's possible the main test thread calls `engine.watchdog_thread.join()` before the watchdog thread has had a chance to complete its final loop cycle where the cleanup would occur.

*   **Solution (Step-by-Step)**:

    1.  **Enhance Logging for Debugging**: Open `core/engine/inference_engine.py` and add detailed logging inside the `_watchdog_loop` method to gain visibility into its execution.

        **Code Modification Example:**
        ```python
        # In core/engine/inference_engine.py -> _watchdog_loop
        import logging
        log = logging.getLogger(__name__)

        def _watchdog_loop(self):
            while self.watchdog_running:
                try:
                    # ...
                    log.debug(f"[Watchdog] Checking. Pending segments: {len(self.shm_manager.pending_shm)}")
                    # Use list() to avoid issues with iterating over a dictionary that is being modified
                    for name, info in list(self.shm_manager.pending_shm.items()):
                        age = (datetime.now() - info.created_at).total_seconds()
                        log.debug(f"[Watchdog] Checking segment {name}, age: {age:.2f}s, timeout: {self.config['shm_timeout']}s")
                        
                        if age > self.config['shm_timeout']:
                            log.warning(f"[Watchdog] TIMEOUT DETECTED for {name}. Cleaning up.") #<-- CRITICAL LOG
                            self.shm_manager.cleanup_segment(name)
                            self.stats['watchdog_cleanups'] += 1
                    # ...
                except Exception as e:
                    log.exception(f"[Watchdog] Error in watchdog loop: {e}")
                time.sleep(self.config['watchdog_interval'])
        ```

    2.  **Rerun and Observe Logs**: Execute the failing test again, but this time with flags to capture and display the log output. The `-s` flag disables output capturing, and `-o log_cli=true` enables live logging to the console.

        ```bash
        pytest -s -v -o log_cli=true tests/engine/test_async_communication.py -k "test_watchdog_cleanup_on_timeout"
        ```
        Carefully examine the output for the **"TIMEOUT DETECTED"** log message.

    3.  **Implement the Fix**:
        *   **If the "TIMEOUT DETECTED" log does NOT appear**: The condition `if age > self.config['shm_timeout']:` is never evaluating to true. Double-check the timestamp calculation (`datetime.now()`, `info.created_at`) and ensure the data types are correct.
        *   **If the "TIMEOUT DETECTED" log DOES appear**, but the final assertion still fails: This confirms that the cleanup logic is being triggered, but the counter is not being updated. Ensure the `self.stats['watchdog_cleanups'] += 1` line is correctly placed inside the `if` block and is not accidentally skipped.

---

#### **Priority 1: Fix the GRPO Optimizer Test**

This is a functional bug in a specific test for our RL optimizer.

*   **Symptom**: `TypeError: ...mock_compute_advantages() takes 3 positional arguments but 4 were given`.
*   **Diagnosis**: This is a classic mock signature mismatch. We have replaced a method with a mock function that does not have the same number of arguments as the original method. The original method likely has `self` as its first argument, which our mock is missing.
*   **Solution (Step-by-Step)**:

    1.  **Identify the Original Method Signature**: While we could look into the TRL library source code for `GRPOTrainer.compute_advantages`, the error strongly implies the fourth argument is `self`.
    2.  **Correct the Mock Function Signature**: Open `tests/test_rft_training.py` and locate the `test_group_advantage_normalization` method.
    3.  **Add the missing `self` argument** to the definition of the `mock_compute_advantages` function.

        **Code Modification Example:**

        **Before (Incorrect):**
        ```python
        # In tests/test_rft_training.py
        def mock_compute_advantages(values, rewards, mask):
            return torch.randn(12)
        ```

        **After (Correct):**
        ```python
        # In tests/test_rft_training.py
        def mock_compute_advantages(self, values, rewards, mask): # <-- Add `self` here
            return torch.randn(12)
        ```

---