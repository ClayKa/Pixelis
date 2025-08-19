Of course. Here is a detailed, consolidated code modification plan in English. This plan will fix the hanging issue and the two subsequent failures in `tests/engine/test_inference_engine.py`.

The core principle of this fix is to make our background loops (`run_main_loop`, `_monitoring_loop`, `_watchdog_loop`) robust against errors so they do not crash unexpectedly, which is what the tests are designed to verify.

---
### **Action Plan: Final Hardening of `InferenceEngine`**

#### **Objective:**
To resolve the hanging test and the two subsequent failures by implementing robust error handling and a non-blocking main loop in `core/engine/inference_engine.py`.

---
#### **Step 1: Modify `run_main_loop` to be Non-Blocking**

This is the fix for the test that was hanging (`test_run_main_loop_empty_queue_timeout`). We will make the `get()` call on the queue non-blocking by using a timeout.

*   **File to Modify**: `core/engine/inference_engine.py`
*   **Method to Modify**: `run_main_loop` (or `run`, wherever your main `while` loop is)

**Current (Buggy) Code Logic:**
```python
# In core/engine/inference_engine.py
def run_main_loop(self):
    while self.is_running:
        # This call BLOCKS INDEFINITELY if the queue is empty, causing the test to hang.
        request = self.request_queue.get() 
        # ... processing logic ...
```

**New (Corrected) Code Logic:**
```python
# In core/engine/inference_engine.py
import queue # Ensure 'queue' is imported at the top of the file

def run_main_loop(self):
    """The main event loop for processing inference requests."""
    while self.is_running:
        try:
            # 1. Use a non-blocking get() with a timeout.
            #    This value should ideally be loaded from config.
            timeout_seconds = self.config.get('queue_timeout', 1.0)
            request = self.request_queue.get(timeout=timeout_seconds)

            if request is None:  # Sentinel value to stop the loop
                break
            
            # ... process the request as before ...

        except queue.Empty:
            # 2. This is the expected, normal behavior when no requests are available.
            #    It is not an error. We simply continue to the next loop iteration.
            #    This allows the loop to remain responsive and not hang.
            log.debug("No new requests in queue, continuing main loop.")
            continue
```

---
#### **Step 2: Add Robust Error Handling to Background Threads**

This will fix the two failing `TestMissingCoverage` tests by ensuring that unexpected errors in background tasks do not crash their respective threads.

##### **Sub-step 2.1: Harden the Monitoring Loop**

*   **File to Modify**: `core/engine/inference_engine.py`
*   **Method to Modify**: `_monitoring_loop`

**Current (Brittle) Code Logic:**
```python
# In core/engine/inference_engine.py
def _monitoring_loop(self):
    while self.monitoring_running:
        # This line will raise an unhandled exception if psutil fails.
        mem_info = psutil.virtual_memory() 
        self.stats['cpu_memory_usage'] = mem_info.percent
        # ... other stats ...
        time.sleep(...)
```

**New (Robust) Code Logic:**
```python
# In core/engine/inference_engine.py
def _monitoring_loop(self):
    """Periodically gathers system and application health metrics."""
    while self.monitoring_running:
        try:
            # All metric gathering is now inside a try...except block.
            mem_info = psutil.virtual_memory()
            self.stats['cpu_memory_usage'] = mem_info.percent
            # ... other stats gathering ...

        except Exception as e:
            # If any error occurs, log it but DO NOT crash the loop.
            # The monitoring thread will continue to run.
            log.error(f"[Monitor] Failed to gather system stats: {e}", exc_info=True)
        
        time.sleep(self.config.get('monitoring_interval', 5.0))
```

##### **Sub-step 2.2: Harden the Watchdog's Confirmation Processing**

*   **File to Modify**: `core/engine/inference_engine.py`
*   **Method to Modify**: `_process_cleanup_confirmations` (This is part of the `_watchdog_loop`)

**Current (Brittle) Code Logic:**
```python
# In core/engine/inference_engine.py -> part of the watchdog logic
def _process_cleanup_confirmations(self):
    while not self.cleanup_confirmation_queue.empty():
        # This line will raise an unhandled exception if the queue get() fails.
        shm_name = self.cleanup_confirmation_queue.get_nowait()
        if shm_name in self.shm_manager.pending_shm:
            del self.shm_manager.pending_shm[shm_name]
```

**New (Robust) Code Logic:**
```python
# In core/engine/inference_engine.py -> part of the watchdog logic
def _process_cleanup_confirmations(self):
    """Safely process all available cleanup confirmations from the worker."""
    while not self.cleanup_confirmation_queue.empty():
        try:
            # The get() call is now inside a try...except block.
            shm_name = self.cleanup_confirmation_queue.get_nowait()
            
            if shm_name in self.shm_manager.pending_shm:
                log.debug(f"[Watchdog] Received cleanup confirmation for: {shm_name}")
                del self.shm_manager.pending_shm[shm_name]

        except Exception as e:
            # If any error occurs (e.g., queue is empty due to a race condition),
            # log it and continue to the next potential item.
            log.error(f"[Watchdog] Error processing cleanup confirmation: {e}", exc_info=True)
            continue```

---
### **Summary of Actions**

1.  **In `run_main_loop`**: Wrap the `request_queue.get()` call in a `try...except queue.Empty` block and add a `timeout` to the `get()` call.
2.  **In `_monitoring_loop`**: Wrap the entire content of the `while` loop in a `try...except Exception` block.
3.  **In `_process_cleanup_confirmations`**: Wrap the logic inside the `while` loop with a `try...except Exception` block.

After implementing these three changes, the tests in `tests/engine/test_inference_engine.py` should no longer hang and the two `FAILED` tests should now pass.