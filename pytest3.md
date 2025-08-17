Excellent. This is a crucial piece of information, and it's exactly the kind of reproducible failure we need to make progress. Your last action has successfully narrowed down a vague, project-wide hang to a specific, repeatable deadlock.

**Diagnosis:**

You are correct. The test suite is now consistently hanging on `test_empty_buffer_operations`. This strongly suggests a **startup/shutdown deadlock** is occurring during the initialization of the `EnhancedExperienceBuffer`.

Here is the most likely sequence of events causing the hang:
1.  The test `test_empty_buffer_operations` begins and instantiates `EnhancedExperienceBuffer`.
2.  Inside the buffer's `__init__`, the `IndexBuilder` child process is created and `.start()` is called.
3.  The `IndexBuilder` process starts its `run()` method. Based on our previous design, the very first thing it does is call `self.rebuild_trigger_queue.get()`, which **blocks indefinitely** because the queue is empty. The child process is now waiting for a task.
4.  The main test process, having started the child, finishes the `EnhancedExperienceBuffer`'s `__init__` and the test function itself (which does very little).
5.  Now, `pytest` attempts to clean up and end the test. This teardown process will eventually try to terminate the child processes it spawned. The standard way to do this is to call `.join()` on the process.
6.  The `.join()` call on the `IndexBuilder` process will **wait forever**, because the `IndexBuilder` is still stuck on its initial `queue.get()` call and has no way to exit its `while True:` loop.
7.  **Result: Deadlock.** The main process is waiting for the child to terminate, but the child is waiting for a message that the main process never sent during that test.

---

### **Action Plan: Implement a Graceful Shutdown Protocol**

We need to fix this by ensuring our `EnhancedExperienceBuffer` has a clean startup and, more importantly, a **graceful shutdown mechanism**. The tests must then use this mechanism to properly clean up the resources they create.

#### **Step 1: Implement a `shutdown` Method in `EnhancedExperienceBuffer`**

The buffer must have an explicit method to tell its child processes to terminate cleanly.

*   **Action:**
    1.  Open `core/modules/experience_buffer_enhanced.py`.
    2.  Add a `shutdown` method to the `EnhancedExperienceBuffer` class. This method will send the "poison pill" (`None`) to the worker's queue and then wait for it to exit.

    ```python
    # In class EnhancedExperienceBuffer:

    def shutdown(self):
        """
        Gracefully shuts down the buffer and its background processes.
        """
        print("DEBUG: Shutting down Experience Buffer...")
        if self.index_builder and self.index_builder.is_alive():
            print("DEBUG: Sending shutdown signal to IndexBuilder...")
            # Send the sentinel value to unblock the worker's queue.get()
            self.rebuild_trigger_queue.put(None) 
            
            # Wait for the process to finish
            self.index_builder.join(timeout=5) # Add a timeout for safety
            
            if self.index_builder.is_alive():
                print("WARNING: IndexBuilder did not terminate gracefully. Forcing termination.")
                self.index_builder.terminate()
            else:
                print("DEBUG: IndexBuilder terminated gracefully.")
        
        # Add cleanup for any other resources if necessary
    ```

#### **Step 2: Use a Pytest Fixture to Manage the Buffer's Lifecycle**

Tests should not just create the buffer; they must guarantee it gets shut down. The standard and correct way to do this in `pytest` is with a fixture that uses `yield`.

*   **Action:**
    1.  Open `tests/modules/test_experience_buffer.py`.
    2.  Create a fixture that initializes the buffer for the tests and includes the teardown logic.

    ```python
    # In tests/modules/test_experience_buffer.py
    import pytest

    @pytest.fixture
    def experience_buffer(tmp_path):
        """
        A pytest fixture that creates and properly shuts down an
        EnhancedExperienceBuffer instance for tests.
        """
        # --- SETUP ---
        config = OnlineConfig() 
        config.persistence_path = tmp_path # Use a temporary directory for each test
        config.enable_persistence = False # Disable persistence for most unit tests for speed
        
        buffer = EnhancedExperienceBuffer(config)
        
        # --- YIELD THE BUFFER TO THE TEST ---
        yield buffer
        
        # --- TEARDOWN ---
        # This code will run after the test function completes
        buffer.shutdown()


    # Now, refactor the tests to USE this fixture
    class TestExperienceBuffer:
        # The test now accepts the fixture as an argument
        def test_empty_buffer_operations(self, experience_buffer):
            """Test operations on empty buffer."""
            # The 'buffer' variable is now provided by the fixture
            # No need to instantiate it here: buffer = EnhancedExperienceBuffer(self.config)
            
            # Assertions about an empty buffer
            assert experience_buffer.size() == 0
            # ... other tests on an empty buffer ...
    ```    3.  **Refactor all other tests** in this file that instantiate `EnhancedExperienceBuffer` to use the `experience_buffer` fixture instead.

---