### **Phase 2: Online Training （TTRL/Test-Time Reinforcement Learning Evolution）**

**Round 1: Asynchronous Architecture**

*   **Task 1: Establish Core Process Files.**
    *   Create two main files: `core/engine/inference_engine.py` for the main inference process and `core/engine/update_worker.py` for the background learning process.

*   **Task 2: Implement Inter-Process Communication (IPC) Queues.**
    *   Instantiate three `torch.multiprocessing.Queue` objects to manage data flow:
        1.  `request_queue`: For incoming inference requests.
        2.  `response_queue`: For sending final predictions back to the user.
        3.  `update_queue`: For sending `UpdateTask` data from the inference engine to the update worker.
        4.  cleanup_confirmation_queue: A lightweight, reverse-direction queue for the update_worker to notify the inference_engine that a shared memory segment has been successfully processed and unlinked.

*   **Task 3: Implement a High-Performance, Stable Tensor Transfer Strategy.**
    *   **Goal:** To safely and efficiently transfer large data tensors (specifically image features) between the main inference process and the background update worker, avoiding common pitfalls of inter-process GPU communication.
    *   **File:** `core/data_structures.py`, `core/engine/inference_engine.py`, `core/engine/update_worker.py`.
    *   **Action 1: Define the Data Transfer Protocol.**
        *   The core protocol will be: **Never pass CUDA tensors directly through a `multiprocessing.Queue`**. All tensors must be transferred via CPU shared memory.
        *   This protocol will be documented in `docs/ARCHITECTURE.md`.
    *   **Action 2: Use CPU Pinned Memory for Efficient Host-to-Device Transfer.**
        *   In the `inference_engine` (where data originates), after a tensor is processed on the GPU, it will be moved to **CPU pinned memory** using `.to('cpu', non_blocking=True).pin_memory()`.
        *   **Rationale:** Pinned memory is a special region of CPU memory that the GPU can access directly via DMA (Direct Memory Access), which makes subsequent transfers from CPU back to GPU (in the `update_worker`) significantly faster.
    *   **Action 3: Implement Transfer via `torch.multiprocessing.shared_memory` with **Stateful Tracking**.**
        *   **File:** `core/engine/inference_engine.py`
        *   **Action:**
            1.  The `inference_engine` will maintain a local dictionary, e.g., `self.pending_shm = {}`, to track all shared memory (SHM) segments it has created but not yet received cleanup confirmation for. The keys will be the SHM names and the values will be their creation timestamps.
            2.  When a tensor is placed into a `shared_memory` segment, its name and timestamp are added to `self.pending_shm`.
            3.  The `UpdateTask` object, containing the SHM metadata, is put on the `update_queue`.
    *   **Action 4: Implement Reconstruction and **Confirmation** in the Worker Process.**
        *   **File:** `core/engine/update_worker.py`
        *   **Action:**
            1.  The `update_worker` fetches the `UpdateTask` and reconstructs the tensor from shared memory.
            2.  ... (uses the tensor for model update) ...
            3.  **After** using the tensor and calling the necessary cleanup/unlink methods on the shared memory segment, the worker **must** put the name of the cleaned SHM segment onto the `cleanup_confirmation_queue`.
    *   **Action 5: Implement the "Watchdog" and Cleanup Logic in the Inference Engine.**
        *   **File:** `core/engine/inference_engine.py`
        *   **Action:** The `inference_engine`'s main loop will include a "watchdog" mechanism that runs periodically. This watchdog will:
            1.  **Process Confirmations:** Non-blockingly read all names from the `cleanup_confirmation_queue` and remove the corresponding entries from its `self.pending_shm` dictionary.
            2.  **Check for Worker Liveness:** Check if the `update_worker` process `.is_alive()`.
            3.  **Enforce Timeouts:** Iterate through its `self.pending_shm`. If any SHM segment's age (current time - creation timestamp) exceeds a configurable timeout, or if the worker process is not alive, the `inference_engine` will:
                a. Log a critical warning: `[Watchdog] Worker appears unresponsive or has crashed. Forcibly cleaning up stale shared memory segment: [shm_name]`.
                b. Proactively call the necessary unlink/cleanup functions on that stale shared memory segment itself.
                c. Remove the entry from `self.pending_shm`.
            *   **This two-pronged approach (confirmation queue + watchdog timer/liveness check) provides a robust lifecycle management system for shared memory resources.**

*   **Task 4: Create Robustness, **Fault-Tolerance**, and Communication Tests.**
    *   **File:** `tests/engine/test_async_communication.py`, `tests/engine/test_ipc.py`.
    *   **Action 1:** Develop `tests/engine/test_async_communication.py` to simulate the full normal workflow...
    *   **Action 2:** Create a Shared Memory Transfer Test in `tests/engine/test_ipc.py`...
        *   *(a, b, c, d remain the same)*
    *   **Action 3: Implement a Fault-Tolerance Test for Worker Failure.**
        *   **Goal:** To verify that the watchdog mechanism correctly cleans up resources after an unexpected worker process termination.
        *   **File:** A new test function within `tests/engine/test_async_communication.py`.
        *   **Action:** The test case must perform the following steps:
            1.  Start the `inference_engine` and `update_worker` processes.
            2.  The parent test process sends a request that causes the `inference_engine` to create a shared memory segment and send an `UpdateTask` to the worker.
            3.  **Immediately and forcefully terminate (`os.kill` or `.terminate()`) the `update_worker` process** before it has a chance to process the task or send a cleanup confirmation.
            4.  Wait for a duration longer than the watchdog's timeout.
            5.  **Assert** that the `inference_engine`'s `pending_shm` dictionary is now empty, proving that the watchdog has detected the stale segment and cleaned it up.
            6.  Attempt to connect to the shared memory segment from the test process and **assert** that it raises a `FileNotFoundError`, confirming it has been correctly unlinked from the system.