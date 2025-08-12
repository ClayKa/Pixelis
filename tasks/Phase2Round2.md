**Round 2: Intelligent Experience Buffer Implementation**

*   **Task 1: Define the Buffer's Core Structure.**
    *   In `core/modules/experience_buffer.py`, create the `ExperienceBuffer` class. Use `collections.deque(maxlen=...)` as the underlying data structure to automatically manage its maximum size, loaded from `configs/training_params.yaml`.
*   **Task 2: Implement Multi-Factor Priority Calculation with value tracking.**
    *   **Goal:** To prioritize experiences based on their initial value and track their long-term contribution in a lightweight manner.
    *   **File:** `core/data_structures.py`, `core/modules/experience_buffer.py`
    *   **Action:**
        *   **Modify Experience Dataclass:** Add two simple integer fields: `retrieval_count: int = 0` and `success_count: int = 0`.
        *   **Initial Priority:** The initial priority calculation remains the same (based on `P_uncertainty` and `P_reward`).
        *   **Value Tracking:** After an experience is retrieved and used in the temporal ensemble, the `inference_engine` will track if the final consensus answer was "correct" (based on high confidence). If so, it will increment the `success_count` of the retrieved experiences. The `retrieval_count` is always incremented.
        *   **Lightweight Value Decay:** Instead of a computationally expensive dynamic recalculation of all priorities, the value decay is applied implicitly. When sampling from the buffer for replay learning, the sampling probability will be influenced by both the initial priority and the long-term success rate (`success_count` / `retrieval_count`). This achieves a similar outcome with a much lower computational cost.
        *   Define an `Experience` dataclass to hold all relevant data for a single event.
        *   Implement a method within the `ExperienceBuffer` class, `_calculate_initial_priority(experience)`, which computes a priority score as a weighted sum of several factors:
            1.  `P_uncertainty`: Based on the prediction's entropy or the confidence score from the voting module.
            2.  `P_reward`: The absolute value of the total reward calculated for this experience (post-hoc).
            3.  `P_age`: A time-based component, initially zero.
*   **Task 3: Implement Hybrid k-NN Retrieval.**
    *   **Goal:** To retrieve more relevant experiences by considering both visual and intentional similarity.
    *   **File:** `core/modules/experience_buffer.py`, `core/engine/inference_engine.py`
    *   **Action:**
        *   **Feature Combination:** When an `Experience` is added to the buffer, create a hybrid embedding.
        *   **Lightweight Hybrid Representation:** Avoid introducing a new text encoder. Reuse the base model's existing embedding capabilities. Create the hybrid embedding by taking a simple weighted average of the global pooled visual features (`v_embed`) and the question's text embedding (`q_embed`), which can be extracted from the model's text encoder: `hybrid_embed = 0.7 * v_embed + 0.3 * q_embed`.
        *   The FAISS index will be built upon these `hybrid_embed` vectors. The retrieval process remains the same but will now naturally find experiences similar in both visual context and user intent.
*   **Task 4: Integrate k-NN Index for Neighbor Retrieval.**
    *   Integrate the `faiss-gpu` library. The `ExperienceBuffer` will maintain a FAISS index synchronized with the visual embeddings of the experiences it stores. Implement `add_to_index` and `search_index` methods.
*   **Task 5: Integrate Hybrid k-NN Index with Strong Consistency Guarantees.**
    *   **Goal:** To ensure the `ExperienceBuffer` (both the deque and the FAISS index) remains perfectly consistent and crash-resistant in a multi-process environment.
    *   **File:** `core/modules/experience_buffer.py`
    *   **Action 1: Implement a Thread-Safe/Process-Safe Write Lock.**
        *   The `ExperienceBuffer` class will be initialized with a `multiprocessing.Lock`.
        *   All methods that **modify** the buffer's state (`add`, `remove`, `update_priorities`) **must** acquire this lock at the beginning of their execution and release it at the end, typically using a `with lock:` statement to ensure it's always released. This enforces a **"single-writer"** principle, preventing concurrent write operations from corrupting the state.
    *   **Action 2: Ensure Atomic Operations for Consistency.**
        *   Within the locked `add` method, the implementation must guarantee that the new experience is successfully added to **both the deque and the FAISS index** before the lock is released. A `try...finally` block should be used.
        *   Similarly, the `remove` method must ensure the experience is removed from both structures atomically.
    *   **Action 3: Implement a Robust, WAL-based Persistence and Recovery Strategy with Asynchronous Indexing.**
        *   **Goal:** To provide a high-performance, crash-proof, and non-blocking mechanism for managing and persisting the `ExperienceBuffer`, ensuring data integrity and 100% availability for read operations.
        *   **Sub-Task 3.1: Design the WAL (Write-Ahead Log) System with a Tiered Technology Strategy.**
            *   **Goal:** To select a WAL implementation that balances simplicity with the high-throughput, low-latency requirements of a production-grade online system.
            *   **Files:** `experience_data.wal` (or `.mdb`/`.db`), `index_operations.wal`, `snapshots/`.
            *   **Action (Technology Choice):**
                1.  **Default Implementation (File-based WAL):** The primary and default implementation will use the **transactional file writing pattern (`tmp -> fsync -> rename`)** for the two WAL files. This provides excellent crash-safety with minimal external dependencies, making it ideal for research and moderate-load scenarios.
                2.  **[NEW] High-Throughput Backend (Embedded KV Store):** The system architecture **must** be designed to accommodate a more performant backend as a drop-in replacement.
                    *   **Selection:** The project will select a lightweight, embedded transactional key-value store, with **LMDB** or **SQLite** being the primary candidates due to their proven stability and high write throughput.
                    *   **Integration Plan:** The `ExperienceBuffer` will interact with the persistence layer through a **dedicated adapter interface** (e.g., `PersistenceAdapter` abstract base class). This allows the main code to be agnostic to the underlying storage mechanism. Implementations like `FilePersistenceAdapter` and `LMDBCPersistenceAdapter` will exist.
                    *   **Configuration:** The choice of persistence backend (`"file"` vs. `"lmdb"`) will be a configurable option in `configs/training_params.yaml`, allowing for easy switching.
        *   **Sub-Task 3.2: Implement Atomic, Sequential WAL Writes.**
            *   **File:** `core/modules/experience_buffer.py` (within the `add` method).
            *   **Protocol:** All write operations (e.g., `add(experience)`) must be protected by a `multiprocessing.Lock` and must execute the following steps in strict, sequential order to guarantee consistency:
                1.  **Write Data Log:** Atomically write the full `experience` object to `experience_data.wal` (using the `tmp -> fsync -> rename` pattern).
                2.  **Write Operation Log:** Atomically write the corresponding `{op: "add", ...}` event to `index_operations.wal`.
                3.  **Update In-Memory State:** Only after both logs are confirmed written to disk, modify the in-memory `deque` and apply the change to the live, in-memory `FAISS index`.
        *   **Sub-Task 3.3: Implement the Asynchronous `IndexBuilder` Worker.**
            *   **Process:** A dedicated background process or thread, the `IndexBuilder`, will be responsible for periodically rebuilding the FAISS index from disk to maintain optimal search performance.
            *   **Trigger:** The `IndexBuilder` will be triggered not by a simple timer, but by the number of new entries in the `index_operations.wal` file (e.g., "rebuild after every 1000 new operations").
            *   **Action (The "Blue-Green" Deployment for the Index):**
                1.  The `IndexBuilder` reads the data from the **last successful snapshot** and applies all operations from the **entire `index_operations.wal` log**.
                2.  It uses this data to build a **completely new, optimized FAISS index** in a temporary file (`new_index.faiss.tmp`).
                3.  Once building is complete, it performs an **atomic `os.rename()`** to swap the temporary file with the current live one (e.g., `os.rename("new_index.faiss.tmp", "live_index.faiss")`).
                4.  The main `ExperienceBuffer` object is then notified to hot-reload this new, updated index into memory for subsequent read operations.
        *   **Sub-Task 3.4: Implement an Integrated Snapshotting and Log Truncation Process.**
            *   **Trigger:** This critical process is **exclusively triggered by the `IndexBuilder`** as the final step of a successful reconstruction run.
            *   **Action:** After the `IndexBuilder` has successfully completed the atomic swap of the FAISS index file:
                1.  It acquires the main `ExperienceBuffer` write lock.
                2.  It saves a new, full snapshot containing the **current state of the `deque`** and a reference to the **newly activated `live_index.faiss`**.
                3.  After the snapshot is successfully written, it **atomically truncates (clears) both** `experience_data.wal` and `index_operations.wal`.
                4.  It releases the write lock.
        *   **Sub-Task 3.5: Implement the Comprehensive, Dual-WAL-Aware Recovery Mechanism.**
            *   **File:** `core/modules/experience_buffer.py` (within the `load_from_disk` method).
            *   **Action:** Upon application startup, the recovery process must follow this strict sequence:
                1.  **Load Snapshot:** Load the most recent complete snapshot. This restores the `deque` and the `FAISS index` to the last known-good state.
                2.  **Replay Data WAL:** Read `experience_data.wal` from start to finish. For each entry, add the full `Experience` object back into the in-memory `deque`.
                3.  **Replay Operations WAL:** Read `index_operations.wal`. For each `add` or `remove` operation, apply it to the in-memory `FAISS index` that was loaded from the snapshot.
                4.  **Final State:** After replaying both logs, the in-memory state of the `deque` and the `FAISS index` are now guaranteed to be fully consistent and up-to-date.
    *   **Action 4: Add Concurrency and Recovery to Unit Tests.**
        *   The `tests/modules/test_experience_buffer.py` must be enhanced.
        *   Create tests that spawn multiple processes trying to write to and read from the same buffer instance simultaneously to verify that the lock prevents data corruption.
        *   Create a test that simulates a crash: add some items, call `save_snapshot`, add more items, then re-initialize the buffer and verify that it correctly recovers all items.
    *   **Action 5: Implement Configurable FAISS Backend.**
        *   **Goal:** To allow flexible deployment on different hardware and efficient handling of varying buffer sizes.
        *   **File:** `core/config_schema.py`, `configs/training_params.yaml`, `core/modules/experience_buffer.py`.
        *   **Implementation:**
            1.  **Configuration Schema:** In `core/config_schema.py`, add a `FaissConfig` dataclass with a field `backend: Literal["gpu", "cpu"] = "gpu"` and other backend-specific parameters (e.g., `n_probes` for IVF).
            2.  **YAML Configuration:** The main config file will now have a section to control the FAISS backend choice at runtime.
            3.  **Conditional Initialization:** In the `ExperienceBuffer`'s `__init__` method, it will read this configuration. An `if-else` block will determine whether to initialize a `faiss.GpuIndex...` or a `faiss.IndexIVFPQ...` (CPU index).
            4.  **Automatic Fallback (Optional but Recommended):** The implementation can include a `try-except` block. It first attempts to initialize the GPU index. If it fails (e.g., due to insufficient VRAM), it logs a warning and automatically falls back to initializing the CPU index, ensuring the program doesn't crash.
  *   **Action 6: Document Engineering Trade-offs in `ARCHITECTURE.md`.**
        *   **Goal:** To provide clear, long-term documentation on the design rationale behind the `ExperienceBuffer`'s complexity, ensuring future maintainability and onboarding efficiency.
        *   **File:** `docs/ARCHITECTURE.md`
        *   **Action:** Create a new, dedicated section in the architecture document titled **"Experience Buffer: High-Reliability Design Decisions"**. This section must explicitly detail the trade-offs made in the pursuit of a production-grade system. It should include, at a minimum, the following points:
            *   **On Durability (WAL + Snapshots):**
                *   **Decision:** "We employ a Write-Ahead Log (WAL) and periodic snapshotting mechanism for data persistence."
                *   **Rationale / Pro:** "This provides maximum crash consistency. No acknowledged write operation will ever be lost, even in the case of a sudden system failure."
                *   **Trade-off / Con:** "The cost of this durability is a minor increase in write latency (due to requiring disk `fsync`) and a higher implementation complexity compared to a simple in-memory buffer."
            *   **On Read Availability (Asynchronous Index Rebuilding):**
                *   **Decision:** "The FAISS index is rebuilt asynchronously in a background process, followed by an atomic swap."
                *   **Rationale / Pro:** "This guarantees that read operations (i.e., k-NN searches) are never blocked and always operate with low latency on a fully consistent index."
                *   **Trade-off / Con:** "The trade-off is that the index is not real-time. There is a configurable delay between when an experience is added and when it becomes searchable in the index."
            *   **On Flexibility (Pluggable Backends):**
                *   **Decision:** "The persistence layer and FAISS backend are accessed through abstract adapter interfaces, allowing for configurable implementations (e.g., File vs. LMDB, CPU vs. GPU)."
                *   **Rationale / Pro:** "This provides maximum long-term flexibility, allowing the system to be deployed on diverse hardware and to scale to higher throughputs by simply changing a configuration line."
                *   **Trade-off / Con:** "The cost of this flexibility is an added layer of abstraction, which slightly increases the cognitive overhead for developers working on the buffer's core logic."


*   **Task 6: Develop Comprehensive Buffer Unit Tests.**
    *   Create `tests/modules/test_experience_buffer.py` to test all public methods: `add`, `sample` (for prioritized sampling), `search_index`, and the priority update logic.