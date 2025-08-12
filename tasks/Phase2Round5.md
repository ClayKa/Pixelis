**Round 5: Main Integration, Observability, and Bootstrapping**

*   **Task 1: Build the Main `infer_and_adapt()` Orchestration Function.**
    *   **File:** `core/engine/inference_engine.py`
    *   **Action:** Tie all previous tasks together into a single, cohesive function that represents one full pass through the online evolution loop: from receiving a request to potentially enqueuing an update task.
*   **Task 2: Implement the Cold Start Bootstrapping Strategy.**
    *   Add a condition at the beginning of `infer_and_adapt()`: `if len(experience_buffer) < config['cold_start_threshold']`.
    *   If true, the system will operate in a "conservative mode": it will bypass the ensemble voting and return its own answer directly. It will not perform learning updates but will add all experiences to the buffer with a high initial priority to rapidly build a useful memory.
*   **Task 3: Integrate Comprehensive Monitoring with Automated Alerting.**
    *   **Goal:** To create a robust, real-time health monitoring system for the online engine.
    *   **File:** `core/engine/inference_engine.py`, `core/engine/update_worker.py`
    *   **Action 1: Track Key Health Indicators.**
        *   In addition to the existing metrics, explicitly track and log these vital signs to wandb:
            *   `update_rate`: The number of model updates performed per minute. A sudden drop to zero indicates a problem.
            *   `faiss_failure_rate`: The percentage of k-NN search calls that fail or return errors.
            *   `mean_kl_divergence`: The average KL divergence per update. A sustained, unusually high value indicates the online policy is drifting too far from its base, which can be dangerous.
            *   `queue_size`: The current size of the `update_queue`. A continuously growing queue indicates the `update_worker` cannot keep up with the inference engine.
    *   **Action 2: Implement an Alerting Mechanism.**
        *   Create a simple alerting module (e.g., `core/modules/alerter.py`) that can send notifications (e.g., via Slack webhook or email).
        *   In the main engine and worker loops, add simple `if` conditions to check for critical thresholds. For example: `if mean_kl_divergence > config['kl_alert_threshold']: alerter.send_alert(...)`.
*   **Task 4: Conduct End-to-End System Testing.**
    *   **Goal:** To validate the functional correctness and short-term stability of the entire online system.
    *   **File:** `scripts/run_online_simulation.py`.
    *   **Action:** This script will serve as the engine for both short and long tests. It will be configurable to run for a specified duration or number of requests. The existing plan to use it for a basic end-to-end test remains.

*   **Task 5: Implement and Automate a Long-Running Stability and Stress Test.**
    *   **Goal:** To rigorously verify the long-term stability of the online learning engine by automatically running it under sustained loading injecting controlled failures, and monitoring for critical resource and performance degradation.
    *   **File:** The test will be executed by the `ci-long-running.yml` workflow, which calls `scripts/run_online_simulation.py` with specific parameters.
    *   **Action 1: Configure the Stress Test Scenario.**
        *   The `run_online_simulation.py` script will be run with parameters for a long duration (e.g., `--duration 8h` for an 8-hour test).
        *   The simulation will generate a continuous, high-frequency stream of mock data to stress the IPC queues and the update worker.
    *   **Action 1.5: Implement Chaos Injection in the Simulation Script.**
        *   **File:** `scripts/run_online_simulation.py`
        *   **Action:** Enhance the simulation script to support chaos engineering principles.
            1.  Add new command-line arguments to the script, such as `--enable-chaos-testing` and `--worker-crash-probability=0.01`.
            2.  When chaos mode is enabled, the main simulation loop will, at each iteration, check against the probability and randomly terminate (`os.kill` or `.terminate()`) the `update_worker` process.
    *   **Action 2: Implement Automated Health Metric Monitoring and Assertion.**
        *   During the long run, the simulation script **must** periodically (e.g., every 5 minutes) query and log the system's health metrics from a shared monitoring utility.
        *   **The CI job will fail** if any of the following assertions are violated at the end of the test:
            1.  **Memory Leak Detection:** Assert that the memory usage of the `inference_engine` and `update_worker` processes has not grown beyond a reasonable threshold (e.g., `final_memory < initial_memory * 1.1`).
            2.  **Queue Size Assertion:** Assert that the `update_queue` size remains bounded and does not continuously grow, which would indicate the worker is falling behind.
            3.  **FAISS Failure Rate Assertion:** Assert that the `faiss_failure_rate` logged to `wandb` remains at or near zero.
            4.  **WAL Growth Assertion:** Assert that the WAL files are being correctly truncated and do not grow indefinitely.
            5.  **Fault Recovery Assertion:** The simulation script must track the number of injected worker crashes. At the end of the test, it must assert that the number of successful resource cleanups (as detected by the `inference_engine`'s watchdog) is equal to the number of injected crashes. This directly verifies that the fault tolerance mechanism from `Phase 2, Round 1` is working as expected.
    *   **Action 3: Integrate with the Monitoring Dashboard.**
        *   The long-running test will log all health metrics to `wandb`.
        *   A dedicated `WandB` dashboard named "Nightly Stability Test" will be created to visualize these metrics over the entire duration of the test, making it easy to spot negative trends (like a slow, creeping memory leak) visually.
        *   The `WandB` dashboard **must** now include a plot that overlays the `update_rate` with event annotations marking each time a "Worker Crash" was injected. This allows for visual confirmation of the system's response to failures.

*   **Task 6: Design and Implement a Worker Process Supervisor for Automatic Restart.**
    *   **Goal:** To move from a passive fault tolerance model (cleanup only) to an active self-healing model by ensuring the `update_worker` process is automatically restarted upon failure.
    *   **File:** `core/engine/inference_engine.py` or a new parent launcher script.
    *   **Action 1: Implement the Supervisor Logic.**
        *   The process responsible for launching the `update_worker` (likely the `inference_engine` or a main launcher script) will act as its supervisor.
        *   The supervisor will hold a reference to the `update_worker` process object.
        *   In its main loop, the supervisor will periodically (e.g., every few seconds) check `update_worker.is_alive()`.
    *   **Action 2: Implement the Restart Mechanism.**
        *   If `is_alive()` returns `False` for a previously running worker, the supervisor will:
            1.  Log a critical error: `[Supervisor] - Update worker process has terminated unexpectedly. Attempting to restart.`
            2.  Wait for the watchdog in the `inference_engine` to complete its cleanup of any stale resources (this is a crucial coordination step).
            3.  Instantiate and start a **new** `update_worker` process to replace the failed one.
    *   **Action 3: Add Restart Logic to the Chaos Test.**
        *   The long-running chaos test becomes the primary validation tool for this mechanism.
        *   When running the chaos test, the monitoring system should observe the following pattern in the WandB logs:
            a. A worker crash is injected.
            b. The `update_rate` metric drops to zero.
            c. After a short delay (the supervisor's check interval), the `update_rate` recovers, proving that the new worker has started and is processing tasks from the queue again.
        *   The test will now also assert that the total number of updates continues to grow throughout the test, despite the injected crashes.