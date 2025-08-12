**Round 3: Core Inference and Gated Learning Mechanisms**

*   **Task 1: Implement the Temporal Ensemble Logic.**
    *   In the main `infer_and_adapt()` function within `core/engine/inference_engine.py`, orchestrate the sequence:
        1.  Get the model's initial prediction.
        2.  Call the `ExperienceBuffer.search_index()` to retrieve k-NN neighbors.
        3.  Pass the initial prediction and neighbor data to the voting module.
*   **Task 2: Implement Configurable Voting Strategies.**
    *   **Goal:** To create a voting system that not only makes decisions but also provides clear, traceable evidence for how each decision was made.
    *   **File:** `core/data_structures.py`, `core/modules/voting.py`.
    *   **Action 1: Define the `VotingResult` Dataclass.**
        *   In `core/data_structures.py`, create a new `@dataclass` named `VotingResult`. It will enforce a strict "contract" for the output of any voting method and must contain the following fields:
            *   `final_answer: Any`
            *   `confidence: float`
            *   `provenance: dict` (The audit trail)
    *   **Action 2: Enhance Voting Methods to Return Provenance.**
        *   In `core/modules/voting.py`, all voting methods (e.g., `hard_majority_vote`, `similarity_weighted_vote`) **must** now return an instance of the `VotingResult` dataclass.
        *   The `provenance` dictionary will contain rich information about the decision process, such as:
            *   `model_self_answer: Any` (The model's own initial answer)
            *   `retrieved_neighbors_count: int`
            *   `neighbor_answers: List[dict]` (A list of neighbor answers with their IDs and similarity scores)
            *   `voting_strategy: str` (The name of the strategy used, e.g., "similarity_weighted")
    *   **Action 3: Utilize Provenance in the Inference Engine.**
        *   In `core/engine/inference_engine.py`, after receiving the `VotingResult` object, the system will log the entire `provenance` dictionary. This creates an invaluable audit trail for every single online decision, which is critical for debugging unexpected behavior or performance degradation.
    *   **Action 4: Implement Comprehensive Unit Tests for Voting Logic.**
        *   **Goal:** To ensure the core decision-making mechanism of the online system is precise, predictable, and correct.
        *   **File:** Create a new test file: `tests/modules/test_voting.py`.
        *   **Action:** This test suite **must** cover all implemented voting strategies. For each strategy (e.g., `hard_majority_vote`, `similarity_weighted_vote`), it should include test cases that:
            1.  **Test for Correct `final_answer`**: Provide mock model predictions and neighbor data, and assert that the method returns the expected consensus answer. Include edge cases like ties.
            2.  **Test for Correct `confidence` Score**: Assert that the calculated confidence score falls within the expected range [0.0, 1.0] and matches the expected value for a given input.
            3.  **Test for Correct `provenance` Structure and Content**: Assert that the returned `VotingResult` object's `provenance` dictionary contains all the required keys (`model_self_answer`, `voting_strategy`, etc.) and that their values match the inputs provided to the test.

*   **Task 3: Implement the Confidence Gating Mechanism.**
    *   **File:** `core/engine/inference_engine.py`
    *   **Action:** After receiving the final answer and confidence score from the `VotingModule`, implement an `if` statement: `if confidence_score > config['confidence_threshold']:` to decide whether to proceed with the learning update. The threshold value must be loaded from the config.
*   **Task 4: Implement a Proportional and Bounded Learning Rate Strategy.**
    *   **Goal:** To replace the discrete dual-mode system with a more elegant, continuous, and safe learning rate policy.
    *   **File:** `core/engine/inference_engine.py`
    *   **Action:**
        *   **Calculate Proportional LR:** When an update is triggered, calculate the learning rate using the proportional formula: `lr_prop = lr_base * (1.0 - confidence_score)`. The `lr_base` is the maximum learning rate, loaded from the config.
        *   **Add Safety Bounds:** To prevent instability from a learning rate that is too high (when `confidence_score` is near 0) or too low (when `confidence_score` is near 1), clip the calculated learning rate within a safe, predefined range.
        *   `final_lr = clip(lr_prop, min=lr_min, max=lr_max)`
        *   The final `final_lr` is a smooth, continuous value that is also bounded. The `lr_min` and `lr_max` values will be loaded from the config file. This combines the elegance of the proportional idea with the stability of a bounded system.
*   **Task 5: Implement a Human-in-the-Loop (HIL) Safety Valve for Initial Rollout.**
    *   **Goal:** To add a layer of human oversight during the critical initial stages of online learning, preventing the system from learning from subtly incorrect or harmful pseudo-labels before the experience buffer is mature.
    *   **File:** `core/engine/inference_engine.py`, `configs/training_params.yaml`, and a new `scripts/human_review_app.py`.
    *   **Action 1: Add HIL Mode to Configuration.**
        *   In `configs/training_params.yaml`, add a new section for HIL, e.g., `online_learning.hil_mode_enabled: true` and `online_learning.hil_review_percentage: 0.02`.
    *   **Action 2: Enhance the Confidence Gating Logic.**
        *   Modify the logic in `Task 3` of this round. When the confidence gate decides an update should be triggered, it will first check if HIL mode is enabled.
        *   If `hil_mode_enabled` is `true`, instead of directly enqueuing the `UpdateTask`, the system will **sample a fraction** (e.g., 2%, based on the config) of these potential updates and enqueue them into a **new, separate `human_review_queue`**. The rest will proceed to the normal `update_queue`.
    *   **Action 3: Create a Simple Human Review Interface.**
        *   In `scripts/human_review_app.py`, use a simple framework like `Gradio` or `Streamlit` to build a review tool.
        *   This tool will fetch tasks from the `human_review_queue`, display the visual input, the question, the model's reasoning trajectory, and the proposed consensus answer (`Answer_final`).
        *   It will provide simple "Approve" / "Reject" buttons for a human expert.
    *   **Action 4: Implement the Feedback Loop.**
        *   If an expert clicks "Approve", the `UpdateTask` is then moved from the `human_review_queue` to the main `update_queue` for normal processing by the `update_worker`.
        *   If "Reject", the task is discarded and logged for later analysis.
    *   **Protocol:** This HIL mode is intended to be active primarily during the initial deployment of the `Pixelis-Online` model, and can be disabled via the configuration file once the system has demonstrated stable and reliable autonomous learning.
