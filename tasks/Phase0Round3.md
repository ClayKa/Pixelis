**Round 3: Modify Model Architecture**

*   **Goal:** To implement a Parameter-Efficient Fine-Tuning (PEFT) strategy using a novel, robust, and verifiable Dynamic Rank Allocation mechanism.
*   **Task 1: Preliminary Full Fine-Tuning.**
    *   **Action:** On a small, representative subset of the training data, perform a brief, full-parameter fine-tuning run to produce a `W_finetuned` checkpoint. The subset will be created by stratified sampling from the full dataset to ensure it maintains the same distribution of task types and difficulty levels.
    *   **Note:** The primary objective of this step is not to achieve a high-performance model, but rather to obtain a meaningful delta_W. Therefore, the training run does not need to reach full convergence and can be stopped early to conserve computational resources.

*   **Task 2: SVD Analysis Script.**
    *   **File:** `scripts/analyze_lora_ranks.py`
    *   **Action:** Implement the script to perform SVD on `delta_W = W_finetuned - W_pretrained`, using randomized SVD for efficiency.

*   **Task 3: Implement Robust Dynamic Rank Configuration.**
    *   **Goal:** To generate a LoRA rank configuration that is not only data-driven but also stable and avoids overfitting.
    *   **File:** `scripts/analyze_lora_ranks.py`, `configs/lora_rank_config.json`
    *   **Action 1: Calculate Raw Ranks.**
        *   For each layer, analyze the distribution of its singular values to determine a raw suggested rank (`r_raw`), e.g., based on the number of values needed to retain 90% of the spectral energy.
    *   **Action 2: Apply Heuristic Constraints and Regularization.**
        *   **Implement Rank Bounding:** Apply strict upper and lower bounds to the calculated ranks. Any rank below a `min_r` (e.g., 4) will be clipped to `min_r`, and any rank above a `max_r` (e.g., 64) will be clipped to `max_r`. This prevents pathologically small or large ranks.
        *   **Implement Rank Smoothing (Optional):** Apply a smoothing function across layers within the same module (e.g., all attention layers in a block) to prevent extreme variance in ranks, which can sometimes lead to instability.
    *   **Action 3: Store Enriched Metadata.**
        *   The output `configs/lora_rank_config.json` will now store not just the final rank, but also the metadata used to derive it, such as `r_raw`, `energy_retained`, and the singular value decay rate. This is invaluable for later analysis and debugging.

*   **Task 4: Integration with PEFT.**
    *   **File:** Model definition files (e.g., in `core/models/`).
    *   **Action:** Implement the logic to dynamically construct the `LoraConfig` from the generated JSON file and wrap the base model. To maintain clean code, this logic should be encapsulated within a factory function or a class method, e.g., create_peft_model_from_config(base_model, rank_config_path).

*   **Task 5: Enhance Unit Testing with Performance Assertions.**
    *   **Goal:** To verify not only the correctness of the dynamic LoRA insertion but also its impact on key performance characteristics.
    *   **File:** `tests/modules/test_model_init.py`
    *   **Action 1: Verify Correct Insertion.**
        *   Write assertions to check that LoRA layers have been correctly and heterogeneously inserted according to the config file and that the model can perform a forward pass.
    *   **Action 2: Add a Memory Usage Assertion.**
        *   Create a test case that initializes the model and measures its total VRAM footprint.
        *   Assert that `torch.cuda.memory_allocated()` is below a predefined, reasonable threshold (e.g., `assert memory_usage < 10 * 1024**3` for a 7B model). This acts as a regression test to prevent future changes from causing unexpected memory bloat.
    *   **Action 3: Add a Latency Assertion.**
        *   Create a test case that performs a single, small-batch inference pass on a sample input.
        *   Measure the wall-clock time for this forward pass.
        *   Assert that the latency is below a predefined threshold (e.g., `assert latency < 500` milliseconds). This ensures that the chosen rank configuration does not make the model unacceptably slow for its intended application.
    *   **Action 4: Persist SVD Analysis Artifacts.**
        *   **Goal:** To save the intermediate and final results of the SVD analysis for easier debugging, visualization, and inclusion in the final paper.
        *   **File:** `scripts/analyze_lora_ranks.py`.
        *   **Action:** Modify the script so that, in addition to the final `lora_rank_config.json`, it also saves the following outputs to a designated directory (e.g., `analysis_outputs/svd/`):
            1.  **Singular Value Plots:** For key layers, save plots (`.png`) of their singular value decay curves.
            2.  **Raw Data:** Save the raw singular values for each layer in a machine-readable format (e.g., `.csv` or `.json`).
            3.  **Delta_W Matrices (Optional):** For a few small, representative layers, save the `delta_W` weight matrices themselves for potential deeper analysis.
        *   **Note on Thresholds:** It must be clearly documented that these performance assertion thresholds are hardware-dependent. For example, the test docstrings or a README.md in the tests/ directory should state: "The following thresholds are benchmarked and set based on a specific hardware configuration (e.g., NVIDIA A100 80GB GPU). They may require adjustment when running tests on different hardware."