**Round 2: Testing for Robustness, Efficiency, and Continual Learning**

*   **Task 1: Test Continual Learning and Domain Adaptation.**
    *   **Action:** Use the final `Pixelis-Online` model for this test.
        1.  **Adaptation Speed:** Present the model with data from a completely new, unseen domain. Plot its performance over time (as it processes more samples) to show how quickly it adapts. Compare this to the static `Pixelis-RFT-Full` model's performance on the same domain.
        2.  **Forgetting Resistance:** Expose the model sequentially to Task A, then Task B, then Task C. After it has learned Task C, re-evaluate its performance on Task A. A small drop in performance demonstrates its ability to resist catastrophic forgetting, thanks to the time-boosted experience replay.
*   **Task 2: Profile Efficiency and Latency.**
    *   **Action:**
        1.  Use `torch.profiler` on the `infer_and_adapt` function of the `pixelis-Online` model.
        2.  **Specifically measure the latency overhead** of the key components: k-NN search (`faiss`), the curiosity reward calculation (dynamics model forward pass), and the coherence reward calculation (embedding comparisons).
        3.  Report the final P99 latency and average memory usage to demonstrate the system's real-world viability.
*   **Task 3: Test Robustness to Noisy Data.**
    *   **Action:** Inject noisy data (e.g., blurred images, irrelevant text prompts) into the online stream for `Pixelis-Online`.
    *   Monitor the **confidence score** and **update trigger rate**. A robust system should show a significant drop in both metrics when faced with noise, demonstrating that the confidence gating mechanism is successfully preventing the model from learning from corrupted signals.
*   **Task 4: Conduct Hyperparameter Sensitivity Analysis.**
    *   **Action:** Isolate the final offline model (`Pixelis-RFT-Full`) for this analysis to reduce computational cost. Systematically vary the values of the two most critical hyperparameters:
        *   **Reward Weights:** Fix one weight (e.g., `w_coherence` = 0.1) and vary the other (`w_curiosity` in a range like [0.01, 0.05, 0.1, 0.2]). Repeat this for the other weight.
        *   **Confidence Threshold:** For the `Pixelis-Online` model, vary the confidence gating threshold (e.g., in a range like [0.6, 0.7, 0.8, 0.9]).
    *   **Goal:** Plot the model's final performance against these parameter values. This demonstrates how sensitive the system is to jejich exact tuning and helps identify a robust operating range. This analysis should be included in the appendix of the final paper.
*   **Task 5: Conduct Tool-Specific Stress Tests.**
    *   **Goal:** To evaluate the robustness of the individual visual operations under challenging, real-world conditions.
    *   **Action 1: Create Augmented Stress-Test Datasets.**
        *   Create a new script `scripts/augment_data_for_stress_test.py`.
        *   This script will take your Custom Capabilities Benchmark and apply a series of challenging visual augmentations:
            *   **For `SEGMENT_OBJECT_AT` / `GET_PROPERTIES` tasks:** Introduce partial occlusion, low-light conditions, and motion blur.
            *   **For `READ_TEXT` tasks:** Apply perspective distortion, varied fonts, and visual noise to the text regions.
            *   **For `TRACK_OBJECT` tasks:** Use videos with rapid motion, camera shake, and temporary full occlusion of the target object.
    *   **Action 2: Evaluate on Stress-Test Sets.**
        *   Run your final models (`Pixelis-RFT-Full` and `Pixelis-Online`) on these new, harder benchmark splits.
        *   Report the performance degradation compared to the clean version to quantify the tools' robustness.
    *   **Action 3 (Optional): Incorporate into Training.**
        *   Add a small fraction of these augmented samples into your SFT and RFT training data to improve the model's robustness from the start.
