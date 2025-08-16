**Round 1: Comprehensive and Focused Ablation Studies**

*   **Task 1: Define a Clean and Powerful Comparison Set.**
    *   **Action1:** Create a separate configuration file in `configs/` for each of the following models to ensure fair and reproducible comparisons. This set is designed to surgically isolate the contribution of each innovation.
        1.  `Pixelis-SFT-Baseline`: The model after only Phase 1, Round 2 (SFT with curriculum). This is your starting point.
        2.  `Pixelis-RFT-Base`: The SFT model after being trained with RL using only the final task reward (`R_final`). This isolates the benefit of RL itself.
        3.  `Pixelis-RFT-Full`: The final offline model, trained with `R_final` + `R_coherence` + `R_curiosity`.
        4.  **`Pixel-Reasoner-Baseline`:** A faithful reimplementation of the original Pixel-Reasoner, trained only on `ZOOM-IN` and `SELECT-FRAME` tasks. This is your direct point of comparison.
        5.  **`Pixelis-Online` (The Hero Model):** The `RFT-Full` model with the complete Phase 2 online "Test-Time Evolution" engine enabled.
    *   **Action 2: Establish and Document the "Fair Comparison Protocol".**
        *   **Goal:** To proactively ensure and document the fairness of the comparison against the key baseline, addressing potential reviewer concerns.
        *   **File:** `docs/BENCHMARKS.md`.
        *   **Action:** A dedicated section titled "Baseline Comparison Fairness Protocol" **must** be created in the benchmarks document. This section will explicitly detail the measures taken to ensure a fair comparison with `Pixel-Reasoner-Baseline`, including:
            1.  **Identical Base Model:** Confirmation that the exact same pre-trained base model and tokenizer were used.
            2.  **Equivalent Computational Budget:** A statement confirming that the baseline was trained with a total computational budget (e.g., GPU hours) comparable to that of the `Pixelis-RFT-Full` model.
            3.  **Hyperparameter Tuning:** A description of the reasonable hyperparameter search conducted for the baseline to ensure it was performing optimally under its own framework.

*   **Task 2: Create a New, Challenging Evaluation Benchmark.**
    *   **File:** `scripts/evaluate.py`, Data preparation scripts.
    *   **Action:** Prepare two sets of evaluation benchmarks:
        1.  **Standard Benchmarks:** Use established benchmarks like V*QA, TallyQA, MM-Vet, MMMU, ScienceQA etc., to compare Pixelis with Pixel-Reasoner-Baseline on its original tasks. This demonstrates the superiority of your training methodology.
        2.  **Custom Capabilities Benchmark:** Create a new, challenging evaluation set by creating a held-out test split (e.g., 10-15%) from the newly synthesized data in Phase 1, Round 1. This benchmark will contain tasks that are impossible to solve without the new visual operations.
*   **Task 3: Implement Tool-Specific Evaluation Metrics.**
    *   **File:** `scripts/evaluate.py`
    *   **Action:** In addition to the general `Exploration Efficiency` and `Coherence Score`, add metrics to evaluate the performance of the new tools on your custom benchmark.
        1.  **Segmentation Accuracy (IoU):** For tasks involving `SEGMENT_OBJECT_AT`, measure the Intersection over Union (IoU) of the predicted mask vs. a ground-truth mask. **Boundary F1-score:** To specifically measure the accuracy of the segmented object's contour.
        2.  **OCR Accuracy (Edit Distance):** For tasks involving `READ_TEXT`, measure the character-level edit distance between the OCR result and the ground-truth text.
        3.  **Tracking Success Rate:** For `TRACK_OBJECT` tasks, measure the percentage of frames where the tracked object's bounding box successfully overlaps with the ground-truth path.
            *   **MOTA (Multiple Object Tracking Accuracy):** A composite metric that accounts for false positives, misses, and ID switches.
            *   **MOTP (Multiple Object Tracking Precision):** Measures the pure localization precision of the tracker.
            *   **ID Switches:** Counts how many times the tracker incorrectly swaps the identity of tracked objects, measuring tracking coherence.

*   **Task 4: Analyze the Sample Efficiency of the SFT Process.**
    *   **Goal:** To demonstrate how efficiently the model learns from the synthesized data, proving the effectiveness of the data quality and training methodology.
    *   **Action 1: Create Data Subsets.**
        *   From your final, filtered "gold-standard" SFT training dataset, create several subsets of varying sizes, for example: 10%, 25%, 50%, and 100%. Ensure the class/difficulty distribution is maintained across subsets.
    *   **Action 1.5: Create a Control Group with Random Sampling.**
        *   **Goal:** To isolate the benefit of the intelligent data curation and curriculum strategy.
        *   **Action:** In addition to the stratified subsets, create a parallel set of "control" subsets (10%, 25%, 50%, 100%) using **pure random sampling** from the unfiltered, un-scored synthesized dataset.
    *   **Action 2: Train Multiple SFT Models.**
        *   Using the exact same SFT training script (`scripts/train.py --mode sft`) and hyperparameters, train a separate `Pixelis-SFT-Baseline` model on each of the data subsets. This action will now be performed for both the stratified subsets and the random-sampled control subsets.
    *   **Action 3: Evaluate and Plot.**
        *   Evaluate each of the trained SFT models on a fixed, held-out validation set.
        *   Plot a 2D graph that shows two curves:
            1.  The performance curve for models trained on the "gold-standard" stratified data subsets.
            2.  The performance curve for models trained on the "control" random-sampled data subsets.
    *   **Hypothesis:** The curve for the gold-standard data will be significantly steeper and will saturate at a higher performance level than the control curve. This will provide powerful evidence that not only the data itself, but also the intelligent filtering, enrichment (e.g., trap samples), and curriculum strategies are critical for sample efficiency.
*   **Task 5: Execute and Analyze Ablation and Comparative Experiments.**
    *   **File:** `scripts/evaluate.py`
    *   **Action:**
        1.  Run all models (including `Pixel-Reasoner-Baseline`) on both standard benchmarks (like V*QA) AND your new custom benchmark.
        2.  **Create a key results table** that shows:
            *   Your `Pixelis-Online` model outperforms `Pixel-Reasoner-Baseline` on the standard benchmarks (showing your training method is superior).
            *   Your `Pixelis-Online` model achieves high scores on the new benchmark, while the `Pixel-Reasoner-Baseline` **fails completely** (as it lacks the necessary tools). This provides knockout evidence of your contribution.
        3.  Present detailed ablation results for the `Pixelis` model series on the new benchmark, showing how `R_curiosity` and `R_coherence` help in solving these more complex tasks.