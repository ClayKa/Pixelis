### **Phase 3: Experiments, Evaluation, and Analysis**

**Round 0: Establish Rigorous Experimental Protocol**

*   **Task 1: Mandate Statistical Significance and Reproducibility.**
    *   **Goal:** To ensure that all reported experimental results are scientifically valid, credible, and reproducible, by adhering to the highest standards of empirical evaluation in machine learning.
    *   **Action:** This protocol shall be applied to all key comparative and ablation experiments outlined in the subsequent rounds of Phase 3.
*   **Task 2: Implement Multi-Seed Experimental Runs.**
    *   **Protocol:** For every model configuration that will be reported in the main results tables of the final paper (especially the ablation set from `Round 1, Task 1`), the **entire end-to-end training process (SFT and/or RFT) must be executed a minimum of three (3) times**.
    *   **Implementation:** Each of these three runs must be identical in its configuration, differing only in the **initial random seed** used for model weight initialization, data shuffling, and other stochastic processes.
    *   **File:** `scripts/run_experiments.sh` (a new shell script).
    *   **Action:** Create a master script that automates this process. It will loop three times, each time calling `python scripts/train.py` with the same configuration file but with a different, predefined random seed (e.g., `--seed 42`, `--seed 84`, `--seed 126`).
*   **Task 3: Enforce Reporting of Aggregated Results.**
    *   **Protocol:** All reported metrics in tables and plots must be presented as an aggregate of the multi-seed runs. The standard reporting format will be **`mean ± standard deviation`** (e.g., `84.3 ± 0.5`).
    *   **File:** `scripts/analyze_results.py` (a new analysis script).
    *   **Action:** Create a dedicated script that automatically collects the evaluation outputs from all seed runs for a given experiment. This script will calculate the mean and standard deviation for each metric and generate the final, properly formatted tables for the paper.
*   **Task 4: Perform Statistical Significance Testing.**
    *   **Protocol:** When comparing the primary hero model (`Pixelis-Online`) against the key baseline (`Pixel-Reasoner-Baseline`), a statistical test should be performed to confirm that the observed performance difference is not due to random chance.
    *   **File:** `scripts/analyze_results.py`
    *   **Action:** Implement a function to perform a **paired t-test or a bootstrap test** on the sets of results from the multi-seed runs of the two models. The resulting p-value should be reported in the paper (e.g., "The improvement of our model over the baseline is statistically significant with p < 0.01").