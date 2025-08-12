**Round 4: Conduct Feasibility Assessment and Resource Planning**

*   **Task 1: Create the Computational Budget Document.**
    *   **Goal:** To provide a realistic, high-level estimation of the computational resources (GPU hours, VRAM, storage) required for the project, ensuring feasibility and guiding resource allocation.
    *   **File:** Create a new document in the project root: `COMPUTE_BUDGET.md`.
    *   **Action:** Populate this document with preliminary, order-of-magnitude estimates for each major computational stage. The document should be treated as a "living document" and updated as experiments provide more accurate numbers.

*   **Task 1.5: Conduct a Mandatory Micro-Benchmark Run.**
Goal: To move from rough estimations to data-driven projections by measuring real-world performance on the target hardware.
Action: Before any large-scale experiments are launched, a short, mandatory benchmark run must be completed. This involves:
Running the SFT script (scripts/train.py --mode sft) for a small, fixed number of steps (e.g., 100 steps) on a single GPU.
Using torch.profiler and nvidia-smi to precisely measure:
Average step time (wall-clock time).
Peak VRAM usage.
Data loader bottleneck analysis (I/O wait time vs. GPU compute time).
The entire COMPUTE_BUDGET.md document must then be updated using these measured figures to extrapolate more accurate total costs. This transforms the document from an "estimation" to a "projection".

*   **Task 2: Estimate SFT Computational Cost.**
    *   **Action:** In `COMPUTE_BUDGET.md`, create a section for Supervised Fine-Tuning (SFT) and estimate the following, assuming a 7B parameter base model:
        *   **Target GPU:** A100 (40GB or 80GB VRAM) or RTX 4090/3090 (24GB VRAM). Note the target hardware.
        *   **VRAM per GPU:** Estimate VRAM usage with gradient checkpointing enabled. (e.g., `~30GB`).
        *   **Batch Size:** Propose a target per-GPU batch size and total effective batch size with gradient accumulation. (e.g., `Per-GPU Batch Size: 8`, `Gradient Accumulation Steps: 4`, `Total Batch Size: 32`).
        *   **Dataset Size:** State the estimated number of samples in the final filtered SFT dataset. (e.g., `~10,000` trajectories).
        *   **Training Time Estimation:**
            *   Estimate training time for a **single epoch**.
            *   Calculate the total estimated GPU hours for a single SFT run, providing a range that includes a contingency buffer.
            *   Propose the total number of epochs. (e.g., `3 epochs`). (e.g., `Time per Epoch: ~10 hours on a single A100`, `Total SFT Time: ~30 GPU hours`).
    		*   **Acknowledge Evaluation Overhead:** **The curriculum learning strategy involves an additional, immediate re-evaluation step upon each difficulty advancement attempt. This will add a minor overhead to the total training time. We are including a small buffer in our estimation to account for this.**
            *   Example: "Based on the micro-benchmark, we project a single SFT run to require ~30 GPU hours. To account for potential I/O bottlenecks and debugging time, the planning budget for one run is set at 40 GPU hours."

*   **Task 3: Estimate SVD Analysis Cost.**
    *   **Action:** In `COMPUTE_BUDGET.md`, create a section for SVD Analysis.
        *   **Note:** Acknowledge that a brief full-parameter fine-tuning is required first. Estimate this cost (e.g., `~2 hours on a single A100`).
        *   **Analysis Cost:** Estimate the time required for the `analyze_lora_ranks.py` script to run, noting that the randomized SVD option is crucial for managing memory and time. (e.g., `~1 hour on a single A100`).
        *   **Note:** Acknowledge that this is a one-time cost per major model architecture and is not repeated for every seed run.
        	*   Estimate the cost for the preliminary full fine-tuning (e.g., ~2-3 hours on a single A100).
        	*   Estimate the script execution time (e.g., ~1 hour on a single A100).

*   **Task 4: Estimate RFT Computational Cost.**
    *   **Action:** In `COMPUTE_BUDGET.md`, create a section for Reinforcement Fine-Tuning (RFT).
        *   **Acknowledge Higher Cost:** Note that RFT is significantly more expensive than SFT due to trajectory generation (inference) at each step.
        *   **Generation vs. Update:** Estimate the number of generated responses per prompt (e.g., `8 responses per prompt`).
        *   **Training Step Estimation:** State the estimated number of total RFT update steps. (e.g., `~5,000 steps`).
        *   **Total Training Time:** Provide a scaled estimate as a range. (e.g., "We estimate RFT to be 5-8x more computationally expensive per sample than SFT. With a dataset of `~15,000` prompts, a full RFT run is estimated to take **~150-200 GPU hours** on a single A100.").
        *   Example: "We project RFT to be 5-8x more computationally expensive than SFT. A full RFT run is projected to take ~150-200 GPU hours, with a planning budget of 220 GPU hours per run."


*   **Task 5: Estimate Online TTRL Simulation and Evaluation Cost.**
    *   **Goal:** To estimate the resources required to validate the performance and stability of the final `Pixelis-Online` model.
    *   **File:** `COMPUTE_BUDGET.md`.
    *   **Action:** Create a new section in the document for "Phase 2 - Online Evolution Simulation".
        1.  **Acknowledge Architectural Overhead:** Note that this phase requires running a multi-process application (`inference_engine` + `update_worker`) and will have a higher CPU and RAM footprint compared to offline training scripts.
        2.  **Estimate "Domain Adaptation" Test Cost:**
            *   **Protocol:** As defined in `Phase 3`, this involves running the model on a new, unseen data stream for a sustained period.
            *   **Estimated Duration:** Propose a reasonable simulation duration, e.g., "The model will be run for **12 hours** on a continuous stream of new data."
            *   **Required GPUs:** This test requires at least 2 GPUs (one for the `inference_engine`, one for the `update_worker`).
            *   **Cost:** `12 hours * 2 GPUs = 24 GPU hours` per run.
        3.  **Estimate "Continual Learning" Test Cost:**
            *   **Protocol:** As defined in `Phase 3`, this involves sequentially exposing the model to different tasks.
            *   **Estimated Duration:** e.g., "The model will be exposed to each of the 3 tasks for **8 hours**, for a total of **24 hours** of online learning."
            *   **Cost:** `24 hours * 2 GPUs = 48 GPU hours` per run.
        4.  **Total Online Test Cost:** Sum the costs, e.g., `(24 + 48) GPU hours = 72 GPU hours` for a single seed run.
        5.  * 	**Estimate Storage Cost:**
            	* Acknowledge Storage Needs: Note that online evolution generates significant artifacts, including experience buffer snapshots, WAL files, and detailed logs.
                * Estimate per Cycle: Estimate the storage footprint for a full experimental cycle (e.g., one full run of all planned online tests).
                * Example: "We estimate that each full online experimental cycle will generate ~500 GB of data artifacts. This must be factored into our cloud storage or local disk planning."

*   **Task 6: Estimate Total Experimental Cost.**
    *   **Action:** In the summary section, the total budget calculation will now explicitly include the online simulation cost.
        *   **Total Budget Formula:** The total budget will be calculated using a more precise, structured formula that distinguishes between one-time costs and multi-seed costs.
        *   **Total_Budget** =
    (One_Time_Setup_Costs)  // Includes SVD Analysis
    + (Core_Experiment_Cost * Num_Seeds) // SFT + RFT runs for main comparison
    + (Ablation_Studies_Cost) // Ablations are typically run with a single seed
    + (Online_Simulation_Cost * Num_Seeds) // Online tests for the final model
    + (Contingency_Budget)
        *   **Example:** "The total projected experimental cost is calculated based on the formula above. We will add a 15% contingency budget to the total calculated cost. This buffer is allocated to cover unforeseen challenges, additional debugging runs, and supplementary experiments that may arise during the research process. The final estimated total budget is in the range of ... A100 GPU hours."