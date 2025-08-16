**Round 2: Supervised Fine-Tuning (SFT) with Enhanced Curriculum Learning**

*   **Task 1: Implement SFT Mode in the Unified Training Script.**
    *   **File:** `scripts/train.py`
    *   **Action:** Modify the script's main function to accept a `--mode sft` argument. Create a dedicated function `run_sft(config)` that will be called when this mode is active. This function will encapsulate all SFT-specific logic.

*   **Task 2: Implement Data Loading with Curriculum-Based Stratification.**
    *   **Goal:** To manage the introduction of training data of increasing difficulty in a way that is robust to training instability.
    *   **File:** `scripts/train.py` (within `run_sft`)
    *   **Action:**
        1.  Create a helper script (`scripts/preprocess_data.py`) to run once before training. This script will load the raw CoTA data, calculate a **composite difficulty score** for each sample (based on trajectory length, action complexity, etc.), and save the data with a new `difficulty` field ("simple", "medium", "hard").
        2.  In `scripts/train.py`, implement a `CurriculumDataset` class that inherits from `torch.utils.data.Dataset`.
        3.  The `CurriculumDataset`'s `__init__` will load the pre-processed data and start with only the "simple" samples. It will have a method `advance_curriculum()` to progressively add "medium" and then "hard" samples to its active data pool.
        4.  Create the `CurriculumManager` Class. Create a new helper class, `CurriculumManager`, within the training script. This manager will be responsible for the state of the curriculum (e.g., current difficulty stage: "simple") and the logic for advancing or rolling back. It will track the validation performance history (e.g., a list of recent validation scores).
*   **Task 3: Implement SFT Model Loading and Configuration.**
    *   **File:** `scripts/train.py` (within `run_sft`)
    *   **Action:**
        1.  Load the base model (e.g., Qwen2.5-VL) using the path specified in `configs/model_arch.yaml`.
        2.  Enable Gradient Checkpointing: When loading the model or configuring the TrainingArguments, explicitly enable gradient checkpointing. This can typically be done by calling `model.gradient_checkpointing_enable()` or setting a `gradient_checkpointing=True` flag. This will significantly reduce VRAM usage at the cost of a minor increase in training time.
        3.  Load the dynamic LoRA rank configuration from `configs/lora_rank_config.json`.
        4.  Use `peft.get_peft_model` to wrap the base model with the dynamically configured LoRA layers. This prepared model is ready for SFT.

*   **Task 4: Configure and Launch SFT Training.**
    *   **File:** `scripts/train.py` (within `run_sft`), configs/training_params.yaml.
    *   **Action 1: Initialize Trainer and Curriculum Manager.**
        *   Initialize the Hugging Face `Trainer` and the new `CurriculumManager`.
    *   **Action 2: Implement a Custom `TrainerCallback`.**
        *   Create a custom callback by inheriting from `transformers.TrainerCallback`. This is the most elegant way to inject custom logic into the training loop without modifying it directly.
        *   This callback will be triggered at the end of each evaluation step (`on_evaluate`).
    *   **Action 3: Implement the Advance-and-Validate Logic within the Callback.**
        *   Inside the `on_evaluate` method of your custom callback:
            1.  **Check for Advancement Trigger:** First, check if it's time to attempt a curriculum advancement (e.g., based on the current step or epoch, as defined in the config).
            2.  **Record Pre-Advance Performance:** If it is time, store the current validation score as `perf_before_advance`.
            3.  **Advance the Curriculum:** Call `CurriculumDataset.advance_curriculum()` to introduce the next difficulty level (e.g., "medium").
            4.  **Trigger Immediate Re-evaluation:** Force the `Trainer` to run another evaluation immediately on the new, harder data mix. Get the new score `perf_after_advance`.
            5.  **Call the Manager to Decide:** Pass both `perf_before_advance` and `perf_after_advance` to the `CurriculumManager`.
    *   **Action 4: Implement the Automatic Failure Rollback Logic in the Manager.**
        *   The `CurriculumManager.decide_and_update(...)` method will contain the core rollback logic:
            1.  **Calculate Performance Drop:** Calculate `delta = perf_after_advance - perf_before_advance`.
            2.  **Check for Catastrophic Drop:** If delta is below a sharp negative threshold, a rollback is triggered. This threshold must be a configurable parameter loaded from configs/training_params.yaml (e.g., curriculum.rollback_threshold: -0.05), not a hardcoded value.
            3.  **Execute Rollback:**
                a. The manager instructs the `CurriculumDataset` to revert to its previous state (e.g., remove the "medium" data).
                b. The manager modifies the training plan, for instance, by doubling the number of steps before the next advancement attempt is made (i.e., "reducing the difficulty step").
            4.  Log the Event to Console and WandB: Log a clear warning message to the console. Crucially, the CurriculumManager must also log a custom event to Weights & Biases, e.g., wandb.log({"curriculum/event": "Rolled back from Medium"}).
    *   **Action 5: Implement Comprehensive Curriculum State Logging.**
        *   **Goal:** To create a clear, visual representation of the curriculum's progression over time for easy analysis.
        *   **File:** `scripts/train.py` (within the custom callback).
        *   **Action:** At every step where `wandb.log` is called, the custom callback **must** also log the current curriculum stage. The stage will be represented numerically (e.g., simple=0, medium=1, hard=2).
        *   **Example log call:** `wandb.log({"curriculum/stage": 1, ...other_metrics})`.
        *   **Outcome:** This enables the creation of a dashboard in the WandB UI that plots the curriculum stage against the training steps, providing an intuitive visualization of how and when the difficulty was increased or rolled back.