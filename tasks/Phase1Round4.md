**Round 4: Offline RL Training Execution, Monitoring, and Analysis**

*   **Task 1: Launch the Focused Offline RL Training.**
    *   **File:** `scripts/train.py`
    *   **Action:** Execute the main training script with the RL configuration. The command will be: `python scripts/train.py --config_path configs/rft_final.yaml --mode rft`.
    *   **Underlying Process:** This command triggers the `run_rft` function, which initializes the GRPO-powered `PPOTrainer` and the focused `RewardOrchestrator` (with Curiosity and Coherence modules) as defined in Round 3.
*   **Task 2: Execute the Core Training Loop.**
    *   **File:** `scripts/train.py` (inside `run_rft`)
    *   **Action:** The script enters a loop that performs the following steps iteratively:
        1.  **Sampling:** Call `ppo_trainer.generate()` to have the current policy generate a batch of reasoning trajectories in response to prompts from the training dataset.
        2.  **Scoring:** For each trajectory, call `reward_orchestrator.calculate_total_reward()` to obtain its dense, multi-component reward tensor.
        3.  **Optimization:** Pass the trajectories and their rewards to `ppo_trainer.step()` to execute one optimization step, updating the policy's LoRA weights. The underlying GRPO logic will ensure only informative samples are used.
*   **Task 3: Implement Comprehensive and Focused Monitoring.**
    *   **File:** `scripts/train.py` (integrated with `wandb` or `TensorBoard`)
    *   **Action:** Within the training loop, log the following critical metrics at each step to gain deep insights into the focused training dynamics:
        1.  **Reward Component Analysis:** Log the average value of `R_final`, `R_curiosity`, and `R_coherence` for each batch. This is crucial to see which reward is driving the behavior at different stages.
        2.  **Policy Stability Metrics:** Track the **KL Divergence** between the policy and the reference model to ensure the policy does not drift too far, too fast.
        3.  **GRPO-Specific Metrics:** Monitor the **GRPO Filtering Rate** (i.e., the percentage of samples in a batch that have non-negligible advantages and are used for updates). This indicates the "richness" of the learning signal.
        4.  **Behavioral Metrics:** Track the average length of generated trajectories and the **Rate of Pixel-space Reasoning (RaPR)** to see if the agent is actively using its visual operations.
        5.  **[ENHANCEMENT]** Implement a moving average tracker for key performance indicators (KPIs) that will be used as triggers for the automated curriculum. For example, track `avg_R_final_last_100_steps` and `success_rate_last_100_steps`.

*   **Task 4: Implement a Phased Reward Curriculum Strategy.**
    *   **Goal:** To replace the fixed, step-based curriculum with a more intelligent, adaptive system that introduces new reward components only when the model is ready.
    *   **File:** `configs/rft_final.yaml`
    *   **Action 1: Define a Trigger-Based Curriculum in the Configuration.**
        *   Modify the `reward_schedule` section to be based on performance metric triggers instead of fixed steps.
        ```yaml
        # In configs/rft_final.yaml
        reward_curriculum:
          stages:
            - name: "Phase1_Learn_Goal"
              weights: { w_final: 1.0, w_coherence: 0.0, w_curiosity: 0.0 }
              exit_condition: # Trigger to move to the next stage
                metric: "success_rate_last_100_steps"
                threshold: 0.70 # e.g., when success rate is consistently above 70%
            
            - name: "Phase2_Learn_Coherence"
              weights: { w_final: 1.0, w_coherence: 0.1, w_curiosity: 0.0 }
              exit_condition:
                metric: "coherence_score_improvement_stagnated" # A more advanced metric
                patience: 5 # e.g., if coherence score hasn't improved for 5 evaluation cycles
            
            - name: "Phase3_Learn_Curiosity"
              weights: { w_final: 1.0, w_coherence: 0.1, w_curiosity: 0.05 }
              # This is the final stage, no exit condition
        ```
    *   **File:** `scripts/train.py` (within `run_rft`)
    *   **Action 2: Implement the Curriculum Manager.**
        *   Create a small helper class or function, `CurriculumManager`, within the training script.
        *   At the end of each evaluation cycle, the training loop will:
            a. Pass the current monitored KPIs (e.g., success rate) to the `CurriculumManager`.
            b. The manager will check if the `exit_condition` for the current curriculum stage has been met. For advanced strategies like `regression_slope_over_n_evals`, the manager must implement robust logic. For example, it will maintain a history of the last `patience` scores for the specified metric, perform a simple linear regression over these points, and check if the calculated slope is below the `slope_threshold`. This prevents premature advancement due to random fluctuations in performance metrics.
            c. If the condition is met, it will advance to the next stage and return the new set of reward weights.
        *   The training loop then updates the `RewardOrchestrator` with these new weights for the subsequent training phase.
*   **Task 5: Systematic Evaluation and Checkpointing.**
    *   **File:** `scripts/train.py`
    *   **Action:**
        1.  Save a model checkpoint (the LoRA adapter weights) at the end of each stage defined in the reward curriculum. This will produce:
            *   `rft_adapters_base/` (only `R_final`)
            *   `rft_adapters_coherence/` (`R_final` + `R_coherence`)
            *   `rft_adapters_full/` (final model with all rewards)
        2.  Also, save a complete training state checkpoint at the end of each stage defined in the reward curriculum. This checkpoint **must** contain not only the LoRA adapter weights, but also the full state of the `PPOTrainer`, the `optimizer`, the `learning rate scheduler`, and crucially, the state of the `CurriculumManager` (including its current stage and performance history). **The primary goal is to ensure that training can be perfectly resumed from any checkpoint, preserving the exact curriculum stage and learning dynamics.**
        3.  These checkpoints are essential for the ablation studies planned in `Phase 3`.
        4.  After the full training is complete, perform a final evaluation on a held-out test set and log the results. This provides the headline performance metrics for the final model.
*   **Task 6: Post-Training Trajectory Analysis.**
    *   **File:** `scripts/analyze_trajectories.py` (a new analysis script)
    *   **Action:** After training, create a dedicated script to use this script for deep, qualitative analysis and visualization for the paper. This script will:
        1.  Load a final trained model.
        2.  Generate sample trajectories on interesting evaluation examples.
        3.  Create visualizations showing how the model's pathfinding differs when trained with vs. without the curiosity and coherence rewards. For example, show a trajectory from the base model that gets stuck in a loop, and a trajectory from the final model that explores efficiently. This provides powerful, intuitive evidence for the paper.
*   **Task 7: Develop an Interactive Training Monitor.**
    *   **Goal:** To create a real-time, interactive dashboard for intuitive monitoring and debugging of the RL training process, complementing the static logs from wandb.
    *   **File:** `scripts/launch_monitor.py` (a new script).
    *   **Action 1: Choose a Framework.**
        *   Select a suitable web framework like `Gradio` or `Streamlit` for its ease of use in creating interactive data science applications.
    *   **Action 1.5: Manage Optional Dependencies.**
        *   The libraries required for the monitor (`Gradio` or `Streamlit`) **must not** be added to the core `requirements.txt`.
        *   They will be listed in a separate file, `requirements.dev.txt` or `requirements.monitor.txt`.
        *   The `README.md` or developer documentation will provide separate instructions for installing these optional dependencies for users who wish to run the interactive monitor. This ensures the core training engine remains lean and free of unnecessary UI dependencies.
    *   **Action 2: Setup Data Communication.**
        *   The main training script (`scripts/train.py`) will be modified to periodically (e.g., every N steps) write a small JSON file or log entry containing the latest batch's key statistics (e.g., average reward breakdown, tool usage frequencies, etc.).
        *   The `launch_monitor.py` script will run as a separate process, read this data file, and update the dashboard components.
    *   **Action 3: Implement Dashboard Components.**
        *   The dashboard will contain several interactive plots and displays:
            *   **Reward Breakdown Pie Chart:** A pie chart showing the real-time contribution of `R_final`, `R_curiosity`, and `R_coherence` to the total reward.
            *   **Tool Usage Bar Chart:** A bar chart showing the frequency of each visual operation being used in the latest batch.
            *   **Key Metric Time-Series:** Live plots of KL divergence, GRPO filtering rate, and average trajectory length over the last few hundred steps.
            *   **Live Trajectory Viewer (Advanced):** A text box where you can see one or two example reasoning trajectories from the most recent batch, allowing you to "peek inside the model's brain" in real-time.

