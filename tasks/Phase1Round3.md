**Round 3: Offline Reinforcement Learning (RL) with GRPO-Powered "Explore & Focus" Shaping**
*   **Remark:** We adopt the GRPO algorithm because it effectively mitigates the vanishing advantages problem, which can be encountered in standard PPO, by using selective sample replay. This leads to more stable learning.

*   **Task 1: Implement RL Mode and Core Infrastructure in the Unified Training Script.**
    *   **File:** `scripts/train.py`
    *   **Action:**
        1.  Add logic to handle `--mode rft`. Create a dedicated function `run_rft(config)`.
        2.  **Model Loading:** Inside `run_rft`, first load the base model. Ensure gradient checkpointing is also enabled here for the RL training phase. Then attach the SFT-trained LoRA adapters from `saved_models/sft_adapters/`.
        3.  **GRPO Integration:** Initialize the TRL `PPOTrainer`. If the library supports GRPO, enable it via configuration. If not, create a `CustomGRPOTrainer` class that inherits from `PPOTrainer` and overrides the loss calculation to implement GRPO's selective sample replay logic.
*   **Task 2: Implement a Performance-Aware Curiosity Reward Module.**
    *   **Goal:** To create an intrinsic reward mechanism that is not only effective at driving exploration but also computationally efficient.
    *   **File:** `core/modules/reward_shaping.py`
    *   **Action 1: Implement an Efficient `DynamicsModel`.**
        *   The `DynamicsModel(nn.Module)` class will be explicitly designed for efficiency. Instead of a large, complex network, it will be a **lightweight MLP** (Multi-Layer Perceptron).
        *   **Implement Low-Rank Adaptation (LoRA):** Apply LoRA adapters to the `DynamicsModel` itself. This is a key optimization. Since the dynamics model only needs to capture the *change* in state embeddings, a low-rank update structure is highly suitable and will dramatically reduce its parameter count and computational cost.
    *   **Action 2: Implement the `CuriosityDrivenRewardModule` Class.**
        *   This class will contain an instance of the efficient, LoRA-enabled `DynamicsModel`.
        *   Its `compute()` method will calculate `R_curiosity` based on the prediction error, as planned.
    *   **Action 3 (Optional): Implement Embedding Caching.**
        *   To further optimize, the `RewardOrchestrator` can maintain a small, in-memory LRU cache.
        *   Before computing the curiosity reward for a given state `s_t` and action `a_t`, it first checks the cache. If the resulting `s_{t+1}` embedding has been computed recently for the same `(s_t, a_t)` pair, it can reuse the cached result, avoiding a forward pass through the `DynamicsModel`.
*   **Task 3: Implement the Trajectory Coherence Reward Module.**
    *   **File:** `core/modules/reward_shaping.py`
    *   **Action:** Create a `TrajectoryCoherenceAnalyzer` class. Its `compute()` method will calculate `R_coherence` for each step in a trajectory based on cosine similarity of embeddings and will penalize repetitive actions.

*   **Task 4: Implement and Enhance the Central Reward Orchestrator.**
    *   **File:** `core/modules/reward_shaping.py`
    *   **Action 1:** Create the `RewardOrchestrator` Class Structure.
        *   Define a class named `RewardOrchestrator`.
        *   Its `__init__` constructor will:
            a. Initialize instances of `CuriosityDrivenRewardModule` and `TrajectoryCoherenceAnalyzer`.
            b. Accept reward weights (e.g., `w_curiosity`, `w_coherence`, `w_penalty`) as arguments, which are passed from the main training script after being read from the config file.
    *   **Action 2:** Implement the Core `calculate_total_reward` Method.
        *   Define the main method, `calculate_total_reward()`, which takes a full trajectory and the final answer's correctness as input.
        *   This method will orchestrate the reward calculation by:
            a. Calling the `coherence_module.compute()` to get the step-wise `R_coherence` rewards.
            b. Calling the `curiosity_module.compute()` to get the step-wise `R_curiosity` rewards.
            c. Calculating the final task reward `R_final` based on correctness.
            d. Calling an internal helper method (see Action 3) to calculate penalties `R_penalty`.
            e. Combining all these components into a final, dense reward tensor using the stored weights.
    *   **Action 3:** Implement a Tool Misuse Penalty System.
        *   Create a private helper method, `_calculate_penalties(trajectory)`.
        *   This method will iterate through the trajectory's action sequence and apply penalties for specific, logical violations related to the new visual operations:
            *   Penalty for incorrect tool usage: e.g., using `TRACK_OBJECT` on a static image.
            *   Penalty for incorrect action sequence: e.g., calling `GET_PROPERTIES` without a preceding `SEGMENT_OBJECT_AT` that provides a valid mask.
            *   Penalty for invalid arguments: e.g., providing an out-of-bounds coordinate to `SEGMENT_OBJECT_AT`.
        *   This enhancement makes the `RewardOrchestrator` "aware" of the new tools and their proper usage patterns, ensuring the agent learns to use them correctly.
    *   **Action 4:** Implement Reward Component Normalization.
        *   **Rationale:** Different reward components (`R_curiosity`, `R_coherence`) will naturally have different scales and variances. Without normalization, a component with a larger numerical range could dominate the learning process, even if its assigned weight is small.
        *   **Implementation:**
            *   The `RewardOrchestrator` will maintain a running estimate of the mean and standard deviation for each reward component (`R_curiosity`, `R_coherence`).
            *   Create a private method `_normalize_rewards(rewards_tensor, component_name)`.
            *   This method will apply z-score normalization to the incoming batch of rewards for a specific component: `normalized_reward = (reward - running_mean) / (running_std + epsilon)`.
            *   Inside the main `calculate_total_reward` method, after computing the raw rewards from each module, pass them through this normalization layer before applying the final weights (`w_curiosity`, `w_coherence`).
    *   **Action 5: Implement Comprehensive Unit Tests for All Reward Components.**
        *   **Goal:** To rigorously verify that each reward component and penalty functions exactly as designed, preventing unintended behaviors during the expensive RL training phase.
        *   **File:** `tests/modules/test_reward_shaping.py`
        *   **Action:** Create a dedicated test file for the reward shaping module. It **must** include, at a minimum, the following test cases:
            1.  **`test_coherence_penalizes_repetition`**: Construct a mock trajectory that contains identical, consecutive actions. Assert that the calculated `R_coherence` for the repeated step is negative.
            2.  **`test_penalty_for_tool_misuse`**: Construct a mock trajectory where a tool requiring a video input (e.g., `TRACK_OBJECT`) is called on a static image. Assert that the `_calculate_penalties` method returns a significant negative penalty value.
            3.  **`test_penalty_for_incorrect_sequence`**: Construct a trajectory where `GET_PROPERTIES` is called without a preceding `SEGMENT_OBJECT_AT`. Assert that a penalty is applied.
            4.  **`test_reward_normalization`**: Pass a batch of rewards with a known mean and variance to the normalization function. Assert that the output has a mean close to 0 and a standard deviation close to 1.


*   **Task 5: Implement the Main GRPO-Powered RL Training Loop.**
    *   **File:** `scripts/train.py` (within `run_rft`)
    *   **Action:**
        1.  Initialize the `RewardOrchestrator`, passing in the weights defined in `configs/training_params.yaml`.
        2.  Begin the main training loop. For each batch of prompts:
        3.  Use `ppo_trainer.generate()` to roll out trajectories from the current policy.
        4.  For each generated trajectory, use the `reward_orchestrator.calculate_total_reward()` to compute its dense reward scores.
        5.  Pass the trajectories and their associated rewards to `ppo_trainer.step()` to perform a GRPO-powered optimization step.
        6.  Implement comprehensive logging to `wandb`. In addition to tracking all reward components, GRPO filtering rate, and KL divergence, the logging **must** also include:
            *   **Raw and Normalized Reward Values:** Log both raw and normalized values for each reward component (e.g., `rewards/raw/curiosity` and `rewards/normalized/curiosity`).
            *   **Action Usage Distribution:** At each logging step, calculate the frequency of each visual operation (`SEGMENT_OBJECT_AT`, `READ_TEXT`, etc.) appearing in the generated batch of trajectories. Log this distribution to wandb (e.g., as a dictionary under `behavior/action_distribution`). **This enables the creation of a stacked bar chart in the WandB dashboard to visualize how the model's tool usage strategy evolves over the course of training.**
            *   Create plots in your wandb dashboard to visualize:
                a. The time-series of both raw and normalized rewards to observe the effect of normalization.
                b. The running mean and standard deviation of each reward component to ensure the normalization statistics are stable.
        7.  Upon completion, save the final, RL-tuned LoRA adapter weights to `saved_models/rft_adapters/`.