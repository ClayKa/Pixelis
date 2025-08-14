# Pixelis Build Roadmap
This project introduces **Pixelis**, a novel vision-language agent designed to reason directly within the pixel space of images and videos.

## Overview
Pixelis is a novel AI framework designed to pioneer the next generation of visual intelligence, enabling models to reason and interact with visual data at a granular level. It achieves this through:
- **Enhanced Pixel-Level Reasoning:** Executes complex, multi-step tasks by directly interacting with image and video content using a versatile toolkit of visual operations, such as SEGMENT_OBJECT_AT, READ_TEXT, and TRACK_OBJECT.
- **Self-Evolving Online Learning (TTRL):** Employs a Test-Time Reinforcement Learning architecture where the model continuously learns and adapts in a live environment. It improves its skills autonomously based on new interactions, guided by a memory of past experiences.
- **Focused Reinforcement Fine-Tuning (RFT):** Utilizes an advanced offline training phase where the model learns not just to be correct, but to be efficient and logical. A custom reward system incorporating task success, trajectory coherence, and exploration curiosity prepares the model for complex, real-world problem-solving.
- **Synthesized Multi-modal Datasets:** Is trained on a rich, diverse dataset of "Chain-of-Thought-Action" (CoTA) trajectories. This unique data, featuring positive examples, logical traps, and self-correction paths, teaches the model to solve complex visual and spatio-temporal reasoning tasks.

## Basic Information
**Base Models**: Qwen2.5-VL(7B);Qwen3(8B)
The script for training the model in ttrl and rft is not qwen2.5-VL 7B. Change it to qwen2.5-VL 7B and write a version of the script; and write another version of Qwen3 (8B)
**Code**: Code feasibility is more important than innovation. Prioritize ensuring the logic of the entire project is clear and that errors can be traced, rather than focusing on fixing errors.
**Datasets for synthesis**: SA1B, FineWeb, STARQA, PartImageNet, MathVista, Ego4D
**Datasets for evaluation**: MM-Vet, MMMU, ViRL39K


## Standard Workflow for Each Step
**Breakdown of the Steps**
**Planning**: Define the project's goals, requirements, and scope. This involves identifying the core problem to solve, defining functional and non-functional requirements, and creating a detailed timeline.

**Deeply Understanding the Problem**: This is a crucial step. Before writing any code, take the time to analyze the root cause of the issue, evaluate the pros and cons of existing solutions, and brainstorm several potential approaches. This phase is the foundation for ensuring the rest of your work is heading in the right direction.

**Problem Solving**: Choose the best solution and break it down into smaller, actionable tasks. This often involves designing the system architecture, algorithms, or data structures, which prepares you for the coding phase.

**Coding**: Translate your designed solution into actual code. During this stage, you should follow established coding standards, write clean and readable code, and add comments where necessary.

**Functional Review**: This step focuses on the functionality of your code. By running unit tests, integration tests, or manual checks, you verify that the code works as expected and successfully solves the problem.

**Code Quality Review**: This step focuses on the quality of the code. It involves checking that the code style conforms to team standards, identifying potential performance issues, ensuring the logic is clear and concise, and verifying it meets maintainability and security requirements.

**Commit**: Save the local changes to your version control system. You should always include a clear and meaningful commit message that explains the purpose and content of the change.

**Push to GitHub**: Synchronize the commits from your local repository to a remote repository (like GitHub). This allows other team members to see, collaborate on, and review your work.


## Listed Development Workflow

Important!!!: When finish one task or one round, or even one phase, replace the ⚪ with the ✅.

### Phase 0: Project Initialization and Setup ✅

- **Round 1: Setup Environment and Codebase** ✅
    - See: `/tasks/Phase0Round1.md`
    - **Task 001: Initialize a new git repository and clone the codebase** ✅
        - Initialize a new git repository (`Pixelis`) and clone the `tiger-ai-lab/pixel-reasoner` codebase to serve as the foundational scaffold for the project.
    - **Task 002: Systematically merge dependencies** ✅
        - Merge dependencies from all three source projects (`pixel-reasoner`, `reason-rft`, `ttrl`), using a tool like `pipdeptree` to analyze the dependency graph, resolve version conflicts, and generate a final `requirements.txt`.
    - **Task 003: Create a dedicated conda environment** ✅
        - Create a dedicated conda environment with Python 3.10 for stability and install all project dependencies from the final `requirements.txt` file.
    - **Task 004: Establish strict environment reproducibility** ✅
        - Export the final, locked software environment to an `environment.yml` file to precisely manage CUDA toolkit and all library versions, ensuring full reproducibility.

- **Round 2: Establish Directory Structure** ✅
    - See: `/tasks/Phase0Round2.md`
    - **Task 001: Create the core project directory structure** ✅
        - Create the core project directory structure, explicitly separating logic (`core/engine`, `core/modules`), configurations (`configs`), scripts, tests, and documentation to ensure a clean and scalable architecture.
    - **Task 002: Implement the Visual Operation Registry** ✅
        - Create a central, decoupled singleton system (`VisualOperationRegistry`) for managing and executing all visual operations, allowing for easy extension and maintenance.
    - **Task 003: Implement Visual Operations as Self-Contained Plugins** ✅
        - Encapsulate the logic for each visual tool (e.g., `SEGMENT_OBJECT_AT`, `READ_TEXT`) in its own independent module within `core/modules/operations/`, with each module registering itself to the central registry.
    - **Task 004: Implement a Structured and Validated Configuration System** ✅
        - Adopt `Hydra` and `OmegaConf` to establish a robust, error-proof system for managing all configurations, using Python `dataclasses` to define strict schemas for validation at startup.
    - **Task 005: Define Core Data Structures with Validation** ✅
        - Create a central `core/data_structures.py` file using `@dataclass` to strictly define the structure and types of core data objects like `Experience` and `UpdateTask`, ensuring data consistency system-wide.

- **Round 3: Modify Model Architecture** ✅
    - See: `/tasks/Phase0Round3.md`
    - **Task 001: Preliminary Full Fine-Tuning** ✅
        - On a small, representative subset of the training data, perform a brief, full-parameter fine-tuning run to produce a `W_finetuned` checkpoint for analysis.
    - **Task 002: SVD Analysis Script** ✅
        - Implement the script `scripts/analyze_lora_ranks.py` to perform Singular Value Decomposition (SVD) on the weight delta (`W_finetuned - W_pretrained`), using randomized SVD for efficiency.
    - **Task 003: Implement Robust Dynamic Rank Configuration** ✅
        - Generate a data-driven LoRA rank configuration by analyzing singular value distributions, applying heuristic constraints (bounding and smoothing), and storing enriched metadata for debugging.
    - **Task 004: Integration with PEFT** ✅
        - Implement the logic in the model definition files to dynamically construct the `LoraConfig` from the generated JSON file and wrap the base model with parameter-efficient LoRA layers.
    - **Task 005: Enhance Unit Testing with Performance Assertions** ✅
        - Add unit tests to verify not only the correct insertion of LoRA layers but also to assert that VRAM usage and inference latency remain below predefined, acceptable thresholds.

- **Round 4: Conduct Feasibility Assessment and Resource Planning** ✅
    - See: `/tasks/Phase0Round4.md`
    - **Task 001: Create the Computational Budget Document** ✅
        - Create a new living document, `COMPUTE_BUDGET.md`, to provide a realistic, high-level estimation of the computational resources required for the project.
  	- **Task 1.5: Conduct a Mandatory Micro-Benchmark Run.** ✅
    - **Task 002: Estimate SFT Computational Cost** ✅
        - In `COMPUTE_BUDGET.md`, estimate the GPU hours, VRAM, and training time required for a single Supervised Fine-Tuning (SFT) run.
    - **Task 003: Estimate SVD Analysis Cost** ✅
        - Estimate the computational cost required for the brief full fine-tuning run and the subsequent SVD analysis script.
    - **Task 004: Estimate RFT Computational Cost** ✅
        - Estimate the significantly higher computational cost for Reinforcement Fine-Tuning (RFT), accounting for trajectory generation at each step.
    - **Task 005: Estimate Online TTRL Simulation and Evaluation Cost** ✅
        - Estimate the resources required to validate the final online model through sustained "Domain Adaptation" and "Continual Learning" simulation tests.
    - **Task 006: Estimate Total Experimental Cost** ✅
        - In the budget summary, calculate the total estimated GPU hours for the entire project, including all training phases, online simulations, and multi-seed experimental runs.

- **Round 5: Establish Project-Wide Reproducibility and Artifact Management Protocol** ⚪
    - See: `/tasks/Phase0Round5.md`
    - **Task 001: Mandate Comprehensive Experiment Tracking and Artifact Versioning** ⚪
        - Standardize on using WandB (Weights & Biases), specifically its Artifacts feature, to version and link every input and output of the experimental process.
    - **Task 002: Implement Automatic Configuration and Environment Logging** ⚪
        - Modify training scripts to automatically save the exact Hydra configuration, `environment.yml`, and `requirements.txt` as WandB Artifacts at the start of every run.
    - **Task 003: Implement Versioning for All Key Inputs and Outputs** ⚪
        - Modify all data processing, training, and evaluation scripts to consume and produce versioned WandB Artifacts for datasets, model checkpoints, and evaluation results.
    - **Task 004: Document the Reproducibility Workflow** ⚪
        - Document the end-to-end artifact-driven workflow, explaining how any result can be traced back to the exact code, environment, configuration, dataset, and model weights that produced it.

- **Round 6: Establish Development Workflow and CI/CD Pipeline** ⚪
    - See: `/tasks/Phase0Round6.md`
    - **Task 001: Implement Pre-Commit Hooks for Code Quality** ⚪
        - Use the `pre-commit` framework to automatically run code formatters (`black`, `isort`), linters (`ruff`), and static type checkers (`mypy`) before any code is committed.
    - **Task 002: Set Up a Continuous Integration (CI) Pipeline** ⚪
        - Use GitHub Actions to create a CI workflow that automatically runs all pre-commit hooks and the entire test suite (`pytest`) on every new pull request to protect the main branch.
    - **Task 003: Enforce a Test Coverage Threshold** ⚪
        - Configure the CI pipeline to fail if the test coverage for the core application logic (`core/modules/` and `core/engine/`) drops below a 70% threshold.

### Phase 1: Offline Training ✅

- **Round 1: CoTA (Chain-of-Thought-Action) Data Synthesis and Enrichment** ✅
    - See: `/tasks/Phase1Round1.md`
    - **Task 001: Establish Data Provenance and Licensing Protocol** ✅
        - Create a centralized document, `docs/DATA_PROVENANCE.md`, to record all external datasets used, ensuring academic integrity and license compliance, and embed provenance metadata into each synthesized sample.
    - **Task 002: Create a script to generate structured CoTA data** ✅
        - Create the script `scripts/generate_cota_data.py` that formats all synthesized reasoning trajectories as structured JSON to eliminate complex parsing and ensure data integrity.
    - **Task 003: Implement data diversity strategies** ✅
        - When calling the generation API, strategically vary parameters like `temperature` and use diverse prompt templates to generate a wide variety of correct reasoning paths for the same task.
    - **Task 004: Augment the Dataset with Advanced Negative Samples** ✅
        - Synthesize not only "Outcome-Negative" samples (wrong final answer) but also advanced "Process-Negative" (Trap) samples that contain subtle perceptual or logical flaws in the reasoning path.
    - **Task 005: Implement a validation function for generated data** ✅
        - Create a validation function within the synthesis script to check the integrity of the returned JSON structure and ensure all generated actions are within a predefined, valid set.
    - **Task 006: Synthesize Training Data for New Visual Operations** ✅
        - Extend the data synthesis script to generate trajectories for new, complex spatial reasoning tasks like geometric comparison (`SEGMENT_OBJECT_AT` + `GET_PROPERTIES`) and spatio-temporal analysis (`TRACK_OBJECT`).
    - **Task 007: Synthesize Iterative Self-Correction Trajectories** ✅
        - Design a new data synthesis template that intentionally introduces an error early in a trajectory, then prompts the LLM to generate a "correctional" reasoning step before proceeding correctly.
    - **Task 008: Implement a Data Quality Scoring and Filtering Pipeline** ✅
        - Create a dedicated script (`scripts/filter_and_score_data.py`) to programmatically clean the synthesized dataset using heuristic filters and model-based quality scoring with consistency checks, ensuring high-fidelity training data.
    - **Task 009: Implement Data Strategy for Hard-Negative Mining** ✅
        - Tag advanced "trap" samples during synthesis and add a `sampling_weight` field to the pre-processing script, then modify the training data loader to use a `WeightedRandomSampler` to oversample these challenging examples.

- **Round 2: Supervised Fine-Tuning (SFT) with Enhanced Curriculum Learning** ✅
    - See: `/tasks/Phase1Round2.md`
    - **Task 001: Implement SFT Mode in the Unified Training Script** ✅
        - Modify the main `scripts/train.py` script to accept a `--mode sft` argument and encapsulate all SFT-specific logic within a dedicated `run_sft` function.
    - **Task 002: Implement Data Loading with Curriculum-Based Stratification** ✅
        - Create a pre-processing script to assign a difficulty score ("simple", "medium", "hard") to each data sample, and implement a `CurriculumDataset` to manage the staged introduction of this data.
    - **Task 003: Implement SFT Model Loading and Configuration** ✅
        - Implement the logic to load the base model, enable gradient checkpointing to reduce VRAM usage, and then wrap the model with the dynamically configured LoRA layers from the PEFT analysis.
    - **Task 004: Configure and Launch SFT Training** ✅
        - Implement a `CurriculumManager` and a custom `TrainerCallback` to manage the training process. The manager will automatically advance the curriculum to harder data but will also trigger a rollback if validation performance drops significantly.

- **Round 3: Offline Reinforcement Learning (RL) with GRPO-Powered "Explore & Focus" Shaping** ✅
    - See: `/tasks/Phase1Round3.md`
    - **Task 001: Implement RL Mode and Core Infrastructure in the Unified Training Script** ✅
        - Add logic to handle a `--mode rft` argument, loading the SFT-trained LoRA adapters and initializing the TRL `PPOTrainer` with GRPO enabled for more stable learning.
    - **Task 002: Implement a Performance-Aware Curiosity Reward Module** ✅
        - Create a computationally efficient curiosity module using a lightweight MLP for the `DynamicsModel`, further optimized with its own LoRA adapters to reduce parameter count.
    - **Task 003: Implement the Trajectory Coherence Reward Module** ✅
        - Create a `TrajectoryCoherenceAnalyzer` class that calculates a reward based on the cosine similarity of embeddings between steps and penalizes repetitive, unproductive actions.
    - **Task 004: Implement and Enhance the Central Reward Orchestrator** ✅
        - Create a `RewardOrchestrator` class that combines the final task reward with curiosity and coherence rewards, applies a tool misuse penalty system, and normalizes all reward components to prevent scale dominance.
    - **Task 005: Implement the Main GRPO-Powered RL Training Loop** ✅
        - Implement the main RL loop that generates trajectories, calculates the combined reward using the orchestrator, and performs an optimization step using the GRPO-powered trainer.
    - **Task 006: Implement Comprehensive Logging to wandb** ✅
        - Log all individual raw and normalized reward components, the GRPO filtering rate, and KL divergence to `wandb` to allow for detailed analysis of the learning dynamics.
    - **Task 007: Implement Reward Component Normalization** ✅
        - The `RewardOrchestrator` will maintain running statistics (mean, std dev) for each reward component and apply z-score normalization to ensure no single component numerically dominates the learning process.

- **Round 4: Offline RL Training Execution, Monitoring, and Analysis** ✅
    - See: `/tasks/Phase1Round4.md`
    - **Task 001: Launch the Focused Offline RL Training** ✅
        - Execute the main training script with the RFT configuration (`--mode rft`) to begin the reinforcement learning phase.
    - **Task 002: Execute the Core Training Loop** ✅
        - The script will enter the main iterative loop of sampling trajectories from the current policy, scoring them with the multi-component reward, and updating the policy's weights via a GRPO optimization step.
    - **Task 003: Implement Comprehensive and Focused Monitoring** ✅
        - Within the training loop, log critical metrics to `wandb`, including the breakdown of all reward components, KL divergence for policy stability, the GRPO filtering rate, and the Rate of Pixel-space Reasoning (RaPR).
    - **Task 004: Implement a Phased Reward Curriculum Strategy** ✅
        - Replace a fixed curriculum with an intelligent, adaptive system where new reward components (coherence, curiosity) are introduced based on performance metric triggers (e.g., success rate exceeding 70%) rather than fixed steps.
    - **Task 005: Systematic Evaluation and Checkpointing** ✅
        - Save a versioned model checkpoint at the end of each stage of the reward curriculum, enabling detailed ablation studies on the contribution of each reward component.
    - **Task 006: Post-Training Trajectory Analysis** ✅
        - Create a dedicated analysis script (`scripts/analyze_trajectories.py`) to generate and visualize sample trajectories, providing qualitative evidence of how the reward system improves the model's reasoning.
    - **Task 007: Develop an Interactive Training Monitor** ✅
        - Create a real-time, interactive dashboard using `Gradio` or `Streamlit` that visualizes key training dynamics, such as the reward breakdown and tool usage frequencies, for intuitive monitoring.

### Phase 2: Online Training (TTRL Evolution) ⚪

- **Round 1: Asynchronous Architecture** ✅
    - See: `/tasks/Phase2Round1.md`
    - **Task 001: Implement Two-Process Architecture** ✅
        - Establish the core `inference_engine.py` for the main user-facing process and `update_worker.py` for the background learning process.
    - **Task 002: Implement Inter-Process Communication** ✅
        - Use `torch.multiprocessing.Queue` to manage the flow of requests, responses, and learning update tasks between the two processes.
    - **Task 003: Implement High-Performance Tensor Transfer** ✅
        - Develop a stable and efficient strategy for transferring large tensors by moving them to CPU pinned memory and then placing them in `torch.multiprocessing.shared_memory`, passing only lightweight metadata through the queue to avoid serialization overhead.
    - **Task 004: Create Robustness and Communication Tests** ✅
        - Develop dedicated unit and integration tests in `tests/engine/` to ensure the asynchronous communication is reliable, can handle edge cases like queue timeouts, and that the shared memory tensor transfer is correct and free of memory leaks.

- **Round 2: Intelligent Experience Buffer Implementation** ✅
    - See: `/tasks/Phase2Round2.md`
    - **Task 001: Define the Buffer's Core Structure** ✅
        - In `core/modules/experience_buffer.py`, create the `ExperienceBuffer` class, using `collections.deque(maxlen=...)` as the underlying data structure to automatically manage its maximum size.
    - **Task 002: Implement Multi-Factor Priority Calculation with value tracking** ✅
        - Implement a priority score for experiences based on a weighted sum of initial uncertainty and reward, and track their long-term value by monitoring their success rate upon retrieval.
    - **Task 003: Implement Hybrid k-NN Retrieval** ✅
        - Create a hybrid embedding for each experience by taking a weighted average of its visual and text features, allowing retrieval of neighbors that are similar in both visual context and user intent.
    - **Task 004: Integrate k-NN Index for Neighbor Retrieval** ✅
        - Integrate the `faiss-gpu` library to maintain an index synchronized with the experience embeddings, implementing efficient `add_to_index` and `search_index` methods.
    - **Task 005: Integrate Hybrid k-NN Index with Strong Consistency Guarantees** ✅
        - Implement a robust, crash-proof persistence system using a **Write-Ahead Log (WAL)** and a "blue-green" index deployment strategy to ensure data integrity, 100% read availability, and safe recovery.
    - **Task 006: Develop Comprehensive Buffer Unit Tests** ✅
        - Create a dedicated test suite in `tests/modules/test_experience_buffer.py` to validate all public methods, including priority sampling, k-NN search, concurrency handling, and the crash recovery mechanism.

- **Round 3: Core Inference and Gated Learning Mechanisms** ✅
    - See: `/tasks/Phase2Round3.md`
    - **Task 001: Implement the Temporal Ensemble Logic** ✅
        - Orchestrate the core inference sequence: get the model's initial prediction, call `ExperienceBuffer.search_index()` to retrieve k-NN neighbors, and pass the data to the voting module.
    - **Task 002: Implement Configurable Voting Strategies** ✅
        - Enhance the voting module to return a structured `VotingResult` object that contains not only the final answer but also a detailed `provenance` dictionary, creating a full audit trail for each decision.
    - **Task 003: Implement the Confidence Gating Mechanism** ✅
        - Implement a mechanism in the inference engine to trigger a learning update only if the confidence score from the temporal ensemble voting module exceeds a configurable threshold.
    - **Task 004: Implement a Proportional and Bounded Learning Rate Strategy** ✅
        - Develop a continuous and adaptive learning rate policy where the LR is proportional to the model's error (`1.0 - confidence_score`) but is safely clipped within a pre-defined `min/max` range to ensure stability.
    - **Task 005: Implement a Human-in-the-Loop (HIL) Safety Valve for Initial Rollout** ✅
        - Add an optional HIL mode that, when enabled, queues a fraction of potential updates for human expert review ("Approve" / "Reject") before they are applied, ensuring an extra layer of safety during initial deployment.

- **Round 4: Focused Reward Calculation and Asynchronous Updates** ✅
    - See: `/tasks/Phase2Round4.md`
    - **Task 001: Integrate the Focused Reward Orchestrator** ✅
        - In the online `inference_engine`, import and use the same `RewardOrchestrator` from the offline phase to calculate the multi-component reward for potential updates.
    - **Task 002: Compute Online Rewards Based on Pseudo-Labels** ✅
        - Use the high-confidence consensus answer from the voting module as a high-quality pseudo-label to calculate the final task reward, `R_final`.
    - **Task 003: Structure and Enqueue the Update Task** ✅
        - Define an `UpdateTask` dataclass to package the experience, the multi-component reward tensor, the determined learning rate, and the original logits (for KL calculation), and push it to the `update_queue`.
    - **Task 004: Implement a Conservative and Stable Model Update Worker** ✅
        - Implement a "Three-Tiered Safety System": a self-regulating KL-divergence penalty (Behavioral Guardrail), gradient clipping (Magnitude Guardrail), and EMA smoothing with decoupled synchronization (Temporal Guardrail).
    - **Task 005: Log Update Contribution Metrics for Post-Hoc Analysis** ✅
        - Log detailed, structured data for each online update, including the experience ID, reward tensor, and KL divergence, to enable deeper offline analysis of what drives performance changes.

- **Round 5: Main Integration, Observability, and Bootstrapping** ✅
    - See: `/tasks/Phase2Round5.md`
    - **Task 001: Build the Main `infer_and_adapt()` Orchestration Function** ✅
        - Tie all previous tasks together into a single, cohesive function that represents one full pass through the online evolution loop, from receiving a request to potentially enqueuing an update task.
    - **Task 002: Implement the Cold Start Bootstrapping Strategy** ✅
        - Develop a "conservative mode" that is active when the experience buffer is not yet mature. In this mode, the system will collect experiences to build its memory but will bypass learning updates.
    - **Task 003: Integrate Comprehensive Monitoring with Automated Alerting** ✅
        - Track and log key system health indicators (e.g., update rate, IPC queue sizes, mean KL divergence) to `wandb`, and implement an automated alerting module for critical issues.
    - **Task 004: Conduct End-to-End System Testing** ✅
        - Create a script (`scripts/run_online_simulation.py`) to serve as a configurable engine for validating the functional correctness and short-term stability of the entire online system.
    - **Task 005: Implement and Automate a Long-Running Stability and Stress Test** ✅
        - Create a dedicated CI workflow (`ci-long-running.yml`) that runs the online system for an extended period (e.g., 8 hours) under sustained load, automatically asserting against memory leaks, unbounded queues, and other signs of instability.
	- **Task 6: Design and Implement a Worker Process Supervisor for Automatic Restart.** ✅

- **Round 6: Implement Security, Privacy, and Compliance Protocols** ✅
    - See: `/tasks/Phase2Round6.md`
    - **Task 001: Create a Central Security and Privacy Policy Document** ✅
        - Establish a formal policy document, `docs/SECURITY_AND_PRIVACY.md`, to serve as the single source of truth for all data handling, user privacy, and system security decisions.
    - **Task 002: Define and Implement the Data Handling Policy for Online Learning** ✅
        - Ensure that raw user inputs are not stored persistently and implement a PII redaction module to process and anonymize all data before it is stored in the `ExperienceBuffer`.
    - **Task 003: Enforce a "Read-Only" Policy for the Public Demonstrator** ✅
        - Implement a configuration flag that completely bypasses the entire learning and update pipeline for the public-facing demo, preventing malicious updates and data contamination.
    - **Task 004: Define Data Retention and Deletion Policies** ✅
        - Implement an automated data pruning task in the experience buffer to enforce a maximum data retention period (e.g., 90 days), with the `timestamp` field in the `Experience` dataclass.
    - **Task 005: Implement Audit Trails** ✅
        - Configure the `update_worker` to maintain a separate, append-only log file (`update_audit.log`) that records key, non-sensitive metadata for every successful model update, providing a clear trail for security reviews.

### Phase 3: Experiments, Evaluation, and Analysis ⚪

- **Round 0: Establish Rigorous Experimental Protocol** ✅
    - See: `/tasks/Phase3Round0.md`
    - **Task 001: Mandate Statistical Significance and Reproducibility** ✅
        - Formally establish a protocol that all key reported experimental results must be scientifically valid, credible, and reproducible by adhering to the highest empirical standards.
    - **Task 002: Implement Multi-Seed Experimental Runs** ✅
        - For every key model configuration, the entire end-to-end training process must be executed a minimum of three times with different random seeds, managed by an automated script.
    - **Task 003: Enforce Reporting of Aggregated Results** ✅
        - All reported metrics in tables and plots must be presented as an aggregate of the multi-seed runs, in the standard format of `mean ± standard deviation`.
    - **Task 004: (Optional but Recommended) Perform Statistical Significance Testing** ✅
        - When comparing primary models against baselines, perform a statistical test (e.g., paired t-test) on the multi-seed results to confirm that observed improvements are statistically significant.

- **Round 1: Comprehensive and Focused Ablation Studies** ⚪
    - See: `/tasks/Phase3Round1.md`
    - **Task 001: Define a Clean and Powerful Comparison Set** ⚪
        - Create a set of version-controlled configuration files for each model in the ablation study, including baselines, SFT-only, RFT-base, RFT-full, and the final online model.
    - **Task 002: Create a New, Challenging Evaluation Benchmark** ⚪
        - Create a new, held-out "Custom Capabilities Benchmark" from the synthesized data that contains tasks impossible to solve without the new visual operations, alongside standard benchmarks.
    - **Task 003: Implement Tool-Specific Evaluation Metrics** ⚪
        - Enhance the evaluation script to measure the performance of new tools on the custom benchmark, including Segmentation IoU/Boundary F1-score, OCR Edit Distance, and MOTA/MOTP for tracking.
    - **Task 004: Analyze the Sample Efficiency of the SFT Process** ⚪
        - Train multiple SFT models on incremental subsets of the training data (10%, 25%, 50%, 100%) and plot the performance curve to provide powerful evidence of sample efficiency.
    - **Task 005: Execute and Analyze Ablation and Comparative Experiments** ⚪
        - Run all models on all benchmarks and create a key results table that demonstrates the superiority over baselines and provides knockout evidence of the new tools' contributions.

- **Round 2: Testing for Robustness, Efficiency, and Continual Learning** ⚪
    - See: `/tasks/Phase3Round2.md`
    - **Task 001: Test Continual Learning and Domain Adaptation** ⚪
        - Evaluate the `Pixelis-Online` model's adaptation speed on new, unseen domains and its resistance to catastrophic forgetting by sequentially exposing it to different tasks.
    - **Task 002: Profile Efficiency and Latency** ⚪
        - Use `torch.profiler` to measure the latency overhead of key online components (e.g., k-NN search) and report the final P99 latency and memory usage to demonstrate real-world viability.
    - **Task 003: Test Robustness to Noisy Data** ⚪
        - Inject noisy data into the online stream and monitor the confidence score and update rate to verify that the confidence gating mechanism successfully prevents learning from corrupted signals.
    - **Task 004: Conduct Hyperparameter Sensitivity Analysis** ⚪
        - Systematically vary the most critical hyperparameters (reward weights, confidence threshold) and plot the impact on performance to identify a robust operating range for the system.
    - **Task 005: Conduct Tool-Specific Stress Tests** ⚪
        - Create augmented stress-test datasets with challenging visual conditions (occlusion, blur, distortion) to quantify the performance degradation and robustness of individual visual operations.

- **Round 3: Human Evaluation of Reasoning Quality** ⚪
    - See: `/tasks/Phase3Round3.md`
    - **Task 001: Define the Evaluation Scope and Hypotheses** ⚪
        - State clear, testable hypotheses for the human evaluation, focusing on whether the new reward components lead to reasoning that is perceived as more coherent and intelligent.
    - **Task 002: Design the Human Evaluation Interface and Protocol** ⚪
        - Build a simple and unbiased UI using `Gradio` or `Streamlit` that presents annotators with blind, side-by-side comparisons of reasoning trajectories from different models.
    - **Task 003: Plan the Data Sampling and Annotation Process** ⚪
        - Randomly sample a diverse set of 100-300 questions for evaluation, with each question being annotated by three different human judges to ensure reliability.
    - **Task 004: Analyze Results and Report Inter-Annotator Agreement** ⚪
        - Collect all ratings, calculate a statistical measure of Inter-Annotator Agreement (e.g., Fleiss' Kappa), and use appropriate statistical tests to determine if observed differences are significant.

- **Round 4: Inference Acceleration & Optimization** ⚪
    - See: `/tasks/Phase3Round4.md`
    - **Task 001: Profile and Identify Bottlenecks** ⚪
        - Use `torch.profiler` to get a detailed performance breakdown of the entire inference pipeline, identifying the most computationally expensive components as targets for optimization.
    - **Task 002: Apply Standard Optimizations** ⚪
        - Accelerate the model using standard, state-of-the-art techniques including `torch.compile`, INT8 post-training quantization (PTQ), and enabling Flash Attention 2.
    - **Task 003: Implement Service-Level Optimizations** ⚪
        - For a high-throughput service architecture, implement features like dynamic batching and multi-level caching (LRU cache) to improve overall throughput and latency.
    - **Task 004: Implement Task-Specific Optimizations** ⚪
        - Experiment with optimizations specific to the online loop, such as using an approximate k-NN index or caching reward computations for revisited states.
    - **Task 005: (Optional) Export to a Dedicated Inference Engine** ⚪
        - For peak performance in a production environment, explore exporting the final, optimized inference graph to a specialized engine like ONNX Runtime or NVIDIA TensorRT.

- **Round 5: Final Analysis, Reporting, and Packaging** ⚪
    - See: `/tasks/Phase3Round5.md`
    - **Task 001: Conduct In-Depth Analysis of Logged Metrics** ⚪
        - Perform a deep dive into the `wandb` logs from all experiments, creating plots that correlate reward signals with behavior and tell a compelling story about the learning process.
    - **Task 002: Perform Qualitative Case Studies** ⚪
        - Select compelling examples and create visualizations comparing the reasoning trajectories of different models to provide powerful, intuitive evidence of the framework's improvements.
    - **Task 003: Perform Systematic Error Mode Analysis** ⚪
        - Use a combination of automated clustering on failure case embeddings and manual interpretation to discover, classify, and report on the system's common error patterns.
    - **Task 004: Create an Interactive Public Demonstrator** ⚪
        - Build and deploy a user-friendly public demo (e.g., on Hugging Face Spaces) that features a side-by-side comparison view, allowing the community to interact with the models and visualize their reasoning.
    - **Task 005: Create Comprehensive Documentation** ⚪
        - Write and finalize all project documentation, including a detailed `ARCHITECTURE.md`, a `BENCHMARKS.md` with all key results, a helpful `TROUBLESHOOTING.md`, and a professionally generated API reference.
    - **Task 006: Package for Release with Mandated Artifact Management** ⚪
        - Create a minimal, end-to-end "reproducibility kit" with a tiny dataset and pre-trained adapters, providing simple, copy-paste commands to allow other researchers to easily verify core results on consumer-grade hardware.