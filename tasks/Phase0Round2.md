**Round 2: Establish Directory Structure**

*   **Task1:** Create the core project directory structure, explicitly separating logic (engine), components (modules), and configurations (configs). Then, configure .gitignore. Immediately populate the .gitignore file. It must include entries to ignore common Python artifacts (__pycache__/, *.pyc), environment files (.env), and crucially, the output directories for large files like saved_models/ and analysis_outputs/ to prevent committing large model weights or data to the repository.
*   **Example:**
    ```
    Pixelis/
    |
    |-- core/
    |   |-- __init__.py
    |   |-- engine/
    |   |   |-- __init__.py
    |   |   |-- inference_engine.py
    |   |   `-- update_worker.py
    |   `-- modules/
    |       |-- operations/
    |       |-- __init__.py
    |       |-- experience_buffer.py
    |       |-- reward_shaping.py
    |       |-- voting.py
    |       `-- dynamics_model.py
    |
    |-- configs/
    |   |-- lora_rank_config.json
    |   |-- model_arch.yaml
    |   `-- training_params.yaml
    |
    |-- scripts/
    |   |-- __init__.py
    |   |-- analyze_lora_ranks.py
    |   |-- generate_cota_data.py
    |   `-- train.py
    |
    |-- tests/
    |   |-- __init__.py
    |   |-- engine/
    |   |   `-- test_async_communication.py
    |   `-- modules/
    |       |-- test_experience_buffer.py
    |       `-- test_reward_shaping.py
    |-- docs/
    |   |-- ARCHITECTURE.md
    |   |-- RISK_MITIGATION_PLAN.md
    |   |-- BENCHMARKS.md
    |   |-- TROUBLESHOOTING.md
    |   `-- API_REFERENCE.md
    |
    |-- saved_models/
    |   `-- .gitkeep
    |
    |-- .gitignore
    |-- environment.yml
    |-- requirements.txt
    `-- README.md

    `core/`: The main source code package for the application.
        `engine/`: Contains high-level logic, process flows, and orchestration.
            `inference_engine.py`: Manages the primary predict-and-adapt online learning loop.
            `update_worker.py`: Handles asynchronous model parameter updates in a separate background process.
        `modules/`: Contains self-contained, reusable components (a functional library).
            `experience_buffer.py`: Encapsulates all logic for the experience replay buffer.
            `reward_shaping.py`: Contains all reward calculation classes (Curiosity, Coherence, Orchestrator).
            `voting.py`: Implements the temporal ensemble voting and confidence scoring logic.
            `dynamics_model.py`: Defines the auxiliary model required by the curiosity reward module.
    `configs/`: The single source of truth for all configurations and hyperparameters.
        `lora_rank_config.json`: Stores the dynamically calculated ranks for LoRA layers.
        `model_arch.yaml`: Defines the base model architecture and its loading parameters.
        `training_params.yaml`: Contains all hyperparameters for SFT/RFT, such as learning rates and reward weights.
    `scripts/`: Holds standalone scripts for performing specific tasks.
        `analyze_lora_ranks.py`: Script to perform SVD analysis on model weights.
        `generate_cota_data.py`: Script to synthesize training data trajectories.
        `train.py`: The main, unified script to launch all training modes (SFT, RFT).
    `tests/`: Contains all unit and integration tests. Its structure mirrors `core/`.
        `test_async_communication.py`: Tests the IPC between the main engine and the update worker.
        `test_experience_buffer.py`: Unit tests for the experience buffer module.
        `test_reward_shaping.py`: Unit tests for the reward calculation logic.
    `saved_models/`: Directory for storing output artifacts, such as trained LoRA adapter weights.
    `environment.yml`: Defines the Conda environment for project reproducibility.
    `requirements.txt`: Defines Python package dependencies for `pip`.
    `README.md`: Provides an overview of the project, setup instructions, and usage guide.
    ```
*   **Task2:** Implement the Visual Operation Registry.
    *   **Goal:** To create a central, decoupled system for managing and executing all visual operations (tools).
    *   **File:** `core/modules/operation_registry.py`
    *   **Action1: Create the Registry Class:** Define a singleton class named `VisualOperationRegistry`. This class will contain an internal dictionary, e.g., `self._operations = {}`.
    *   **Action2: Implement `register` Method:** Create a public method, `register(self, operation_name: str, operation_class)`, which will be used by individual tool plugins to add themselves to the `_operations` dictionary.
    *   **Action3: Implement `execute` Method:** Create the main execution method, `execute(self, operation_name: str, **kwargs)`. This method will:
        *   a. Look up the `operation_name` in the `_operations` dictionary.
        *   b. If found, instantiate the corresponding class and call its `run(**kwargs)` method.
        *   c. If not found, raise a `NotImplementedError` or handle the error gracefully.
    *   **Action4: Instantiate Global Object:** Create a single, global instance of the registry (e.g., `registry = VisualOperationRegistry()`) that can be imported and used by other parts of the system.

*   **Task3:** Implement Visual Operations as Self-Contained Plugins.
    *   **Goal:** To encapsulate the logic for each visual tool in its own independent module, making the system easy to extend.
    *   **File Location:** All individual operation files will reside in the `core/modules/operations/` directory.
    *   **Action1: Create a Base Class (Optional but Recommended):** In `core/modules/operations/base_operation.py`, create an abstract base class `BaseOperation` with an abstract method `run(self, **kwargs)`. All specific operations will inherit from this class to ensure a consistent interface.
    *   **Action2: Implement Each Operation:** For each tool (e.g., `SEGMENT_OBJECT_AT`), create its own file (e.g., `segment_object.py`). Inside this file:
        *   a. Define a class (e.g., `SegmentObjectOperation(BaseOperation)`) that contains all the logic for this tool.
        *   b. Implement the `run(self, **kwargs)` method, which takes the necessary arguments (e.g., `point`) and returns the result.
        *   c. **Register the Plugin:** At the bottom of the file, import the global registry instance and register the new operation class:
            ```python
            from ..operation_registry import registry
            # ... class definition ...
            registry.register('SEGMENT_OBJECT_AT', SegmentObjectOperation)
            ```
    *   **Action3:** Repeat this process for all other visual operations (`READ_TEXT`, `GET_PROPERTIES`, `TRACK_OBJECT`, etc.).

*   **Task 4:** Implement a Structured and Validated Configuration System.
    *   **Goal:** To establish a robust, error-proof system for managing all project configurations and experiments, ensuring consistency and reproducibility.
    *   **Action 1: Add Dependencies.** The project will adopt the `Hydra` and `OmegaConf` libraries. These will be added as dependencies to the `requirements.txt` and `environment.yml` files.
    *   **Action 2: Define Configuration Schemas.** A central Python file (e.g., `core/config_schema.py`) will be created to define the strict structure, expected data types, and default values for all configuration parameters. This will be done using Python's `dataclass` feature. Separate dataclasses will be defined for different configuration sections, such as `ModelConfig`, `TrainingConfig`, and `RewardConfig`. This file serves as the single source of truth for the project's entire configuration contract.
    *   **Action 3: Structure YAML Files According to Schemas.** All `.yaml` files within the `configs/` directory must be structured to exactly match the hierarchy and keys defined in the configuration schemas. This ensures consistency across all experiments.
    *   **Action 4: Integrate Hydra into the Main Training Script.** The main entry point for training, `scripts/train.py`, will be refactored to use `Hydra`. The script will be decorated with `@hydra.main`, pointing to the `configs/` directory and the schema definitions. This integration will provide two key benefits automatically:
        *   **Startup Validation:** Hydra will validate the loaded YAML config against the predefined schema at the moment the script is launched. Any mismatch, typo, or type error will cause an immediate and informative failure, preventing wasted computation.
        *   **Command-Line Overrides:** It will enable the ability to easily override any parameter from the command line for rapid experimentation, without needing to edit the YAML files.

*   **Task 5:** Define Core Data Structures with Validation.
    *   **Goal:** To ensure data consistency and type safety for all data objects passed between different modules and processes.
    *   **File:** Create a new file, `core/data_structures.py`.
    *   **Action1: Define Dataclasses:** In this file, use Python's `@dataclass` to strictly define the structure and types of the core data objects:
    *   **Action2: Experience:** Will contain fields like `image_features`, `question_text`, `trajectory`, `model_confidence`, `timestamp`, etc.
    *   **Action3: UpdateTask:** Will contain an `Experience` object, a `reward_tensor`, a `learning_rate`, and crucially, the `original_logits` from the policy before the update.
    *   **Action4: Generate JSON Schemas:** For an even higher level of validation, especially if you plan to save/load these objects, write a small utility script to automatically generate `jsonschema` definitions from these dataclasses.
    *   **Action5: Integrate:** All other parts of the system (engine, modules) must import these dataclasses and use them for type hinting and object creation, ensuring system-wide consistency.