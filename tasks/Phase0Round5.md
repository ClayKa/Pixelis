**Round 5: Establish Project-Wide Reproducibility and Artifact Management Protocol**

*   **Task 1: Mandate Comprehensive Experiment Tracking and Artifact Versioning.**
    *   **Goal:** To move beyond simple metric logging and establish a robust system for versioning and linking every single input and output of the experimental process, creating an unbreakable chain of reproducibility.
    *   **Tooling:** The project will standardize on using **WandB (Weights & Biases)** not just for logging, but specifically for its **Artifacts** feature.
*   **Task 2: Implement Automatic Configuration and Environment Logging.**
    *   **File:** `scripts/train.py` (and other key scripts).
    *   **Action:**
        1.  At the beginning of every experimental run (`wandb.init`), the script **must** automatically save key files as WandB Artifacts.
        2.  **Configuration Artifact:** The exact Hydra configuration object (`cfg`) used for the run must be saved.
        3.  **Environment Artifact:** The `environment.yml` and `requirements.txt` files must be saved to capture the exact software environment.
        4.  **Code Artifact:** The current git commit hash will be automatically logged by `wandb`, linking the run to the exact version of the source code.
*   **Task 3: Implement Versioning for All Key Inputs and Outputs.**
    *   **File:** `scripts/filter_and_score_data.py`, `scripts/train.py`, `scripts/evaluate.py`.
    *   **Action:** Modify all scripts that produce or consume critical data to use WandB Artifacts for versioning.
        1.  **Datasets as Input Artifacts:** The `scripts/train.py` script will be modified to accept a WandB Artifact address for the training dataset. It will log this address, ensuring that every training run is explicitly linked to a specific, versioned dataset. The `scripts/filter_and_score_data.py` will be responsible for creating and logging this dataset artifact in the first place.
        2.  **Model Checkpoints as Output Artifacts:** After every training run (or at each checkpoint), the `scripts/train.py` script **must** save the LoRA adapter weights as a new WandB Artifact, versioned and linked to the run that produced it.
        3.  **Evaluation Results as Output Artifacts:** The `scripts/evaluate.py` script will take a model artifact address as input and log its output (e.g., results tables in `.json` or `.csv` format) as a final, versioned artifact linked to both the model and the dataset it was evaluated on.
*   **Task 4: Document the Reproducibility Workflow.**
    *   **File:** Create a new document: `docs/REPRODUCIBILITY.md`.
    *   **Action 1: Document the End-to-End Workflow.** Explain how any result reported in the paper can be traced back to its WandB run, which in turn contains links to the exact code, environment, configuration, dataset, and model weights used to generate it.
    *   **Action 2: Define and Mandate Artifact Naming Conventions.**
        *   **Goal:** To maintain a clean, organized, and searchable Artifact repository in Weights & Biases.
        *   **Action:** The `REPRODUCIBILITY.md` document **must** define a strict naming convention for all project artifacts. All scripts must adhere to this convention when logging artifacts. The convention is as follows:
            *   **Datasets:** `dataset-<name>-<version>` (e.g., `dataset-cota_sft-v1`, `dataset-cota_rft-v1`). The `<name>` should be descriptive of the data's purpose, and `<version>` should be incremented for each significant change.
            *   **Models:** `model-<run_id>` (e.g., `model-2x3y5z7q`). The model artifact should be named using the unique ID of the WandB run that produced it. This creates an unambiguous link.
            *   **Evaluation Results:** `eval-<model_artifact_name>-on-<dataset_artifact_name>` (e.g., `eval-model-2x3y5z7q-on-dataset-custom_benchmark-v2`). This name explicitly records which model version was evaluated on which dataset version.
    *   **Action 3: Include a Workflow Visualization.**
        *   **Goal:** To provide an intuitive, high-level overview of the entire artifact-driven process.
        *   **Action:** The `REPRODUCIBILITY.md` document **must** include a simple flowchart diagram illustrating the chain of reproducibility. This diagram will visually represent the dependencies between scripts and artifacts.
        *   **Example Diagram Flow:**
            ```mermaid
            graph TD
                A["scripts/generate_cota_data.py"] -->|produces| B(Artifact: dataset-cota-sft-v1);
                B -->|is consumed by| C["scripts/train.py"];
                C -->|produces| D(Artifact: model-2x3y5z7q);
                B -->|is also consumed by| E["scripts/evaluate.py"];
                D -->|is consumed by| E;
                E -->|produces| F(Artifact: eval-model-2x3y5z7q-on-dataset-cota-sft-v1);
            ```
            *(Note: This Mermaid syntax can be directly embedded in Markdown files on platforms like GitHub to render the diagram).*
