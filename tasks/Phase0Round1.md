**Phase 0: Project Initialization and Setup**
**Round 1: Setup Environment and Codebase**

*   **Task1:** Initialize a new git repository (`Pixelis`) and clone the `tiger-ai-lab/pixel-reasoner` codebase as the foundational scaffold.
*   **Task2:** Systematically merge dependencies from all three source projects (`pixel-reasoner`, `reason-rft`, `ttrl`), using a tool like `pipdeptree` to analyze the dependency graph and resolve any version conflicts. Finally you should get an ‘requirements.txt’.
*   **Task3:** Create a dedicated conda environment with Python 3.10 (for stability) and install dependencies from the final `requirements.txt`.
*   **Task4:** Establish strict environment reproducibility by exporting the final locked environment to an `environment.yml` file to manage CUDA toolkit and all library versions precisely.