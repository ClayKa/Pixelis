**Round 6: Establish Development Workflow and CI/CD Pipeline**

*   **Task 1: Implement Pre-Commit Hooks for Code Quality.**
    *   **Goal:** To automatically enforce a consistent code style and prevent common errors before code is ever committed to the repository.
    *   **Tooling:** Use the `pre-commit` framework.
    *   **Action:**
        1.  Create a `.pre-commit-config.yaml` file in the project root.
        2.  Configure it to run a suite of essential hooks on every `git commit`:
            *   **Formatting:** `black` and `isort` to auto-format code.
            *   **Linting & Import Sorting:** `ruff`. **Note: `ruff` is an extremely fast linter that can also handle import sorting, replacing the need for separate tools like `flake8` and `isort`. This choice simplifies configuration and significantly speeds up the pre-commit checks.**
            *   **Type Checking:** `mypy` to perform static type analysis.
        3.  Update the `README.md` and `docs/` with instructions for developers to install and use the pre-commit hooks.

*   **Task 2: Set Up a Continuous Integration (CI) Pipeline.**
    *   **Goal:** To automatically run all tests on every new commit or pull request, ensuring the integrity of the main branch is never compromised.
    *   **Tooling:** Use **GitHub Actions**.
    *   **File:** Create `.github/workflows/ci.yml`.
    *   **Action:** Configure the CI workflow to perform the following steps:
        1.  Check out the code.
        2.  Set up the Python environment and install dependencies. **Crucially, implement dependency caching using the `actions/cache` action.** The cache key should be based on a hash of the `requirements.txt` or `environment.yml` file. This ensures that dependencies are only re-installed when the requirements file changes, dramatically speeding up CI run times.
        3.  Run all pre-commit hooks to check for quality issues.
        4.  Execute the entire test suite using `pytest`.
        5.  Generate a test coverage report using `pytest-cov`.
        6.  The CI pipeline will fail if any of these steps fail, preventing broken code from being merged.
*   **[NEW] Future Optimization Note:**
        *   A note should be added to `docs/ARCHITECTURE.md` or a developer guide: "As the test suite grows, we should consider **parallelizing the test execution** across multiple CI jobs to maintain fast feedback cycles. This can be achieved using a matrix strategy in GitHub Actions."

*   **Task 3: Enforce a Test Coverage Threshold.**
    *   **Goal:** To ensure that the most critical parts of the codebase are well-tested.
    *   **Action1:**
        1.  Configure the CI pipeline to fail if the test coverage for the core modules (`core/modules/` and `core/engine/`) drops below **70%**. The `pytest-cov` configuration (e.g., in `pyproject.toml` or `.coveragerc`) **must be set up to omit files that do not require testing**, such as `__init__.py` files, configuration schema definitions (`core/config_schema.py`), and pure data structure files (`core/data_structures.py`). This ensures the coverage metric accurately reflects the testing of core business logic.
        2.  This creates a strong incentive for all new code to be accompanied by corresponding unit or integration tests.
    *   **Action2: Document the Code Review Mandate for Test Quality.**
        *   **File:** A new `docs/DEVELOPMENT_WORKFLOW.md` or within the `README.md`.
        *   **Action:** This document must explicitly state the project's policy on code review. It must emphasize that **test coverage is a necessary but not sufficient condition for merging a pull request.** Reviewers are required to not only review the implementation code but also the accompanying tests to ensure they are meaningful, contain relevant assertions, and cover important edge cases, rather than merely existing to increase the coverage percentage.