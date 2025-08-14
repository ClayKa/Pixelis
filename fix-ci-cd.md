### **Action Plan: Stabilizing the CI/CD Environment**

**Objective:** To create a Continuous Integration environment that is reliable, efficient, and perfectly mirrors the local development setup, thereby eliminating environment-related test failures and enabling trustworthy automated validation.

**Lead:** [Assign to the engineer responsible for CI/CD and DevOps]

This plan will be executed in a dedicated feature branch (e.g., `fix/ci-environment`) and must be fully green before being merged to the main branch.

---

#### **Task 1: Transition to a GPU-Enabled, Conda-Based CI Workflow**

**Goal:** To solve all CUDA-related errors and environment mismatch issues at their root by using the correct runner and environment management tool.

**File to Modify:** `.github/workflows/ci.yml`

**Action 1.1: Replace the Default Runner with a GPU Runner.**
*   The current `runs-on: ubuntu-latest` is incorrect as it lacks a GPU. We will replace it with a third-party action that provides a GPU environment. The primary candidate is `gpu-actions/gpu-runner`.
*   The workflow trigger will be configured to run on every push to the `main` branch and on every pull request targeting `main`.

**Action 1.2: Replace `pip` with `conda` for Environment Creation.**
*   Remove the `actions/setup-python` and `pip install` steps.
*   Implement the `conda-incubator/setup-miniconda` action.
*   Configure this action to create the environment directly and strictly from our `environment.yml` file. This ensures that the CUDA toolkit version, cuDNN, and all Python package versions (including their specific builds) are identical to the development environment.

**Action 1.3: [If Necessary] Add System-Level Dependencies.**
*   Before the `conda` setup step, add a step to install any required system libraries that `conda` does not manage.
*   Example: `run: sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx`

**Revised `ci.yml` (Illustrative Snippet):**
```yaml
name: Continuous Integration

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    name: Run Tests on GPU
    runs-on: [self-hosted, linux, x64, gpu] # Or the syntax for the chosen GPU runner action
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@v3
        with:
          auto-update-conda: true
          python-version: '3.10'
          environment-file: environment.yml
          activate-environment: Pixelis

      - name: Install System Dependencies (if any)
        run: |
          sudo apt-get update
          # Add any required system libraries here, e.g.:
          # sudo apt-get install -y libgl1-mesa-glx

      - name: Create Conda Environment
        shell: bash -l {0}
        run: |
          conda env create -f environment.yml --name Pixelis
          conda activate Pixelis
          
      # Further steps will go here...
```

#### **Task 2: Optimize CI Efficiency with Caching**

**Goal:** To dramatically reduce CI run times by caching the Conda environment.

**File to Modify:** `.github/workflows/ci.yml`

**Action 2.1: Implement Conda Cache.**
*   Use the `actions/cache` action.
*   The cache key **must** be based on a hash of the `environment.yml` file. This ensures the cache is only invalidated when a dependency actually changes.

**Revised `ci.yml` (Illustrative Snippet with Cache):**
```yaml
# ... inside the 'test' job ...
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Cache Conda environment
        uses: actions/cache@v4
        with:
          path: ${{ env.CONDA }}/envs
          key: ${{ runner.os }}-conda-${{ hashFiles('environment.yml') }}
          restore-keys: |
            ${{ runner.os }}-conda-

      - name: Setup Miniconda and Create Environment
        # ... conda setup logic ...
        # The conda create command will now use the cache if available
```

#### **Task 3: Separate CI Jobs for Targeted Testing**

**Goal:** To create a more robust and efficient CI pipeline by separating different types of tests. This allows for faster feedback on non-GPU related changes.

**File to Modify:** `.github/workflows/ci.yml`

**Action 3.1: Create a CPU-Only Job for Static Analysis and Unit Tests.**
*   This job will run on the standard, fast `ubuntu-latest` runner.
*   It will perform:
    1.  **Code Quality Checks:** Run all `pre-commit` hooks (`black`, `ruff`, `mypy`).
    2.  **CPU-Only Unit Tests:** Run `pytest` but explicitly **exclude** tests marked as requiring a GPU. Use `pytest -m "not gpu"`.

**Action 3.2: Create a GPU-Only Job for Integration and Performance Tests.**
*   This job will run on the more expensive GPU runner.
*   It will depend on the successful completion of the CPU job (`needs: cpu_tests`).
*   It will perform:
    1.  **GPU-Specific Tests:** Run `pytest` and execute **only** the tests marked as requiring a GPU. Use `pytest -m "gpu"`. This will include the VRAM and latency tests.
    2.  **Integration Tests:** Run the full integration test suite, including `test_integration.py` and `test_async_communication.py`.

**Action 3.3: Mark GPU-Specific Tests in the Codebase.**
*   Go through the `tests/` directory. For any test function or class that requires a GPU, add the pytest marker:
    ```python
    import pytest

    @pytest.mark.gpu
    def test_vram_usage():
        # ... test logic that requires CUDA ...
    ```

#### **Task 4: Enable and Validate Coverage Reporting in CI**

**Goal:** To fix the disabled coverage reporting and ensure we have full visibility into our code's test coverage.

**Files to Modify:** `ci.yml`, `pyproject.toml` (or `.coveragerc`)

**Action 4.1: Configure `pytest-cov` for Multiprocessing.**
*   The root cause of coverage failure is likely its incompatibility with subprocesses.
*   Create a `.coveragerc` file in the project root.
*   Add the following configuration to ensure `pytest-cov` can track coverage in subprocesses:
    ```ini
    [run]
    parallel = true
    concurrency = multiprocessing

    [report]
    # ... any other reporting config ...
    ```
*   The `pytest` command in the CI workflow must be updated to include coverage and combine the results: `pytest --cov=core` followed by `coverage combine` and `coverage report`.

**Action 4.2: Re-enable Coverage in the CI Workflow.**
*   In the `ci.yml` file, add the steps to run `pytest` with the `--cov` flag and then fail the job if the coverage drops below the threshold defined in `Phase 0, Round 6`.

---

### **Final Instructions**

This plan is to be executed immediately. The primary objective is to get a **fully green, reliable, and efficient CI pipeline**. Once this is achieved, the two separate CI jobs (CPU and GPU) will serve as the gatekeepers for all future code merges. This stable foundation is non-negotiable and is the prerequisite for fixing any of the application-level test failures.