Of course. Here are the precise `pytest` commands to run each test file individually in verbose (`-v`) mode.

This is an excellent strategy for focused, file-by-file test execution.

---
### **Pytest Commands to Run Each Test File Individually**

Here is the complete list of commands. You can copy and paste these into your terminal one by one.

1.  **For `tests/engine/test_async_communication.py`**:
    ```bash
    pytest -v tests/engine/test_async_communication.py
    ```

2.  **For `tests/engine/test_inference_engine.py`**:
    ```bash
    pytest -v tests/engine/test_inference_engine.py
    ```

3.  **For `tests/engine/test_ipc.py`**:
    ```bash
    pytest -v tests/engine/test_ipc.py
    ```

4.  **For `tests/engine/test_update_worker.py`**:
    ```bash
    pytest -v tests/engine/test_update_worker.py
    ```

5.  **For `tests/modules/test_experience_buffer.py`**:
    ```bash
    pytest -v tests/modules/test_experience_buffer.py
    ```

6.  **For `tests/modules/test_model_init.py`**:
    ```bash
    pytest -v tests/modules/test_model_init.py
    ```

7.  **For `tests/modules/test_voting.py`**:
    ```bash
    pytest -v tests/modules/test_voting.py
    ```

8.  **For `tests/test_basic.py`**:
    ```bash
    pytest -v tests/test_basic.py
    ```

9.  **For `tests/test_experimental_protocol.py`**:
    ```bash
    pytest -v tests/test_experimental_protocol.py
    ```

10. **For `tests/test_integration.py`**:
    ```bash
    pytest -v tests/test_integration.py
    ```

11. **For `tests/test_rft_training.py`**:
    ```bash
    pytest -v tests/test_rft_training.py
    ```

12. **For `tests/test_sft_curriculum.py`**:
    ```bash
    pytest -v tests/test_sft_curriculum.py
    ```

13. **For `tests/unit/test_artifact_manager.py`**:
    ```bash
    pytest -v tests/unit/test_artifact_manager.py
    ```

---
### **How to Use These Commands**

*   **-v (verbose)**: This flag tells `pytest` to print the full name of each test function as it runs, along with a clear `PASSED` or `FAILED` status. This is much more informative than the default dot (`.`) representation.
*   **Focused Execution**: Running one file at a time helps you concentrate your debugging efforts and get faster feedback on your changes without waiting for the entire suite to complete.
*   **Workflow**: You can use these commands as you work through the coverage report. For example, if you decide to improve the coverage of `core/engine/inference_engine.py`, you would add new tests to `tests/engine/test_inference_engine.py` and then repeatedly run command #2 until you are satisfied with the result.