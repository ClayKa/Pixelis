Your setup (Windows with WSL2 and an RTX 4090) is excellent for this kind of work. All PyTorch and CUDA operations will happen within the WSL2 environment, so the process is standard.

---

### **Action Plan: Priority One (P1) - Core Algorithm Fixes**

This is the most critical task. The errors here are causing numerous knock-on (cascading) failures throughout the test suite. Fixing these will have the biggest impact.

#### **Objective 1: Fix Symptom #1 - `RuntimeError: Tensor Device Mismatch`**

**Diagnosis:**
This error, `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`, occurs when a PyTorch operation (like `torch.cat`) is given tensors that live on different hardware. In our case, one tensor is on the GPU (`cuda:0`) and another is on the CPU (`cpu`). The tracebacks point to the reward calculation modules.

This typically happens when model parameters are on the GPU, but input data created on-the-fly in a test (like action embeddings) remains on the CPU by default.

**Step-by-Step Solution:**

1.  **Open the Target File:** Navigate to and open `core/modules/reward_shaping.py`.

2.  **Locate the `forward` method** inside the `CuriosityRewardModule` class. This is where the error originates.

    ```python
    # Inside core/modules/reward_shaping.py -> CuriosityRewardModule class
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ ... docstring ... """
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        # Forward model: predict next state
        # --- THIS IS THE LINE CAUSING THE ERROR ---
        state_action = torch.cat([state_feat, action], dim=-1) 
        # ... rest of the function
    ```

3.  **Implement a Robust Device-Syncing Fix:** The best practice is to determine the correct device from the model's parameters (which are authoritative) and ensure all inputs are moved to that device before any operations are performed.

    *   **Modify the `forward` method** as follows. The changes are commented.

    ```python
    # Inside core/modules/reward_shaping.py -> CuriosityRewardModule class

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ ... docstring ... """
        
        # --- FIX STEP 1: DETERMINE THE TARGET DEVICE ---
        # Get the target device from the model itself. This is the most reliable way.
        # We can pick any parameter, e.g., from the feature_encoder.
        target_device = next(self.feature_encoder.parameters()).device

        # --- FIX STEP 2: MOVE ALL INPUT TENSORS TO THE TARGET DEVICE ---
        # It's safer to move all tensors, even if some might already be there.
        state = state.to(target_device)
        action = action.to(target_device)
        next_state = next_state.to(target_device)

        # Now, all subsequent operations are guaranteed to be on the same device.
        
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        # Forward model: predict next state
        state_action = torch.cat([state_feat, action], dim=-1)
        # ... rest of the function
    ```4.  **Repeat for Other Modules if Necessary:** The test logs show this error also occurs in `scripts/train_rft.py`. Examine the `LightweightDynamicsModel`'s `forward` method in that file and apply the exact same fix: determine the target device from a model parameter and move all input tensors to that device at the beginning of the method.

#### **Objective 2: Fix Symptom #2 - `RuntimeError: Tensor Shape Mismatch`**

**Diagnosis:**
The error `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 512x256)` is a clear indicator of a shape mismatch during a matrix multiplication, which is the core operation of a `torch.nn.Linear` layer. The logs show this happens inside the `update_worker` when `_process_update` calls `self.model(...)`. The test model's first layer (`fc1`) expects an input with 512 features, but it's receiving a tensor with 256 features.

**Step-by-Step Solution:**

This requires debugging, as the cause is less obvious. We will use the `pdb` debugger inside the WSL2 terminal.

1.  **Open the Target File:** Navigate to and open `core/engine/update_worker.py`.

2.  **Set a Breakpoint:** Locate the `_process_update` method. Just before the line where the model is called, insert the Python debugger breakpoint.

    ```python
    # Inside core/engine/update_worker.py -> UpdateWorker class -> _process_update method

    def _process_update(self, task: UpdateTask):
        try:
            # ... (code for reconstructing tensor from shared memory might be here) ...

            # --- DEBUGGING STEP: SET BREAKPOINT HERE ---
            import pdb; pdb.set_trace()

            # The error happens on the next line
            outputs = self.model(
                image_features=task.experience.image_features,
                # ... other arguments
            )
            # ...
    ```

3.  **Run the Failing Test with Debugger Support:** In your WSL2 terminal, run one of the specific tests that fails with this error. The `-s` flag is important to allow the debugger to interact with your terminal.

    ```bash
    pytest -k "test_process_update_with_shared_memory" -s
    ```
    *   `-k` isolates the specific test.
    *   `-s` disables output capturing so you can interact with `pdb`.

4.  **Debug the Code:** The test will run and then pause at your `pdb.set_trace()` line, giving you a `(Pdb)` prompt. Now, investigate the tensor shape:

    *   At the `(Pdb)` prompt, type:
        ```
        p task.experience.image_features.shape
        ```
        (You can use `p` as a shorthand for `print`).
    *   **Hypothesis:** You will likely see `torch.Size([1, 256])`. This confirms the tensor has the wrong shape *before* being passed to the model.
    *   Now, you need to trace back where `task.experience.image_features` came from. Is it being reconstructed from shared memory in this method? If so, inspect the `shm_info` metadata that was used for reconstruction. The error is likely in the reconstruction logic or in the metadata itself.
    *   Also check the model's expected shape:
        ```
        p self.model.fc1.in_features
        ```
        This should print `512`.

5.  **Identify and Fix the Root Cause:** Based on your debugging, the cause is likely one of these:
    *   **Case A: Incorrect Reconstruction Logic:** The code that reconstructs the tensor from shared memory is using the wrong shape information.
    *   **Case B: Incorrect Metadata in Test:** The test that creates the `UpdateTask` is putting incorrect shape information into the `shm_info` metadata.
    *   **Case C: Incorrect Data in Test:** The test is simply creating a tensor with the wrong shape to begin with. The logs for `test_worker_queue_processing` show `torch.randn(1, 512)`, which is correct, but the error still happens. This strongly suggests the problem is in the `update_worker`'s handling of the data, not the test's creation of it.

    **The most probable fix is in the `_process_update` method itself.** Find the code block that handles shared memory and ensure it correctly uses the `shape` from `task.experience.metadata['shm_info']` when creating the tensor view from the shared memory buffer.