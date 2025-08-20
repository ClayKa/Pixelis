Of course. Here is the highly detailed, execution-focused engineering plan without the verification sections.

---

### **Engineering Action Plan: Hardening the Pixelis Codebase for Robust Experimentation**

This document outlines the mandatory engineering tasks required to mitigate risks related to reproducibility, training stability, and long-term maintainability before commencing large-scale experiments.

#### **Action Item 1: Enforce Full Determinism (Highest Priority)**

*   **Goal:** To eliminate hidden sources of randomness and ensure that experiments with the same configuration and seed are statistically reproducible.

*   **Implementation Steps:**

    1.  **Create a Centralized Reproducibility Utility Module:**
        *   Navigate to the `core/` directory and create a new subdirectory named `utils/`.
        *   Inside `core/utils/`, create a new file named `reproducibility.py`.
        *   Add an `__init__.py` file to both `core/utils/` and `core/` if they don't exist, to make them importable packages.

    2.  **Implement the Global Seeding Function:**
        *   In `core/utils/reproducibility.py`, define a function `set_global_seed(seed: int)`.
        *   Inside this function, add the following lines of code to set the seed for all relevant libraries:
            ```python
            import os
            import random
            import numpy as np
            import torch

            def set_global_seed(seed: int):
                """Sets the random seed for all relevant libraries to ensure reproducibility."""
                os.environ['PYTHONHASHSEED'] = str(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                # If using CUDA
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed) # Important for multi-GPU setups
            ```

    3.  **Implement the Deterministic Mode Function:**
        *   In the same `core/utils/reproducibility.py` file, define a second function `enable_deterministic_mode()`.
        *   Inside this function, add the code to configure PyTorch and cuDNN for deterministic, non-benchmark operations. This is critical for ensuring identical results between runs.
            ```python
            import torch

            def enable_deterministic_mode():
                """
                Enforces deterministic behavior in PyTorch/cuDNN.
                Note: This can have a performance cost.
                """
                if torch.cuda.is_available():
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            ```

    4.  **Integrate Seeding and Determinism into Entry-Point Scripts:**
        *   Open the main training script, `scripts/train.py`.
        *   At the top of the file, import the newly created functions: `from core.utils.reproducibility import set_global_seed, enable_deterministic_mode`.
        *   Locate the main function, decorated with `@hydra.main`.
        *   As the very first step inside this function, add the logic to read from the Hydra config (`cfg`) and call the functions.
            ```python
            # In scripts/train.py
            @hydra.main(config_path="../configs", config_name="training_params", version_base=None)
            def main(cfg: DictConfig):
                # --- Reproducibility Setup ---
                if "seed" in cfg.training:
                    set_global_seed(cfg.training.seed)
                
                if cfg.training.get("deterministic_mode", False):
                    enable_deterministic_mode()
                # -----------------------------

                # ... rest of the main function ...
            ```
        *   Repeat this integration step for all other key entry-point scripts, such as `scripts/evaluate.py` and `scripts/analyze_lora_ranks.py`.

    5.  **Configure DataLoader for Deterministic Worker Initialization:**
        *   Open `core/utils/reproducibility.py` again.
        *   Add a new function `seed_worker(worker_id: int)` which will be passed to the `DataLoader`.
            ```python
            import torch
            import numpy as np
            import random

            def seed_worker(worker_id: int):
                """
                Initializes each DataLoader worker with a unique but predictable seed.
                This is essential for reproducible data loading and augmentation.
                """
                # The worker seed is derived from the main process's initial seed.
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            ```
        *   Now, in any part of your code where you instantiate a `torch.utils.data.DataLoader` (likely within `scripts/train.py`), import `seed_worker`.
        *   When creating the `DataLoader` instance, pass this function to the `worker_init_fn` argument. You will also need to create a `torch.Generator` and pass it to the `generator` argument for full reproducibility.
            ```python
            # In scripts/train.py or your data module
            from torch.utils.data import DataLoader
            from core.utils.reproducibility import seed_worker

            # Create a generator object and seed it for reproducible shuffling/sampling
            g = torch.Generator()
            g.manual_seed(cfg.training.seed)

            data_loader = DataLoader(
                your_dataset,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                worker_init_fn=seed_worker,
                generator=g
            )
            ```

---

#### **Action Item 2: Harden the Curriculum Management System (High Priority)**

*   **Goal:** To make the curriculum advancement logic robust against noisy evaluation metrics, preventing training instability.

*   **Implementation Steps:**

    1.  **Enhance `CurriculumManager` State with History Tracking:**
        *   Open `scripts/train.py` and locate the `CurriculumManager` class.
        *   In its `__init__` method, initialize a dictionary of deques to store the history of evaluation metrics. The size of the deque should be configurable.
            ```python
            from collections import deque

            class CurriculumManager:
                def __init__(self, config):
                    self.config = config
                    self.metric_history = {
                        metric_name: deque(maxlen=config.smoothing_window_size)
                        for metric_name in self._get_tracked_metrics()
                    }
                    self.cooldown_counter = 0
                    # ... other initializations ...
            ```

    2.  **Implement Metric Smoothing Logic:**
        *   Within the `CurriculumManager` class, create a private method `_get_smoothed_metric(self, metric_name: str) -> float`.
        *   This method will access `self.metric_history`, and if there is enough data, it will calculate and return a smoothed value (e.g., Simple Moving Average).
            ```python
            import numpy as np

            def _get_smoothed_metric(self, metric_name: str):
                history = self.metric_history.get(metric_name)
                if not history or len(history) < self.config.smoothing_window_size:
                    return None # Not enough data to compute a smoothed value
                return np.mean(history)
            ```

    3.  **Refactor Decision Logic to Use Smoothed Metrics and Patience:**
        *   Modify the main decision-making method (e.g., `check_for_advancement`).
        *   This method will now first update its history with the latest raw score.
        *   Then, it will call `_get_smoothed_metric` to get the smoothed value.
        *   The logic will check if the smoothed value has been above the threshold for `patience_cycles` consecutive times.
            ```python
            # Inside CurriculumManager
            def check_for_advancement(self, latest_metrics):
                # Update history first
                for name, value in latest_metrics.items():
                    if name in self.metric_history:
                        self.metric_history[name].append(value)
                
                # Check cooldown
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1
                    return False # Still in cooldown

                # ... logic to get current stage's exit condition ...
                smoothed_value = self._get_smoothed_metric(exit_condition.metric)
                
                if smoothed_value is None or smoothed_value < exit_condition.threshold:
                    self.patience_counter = 0 # Reset patience if condition is not met
                    return False
                
                # If condition is met, increment patience counter
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.patience_cycles:
                    # Trigger advancement and reset patience
                    self.patience_counter = 0
                    self.cooldown_counter = self.config.cooldown_cycles # Start cooldown
                    return True
                
                return False
            ```

    4.  **Integrate Cooldown Mechanism:**
        *   As shown in the snippet above, after any successful advancement or a rollback decision, set `self.cooldown_counter` to the configured value from `self.config.cooldown_cycles`.
        *   The check at the beginning of the decision logic ensures no further changes happen during the cooldown period.

    5.  **Update Configuration Files:**
        *   Open `configs/training_params.yaml`.
        *   Under the `curriculum` section, add the new parameters for controlling this hardened logic.
            ```yaml
            # In configs/training_params.yaml
            curriculum:
              # ... existing curriculum stages ...
              rollback_threshold: -0.05
              smoothing_window_size: 3
              patience_cycles: 2
              cooldown_cycles: 3
            ```
        *   Open `core/config_schema.py` and update the corresponding dataclass to include these new fields with their types and default values.

---

#### **Action Item 3: Enhance Debuggability with Distributed Tracing (Long-Term/Optional)**

*   **Goal:** To enable in-depth debugging by linking all logs related to a single user request.

*   **Implementation Steps:**

    1.  **Create a Context Utility Module:**
        *   In the `core/utils/` directory, create a new file named `context.py`.
        *   In this file, define the `ContextVar` that will hold the trace ID.
            ```python
            # In core/utils/context.py
            from contextvars import ContextVar
            from typing import Optional

            # The variable will hold a string trace_id, or None if not set.
            trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
            ```

    2.  **Integrate Trace ID Generation and Context Setting:**
        *   Open `core/engine/inference_engine.py`.
        *   At the top, import the context variable: `from core.utils.context import trace_id_var`.
        *   Locate the main entry point for an inference request, `infer_and_adapt`.
        *   As the first step, generate a UUID and set it in the context variable.
            ```python
            import uuid

            class InferenceEngine:
                def infer_and_adapt(self, request):
                    # 1. Generate and set the trace_id for this request's context
                    trace_id = str(uuid.uuid4())
                    token = trace_id_var.set(trace_id)

                    try:
                        # ... all the core logic of inference, retrieval, voting ...
                    finally:
                        # 2. Reset the context variable to its previous state
                        trace_id_var.reset(token)
            ```

    3.  **Implement a Custom Logging Formatter:**
        *   In a utility file (e.g., `core/utils/logging.py`), define a custom `logging.Formatter` class.
        *   This class will override the `format` method to inject the `trace_id` from the context variable into the log record before the message is formatted.
            ```python
            # In core/utils/logging.py
            import logging
            from core.utils.context import trace_id_var

            class TracingFormatter(logging.Formatter):
                def format(self, record):
                    # Get the trace_id from the context. It will be None if not in a request.
                    record.trace_id = trace_id_var.get()
                    return super().format(record)
            ```

    4.  **Apply the Custom Formatter in Your Logging Configuration:**
        *   In the part of your code where you set up the root logger (this might be in your main script or a dedicated logging setup function), instantiate and apply this custom formatter.
            ```python
            import logging
            from core.utils.logging import TracingFormatter

            # Get the root logger
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Get the handler (e.g., StreamHandler)
            handler = logging.StreamHandler()
            
            # Create the formatter with a format string that includes the new trace_id field
            # Use a placeholder like '-' for logs outside a request context.
            formatter = TracingFormatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] [%(trace_id)s | %(threadName)s] - %(message)s'
            )
            
            # Set the formatter for the handler
            handler.setFormatter(formatter)
            
            # Add the handler to the logger
            if not logger.handlers:
                logger.addHandler(handler)
            ```
        *   This setup ensures that any log message emitted anywhere in your code (`logging.info(...)`) will automatically be enriched with the trace ID if one has been set for the current execution context.