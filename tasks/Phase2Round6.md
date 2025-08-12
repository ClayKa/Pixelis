**Round 6: Implement Security, Privacy, and Compliance Protocols**

*   **Task 1: Create a Central Security and Privacy Policy Document.**
    *   **Goal:** To establish a clear, comprehensive, and transparent policy regarding data handling, user privacy, and system security for the entire project.
    *   **File:** Create a new document: `docs/SECURITY_AND_PRIVACY.md`.
    *   **Action:** This document will serve as the single source of truth for all privacy- and security-related decisions. It must be created and reviewed before any public-facing component (like the online demo) is deployed.
*   **Task 2: Define and Implement the Data Handling Policy for Online Learning.**
    *   **Goal:** To ensure the online learning process does not store or learn from sensitive user data.
    *   **File:** `docs/SECURITY_AND_PRIVACY.md` (Policy), `core/engine/inference_engine.py` (Implementation).
    *   **Action (Policy):** The policy document must explicitly state:
        1.  **No Persistent Storage of User Inputs:** The raw user inputs (images, videos, text queries) used for online inference will **not be stored persistently** in any long-term log or database. They are only held in memory for the duration of a single request.
        2.  **Learning from Anonymized Data Only:** The `Experience` objects saved to the `ExperienceBuffer` must be stripped of any potential Personally Identifiable Information (PII).
    *   **Action (Implementation):**
        *   Implement a **PII redaction/anonymization** module that processes all text data before it's stored in the `Experience` dataclass.
        *   For image data, confirm that no sensitive EXIF metadata is retained.
*   **Task 3: Enforce a "Read-Only" Policy for the Public Demonstrator.**
    *   **Goal:** To ensure that the publicly accessible interactive demo cannot trigger real, persistent updates to the production model, preventing malicious attacks and data contamination.
    *   **File:** `docs/SECURITY_AND_PRIVACY.md` (Policy), `scripts/launch_demo.py` (Implementation).
    *   **Action (Policy):** The policy must state that the public demo operates in a "sandboxed" or "read-only" mode.
    *   **Action (Implementation):**
        *   When launching the Gradio/Streamlit demo, the script will initialize the `InferenceEngine` with a special configuration flag, e.g., `read_only_mode=True`.
        *   Inside the `infer_and_adapt` function, there will be a check: `if self.config.read_only_mode: return response`. This will ensure that the entire learning and update pipeline (Confidence Gating, Reward Calculation, Enqueuing UpdateTask) is **completely bypassed**.
*   **Task 4: Define Data Retention and Deletion Policies.**
    *   **Goal:** To comply with data privacy regulations like GDPR/CCPA and to manage storage costs.
    *   **File:** `docs/SECURITY_AND_PRIVACY.md` (Policy), `core/modules/experience_buffer.py` (Implementation).
    *   **Action (Policy):** The policy document will define a maximum retention period for all data stored in the `ExperienceBuffer` (e.g., 90 days).
    *   **Action (Implementation):**
        *   The `Experience` dataclass will have a mandatory `timestamp` field.
        *   The `ExperienceBuffer` will include a periodic background task, `prune_old_experiences()`, that automatically removes any experience older than the defined retention period.

*   **Task 5: Implement Audit Trails.**
    *   **Goal:** To maintain a secure, non-repudiable log of all actions that modify the model's state.
    *   **File:** `core/engine/update_worker.py`
    *   **Action:** The `update_worker` will maintain a separate, append-only log file (`update_audit.log`). For every successful model update, it will log key, non-sensitive metadata, such as the `timestamp`, the `Experience` unique ID that triggered the update, the calculated confidence score, and the resulting KL divergence. This provides a clear audit trail for debugging and security reviews.