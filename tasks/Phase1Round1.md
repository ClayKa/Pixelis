### **Phase 1: Offline Training**

**Round 1: CoTA (Chain-of-Thought-Action) Data Synthesis and Enrichment**

*   **Task 1: Establish Data Provenance and Licensing Protocol.**
    *   **Goal:** To create a clear, centralized record of all external datasets used in the project, ensuring academic integrity, reproducibility, and compliance with data usage licenses.
    *   **File:** Create a new document in the `docs/` directory: `docs/DATA_PROVENANCE.md`.
    *   **Action 1: Create a Master Datasource Table.**
        *   In `DATA_PROVENANCE.md`, create a master table with the following columns: `Dataset Name`, `Version`, `Original Source (URL)`, `License Type`, `Primary Use Case`, and `Citation`. （）
    *   **Action 2: Populate the Datasource Table.**
        *   Systematically fill this table for every external dataset you plan to use for data synthesis. For example:
            | Dataset Name | Version | Original Source (URL) | License Type | Primary Use Case | Citation |
            | :--- | :--- | :--- | :--- | :--- | :--- |

            | **SA1B** | 1.0 | https://segment-anything.com/dataset | Apache 2.0 | Training and evaluating promptable, general-purpose object segmentation models. | Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. *arXiv preprint arXiv:2304.02643*. |
            | **FineWeb** | 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb | Common Crawl Terms of Use | A high-quality, large-scale pre-training corpus of filtered web data for Large Language Models. | Penedo, G., Malpure, A., Al-Khateeb, O., Al-Ghamdi, S., Alyafeai, Z., Almazrouei, S., & Launay, J. (2024). The FineWeb dataset. *arXiv preprint arXiv:2406.02397*. |
            | **STARQA** | N/A | https://st-vqa.github.io/star/ | CC BY-NC-SA 4.0 | A benchmark for evaluating situational and spatiotemporal reasoning of models in real-world videos. | Wu, B., Yu, S., Chen, Z., Tenenbaum, J. B., & Gan, C. (2024). STAR: A Benchmark for Situated Reasoning in Real-World Videos. *arXiv preprint arXiv:2405.09711*. |
            | **PartImageNet** | N/A | https://partimagenet.github.io/ | Custom (Non-commercial research) | A large-scale, high-quality dataset for training and evaluating fine-grained, part-level object segmentation. | He, Y., Li, Y., Yuan, H., Li, C., Zhang, L., & Zhang, R. (2022). PartImageNet: A Large, High-Quality Dataset of Part Segmentations. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. |
            | **MathVista** | N/A | https://mathvista.github.io/ | CC BY-NC 4.0 | Evaluating the mathematical reasoning capabilities of foundation models in diverse visual contexts. | Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., ... & Gao, J. (2023). MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. *arXiv preprint arXiv:2310.02255*. |
            | **Ego4D** | v2 | https://ego4d-data.org/ | Custom (Ego4D Non-commercial) | A massive-scale, egocentric (first-person) video understanding benchmark across a range of tasks. | Grauman, K., Westbury, A., Byrne, E., Chavis, C., Furnari, A., Girdhar, R., ... & Batra, D. (2022). Ego4D: Exocentric 4D Perception. *In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. |
    *   **Action 3: Implement Provenance Tracking in the Synthesis Pipeline.**
        *   Modify the `scripts/generate_cota_data.py` script.
        *   When generating each new trajectory, the script **must** record the `source_dataset_name` and `original_sample_id` from the source dataset.
        *   This metadata must be saved as part of the final structured JSON for each synthesized sample, e.g., `"provenance": {"source": "COCO", "id": "000000397133"}`.

*   **Task2:** Create a script `scripts/generate_cota_data.py` that formats all output as structured JSON to eliminate complex parsing.

*   **Task3:** Implement data diversity strategies when calling the generation API:
    *   Vary the `temperature` parameter to control creative vs. factual trajectories.
    *   Use multiple, diverse prompt templates for the same task to generate varied reasoning paths (CoTAs).
*   **Task 4:** Augment the Dataset with Advanced Negative Samples.
    *   **Action 1: Generate "Outcome-Negative" Samples.** This is the baseline negative sample. The final answer is deliberately wrong, but the reasoning path may or may not be flawed.
    *   **Action 2: Generate "Process-Negative" (Trap) Samples.** This is the advanced negative sample. The final answer is wrong because of a subtle flaw in the reasoning process. Implement two primary trap types:
        *   **Perceptual Trap:** The trajectory correctly uses a visual tool but then misinterprets the result (e.g., zooms on a license plate but reads "ABC" as "ABD").
        *   **Logical Trap:** The trajectory gathers all visual evidence correctly but makes an incorrect logical deduction in the final textual reasoning step.

*   **Task5:** Implement a validation function to check the integrity of the returned JSON and ensure all actions are within a predefined set.

*   **Task6:** Synthesize Training Data for New Visual Operations.
    *   **Action:** Extend the `scripts/generate_cota_data.py` script (or create a new one) to generate trajectories for new, more complex spatial reasoning tasks.
    *   **Sub-Task 6.1: Task Design for `SEGMENT_OBJECT_AT` + `GET_PROPERTIES`.**
        *   **Task Type:** "Geometric Comparison".
        *   **Example Question:** "Which object is larger: the one at coordinate (x1, y1) or the one at (x2, y2)?"
        *   **Synthesis Strategy:**
            1.  **Input:** An image from a dataset with instance segmentation annotations (e.g., COCO).
            2.  **Object Selection:** Randomly select two distinct object instances from the image's annotation data.
            3.  **Coordinate & Ground Truth Generation:** Extract the center coordinates (`coord_A`, `coord_B`) and pixel areas (`area_A`, `area_B`) for both objects. Determine the ground truth answer (e.g., "the object at `coord_A` is larger").
            4.  **Question & Trajectory Generation:** Use a template to generate a question based on the coordinates. Then, synthesize a reasoning trace that sequentially calls `SEGMENT_OBJECT_AT(coord_A)`, `GET_PROPERTIES(mask_A)`, `SEGMENT_OBJECT_AT(coord_B)`, `GET_PROPERTIES(mask_B)`, followed by a textual comparison and the final answer.
    *   **Sub-Task 6.2: Task Design for `READ_TEXT`.**
        *   **Task Type:** "Targeted Information Extraction".
        *   **Example Question:** "What is the expiration date printed on the milk carton?"
            1.  **Input:** An image from a dataset with OCR annotations (e.g., InfographicsVQA, M6Doc).
            2.  **Region Selection:** Select a text region with its corresponding bounding box (bbox) and ground truth text (gt_text).
            3.  **Question & Trajectory Generation:** Prompt an LLM (e.g., GPT-4) to generate a natural language question whose answer is the `gt_text` (e.g., "What is the title of the chart?"). Then, synthesize a trajectory that calls `READ_TEXT(bbox)` and uses the returned text to answer the question.
    *   **Sub-Task 6.3: Task Design for `TRACK_OBJECT`.**
        *   **Task Type:** "Spatio-Temporal State Analysis".
        *   **Example Question:** "Did the person wearing the red hat ever leave the designated blue area in the video?"
        *   **Synthesis Strategy:**
            1.  **Input:** A video clip from a dataset with object tracking annotations (e.g., MOT17, LaSOT).
            2.  **Object & Region Selection:** Select a tracked object path (`tracked_path`) and define a static spatial region (`region_of_interest`).
            3.  **Ground Truth Calculation:** Programmatically check if the `tracked_path` ever intersects with the `region_of_interest`. This determines the "Yes/No" ground truth answer.
            4.  **Question & Trajectory Generation:** Generate a question like "Did the tracked person ever enter the blue square?". Synthesize a trace that gets the initial object mask, calls `TRACK_OBJECT` to get the path, and then programmatically reasons over the path to find the final answer.
*   **Task 7: Synthesize Iterative Self-Correction Trajectories.**
    *   **Action:** Design a new data synthesis template.
    *   **Synthesis Strategy:**
        *   Start with a correct trajectory.
        *   Intentionally insert an error in an early step (e.g., use `SEGMENT_OBJECT_AT` on a wrong coordinate).
        *   Prompt the LLM to generate a "correctional" reasoning step, such as: "[Text] That doesn't seem right, the object I found is not what I was looking for. I will try a different location."
        *   Follow this with the correct visual operation and continue the rest of the original correct trajectory.

*   **Task 8: Implement a Data Quality Scoring and Filtering Pipeline.**
    *   **Goal:** To programmatically clean the synthesized dataset and remove low-quality or erroneous samples, ensuring the model is trained on high-fidelity data.
    *   **File:** `scripts/filter_and_score_data.py` (a new, dedicated script).
    *   **Action 1: Implement Heuristic Filters.**
        *   The script will first apply a series of rule-based checks to remove obviously flawed samples:
            *   Filter out trajectories with excessively long or short reasoning paths.
            *   Validate and discard any samples with malformed JSON or incorrect action syntax.
            *   Filter out samples where the final answer is missing or not in the expected format.
    *   **Action 2: Implement Model-Based Quality Scoring with Consistency Checks.**
        *   **Goal:** To use a powerful "judge" model to score data quality while ensuring the judge's own ratings are reliable and consistent.
        *   **Sub-Task 2.1: Initial Scoring.**
            *   For each sample, prompt the judge model (e.g., GPT-4o) to rate its logical coherence and correctness on a scale (e.g., 1-5).
        *   **Sub-Task 2.2: Implement Consistency Thresholding.**
            *   To ensure the judge model is consistent, for a small, random subset of the data (e.g., 1%), **run the scoring prompt three (3) times** with a higher temperature.
            *   Calculate the standard deviation of the scores for each of these samples.
            *   If the judge model's ratings for the same sample show a high variance (e.g., std dev > 1.0), it indicates the prompt or the model's judgment is unreliable. This will trigger an alert for manual review of the scoring prompt and process. Only samples judged consistently will proceed.
    *   **Action 3: Create the Final, Filtered and Validated Dataset.**
        *   **Goal:** To construct the final "gold-standard" dataset by applying score thresholds and performing crucial distribution checks.
        *   **Sub-Task 3.1: Apply Quality Score Threshold.**
            *   Filter the dataset to keep only samples that pass the heuristic filters and have a quality score above a predefined threshold (e.g., >= 4.0). This forms the set of "high-confidence synthesized samples".
        *   **Sub-Task 3.2: Perform Distribution Checks.**
            *   After filtering, the script **must** perform a distribution analysis to prevent unintentional data bias. It will calculate and report the distribution of key attributes in the final dataset compared to the initial unfiltered dataset. This includes:
                1.  **Task Type Distribution:** The percentage of samples for "Geometric Comparison", "OCR", "Tracking", etc.
                2.  **Trajectory Type Distribution:** The percentage of "Positive Samples", "Trap Samples", and "Self-Correction Samples".
                3.  **Action Usage Distribution:** The frequency of each visual operation (`SEGMENT_OBJECT_AT`, `READ_TEXT`, etc.) being used.
        *   **Sub-Task 3.3: Enforce Minimum Sample Counts.**
            *   The script will check if the number of samples for any critical category (e.g., any specific task type or the "Trap Samples" category) has fallen below a predefined minimum threshold after filtering.
            *   If a category is under-represented, it will trigger a warning, indicating that more data of that type needs to be synthesized to avoid distributional shift and ensure the model is adequately trained on all desired behaviors.
*   **Task 9: Implement Data Strategy for Hard-Negative Mining.**
    *   **Goal:** To move beyond simple data filtering and implement an intelligent data sampling strategy that prioritizes the most informative "trap" samples during training.
    *   **File:** `scripts/preprocess_data.py` (the same script used for difficulty scoring).
    *   **Action 1: Tag and Analyze Trap Samples.**
        *   When synthesizing "Process-Negative" (Trap) Samples in `Task 4`, the `scripts/generate_cota_data.py` script **must** now add a specific tag to the output JSON, e.g., `"sample_type": "trap_perceptual"` or `"sample_type": "trap_logical"`.
        *   The `preprocess_data.py` script will be enhanced to calculate and report the frequency of each trap type in the synthesized dataset.
    *   **Action 2: Prepare for Weighted Sampling.**
        *   The pre-processing script will add a `sampling_weight` field to every sample in the final "gold-standard" dataset.
        *   By default, this weight is `1.0` for all samples.
        *   For samples tagged as `trap_perceptual` or `trap_logical`, their weight can be increased (e.g., to `1.5` or `2.0`, a configurable parameter). This constitutes a simple but effective hard-mining strategy.
    *   **Action 3: Implement Weighted Data Loading.**
        *   **File:** `scripts/train.py` (within the `CurriculumDataset` or a new `WeightedSampler`).
        *   **Action:** Modify the data loader for both SFT and RFT phases to use a `WeightedRandomSampler`. This sampler will use the `sampling_weight` field to oversample the more challenging "trap" samples during training, forcing the model to spend more effort learning to overcome them.