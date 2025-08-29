# =====================================================================
# MASTER PROMPT TEMPLATE: Fine-grained Detail Perception Task Generation
# =====================================================================
# This prompt guides a powerful LLM (like GPT-4o or Gemini 2.5 Pro) to act as an
# expert data generator. It will create a complete "Chain-of-Thought-Action"
# (CoTA) sample for a task requiring fine-grained visual detail perception.
# =====================================================================

**## YOUR ROLE:**
You are an expert AI data annotator. Your primary goal is to generate a diverse and high-quality training sample for a junior vision-language model. The task you create must teach the junior model how to use a `ZOOM-IN` tool to perceive fine details in an image.

**## GLOBAL INSTRUCTIONS:**
1.  **Choose a Difficulty Level:** First, you **must** randomly select a difficulty level for the task you are about to create, following this exact probability distribution:
    *   **Easy (Presence): 40% chance**
    *   **Medium (Reading): 40% chance**
    *   **Hard (Attributes): 20% chance**
2.  **Select Appropriate Context:** Based on your chosen difficulty, select the relevant information from the `AVAILABLE CONTEXTS` section below. You **must** use the context that matches the difficulty level.
3.  **Generate a Natural Question:** Create a question that a human would ask and that can only be answered by using the detail from your selected context.
4.  **Formulate a Logical Trajectory:** Write a step-by-step reasoning process. This process **must** include a `ZOOM-IN` action that uses the bounding box (`bbox`) from your selected context. The thoughts should be clear and logical.
5.  **Provide the Final Answer:** The final answer must be a direct result of the information revealed after the zoom.
6.  **Strict JSON Output:** Your entire output **must** be a single, valid JSON object. Do not include any explanatory text, apologies, or any characters before or after the JSON code block.

**## DIFFICULTY LEVEL DEFINITIONS:**
*   **Easy (Presence):** The task is about identifying the **presence or count** of a very small object. Use the `[CONTEXT FOR EASY TASK]` for this.
*   **Medium (Reading):** The task is about **reading** small, embedded text that is illegible in the full view. Use the `[CONTEXT FOR MEDIUM TASK]` for this.
*   **Hard (Attributes):** The task is about describing a **fine-grained attribute**, such as the specific pattern, texture, or condition of an object. Use the `[CONTEXT FOR HARD TASK]` for this.

**## JSON OUTPUT SCHEMA:**
```json
{
  "question": "string",
  "difficulty": "string (must be 'Easy', 'Medium', or 'Hard')",
  "trajectory": [
    {
      "type": "thought",
      "content": "string"
    },
    {
      "type": "action",
      "name": "ZOOM-IN",
      "parameters": {
        "bbox": "[x1, y1, x2, y2]"
      }
    },
    {
      "type": "thought",
      "content": "string"
    }
  ],
  "final_answer": "string"
}```
---
**## AVAILABLE CONTEXTS (CHOOSE ONE BASED ON YOUR DIFFICULTY SELECTION):**

**[CONTEXT FOR EASY TASK]**
- Source Dataset: {easy_source_dataset}
- Point of Interest BBox: {easy_bbox}
- Detail within BBox: "{easy_detail_description}"

**[CONTEXT FOR MEDIUM TASK]**
- Source Dataset: {medium_source_dataset}
- Point of Interest BBox: {medium_bbox}
- Text within BBox: "{medium_text_content}"

**[CONTEXT FOR HARD TASK]**
- Source Dataset: {hard_source_dataset}
- Point of Interest BBox: {hard_bbox}
- Detail within BBox: "{hard_detail_description}"

**## YOUR TASK:**
Begin by choosing a difficulty level according to the specified probabilities. Then, select the corresponding context and generate the complete JSON output.

**YOUR JSON OUTPUT:**