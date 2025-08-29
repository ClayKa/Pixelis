### **`prompts/geometric_reasoning.md` (Copy-pastable Content)**

```markdown
# =====================================================================
# MASTER PROMPT TEMPLATE: Geometric & Property Reasoning Task Generation
# =====================================================================
# This prompt guides a powerful LLM (like GPT-4o or Gemini 2.5 Pro) to act as an
# expert data generator. It will create a "Chain-of-Thought-Action" (CoTA)
# sample for a task requiring segmentation of objects and reasoning about
# their physical properties. The core operations are SEGMENT_OBJECT_AT and
# GET_PROPERTIES.
# =====================================================================

**## YOUR ROLE:**
You are an expert AI data annotator specializing in complex visual reasoning. Your goal is to generate a high-quality training sample that teaches a junior vision-language model how to use a `SEGMENT_OBJECT_AT` tool and a `GET_PROPERTIES` tool in sequence to answer questions about the geometric or physical attributes of objects in an image.

**## GLOBAL INSTRUCTIONS:**
1.  **Choose a Difficulty Level:** First, you **must** randomly select a difficulty level for the task you are about to create, following this exact probability distribution:
    *   **Easy (Direct Comparison): 40% chance**
    *   **Medium (Filtered Comparison): 40% chance**
    *   **Hard (Hierarchical/Conditional Comparison): 20% chance**
2.  **Select Appropriate Context:** Based on your chosen difficulty, select the relevant information from the `AVAILABLE CONTEXTS` section below.
3.  **Generate a Natural Question:** Create a question that requires comparing the specified properties of the target objects.
4.  **Formulate a Logical Trajectory:** Write a complete, step-by-step reasoning process. The trajectory **must** demonstrate a clear, sequential use of `SEGMENT_OBJECT_AT` followed by `GET_PROPERTIES` for **each** object being analyzed. The final thought must perform the comparison based on the retrieved properties.
5.  **Provide the Final Answer:** The final answer must be a direct conclusion from the evidence gathered in the trajectory.
6.  **Strict JSON Output:** Your entire output **must** be a single, valid JSON object. Do not include any text before or after the JSON code block.

**## DIFFICULTY LEVEL DEFINITIONS:**
*   **Easy (Direct Comparison):** The task is a straightforward comparison between two specified objects. Use the `[CONTEXT FOR EASY TASK]` for this.
*   **Medium (Filtered Comparison):** The task involves three or more objects, but the question requires the model to first filter or identify the correct objects to compare based on a property (e.g., "Which of the two *cars* is larger?"). Use the `[CONTEXT FOR MEDIUM TASK]` for this.
*   **Hard (Hierarchical/Conditional Comparison):** The task requires reasoning about the parts of a single object or a conditional comparison. This is the most complex scenario. Use the `[CONTEXT FOR HARD TASK]` for this.

**## JSON OUTPUT SCHEMA:**
```json
{
  "question": "string",
  "difficulty": "string (must be 'Easy', 'Medium', or 'Hard')",
  "trajectory": [
    { "type": "thought", "content": "Initial thought to break down the problem." },
    { "type": "action", "name": "SEGMENT_OBJECT_AT", "parameters": { "point": "[x, y]" } },
    { "type": "thought", "content": "Now that I have the mask for the first object, I will get its properties." },
    { "type": "action", "name": "GET_PROPERTIES", "parameters": { "mask_id": "from_previous_step" } },
    { "type": "thought", "content": "Now I will do the same for the second object." },
    { "type": "action", "name": "SEGMENT_OBJECT_AT", "parameters": { "point": "[x, y]" } },
    { "type": "thought", "content": "Now that I have the mask for the second object, I will get its properties." },
    { "type": "action", "name": "GET_PROPERTIES", "parameters": { "mask_id": "from_previous_step" } },
    { "type": "thought", "content": "I have the properties for both objects. Now I can compare them to answer the question." }
  ],
  "final_answer": "string"
}
```
---
**## AVAILABLE CONTEXTS (CHOOSE ONE BASED ON YOUR DIFFICULTY SELECTION):**

**[CONTEXT FOR EASY TASK]**
- Source Dataset: {easy_source_dataset}
- Object A:
  - Class Name: "{easy_object_A_class}"
  - Location Point: {easy_object_A_point}
  - Property to Compare (Area): {easy_object_A_area}
- Object B:
  - Class Name: "{easy_object_B_class}"
  - Location Point: {easy_object_B_point}
  - Property to Compare (Area): {easy_object_B_area}

**[CONTEXT FOR MEDIUM TASK]**
- Source Dataset: {medium_source_dataset}
- Candidate Objects:
  - Object 1: { "class_name": "{medium_object_1_class}", "point": {medium_object_1_point}, "area": {medium_object_1_area} }
  - Object 2: { "class_name": "{medium_object_2_class}", "point": {medium_object_2_point}, "area": {medium_object_2_area} }
  - Object 3 (Distractor): { "class_name": "{medium_object_3_class}", "point": {medium_object_3_point}, "area": {medium_object_3_area} }
- Task Goal: "Compare the area of the two objects that are of class '{medium_target_class}'."

**[CONTEXT FOR HARD TASK]**
- Source Dataset: {hard_source_dataset}
- Main Object:
  - Class Name: "{hard_main_object_class}"
- Component Parts:
  - Part 1 (e.g., 'head'): { "part_name": "{hard_part_1_name}", "point": {hard_part_1_point}, "area": {hard_part_1_area} }
  - Part 2 (e.g., 'wing'): { "part_name": "{hard_part_2_name}", "point": {hard_part_2_point}, "area": {hard_part_2_area} }
- Task Goal: "Compare the size of the '{hard_part_1_name}' to the '{hard_part_2_name}' of the main object."

**## YOUR TASK:**
Begin by choosing a difficulty level according to the specified probabilities. Then, select the corresponding context and generate the complete JSON output.

**YOUR JSON OUTPUT:**