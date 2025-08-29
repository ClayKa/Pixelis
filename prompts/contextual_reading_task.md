### **`prompts/contextual_reading.md` (Copy-pastable Content)**
```markdown
# =====================================================================
# MASTER PROMPT TEMPLATE: Text-in-Context Reading Task Generation
# =====================================================================
# This prompt guides a powerful LLM (like GPT-4o or Gemini 2.5 Pro) to act as an
# expert data generator. It will create a "Chain-of-Thought-Action" (CoTA)
# sample for a task that requires reading specific text from a visually
# rich image. The core operation is READ-TEXT.
# =====================================================================

**## YOUR ROLE:**
You are an expert AI data annotator specializing in visual document understanding. Your goal is to generate a high-quality training sample that teaches a junior vision-language model how to use a `READ-TEXT` tool to accurately extract specific information from images like infographics, documents, and real-world scenes.

**## GLOBAL INSTRUCTIONS:**
1.  **Choose a Difficulty Level:** First, you **must** randomly select a difficulty level for the task you are about to create, following this exact probability distribution:
    *   **Easy (Salient Text): 40% chance**
    *   **Medium (Targeted Extraction): 40% chance**
    *   **Hard (Complex Layout/Distortion): 20% chance**
2.  **Select Appropriate Context:** Based on your chosen difficulty, select the relevant information from the `AVAILABLE CONTEXTS` section below.
3.  **Generate a Natural Question:** Create a question whose answer is the `Target Text Content`. The question must be about the *meaning* of the text, not just its location (e.g., ask "What is the title?" not "What text is in this box?").
4.  **Formulate a Logical Trajectory:** Write a step-by-step reasoning process. The trajectory **must** include a `READ-TEXT` action that uses the bounding box (`bbox`) or polygon from your selected context.
5.  **Provide the Final Answer:** The final answer must be the verbatim text from the `Target Text Content`.
6.  **Strict JSON Output:** Your entire output **must** be a single, valid JSON object. Do not include any text before or after the JSON code block.

**## DIFFICULTY LEVEL DEFINITIONS:**
*   **Easy (Salient Text):** The task is to read a large, prominent piece of text like a main title or a large sign. The text is clearly legible and easy to locate. Use the `[CONTEXT FOR EASY TASK]` for this.
*   **Medium (Targeted Extraction):** The task is to find and read a specific piece of information from a dense document or table, such as a date, a value, or a name. This requires locating the correct field before reading. Use the `[CONTEXT FOR MEDIUM TASK]` for this.
*   **Hard (Complex Layout/Distortion):** The task is to read text that is either non-linearly arranged (e.g., on a complex chart), arbitrary-shaped (curved, rotated), or visually distorted. Use the `[CONTEXT FOR HARD TASK]` for this.

**## JSON OUTPUT SCHEMA:**
```json
{
  "question": "string",
  "difficulty": "string (must be 'Easy', 'Medium', or 'Hard')",
  "trajectory": [
    { "type": "thought", "content": "string" },
    { "type": "action", "name": "READ-TEXT", "parameters": { "bbox": "[x1, y1, x2, y2]" } },
    { "type": "thought", "content": "string" }
  ],
  "final_answer": "string"
}
```
---
**## AVAILABLE CONTEXTS (CHOOSE ONE BASED ON YOUR DIFFICULTY SELECTION):**

**[CONTEXT FOR EASY TASK]**
- Source Dataset: {easy_source_dataset}
- Context Description: "The target text is a large, clear title at the top of an infographic."
- Target Text BBox: {easy_bbox}
- Target Text Content: "{easy_text_content}"

**[CONTEXT FOR MEDIUM TASK]**
- Source Dataset: {medium_source_dataset}
- Context Description: "The target text is a value located in the 'Net Profit' row of a financial table within a scanned document."
- Target Text BBox: {medium_bbox}
- Target Text Content: "{medium_text_content}"

**[CONTEXT FOR HARD TASK]**
- Source Dataset: {hard_source_dataset}
- Context Description: "The target text is part of a curved sign on a building in a real-world photograph."
- Target Text Polygon: "{hard_polygon}"
- Target Text Content: "{hard_text_content}"

**## YOUR TASK:**
Begin by choosing a difficulty level according to the specified probabilities. Then, select the corresponding context and generate the complete JSON output.

**YOUR JSON OUTPUT:**