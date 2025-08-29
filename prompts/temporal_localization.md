# =====================================================================
# MASTER PROMPT TEMPLATE: Temporal Moment Localization Task Generation
# =====================================================================
# This prompt guides a powerful LLM (like GPT-4o or Gemini 2.5 Pro) to act as an
# expert data generator. It will create a complete "Chain-of-Thought-Action"
# (CoTA) sample for a task requiring precise temporal localization of an
# event within a video. The core operation is SELECT-FRAME.
# =====================================================================

**## YOUR ROLE:**
You are an expert AI data annotator for a video understanding project. Your primary goal is to generate a diverse and high-quality training sample that teaches a junior vision-language model how to use a `SELECT-FRAME` tool to find specific moments in a video based on a natural language query.

**## GLOBAL INSTRUCTIONS:**
1.  **Choose a Difficulty Level:** First, you **must** randomly select a difficulty level for the task you are about to create, following this exact probability distribution:
    *   **Easy (Direct Action): 40% chance**
    *   **Medium (Inferential): 40% chance**
    *   **Hard (Procedural Context): 20% chance**
2.  **Select Appropriate Context:** Based on your chosen difficulty, select the relevant information from the `AVAILABLE CONTEXTS` section below. You **must** use the context that matches the difficulty level.
3.  **Generate a Natural Question:** Create a question that a human would ask, which requires finding the target event.
4.  **Formulate a Logical Trajectory:** Write a step-by-step reasoning process. This process **must** include a `SELECT-FRAME` action using the `start_time_sec` and `end_time_sec` from your selected context. The thoughts should justify why this specific time window is being selected.
5.  **Provide the Final Answer:** The final answer must confirm that the event was found within the selected time window.
6.  **Strict JSON Output:** Your entire output **must** be a single, valid JSON object. Do not include any explanatory text or any characters before or after the JSON code block.

**## DIFFICULTY LEVEL DEFINITIONS:**
*   **Easy (Direct Action):** The task is to find a single, physically clear, and easily describable action. Use the `[CONTEXT FOR EASY TASK]` for this.
*   **Medium (Inferential):** The task requires understanding dialogue, plot, or abstract concepts to locate the moment. The event is less about a physical action and more about a narrative or social event. Use the `[CONTEXT FOR MEDIUM TASK]` for this.
*   **Hard (Procedural Context):** The task is to locate a specific, fine-grained **sub-step** within a much longer, complex procedural video. The question might require understanding the broader context of the procedure to identify the correct moment. Use the `[CONTEXT FOR HARD TASK]` for this.

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
      "name": "SELECT-FRAME",
      "parameters": {
        "start_time_sec": float,
        "end_time_sec": float
      }
    },
    {
      "type": "thought",
      "content": "string"
    }
  ],
  "final_answer": "string"
}
```
---
**## AVAILABLE CONTEXTS (CHOOSE ONE BASED ON YOUR DIFFICULTY SELECTION):**

**[CONTEXT FOR EASY TASK]**
- Source Dataset: {easy_source_dataset}
- General Video Description: "{easy_video_description}"
- Target Event:
  - Description: "{easy_event_description}"
  - Start Time: {easy_start_time}
  - End Time: {easy_end_time}

**[CONTEXT FOR MEDIUM TASK]**
- Source Dataset: {medium_source_dataset}
- General Video Description: "{medium_video_description}"
- Target Event:
  - Description: "{medium_event_description}"
  - Start Time: {medium_start_time}
  - End Time: {medium_end_time}

**[CONTEXT FOR HARD TASK]**
- Source Dataset: {hard_source_dataset}
- General Video Description: "{hard_video_description}"
- **Overall Procedure Goal**: "{hard_overall_goal}"
- Target Event (A specific sub-step):
  - Description: "{hard_event_description}"
  - Start Time: {hard_start_time}
  - End Time: {hard_end_time}

**## YOUR TASK:**
Begin by choosing a difficulty level according to the specified probabilities. Then, select the corresponding context and generate the complete JSON output.

**YOUR JSON OUTPUT:**