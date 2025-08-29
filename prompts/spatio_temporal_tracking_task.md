### **`prompts/spatio_temporal_tracking.md` (Copy-pastable Content)**
```markdown
# =====================================================================
# MASTER PROMPT TEMPLATE: Spatio-Temporal Tracking Task Generation
# =====================================================================
# This prompt guides a powerful LLM (like GPT-4o or Gemini 2.5 Pro) to act as an
# expert data generator. It will create a "Chain-of-Thought-Action" (CoTA)
# sample for a task that requires tracking an object's movement over time in a
# video. The core operation is TRACK_OBJECT.
# =====================================================================

**## YOUR ROLE:**
You are an expert AI data annotator for a video understanding project. Your goal is to generate a high-quality training sample that teaches a junior vision-language model how to use a `TRACK_OBJECT` tool to follow an object through a video and answer questions about its trajectory.

**## GLOBAL INSTRUCTIONS:**
1.  **Choose a Difficulty Level:** First, you **must** randomly select a difficulty level for the task you are about to create, following this exact probability distribution:
    *   **Easy (Existence & Basic State): 40% chance**
    *   **Medium (Zone Interaction): 40% chance**
    *   **Hard (Multi-Object Relational): 20% chance**
2.  **Select Appropriate Context:** Based on your chosen difficulty, select the relevant information from the `AVAILABLE CONTEXTS` section below.
3.  **Generate a Natural Question:** Create a question that can only be answered by analyzing the full trajectory of the target object(s).
4.  **Formulate a Logical Trajectory:** Write a step-by-step reasoning process. The trajectory **must** include a `TRACK_OBJECT` action, starting from the provided initial mask. The subsequent thoughts should analyze the returned trajectory to answer the question.
5.  **Provide the Final Answer:** The final answer must be a direct conclusion from the trajectory analysis.
6.  **Strict JSON Output:** Your entire output **must** be a single, valid JSON object. Do not include any text before or after the JSON code block.

**## DIFFICULTY LEVEL DEFINITIONS:**
*   **Easy (Existence & Basic State):** The task is to track a single object and answer a simple question about its existence or a basic state change (e.g., "Does the car ever leave the frame?"). Use the `[CONTEXT FOR EASY TASK]` for this.
*   **Medium (Zone Interaction):** The task is to track a single object and determine if its trajectory interacts with a predefined spatial region (e.g., "Did the person ever step onto the grass?"). Use the `[CONTEXT FOR MEDIUM TASK]` for this.
*   **Hard (Multi-Object Relational):** The task is to track **two** different objects and answer a question about their relative spatial or temporal relationship (e.g., "Which person crossed the finish line first?"). This is the most complex scenario. Use the `[CONTEXT FOR HARD TASK]` for this.

**## JSON OUTPUT SCHEMA:**
```json
{
  "question": "string",
  "difficulty": "string (must be 'Easy', 'Medium', or 'Hard')",
  "trajectory": [
    { "type": "thought", "content": "Initial thought to identify the target object(s)." },
    { "type": "action", "name": "TRACK_OBJECT", "parameters": { "initial_mask": "mask_for_object_A" } },
    // For Hard difficulty, there will be a second TRACK_OBJECT action here
    { "type": "thought", "content": "Now that I have the trajectory/trajectories, I will analyze them to answer the question." }
  ],
  "final_answer": "string"
}
```
---
**## AVAILABLE CONTEXTS (CHOOSE ONE BASED ON YOUR DIFFICULTY SELECTION):**

**[CONTEXT FOR EASY TASK]**
- Source Dataset: {easy_source_dataset}
- Video Description: "{easy_video_description}"
- Target Object:
  - Initial Mask: "{easy_object_mask_A}" (This is a placeholder for the actual mask data)
  - Ground Truth Trajectory Summary: "The object remains visible throughout the entire clip."

**[CONTEXT FOR MEDIUM TASK]**
- Source Dataset: {medium_source_dataset}
- Video Description: "{medium_video_description}"
- Target Object:
  - Initial Mask: "{medium_object_mask_A}"
- Spatial Zone:
  - Zone Name: "the designated safe zone"
  - Zone Coordinates (Polygon): "{medium_zone_polygon}"
- Ground Truth Trajectory Summary: "The object's path intersects with the spatial zone between frames 150 and 220."

**[CONTEXT FOR HARD TASK]**
- Source Dataset: {hard_source_dataset}
- Video Description: "{hard_video_description}"
- Target Object A:
  - Initial Mask: "{hard_object_mask_A}"
  - Object Name: "{hard_object_name_A}"
- Target Object B:
  - Initial Mask: "{hard_object_mask_B}"
  - Object Name: "{hard_object_name_B}"
- Ground Truth Trajectory Summary: "Object A's trajectory crosses the y-coordinate 500 at frame 180. Object B's trajectory crosses the same line at frame 210. Therefore, Object A crossed first."

**## YOUR TASK:**
Begin by choosing a difficulty level according to the specified probabilities. Then, select the corresponding context and generate the complete JSON output.

**YOUR JSON OUTPUT:**