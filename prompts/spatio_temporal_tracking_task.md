**## YOUR ROLE:**
You are an expert AI data annotator specializing in video understanding. Your task is to generate a high-quality "Chain-of-Thought-Action" (CoTA) training sample that uses the `TRACK_OBJECT` tool.

**## CRITICAL INSTRUCTIONS:**
1.  Your entire output **MUST** be a single, valid JSON object.
2.  The `question` you generate must be a natural question based on the `TASK GOAL`.
3.  The `final_answer` in your JSON **MUST BE THE EXACT, VERBATIM** string provided in the `GROUND TRUTH CONCLUSION`. This is the most important rule.
4.  The `trajectory` must show a clear process of identifying the target(s), using the `TRACK_OBJECT` tool for each, and then analyzing the resulting trajectory/trajectories.
5.  The `initial_mask` parameter must match the placeholder from the context.

**## PERFECT EXAMPLE (Follow this format and logic exactly):**
---
**EXAMPLE CONTEXT:**
- Source Dataset: MOT20
- Task Goal: "Determine which person, the one in the red shirt or the one in the blue shirt, crosses the horizontal line at y=300 first."
- Ground Truth Conclusion: "The person in the red shirt crosses the line first."
- Object A: {{{{"name": "person in red shirt", "initial_mask": "mask_for_person_A"}}}
- Object B: {{{{"name": "person in blue shirt", "initial_mask": "mask_for_person_B"}}}

**EXAMPLE JSON OUTPUT:**
```json
{{{{
  "question": "Which person crosses the horizontal line in the middle of the screen first: the one in the red shirt or the one in the blue shirt?",
  "difficulty": "Hard",
  "trajectory": [
    {{{{
      "type": "thought",
      "content": "To answer this, I need to track two different people and compare when their paths cross a specific line. I'll start by tracking the person in the red shirt."
    }}}},
    {{{{
      "type": "action",
      "name": "TRACK_OBJECT",
      "parameters": {{{{
        "initial_mask": "mask_for_person_A"
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "I have the trajectory for the person in the red shirt. Now I need to track the person in the blue shirt to compare."
    }}}},
    {{{{
      "type": "action",
      "name": "TRACK_OBJECT",
      "parameters": {{{{
        "initial_mask": "mask_for_person_B"
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "I now have the trajectories for both people. By analyzing their paths, I can determine that the red shirt person's bounding box crosses the y=300 line at an earlier frame than the blue shirt person's. Therefore, the person in the red shirt was first."
    }}}}
  ],
  "final_answer": "The person in the red shirt crosses the line first."
}}}}
```
---

**## YOUR ASSIGNMENT:**
Now, generate a new, unique JSON output for the following context. Remember to follow all critical instructions perfectly.

**## CONTEXT FOR YOUR TASK:**
- Source Dataset: {source_dataset}
- Video Description: "{video_description}"
- Task Goal: "{task_goal}"
- Ground Truth Conclusion: "{ground_truth_conclusion}"
- Object A: {{object_A_details_json}}
- (Optional) Object B: {{object_B_details_json}}
- (Optional) Spatial Zone: {{spatial_zone_json}}

**YOUR JSON OUTPUT:**