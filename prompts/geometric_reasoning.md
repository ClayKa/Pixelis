**## YOUR ROLE:**
You are an expert AI data annotator specializing in complex visual reasoning. Your task is to generate a high-quality "Chain-of-Thought-Action" (CoTA) training sample that uses `SEGMENT_OBJECT_AT` and `GET_PROPERTIES` tools.

**## CRITICAL INSTRUCTIONS:**
1.  Your entire output **MUST** be a single, valid JSON object.
2.  The `question` you generate must be a natural question based on the `TASK GOAL`.
3.  The `trajectory` must show a step-by-step process of segmenting each required object and then getting its properties before making a final comparison.
4.  The `final_answer` in your JSON **MUST BE THE EXACT, VERBATIM** string provided in the `GROUND TRUTH CONCLUSION`. This is the most important rule.
5.  The `point` parameter in each `SEGMENT_OBJECT_AT` action must exactly match the `Location Point` from the context.

**## PERFECT EXAMPLE (Follow this format and logic exactly):**
---
**EXAMPLE CONTEXT:**
- Source Dataset: COCO
- Task Goal: "Compare the size of the cat and the dog."
- Ground Truth Conclusion: "The dog is larger than the cat."
- Object A: {{{{ "class_name": "cat", "point": [250, 300], "area": 12800 }}}
- Object B: {{{{ "class_name": "dog", "point": [680, 450], "area": 15210 }}}

**EXAMPLE JSON OUTPUT:**
```json
{{{{
  "question": "Which animal in the image is larger, the cat or the dog?",
  "difficulty": "Easy",
  "trajectory": [
    {{{{
      "type": "thought",
      "content": "To answer this question, I need to determine the size of two objects: the cat and the dog. I will start by segmenting the cat."
    }}}},
    {{{{
      "type": "action",
      "name": "SEGMENT_OBJECT_AT",
      "parameters": {{{{
        "point": ""
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "Now that I have the mask for the cat, I will get its properties to find its area."
    }}}},
    {{{{
      "type": "action",
      "name": "GET_PROPERTIES",
      "parameters": {{{{
        "mask_id": "from_previous_step"
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "Okay, I have the cat's area. Now I need to do the same for the dog."
    }}}},
    {{{{
      "type": "action",
      "name": "SEGMENT_OBJECT_AT",
      "parameters": {{{{
        "point": ""
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "I have the dog's mask. I will now get its properties."
    }}}},
    {{{{
      "type": "action",
      "name": "GET_PROPERTIES",
      "parameters": {{{{
        "mask_id": "from_previous_step"
      }}}}
    }}}},
    {{{{
      "type": "thought",
      "content": "I have the properties for both animals. The cat's area is 12800 and the dog's area is 15210. Therefore, the dog is larger."
    }}}}
  ],
  "final_answer": "The dog is larger than the cat."
}}}}
```
---

**## YOUR ASSIGNMENT:**
Now, generate a new, unique JSON output for the following context. Remember to follow all critical instructions perfectly.

**## CONTEXT FOR YOUR TASK:**
- Source Dataset: {source_dataset}
- Task Goal: "{task_goal}"
- Ground Truth Conclusion: "{ground_truth_conclusion}"
- Object A: {{object_A_details_json}}
- Object B: {{object_B_details_json}}
- (Optional) Object C (Distractor): {object_C_details_json}

**YOUR JSON OUTPUT:**