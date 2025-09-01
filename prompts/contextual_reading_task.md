**## YOUR ROLE:**
You are an expert AI data annotator specializing in visual document understanding. Your task is to generate a high-quality "Chain-of-Thought-Action" (CoTA) training sample that uses the `READ-TEXT` tool.

**## CRITICAL INSTRUCTIONS:**
1.  Your entire output **MUST** be a single, valid JSON object.
2.  The `question` you generate must be a natural question whose answer is the `GROUND TRUTH TEXT`.
3.  The `final_answer` in your JSON **MUST BE THE EXACT, VERBATIM** string provided in the `GROUND TRUTH TEXT`. This is the most important rule.
4.  The `trajectory` must contain a logical thought process that justifies reading the text from the specified location.
5.  The `bbox` or `polygon` parameter in the `READ-TEXT` action must exactly match the `TARGET GEOMETRY` from the context.

**## PERFECT EXAMPLE (Follow this format and logic exactly):**
---
**EXAMPLE CONTEXT:**
- Source Dataset: InfographicsVQA
- Context Description: "The target text is a large, clear title at the top of an infographic."
- Target Geometry (BBox): [100, 50, 800, 150]
- Ground Truth Text: "Annual Sales Report 2024"

**EXAMPLE JSON OUTPUT:**
```json
{{
  "question": "What is the title of this report?",
  "difficulty": "Easy",
  "trajectory": [
    {{
      "type": "thought",
      "content": "The user is asking for the title of the document. Based on the layout, the title is the large text at the very top. I will read the text within the specified bounding box to get the title."
    }},
    {{
      "type": "action",
      "name": "READ-TEXT",
      "parameters": {{
        "bbox": ""
      }}
    }},
    {{
      "type": "thought",
      "content": "The `READ-TEXT` action returned the string 'Annual Sales Report 2024'. This directly answers the user's question."
    }}
  ],
  "final_answer": "Annual Sales Report 2024"
}}
```
---

**## YOUR ASSIGNMENT:**
Now, generate a new, unique JSON output for the following context. Remember to follow all critical instructions perfectly.

**## CONTEXT FOR YOUR TASK:**
- Source Dataset: {source_dataset}
- Context Description: "{context_description}"
- Target Geometry (BBox or Polygon): {target_geometry}
- Ground Truth Text: "{ground_truth_text}"

**YOUR JSON OUTPUT:**