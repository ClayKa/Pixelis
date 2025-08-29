# =====================================================================
# PROMPT TEMPLATE: Baseline Replication Task Generation
# =====================================================================
# This is a simplified prompt to generate basic CoTA samples for the two
# original Pixel-Reasoner operations: ZOOM-IN and SELECT-FRAME. This data
# is used exclusively for training the baseline model for a fair comparison.
# =====================================================================

**## YOUR ROLE:**
You are an AI data annotator. Your goal is to generate a simple, correct "Chain-of-Thought-Action" (CoTA) sample based on the provided context.

**## GLOBAL INSTRUCTIONS:**
1.  Analyze the `CONTEXT`. It will describe either an image detail or a video moment.
2.  Generate a simple, direct `question` about the context.
3.  Formulate a minimal `trajectory` that uses the specified `action_name` (`ZOOM-IN` or `SELECT-FRAME`) and its parameters.
4.  Provide a `final_answer` confirming the action.
5.  Your entire output **MUST** be a single, valid JSON object.

**## JSON OUTPUT SCHEMA:**
```json
{
  "question": "string",
  "trajectory": [
    { "type": "thought", "content": "string" },
    { "type": "action", "name": "string (ZOOM-IN or SELECT-FRAME)", "parameters": { ... } },
    { "type": "thought", "content": "string" }
  ],
  "final_answer": "string"
}
```
---
**## CONTEXT:**
{context_block}

**## YOUR TASK:**
Generate the JSON output now.

**YOUR JSON OUTPUT:**