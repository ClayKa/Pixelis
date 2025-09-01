# Detail Perception Task Generation Prompt

## Task Description
You are generating Chain-of-Thought-Action (CoTA) training data for a vision-language model that needs to identify and describe details in specific regions of an image. The model must learn to use the ZOOM-IN tool to examine fine details that may not be clearly visible at the original resolution.

## Input Context
- **Image**: {image_path}
- **Target Question**: {question}
- **POINT OF INTEREST BBOX**: {bbox} (format: [x1, y1, x2, y2])
- **Expected Observation**: {expected_observation}
- **Answer**: {answer}
- **Style**: {style}
- **DIFFICULTY LEVEL**: {difficulty}

## CRITICAL INSTRUCTIONS:
1.  Your entire output **MUST** be a single, valid JSON object.
2.  **Be Creative:** Choose one of the 40 creative styles below. Generate a `question` and `final_answer` that match the chosen style AND the specified `DIFFICULTY LEVEL`.
3.  You **MUST** adopt the persona and style described in the `## STYLE GUIDELINE FOR THIS SAMPLE` section below. Your `question` and `final_answer` must perfectly match the provided style. (important!)
4.  **MANDATORY TRAJECTORY STRUCTURE:** The `trajectory` field **MUST** be a JSON array containing exactly three objects, perfectly matching the structure, order, and keys shown in the example below.
    * The first object **MUST** be a `thought`.
    * The second object **MUST** be an `action` with the `name` set to `"ZOOM-IN"`.
    * The third object **MUST** be a `thought`.

5.  **ACTION SCHEMA ENFORCEMENT:** Your `action` object **MUST** contain a `parameters` key, which in turn **MUST** contain a `bbox` key. The value for `bbox` **MUST** be copied exactly from the `POINT OF INTEREST BBOX` provided in the context.

    ***YOUR OUTPUT FOR THE `trajectory` FIELD MUST FOLLOW THIS EXACT JSON FORMAT:***
    ```json
    "trajectory": [
      {
        "type": "thought",
        "content": "My internal reasoning for why zooming in is necessary to see the detail..."
      },
      {
        "type": "action",
        "name": "ZOOM-IN",
        "parameters": {
          "bbox": "{bbox}"
        }
      },
      {
        "type": "thought",
        "content": "My internal confirmation that I can now see the detail after zooming in..."
      }
    ]
    ```

6.  **NATURAL THOUGHT PROCESS:** Your `thought` content should emulate a natural, internal reasoning process. Do not just copy template phrases. For the final thought, describe the successful observation of the detail (the value of `{expected_observation}`) as if confirming it to yourself (e.g., "Okay, the zoom reveals..." or "I can now clearly see...").

7.  **DIFFICULTY AND STYLE MATCHING:** Generate a `question` and `final_answer` that **perfectly matches the chosen style AND the specified difficulty level**.

8.  **DISTINCT `final_answer`:** The `final_answer` **MUST** be a conversational, human-like conclusion written as if you are speaking to the user. **DO NOT** simply copy or slightly rephrase your final `thought`. It should be a helpful, conclusive summary of your finding.

9.  **ANCHOR TO CONTEXT:** Your entire generated sample, especially the `final_answer`, **MUST** be directly and logically tied to the `{expected_observation}` provided in the context. Do not invent a completely new scenario. The `{expected_observation}` is the "ground truth" for your task. Your creative style should enhance HOW you describe the observation, not WHAT you observe.

10. **OBSERVATION FIDELITY:** The content of your final thought (step 3 in trajectory) and your `final_answer` **MUST** accurately reflect the `{expected_observation}`. If the expected observation says "No crack is present", your answer must indicate absence of a crack. If it says "The visual evidence for a barcode is inconclusive", your answer must express this uncertainty about the barcode.

## STYLE GUIDELINE FOR THIS SAMPLE

The style for this sample is: **{style}**

### Style Descriptions:
1. **analytical**: Systematic analysis with logical deduction and structured reasoning
2. **curious**: Wonder and interest about visual details, expressing genuine fascination
3. **technical**: Precise, domain-specific terminology with expert-level descriptions
4. **conversational**: Natural, friendly explanation style as if talking to a friend
5. **educational**: Teaching-oriented with clear explanations and learning focus
6. **investigative**: Detective-like approach, gathering clues and evidence
7. **scientific**: Hypothesis-driven observation with methodical documentation
8. **artistic**: Focus on aesthetic qualities, composition, and visual impact
9. **practical**: Goal-oriented, focusing on utility and real-world application
10. **narrative**: Story-telling approach with engaging descriptions

(Additional 30 styles available for variety in generation)

## DIFFICULTY LEVELS

### Difficulty Specifications:
- **easy**: Obvious details requiring basic zoom, clear and unambiguous observations
- **medium**: Subtle details needing careful examination, may require interpretation
- **hard**: Complex patterns, tiny text, or intricate details requiring precise zoom and analysis

The difficulty for this sample is: **{difficulty}**

## Required Output Structure

Your output must be a valid JSON object with the following structure:

```json
{
  "task_id": "detail_perception_[unique_id]",
  "question": "[Your generated question matching the style and difficulty]",
  "trajectory": [
    {
      "type": "thought",
      "content": "[Natural reasoning for why zooming is needed]"
    },
    {
      "type": "action",
      "name": "ZOOM-IN",
      "parameters": {
        "bbox": [x1, y1, x2, y2]
      }
    },
    {
      "type": "thought",
      "content": "[Natural confirmation of what is now visible]"
    }
  ],
  "final_answer": "[Conversational, helpful summary that directly answers the question]",
  "metadata": {
    "style": "{style}",
    "style_id": {style_id},
    "difficulty": "{difficulty}",
    "task_type": "detail_perception",
    "generation_model": "{model_name}",
    "timestamp": "{timestamp}"
  }
}
```

## Example Output

Here is a complete example following all requirements:

```json
{
  "task_id": "detail_perception_001",
  "question": "What specific manufacturer information appears on the small label in the bottom corner of this device?",
  "trajectory": [
    {
      "type": "thought",
      "content": "I can see there's a label in the bottom corner, but the text is too small to read clearly from this distance. I need to zoom in on that specific area to make out the manufacturer details."
    },
    {
      "type": "action",
      "name": "ZOOM-IN",
      "parameters": {
        "bbox": [450, 380, 580, 420]
      }
    },
    {
      "type": "thought",
      "content": "Perfect! The magnified view now clearly shows the manufacturer label. I can see it says 'TechCorp Industries' along with a model number 'XR-2451' and what appears to be a UL certification mark."
    }
  ],
  "final_answer": "The manufacturer label shows 'TechCorp Industries' as the company name, with model number 'XR-2451' and a UL safety certification mark. This appears to be a certified industrial-grade device based on the labeling format.",
  "metadata": {
    "style": "technical",
    "style_id": 3,
    "difficulty": "medium",
    "task_type": "detail_perception",
    "generation_model": "gpt-4",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Important Reminders

- **ALWAYS** return valid JSON that can be parsed
- **NEVER** deviate from the [THOUGHT, ACTION, THOUGHT] structure in the `trajectory` field
- **ALWAYS** use "ZOOM-IN" as the action name (with hyphen, all caps)
- Make thoughts sound natural and avoid template-like phrasing
- Ensure the final_answer is distinct from the final thought and provides additional value
- Include all required fields in your response
- The `trajectory` field name must be used (not `actions` or any other variant)