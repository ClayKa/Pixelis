# Question Framing Implementation Summary

## Overview
Successfully implemented a programmatic question framing library to eliminate repetitive question structures in the CoTA data generation process. This addresses the issue where questions were following predictable patterns like "To do X, I need to Y..." or "In this X, I see Y...".

## Implementation Details

### 1. Created QUESTION_FRAMING_TEMPLATES Library
**File**: `core/data_generation/detail_perception.py`

Added 30 diverse question templates as a class attribute that use placeholders for dynamic content:
- `{task_goal}`: The specific objective extracted from the expected observation
- `{location_info}`: The bounding box coordinates

Example templates:
```python
"I need to {task_goal}. Can you please zoom in on the area at {location_info} and tell me what you find?"
"Looking at {location_info}, my task is to {task_goal}. What's visible there?"
"Quick check: Can you {task_goal} by examining {location_info}?"
```

### 2. Implemented Dynamic Task Goal Extraction
**Method**: `_extract_task_goal(expected_observation: str) -> str`

This method intelligently extracts a concise task goal from the expected observation:
- Handles negative observations: "No crack is present" → "verify the absence of crack"
- Handles ambiguous observations: "might be a defect" → "determine if this could be a defect"
- Handles specific patterns: "The serial number reads ABC-123" → "identify the serial number"
- Handles visible items: "Multiple scratch patterns are visible" → "identify the multiple scratch patterns"

### 3. Modified Context Placeholder Building
**Method**: `_build_context_placeholders()`

Updated to:
1. Extract task goal from expected observation
2. Format location info without redundant "at" (just "coordinates [x1, y1, x2, y2]")
3. Randomly select a question template
4. Generate varied questions by combining template with task goal and location

### 4. Results Achieved

#### Question Diversity
- **Before**: Questions had repetitive patterns with low diversity
- **After**: 56% unique starting patterns (exceeds 50% target)
- Eliminated old repetitive patterns ("To do...", "In this...")

#### Natural Language Flow
- Fixed double "at" issue in location descriptions
- Questions now read naturally without awkward constructions
- Maintains grammatical correctness across all templates

#### Task Goal Extraction
- Successfully handles 5 major observation types:
  - Negative observations (absence of objects)
  - Ambiguous observations (uncertainty)
  - Reading patterns (serial numbers, text)
  - Visible items (objects, patterns)
  - Generic observations

## Testing Results

### Question Diversity Test: ✅ PASSED
- Generated 50 test questions
- 56% had unique starting patterns
- 0 questions used old repetitive patterns

### Natural Language Flow Test: ✅ PASSED
- All generated questions have proper grammar
- No double prepositions or awkward constructions
- Natural reading flow maintained

### Task Goal Extraction: ✅ WORKING
- Correctly extracts goals from various observation types
- Handles edge cases like negatives and ambiguity
- Provides fallback for unrecognized patterns

## Impact on Data Quality

The implementation successfully:
1. **Increases linguistic diversity** - Questions now have varied structures
2. **Maintains semantic accuracy** - Task goals accurately reflect observations
3. **Improves naturalness** - Questions read like natural human inquiries
4. **Preserves functionality** - All existing validation and generation logic continues to work

## Files Modified

1. **`core/data_generation/detail_perception.py`**
   - Added QUESTION_FRAMING_TEMPLATES (30 templates)
   - Added _extract_task_goal() method
   - Modified _build_context_placeholders() to use templates
   - Updated mock context generation

## Conclusion

The question framing library successfully eliminates repetitive patterns while maintaining high-quality, semantically accurate question generation. The system now produces diverse, natural-sounding questions that will improve the training data quality for the model.