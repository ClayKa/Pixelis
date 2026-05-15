#!/usr/bin/env python3
"""
Test script to verify the improved validation logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the necessary imports to avoid dependencies
class MockLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = MockLogger()

# Test cases for the improved validation
test_cases = [
    {
        "name": "Case-insensitive keys (finalAnswer instead of final_answer)",
        "response": {
            "question": "What detail can you see?",
            "finalAnswer": "I can see a logo",  # camelCase
            "trajectory": [
                {"type": "thought", "content": "Let me examine the area"},
                {"type": "action", "name": "ZOOM-IN", "parameters": {"bbox": [10, 20, 30, 40]}},
                {"type": "thought", "content": "I can now see a logo"}
            ]
        },
        "expected": True
    },
    {
        "name": "Flexible trajectory length (5 steps instead of 3)",
        "response": {
            "question": "What detail can you see?",
            "final_answer": "I can see a barcode",
            "trajectory": [
                {"type": "thought", "content": "Let me examine the area"},
                {"type": "thought", "content": "I need to zoom in closer"},
                {"type": "action", "name": "ZOOM-IN", "parameters": {"bbox": [10, 20, 30, 40]}},
                {"type": "thought", "content": "Now I can see more clearly"},
                {"type": "thought", "content": "I can see a barcode"}
            ]
        },
        "expected": True
    },
    {
        "name": "Action not in second position (T-T-A-T pattern)",
        "response": {
            "question": "What's in this region?",
            "final_answer": "A serial number",
            "trajectory": [
                {"type": "thought", "content": "I need to analyze this region"},
                {"type": "thought", "content": "Let me zoom in for a better view"},
                {"type": "action", "name": "ZOOM-IN", "parameters": {"bbox": [50, 60, 70, 80]}},
                {"type": "thought", "content": "I can see a serial number"}
            ]
        },
        "expected": True
    },
    {
        "name": "Missing action (should fail)",
        "response": {
            "question": "What do you see?",
            "final_answer": "Something",
            "trajectory": [
                {"type": "thought", "content": "Looking"},
                {"type": "thought", "content": "I see something"}
            ]
        },
        "expected": False
    },
    {
        "name": "Trajectory too short (should fail)",
        "response": {
            "question": "What's there?",
            "final_answer": "An object",
            "trajectory": [
                {"type": "action", "name": "ZOOM-IN", "parameters": {"bbox": [1, 2, 3, 4]}},
                {"type": "thought", "content": "I see an object"}
            ]
        },
        "expected": False
    }
]

def validate_response_simplified(llm_response):
    """Simplified version of the validation logic for testing."""
    
    # 1. Basic structural validation
    if not isinstance(llm_response, dict):
        logger.warning(f"LLM response is not a dictionary: {type(llm_response)}")
        return False
    
    # Normalize all keys to lowercase
    try:
        normalized_response = {k.lower(): v for k, v in llm_response.items()}
    except AttributeError:
        logger.warning("Validation failed: LLM output was not a valid dictionary.")
        return False
    
    # Check for required fields
    if 'question' not in normalized_response:
        logger.warning(f"LLM response missing 'question'. Got keys: {list(normalized_response.keys())}")
        return False
    
    # Handle both 'final_answer' and 'finalanswer' cases
    if 'final_answer' not in normalized_response and 'finalanswer' not in normalized_response:
        logger.warning(f"LLM response missing 'final_answer'. Got keys: {list(normalized_response.keys())}")
        return False
    
    # Get trajectory
    trajectory = normalized_response.get('trajectory', llm_response.get('trajectory', []))
    
    if not trajectory or not isinstance(trajectory, list):
        logger.warning(f"Validation failed: Trajectory is not a list.")
        return False
    
    # Check minimum length
    min_trajectory_length = 3
    if len(trajectory) < min_trajectory_length:
        logger.warning(f"Validation failed: Trajectory must have at least {min_trajectory_length} steps. Got {len(trajectory)}")
        return False
    
    # Check for at least one action
    action_exists = any(
        step.get('type') == 'action' 
        for step in trajectory 
        if isinstance(step, dict)
    )
    
    if not action_exists:
        logger.warning("Validation failed: Trajectory must contain at least one 'action' step.")
        return False
    
    # Find and validate action
    for step in trajectory:
        if isinstance(step, dict) and step.get('type') == 'action':
            action_name = step.get('name', '').upper().replace('_', '-')
            if action_name != 'ZOOM-IN':
                logger.warning(f"Validation failed: Action name must be 'ZOOM-IN'. Got: '{step.get('name')}'")
                return False
            
            parameters = step.get('parameters')
            if not parameters or not isinstance(parameters, dict):
                logger.warning(f"Validation failed: ZOOM-IN action missing valid parameters")
                return False
            
            bbox = parameters.get('bbox')
            if not bbox or not (isinstance(bbox, list) and len(bbox) == 4):
                logger.warning(f"Validation failed: Invalid bbox format")
                return False
            break
    
    # Check for at least one thought with content
    thought_with_content = any(
        step.get('type') == 'thought' and step.get('content')
        for step in trajectory
        if isinstance(step, dict)
    )
    
    if not thought_with_content:
        logger.warning("Validation failed: Trajectory must contain at least one thought with content")
        return False
    
    logger.info("Validation passed!")
    return True

# Run tests
print("=" * 60)
print("Testing Improved Validation Logic")
print("=" * 60)

passed = 0
failed = 0

for test_case in test_cases:
    print(f"\nTest: {test_case['name']}")
    print("-" * 40)
    
    result = validate_response_simplified(test_case['response'])
    expected = test_case['expected']
    
    if result == expected:
        print(f"✅ PASSED (Expected: {expected}, Got: {result})")
        passed += 1
    else:
        print(f"❌ FAILED (Expected: {expected}, Got: {result})")
        failed += 1

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 60)

if failed == 0:
    print("\n🎉 All tests passed! The validation improvements are working correctly.")
else:
    print(f"\n⚠️  {failed} test(s) failed. Please review the validation logic.")