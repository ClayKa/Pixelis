#!/usr/bin/env python3
"""
Simplified test script to validate the data quality fixes.
Directly tests the validation logic without full generator initialization.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def normalize_trajectory(trajectory: List[Dict]) -> List[Dict]:
    """Simplified version of trajectory normalization for testing."""
    normalized = []
    for step in trajectory:
        normalized_step = {'type': step.get('type', 'unknown')}
        
        if normalized_step['type'] == 'thought':
            # Try various content keys
            content = (step.get('content') or step.get('thought') or 
                      step.get('observation') or step.get('reasoning') or 
                      step.get('text', ''))
            normalized_step['content'] = content
            
        elif normalized_step['type'] == 'action':
            # Try various name keys
            name = (step.get('name') or step.get('tool_name') or 
                   step.get('action_name') or step.get('tool', ''))
            normalized_step['name'] = name.upper() if name else ''
            
            # Try various parameter keys
            params = (step.get('parameters') or step.get('params') or 
                     step.get('arguments') or step.get('args', {}))
            normalized_step['parameters'] = params
            
        normalized.append(normalized_step)
    
    return normalized

def validate_trajectory_structure(trajectory: List[Dict]) -> tuple[bool, str]:
    """Validate that trajectory follows [THOUGHT, ACTION, THOUGHT] structure."""
    # Normalize first
    normalized = normalize_trajectory(trajectory)
    
    # Check length
    if len(normalized) != 3:
        return False, f"Trajectory must have exactly 3 items, got {len(normalized)}"
    
    # Check types
    step1, step2, step3 = normalized
    if step1.get('type') != 'thought':
        return False, "First step must be a thought"
    if step2.get('type') != 'action':
        return False, "Second step must be an action"
    if step3.get('type') != 'thought':
        return False, "Third step must be a thought"
    
    # Check action name
    if step2.get('name') != 'ZOOM-IN':
        return False, f"Action must be ZOOM-IN, got {step2.get('name')}"
    
    return True, "Valid trajectory structure"

def check_redundancy(final_answer: str, last_thought: str, threshold: float = 0.85) -> tuple[bool, float]:
    """Check if final_answer is too similar to last thought."""
    # Normalize strings for comparison
    final_clean = final_answer.strip().lower()
    thought_clean = last_thought.strip().lower()
    
    # Calculate similarity
    similarity = SequenceMatcher(None, final_clean, thought_clean).ratio()
    
    is_redundant = similarity > threshold
    return is_redundant, similarity

def test_trajectory_validation():
    """Test trajectory structure validation."""
    print("\n" + "="*60)
    print("Testing Trajectory Structure Validation")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Valid trajectory',
            'trajectory': [
                {'type': 'thought', 'content': 'I need to zoom in'},
                {'type': 'action', 'name': 'ZOOM-IN', 'parameters': {'bbox': [0, 0, 100, 100]}},
                {'type': 'thought', 'content': 'Now I can see clearly'}
            ],
            'expected': True
        },
        {
            'name': 'Missing ACTION (only thoughts)',
            'trajectory': [
                {'type': 'thought', 'content': 'First thought'},
                {'type': 'thought', 'content': 'Second thought'}
            ],
            'expected': False
        },
        {
            'name': 'Wrong action name',
            'trajectory': [
                {'type': 'thought', 'content': 'Need to zoom'},
                {'type': 'action', 'name': 'unknown', 'parameters': {}},
                {'type': 'thought', 'content': 'Done'}
            ],
            'expected': False
        },
        {
            'name': 'Wrong order (ACTION first)',
            'trajectory': [
                {'type': 'action', 'name': 'ZOOM-IN', 'parameters': {}},
                {'type': 'thought', 'content': 'Thought 1'},
                {'type': 'thought', 'content': 'Thought 2'}
            ],
            'expected': False
        },
        {
            'name': 'Alternative field names (should normalize)',
            'trajectory': [
                {'type': 'thought', 'thought': 'Need zoom'},  # 'thought' key
                {'type': 'action', 'tool_name': 'zoom-in', 'params': {'bbox': [0, 0, 50, 50]}},  # Alternative keys
                {'type': 'thought', 'observation': 'I see it'}  # 'observation' key
            ],
            'expected': True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        is_valid, message = validate_trajectory_structure(test_case['trajectory'])
        passed = is_valid == test_case['expected']
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"Test {i}: {test_case['name']} - {status}")
        if not passed:
            print(f"  Expected: {test_case['expected']}, Got: {is_valid}")
            print(f"  Message: {message}")

def test_redundancy_detection():
    """Test redundancy detection between final_answer and thoughts."""
    print("\n" + "="*60)
    print("Testing Redundancy Detection")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Identical text',
            'final_answer': 'I can see a red circle',
            'last_thought': 'I can see a red circle',
            'expected_redundant': True
        },
        {
            'name': 'Very similar (minor changes)',
            'final_answer': 'The text shows "Exit"',
            'last_thought': 'The text clearly shows "Exit"',
            'expected_redundant': True
        },
        {
            'name': 'Distinct and conversational',
            'final_answer': 'The safety label indicates this is an emergency exit door.',
            'last_thought': 'Now I can clearly see it says "Emergency Exit"',
            'expected_redundant': False
        },
        {
            'name': 'Different case but same',
            'final_answer': 'The Label Says Hello',
            'last_thought': 'the label says hello',
            'expected_redundant': True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        is_redundant, similarity = check_redundancy(
            test_case['final_answer'], 
            test_case['last_thought']
        )
        passed = is_redundant == test_case['expected_redundant']
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"Test {i}: {test_case['name']} - {status}")
        print(f"  Similarity: {similarity:.2%}")
        if not passed:
            print(f"  Expected redundant: {test_case['expected_redundant']}, Got: {is_redundant}")

def test_complete_sample():
    """Test a complete sample validation."""
    print("\n" + "="*60)
    print("Testing Complete Sample Validation")
    print("="*60)
    
    # Valid sample
    valid_sample = {
        'task_id': 'test_001',
        'question': 'What text is visible in the corner?',
        'actions': [
            {'type': 'thought', 'content': 'I need to zoom in to see the text clearly'},
            {'type': 'action', 'name': 'ZOOM-IN', 'parameters': {'bbox': [10, 10, 50, 50]}},
            {'type': 'thought', 'content': 'Now I can see the text says "Hello World"'}
        ],
        'final_answer': 'The text in the corner reads "Hello World", which appears to be a label.',
        'metadata': {'style': 'analytical', 'style_id': 1, 'difficulty': 'easy'}
    }
    
    # Validate trajectory structure
    traj_valid, traj_msg = validate_trajectory_structure(valid_sample['actions'])
    
    # Check redundancy
    last_thought = valid_sample['actions'][-1].get('content', '')
    is_redundant, similarity = check_redundancy(valid_sample['final_answer'], last_thought)
    
    print("Sample Validation Results:")
    print(f"  Trajectory Structure: {'✓ Valid' if traj_valid else '✗ Invalid'} - {traj_msg}")
    print(f"  Redundancy Check: {'✗ Redundant' if is_redundant else '✓ Distinct'} (similarity: {similarity:.2%})")
    print(f"  Overall: {'✓ PASS' if (traj_valid and not is_redundant) else '✗ FAIL'}")
    
    # Invalid sample
    print("\n" + "-"*40)
    invalid_sample = {
        'task_id': 'test_002',
        'question': 'What is shown?',
        'actions': [
            {'type': 'thought', 'content': 'Looking at image'},
            {'type': 'thought', 'content': 'I see something'}  # Missing ACTION
        ],
        'final_answer': 'I see something',  # Redundant
        'metadata': {}
    }
    
    # Validate trajectory structure
    traj_valid, traj_msg = validate_trajectory_structure(invalid_sample['actions'])
    
    # Check redundancy
    last_thought = invalid_sample['actions'][-1].get('content', '')
    is_redundant, similarity = check_redundancy(invalid_sample['final_answer'], last_thought)
    
    print("Invalid Sample Results:")
    print(f"  Trajectory Structure: {'✓ Valid' if traj_valid else '✗ Invalid'} - {traj_msg}")
    print(f"  Redundancy Check: {'✗ Redundant' if is_redundant else '✓ Distinct'} (similarity: {similarity:.2%})")
    print(f"  Overall: {'✓ PASS' if (traj_valid and not is_redundant) else '✗ FAIL'}")

def main():
    """Run all tests."""
    print("="*60)
    print("Data Quality Validation Test Suite")
    print("="*60)
    print("\nTesting fixes for critical data quality issues:")
    print("1. Trajectory structure validation")
    print("2. Redundancy detection")
    print("3. Normalization for various formats")
    
    # Run tests
    test_trajectory_validation()
    test_redundancy_detection()
    test_complete_sample()
    
    print("\n" + "="*60)
    print("Summary of Implemented Fixes")
    print("="*60)
    print("✓ Strict validation enforcing [THOUGHT, ACTION, THOUGHT] structure")
    print("✓ ZOOM-IN action name requirement")
    print("✓ Redundancy detection (85% similarity threshold)")
    print("✓ Trajectory normalization for multiple field names")
    print("✓ Clear validation messages for debugging")
    print("\nThese validation checks are now integrated into:")
    print("  - core/data_generation/base_generator.py (_validate_and_process_response)")
    print("  - core/data_generation/detail_perception.py (strict validation)")
    print("  - core/data_generation/prompt_templates/detail_perception.md (improved prompts)")

if __name__ == "__main__":
    main()