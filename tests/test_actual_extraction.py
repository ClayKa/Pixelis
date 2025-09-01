#!/usr/bin/env python3
"""
Test the actual extraction logic from detail_perception.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator

# Create a mock generator to test the extraction method
class MockDetailPerception(DetailPerceptionTaskGenerator):
    def __init__(self):
        # Skip normal initialization
        pass

# Test cases
test_cases = [
    ("a tiny logo on the object's edge is clearly visible", "identify the tiny logo"),
    ("No crack is present on the surface", "verify the absence of crack"),
    ("A dark shape that might be a defect, but it's unclear", "determine if this could be a defect"),
    ("The serial number reads ABC-123", "identify the serial number"),
    ("Multiple scratch patterns are visible", "identify the multiple scratch patterns"),
    ("Something unusual appears in the corner", "examine what can be seen")
]

generator = MockDetailPerception()

print("Testing actual extraction logic from detail_perception.py:")
print("="*60)

for observation, expected in test_cases:
    extracted = generator._extract_task_goal(observation)
    
    # Check if it matches (case-insensitive comparison)
    match = expected.lower() in extracted.lower() or extracted.lower() in expected.lower()
    status = "✅" if match else "❌"
    
    print(f"\nObservation: '{observation}'")
    print(f"Expected: '{expected}'")
    print(f"Extracted: '{extracted}' {status}")

print("\n" + "="*60)
print("Test complete!")