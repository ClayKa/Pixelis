#!/usr/bin/env python3
"""Debug script to check which lines are covered by specific tests."""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from core.modules.operations.read_text import ReadTextOperation

def test_numpy_3d_branch():
    """Test if numpy 3D conversion hits the expected lines."""
    print("Testing numpy 3D conversion...")
    
    operation = ReadTextOperation()
    
    # Create 3D numpy array (HWC format)
    image_3d = np.random.rand(50, 60, 3)
    print(f"Input image shape: {image_3d.shape}")
    print(f"Input image type: {type(image_3d)}")
    print(f"Is numpy array? {isinstance(image_3d, np.ndarray)}")
    print(f"Length of shape: {len(image_3d.shape)}")
    
    # Call preprocess
    result = operation.preprocess(image=image_3d)
    
    print(f"Output image shape: {result['image'].shape}")
    print(f"Output image type: {type(result['image'])}")
    
    # Expected: should convert HWC (50, 60, 3) to CHW (3, 50, 60)
    expected_shape = (3, 50, 60)
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape matches expected: {result['image'].shape == expected_shape}")

def test_confidence_no_words():
    """Test confidence calculation with no words."""
    print("\nTesting confidence calculation with no words...")
    
    # This should be covered by the patched test, but let's verify the logic
    words = []  # Empty list
    
    if words:
        overall_confidence = np.mean([w['confidence'] for w in words])
        print("Branch: words exist")
    else:
        overall_confidence = 0.0  # This is line 206
        print("Branch: no words - confidence = 0.0")
    
    print(f"Overall confidence: {overall_confidence}")

if __name__ == "__main__":
    test_numpy_3d_branch()
    test_confidence_no_words()