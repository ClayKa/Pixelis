#!/usr/bin/env python3
"""
Debug version to identify where track_object tests are failing.
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from unittest.mock import patch


def debug_track_test():
    """Debug test execution step by step."""
    try:
        print("=== DEBUG TRACK_OBJECT TEST ===")
        
        from core.modules.operations.track_object import TrackObjectOperation
        print("✓ Import successful")
        
        # Test data setup
        test_frame = torch.rand(3, 100, 100)
        test_frames = [torch.rand(3, 100, 100) for _ in range(3)]
        test_mask = torch.zeros(100, 100)
        test_mask[40:60, 30:70] = 1.0
        test_bbox = [30, 40, 70, 60]
        print("✓ Test data created")
        
        # Test initialization
        operation = TrackObjectOperation()
        print("✓ Operation created")
        
        # Test validation with valid inputs
        print("Testing validation with valid inputs...")
        result = operation.validate_inputs(frames=test_frames, init_mask=test_mask)
        print(f"Validation result: {result}")
        
        # Test run method with valid inputs
        print("Testing run method...")
        try:
            result = operation.run(
                frame=test_frame,
                init_mask=test_mask,
                track_id="debug_test"
            )
            print(f"✓ Run successful, result keys: {result.keys()}")
        except Exception as e:
            print(f"❌ Run failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("✓ Debug test completed")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_track_test()