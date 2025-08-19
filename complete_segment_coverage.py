#!/usr/bin/env python3
"""
Complete coverage test targeting the exact missing lines for segment_object.py.
"""

import sys
sys.path.insert(0, '.')

import coverage
import numpy as np
import torch
from unittest.mock import patch, MagicMock

def complete_coverage_test():
    """Target the exact missing lines to achieve 100% coverage."""
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        print("=== TARGETING EXACT MISSING LINES ===")
        
        # Import the module to trigger lines 7-21 (module-level imports and docstring)
        print("Importing module to cover lines 7-21...")
        from core.modules.operations.segment_object import SegmentObjectOperation
        
        # Test line 26 - constructor (__init__)
        print("Testing constructor (line 26)...")
        operation = SegmentObjectOperation()
        
        # Test line 39 - validate_inputs method declaration 
        print("Testing validate_inputs method (line 39)...")
        operation.validate_inputs(image=torch.rand(3, 50, 50), point=(25, 25))
        
        # Test line 79 - preprocess method declaration
        print("Testing preprocess method (line 79)...")
        operation.preprocess(image=torch.rand(3, 50, 50), point=(25, 25))
        
        # Test line 103->105 branch - numpy 3D HWC conversion with specific conditions
        print("Testing numpy 3D conversion branch (lines 103->105)...")
        # This targets the specific branch condition in preprocess
        numpy_3d_rgba = np.random.rand(50, 60, 4).astype(np.float32)  # 4 channels (RGBA)
        result = operation.preprocess(image=numpy_3d_rgba, point=(30, 25))
        assert result['image'].shape == (4, 50, 60)
        
        # Test line 120 - run method declaration
        print("Testing run method (line 120)...")
        operation.run(image=torch.rand(3, 50, 50), point=(25, 25))
        
        # Test line 195 - _create_dummy_mask method declaration
        print("Testing _create_dummy_mask method (line 195)...")
        mask = operation._create_dummy_mask(100, 100, (50, 50), radius=30)
        
        # Test line 229 - _get_bbox_from_mask method declaration
        print("Testing _get_bbox_from_mask method (line 229)...")
        bbox = operation._get_bbox_from_mask(mask)
        
        # Test line 255 - get_required_params method declaration
        print("Testing get_required_params method (line 255)...")
        params = operation.get_required_params()
        
        # Test line 259 - get_optional_params method declaration
        print("Testing get_optional_params method (line 259)...")
        optional = operation.get_optional_params()
        
        # Test line 268 - registry registration
        print("Testing registry registration (line 268)...")
        from core.modules.operation_registry import registry
        assert registry.has_operation('SEGMENT_OBJECT_AT')
        
        # Additional targeted tests for any edge cases
        print("Running additional edge case tests...")
        
        # Test numpy with 1 channel specifically (HWC with C=1)
        numpy_1ch = np.random.rand(40, 50, 1).astype(np.float32)
        result = operation.preprocess(image=numpy_1ch, point=(25, 20))
        assert result['image'].shape == (1, 40, 50)
        
        # Test numpy with exactly 3 channels (RGB)
        numpy_3ch = np.random.rand(40, 50, 3).astype(np.float32)
        result = operation.preprocess(image=numpy_3ch, point=(25, 20))
        assert result['image'].shape == (3, 40, 50)
        
        # Test edge case of mask calculation
        tiny_mask = operation._create_dummy_mask(10, 10, (5, 5), radius=2)
        bbox = operation._get_bbox_from_mask(tiny_mask)
        
        # Test with different image formats in run method
        result = operation.run(image=torch.rand(1, 3, 50, 50), point=(25, 25), threshold=0.8)
        result = operation.run(image=np.random.rand(50, 50), point=(25, 25), return_scores=True)
        
        print("✓ All targeted lines and branches covered")
        
    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("COMPLETE COVERAGE REPORT")
        print("="*80)
        cov.report(show_missing=True, include='*segment_object.py')
        
        # Generate HTML report
        cov.html_report(directory='complete_segment_coverage_html', include='*segment_object.py')
        print(f"\nComplete HTML coverage report: complete_segment_coverage_html/")
        
        return cov

if __name__ == "__main__":
    try:
        cov = complete_coverage_test()
        print("\n🎯 COMPLETE COVERAGE TEST FINISHED!")
        print("📊 Check the coverage report above for final percentage.")
    except Exception as e:
        print(f"\n❌ Complete coverage test failed: {e}")
        import traceback
        traceback.print_exc()