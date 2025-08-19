#!/usr/bin/env python3
"""
Final comprehensive test to achieve exactly 100% coverage for segment_object.py.
Targets every single line and branch including all error conditions.
"""

import sys
sys.path.insert(0, '.')

import coverage
import numpy as np
import torch
from unittest.mock import patch, MagicMock

def final_100_percent_test():
    """Achieve exactly 100% coverage by testing every single line and branch."""
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        print("=== FINAL 100% COVERAGE TEST ===")
        
        # Import module - covers lines 1-11 (imports and module docstring)
        from core.modules.operations.segment_object import SegmentObjectOperation
        
        # Create operation instance - covers lines 21-24 (__init__)
        operation = SegmentObjectOperation()
        
        # === VALIDATION ERROR BRANCHES ===
        print("Testing all validation error branches...")
        
        # Lines 51-52: Missing image error
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(point=(25, 25))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'image'")
        
        # Lines 55-56: Missing point error  
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 50, 50))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'point'")
        
        # Lines 60-61: Invalid point type error
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 50, 50), point="invalid")
            assert not result
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
        
        # Lines 67-68: Invalid tensor dimensions error
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(50, 50), point=(25, 25))  # 2D tensor
            assert not result
            mock_error.assert_called_once_with("Image tensor must be 3D (CHW) or 4D (BCHW)")
        
        # Lines 71-72: Invalid numpy dimensions error
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=np.random.rand(50), point=(25, 25))  # 1D array
            assert not result
            mock_error.assert_called_once_with("Image array must be 2D (HW) or 3D (HWC)")
        
        # Lines 74-75: Invalid image type error
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image="invalid", point=(25, 25))
            assert not result
            mock_error.assert_called_once_with("Image must be a torch.Tensor or numpy.ndarray")
        
        print("✓ All validation error branches covered")
        
        # === PREPROCESSING BRANCHES ===
        print("Testing all preprocessing branches...")
        
        # Lines 103->105: Test the branch condition in numpy 3D conversion
        # This specifically targets the condition: image.shape[-1] in [1, 3, 4]
        
        # Test with 1 channel (should trigger line 104)
        numpy_1ch = np.random.rand(40, 50, 1).astype(np.float32)
        result = operation.preprocess(image=numpy_1ch, point=(25, 20))
        assert result['image'].shape == (1, 40, 50)
        
        # Test with 3 channels (should trigger line 104)  
        numpy_3ch = np.random.rand(40, 50, 3).astype(np.float32)
        result = operation.preprocess(image=numpy_3ch, point=(25, 20))
        assert result['image'].shape == (3, 40, 50)
        
        # Test with 4 channels (should trigger line 104)
        numpy_4ch = np.random.rand(40, 50, 4).astype(np.float32)
        result = operation.preprocess(image=numpy_4ch, point=(25, 20))
        assert result['image'].shape == (4, 40, 50)
        
        # Line 116: Point outside bounds error
        try:
            operation.preprocess(image=torch.rand(3, 50, 50), point=(100, 25))  # x=100 > w=50
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Point (100, 25) is outside image bounds" in str(e)
        
        print("✓ All preprocessing branches covered")
        
        # === RUN METHOD ERROR BRANCH ===
        print("Testing run method error branch...")
        
        # Line 140: Invalid inputs error in run method
        try:
            operation.run()  # No arguments
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert str(e) == "Invalid inputs for segment operation"
        
        print("✓ Run method error branch covered")
        
        # === BBOX EMPTY MASK BRANCH ===
        print("Testing bbox empty mask branch...")
        
        # Line 243: Empty mask case in _get_bbox_from_mask
        empty_mask = torch.zeros(50, 50)
        bbox = operation._get_bbox_from_mask(empty_mask)
        assert bbox == [0, 0, 0, 0]
        
        print("✓ Empty mask branch covered")
        
        # === COMPLETE WORKFLOW TESTS ===
        print("Testing complete workflows...")
        
        # Test all valid input combinations to ensure full execution paths
        test_cases = [
            # CHW tensor
            {'image': torch.rand(3, 50, 50), 'point': (25, 25)},
            # BCHW tensor
            {'image': torch.rand(1, 3, 50, 50), 'point': (25, 25)},
            # Numpy 2D
            {'image': np.random.rand(50, 50), 'point': (25, 25)},
            # Numpy 3D - 1 channel
            {'image': np.random.rand(50, 50, 1), 'point': (25, 25)},
            # Numpy 3D - 3 channels
            {'image': np.random.rand(50, 50, 3), 'point': (25, 25)},
            # Numpy 3D - 4 channels
            {'image': np.random.rand(50, 50, 4), 'point': (25, 25)},
            # With optional parameters
            {'image': torch.rand(3, 50, 50), 'point': (25, 25), 'threshold': 0.7, 'return_scores': True},
            {'image': torch.rand(3, 50, 50), 'point': (25, 25), 'return_scores': False},
        ]
        
        for i, case in enumerate(test_cases):
            result = operation.run(**case)
            assert isinstance(result, dict)
            assert 'mask' in result
            assert 'bbox' in result
            assert 'area' in result
            assert 'confidence' in result
            assert 'object_id' in result
            assert 'point' in result
            
            if case.get('return_scores', False):
                assert 'scores' in result
            elif 'return_scores' in case and not case['return_scores']:
                assert 'scores' not in result
        
        # Test utility methods
        params = operation.get_required_params()
        assert params == ['image', 'point']
        
        optional = operation.get_optional_params()
        expected = {'threshold': 0.5, 'return_scores': False}
        assert optional == expected
        
        # Test registry
        from core.modules.operation_registry import registry
        assert registry.has_operation('SEGMENT_OBJECT_AT')
        
        # Test edge cases
        # Very small image
        tiny_result = operation.run(image=torch.rand(3, 5, 5), point=(2, 2))
        assert isinstance(tiny_result, dict)
        
        # Boundary points
        boundary_result = operation.run(image=torch.rand(3, 100, 100), point=(0, 0))
        assert isinstance(boundary_result, dict)
        
        boundary_result = operation.run(image=torch.rand(3, 100, 100), point=(99, 99))
        assert isinstance(boundary_result, dict)
        
        print("✓ All workflow tests completed")
        
        print("\n=== ALL COMPREHENSIVE TESTS COMPLETED ===")
        
    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("FINAL 100% COVERAGE REPORT")
        print("="*80)
        cov.report(show_missing=True, include='*segment_object.py')
        
        # Generate HTML report
        cov.html_report(directory='final_segment_100_coverage_html', include='*segment_object.py')
        print(f"\nFinal 100% HTML coverage report: final_segment_100_coverage_html/")
        
        return cov

if __name__ == "__main__":
    try:
        cov = final_100_percent_test()
        print("\n🎯 FINAL 100% COVERAGE TEST COMPLETED!")
        print("🏆 Segment object operation should now have 100% test coverage!")
    except Exception as e:
        print(f"\n❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()