#!/usr/bin/env python3
"""
Ultimate comprehensive coverage test for segment_object.py to achieve 100% coverage.
Includes all test cases plus the specific branch 103->105 for numpy with unsupported channels.
"""

import sys
sys.path.insert(0, '.')

import coverage
import numpy as np
import torch
from unittest.mock import patch, MagicMock

def ultimate_coverage_test():
    """Ultimate test achieving 100% coverage including all branches."""
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        print("=== ULTIMATE 100% COVERAGE TEST ===")
        
        from core.modules.operations.segment_object import SegmentObjectOperation
        operation = SegmentObjectOperation()
        
        print("Testing ALL validation error branches...")
        
        # Lines 51-52: Missing image
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(point=(25, 25))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'image'")
        
        # Lines 55-56: Missing point  
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 50, 50))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'point'")
        
        # Lines 60-61: Invalid point type
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 50, 50), point="invalid")
            assert not result
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
        
        # Lines 60-61: Invalid point length
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 50, 50), point=(1, 2, 3))
            assert not result
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
        
        # Lines 67-68: Invalid tensor dimensions
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(50, 50), point=(25, 25))  # 2D tensor
            assert not result
            mock_error.assert_called_once_with("Image tensor must be 3D (CHW) or 4D (BCHW)")
        
        # Lines 71-72: Invalid numpy dimensions
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=np.random.rand(50), point=(25, 25))  # 1D array
            assert not result
            mock_error.assert_called_once_with("Image array must be 2D (HW) or 3D (HWC)")
        
        # Lines 74-75: Invalid image type
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image="invalid", point=(25, 25))
            assert not result
            mock_error.assert_called_once_with("Image must be a torch.Tensor or numpy.ndarray")
        
        # Valid inputs
        assert operation.validate_inputs(image=torch.rand(3, 50, 50), point=(25, 25))
        assert operation.validate_inputs(image=np.random.rand(50, 50), point=(25, 25))
        
        print("✓ All validation branches covered")
        
        print("Testing ALL preprocessing branches...")
        
        # Point conversion
        result = operation.preprocess(image=torch.rand(3, 50, 50), point=(25.7, 30.9))
        assert result['point'] == (25, 30)
        
        # Torch tensor - no conversion
        torch_img = torch.rand(3, 50, 50)
        result = operation.preprocess(image=torch_img, point=(25, 30))
        assert torch.equal(result['image'], torch_img)
        
        # Numpy 2D conversion
        numpy_2d = np.random.rand(50, 60).astype(np.float32)
        result = operation.preprocess(image=numpy_2d, point=(30, 25))
        assert result['image'].shape == (1, 50, 60)
        assert result['image'].dtype == torch.float32
        
        # Numpy 3D HWC conversion - supported channels (1, 3, 4)
        for channels in [1, 3, 4]:
            numpy_3d = np.random.rand(40, 50, channels).astype(np.float32)
            result = operation.preprocess(image=numpy_3d, point=(25, 20))
            assert result['image'].shape == (channels, 40, 50)  # CHW format
            assert result['image'].dtype == torch.float32
        
        # *** CRITICAL: Lines 103->105 - numpy 3D with UNSUPPORTED channels ***
        # This is the missing branch! Channels NOT in [1, 3, 4]
        print("Testing CRITICAL branch 103->105: unsupported channel counts...")
        
        for channels in [2, 5, 6, 8]:  # NOT in [1, 3, 4]
            numpy_unsupported = np.random.rand(30, 40, channels).astype(np.float32)
            result = operation.preprocess(image=numpy_unsupported, point=(20, 15))
            # When channels not in [1,3,4], line 104 is SKIPPED, goes to line 105
            # Shape should NOT be permuted - remains HWC
            assert result['image'].shape == (30, 40, channels)  # Original HWC format
            assert result['image'].dtype == torch.float32
            print(f"  ✓ {channels} channels: {numpy_unsupported.shape} -> {result['image'].shape}")
        
        # Point bounds checking
        try:
            operation.preprocess(image=torch.rand(3, 50, 50), point=(100, 25))  # Outside bounds
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Point (100, 25) is outside image bounds" in str(e)
        
        print("✓ All preprocessing branches covered including 103->105")
        
        print("Testing run method error branch...")
        
        # Line 140: Invalid inputs in run
        try:
            operation.run()  # No arguments
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert str(e) == "Invalid inputs for segment operation"
        
        print("✓ Run method error branch covered")
        
        print("Testing helper methods...")
        
        # _create_dummy_mask
        mask = operation._create_dummy_mask(100, 100, (50, 50), radius=30)
        assert mask.shape == (100, 100)
        assert torch.all((mask >= 0) & (mask <= 1))
        assert mask[50, 50].item() == 1.0  # Center should be 1
        
        # _get_bbox_from_mask - normal case
        mask = torch.zeros(100, 100)
        mask[20:60, 30:80] = 1.0
        bbox = operation._get_bbox_from_mask(mask)
        assert bbox == [30, 20, 79, 59]
        
        # _get_bbox_from_mask - empty mask (line 243)
        empty_mask = torch.zeros(100, 100)
        bbox = operation._get_bbox_from_mask(empty_mask)
        assert bbox == [0, 0, 0, 0]
        
        print("✓ Helper methods covered")
        
        print("Testing utility methods...")
        
        params = operation.get_required_params()
        assert params == ['image', 'point']
        
        optional = operation.get_optional_params()
        expected = {'threshold': 0.5, 'return_scores': False}
        assert optional == expected
        
        print("✓ Utility methods covered")
        
        print("Testing registry integration...")
        
        from core.modules.operation_registry import registry
        assert registry.has_operation('SEGMENT_OBJECT_AT')
        operation_class = registry.get_operation_class('SEGMENT_OBJECT_AT')
        assert operation_class == SegmentObjectOperation
        
        print("✓ Registry integration covered")
        
        print("Testing complete workflows...")
        
        # Test all image formats including unsupported channel counts
        test_cases = [
            torch.rand(3, 50, 50),      # CHW tensor
            torch.rand(1, 3, 50, 50),   # BCHW tensor
            np.random.rand(50, 50),     # 2D numpy
            np.random.rand(50, 50, 1),  # 3D numpy 1 channel
            np.random.rand(50, 50, 3),  # 3D numpy 3 channels 
            np.random.rand(50, 50, 4),  # 3D numpy 4 channels
            np.random.rand(50, 50, 2),  # 3D numpy 2 channels (unsupported!)
            np.random.rand(50, 50, 5),  # 3D numpy 5 channels (unsupported!)
        ]
        
        for i, image in enumerate(test_cases):
            result = operation.run(image=image, point=(25, 25))
            assert isinstance(result, dict)
            assert 'mask' in result
            assert 'bbox' in result
            assert 'area' in result
            assert 'confidence' in result
            assert 'object_id' in result
            assert 'point' in result
            print(f"  ✓ Format {i+1}: {type(image).__name__} {getattr(image, 'shape', 'N/A')}")
        
        # Test with optional parameters
        result = operation.run(
            image=torch.rand(3, 50, 50),
            point=(25, 25),
            threshold=0.7,
            return_scores=True
        )
        assert 'scores' in result
        
        result = operation.run(
            image=torch.rand(3, 50, 50),
            point=(25, 25),
            return_scores=False
        )
        assert 'scores' not in result
        
        # Test edge cases
        tiny_result = operation.run(image=torch.rand(3, 5, 5), point=(2, 2))
        assert isinstance(tiny_result, dict)
        
        boundary_result = operation.run(image=torch.rand(3, 100, 100), point=(0, 0))
        assert isinstance(boundary_result, dict)
        
        boundary_result = operation.run(image=torch.rand(3, 100, 100), point=(99, 99))
        assert isinstance(boundary_result, dict)
        
        print("✓ Complete workflows covered")
        
        # Test debug logging
        with patch.object(operation.logger, 'debug') as mock_debug:
            operation.run(image=torch.rand(3, 50, 50), point=(25, 25))
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert "Segmented object at point" in call_args
        
        print("✓ Debug logging covered")
        
        print("\n=== ALL COMPREHENSIVE TESTS COMPLETED ===")
        
    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("ULTIMATE FINAL COVERAGE REPORT")
        print("="*80)
        cov.report(show_missing=True, include='*segment_object.py')
        
        # Generate HTML report
        cov.html_report(directory='ultimate_segment_coverage_html', include='*segment_object.py')
        print(f"\nUltimate HTML coverage report: ultimate_segment_coverage_html/")
        
        return cov

if __name__ == "__main__":
    try:
        cov = ultimate_coverage_test()
        print("\n🏆 ULTIMATE COVERAGE TEST COMPLETED!")
        print("🎯 Segment object operation should now have 100% test coverage!")
    except Exception as e:
        print(f"\n❌ Ultimate test failed: {e}")
        import traceback
        traceback.print_exc()