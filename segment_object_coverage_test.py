#!/usr/bin/env python3
"""
Comprehensive coverage test for segment_object.py to achieve 100% coverage.
"""

import sys
sys.path.insert(0, '.')

import coverage
import numpy as np
import torch
from unittest.mock import patch
from core.modules.operations.segment_object import SegmentObjectOperation

def run_comprehensive_segment_tests():
    """Run all tests to achieve complete coverage for segment_object.py."""
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        operation = SegmentObjectOperation()
        
        print("=== Testing initialization and model loading ===")
        
        # Test __init__ (lines 22-24)
        assert operation.model is None
        print("✓ __init__ tested")
        
        # Test _load_model first call (lines 32-37)
        with patch.object(operation.logger, 'info') as mock_info:
            operation._load_model()
            mock_info.assert_called_once_with("Loading segmentation model...")
        
        assert operation.model == "placeholder_model"
        print("✓ _load_model first call tested")
        
        # Test _load_model already loaded (line 32)
        with patch.object(operation.logger, 'info') as mock_info:
            operation._load_model()  # Should not call logger again
            mock_info.assert_not_called()
        print("✓ _load_model already loaded tested")
        
        print("=== Testing all validation branches ===")
        
        # Test missing image (lines 50-52)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(point=(50, 60))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'image'")
        
        # Test missing point (lines 54-56)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 100, 100))
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'point'")
        
        # Test invalid point type (lines 59-61)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 100, 100), point="invalid")
            assert not result
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
        
        # Test invalid point length (lines 59-61)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(3, 100, 100), point=(1, 2, 3))
            assert not result
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
        
        # Test invalid torch tensor dimensions (lines 66-68)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=torch.rand(100, 100), point=(50, 60))  # 2D instead of 3D/4D
            assert not result
            mock_error.assert_called_once_with("Image tensor must be 3D (CHW) or 4D (BCHW)")
        
        # Test invalid numpy array dimensions (lines 70-72)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image=np.random.rand(100), point=(50, 60))  # 1D instead of 2D/3D
            assert not result
            mock_error.assert_called_once_with("Image array must be 2D (HW) or 3D (HWC)")
        
        # Test invalid image type (lines 74-75)
        with patch.object(operation.logger, 'error') as mock_error:
            result = operation.validate_inputs(image="invalid", point=(50, 60))
            assert not result
            mock_error.assert_called_once_with("Image must be a torch.Tensor or numpy.ndarray")
        
        # Test valid inputs (line 77)
        result = operation.validate_inputs(image=torch.rand(3, 100, 100), point=(50, 60))
        assert result
        
        result = operation.validate_inputs(image=np.random.rand(100, 100), point=(50, 60))
        assert result
        
        print("✓ All validation branches tested")
        
        print("=== Testing preprocessing branches ===")
        
        # Test point conversion (lines 94-95)
        result = operation.preprocess(image=torch.rand(3, 100, 100), point=(50.7, 60.9))
        assert result['point'] == (50, 60)
        
        # Test torch tensor - no conversion (lines 98-99)
        torch_img = torch.rand(3, 100, 100)
        result = operation.preprocess(image=torch_img, point=(50, 60))
        assert torch.equal(result['image'], torch_img)
        
        # Test numpy 2D conversion (lines 101-102)
        numpy_2d = np.random.rand(100, 100)
        result = operation.preprocess(image=numpy_2d, point=(50, 60))
        assert result['image'].shape == (1, 100, 100)
        assert result['image'].dtype == torch.float32
        
        # Test numpy 3D HWC conversion (lines 103-104)
        numpy_3d = np.random.rand(80, 90, 3)  # HWC format
        result = operation.preprocess(image=numpy_3d, point=(40, 50))
        assert result['image'].shape == (3, 80, 90)  # Should convert to CHW
        assert result['image'].dtype == torch.float32
        
        # Test image dimensions CHW (lines 108-109)
        result = operation.preprocess(image=torch.rand(3, 100, 100), point=(30, 40))
        assert 'image' in result and 'point' in result
        
        # Test image dimensions BCHW (lines 110-111) 
        result = operation.preprocess(image=torch.rand(1, 3, 100, 100), point=(30, 40))
        assert 'image' in result and 'point' in result
        
        # Test point within bounds (lines 114-117)
        result = operation.preprocess(image=torch.rand(3, 100, 100), point=(50, 60))
        assert result['point'] == (50, 60)
        
        # Test point outside bounds (lines 115-116)
        try:
            operation.preprocess(image=torch.rand(3, 100, 100), point=(150, 60))
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Point (150, 60) is outside image bounds" in str(e)
        
        print("✓ All preprocessing branches tested")
        
        print("=== Testing run method ===")
        
        # Test invalid inputs (lines 139-140)
        try:
            operation.run(image=torch.rand(3, 100, 100))  # Missing point
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert str(e) == "Invalid inputs for segment operation"
        
        # Test basic execution CHW (lines 142-193)
        result = operation.run(image=torch.rand(3, 100, 100), point=(50, 60))
        expected_keys = ['mask', 'bbox', 'area', 'confidence', 'object_id', 'point']
        for key in expected_keys:
            assert key in result
        
        assert result['point'] == (50, 60)
        assert result['confidence'] == 0.95
        assert isinstance(result['mask'], torch.Tensor)
        assert isinstance(result['bbox'], list)
        assert len(result['bbox']) == 4
        
        # Test basic execution BCHW (lines 160-161)
        result = operation.run(image=torch.rand(1, 3, 100, 100), point=(50, 60))
        assert 'mask' in result and 'bbox' in result
        
        # Test with return_scores=True (lines 185-186)
        result = operation.run(
            image=torch.rand(3, 100, 100), 
            point=(50, 60), 
            return_scores=True
        )
        assert 'scores' in result
        assert isinstance(result['scores'], torch.Tensor)
        
        # Test with return_scores=False (lines 185-186)
        result = operation.run(
            image=torch.rand(3, 100, 100), 
            point=(50, 60), 
            return_scores=False
        )
        assert 'scores' not in result
        
        # Test debug logging (lines 188-191)
        with patch.object(operation.logger, 'debug') as mock_debug:
            result = operation.run(image=torch.rand(3, 100, 100), point=(50, 60))
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert "Segmented object at point" in call_args
        
        print("✓ Run method tested")
        
        print("=== Testing helper methods ===")
        
        # Test _create_dummy_mask (lines 214-227)
        mask = operation._create_dummy_mask(100, 120, (60, 50), radius=30)
        assert mask.shape == (100, 120)
        assert torch.all((mask >= 0) & (mask <= 1))  # Binary mask
        assert mask[50, 60].item() == 1.0  # Center should be 1
        
        # Test _get_bbox_from_mask normal case (lines 240-253)
        mask = torch.zeros(100, 100)
        mask[20:60, 30:80] = 1.0  # Rectangle
        bbox = operation._get_bbox_from_mask(mask)
        assert bbox == [30, 20, 79, 59]
        
        # Test _get_bbox_from_mask empty mask (lines 242-243)
        empty_mask = torch.zeros(100, 100)
        bbox = operation._get_bbox_from_mask(empty_mask)
        assert bbox == [0, 0, 0, 0]
        
        print("✓ Helper methods tested")
        
        print("=== Testing utility methods ===")
        
        # Test get_required_params (lines 256-257)
        params = operation.get_required_params()
        assert params == ['image', 'point']
        
        # Test get_optional_params (lines 261-264)
        optional = operation.get_optional_params()
        expected = {'threshold': 0.5, 'return_scores': False}
        assert optional == expected
        
        print("✓ Utility methods tested")
        
        print("=== Testing registry integration ===")
        
        # Test registry registration (lines 268-286)
        from core.modules.operation_registry import registry
        assert registry.has_operation('SEGMENT_OBJECT_AT')
        operation_class = registry.get_operation_class('SEGMENT_OBJECT_AT')
        assert operation_class == SegmentObjectOperation
        
        print("✓ Registry integration tested")
        
        print("=== Testing comprehensive edge cases ===")
        
        # Test different image formats
        formats = [
            torch.rand(3, 50, 60),      # CHW
            torch.rand(1, 3, 50, 60),   # BCHW
            torch.rand(2, 3, 50, 60),   # BCHW batch
            np.random.rand(50, 60),     # 2D numpy
            np.random.rand(50, 60, 1),  # 3D numpy 1 channel
            np.random.rand(50, 60, 3),  # 3D numpy 3 channels
            np.random.rand(50, 60, 4),  # 3D numpy 4 channels
        ]
        
        for i, img in enumerate(formats):
            result = operation.run(image=img, point=(25, 20))
            assert isinstance(result, dict)
            assert 'mask' in result
            print(f"✓ Format {i+1}/{len(formats)} tested")
        
        # Test edge boundary points
        edge_points = [(0, 0), (99, 0), (0, 99), (99, 99)]
        for point in edge_points:
            result = operation.run(image=torch.rand(3, 100, 100), point=point)
            assert isinstance(result, dict)
        
        print("✓ Edge cases tested")
        
        print("\n=== ALL COMPREHENSIVE TESTS COMPLETED ===")
        
    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("FINAL COVERAGE REPORT")
        print("="*80)
        cov.report(show_missing=True, include='*segment_object.py')
        
        # Generate HTML report
        cov.html_report(directory='segment_coverage_html', include='*segment_object.py')
        print(f"\nHTML coverage report generated in: segment_coverage_html/")
        
        return cov

if __name__ == "__main__":
    try:
        cov = run_comprehensive_segment_tests()
        print("\n🎉 All comprehensive segment object tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()