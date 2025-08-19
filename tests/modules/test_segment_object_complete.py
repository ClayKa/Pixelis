"""
Complete test suite for segment_object.py achieving 100% coverage.

This test file comprehensively covers all functionality in the SegmentObjectOperation class,
including all branches, error conditions, and edge cases to achieve complete test coverage.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import logging
import hashlib

import sys
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.operations.segment_object import SegmentObjectOperation


class TestSegmentObjectOperation(unittest.TestCase):
    """Test the SegmentObjectOperation class comprehensively."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = SegmentObjectOperation()
        # Create sample image data
        self.sample_image_chw = torch.rand(3, 100, 100)  # CHW format
        self.sample_image_bchw = torch.rand(1, 3, 100, 100)  # BCHW format
        self.sample_image_numpy_2d = np.random.rand(100, 100)  # Grayscale
        self.sample_image_numpy_3d = np.random.rand(100, 100, 3)  # HWC
        self.sample_point = (50, 60)  # Valid point within image bounds

    def test_init(self):
        """Test __init__ method - lines 22-24."""
        self.assertIsNone(self.operation.model)
        self.assertIsInstance(self.operation.logger, logging.Logger)

    def test_load_model_first_call(self):
        """Test _load_model method first call - lines 32-37."""
        # Initially model is None
        self.assertIsNone(self.operation.model)
        
        # Mock the logger to verify info message
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            mock_info.assert_called_once_with("Loading segmentation model...")
        
        # After loading, model should be set
        self.assertEqual(self.operation.model, "placeholder_model")

    def test_load_model_already_loaded(self):
        """Test _load_model method when already loaded - line 32."""
        # Set model to something
        self.operation.model = "already_loaded"
        
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            # Should not call info since model is already loaded
            mock_info.assert_not_called()
        
        # Should remain unchanged
        self.assertEqual(self.operation.model, "already_loaded")

    def test_validate_inputs_missing_image(self):
        """Test validate_inputs with missing image parameter - lines 50-52."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(point=self.sample_point)
            mock_error.assert_called_once_with("Missing required parameter: 'image'")
            self.assertFalse(result)

    def test_validate_inputs_missing_point(self):
        """Test validate_inputs with missing point parameter - lines 54-56."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw)
            mock_error.assert_called_once_with("Missing required parameter: 'point'")
            self.assertFalse(result)

    def test_validate_inputs_invalid_point_not_tuple_or_list(self):
        """Test validate_inputs with invalid point type - lines 59-61."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw, point="invalid")
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
            self.assertFalse(result)

    def test_validate_inputs_invalid_point_wrong_length(self):
        """Test validate_inputs with point wrong length - lines 59-61."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw, point=(1, 2, 3))
            mock_error.assert_called_once_with("'point' must be a tuple or list of (x, y)")
            self.assertFalse(result)

    def test_validate_inputs_invalid_torch_tensor_wrong_dimensions(self):
        """Test validate_inputs with invalid torch tensor dimensions - lines 66-68."""
        # Create tensor with wrong dimensions (2D instead of 3D/4D)
        invalid_tensor = torch.rand(100, 100)  # 2D tensor
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=invalid_tensor, point=self.sample_point)
            mock_error.assert_called_once_with("Image tensor must be 3D (CHW) or 4D (BCHW)")
            self.assertFalse(result)

    def test_validate_inputs_invalid_numpy_array_wrong_dimensions(self):
        """Test validate_inputs with invalid numpy array dimensions - lines 70-72."""
        # Create array with wrong dimensions (1D instead of 2D/3D)
        invalid_array = np.random.rand(100)  # 1D array
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=invalid_array, point=self.sample_point)
            mock_error.assert_called_once_with("Image array must be 2D (HW) or 3D (HWC)")
            self.assertFalse(result)

    def test_validate_inputs_invalid_image_type(self):
        """Test validate_inputs with invalid image type - lines 74-75."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image="invalid", point=self.sample_point)
            mock_error.assert_called_once_with("Image must be a torch.Tensor or numpy.ndarray")
            self.assertFalse(result)

    def test_validate_inputs_valid_torch_tensor_chw(self):
        """Test validate_inputs with valid torch tensor CHW - line 77."""
        result = self.operation.validate_inputs(image=self.sample_image_chw, point=self.sample_point)
        self.assertTrue(result)

    def test_validate_inputs_valid_torch_tensor_bchw(self):
        """Test validate_inputs with valid torch tensor BCHW - line 77."""
        result = self.operation.validate_inputs(image=self.sample_image_bchw, point=self.sample_point)
        self.assertTrue(result)

    def test_validate_inputs_valid_numpy_2d(self):
        """Test validate_inputs with valid numpy 2D array - line 77."""
        result = self.operation.validate_inputs(image=self.sample_image_numpy_2d, point=self.sample_point)
        self.assertTrue(result)

    def test_validate_inputs_valid_numpy_3d(self):
        """Test validate_inputs with valid numpy 3D array - line 77."""
        result = self.operation.validate_inputs(image=self.sample_image_numpy_3d, point=self.sample_point)
        self.assertTrue(result)

    def test_preprocess_point_conversion(self):
        """Test preprocess point conversion to integers - lines 94-95."""
        result = self.operation.preprocess(image=self.sample_image_chw, point=(50.7, 60.9))
        self.assertEqual(result['point'], (50, 60))

    def test_preprocess_torch_tensor_no_conversion(self):
        """Test preprocess with torch tensor - no conversion needed - line 98-99."""
        result = self.operation.preprocess(image=self.sample_image_chw, point=self.sample_point)
        self.assertTrue(torch.equal(result['image'], self.sample_image_chw))

    def test_preprocess_numpy_2d_conversion(self):
        """Test preprocess with numpy 2D array conversion - lines 101-102."""
        result = self.operation.preprocess(image=self.sample_image_numpy_2d, point=self.sample_point)
        expected_shape = (1, 100, 100)  # Should add channel dimension
        self.assertEqual(result['image'].shape, expected_shape)
        self.assertEqual(result['image'].dtype, torch.float32)

    def test_preprocess_numpy_3d_hwc_conversion(self):
        """Test preprocess with numpy 3D HWC array conversion - lines 103-104."""
        # Create array with channels last (HWC format)
        image_hwc = np.random.rand(80, 90, 3)  # HWC format
        result = self.operation.preprocess(image=image_hwc, point=(40, 50))
        expected_shape = (3, 80, 90)  # Should convert to CHW
        self.assertEqual(result['image'].shape, expected_shape)
        self.assertEqual(result['image'].dtype, torch.float32)

    def test_preprocess_numpy_3d_grayscale_hwc_conversion(self):
        """Test preprocess with numpy 3D grayscale (HWC with 1 channel) conversion - lines 103-104."""
        # Create array with 1 channel (grayscale in HWC format)
        image_hwc_gray = np.random.rand(80, 90, 1)  # HWC format with 1 channel
        result = self.operation.preprocess(image=image_hwc_gray, point=(40, 50))
        expected_shape = (1, 80, 90)  # Should convert to CHW
        self.assertEqual(result['image'].shape, expected_shape)
        self.assertEqual(result['image'].dtype, torch.float32)

    def test_preprocess_image_dimensions_chw(self):
        """Test preprocess image dimension extraction for CHW - lines 108-109."""
        result = self.operation.preprocess(image=self.sample_image_chw, point=(30, 40))
        # Check that preprocessing completed without error
        self.assertIn('image', result)
        self.assertIn('point', result)

    def test_preprocess_image_dimensions_bchw(self):
        """Test preprocess image dimension extraction for BCHW - lines 110-111."""
        result = self.operation.preprocess(image=self.sample_image_bchw, point=(30, 40))
        # Check that preprocessing completed without error
        self.assertIn('image', result)
        self.assertIn('point', result)

    def test_preprocess_point_within_bounds(self):
        """Test preprocess with point within image bounds - lines 114-117."""
        # Point within bounds should not raise exception
        result = self.operation.preprocess(image=self.sample_image_chw, point=(50, 60))
        self.assertEqual(result['point'], (50, 60))

    def test_preprocess_point_outside_bounds_x_negative(self):
        """Test preprocess with point outside bounds (x negative) - lines 115-116."""
        with self.assertRaises(ValueError) as context:
            self.operation.preprocess(image=self.sample_image_chw, point=(-5, 60))
        self.assertIn("Point (-5, 60) is outside image bounds", str(context.exception))

    def test_preprocess_point_outside_bounds_y_negative(self):
        """Test preprocess with point outside bounds (y negative) - lines 115-116."""
        with self.assertRaises(ValueError) as context:
            self.operation.preprocess(image=self.sample_image_chw, point=(50, -10))
        self.assertIn("Point (50, -10) is outside image bounds", str(context.exception))

    def test_preprocess_point_outside_bounds_x_too_large(self):
        """Test preprocess with point outside bounds (x too large) - lines 115-116."""
        with self.assertRaises(ValueError) as context:
            self.operation.preprocess(image=self.sample_image_chw, point=(150, 60))
        self.assertIn("Point (150, 60) is outside image bounds", str(context.exception))

    def test_preprocess_point_outside_bounds_y_too_large(self):
        """Test preprocess with point outside bounds (y too large) - lines 115-116."""
        with self.assertRaises(ValueError) as context:
            self.operation.preprocess(image=self.sample_image_chw, point=(50, 150))
        self.assertIn("Point (50, 150) is outside image bounds", str(context.exception))

    def test_run_invalid_inputs(self):
        """Test run method with invalid inputs - lines 139-140."""
        with self.assertRaises(ValueError) as context:
            self.operation.run(image=self.sample_image_chw)  # Missing point
        self.assertEqual(str(context.exception), "Invalid inputs for segment operation")

    def test_run_basic_execution_chw(self):
        """Test run method basic execution with CHW image - lines 142-193."""
        with patch.object(self.operation, '_load_model') as mock_load:
            result = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
            mock_load.assert_called_once()
        
        # Check basic result structure
        expected_keys = ['mask', 'bbox', 'area', 'confidence', 'object_id', 'point']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check specific values
        self.assertEqual(result['point'], self.sample_point)
        self.assertEqual(result['confidence'], 0.95)
        self.assertIsInstance(result['mask'], torch.Tensor)
        self.assertIsInstance(result['bbox'], list)
        self.assertEqual(len(result['bbox']), 4)

    def test_run_basic_execution_bchw(self):
        """Test run method basic execution with BCHW image - lines 160-161."""
        result = self.operation.run(image=self.sample_image_bchw, point=self.sample_point)
        
        # Should handle BCHW format correctly
        self.assertIn('mask', result)
        self.assertIn('bbox', result)

    def test_run_with_custom_parameters(self):
        """Test run method with custom parameters - lines 150-151."""
        result = self.operation.run(
            image=self.sample_image_chw,
            point=self.sample_point,
            threshold=0.7,
            return_scores=True
        )
        
        # Should include scores when requested
        self.assertIn('scores', result)
        self.assertIsInstance(result['scores'], torch.Tensor)

    def test_run_without_scores(self):
        """Test run method without scores - lines 185-186."""
        result = self.operation.run(
            image=self.sample_image_chw,
            point=self.sample_point,
            return_scores=False
        )
        
        # Should not include scores when not requested
        self.assertNotIn('scores', result)

    def test_run_object_id_generation(self):
        """Test run method object ID generation - lines 173-174."""
        result1 = self.operation.run(image=self.sample_image_chw, point=(30, 40))
        result2 = self.operation.run(image=self.sample_image_chw, point=(50, 60))
        
        # Different points should generate different object IDs
        self.assertNotEqual(result1['object_id'], result2['object_id'])
        
        # Object ID should be 8 characters (MD5 hash truncated)
        self.assertEqual(len(result1['object_id']), 8)

    def test_run_area_calculation(self):
        """Test run method area calculation - line 170."""
        result = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
        
        # Area should be positive integer
        self.assertIsInstance(result['area'], (int, float))
        self.assertGreater(result['area'], 0)

    def test_run_debug_logging(self):
        """Test run method debug logging - lines 188-191."""
        with patch.object(self.operation.logger, 'debug') as mock_debug:
            result = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
            
            # Should log debug message with segmentation details
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            self.assertIn("Segmented object at point", call_args)
            self.assertIn("area=", call_args)
            self.assertIn("bbox=", call_args)
            self.assertIn("id=", call_args)

    def test_create_dummy_mask(self):
        """Test _create_dummy_mask method - lines 214-227."""
        height, width = 100, 120
        point = (60, 50)
        radius = 30
        
        mask = self.operation._create_dummy_mask(height, width, point, radius)
        
        # Check mask properties
        self.assertEqual(mask.shape, (height, width))
        self.assertTrue(torch.all((mask >= 0) & (mask <= 1)))  # Binary mask
        
        # Check that center point is included
        x, y = point
        self.assertEqual(mask[y, x].item(), 1.0)  # Center should be 1
        
        # Check that distant points are excluded
        self.assertEqual(mask[0, 0].item(), 0.0)  # Corner should be 0

    def test_create_dummy_mask_different_radius(self):
        """Test _create_dummy_mask method with different radius - lines 214-227."""
        height, width = 80, 80
        point = (40, 40)
        
        # Test with small radius
        mask_small = self.operation._create_dummy_mask(height, width, point, radius=10)
        
        # Test with large radius
        mask_large = self.operation._create_dummy_mask(height, width, point, radius=30)
        
        # Large radius should include more pixels
        self.assertGreater(mask_large.sum(), mask_small.sum())

    def test_get_bbox_from_mask_normal_case(self):
        """Test _get_bbox_from_mask method with normal mask - lines 240-253."""
        # Create a simple rectangular mask
        mask = torch.zeros(100, 100)
        mask[20:60, 30:80] = 1.0  # Rectangle from (20,30) to (59,79)
        
        bbox = self.operation._get_bbox_from_mask(mask)
        
        # Expected bbox [x1, y1, x2, y2]
        self.assertEqual(bbox, [30, 20, 79, 59])

    def test_get_bbox_from_mask_empty_mask(self):
        """Test _get_bbox_from_mask method with empty mask - lines 242-243."""
        # Create empty mask
        mask = torch.zeros(100, 100)
        
        bbox = self.operation._get_bbox_from_mask(mask)
        
        # Should return [0, 0, 0, 0] for empty mask
        self.assertEqual(bbox, [0, 0, 0, 0])

    def test_get_bbox_from_mask_single_pixel(self):
        """Test _get_bbox_from_mask method with single pixel - lines 245-252."""
        # Create mask with single pixel
        mask = torch.zeros(100, 100)
        mask[45, 55] = 1.0  # Single pixel at (45, 55)
        
        bbox = self.operation._get_bbox_from_mask(mask)
        
        # Should return bbox of single pixel
        self.assertEqual(bbox, [55, 45, 55, 45])

    def test_get_required_params(self):
        """Test get_required_params method - lines 256-257."""
        result = self.operation.get_required_params()
        self.assertEqual(result, ['image', 'point'])

    def test_get_optional_params(self):
        """Test get_optional_params method - lines 261-264."""
        result = self.operation.get_optional_params()
        expected = {
            'threshold': 0.5,
            'return_scores': False
        }
        self.assertEqual(result, expected)

    def test_registry_registration(self):
        """Test that the operation is properly registered - lines 268-286."""
        from core.modules.operation_registry import registry
        
        # Check if SEGMENT_OBJECT_AT is registered
        self.assertTrue(registry.has_operation('SEGMENT_OBJECT_AT'))
        
        # Check if we can get the operation class through registry
        operation_class = registry.get_operation_class('SEGMENT_OBJECT_AT')
        self.assertEqual(operation_class, SegmentObjectOperation)
        
        # Check if we can create an instance by instantiating the class
        operation = operation_class()
        self.assertIsInstance(operation, SegmentObjectOperation)

    def test_full_workflow_integration(self):
        """Test complete workflow integration with all components."""
        test_cases = [
            {
                'name': 'CHW_tensor_center_point',
                'image': torch.rand(3, 60, 80),
                'point': (40, 30),
                'kwargs': {}
            },
            {
                'name': 'BCHW_tensor_corner_point',
                'image': torch.rand(1, 3, 70, 70),
                'point': (10, 10),
                'kwargs': {'return_scores': True, 'threshold': 0.7}
            },
            {
                'name': 'numpy_2d_edge_point',
                'image': np.random.rand(50, 60),
                'point': (55, 25),
                'kwargs': {'return_scores': False}
            },
            {
                'name': 'numpy_3d_hwc_center',
                'image': np.random.rand(40, 50, 3),
                'point': (25, 20),
                'kwargs': {}
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                result = self.operation.run(
                    image=case['image'],
                    point=case['point'],
                    **case['kwargs']
                )
                
                # Verify result structure
                self.assertIsInstance(result, dict)
                expected_keys = ['mask', 'bbox', 'area', 'confidence', 'object_id', 'point']
                for key in expected_keys:
                    self.assertIn(key, result)
                
                # Check optional scores
                if case['kwargs'].get('return_scores', False):
                    self.assertIn('scores', result)
                else:
                    self.assertNotIn('scores', result)

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal size image
        tiny_image = torch.rand(3, 5, 5)
        result = self.operation.run(image=tiny_image, point=(2, 2))
        self.assertIsInstance(result, dict)
        
        # Test with point at image boundary
        boundary_point = (0, 0)  # Top-left corner
        result = self.operation.run(image=self.sample_image_chw, point=boundary_point)
        self.assertIsInstance(result, dict)
        
        # Test with point at opposite boundary
        boundary_point = (99, 99)  # Bottom-right corner (within 100x100 image)
        result = self.operation.run(image=self.sample_image_chw, point=boundary_point)
        self.assertIsInstance(result, dict)

    def test_model_lazy_loading_behavior(self):
        """Test model lazy loading behavior across multiple calls."""
        # First call should load model
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation.run(image=self.sample_image_chw, point=self.sample_point)
            mock_info.assert_called_once_with("Loading segmentation model...")
        
        # Second call should not load model again
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation.run(image=self.sample_image_chw, point=self.sample_point)
            mock_info.assert_not_called()

    def test_error_handling_and_logging(self):
        """Test error handling and logging throughout the operation."""
        # Test various error conditions with proper logging
        test_cases = [
            ({'point': self.sample_point}, "Missing required parameter: 'image'"),
            ({'image': self.sample_image_chw}, "Missing required parameter: 'point'"),
            ({'image': "invalid", 'point': self.sample_point}, "Image must be a torch.Tensor or numpy.ndarray"),
            ({'image': self.sample_image_chw, 'point': "invalid"}, "'point' must be a tuple or list of (x, y)"),
        ]
        
        for kwargs, expected_error in test_cases:
            with self.subTest(kwargs=kwargs):
                with patch.object(self.operation.logger, 'error') as mock_error:
                    result = self.operation.validate_inputs(**kwargs)
                    self.assertFalse(result)
                    mock_error.assert_called()

    def test_comprehensive_image_format_handling(self):
        """Test comprehensive image format handling."""
        # Test all supported image formats
        formats = [
            ('torch_chw', torch.rand(3, 40, 50)),
            ('torch_bchw', torch.rand(1, 3, 40, 50)),
            ('torch_bchw_batch', torch.rand(2, 3, 40, 50)),
            ('numpy_2d', np.random.rand(40, 50)),
            ('numpy_3d_1ch', np.random.rand(40, 50, 1)),
            ('numpy_3d_3ch', np.random.rand(40, 50, 3)),
            ('numpy_3d_4ch', np.random.rand(40, 50, 4)),
        ]
        
        for format_name, image in formats:
            with self.subTest(format=format_name):
                result = self.operation.run(image=image, point=(25, 20))
                self.assertIsInstance(result, dict)
                self.assertIn('mask', result)
                self.assertIn('bbox', result)

    def test_mask_and_bbox_consistency(self):
        """Test that mask and bbox are consistent."""
        result = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
        
        mask = result['mask']
        bbox = result['bbox']
        area = result['area']
        
        # Mask area should match calculated area
        self.assertEqual(mask.sum().item(), area)
        
        # Bbox should contain non-zero elements of mask
        if area > 0:  # Only check if mask is non-empty
            x1, y1, x2, y2 = bbox
            mask_region = mask[y1:y2+1, x1:x2+1]
            self.assertGreater(mask_region.sum().item(), 0)

    def test_reproducibility_with_same_inputs(self):
        """Test that same inputs produce same outputs."""
        # Run same operation twice
        result1 = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
        result2 = self.operation.run(image=self.sample_image_chw, point=self.sample_point)
        
        # Results should be identical
        self.assertEqual(result1['object_id'], result2['object_id'])
        self.assertEqual(result1['area'], result2['area'])
        self.assertEqual(result1['bbox'], result2['bbox'])
        self.assertTrue(torch.equal(result1['mask'], result2['mask']))


if __name__ == '__main__':
    # Run with verbose output to see all test cases
    unittest.main(verbosity=2)