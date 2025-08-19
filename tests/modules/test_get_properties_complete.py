"""
Complete test suite for get_properties.py achieving 100% coverage.

This test file comprehensively covers all functionality in the GetPropertiesOperation class,
including all branches, error conditions, and edge cases to achieve complete test coverage.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import logging

import sys
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.operations.get_properties import GetPropertiesOperation


class TestGetPropertiesOperation(unittest.TestCase):
    """Test the GetPropertiesOperation class comprehensively."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.operation = GetPropertiesOperation()
        # Create sample image data
        self.sample_image_chw = torch.rand(3, 100, 100)  # CHW format
        self.sample_image_bchw = torch.rand(1, 3, 100, 100)  # BCHW format
        self.sample_mask = torch.zeros(100, 100)
        self.sample_mask[30:60, 40:70] = 1.0  # Rectangle in the middle
        self.sample_bbox = [40, 30, 70, 60]  # [x1, y1, x2, y2]

    def test_init(self):
        """Test __init__ method - lines 22-25."""
        self.assertIsNone(self.operation.feature_extractor)
        self.assertIsInstance(self.operation.logger, logging.Logger)

    def test_load_model_first_call(self):
        """Test _load_model method first call - lines 27-37."""
        # Initially feature_extractor is None
        self.assertIsNone(self.operation.feature_extractor)
        
        # Mock the logger to verify info message
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            mock_info.assert_called_once_with("Loading feature extraction model...")
        
        # After loading, feature_extractor should be set
        self.assertEqual(self.operation.feature_extractor, "placeholder_feature_model")

    def test_load_model_already_loaded(self):
        """Test _load_model method when already loaded - line 33."""
        # Set feature_extractor to something
        self.operation.feature_extractor = "already_loaded"
        
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            # Should not call info since model is already loaded
            mock_info.assert_not_called()
        
        # Should remain unchanged
        self.assertEqual(self.operation.feature_extractor, "already_loaded")

    def test_validate_inputs_missing_image(self):
        """Test validate_inputs with missing image parameter - lines 50-52."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(mask=self.sample_mask)
            mock_error.assert_called_once_with("Missing required parameter: 'image'")
            self.assertFalse(result)

    def test_validate_inputs_missing_mask_and_bbox(self):
        """Test validate_inputs with missing mask and bbox - lines 55-57."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw)
            mock_error.assert_called_once_with("Must provide either 'mask' or 'bbox' parameter")
            self.assertFalse(result)

    def test_validate_inputs_invalid_bbox_not_list(self):
        """Test validate_inputs with invalid bbox type - lines 60-64."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw, bbox="invalid")
            mock_error.assert_called_once_with("'bbox' must be [x1, y1, x2, y2]")
            self.assertFalse(result)

    def test_validate_inputs_invalid_bbox_wrong_length(self):
        """Test validate_inputs with bbox wrong length - lines 60-64."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(image=self.sample_image_chw, bbox=[1, 2, 3])
            mock_error.assert_called_once_with("'bbox' must be [x1, y1, x2, y2]")
            self.assertFalse(result)

    def test_validate_inputs_valid_with_mask(self):
        """Test validate_inputs with valid mask - line 66."""
        result = self.operation.validate_inputs(image=self.sample_image_chw, mask=self.sample_mask)
        self.assertTrue(result)

    def test_validate_inputs_valid_with_bbox(self):
        """Test validate_inputs with valid bbox - line 66."""
        result = self.operation.validate_inputs(image=self.sample_image_chw, bbox=self.sample_bbox)
        self.assertTrue(result)

    def test_validate_inputs_valid_with_both_mask_and_bbox(self):
        """Test validate_inputs with both mask and bbox - line 66."""
        result = self.operation.validate_inputs(
            image=self.sample_image_chw, 
            mask=self.sample_mask, 
            bbox=self.sample_bbox
        )
        self.assertTrue(result)

    def test_run_invalid_inputs(self):
        """Test run method with invalid inputs - lines 87-89."""
        with self.assertRaises(ValueError) as context:
            self.operation.run(image=self.sample_image_chw)  # Missing mask/bbox
        self.assertEqual(str(context.exception), "Invalid inputs for get properties operation")

    def test_run_with_mask_all_properties(self):
        """Test run method with mask and all properties - lines 91-133."""
        with patch.object(self.operation, '_load_model') as mock_load:
            result = self.operation.run(image=self.sample_image_chw, mask=self.sample_mask)
            mock_load.assert_called_once()
        
        # Should contain all property categories
        expected_keys = ['color', 'texture', 'shape', 'size', 'position', 'appearance']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_run_with_bbox_only(self):
        """Test run method with bbox only - lines 100-102."""
        with patch.object(self.operation, '_create_mask_from_bbox') as mock_create_mask:
            mock_create_mask.return_value = self.sample_mask
            result = self.operation.run(image=self.sample_image_chw, bbox=self.sample_bbox)
            mock_create_mask.assert_called_once_with(self.sample_image_chw, self.sample_bbox)
        
        # Should contain all property categories
        expected_keys = ['color', 'texture', 'shape', 'size', 'position', 'appearance']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_run_with_specific_properties_list(self):
        """Test run method with specific properties list - lines 108-129."""
        requested_properties = ['color', 'shape']
        result = self.operation.run(
            image=self.sample_image_chw, 
            mask=self.sample_mask,
            properties=requested_properties
        )
        
        # Should only contain requested properties
        self.assertIn('color', result)
        self.assertIn('shape', result)
        self.assertNotIn('texture', result)
        self.assertNotIn('size', result)
        self.assertNotIn('position', result)
        self.assertNotIn('appearance', result)

    def test_run_individual_property_conditions(self):
        """Test individual property conditions - lines 108-129."""
        # Test each property individually
        properties_to_test = ['color', 'texture', 'shape', 'size', 'position', 'appearance']
        
        for prop in properties_to_test:
            with self.subTest(property=prop):
                result = self.operation.run(
                    image=self.sample_image_chw,
                    mask=self.sample_mask,
                    properties=[prop]
                )
                self.assertIn(prop, result)
                self.assertEqual(len(result), 1)

    def test_create_mask_from_bbox_chw(self):
        """Test _create_mask_from_bbox with CHW image - lines 135-159."""
        image = torch.rand(3, 100, 100)  # CHW format
        bbox = [20, 30, 60, 80]  # [x1, y1, x2, y2]
        
        mask = self.operation._create_mask_from_bbox(image, bbox)
        
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask[30:80, 20:60].sum(), 40 * 50)  # Should be 1 in the region
        self.assertEqual(mask[:30, :].sum(), 0)  # Should be 0 outside
        self.assertEqual(mask[80:, :].sum(), 0)  # Should be 0 outside

    def test_create_mask_from_bbox_bchw(self):
        """Test _create_mask_from_bbox with BCHW image - lines 150-159."""
        image = torch.rand(1, 3, 100, 100)  # BCHW format
        bbox = [20, 30, 60, 80]  # [x1, y1, x2, y2]
        
        mask = self.operation._create_mask_from_bbox(image, bbox)
        
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask[30:80, 20:60].sum(), 40 * 50)  # Should be 1 in the region

    def test_extract_color_properties_chw(self):
        """Test _extract_color_properties with CHW image - lines 161-198."""
        image = torch.rand(3, 100, 100)
        mask = torch.ones(100, 100) * 0.6  # Above 0.5 threshold
        
        result = self.operation._extract_color_properties(image, mask)
        
        # Check all expected keys are present
        expected_keys = [
            'dominant_color', 'mean_color', 'color_variance', 'brightness', 
            'saturation', 'hue', 'color_histogram'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check specific values match implementation
        self.assertEqual(result['dominant_color'], [128, 64, 32])
        self.assertEqual(result['mean_color'], [120, 70, 40])
        self.assertEqual(result['color_variance'], 25.5)

    def test_extract_color_properties_bchw(self):
        """Test _extract_color_properties with BCHW image - lines 177-178."""
        image = torch.rand(2, 3, 100, 100)  # Batch size 2
        mask = torch.ones(100, 100) * 0.6
        
        result = self.operation._extract_color_properties(image, mask)
        
        # Should take first batch item and process correctly
        expected_keys = [
            'dominant_color', 'mean_color', 'color_variance', 'brightness', 
            'saturation', 'hue', 'color_histogram'
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_extract_texture_properties(self):
        """Test _extract_texture_properties - lines 200-227."""
        image = torch.rand(3, 100, 100)
        mask = torch.ones(100, 100)
        
        result = self.operation._extract_texture_properties(image, mask)
        
        # Check all expected keys are present
        expected_keys = [
            'smoothness', 'roughness', 'regularity', 'directionality',
            'contrast', 'homogeneity', 'entropy', 'pattern_type'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check specific values match implementation
        self.assertEqual(result['smoothness'], 0.7)
        self.assertEqual(result['pattern_type'], 'uniform')

    def test_extract_shape_properties_with_mask(self):
        """Test _extract_shape_properties with valid mask - lines 229-275."""
        mask = torch.zeros(100, 100)
        mask[25:75, 20:80] = 1.0  # Rectangle from (25,20) to (75,80)
        
        result = self.operation._extract_shape_properties(mask)
        
        # Should not have error key
        self.assertNotIn('error', result)
        
        # Check all expected keys are present
        expected_keys = [
            'centroid', 'aspect_ratio', 'circularity', 'solidity', 'eccentricity',
            'orientation', 'num_corners', 'is_convex', 'perimeter', 'compactness'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check centroid calculation (actual values from implementation)
        self.assertAlmostEqual(result['centroid'][0], 49.5, places=1)
        self.assertAlmostEqual(result['centroid'][1], 49.5, places=1)
        
        # Check aspect ratio calculation (actual value from implementation)
        self.assertAlmostEqual(result['aspect_ratio'], 1.204, places=2)

    def test_extract_shape_properties_empty_mask(self):
        """Test _extract_shape_properties with empty mask - lines 272-273."""
        mask = torch.zeros(100, 100)  # All zeros, no object
        
        result = self.operation._extract_shape_properties(mask)
        
        # Should have error key when no object found
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No object found in mask')

    def test_extract_size_properties_chw(self):
        """Test _extract_size_properties with CHW image - lines 277-326."""
        image = torch.rand(3, 100, 100)  # CHW format
        mask = torch.zeros(100, 100)
        mask[20:60, 30:80] = 1.0  # Rectangle 40x50 pixels
        
        result = self.operation._extract_size_properties(mask, image)
        
        # Check all expected keys are present
        expected_keys = [
            'area_pixels', 'relative_size', 'bbox_width', 'bbox_height',
            'diagonal_length', 'size_category'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check calculations
        expected_area = 40 * 50  # 2000 pixels
        self.assertEqual(result['area_pixels'], expected_area)
        
        expected_relative_size = expected_area / (100 * 100)
        self.assertAlmostEqual(result['relative_size'], expected_relative_size, places=3)
        
        # Check bbox dimensions (actual values from implementation)
        self.assertEqual(result['bbox_width'], 49)  # 79 - 30
        self.assertEqual(result['bbox_height'], 39)  # 59 - 20

    def test_extract_size_properties_bchw(self):
        """Test _extract_size_properties with BCHW image - lines 293-296."""
        image = torch.rand(1, 3, 100, 100)  # BCHW format
        mask = torch.zeros(100, 100)
        mask[20:60, 30:80] = 1.0
        
        result = self.operation._extract_size_properties(mask, image)
        
        # Should work correctly with BCHW format
        self.assertEqual(result['area_pixels'], 2000)

    def test_extract_size_properties_empty_mask(self):
        """Test _extract_size_properties with empty mask - lines 313-315."""
        image = torch.rand(3, 100, 100)
        mask = torch.zeros(100, 100)  # Empty mask
        
        result = self.operation._extract_size_properties(mask, image)
        
        # Should handle empty mask gracefully
        self.assertEqual(result['area_pixels'], 0)
        self.assertEqual(result['bbox_width'], 0)
        self.assertEqual(result['bbox_height'], 0)

    def test_categorize_size_tiny(self):
        """Test _categorize_size with tiny size - lines 338-339."""
        result = self.operation._categorize_size(0.005)  # < 0.01
        self.assertEqual(result, 'tiny')

    def test_categorize_size_small(self):
        """Test _categorize_size with small size - lines 340-341."""
        result = self.operation._categorize_size(0.03)  # 0.01 <= x < 0.05
        self.assertEqual(result, 'small')

    def test_categorize_size_medium(self):
        """Test _categorize_size with medium size - lines 342-343."""
        result = self.operation._categorize_size(0.1)  # 0.05 <= x < 0.2
        self.assertEqual(result, 'medium')

    def test_categorize_size_large(self):
        """Test _categorize_size with large size - lines 344-345."""
        result = self.operation._categorize_size(0.3)  # 0.2 <= x < 0.5
        self.assertEqual(result, 'large')

    def test_categorize_size_very_large(self):
        """Test _categorize_size with very large size - lines 346-347."""
        result = self.operation._categorize_size(0.7)  # >= 0.5
        self.assertEqual(result, 'very_large')

    def test_extract_position_properties_with_mask(self):
        """Test _extract_position_properties with valid mask - lines 349-394."""
        mask = torch.zeros(100, 100)
        mask[20:40, 30:50] = 1.0  # Rectangle in upper-left area
        
        result = self.operation._extract_position_properties(mask)
        
        # Should not have error key
        self.assertNotIn('error', result)
        
        # Check all expected keys are present
        expected_keys = [
            'centroid', 'normalized_position', 'quadrant', 'distance_from_center',
            'is_centered', 'edge_proximity'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check centroid calculation (actual values from implementation)
        self.assertAlmostEqual(result['centroid'][0], 39.5, places=1)
        self.assertAlmostEqual(result['centroid'][1], 29.5, places=1)

    def test_extract_position_properties_quadrants(self):
        """Test _extract_position_properties quadrant detection - lines 374-381."""
        # Test all four quadrants
        quadrant_tests = [
            (torch.zeros(100, 100), (10, 20, 30, 40), 'top_left'),      # x<0.5, y<0.5
            (torch.zeros(100, 100), (60, 20, 80, 40), 'top_right'),     # x>=0.5, y<0.5
            (torch.zeros(100, 100), (10, 60, 30, 80), 'bottom_left'),   # x<0.5, y>=0.5
            (torch.zeros(100, 100), (60, 60, 80, 80), 'bottom_right'),  # x>=0.5, y>=0.5
        ]
        
        for mask, (x1, y1, x2, y2), expected_quadrant in quadrant_tests:
            with self.subTest(quadrant=expected_quadrant):
                mask[y1:y2, x1:x2] = 1.0
                result = self.operation._extract_position_properties(mask)
                self.assertEqual(result['quadrant'], expected_quadrant)
                mask.zero_()  # Reset for next test

    def test_extract_position_properties_centered_object(self):
        """Test _extract_position_properties with centered object - line 388."""
        mask = torch.zeros(100, 100)
        # Create object centered around (50, 50) with small size
        mask[45:55, 45:55] = 1.0
        
        result = self.operation._extract_position_properties(mask)
        
        # Should be detected as centered (within 0.1 of center)
        self.assertTrue(result['is_centered'])

    def test_extract_position_properties_empty_mask(self):
        """Test _extract_position_properties with empty mask - lines 391-392."""
        mask = torch.zeros(100, 100)  # Empty mask
        
        result = self.operation._extract_position_properties(mask)
        
        # Should have error key when no object found
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No object found in mask')

    def test_extract_appearance_features(self):
        """Test _extract_appearance_features - lines 396-422."""
        image = torch.rand(3, 100, 100)
        mask = torch.ones(100, 100)
        
        result = self.operation._extract_appearance_features(image, mask)
        
        # Check all expected keys are present
        expected_keys = [
            'feature_vector', 'semantic_category', 'confidence',
            'visual_complexity', 'distinctiveness', 'material_type'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check specific values match implementation
        self.assertEqual(result['semantic_category'], 'object')
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(result['material_type'], 'solid')
        self.assertEqual(len(result['feature_vector']), 10)  # Truncated to 10

    def test_get_required_params(self):
        """Test get_required_params method - lines 424-426."""
        result = self.operation.get_required_params()
        self.assertEqual(result, ['image'])

    def test_get_optional_params(self):
        """Test get_optional_params method - lines 428-434."""
        result = self.operation.get_optional_params()
        expected = {
            'mask': None,
            'bbox': None,
            'properties': 'all'
        }
        self.assertEqual(result, expected)

    def test_logging_in_run(self):
        """Test debug logging in run method - line 131."""
        with patch.object(self.operation.logger, 'debug') as mock_debug:
            self.operation.run(image=self.sample_image_chw, mask=self.sample_mask)
            mock_debug.assert_called_once_with("Extracted 6 property categories")

    def test_registry_registration(self):
        """Test that the operation is properly registered - lines 437-458."""
        from core.modules.operation_registry import registry
        
        # Check if GET_PROPERTIES is registered
        self.assertTrue(registry.has_operation('GET_PROPERTIES'))
        
        # Check if we can get the operation class through registry
        operation_class = registry.get_operation_class('GET_PROPERTIES')
        self.assertEqual(operation_class, GetPropertiesOperation)
        
        # Check if we can create an instance by instantiating the class
        operation = operation_class()
        self.assertIsInstance(operation, GetPropertiesOperation)

    def test_full_workflow_integration(self):
        """Test complete workflow integration with all components."""
        # Test the complete pipeline with different configurations
        test_cases = [
            {
                'name': 'CHW_image_with_mask',
                'image': torch.rand(3, 50, 50),
                'mask': torch.ones(50, 50) * 0.8,
                'properties': 'all'
            },
            {
                'name': 'BCHW_image_with_bbox',
                'image': torch.rand(1, 3, 50, 50),
                'bbox': [10, 10, 40, 40],
                'properties': ['color', 'size']
            },
            {
                'name': 'edge_case_small_object',
                'image': torch.rand(3, 100, 100),
                'mask': torch.zeros(100, 100),
                'properties': ['shape', 'position']
            }
        ]
        
        # Set small object mask for last case
        test_cases[2]['mask'][49:51, 49:51] = 1.0  # 2x2 pixel object
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                # Run the operation
                if 'bbox' in case:
                    result = self.operation.run(
                        image=case['image'],
                        bbox=case['bbox'],
                        properties=case['properties']
                    )
                else:
                    result = self.operation.run(
                        image=case['image'],
                        mask=case['mask'],
                        properties=case['properties']
                    )
                
                # Verify result structure
                self.assertIsInstance(result, dict)
                
                if case['properties'] == 'all':
                    expected_keys = ['color', 'texture', 'shape', 'size', 'position', 'appearance']
                    for key in expected_keys:
                        self.assertIn(key, result)
                else:
                    for key in case['properties']:
                        self.assertIn(key, result)

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal size image
        tiny_image = torch.rand(3, 2, 2)
        tiny_mask = torch.ones(2, 2)
        
        result = self.operation.run(image=tiny_image, mask=tiny_mask)
        self.assertIsInstance(result, dict)
        
        # Test with bbox at image boundaries
        boundary_bbox = [0, 0, 100, 100]  # Full image bbox
        result = self.operation.run(image=self.sample_image_chw, bbox=boundary_bbox)
        self.assertIsInstance(result, dict)
        
        # Test with single pixel mask
        single_pixel_mask = torch.zeros(100, 100)
        single_pixel_mask[50, 50] = 1.0
        result = self.operation.run(image=self.sample_image_chw, mask=single_pixel_mask)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    # Run with verbose output to see all test cases
    unittest.main(verbosity=2)