"""
Get Properties Operation

Implements the GET_PROPERTIES visual operation for extracting object properties.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from ..operation_registry import BaseOperation, registry


class GetPropertiesOperation(BaseOperation):
    """
    Extracts visual properties of an object or region in an image.
    
    This operation analyzes a specified object or region and returns
    various visual properties such as color, texture, shape, size,
    and spatial relationships.
    """
    
    def __init__(self):
        """Initialize the get properties operation."""
        super().__init__()
        self.feature_extractor = None  # Will be loaded lazily
    
    def _load_model(self):
        """
        Lazily load the feature extraction model.
        
        This could be a vision transformer or CNN for feature extraction.
        """
        if self.feature_extractor is None:
            self.logger.info("Loading feature extraction model...")
            # Placeholder for actual model loading
            # Example: self.feature_extractor = load_vit_model()
            self.feature_extractor = "placeholder_feature_model"
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input arguments.
        
        Required:
            - image: Image tensor or array
            - Either 'mask' or 'bbox' to specify the object/region
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if 'image' not in kwargs:
            self.logger.error("Missing required parameter: 'image'")
            return False
        
        # Must have either mask or bbox
        if 'mask' not in kwargs and 'bbox' not in kwargs:
            self.logger.error("Must provide either 'mask' or 'bbox' parameter")
            return False
        
        # Validate bbox format if provided
        if 'bbox' in kwargs:
            bbox = kwargs['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                self.logger.error("'bbox' must be [x1, y1, x2, y2]")
                return False
        
        return True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the property extraction operation.
        
        Args:
            image: Image tensor (CHW or BCHW format)
            mask: Optional binary mask defining the object
            bbox: Optional [x1, y1, x2, y2] bounding box
            properties: List of specific properties to extract
            
        Returns:
            Dictionary containing various object properties:
                - color: Dominant colors and color statistics
                - texture: Texture descriptors
                - shape: Shape characteristics
                - size: Size metrics (area, dimensions)
                - position: Spatial location and relationships
                - appearance: General appearance features
        """
        # Validate inputs
        if not self.validate_inputs(**kwargs):
            raise ValueError("Invalid inputs for get properties operation")
        
        # Load model if needed
        self._load_model()
        
        # Extract inputs
        image = kwargs['image']
        mask = kwargs.get('mask', None)
        bbox = kwargs.get('bbox', None)
        requested_properties = kwargs.get('properties', 'all')
        
        # Create mask from bbox if only bbox is provided
        if mask is None and bbox is not None:
            mask = self._create_mask_from_bbox(image, bbox)
        
        # Extract all properties
        properties = {}
        
        # Color properties
        if requested_properties == 'all' or 'color' in requested_properties:
            properties['color'] = self._extract_color_properties(image, mask)
        
        # Texture properties
        if requested_properties == 'all' or 'texture' in requested_properties:
            properties['texture'] = self._extract_texture_properties(image, mask)
        
        # Shape properties
        if requested_properties == 'all' or 'shape' in requested_properties:
            properties['shape'] = self._extract_shape_properties(mask)
        
        # Size properties
        if requested_properties == 'all' or 'size' in requested_properties:
            properties['size'] = self._extract_size_properties(mask, image)
        
        # Position properties
        if requested_properties == 'all' or 'position' in requested_properties:
            properties['position'] = self._extract_position_properties(mask)
        
        # Appearance features (using feature extractor)
        if requested_properties == 'all' or 'appearance' in requested_properties:
            properties['appearance'] = self._extract_appearance_features(image, mask)
        
        self.logger.debug(f"Extracted {len(properties)} property categories")
        
        return properties
    
    def _create_mask_from_bbox(
        self,
        image: torch.Tensor,
        bbox: List[int]
    ) -> torch.Tensor:
        """
        Create a binary mask from a bounding box.
        
        Args:
            image: Image tensor
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Binary mask tensor
        """
        if len(image.shape) == 3:  # CHW
            _, h, w = image.shape
        else:  # BCHW
            _, _, h, w = image.shape
        
        x1, y1, x2, y2 = bbox
        mask = torch.zeros((h, w))
        mask[y1:y2, x1:x2] = 1.0
        
        return mask
    
    def _extract_color_properties(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract color-related properties.
        
        Args:
            image: Image tensor
            mask: Binary mask
            
        Returns:
            Color properties dictionary
        """
        # Apply mask to image
        if len(image.shape) == 4:  # BCHW
            image = image[0]  # Take first batch item
        
        # Get masked pixels
        mask_bool = mask > 0.5
        
        # Calculate color statistics (placeholder values)
        color_props = {
            'dominant_color': [128, 64, 32],  # RGB values
            'mean_color': [120, 70, 40],
            'color_variance': 25.5,
            'brightness': 0.65,
            'saturation': 0.45,
            'hue': 30.0,
            'color_histogram': {
                'red': [0.1, 0.3, 0.4, 0.2],  # Simplified histogram bins
                'green': [0.2, 0.4, 0.3, 0.1],
                'blue': [0.3, 0.3, 0.2, 0.2]
            }
        }
        
        return color_props
    
    def _extract_texture_properties(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract texture-related properties.
        
        Args:
            image: Image tensor
            mask: Binary mask
            
        Returns:
            Texture properties dictionary
        """
        # Placeholder texture analysis
        texture_props = {
            'smoothness': 0.7,
            'roughness': 0.3,
            'regularity': 0.6,
            'directionality': 45.0,  # Dominant direction in degrees
            'contrast': 0.8,
            'homogeneity': 0.65,
            'entropy': 2.3,
            'pattern_type': 'uniform'  # Could be: uniform, striped, spotted, etc.
        }
        
        return texture_props
    
    def _extract_shape_properties(
        self,
        mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract shape-related properties.
        
        Args:
            mask: Binary mask
            
        Returns:
            Shape properties dictionary
        """
        # Calculate shape metrics (placeholder values)
        nonzero = torch.nonzero(mask)
        
        if len(nonzero) > 0:
            # Calculate centroid
            centroid_y = nonzero[:, 0].float().mean().item()
            centroid_x = nonzero[:, 1].float().mean().item()
            
            # Calculate bounding box
            min_y = nonzero[:, 0].min().item()
            max_y = nonzero[:, 0].max().item()
            min_x = nonzero[:, 1].min().item()
            max_x = nonzero[:, 1].max().item()
            
            width = max_x - min_x
            height = max_y - min_y
            aspect_ratio = width / max(height, 1)
            
            shape_props = {
                'centroid': [centroid_x, centroid_y],
                'aspect_ratio': aspect_ratio,
                'circularity': 0.75,  # How circular the shape is
                'solidity': 0.85,  # Ratio of area to convex hull area
                'eccentricity': 0.4,  # How elongated the shape is
                'orientation': 30.0,  # Principal axis orientation in degrees
                'num_corners': 4,  # Estimated number of corners
                'is_convex': False,
                'perimeter': 250.0,
                'compactness': 0.7
            }
        else:
            shape_props = {'error': 'No object found in mask'}
        
        return shape_props
    
    def _extract_size_properties(
        self,
        mask: torch.Tensor,
        image: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract size-related properties.
        
        Args:
            mask: Binary mask
            image: Image tensor for reference
            
        Returns:
            Size properties dictionary
        """
        # Get image dimensions
        if len(image.shape) == 3:  # CHW
            _, h, w = image.shape
        else:  # BCHW
            _, _, h, w = image.shape
        
        # Calculate size metrics
        area_pixels = mask.sum().item()
        total_pixels = h * w
        relative_size = area_pixels / total_pixels
        
        # Get bounding box dimensions
        nonzero = torch.nonzero(mask)
        if len(nonzero) > 0:
            min_y = nonzero[:, 0].min().item()
            max_y = nonzero[:, 0].max().item()
            min_x = nonzero[:, 1].min().item()
            max_x = nonzero[:, 1].max().item()
            
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
        else:
            bbox_width = 0
            bbox_height = 0
        
        size_props = {
            'area_pixels': int(area_pixels),
            'relative_size': relative_size,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'diagonal_length': np.sqrt(bbox_width**2 + bbox_height**2),
            'size_category': self._categorize_size(relative_size)
        }
        
        return size_props
    
    def _categorize_size(self, relative_size: float) -> str:
        """
        Categorize object size.
        
        Args:
            relative_size: Size relative to image
            
        Returns:
            Size category string
        """
        if relative_size < 0.01:
            return 'tiny'
        elif relative_size < 0.05:
            return 'small'
        elif relative_size < 0.2:
            return 'medium'
        elif relative_size < 0.5:
            return 'large'
        else:
            return 'very_large'
    
    def _extract_position_properties(
        self,
        mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract position-related properties.
        
        Args:
            mask: Binary mask
            
        Returns:
            Position properties dictionary
        """
        h, w = mask.shape
        nonzero = torch.nonzero(mask)
        
        if len(nonzero) > 0:
            centroid_y = nonzero[:, 0].float().mean().item()
            centroid_x = nonzero[:, 1].float().mean().item()
            
            # Normalize to [0, 1]
            norm_x = centroid_x / w
            norm_y = centroid_y / h
            
            # Determine quadrant
            if norm_x < 0.5 and norm_y < 0.5:
                quadrant = 'top_left'
            elif norm_x >= 0.5 and norm_y < 0.5:
                quadrant = 'top_right'
            elif norm_x < 0.5 and norm_y >= 0.5:
                quadrant = 'bottom_left'
            else:
                quadrant = 'bottom_right'
            
            position_props = {
                'centroid': [centroid_x, centroid_y],
                'normalized_position': [norm_x, norm_y],
                'quadrant': quadrant,
                'distance_from_center': np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2),
                'is_centered': abs(norm_x - 0.5) < 0.1 and abs(norm_y - 0.5) < 0.1,
                'edge_proximity': min(norm_x, norm_y, 1-norm_x, 1-norm_y)
            }
        else:
            position_props = {'error': 'No object found in mask'}
        
        return position_props
    
    def _extract_appearance_features(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract high-level appearance features using the feature extractor.
        
        Args:
            image: Image tensor
            mask: Binary mask
            
        Returns:
            Appearance features dictionary
        """
        # Placeholder for feature extraction
        # In production, this would use the loaded model
        appearance_props = {
            'feature_vector': np.random.randn(128).tolist()[:10],  # Truncated for demo
            'semantic_category': 'object',  # Predicted category
            'confidence': 0.85,
            'visual_complexity': 0.6,
            'distinctiveness': 0.7,
            'material_type': 'solid',  # Could be: solid, transparent, metallic, etc.
        }
        
        return appearance_props
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        return ['image']  # Either mask or bbox is also required, checked in validate
    
    def get_optional_params(self) -> Dict[str, Any]:
        """Get dictionary of optional parameters with defaults."""
        return {
            'mask': None,
            'bbox': None,
            'properties': 'all'  # Can be 'all' or list of specific properties
        }


# Register the operation with the global registry
registry.register(
    'GET_PROPERTIES',
    GetPropertiesOperation,
    metadata={
        'description': 'Extract visual properties of an object or region',
        'category': 'analysis',
        'input_types': {
            'image': 'torch.Tensor or numpy.ndarray',
            'mask': 'Optional[torch.Tensor]',
            'bbox': 'Optional[List[int]]'
        },
        'output_types': {
            'color': 'Dict',
            'texture': 'Dict',
            'shape': 'Dict',
            'size': 'Dict',
            'position': 'Dict',
            'appearance': 'Dict'
        }
    }
)