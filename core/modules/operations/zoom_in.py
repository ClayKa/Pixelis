"""
Zoom In Operation

Implements the ZOOM_IN visual operation for focusing on specific image regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from ..operation_registry import BaseOperation, registry


class ZoomInOperation(BaseOperation):
    """
    Zooms into a specific region of an image with optional enhancement.
    
    This operation crops and potentially upscales a region of interest,
    allowing for detailed examination of specific image areas.
    """
    
    def __init__(self):
        """Initialize the zoom in operation."""
        super().__init__()
        self.upscaler = None  # Will be loaded lazily for super-resolution
    
    def _load_upscaler(self):
        """
        Lazily load the upscaling/super-resolution model.
        
        This could be ESRGAN, Real-ESRGAN, or another upscaling model.
        """
        if self.upscaler is None:
            self.logger.info("Loading upscaling model...")
            # Placeholder for actual model loading
            # Example: self.upscaler = load_esrgan_model()
            self.upscaler = "placeholder_upscaler"
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input arguments.
        
        Required:
            - image: Image tensor or array
            - Either 'center' + 'zoom_factor' or 'region'
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if 'image' not in kwargs:
            self.logger.error("Missing required parameter: 'image'")
            return False
        
        # Check for zoom specification
        has_center_zoom = 'center' in kwargs and 'zoom_factor' in kwargs
        has_region = 'region' in kwargs
        
        if not has_center_zoom and not has_region:
            self.logger.error(
                "Must provide either ('center' and 'zoom_factor') or 'region'"
            )
            return False
        
        # Validate center format
        if 'center' in kwargs:
            center = kwargs['center']
            if not isinstance(center, (tuple, list)) or len(center) != 2:
                self.logger.error("'center' must be (x, y) tuple")
                return False
        
        # Validate zoom factor
        if 'zoom_factor' in kwargs:
            zoom = kwargs['zoom_factor']
            if not isinstance(zoom, (int, float)) or zoom <= 0:
                self.logger.error("'zoom_factor' must be positive number")
                return False
        
        # Validate region format
        if 'region' in kwargs:
            region = kwargs['region']
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                self.logger.error("'region' must be [x1, y1, x2, y2]")
                return False
        
        return True
    
    def preprocess(self, **kwargs) -> Dict[str, Any]:
        """
        Preprocess inputs and calculate zoom region.
        
        Args:
            **kwargs: Raw input arguments
            
        Returns:
            Preprocessed arguments with calculated region
        """
        processed = kwargs.copy()
        
        # Convert image to tensor
        image = kwargs['image']
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = torch.from_numpy(image).unsqueeze(0)
            elif len(image.shape) == 3:  # HWC
                image = torch.from_numpy(image).permute(2, 0, 1)
            processed['image'] = image.float()
        
        # Get image dimensions
        if len(processed['image'].shape) == 3:  # CHW
            _, h, w = processed['image'].shape
        else:  # BCHW
            _, _, h, w = processed['image'].shape
        
        # Calculate region from center and zoom_factor if needed
        if 'center' in kwargs and 'zoom_factor' in kwargs:
            center_x, center_y = kwargs['center']
            zoom_factor = kwargs['zoom_factor']
            
            # Calculate zoom window size
            window_w = int(w / zoom_factor)
            window_h = int(h / zoom_factor)
            
            # Calculate region bounds
            x1 = int(center_x - window_w / 2)
            y1 = int(center_y - window_h / 2)
            x2 = int(center_x + window_w / 2)
            y2 = int(center_y + window_h / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            
            processed['region'] = [x1, y1, x2, y2]
        
        # Validate region bounds
        if 'region' in processed:
            x1, y1, x2, y2 = processed['region']
            x1 = max(0, min(x1, w))
            x2 = max(x1 + 1, min(x2, w))  # Ensure x2 > x1
            y1 = max(0, min(y1, h))
            y2 = max(y1 + 1, min(y2, h))  # Ensure y2 > y1
            processed['region'] = [int(x1), int(y1), int(x2), int(y2)]
        
        return processed
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the zoom operation.
        
        Args:
            image: Image tensor (CHW or BCHW format)
            center: (x, y) center point for zoom
            zoom_factor: Zoom level (e.g., 2.0 for 2x zoom)
            region: Alternative [x1, y1, x2, y2] region specification
            target_size: Optional target output size (width, height)
            enhance: Whether to apply super-resolution enhancement
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Dictionary containing:
                - zoomed_image: The zoomed/cropped image
                - region: The actual region that was zoomed [x1, y1, x2, y2]
                - zoom_level: Effective zoom level applied
                - original_size: Original image dimensions
                - output_size: Output image dimensions
                - metadata: Additional zoom metadata
        """
        # Validate and preprocess
        if not self.validate_inputs(**kwargs):
            raise ValueError("Invalid inputs for zoom operation")
        
        processed = self.preprocess(**kwargs)
        
        # Extract parameters
        image = processed['image']
        region = processed['region']
        target_size = processed.get('target_size', None)
        enhance = processed.get('enhance', False)
        maintain_aspect = processed.get('maintain_aspect', True)
        interpolation_mode = processed.get('interpolation', 'bilinear')
        
        # Get original dimensions
        if len(image.shape) == 3:  # CHW
            c, h, w = image.shape
            batch_dim = False
        else:  # BCHW
            b, c, h, w = image.shape
            batch_dim = True
        
        original_size = (w, h)
        
        # Extract region
        x1, y1, x2, y2 = region
        if batch_dim:
            zoomed = image[:, :, y1:y2, x1:x2]
        else:
            zoomed = image[:, y1:y2, x1:x2]
        
        # Calculate effective zoom level
        region_w = x2 - x1
        region_h = y2 - y1
        zoom_x = w / region_w if region_w > 0 else 1
        zoom_y = h / region_h if region_h > 0 else 1
        effective_zoom = (zoom_x + zoom_y) / 2
        
        # Determine output size
        if target_size is not None:
            target_w, target_h = target_size
            
            if maintain_aspect:
                # Maintain aspect ratio
                aspect = region_w / region_h
                if target_w / target_h > aspect:
                    target_w = int(target_h * aspect)
                else:
                    target_h = int(target_w / aspect)
        else:
            # Default: scale back to original image size
            target_w = w
            target_h = h
        
        # Resize if needed
        current_h = y2 - y1
        current_w = x2 - x1
        
        if current_w != target_w or current_h != target_h:
            # Add batch dimension if needed for interpolation
            if not batch_dim:
                zoomed = zoomed.unsqueeze(0)
            
            # Perform interpolation
            zoomed = F.interpolate(
                zoomed,
                size=(target_h, target_w),
                mode=interpolation_mode,
                align_corners=False if interpolation_mode in ['bilinear', 'bicubic'] else None
            )
            
            # Remove batch dimension if it was added
            if not batch_dim:
                zoomed = zoomed.squeeze(0)
        
        # Apply enhancement if requested
        if enhance:
            zoomed = self._apply_enhancement(zoomed)
        
        # Calculate focus metrics
        focus_metrics = self._calculate_focus_metrics(zoomed)
        
        # Prepare result
        result = {
            'zoomed_image': zoomed,
            'region': region,
            'zoom_level': effective_zoom,
            'original_size': original_size,
            'output_size': (target_w, target_h),
            'metadata': {
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'region_size': (region_w, region_h),
                'scale_factor': target_w / region_w if region_w > 0 else 1,
                'enhanced': enhance,
                'focus_quality': focus_metrics
            }
        }
        
        self.logger.debug(
            f"Zoomed to region {region} with {effective_zoom:.2f}x zoom, "
            f"output size {target_w}x{target_h}"
        )
        
        return result
    
    def _apply_enhancement(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply super-resolution enhancement to the zoomed image.
        
        Args:
            image: Zoomed image tensor
            
        Returns:
            Enhanced image tensor
        """
        # Load upscaler if needed
        if self.upscaler is None:
            self._load_upscaler()
        
        # Placeholder for actual enhancement
        # In production, this would use the upscaling model
        # Example: enhanced = self.upscaler.enhance(image)
        
        # For now, apply simple sharpening
        enhanced = self._sharpen_image(image)
        
        return enhanced
    
    def _sharpen_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply simple sharpening filter.
        
        Args:
            image: Image tensor
            
        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        
        # Reshape kernel for conv2d
        kernel = kernel.view(1, 1, 3, 3)
        
        # Apply to each channel
        if len(image.shape) == 3:  # CHW
            c, h, w = image.shape
            sharpened = []
            for i in range(c):
                channel = image[i:i+1].unsqueeze(0)  # 1x1xHxW
                sharp_channel = F.conv2d(channel, kernel, padding=1)
                sharpened.append(sharp_channel.squeeze(0))
            result = torch.stack(sharpened, dim=0)
        else:  # BCHW
            b, c, h, w = image.shape
            # Apply per channel
            kernel = kernel.repeat(c, 1, 1, 1)  # CxCx3x3
            result = F.conv2d(image, kernel, padding=1, groups=c)
        
        # Clip values to valid range
        result = torch.clamp(result, 0, 255) if image.max() > 1 else torch.clamp(result, 0, 1)
        
        return result
    
    def _calculate_focus_metrics(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Calculate focus quality metrics for the zoomed region.
        
        Args:
            image: Zoomed image tensor
            
        Returns:
            Dictionary of focus metrics
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[0] > 1:  # CHW with multiple channels
            gray = torch.mean(image, dim=0)
        elif len(image.shape) == 4 and image.shape[1] > 1:  # BCHW
            gray = torch.mean(image, dim=1)
        else:
            gray = image
        
        # Calculate Laplacian for sharpness
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        if len(gray.shape) == 2:  # HW
            gray = gray.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        elif len(gray.shape) == 3:  # 1xHxW or BxHxW
            if gray.shape[0] != 1:
                gray = gray.unsqueeze(1)  # Bx1xHxW
            else:
                gray = gray.unsqueeze(0)  # 1x1xHxW
        
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
        sharpness = laplacian.var().item()
        
        # Calculate contrast
        min_val = image.min().item()
        max_val = image.max().item()
        contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
        
        # Calculate edge density
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = (edge_magnitude > edge_magnitude.mean()).float().mean().item()
        
        metrics = {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'quality_score': float((sharpness * contrast * edge_density) ** (1/3))  # Geometric mean
        }
        
        return metrics
    
    def create_zoom_sequence(
        self,
        image: torch.Tensor,
        center: Tuple[int, int],
        zoom_levels: List[float],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a sequence of zoomed images at different levels.
        
        Args:
            image: Source image
            center: Center point for zoom
            zoom_levels: List of zoom factors
            **kwargs: Additional parameters for zoom operation
            
        Returns:
            List of zoom results
        """
        sequence = []
        
        for zoom_factor in zoom_levels:
            result = self.run(
                image=image,
                center=center,
                zoom_factor=zoom_factor,
                **kwargs
            )
            result['zoom_factor_requested'] = zoom_factor
            sequence.append(result)
        
        return sequence
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        return ['image']  # Plus either center+zoom_factor or region
    
    def get_optional_params(self) -> Dict[str, Any]:
        """Get dictionary of optional parameters with defaults."""
        return {
            'center': None,
            'zoom_factor': None,
            'region': None,
            'target_size': None,
            'enhance': False,
            'maintain_aspect': True,
            'interpolation': 'bilinear'  # Options: nearest, bilinear, bicubic
        }


# Register the operation with the global registry
registry.register(
    'ZOOM_IN',
    ZoomInOperation,
    metadata={
        'description': 'Zoom into a specific region of an image',
        'category': 'transformation',
        'input_types': {
            'image': 'torch.Tensor or numpy.ndarray',
            'center': 'Optional[Tuple[int, int]]',
            'zoom_factor': 'Optional[float]',
            'region': 'Optional[List[int]]'
        },
        'output_types': {
            'zoomed_image': 'torch.Tensor',
            'region': 'List[int]',
            'zoom_level': 'float',
            'metadata': 'Dict'
        }
    }
)