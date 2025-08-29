# core/dataloaders/part_imagenet_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from PIL import Image

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PartImageNetLoader(BaseLoader):
    """
    [REVISED]
    A concrete data loader for the PartImageNet dataset, adapted for a setup
    without a central metadata file. It pairs images with PNG annotation masks
    based on matching filenames and extracts raw instance IDs from the masks.
    
    This loader handles PartImageNet's structure:
    1. A directory of original JPEG images
    2. A parallel directory of PNG segmentation masks
    
    Each mask's pixel values directly encode part/instance IDs.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        [REVISED]
        Initialize the PartImageNetLoader.
        
        Args:
            config: Configuration dictionary containing 'image_path' and 'mask_path'
        """
        # Validate required config keys before calling super().__init__
        required_keys = ['image_path', 'mask_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"PartImageNetLoader config must include '{key}'")
        
        # Set up paths before calling super().__init__
        self.image_dir = Path(config['image_path'])
        self.mask_dir = Path(config['mask_path'])
        
        # Validate paths exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        # [REVISED] The __init__ method is now simplified, as it no longer
        # needs to load a metadata file. The parent's __init__ is sufficient.
        super().__init__(config)

    def _build_index(self) -> List[Tuple[Path, Path]]:
        """
        Scans the image and mask directories, finds all corresponding
        image-mask pairs based on identical filenames (sans extension),
        and builds an index of these pairs.
        
        Returns:
            List of tuples containing (image_path, mask_path) pairs
        """
        # Find all image files (support multiple extensions)
        image_files = []
        for pattern in ['**/*.jpg', '**/*.jpeg', '**/*.JPG', '**/*.JPEG']:
            image_files.extend(self.image_dir.glob(pattern))
        
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files")
        
        # Build index by pairing images with masks
        index = []
        matched_count = 0
        
        for image_file in image_files:
            # Construct corresponding mask path
            relative_path = image_file.relative_to(self.image_dir)
            mask_file = (self.mask_dir / relative_path).with_suffix('.png')
            
            # Only include if mask file exists
            if mask_file.is_file():
                index.append((image_file, mask_file))
                matched_count += 1
            else:
                logger.debug(f"No mask found for {image_file.name}")
        
        logger.info(f"Successfully matched {matched_count} images with masks")
        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        [REVISED]
        Retrieves a single image-mask pair and extracts all available segmentation
        information directly from the mask's raw pixel values.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with mask path and instance IDs
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get image and mask paths
        image_path, mask_path = self._index[index]
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=image_path.stem,
            media_path=image_path,
            media_type="image"
        )
        
        # [REVISED LOGIC]
        # Instead of looking up part labels, we parse the mask to provide the raw
        # instance IDs (pixel values) available for this image.
        try:
            with Image.open(mask_path) as mask_image:
                mask_array = np.array(mask_image)
            
            # Find all unique non-zero pixel values. Each value is a part/object ID.
            # Note: In PartImageNet, 158 is commonly used as background, so we exclude it too
            unique_values = np.unique(mask_array)
            # Filter out background values (0 and commonly 158)
            instance_ids = [int(val) for val in unique_values if val not in [0, 158]]
            
            # The standardized annotation now provides the mask's path and the list of
            # raw instance IDs it contains.
            sample['annotations']['segmentation_mask_path'] = str(mask_path.resolve())
            sample['annotations']['available_instance_ids'] = instance_ids
            
            # Also provide mask shape for reference
            sample['annotations']['mask_shape'] = mask_array.shape
            
            # Optionally provide more detailed information about each instance
            instance_info = []
            for instance_id in instance_ids:
                instance_mask = (mask_array == instance_id).astype(np.uint8)
                bbox = self._calculate_bbox_from_mask(instance_mask)
                area = int(np.sum(instance_mask))
                
                instance_info.append({
                    'instance_id': instance_id,
                    'bbox': bbox,
                    'area': area,
                    'pixel_ratio': float(area / mask_array.size)
                })
            
            sample['annotations']['instance_details'] = instance_info
            sample['annotations']['num_instances'] = len(instance_ids)
            
        except Exception as e:
            logger.error(f"Failed to parse mask file {mask_path} for sample {image_path.stem}. Error: {e}")
            sample['annotations']['segmentation_mask_path'] = None
            sample['annotations']['available_instance_ids'] = []
            sample['annotations']['instance_details'] = []
            sample['annotations']['num_instances'] = 0
            sample['annotations']['error'] = str(e)
        
        return sample

    def _calculate_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """Calculate bounding box [x, y, width, height] from binary mask."""
        if not np.any(mask):
            return [0.0, 0.0, 0.0, 0.0]
        
        # Find coordinates of object pixels
        rows, cols = np.where(mask > 0)
        
        if len(rows) == 0 or len(cols) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Calculate bounding box
        x_min, x_max = float(np.min(cols)), float(np.max(cols))
        y_min, y_max = float(np.min(rows)), float(np.max(rows))
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return [x_min, y_min, width, height]

    def get_samples_by_class(self, class_id: str) -> List[Dict[str, Any]]:
        """
        Get all samples for a specific ImageNet class ID.
        
        Args:
            class_id: ImageNet class ID (e.g., 'n01440764')
            
        Returns:
            List of sample dictionaries for the specified class
        """
        samples = []
        for i in range(len(self)):
            image_path, _ = self._index[i]
            sample_class = image_path.stem.split('_')[0]
            if sample_class == class_id:
                samples.append(self.get_item(i))
        
        return samples

    def get_class_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the classes in the dataset.
        
        Returns:
            Dictionary with class distribution and statistics
        """
        class_counts = {}
        total_samples = len(self._index)
        
        for image_path, _ in self._index:
            class_id = image_path.stem.split('_')[0]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Calculate statistics
        counts = list(class_counts.values())
        
        return {
            'total_samples': total_samples,
            'total_classes': len(class_counts),
            'class_distribution': class_counts,
            'samples_per_class': {
                'min': min(counts) if counts else 0,
                'max': max(counts) if counts else 0,
                'avg': sum(counts) / len(counts) if counts else 0
            },
            'top_classes': sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def get_mask_statistics(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze mask properties across a sample of the dataset.
        
        Args:
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary with mask statistics
        """
        if not self._index:
            return {'error': 'No samples available'}
        
        sample_indices = np.linspace(0, len(self._index) - 1, 
                                   min(sample_size, len(self._index)), dtype=int)
        
        instance_counts = []
        all_instance_ids = []
        instance_area_ratios = []
        
        for idx in sample_indices:
            try:
                _, mask_path = self._index[idx]
                mask_array = np.array(Image.open(mask_path))
                
                # Get all non-zero values (instance IDs), excluding common background (158)
                unique_vals = np.unique(mask_array)
                instance_ids = [int(val) for val in unique_vals if val not in [0, 158]]
                instance_counts.append(len(instance_ids))
                all_instance_ids.extend(instance_ids)
                
                # Calculate area ratios for each instance
                for instance_id in instance_ids:
                    area_ratio = np.sum(mask_array == instance_id) / mask_array.size
                    instance_area_ratios.append(area_ratio)
                    
            except Exception as e:
                logger.debug(f"Error analyzing mask at index {idx}: {e}")
                continue
        
        return {
            'samples_analyzed': len(sample_indices),
            'instance_count_distribution': {
                'min': min(instance_counts) if instance_counts else 0,
                'max': max(instance_counts) if instance_counts else 0,
                'avg': np.mean(instance_counts) if instance_counts else 0,
                'std': np.std(instance_counts) if instance_counts else 0
            },
            'instance_ids': {
                'unique_count': len(set(all_instance_ids)),
                'range': [min(all_instance_ids), max(all_instance_ids)] if all_instance_ids else [0, 0]
            },
            'instance_area_ratios': {
                'min': min(instance_area_ratios) if instance_area_ratios else 0,
                'max': max(instance_area_ratios) if instance_area_ratios else 0,
                'avg': np.mean(instance_area_ratios) if instance_area_ratios else 0,
                'std': np.std(instance_area_ratios) if instance_area_ratios else 0
            }
        }