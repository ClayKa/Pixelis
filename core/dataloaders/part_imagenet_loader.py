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
    A concrete data loader for the PartImageNet dataset.

    This loader is designed to handle PartImageNet's structure, which consists of:
    1. A directory of original JPEG images
    2. A parallel directory of PNG files for segmentation masks, where pixel
       values encode object vs. background (binary segmentation)
    
    Each PNG mask contains exactly 2 unique pixel values:
    - Background pixels (typically value 158)
    - Foreground/object pixels (class-specific value)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PartImageNetLoader.
        
        Args:
            config: Configuration dictionary containing 'path' and 'annotation_path'
        """
        # Validate required config keys before calling super().__init__
        required_keys = ['path', 'annotation_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"PartImageNetLoader config must include '{key}'")
        
        # Set up paths before calling super().__init__
        self.images_path = Path(config['path'])
        self.annotation_path = Path(config['annotation_path'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotation_path}")
        
        # Load optional metadata file for part ID to label mapping
        self.part_id_to_label = {}
        if 'metadata_file' in config:
            metadata_path = Path(config['metadata_file'])
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.part_id_to_label = json.load(f)
                    logger.info(f"Loaded part ID mappings from {metadata_path}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not load metadata file {metadata_path}: {e}")
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Tuple[Path, Path]]:
        """
        Build index by pairing images with their corresponding PNG annotation masks.
        
        Returns:
            List of tuples containing (image_path, annotation_path) pairs
        """
        # Find all image files (JPEG format)
        image_files = list(self.images_path.glob('*.JPEG'))
        if not image_files:
            # Fallback to other common extensions
            for ext in ['*.jpg', '*.jpeg', '*.JPG']:
                image_files.extend(self.images_path.glob(ext))
        
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files")
        
        # Build index by pairing images with annotations
        index = []
        matched_count = 0
        
        for image_path in image_files:
            # Construct expected annotation path
            annotation_file = self.annotation_path / f"{image_path.stem}.png"
            
            # Only include if annotation file exists
            if annotation_file.exists():
                index.append((image_path, annotation_file))
                matched_count += 1
            else:
                logger.debug(f"No annotation found for {image_path.name}")
        
        logger.info(f"Successfully matched {matched_count} images with annotations")
        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sample by index with binary segmentation mask.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with binary segmentation mask
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get image and annotation paths
        image_path, annotation_path = self._index[index]
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=image_path.stem,
            media_path=image_path,
            media_type="image"
        )
        
        # Parse the PNG annotation mask
        try:
            # Load PNG mask and convert to numpy array
            mask_image = Image.open(annotation_path)
            mask_array = np.array(mask_image)
            
            # Get unique pixel values
            unique_values = np.unique(mask_array)
            
            if len(unique_values) < 2:
                logger.warning(f"Mask {annotation_path} has only {len(unique_values)} unique values")
                # Create empty annotations for invalid masks
                sample['annotations'].update({
                    'part_level_segmentation': [],
                    'num_parts': 0,
                    'mask_info': {
                        'unique_values': unique_values.tolist(),
                        'background_value': None,
                        'object_value': None
                    }
                })
                return sample
            
            # Identify background and object values
            # Background is typically the most frequent value (often 158)
            value_counts = [(val, np.sum(mask_array == val)) for val in unique_values]
            value_counts.sort(key=lambda x: x[1], reverse=True)
            
            background_value = value_counts[0][0]
            object_value = value_counts[1][0] if len(value_counts) > 1 else unique_values[unique_values != background_value][0]
            
            # Create binary mask for the object
            object_mask = (mask_array == object_value).astype(np.uint8)
            
            # Calculate bounding box and area
            bbox = self._calculate_bbox_from_mask(object_mask)
            area = int(np.sum(object_mask))
            
            # Get class label from filename (ImageNet class ID)
            class_id = image_path.stem.split('_')[0]  # e.g., 'n01440764' from 'n01440764_10029'
            part_label = self.part_id_to_label.get(class_id, class_id)
            
            # Create standardized annotation
            part_annotation = {
                'annotation_id': 0,  # Single object per image
                'class_id': class_id,
                'part_label': part_label,
                'pixel_value': int(object_value),
                'bbox': bbox,
                'area': area,
                'segmentation_mask': object_mask,
                'mask_shape': object_mask.shape
            }
            
            # Add PartImageNet specific annotations
            sample['annotations'].update({
                'part_level_segmentation': [part_annotation],
                'num_parts': 1,
                'mask_info': {
                    'unique_values': unique_values.tolist(),
                    'background_value': int(background_value),
                    'object_value': int(object_value),
                    'mask_size': mask_array.shape,
                    'background_ratio': float(value_counts[0][1] / mask_array.size),
                    'object_ratio': float(area / mask_array.size)
                },
                'dataset_info': {
                    'task_type': 'binary_segmentation',
                    'source': 'PartImageNet',
                    'class_id': class_id,
                    'has_hierarchical_parts': False,
                    'encoding': 'binary_pixel_values'
                }
            })
            
        except (IOError, OSError) as e:
            logger.error(f"Error loading mask {annotation_path}: {e}")
            # Return sample with empty annotations
            sample['annotations'].update({
                'part_level_segmentation': [],
                'num_parts': 0,
                'mask_info': {'error': str(e)}
            })
        
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
        
        background_values = []
        object_values = []
        object_ratios = []
        unique_value_counts = []
        
        for idx in sample_indices:
            try:
                _, annotation_path = self._index[idx]
                mask_array = np.array(Image.open(annotation_path))
                unique_values = np.unique(mask_array)
                unique_value_counts.append(len(unique_values))
                
                if len(unique_values) >= 2:
                    value_counts = [(val, np.sum(mask_array == val)) for val in unique_values]
                    value_counts.sort(key=lambda x: x[1], reverse=True)
                    
                    background_val = value_counts[0][0]
                    object_val = value_counts[1][0]
                    
                    background_values.append(background_val)
                    object_values.append(object_val)
                    
                    object_ratio = value_counts[1][1] / mask_array.size
                    object_ratios.append(object_ratio)
                    
            except Exception as e:
                logger.debug(f"Error analyzing mask at index {idx}: {e}")
                continue
        
        return {
            'samples_analyzed': len(sample_indices),
            'unique_value_distribution': {
                'min': min(unique_value_counts) if unique_value_counts else 0,
                'max': max(unique_value_counts) if unique_value_counts else 0,
                'avg': np.mean(unique_value_counts) if unique_value_counts else 0
            },
            'background_values': {
                'unique': list(set(background_values)),
                'most_common': max(set(background_values), key=background_values.count) if background_values else None
            },
            'object_values': {
                'range': [min(object_values), max(object_values)] if object_values else [0, 0],
                'unique_count': len(set(object_values))
            },
            'object_ratio_stats': {
                'min': min(object_ratios) if object_ratios else 0,
                'max': max(object_ratios) if object_ratios else 0,
                'avg': np.mean(object_ratios) if object_ratios else 0,
                'std': np.std(object_ratios) if object_ratios else 0
            }
        }