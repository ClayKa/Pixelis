# core/dataloaders/sa1b_streaming_loader.py

"""
Streaming version of SA-1B dataset loader that handles massive JSON files efficiently.

This loader uses ijson for streaming JSON parsing to avoid loading the entire
multi-gigabyte sa_1b.json file into memory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import ijson

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Sa1bStreamingLoader(BaseLoader):
    """
    Memory-efficient loader for SA-1B dataset using streaming JSON parsing.
    
    This loader processes the massive sa_1b.json file in a streaming fashion,
    building the index without loading the entire file into memory.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SA-1B streaming loader.
        
        Expected config:
        {
            'name': 'sa1b',
            'path': '/path/to/sa1b/images',
            'annotation_file': '/path/to/sa1b/sa_1b.json'  # The massive JSON file
        }
        """
        # Validate config
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Sa1bStreamingLoader config must include '{key}'")
        
        self.name = config.get('name', 'sa1b')
        self.images_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Initialize storage for parsed data
        self.image_name_to_info = {}
        self.image_id_to_annotations = defaultdict(list)
        
        # Build index using streaming parser
        super().__init__(config)
        
        logger.info(f"Initialized SA-1B streaming loader with {len(self._index)} samples")
    
    def _build_index(self) -> List[str]:
        """
        Build index using streaming JSON parser to handle massive files.
        
        This method makes two passes through the JSON file:
        1. First pass: Build image_name_to_info mapping
        2. Second pass: Build image_id_to_annotations mapping
        
        Returns:
            List of image IDs that have both image files and annotations
        """
        logger.info("Starting streaming parse of SA-1B annotations...")
        
        # Phase 1: Stream parse to build image mapping
        logger.info("Phase 1: Building image index...")
        image_count = 0
        
        try:
            with open(self.annotation_file, 'rb') as f:
                # Parse the 'images' array in streaming fashion
                parser = ijson.items(f, 'images.item')
                for item in parser:
                    file_name = item.get('file_name', '')
                    if file_name:
                        self.image_name_to_info[file_name] = {
                            'id': item.get('id'),
                            'width': item.get('width'),
                            'height': item.get('height'),
                            'file_name': file_name
                        }
                        image_count += 1
                        
                        # Log progress every 10000 images
                        if image_count % 10000 == 0:
                            logger.info(f"  Processed {image_count} images...")
        
        except ijson.JSONError as e:
            logger.error(f"Error parsing images section: {e}")
            raise
        
        logger.info(f"Phase 1 complete: Found {image_count} images in annotations")
        
        # Phase 2: Stream parse to build annotations mapping
        logger.info("Phase 2: Building annotations index...")
        annotation_count = 0
        
        try:
            with open(self.annotation_file, 'rb') as f:
                # Parse the 'annotations' array in streaming fashion
                parser = ijson.items(f, 'annotations.item')
                for item in parser:
                    image_id = item.get('image_id')
                    if image_id is not None:
                        # Store minimal annotation data to save memory
                        annotation = {
                            'id': item.get('id'),
                            'category_id': item.get('category_id'),
                            'bbox': item.get('bbox'),
                            'area': item.get('area', 0),
                            'segmentation': item.get('segmentation'),
                            'iscrowd': item.get('iscrowd', 0)
                        }
                        self.image_id_to_annotations[image_id].append(annotation)
                        annotation_count += 1
                        
                        # Log progress every 100000 annotations
                        if annotation_count % 100000 == 0:
                            logger.info(f"  Processed {annotation_count} annotations...")
        
        except ijson.JSONError as e:
            logger.error(f"Error parsing annotations section: {e}")
            raise
        
        logger.info(f"Phase 2 complete: Found {annotation_count} annotations")
        
        # Phase 3: Match with local image files
        logger.info("Phase 3: Matching with local image files...")
        
        # Get all available image files
        available_images = set()
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.images_path.glob(ext):
                available_images.add(img_path.name)
        
        logger.info(f"Found {len(available_images)} local image files")
        
        # Build final index by matching annotations with available images
        final_index = []
        for img_name, img_info in self.image_name_to_info.items():
            if img_name in available_images:
                image_id = img_info['id']
                # Only include if we have annotations for this image
                if image_id in self.image_id_to_annotations:
                    final_index.append(image_id)
        
        logger.info(f"Final index contains {len(final_index)} samples with both images and annotations")
        
        # Sort for reproducibility
        final_index.sort()
        
        return final_index
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing image path and annotations
        """
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range for dataset with {len(self._index)} samples")
        
        image_id = self._index[index]
        
        # Find image info by ID
        image_info = None
        image_name = None
        for name, info in self.image_name_to_info.items():
            if info['id'] == image_id:
                image_info = info
                image_name = name
                break
        
        if not image_info:
            raise KeyError(f"Image ID {image_id} not found in image mapping")
        
        # Build image path
        image_path = self.images_path / image_name
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # Process segmentation masks
        segmentation_masks = []
        for ann in annotations:
            if ann.get('segmentation'):
                segmentation_masks.append({
                    'id': ann['id'],
                    'category_id': ann.get('category_id', 1),
                    'segmentation': ann['segmentation'],
                    'bbox': ann.get('bbox', []),
                    'area': ann.get('area', 0),
                    'iscrowd': ann.get('iscrowd', 0)
                })
        
        # Build sample
        sample = {
            'sample_id': f"sa1b_{image_id}",
            'media_path': str(image_path),
            'dataset': 'sa1b',
            
            # Image metadata
            'width': image_info.get('width', 0),
            'height': image_info.get('height', 0),
            
            # Annotations
            'annotations': {
                'segmentation_masks': segmentation_masks,
                'num_objects': len(segmentation_masks)
            }
        }
        
        return sample
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Estimate current memory usage of the loader.
        
        Returns:
            Dictionary with memory usage statistics
        """
        import sys
        
        # Estimate memory usage
        image_info_size = sys.getsizeof(self.image_name_to_info)
        annotations_size = sys.getsizeof(self.image_id_to_annotations)
        index_size = sys.getsizeof(self._index)
        
        # Estimate deep size for nested structures
        for annotations_list in self.image_id_to_annotations.values():
            annotations_size += sys.getsizeof(annotations_list)
            for ann in annotations_list:
                annotations_size += sys.getsizeof(ann)
        
        total_size = image_info_size + annotations_size + index_size
        
        return {
            'image_info_bytes': image_info_size,
            'annotations_bytes': annotations_size,
            'index_bytes': index_size,
            'total_bytes': total_size,
            'total_mb': total_size / (1024 * 1024)
        }


class Sa1bStreamingSegmentLoader(Sa1bStreamingLoader):
    """
    Streaming SA-1B loader specifically for segmentation tasks.
    
    This variant focuses on segmentation annotations and provides
    additional utilities for mask processing.
    """
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get item with enhanced segmentation information.
        """
        sample = super().get_item(index)
        
        # Add segmentation-specific fields
        if 'annotations' in sample and 'segmentation_masks' in sample['annotations']:
            masks = sample['annotations']['segmentation_masks']
            
            # Add statistics
            sample['annotations']['mask_stats'] = {
                'num_masks': len(masks),
                'total_area': sum(m.get('area', 0) for m in masks),
                'has_crowd': any(m.get('iscrowd', 0) for m in masks)
            }
            
            # Categorize masks by size
            small_masks = [m for m in masks if m.get('area', 0) < 1000]
            medium_masks = [m for m in masks if 1000 <= m.get('area', 0) < 10000]
            large_masks = [m for m in masks if m.get('area', 0) >= 10000]
            
            sample['annotations']['mask_distribution'] = {
                'small': len(small_masks),
                'medium': len(medium_masks),
                'large': len(large_masks)
            }
        
        return sample