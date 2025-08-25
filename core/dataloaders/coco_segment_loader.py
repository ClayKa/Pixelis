# core/dataloaders/coco_segment_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class CocoSegmentLoader(BaseLoader):
    """
    A concrete data loader for the COCO 2017 dataset, focused on instance segmentation.
    
    This loader reads from the shared COCO+LVIS image directory but only parses 
    COCO-specific annotations. It's optimized for segmentation and property-based 
    tasks, providing clean instance segmentation data with COCO's 80 categories.
    
    Key features:
    - Loads COCO 2017 instance segmentation annotations
    - Efficient lookup structures for image metadata and annotations
    - Category information with human-readable names
    - Polygon and RLE segmentation support
    - Quality filtering based on area and crowd flags
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CocoSegmentLoader.
        
        Args:
            config: Configuration dictionary containing 'path', 'annotation_file'
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"CocoSegmentLoader config must include '{key}'")
        
        self.images_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Configuration options for filtering
        self.min_area = config.get('min_area', 0)  # Minimum area threshold
        self.include_crowd = config.get('include_crowd', False)  # Include crowd annotations
        
        # Initialize lookup structures (populated in _build_index)
        self._image_id_to_info = {}
        self._image_id_to_annotations = defaultdict(list)
        self._category_id_to_info = {}
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[int]:
        """
        Load COCO annotations and build efficient lookup structures.
        
        Returns:
            List of image IDs that have annotations
        """
        logger.info(f"Loading COCO annotations from {self.annotation_file}")
        
        # Load COCO JSON file
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Build image lookup
        images = coco_data.get('images', [])
        for img in images:
            self._image_id_to_info[img['id']] = img
        
        logger.info(f"Loaded {len(images)} image entries")
        
        # Build category lookup
        categories = coco_data.get('categories', [])
        for cat in categories:
            self._category_id_to_info[cat['id']] = cat
        
        logger.info(f"Loaded {len(categories)} categories")
        
        # Build annotations lookup with filtering
        annotations = coco_data.get('annotations', [])
        valid_image_ids = set()
        total_annotations = 0
        filtered_annotations = 0
        
        for ann in annotations:
            # Apply filters
            if ann.get('area', 0) < self.min_area:
                continue
            
            if not self.include_crowd and ann.get('iscrowd', 0):
                continue
            
            # Check if image exists in our image set
            image_id = ann['image_id']
            if image_id in self._image_id_to_info:
                self._image_id_to_annotations[image_id].append(ann)
                valid_image_ids.add(image_id)
                filtered_annotations += 1
            
            total_annotations += 1
        
        logger.info(f"Filtered {filtered_annotations}/{total_annotations} annotations")
        logger.info(f"Found {len(valid_image_ids)} images with valid annotations")
        
        # Apply additional filtering: only include images that actually exist
        existing_image_ids = []
        for image_id in valid_image_ids:
            image_info = self._image_id_to_info[image_id]
            image_path = self.images_path / image_info['file_name']
            if image_path.exists():
                existing_image_ids.append(image_id)
        
        logger.info(f"Found {len(existing_image_ids)} images that exist on disk")
        
        return sorted(existing_image_ids)

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sample by index with COCO segmentation annotations.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with COCO instance segmentation
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get image ID and info
        image_id = self._index[index]
        image_info = self._image_id_to_info[image_id]
        
        # Construct image path
        image_path = self.images_path / image_info['file_name']
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=str(image_id),
            media_path=image_path,
            media_type="image"
        )
        
        # Process COCO annotations
        raw_annotations = self._image_id_to_annotations[image_id]
        instance_segmentations = []
        total_area = 0
        category_counts = defaultdict(int)
        
        for ann in raw_annotations:
            category_id = ann['category_id']
            category_info = self._category_id_to_info.get(category_id, {})
            
            # Process segmentation data
            segmentation = ann.get('segmentation', [])
            segmentation_type = 'polygon'
            if isinstance(segmentation, dict):
                # RLE format
                segmentation_type = 'rle'
            elif isinstance(segmentation, list) and len(segmentation) > 0:
                # Polygon format
                segmentation_type = 'polygon'
            
            # Create instance annotation
            instance_annotation = {
                'annotation_id': ann['id'],
                'category_id': category_id,
                'category_name': category_info.get('name', f'category_{category_id}'),
                'supercategory': category_info.get('supercategory', 'unknown'),
                'area_pixels': ann.get('area', 0),
                'bbox': ann.get('bbox', []),  # [x, y, width, height]
                'segmentation': segmentation,
                'segmentation_type': segmentation_type,
                'is_crowd': bool(ann.get('iscrowd', 0)),
                'center_point': self._calculate_center_point(ann.get('bbox', [])),
                'geometric_properties': {
                    'width': ann.get('bbox', [0, 0, 0, 0])[2] if len(ann.get('bbox', [])) >= 3 else 0,
                    'height': ann.get('bbox', [0, 0, 0, 0])[3] if len(ann.get('bbox', [])) >= 4 else 0,
                    'aspect_ratio': self._calculate_aspect_ratio(ann.get('bbox', [])),
                    'relative_area': ann.get('area', 0) / (image_info['width'] * image_info['height']) if image_info.get('width', 0) > 0 and image_info.get('height', 0) > 0 else 0
                }
            }
            
            instance_segmentations.append(instance_annotation)
            total_area += ann.get('area', 0)
            category_counts[category_info.get('name', f'category_{category_id}')] += 1
        
        # Add COCO-specific annotations
        sample['annotations'].update({
            'coco_instance_segmentation': instance_segmentations,
            'num_instances': len(instance_segmentations),
            'total_segmented_area': total_area,
            'coverage_ratio': total_area / (image_info['width'] * image_info['height']) if image_info.get('width', 0) > 0 and image_info.get('height', 0) > 0 else 0,
            'category_distribution': dict(category_counts),
            'unique_categories': len(category_counts),
            'image_metadata': {
                'coco_image_id': image_id,
                'file_name': image_info['file_name'],
                'width': image_info.get('width', 0),
                'height': image_info.get('height', 0),
                'license': image_info.get('license'),
                'coco_url': image_info.get('coco_url'),
                'date_captured': image_info.get('date_captured')
            },
            'dataset_info': {
                'task_type': 'coco_instance_segmentation',
                'source': 'COCO2017',
                'suitable_for_segment_object_at': True,
                'suitable_for_get_properties': True,
                'has_category_names': True,
                'has_supercategories': True,
                'num_categories': len(self._category_id_to_info),
                'segmentation_formats': ['polygon', 'rle'],
                'filtering_applied': {
                    'min_area': self.min_area,
                    'include_crowd': self.include_crowd
                }
            }
        })
        
        return sample

    def _calculate_center_point(self, bbox: List[float]) -> List[float]:
        """Calculate center point from bounding box [x, y, width, height]."""
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return [0.0, 0.0]
        return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]

    def _calculate_aspect_ratio(self, bbox: List[float]) -> float:
        """Calculate aspect ratio from bounding box [x, y, width, height]."""
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return 0.0
        return bbox[2] / bbox[3]  # width / height

    def get_samples_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        """
        Get samples containing instances of a specific category.
        
        Args:
            category_name: Name of the category (e.g., 'person', 'car', 'dog')
            
        Returns:
            List of samples containing the specified category
        """
        matching_samples = []
        
        for i in range(len(self)):
            sample = self.get_item(i)
            category_dist = sample['annotations']['category_distribution']
            
            if category_name in category_dist:
                matching_samples.append(sample)
        
        return matching_samples

    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about category distribution in the dataset.
        
        Returns:
            Dictionary with category statistics
        """
        category_counts = defaultdict(int)
        total_instances = 0
        
        # Sample a subset for efficiency
        sample_size = min(1000, len(self._index))
        indices = range(0, len(self._index), max(1, len(self._index) // sample_size))
        
        for i in indices:
            image_id = self._index[i]
            annotations = self._image_id_to_annotations[image_id]
            
            for ann in annotations:
                category_id = ann['category_id']
                category_info = self._category_id_to_info.get(category_id, {})
                category_name = category_info.get('name', f'category_{category_id}')
                category_counts[category_name] += 1
                total_instances += 1
        
        # Scale up to full dataset
        scaling_factor = len(self._index) / sample_size if sample_size > 0 else 1
        
        return {
            'total_categories': len(self._category_id_to_info),
            'total_instances_estimated': int(total_instances * scaling_factor),
            'samples_analyzed': sample_size,
            'category_distribution': dict(category_counts),
            'most_common_categories': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'rare_categories': sorted(category_counts.items(), key=lambda x: x[1])[:10],
            'available_categories': list(self._category_id_to_info.values())
        }

    def get_samples_by_complexity(self, min_instances: int = 1, max_instances: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get samples filtered by number of instances (complexity).
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances (None for no limit)
            
        Returns:
            List of samples meeting the complexity criteria
        """
        matching_samples = []
        
        for i in range(len(self)):
            image_id = self._index[i]
            num_instances = len(self._image_id_to_annotations[image_id])
            
            if num_instances >= min_instances:
                if max_instances is None or num_instances <= max_instances:
                    sample = self.get_item(i)
                    matching_samples.append(sample)
        
        return matching_samples

    def get_supercategory_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of instances by supercategory.
        
        Returns:
            Dictionary with supercategory statistics
        """
        supercategory_counts = defaultdict(int)
        
        # Sample for efficiency
        sample_size = min(500, len(self._index))
        indices = range(0, len(self._index), max(1, len(self._index) // sample_size))
        
        for i in indices:
            image_id = self._index[i]
            annotations = self._image_id_to_annotations[image_id]
            
            for ann in annotations:
                category_id = ann['category_id']
                category_info = self._category_id_to_info.get(category_id, {})
                supercategory = category_info.get('supercategory', 'unknown')
                supercategory_counts[supercategory] += 1
        
        return {
            'supercategory_distribution': dict(supercategory_counts),
            'num_supercategories': len(supercategory_counts),
            'samples_analyzed': sample_size
        }

    def get_geometric_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive geometric statistics for spatial reasoning.
        
        Returns:
            Dictionary with geometric analysis statistics
        """
        areas = []
        aspect_ratios = []
        relative_areas = []
        bbox_sizes = []
        
        # Sample for efficiency
        sample_size = min(200, len(self._index))
        indices = range(0, len(self._index), max(1, len(self._index) // sample_size))
        
        for i in indices:
            image_id = self._index[i]
            image_info = self._image_id_to_info[image_id]
            image_area = image_info.get('width', 1) * image_info.get('height', 1)
            
            annotations = self._image_id_to_annotations[image_id]
            
            for ann in annotations:
                area = ann.get('area', 0)
                bbox = ann.get('bbox', [])
                
                areas.append(area)
                
                if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                    aspect_ratios.append(bbox[2] / bbox[3])
                    bbox_sizes.append(bbox[2] * bbox[3])  # bbox area
                
                if image_area > 0:
                    relative_areas.append(area / image_area)
        
        return {
            'samples_analyzed': sample_size,
            'total_instances_analyzed': len(areas),
            'area_statistics': {
                'min_pixels': min(areas) if areas else 0,
                'max_pixels': max(areas) if areas else 0,
                'avg_pixels': sum(areas) / len(areas) if areas else 0,
                'median_pixels': sorted(areas)[len(areas) // 2] if areas else 0
            },
            'aspect_ratio_statistics': {
                'min': min(aspect_ratios) if aspect_ratios else 0,
                'max': max(aspect_ratios) if aspect_ratios else 0,
                'avg': sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0
            },
            'relative_area_statistics': {
                'min': min(relative_areas) if relative_areas else 0,
                'max': max(relative_areas) if relative_areas else 0,
                'avg': sum(relative_areas) / len(relative_areas) if relative_areas else 0
            },
            'bbox_size_statistics': {
                'min': min(bbox_sizes) if bbox_sizes else 0,
                'max': max(bbox_sizes) if bbox_sizes else 0,
                'avg': sum(bbox_sizes) / len(bbox_sizes) if bbox_sizes else 0
            }
        }