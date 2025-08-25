# core/dataloaders/lvis_segment_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class LvisSegmentLoader(BaseLoader):
    """
    A concrete data loader for the LVIS v1 dataset.
    
    This loader reads from the shared COCO+LVIS image directory but only parses 
    LVIS-specific annotations. LVIS provides a much larger vocabulary with 1000+ 
    categories including many "long-tail" rare categories, making it ideal for 
    diverse segmentation and property analysis tasks.
    
    Key features:
    - Loads LVIS v1 instance segmentation annotations
    - Large vocabulary with 1000+ categories and long-tail distribution
    - Frequency-based category classification (common/frequent/rare)
    - Cross-references with COCO images for complete metadata
    - Quality filtering based on area and segmentation quality
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LvisSegmentLoader.
        
        Args:
            config: Configuration dictionary containing 'path', 'annotation_file'
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"LvisSegmentLoader config must include '{key}'")
        
        self.images_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Configuration options for filtering
        self.min_area = config.get('min_area', 0)  # Minimum area threshold
        
        # Initialize lookup structures (populated in _build_index)
        self._image_id_to_info = {}
        self._image_id_to_annotations = defaultdict(list)
        self._category_id_to_info = {}
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[int]:
        """
        Load LVIS annotations and build efficient lookup structures.
        
        Returns:
            List of image IDs that have annotations
        """
        logger.info(f"Loading LVIS annotations from {self.annotation_file}")
        
        # Load LVIS JSON file
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            lvis_data = json.load(f)
        
        # Build image lookup - LVIS uses the same images as COCO
        images = lvis_data.get('images', [])
        for img in images:
            self._image_id_to_info[img['id']] = img
        
        logger.info(f"Loaded {len(images)} image entries")
        
        # Build category lookup - LVIS has much more categories than COCO
        categories = lvis_data.get('categories', [])
        for cat in categories:
            self._category_id_to_info[cat['id']] = cat
        
        logger.info(f"Loaded {len(categories)} categories")
        
        # Build annotations lookup with filtering
        annotations = lvis_data.get('annotations', [])
        valid_image_ids = set()
        total_annotations = 0
        filtered_annotations = 0
        
        for ann in annotations:
            # Apply area filter
            if ann.get('area', 0) < self.min_area:
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
        Retrieve a single sample by index with LVIS segmentation annotations.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with LVIS instance segmentation
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
        
        # Process LVIS annotations
        raw_annotations = self._image_id_to_annotations[image_id]
        instance_segmentations = []
        total_area = 0
        category_counts = defaultdict(int)
        frequency_distribution = defaultdict(int)  # LVIS-specific: track category frequencies
        
        for ann in raw_annotations:
            category_id = ann['category_id']
            category_info = self._category_id_to_info.get(category_id, {})
            
            # LVIS-specific: Get frequency information
            frequency = category_info.get('frequency', 'unknown')
            
            # Process segmentation data - LVIS typically uses RLE format
            segmentation = ann.get('segmentation', {})
            segmentation_type = 'rle'  # LVIS primarily uses RLE format
            
            # Create instance annotation with LVIS-specific fields
            instance_annotation = {
                'annotation_id': ann['id'],
                'category_id': category_id,
                'category_name': category_info.get('name', f'category_{category_id}'),
                'synset': category_info.get('synset'),  # LVIS-specific: WordNet synset
                'synonyms': category_info.get('synonyms', []),  # LVIS-specific: alternative names
                'def': category_info.get('def'),  # LVIS-specific: definition
                'frequency': frequency,  # LVIS-specific: r(are), c(ommon), f(requent)
                'area_pixels': ann.get('area', 0),
                'bbox': ann.get('bbox', []),  # [x, y, width, height]
                'segmentation_rle': segmentation,
                'segmentation_type': segmentation_type,
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
            frequency_distribution[frequency] += 1
        
        # Add LVIS-specific annotations
        sample['annotations'].update({
            'lvis_instance_segmentation': instance_segmentations,
            'num_instances': len(instance_segmentations),
            'total_segmented_area': total_area,
            'coverage_ratio': total_area / (image_info['width'] * image_info['height']) if image_info.get('width', 0) > 0 and image_info.get('height', 0) > 0 else 0,
            'category_distribution': dict(category_counts),
            'frequency_distribution': dict(frequency_distribution),  # LVIS-specific
            'unique_categories': len(category_counts),
            'has_rare_categories': frequency_distribution.get('r', 0) > 0,  # LVIS-specific
            'has_common_categories': frequency_distribution.get('c', 0) > 0,  # LVIS-specific
            'image_metadata': {
                'lvis_image_id': image_id,
                'file_name': image_info['file_name'],
                'width': image_info.get('width', 0),
                'height': image_info.get('height', 0),
                'license': image_info.get('license'),
                'coco_url': image_info.get('coco_url'),  # LVIS shares COCO images
                'flickr_url': image_info.get('flickr_url'),
                'date_captured': image_info.get('date_captured')
            },
            'dataset_info': {
                'task_type': 'lvis_instance_segmentation',
                'source': 'LVIS-v1',
                'suitable_for_segment_object_at': True,
                'suitable_for_get_properties': True,
                'has_category_names': True,
                'has_synonyms': True,
                'has_definitions': True,
                'has_frequency_info': True,
                'num_categories': len(self._category_id_to_info),
                'segmentation_format': 'rle',
                'long_tail_vocabulary': True,
                'filtering_applied': {
                    'min_area': self.min_area
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

    def get_samples_by_frequency(self, frequency: str) -> List[Dict[str, Any]]:
        """
        Get samples containing instances of a specific frequency class.
        
        Args:
            frequency: 'r' (rare), 'c' (common), or 'f' (frequent)
            
        Returns:
            List of samples containing instances of the specified frequency
        """
        matching_samples = []
        
        for i in range(len(self)):
            sample = self.get_item(i)
            freq_dist = sample['annotations']['frequency_distribution']
            
            if freq_dist.get(frequency, 0) > 0:
                matching_samples.append(sample)
        
        return matching_samples

    def get_rare_category_samples(self) -> List[Dict[str, Any]]:
        """
        Get samples that contain rare categories (long-tail).
        
        Returns:
            List of samples with rare category instances
        """
        return self.get_samples_by_frequency('r')

    def get_samples_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        """
        Get samples containing instances of a specific category.
        
        Args:
            category_name: Name of the category
            
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

    def get_frequency_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about frequency distribution in the dataset.
        
        Returns:
            Dictionary with LVIS-specific frequency statistics
        """
        frequency_counts = defaultdict(int)
        category_by_frequency = defaultdict(list)
        total_instances = 0
        
        # Get frequency info from categories
        for cat_info in self._category_id_to_info.values():
            frequency = cat_info.get('frequency', 'unknown')
            category_by_frequency[frequency].append(cat_info.get('name', f"category_{cat_info['id']}"))
        
        # Sample instances for statistics
        sample_size = min(1000, len(self._index))
        indices = range(0, len(self._index), max(1, len(self._index) // sample_size))
        
        for i in indices:
            image_id = self._index[i]
            annotations = self._image_id_to_annotations[image_id]
            
            for ann in annotations:
                category_id = ann['category_id']
                category_info = self._category_id_to_info.get(category_id, {})
                frequency = category_info.get('frequency', 'unknown')
                frequency_counts[frequency] += 1
                total_instances += 1
        
        return {
            'instance_frequency_distribution': dict(frequency_counts),
            'category_frequency_distribution': {
                freq: len(cats) for freq, cats in category_by_frequency.items()
            },
            'rare_categories': category_by_frequency.get('r', []),
            'common_categories': category_by_frequency.get('c', []),
            'frequent_categories': category_by_frequency.get('f', []),
            'num_rare_categories': len(category_by_frequency.get('r', [])),
            'num_common_categories': len(category_by_frequency.get('c', [])),
            'num_frequent_categories': len(category_by_frequency.get('f', [])),
            'total_instances_analyzed': total_instances,
            'samples_analyzed': sample_size
        }

    def get_vocabulary_diversity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about vocabulary diversity unique to LVIS.
        
        Returns:
            Dictionary with vocabulary diversity statistics
        """
        categories_with_synonyms = 0
        categories_with_definitions = 0
        total_synonyms = 0
        
        for cat_info in self._category_id_to_info.values():
            if cat_info.get('synonyms'):
                categories_with_synonyms += 1
                total_synonyms += len(cat_info['synonyms'])
            
            if cat_info.get('def'):
                categories_with_definitions += 1
        
        return {
            'total_categories': len(self._category_id_to_info),
            'categories_with_synonyms': categories_with_synonyms,
            'categories_with_definitions': categories_with_definitions,
            'total_synonym_count': total_synonyms,
            'avg_synonyms_per_category': total_synonyms / categories_with_synonyms if categories_with_synonyms > 0 else 0,
            'synonym_coverage_ratio': categories_with_synonyms / len(self._category_id_to_info) if self._category_id_to_info else 0,
            'definition_coverage_ratio': categories_with_definitions / len(self._category_id_to_info) if self._category_id_to_info else 0
        }

    def get_samples_with_high_diversity(self, min_categories: int = 3) -> List[Dict[str, Any]]:
        """
        Get samples with high category diversity (multiple different categories).
        
        Args:
            min_categories: Minimum number of unique categories required
            
        Returns:
            List of samples with high category diversity
        """
        diverse_samples = []
        
        for i in range(len(self)):
            sample = self.get_item(i)
            unique_categories = sample['annotations']['unique_categories']
            
            if unique_categories >= min_categories:
                diverse_samples.append(sample)
        
        return diverse_samples

    def get_geometric_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive geometric statistics for spatial reasoning.
        
        Returns:
            Dictionary with geometric analysis statistics
        """
        areas = []
        aspect_ratios = []
        relative_areas = []
        frequency_areas = defaultdict(list)  # Areas by frequency class
        
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
                
                # Get frequency for this annotation
                category_id = ann['category_id']
                category_info = self._category_id_to_info.get(category_id, {})
                frequency = category_info.get('frequency', 'unknown')
                
                areas.append(area)
                frequency_areas[frequency].append(area)
                
                if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                    aspect_ratios.append(bbox[2] / bbox[3])
                
                if image_area > 0:
                    relative_areas.append(area / image_area)
        
        # Calculate frequency-specific statistics
        frequency_stats = {}
        for freq, freq_areas in frequency_areas.items():
            if freq_areas:
                frequency_stats[freq] = {
                    'count': len(freq_areas),
                    'avg_area': sum(freq_areas) / len(freq_areas),
                    'min_area': min(freq_areas),
                    'max_area': max(freq_areas)
                }
        
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
            'frequency_based_statistics': frequency_stats
        }