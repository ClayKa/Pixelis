# core/dataloaders/sa1b_segment_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Sa1bSegmentLoader(BaseLoader):
    """
    A concrete data loader for the Segment Anything (SA-1B) dataset, specifically
    tailored to provide samples for segmentation and property-based tasks.

    This loader uses the same core logic as a general SA-1B loader (individual JSON
    files per image) but ensures its output contract is optimized for downstream
    generators like the GeometricComparisonTaskGenerator that require clean,
    usable segmentation data.
    
    Key optimizations for segmentation tasks:
    - Pre-calculated center points for SEGMENT_OBJECT_AT operations
    - Configurable minimum area filtering to remove tiny, unusable masks
    - Specialized annotation format optimized for spatial reasoning
    - Quality-based mask filtering based on stability scores and predicted IoU
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Sa1bSegmentLoader.
        
        Args:
            config: Configuration dictionary containing 'path' and 'annotations_path'
                   plus optional segmentation-specific parameters
        """
        # Validate required config keys before calling super().__init__
        required_keys = ['path', 'annotations_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Sa1bSegmentLoader config must include '{key}'")
        
        # Set up paths before calling super().__init__
        self.images_path = Path(config['path'])
        self.annotations_path = Path(config['annotations_path'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_path}")
        
        # Segmentation-specific configuration parameters
        self.min_pixel_area = config.get('min_pixel_area', 100)  # Filter tiny masks
        self.min_stability_score = config.get('min_stability_score', 0.5)  # Quality threshold
        self.min_predicted_iou = config.get('min_predicted_iou', 0.5)  # Quality threshold
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build index by matching available images with their annotation files,
        optimized for segmentation task generation.
        
        Returns:
            List of dictionaries containing image and annotation paths with metadata
        """
        # Find all image files
        image_files = list(self.images_path.glob('*.jpg'))
        if not image_files:
            # Fallback to other extensions
            for ext in ['*.jpeg', '*.JPG', '*.JPEG']:
                image_files.extend(self.images_path.glob(ext))
        
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files")
        
        # Build index by matching images to annotations
        index = []
        matched_count = 0
        usable_masks_count = 0
        
        for image_path in image_files:
            # Derive annotation filename from image filename
            image_stem = image_path.stem  # e.g., "sa_3624991"
            annotation_file = self.annotations_path / f"{image_stem}.json"
            
            # Only include if annotation file exists
            if annotation_file.exists():
                try:
                    # Load annotation to get metadata and filter quality
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotation_data = json.load(f)
                    
                    # Extract image metadata
                    image_info = annotation_data.get('image', {})
                    annotations = annotation_data.get('annotations', [])
                    
                    # Pre-filter annotations for segmentation quality
                    usable_annotations = []
                    for ann in annotations:
                        # Apply quality filters
                        area = ann.get('area', 0)
                        stability_score = ann.get('stability_score', 0.0)
                        predicted_iou = ann.get('predicted_iou', 0.0)
                        
                        if (area >= self.min_pixel_area and 
                            stability_score >= self.min_stability_score and 
                            predicted_iou >= self.min_predicted_iou):
                            
                            # Pre-calculate center point for SEGMENT_OBJECT_AT operations
                            bbox = ann.get('bbox', [0, 0, 0, 0])
                            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                                center_x = bbox[0] + bbox[2] / 2
                                center_y = bbox[1] + bbox[3] / 2
                                ann['center_point'] = [center_x, center_y]
                                usable_annotations.append(ann)
                    
                    # Only include images with usable segmentation masks
                    if usable_annotations:
                        # Create index entry with all necessary information
                        index_entry = {
                            'image_id': image_info.get('image_id', image_stem),
                            'image_path': str(image_path),
                            'annotation_path': str(annotation_file),
                            'width': image_info.get('width', 0),
                            'height': image_info.get('height', 0),
                            'file_name': image_info.get('file_name', image_path.name),
                            'num_annotations': len(annotations),
                            'num_usable_annotations': len(usable_annotations),
                            'annotations': usable_annotations  # Only store usable annotations
                        }
                        index.append(index_entry)
                        matched_count += 1
                        usable_masks_count += len(usable_annotations)
                    else:
                        logger.debug(f"No usable masks found for {image_path.name}")
                    
                except (json.JSONDecodeError, IOError, KeyError) as e:
                    logger.warning(f"Error reading annotation {annotation_file}: {e}")
                    continue
        
        logger.info(f"Successfully matched {matched_count} images with {usable_masks_count} usable masks")
        logger.info(f"Applied filters: min_area={self.min_pixel_area}, "
                   f"min_stability={self.min_stability_score}, min_iou={self.min_predicted_iou}")
        
        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sample by index with segmentation-optimized annotations.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with segmentation-focused annotations
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})\")")
        
        # Get index entry
        entry = self._index[index]
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=str(entry['image_id']),
            media_path=Path(entry['image_path']),
            media_type="image"
        )
        
        # Process annotations with segmentation focus
        segmentation_instances = []
        total_area = 0
        quality_scores = []
        
        for ann in entry['annotations']:
            # Create segmentation-focused annotation
            processed_annotation = {
                'instance_id': ann.get('id', 0),
                'area_pixels': ann.get('area', 0),
                'bbox': ann.get('bbox', []),
                'segmentation_mask_rle': ann.get('segmentation', {}),
                'center_point': ann.get('center_point', [0, 0]),  # Pre-calculated for SEGMENT_OBJECT_AT
                'quality_metrics': {
                    'stability_score': ann.get('stability_score', 0.0),
                    'predicted_iou': ann.get('predicted_iou', 0.0),
                    'point_coords': ann.get('point_coords', []),
                    'crop_box': ann.get('crop_box', [])
                },
                'geometric_properties': {
                    'width': ann.get('bbox', [0, 0, 0, 0])[2] if len(ann.get('bbox', [])) >= 3 else 0,
                    'height': ann.get('bbox', [0, 0, 0, 0])[3] if len(ann.get('bbox', [])) >= 4 else 0,
                    'aspect_ratio': self._calculate_aspect_ratio(ann.get('bbox', [])),
                    'relative_area': ann.get('area', 0) / (entry['width'] * entry['height']) if entry['width'] > 0 and entry['height'] > 0 else 0
                }
            }
            
            segmentation_instances.append(processed_annotation)
            total_area += ann.get('area', 0)
            quality_scores.append(ann.get('stability_score', 0.0))
        
        # Add segmentation-focused annotations
        sample['annotations'].update({
            'instance_segmentation': segmentation_instances,
            'num_instances': len(segmentation_instances),
            'total_segmented_area': total_area,
            'coverage_ratio': total_area / (entry['width'] * entry['height']) if entry['width'] > 0 and entry['height'] > 0 else 0,
            'quality_statistics': {
                'avg_stability_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                'min_stability_score': min(quality_scores) if quality_scores else 0.0,
                'max_stability_score': max(quality_scores) if quality_scores else 0.0,
                'usable_mask_ratio': len(segmentation_instances) / entry['num_annotations'] if entry['num_annotations'] > 0 else 0.0
            },
            'image_metadata': {
                'sa_image_id': entry['image_id'],
                'original_width': entry['width'],
                'original_height': entry['height'],
                'file_name': entry['file_name'],
                'total_annotations': entry['num_annotations'],
                'usable_annotations': entry['num_usable_annotations']
            },
            'dataset_info': {
                'task_type': 'instance_segmentation_optimized',
                'suitable_for_zoom': False,  # This is for segmentation, not zoom
                'suitable_for_segment_object_at': True,
                'suitable_for_get_properties': True,
                'source': 'SA-1B-Segment',
                'mask_format': 'rle',
                'has_center_points': True,
                'has_quality_metrics': True,
                'filtering_applied': {
                    'min_pixel_area': self.min_pixel_area,
                    'min_stability_score': self.min_stability_score,
                    'min_predicted_iou': self.min_predicted_iou
                }
            }
        })
        
        return sample

    def _calculate_aspect_ratio(self, bbox: List[float]) -> float:
        """Calculate aspect ratio from bounding box [x, y, width, height]."""
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return 0.0
        return bbox[2] / bbox[3]  # width / height

    def get_high_quality_instances(self, min_stability_score: float = 0.95, 
                                  min_predicted_iou: float = 0.9) -> List[Dict[str, Any]]:
        """
        Get samples containing high-quality segmentation instances.
        
        Args:
            min_stability_score: Minimum stability score threshold
            min_predicted_iou: Minimum predicted IoU threshold
            
        Returns:
            List of samples with high-quality instances
        """
        high_quality_samples = []
        
        for i in range(len(self)):
            entry = self._index[i]
            
            # Check if this sample has any high-quality instances
            high_quality_instances = []
            for ann in entry['annotations']:
                stability = ann.get('stability_score', 0.0)
                iou = ann.get('predicted_iou', 0.0)
                
                if stability >= min_stability_score and iou >= min_predicted_iou:
                    high_quality_instances.append(ann)
            
            # If there are high-quality instances, include the sample
            if high_quality_instances:
                sample = self.get_item(i)
                # Filter to only include high-quality instances
                sample['annotations']['instance_segmentation'] = [
                    inst for inst in sample['annotations']['instance_segmentation']
                    if inst['quality_metrics']['stability_score'] >= min_stability_score
                    and inst['quality_metrics']['predicted_iou'] >= min_predicted_iou
                ]
                high_quality_samples.append(sample)
        
        return high_quality_samples

    def get_samples_by_area_range(self, min_area: int, max_area: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get samples filtered by segmentation instance area.
        
        Args:
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (None for no limit)
            
        Returns:
            List of sample dictionaries meeting the area criteria
        """
        filtered_samples = []
        
        for i in range(len(self)):
            entry = self._index[i]
            
            # Check if this sample has instances in the desired area range
            matching_instances = []
            for ann in entry['annotations']:
                area = ann.get('area', 0)
                if area >= min_area and (max_area is None or area <= max_area):
                    matching_instances.append(ann)
            
            if matching_instances:
                sample = self.get_item(i)
                # Filter to only include matching instances
                sample['annotations']['instance_segmentation'] = [
                    inst for inst in sample['annotations']['instance_segmentation']
                    if min_area <= inst['area_pixels'] <= (max_area or float('inf'))
                ]
                filtered_samples.append(sample)
        
        return filtered_samples

    def get_geometric_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about geometric properties for spatial reasoning tasks.
        
        Returns:
            Dictionary with geometric analysis statistics
        """
        if not self._index:
            return {'error': 'No samples available'}
        
        total_instances = 0
        areas = []
        aspect_ratios = []
        relative_areas = []
        stability_scores = []
        predicted_ious = []
        coverage_ratios = []
        
        # Analyze a sample of images for statistics
        sample_size = min(100, len(self._index))
        for i in range(0, len(self._index), max(1, len(self._index) // sample_size)):
            entry = self._index[i]
            
            instance_areas_for_image = []
            image_area = entry['width'] * entry['height']
            
            for ann in entry['annotations']:
                area = ann.get('area', 0)
                bbox = ann.get('bbox', [])
                
                areas.append(area)
                instance_areas_for_image.append(area)
                
                if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                    aspect_ratios.append(bbox[2] / bbox[3])
                
                if image_area > 0:
                    relative_areas.append(area / image_area)
                
                stability_scores.append(ann.get('stability_score', 0.0))
                predicted_ious.append(ann.get('predicted_iou', 0.0))
                total_instances += 1
            
            # Image-level coverage ratio
            if image_area > 0:
                coverage_ratios.append(sum(instance_areas_for_image) / image_area)
        
        return {
            'total_samples_analyzed': sample_size,
            'total_instances': total_instances,
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
            'quality_statistics': {
                'avg_stability_score': sum(stability_scores) / len(stability_scores) if stability_scores else 0,
                'avg_predicted_iou': sum(predicted_ious) / len(predicted_ious) if predicted_ious else 0
            },
            'coverage_statistics': {
                'avg_coverage_ratio': sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0,
                'min_coverage_ratio': min(coverage_ratios) if coverage_ratios else 0,
                'max_coverage_ratio': max(coverage_ratios) if coverage_ratios else 0
            },
            'filtering_impact': {
                'min_pixel_area_applied': self.min_pixel_area,
                'min_stability_score_applied': self.min_stability_score,
                'min_predicted_iou_applied': self.min_predicted_iou
            }
        }

    def get_samples_suitable_for_geometric_comparison(self, min_instances: int = 2,
                                                     max_instances: int = 10) -> List[Dict[str, Any]]:
        """
        Get samples particularly suitable for geometric comparison tasks.
        
        Args:
            min_instances: Minimum number of segmentation instances required
            max_instances: Maximum number of instances allowed
            
        Returns:
            List of samples suitable for geometric comparison
        """
        suitable_samples = []
        
        for i in range(len(self)):
            entry = self._index[i]
            num_instances = entry['num_usable_annotations']
            
            if min_instances <= num_instances <= max_instances:
                sample = self.get_item(i)
                
                # Additional suitability checks
                instances = sample['annotations']['instance_segmentation']
                
                # Check for diverse sizes (avoid all tiny or all huge objects)
                areas = [inst['area_pixels'] for inst in instances]
                if len(areas) >= 2:
                    area_ratio = max(areas) / min(areas) if min(areas) > 0 else 0
                    # Include if there's reasonable size diversity
                    if 1.5 <= area_ratio <= 50:  # Not too similar, not too extreme
                        suitable_samples.append(sample)
        
        return suitable_samples