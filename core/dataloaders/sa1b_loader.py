# core/dataloaders/sa1b_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Sa1bLoader(BaseLoader):
    """
    A concrete data loader for the Segment Anything (SA-1B) dataset.

    This loader is designed to handle SA-1B's structure with individual 
    annotation files per image:
    1.  Individual JSON files for each image containing segmentation masks
    2.  Images distributed in an images directory
    3.  RLE-encoded segmentation masks for instance segmentation

    The loader builds an index by matching available images with their
    corresponding annotation files to ensure only complete samples are loaded.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Sa1bLoader.
        
        Args:
            config: Configuration dictionary containing 'path' and 'annotations_path'
        """
        # Validate required config keys before calling super().__init__
        required_keys = ['path', 'annotations_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Sa1bLoader config must include '{key}'")
        
        # Set up paths before calling super().__init__
        self.images_path = Path(config['path'])
        self.annotations_path = Path(config['annotations_path'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_path}")
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build index by matching available images with their annotation files.
        
        Returns:
            List of dictionaries containing image and annotation paths and metadata
        """
        # Find all image files
        image_files = list(self.images_path.glob('*.jpg'))
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Build index by matching images to annotations
        index = []
        matched_count = 0
        
        for image_path in image_files:
            # Derive annotation filename from image filename
            image_stem = image_path.stem  # e.g., "sa_1062875"
            annotation_file = self.annotations_path / f"{image_stem}.json"
            
            # Only include if annotation file exists
            if annotation_file.exists():
                try:
                    # Load annotation to get metadata
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotation_data = json.load(f)
                    
                    # Extract image metadata
                    image_info = annotation_data.get('image', {})
                    annotations = annotation_data.get('annotations', [])
                    
                    # Create index entry with all necessary information
                    index_entry = {
                        'image_id': image_info.get('image_id', image_stem),
                        'image_path': str(image_path),
                        'annotation_path': str(annotation_file),
                        'width': image_info.get('width', 0),
                        'height': image_info.get('height', 0),
                        'file_name': image_info.get('file_name', image_path.name),
                        'num_annotations': len(annotations),
                        'annotations': annotations
                    }
                    index.append(index_entry)
                    matched_count += 1
                    
                except (json.JSONDecodeError, IOError, KeyError) as e:
                    logger.warning(f"Error reading annotation {annotation_file}: {e}")
                    continue
        
        logger.info(f"Successfully matched {matched_count} images with annotations")
        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sample by index with segmentation annotations.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with segmentation masks
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get index entry
        entry = self._index[index]
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=str(entry['image_id']),
            media_path=Path(entry['image_path']),
            media_type="image"
        )
        
        # Process annotations
        segmentation_masks = []
        total_area = 0
        
        for ann in entry['annotations']:
            processed_annotation = {
                'annotation_id': ann.get('id', 0),
                'bbox': ann.get('bbox', []),
                'area': ann.get('area', 0),
                'segmentation_rle': ann.get('segmentation', {}),
                'predicted_iou': ann.get('predicted_iou', 0.0),
                'stability_score': ann.get('stability_score', 0.0),
                'point_coords': ann.get('point_coords', []),
                'crop_box': ann.get('crop_box', [])
            }
            segmentation_masks.append(processed_annotation)
            total_area += ann.get('area', 0)
        
        # Add SA-1B specific annotations
        sample['annotations'].update({
            'instance_segmentation': segmentation_masks,
            'num_masks': len(segmentation_masks),
            'total_area': total_area,
            'image_metadata': {
                'sa_image_id': entry['image_id'],
                'original_width': entry['width'],
                'original_height': entry['height'],
                'file_name': entry['file_name']
            },
            'dataset_info': {
                'task_type': 'instance_segmentation',
                'suitable_for_zoom': True,
                'source': 'SA-1B',
                'mask_format': 'rle',
                'has_point_prompts': True,
                'avg_stability_score': self._calculate_avg_stability_score(entry['annotations']),
                'avg_predicted_iou': self._calculate_avg_predicted_iou(entry['annotations'])
            }
        })
        
        return sample

    def _calculate_avg_stability_score(self, annotations: List[Dict[str, Any]]) -> float:
        """Calculate average stability score for all masks in the image."""
        if not annotations:
            return 0.0
        
        scores = [ann.get('stability_score', 0.0) for ann in annotations]
        return round(sum(scores) / len(scores), 4)

    def _calculate_avg_predicted_iou(self, annotations: List[Dict[str, Any]]) -> float:
        """Calculate average predicted IoU for all masks in the image."""
        if not annotations:
            return 0.0
        
        ious = [ann.get('predicted_iou', 0.0) for ann in annotations]
        return round(sum(ious) / len(ious), 4)

    def get_sample_by_image_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a sample by its SA-1B image ID.
        
        Args:
            image_id: The SA-1B image ID to search for
            
        Returns:
            Sample dictionary if found, None otherwise
        """
        for i, entry in enumerate(self._index):
            if entry['image_id'] == image_id:
                return self.get_item(i)
        
        return None

    def get_high_quality_samples(self, min_stability_score: float = 0.95, 
                                min_predicted_iou: float = 0.9) -> List[Dict[str, Any]]:
        """
        Get samples with high-quality segmentation masks.
        
        Args:
            min_stability_score: Minimum stability score threshold
            min_predicted_iou: Minimum predicted IoU threshold
            
        Returns:
            List of high-quality sample dictionaries
        """
        high_quality_samples = []
        
        for i in range(len(self)):
            entry = self._index[i]
            
            # Check if this sample has high-quality masks
            high_quality_annotations = []
            for ann in entry['annotations']:
                stability = ann.get('stability_score', 0.0)
                iou = ann.get('predicted_iou', 0.0)
                
                if stability >= min_stability_score and iou >= min_predicted_iou:
                    high_quality_annotations.append(ann)
            
            # If there are high-quality masks, include the sample
            if high_quality_annotations:
                sample = self.get_item(i)
                high_quality_samples.append(sample)
        
        return high_quality_samples

    def get_samples_by_mask_count(self, min_masks: int = 1, max_masks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get samples filtered by number of segmentation masks.
        
        Args:
            min_masks: Minimum number of masks required
            max_masks: Maximum number of masks allowed (None for no limit)
            
        Returns:
            List of sample dictionaries meeting the criteria
        """
        filtered_samples = []
        
        for i in range(len(self)):
            entry = self._index[i]
            num_masks = entry['num_annotations']
            
            if num_masks >= min_masks:
                if max_masks is None or num_masks <= max_masks:
                    sample = self.get_item(i)
                    filtered_samples.append(sample)
        
        return filtered_samples

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self._index:
            return {
                'total_images': 0,
                'total_masks': 0,
                'avg_masks_per_image': 0,
                'avg_stability_score': 0,
                'avg_predicted_iou': 0
            }
        
        total_images = len(self._index)
        total_masks = 0
        total_area = 0
        stability_scores = []
        predicted_ious = []
        mask_counts = []
        
        # Sample a subset for detailed statistics (to avoid processing all 25k images)
        sample_size = min(100, total_images)
        sample_indices = range(0, total_images, max(1, total_images // sample_size))
        
        for i in sample_indices:
            entry = self._index[i]
            num_masks = entry['num_annotations']
            mask_counts.append(num_masks)
            total_masks += num_masks
            
            for ann in entry['annotations']:
                total_area += ann.get('area', 0)
                stability_scores.append(ann.get('stability_score', 0.0))
                predicted_ious.append(ann.get('predicted_iou', 0.0))
        
        sample_count = len(list(sample_indices))
        
        return {
            'total_images': total_images,
            'total_masks': int(total_masks * total_images / sample_count) if sample_count > 0 else 0,
            'avg_masks_per_image': sum(mask_counts) / len(mask_counts) if mask_counts else 0,
            'avg_stability_score': sum(stability_scores) / len(stability_scores) if stability_scores else 0,
            'avg_predicted_iou': sum(predicted_ious) / len(predicted_ious) if predicted_ious else 0,
            'avg_mask_area': total_area / sum(mask_counts) if sum(mask_counts) > 0 else 0,
            'image_resolution_stats': self._get_resolution_stats(),
            'mask_count_distribution': {
                'min': min(mask_counts) if mask_counts else 0,
                'max': max(mask_counts) if mask_counts else 0,
                'median': sorted(mask_counts)[len(mask_counts) // 2] if mask_counts else 0
            },
            'sample_size_used': sample_count
        }

    def _get_resolution_stats(self) -> Dict[str, Any]:
        """Get statistics about image resolutions."""
        widths = []
        heights = []
        
        # Sample a subset for resolution stats
        sample_size = min(50, len(self._index))
        for i in range(0, len(self._index), max(1, len(self._index) // sample_size)):
            entry = self._index[i]
            widths.append(entry['width'])
            heights.append(entry['height'])
        
        if not widths:
            return {'avg_width': 0, 'avg_height': 0}
        
        return {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }