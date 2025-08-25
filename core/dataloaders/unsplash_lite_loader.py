# core/dataloaders/unsplash_lite_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_loader import BaseLoader


class UnsplashLiteLoader(BaseLoader):
    """
    A concrete data loader for the Unsplash-Lite 25k dataset.

    This loader handles the dataset's structure, which consists of a directory of
    high-resolution images and optional annotation files. It can operate in two modes:
    1. Simple mode: Just loads images without annotations (as per original spec)
    2. Enhanced mode: Loads images with bounding box annotations and descriptions
    
    Each image file is treated as an independent sample.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UnsplashLiteLoader.
        
        Args:
            config: Configuration dictionary containing 'path' and optional 'annotations_path'
        """
        # Validate required config keys before calling super().__init__
        if 'path' not in config:
            raise ValueError("UnsplashLiteLoader config must include 'path'")
        
        # Set up paths before calling super().__init__
        self.images_path = Path(config['path'])
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        
        # Optional annotations directory
        self.annotations_path = None
        self.use_annotations = False
        if 'annotations_path' in config:
            self.annotations_path = Path(config['annotations_path'])
            if self.annotations_path.exists():
                self.use_annotations = True
                print(f"Found annotations directory: {self.annotations_path}")
            else:
                print(f"Warning: Annotations directory not found: {self.annotations_path}")
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Path]:
        """
        Build index by scanning the images directory for valid image files.
        
        Returns:
            List of Path objects pointing to valid image files
        """
        # Supported image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        image_files = []
        
        # Scan for all supported image types
        for pattern in image_extensions:
            found_files = list(self.images_path.glob(pattern))
            image_files.extend(found_files)
        
        # Filter to ensure they are actually files
        valid_image_files = [f for f in image_files if f.is_file()]
        
        # Sort for consistent ordering
        valid_image_files.sort()
        
        print(f"Found {len(valid_image_files)} image files in {self.images_path}")
        
        # If annotations are enabled, filter to only include images with annotations
        if self.use_annotations:
            annotated_images = []
            for image_path in valid_image_files:
                annotation_file = self.annotations_path / f"{image_path.stem}.json"
                if annotation_file.exists():
                    annotated_images.append(image_path)
            
            print(f"Found {len(annotated_images)} images with annotations")
            return annotated_images
        
        return valid_image_files

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single image sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get the image path
        image_path = self._index[index]
        
        # Extract sample ID from filename (without extension)
        sample_id = image_path.stem
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=sample_id,
            media_path=image_path,
            media_type="image"
        )
        
        # Initialize annotations
        annotations = {
            'dataset_info': {
                'task_type': 'high_resolution_image_collection',
                'suitable_for_zoom': True,
                'source': 'Unsplash-Lite-25k',
                'has_annotations': self.use_annotations
            }
        }
        
        # Load annotations if available
        if self.use_annotations:
            annotation_file = self.annotations_path / f"{sample_id}.json"
            if annotation_file.exists():
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotation_data = json.load(f)
                    
                    # Extract annotation information
                    annotations.update({
                        'annotator_model': annotation_data.get('annotator_model', ''),
                        'bounding_boxes': annotation_data.get('annotations', []),
                        'num_annotations': len(annotation_data.get('annotations', [])),
                        'regions': self._process_annotations(annotation_data.get('annotations', []))
                    })
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Error reading annotation file {annotation_file}: {e}")
        
        # Add annotations to sample
        sample['annotations'].update(annotations)
        
        return sample

    def _process_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw annotations into standardized format.
        
        Args:
            annotations: List of raw annotation dictionaries
            
        Returns:
            List of processed annotation dictionaries
        """
        processed = []
        
        for i, ann in enumerate(annotations):
            if 'box' in ann and 'desc' in ann:
                # Extract bounding box coordinates
                box = ann['box']
                if len(box) >= 4:
                    processed_ann = {
                        'region_id': i,
                        'description': ann['desc'],
                        'bbox': {
                            'x': box[0],
                            'y': box[1], 
                            'width': box[2] - box[0],
                            'height': box[3] - box[1]
                        },
                        'confidence': 1.0  # Annotations don't include confidence scores
                    }
                    processed.append(processed_ann)
        
        return processed

    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an image sample by its ID.
        
        Args:
            image_id: The image ID to search for
            
        Returns:
            Sample dictionary if found, None otherwise
        """
        for i, image_path in enumerate(self._index):
            if image_path.stem == image_id:
                return self.get_item(i)
        
        return None

    def get_annotated_images(self) -> List[Dict[str, Any]]:
        """
        Get all images that have annotations (only works if annotations are enabled).
        
        Returns:
            List of sample dictionaries for images with annotations
        """
        if not self.use_annotations:
            return []
        
        annotated_samples = []
        for i in range(len(self)):
            sample = self.get_item(i)
            if sample['annotations'].get('num_annotations', 0) > 0:
                annotated_samples.append(sample)
        
        return annotated_samples

    def get_images_by_annotation_count(self, min_count: int = 1) -> List[Dict[str, Any]]:
        """
        Get images that have at least the specified number of annotations.
        
        Args:
            min_count: Minimum number of annotations required
            
        Returns:
            List of sample dictionaries meeting the criteria
        """
        if not self.use_annotations:
            return []
        
        filtered_samples = []
        for i in range(len(self)):
            sample = self.get_item(i)
            if sample['annotations'].get('num_annotations', 0) >= min_count:
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
                'annotations_enabled': self.use_annotations,
                'avg_annotations_per_image': 0,
                'total_annotations': 0,
                'annotation_coverage': 0.0
            }
        
        total_images = len(self._index)
        
        if not self.use_annotations:
            return {
                'total_images': total_images,
                'annotations_enabled': False,
                'image_extensions': self._get_image_extensions(),
                'avg_file_size_mb': self._calculate_avg_file_size()
            }
        
        # Calculate annotation statistics from a sample
        sample_size = min(50, total_images)
        sample_indices = range(0, total_images, max(1, total_images // sample_size))
        
        total_annotations = 0
        images_with_annotations = 0
        
        for i in sample_indices:
            try:
                sample = self.get_item(i)
                ann_count = sample['annotations'].get('num_annotations', 0)
                total_annotations += ann_count
                if ann_count > 0:
                    images_with_annotations += 1
            except Exception:
                continue
        
        sample_count = len(list(sample_indices))
        
        return {
            'total_images': total_images,
            'annotations_enabled': self.use_annotations,
            'total_annotations': int(total_annotations * total_images / sample_count) if sample_count > 0 else 0,
            'avg_annotations_per_image': total_annotations / sample_count if sample_count > 0 else 0,
            'annotation_coverage': images_with_annotations / sample_count if sample_count > 0 else 0,
            'image_extensions': self._get_image_extensions(),
            'sample_size_used': sample_count
        }

    def _get_image_extensions(self) -> Dict[str, int]:
        """Get distribution of image file extensions."""
        extensions = {}
        for image_path in self._index:
            ext = image_path.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        return extensions

    def _calculate_avg_file_size(self) -> float:
        """Calculate average file size in MB from a sample."""
        if not self._index:
            return 0.0
        
        # Sample a few files to estimate average size
        sample_size = min(10, len(self._index))
        total_size = 0
        
        for i in range(0, len(self._index), max(1, len(self._index) // sample_size)):
            try:
                file_path = self._index[i]
                size_bytes = file_path.stat().st_size
                total_size += size_bytes
            except Exception:
                continue
        
        avg_size_bytes = total_size / sample_size if sample_size > 0 else 0
        return round(avg_size_bytes / (1024 * 1024), 2)  # Convert to MB