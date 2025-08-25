# core/dataloaders/icdar_art_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class IcdarArTLoader(BaseLoader):
    """
    A concrete data loader for the ICDAR 2019 ArT (Arbitrary-Shaped Text) dataset.

    This loader handles the dataset's structure, which consists of:
    1. A single, large JSON file containing annotations for all training images.
    2. A directory of image files whose names are the keys in the annotation file.
    
    The loader efficiently parses the train_labels.json file, builds a robust index,
    and provides standardized sample dictionaries that link each image to its 
    corresponding text transcriptions and polygon bounding boxes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ICDAR ArT loader.
        
        Args:
            config: Configuration dictionary containing:
                - name: Name of the datasource
                - path: Path to the directory containing images  
                - annotation_file: Path to the train_labels.json file
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"IcdarArTLoader config must include '{key}'")
        
        # Store paths
        self.images_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Will be populated by _build_index
        self._annotations_map: Dict[str, List[Dict[str, Any]]] = {}
        
        # Call parent init (which will call _build_index)
        super().__init__(config)

    def _build_index(self) -> List[str]:
        """
        Load the main train_labels.json file once during initialization and create 
        a simple, lightweight index of all available image samples.
        
        The train_labels.json file is a large dictionary where keys are image 
        identifiers (e.g., "gt_1726") without file extensions, and values are 
        lists of annotations. We need to filter to only include entries where 
        corresponding image files actually exist.
        
        Returns:
            List[str]: List of image identifiers that have corresponding image files
        """
        logger.info(f"Building index for ICDAR ArT dataset from {self.annotation_file}")
        
        # Load the entire JSON file containing all annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)
        
        logger.info(f"Loaded annotations for {len(raw_annotations)} entries")
        
        # Filter to only include entries with existing image files
        # The annotation keys are without extension, but image files have .jpg extension
        valid_image_ids = []
        self._annotations_map = {}
        
        for image_id in raw_annotations:
            # Try to find corresponding image file (.jpg extension)
            image_path = self.images_path / f"{image_id}.jpg"
            if image_path.exists():
                valid_image_ids.append(image_id)
                self._annotations_map[image_id] = raw_annotations[image_id]
            else:
                logger.debug(f"No image file found for annotation key: {image_id}")
        
        logger.info(f"Found {len(valid_image_ids)} valid image samples with existing files")
        
        return valid_image_ids

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single image and its annotations by index and format it into 
        the project's standardized dictionary.
        
        Args:
            index: The integer index of the sample in the _index list
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary containing:
                - source_dataset: Name of the dataset
                - sample_id: Image identifier (without extension)
                - media_type: "image"
                - media_path: Absolute path to the image file
                - width, height: Image dimensions
                - annotations: Dictionary containing scene_text annotations
        """
        # Retrieve image identifier from the index
        image_id = self._index[index]
        
        # Construct full path to the image file (add .jpg extension)
        image_path = self.images_path / f"{image_id}.jpg"
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=image_id,
            media_path=image_path,
            media_type="image"
        )
        
        # Retrieve raw annotations for this image
        raw_annotations = self._annotations_map[image_id]
        
        # Adapt and populate annotations (the "translation" step)
        standardized_annotations = []
        for ann in raw_annotations:
            # Skip illegible text annotations
            if ann.get('illegibility', False):
                continue
                
            # Convert to standardized format
            standardized_annotations.append({
                'text': ann['transcription'],
                'bbox_polygon': ann['points'],  # List of [x, y] coordinate pairs
                'language': ann.get('language', 'unknown')  # Handle optional language key
            })
        
        # Add scene_text annotations to the sample
        sample['annotations']['scene_text'] = standardized_annotations
        
        return sample