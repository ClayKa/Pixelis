# core/dataloaders/flickr30k_loader.py

import ast
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Flickr30kLoader(BaseLoader):
    """
    A concrete data loader for the Flickr30k dataset.

    This loader is responsible for:
    1. Parsing the main annotation CSV file, which contains image names and
       their corresponding captions (typically 5 captions per image).
    2. Grouping captions by their associated image.
    3. Building an index of all available images.
    4. Translating a raw Flickr30k sample into the project's standard format.
    
    The Flickr30k dataset provides rich image-caption pairs that are ideal for
    ZOOM-IN tasks, where the detailed captions can help identify interesting
    regions or objects to focus on within the images.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Flickr30k loader.
        
        Args:
            config: Configuration dictionary containing:
                - name: Name of the datasource
                - path: Path to the directory containing images
                - annotation_file: Path to the CSV annotation file
                - split: Optional split to filter ('train', 'val', 'test')
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Flickr30kLoader config must include '{key}'")
        
        # Store paths and configuration
        self.images_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        self.split_filter = config.get('split', None)  # Optional split filtering
        
        # Validate paths exist
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Will be populated by _build_index
        self._image_to_captions: Dict[str, List[str]] = defaultdict(list)
        self._image_to_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Call parent init (which will call _build_index)
        super().__init__(config)

    def _build_index(self) -> List[str]:
        """
        Read the entire annotation CSV file once, parse its contents, group the 
        captions by image, and create a lightweight index of all unique images 
        in the dataset.
        
        Returns:
            List[str]: List of all image filenames that have captions
        """
        logger.info(f"Building index for Flickr30k dataset from {self.annotation_file}")
        
        # Parse the CSV file
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            processed_count = 0
            filtered_count = 0
            
            for row in reader:
                # Extract fields from the CSV row
                raw_captions = row['raw']
                filename = row['filename']
                split = row['split']
                img_id = row['img_id']
                sentids = row['sentids']
                
                # Apply split filtering if specified
                if self.split_filter and split != self.split_filter:
                    filtered_count += 1
                    continue
                
                # Parse the raw captions (stored as a JSON-like string)
                try:
                    # The raw field contains a string representation of a Python list
                    captions_list = ast.literal_eval(raw_captions)
                    
                    # Ensure we have a list of captions
                    if not isinstance(captions_list, list):
                        logger.warning(f"Unexpected captions format for {filename}: {type(captions_list)}")
                        continue
                    
                    # Store captions for this image
                    self._image_to_captions[filename].extend(captions_list)
                    
                    # Store additional metadata
                    self._image_to_metadata[filename] = {
                        'img_id': int(img_id),
                        'split': split,
                        'sentids': ast.literal_eval(sentids) if sentids else []
                    }
                    
                    processed_count += 1
                    
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse captions for {filename}: {e}")
                    continue
        
        logger.info(f"Processed {processed_count} images, filtered out {filtered_count} by split")
        
        # Filter to only include images that actually exist
        valid_images = []
        missing_count = 0
        
        for image_filename in self._image_to_captions.keys():
            image_path = self.images_path / image_filename
            if image_path.exists():
                valid_images.append(image_filename)
            else:
                logger.debug(f"Image file not found: {image_path}")
                missing_count += 1
        
        if missing_count > 0:
            logger.warning(f"{missing_count} images referenced in annotations but not found in directory")
        
        logger.info(f"Successfully indexed {len(valid_images)} valid images")
        
        return valid_images

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single image and its complete set of annotations by index, 
        and format it into the project's standardized dictionary.
        
        Args:
            index: The integer index of the sample in the _index list
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary containing:
                - source_dataset: Name of the dataset
                - sample_id: Image ID (filename without extension)
                - media_type: "image"
                - media_path: Absolute path to the image file
                - width, height: Image dimensions
                - annotations: Dictionary containing captions and metadata
        """
        # Retrieve Image Filename
        image_filename = self._index[index]
        
        # Extract sample ID (filename without extension)
        sample_id = Path(image_filename).stem
        
        # Construct Image Path
        image_path = self.images_path / image_filename
        
        # Create Base Structure
        sample = self._get_standardized_base(
            sample_id=sample_id,
            media_path=image_path,
            media_type="image"
        )
        
        # Retrieve Captions and Metadata
        captions = self._image_to_captions[image_filename]
        metadata = self._image_to_metadata.get(image_filename, {})
        
        # Adapt and Populate Annotations (The "Translation" Step)
        sample['annotations']['captions'] = captions
        sample['annotations']['num_captions'] = len(captions)
        
        # Add Flickr30k-specific metadata
        if metadata:
            sample['annotations']['flickr_img_id'] = metadata.get('img_id')
            sample['annotations']['split'] = metadata.get('split')
            sample['annotations']['sentence_ids'] = metadata.get('sentids', [])
        
        # Add dataset-specific information useful for ZOOM-IN tasks
        sample['annotations']['dataset_info'] = {
            'task_type': 'image_captioning',
            'suitable_for_zoom': True,  # Flickr30k images are good for zoom tasks
            'caption_quality': 'high',  # Human-annotated captions
            'avg_caption_length': sum(len(cap.split()) for cap in captions) / len(captions) if captions else 0
        }
        
        return sample

    def get_captions_for_image(self, image_filename: str) -> List[str]:
        """
        Utility method to get all captions for a specific image.
        
        Args:
            image_filename: Name of the image file
            
        Returns:
            List of captions for the image
        """
        return self._image_to_captions.get(image_filename, [])

    def get_images_by_split(self, split: str) -> List[str]:
        """
        Utility method to get all images from a specific split.
        
        Args:
            split: Split name ('train', 'val', 'test')
            
        Returns:
            List of image filenames in the specified split
        """
        return [
            filename for filename in self._image_to_captions.keys()
            if self._image_to_metadata.get(filename, {}).get('split') == split
        ]

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_images = len(self._image_to_captions)
        total_captions = sum(len(captions) for captions in self._image_to_captions.values())
        
        # Split statistics
        split_counts = defaultdict(int)
        for metadata in self._image_to_metadata.values():
            split_counts[metadata.get('split', 'unknown')] += 1
        
        # Caption length statistics
        all_caption_lengths = []
        for captions in self._image_to_captions.values():
            all_caption_lengths.extend([len(cap.split()) for cap in captions])
        
        avg_caption_length = sum(all_caption_lengths) / len(all_caption_lengths) if all_caption_lengths else 0
        
        return {
            'total_images': total_images,
            'total_captions': total_captions,
            'avg_captions_per_image': total_captions / total_images if total_images > 0 else 0,
            'split_distribution': dict(split_counts),
            'avg_caption_length_words': avg_caption_length,
            'filter_applied': self.split_filter
        }