# core/dataloaders/textcaps_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List

from .base_loader import BaseLoader


class TextCapsLoader(BaseLoader):
    """
    A concrete data loader for the TextCaps dataset.

    This loader is responsible for handling the dataset's structure, which consists of:
    1.  A primary annotation file containing image IDs and their corresponding captions.
    2.  A corresponding pre-computed OCR file containing text tokens and bounding boxes for each image.
    3.  A directory of images.

    The loader's primary role is to create a unified sample that links an image
    to both its descriptive captions and the foundational OCR tokens.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TextCapsLoader.
        
        Args:
            config: Configuration dictionary containing required paths
        """
        # Validate required config keys before calling super().__init__
        required_keys = ['annotation_file', 'ocr_file', 'image_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"TextCapsLoader config must include '{key}'")
        
        # Set up paths before calling super().__init__
        self.annotation_file = Path(config['annotation_file'])
        self.ocr_file = Path(config['ocr_file'])
        self.image_path = Path(config['image_path'])
        
        # Validate paths exist
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.ocr_file.exists():
            raise FileNotFoundError(f"OCR file not found: {self.ocr_file}")
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        
        # Initialize OCR lookup dictionary
        self._image_id_to_ocr = {}
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build index by loading annotation and OCR files, creating efficient OCR lookup.
        
        Returns:
            List of annotation dictionaries from the data field
        """
        # Load caption annotations
        print(f"Loading annotations from {self.annotation_file}")
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Extract the data list
        if 'data' not in annotation_data:
            raise ValueError(f"Invalid annotation file format: missing 'data' key")
        
        caption_samples = annotation_data['data']
        print(f"Found {len(caption_samples)} annotation samples")
        
        # Load and pre-process OCR data for efficient lookup
        print(f"Loading OCR data from {self.ocr_file}")
        with open(self.ocr_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Extract OCR data
        if 'data' not in ocr_data:
            raise ValueError(f"Invalid OCR file format: missing 'data' key")
        
        ocr_samples = ocr_data['data']
        print(f"Found {len(ocr_samples)} OCR samples")
        
        # Create efficient OCR lookup dictionary
        for ocr_sample in ocr_samples:
            image_id = ocr_sample['image_id']
            self._image_id_to_ocr[image_id] = {
                'ocr_tokens': ocr_sample.get('ocr_tokens', []),
                'ocr_info': ocr_sample.get('ocr_info', [])
            }
        
        print(f"Built OCR lookup for {len(self._image_id_to_ocr)} images")
        
        # Filter caption samples to only include those with existing images and OCR data
        valid_samples = []
        for sample in caption_samples:
            image_id = sample['image_id']
            
            # Check if image file exists
            image_file = self.image_path / f"{image_id}.jpg"
            if not image_file.exists():
                continue
            
            # Check if OCR data exists
            if image_id not in self._image_id_to_ocr:
                continue
            
            valid_samples.append(sample)
        
        print(f"Found {len(valid_samples)} valid samples with images and OCR data")
        return valid_samples

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sample by index with captions and OCR data.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get the caption sample
        caption_sample = self._index[index]
        image_id = caption_sample['image_id']
        
        # Construct image path
        image_file = self.image_path / f"{image_id}.jpg"
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=image_id,
            media_path=image_file,
            media_type="image"
        )
        
        # Get OCR data for this image
        ocr_data = self._image_id_to_ocr.get(image_id, {})
        
        # Process OCR tokens and bounding boxes
        ocr_tokens = []
        for i, token in enumerate(ocr_data.get('ocr_tokens', [])):
            ocr_info = ocr_data.get('ocr_info', [])
            bbox_info = ocr_info[i] if i < len(ocr_info) else {}
            
            # Extract bounding box coordinates
            bbox = bbox_info.get('bounding_box', {})
            
            ocr_token_entry = {
                'text': token,
                'bbox': {
                    'x': bbox.get('top_left_x', 0),
                    'y': bbox.get('top_left_y', 0),
                    'width': bbox.get('width', 0),
                    'height': bbox.get('height', 0)
                },
                'confidence': 1.0  # TextCaps doesn't provide confidence scores
            }
            ocr_tokens.append(ocr_token_entry)
        
        # Add TextCaps-specific annotations
        sample['annotations'].update({
            'captions': caption_sample.get('reference_strs', []),
            'primary_caption': caption_sample.get('caption_str', ''),
            'num_captions': len(caption_sample.get('reference_strs', [])),
            'ocr_tokens': ocr_tokens,
            'num_ocr_tokens': len(ocr_tokens),
            'image_classes': caption_sample.get('image_classes', []),
            'set_name': caption_sample.get('set_name', ''),
            'flickr_original_url': caption_sample.get('flickr_original_url', ''),
            'image_width': caption_sample.get('image_width'),
            'image_height': caption_sample.get('image_height'),
            'dataset_info': {
                'task_type': 'image_text_captioning',
                'suitable_for_zoom': True,
                'text_grounded': True,
                'avg_caption_length': self._calculate_avg_caption_length(caption_sample.get('reference_strs', [])),
                'ocr_coverage': len(ocr_tokens) > 0
            }
        })
        
        return sample

    def _calculate_avg_caption_length(self, captions: List[str]) -> float:
        """
        Calculate average caption length in words.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Average length in words
        """
        if not captions:
            return 0.0
        
        total_words = sum(len(caption.split()) for caption in captions)
        return round(total_words / len(captions), 2)

    def get_captions_for_image(self, image_id: str) -> List[str]:
        """
        Get all captions for a specific image ID.
        
        Args:
            image_id: The image ID to search for
            
        Returns:
            List of captions for the image
            
        Raises:
            ValueError: If image ID not found
        """
        for sample in self._index:
            if sample['image_id'] == image_id:
                return sample.get('reference_strs', [])
        
        raise ValueError(f"Image ID '{image_id}' not found")

    def get_ocr_tokens_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """
        Get OCR tokens for a specific image ID.
        
        Args:
            image_id: The image ID to search for
            
        Returns:
            List of OCR token dictionaries
            
        Raises:
            ValueError: If image ID not found
        """
        if image_id not in self._image_id_to_ocr:
            raise ValueError(f"OCR data for image ID '{image_id}' not found")
        
        ocr_data = self._image_id_to_ocr[image_id]
        ocr_tokens = []
        
        for i, token in enumerate(ocr_data.get('ocr_tokens', [])):
            ocr_info = ocr_data.get('ocr_info', [])
            bbox_info = ocr_info[i] if i < len(ocr_info) else {}
            bbox = bbox_info.get('bounding_box', {})
            
            ocr_token_entry = {
                'text': token,
                'bbox': {
                    'x': bbox.get('top_left_x', 0),
                    'y': bbox.get('top_left_y', 0),
                    'width': bbox.get('width', 0),
                    'height': bbox.get('height', 0)
                },
                'confidence': 1.0
            }
            ocr_tokens.append(ocr_token_entry)
        
        return ocr_tokens

    def get_images_by_class(self, image_class: str) -> List[Dict[str, Any]]:
        """
        Get all images that belong to a specific class.
        
        Args:
            image_class: The image class to filter by
            
        Returns:
            List of sample dictionaries from that class
        """
        class_images = []
        
        for sample in self._index:
            if image_class in sample.get('image_classes', []):
                class_images.append(sample)
        
        return class_images

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self._index:
            return {
                'total_images': 0,
                'total_captions': 0,
                'avg_captions_per_image': 0,
                'avg_caption_length_words': 0,
                'total_ocr_tokens': 0,
                'avg_ocr_tokens_per_image': 0,
                'image_classes_distribution': {},
                'ocr_coverage': 0.0
            }
        
        # Calculate statistics from a sample to avoid processing everything
        sample_size = min(100, len(self._index))
        sample_indices = range(0, len(self._index), len(self._index) // sample_size)
        
        total_captions = 0
        total_caption_words = 0
        total_ocr_tokens = 0
        image_classes = {}
        images_with_ocr = 0
        
        for i in sample_indices:
            try:
                sample = self._index[i]
                
                # Caption statistics
                captions = sample.get('reference_strs', [])
                total_captions += len(captions)
                total_caption_words += sum(len(caption.split()) for caption in captions)
                
                # OCR statistics
                image_id = sample['image_id']
                if image_id in self._image_id_to_ocr:
                    ocr_data = self._image_id_to_ocr[image_id]
                    ocr_tokens = ocr_data.get('ocr_tokens', [])
                    total_ocr_tokens += len(ocr_tokens)
                    if len(ocr_tokens) > 0:
                        images_with_ocr += 1
                
                # Image class statistics
                for img_class in sample.get('image_classes', []):
                    image_classes[img_class] = image_classes.get(img_class, 0) + 1
                    
            except Exception:
                continue
        
        sample_count = len(sample_indices)
        
        return {
            'total_images': len(self._index),
            'total_captions': int(total_captions * len(self._index) / sample_count) if sample_count > 0 else 0,
            'avg_captions_per_image': total_captions / sample_count if sample_count > 0 else 0,
            'avg_caption_length_words': total_caption_words / total_captions if total_captions > 0 else 0,
            'total_ocr_tokens': int(total_ocr_tokens * len(self._index) / sample_count) if sample_count > 0 else 0,
            'avg_ocr_tokens_per_image': total_ocr_tokens / sample_count if sample_count > 0 else 0,
            'image_classes_distribution': dict(sorted(image_classes.items(), key=lambda x: x[1], reverse=True)[:10]),
            'ocr_coverage': images_with_ocr / sample_count if sample_count > 0 else 0.0,
            'sample_size_used': sample_count
        }