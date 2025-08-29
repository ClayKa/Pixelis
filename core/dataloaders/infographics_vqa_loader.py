# core/dataloaders/infographics_vqa_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class InfographicsVqaLoader(BaseLoader):
    """
    A concrete data loader for the InfographicsVQA dataset.

    This loader is responsible for handling the dataset's structure, which is
    analogous to DocVQA, consisting of:
    1. A primary QA annotation file (questions and answers).
    2. A corresponding pre-computed OCR file (text tokens and bounding boxes).
    3. A directory of infographic image files.
    
    The loader efficiently parses the QA annotations and the corresponding 
    pre-computed OCR files, and adapts them into the project's standardized 
    sample format.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the InfographicsVQA loader.
        
        Args:
            config: Configuration dictionary containing:
                - name: Name of the datasource
                - image_path: Path to the directory containing images
                - annotation_file: Path to the QA annotations JSON file
                - ocr_file: Path to the OCR JSON file
        """
        # Validate required config keys
        required_keys = ['image_path', 'annotation_file', 'ocr_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"InfographicsVqaLoader config must include '{key}'")
        
        # Store paths
        self.image_path = Path(config['image_path'])
        self.annotation_file = Path(config['annotation_file'])
        self.ocr_path = Path(config['ocr_path'])
        
        # Validate primary paths exist first
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Validate OCR path exists, then check if it's a directory
        if not self.ocr_path.exists():
            raise FileNotFoundError(f"OCR directory not found: {self.ocr_path}")
        if not self.ocr_path.is_dir():
            raise ValueError(f"OCR path must be a directory: {self.ocr_path}")
        
        self.ocr_dir = self.ocr_path
        
        # Will be populated by _build_index
        self._image_id_to_ocr: Dict[str, Dict] = {}
        
        # Call parent init (which will call _build_index)
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load the main QA annotation and OCR files once during initialization, 
        create an efficient lookup for OCR data, and build a lightweight index 
        of all available samples.
        
        Returns:
            List[Dict[str, Any]]: List of QA dictionaries from the annotation file
        """
        logger.info(f"Building index for InfographicsVQA dataset from {self.annotation_file}")
        
        # Load QA annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # Extract the data list (the actual QA samples)
        if 'data' not in qa_data:
            raise ValueError(f"Annotation file {self.annotation_file} missing 'data' field")
        
        qa_samples = qa_data['data']
        logger.info(f"Loaded {len(qa_samples)} QA samples from annotations")
        
        # Load and pre-process OCR data
        self._load_ocr_data(qa_samples)
        
        return qa_samples

    def _load_ocr_data(self, qa_samples: List[Dict[str, Any]]) -> None:
        """
        Load OCR data using the corrected logic that derives image_id from filename.
        
        Args:
            qa_samples: List of QA samples containing image references
        """
        logger.info("Starting aggregation of OCR files...")
        
        # Step 1: Create a mapping from the QA annotation's image identifier to the QA sample
        qa_image_id_map = {}
        for qa_sample in qa_samples:
            image_filename = qa_sample.get('image_local_name', '')
            if image_filename:
                # For InfographicsVQA, derive the stem (e.g., "20471.jpeg" -> "20471")
                image_stem = Path(image_filename).stem
                qa_image_id_map[image_stem] = qa_sample
        
        # Step 2: Aggregate OCR data using filename-based mapping
        ocr_files = list(self.ocr_dir.glob('*.json'))
        logger.info(f"Found {len(ocr_files)} OCR files to process")
        
        for ocr_file in ocr_files:
            try:
                # Step A: Derive the image_id from the OCR filename
                # e.g., "/path/to/ocr/20471.json" -> "20471"
                image_id_from_filename = ocr_file.stem
                
                # Step B: Check if this ID exists in our list of QA samples
                if image_id_from_filename in qa_image_id_map:
                    with open(ocr_file, 'r', encoding='utf-8') as f:
                        ocr_data = json.load(f)
                    
                    # Step C: Use the correct image identifier as the key for lookup
                    # The key must match what get_item() will use for lookup (full filename)
                    qa_sample = qa_image_id_map[image_id_from_filename]
                    final_image_key = qa_sample['image_local_name']  # e.g., "20471.jpeg"
                    
                    self._image_id_to_ocr[final_image_key] = ocr_data
                else:
                    logger.debug(f"Skipping OCR file {ocr_file.name} as it does not correspond to any known QA sample.")
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading or parsing OCR file {ocr_file}: {e}")
                continue
        
        logger.info(f"Successfully aggregated OCR data for {len(self._image_id_to_ocr)} unique images")
        
        # Step 3: Validate coverage for QA samples
        missing_count = 0
        found_count = 0
        
        for sample in qa_samples:
            image_filename = sample.get('image_local_name', '')
            if image_filename:
                if image_filename in self._image_id_to_ocr:
                    found_count += 1
                else:
                    missing_count += 1
                    logger.debug(f"No OCR data found for image: {image_filename}")
        
        logger.info(f"OCR coverage: {found_count} found, {missing_count} missing from QA samples")
    
    def _extract_image_id_from_key(self, key: str) -> str:
        """
        Extract image ID from various OCR key formats.
        
        Args:
            key: Key from OCR JSON file (could be image path, filename, or ID)
            
        Returns:
            Standardized image ID
        """
        # Handle different key formats common in InfographicsVQA:
        # - "infographicsVQA_images/12345.png" -> "12345.png" or "12345"
        # - "12345.png" -> "12345.png" 
        # - "12345" -> "12345"
        
        if '/' in key:
            # Extract filename from path
            return Path(key).name
        else:
            # Return as-is (could be filename or stem)
            return key

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single QA sample by its index and format it into the project's 
        standardized dictionary by combining information from the QA annotations, 
        OCR results, and the image file.
        
        Args:
            index: The integer index of the sample in the _index list
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary containing:
                - source_dataset: Name of the dataset
                - sample_id: Question ID
                - media_type: "image"
                - media_path: Absolute path to the image file
                - width, height: Image dimensions
                - annotations: Dictionary containing question, answers, and ocr_tokens
        """
        # Retrieve QA Sample
        qa_sample = self._index[index]
        
        # Extract Key Information
        question = qa_sample.get('question', '')
        answers = qa_sample.get('answers', [])
        question_id = qa_sample.get('questionId', index)
        image_filename = qa_sample.get('image_local_name', '')
        
        if not image_filename:
            raise ValueError(f"Sample at index {index} missing 'image_local_name' field")
        
        # Construct Image Path
        image_path = self.image_path / image_filename
        
        # Create Base Structure
        sample = self._get_standardized_base(
            sample_id=str(question_id),
            media_path=image_path,
            media_type="image"
        )
        
        # Retrieve OCR Data
        ocr_data = self._image_id_to_ocr.get(image_filename, {})
        
        # Adapt and Populate Annotations (The "Translation" Step)
        sample['annotations']['question'] = question
        sample['annotations']['answers'] = answers
        
        # Process OCR tokens - extract text and bounding boxes
        ocr_tokens = self._extract_ocr_tokens(ocr_data)
        sample['annotations']['ocr_tokens'] = ocr_tokens
        
        return sample

    def _extract_ocr_tokens(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract OCR tokens and their bounding boxes from the OCR data.
        
        Args:
            ocr_data: Raw OCR data for an image
            
        Returns:
            List of dictionaries containing OCR token information
        """
        ocr_tokens = []
        
        if not ocr_data:
            return ocr_tokens
        
        # Handle different OCR data formats
        # Format 1: AWS Textract format (most common for InfographicsVQA)
        if 'LINE' in ocr_data:
            for line in ocr_data['LINE']:
                if 'Text' in line and 'Geometry' in line:
                    # Extract text and bounding box
                    text = line['Text']
                    geometry = line['Geometry']
                    
                    # Get bounding box coordinates
                    if 'BoundingBox' in geometry:
                        bbox = geometry['BoundingBox']
                        # Convert normalized coordinates to a standard format
                        ocr_tokens.append({
                            'text': text,
                            'bbox': [
                                bbox.get('Left', 0),
                                bbox.get('Top', 0), 
                                bbox.get('Width', 0),
                                bbox.get('Height', 0)
                            ],
                            'confidence': line.get('Confidence', 0)
                        })
        
        # Format 2: WORD level tokens (if available)
        elif 'WORD' in ocr_data:
            for word in ocr_data['WORD']:
                if 'Text' in word and 'Geometry' in word:
                    text = word['Text']
                    geometry = word['Geometry']
                    
                    if 'BoundingBox' in geometry:
                        bbox = geometry['BoundingBox']
                        ocr_tokens.append({
                            'text': text,
                            'bbox': [
                                bbox.get('Left', 0),
                                bbox.get('Top', 0),
                                bbox.get('Width', 0), 
                                bbox.get('Height', 0)
                            ],
                            'confidence': word.get('Confidence', 0)
                        })
        
        # Format 3: Alternative structure (fallback)
        elif 'words' in ocr_data:
            for word_info in ocr_data['words']:
                if isinstance(word_info, dict) and 'text' in word_info:
                    ocr_tokens.append({
                        'text': word_info['text'],
                        'bbox': word_info.get('bbox', [0, 0, 0, 0]),
                        'confidence': word_info.get('confidence', 0)
                    })
        
        return ocr_tokens