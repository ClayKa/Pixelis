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
        required_keys = ['image_path', 'annotation_file', 'ocr_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"InfographicsVqaLoader config must include '{key}'")
        
        # Store paths
        self.image_path = Path(config['image_path'])
        self.annotation_file = Path(config['annotation_file'])
        self.ocr_file = Path(config['ocr_file'])
        
        # Determine if ocr_file is a directory or a file
        if self.ocr_file.is_dir():
            self.ocr_dir = self.ocr_file
            self.ocr_file = None  # No single OCR file
        else:
            self.ocr_dir = self.ocr_file.parent
        
        # Validate paths exist
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if self.ocr_file and not self.ocr_file.exists():
            raise FileNotFoundError(f"OCR file not found: {self.ocr_file}")
        if not self.ocr_dir.exists():
            raise FileNotFoundError(f"OCR directory not found: {self.ocr_dir}")
        
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
        Load and pre-process OCR data for efficiency. Create an internal lookup 
        dictionary that maps each image identifier to its corresponding OCR results.
        
        For InfographicsVQA, OCR data is typically stored as individual JSON files
        per image in the OCR directory.
        
        Args:
            qa_samples: List of QA samples containing image references
        """
        logger.info(f"Loading OCR data from {self.ocr_dir}")
        
        # Get unique image filenames from QA samples
        image_filenames = set()
        for sample in qa_samples:
            if 'image_local_name' in sample:
                image_filenames.add(sample['image_local_name'])
        
        logger.info(f"Processing OCR data for {len(image_filenames)} unique images")
        
        # First try to load from a consolidated OCR file if it exists
        consolidated_ocr_data = {}
        if self.ocr_file and self.ocr_file.exists():
            try:
                with open(self.ocr_file, 'r', encoding='utf-8') as f:
                    consolidated_ocr_data = json.load(f)
                logger.info("Loaded consolidated OCR data")
            except Exception as e:
                logger.warning(f"Failed to load consolidated OCR file {self.ocr_file}: {e}")
        
        # Load OCR data for each image
        loaded_count = 0
        missing_count = 0
        
        for image_filename in image_filenames:
            image_stem = Path(image_filename).stem
            ocr_data = None
            
            # Try consolidated OCR data first
            if consolidated_ocr_data:
                if image_filename in consolidated_ocr_data:
                    ocr_data = consolidated_ocr_data[image_filename]
                elif image_stem in consolidated_ocr_data:
                    ocr_data = consolidated_ocr_data[image_stem]
            
            # If not found in consolidated data, try individual OCR file
            if not ocr_data:
                ocr_file_path = self.ocr_dir / f"{image_stem}.json"
                if ocr_file_path.exists():
                    try:
                        with open(ocr_file_path, 'r', encoding='utf-8') as f:
                            ocr_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load individual OCR file {ocr_file_path}: {e}")
                        ocr_data = {}
                else:
                    logger.debug(f"No OCR file found for image: {image_filename}")
                    ocr_data = {}
            
            # Store the OCR data
            self._image_id_to_ocr[image_filename] = ocr_data
            
            if ocr_data:
                loaded_count += 1
            else:
                missing_count += 1
        
        logger.info(f"Successfully loaded OCR data for {loaded_count} images, {missing_count} missing")

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