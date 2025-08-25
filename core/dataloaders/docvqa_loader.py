# core/dataloaders/docvqa_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class DocVqaLoader(BaseLoader):
    """
    A concrete data loader for the Single Page Document VQA (SP-DocVQA) dataset.

    This loader is responsible for:
    1. Loading the primary QA annotation file.
    2. Loading the corresponding pre-computed OCR file.
    3. Building an index of all available question-answer samples.
    4. Translating a raw DocVQA sample into the project's standard format,
       combining information from the QA, OCR, and image files.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DocVQA loader.
        
        Args:
            config: Configuration dictionary containing:
                - name: Name of the datasource
                - image_path: Path to the directory containing images
                - annotation_file: Path to the QA annotations JSON file
                - ocr_path: Path to the directory containing OCR JSON files
        """
        # Validate required config keys
        required_keys = ['image_path', 'annotation_file', 'ocr_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"DocVqaLoader config must include '{key}'")
        
        # Store paths
        self.image_path = Path(config['image_path'])
        self.annotation_file = Path(config['annotation_file'])
        self.ocr_path = Path(config['ocr_path'])
        
        # Validate paths exist
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.ocr_path.exists():
            raise FileNotFoundError(f"OCR directory not found: {self.ocr_path}")
        
        # Will be populated by _build_index
        self._image_id_to_ocr: Dict[str, Dict] = {}
        
        # Call parent init (which will call _build_index)
        super().__init__(config)
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load and build index from annotation files.
        
        Returns:
            List of QA samples from the annotation file
        """
        logger.info(f"Building index for DocVQA dataset from {self.annotation_file}")
        
        # Load QA annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # Extract the data list (the actual QA samples)
        if 'data' not in qa_data:
            raise ValueError(f"Annotation file {self.annotation_file} missing 'data' field")
        
        qa_samples = qa_data['data']
        logger.info(f"Loaded {len(qa_samples)} QA samples from annotations")
        
        # Pre-load OCR data for efficient lookup
        self._preload_ocr_data(qa_samples)
        
        return qa_samples
    
    def _preload_ocr_data(self, qa_samples: List[Dict[str, Any]]) -> None:
        """
        Pre-load OCR data for all images referenced in QA samples.
        
        Args:
            qa_samples: List of QA samples containing image references
        """
        # Get unique image IDs from QA samples
        image_ids = set()
        for sample in qa_samples:
            if 'image' in sample:
                # Extract image ID from path (e.g., "documents/xnbl0037_1.png" -> "xnbl0037_1")
                image_filename = Path(sample['image']).stem
                image_ids.add(image_filename)
        
        logger.info(f"Pre-loading OCR data for {len(image_ids)} unique images")
        
        # Load OCR data for each unique image
        loaded_count = 0
        missing_count = 0
        
        for image_id in image_ids:
            ocr_file = self.ocr_path / f"{image_id}.json"
            
            if ocr_file.exists():
                try:
                    with open(ocr_file, 'r', encoding='utf-8') as f:
                        ocr_data = json.load(f)
                        self._image_id_to_ocr[image_id] = ocr_data
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load OCR file {ocr_file}: {e}")
                    missing_count += 1
            else:
                logger.debug(f"OCR file not found: {ocr_file}")
                missing_count += 1
        
        logger.info(f"Successfully loaded OCR data for {loaded_count} images, {missing_count} missing")
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single QA sample by its index and format it.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Standardized sample dictionary with DocVQA-specific annotations
        """
        # Get the QA sample from index
        qa_sample = self._index[index]
        
        # Extract key information
        question = qa_sample.get('question', '')
        answers = qa_sample.get('answers', [])
        question_id = qa_sample.get('questionId', index)
        question_types = qa_sample.get('question_types', [])
        
        # Extract image path and ID
        image_relative = qa_sample.get('image', '')
        if not image_relative:
            raise ValueError(f"Sample at index {index} missing 'image' field")
        
        # Remove 'documents/' prefix if present
        image_filename = Path(image_relative).name
        image_id = Path(image_relative).stem
        
        # Construct full image path
        image_full_path = self.image_path / image_filename
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=question_id,
            media_path=image_full_path,
            media_type="image"
        )
        
        # Add DocVQA-specific annotations
        sample['annotations']['question'] = question
        sample['annotations']['answers'] = answers
        sample['annotations']['question_types'] = question_types
        
        # Add document metadata if available
        if 'ucsf_document_id' in qa_sample:
            sample['annotations']['document_id'] = qa_sample['ucsf_document_id']
        if 'ucsf_document_page_no' in qa_sample:
            sample['annotations']['page_number'] = qa_sample['ucsf_document_page_no']
        
        # Add OCR data if available
        if image_id in self._image_id_to_ocr:
            ocr_data = self._image_id_to_ocr[image_id]
            ocr_tokens = self._extract_ocr_tokens(ocr_data)
            sample['annotations']['ocr_tokens'] = ocr_tokens
            
            # Optional: Find answer bounding boxes within OCR tokens
            answer_bboxes = self._find_answer_bboxes(answers, ocr_tokens)
            if answer_bboxes:
                sample['annotations']['answer_bboxes'] = answer_bboxes
        else:
            logger.debug(f"No OCR data available for image {image_id}")
            sample['annotations']['ocr_tokens'] = []
        
        return sample
    
    def _extract_ocr_tokens(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract OCR tokens with bounding boxes from raw OCR data.
        
        Args:
            ocr_data: Raw OCR data from JSON file
            
        Returns:
            List of dictionaries containing text and bbox for each word
        """
        tokens = []
        
        # Check if OCR data has the expected structure
        if 'recognitionResults' not in ocr_data:
            logger.warning("OCR data missing 'recognitionResults' field")
            return tokens
        
        # Process each page in the OCR results
        for page_result in ocr_data['recognitionResults']:
            if 'lines' not in page_result:
                continue
                
            page_num = page_result.get('page', 1)
            
            # Process each line of text
            for line in page_result['lines']:
                if 'words' not in line:
                    continue
                
                # Process each word in the line
                for word in line['words']:
                    if 'text' in word and 'boundingBox' in word:
                        # Convert bounding box format (8 points to x,y,w,h)
                        bbox_points = word['boundingBox']
                        if len(bbox_points) >= 8:
                            x_coords = bbox_points[0::2]  # Even indices
                            y_coords = bbox_points[1::2]  # Odd indices
                            
                            x_min = min(x_coords)
                            y_min = min(y_coords)
                            x_max = max(x_coords)
                            y_max = max(y_coords)
                            
                            tokens.append({
                                'text': word['text'],
                                'bbox': [x_min, y_min, x_max, y_max],
                                'page': page_num,
                                'confidence': word.get('confidence', 1.0)
                            })
        
        return tokens
    
    def _find_answer_bboxes(self, answers: List[str], ocr_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find bounding boxes for answer phrases within OCR tokens.
        
        Args:
            answers: List of answer strings
            ocr_tokens: List of OCR tokens with text and bounding boxes
            
        Returns:
            List of dictionaries containing answer text and corresponding bbox
        """
        answer_bboxes = []
        
        # Create a simple text index for fast lookup
        token_texts = [token['text'].lower() for token in ocr_tokens]
        
        for answer in answers:
            answer_lower = answer.lower()
            answer_words = answer_lower.split()
            
            if not answer_words:
                continue
            
            # Try to find consecutive tokens matching the answer
            for i in range(len(token_texts) - len(answer_words) + 1):
                # Check if tokens match answer words
                if all(token_texts[i + j] == answer_words[j] for j in range(len(answer_words))):
                    # Found matching tokens, compute combined bbox
                    matching_tokens = ocr_tokens[i:i + len(answer_words)]
                    
                    # Compute union of bounding boxes
                    x_mins = [t['bbox'][0] for t in matching_tokens]
                    y_mins = [t['bbox'][1] for t in matching_tokens]
                    x_maxs = [t['bbox'][2] for t in matching_tokens]
                    y_maxs = [t['bbox'][3] for t in matching_tokens]
                    
                    combined_bbox = [
                        min(x_mins),
                        min(y_mins),
                        max(x_maxs),
                        max(y_maxs)
                    ]
                    
                    answer_bboxes.append({
                        'answer': answer,
                        'bbox': combined_bbox,
                        'token_indices': list(range(i, i + len(answer_words)))
                    })
                    break  # Found first match for this answer
        
        return answer_bboxes