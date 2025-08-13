"""
Read Text Operation

Implements the READ_TEXT visual operation for optical character recognition (OCR).
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..operation_registry import BaseOperation, registry


class ReadTextOperation(BaseOperation):
    """
    Reads and extracts text from a specified region in an image.
    
    This operation performs OCR on either the entire image or a specified
    region, returning the extracted text along with confidence scores and
    bounding boxes for detected text regions.
    """
    
    def __init__(self):
        """Initialize the read text operation."""
        super().__init__()
        self.ocr_model = None  # Will be loaded lazily
    
    def _load_model(self):
        """
        Lazily load the OCR model.
        
        This could be TrOCR, PaddleOCR, or another OCR model.
        """
        if self.ocr_model is None:
            self.logger.info("Loading OCR model...")
            # Placeholder for actual model loading
            # Example: self.ocr_model = load_trocr_model()
            self.ocr_model = "placeholder_ocr_model"
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input arguments.
        
        Required:
            - image: Image tensor or array
            
        Optional:
            - region: [x1, y1, x2, y2] bounding box for specific region
            - language: Language code for OCR
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if 'image' not in kwargs:
            self.logger.error("Missing required parameter: 'image'")
            return False
        
        # Validate image format
        image = kwargs['image']
        if not isinstance(image, (torch.Tensor, np.ndarray)):
            self.logger.error("Image must be a torch.Tensor or numpy.ndarray")
            return False
        
        # Validate region if provided
        if 'region' in kwargs:
            region = kwargs['region']
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                self.logger.error("'region' must be [x1, y1, x2, y2]")
                return False
            
            x1, y1, x2, y2 = region
            if x1 >= x2 or y1 >= y2:
                self.logger.error("Invalid region: x1 < x2 and y1 < y2 required")
                return False
        
        return True
    
    def preprocess(self, **kwargs) -> Dict[str, Any]:
        """
        Preprocess inputs.
        
        Converts image to appropriate format and validates region bounds.
        
        Args:
            **kwargs: Raw input arguments
            
        Returns:
            Preprocessed arguments
        """
        processed = kwargs.copy()
        
        # Convert image to tensor if needed
        image = kwargs['image']
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = torch.from_numpy(image).unsqueeze(0)
            elif len(image.shape) == 3:  # HWC
                image = torch.from_numpy(image).permute(2, 0, 1)
            processed['image'] = image.float()
        
        # Get image dimensions
        if len(processed['image'].shape) == 3:  # CHW
            _, h, w = processed['image'].shape
        else:  # BCHW
            _, _, h, w = processed['image'].shape
        
        # Validate and clip region if provided
        if 'region' in processed:
            x1, y1, x2, y2 = processed['region']
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            processed['region'] = [int(x1), int(y1), int(x2), int(y2)]
        
        return processed
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the OCR operation.
        
        Args:
            image: Image tensor (CHW or BCHW format)
            region: Optional [x1, y1, x2, y2] bounding box for specific region
            language: Optional language code (default: 'en')
            return_confidence: Whether to return confidence scores
            return_boxes: Whether to return text bounding boxes
            
        Returns:
            Dictionary containing:
                - text: Extracted text string
                - lines: List of detected text lines
                - words: List of detected words with positions
                - confidence: Overall confidence score
                - boxes: Bounding boxes for each text element (if requested)
                - language: Detected or specified language
        """
        # Validate and preprocess inputs
        if not self.validate_inputs(**kwargs):
            raise ValueError("Invalid inputs for read text operation")
        
        processed = self.preprocess(**kwargs)
        
        # Load model if needed
        self._load_model()
        
        # Extract processed inputs
        image = processed['image']
        region = processed.get('region', None)
        language = processed.get('language', 'en')
        return_confidence = processed.get('return_confidence', True)
        return_boxes = processed.get('return_boxes', True)
        
        # Crop image to region if specified
        if region is not None:
            x1, y1, x2, y2 = region
            if len(image.shape) == 3:  # CHW
                image_crop = image[:, y1:y2, x1:x2]
            else:  # BCHW
                image_crop = image[:, :, y1:y2, x1:x2]
        else:
            image_crop = image
        
        # Placeholder for actual OCR
        # In production, this would use the loaded OCR model
        # Example: text_results = self.ocr_model.recognize(image_crop)
        
        # Generate dummy results for demonstration
        text_lines = [
            "This is a sample text",
            "detected by the OCR system",
            "in the specified region"
        ]
        
        words = []
        current_y = 10
        for line in text_lines:
            line_words = line.split()
            current_x = 10
            for word in line_words:
                word_info = {
                    'text': word,
                    'confidence': np.random.uniform(0.85, 0.99),
                    'position': [current_x, current_y]
                }
                if return_boxes:
                    # Estimate word width (placeholder)
                    word_width = len(word) * 8
                    word_height = 12
                    word_info['box'] = [
                        current_x,
                        current_y,
                        current_x + word_width,
                        current_y + word_height
                    ]
                words.append(word_info)
                current_x += len(word) * 8 + 5
            current_y += 20
        
        # Combine all text
        full_text = ' '.join(text_lines)
        
        # Calculate overall confidence
        if words:
            overall_confidence = np.mean([w['confidence'] for w in words])
        else:
            overall_confidence = 0.0
        
        result = {
            'text': full_text,
            'lines': text_lines,
            'words': words,
            'language': language,
            'num_words': len(words),
            'num_lines': len(text_lines)
        }
        
        if return_confidence:
            result['confidence'] = float(overall_confidence)
        
        if return_boxes and region is not None:
            # Adjust boxes to global coordinates if region was specified
            x1, y1, _, _ = region
            for word in words:
                if 'box' in word:
                    word['box'][0] += x1
                    word['box'][1] += y1
                    word['box'][2] += x1
                    word['box'][3] += y1
                word['position'][0] += x1
                word['position'][1] += y1
        
        self.logger.debug(
            f"Extracted {len(words)} words in {len(text_lines)} lines "
            f"with confidence {overall_confidence:.2f}"
        )
        
        return result
    
    def postprocess(self, result: Any) -> Any:
        """
        Postprocess the operation result.
        
        Clean up text and format output.
        
        Args:
            result: Raw operation result
            
        Returns:
            Postprocessed result
        """
        if isinstance(result, dict) and 'text' in result:
            # Clean up text: remove extra spaces, normalize whitespace
            result['text'] = ' '.join(result['text'].split())
            
            # Clean up lines
            if 'lines' in result:
                result['lines'] = [' '.join(line.split()) for line in result['lines']]
        
        return result
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        return ['image']
    
    def get_optional_params(self) -> Dict[str, Any]:
        """Get dictionary of optional parameters with defaults."""
        return {
            'region': None,
            'language': 'en',
            'return_confidence': True,
            'return_boxes': True
        }


# Register the operation with the global registry
registry.register(
    'READ_TEXT',
    ReadTextOperation,
    metadata={
        'description': 'Extract text from an image or image region using OCR',
        'category': 'text_extraction',
        'input_types': {
            'image': 'torch.Tensor or numpy.ndarray',
            'region': 'Optional[List[int]]',
            'language': 'str'
        },
        'output_types': {
            'text': 'str',
            'lines': 'List[str]',
            'words': 'List[Dict]',
            'confidence': 'float'
        }
    }
)