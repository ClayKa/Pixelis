"""
Targeted OCR Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for contextual text reading
and OCR tasks using READ_TEXT operations with spatial targeting.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class TargetedOCRTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for targeted OCR and contextual text reading tasks.
    
    This generator creates tasks that require:
    - Reading text from specific regions of images
    - Understanding contextual relationships between text elements
    - Extracting structured information from documents
    - Multi-lingual text recognition
    - Scene text understanding
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the targeted OCR task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders mapping
        self.difficulty_loaders = {
            'easy': ['textvqa_train', 'textcaps_train'],
            'medium': ['infographics_vqa_train', 'docvqa_train'],
            'hard': ['chartqa_train', 'hierarchical_text_train']
        }
        
        # Text extraction scenarios
        self.text_scenarios = [
            'sign_reading', 'document_extraction', 'menu_parsing',
            'receipt_analysis', 'chart_reading', 'interface_text',
            'handwritten_notes', 'multi_column_layout', 'table_extraction'
        ]
        
        # OCR challenges by difficulty
        self.ocr_challenges = {
            'easy': ['clear_text', 'high_contrast', 'single_line'],
            'medium': ['partial_occlusion', 'curved_text', 'multiple_fonts'],
            'hard': ['low_resolution', 'distorted_text', 'artistic_fonts', 'multilingual']
        }
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """
        Build context placeholders for the OCR task prompt.
        
        Returns:
            Dictionary mapping placeholder names to their values
        """
        placeholders = {}
        
        # Build context for each difficulty level
        placeholders.update(self._build_easy_context())
        placeholders.update(self._build_medium_context())
        placeholders.update(self._build_hard_context())
        
        # Add general context
        placeholders['task_description'] = self._generate_task_description()
        placeholders['available_operations'] = self._get_available_operations()
        placeholders['output_format'] = self._get_output_format()
        placeholders['ocr_requirements'] = self._get_ocr_requirements()
        
        # Track source datasets for provenance
        placeholders['source_datasets'] = self._get_active_datasets()
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build context for easy OCR tasks."""
        context = {}
        
        # Try to get TextVQA or TextCaps loader
        loader = None
        for loader_name in ['textvqa_train', 'textcaps_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            logger.warning("No easy OCR loader available, using mock data")
            return self._build_mock_easy_context()
        
        try:
            # Sample an image with text
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build easy context
            context['easy_source_dataset'] = loader_name.replace('_', ' ').title()
            context['easy_image_type'] = random.choice([
                'street sign', 'product label', 'book cover', 
                'storefront', 'poster', 'digital display'
            ])
            context['easy_text_location'] = self._describe_text_location('easy', sample)
            context['easy_text_characteristics'] = ', '.join(random.sample(self.ocr_challenges['easy'], 2))
            context['easy_extraction_goal'] = random.choice([
                'read the main text',
                'extract the title',
                'identify the brand name',
                'read the price information',
                'extract the date'
            ])
            context['easy_expected_text_length'] = random.choice(['word', 'phrase', 'short sentence'])
            context['easy_language'] = 'English'
            context['easy_confidence_threshold'] = '0.8'
            
        except Exception as e:
            logger.error(f"Error building easy OCR context: {e}")
            return self._build_mock_easy_context()
        
        return context
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build context for medium difficulty OCR tasks."""
        context = {}
        
        # Try document-oriented loaders
        loader = None
        for loader_name in ['infographics_vqa_train', 'docvqa_train', 'fineweb_edu_10bt']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            logger.warning("No medium OCR loader available, using mock data")
            return self._build_mock_medium_context()
        
        try:
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build medium context
            context['medium_source_dataset'] = loader_name.replace('_', ' ').title()
            context['medium_document_type'] = random.choice([
                'infographic', 'invoice', 'form', 'presentation slide',
                'technical diagram', 'newspaper article', 'menu'
            ])
            context['medium_layout_complexity'] = random.choice([
                'multi-column layout',
                'mixed text and graphics',
                'hierarchical structure',
                'tabular format'
            ])
            context['medium_text_regions'] = str(random.randint(3, 7))
            context['medium_extraction_task'] = random.choice([
                'extract all heading texts',
                'read specific data fields',
                'identify and extract key-value pairs',
                'parse structured information',
                'extract text following a pattern'
            ])
            context['medium_text_challenges'] = ', '.join(random.sample(self.ocr_challenges['medium'], 2))
            context['medium_required_structure'] = random.choice([
                'maintain reading order',
                'preserve hierarchical relationships',
                'group related text elements',
                'separate columns correctly'
            ])
            context['medium_languages'] = random.choice([
                'English only', 'English with technical terms',
                'Mixed English and numbers', 'English with symbols'
            ])
            
        except Exception as e:
            logger.error(f"Error building medium OCR context: {e}")
            return self._build_mock_medium_context()
        
        return context
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build context for hard OCR tasks."""
        context = {}
        
        # Try complex document loaders
        loader = None
        for loader_name in ['chartqa_train', 'hierarchical_text_train', 'mathvista_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            # Fallback to any available loader
            if self.loaders:
                loader = list(self.loaders.values())[0]
            else:
                logger.warning("No hard OCR loader available, using mock data")
                return self._build_mock_hard_context()
        
        try:
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build hard context
            context['hard_source_dataset'] = loader_name.replace('_', ' ').title() if loader_name else 'Complex Documents'
            context['hard_document_complexity'] = random.choice([
                'scientific paper with equations',
                'multilingual document',
                'historical manuscript',
                'complex chart with annotations',
                'handwritten mathematical notes',
                'degraded or damaged document',
                'artistic typography design'
            ])
            context['hard_ocr_challenges'] = ', '.join(random.sample(self.ocr_challenges['hard'], 3))
            context['hard_extraction_requirements'] = random.choice([
                'extract and structure hierarchical information',
                'parse complex mathematical expressions',
                'maintain precise spatial relationships',
                'handle multiple scripts and languages',
                'reconstruct fragmented text',
                'extract text from low-quality regions'
            ])
            context['hard_text_regions'] = str(random.randint(8, 15))
            context['hard_special_handling'] = random.choice([
                'handle overlapping text regions',
                'detect and correct OCR errors',
                'infer missing characters from context',
                'normalize text variations',
                'handle vertical and rotated text'
            ])
            context['hard_output_structure'] = random.choice([
                'nested JSON with spatial metadata',
                'structured table with row/column indices',
                'hierarchical tree of text elements',
                'graph of text relationships'
            ])
            context['hard_accuracy_requirement'] = 'character-level precision with confidence scores'
            context['hard_post_processing'] = random.choice([
                'spell checking and correction',
                'semantic validation',
                'format normalization',
                'entity recognition and linking'
            ])
            
        except Exception as e:
            logger.error(f"Error building hard OCR context: {e}")
            return self._build_mock_hard_context()
        
        return context
    
    def _describe_text_location(self, difficulty: str, sample: Dict) -> str:
        """Generate description of text location in image."""
        locations = {
            'easy': [
                'center of the image',
                'top portion',
                'clearly visible in the foreground',
                'on a flat surface'
            ],
            'medium': [
                'distributed across multiple regions',
                'in various sections of the document',
                'within structured layouts',
                'mixed with graphical elements'
            ],
            'hard': [
                'scattered throughout complex layouts',
                'partially obscured regions',
                'at various orientations and scales',
                'embedded in dense visual contexts'
            ]
        }
        return random.choice(locations.get(difficulty, locations['medium']))
    
    def _generate_task_description(self) -> str:
        """Generate overall task description."""
        return """Extract and process text from images using targeted OCR operations. 
        The task requires identifying text regions, applying appropriate OCR techniques, 
        and structuring the extracted information according to the specified requirements."""
    
    def _get_available_operations(self) -> str:
        """Get available OCR operations."""
        return """
        - READ_TEXT(bbox): Extract text from specified bounding box
        - READ_TEXT_REGION(region_type): Extract text from semantic region
        - DETECT_TEXT_AREAS(): Identify all text-containing regions
        - OCR_WITH_CORRECTION(bbox, language): OCR with error correction
        - EXTRACT_STRUCTURED_TEXT(template): Extract text matching template
        """
    
    def _get_output_format(self) -> str:
        """Get expected output format."""
        return """
        {
            "trajectory": [
                {
                    "step": int,
                    "thought": "reasoning about text location and extraction strategy",
                    "action": "READ_TEXT or variant",
                    "parameters": {"bbox": [x1, y1, x2, y2], "options": {...}},
                    "result": "extracted text content"
                }
            ],
            "extracted_texts": {
                "region_1": {"text": "...", "confidence": float},
                "region_2": {"text": "...", "confidence": float}
            },
            "structured_output": "formatted according to requirements",
            "final_answer": "complete extracted and processed text"
        }
        """
    
    def _get_ocr_requirements(self) -> str:
        """Get OCR-specific requirements."""
        return """
        - Maintain original text formatting when relevant
        - Preserve spatial relationships between text elements
        - Include confidence scores for extracted text
        - Handle special characters and punctuation correctly
        - Apply appropriate language models for correction
        """
    
    def _get_active_datasets(self) -> List[str]:
        """Get list of active OCR datasets."""
        active = []
        ocr_datasets = [
            'textvqa', 'textcaps', 'docvqa', 'infographics',
            'chartqa', 'fineweb', 'hierarchical_text', 'mathvista'
        ]
        for loader_name in self.loaders.keys():
            if any(dataset in loader_name.lower() for dataset in ocr_datasets):
                active.append(loader_name)
        return active
    
    # Mock context builders
    def _build_mock_easy_context(self) -> Dict[str, str]:
        """Build mock easy OCR context."""
        return {
            'easy_source_dataset': 'TextVQA (mock)',
            'easy_image_type': 'street sign',
            'easy_text_location': 'center of the image',
            'easy_text_characteristics': 'clear_text, high_contrast',
            'easy_extraction_goal': 'read the main text',
            'easy_expected_text_length': 'phrase',
            'easy_language': 'English',
            'easy_confidence_threshold': '0.8'
        }
    
    def _build_mock_medium_context(self) -> Dict[str, str]:
        """Build mock medium OCR context."""
        return {
            'medium_source_dataset': 'DocVQA (mock)',
            'medium_document_type': 'invoice',
            'medium_layout_complexity': 'multi-column layout',
            'medium_text_regions': '5',
            'medium_extraction_task': 'extract key-value pairs',
            'medium_text_challenges': 'partial_occlusion, multiple_fonts',
            'medium_required_structure': 'maintain reading order',
            'medium_languages': 'English only'
        }
    
    def _build_mock_hard_context(self) -> Dict[str, str]:
        """Build mock hard OCR context."""
        return {
            'hard_source_dataset': 'ChartQA (mock)',
            'hard_document_complexity': 'scientific paper with equations',
            'hard_ocr_challenges': 'low_resolution, distorted_text, multilingual',
            'hard_extraction_requirements': 'parse complex mathematical expressions',
            'hard_text_regions': '12',
            'hard_special_handling': 'handle overlapping text regions',
            'hard_output_structure': 'nested JSON with spatial metadata',
            'hard_accuracy_requirement': 'character-level precision with confidence scores',
            'hard_post_processing': 'semantic validation'
        }