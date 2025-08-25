# core/dataloaders/hiertext_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import mmap

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class HierTextLoader(BaseLoader):
    """
    A concrete data loader for the HierText dataset.

    This loader is specifically designed to handle HierText's structure, which consists of:
    1. A single, large JSON file containing annotations for all images.
    2. A directory of image files whose names correspond to the `image_id` in the annotations.
    
    HierText provides hierarchical text annotations with paragraph->line->word structure,
    including bounding box vertices and legibility information.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HierText loader.
        
        Args:
            config: Configuration dictionary containing:
                - name: Name of the datasource
                - image_path: Path to the directory containing images
                - annotation_file: Path to the JSON annotations file
        """
        # Validate required config keys
        required_keys = ['image_path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"HierTextLoader config must include '{key}'")
        
        # Store paths
        self.image_path = Path(config['image_path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Will be populated by _build_index
        self._id_to_annotation: Dict[str, Dict] = {}
        
        # Call parent init (which will call _build_index)
        super().__init__(config)
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load and build index from the HierText JSON file.
        
        Due to the large size of the file, we use memory-efficient parsing.
        
        Returns:
            List of annotation dictionaries
        """
        logger.info(f"Building index for HierText dataset from {self.annotation_file}")
        logger.info("Note: This may take a while due to the large file size...")
        
        annotations = []
        
        try:
            # Load the JSON file
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                logger.info("Loading HierText annotations...")
                data = json.load(f)
            
            # Extract annotations array
            if 'annotations' not in data:
                raise ValueError(f"Annotation file {self.annotation_file} missing 'annotations' field")
            
            raw_annotations = data['annotations']
            logger.info(f"Found {len(raw_annotations)} annotations in HierText dataset")
            
            # Process each annotation
            for ann in raw_annotations:
                if 'image_id' not in ann:
                    logger.warning("Annotation missing 'image_id', skipping")
                    continue
                
                image_id = ann['image_id']
                
                # Store full annotation for later retrieval
                self._id_to_annotation[image_id] = ann
                
                # Add to index (store minimal info)
                annotations.append({
                    'image_id': image_id,
                    'has_paragraphs': 'paragraphs' in ann and len(ann.get('paragraphs', [])) > 0
                })
            
            logger.info(f"Successfully indexed {len(annotations)} valid annotations")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
        except MemoryError:
            logger.error("File too large to load into memory. Consider using streaming JSON parser.")
            raise
        
        return annotations
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single annotation by its index and format it.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Standardized sample dictionary with HierText-specific annotations
        """
        # Get the annotation info from index
        ann_info = self._index[index]
        image_id = ann_info['image_id']
        
        # Get full annotation
        if image_id not in self._id_to_annotation:
            raise KeyError(f"Annotation for image_id {image_id} not found in cache")
        
        raw_annotation = self._id_to_annotation[image_id]
        
        # Construct image path (assuming .jpg extension)
        image_filename = f"{image_id}.jpg"
        image_full_path = self.image_path / image_filename
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=image_id,
            media_path=image_full_path,
            media_type="image"
        )
        
        # Add HierText-specific annotations
        # Store the full hierarchical structure
        sample['annotations']['hierarchical_text'] = raw_annotation.get('paragraphs', [])
        
        # Extract flattened word list for easier processing
        all_words = self._extract_flat_words(raw_annotation)
        sample['annotations']['flat_word_list'] = all_words
        
        # Add summary statistics
        sample['annotations']['num_paragraphs'] = len(raw_annotation.get('paragraphs', []))
        sample['annotations']['num_words'] = len(all_words)
        
        # Extract all text as a single string (useful for some tasks)
        all_text = self._extract_all_text(raw_annotation)
        sample['annotations']['full_text'] = all_text
        
        return sample
    
    def _extract_flat_words(self, annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract a flattened list of all words from the hierarchical structure.
        
        Args:
            annotation: Raw HierText annotation dictionary
            
        Returns:
            List of word dictionaries with text, vertices, and metadata
        """
        all_words = []
        
        for para_idx, paragraph in enumerate(annotation.get('paragraphs', [])):
            para_legible = paragraph.get('legible', True)
            para_vertices = paragraph.get('vertices', [])
            
            for line_idx, line in enumerate(paragraph.get('lines', [])):
                line_text = line.get('text', '')
                line_vertices = line.get('vertices', [])
                line_legible = line.get('legible', para_legible)
                
                for word_idx, word in enumerate(line.get('words', [])):
                    word_entry = {
                        'text': word.get('text', ''),
                        'vertices': word.get('vertices', []),
                        'legible': word.get('legible', line_legible),
                        'paragraph_idx': para_idx,
                        'line_idx': line_idx,
                        'word_idx': word_idx,
                        'line_text': line_text  # Context
                    }
                    
                    # Convert vertices to bbox if possible
                    if word_entry['vertices']:
                        bbox = self._vertices_to_bbox(word_entry['vertices'])
                        if bbox:
                            word_entry['bbox'] = bbox
                    
                    all_words.append(word_entry)
        
        return all_words
    
    def _extract_all_text(self, annotation: Dict[str, Any]) -> str:
        """
        Extract all text from the annotation as a single string.
        
        Args:
            annotation: Raw HierText annotation dictionary
            
        Returns:
            All text concatenated with appropriate separators
        """
        paragraphs_text = []
        
        for paragraph in annotation.get('paragraphs', []):
            lines_text = []
            
            for line in paragraph.get('lines', []):
                line_text = line.get('text', '')
                if line_text:
                    lines_text.append(line_text)
            
            if lines_text:
                paragraphs_text.append(' '.join(lines_text))
        
        return '\n\n'.join(paragraphs_text)
    
    def _vertices_to_bbox(self, vertices: List[List[int]]) -> Optional[List[int]]:
        """
        Convert polygon vertices to axis-aligned bounding box.
        
        Args:
            vertices: List of [x, y] coordinates
            
        Returns:
            [x_min, y_min, x_max, y_max] or None if vertices are invalid
        """
        if not vertices or len(vertices) < 2:
            return None
        
        try:
            x_coords = [v[0] for v in vertices if len(v) >= 2]
            y_coords = [v[1] for v in vertices if len(v) >= 2]
            
            if not x_coords or not y_coords:
                return None
            
            return [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords)
            ]
        except (TypeError, ValueError):
            return None


class HierTextStreamingLoader(HierTextLoader):
    """
    Alternative implementation using streaming JSON parsing for very large files.
    
    This is more memory-efficient but slower for random access.
    Note: This requires the ijson library (pip install ijson).
    """
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build index using streaming JSON parser to handle very large files.
        
        Returns:
            List of annotation dictionaries
        """
        logger.info(f"Building index for HierText dataset using streaming parser")
        logger.info(f"Reading from {self.annotation_file}")
        
        annotations = []
        
        try:
            import ijson
        except ImportError:
            logger.warning("ijson not installed, falling back to standard loader")
            return super()._build_index()
        
        try:
            with open(self.annotation_file, 'rb') as f:
                # Parse annotations array incrementally
                parser = ijson.items(f, 'annotations.item')
                
                for idx, ann in enumerate(parser):
                    if idx % 1000 == 0:
                        logger.info(f"Processed {idx} annotations...")
                    
                    if 'image_id' not in ann:
                        continue
                    
                    image_id = ann['image_id']
                    
                    # Store full annotation
                    self._id_to_annotation[image_id] = ann
                    
                    # Add to index
                    annotations.append({
                        'image_id': image_id,
                        'has_paragraphs': 'paragraphs' in ann and len(ann.get('paragraphs', [])) > 0
                    })
            
            logger.info(f"Successfully indexed {len(annotations)} annotations using streaming parser")
            
        except Exception as e:
            logger.error(f"Error during streaming parse: {e}")
            raise
        
        return annotations