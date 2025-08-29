"""
Geometric Comparison Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for geometric reasoning
and property comparison tasks using SEGMENT_OBJECT_AT and GET_PROPERTIES operations.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class GeometricComparisonTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for geometric reasoning and object property comparison.
    
    This generator creates tasks that require:
    - Segmenting objects at specific pixel coordinates
    - Extracting geometric properties (size, shape, orientation)
    - Comparing properties between multiple objects
    - Spatial relationship reasoning
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the geometric comparison task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders mapping
        self.difficulty_loaders = {
            'easy': ['coco2017_train'],
            'medium': ['lvis_v1_train', 'ade20k_train'],
            'hard': ['part_imagenet_train', 'partimagenet_train']
        }
        
        # Property types for comparison
        self.property_types = [
            'size', 'shape', 'orientation', 'texture', 
            'relative_position', 'aspect_ratio', 'symmetry'
        ]
        
        # Comparison operators
        self.comparison_ops = [
            'larger', 'smaller', 'equal', 'similar',
            'above', 'below', 'left_of', 'right_of',
            'more_complex', 'simpler', 'rounder', 'more_elongated'
        ]
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """
        Build context placeholders for the geometric reasoning prompt.
        
        Returns:
            Dictionary mapping placeholder names to their values
        """
        placeholders = {}
        
        # Randomly select difficulty level with weighted distribution
        difficulty_weights = {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}
        difficulty = random.choices(
            list(difficulty_weights.keys()),
            weights=list(difficulty_weights.values())
        )[0]
        
        # Build context for each difficulty level
        placeholders.update(self._build_easy_context())
        placeholders.update(self._build_medium_context())
        placeholders.update(self._build_hard_context())
        
        # Add general context
        placeholders['task_description'] = self._generate_task_description(difficulty)
        placeholders['available_operations'] = self._get_available_operations()
        placeholders['output_format'] = self._get_output_format()
        
        # Track source datasets for provenance
        placeholders['source_datasets'] = self._get_active_datasets()
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build context for easy difficulty level."""
        context = {}
        
        # Get COCO loader
        loader_name = 'coco2017_train'
        if loader_name not in self.loaders:
            logger.warning(f"Loader {loader_name} not available, using mock data")
            return self._build_mock_easy_context()
        
        loader = self.loaders[loader_name]
        
        try:
            # Sample an image with multiple objects
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Extract relevant information
            if 'annotations' in sample and len(sample['annotations']) >= 2:
                # Select two objects for comparison
                obj1, obj2 = random.sample(sample['annotations'], 2)
                
                context['easy_source_dataset'] = 'COCO 2017'
                context['easy_image_description'] = self._describe_image(sample)
                context['easy_object_A_class'] = obj1.get('category_name', 'object')
                context['easy_object_A_bbox'] = str(obj1.get('bbox', [100, 100, 200, 200]))
                context['easy_object_B_class'] = obj2.get('category_name', 'object')
                context['easy_object_B_bbox'] = str(obj2.get('bbox', [300, 300, 400, 400]))
                context['easy_property_to_compare'] = random.choice(['size', 'position', 'shape'])
                context['easy_expected_operations'] = 'SEGMENT_OBJECT_AT, GET_PROPERTIES'
                
            else:
                # Fallback to mock if insufficient annotations
                return self._build_mock_easy_context()
                
        except Exception as e:
            logger.error(f"Error building easy context: {e}")
            return self._build_mock_easy_context()
        
        return context
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build context for medium difficulty level."""
        context = {}
        
        # Try LVIS or ADE20K loader
        for loader_name in ['lvis_v1_train', 'ade20k_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        else:
            logger.warning("No medium difficulty loader available, using mock data")
            return self._build_mock_medium_context()
        
        try:
            # Sample an image with fine-grained categories
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build medium difficulty context
            context['medium_source_dataset'] = loader_name.replace('_', ' ').upper()
            context['medium_image_description'] = self._describe_complex_scene(sample)
            context['medium_num_objects'] = str(random.randint(3, 5))
            context['medium_comparison_type'] = random.choice([
                'relative spatial arrangement',
                'shape complexity comparison',
                'size hierarchy determination'
            ])
            context['medium_spatial_constraint'] = random.choice([
                'within the left half of the image',
                'in the foreground',
                'arranged horizontally'
            ])
            context['medium_property_set'] = ', '.join(random.sample(self.property_types, 3))
            context['medium_expected_operations'] = 'Multiple SEGMENT_OBJECT_AT, GET_PROPERTIES, spatial analysis'
            
        except Exception as e:
            logger.error(f"Error building medium context: {e}")
            return self._build_mock_medium_context()
        
        return context
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build context for hard difficulty level."""
        context = {}
        
        # Try PartImageNet loader for part-level reasoning
        loader_name = 'part_imagenet_train'
        if loader_name not in self.loaders:
            # Try alternative name
            loader_name = 'partimagenet_train'
        
        if loader_name not in self.loaders:
            logger.warning("No hard difficulty loader available, using mock data")
            return self._build_mock_hard_context()
        
        loader = self.loaders[loader_name]
        
        try:
            # Sample an image with part annotations
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build hard difficulty context with part-level reasoning
            context['hard_source_dataset'] = 'PartImageNet'
            context['hard_image_description'] = self._describe_part_level_image(sample)
            context['hard_main_object'] = sample.get('category_name', 'complex object')
            context['hard_num_parts'] = str(random.randint(4, 8))
            context['hard_reasoning_type'] = random.choice([
                'hierarchical part-whole relationships',
                'symmetry analysis across parts',
                'geometric pattern recognition',
                'structural integrity assessment'
            ])
            context['hard_comparison_criteria'] = random.choice([
                'bilateral symmetry of parts',
                'proportional relationships between components',
                'geometric regularity of arrangements',
                'fractal-like self-similarity'
            ])
            context['hard_analysis_depth'] = random.choice([
                'pixel-level precision',
                'sub-part decomposition',
                'multi-scale analysis'
            ])
            context['hard_expected_operations'] = 'Nested SEGMENT_OBJECT_AT, recursive GET_PROPERTIES, complex spatial reasoning'
            
        except Exception as e:
            logger.error(f"Error building hard context: {e}")
            return self._build_mock_hard_context()
        
        return context
    
    def _describe_image(self, sample: Dict) -> str:
        """Generate a natural language description of an image."""
        if 'caption' in sample:
            return sample['caption']
        
        # Generate description from annotations
        if 'annotations' in sample:
            num_objects = len(sample['annotations'])
            categories = list(set(ann.get('category_name', 'object') 
                                for ann in sample['annotations']))
            return f"An image containing {num_objects} objects including {', '.join(categories[:3])}"
        
        return "An image with multiple objects for comparison"
    
    def _describe_complex_scene(self, sample: Dict) -> str:
        """Generate description for medium difficulty scene."""
        base_desc = self._describe_image(sample)
        complexity_additions = [
            "with overlapping objects",
            "featuring varied scales and perspectives",
            "with complex spatial arrangements",
            "containing both foreground and background elements"
        ]
        return f"{base_desc} {random.choice(complexity_additions)}"
    
    def _describe_part_level_image(self, sample: Dict) -> str:
        """Generate description for hard difficulty with part-level details."""
        if 'part_annotations' in sample:
            num_parts = len(sample.get('part_annotations', []))
            return f"A detailed image of a {sample.get('category_name', 'complex object')} with {num_parts} annotated parts showing intricate geometric relationships"
        
        return f"A high-resolution image suitable for part-level geometric analysis"
    
    def _generate_task_description(self, difficulty: str) -> str:
        """Generate a task description based on difficulty."""
        descriptions = {
            'easy': "Compare basic geometric properties between two clearly visible objects",
            'medium': "Analyze spatial relationships and properties among multiple objects with specific constraints",
            'hard': "Perform complex part-level geometric reasoning with hierarchical analysis"
        }
        return descriptions.get(difficulty, descriptions['medium'])
    
    def _get_available_operations(self) -> str:
        """Get the list of available operations for this task type."""
        return """
        - SEGMENT_OBJECT_AT(x, y): Segment object at pixel coordinates
        - GET_PROPERTIES(object_id): Extract geometric properties of segmented object
        - COMPARE(property, object1, object2): Compare specific property between objects
        - SPATIAL_RELATION(object1, object2): Determine spatial relationship
        """
    
    def _get_output_format(self) -> str:
        """Get the expected output format specification."""
        return """
        {
            "trajectory": [
                {
                    "step": int,
                    "thought": "reasoning about what to do",
                    "action": "operation_name",
                    "parameters": {...},
                    "result": "operation output"
                }
            ],
            "final_answer": "conclusion based on geometric analysis",
            "extracted_properties": {
                "object1": {...},
                "object2": {...}
            },
            "comparison_result": "detailed comparison outcome"
        }
        """
    
    def _get_active_datasets(self) -> List[str]:
        """Get list of datasets being used."""
        active = []
        for loader_name in self.loaders.keys():
            if any(loader_name.startswith(prefix) for prefix in ['coco', 'lvis', 'ade20k', 'part']):
                active.append(loader_name)
        return active
    
    # Mock context builders for fallback
    def _build_mock_easy_context(self) -> Dict[str, str]:
        """Build mock context for easy level when loaders unavailable."""
        return {
            'easy_source_dataset': 'COCO 2017 (mock)',
            'easy_image_description': 'An image with a cat and a dog on a sofa',
            'easy_object_A_class': 'cat',
            'easy_object_A_bbox': '[120, 80, 220, 180]',
            'easy_object_B_class': 'dog',
            'easy_object_B_bbox': '[250, 100, 380, 240]',
            'easy_property_to_compare': 'size',
            'easy_expected_operations': 'SEGMENT_OBJECT_AT, GET_PROPERTIES'
        }
    
    def _build_mock_medium_context(self) -> Dict[str, str]:
        """Build mock context for medium level when loaders unavailable."""
        return {
            'medium_source_dataset': 'LVIS (mock)',
            'medium_image_description': 'A kitchen scene with multiple utensils and appliances',
            'medium_num_objects': '4',
            'medium_comparison_type': 'relative spatial arrangement',
            'medium_spatial_constraint': 'on the countertop',
            'medium_property_set': 'size, position, shape',
            'medium_expected_operations': 'Multiple SEGMENT_OBJECT_AT, GET_PROPERTIES'
        }
    
    def _build_mock_hard_context(self) -> Dict[str, str]:
        """Build mock context for hard level when loaders unavailable."""
        return {
            'hard_source_dataset': 'PartImageNet (mock)',
            'hard_image_description': 'A detailed view of a bicycle with visible components',
            'hard_main_object': 'bicycle',
            'hard_num_parts': '6',
            'hard_reasoning_type': 'hierarchical part-whole relationships',
            'hard_comparison_criteria': 'proportional relationships between components',
            'hard_analysis_depth': 'sub-part decomposition',
            'hard_expected_operations': 'Nested SEGMENT_OBJECT_AT, recursive GET_PROPERTIES'
        }