"""
Detail Perception Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for fine-grained
visual perception tasks requiring attention to subtle details.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class DetailPerceptionTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for fine-grained detail perception tasks.
    
    This generator creates tasks that require:
    - Identifying subtle visual differences
    - Detecting fine-grained attributes
    - Recognizing small visual anomalies
    - Comparing minute details between objects
    - Understanding texture and material properties
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the detail perception task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders
        self.difficulty_loaders = {
            'easy': ['stanford_cars_train', 'cub200_train'],
            'medium': ['inat2021_train', 'met_artwork_train'],
            'hard': ['funginet_train', 'plant_pathology_train']
        }
        
        # Detail perception scenarios
        self.perception_scenarios = [
            'fine_grained_classification', 'defect_detection', 'quality_assessment',
            'material_recognition', 'texture_analysis', 'pattern_matching',
            'subtle_difference_detection', 'attribute_verification'
        ]
        
        # Detail types by difficulty
        self.detail_types = {
            'easy': ['color_variations', 'shape_differences', 'size_comparisons'],
            'medium': ['texture_patterns', 'surface_properties', 'structural_features'],
            'hard': ['microscopic_features', 'material_composition', 'wear_patterns']
        }
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """Build context placeholders for detail perception tasks."""
        placeholders = {}
        
        # Build difficulty-specific contexts
        placeholders.update(self._build_easy_context())
        placeholders.update(self._build_medium_context())
        placeholders.update(self._build_hard_context())
        
        # General context
        placeholders['task_description'] = "Analyze and identify fine-grained visual details and subtle differences"
        placeholders['available_operations'] = "EXAMINE_DETAIL(region), COMPARE_ATTRIBUTES(obj1, obj2), DETECT_ANOMALY(area)"
        placeholders['output_format'] = self._get_output_format()
        placeholders['source_datasets'] = self._get_active_datasets()
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build easy detail perception context."""
        return {
            'easy_source_dataset': 'Stanford Cars',
            'easy_object_category': 'vehicle models',
            'easy_detail_focus': 'distinguishing features between car models',
            'easy_attributes': 'grille design, headlight shape, badge placement',
            'easy_comparison_type': 'model identification',
            'easy_visual_cues': 'prominent design elements',
            'easy_difficulty': 'clear distinguishing features'
        }
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build medium detail perception context."""
        return {
            'medium_source_dataset': 'iNaturalist 2021',
            'medium_domain': 'species identification',
            'medium_detail_level': 'subspecies variations',
            'medium_key_features': 'wing patterns, markings, coloration',
            'medium_challenge': 'subtle morphological differences',
            'medium_context_importance': 'habitat and behavior cues',
            'medium_expertise_level': 'naturalist knowledge required'
        }
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build hard detail perception context."""
        return {
            'hard_source_dataset': 'Plant Pathology',
            'hard_analysis_type': 'disease detection and classification',
            'hard_microscopic_details': 'cellular-level features',
            'hard_pattern_complexity': 'complex symptom patterns',
            'hard_temporal_aspect': 'disease progression stages',
            'hard_differential_diagnosis': 'multiple possible conditions',
            'hard_expertise_requirement': 'expert-level domain knowledge'
        }
    
    def _get_output_format(self) -> str:
        """Get output format."""
        return """{
            "trajectory": [{"step": int, "action": "EXAMINE_DETAIL", "region": [...], "observation": "..."}],
            "detected_features": {"feature_name": "description", ...},
            "confidence_per_feature": {"feature": float, ...},
            "diagnostic_reasoning": "step-by-step analysis",
            "final_answer": "detailed conclusion"
        }"""
    
    def _get_active_datasets(self) -> List[str]:
        """Get active datasets."""
        return [name for name in self.loaders.keys() if any(
            d in name.lower() for d in ['stanford', 'cub', 'inat', 'fungi', 'plant']
        )]