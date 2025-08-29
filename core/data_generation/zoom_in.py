"""
Zoom-In Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for detail perception
tasks using ZOOM_IN operations for progressive refinement.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class ZoomInTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for zoom-in based detail perception tasks.
    
    This generator creates tasks that require:
    - Progressive zooming to reveal details
    - Multi-scale visual reasoning
    - Finding small objects in large scenes
    - Reading fine text or symbols
    - Identifying detailed patterns or textures
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the zoom-in task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders
        self.difficulty_loaders = {
            'easy': ['coco2017_train', 'visual_genome_train'],
            'medium': ['open_images_v6_train', 'objects365_train'],
            'hard': ['sa1b_train', 'megadepth_train']
        }
        
        # Zoom scenarios
        self.zoom_scenarios = [
            'find_small_object', 'read_distant_text', 'inspect_texture',
            'identify_pattern', 'count_small_items', 'verify_details',
            'locate_defect', 'examine_component'
        ]
        
        # Zoom strategies
        self.zoom_strategies = {
            'easy': ['single_zoom', 'center_focus'],
            'medium': ['multi_step_zoom', 'quadrant_search'],
            'hard': ['recursive_zoom', 'adaptive_refinement', 'multi_region_analysis']
        }
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """Build context placeholders for zoom-in tasks."""
        placeholders = {}
        
        # Build difficulty-specific contexts
        placeholders.update(self._build_easy_context())
        placeholders.update(self._build_medium_context())
        placeholders.update(self._build_hard_context())
        
        # General context
        placeholders['task_description'] = "Use progressive zoom-in operations to identify and analyze fine details"
        placeholders['available_operations'] = "ZOOM_IN(bbox), ANALYZE_REGION(bbox), ENHANCE_DETAIL(region)"
        placeholders['output_format'] = self._get_output_format()
        placeholders['source_datasets'] = self._get_active_datasets()
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build easy zoom context."""
        return {
            'easy_source_dataset': 'COCO 2017',
            'easy_image_description': 'A room with various objects',
            'easy_target_object': 'a clock on the wall',
            'easy_zoom_levels': '1-2',
            'easy_detail_to_find': 'the time shown',
            'easy_initial_visibility': 'partially visible',
            'easy_expected_strategy': 'single zoom to target'
        }
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build medium zoom context."""
        return {
            'medium_source_dataset': 'Open Images V6',
            'medium_scene_complexity': 'crowded marketplace',
            'medium_search_target': 'specific product labels',
            'medium_zoom_levels': '2-3',
            'medium_search_regions': 'multiple areas',
            'medium_occlusion_level': 'moderate',
            'medium_strategy': 'systematic quadrant search'
        }
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build hard zoom context."""
        return {
            'hard_source_dataset': 'SA-1B',
            'hard_image_resolution': 'ultra-high resolution',
            'hard_analysis_type': 'microscopic detail inspection',
            'hard_zoom_levels': '4-5',
            'hard_pattern_complexity': 'fractal-like structures',
            'hard_multi_scale_reasoning': 'required',
            'hard_adaptive_strategy': 'context-dependent zooming'
        }
    
    def _get_output_format(self) -> str:
        """Get output format."""
        return """{
            "trajectory": [{"step": int, "action": "ZOOM_IN", "bbox": [...], "finding": "..."}],
            "zoom_sequence": [[x1,y1,x2,y2], ...],
            "details_found": {...},
            "final_answer": "..."
        }"""
    
    def _get_active_datasets(self) -> List[str]:
        """Get active datasets."""
        return [name for name in self.loaders.keys() if any(
            d in name.lower() for d in ['coco', 'visual', 'open_images', 'sa1b']
        )]