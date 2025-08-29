"""
Select Frame Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for temporal localization
tasks using SELECT_FRAME operations in video sequences.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class SelectFrameTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for frame selection and temporal localization tasks.
    
    This generator creates tasks that require:
    - Finding specific moments in videos
    - Temporal event detection
    - Key frame identification
    - Action boundary detection
    - Temporal grounding of descriptions
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the select frame task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders
        self.difficulty_loaders = {
            'easy': ['activitynet_train', 'kinetics_train'],
            'medium': ['charades_train', 'something_something_train'],
            'hard': ['ego4d_nlq_train', 'dense_video_captioning_train']
        }
        
        # Temporal localization scenarios
        self.temporal_scenarios = [
            'action_start', 'action_peak', 'action_end',
            'state_change', 'object_appearance', 'interaction_moment',
            'critical_event', 'transition_point'
        ]
        
        # Frame selection strategies
        self.selection_strategies = {
            'easy': ['linear_search', 'binary_search'],
            'medium': ['sliding_window', 'attention_based'],
            'hard': ['multi_scale_temporal', 'hierarchical_refinement']
        }
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """Build context placeholders for frame selection tasks."""
        placeholders = {}
        
        # Build difficulty-specific contexts
        placeholders.update(self._build_easy_context())
        placeholders.update(self._build_medium_context())
        placeholders.update(self._build_hard_context())
        
        # General context
        placeholders['task_description'] = "Identify and select specific frames from video sequences based on temporal queries"
        placeholders['available_operations'] = "SELECT_FRAME(criteria), COMPARE_FRAMES(f1, f2), TEMPORAL_SCAN(range)"
        placeholders['output_format'] = self._get_output_format()
        placeholders['source_datasets'] = self._get_active_datasets()
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build easy frame selection context."""
        return {
            'easy_source_dataset': 'ActivityNet',
            'easy_video_description': 'A person cooking in kitchen',
            'easy_target_moment': 'when the person starts cutting vegetables',
            'easy_video_duration': '30 seconds',
            'easy_num_frames': '900',
            'easy_temporal_cue': 'clear action boundary',
            'easy_search_strategy': 'linear scan'
        }
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build medium frame selection context."""
        return {
            'medium_source_dataset': 'Charades',
            'medium_video_complexity': 'multiple overlapping activities',
            'medium_query': 'the moment when person A hands object to person B',
            'medium_temporal_ambiguity': 'moderate',
            'medium_num_candidates': '5-10 possible frames',
            'medium_refinement_needed': 'yes',
            'medium_context_window': '2-second window'
        }
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build hard frame selection context."""
        return {
            'hard_source_dataset': 'Ego4D NLQ',
            'hard_query_type': 'natural language temporal grounding',
            'hard_query': 'the first time I looked at my phone after entering the room',
            'hard_video_length': '5 minutes',
            'hard_temporal_precision': 'sub-second accuracy required',
            'hard_distractors': 'multiple similar events',
            'hard_reasoning_depth': 'causal and contextual understanding needed'
        }
    
    def _get_output_format(self) -> str:
        """Get output format."""
        return """{
            "trajectory": [{"step": int, "action": "SELECT_FRAME", "frame_num": int, "reasoning": "..."}],
            "selected_frames": [frame_numbers],
            "temporal_boundaries": {"start": int, "end": int},
            "confidence_scores": [...],
            "final_answer": "Frame X at timestamp Y"
        }"""
    
    def _get_active_datasets(self) -> List[str]:
        """Get active datasets."""
        return [name for name in self.loaders.keys() if any(
            d in name.lower() for d in ['activitynet', 'kinetics', 'charades', 'ego4d']
        )]