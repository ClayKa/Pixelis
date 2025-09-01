"""
Select Frame Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for temporal localization
tasks using SELECT_FRAME operations in video sequences.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator
from .style_definitions import UNIVERSAL_STYLES, TEMPORAL_STYLES

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
        
        # [NEW] Combine universal and temporal-specific styles
        self.styles = UNIVERSAL_STYLES + TEMPORAL_STYLES
        
        # Define difficulty-specific loaders (from manifest)
        self.difficulty_loaders = {
            'easy': ['starqa_train', 'didemo_train'],
            'medium': ['msrvtt_train', 'activitynet_captions_train'],
            'hard': ['assembly101_train']
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
        """
        Build context placeholders for the temporal localization prompt.
        SIMPLIFIED VERSION: Only provides the required placeholders.
        """
        logger.debug(f"Building context for select frame task")
        
        # Get a loader
        loader_name = None
        loader = None
        for name in ['starqa_train', 'didemo_train', 'msrvtt_train', 'activitynet_captions_train', 'assembly101_train']:
            if name in self.loaders:
                loader = self.loaders[name]
                loader_name = name
                break
        
        # [CRITICAL FIX] Ensure unique sample selection
        max_attempts = 10
        unique_sample_found = False
        
        for attempt in range(max_attempts):
            # Create a unique ID for this potential sample
            video_idx = random.randint(0, 10000)
            unique_id = f"selectframe_{loader_name if loader_name else 'mock'}_{video_idx}"
            
            # Check if already used
            if unique_id not in self.used_source_sample_ids:
                self.used_source_sample_ids.add(unique_id)
                unique_sample_found = True
                logger.debug(f"Found unique select frame sample: {unique_id} (attempt {attempt + 1})")
                break
            else:
                logger.debug(f"Select frame sample {unique_id} already used, trying another...")
        
        if not unique_sample_found:
            logger.warning(f"Could not find unique select frame sample after {max_attempts} attempts")
        
        # [NEW DIVERSITY LOGIC] Randomize temporal scenario and strategy
        scenario = random.choice(self.temporal_scenarios)
        difficulty = random.choice(['easy', 'medium', 'hard'])
        strategy = random.choice(self.selection_strategies[difficulty])
        
        logger.debug(f"Temporal variation: {scenario} using {strategy} strategy")
        
        # Generate a temporal event
        events = [
            ("The person picks up a tool", 32.5, 34.1),
            ("The chef adds salt to the pot", 45.2, 46.8),
            ("The dog jumps over the fence", 12.3, 14.5),
            ("The car passes the traffic light", 8.7, 10.2),
            ("The player scores a goal", 67.8, 69.5),
            ("The bird lands on the branch", 23.4, 24.9)
        ]
        
        event_description, start_time, end_time = random.choice(events)
        
        video_descriptions = [
            "A person is shown assembling a piece of furniture in a workshop.",
            "A chef is preparing a complex dish in a professional kitchen.",
            "A dog is playing in a backyard with various obstacles.",
            "Traffic is flowing through a busy intersection.",
            "A soccer match is in progress with multiple players on the field.",
            "Birds are flying and perching in a garden."
        ]
        
        video_description = random.choice(video_descriptions)
        
        # Adjust event based on scenario
        scenario_events = {
            'action_start': ("The action begins", start_time, start_time + 0.5),
            'action_peak': ("The critical moment occurs", (start_time + end_time) / 2, (start_time + end_time) / 2 + 0.5),
            'action_end': ("The action concludes", end_time - 0.5, end_time),
            'state_change': ("The state transitions", start_time, end_time),
            'object_appearance': ("The object becomes visible", start_time, start_time + 1.0),
            'interaction_moment': ("The interaction happens", (start_time + end_time) / 2, (start_time + end_time) / 2 + 1.0),
            'critical_event': (event_description, start_time, end_time),
            'transition_point': ("The scene transitions", end_time, end_time + 1.5)
        }
        
        if scenario in scenario_events:
            event_description, start_time, end_time = scenario_events[scenario]
        
        overall_goals = [
            "Complete the furniture assembly",
            "Prepare the dish for service",
            "Navigate the obstacle course",
            "Monitor traffic flow",
            "Score during the match",
            "Document bird behavior"
        ]
        
        overall_goal = random.choice(overall_goals)
        
        # [CRITICAL NEW LOGIC] Dynamic Style Forcing  
        chosen_style = random.choice(self.styles)
        logger.debug(f"Selected temporal style: {chosen_style['name']}")
        
        # Build the required placeholders
        placeholders = {
            'source_dataset': loader_name.replace('_', ' ').title() if loader_name else "STARQA",
            'video_description': video_description,
            'ground_truth_event_description': event_description,
            'start_time': start_time,
            'end_time': end_time,
            'overall_goal': overall_goal
        }
        
        logger.debug(f"Constructed placeholders: {list(placeholders.keys())}")
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build easy frame selection context."""
        context = {}
        
        # Try easy dataset loaders from manifest (STARQA, DiDeMo)
        loader = None
        loader_name = None
        for name in ['starqa_train', 'didemo_train']:
            if name in self.loaders:
                loader = self.loaders[name]
                loader_name = name
                break
        
        if not loader:
            logger.warning("Datasources 'starqa_train' and 'didemo_train' not found for Easy difficulty. Check manifest.")
            # Return mock data
            return {
                'easy_source_dataset': 'STARQA (mock)',
                'easy_video_description': 'A person performing an action',
                'easy_target_moment': 'when the action starts',
                'easy_video_duration': '30 seconds',
                'easy_num_frames': '900',
                'easy_temporal_cue': 'clear action boundary',
                'easy_search_strategy': 'linear scan'
            }
        
        # Build context from actual loader
        context['easy_source_dataset'] = loader_name.replace('_', ' ').title()
        context['easy_video_description'] = 'A video with clear temporal events'
        context['easy_target_moment'] = 'a specific action or event'
        context['easy_video_duration'] = '30 seconds'
        context['easy_num_frames'] = '900'
        context['easy_temporal_cue'] = 'clear action boundary'
        context['easy_search_strategy'] = 'linear scan'
        
        return context
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build medium frame selection context."""
        context = {}
        
        # Try medium dataset loaders from manifest (MSRVTT, ActivityNet Captions)
        loader = None
        loader_name = None
        for name in ['msrvtt_train', 'activitynet_captions_train']:
            if name in self.loaders:
                loader = self.loaders[name]
                loader_name = name
                break
        
        if not loader:
            logger.warning("Datasources 'msrvtt_train' and 'activitynet_captions_train' not found for Medium difficulty. Check manifest.")
            # Return mock data
            return {
                'medium_source_dataset': 'MSRVTT (mock)',
                'medium_video_complexity': 'multiple overlapping activities',
                'medium_query': 'complex temporal query',
                'medium_temporal_ambiguity': 'moderate',
                'medium_num_candidates': '5-10 possible frames',
                'medium_refinement_needed': 'yes',
                'medium_context_window': '2-second window'
            }
        
        # Build context from actual loader
        context['medium_source_dataset'] = loader_name.replace('_', ' ').title()
        context['medium_video_complexity'] = 'multiple events and captions'
        context['medium_query'] = 'temporally grounded description'
        context['medium_temporal_ambiguity'] = 'moderate'
        context['medium_num_candidates'] = '5-10 possible frames'
        context['medium_refinement_needed'] = 'yes'
        context['medium_context_window'] = '2-second window'
        
        return context
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build hard frame selection context."""
        context = {}
        
        # Try hard dataset loader from manifest (Assembly101)
        loader = None
        loader_name = 'assembly101_train'
        if loader_name in self.loaders:
            loader = self.loaders[loader_name]
        
        if not loader:
            logger.warning("Datasource 'assembly101_train' not found for Hard difficulty. Check manifest.")
            # Return mock data
            return {
                'hard_source_dataset': 'Assembly101 (mock)',
                'hard_query_type': 'precise action timing',
                'hard_query': 'exact moment of assembly step',
                'hard_video_length': '5 minutes',
                'hard_temporal_precision': 'sub-second accuracy required',
                'hard_distractors': 'repetitive assembly actions',
                'hard_reasoning_depth': 'understanding assembly sequence'
            }
        
        # Build context from actual loader
        context['hard_source_dataset'] = 'Assembly101'
        context['hard_query_type'] = 'fine-grained action timing'
        context['hard_query'] = 'precise assembly step timing'
        context['hard_video_length'] = '5 minutes'
        context['hard_temporal_precision'] = 'sub-second accuracy required'
        context['hard_distractors'] = 'similar assembly actions'
        context['hard_reasoning_depth'] = 'procedural understanding needed'
        
        return context
    
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
            d in name.lower() for d in ['starqa', 'didemo', 'msrvtt', 'activitynet', 'assembly']
        )]