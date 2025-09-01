"""
Spatio-Temporal Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for video understanding
and object tracking tasks using TRACK_OBJECT and temporal analysis operations.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator
from .style_definitions import UNIVERSAL_STYLES, TRACKING_STYLES

logger = logging.getLogger(__name__)


class SpatioTemporalTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for spatio-temporal reasoning and object tracking in videos.
    
    This generator creates tasks that require:
    - Tracking objects across video frames
    - Understanding temporal dynamics and motion patterns
    - Analyzing spatial transformations over time
    - Detecting and reasoning about events and activities
    - Multi-object tracking and interaction analysis
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the spatio-temporal task generator."""
        super().__init__(loaders, config, global_config)
        
        # [NEW] Combine universal and tracking-specific styles
        self.styles = UNIVERSAL_STYLES + TRACKING_STYLES
        
        # Define difficulty-specific loaders mapping (from manifest)
        self.difficulty_loaders = {
            'easy': ['mot20_train', 'uvo_dense_train'],
            'medium': ['epic_kitchens_visor_train'],
            'hard': ['vis2022_train']
        }
        
        # Tracking scenarios
        self.tracking_scenarios = [
            'single_object_tracking',
            'multi_object_tracking',
            'occlusion_handling',
            'identity_preservation',
            'trajectory_prediction',
            'interaction_analysis',
            'event_detection',
            'activity_recognition'
        ]
        
        # Motion patterns
        self.motion_patterns = [
            'linear_motion', 'circular_motion', 'zigzag_pattern',
            'acceleration', 'deceleration', 'sudden_stops',
            'direction_changes', 'group_movement', 'formation_changes'
        ]
        
        # Temporal reasoning types
        self.temporal_reasoning = {
            'easy': ['object_presence', 'simple_counting', 'basic_motion'],
            'medium': ['speed_estimation', 'trajectory_analysis', 'event_ordering'],
            'hard': ['complex_interactions', 'causal_reasoning', 'prediction', 'anomaly_detection']
        }
    
    def _build_context_placeholders(self) -> Dict[str, str]:
        """
        Build context placeholders for the spatio-temporal reasoning prompt.
        
        Returns:
            Dictionary mapping placeholder names to their values
        """
        import json
        
        # [CRITICAL FIX] Ensure unique sample selection
        source_datasets = ['MOT20', 'UVO Dense', 'EPIC-KITCHENS VISOR', 'VIS2022']
        source_dataset = random.choice(source_datasets)
        
        max_attempts = 10
        unique_sample_found = False
        
        for attempt in range(max_attempts):
            # Create a unique ID for this potential sample
            video_idx = random.randint(0, 10000)
            unique_id = f"spatiotemporal_{source_dataset}_{video_idx}"
            
            # Check if already used
            if unique_id not in self.used_source_sample_ids:
                self.used_source_sample_ids.add(unique_id)
                unique_sample_found = True
                logger.debug(f"Found unique spatio-temporal sample: {unique_id} (attempt {attempt + 1})")
                break
            else:
                logger.debug(f"Spatio-temporal sample {unique_id} already used, trying another...")
        
        if not unique_sample_found:
            logger.warning(f"Could not find unique spatio-temporal sample after {max_attempts} attempts")
        
        # [NEW DIVERSITY LOGIC] Randomize tracking scenario and motion pattern
        scenario = random.choice(self.tracking_scenarios)
        motion_pattern = random.choice(self.motion_patterns)
        difficulty = random.choice(['easy', 'medium', 'hard'])
        reasoning_type = random.choice(self.temporal_reasoning[difficulty])
        
        logger.debug(f"Spatio-temporal variation: {scenario} with {motion_pattern} ({reasoning_type})")
        
        # Generate video description based on scenario
        scenario_descriptions = {
            'single_object_tracking': [
                "A single vehicle moving through traffic",
                "A person walking through a crowded area",
                "An animal moving across the field"
            ],
            'multi_object_tracking': [
                "Multiple pedestrians crossing an intersection",
                "Several players on a sports field",
                "A group of vehicles in a parking lot"
            ],
            'occlusion_handling': [
                "Objects passing behind obstacles",
                "People walking through doorways",
                "Vehicles passing under bridges"
            ],
            'interaction_analysis': [
                "Two people having a conversation",
                "Players passing a ball",
                "Vehicles merging in traffic"
            ]
        }
        
        video_description = random.choice(scenario_descriptions.get(
            scenario, 
            ["A complex scene with multiple moving objects"]
        ))
        
        # Generate task goal
        task_goals = [
            "Track the person in red from start to finish",
            "Count how many times the objects cross paths",
            "Determine when Object A enters the defined zone",
            "Analyze the interaction between the two main subjects",
            "Monitor the movement pattern of the tracked vehicle"
        ]
        task_goal = random.choice(task_goals)
        
        # Generate object details
        object_A_details = {
            "name": random.choice(["person in red shirt", "blue car", "cyclist with helmet"]),
            "initial_mask": f"mask_for_object_{random.randint(1, 100)}"
        }
        
        object_B_details = {
            "name": random.choice(["person in blue shirt", "white van", "pedestrian with bag"]),
            "initial_mask": f"mask_for_object_{random.randint(101, 200)}"
        }
        
        # Generate spatial zone
        spatial_zone = {
            "zone_type": random.choice(["bounding_box", "polygon", "circle"]),
            "coordinates": f"[{random.randint(100, 300)}, {random.randint(100, 300)}, {random.randint(400, 600)}, {random.randint(400, 600)}]"
        }
        
        # Generate ground truth conclusion based on reasoning type
        reasoning_conclusions = {
            'object_presence': "Object A was present in frames 10-45 and 67-89",
            'simple_counting': "Object B appeared 4 times in the video",
            'basic_motion': "The tracked object moved from left to right",
            'speed_estimation': "Object A maintained an average speed of 2.5 m/s",
            'trajectory_analysis': "The object followed a curved path with 3 direction changes",
            'event_ordering': "Event A occurred at frame 23, followed by Event B at frame 45",
            'complex_interactions': "Objects A and B engaged in 3 distinct interaction phases",
            'causal_reasoning': "Object A's movement triggered Object B's response at frame 34",
            'prediction': "Based on trajectory, Object A will reach the zone in 2.3 seconds",
            'anomaly_detection': "Unusual behavior detected at frames 56-61"
        }
        
        ground_truth_conclusion = reasoning_conclusions.get(
            reasoning_type,
            "The tracked object completed its trajectory successfully"
        )
        
        # [CRITICAL NEW LOGIC] Dynamic Style Forcing
        chosen_style = random.choice(self.styles)
        logger.debug(f"Selected spatio-temporal style: {chosen_style['name']}")
        
        placeholders = {
            'source_dataset': source_dataset,
            'video_description': video_description,
            'task_goal': task_goal,
            'ground_truth_conclusion': ground_truth_conclusion,
            'object_A_details_json': json.dumps(object_A_details),
            'object_B_details_json': json.dumps(object_B_details),
            'spatial_zone_json': json.dumps(spatial_zone)
        }
        
        return placeholders
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build context for easy tracking tasks."""
        context = {}
        
        # Try easy dataset loaders from manifest (MOT20, UVO)
        loader = None
        for loader_name in ['mot20_train', 'uvo_dense_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            logger.warning("Datasources 'mot20_train' and 'uvo_dense_train' not found for Easy difficulty. Check manifest.")
            return self._build_mock_easy_context()
        
        try:
            # Sample a video sequence
            sample = loader.get_item(random.randint(0, min(100, len(loader) - 1)))
            
            # Build easy tracking context
            context['easy_source_dataset'] = loader_name.replace('_', ' ').title()
            context['easy_video_description'] = self._describe_video_scene('easy', sample)
            context['easy_num_frames'] = str(random.randint(30, 60))
            context['easy_fps'] = str(random.choice([15, 24, 30]))
            context['easy_num_objects'] = str(random.randint(1, 3))
            context['easy_object_types'] = random.choice([
                'pedestrians', 'vehicles', 'cyclists', 'animals'
            ])
            context['easy_tracking_task'] = random.choice([
                'track a single person walking',
                'follow a vehicle through the scene',
                'monitor object entrance and exit',
                'count objects passing a region'
            ])
            context['easy_motion_complexity'] = 'simple linear motion'
            context['easy_occlusion_level'] = 'minimal occlusions'
            context['easy_expected_challenges'] = 'scale changes, illumination variations'
            context['easy_analysis_requirements'] = random.choice([
                'determine object trajectory',
                'measure object presence duration',
                'identify motion direction',
                'detect stops and starts'
            ])
            
        except Exception as e:
            logger.error(f"Error building easy spatio-temporal context: {e}")
            return self._build_mock_easy_context()
        
        return context
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build context for medium difficulty tracking tasks."""
        context = {}
        
        # Try medium dataset loaders from manifest (EPIC-KITCHENS)
        loader = None
        for loader_name in ['epic_kitchens_visor_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            logger.warning("Datasource 'epic_kitchens_visor_train' not found for Medium difficulty. Check manifest.")
            return self._build_mock_medium_context()
        
        try:
            sample = loader.get_item(random.randint(0, min(100, len(loader) - 1)))
            
            # Build medium tracking context
            context['medium_source_dataset'] = loader_name.replace('_', ' ').upper()
            context['medium_video_type'] = random.choice([
                'first-person activity video',
                'surveillance footage',
                'sports recording',
                'traffic monitoring',
                'indoor scene video'
            ])
            context['medium_duration_seconds'] = str(random.randint(5, 15))
            context['medium_num_objects'] = str(random.randint(4, 8))
            context['medium_tracking_complexity'] = random.choice([
                'multiple overlapping trajectories',
                'frequent occlusions and reappearances',
                'similar-looking objects',
                'varying object scales',
                'camera motion compensation needed'
            ])
            context['medium_interaction_types'] = random.choice([
                'object handoffs',
                'group formations and splits',
                'collision avoidance',
                'following behaviors',
                'synchronized movements'
            ])
            context['medium_temporal_analysis'] = random.choice([
                'identify temporal patterns',
                'detect recurring events',
                'analyze speed variations',
                'track state changes',
                'measure interaction durations'
            ])
            context['medium_spatial_constraints'] = random.choice([
                'track within regions of interest',
                'monitor boundary crossings',
                'maintain relative positions',
                'analyze spatial distributions'
            ])
            context['medium_output_requirements'] = 'frame-by-frame tracking with confidence scores'
            
        except Exception as e:
            logger.error(f"Error building medium spatio-temporal context: {e}")
            return self._build_mock_medium_context()
        
        return context
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build context for hard tracking tasks."""
        context = {}
        
        # Try hard dataset loaders from manifest (VIS2022)
        loader = None
        for loader_name in ['vis2022_train']:
            if loader_name in self.loaders:
                loader = self.loaders[loader_name]
                break
        
        if not loader:
            logger.warning("Datasource 'vis2022_train' not found for Hard difficulty. Check manifest.")
            return self._build_mock_hard_context()
        
        try:
            sample = loader.get_item(random.randint(0, min(100, len(loader) - 1)))
            
            # Build hard tracking context
            context['hard_source_dataset'] = loader_name.replace('_', ' ').title()
            context['hard_scenario_complexity'] = random.choice([
                'dense urban traffic with 50+ objects',
                'nighttime/adverse weather conditions',
                'highly dynamic sports scene',
                'crowded pedestrian environment',
                'multi-camera synchronized tracking',
                'long-term tracking with identity switches'
            ])
            context['hard_video_duration'] = str(random.randint(20, 60))
            context['hard_num_objects'] = str(random.randint(15, 50))
            context['hard_tracking_challenges'] = ', '.join(random.sample([
                'severe occlusions',
                'dramatic scale changes',
                'fast motion blur',
                'similar appearances',
                'crossing trajectories',
                'entering/leaving frame',
                'deformable objects',
                'reflection/shadow confusion'
            ], 4))
            context['hard_advanced_analysis'] = random.choice([
                'predict future trajectories',
                'detect anomalous behaviors',
                'analyze group dynamics',
                'identify causal relationships',
                'recognize complex activities',
                'track through multiple cameras'
            ])
            context['hard_temporal_reasoning'] = random.choice([
                'long-term trajectory prediction with uncertainty',
                'multi-scale temporal pattern recognition',
                'event causality chain analysis',
                'abnormal behavior detection',
                'anticipatory motion planning'
            ])
            context['hard_spatial_analysis'] = random.choice([
                '3D trajectory reconstruction',
                'multi-view geometry estimation',
                'occlusion reasoning and recovery',
                'spatial interaction graphs',
                'formation analysis and clustering'
            ])
            context['hard_performance_metrics'] = 'MOTA, MOTP, IDF1, track fragmentation, identity switches'
            context['hard_output_complexity'] = random.choice([
                'hierarchical tracking with sub-parts',
                'probabilistic trajectories with uncertainty',
                'interaction graphs with temporal edges',
                'multi-hypothesis tracking results'
            ])
            
        except Exception as e:
            logger.error(f"Error building hard spatio-temporal context: {e}")
            return self._build_mock_hard_context()
        
        return context
    
    def _describe_video_scene(self, difficulty: str, sample: Dict) -> str:
        """Generate description of video scene based on difficulty."""
        descriptions = {
            'easy': [
                'A clear outdoor scene with good lighting',
                'An indoor corridor with steady camera',
                'A parking lot with sparse traffic',
                'A sidewalk with pedestrians walking'
            ],
            'medium': [
                'A busy intersection with multiple object types',
                'A sports field with player interactions',
                'A shopping mall with crowds',
                'A construction site with machinery and workers'
            ],
            'hard': [
                'A complex urban environment at rush hour',
                'A crowded stadium during an event',
                'A multi-level parking garage with poor lighting',
                'An aerial view of dense traffic patterns'
            ]
        }
        
        base_desc = random.choice(descriptions.get(difficulty, descriptions['medium']))
        
        if 'scene_description' in sample:
            return f"{base_desc}. {sample['scene_description']}"
        
        return base_desc
    
    def _generate_task_description(self) -> str:
        """Generate overall task description."""
        return """Perform spatio-temporal analysis on video sequences using object tracking 
        and temporal reasoning operations. The task requires identifying objects, tracking them 
        across frames, analyzing their motion patterns, and understanding temporal relationships 
        and spatial interactions."""
    
    def _get_available_operations(self) -> str:
        """Get available spatio-temporal operations."""
        return """
        - TRACK_OBJECT(object_id, start_frame, end_frame): Track object across frames
        - DETECT_OBJECTS(frame_num): Detect all objects in a specific frame
        - ANALYZE_TRAJECTORY(track_id): Analyze motion pattern of tracked object
        - COMPUTE_VELOCITY(track_id, frame_range): Calculate object velocity
        - DETECT_INTERACTION(obj1_id, obj2_id, frame_range): Detect object interactions
        - PREDICT_TRAJECTORY(track_id, future_frames): Predict future motion
        - TEMPORAL_SEGMENTATION(): Segment video into temporal events
        """
    
    def _get_output_format(self) -> str:
        """Get expected output format."""
        return """
        {
            "trajectory": [
                {
                    "step": int,
                    "thought": "reasoning about tracking strategy",
                    "action": "TRACK_OBJECT or other operation",
                    "parameters": {"object_id": ..., "frames": ...},
                    "result": "tracking results"
                }
            ],
            "tracks": {
                "object_1": {
                    "frames": [frame_ids],
                    "bboxes": [[x,y,w,h], ...],
                    "confidence": [scores],
                    "trajectory": "motion description"
                }
            },
            "temporal_analysis": {
                "events": [...],
                "interactions": [...],
                "patterns": [...]
            },
            "final_answer": "comprehensive spatio-temporal analysis"
        }
        """
    
    def _get_tracking_requirements(self) -> str:
        """Get tracking-specific requirements."""
        return """
        - Maintain consistent object identities across frames
        - Handle occlusions and reappearances gracefully
        - Provide confidence scores for all tracks
        - Detect and report track fragmentations
        - Include motion statistics (speed, direction, acceleration)
        - Identify key events and state changes
        """
    
    def _get_active_datasets(self) -> List[str]:
        """Get list of active video/tracking datasets."""
        active = []
        video_datasets = [
            'mot', 'ego4d', 'tao', 'bdd100k', 'waymo', 'youtube_vis',
            'nuscenes', 'kitti', 'davis', 'virat'
        ]
        for loader_name in self.loaders.keys():
            if any(dataset in loader_name.lower() for dataset in video_datasets):
                active.append(loader_name)
        return active
    
    # Mock context builders
    def _build_mock_easy_context(self) -> Dict[str, str]:
        """Build mock easy tracking context."""
        return {
            'easy_source_dataset': 'MOT17 (mock)',
            'easy_video_description': 'A clear outdoor scene with good lighting',
            'easy_num_frames': '45',
            'easy_fps': '30',
            'easy_num_objects': '2',
            'easy_object_types': 'pedestrians',
            'easy_tracking_task': 'track a single person walking',
            'easy_motion_complexity': 'simple linear motion',
            'easy_occlusion_level': 'minimal occlusions',
            'easy_expected_challenges': 'scale changes, illumination variations',
            'easy_analysis_requirements': 'determine object trajectory'
        }
    
    def _build_mock_medium_context(self) -> Dict[str, str]:
        """Build mock medium tracking context."""
        return {
            'medium_source_dataset': 'EGO4D (mock)',
            'medium_video_type': 'first-person activity video',
            'medium_duration_seconds': '10',
            'medium_num_objects': '6',
            'medium_tracking_complexity': 'multiple overlapping trajectories',
            'medium_interaction_types': 'object handoffs',
            'medium_temporal_analysis': 'identify temporal patterns',
            'medium_spatial_constraints': 'track within regions of interest',
            'medium_output_requirements': 'frame-by-frame tracking with confidence scores'
        }
    
    def _build_mock_hard_context(self) -> Dict[str, str]:
        """Build mock hard tracking context."""
        return {
            'hard_source_dataset': 'BDD100K (mock)',
            'hard_scenario_complexity': 'dense urban traffic with 50+ objects',
            'hard_video_duration': '30',
            'hard_num_objects': '25',
            'hard_tracking_challenges': 'severe occlusions, fast motion blur, crossing trajectories, similar appearances',
            'hard_advanced_analysis': 'predict future trajectories',
            'hard_temporal_reasoning': 'long-term trajectory prediction with uncertainty',
            'hard_spatial_analysis': '3D trajectory reconstruction',
            'hard_performance_metrics': 'MOTA, MOTP, IDF1, track fragmentation',
            'hard_output_complexity': 'probabilistic trajectories with uncertainty'
        }