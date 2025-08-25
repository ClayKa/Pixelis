# core/dataloaders/starqa_loader.py

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class StarqaLoader(BaseLoader):
    """
    A concrete data loader for the STARQA (Situated Reasoning in Real-World Videos) dataset.
    
    This loader parses the main question-answering annotations and is designed to
    optionally integrate supplementary fine-grained annotations (person bboxes, 
    object bboxes and relationships, keyframe IDs, and video segments) to create
    richer, multi-modal sample representations for situated video understanding.
    
    Key features:
    - Loads STARQA QA annotations with temporal bounds and structured questions
    - Optional integration of person pose and bbox annotations
    - Optional object detection and relationship annotations
    - Keyframe and video segment information
    - Supports multiple choice questions with structured programs
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the StarqaLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'path': Path to video directory
                - 'annotation_file': Path to main QA annotation file (e.g., STAR_train.json)
                - 'supplementary_paths': Optional dict with paths to additional annotation files:
                  - 'person_bbox': Path to person bbox pickle file
                  - 'object_bbox_and_relationship': Path to object bbox/relationship pickle file
                  - 'video_keyframe_ids': Path to keyframe IDs CSV
                  - 'video_segments': Path to video segments CSV
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"StarqaLoader config must include '{key}'")
        
        self.videos_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.videos_path.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Handle supplementary paths
        self.supplementary_paths = config.get('supplementary_paths', {})
        
        # Initialize lookup structures for supplementary data
        self._person_bbox_data = {}
        self._object_relationship_data = {}
        self._video_keyframes = defaultdict(list)
        self._video_segments = defaultdict(dict)
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load STARQA annotations and discover supplementary annotation files.
        
        Returns:
            List of QA sample dictionaries
        """
        logger.info(f"Loading STARQA annotations from {self.annotation_file}")
        
        # Load main QA annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            qa_samples = json.load(f)
        
        logger.info(f"Loaded {len(qa_samples)} QA samples")
        
        # Load supplementary annotations if available
        self._load_supplementary_data()
        
        # Filter samples to only include those with existing video files
        existing_samples = []
        for sample in qa_samples:
            video_id = sample.get('video_id', '')
            video_path = self.videos_path / f"{video_id}.mp4"
            
            if video_path.exists():
                existing_samples.append(sample)
        
        logger.info(f"Found {len(existing_samples)} samples with existing video files")
        
        return existing_samples

    def _load_supplementary_data(self):
        """Load optional supplementary annotation data."""
        
        # Load person bbox data
        person_bbox_path = self.supplementary_paths.get('person_bbox')
        if person_bbox_path and Path(person_bbox_path).exists():
            try:
                with open(person_bbox_path, 'rb') as f:
                    self._person_bbox_data = pickle.load(f)
                logger.info(f"Loaded person bbox data for {len(self._person_bbox_data)} keyframes")
            except Exception as e:
                logger.warning(f"Failed to load person bbox data: {e}")
        
        # Load object bbox and relationship data
        object_relationship_path = self.supplementary_paths.get('object_bbox_and_relationship')
        if object_relationship_path and Path(object_relationship_path).exists():
            try:
                with open(object_relationship_path, 'rb') as f:
                    self._object_relationship_data = pickle.load(f)
                logger.info(f"Loaded object/relationship data for {len(self._object_relationship_data)} keyframes")
            except Exception as e:
                logger.warning(f"Failed to load object/relationship data: {e}")
        
        # Load video keyframes (CSV format)
        keyframes_path = self.supplementary_paths.get('video_keyframe_ids')
        if keyframes_path and Path(keyframes_path).exists():
            try:
                self._load_video_keyframes(keyframes_path)
                logger.info(f"Loaded keyframe data for {len(self._video_keyframes)} videos")
            except Exception as e:
                logger.warning(f"Failed to load keyframe data: {e}")
        
        # Load video segments (CSV format)
        segments_path = self.supplementary_paths.get('video_segments')
        if segments_path and Path(segments_path).exists():
            try:
                self._load_video_segments(segments_path)
                logger.info(f"Loaded segment data for {len(self._video_segments)} videos")
            except Exception as e:
                logger.warning(f"Failed to load segment data: {e}")

    def _load_video_keyframes(self, keyframes_path: str):
        """Load video keyframe IDs from CSV."""
        import csv
        with open(keyframes_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get('video_id', '')
                keyframe_id = row.get('keyframe_id', '')
                if video_id and keyframe_id:
                    self._video_keyframes[video_id].append(keyframe_id)

    def _load_video_segments(self, segments_path: str):
        """Load video segment information from CSV."""
        import csv
        with open(segments_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get('video_id', '')
                if video_id:
                    # Store all segment information for this video
                    self._video_segments[video_id] = dict(row)

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single QA sample with optional supplementary annotations.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with STARQA video QA data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get QA sample from pre-built index
        qa_sample = self._index[index]
        
        # Extract key information
        video_id = qa_sample['video_id']
        question_id = qa_sample.get('question_id', f'q_{index}')
        
        # Construct video path
        video_path = self.videos_path / f"{video_id}.mp4"
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=question_id,
            media_path=video_path,
            media_type="video"
        )
        
        # Process core QA annotations
        question = qa_sample.get('question', '')
        answer = qa_sample.get('answer', '')
        start_time = qa_sample.get('start', 0.0)
        end_time = qa_sample.get('end', 0.0)
        
        # Process choices and question programs
        choices = qa_sample.get('choices', [])
        question_program = qa_sample.get('question_program', [])
        situations = qa_sample.get('situations', {})
        
        # Add STARQA-specific annotations
        sample['annotations'].update({
            'starqa_situated_video_qa': {
                'question_id': question_id,
                'video_id': video_id,
                'question': question,
                'answer': answer,
                'temporal_bounds': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time if end_time > start_time else 0.0
                },
                'question_structure': {
                    'question_program': question_program,
                    'program_length': len(question_program),
                    'functions_used': [step.get('function') for step in question_program if 'function' in step]
                },
                'choices': choices,
                'num_choices': len(choices),
                'situations': situations,
                'num_situations': len(situations)
            },
            'video_metadata': {
                'video_filename': f"{video_id}.mp4",
                'duration_seconds': end_time - start_time if end_time > start_time else 0.0,
                'temporal_bounds': [start_time, end_time]
            },
            'dataset_info': {
                'task_type': 'situated_video_qa',
                'source': 'STARQA',
                'suitable_for_select_frame': True,
                'suitable_for_temporal_reasoning': True,
                'suitable_for_spatial_reasoning': True,
                'has_structured_programs': len(question_program) > 0,
                'has_multiple_choice': len(choices) > 0,
                'has_situation_grounding': len(situations) > 0,
                'video_format': 'mp4',
                'supplementary_data_available': self._get_available_supplementary_data(video_id)
            }
        })
        
        # Add supplementary annotations if available
        supplementary_data = self._get_supplementary_annotations(video_id)
        if supplementary_data:
            sample['annotations']['starqa_situated_video_qa']['supplementary_annotations'] = supplementary_data
        
        return sample

    def _get_available_supplementary_data(self, video_id: str) -> Dict[str, bool]:
        """Check what supplementary data is available for a video."""
        return {
            'person_bboxes': any(f"{video_id}.mp4/" in key for key in self._person_bbox_data.keys()),
            'object_relationships': any(f"{video_id}.mp4/" in key for key in self._object_relationship_data.keys()),
            'keyframes': video_id in self._video_keyframes,
            'segments': video_id in self._video_segments
        }

    def _get_supplementary_annotations(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get supplementary annotations for a specific video."""
        supplementary = {}
        
        # Get person bbox data for this video
        person_data = self._get_person_annotations(video_id)
        if person_data:
            supplementary['person_annotations'] = person_data
        
        # Get object and relationship data for this video
        object_data = self._get_object_relationship_annotations(video_id)
        if object_data:
            supplementary['object_relationship_annotations'] = object_data
        
        # Get keyframes for this video
        if video_id in self._video_keyframes:
            supplementary['keyframes'] = self._video_keyframes[video_id]
        
        # Get segment information for this video
        if video_id in self._video_segments:
            supplementary['segment_info'] = self._video_segments[video_id]
        
        return supplementary if supplementary else None

    def _get_person_annotations(self, video_id: str) -> List[Dict[str, Any]]:
        """Extract person annotations for a specific video."""
        person_annotations = []
        
        for key, data in self._person_bbox_data.items():
            if key.startswith(f"{video_id}.mp4/"):
                frame_id = key.split('/')[-1]
                
                annotation = {
                    'frame_id': frame_id,
                    'bboxes': data.get('bbox', []).tolist() if hasattr(data.get('bbox', []), 'tolist') else data.get('bbox', []),
                    'bbox_scores': data.get('bbox_score', []).tolist() if hasattr(data.get('bbox_score', []), 'tolist') else data.get('bbox_score', []),
                    'bbox_size': data.get('bbox_size', (0, 0)),
                    'bbox_mode': data.get('bbox_mode', 'xyxy'),
                    'has_keypoints': 'keypoints' in data
                }
                
                if 'keypoints' in data and data['keypoints'] is not None:
                    keypoints = data['keypoints']
                    if hasattr(keypoints, 'tolist'):
                        annotation['keypoints'] = keypoints.tolist()
                    else:
                        annotation['keypoints'] = keypoints
                
                person_annotations.append(annotation)
        
        return person_annotations

    def _get_object_relationship_annotations(self, video_id: str) -> List[Dict[str, Any]]:
        """Extract object and relationship annotations for a specific video."""
        object_annotations = []
        
        for key, data_list in self._object_relationship_data.items():
            if key.startswith(f"{video_id}.mp4/"):
                frame_id = key.split('/')[-1]
                
                frame_annotation = {
                    'frame_id': frame_id,
                    'objects': []
                }
                
                if isinstance(data_list, list):
                    for obj_data in data_list:
                        if isinstance(obj_data, dict):
                            obj_annotation = {
                                'class': obj_data.get('class', 'unknown'),
                                'bbox': obj_data.get('bbox'),
                                'attention_relationship': obj_data.get('attention_relationship'),
                                'spatial_relationship': obj_data.get('spatial_relationship'),
                                'contacting_relationship': obj_data.get('contacting_relationship'),
                                'metadata': obj_data.get('metadata', {}),
                                'visible': obj_data.get('visible', True)
                            }
                            frame_annotation['objects'].append(obj_annotation)
                
                object_annotations.append(frame_annotation)
        
        return object_annotations

    def get_samples_by_question_type(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get samples that use a specific function in their question program.
        
        Args:
            function_name: Name of the function (e.g., 'Filter_Actions_with_Verb')
            
        Returns:
            List of samples using the specified function
        """
        matching_samples = []
        
        for i, qa_sample in enumerate(self._index):
            question_program = qa_sample.get('question_program', [])
            functions_used = [step.get('function') for step in question_program if 'function' in step]
            
            if function_name in functions_used:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_by_duration(self, min_duration: float = 0.0, max_duration: float = float('inf')) -> List[Dict[str, Any]]:
        """
        Get samples filtered by temporal duration.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            List of samples within the duration range
        """
        matching_samples = []
        
        for i, qa_sample in enumerate(self._index):
            start_time = qa_sample.get('start', 0.0)
            end_time = qa_sample.get('end', 0.0)
            duration = end_time - start_time if end_time > start_time else 0.0
            
            if min_duration <= duration <= max_duration:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_question_type_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about question types and program structures.
        
        Returns:
            Dictionary with question type analysis
        """
        function_counts = defaultdict(int)
        program_lengths = []
        choice_counts = []
        situation_counts = []
        
        for qa_sample in self._index:
            # Analyze question programs
            question_program = qa_sample.get('question_program', [])
            program_lengths.append(len(question_program))
            
            for step in question_program:
                function_name = step.get('function', 'unknown')
                function_counts[function_name] += 1
            
            # Analyze choices
            choices = qa_sample.get('choices', [])
            choice_counts.append(len(choices))
            
            # Analyze situations
            situations = qa_sample.get('situations', {})
            situation_counts.append(len(situations))
        
        return {
            'total_samples': len(self._index),
            'function_usage': dict(function_counts),
            'most_common_functions': sorted(function_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'program_length_statistics': {
                'avg_length': sum(program_lengths) / len(program_lengths) if program_lengths else 0,
                'min_length': min(program_lengths) if program_lengths else 0,
                'max_length': max(program_lengths) if program_lengths else 0,
                'median_length': sorted(program_lengths)[len(program_lengths) // 2] if program_lengths else 0
            },
            'choice_statistics': {
                'avg_choices': sum(choice_counts) / len(choice_counts) if choice_counts else 0,
                'min_choices': min(choice_counts) if choice_counts else 0,
                'max_choices': max(choice_counts) if choice_counts else 0
            },
            'situation_statistics': {
                'avg_situations': sum(situation_counts) / len(situation_counts) if situation_counts else 0,
                'samples_with_situations': sum(1 for count in situation_counts if count > 0),
                'max_situations': max(situation_counts) if situation_counts else 0
            }
        }

    def get_supplementary_data_coverage(self) -> Dict[str, Any]:
        """
        Get statistics about supplementary data coverage.
        
        Returns:
            Dictionary with coverage statistics
        """
        person_coverage = 0
        object_coverage = 0
        keyframe_coverage = 0
        segment_coverage = 0
        
        unique_videos = set(qa_sample['video_id'] for qa_sample in self._index)
        
        for video_id in unique_videos:
            available_data = self._get_available_supplementary_data(video_id)
            
            if available_data['person_bboxes']:
                person_coverage += 1
            if available_data['object_relationships']:
                object_coverage += 1
            if available_data['keyframes']:
                keyframe_coverage += 1
            if available_data['segments']:
                segment_coverage += 1
        
        total_videos = len(unique_videos)
        
        return {
            'total_unique_videos': total_videos,
            'person_bbox_coverage': {
                'videos_covered': person_coverage,
                'coverage_percentage': (person_coverage / total_videos * 100) if total_videos > 0 else 0,
                'total_keyframes': len(self._person_bbox_data)
            },
            'object_relationship_coverage': {
                'videos_covered': object_coverage,
                'coverage_percentage': (object_coverage / total_videos * 100) if total_videos > 0 else 0,
                'total_keyframes': len(self._object_relationship_data)
            },
            'keyframe_coverage': {
                'videos_covered': keyframe_coverage,
                'coverage_percentage': (keyframe_coverage / total_videos * 100) if total_videos > 0 else 0
            },
            'segment_coverage': {
                'videos_covered': segment_coverage,
                'coverage_percentage': (segment_coverage / total_videos * 100) if total_videos > 0 else 0
            }
        }

    def get_temporal_distribution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about temporal properties of the QA samples.
        
        Returns:
            Dictionary with temporal statistics
        """
        durations = []
        start_times = []
        end_times = []
        
        for qa_sample in self._index:
            start_time = qa_sample.get('start', 0.0)
            end_time = qa_sample.get('end', 0.0)
            
            if end_time > start_time:
                duration = end_time - start_time
                durations.append(duration)
                start_times.append(start_time)
                end_times.append(end_time)
        
        return {
            'total_samples': len(self._index),
            'samples_with_temporal_bounds': len(durations),
            'duration_statistics': {
                'avg_duration': sum(durations) / len(durations) if durations else 0.0,
                'min_duration': min(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0.0,
                'median_duration': sorted(durations)[len(durations) // 2] if durations else 0.0
            },
            'start_time_statistics': {
                'avg_start': sum(start_times) / len(start_times) if start_times else 0.0,
                'min_start': min(start_times) if start_times else 0.0,
                'max_start': max(start_times) if start_times else 0.0
            },
            'end_time_statistics': {
                'avg_end': sum(end_times) / len(end_times) if end_times else 0.0,
                'min_end': min(end_times) if end_times else 0.0,
                'max_end': max(end_times) if end_times else 0.0
            }
        }