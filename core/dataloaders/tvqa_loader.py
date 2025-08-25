# core/dataloaders/tvqa_loader.py

"""
Data loader for TVQA dataset.

TVQA provides timestamps in seconds rather than frame indices,
demonstrating the timestamp-to-frame conversion functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from .base_loader import BaseLoader
from .frame_aware_loader import FrameAwareLoaderMixin
from .timestamp_utils import TimestampConverter

logger = logging.getLogger(__name__)


class TVQALoader(BaseLoader):
    """
    Loader for TVQA dataset with timestamp-based annotations.
    
    TVQA contains:
    - Video clips from TV shows
    - Questions about the clips
    - Timestamp ranges for relevant segments
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TVQA loader.
        
        Expected config:
        {
            'name': 'tvqa',
            'path': '/path/to/tvqa/videos',
            'annotation_file': '/path/to/tvqa_qa_release/tvqa_train.jsonl'
        }
        """
        self.name = config.get('name', 'tvqa')
        self.video_dir = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Load annotations BEFORE calling super().__init__
        self._annotations = []
        self._load_annotations()
        
        # Now call parent init which will call _build_index
        super().__init__(config)
        
        logger.info(f"Loaded TVQA dataset with {len(self._annotations)} samples")
    
    def _load_annotations(self):
        """Load TVQA annotations from JSONL file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        with open(self.annotation_file, 'r') as f:
            for line in f:
                if line.strip():
                    self._annotations.append(json.loads(line))
    
    def _build_index(self):
        """Build index for the dataset (required by BaseLoader)."""
        # For TVQA, the index is simply the list of annotations
        # Each annotation is already a complete sample
        return list(range(len(self._annotations)))
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._annotations)
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - sample_id: Unique identifier
            - media_path: Path to video file
            - question: Question text
            - answers: List of answer options
            - correct_answer: Index of correct answer
            - timestamps: Start and end time in seconds
            - annotations: Structured annotation data
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} samples")
        
        ann = self._annotations[index]
        
        # Extract video information
        video_name = ann.get('vid_name', '')
        show_name = ann.get('show', '')
        
        # Build video path
        # TVQA structure: show_name/video_name.mp4
        video_path = self.video_dir / show_name / f"{video_name}.mp4"
        
        # Check for alternative extensions if mp4 doesn't exist
        if not video_path.exists():
            for ext in ['.mkv', '.avi', '.mov']:
                alt_path = video_path.with_suffix(ext)
                if alt_path.exists():
                    video_path = alt_path
                    break
        
        # Extract timestamps (TVQA provides these in seconds)
        ts_start = float(ann.get('ts_start', 0))
        ts_end = float(ann.get('ts_end', ts_start + 5))  # Default 5 second clip
        
        # Build structured sample
        sample = {
            'sample_id': f"tvqa_{ann.get('qid', index)}",
            'media_path': str(video_path),
            'dataset': 'tvqa',
            
            # Question and answers
            'question': ann.get('q', ''),
            'answers': [
                ann.get('a0', ''),
                ann.get('a1', ''),
                ann.get('a2', ''),
                ann.get('a3', ''),
                ann.get('a4', '')
            ],
            'correct_answer': ann.get('answer_idx', -1),
            
            # Subtitle information if available
            'subtitles': ann.get('sub', ''),
            
            # Timestamps in seconds
            'timestamps': {
                'start': ts_start,
                'end': ts_end,
                'duration': ts_end - ts_start
            },
            
            # Structured annotations
            'annotations': {
                'question_type': self._classify_question_type(ann.get('q', '')),
                'temporal_segment': {
                    'start_time': ts_start,
                    'end_time': ts_end,
                    'duration': ts_end - ts_start
                },
                'metadata': {
                    'show': show_name,
                    'video_name': video_name,
                    'question_id': ann.get('qid', ''),
                    'located': ann.get('located', True)  # Whether timestamp is provided
                }
            }
        }
        
        return sample
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classify the type of question.
        
        Simple heuristic classification based on question words.
        """
        q_lower = question.lower()
        
        if q_lower.startswith('what'):
            return 'what'
        elif q_lower.startswith('who'):
            return 'who'
        elif q_lower.startswith('where'):
            return 'where'
        elif q_lower.startswith('when'):
            return 'when'
        elif q_lower.startswith('why'):
            return 'why'
        elif q_lower.startswith('how'):
            return 'how'
        elif 'what happens' in q_lower:
            return 'event'
        else:
            return 'other'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self),
            'shows': set(),
            'total_duration': 0,
            'question_types': {},
            'avg_clip_duration': 0
        }
        
        durations = []
        
        for ann in self._annotations:
            # Collect shows
            show = ann.get('show', '')
            if show:
                stats['shows'].add(show)
            
            # Collect durations
            ts_start = float(ann.get('ts_start', 0))
            ts_end = float(ann.get('ts_end', ts_start))
            duration = ts_end - ts_start
            if duration > 0:
                durations.append(duration)
                stats['total_duration'] += duration
            
            # Count question types
            q_type = self._classify_question_type(ann.get('q', ''))
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
        
        stats['num_shows'] = len(stats['shows'])
        stats['shows'] = list(stats['shows'])[:10]  # Keep only first 10 for brevity
        
        if durations:
            stats['avg_clip_duration'] = sum(durations) / len(durations)
            stats['min_duration'] = min(durations)
            stats['max_duration'] = max(durations)
        
        return stats


class FrameAwareTVQALoader(FrameAwareLoaderMixin, TVQALoader):
    """
    TVQA loader with frame extraction capabilities.
    
    This loader demonstrates automatic timestamp-to-frame conversion.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        extract_frames: bool = False,
        sampling_strategy: str = "uniform_8",
        **kwargs
    ):
        """
        Initialize frame-aware TVQA loader.
        
        Args:
            config: Dataset configuration
            extract_frames: Whether to extract actual frames
            sampling_strategy: How to sample frames from segments
            **kwargs: Additional arguments for frame extraction
        """
        # Initialize with timestamp converter
        kwargs['timestamp_converter'] = TimestampConverter(
            default_fps=24.0,  # Common for TV shows
            round_mode="round",
            normalize=True
        )
        
        super().__init__(config, extract_frames=extract_frames, **kwargs)
        self.sampling_strategy = sampling_strategy
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get item with optional frame extraction from timestamps.
        """
        # Get base sample
        sample = TVQALoader.get_item(self, index)
        
        # Extract frames if enabled
        if self.extract_frames and 'timestamps' in sample:
            video_path = Path(sample['media_path'])
            
            # Use the timestamp-based extraction method
            frames = self._extract_frames_for_timestamp(
                video_path,
                sample['timestamps']['start'],
                sample['timestamps']['end'],
                self.sampling_strategy
            )
            
            if frames is not None:
                # Add frame data to sample
                sample = self._add_frame_data_to_sample(
                    sample, frames, self.sampling_strategy
                )
                
                # Also add the calculated frame indices for reference
                video_info = self.get_video_info(video_path)
                if video_info and 'fps' in video_info:
                    fps = video_info['fps']
                    start_frame, end_frame = self._timestamp_converter.convert_range(
                        sample['timestamps']['start'],
                        sample['timestamps']['end'],
                        fps,
                        video_path
                    )
                    sample['frames']['calculated_indices'] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'fps': fps
                    }
        
        return sample


def create_tvqa_loader(
    config: Dict[str, Any],
    with_frames: bool = False,
    **frame_kwargs
) -> TVQALoader:
    """
    Factory function to create TVQA loader with optional frame extraction.
    
    Args:
        config: Dataset configuration
        with_frames: Whether to create frame-aware version
        **frame_kwargs: Arguments for frame extraction
    
    Returns:
        TVQALoader or FrameAwareTVQALoader instance
    """
    if with_frames:
        return FrameAwareTVQALoader(
            config,
            extract_frames=True,
            **frame_kwargs
        )
    else:
        return TVQALoader(config)