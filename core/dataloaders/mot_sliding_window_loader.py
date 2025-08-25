# core/dataloaders/mot_sliding_window_loader.py

"""
Enhanced MOT loader with sliding window sampling strategy.

This loader creates multiple overlapping clips from long MOT sequences,
dramatically increasing the number of training samples available.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MotSlidingWindowLoader(BaseLoader):
    """
    MOT loader with configurable sliding window sampling.
    
    Instead of treating each long video as a single sample, this loader
    creates multiple shorter clips using a sliding window approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MOT loader with sliding window configuration.
        
        Expected config:
        {
            'name': 'mot20_train',
            'path': '/path/to/MOT20/train/',
            'sampling_strategy': {
                'type': 'sliding_window',
                'clip_duration_frames': 300,  # 10 seconds at 30fps
                'stride_frames': 150,         # 50% overlap
                'min_clip_frames': 100        # Minimum frames for valid clip
            }
        }
        """
        self.name = config.get('name', 'mot')
        self.base_path = Path(config['path'])
        
        # Validate path
        if not self.base_path.exists():
            raise FileNotFoundError(f"MOT dataset path not found: {self.base_path}")
        
        # Parse sampling strategy
        self.sampling_config = config.get('sampling_strategy', {})
        self.sampling_type = self.sampling_config.get('type', 'full_sequence')
        
        # Default values for sliding window
        self.clip_duration = self.sampling_config.get('clip_duration_frames', 300)
        self.stride = self.sampling_config.get('stride_frames', 150)
        self.min_clip_frames = self.sampling_config.get('min_clip_frames', 100)
        
        # Storage for sequence metadata
        self.sequence_metadata = {}
        
        # Build index
        super().__init__(config)
        
        logger.info(
            f"Initialized MOT loader with {self.sampling_type} sampling: "
            f"{len(self._index)} clips from {len(self.sequence_metadata)} sequences"
        )
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build index of video clips using sliding window strategy.
        
        Returns:
            List of clip descriptors, each containing:
            - sequence_id: Name of the source sequence
            - sequence_path: Path to the sequence directory
            - start_frame: Starting frame of the clip
            - end_frame: Ending frame of the clip
        """
        logger.info(f"Scanning MOT dataset at {self.base_path}")
        
        # First, find all valid sequences
        valid_sequences = self._find_valid_sequences()
        
        if self.sampling_type == 'full_sequence':
            # Traditional approach: one sample per sequence
            return self._build_full_sequence_index(valid_sequences)
        elif self.sampling_type == 'sliding_window':
            # New approach: multiple clips per sequence
            return self._build_sliding_window_index(valid_sequences)
        else:
            raise ValueError(f"Unknown sampling type: {self.sampling_type}")
    
    def _find_valid_sequences(self) -> List[Path]:
        """Find all valid MOT sequences in the dataset."""
        valid_sequences = []
        
        for sequence_path in self.base_path.iterdir():
            if not sequence_path.is_dir():
                continue
            
            # Check for required subdirectories and files
            img_dir = sequence_path / 'img1'
            gt_file = sequence_path / 'gt' / 'gt.txt'
            
            if img_dir.is_dir() and gt_file.is_file():
                valid_sequences.append(sequence_path)
                
                # Load and cache sequence metadata
                self._load_sequence_metadata(sequence_path)
            else:
                logger.debug(f"Skipping invalid sequence: {sequence_path.name}")
        
        logger.info(f"Found {len(valid_sequences)} valid sequences")
        return valid_sequences
    
    def _load_sequence_metadata(self, sequence_path: Path):
        """Load and cache metadata for a sequence."""
        sequence_id = sequence_path.name
        gt_file = sequence_path / 'gt' / 'gt.txt'
        
        try:
            # Load ground truth to determine frame range
            df = pd.read_csv(
                gt_file,
                header=None,
                names=['frame_id', 'object_id', 'bb_left', 'bb_top', 
                      'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
            )
            
            min_frame = int(df['frame_id'].min())
            max_frame = int(df['frame_id'].max())
            num_objects = df['object_id'].nunique()
            
            self.sequence_metadata[sequence_id] = {
                'path': sequence_path,
                'min_frame': min_frame,
                'max_frame': max_frame,
                'total_frames': max_frame - min_frame + 1,
                'num_objects': num_objects
            }
            
        except Exception as e:
            logger.warning(f"Error loading metadata for {sequence_id}: {e}")
            self.sequence_metadata[sequence_id] = {
                'path': sequence_path,
                'min_frame': 1,
                'max_frame': 1,
                'total_frames': 1,
                'num_objects': 0
            }
    
    def _build_full_sequence_index(self, sequences: List[Path]) -> List[Dict[str, Any]]:
        """Build traditional index with one entry per sequence."""
        index = []
        
        for seq_path in sequences:
            seq_id = seq_path.name
            metadata = self.sequence_metadata[seq_id]
            
            index.append({
                'sequence_id': seq_id,
                'sequence_path': seq_path,
                'start_frame': metadata['min_frame'],
                'end_frame': metadata['max_frame']
            })
        
        return index
    
    def _build_sliding_window_index(self, sequences: List[Path]) -> List[Dict[str, Any]]:
        """Build index with multiple clips per sequence using sliding window."""
        index = []
        total_clips = 0
        
        for seq_path in sequences:
            seq_id = seq_path.name
            metadata = self.sequence_metadata[seq_id]
            
            min_frame = metadata['min_frame']
            max_frame = metadata['max_frame']
            total_frames = metadata['total_frames']
            
            # Skip sequences that are too short
            if total_frames < self.min_clip_frames:
                logger.debug(
                    f"Skipping {seq_id}: {total_frames} frames < "
                    f"minimum {self.min_clip_frames}"
                )
                continue
            
            # Generate clip windows
            clips_for_sequence = 0
            current_start = min_frame
            
            while current_start <= max_frame:
                # Calculate end frame for this clip
                current_end = min(current_start + self.clip_duration - 1, max_frame)
                
                # Check if clip is long enough
                clip_length = current_end - current_start + 1
                if clip_length >= self.min_clip_frames:
                    index.append({
                        'sequence_id': seq_id,
                        'sequence_path': seq_path,
                        'start_frame': current_start,
                        'end_frame': current_end,
                        'clip_index': clips_for_sequence  # Track which clip this is
                    })
                    clips_for_sequence += 1
                
                # Move to next window
                current_start += self.stride
                
                # Stop if we can't create a full minimum-size clip
                if current_start + self.min_clip_frames - 1 > max_frame:
                    break
            
            total_clips += clips_for_sequence
            logger.debug(f"Created {clips_for_sequence} clips from {seq_id}")
        
        logger.info(
            f"Created {total_clips} clips from {len(sequences)} sequences "
            f"(average {total_clips/len(sequences):.1f} clips per sequence)"
        )
        
        return index
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get a single clip with its tracking annotations.
        
        Args:
            index: Index of the clip to retrieve
            
        Returns:
            Standardized sample dictionary with tracking data for the clip
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get clip descriptor
        clip_info = self._index[index]
        sequence_id = clip_info['sequence_id']
        sequence_path = clip_info['sequence_path']
        start_frame = clip_info['start_frame']
        end_frame = clip_info['end_frame']
        
        # Build sample ID that includes clip information
        if 'clip_index' in clip_info:
            sample_id = f"{sequence_id}_clip{clip_info['clip_index']:03d}"
        else:
            sample_id = sequence_id
        
        # Path to images
        img_dir = sequence_path / 'img1'
        
        # Create base structure
        sample = {
            'sample_id': sample_id,
            'media_path': str(img_dir),
            'dataset': 'mot',
            'media_type': 'video_frames',
            
            # Include clip metadata
            'clip_info': {
                'sequence_id': sequence_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': end_frame - start_frame + 1
            }
        }
        
        # Load and filter ground truth for this clip
        gt_file = sequence_path / 'gt' / 'gt.txt'
        
        try:
            # Load ground truth
            df = pd.read_csv(
                gt_file,
                header=None,
                names=['frame_id', 'object_id', 'bb_left', 'bb_top',
                      'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
            )
            
            # CRITICAL: Filter to only include frames in this clip
            df_clip = df[(df['frame_id'] >= start_frame) & 
                        (df['frame_id'] <= end_frame)].copy()
            
            # Normalize frame IDs to start from 0 within the clip
            df_clip['clip_frame_id'] = df_clip['frame_id'] - start_frame
            
            # Group by object_id to get trajectories
            object_trajectories = []
            
            for obj_id, obj_df in df_clip.groupby('object_id'):
                trajectory = []
                
                for _, row in obj_df.iterrows():
                    trajectory.append({
                        'frame': int(row['clip_frame_id']),  # Normalized frame ID
                        'original_frame': int(row['frame_id']),  # Original frame ID
                        'bbox': [
                            float(row['bb_left']),
                            float(row['bb_top']),
                            float(row['bb_width']),
                            float(row['bb_height'])
                        ],
                        'confidence': float(row['conf']) if pd.notna(row['conf']) else 1.0
                    })
                
                object_trajectories.append({
                    'object_id': int(obj_id),
                    'trajectory': sorted(trajectory, key=lambda x: x['frame']),
                    'num_frames': len(trajectory)
                })
            
            # Add tracking annotations
            sample['annotations'] = {
                'tracking': {
                    'trajectories': object_trajectories,
                    'num_objects': len(object_trajectories),
                    'num_frames_annotated': df_clip['frame_id'].nunique()
                }
            }
            
        except Exception as e:
            logger.warning(f"Error loading annotations for {sample_id}: {e}")
            sample['annotations'] = {
                'tracking': {
                    'trajectories': [],
                    'num_objects': 0,
                    'num_frames_annotated': 0
                }
            }
        
        return sample
    
    def get_clip_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated clips.
        
        Returns:
            Dictionary with clip distribution statistics
        """
        if self.sampling_type != 'sliding_window':
            return {'sampling_type': self.sampling_type}
        
        clips_per_sequence = {}
        for clip in self._index:
            seq_id = clip['sequence_id']
            clips_per_sequence[seq_id] = clips_per_sequence.get(seq_id, 0) + 1
        
        clip_lengths = [
            clip['end_frame'] - clip['start_frame'] + 1 
            for clip in self._index
        ]
        
        return {
            'sampling_type': self.sampling_type,
            'total_clips': len(self._index),
            'total_sequences': len(clips_per_sequence),
            'clips_per_sequence': {
                'mean': sum(clips_per_sequence.values()) / len(clips_per_sequence),
                'min': min(clips_per_sequence.values()),
                'max': max(clips_per_sequence.values())
            },
            'clip_lengths': {
                'mean': sum(clip_lengths) / len(clip_lengths) if clip_lengths else 0,
                'min': min(clip_lengths) if clip_lengths else 0,
                'max': max(clip_lengths) if clip_lengths else 0
            },
            'config': {
                'clip_duration': self.clip_duration,
                'stride': self.stride,
                'overlap_ratio': 1.0 - (self.stride / self.clip_duration)
            }
        }


def create_mot_loader(config: Dict[str, Any]) -> BaseLoader:
    """
    Factory function to create MOT loader with appropriate sampling strategy.
    
    Args:
        config: Loader configuration
        
    Returns:
        MotLoader or MotSlidingWindowLoader based on config
    """
    sampling_config = config.get('sampling_strategy', {})
    sampling_type = sampling_config.get('type', 'full_sequence')
    
    if sampling_type == 'sliding_window':
        return MotSlidingWindowLoader(config)
    else:
        # Import original loader for backward compatibility
        from .mot_loader import MotLoader
        return MotLoader(config)