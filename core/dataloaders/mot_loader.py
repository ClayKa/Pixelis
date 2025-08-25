# core/dataloaders/mot_loader.py

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MotLoader(BaseLoader):
    """
    A concrete data loader for Multi-Object Tracking (MOT) datasets like MOT20.
    
    This loader is designed to handle the specific MOT directory structure, where
    each video sequence has its own folder containing:
    1. An `img1/` subdirectory with individual frames.
    2. A `gt/` subdirectory with a `gt.txt` file for ground truth annotations.
    
    It focuses on parsing the ground truth data to extract object trajectories.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MotLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'path': Path to the directory containing sequence folders
        """
        # Validate required config keys
        required_keys = ['path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"MotLoader config must include '{key}'")
        
        self.base_path = Path(config['path'])
        
        # Validate path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_path}")
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Base path is not a directory: {self.base_path}")
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Path]:
        """
        Scan the base directory, discover all valid video sequence subdirectories,
        and create an index of them.
        
        A valid sequence must contain both an image folder (`img1/`) and 
        a ground truth file (`gt/gt.txt`).
        
        Returns:
            List of paths to valid sequence directories
        """
        logger.info(f"Scanning MOT dataset at {self.base_path}")
        
        valid_sequences = []
        
        # Iterate through all subdirectories in the base path
        for sequence_path in self.base_path.iterdir():
            if not sequence_path.is_dir():
                continue
                
            # Check for required subdirectories and files
            img_dir = sequence_path / 'img1'
            gt_file = sequence_path / 'gt' / 'gt.txt'
            
            # Validate sequence structure
            if img_dir.is_dir() and gt_file.is_file():
                valid_sequences.append(sequence_path)
                logger.debug(f"Found valid sequence: {sequence_path.name}")
            else:
                logger.debug(f"Skipping invalid sequence {sequence_path.name}: "
                           f"img1={img_dir.exists()}, gt.txt={gt_file.exists()}")
        
        logger.info(f"Found {len(valid_sequences)} valid sequences")
        return valid_sequences

    def _create_mot_base_structure(self, sample_id: str, img_dir: Path) -> Dict[str, Any]:
        """
        Create the base standardized structure for MOT samples.
        
        Since MOT sequences are directories of images (not single files), 
        we need custom logic instead of using BaseLoader's _get_standardized_base.
        
        Args:
            sample_id: The sequence identifier
            img_dir: Path to the img1 directory containing frames
            
        Returns:
            Dictionary with the basic standardized structure
        """
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found for sample_id '{sample_id}' in source '{self.source_name}'. "
                f"Checked path: {img_dir}"
            )
        
        # Check if there are any image files in the directory
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        if not image_files:
            raise FileNotFoundError(
                f"No image files found in directory for sample_id '{sample_id}' in source '{self.source_name}'. "
                f"Checked path: {img_dir}"
            )
        
        return {
            "source_dataset": self.source_name,
            "sample_id": sample_id,
            "media_type": "video",
            "media_path": str(img_dir.resolve()),  # Use absolute path to image directory
            "width": None,   # Video dimensions not extracted for directory-based sequences
            "height": None,  # Video dimensions not extracted for directory-based sequences
            "annotations": {}  # Initialize an empty dict for annotations
        }

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single sequence (video), parse its ground truth file to 
        reconstruct all object trajectories, and format this into the 
        standardized dictionary.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with MOT tracking data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get sequence path from index
        sequence_path = self._index[index]
        sequence_id = sequence_path.name
        
        # For MOT, we treat the img1 folder as the "video" media
        img_dir = sequence_path / 'img1'
        
        # Create base standardized structure manually since BaseLoader expects files, not directories
        sample = self._create_mot_base_structure(sequence_id, img_dir)
        
        # Parse the ground truth file
        gt_file = sequence_path / 'gt' / 'gt.txt'
        object_tracks = self._parse_ground_truth(gt_file)
        
        # Get sequence statistics
        frame_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        
        # Add MOT-specific annotations
        sample['annotations'].update({
            'multi_object_tracking': {
                'sequence_id': sequence_id,
                'object_tracks': object_tracks,
                'num_objects': len(object_tracks),
                'num_frames': frame_count,
                'track_statistics': self._analyze_tracks(object_tracks),
                'sequence_info': {
                    'img_directory': str(img_dir),
                    'gt_file': str(gt_file),
                    'sequence_name': sequence_id
                }
            },
            'video_metadata': {
                'frame_directory': str(img_dir),
                'total_frames': frame_count,
                'sequence_duration_frames': frame_count
            },
            'dataset_info': {
                'task_type': 'multi_object_tracking',
                'source': 'MOT',
                'suitable_for_tracking': True,
                'suitable_for_temporal_reasoning': True,
                'has_object_trajectories': len(object_tracks) > 0,
                'has_frame_annotations': True,
                'annotation_format': 'mot_format'
            }
        })
        
        return sample

    def _parse_ground_truth(self, gt_file: Path) -> Dict[int, List[Dict[str, Any]]]:
        """
        Parse the MOT format ground truth file and group annotations by object ID.
        
        MOT format columns:
        frame_id, object_id, bb_left, bb_top, bb_width, bb_height, confidence, class, visibility
        
        Args:
            gt_file: Path to the gt.txt file
            
        Returns:
            Dictionary mapping object_id to list of frame annotations
        """
        try:
            # Read the ground truth file with pandas
            # MOT format is CSV-like with no headers
            df = pd.read_csv(
                gt_file, 
                header=None,
                names=['frame_id', 'object_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'confidence', 'class', 'visibility']
            )
            
            # Group by object_id to reconstruct trajectories
            tracks = defaultdict(list)
            
            for _, row in df.iterrows():
                object_id = int(row['object_id'])
                
                # Create standardized annotation for this frame
                annotation = {
                    'frame_id': int(row['frame_id']),
                    'bbox': [
                        float(row['bb_left']), 
                        float(row['bb_top']), 
                        float(row['bb_width']), 
                        float(row['bb_height'])
                    ],
                    'confidence': float(row['confidence']),
                    'class': int(row['class']) if pd.notna(row['class']) else -1,
                    'visibility': float(row['visibility']) if pd.notna(row['visibility']) else 1.0
                }
                
                tracks[object_id].append(annotation)
            
            # Sort each track by frame_id for temporal consistency
            for object_id in tracks:
                tracks[object_id].sort(key=lambda x: x['frame_id'])
            
            logger.debug(f"Parsed {len(tracks)} object tracks from {gt_file}")
            return dict(tracks)
            
        except Exception as e:
            logger.warning(f"Failed to parse ground truth file {gt_file}: {e}")
            return {}

    def _analyze_tracks(self, tracks: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze object tracks to provide statistics.
        
        Args:
            tracks: Dictionary of object tracks
            
        Returns:
            Dictionary with track analysis metrics
        """
        if not tracks:
            return {
                'avg_track_length': 0.0,
                'min_track_length': 0,
                'max_track_length': 0,
                'total_detections': 0,
                'objects_per_frame': 0.0
            }
        
        track_lengths = [len(track) for track in tracks.values()]
        total_detections = sum(track_lengths)
        
        # Calculate frame span
        all_frames = set()
        for track in tracks.values():
            for annotation in track:
                all_frames.add(annotation['frame_id'])
        
        frame_span = len(all_frames) if all_frames else 1
        
        return {
            'avg_track_length': sum(track_lengths) / len(track_lengths),
            'min_track_length': min(track_lengths),
            'max_track_length': max(track_lengths),
            'total_detections': total_detections,
            'objects_per_frame': total_detections / frame_span,
            'frame_span': frame_span,
            'unique_objects': len(tracks)
        }

    def get_samples_by_object_count(self, min_objects: int = 1, max_objects: int = float('inf')) -> List[Dict[str, Any]]:
        """
        Get samples filtered by number of tracked objects.
        
        Args:
            min_objects: Minimum number of objects
            max_objects: Maximum number of objects
            
        Returns:
            List of samples within the object count range
        """
        matching_samples = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            num_objects = sample['annotations']['multi_object_tracking']['num_objects']
            
            if min_objects <= num_objects <= max_objects:
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_by_duration(self, min_frames: int = 1, max_frames: int = float('inf')) -> List[Dict[str, Any]]:
        """
        Get samples filtered by sequence duration (number of frames).
        
        Args:
            min_frames: Minimum number of frames
            max_frames: Maximum number of frames
            
        Returns:
            List of samples within the frame count range
        """
        matching_samples = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            num_frames = sample['annotations']['multi_object_tracking']['num_frames']
            
            if min_frames <= num_frames <= max_frames:
                matching_samples.append(sample)
        
        return matching_samples

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the tracking dataset.
        
        Returns:
            Dictionary with dataset-level tracking statistics
        """
        total_objects = 0
        total_detections = 0
        total_frames = 0
        sequence_lengths = []
        object_counts = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            mot_data = sample['annotations']['multi_object_tracking']
            
            num_objects = mot_data['num_objects']
            num_frames = mot_data['num_frames']
            track_stats = mot_data['track_statistics']
            
            total_objects += num_objects
            total_detections += track_stats['total_detections']
            total_frames += num_frames
            sequence_lengths.append(num_frames)
            object_counts.append(num_objects)
        
        return {
            'total_sequences': len(self._index),
            'total_unique_objects': total_objects,
            'total_detections': total_detections,
            'total_frames': total_frames,
            'avg_objects_per_sequence': total_objects / len(self._index) if self._index else 0,
            'avg_frames_per_sequence': total_frames / len(self._index) if self._index else 0,
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'sequence_length_distribution': {
                'min': min(sequence_lengths) if sequence_lengths else 0,
                'max': max(sequence_lengths) if sequence_lengths else 0,
                'avg': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
            },
            'object_count_distribution': {
                'min': min(object_counts) if object_counts else 0,
                'max': max(object_counts) if object_counts else 0,
                'avg': sum(object_counts) / len(object_counts) if object_counts else 0
            }
        }

    def get_sequence_by_name(self, sequence_name: str) -> Dict[str, Any]:
        """
        Get a specific sequence by its name.
        
        Args:
            sequence_name: Name of the sequence (e.g., "MOT20-01")
            
        Returns:
            Sample dictionary for the specified sequence
            
        Raises:
            ValueError: If sequence name is not found
        """
        for i, sequence_path in enumerate(self._index):
            if sequence_path.name == sequence_name:
                return self.get_item(i)
        
        raise ValueError(f"Sequence '{sequence_name}' not found in dataset")

    def list_sequence_names(self) -> List[str]:
        """
        Get a list of all sequence names in the dataset.
        
        Returns:
            List of sequence names
        """
        return [sequence_path.name for sequence_path in self._index]