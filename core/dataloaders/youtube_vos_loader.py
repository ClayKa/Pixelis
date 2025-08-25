# core/dataloaders/youtube_vos_loader.py

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class YouTubeVOSLoader(BaseLoader):
    """
    A concrete data loader for the YouTube Video Object Segmentation (VOS) dataset.
    
    This loader is designed to handle the typical VOS structure:
    1. A single, large `instances.json` file containing all metadata and annotations.
    2. A main directory containing subdirectories for each video, where each
       subdirectory holds the extracted image frames.
    
    It builds an efficient index by pre-processing the main JSON file to group
    annotations by video.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YouTubeVOSLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'path': Path to the directory containing video subdirectories
                - 'annotation_file': Path to the instances.json file
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"YouTubeVOSLoader config must include '{key}'")
        
        self.frames_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        
        # Validate paths exist
        if not self.frames_path.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Initialize lookup structures
        self._video_id_to_info = {}
        self._category_id_to_name = {}
        self._video_id_to_annotations = defaultdict(list)
        self._video_name_to_id = {}  # Map video name to ID for directory matching
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[int]:
        """
        Load the large instances.json file once, parse its contents, and restructure
        the flat annotation list into an efficient, video-centric lookup.
        
        Returns:
            List of valid video IDs found in the dataset
        """
        logger.info(f"Loading annotations from {self.annotation_file}")
        
        # Step 1: Load the entire JSON
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        # Step 2: Create lookups
        self._process_videos(data.get('videos', []))
        self._process_categories(data.get('categories', []))
        self._process_annotations(data.get('annotations', []))
        
        logger.info(f"Processed {len(self._video_id_to_info)} videos, "
                   f"{len(self._category_id_to_name)} categories, "
                   f"{len(self._video_id_to_annotations)} videos with annotations")
        
        # Step 3: Validate against image folders
        valid_video_ids = self._validate_against_disk()
        
        logger.info(f"Found {len(valid_video_ids)} valid videos with both annotations and frames")
        
        # Sort for consistent ordering
        return sorted(valid_video_ids)

    def _process_videos(self, videos: List[Dict[str, Any]]):
        """
        Process video metadata from the JSON.
        
        Args:
            videos: List of video dictionaries from instances.json
        """
        for video in videos:
            video_id = video['id']
            self._video_id_to_info[video_id] = video
            
            # Extract video name from file_names for directory matching
            if 'file_names' in video and video['file_names']:
                # Format is typically "video_name/frame_000000.jpg"
                first_frame = video['file_names'][0]
                video_name = first_frame.split('/')[0] if '/' in first_frame else str(video_id)
                self._video_name_to_id[video_name] = video_id

    def _process_categories(self, categories: List[Dict[str, Any]]):
        """
        Process category information from the JSON.
        
        Args:
            categories: List of category dictionaries from instances.json
        """
        for category in categories:
            self._category_id_to_name[category['id']] = category.get('name', f"category_{category['id']}")

    def _process_annotations(self, annotations: List[Dict[str, Any]]):
        """
        Process and group annotations by video ID.
        
        Args:
            annotations: List of annotation dictionaries from instances.json
        """
        for ann in annotations:
            video_id = ann['video_id']
            self._video_id_to_annotations[video_id].append(ann)

    def _validate_against_disk(self) -> List[int]:
        """
        Validate that videos have corresponding frame directories on disk.
        
        Returns:
            List of video IDs that have both annotations and frame directories
        """
        valid_video_ids = []
        
        # Get all subdirectories in the frames path
        existing_dirs = {d.name for d in self.frames_path.iterdir() if d.is_dir()}
        
        for video_id, video_info in self._video_id_to_info.items():
            # Check if this video has annotations
            if video_id not in self._video_id_to_annotations:
                logger.debug(f"Video {video_id} has no annotations, skipping")
                continue
            
            # Check if frame directory exists
            video_name = self._get_video_name(video_id, video_info)
            if video_name in existing_dirs:
                valid_video_ids.append(video_id)
            else:
                logger.debug(f"Video {video_id} ({video_name}) has no frame directory, skipping")
        
        return valid_video_ids

    def _get_video_name(self, video_id: int, video_info: Dict[str, Any]) -> str:
        """
        Extract video name from video info for directory matching.
        
        Args:
            video_id: Video ID
            video_info: Video metadata dictionary
            
        Returns:
            Video name for directory lookup
        """
        if 'file_names' in video_info and video_info['file_names']:
            # Extract from first frame path
            first_frame = video_info['file_names'][0]
            return first_frame.split('/')[0] if '/' in first_frame else str(video_id)
        return str(video_id)

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single video and its complete set of object trajectories
        (annotations) and format it into the standardized dictionary.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with VOS data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Step 1: Retrieve video ID
        video_id = self._index[index]
        
        # Step 2: Get video info
        video_info = self._video_id_to_info[video_id]
        
        # Step 3: Construct frame directory path
        video_name = self._get_video_name(video_id, video_info)
        frame_dir = self.frames_path / video_name
        
        # Step 4: Create base structure
        sample = self._create_vos_base_structure(video_id, frame_dir)
        
        # Step 5: Retrieve and adapt annotations
        raw_annotations = self._video_id_to_annotations[video_id]
        tracks = self._process_video_annotations(raw_annotations)
        
        # Get video statistics
        num_frames = len(video_info.get('file_names', []))
        
        # Add VOS-specific annotations
        sample['annotations'].update({
            'video_object_segmentation': {
                'video_id': video_id,
                'video_name': video_name,
                'object_tracks_vos': tracks,
                'num_objects': len(tracks),
                'num_frames': num_frames,
                'frame_files': video_info.get('file_names', []),
                'video_metadata': {
                    'width': video_info.get('width'),
                    'height': video_info.get('height'),
                    'length': video_info.get('length', num_frames)
                },
                'tracking_statistics': self._calculate_tracking_statistics(tracks, num_frames)
            },
            'video_metadata': {
                'frame_directory': str(frame_dir),
                'total_frames': num_frames,
                'resolution': [video_info.get('width'), video_info.get('height')]
            },
            'dataset_info': {
                'task_type': 'video_object_segmentation',
                'source': 'YouTube-VOS',
                'suitable_for_tracking': True,
                'suitable_for_segmentation': True,
                'suitable_for_temporal_reasoning': True,
                'has_pixel_masks': True,
                'annotation_format': 'rle_masks'
            }
        })
        
        return sample

    def _create_vos_base_structure(self, video_id: int, frame_dir: Path) -> Dict[str, Any]:
        """
        Create the base standardized structure for VOS samples.
        
        Args:
            video_id: The video identifier
            frame_dir: Path to the frame directory
            
        Returns:
            Dictionary with the basic standardized structure
        """
        if not frame_dir.exists():
            logger.warning(
                f"Frame directory not found for video_id '{video_id}' in source '{self.source_name}'. "
                f"Path: {frame_dir}"
            )
        
        return {
            "source_dataset": self.source_name,
            "sample_id": str(video_id),
            "media_type": "video",
            "media_path": str(frame_dir.resolve()),
            "width": None,   # Will be filled from video_info if available
            "height": None,  # Will be filled from video_info if available
            "annotations": {}
        }

    def _process_video_annotations(self, raw_annotations: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Process raw annotations into structured tracks.
        
        Args:
            raw_annotations: List of raw annotation dictionaries
            
        Returns:
            Dictionary mapping object IDs to their track information
        """
        tracks = {}
        
        for ann in raw_annotations:
            object_id = ann['id']
            category_id = ann.get('category_id', 0)
            
            # Build trajectory from segmentations
            trajectory = []
            segmentations = ann.get('segmentations', [])
            
            for frame_idx, mask_data in enumerate(segmentations):
                if mask_data is not None:  # Skip frames where object is not visible
                    trajectory.append({
                        'frame_index': frame_idx,
                        'mask_rle': mask_data,  # Keep RLE format for efficiency
                        'bbox': ann.get('bboxes', [None] * len(segmentations))[frame_idx]
                    })
            
            tracks[object_id] = {
                'category_id': category_id,
                'category_name': self._category_id_to_name.get(category_id, f'category_{category_id}'),
                'trajectory': trajectory,
                'num_frames_visible': len(trajectory),
                'first_frame': trajectory[0]['frame_index'] if trajectory else None,
                'last_frame': trajectory[-1]['frame_index'] if trajectory else None,
                'areas': ann.get('areas', []),  # Area for each frame if available
                'iscrowd': ann.get('iscrowd', 0)
            }
        
        return tracks

    def _calculate_tracking_statistics(self, tracks: Dict[int, Dict[str, Any]], num_frames: int) -> Dict[str, Any]:
        """
        Calculate statistics about the tracks in this video.
        
        Args:
            tracks: Dictionary of object tracks
            num_frames: Total number of frames in the video
            
        Returns:
            Dictionary with tracking statistics
        """
        if not tracks:
            return {
                'avg_track_length': 0.0,
                'min_track_length': 0,
                'max_track_length': 0,
                'total_annotations': 0,
                'density': 0.0
            }
        
        track_lengths = [track['num_frames_visible'] for track in tracks.values()]
        total_annotations = sum(track_lengths)
        
        return {
            'avg_track_length': sum(track_lengths) / len(track_lengths),
            'min_track_length': min(track_lengths),
            'max_track_length': max(track_lengths),
            'total_annotations': total_annotations,
            'density': total_annotations / (num_frames * len(tracks)) if num_frames > 0 else 0.0,
            'unique_objects': len(tracks)
        }

    @staticmethod
    def decode_rle(rle: Dict[str, Any], height: int, width: int) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoding) mask to binary numpy array.
        
        This is a utility function for downstream tasks that need the actual mask.
        
        Args:
            rle: RLE dictionary with 'size' and 'counts'
            height: Height of the mask
            width: Width of the mask
            
        Returns:
            Binary mask as numpy array
        """
        try:
            # Import pycocotools for RLE decoding
            from pycocotools import mask as mask_utils
            
            # Ensure RLE is in correct format
            if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
                # Decode RLE to binary mask
                mask = mask_utils.decode(rle)
                return mask.astype(np.uint8)
            else:
                logger.warning(f"Invalid RLE format: {rle}")
                return np.zeros((height, width), dtype=np.uint8)
                
        except ImportError:
            logger.warning("pycocotools not installed, returning empty mask")
            return np.zeros((height, width), dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Failed to decode RLE: {e}")
            return np.zeros((height, width), dtype=np.uint8)

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
            num_objects = sample['annotations']['video_object_segmentation']['num_objects']
            
            if min_objects <= num_objects <= max_objects:
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_by_duration(self, min_frames: int = 1, max_frames: int = float('inf')) -> List[Dict[str, Any]]:
        """
        Get samples filtered by video duration (number of frames).
        
        Args:
            min_frames: Minimum number of frames
            max_frames: Maximum number of frames
            
        Returns:
            List of samples within the frame count range
        """
        matching_samples = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            num_frames = sample['annotations']['video_object_segmentation']['num_frames']
            
            if min_frames <= num_frames <= max_frames:
                matching_samples.append(sample)
        
        return matching_samples

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset-level statistics
        """
        total_videos = len(self._index)
        total_objects = 0
        total_annotations = 0
        total_frames = 0
        
        category_counts = defaultdict(int)
        video_lengths = []
        object_counts = []
        
        for video_id in self._index:
            video_info = self._video_id_to_info[video_id]
            annotations = self._video_id_to_annotations[video_id]
            
            num_frames = len(video_info.get('file_names', []))
            num_objects = len(annotations)
            
            total_frames += num_frames
            total_objects += num_objects
            video_lengths.append(num_frames)
            object_counts.append(num_objects)
            
            # Count categories
            for ann in annotations:
                category_id = ann.get('category_id', 0)
                category_counts[self._category_id_to_name.get(category_id, f'category_{category_id}')] += 1
                
                # Count non-null segmentations
                segmentations = ann.get('segmentations', [])
                total_annotations += sum(1 for seg in segmentations if seg is not None)
        
        return {
            'total_videos': total_videos,
            'total_unique_objects': total_objects,
            'total_annotations': total_annotations,
            'total_frames': total_frames,
            'avg_objects_per_video': total_objects / total_videos if total_videos > 0 else 0,
            'avg_frames_per_video': total_frames / total_videos if total_videos > 0 else 0,
            'avg_annotations_per_frame': total_annotations / total_frames if total_frames > 0 else 0,
            'video_length_distribution': {
                'min': min(video_lengths) if video_lengths else 0,
                'max': max(video_lengths) if video_lengths else 0,
                'avg': sum(video_lengths) / len(video_lengths) if video_lengths else 0
            },
            'object_count_distribution': {
                'min': min(object_counts) if object_counts else 0,
                'max': max(object_counts) if object_counts else 0,
                'avg': sum(object_counts) / len(object_counts) if object_counts else 0
            },
            'category_distribution': dict(category_counts),
            'num_categories': len(self._category_id_to_name)
        }

    def get_category_statistics(self) -> Dict[str, int]:
        """
        Get the distribution of categories in the dataset.
        
        Returns:
            Dictionary mapping category names to their counts
        """
        category_counts = defaultdict(int)
        
        for video_id in self._index:
            annotations = self._video_id_to_annotations[video_id]
            for ann in annotations:
                category_id = ann.get('category_id', 0)
                category_name = self._category_id_to_name.get(category_id, f'category_{category_id}')
                category_counts[category_name] += 1
        
        return dict(category_counts)

    def get_video_by_name(self, video_name: str) -> Dict[str, Any]:
        """
        Get a specific video sample by its name.
        
        Args:
            video_name: Name of the video directory
            
        Returns:
            Sample dictionary for the specified video
            
        Raises:
            ValueError: If video name is not found
        """
        if video_name not in self._video_name_to_id:
            raise ValueError(f"Video '{video_name}' not found in dataset")
        
        video_id = self._video_name_to_id[video_name]
        
        try:
            index = self._index.index(video_id)
            return self.get_item(index)
        except ValueError:
            raise ValueError(f"Video '{video_name}' (ID: {video_id}) not in valid index")

    def list_video_names(self) -> List[str]:
        """
        Get a list of all video names in the dataset.
        
        Returns:
            List of video directory names
        """
        names = []
        for video_id in self._index:
            video_info = self._video_id_to_info[video_id]
            video_name = self._get_video_name(video_id, video_info)
            names.append(video_name)
        return names

    def list_categories(self) -> List[str]:
        """
        Get a list of all category names in the dataset.
        
        Returns:
            List of category names
        """
        return list(self._category_id_to_name.values())