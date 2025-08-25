# core/dataloaders/assembly101_loader.py

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Assembly101Loader(BaseLoader):
    """
    A concrete data loader for the Assembly101 dataset.

    This loader is designed to work with a structure where annotations are provided
    in CSV files (e.g., train.csv, validation.csv, test.csv), with each row representing a
    specific action segment within a video, identified by start and end frames.
    
    Assembly101 is a large-scale multi-view video dataset for learning and benchmarking
    a diverse set of procedural activities, focused on toy assembly tasks.
    """

    def _build_index(self) -> List[Dict]:
        """
        Load the main annotation CSV file for a specific split (e.g., train),
        validate its entries against the locally available video files, and build
        a lightweight index. Each element in the index represents one unique action segment.
        
        Returns:
            List[Dict]: A list of annotation dictionaries for valid action segments.
        """
        # Get the annotation file path from config
        if 'annotation_file' not in self.config:
            raise ValueError(f"Assembly101Loader requires 'annotation_file' in config")
        
        annotation_file_path = Path(self.config['annotation_file'])
        
        if not annotation_file_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file_path}"
            )
        
        # Get the video directory path
        if 'path' not in self.config:
            raise ValueError(f"Assembly101Loader requires 'path' (video directory) in config")
        
        video_dir_path = Path(self.config['path'])
        
        if not video_dir_path.is_dir():
            raise FileNotFoundError(
                f"Video directory not found: {video_dir_path}"
            )
        
        # Load annotations with pandas
        logger.info(f"Loading annotations from {annotation_file_path}")
        df = pd.read_csv(annotation_file_path)
        logger.info(f"Loaded {len(df)} action segments from CSV")
        
        # Build a set of available video files for fast lookup
        # Since videos are in subdirectories, we need to traverse them
        logger.info("Building video file index for validation...")
        available_videos = set()
        
        # The video paths in CSV are relative to the videos directory
        # Format: subdirectory/camera_rgb.mp4
        for subdir in video_dir_path.iterdir():
            if subdir.is_dir():
                for video_file in subdir.glob("*.mp4"):
                    # Store relative path from video_dir_path
                    relative_path = f"{subdir.name}/{video_file.name}"
                    available_videos.add(relative_path)
        
        logger.info(f"Found {len(available_videos)} video files in directory")
        
        # Validate against videos and build index
        self._index = []
        missing_videos = set()
        
        # Group by video to check each unique video only once
        unique_videos = df['video'].unique()
        valid_videos = set()
        
        for video_path in unique_videos:
            if video_path in available_videos:
                valid_videos.add(video_path)
            else:
                missing_videos.add(video_path)
        
        # Now filter the dataframe to only include rows with valid videos
        valid_df = df[df['video'].isin(valid_videos)]
        
        # Convert each valid row to a dictionary and add to index
        for index, row in valid_df.iterrows():
            self._index.append(row.to_dict())
        
        if missing_videos:
            logger.warning(
                f"Found {len(missing_videos)} unique videos in annotations but not in video directory. "
                f"First 10 missing: {list(missing_videos)[:10]}"
            )
        
        logger.info(
            f"Built index with {len(self._index)} valid action segments "
            f"(out of {len(df)} total annotations)"
        )
        
        return self._index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single action segment annotation from the index and format it
        into the project's standardized dictionary.
        
        Args:
            index (int): The integer index of the sample in the self._index list.
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary with video path and
                          action segment information.
        """
        # Retrieve annotation row from the index
        ann_row = self._index[index]
        
        # Extract key information
        video_filename = ann_row['video']
        start_frame = ann_row['start_frame']
        end_frame = ann_row['end_frame']
        
        # Handle start/end frames that might be strings with leading zeros
        if isinstance(start_frame, str):
            start_frame = int(start_frame)
        if isinstance(end_frame, str):
            end_frame = int(end_frame)
        
        # Create a unique sample ID
        # Use the annotation id if available, otherwise create from video and frames
        if 'id' in ann_row:
            sample_id = f"segment_{ann_row['id']}"
        else:
            video_stem = Path(video_filename).stem
            sample_id = f"{video_stem}_{start_frame:06d}_{end_frame:06d}"
        
        # Construct the full path to the video file
        video_path = Path(self.config['path']) / video_filename
        
        # Create base structure using helper method
        sample = self._get_standardized_base(
            sample_id=sample_id,
            media_path=video_path,
            media_type="video"
        )
        
        # Populate annotations with action segment information
        sample['annotations']['action_segment'] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': end_frame - start_frame + 1
        }
        
        # Add action information if available
        if 'action_cls' in ann_row:
            sample['annotations']['action_segment']['action'] = ann_row['action_cls']
        
        if 'verb_cls' in ann_row and 'noun_cls' in ann_row:
            sample['annotations']['action_segment']['verb'] = ann_row['verb_cls']
            sample['annotations']['action_segment']['noun'] = ann_row['noun_cls']
        
        # Add action IDs if available
        if 'action_id' in ann_row:
            sample['annotations']['action_segment']['action_id'] = ann_row['action_id']
        
        if 'verb_id' in ann_row:
            sample['annotations']['action_segment']['verb_id'] = ann_row['verb_id']
        
        if 'noun_id' in ann_row:
            sample['annotations']['action_segment']['noun_id'] = ann_row['noun_id']
        
        # Add toy information if available
        if 'toy_id' in ann_row and ann_row['toy_id'] != '-':
            sample['annotations']['action_segment']['toy_id'] = ann_row['toy_id']
        
        if 'toy_name' in ann_row and ann_row['toy_name'] != '-':
            sample['annotations']['action_segment']['toy_name'] = ann_row['toy_name']
        
        # Add metadata flags
        if 'is_shared' in ann_row:
            sample['annotations']['is_shared'] = bool(ann_row['is_shared'])
        
        if 'is_RGB' in ann_row:
            sample['annotations']['is_RGB'] = bool(ann_row['is_RGB'])
        
        # Store the full annotation ID for reference
        if 'id' in ann_row:
            sample['annotations']['annotation_id'] = ann_row['id']
        
        return sample