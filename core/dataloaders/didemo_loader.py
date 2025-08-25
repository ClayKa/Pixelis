# core/dataloaders/didemo_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class DiDeMoLoader(BaseLoader):
    """
    A concrete data loader for the DiDeMo (Distinct Describable Moments) dataset.

    This loader is responsible for:
    1. Parsing the main annotation JSON file, which contains a list of
       "moment" annotations (description, video_id, timestamp).
    2. Building an index of all moments for which a corresponding video file exists.
    3. Providing standardized samples that associate a video with a specific
       describable moment.
    
    The DiDeMo dataset contains videos with natural language descriptions of 
    specific moments, where each moment is typically 5 seconds long.
    """

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load the main annotation JSON file once, validate its entries against
        the locally available video files, and build a lightweight index of all valid "moment" samples.
        
        Returns:
            List[Dict[str, Any]]: A list of moment annotation dictionaries for which video files exist.
        """
        # Get the annotation file path from config
        if 'annotation_file' not in self.config:
            raise ValueError(f"DiDeMoLoader requires 'annotation_file' in config")
        
        annotation_file_path = Path(self.config['annotation_file'])
        
        if not annotation_file_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file_path}"
            )
        
        # Load the annotation JSON file
        logger.info(f"Loading annotations from {annotation_file_path}")
        with open(annotation_file_path, 'r') as f:
            annotations_list = json.load(f)
        
        # Get the video directory path
        if 'path' not in self.config:
            raise ValueError(f"DiDeMoLoader requires 'path' (video directory) in config")
        
        video_dir_path = Path(self.config['path'])
        
        if not video_dir_path.is_dir():
            raise FileNotFoundError(
                f"Video directory not found: {video_dir_path}"
            )
        
        # Build a set of existing video files for faster lookup
        logger.info("Building video file index for fast lookup...")
        existing_videos = set()
        video_extensions = ['.mp4', '.avi', '.mov', '.m4v', '.mkv']  # Common video formats in DiDeMo
        
        # Create a mapping from stem to actual filename
        video_stem_map = {}
        for ext in video_extensions:
            for video_file in video_dir_path.glob(f"*{ext}"):
                existing_videos.add(video_file.name)
                video_stem_map[video_file.stem] = video_file.name
        
        logger.info(f"Found {len(existing_videos)} video files in directory")
        
        # Validate against videos and build index
        self._index = []
        missing_videos = []
        
        for moment_ann in annotations_list:
            video_filename = moment_ann['video']
            
            # Check if video exists directly
            if video_filename in existing_videos:
                self._index.append(moment_ann)
            else:
                # Try to find by stem (in case extension differs)
                video_stem = Path(video_filename).stem
                if video_stem in video_stem_map:
                    # Update the annotation with the correct filename
                    moment_ann['video'] = video_stem_map[video_stem]
                    self._index.append(moment_ann)
                else:
                    missing_videos.append(video_filename)
        
        if missing_videos:
            logger.warning(
                f"Found {len(missing_videos)} moments with missing videos. "
                f"First 10 missing: {missing_videos[:10]}"
            )
        
        logger.info(
            f"Built index with {len(self._index)} valid moment samples "
            f"(out of {len(annotations_list)} total annotations)"
        )
        
        return self._index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single "moment" annotation from the index and format it 
        into the project's standardized dictionary.
        
        Args:
            index (int): The integer index of the sample in the self._index list.
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary with video path and
                          moment description with timestamps.
        """
        # Retrieve moment annotation from the index
        moment_ann = self._index[index]
        
        # Extract key information
        video_filename = moment_ann['video']
        description = moment_ann['description']
        annotation_id = moment_ann.get('annotation_id', f"moment_{index}")
        
        # Handle timestamps - DiDeMo has multiple annotator timestamps
        # Use train_times if available (consensus), otherwise use first annotation
        if 'train_times' in moment_ann and moment_ann['train_times']:
            # train_times is typically a single consensus timestamp
            timestamp = moment_ann['train_times'][0]
        elif 'times' in moment_ann and moment_ann['times']:
            # Use the first annotator's timestamp as fallback
            timestamp = moment_ann['times'][0]
        else:
            # Default to unknown timestamp
            logger.warning(f"No valid timestamp found for annotation_id: {annotation_id}")
            timestamp = [0, 0]  # Default to first 5-second segment
        
        # Convert timestamps from segment indices to seconds
        # DiDeMo uses 5-second segments, so segment [1, 2] means seconds [5, 15]
        start_sec = timestamp[0] * 5
        end_sec = (timestamp[1] + 1) * 5  # +1 because end is inclusive
        
        # Create a unique sample ID
        sample_id = f"{Path(video_filename).stem}_{annotation_id}"
        
        # Construct the full path to the video file
        video_path = Path(self.config['path']) / video_filename
        
        # Create base structure using helper method
        sample = self._get_standardized_base(
            sample_id=sample_id,
            media_path=video_path,
            media_type="video"
        )
        
        # Populate annotations with moment-specific information
        sample['annotations']['moment'] = {
            'timestamp_sec': [start_sec, end_sec],
            'segment_indices': timestamp,  # Original segment indices
            'description': description
        }
        
        # Add additional metadata if available
        if 'reference_description' in moment_ann and moment_ann['reference_description']:
            sample['annotations']['reference_description'] = moment_ann['reference_description']
        
        if 'context' in moment_ann and moment_ann['context']:
            sample['annotations']['context'] = moment_ann['context']
        
        # Include all annotator timestamps for analysis if needed
        if 'times' in moment_ann:
            sample['annotations']['all_annotations'] = [
                {
                    'segment_indices': t,
                    'timestamp_sec': [t[0] * 5, (t[1] + 1) * 5]
                }
                for t in moment_ann['times']
            ]
        
        sample['annotations']['annotation_id'] = annotation_id
        
        return sample