# core/dataloaders/activitynet_captions_loader.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ActivityNetCaptionsLoader(BaseLoader):
    """
    A concrete data loader for the ActivityNet Captions dataset.

    This loader is responsible for:
    1. Parsing the main annotation JSON file, which maps video IDs to a list of
       timestamped sentence captions.
    2. Building an index of all video IDs for which both annotations and a
       video file exist.
    3. Providing standardized samples that associate a video with its dense
       event descriptions.
    """

    def _build_index(self) -> List[str]:
        """
        Load the main annotation JSON file once, validate its entries against
        the locally available video files, and build a lightweight index of valid video IDs.
        
        Returns:
            List[str]: A list of unique video IDs for which both annotations and video files exist.
        """
        # Get the annotation file path from config
        if 'annotation_file' not in self.config:
            raise ValueError(f"ActivityNetCaptionsLoader requires 'annotation_file' in config")
        
        annotation_file_path = Path(self.config['annotation_file'])
        
        if not annotation_file_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file_path}"
            )
        
        # Load the annotation JSON file
        logger.info(f"Loading annotations from {annotation_file_path}")
        with open(annotation_file_path, 'r') as f:
            annotations_list = json.load(f)
        
        # Convert list to dictionary for O(1) lookup
        # Store annotations in internal map for later retrieval
        self._annotations_map = {}
        for ann in annotations_list:
            video_id = ann['video_id']
            self._annotations_map[video_id] = ann
        
        # Get the video directory path
        if 'path' not in self.config:
            raise ValueError(f"ActivityNetCaptionsLoader requires 'path' (video directory) in config")
        
        video_dir_path = Path(self.config['path'])
        
        if not video_dir_path.is_dir():
            raise FileNotFoundError(
                f"Video directory not found: {video_dir_path}"
            )
        
        # Validate against videos and build index
        self._index = []
        missing_videos = []
        
        for video_id in self._annotations_map.keys():
            # Get the expected video filename from annotation
            video_filename = self._annotations_map[video_id].get('video', f"{video_id}.mp4")
            
            # Construct full path to video file
            video_file_path = video_dir_path / video_filename
            
            # Check for existence - also try .mkv extension if .mp4 not found
            if video_file_path.is_file():
                self._index.append(video_id)
            elif video_file_path.with_suffix('.mkv').is_file():
                # Update the stored video filename for later use
                self._annotations_map[video_id]['video'] = video_filename.replace('.mp4', '.mkv')
                self._index.append(video_id)
            else:
                missing_videos.append(video_id)
        
        if missing_videos:
            logger.warning(
                f"Found {len(missing_videos)} videos in annotations but not in video directory. "
                f"First 10 missing: {missing_videos[:10]}"
            )
        
        logger.info(
            f"Built index with {len(self._index)} valid samples "
            f"(out of {len(self._annotations_map)} annotated videos)"
        )
        
        return self._index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single video and its complete set of timestamped captions
        and format it into the project's standardized dictionary.
        
        Args:
            index (int): The integer index of the sample in the self._index list.
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary with video path and
                          timestamped event descriptions.
        """
        # Retrieve video ID from the index
        video_id = self._index[index]
        
        # Get the raw annotation data for this video
        raw_ann = self._annotations_map[video_id]
        
        # Construct the full path to the video file
        video_filename = raw_ann.get('video', f"{video_id}.mp4")
        video_path = Path(self.config['path']) / video_filename
        
        # Create base structure using helper method
        sample = self._get_standardized_base(
            sample_id=video_id,
            media_path=video_path,
            media_type="video"
        )
        
        # Combine timestamps and sentences into a list of event dictionaries
        events = []
        timestamps = raw_ann.get('timestamps', [])
        sentences = raw_ann.get('sentences', [])
        
        # Ensure we have matching timestamps and sentences
        if len(timestamps) != len(sentences):
            logger.warning(
                f"Mismatch between timestamps ({len(timestamps)}) and sentences ({len(sentences)}) "
                f"for video_id: {video_id}"
            )
            # Use the minimum length to avoid index errors
            min_length = min(len(timestamps), len(sentences))
            timestamps = timestamps[:min_length]
            sentences = sentences[:min_length]
        
        for i, (timestamp, sentence) in enumerate(zip(timestamps, sentences)):
            events.append({
                'timestamp_sec': timestamp,  # [start, end] pair
                'description': sentence
            })
        
        # Add annotations to the sample
        sample['annotations']['duration_sec'] = raw_ann.get('duration', None)
        sample['annotations']['timed_events'] = events
        sample['annotations']['source'] = raw_ann.get('source', 'ActivityNet_Captions')
        
        # Optionally add the full concatenated caption if it exists
        if 'caption' in raw_ann:
            sample['annotations']['full_caption'] = raw_ann['caption']
        
        return sample