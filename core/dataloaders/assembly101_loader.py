# core/dataloaders/assembly101_loader.py

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Assembly101Loader(BaseLoader):
    """
    [REVISED]
    A concrete data loader for the Assembly101 dataset.

    This loader is designed to work with the official annotation structure, which
    separates action event logs (train.csv, etc.) from action class definitions
    (actions.csv). It first loads the action metadata as a lookup table, then
    builds an index from a specific split file (e.g., train.csv), ensuring that
    each sample is enriched with human-readable labels.
    
    Assembly101 is a large-scale multi-view video dataset for learning and benchmarking
    a diverse set of procedural activities, focused on toy assembly tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Assembly101Loader with action metadata support."""
        # Initialize lookup dictionaries
        self._verb_id_to_cls = {}
        self._noun_id_to_cls = {}
        
        super().__init__(config)

    def _build_index(self) -> List[Dict]:
        """
        [REVISED]
        First load and process the actions.csv metadata, then load a specific split file 
        (e.g., train.csv), validate its entries against available video data, and build 
        an index of valid action segments.
        
        Returns:
            List[Dict]: A list of annotation dictionaries for valid action segments.
        """
        # Step 1: Load and Process Action Metadata (if available)
        if 'action_metadata_file' in self.config:
            metadata_path = Path(self.config['action_metadata_file'])
            if metadata_path.is_file():
                logger.info(f"Loading action metadata from {metadata_path}")
                actions_df = pd.read_csv(metadata_path)
                
                # Create internal lookup dictionaries for efficient translation
                # Create a verb ID -> verb class name mapping
                if 'verb_id' in actions_df.columns and 'verb_cls' in actions_df.columns:
                    temp_dict = pd.Series(
                        actions_df.verb_cls.values, 
                        index=actions_df.verb_id
                    ).to_dict()
                    # Store both string and integer versions of keys for flexibility
                    self._verb_id_to_cls = {}
                    for k, v in temp_dict.items():
                        self._verb_id_to_cls[k] = v  # Original key (might be int)
                        self._verb_id_to_cls[str(k)] = v  # String version
                        # Also handle zero-padded versions if k is an integer
                        if isinstance(k, (int, float)):
                            # Store common zero-padded versions
                            self._verb_id_to_cls[f"{int(k):04d}"] = v  # 4-digit padding
                    logger.info(f"Loaded {len(temp_dict)} verb mappings")
                
                # Create a noun ID -> noun class name mapping  
                if 'noun_id' in actions_df.columns and 'noun_cls' in actions_df.columns:
                    temp_dict = pd.Series(
                        actions_df.noun_cls.values,
                        index=actions_df.noun_id
                    ).to_dict()
                    # Store both string and integer versions of keys for flexibility
                    self._noun_id_to_cls = {}
                    for k, v in temp_dict.items():
                        self._noun_id_to_cls[k] = v  # Original key (might be int)
                        self._noun_id_to_cls[str(k)] = v  # String version
                        # Also handle zero-padded versions if k is an integer
                        if isinstance(k, (int, float)):
                            # Store common zero-padded versions
                            self._noun_id_to_cls[f"{int(k):04d}"] = v  # 4-digit padding
                    logger.info(f"Loaded {len(temp_dict)} noun mappings")
            else:
                logger.warning(f"Action metadata file not found: {metadata_path}")
        
        # Step 2: Load Split Annotations
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
        
        # Step 3: Validate Against Videos and Build Index
        # Load annotations with pandas
        logger.info(f"Loading split annotations from {annotation_file_path}")
        split_df = pd.read_csv(annotation_file_path)
        logger.info(f"Loaded {len(split_df)} action segments from CSV")
        
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
        unique_videos = split_df['video'].unique()
        valid_videos = set()
        
        for video_path in unique_videos:
            if video_path in available_videos:
                valid_videos.add(video_path)
            else:
                missing_videos.add(video_path)
        
        # Now filter the dataframe to only include rows with valid videos
        valid_df = split_df[split_df['video'].isin(valid_videos)]
        
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
            f"(out of {len(split_df)} total annotations)"
        )
        
        # Step 4: Return the Index
        return self._index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        [REVISED]
        Retrieve a single action segment annotation, use the pre-built lookup tables to 
        translate its numeric IDs into text, and format everything into the project's 
        standardized dictionary.
        
        Args:
            index (int): The integer index of the sample in the self._index list.
            
        Returns:
            Dict[str, Any]: A standardized sample dictionary with video path and
                          enriched action segment information.
        """
        # Step 1: Retrieve annotation row from the index
        ann_row = self._index[index]
        
        # Step 2: Extract key information
        video_filename = ann_row['video']
        start_frame = ann_row['start_frame']
        end_frame = ann_row['end_frame']
        
        # Handle start/end frames that might be strings with leading zeros
        if isinstance(start_frame, str):
            start_frame = int(start_frame)
        if isinstance(end_frame, str):
            end_frame = int(end_frame)
        
        # Step 3: Create a unique sample ID
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
        
        # Step 4: [REVISED] Adapt and Populate Annotations with Translation
        # Perform Translation: Use the lookup dictionaries created in _build_index to get human-readable labels
        verb_text = None
        noun_text = None
        
        if 'verb_id' in ann_row and ann_row['verb_id'] is not None:
            verb_id = ann_row['verb_id']
            verb_text = self._verb_id_to_cls.get(verb_id, 'unknown_verb')
        
        if 'noun_id' in ann_row and ann_row['noun_id'] is not None:
            noun_id = ann_row['noun_id']
            noun_text = self._noun_id_to_cls.get(noun_id, 'unknown_noun')
        
        # Populate the standardized sample with rich annotation dictionary
        sample['annotations']['action_segment'] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': end_frame - start_frame + 1
        }
        
        # Add verb and noun IDs if available
        if 'verb_id' in ann_row and ann_row['verb_id'] is not None:
            sample['annotations']['action_segment']['verb_id'] = ann_row['verb_id']
            sample['annotations']['action_segment']['verb_class'] = verb_text
        
        if 'noun_id' in ann_row and ann_row['noun_id'] is not None:
            sample['annotations']['action_segment']['noun_id'] = ann_row['noun_id']
            sample['annotations']['action_segment']['noun_class'] = noun_text
        
        # Create full description if both verb and noun are available
        if verb_text and noun_text:
            sample['annotations']['action_segment']['full_description'] = f"{verb_text} {noun_text}"
        
        # Add raw action class if available (for backwards compatibility)
        if 'action_cls' in ann_row:
            sample['annotations']['action_segment']['action'] = ann_row['action_cls']
        
        # If verb_cls and noun_cls are already in the CSV (pre-translated), use them
        if 'verb_cls' in ann_row and ann_row['verb_cls']:
            sample['annotations']['action_segment']['verb_class'] = ann_row['verb_cls']
        
        if 'noun_cls' in ann_row and ann_row['noun_cls']:
            sample['annotations']['action_segment']['noun_class'] = ann_row['noun_cls']
        
        # Add action ID if available
        if 'action_id' in ann_row:
            sample['annotations']['action_segment']['action_id'] = ann_row['action_id']
        
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