# core/dataloaders/robust_activitynet_loader.py

import logging
from pathlib import Path
from typing import Any, Dict, List

from .robust_base_loader import RobustBaseLoader, DataLoadError

logger = logging.getLogger(__name__)


class RobustActivityNetCaptionsLoader(RobustBaseLoader):
    """
    Robust version of ActivityNetCaptionsLoader with comprehensive error handling.
    
    This loader can gracefully handle:
    - Corrupted JSON files
    - Missing video files
    - Malformed annotations
    - Encoding issues
    """
    
    def _build_index(self) -> List[str]:
        """
        Build index with robust error handling and validation.
        """
        # Validate configuration
        if 'annotation_file' not in self.config:
            raise ValueError(f"ActivityNetCaptionsLoader requires 'annotation_file' in config")
        
        if 'path' not in self.config:
            raise ValueError(f"ActivityNetCaptionsLoader requires 'path' (video directory) in config")
        
        annotation_file_path = Path(self.config['annotation_file'])
        video_dir_path = Path(self.config['path'])
        
        if not video_dir_path.is_dir():
            raise DataLoadError(
                f"Video directory not found or not accessible",
                file_path=video_dir_path
            )
        
        # Load annotations with comprehensive error handling
        logger.info(f"Loading annotations from {annotation_file_path}")
        
        try:
            annotations_list = self._load_json_safe(
                annotation_file_path,
                required_fields=None  # ActivityNet uses array format, not object
            )
            
            # Validate that it's a list
            if not isinstance(annotations_list, list):
                raise DataLoadError(
                    f"Expected JSON array but got {type(annotations_list).__name__}",
                    file_path=annotation_file_path
                )
            
        except DataLoadError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise DataLoadError(
                f"Failed to load annotation file",
                file_path=annotation_file_path,
                original_error=e
            )
        
        logger.info(f"Loaded {len(annotations_list)} annotations from JSON")
        
        # Convert to dictionary for O(1) lookup
        self._annotations_map = {}
        malformed_annotations = []
        
        for idx, ann in enumerate(annotations_list):
            try:
                # Validate annotation structure
                if not isinstance(ann, dict):
                    malformed_annotations.append(f"Index {idx}: Not a dictionary")
                    continue
                
                if 'video_id' not in ann:
                    malformed_annotations.append(f"Index {idx}: Missing 'video_id'")
                    continue
                
                video_id = ann['video_id']
                
                # Validate critical fields
                if 'timestamps' not in ann or 'sentences' not in ann:
                    self._validation_errors.append({
                        'type': 'incomplete_annotation',
                        'video_id': video_id,
                        'missing': [
                            k for k in ['timestamps', 'sentences'] 
                            if k not in ann
                        ]
                    })
                    if not self.skip_on_error:
                        continue
                
                # Check for data consistency
                if len(ann.get('timestamps', [])) != len(ann.get('sentences', [])):
                    self._validation_errors.append({
                        'type': 'mismatched_timestamps_sentences',
                        'video_id': video_id,
                        'num_timestamps': len(ann.get('timestamps', [])),
                        'num_sentences': len(ann.get('sentences', []))
                    })
                
                self._annotations_map[video_id] = ann
                
            except Exception as e:
                logger.warning(f"Error processing annotation at index {idx}: {e}")
                if not self.skip_on_error:
                    malformed_annotations.append(f"Index {idx}: {str(e)}")
        
        if malformed_annotations:
            logger.warning(
                f"Found {len(malformed_annotations)} malformed annotations. "
                f"First 5: {malformed_annotations[:5]}"
            )
        
        # Build video file index for fast validation
        logger.info("Validating video files...")
        self._index = []
        missing_videos = []
        corrupted_videos = []
        
        for video_id, ann in self._annotations_map.items():
            try:
                # Get expected video filename
                video_filename = ann.get('video', f"{video_id}.mp4")
                video_path = video_dir_path / video_filename
                
                # Try alternative extensions if primary doesn't exist
                if not video_path.exists():
                    for ext in ['.mkv', '.avi', '.mov', '.webm']:
                        alt_path = video_path.with_suffix(ext)
                        if alt_path.exists():
                            video_path = alt_path
                            ann['video'] = alt_path.name
                            break
                
                # Validate the video file
                if self._validate_media_file(video_path, 'video', video_id):
                    self._index.append(video_id)
                else:
                    if video_path.exists():
                        corrupted_videos.append(video_id)
                    else:
                        missing_videos.append(video_id)
                        
            except Exception as e:
                logger.warning(f"Error validating video for {video_id}: {e}")
                self._corrupted_samples.append(video_id)
        
        # Report validation results
        if missing_videos:
            logger.warning(
                f"Found {len(missing_videos)} videos in annotations but not on disk. "
                f"First 10: {missing_videos[:10]}"
            )
        
        if corrupted_videos:
            logger.warning(
                f"Found {len(corrupted_videos)} corrupted or inaccessible videos. "
                f"First 10: {corrupted_videos[:10]}"
            )
        
        logger.info(
            f"Index built with {len(self._index)} valid samples "
            f"(out of {len(self._annotations_map)} annotated videos)"
        )
        
        return self._index
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get a single item with comprehensive error handling.
        """
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range for dataset with {len(self._index)} samples")
        
        # Get video ID from index
        video_id = self._index[index]
        
        try:
            # Get annotation data
            raw_ann = self._annotations_map[video_id]
            
            # Construct video path
            video_filename = raw_ann.get('video', f"{video_id}.mp4")
            video_path = Path(self.config['path']) / video_filename
            
            # Create base structure with error handling
            sample = self._get_standardized_base_safe(
                sample_id=video_id,
                media_path=video_path,
                media_type="video"
            )
            
            # Build events with validation
            events = []
            timestamps = raw_ann.get('timestamps', [])
            sentences = raw_ann.get('sentences', [])
            
            # Handle mismatched lengths gracefully
            min_length = min(len(timestamps), len(sentences))
            if len(timestamps) != len(sentences):
                logger.debug(
                    f"Timestamp/sentence mismatch for {video_id}: "
                    f"{len(timestamps)} timestamps, {len(sentences)} sentences. "
                    f"Using first {min_length} pairs."
                )
            
            for i in range(min_length):
                try:
                    # Validate timestamp format
                    ts = timestamps[i]
                    if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                        logger.warning(f"Invalid timestamp format at index {i} for {video_id}: {ts}")
                        continue
                    
                    # Validate timestamp values
                    start, end = ts[0], ts[1]
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        logger.warning(f"Non-numeric timestamp at index {i} for {video_id}: {ts}")
                        continue
                    
                    if start < 0 or end < start:
                        logger.warning(f"Invalid timestamp range at index {i} for {video_id}: {ts}")
                        continue
                    
                    events.append({
                        'timestamp_sec': [float(start), float(end)],
                        'description': str(sentences[i])  # Ensure it's a string
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing event {i} for {video_id}: {e}")
                    if not self.skip_on_error:
                        raise
            
            # Add annotations to sample
            sample['annotations']['duration_sec'] = raw_ann.get('duration', None)
            sample['annotations']['timed_events'] = events
            sample['annotations']['source'] = raw_ann.get('source', 'ActivityNet_Captions')
            
            if 'caption' in raw_ann:
                sample['annotations']['full_caption'] = str(raw_ann['caption'])
            
            return sample
            
        except Exception as e:
            raise DataLoadError(
                f"Failed to load sample",
                sample_id=video_id,
                original_error=e
            )