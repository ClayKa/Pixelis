# core/dataloaders/msrvtt_loader.py

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MsrVttLoader(BaseLoader):
    """
    A concrete data loader for the MSR-VTT (Microsoft Research Video to Text) dataset.
    
    This loader handles MSR-VTT's multi-file structure which involves:
    1. A main annotation file (msrvtt_train_9k.json) containing video info and timestamps
    2. A category mapping file (category.txt) with category names
    3. A raw captions file (raw-captions.pkl) with additional captions
    4. A directory of video files (.mp4)
    
    The loader builds a unified index by pre-processing and merging these sources,
    optimized for video captioning and temporal reasoning tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MsrVttLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'path': Path to video directory
                - 'annotation_file': Path to main annotation JSON
                - 'category_file': Path to category mapping file
                - 'raw_captions_file': Path to raw captions pickle file (optional)
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_file', 'category_file']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"MsrVttLoader config must include '{key}'")
        
        self.videos_path = Path(config['path'])
        self.annotation_file = Path(config['annotation_file'])
        self.category_file = Path(config['category_file'])
        self.raw_captions_file = Path(config['raw_captions_file']) if 'raw_captions_file' in config else None
        
        # Validate paths exist
        if not self.videos_path.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_path}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.category_file.exists():
            raise FileNotFoundError(f"Category file not found: {self.category_file}")
        if self.raw_captions_file and not self.raw_captions_file.exists():
            raise FileNotFoundError(f"Raw captions file not found: {self.raw_captions_file}")
        
        # Initialize lookup structures (populated in _build_index)
        self._category_id_to_name = {}
        self._video_id_to_raw_captions = {}
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Load and merge all annotation files into a unified index.
        
        Returns:
            List of enriched video dictionaries with all metadata
        """
        logger.info(f"Loading MSR-VTT annotations from {self.annotation_file}")
        
        # Step 1: Load category mapping
        self._load_category_mapping()
        logger.info(f"Loaded {len(self._category_id_to_name)} categories")
        
        # Step 2: Load raw captions (optional)
        if self.raw_captions_file:
            self._load_raw_captions()
            logger.info(f"Loaded raw captions for {len(self._video_id_to_raw_captions)} videos")
        
        # Step 3: Load main annotations and build enriched index
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        enriched_index = []
        existing_videos = 0
        
        for video_info in video_data:
            # Check if video file exists on disk
            video_path = self.videos_path / video_info['video']
            if not video_path.exists():
                continue
                
            # Create enriched video dictionary
            enriched_video = video_info.copy()
            
            # Add human-readable category name
            category_id = video_info.get('category', -1)
            enriched_video['category_name'] = self._category_id_to_name.get(
                category_id, f'category_{category_id}'
            )
            
            # Add raw captions if available
            video_id = video_info['video_id']
            if video_id in self._video_id_to_raw_captions:
                enriched_video['raw_captions'] = self._video_id_to_raw_captions[video_id]
            else:
                enriched_video['raw_captions'] = []
            
            # Calculate video duration
            start_time = video_info.get('start time', 0.0)
            end_time = video_info.get('end time', 0.0)
            enriched_video['duration'] = end_time - start_time if end_time > start_time else 0.0
            
            enriched_index.append(enriched_video)
            existing_videos += 1
        
        logger.info(f"Built index with {len(enriched_index)} videos")
        logger.info(f"Found {existing_videos} videos existing on disk")
        
        return enriched_index

    def _load_category_mapping(self):
        """Load category ID to name mapping from category file."""
        with open(self.category_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    category_name, category_id = parts
                    self._category_id_to_name[int(category_id)] = category_name

    def _load_raw_captions(self):
        """Load raw captions from pickle file (optional)."""
        try:
            with open(self.raw_captions_file, 'rb') as f:
                raw_captions_data = pickle.load(f)
                
            # Handle different possible formats of the pickle file
            if isinstance(raw_captions_data, dict):
                self._video_id_to_raw_captions = raw_captions_data
            elif isinstance(raw_captions_data, list):
                # If it's a list, try to map it to video IDs
                for i, captions in enumerate(raw_captions_data):
                    video_id = f"video{i}"
                    self._video_id_to_raw_captions[video_id] = captions
            else:
                logger.warning(f"Unknown format for raw captions file: {type(raw_captions_data)}")
                
        except Exception as e:
            logger.warning(f"Failed to load raw captions: {e}")
            self._video_id_to_raw_captions = {}

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single video sample with all associated metadata.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with MSR-VTT video captioning data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get enriched video info from pre-built index
        video_info = self._index[index]
        
        # Construct video path
        video_path = self.videos_path / video_info['video']
        
        # Create base standardized structure
        sample = self._get_standardized_base(
            sample_id=video_info['video_id'],
            media_path=video_path,
            media_type="video"
        )
        
        # Process captions - MSR-VTT has multiple captions per video
        captions = video_info.get('caption', [])
        raw_captions = video_info.get('raw_captions', [])
        all_captions = captions + raw_captions
        
        # Calculate caption statistics
        caption_stats = self._analyze_captions(all_captions)
        
        # Add MSR-VTT-specific annotations
        sample['annotations'].update({
            'msrvtt_video_captioning': {
                'video_id': video_info['video_id'],
                'primary_captions': captions,
                'raw_captions': raw_captions,
                'all_captions': all_captions,
                'num_captions': len(all_captions),
                'caption_statistics': caption_stats,
                'temporal_info': {
                    'start_time': video_info.get('start time', 0.0),
                    'end_time': video_info.get('end time', 0.0),
                    'duration': video_info.get('duration', 0.0)
                },
                'category_info': {
                    'category_id': video_info.get('category', -1),
                    'category_name': video_info.get('category_name', 'unknown')
                },
                'source_info': {
                    'source': video_info.get('source', 'MSR-VTT'),
                    'original_url': video_info.get('url'),
                    'dataset_id': video_info.get('id', -1)
                }
            },
            'video_metadata': {
                'video_filename': video_info['video'],
                'duration_seconds': video_info.get('duration', 0.0),
                'temporal_bounds': [
                    video_info.get('start time', 0.0),
                    video_info.get('end time', 0.0)
                ]
            },
            'dataset_info': {
                'task_type': 'video_captioning',
                'source': 'MSR-VTT',
                'suitable_for_select_frame': True,
                'suitable_for_temporal_reasoning': True,
                'has_multiple_captions': len(all_captions) > 1,
                'has_temporal_bounds': video_info.get('duration', 0.0) > 0,
                'has_category_labels': True,
                'num_categories': len(self._category_id_to_name),
                'video_format': 'mp4'
            }
        })
        
        return sample

    def _analyze_captions(self, captions: List[str]) -> Dict[str, Any]:
        """
        Analyze caption statistics for diversity and complexity metrics.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Dictionary with caption analysis metrics
        """
        if not captions:
            return {
                'avg_length': 0.0,
                'min_length': 0,
                'max_length': 0,
                'total_words': 0,
                'unique_words': 0,
                'avg_words_per_caption': 0.0
            }
        
        lengths = [len(caption) for caption in captions]
        word_counts = [len(caption.split()) for caption in captions]
        all_words = ' '.join(captions).lower().split()
        unique_words = set(all_words)
        
        return {
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'avg_words_per_caption': sum(word_counts) / len(word_counts),
            'vocabulary_diversity': len(unique_words) / len(all_words) if all_words else 0.0
        }

    def get_samples_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        """
        Get samples from a specific category.
        
        Args:
            category_name: Name of the category (e.g., 'music', 'sports/actions')
            
        Returns:
            List of samples from the specified category
        """
        matching_samples = []
        
        for i, video_info in enumerate(self._index):
            if video_info.get('category_name') == category_name:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about category distribution in the dataset.
        
        Returns:
            Dictionary with category statistics
        """
        category_counts = {}
        for video_info in self._index:
            category = video_info.get('category_name', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_categories': len(self._category_id_to_name),
            'total_videos': len(self._index),
            'category_distribution': category_counts,
            'most_common_categories': sorted(
                category_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            'available_categories': list(self._category_id_to_name.values())
        }

    def get_samples_by_duration(self, min_duration: float = 0.0, max_duration: float = float('inf')) -> List[Dict[str, Any]]:
        """
        Get samples filtered by video duration.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            List of samples within the duration range
        """
        matching_samples = []
        
        for i, video_info in enumerate(self._index):
            duration = video_info.get('duration', 0.0)
            if min_duration <= duration <= max_duration:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_caption_diversity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about caption diversity and complexity.
        
        Returns:
            Dictionary with caption analysis statistics
        """
        all_caption_stats = []
        total_captions = 0
        total_unique_words = set()
        
        for video_info in self._index:
            captions = video_info.get('caption', []) + video_info.get('raw_captions', [])
            if captions:
                stats = self._analyze_captions(captions)
                all_caption_stats.append(stats)
                total_captions += len(captions)
                
                # Collect unique words across all captions
                words = ' '.join(captions).lower().split()
                total_unique_words.update(words)
        
        if not all_caption_stats:
            return {'error': 'No captions found'}
        
        # Calculate aggregate statistics
        avg_lengths = [stats['avg_length'] for stats in all_caption_stats]
        avg_words = [stats['avg_words_per_caption'] for stats in all_caption_stats]
        vocab_diversities = [stats['vocabulary_diversity'] for stats in all_caption_stats]
        
        return {
            'total_videos_with_captions': len(all_caption_stats),
            'total_captions': total_captions,
            'total_unique_words': len(total_unique_words),
            'avg_captions_per_video': total_captions / len(all_caption_stats),
            'caption_length_statistics': {
                'avg_length': sum(avg_lengths) / len(avg_lengths),
                'min_avg_length': min(avg_lengths),
                'max_avg_length': max(avg_lengths)
            },
            'word_statistics': {
                'avg_words_per_caption': sum(avg_words) / len(avg_words),
                'vocabulary_diversity': sum(vocab_diversities) / len(vocab_diversities)
            }
        }

    def get_temporal_distribution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about temporal properties of the videos.
        
        Returns:
            Dictionary with temporal statistics
        """
        durations = []
        start_times = []
        end_times = []
        
        for video_info in self._index:
            duration = video_info.get('duration', 0.0)
            start_time = video_info.get('start time', 0.0)
            end_time = video_info.get('end time', 0.0)
            
            if duration > 0:
                durations.append(duration)
            if start_time > 0:
                start_times.append(start_time)
            if end_time > 0:
                end_times.append(end_time)
        
        return {
            'total_videos': len(self._index),
            'videos_with_duration': len(durations),
            'duration_statistics': {
                'avg_duration': sum(durations) / len(durations) if durations else 0.0,
                'min_duration': min(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0.0,
                'median_duration': sorted(durations)[len(durations) // 2] if durations else 0.0
            },
            'temporal_bounds': {
                'avg_start_time': sum(start_times) / len(start_times) if start_times else 0.0,
                'avg_end_time': sum(end_times) / len(end_times) if end_times else 0.0
            }
        }