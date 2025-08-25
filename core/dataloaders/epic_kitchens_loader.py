# core/dataloaders/epic_kitchens_loader.py

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class EpicKitchensVisorLoader(BaseLoader):
    """
    A concrete data loader for the EPIC-KITCHENS dataset with VISOR annotations.
    
    This loader is designed to handle the dataset's specific structure, which includes:
    1. A main directory for RGB frames, split by train/val and video ID.
    2. A parallel annotations directory with both sparse ground truth and dense interpolations.
    3. Critical metadata files for class and frame mapping.
    
    It builds an index of all videos and provides a unified sample dictionary
    that merges all available data for a given video.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EpicKitchensVisorLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'image_path': Path to the directory containing train/val subfolders of frames
                - 'sparse_annotation_path': Path to sparse annotation directory
                - 'dense_annotation_path': Path to dense annotation directory
                - 'class_mapping_file': Path to noun classes CSV file
                - 'frame_mapping_file': Path to frame mapping JSON file
        """
        # Validate required config keys
        required_keys = [
            'image_path', 
            'sparse_annotation_path', 
            'dense_annotation_path',
            'class_mapping_file', 
            'frame_mapping_file'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"EpicKitchensVisorLoader config must include '{key}'")
        
        self.image_path = Path(config['image_path'])
        self.sparse_annotation_path = Path(config['sparse_annotation_path'])
        self.dense_annotation_path = Path(config['dense_annotation_path'])
        self.class_mapping_file = Path(config['class_mapping_file'])
        self.frame_mapping_file = Path(config['frame_mapping_file'])
        
        # Validate paths exist
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not self.sparse_annotation_path.exists():
            raise FileNotFoundError(f"Sparse annotation directory not found: {self.sparse_annotation_path}")
        if not self.dense_annotation_path.exists():
            raise FileNotFoundError(f"Dense annotation directory not found: {self.dense_annotation_path}")
        if not self.class_mapping_file.exists():
            raise FileNotFoundError(f"Class mapping file not found: {self.class_mapping_file}")
        if not self.frame_mapping_file.exists():
            raise FileNotFoundError(f"Frame mapping file not found: {self.frame_mapping_file}")
        
        # Initialize metadata dictionaries
        self._class_id_to_name = {}
        self._frame_mapping = {}
        self._video_id_to_split = {}  # Track which split each video belongs to
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[str]:
        """
        Perform a one-time scan to discover all valid video sequences and pre-load
        all essential metadata into memory for efficient access.
        
        Returns:
            List of valid video IDs found in the dataset
        """
        logger.info(f"Building index for EPIC-KITCHENS dataset at {self.image_path}")
        
        # Step 1: Load metadata first
        self._load_class_mapping()
        self._load_frame_mapping()
        
        # Step 2: Discover video IDs from rgb_frames directory
        valid_video_ids = []
        
        for split in ['train', 'val']:
            split_path = self.image_path / split
            if not split_path.exists():
                logger.warning(f"Split directory not found: {split_path}")
                continue
            
            # Scan for video directories
            for video_dir in split_path.iterdir():
                if not video_dir.is_dir():
                    continue
                
                video_id = video_dir.name
                
                # Step 3: Validate that annotations exist
                if self._has_annotations(video_id, split):
                    valid_video_ids.append(video_id)
                    self._video_id_to_split[video_id] = split
                    logger.debug(f"Found valid video: {video_id} in {split} split")
                else:
                    logger.debug(f"Skipping video {video_id}: no annotations found")
        
        logger.info(f"Found {len(valid_video_ids)} valid videos with annotations")
        
        # Sort for consistent ordering
        return sorted(valid_video_ids)

    def _load_class_mapping(self):
        """Load the class mapping from CSV file."""
        try:
            df = pd.read_csv(self.class_mapping_file)
            
            # Assuming columns are 'noun_id' and 'noun' (or similar)
            # Adjust column names based on actual CSV structure
            if 'noun_id' in df.columns and 'noun' in df.columns:
                for _, row in df.iterrows():
                    self._class_id_to_name[int(row['noun_id'])] = str(row['noun'])
            elif 'id' in df.columns and 'name' in df.columns:
                for _, row in df.iterrows():
                    self._class_id_to_name[int(row['id'])] = str(row['name'])
            else:
                # Try to infer from first two columns
                for _, row in df.iterrows():
                    self._class_id_to_name[int(row.iloc[0])] = str(row.iloc[1])
            
            logger.info(f"Loaded {len(self._class_id_to_name)} class mappings")
            
        except Exception as e:
            logger.warning(f"Failed to load class mapping: {e}")
            self._class_id_to_name = {}

    def _load_frame_mapping(self):
        """Load the frame mapping from JSON file."""
        try:
            with open(self.frame_mapping_file, 'r') as f:
                self._frame_mapping = json.load(f)
            
            logger.info(f"Loaded frame mapping with {len(self._frame_mapping)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load frame mapping: {e}")
            self._frame_mapping = {}

    def _has_annotations(self, video_id: str, split: str) -> bool:
        """
        Check if a video has at least one annotation file (sparse or dense).
        
        Args:
            video_id: Video identifier
            split: Train or val split
            
        Returns:
            True if annotations exist for the video
        """
        # Check for sparse annotations
        sparse_file = self.sparse_annotation_path / split / f"{video_id}.json"
        if sparse_file.exists():
            return True
        
        # Check for dense annotations
        dense_file = self.dense_annotation_path / split / f"{video_id}.json"
        if dense_file.exists():
            return True
        
        return False

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve all data for a single video ID, combine the sparse and dense
        annotations, and format everything into the standardized dictionary.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with EPIC-KITCHENS data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Step 1: Retrieve video ID
        video_id = self._index[index]
        
        # Step 2: Determine split
        split = self._video_id_to_split[video_id]
        
        # Step 3: Construct frame directory path
        frame_dir = self.image_path / split / video_id
        
        # Step 4: Create base structure
        sample = self._create_epic_base_structure(video_id, frame_dir)
        
        # Step 5: Load and adapt annotations
        sparse_annotations = self._load_sparse_annotations(video_id, split)
        dense_annotations = self._load_dense_annotations(video_id, split)
        
        # Translate frame references and class IDs
        sparse_annotations = self._translate_annotations(sparse_annotations)
        dense_annotations = self._translate_annotations(dense_annotations)
        
        # Add EPIC-KITCHENS-specific annotations
        sample['annotations'].update({
            'epic_kitchens_visor': {
                'video_id': video_id,
                'split': split,
                'sparse_ground_truth': sparse_annotations,
                'dense_interpolations': dense_annotations,
                'num_sparse_annotations': len(sparse_annotations) if isinstance(sparse_annotations, list) else 0,
                'num_dense_annotations': len(dense_annotations) if isinstance(dense_annotations, list) else 0,
                'has_sparse': sparse_annotations is not None,
                'has_dense': dense_annotations is not None,
                'annotation_statistics': self._calculate_annotation_statistics(
                    sparse_annotations, dense_annotations
                )
            },
            'video_metadata': {
                'frame_directory': str(frame_dir),
                'split': split,
                'num_frames': len(list(frame_dir.glob('*.jpg'))) if frame_dir.exists() else 0
            },
            'dataset_info': {
                'task_type': 'egocentric_video_segmentation',
                'source': 'EPIC-KITCHENS-VISOR',
                'suitable_for_tracking': True,
                'suitable_for_segmentation': True,
                'suitable_for_temporal_reasoning': True,
                'has_sparse_annotations': sparse_annotations is not None,
                'has_dense_annotations': dense_annotations is not None,
                'annotation_format': 'visor_format'
            }
        })
        
        return sample

    def _create_epic_base_structure(self, video_id: str, frame_dir: Path) -> Dict[str, Any]:
        """
        Create the base standardized structure for EPIC-KITCHENS samples.
        
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
            "sample_id": video_id,
            "media_type": "video",
            "media_path": str(frame_dir.resolve()),
            "width": None,   # Video dimensions not extracted by default
            "height": None,  # Video dimensions not extracted by default
            "annotations": {}
        }

    def _load_sparse_annotations(self, video_id: str, split: str) -> Optional[Any]:
        """
        Load sparse ground truth annotations for a video.
        
        Args:
            video_id: Video identifier
            split: Train or val split
            
        Returns:
            Parsed sparse annotations or None if not found
        """
        sparse_file = self.sparse_annotation_path / split / f"{video_id}.json"
        
        if not sparse_file.exists():
            return None
        
        try:
            with open(sparse_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load sparse annotations for {video_id}: {e}")
            return None

    def _load_dense_annotations(self, video_id: str, split: str) -> Optional[Any]:
        """
        Load dense interpolated annotations for a video.
        
        Args:
            video_id: Video identifier
            split: Train or val split
            
        Returns:
            Parsed dense annotations or None if not found
        """
        dense_file = self.dense_annotation_path / split / f"{video_id}.json"
        
        if not dense_file.exists():
            return None
        
        try:
            with open(dense_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load dense annotations for {video_id}: {e}")
            return None

    def _translate_annotations(self, annotations: Any) -> Any:
        """
        Translate frame references and class IDs in annotations using metadata.
        
        This is the critical linking step that ensures all references in the
        annotations are correctly mapped to actual filenames and class names.
        
        Args:
            annotations: Raw annotations to translate
            
        Returns:
            Translated annotations with proper frame filenames and class names
        """
        if annotations is None:
            return None
        
        # Deep copy to avoid modifying original
        if isinstance(annotations, dict):
            translated = {}
            for key, value in annotations.items():
                # Translate frame references
                if 'frame' in key.lower() and isinstance(value, (str, int)):
                    frame_key = str(value)
                    if frame_key in self._frame_mapping:
                        translated[key] = self._frame_mapping[frame_key]
                    else:
                        translated[key] = value
                # Translate class IDs
                elif 'class' in key.lower() and 'id' in key.lower() and isinstance(value, int):
                    if value in self._class_id_to_name:
                        translated[key] = self._class_id_to_name[value]
                        translated[f"{key}_original"] = value  # Keep original ID too
                    else:
                        translated[key] = value
                # Recursively translate nested structures
                elif isinstance(value, (dict, list)):
                    translated[key] = self._translate_annotations(value)
                else:
                    translated[key] = value
            return translated
        
        elif isinstance(annotations, list):
            return [self._translate_annotations(item) for item in annotations]
        
        else:
            return annotations

    def _calculate_annotation_statistics(self, sparse: Any, dense: Any) -> Dict[str, Any]:
        """
        Calculate statistics about the annotations.
        
        Args:
            sparse: Sparse annotations
            dense: Dense annotations
            
        Returns:
            Dictionary with annotation statistics
        """
        stats = {
            'total_annotations': 0,
            'sparse_count': 0,
            'dense_count': 0,
            'annotation_density': 0.0
        }
        
        if sparse is not None:
            if isinstance(sparse, list):
                stats['sparse_count'] = len(sparse)
            elif isinstance(sparse, dict):
                stats['sparse_count'] = 1
        
        if dense is not None:
            if isinstance(dense, list):
                stats['dense_count'] = len(dense)
            elif isinstance(dense, dict):
                stats['dense_count'] = 1
        
        stats['total_annotations'] = stats['sparse_count'] + stats['dense_count']
        
        # Calculate density (dense / sparse ratio)
        if stats['sparse_count'] > 0:
            stats['annotation_density'] = stats['dense_count'] / stats['sparse_count']
        
        return stats

    def get_samples_by_split(self, split: str) -> List[Dict[str, Any]]:
        """
        Get all samples from a specific split (train or val).
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            List of samples from the specified split
        """
        matching_samples = []
        
        for i, video_id in enumerate(self._index):
            if self._video_id_to_split[video_id] == split:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_with_both_annotations(self) -> List[Dict[str, Any]]:
        """
        Get samples that have both sparse and dense annotations.
        
        Returns:
            List of samples with complete annotations
        """
        matching_samples = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            epic_data = sample['annotations']['epic_kitchens_visor']
            
            if epic_data['has_sparse'] and epic_data['has_dense']:
                matching_samples.append(sample)
        
        return matching_samples

    def get_annotation_coverage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about annotation coverage across the dataset.
        
        Returns:
            Dictionary with coverage statistics
        """
        total_videos = len(self._index)
        train_count = sum(1 for vid in self._index if self._video_id_to_split[vid] == 'train')
        val_count = sum(1 for vid in self._index if self._video_id_to_split[vid] == 'val')
        
        # Count annotation types
        sparse_only = 0
        dense_only = 0
        both = 0
        
        for i in range(len(self._index)):
            video_id = self._index[i]
            split = self._video_id_to_split[video_id]
            
            has_sparse = (self.sparse_annotation_path / split / f"{video_id}.json").exists()
            has_dense = (self.dense_annotation_path / split / f"{video_id}.json").exists()
            
            if has_sparse and has_dense:
                both += 1
            elif has_sparse:
                sparse_only += 1
            elif has_dense:
                dense_only += 1
        
        return {
            'total_videos': total_videos,
            'split_distribution': {
                'train': train_count,
                'val': val_count,
                'train_percentage': (train_count / total_videos * 100) if total_videos > 0 else 0,
                'val_percentage': (val_count / total_videos * 100) if total_videos > 0 else 0
            },
            'annotation_coverage': {
                'sparse_only': sparse_only,
                'dense_only': dense_only,
                'both_annotations': both,
                'sparse_coverage_percentage': ((sparse_only + both) / total_videos * 100) if total_videos > 0 else 0,
                'dense_coverage_percentage': ((dense_only + both) / total_videos * 100) if total_videos > 0 else 0,
                'complete_coverage_percentage': (both / total_videos * 100) if total_videos > 0 else 0
            },
            'metadata_statistics': {
                'num_classes': len(self._class_id_to_name),
                'num_frame_mappings': len(self._frame_mapping)
            }
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to their counts
        """
        class_counts = {}
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            epic_data = sample['annotations']['epic_kitchens_visor']
            
            # Count classes in sparse annotations
            if epic_data['sparse_ground_truth']:
                self._count_classes_in_annotations(epic_data['sparse_ground_truth'], class_counts)
            
            # Count classes in dense annotations
            if epic_data['dense_interpolations']:
                self._count_classes_in_annotations(epic_data['dense_interpolations'], class_counts)
        
        return class_counts

    def _count_classes_in_annotations(self, annotations: Any, class_counts: Dict[str, int]):
        """
        Recursively count class occurrences in annotations.
        
        Args:
            annotations: Annotations to search for classes
            class_counts: Dictionary to update with counts
        """
        if isinstance(annotations, dict):
            for key, value in annotations.items():
                if 'class' in key.lower() and isinstance(value, str):
                    class_counts[value] = class_counts.get(value, 0) + 1
                elif isinstance(value, (dict, list)):
                    self._count_classes_in_annotations(value, class_counts)
        elif isinstance(annotations, list):
            for item in annotations:
                self._count_classes_in_annotations(item, class_counts)

    def list_available_videos(self) -> List[str]:
        """
        Get a list of all available video IDs.
        
        Returns:
            List of video IDs
        """
        return list(self._index)

    def get_video_by_id(self, video_id: str) -> Dict[str, Any]:
        """
        Get a specific video sample by its ID.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Sample dictionary for the specified video
            
        Raises:
            ValueError: If video ID is not found
        """
        try:
            index = self._index.index(video_id)
            return self.get_item(index)
        except ValueError:
            raise ValueError(f"Video ID '{video_id}' not found in dataset")