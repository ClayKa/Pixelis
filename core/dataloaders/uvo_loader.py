# core/dataloaders/uvo_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class UvoLoader(BaseLoader):
    """
    A concrete data loader for the UVO (Unidentified Video Objects) dataset.
    
    This loader is designed to navigate UVO's rich and complex annotation structure,
    which is split across multiple directories for different annotation types
    (e.g., masks, expressions, relationships) and data splits (dense vs. sparse).
    
    It builds a unified index that allows flexible retrieval and combination of these
    disparate data sources into a single, cohesive sample.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UvoLoader.
        
        Args:
            config: Configuration dictionary containing:
                - 'path': Path to the directory containing video files/frames
                - 'annotation_path': Path to the top-level annotation directory
        """
        # Validate required config keys
        required_keys = ['path', 'annotation_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"UvoLoader config must include '{key}'")
        
        self.video_path = Path(config['path'])
        self.annotation_path = Path(config['annotation_path'])
        
        # Validate paths exist
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_path}")
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_path}")
        
        # Initialize lookup structures for different annotation types
        # These will be populated in _build_index()
        self._video_id_to_dense_mask_path = {}
        self._video_id_to_sparse_mask_path = {}
        self._video_id_to_expression_path = {}
        self._video_id_to_relationship_path = {}
        self._video_id_to_box_annotation_path = {}
        self._video_id_to_additional_annotations = defaultdict(dict)
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[str]:
        """
        Scan the primary data directories (dense and/or sparse), discover all unique
        video IDs, and build comprehensive lookup tables for all annotation types.
        
        Returns:
            List of unique video IDs found in the dataset
        """
        logger.info(f"Scanning UVO dataset at {self.video_path}")
        
        # Step 1: Discover video IDs from the primary video directory
        video_ids = set()
        
        # Scan for video files or frame directories
        for item in self.video_path.iterdir():
            if item.is_file():
                # Video files (e.g., .mp4, .avi)
                if item.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    video_id = item.stem
                    video_ids.add(video_id)
            elif item.is_dir():
                # Frame directories
                video_id = item.name
                video_ids.add(video_id)
        
        logger.info(f"Found {len(video_ids)} video IDs from video directory")
        
        # Step 2: Investigate and index all annotation types
        self._index_annotation_files(video_ids)
        
        # Convert to sorted list for consistent ordering
        return sorted(list(video_ids))

    def _index_annotation_files(self, video_ids: set):
        """
        Scan the annotation directory and build lookup tables for all annotation types.
        
        Args:
            video_ids: Set of video IDs to look for annotations for
        """
        logger.info(f"Indexing annotation files in {self.annotation_path}")
        
        # Scan all subdirectories in the annotation path
        for annotation_dir in self.annotation_path.iterdir():
            if not annotation_dir.is_dir():
                continue
                
            dir_name = annotation_dir.name.lower()
            logger.debug(f"Processing annotation directory: {dir_name}")
            
            # Handle different annotation types based on directory names
            if 'dense' in dir_name and 'mask' in dir_name or dir_name == 'uvo-dense':
                self._index_mask_files(annotation_dir, self._video_id_to_dense_mask_path, 'dense')
            elif 'sparse' in dir_name and 'mask' in dir_name or dir_name == 'uvo-sparse':
                self._index_mask_files(annotation_dir, self._video_id_to_sparse_mask_path, 'sparse')
            elif 'expression' in dir_name:
                self._index_json_files(annotation_dir, self._video_id_to_expression_path, 'expressions')
            elif 'rel' in dir_name or 'relationship' in dir_name:
                self._index_json_files(annotation_dir, self._video_id_to_relationship_path, 'relationships')
            elif 'box' in dir_name:
                self._index_json_files(annotation_dir, self._video_id_to_box_annotation_path, 'box_annotations')
            else:
                # Handle other annotation types generically
                self._index_additional_annotations(annotation_dir, dir_name)

    def _index_mask_files(self, annotation_dir: Path, lookup_dict: Dict[str, Path], mask_type: str):
        """
        Index mask annotation files (for dense/sparse masks).
        
        Args:
            annotation_dir: Directory containing mask annotations
            lookup_dict: Dictionary to store video_id -> path mappings
            mask_type: Type of mask ('dense' or 'sparse')
        """
        # Look for JSON files or subdirectories containing mask data
        for item in annotation_dir.rglob('*'):
            if item.is_file() and item.suffix == '.json':
                # Extract video ID from filename or parent directory
                video_id = self._extract_video_id(item)
                if video_id:
                    lookup_dict[video_id] = item
                    logger.debug(f"Found {mask_type} mask for {video_id}: {item}")
            elif item.is_dir():
                # Directory-based masks (e.g., one directory per video with PNG files)
                video_id = item.name
                if video_id and (item / 'masks').exists() or list(item.glob('*.png')):
                    lookup_dict[video_id] = item
                    logger.debug(f"Found {mask_type} mask directory for {video_id}: {item}")

    def _index_json_files(self, annotation_dir: Path, lookup_dict: Dict[str, Path], annotation_type: str):
        """
        Index JSON annotation files.
        
        Args:
            annotation_dir: Directory containing JSON annotations
            lookup_dict: Dictionary to store video_id -> path mappings
            annotation_type: Type of annotation
        """
        for json_file in annotation_dir.rglob('*.json'):
            video_id = self._extract_video_id(json_file)
            if video_id:
                lookup_dict[video_id] = json_file
                logger.debug(f"Found {annotation_type} for {video_id}: {json_file}")

    def _index_additional_annotations(self, annotation_dir: Path, dir_name: str):
        """
        Index additional annotation types generically.
        
        Args:
            annotation_dir: Directory containing annotations
            dir_name: Name of the annotation directory
        """
        for item in annotation_dir.rglob('*'):
            if item.is_file() and item.suffix in ['.json', '.txt', '.xml']:
                video_id = self._extract_video_id(item)
                if video_id:
                    self._video_id_to_additional_annotations[video_id][dir_name] = item
                    logger.debug(f"Found {dir_name} annotation for {video_id}: {item}")

    def _extract_video_id(self, file_path: Path) -> Optional[str]:
        """
        Extract video ID from a file path.
        
        This method handles various naming conventions used in UVO dataset.
        
        Args:
            file_path: Path to extract video ID from
            
        Returns:
            Extracted video ID or None if extraction fails
        """
        # Try different extraction strategies
        
        # Strategy 1: Use parent directory name if it looks like a video ID
        parent_name = file_path.parent.name
        if parent_name and parent_name not in ['masks', 'annotations', 'dense', 'sparse', 'uvo-dense', 'uvo-sparse', 'expressions', 'rel_annotations', 'box_annotations', 'temporal_annotations']:
            # Check if parent name looks like a video ID
            if parent_name.startswith('video') or parent_name.replace('_', '').replace('-', '').isdigit():
                return parent_name
        
        # Strategy 2: Use filename stem directly if it looks like a video ID
        stem = file_path.stem
        if stem and not stem.startswith('.') and stem not in ['masks', 'annotations']:
            # Prefer stems that look like video IDs
            if stem.startswith('video') or '_' in stem:
                return stem
        
        # Strategy 3: Extract from filename with pattern matching
        # Handle patterns like "video_123.json", "123_masks.json", etc.
        if stem:
            name_parts = stem.split('_')
            if len(name_parts) >= 2:
                # Try to find a video ID part
                for part in name_parts:
                    if part.isdigit() or (part.startswith('video') and len(part) > 5):
                        return stem  # Return full stem for complex patterns
        
        # Strategy 4: Fallback to stem if nothing else works
        if stem and not stem.startswith('.'):
            return stem
        
        return None

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve all available information for a single video ID, intelligently
        combining data from core annotations (masks) and any available supplementary files.
        
        Args:
            index: Sample index
            
        Returns:
            Standardized sample dictionary with UVO tracking and annotation data
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get video ID from index
        video_id = self._index[index]
        
        # Construct video path
        video_file = self._find_video_file(video_id)
        
        # Determine if this is a video file or frame directory
        if video_file.is_file():
            media_type = "video"
            media_path = video_file
        else:
            media_type = "video"  # Treat frame directories as videos
            media_path = video_file
        
        # Create base standardized structure
        sample = self._create_uvo_base_structure(video_id, media_path, media_type)
        
        # Load core annotations (prioritize dense over sparse)
        mask_data, mask_source = self._load_mask_annotations(video_id)
        
        # Load supplementary annotations
        expressions = self._load_expressions(video_id)
        relationships = self._load_relationships(video_id)
        box_annotations = self._load_box_annotations(video_id)
        additional_annotations = self._load_additional_annotations(video_id)
        
        # Add UVO-specific annotations
        sample['annotations'].update({
            'unidentified_video_objects': {
                'video_id': video_id,
                'object_tracks': mask_data.get('tracks', {}),
                'mask_source': mask_source,
                'num_objects': len(mask_data.get('tracks', {})),
                'tracking_statistics': self._analyze_uvo_tracks(mask_data.get('tracks', {})),
                'expressions': expressions,
                'relationships': relationships,
                'box_annotations': box_annotations,
                'additional_annotations': additional_annotations,
                'annotation_coverage': {
                    'has_masks': bool(mask_data.get('tracks')),
                    'has_expressions': bool(expressions),
                    'has_relationships': bool(relationships),
                    'has_box_annotations': bool(box_annotations),
                    'has_additional': bool(additional_annotations)
                }
            },
            'video_metadata': {
                'video_filename': video_file.name if video_file.is_file() else None,
                'frame_directory': str(video_file) if video_file.is_dir() else None,
                'media_type': media_type
            },
            'dataset_info': {
                'task_type': 'unidentified_video_object_tracking',
                'source': 'UVO',
                'suitable_for_tracking': True,
                'suitable_for_object_discovery': True,
                'suitable_for_temporal_reasoning': True,
                'has_rich_annotations': bool(expressions or relationships or additional_annotations),
                'annotation_completeness': self._calculate_annotation_completeness(
                    mask_data, expressions, relationships, box_annotations, additional_annotations
                )
            }
        })
        
        return sample

    def _create_uvo_base_structure(self, video_id: str, media_path: Path, media_type: str) -> Dict[str, Any]:
        """
        Create the base standardized structure for UVO samples.
        
        Args:
            video_id: The video identifier
            media_path: Path to the video file or frame directory
            media_type: Type of media
            
        Returns:
            Dictionary with the basic standardized structure
        """
        # Check if media path exists
        if not media_path.exists():
            raise FileNotFoundError(
                f"Media path not found for video_id '{video_id}' in source '{self.source_name}'. "
                f"Checked path: {media_path}"
            )
        
        return {
            "source_dataset": self.source_name,
            "sample_id": video_id,
            "media_type": media_type,
            "media_path": str(media_path.resolve()),
            "width": None,   # Video dimensions not extracted by default
            "height": None,  # Video dimensions not extracted by default
            "annotations": {}
        }

    def _find_video_file(self, video_id: str) -> Path:
        """
        Find the video file or frame directory for a given video ID.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to the video file or frame directory
        """
        # Try as video file first
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_file = self.video_path / f"{video_id}{ext}"
            if video_file.exists():
                return video_file
        
        # Try as frame directory
        frame_dir = self.video_path / video_id
        if frame_dir.exists() and frame_dir.is_dir():
            return frame_dir
        
        # Fallback - return the expected path even if it doesn't exist
        # The error will be caught in _create_uvo_base_structure
        return self.video_path / video_id

    def _load_mask_annotations(self, video_id: str) -> tuple[Dict[str, Any], str]:
        """
        Load mask annotations for a video, prioritizing dense over sparse.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Tuple of (mask_data, source_type)
        """
        # Prioritize dense masks
        if video_id in self._video_id_to_dense_mask_path:
            mask_data = self._parse_mask_file(self._video_id_to_dense_mask_path[video_id])
            return mask_data, 'dense'
        
        # Fallback to sparse masks
        if video_id in self._video_id_to_sparse_mask_path:
            mask_data = self._parse_mask_file(self._video_id_to_sparse_mask_path[video_id])
            return mask_data, 'sparse'
        
        # No masks found
        return {'tracks': {}}, 'none'

    def _parse_mask_file(self, mask_path: Path) -> Dict[str, Any]:
        """
        Parse a mask annotation file (JSON or directory).
        
        Args:
            mask_path: Path to the mask annotation
            
        Returns:
            Dictionary containing parsed mask data
        """
        try:
            if mask_path.is_file() and mask_path.suffix == '.json':
                # JSON-based mask annotations
                with open(mask_path, 'r') as f:
                    data = json.load(f)
                return self._convert_json_to_tracks(data)
            
            elif mask_path.is_dir():
                # Directory-based mask annotations
                return self._convert_directory_to_tracks(mask_path)
            
        except Exception as e:
            logger.warning(f"Failed to parse mask file {mask_path}: {e}")
        
        return {'tracks': {}}

    def _convert_json_to_tracks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON mask data to standardized track format.
        
        Args:
            data: Raw JSON mask data
            
        Returns:
            Dictionary with standardized track format
        """
        tracks = {}
        
        # Handle different JSON structures common in UVO
        if 'annotations' in data:
            for annotation in data['annotations']:
                obj_id = annotation.get('id', annotation.get('object_id', 0))
                frame_id = annotation.get('frame_id', annotation.get('image_id', 0))
                
                if obj_id not in tracks:
                    tracks[obj_id] = []
                
                # Convert segmentation or bbox to standard format
                track_entry = {
                    'frame_id': frame_id,
                    'segmentation': annotation.get('segmentation'),
                    'bbox': annotation.get('bbox'),
                    'area': annotation.get('area'),
                    'category_id': annotation.get('category_id')
                }
                
                tracks[obj_id].append(track_entry)
        
        # Sort tracks by frame_id for temporal consistency
        for obj_id in tracks:
            tracks[obj_id].sort(key=lambda x: x['frame_id'])
        
        return {'tracks': tracks, 'metadata': data.get('info', {})}

    def _convert_directory_to_tracks(self, mask_dir: Path) -> Dict[str, Any]:
        """
        Convert directory-based mask data to standardized track format.
        
        Args:
            mask_dir: Directory containing mask files
            
        Returns:
            Dictionary with standardized track format
        """
        tracks = {}
        
        # Scan for mask files (PNG, etc.)
        for mask_file in mask_dir.glob('*.png'):
            # Extract frame and object information from filename
            # This is dataset-specific and may need adjustment
            filename = mask_file.stem
            
            # Common patterns: "frame_001_obj_01.png", "001_01.png", etc.
            parts = filename.replace('_', '-').split('-')
            if len(parts) >= 2:
                try:
                    frame_id = int(parts[0])
                    obj_id = int(parts[1])
                    
                    if obj_id not in tracks:
                        tracks[obj_id] = []
                    
                    tracks[obj_id].append({
                        'frame_id': frame_id,
                        'mask_file': str(mask_file),
                        'segmentation': None,  # Would need to load PNG to get actual segmentation
                        'bbox': None
                    })
                except ValueError:
                    continue
        
        # Sort tracks by frame_id
        for obj_id in tracks:
            tracks[obj_id].sort(key=lambda x: x['frame_id'])
        
        return {'tracks': tracks}

    def _load_expressions(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load expression annotations for a video."""
        if video_id not in self._video_id_to_expression_path:
            return None
        
        try:
            with open(self._video_id_to_expression_path[video_id], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load expressions for {video_id}: {e}")
            return None

    def _load_relationships(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load relationship annotations for a video."""
        if video_id not in self._video_id_to_relationship_path:
            return None
        
        try:
            with open(self._video_id_to_relationship_path[video_id], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load relationships for {video_id}: {e}")
            return None

    def _load_box_annotations(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load bounding box annotations for a video."""
        if video_id not in self._video_id_to_box_annotation_path:
            return None
        
        try:
            with open(self._video_id_to_box_annotation_path[video_id], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load box annotations for {video_id}: {e}")
            return None

    def _load_additional_annotations(self, video_id: str) -> Dict[str, Any]:
        """Load additional annotations for a video."""
        additional = {}
        
        if video_id in self._video_id_to_additional_annotations:
            for annotation_type, file_path in self._video_id_to_additional_annotations[video_id].items():
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            additional[annotation_type] = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            additional[annotation_type] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load {annotation_type} for {video_id}: {e}")
        
        return additional

    def _analyze_uvo_tracks(self, tracks: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze UVO tracks to provide statistics.
        
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
                'unique_objects': 0
            }
        
        track_lengths = [len(track) for track in tracks.values()]
        total_detections = sum(track_lengths)
        
        return {
            'avg_track_length': sum(track_lengths) / len(track_lengths),
            'min_track_length': min(track_lengths),
            'max_track_length': max(track_lengths),
            'total_detections': total_detections,
            'unique_objects': len(tracks)
        }

    def _calculate_annotation_completeness(self, mask_data, expressions, relationships, box_annotations, additional) -> float:
        """Calculate how complete the annotations are for this sample (0.0 to 1.0)."""
        completeness_factors = [
            1.0 if mask_data.get('tracks') else 0.0,  # Core masks
            1.0 if expressions else 0.0,              # Expressions
            1.0 if relationships else 0.0,            # Relationships
            1.0 if box_annotations else 0.0,          # Box annotations
            1.0 if additional else 0.0                # Additional annotations
        ]
        
        return sum(completeness_factors) / len(completeness_factors)

    def get_samples_by_annotation_completeness(self, min_completeness: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get samples filtered by annotation completeness.
        
        Args:
            min_completeness: Minimum completeness threshold (0.0 to 1.0)
            
        Returns:
            List of samples meeting the completeness threshold
        """
        matching_samples = []
        
        for i in range(len(self._index)):
            sample = self.get_item(i)
            completeness = sample['annotations']['dataset_info']['annotation_completeness']
            
            if completeness >= min_completeness:
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_with_expressions(self) -> List[Dict[str, Any]]:
        """Get samples that have expression annotations."""
        matching_samples = []
        
        for i, video_id in enumerate(self._index):
            if video_id in self._video_id_to_expression_path:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_samples_with_relationships(self) -> List[Dict[str, Any]]:
        """Get samples that have relationship annotations."""
        matching_samples = []
        
        for i, video_id in enumerate(self._index):
            if video_id in self._video_id_to_relationship_path:
                sample = self.get_item(i)
                matching_samples.append(sample)
        
        return matching_samples

    def get_annotation_coverage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about annotation coverage across the dataset.
        
        Returns:
            Dictionary with coverage statistics
        """
        total_videos = len(self._index)
        dense_masks = len(self._video_id_to_dense_mask_path)
        sparse_masks = len(self._video_id_to_sparse_mask_path)
        expressions = len(self._video_id_to_expression_path)
        relationships = len(self._video_id_to_relationship_path)
        box_annotations = len(self._video_id_to_box_annotation_path)
        
        return {
            'total_videos': total_videos,
            'mask_coverage': {
                'dense_masks': dense_masks,
                'sparse_masks': sparse_masks,
                'total_with_masks': len(set(self._video_id_to_dense_mask_path.keys()) | 
                                       set(self._video_id_to_sparse_mask_path.keys())),
                'dense_percentage': (dense_masks / total_videos * 100) if total_videos > 0 else 0,
                'sparse_percentage': (sparse_masks / total_videos * 100) if total_videos > 0 else 0
            },
            'supplementary_coverage': {
                'expressions': expressions,
                'relationships': relationships,
                'box_annotations': box_annotations,
                'expression_percentage': (expressions / total_videos * 100) if total_videos > 0 else 0,
                'relationship_percentage': (relationships / total_videos * 100) if total_videos > 0 else 0,
                'box_annotation_percentage': (box_annotations / total_videos * 100) if total_videos > 0 else 0
            },
            'annotation_types_discovered': list(set(
                key 
                for vid in self._video_id_to_additional_annotations
                for key in self._video_id_to_additional_annotations[vid].keys()
            ))
        }

    def list_available_annotation_types(self) -> List[str]:
        """
        List all available annotation types discovered in the dataset.
        
        Returns:
            List of annotation type names
        """
        types = ['masks']  # Always present conceptually
        
        if self._video_id_to_expression_path:
            types.append('expressions')
        if self._video_id_to_relationship_path:
            types.append('relationships')
        if self._video_id_to_box_annotation_path:
            types.append('box_annotations')
        
        # Add additional annotation types
        for video_annotations in self._video_id_to_additional_annotations.values():
            types.extend(video_annotations.keys())
        
        return sorted(list(set(types)))