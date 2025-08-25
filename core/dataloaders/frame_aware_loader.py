# core/dataloaders/frame_aware_loader.py

"""
Frame-aware loader mixin for SELECT-FRAME tasks.

This module provides a mixin class that adds frame extraction capabilities
to existing video loaders like Assembly101Loader, StarQALoader, etc.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np

from ..modules.frame_extractor import FrameExtractor, ExtractionBackend
from .timestamp_utils import TimestampConverter, timestamp_range_to_frames

logger = logging.getLogger(__name__)


class FrameAwareLoaderMixin:
    """
    Mixin class that adds frame extraction capabilities to video loaders.
    
    This can be mixed into any loader that deals with video frames to provide:
    - Automatic frame extraction based on annotations
    - Multiple extraction strategies (all frames, keyframes, sampling)
    - Efficient caching and batch processing
    """
    
    def __init__(self, *args, extract_frames: bool = False, 
                 extraction_backend: ExtractionBackend = ExtractionBackend.AUTO,
                 frame_cache_size: int = 100, 
                 gpu_id: int = -1, 
                 timestamp_converter: Optional[TimestampConverter] = None,
                 **kwargs):
        """
        Initialize frame-aware loader.
        
        Args:
            extract_frames: Whether to extract actual frames or just return metadata
            extraction_backend: Which backend to use for frame extraction
            frame_cache_size: Number of frame batches to cache
            gpu_id: GPU device ID for hardware acceleration
            timestamp_converter: Optional converter for timestamp-to-frame conversion
        """
        super().__init__(*args, **kwargs)
        
        self.extract_frames = extract_frames
        self._frame_extractor = None
        self._timestamp_converter = timestamp_converter or TimestampConverter()
        
        if self.extract_frames:
            self._frame_extractor = FrameExtractor(
                backend=extraction_backend,
                cache_size=frame_cache_size,
                gpu_id=gpu_id
            )
            logger.info(f"Frame extraction enabled with {extraction_backend.value} backend")
    
    def _extract_frames_for_sample(self, video_path: Path, 
                                  start_frame: int, end_frame: int,
                                  sampling_strategy: str = "all") -> Optional[np.ndarray]:
        """
        Extract frames for a sample based on the sampling strategy.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            sampling_strategy: How to sample frames:
                - "all": Extract all frames in range
                - "uniform_N": Uniformly sample N frames (e.g., "uniform_8")
                - "keyframes": Extract keyframes only (if available)
                - "endpoints": Extract only start and end frames
                - "middle": Extract only the middle frame
                
        Returns:
            Extracted frames as numpy array or None if extraction disabled
        """
        if not self.extract_frames or self._frame_extractor is None:
            return None
        
        try:
            if sampling_strategy == "all":
                # Extract all frames in the range
                frames = self._frame_extractor.extract_frame_range(
                    video_path, start_frame, end_frame
                )
                
            elif sampling_strategy.startswith("uniform_"):
                # Uniform sampling
                n_samples = int(sampling_strategy.split("_")[1])
                total_frames = end_frame - start_frame + 1
                
                if total_frames <= n_samples:
                    # If range is smaller than samples, get all frames
                    frames = self._frame_extractor.extract_frame_range(
                        video_path, start_frame, end_frame
                    )
                else:
                    # Sample uniformly
                    indices = np.linspace(start_frame, end_frame, n_samples, dtype=int)
                    frames = self._frame_extractor.extract_frames(video_path, indices.tolist())
                    
            elif sampling_strategy == "endpoints":
                # Only start and end frames
                frames = self._frame_extractor.extract_frames(
                    video_path, [start_frame, end_frame]
                )
                
            elif sampling_strategy == "middle":
                # Only middle frame
                middle_frame = (start_frame + end_frame) // 2
                frames = self._frame_extractor.extract_frames(
                    video_path, [middle_frame]
                )
                
            elif sampling_strategy == "keyframes":
                # Extract keyframes if available (would need to be implemented based on dataset)
                logger.warning("Keyframe extraction not yet implemented, falling back to uniform_8")
                return self._extract_frames_for_sample(
                    video_path, start_frame, end_frame, "uniform_8"
                )
                
            else:
                logger.warning(f"Unknown sampling strategy: {sampling_strategy}, using 'all'")
                frames = self._frame_extractor.extract_frame_range(
                    video_path, start_frame, end_frame
                )
            
            return frames
            
        except Exception as e:
            logger.error(
                f"Failed to extract frames [{start_frame}, {end_frame}] "
                f"from {video_path}: {e}"
            )
            if self.skip_on_error if hasattr(self, 'skip_on_error') else False:
                return None
            raise
    
    def _add_frame_data_to_sample(self, sample: Dict[str, Any], 
                                 frames: Optional[np.ndarray],
                                 sampling_strategy: str = "all") -> Dict[str, Any]:
        """
        Add extracted frame data to a sample dictionary.
        
        Args:
            sample: The base sample dictionary
            frames: Extracted frames as numpy array
            sampling_strategy: The strategy used for extraction
            
        Returns:
            Updated sample dictionary with frame data
        """
        if frames is not None:
            sample['frames'] = {
                'data': frames,
                'shape': frames.shape,  # (num_frames, height, width, channels)
                'dtype': str(frames.dtype),
                'sampling_strategy': sampling_strategy,
                'num_frames': frames.shape[0]
            }
            
            # Add frame-level metadata if available
            if 'annotations' in sample and 'action_segment' in sample['annotations']:
                segment = sample['annotations']['action_segment']
                if 'start_frame' in segment and 'end_frame' in segment:
                    sample['frames']['original_range'] = [
                        segment['start_frame'], 
                        segment['end_frame']
                    ]
        
        return sample
    
    def preload_video_frames(self, video_path: Path, 
                            segments: List[Tuple[int, int]]):
        """
        Preload frames for multiple segments from the same video.
        
        This is useful for efficient batch processing when you know
        you'll need multiple segments from the same video.
        
        Args:
            video_path: Path to the video file
            segments: List of (start_frame, end_frame) tuples
        """
        if self.extract_frames and self._frame_extractor:
            self._frame_extractor.preload_frames(video_path, segments)
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video metadata.
        
        Returns:
            Dictionary with video information (fps, frame_count, etc.)
        """
        if self._frame_extractor:
            info = self._frame_extractor.get_video_info(video_path)
            # Cache FPS for this video in the timestamp converter
            if info and 'fps' in info:
                self._timestamp_converter._fps_cache[str(video_path)] = info['fps']
            return info
        else:
            logger.warning("Frame extraction not enabled, cannot get video info")
            return {}
    
    def _extract_frames_for_timestamp(
        self,
        video_path: Path,
        start_time: Union[float, str],
        end_time: Union[float, str],
        sampling_strategy: str = "all"
    ) -> Optional[np.ndarray]:
        """
        Extract frames for a timestamp range.
        
        This method converts timestamps to frame indices and then extracts frames.
        
        Args:
            video_path: Path to the video file
            start_time: Start timestamp in seconds or string format
            end_time: End timestamp in seconds or string format
            sampling_strategy: How to sample frames
            
        Returns:
            Extracted frames as numpy array or None if extraction disabled
        """
        if not self.extract_frames or self._frame_extractor is None:
            return None
        
        # Get video info to retrieve FPS
        video_info = self.get_video_info(video_path)
        if not video_info or 'fps' not in video_info:
            logger.error(f"Could not get FPS for video: {video_path}")
            return None
        
        fps = video_info['fps']
        
        # Convert timestamps to frame indices
        start_frame, end_frame = self._timestamp_converter.convert_range(
            start_time, end_time, fps, video_path, inclusive=True
        )
        
        # Validate against actual frame count
        frame_count = video_info.get('frame_count', float('inf'))
        if start_frame >= frame_count:
            logger.warning(
                f"Start frame {start_frame} beyond video length {frame_count}"
            )
            return None
        
        end_frame = min(end_frame, frame_count - 1)
        
        # Extract frames using the existing method
        return self._extract_frames_for_sample(
            video_path, start_frame, end_frame, sampling_strategy
        )
    
    def _convert_timestamps_to_frames(
        self,
        timestamps: List[Union[float, str]],
        video_path: Path
    ) -> List[int]:
        """
        Convert a list of timestamps to frame indices.
        
        Args:
            timestamps: List of timestamps in seconds or string format
            video_path: Path to the video file (for FPS lookup)
            
        Returns:
            List of frame indices
        """
        # Get video info to retrieve FPS
        video_info = self.get_video_info(video_path)
        if not video_info or 'fps' not in video_info:
            logger.error(f"Could not get FPS for video: {video_path}")
            return []
        
        fps = video_info['fps']
        frame_count = video_info.get('frame_count', float('inf'))
        
        frame_indices = []
        for timestamp in timestamps:
            frame_idx = self._timestamp_converter.convert(
                timestamp, fps, video_path
            )
            # Validate frame index
            if 0 <= frame_idx < frame_count:
                frame_indices.append(frame_idx)
            else:
                logger.warning(
                    f"Frame index {frame_idx} out of bounds for video with "
                    f"{frame_count} frames"
                )
        
        return frame_indices


class FrameAwareAssembly101Loader(FrameAwareLoaderMixin):
    """
    Example of Assembly101Loader with frame extraction capabilities.
    
    This demonstrates how to combine the mixin with an existing loader.
    """
    
    def __init__(self, config: Dict[str, Any], extract_frames: bool = False,
                 sampling_strategy: str = "uniform_8", **kwargs):
        """
        Initialize Assembly101 loader with frame extraction.
        
        Args:
            config: Loader configuration
            extract_frames: Whether to extract actual frames
            sampling_strategy: How to sample frames from segments
        """
        # Import the original loader
        from .assembly101_loader import Assembly101Loader
        
        # Initialize both the mixin and the original loader
        self.sampling_strategy = sampling_strategy
        
        # Create a temporary class that inherits from both
        class _TempLoader(FrameAwareLoaderMixin, Assembly101Loader):
            pass
        
        # Initialize through the temporary class
        _TempLoader.__init__(self, config, extract_frames=extract_frames, **kwargs)
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get item with optional frame extraction.
        """
        # Get the base sample from the original loader
        sample = super().get_item(index)
        
        # If frame extraction is enabled, extract frames
        if self.extract_frames and 'annotations' in sample:
            action_segment = sample['annotations'].get('action_segment', {})
            
            if 'start_frame' in action_segment and 'end_frame' in action_segment:
                video_path = Path(sample['media_path'])
                start_frame = action_segment['start_frame']
                end_frame = action_segment['end_frame']
                
                # Extract frames
                frames = self._extract_frames_for_sample(
                    video_path, start_frame, end_frame, 
                    self.sampling_strategy
                )
                
                # Add frame data to sample
                sample = self._add_frame_data_to_sample(
                    sample, frames, self.sampling_strategy
                )
        
        return sample


def create_frame_aware_loader(loader_class, extract_frames: bool = False,
                             sampling_strategy: str = "uniform_8",
                             **extractor_kwargs):
    """
    Factory function to create a frame-aware version of any video loader.
    
    Args:
        loader_class: The original loader class (e.g., Assembly101Loader)
        extract_frames: Whether to extract frames
        sampling_strategy: How to sample frames
        **extractor_kwargs: Additional arguments for FrameExtractor
        
    Returns:
        A new loader class with frame extraction capabilities
    """
    
    class FrameAwareLoader(FrameAwareLoaderMixin, loader_class):
        """Dynamic frame-aware loader class."""
        
        def __init__(self, config: Dict[str, Any], **kwargs):
            # Merge extractor kwargs
            kwargs.update(extractor_kwargs)
            kwargs['extract_frames'] = extract_frames
            
            # Store sampling strategy
            self.sampling_strategy = sampling_strategy
            
            # Initialize both parent classes
            super().__init__(config, **kwargs)
        
        def get_item(self, index: int) -> Dict[str, Any]:
            """Get item with optional frame extraction."""
            # Get base sample
            sample = loader_class.get_item(self, index)
            
            # Extract frames if enabled
            if self.extract_frames:
                # Look for frame information in annotations
                frame_info = None
                
                # Check different possible locations for frame info
                if 'annotations' in sample:
                    annotations = sample['annotations']
                    
                    # Assembly101 style
                    if 'action_segment' in annotations:
                        segment = annotations['action_segment']
                        if 'start_frame' in segment and 'end_frame' in segment:
                            frame_info = (segment['start_frame'], segment['end_frame'])
                    
                    # STARQA style (might have different structure)
                    elif 'start_frame' in annotations and 'end_frame' in annotations:
                        frame_info = (annotations['start_frame'], annotations['end_frame'])
                
                # Extract frames if we found frame information
                if frame_info:
                    video_path = Path(sample['media_path'])
                    start_frame, end_frame = frame_info
                    
                    frames = self._extract_frames_for_sample(
                        video_path, start_frame, end_frame,
                        self.sampling_strategy
                    )
                    
                    sample = self._add_frame_data_to_sample(
                        sample, frames, self.sampling_strategy
                    )
            
            return sample
    
    # Set a meaningful name for the class
    FrameAwareLoader.__name__ = f"FrameAware{loader_class.__name__}"
    
    return FrameAwareLoader