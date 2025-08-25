# core/modules/frame_extractor.py

"""
Frame extraction module for SELECT-FRAME tasks.

This module provides efficient frame extraction from videos using multiple backends
(OpenCV, PyAV, decord) with optimizations for batch processing and caching.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExtractionBackend(Enum):
    """Available backends for frame extraction."""
    OPENCV = "opencv"
    PYAV = "pyav"
    DECORD = "decord"
    AUTO = "auto"  # Automatically select best available


class FrameExtractorBase(ABC):
    """Abstract base class for frame extraction backends."""
    
    @abstractmethod
    def extract_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """
        Extract specific frames from a video.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of 0-based frame indices to extract
            
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        pass
    
    @abstractmethod
    def extract_frame_range(self, video_path: Path, start_frame: int, 
                           end_frame: int) -> np.ndarray:
        """
        Extract a continuous range of frames.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (inclusive)
            
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        pass
    
    @abstractmethod
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get video metadata.
        
        Returns:
            Dictionary with keys: 'fps', 'frame_count', 'width', 'height', 'duration'
        """
        pass


class OpenCVExtractor(FrameExtractorBase):
    """Frame extractor using OpenCV (cv2)."""
    
    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError(
                "OpenCV not installed. Install with: pip install opencv-python"
            )
    
    def extract_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """Extract specific frames using OpenCV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        try:
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            for idx in sorted(frame_indices):
                if idx >= total_frames:
                    logger.warning(
                        f"Frame {idx} out of range for video with {total_frames} frames"
                    )
                    continue
                
                # Seek to frame
                cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame {idx} from {video_path}")
            
            if not frames:
                raise RuntimeError(f"No frames could be extracted from {video_path}")
            
            return np.array(frames)
            
        finally:
            cap.release()
    
    def extract_frame_range(self, video_path: Path, start_frame: int, 
                           end_frame: int) -> np.ndarray:
        """Extract continuous frame range using OpenCV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        try:
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_frames - 1))
            
            # Seek to start frame
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
                    break
            
            if not frames:
                raise RuntimeError(
                    f"No frames could be extracted from range [{start_frame}, {end_frame}]"
                )
            
            return np.array(frames)
            
        finally:
            cap.release()
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using OpenCV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        try:
            info = {
                'fps': cap.get(self.cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
            return info
            
        finally:
            cap.release()


class PyAVExtractor(FrameExtractorBase):
    """Frame extractor using PyAV (more efficient for random access)."""
    
    def __init__(self):
        try:
            import av
            self.av = av
        except ImportError:
            raise ImportError(
                "PyAV not installed. Install with: pip install av"
            )
    
    def extract_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """Extract specific frames using PyAV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        container = self.av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            
            # Configure stream for better seeking
            stream.codec_context.skip_frame = 'NONKEY'
            
            frames = []
            frame_indices_set = set(frame_indices)
            
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in frame_indices_set:
                    # Convert to RGB numpy array
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                    
                    # Remove from set to track completion
                    frame_indices_set.remove(frame_idx)
                    
                    # Stop if we've collected all frames
                    if not frame_indices_set:
                        break
            
            if not frames:
                raise RuntimeError(f"No frames could be extracted from {video_path}")
            
            # Sort frames back to requested order
            frame_dict = dict(zip(
                [idx for idx in frame_indices if idx < len(frames)], 
                frames
            ))
            sorted_frames = [frame_dict[idx] for idx in sorted(frame_dict.keys())]
            
            return np.array(sorted_frames)
            
        finally:
            container.close()
    
    def extract_frame_range(self, video_path: Path, start_frame: int, 
                           end_frame: int) -> np.ndarray:
        """Extract continuous frame range using PyAV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        container = self.av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            
            frames = []
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_frame:
                    continue
                if frame_idx > end_frame:
                    break
                    
                # Convert to RGB numpy array
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
            
            if not frames:
                raise RuntimeError(
                    f"No frames could be extracted from range [{start_frame}, {end_frame}]"
                )
            
            return np.array(frames)
            
        finally:
            container.close()
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using PyAV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        container = self.av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            
            # Calculate frame count if not available
            if stream.frames > 0:
                frame_count = stream.frames
            else:
                # Count frames manually
                frame_count = sum(1 for _ in container.decode(stream))
                container.seek(0)  # Reset
            
            info = {
                'fps': float(stream.average_rate),
                'frame_count': frame_count,
                'width': stream.width,
                'height': stream.height,
                'duration': float(stream.duration * stream.time_base) if stream.duration else 0
            }
            return info
            
        finally:
            container.close()


class DecordExtractor(FrameExtractorBase):
    """Frame extractor using Decord (optimized for deep learning)."""
    
    def __init__(self, gpu_id: int = -1):
        """
        Initialize Decord extractor.
        
        Args:
            gpu_id: GPU device ID for hardware acceleration (-1 for CPU)
        """
        try:
            import decord
            from decord import VideoReader, cpu, gpu
            self.decord = decord
            self.VideoReader = VideoReader
            self.ctx = gpu(gpu_id) if gpu_id >= 0 else cpu(0)
        except ImportError:
            raise ImportError(
                "Decord not installed. Install with: pip install decord"
            )
    
    def extract_frames(self, video_path: Path, frame_indices: List[int]) -> np.ndarray:
        """Extract specific frames using Decord."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        vr = self.VideoReader(str(video_path), ctx=self.ctx)
        
        # Decord can batch extract frames efficiently
        frames = vr.get_batch(frame_indices).asnumpy()
        
        return frames
    
    def extract_frame_range(self, video_path: Path, start_frame: int, 
                           end_frame: int) -> np.ndarray:
        """Extract continuous frame range using Decord."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        vr = self.VideoReader(str(video_path), ctx=self.ctx)
        
        # Create list of indices for the range
        frame_indices = list(range(start_frame, min(end_frame + 1, len(vr))))
        
        if not frame_indices:
            raise RuntimeError(
                f"No valid frames in range [{start_frame}, {end_frame}] "
                f"for video with {len(vr)} frames"
            )
        
        frames = vr.get_batch(frame_indices).asnumpy()
        
        return frames
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using Decord."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        vr = self.VideoReader(str(video_path), ctx=self.ctx)
        
        info = {
            'fps': vr.get_avg_fps(),
            'frame_count': len(vr),
            'width': vr[0].shape[1],  # Get from first frame
            'height': vr[0].shape[0],
            'duration': len(vr) / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0
        }
        return info


class FrameExtractor:
    """
    Main frame extractor with automatic backend selection and caching.
    
    This class provides a unified interface for frame extraction with:
    - Automatic backend selection based on availability
    - Frame caching for repeated access
    - Batch processing optimizations
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, backend: ExtractionBackend = ExtractionBackend.AUTO,
                 cache_size: int = 100, gpu_id: int = -1):
        """
        Initialize frame extractor.
        
        Args:
            backend: Which backend to use (AUTO selects best available)
            cache_size: Number of frame batches to cache
            gpu_id: GPU device ID for hardware acceleration (Decord only)
        """
        self.backend_type = backend
        self.cache_size = cache_size
        self.gpu_id = gpu_id
        
        # Frame cache: video_path -> {frame_range -> frames}
        self._cache: Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {}
        self._cache_order: List[Tuple[str, Tuple[int, ...]]] = []
        
        # Initialize backend
        self._backend = self._initialize_backend()
        logger.info(f"Initialized FrameExtractor with {self._backend.__class__.__name__}")
    
    def _initialize_backend(self) -> FrameExtractorBase:
        """Initialize the appropriate backend based on configuration."""
        
        if self.backend_type == ExtractionBackend.AUTO:
            # Try backends in order of preference
            backends_to_try = [
                (ExtractionBackend.DECORD, lambda: DecordExtractor(self.gpu_id)),
                (ExtractionBackend.PYAV, lambda: PyAVExtractor()),
                (ExtractionBackend.OPENCV, lambda: OpenCVExtractor())
            ]
            
            for backend_type, initializer in backends_to_try:
                try:
                    backend = initializer()
                    logger.info(f"Auto-selected {backend_type.value} backend")
                    return backend
                except ImportError as e:
                    logger.debug(f"Backend {backend_type.value} not available: {e}")
                    continue
            
            raise RuntimeError(
                "No video processing backend available. "
                "Install one of: opencv-python, av, or decord"
            )
        
        # Specific backend requested
        if self.backend_type == ExtractionBackend.OPENCV:
            return OpenCVExtractor()
        elif self.backend_type == ExtractionBackend.PYAV:
            return PyAVExtractor()
        elif self.backend_type == ExtractionBackend.DECORD:
            return DecordExtractor(self.gpu_id)
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")
    
    def extract_frames(self, video_path: Union[str, Path], 
                      frame_indices: Union[List[int], np.ndarray],
                      use_cache: bool = True) -> np.ndarray:
        """
        Extract specific frames from a video.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of 0-based frame indices to extract
            use_cache: Whether to use caching
            
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        video_path = Path(video_path)
        frame_indices = sorted(list(frame_indices))
        
        # Check cache
        if use_cache:
            cache_key = (str(video_path), tuple(frame_indices))
            if str(video_path) in self._cache:
                if tuple(frame_indices) in self._cache[str(video_path)]:
                    logger.debug(f"Cache hit for {len(frame_indices)} frames from {video_path.name}")
                    return self._cache[str(video_path)][tuple(frame_indices)]
        
        # Extract frames
        try:
            frames = self._backend.extract_frames(video_path, frame_indices)
            
            # Update cache
            if use_cache:
                self._update_cache(str(video_path), tuple(frame_indices), frames)
            
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            raise
    
    def extract_frame_range(self, video_path: Union[str, Path],
                           start_frame: int, end_frame: int,
                           use_cache: bool = True) -> np.ndarray:
        """
        Extract a continuous range of frames.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (inclusive)
            use_cache: Whether to use caching
            
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        video_path = Path(video_path)
        
        # Check cache
        if use_cache:
            cache_key = (str(video_path), (start_frame, end_frame))
            if str(video_path) in self._cache:
                if (start_frame, end_frame) in self._cache[str(video_path)]:
                    logger.debug(
                        f"Cache hit for frames [{start_frame}, {end_frame}] from {video_path.name}"
                    )
                    return self._cache[str(video_path)][(start_frame, end_frame)]
        
        # Extract frames
        try:
            frames = self._backend.extract_frame_range(video_path, start_frame, end_frame)
            
            # Update cache
            if use_cache:
                self._update_cache(str(video_path), (start_frame, end_frame), frames)
            
            return frames
            
        except Exception as e:
            logger.error(
                f"Failed to extract frame range [{start_frame}, {end_frame}] from {video_path}: {e}"
            )
            raise
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get video metadata.
        
        Returns:
            Dictionary with keys: 'fps', 'frame_count', 'width', 'height', 'duration'
        """
        video_path = Path(video_path)
        return self._backend.get_video_info(video_path)
    
    def _update_cache(self, video_path: str, frame_key: Tuple[int, ...], 
                     frames: np.ndarray):
        """Update the frame cache with LRU eviction."""
        
        # Initialize cache for this video if needed
        if video_path not in self._cache:
            self._cache[video_path] = {}
        
        # Add to cache
        self._cache[video_path][frame_key] = frames
        self._cache_order.append((video_path, frame_key))
        
        # Evict oldest if cache is full
        while len(self._cache_order) > self.cache_size:
            old_video, old_key = self._cache_order.pop(0)
            if old_video in self._cache and old_key in self._cache[old_video]:
                del self._cache[old_video][old_key]
                # Clean up empty video entries
                if not self._cache[old_video]:
                    del self._cache[old_video]
    
    def clear_cache(self):
        """Clear the frame cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.debug("Frame cache cleared")
    
    def preload_frames(self, video_path: Union[str, Path],
                      frame_ranges: List[Tuple[int, int]]):
        """
        Preload multiple frame ranges for efficient batch processing.
        
        Args:
            video_path: Path to the video file
            frame_ranges: List of (start_frame, end_frame) tuples
        """
        video_path = Path(video_path)
        
        for start_frame, end_frame in frame_ranges:
            try:
                self.extract_frame_range(video_path, start_frame, end_frame, use_cache=True)
            except Exception as e:
                logger.warning(
                    f"Failed to preload frames [{start_frame}, {end_frame}] "
                    f"from {video_path}: {e}"
                )


# Singleton instance for convenience
_default_extractor = None

def get_default_extractor() -> FrameExtractor:
    """Get the default frame extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = FrameExtractor()
    return _default_extractor