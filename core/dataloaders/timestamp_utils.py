# core/dataloaders/timestamp_utils.py

"""
Utilities for converting between timestamps and frame indices.

This module provides functions to handle various timestamp formats
and convert them to frame indices based on video FPS.
"""

import logging
from typing import Union, Tuple, List, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def timestamp_to_frame(timestamp: float, fps: float, round_mode: str = "round") -> int:
    """
    Convert a timestamp in seconds to a frame index.
    
    Args:
        timestamp: Time in seconds
        fps: Frames per second of the video
        round_mode: How to round the frame index:
            - "round": Round to nearest frame (default)
            - "floor": Round down to previous frame
            - "ceil": Round up to next frame
    
    Returns:
        Frame index (0-based)
    
    Examples:
        >>> timestamp_to_frame(2.5, 30.0)  # 2.5 seconds at 30 fps
        75
        >>> timestamp_to_frame(2.5, 29.97)  # NTSC video
        75
    """
    if timestamp < 0:
        raise ValueError(f"Timestamp cannot be negative: {timestamp}")
    
    if fps <= 0:
        raise ValueError(f"FPS must be positive: {fps}")
    
    frame_float = timestamp * fps
    
    if round_mode == "round":
        return int(round(frame_float))
    elif round_mode == "floor":
        return int(np.floor(frame_float))
    elif round_mode == "ceil":
        return int(np.ceil(frame_float))
    else:
        raise ValueError(f"Invalid round_mode: {round_mode}")


def frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """
    Convert a frame index to a timestamp in seconds.
    
    Args:
        frame_idx: Frame index (0-based)
        fps: Frames per second of the video
    
    Returns:
        Timestamp in seconds
    
    Examples:
        >>> frame_to_timestamp(75, 30.0)
        2.5
    """
    if frame_idx < 0:
        raise ValueError(f"Frame index cannot be negative: {frame_idx}")
    
    if fps <= 0:
        raise ValueError(f"FPS must be positive: {fps}")
    
    return frame_idx / fps


def timestamp_range_to_frames(
    start_time: float, 
    end_time: float, 
    fps: float,
    inclusive: bool = True
) -> Tuple[int, int]:
    """
    Convert a timestamp range to frame indices.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Frames per second
        inclusive: Whether end frame is inclusive
    
    Returns:
        Tuple of (start_frame, end_frame)
    
    Examples:
        >>> timestamp_range_to_frames(1.0, 2.0, 30.0)
        (30, 60)  # Frames 30-60 inclusive
    """
    if start_time > end_time:
        raise ValueError(f"Start time ({start_time}) cannot be after end time ({end_time})")
    
    start_frame = timestamp_to_frame(start_time, fps, round_mode="floor")
    
    if inclusive:
        # For inclusive ranges, round up to include partial frames
        end_frame = timestamp_to_frame(end_time, fps, round_mode="ceil")
    else:
        # For exclusive ranges, round down
        end_frame = timestamp_to_frame(end_time, fps, round_mode="floor")
    
    return start_frame, end_frame


def frames_to_timestamp_range(
    start_frame: int,
    end_frame: int,
    fps: float
) -> Tuple[float, float]:
    """
    Convert frame indices to a timestamp range.
    
    Args:
        start_frame: Start frame index
        end_frame: End frame index (inclusive)
        fps: Frames per second
    
    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    start_time = frame_to_timestamp(start_frame, fps)
    # Add 1 to end_frame because it's inclusive
    end_time = frame_to_timestamp(end_frame + 1, fps)
    
    return start_time, end_time


def parse_timestamp_string(timestamp_str: str) -> float:
    """
    Parse various timestamp string formats to seconds.
    
    Supports formats:
        - "HH:MM:SS.mmm" (e.g., "01:23:45.678")
        - "MM:SS.mmm" (e.g., "23:45.678")
        - "SS.mmm" (e.g., "45.678")
        - "HH:MM:SS" (e.g., "01:23:45")
        - "MM:SS" (e.g., "23:45")
        - Plain number (e.g., "45", "45.678")
    
    Args:
        timestamp_str: Timestamp string
    
    Returns:
        Time in seconds
    
    Examples:
        >>> parse_timestamp_string("01:23:45.678")
        5025.678
        >>> parse_timestamp_string("23:45")
        1425.0
    """
    timestamp_str = timestamp_str.strip()
    
    # Try to parse as plain number first
    try:
        return float(timestamp_str)
    except ValueError:
        pass
    
    # Parse time format
    parts = timestamp_str.split(":")
    
    if len(parts) == 1:
        # Just seconds
        return float(parts[0])
    elif len(parts) == 2:
        # MM:SS
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")


def handle_variable_fps(
    timestamps: List[float],
    fps_list: List[float],
    segment_boundaries: Optional[List[float]] = None
) -> List[int]:
    """
    Handle videos with variable frame rates.
    
    Some videos (especially web videos) may have variable frame rates.
    This function handles conversion when FPS changes throughout the video.
    
    Args:
        timestamps: List of timestamps to convert
        fps_list: List of FPS values for different segments
        segment_boundaries: Time boundaries where FPS changes (in seconds)
    
    Returns:
        List of frame indices
    
    Note:
        If segment_boundaries is None, assumes uniform FPS (uses fps_list[0])
    """
    if not fps_list:
        raise ValueError("fps_list cannot be empty")
    
    if segment_boundaries is None:
        # Uniform FPS
        uniform_fps = fps_list[0]
        return [timestamp_to_frame(t, uniform_fps) for t in timestamps]
    
    frame_indices = []
    cumulative_frames = 0
    
    for timestamp in timestamps:
        # Find which segment this timestamp belongs to
        segment_idx = 0
        for i, boundary in enumerate(segment_boundaries):
            if timestamp >= boundary:
                segment_idx = i + 1
            else:
                break
        
        # Ensure we don't exceed fps_list bounds
        segment_idx = min(segment_idx, len(fps_list) - 1)
        
        # Calculate frame index within this segment
        if segment_idx == 0:
            segment_start_time = 0
        else:
            segment_start_time = segment_boundaries[segment_idx - 1]
        
        time_in_segment = timestamp - segment_start_time
        fps = fps_list[segment_idx]
        
        # Calculate cumulative frames from previous segments
        if segment_idx > 0:
            for i in range(segment_idx):
                if i < len(segment_boundaries):
                    segment_duration = segment_boundaries[i] - (
                        0 if i == 0 else segment_boundaries[i-1]
                    )
                    cumulative_frames += int(segment_duration * fps_list[i])
        
        frame_in_segment = timestamp_to_frame(time_in_segment, fps)
        frame_indices.append(cumulative_frames + frame_in_segment)
    
    return frame_indices


def normalize_fps(fps: float) -> float:
    """
    Normalize common FPS values to their standard representations.
    
    Handles common video standards:
        - NTSC: 29.97, 59.94, 119.88
        - PAL: 25, 50
        - Film: 24, 48
        - NTSC Film: 23.976, 47.952
    
    Args:
        fps: Raw FPS value
    
    Returns:
        Normalized FPS value
    
    Examples:
        >>> normalize_fps(29.970029)  # Imprecise NTSC
        29.97
        >>> normalize_fps(23.976024)  # Imprecise NTSC Film
        23.976
    """
    # Common FPS standards with tolerance
    standards = {
        23.976: (23.9, 24.0),   # NTSC Film
        24.0: (23.99, 24.01),    # Film
        25.0: (24.99, 25.01),    # PAL
        29.97: (29.9, 30.0),     # NTSC
        30.0: (29.99, 30.01),    # Standard
        47.952: (47.9, 48.0),    # NTSC Film 2x
        48.0: (47.99, 48.01),    # Film 2x
        50.0: (49.99, 50.01),    # PAL 2x
        59.94: (59.9, 60.0),     # NTSC 2x
        60.0: (59.99, 60.01),    # Standard 2x
        119.88: (119.8, 120.0),  # NTSC 4x
        120.0: (119.99, 120.01)  # Standard 4x
    }
    
    for standard_fps, (min_fps, max_fps) in standards.items():
        if min_fps <= fps < max_fps:
            return standard_fps
    
    # Return as-is if not a standard value
    return fps


class TimestampConverter:
    """
    Helper class for consistent timestamp-to-frame conversion across a dataset.
    
    This class maintains FPS information and provides convenient methods
    for converting timestamps to frames with consistent settings.
    """
    
    def __init__(
        self,
        default_fps: float = 30.0,
        round_mode: str = "round",
        normalize: bool = True
    ):
        """
        Initialize the converter.
        
        Args:
            default_fps: Default FPS to use if not specified
            round_mode: Default rounding mode for conversions
            normalize: Whether to normalize FPS values to standards
        """
        self.default_fps = default_fps
        self.round_mode = round_mode
        self.normalize = normalize
        self._fps_cache = {}
    
    def get_fps(
        self, 
        video_path: Optional[Path] = None,
        fps: Optional[float] = None
    ) -> float:
        """
        Get FPS for a video, using cache if available.
        
        Args:
            video_path: Path to video file (for caching)
            fps: Explicitly provided FPS
        
        Returns:
            FPS value (normalized if enabled)
        """
        if fps is not None:
            fps_value = fps
        elif video_path and str(video_path) in self._fps_cache:
            fps_value = self._fps_cache[str(video_path)]
        else:
            fps_value = self.default_fps
            
        if self.normalize:
            fps_value = normalize_fps(fps_value)
        
        # Cache the value if we have a path
        if video_path:
            self._fps_cache[str(video_path)] = fps_value
        
        return fps_value
    
    def convert(
        self,
        timestamp: Union[float, str],
        fps: Optional[float] = None,
        video_path: Optional[Path] = None
    ) -> int:
        """
        Convert a timestamp to frame index.
        
        Args:
            timestamp: Timestamp in seconds or string format
            fps: FPS value (uses default if not provided)
            video_path: Video path for FPS caching
        
        Returns:
            Frame index
        """
        # Parse string timestamps
        if isinstance(timestamp, str):
            timestamp = parse_timestamp_string(timestamp)
        
        fps = self.get_fps(video_path, fps)
        return timestamp_to_frame(timestamp, fps, self.round_mode)
    
    def convert_range(
        self,
        start_time: Union[float, str],
        end_time: Union[float, str],
        fps: Optional[float] = None,
        video_path: Optional[Path] = None,
        inclusive: bool = True
    ) -> Tuple[int, int]:
        """
        Convert a timestamp range to frame indices.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            fps: FPS value
            video_path: Video path for FPS caching
            inclusive: Whether end frame is inclusive
        
        Returns:
            Tuple of (start_frame, end_frame)
        """
        # Parse string timestamps
        if isinstance(start_time, str):
            start_time = parse_timestamp_string(start_time)
        if isinstance(end_time, str):
            end_time = parse_timestamp_string(end_time)
        
        fps = self.get_fps(video_path, fps)
        return timestamp_range_to_frames(start_time, end_time, fps, inclusive)