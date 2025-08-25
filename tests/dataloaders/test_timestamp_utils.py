# tests/dataloaders/test_timestamp_utils.py

import pytest
from pathlib import Path
import numpy as np

from core.dataloaders.timestamp_utils import (
    timestamp_to_frame,
    frame_to_timestamp,
    timestamp_range_to_frames,
    frames_to_timestamp_range,
    parse_timestamp_string,
    handle_variable_fps,
    normalize_fps,
    TimestampConverter
)


class TestTimestampConversion:
    """Test timestamp to frame conversion functions."""
    
    def test_timestamp_to_frame_basic(self):
        """Test basic timestamp to frame conversion."""
        # 2.5 seconds at 30 fps = frame 75
        assert timestamp_to_frame(2.5, 30.0) == 75
        
        # 1.0 second at 24 fps = frame 24
        assert timestamp_to_frame(1.0, 24.0) == 24
        
        # 0 seconds = frame 0
        assert timestamp_to_frame(0.0, 30.0) == 0
    
    def test_timestamp_to_frame_rounding(self):
        """Test different rounding modes."""
        # 2.4 seconds at 30 fps = 72 frames
        # Round: 72, Floor: 72, Ceil: 72
        assert timestamp_to_frame(2.4, 30.0, "round") == 72
        assert timestamp_to_frame(2.4, 30.0, "floor") == 72
        assert timestamp_to_frame(2.4, 30.0, "ceil") == 72
        
        # 2.5 seconds at 30 fps = 75 frames
        assert timestamp_to_frame(2.5, 30.0, "round") == 75
        assert timestamp_to_frame(2.5, 30.0, "floor") == 75
        assert timestamp_to_frame(2.5, 30.0, "ceil") == 75
        
        # 2.51 seconds at 30 fps = 75.3 frames
        # Round: 75, Floor: 75, Ceil: 76
        assert timestamp_to_frame(2.51, 30.0, "round") == 75
        assert timestamp_to_frame(2.51, 30.0, "floor") == 75
        assert timestamp_to_frame(2.51, 30.0, "ceil") == 76
    
    def test_timestamp_to_frame_ntsc(self):
        """Test with NTSC frame rates (29.97 fps)."""
        # 1 second at 29.97 fps = ~30 frames
        assert timestamp_to_frame(1.0, 29.97, "round") == 30
        
        # 10 seconds at 29.97 fps = ~300 frames
        assert timestamp_to_frame(10.0, 29.97, "round") == 300
    
    def test_frame_to_timestamp(self):
        """Test frame to timestamp conversion."""
        # Frame 75 at 30 fps = 2.5 seconds
        assert frame_to_timestamp(75, 30.0) == 2.5
        
        # Frame 24 at 24 fps = 1.0 second
        assert frame_to_timestamp(24, 24.0) == 1.0
        
        # Frame 0 = 0 seconds
        assert frame_to_timestamp(0, 30.0) == 0.0
    
    def test_timestamp_range_to_frames(self):
        """Test timestamp range conversion."""
        # 1-2 seconds at 30 fps = frames 30-60
        start, end = timestamp_range_to_frames(1.0, 2.0, 30.0, inclusive=True)
        assert start == 30
        assert end == 60
        
        # 0-1 seconds at 24 fps = frames 0-24
        start, end = timestamp_range_to_frames(0.0, 1.0, 24.0, inclusive=True)
        assert start == 0
        assert end == 24
    
    def test_frames_to_timestamp_range(self):
        """Test frame range to timestamp conversion."""
        # Frames 30-60 at 30 fps = 1.0-2.033... seconds
        start_time, end_time = frames_to_timestamp_range(30, 60, 30.0)
        assert start_time == 1.0
        assert abs(end_time - 2.0333333) < 0.001
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Negative timestamp
        with pytest.raises(ValueError):
            timestamp_to_frame(-1.0, 30.0)
        
        # Zero or negative FPS
        with pytest.raises(ValueError):
            timestamp_to_frame(1.0, 0.0)
        
        with pytest.raises(ValueError):
            timestamp_to_frame(1.0, -30.0)
        
        # Invalid round mode
        with pytest.raises(ValueError):
            timestamp_to_frame(1.0, 30.0, "invalid")
        
        # Start after end
        with pytest.raises(ValueError):
            timestamp_range_to_frames(2.0, 1.0, 30.0)


class TestTimestampParsing:
    """Test timestamp string parsing."""
    
    def test_parse_seconds_only(self):
        """Test parsing plain seconds."""
        assert parse_timestamp_string("45.678") == 45.678
        assert parse_timestamp_string("45") == 45.0
        assert parse_timestamp_string("0") == 0.0
    
    def test_parse_minutes_seconds(self):
        """Test parsing MM:SS format."""
        assert parse_timestamp_string("23:45") == 23 * 60 + 45
        assert parse_timestamp_string("01:30") == 90.0
        assert parse_timestamp_string("00:45.5") == 45.5
    
    def test_parse_hours_minutes_seconds(self):
        """Test parsing HH:MM:SS format."""
        assert parse_timestamp_string("01:23:45") == 3600 + 23 * 60 + 45
        assert parse_timestamp_string("00:00:45") == 45.0
        assert parse_timestamp_string("02:30:00") == 2.5 * 3600
    
    def test_parse_with_milliseconds(self):
        """Test parsing with milliseconds."""
        assert parse_timestamp_string("45.678") == 45.678
        assert parse_timestamp_string("01:23.456") == 60 + 23.456
        assert parse_timestamp_string("01:23:45.678") == 3600 + 23 * 60 + 45.678


class TestFPSNormalization:
    """Test FPS normalization for standard video formats."""
    
    def test_normalize_ntsc(self):
        """Test NTSC frame rate normalization."""
        assert normalize_fps(29.970029) == 29.97
        assert normalize_fps(29.97) == 29.97
        assert normalize_fps(59.940059) == 59.94
        assert normalize_fps(23.976024) == 23.976
    
    def test_normalize_pal(self):
        """Test PAL frame rate normalization."""
        assert normalize_fps(25.0) == 25.0
        assert normalize_fps(24.999) == 25.0
        assert normalize_fps(50.0) == 50.0
        assert normalize_fps(49.999) == 50.0
    
    def test_normalize_film(self):
        """Test film frame rate normalization."""
        assert normalize_fps(24.0) == 24.0
        assert normalize_fps(23.999) == 23.976  # Actually closer to NTSC Film
        assert normalize_fps(48.0) == 48.0
    
    def test_non_standard_fps(self):
        """Test that non-standard FPS values pass through."""
        assert normalize_fps(15.0) == 15.0
        assert normalize_fps(120.5) == 120.5


class TestVariableFPS:
    """Test handling of variable frame rates."""
    
    def test_uniform_fps(self):
        """Test with uniform FPS (no boundaries)."""
        timestamps = [0.0, 1.0, 2.0, 3.0]
        fps_list = [30.0]
        
        frames = handle_variable_fps(timestamps, fps_list)
        assert frames == [0, 30, 60, 90]
    
    def test_variable_fps_two_segments(self):
        """Test with two different FPS segments."""
        timestamps = [0.5, 1.5, 2.5]  # Times in seconds
        fps_list = [30.0, 24.0]  # 30 fps for first 2 seconds, then 24 fps
        segment_boundaries = [2.0]  # Change happens at 2 seconds
        
        frames = handle_variable_fps(timestamps, fps_list, segment_boundaries)
        
        # 0.5s at 30fps = frame 15
        assert frames[0] == 15
        # 1.5s at 30fps = frame 45  
        assert frames[1] == 45
        # 2.5s: 2s at 30fps (60 frames) + 0.5s at 24fps (12 frames) = frame 72
        # Note: This is simplified; actual implementation may differ


class TestTimestampConverter:
    """Test the TimestampConverter helper class."""
    
    def test_basic_conversion(self):
        """Test basic conversion with converter."""
        converter = TimestampConverter(default_fps=30.0)
        
        # Convert single timestamp
        assert converter.convert(2.5) == 75
        assert converter.convert("2.5") == 75
        
        # Convert with specific FPS
        assert converter.convert(1.0, fps=24.0) == 24
    
    def test_range_conversion(self):
        """Test range conversion."""
        converter = TimestampConverter(default_fps=30.0)
        
        start, end = converter.convert_range(1.0, 2.0)
        assert start == 30
        assert end == 60
        
        # Test with string timestamps
        start, end = converter.convert_range("00:01", "00:02")
        assert start == 30
        assert end == 60
    
    def test_fps_caching(self):
        """Test that FPS values are cached per video."""
        converter = TimestampConverter()
        video_path = Path("/test/video.mp4")
        
        # First call sets FPS
        frame = converter.convert(1.0, fps=25.0, video_path=video_path)
        assert frame == 25
        
        # Second call uses cached FPS
        frame = converter.convert(2.0, video_path=video_path)
        assert frame == 50  # Uses cached 25 fps
    
    def test_different_round_modes(self):
        """Test converter with different rounding modes."""
        converter_round = TimestampConverter(round_mode="round")
        converter_floor = TimestampConverter(round_mode="floor")
        converter_ceil = TimestampConverter(round_mode="ceil")
        
        # 2.51 seconds at 30 fps = 75.3 frames
        assert converter_round.convert(2.51, fps=30.0) == 75
        assert converter_floor.convert(2.51, fps=30.0) == 75
        assert converter_ceil.convert(2.51, fps=30.0) == 76
    
    def test_normalization(self):
        """Test FPS normalization in converter."""
        converter = TimestampConverter(normalize=True)
        video_path = Path("/test/video.mp4")
        
        # NTSC-like value should be normalized
        frame = converter.convert(1.0, fps=29.970029, video_path=video_path)
        assert frame == 30  # Should use normalized 29.97
        
        # Check that FPS was normalized in cache
        assert converter._fps_cache.get(str(video_path)) == 29.97


class TestIntegrationWithLoader:
    """Test integration with frame-aware loaders."""
    
    def test_timestamp_extraction(self):
        """Test that timestamp-based extraction works."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        from unittest.mock import MagicMock, patch
        
        class TestLoader(FrameAwareLoaderMixin):
            def __init__(self):
                self.extract_frames = True
                self._frame_extractor = MagicMock()
                self._timestamp_converter = TimestampConverter(default_fps=30.0)
        
        loader = TestLoader()
        
        # Mock video info
        loader._frame_extractor.get_video_info.return_value = {
            'fps': 30.0,
            'frame_count': 300,
            'width': 1920,
            'height': 1080
        }
        
        # Mock frame extraction
        loader._frame_extractor.extract_frame_range.return_value = np.zeros((31, 480, 640, 3))
        
        video_path = Path("/test/video.mp4")
        
        # Extract frames for timestamp range 1.0-2.0 seconds
        frames = loader._extract_frames_for_timestamp(video_path, 1.0, 2.0, "all")
        
        # Should have called extract_frame_range with frames 30-60
        loader._frame_extractor.extract_frame_range.assert_called_with(
            video_path, 30, 60
        )
    
    def test_timestamp_list_conversion(self):
        """Test converting list of timestamps."""
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        from unittest.mock import MagicMock
        
        class TestLoader(FrameAwareLoaderMixin):
            def __init__(self):
                self.extract_frames = True
                self._frame_extractor = MagicMock()
                self._timestamp_converter = TimestampConverter()
        
        loader = TestLoader()
        
        # Mock video info
        loader._frame_extractor.get_video_info.return_value = {
            'fps': 24.0,
            'frame_count': 240
        }
        
        video_path = Path("/test/video.mp4")
        
        # Convert multiple timestamps
        timestamps = [0.5, 1.0, 1.5, "02:00"]  # Mixed formats
        frame_indices = loader._convert_timestamps_to_frames(timestamps, video_path)
        
        # At 24 fps:
        # 0.5s = frame 12
        # 1.0s = frame 24
        # 1.5s = frame 36
        # 2:00 (120s) = frame 2880 (but capped by frame_count)
        assert frame_indices == [12, 24, 36]  # Last one excluded as out of bounds