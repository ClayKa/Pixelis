#!/usr/bin/env python3
"""
Demonstration of timestamp-to-frame conversion for video datasets.

This script shows how loaders handle datasets like TVQA that provide
timestamps in seconds rather than frame indices.
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataloaders.timestamp_utils import (
    TimestampConverter,
    timestamp_to_frame,
    parse_timestamp_string,
    normalize_fps
)
from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
from core.dataloaders.tvqa_loader import TVQALoader, FrameAwareTVQALoader


def demonstrate_basic_conversion():
    """Demonstrate basic timestamp to frame conversion."""
    
    print("=" * 70)
    print("TIMESTAMP TO FRAME CONVERSION")
    print("=" * 70)
    print()
    
    print("1. BASIC CONVERSIONS")
    print("-" * 40)
    
    # Common frame rates
    fps_examples = [
        (24.0, "Film"),
        (25.0, "PAL"),
        (29.97, "NTSC"),
        (30.0, "Standard"),
        (60.0, "High Frame Rate")
    ]
    
    timestamp = 2.5  # 2.5 seconds
    
    print(f"Converting {timestamp} seconds to frames at various frame rates:")
    for fps, label in fps_examples:
        frame = timestamp_to_frame(timestamp, fps)
        print(f"  {label:15} ({fps:6.2f} fps): Frame {frame}")
    
    print()
    print("2. TIMESTAMP STRING PARSING")
    print("-" * 40)
    
    timestamp_formats = [
        "45.678",           # Plain seconds with milliseconds
        "01:23",            # MM:SS
        "01:23:45",         # HH:MM:SS
        "00:02:30.500",     # HH:MM:SS.mmm
    ]
    
    print("Parsing various timestamp formats:")
    for ts_str in timestamp_formats:
        seconds = parse_timestamp_string(ts_str)
        frame_30fps = timestamp_to_frame(seconds, 30.0)
        print(f"  '{ts_str:15}' -> {seconds:8.3f} seconds -> Frame {frame_30fps} @ 30fps")
    
    print()
    print("3. FPS NORMALIZATION")
    print("-" * 40)
    
    print("Normalizing imprecise FPS values to standards:")
    imprecise_fps = [
        (29.970029, "NTSC (imprecise)"),
        (23.976024, "NTSC Film (imprecise)"),
        (59.940059, "NTSC 2x (imprecise)"),
        (25.000001, "PAL (imprecise)")
    ]
    
    for fps_val, label in imprecise_fps:
        normalized = normalize_fps(fps_val)
        print(f"  {fps_val:10.6f} ({label:20}) -> {normalized:7.3f}")


def demonstrate_rounding_modes():
    """Show how different rounding modes affect frame calculation."""
    
    print()
    print("4. ROUNDING MODE EFFECTS")
    print("-" * 40)
    
    # Timestamp that doesn't align perfectly with frames
    timestamp = 2.517  # 2.517 seconds
    fps = 30.0  # Results in 75.51 frames
    
    modes = ["floor", "round", "ceil"]
    
    print(f"Converting {timestamp} seconds at {fps} fps (= {timestamp * fps:.2f} frames):")
    for mode in modes:
        frame = timestamp_to_frame(timestamp, fps, mode)
        print(f"  {mode:6} mode: Frame {frame}")
    
    print("\nThis affects which frame is selected from the video!")


def demonstrate_timestamp_converter():
    """Show the TimestampConverter helper class in action."""
    
    print()
    print("5. TIMESTAMP CONVERTER CLASS")
    print("-" * 40)
    
    # Create converter with defaults
    converter = TimestampConverter(
        default_fps=24.0,  # Common for movies
        round_mode="round",
        normalize=True
    )
    
    print("Using TimestampConverter with cached FPS:")
    
    # Simulate different videos
    videos = [
        ("movie.mp4", 23.976, "01:23:45"),     # NTSC Film
        ("tvshow.mp4", 29.97, "00:42:30"),     # NTSC TV
        ("youtube.mp4", 30.0, "05:15"),        # Standard web video
    ]
    
    for video_name, fps, timestamp_str in videos:
        video_path = Path(f"/videos/{video_name}")
        
        # Convert timestamp for this video
        frame = converter.convert(timestamp_str, fps=fps, video_path=video_path)
        seconds = parse_timestamp_string(timestamp_str)
        
        print(f"  {video_name:12} @ {fps:6.2f} fps:")
        print(f"    Timestamp: {timestamp_str:8} ({seconds:.1f}s) -> Frame {frame}")
        
        # Second call uses cached FPS
        frame2 = converter.convert("00:00:10", video_path=video_path)
        print(f"    10 seconds (cached FPS) -> Frame {frame2}")


def demonstrate_tvqa_style_annotations():
    """Show how TVQA-style timestamp annotations are handled."""
    
    print()
    print("6. TVQA-STYLE TIMESTAMP ANNOTATIONS")
    print("-" * 40)
    
    # Example TVQA annotation
    tvqa_annotation = {
        "qid": "12345",
        "vid_name": "friends_s01e01_seg02",
        "ts_start": 47.3,   # Start at 47.3 seconds
        "ts_end": 52.8,      # End at 52.8 seconds
        "q": "What is Ross doing when Monica enters?",
        "a0": "Eating pizza",
        "a1": "Reading a book",
        "a2": "Talking on the phone",
        "a3": "Watching TV",
        "a4": "Cooking dinner",
        "answer_idx": 2
    }
    
    print("TVQA annotation with timestamps in seconds:")
    print(f"  Video: {tvqa_annotation['vid_name']}")
    print(f"  Segment: {tvqa_annotation['ts_start']:.1f}s - {tvqa_annotation['ts_end']:.1f}s")
    print(f"  Duration: {tvqa_annotation['ts_end'] - tvqa_annotation['ts_start']:.1f} seconds")
    
    # Convert to frames at different frame rates
    print("\nConverted to frame indices:")
    
    for fps, source in [(23.976, "Film"), (29.97, "NTSC TV"), (25.0, "PAL TV")]:
        start_frame = timestamp_to_frame(tvqa_annotation['ts_start'], fps, "floor")
        end_frame = timestamp_to_frame(tvqa_annotation['ts_end'], fps, "ceil")
        num_frames = end_frame - start_frame + 1
        
        print(f"  At {fps:6.3f} fps ({source:8}): Frames {start_frame:4} - {end_frame:4} "
              f"({num_frames} frames)")


def demonstrate_loader_integration():
    """Show how timestamp conversion integrates with data loaders."""
    
    print()
    print("7. INTEGRATION WITH DATA LOADERS")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # Create mock TVQA structure
        video_dir = base_path / "videos" / "friends"
        video_dir.mkdir(parents=True)
        
        # Create mock annotation file
        annotations = [
            {
                "qid": "1",
                "show": "friends",
                "vid_name": "s01e01_seg01",
                "ts_start": 10.5,
                "ts_end": 15.3,
                "q": "Test question 1?",
                "a0": "Answer 0", "a1": "Answer 1", "a2": "Answer 2",
                "a3": "Answer 3", "a4": "Answer 4",
                "answer_idx": 2
            },
            {
                "qid": "2",
                "show": "friends",
                "vid_name": "s01e01_seg02",
                "ts_start": 120.0,
                "ts_end": 125.5,
                "q": "Test question 2?",
                "a0": "Answer 0", "a1": "Answer 1", "a2": "Answer 2",
                "a3": "Answer 3", "a4": "Answer 4",
                "answer_idx": 0
            }
        ]
        
        anno_file = base_path / "tvqa_train.jsonl"
        with open(anno_file, 'w') as f:
            for ann in annotations:
                f.write(json.dumps(ann) + '\n')
        
        # Create mock video files
        for ann in annotations:
            video_path = video_dir / f"{ann['vid_name']}.mp4"
            video_path.write_bytes(b"mock video content")
        
        # Configuration
        config = {
            'name': 'tvqa_demo',
            'path': str(base_path / "videos"),
            'annotation_file': str(anno_file)
        }
        
        print("Testing standard TVQA loader (timestamps only):")
        loader = TVQALoader(config)
        sample = loader.get_item(0)
        
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"  Timestamps: {sample['timestamps']['start']:.1f}s - "
              f"{sample['timestamps']['end']:.1f}s")
        print(f"  Duration: {sample['timestamps']['duration']:.1f}s")
        
        print("\nTesting frame-aware TVQA loader (with conversion):")
        
        # Mock frame extraction to avoid actual video processing
        from unittest.mock import MagicMock
        
        frame_loader = FrameAwareTVQALoader(config, extract_frames=False)
        
        # Mock the video info to provide FPS
        frame_loader._frame_extractor = MagicMock()
        frame_loader._frame_extractor.get_video_info.return_value = {
            'fps': 29.97,  # NTSC TV standard
            'frame_count': 3000,
            'width': 1280,
            'height': 720
        }
        
        # Manually calculate what frames would be extracted
        sample = frame_loader.get_item(0)
        fps = 29.97
        
        start_frame, end_frame = frame_loader._timestamp_converter.convert_range(
            sample['timestamps']['start'],
            sample['timestamps']['end'],
            fps
        )
        
        print(f"  Converted to frames at {fps} fps:")
        print(f"    Start: {sample['timestamps']['start']:.1f}s -> Frame {start_frame}")
        print(f"    End: {sample['timestamps']['end']:.1f}s -> Frame {end_frame}")
        print(f"    Total frames: {end_frame - start_frame + 1}")


def demonstrate_edge_cases():
    """Demonstrate handling of edge cases."""
    
    print()
    print("8. EDGE CASES AND VALIDATION")
    print("-" * 40)
    
    converter = TimestampConverter()
    
    print("Handling out-of-bounds timestamps:")
    
    # Mock video info
    video_info = {
        'fps': 30.0,
        'frame_count': 150,  # 5 seconds of video at 30 fps
        'duration': 5.0
    }
    
    test_cases = [
        (4.9, "Near end of video"),
        (5.0, "Exactly at end"),
        (5.1, "Beyond video length"),
        (0.0, "Start of video"),
        (-0.1, "Before video start (invalid)")
    ]
    
    print(f"Video: {video_info['frame_count']} frames @ {video_info['fps']} fps "
          f"({video_info['duration']}s)")
    
    for timestamp, description in test_cases:
        try:
            if timestamp >= 0:
                frame = timestamp_to_frame(timestamp, video_info['fps'])
                if frame < video_info['frame_count']:
                    status = f"Frame {frame} ✓"
                else:
                    status = f"Frame {frame} (out of bounds!)"
            else:
                frame = timestamp_to_frame(timestamp, video_info['fps'])
                status = f"Frame {frame}"
        except ValueError as e:
            status = f"Error: {str(e)}"
        
        print(f"  {timestamp:5.1f}s - {description:25} -> {status}")


if __name__ == "__main__":
    print("Starting timestamp conversion demonstration...\n")
    
    try:
        demonstrate_basic_conversion()
        demonstrate_rounding_modes()
        demonstrate_timestamp_converter()
        demonstrate_tvqa_style_annotations()
        demonstrate_loader_integration()
        demonstrate_edge_cases()
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
The timestamp-to-frame conversion system provides:

1. **Accurate Conversion**: Handles any FPS value with configurable rounding
2. **Format Flexibility**: Parses "HH:MM:SS.mmm", "MM:SS", plain seconds
3. **FPS Normalization**: Recognizes standard rates (NTSC, PAL, Film)
4. **Caching**: Remembers FPS per video for efficiency
5. **Validation**: Checks frame indices against video bounds
6. **Integration**: Seamlessly works with frame-aware loaders

This enables datasets like TVQA, MSRVTT, and others that use timestamps
to work with the same frame extraction infrastructure as frame-based datasets.
        """)
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()