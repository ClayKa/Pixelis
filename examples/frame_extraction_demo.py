#!/usr/bin/env python3
"""
Demonstration of frame extraction capabilities for SELECT-FRAME tasks.

This script shows how the frame extraction system works with video loaders
to efficiently extract and process video frames.
"""

import sys
import time
import tempfile
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.modules.frame_extractor import (
    FrameExtractor, ExtractionBackend, get_default_extractor
)
from core.dataloaders.frame_aware_loader import (
    create_frame_aware_loader, FrameAwareLoaderMixin
)


def create_test_video(output_path: Path, width: int = 640, height: int = 480,
                     fps: int = 30, duration: int = 5):
    """
    Create a test video using OpenCV.
    
    The video will have frame numbers drawn on each frame for verification.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, using mock video file")
        output_path.write_bytes(b"Mock video content")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_idx in range(total_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient based on frame number
        gradient = int(255 * frame_idx / total_frames)
        frame[:, :] = [gradient, 128, 255 - gradient]
        
        # Draw frame number
        text = f"Frame {frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Draw text with background
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, 2, (255, 255, 255), 3)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video with {total_frames} frames at {output_path}")


def demonstrate_basic_extraction():
    """Demonstrate basic frame extraction capabilities."""
    
    print("=" * 70)
    print("FRAME EXTRACTION DEMONSTRATION")
    print("=" * 70)
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video
        video_path = Path(tmpdir) / "test_video.mp4"
        create_test_video(video_path, fps=30, duration=3)  # 90 frames total
        
        print("1. BACKEND SELECTION")
        print("-" * 40)
        
        # Try different backends
        for backend in [ExtractionBackend.AUTO, ExtractionBackend.OPENCV]:
            try:
                extractor = FrameExtractor(backend=backend)
                print(f"✓ {backend.value}: Available")
                
                # Get video info
                info = extractor.get_video_info(video_path)
                print(f"  Video info: {info['frame_count']} frames, "
                      f"{info['width']}x{info['height']}, "
                      f"{info['fps']:.1f} fps")
                break
            except Exception as e:
                print(f"✗ {backend.value}: {e}")
        
        print()
        print("2. FRAME EXTRACTION METHODS")
        print("-" * 40)
        
        # Extract specific frames
        print("Extracting specific frames [0, 15, 30, 45]...")
        start_time = time.time()
        frames = extractor.extract_frames(video_path, [0, 15, 30, 45])
        extraction_time = time.time() - start_time
        print(f"  ✓ Extracted {frames.shape[0]} frames in {extraction_time:.3f}s")
        print(f"  Shape: {frames.shape} (frames, height, width, channels)")
        
        # Extract frame range
        print("\nExtracting frame range [20-40]...")
        start_time = time.time()
        frames_range = extractor.extract_frame_range(video_path, 20, 40)
        extraction_time = time.time() - start_time
        print(f"  ✓ Extracted {frames_range.shape[0]} frames in {extraction_time:.3f}s")
        
        print()
        print("3. CACHING PERFORMANCE")
        print("-" * 40)
        
        # First extraction (no cache)
        print("First extraction [10-20] (no cache)...")
        start_time = time.time()
        frames1 = extractor.extract_frame_range(video_path, 10, 20)
        time1 = time.time() - start_time
        print(f"  Time: {time1:.3f}s")
        
        # Second extraction (with cache)
        print("Second extraction [10-20] (with cache)...")
        start_time = time.time()
        frames2 = extractor.extract_frame_range(video_path, 10, 20)
        time2 = time.time() - start_time
        print(f"  Time: {time2:.3f}s")
        print(f"  ✓ Speedup: {time1/time2:.1f}x")
        
        # Verify frames are identical
        assert np.array_equal(frames1, frames2), "Cached frames don't match!"
        print("  ✓ Cache consistency verified")
        
        print()
        print("4. SAMPLING STRATEGIES")
        print("-" * 40)
        
        # Demonstrate different sampling strategies
        from core.dataloaders.frame_aware_loader import FrameAwareLoaderMixin
        
        # Create a mock loader with frame extraction
        class MockLoader(FrameAwareLoaderMixin):
            def __init__(self):
                self.extract_frames = True
                self._frame_extractor = extractor
        
        loader = MockLoader()
        
        strategies = [
            ("all", "All frames in range"),
            ("uniform_8", "8 uniformly sampled frames"),
            ("endpoints", "Only start and end frames"),
            ("middle", "Only middle frame")
        ]
        
        for strategy, description in strategies:
            frames = loader._extract_frames_for_sample(
                video_path, 30, 60, strategy
            )
            print(f"  {strategy:12} - {description:30} : {frames.shape[0]} frames")
        
        print()
        print("5. ERROR HANDLING")
        print("-" * 40)
        
        # Try to extract out-of-range frames
        print("Attempting to extract out-of-range frames [100, 200]...")
        try:
            info = extractor.get_video_info(video_path)
            max_frame = info['frame_count'] - 1
            
            # This should handle gracefully
            frames = extractor.extract_frames(video_path, [100, 200])
            print(f"  ✓ Handled gracefully, extracted {frames.shape[0]} valid frames")
        except Exception as e:
            print(f"  ✓ Error handled: {str(e)[:50]}...")
        
        # Try non-existent file
        print("Attempting to extract from non-existent file...")
        try:
            frames = extractor.extract_frames(Path("/non/existent.mp4"), [0])
            print("  ✗ Should have raised an error!")
        except FileNotFoundError:
            print("  ✓ FileNotFoundError raised as expected")


def demonstrate_loader_integration():
    """Demonstrate integration with data loaders."""
    
    print()
    print("6. LOADER INTEGRATION")
    print("-" * 40)
    
    # Create mock Assembly101-style data
    with tempfile.TemporaryDirectory() as tmpdir:
        import pandas as pd
        
        # Create video
        video_dir = Path(tmpdir) / "videos" / "session1"
        video_dir.mkdir(parents=True)
        video_path = video_dir / "test_video.mp4"
        create_test_video(video_path, fps=30, duration=3)
        
        # Create annotation
        annotations = pd.DataFrame([
            {
                'video': 'session1/test_video.mp4',
                'start_frame': 10,
                'end_frame': 20,
                'action_cls': 'test_action'
            },
            {
                'video': 'session1/test_video.mp4',
                'start_frame': 40,
                'end_frame': 60,
                'action_cls': 'another_action'
            }
        ])
        
        anno_file = Path(tmpdir) / "annotations.csv"
        annotations.to_csv(anno_file, index=False)
        
        # Create config
        config = {
            'name': 'test_dataset',
            'path': str(Path(tmpdir) / "videos"),
            'annotation_file': str(anno_file)
        }
        
        print("Testing frame-aware Assembly101 loader...")
        
        # Import and create frame-aware version
        from core.dataloaders.assembly101_loader import Assembly101Loader
        FrameAwareAssembly101 = create_frame_aware_loader(
            Assembly101Loader,
            extract_frames=True,
            sampling_strategy="uniform_8"
        )
        
        # Initialize loader
        loader = FrameAwareAssembly101(config)
        print(f"  ✓ Loader initialized with {len(loader)} samples")
        
        # Get a sample with frames
        sample = loader.get_item(0)
        
        print(f"  Sample keys: {list(sample.keys())}")
        
        if 'frames' in sample:
            frame_data = sample['frames']
            print(f"  ✓ Frames extracted: {frame_data['shape']}")
            print(f"    - Strategy: {frame_data['sampling_strategy']}")
            print(f"    - Original range: {frame_data.get('original_range', 'N/A')}")
            print(f"    - Data type: {frame_data['dtype']}")
        else:
            print("  ✗ No frames in sample (extraction may be disabled)")


def demonstrate_performance_comparison():
    """Compare performance of different backends and strategies."""
    
    print()
    print("7. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a longer test video
        video_path = Path(tmpdir) / "perf_test.mp4"
        create_test_video(video_path, fps=30, duration=10)  # 300 frames
        
        # Test different scenarios
        scenarios = [
            ("Sequential access", [(i, i+10) for i in range(0, 100, 20)]),
            ("Random access", [(50, 60), (150, 160), (20, 30), (250, 260), (100, 110)]),
            ("Large ranges", [(0, 50), (100, 150), (200, 250)])
        ]
        
        for scenario_name, ranges in scenarios:
            print(f"\n{scenario_name}:")
            
            # Try with caching
            extractor_cached = FrameExtractor(cache_size=100)
            start_time = time.time()
            
            for start, end in ranges:
                frames = extractor_cached.extract_frame_range(video_path, start, end)
            
            cached_time = time.time() - start_time
            print(f"  With cache: {cached_time:.3f}s")
            
            # Try without caching
            extractor_no_cache = FrameExtractor(cache_size=0)
            start_time = time.time()
            
            for start, end in ranges:
                frames = extractor_no_cache.extract_frame_range(video_path, start, end)
            
            no_cache_time = time.time() - start_time
            print(f"  No cache:   {no_cache_time:.3f}s")
            
            if cached_time < no_cache_time:
                print(f"  ✓ Cache speedup: {no_cache_time/cached_time:.2f}x")


if __name__ == "__main__":
    print("Starting frame extraction demonstration...\n")
    
    try:
        demonstrate_basic_extraction()
        demonstrate_loader_integration()
        demonstrate_performance_comparison()
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
The frame extraction system provides:

1. **Multiple Backends**: OpenCV, PyAV, and Decord support
2. **Automatic Selection**: Chooses best available backend
3. **Efficient Caching**: LRU cache for repeated access
4. **Flexible Sampling**: Multiple strategies for frame selection
5. **Loader Integration**: Seamless integration with existing loaders
6. **Error Handling**: Graceful handling of edge cases

Key Implementation Details:
- Frame indices are 0-based
- Ranges are inclusive [start, end]
- Frames are returned as RGB numpy arrays
- Shape: (num_frames, height, width, 3)
- Cache provides significant speedup for repeated access

This system enables efficient frame extraction for SELECT-FRAME tasks
while maintaining flexibility and performance.
        """)
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()