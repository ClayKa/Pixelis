#!/usr/bin/env python3
"""
Demonstration of robust data loader capabilities for handling edge cases.

This script shows how the robust loaders handle various error conditions
and provide detailed error reporting.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataloaders.robust_activitynet_loader import RobustActivityNetCaptionsLoader
from core.dataloaders.robust_base_loader import DataLoadError


def create_test_dataset(base_path: Path):
    """Create a test dataset with various issues for demonstration."""
    
    # Create directories
    anno_dir = base_path / "annotations"
    video_dir = base_path / "videos"
    anno_dir.mkdir()
    video_dir.mkdir()
    
    # Create annotations with various issues
    annotations = [
        # Valid annotation with existing video
        {
            "video_id": "valid_video_1",
            "video": "valid_video_1.mp4",
            "timestamps": [[0, 5], [5, 10], [10, 15]],
            "sentences": ["First segment", "Second segment", "Third segment"],
            "duration": 15.0,
            "caption": "A complete valid video annotation"
        },
        
        # Valid annotation but video file missing
        {
            "video_id": "missing_video",
            "video": "missing_video.mp4",
            "timestamps": [[0, 3], [3, 6]],
            "sentences": ["This video", "doesn't exist"],
            "duration": 6.0
        },
        
        # Mismatched timestamps and sentences
        {
            "video_id": "mismatched_video",
            "video": "mismatched_video.mp4",
            "timestamps": [[0, 2], [2, 4], [4, 6]],
            "sentences": ["Only two sentences", "here"],  # 3 timestamps, 2 sentences
            "duration": 6.0
        },
        
        # Missing required fields
        {
            "video_id": "incomplete_video",
            "video": "incomplete_video.mp4",
            # Missing timestamps and sentences
            "duration": 10.0
        },
        
        # Invalid timestamp formats
        {
            "video_id": "bad_timestamps",
            "video": "bad_timestamps.mp4",
            "timestamps": [[0, 5], "invalid", [-1, 3], [5, 2]],  # Various invalid formats
            "sentences": ["One", "Two", "Three", "Four"],
            "duration": 10.0
        },
        
        # Video with different extension
        {
            "video_id": "different_extension",
            "video": "different_extension.mp4",  # Says mp4 but we'll create mkv
            "timestamps": [[0, 5]],
            "sentences": ["Different format"],
            "duration": 5.0
        }
    ]
    
    # Save annotations
    anno_file = anno_dir / "test_annotations.json"
    with open(anno_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Create some video files
    (video_dir / "valid_video_1.mp4").write_bytes(b"Valid video content")
    (video_dir / "mismatched_video.mp4").write_bytes(b"Video with issues")
    (video_dir / "incomplete_video.mp4").touch()  # Empty file (corrupted)
    (video_dir / "bad_timestamps.mp4").write_bytes(b"Video content")
    (video_dir / "different_extension.mkv").write_bytes(b"MKV video")  # Different extension
    
    return anno_file, video_dir


def demonstrate_robust_loading():
    """Demonstrate robust loader capabilities."""
    
    print("=" * 70)
    print("ROBUST DATA LOADER DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create temporary test dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        anno_file, video_dir = create_test_dataset(base_path)
        
        # Configuration
        config = {
            "name": "demo_dataset",
            "path": str(video_dir),
            "annotation_file": str(anno_file)
        }
        
        print("1. STRICT MODE (skip_on_error=False)")
        print("-" * 40)
        print("This mode validates strictly but excludes problematic samples from the index.")
        print()
        
        try:
            loader_strict = RobustActivityNetCaptionsLoader(config, skip_on_error=False)
            print(f"✓ Successfully initialized loader")
            print(f"  - Total valid samples: {len(loader_strict)}")
            print(f"  - Corrupted samples skipped: {len(loader_strict._corrupted_samples)}")
            print(f"  - Validation errors found: {len(loader_strict._validation_errors)}")
            print()
            
            # Get error summary
            summary = loader_strict.get_error_summary()
            print("Error Summary:")
            for error_type, count in summary['error_types'].items():
                print(f"  - {error_type}: {count} occurrences")
            print()
            
            # Try to load valid samples
            if len(loader_strict) > 0:
                print("Loading first valid sample:")
                sample = loader_strict.get_item(0)
                print(f"  - Sample ID: {sample['sample_id']}")
                print(f"  - Media path: {Path(sample['media_path']).name}")
                print(f"  - Events: {len(sample['annotations']['timed_events'])} segments")
            print()
            
        except DataLoadError as e:
            print(f"✗ Failed to initialize: {e}")
            print()
        
        print("2. LENIENT MODE (skip_on_error=True)")
        print("-" * 40)
        print("This mode attempts to load as much data as possible, logging issues.")
        print()
        
        loader_lenient = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        print(f"✓ Successfully initialized loader")
        print(f"  - Total valid samples: {len(loader_lenient)}")
        print(f"  - Corrupted samples skipped: {len(loader_lenient._corrupted_samples)}")
        print(f"  - Validation errors found: {len(loader_lenient._validation_errors)}")
        print()
        
        # Demonstrate safe_get_item
        print("3. SAFE LOADING (safe_get_item)")
        print("-" * 40)
        print("This method returns None for problematic samples instead of raising errors.")
        print()
        
        for i in range(min(3, len(loader_lenient))):
            sample = loader_lenient.safe_get_item(i)
            if sample:
                print(f"✓ Sample {i}: {sample['sample_id']}")
            else:
                print(f"✗ Sample {i}: Failed to load")
        
        # Try invalid index
        print(f"Attempting invalid index (999):")
        sample = loader_lenient.safe_get_item(999)
        if sample is None:
            print("  ✓ Returned None as expected")
        print()
        
        print("4. DETAILED ERROR INFORMATION")
        print("-" * 40)
        summary = loader_lenient.get_error_summary()
        
        print(f"Total errors encountered: {summary['total_errors']}")
        print(f"Corrupted samples: {summary['corrupted_samples']}")
        print()
        
        print("Error type breakdown:")
        for error_type, count in summary['error_types'].items():
            print(f"  - {error_type}: {count}")
        print()
        
        if summary['sample_errors']:
            print("Example errors (first 3):")
            for i, error in enumerate(summary['sample_errors'][:3], 1):
                print(f"  {i}. Type: {error.get('type', 'unknown')}")
                if 'sample_id' in error:
                    print(f"     Sample: {error['sample_id']}")
                if 'path' in error:
                    print(f"     Path: {Path(error['path']).name}")
                if 'error' in error:
                    print(f"     Details: {error['error']}")
        
        print()
        print("5. KEY FEATURES DEMONSTRATED")
        print("-" * 40)
        print("✓ Corrupted JSON handling with line numbers")
        print("✓ Missing file detection and reporting")
        print("✓ Malformed annotation handling")
        print("✓ Mismatched data validation")
        print("✓ Alternative file extension discovery")
        print("✓ Empty/corrupted file detection")
        print("✓ Comprehensive error summaries")
        print("✓ Safe loading mode for training resilience")
        print()
        
        print("=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)


def demonstrate_corrupted_json():
    """Demonstrate handling of corrupted JSON files."""
    
    print()
    print("CORRUPTED JSON HANDLING DEMONSTRATION")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        video_dir = base_path / "videos"
        video_dir.mkdir()
        
        # Create various corrupted JSON files
        test_cases = [
            ("invalid_syntax.json", '{invalid json: true, "missing": quotes}'),
            ("truncated.json", '[{"video_id": "test", "timestamps": [[0, 5'),
            ("wrong_encoding.json", b'\xff\xfe{"test": "data"}'),  # UTF-16 BOM
            ("empty.json", ""),
            ("not_json.json", "This is plain text, not JSON")
        ]
        
        for filename, content in test_cases:
            anno_file = base_path / filename
            
            if isinstance(content, bytes):
                anno_file.write_bytes(content)
            else:
                anno_file.write_text(content)
            
            config = {
                "name": f"test_{filename}",
                "path": str(video_dir),
                "annotation_file": str(anno_file)
            }
            
            print(f"\nTesting: {filename}")
            try:
                loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
                print(f"  ✓ Handled gracefully (loaded with warnings)")
            except DataLoadError as e:
                error_msg = str(e)
                if "line" in error_msg and "column" in error_msg:
                    print(f"  ✓ Detailed error location provided")
                elif "encoding" in error_msg.lower():
                    print(f"  ✓ Encoding issue detected")
                elif "JSON" in error_msg:
                    print(f"  ✓ JSON format issue detected")
                print(f"     Error: {error_msg[:100]}...")


if __name__ == "__main__":
    print("Starting robust loader demonstration...\n")
    
    try:
        demonstrate_robust_loading()
        demonstrate_corrupted_json()
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
The robust loaders provide:

1. **Early Detection**: Issues are caught during initialization, not during training
2. **Detailed Context**: Errors include file paths, line numbers, and sample IDs
3. **Graceful Degradation**: Training can continue with valid samples
4. **Recovery Mechanisms**: Automatic handling of common issues
5. **Comprehensive Reporting**: Full error summaries for debugging

This ensures your training pipeline is resilient to real-world data quality issues.
        """)
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()