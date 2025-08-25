# tests/dataloaders/test_edge_cases.py

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from core.dataloaders.robust_base_loader import RobustBaseLoader, DataLoadError
from core.dataloaders.robust_activitynet_loader import RobustActivityNetCaptionsLoader


class TestEdgeCases:
    """Test suite for edge cases and error handling in data loaders."""
    
    # Test corrupted JSON files
    
    def test_corrupted_json_file(self, tmp_path):
        """Test handling of corrupted JSON files."""
        # Create a corrupted JSON file
        annotation_file = tmp_path / "corrupted.json"
        annotation_file.write_text("{invalid json: true, missing quotes}")
        
        config = {
            "name": "test_dataset",
            "path": str(tmp_path / "videos"),
            "annotation_file": str(annotation_file)
        }
        
        # Create video directory
        (tmp_path / "videos").mkdir()
        
        # Should raise DataLoadError with helpful message
        with pytest.raises(DataLoadError) as exc_info:
            loader = RobustActivityNetCaptionsLoader(config)
        
        error = exc_info.value
        assert "Invalid JSON format" in str(error)
        assert "line" in str(error)  # Should include line number
        assert str(annotation_file) in str(error)
    
    def test_json_with_wrong_encoding(self, tmp_path):
        """Test handling of JSON files with non-UTF8 encoding."""
        # Create a JSON file with latin-1 encoding
        annotation_file = tmp_path / "latin1.json"
        data = [{"video_id": "test", "text": "café"}]  # Contains non-ASCII
        
        with open(annotation_file, 'w', encoding='latin-1') as f:
            json.dump(data, f)
        
        config = {
            "name": "test_dataset",
            "path": str(tmp_path / "videos"),
            "annotation_file": str(annotation_file),
        }
        
        # Create video directory
        (tmp_path / "videos").mkdir()
        
        # Should handle gracefully with warning
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        # Should load successfully (with warning logged)
        assert len(loader._index) == 0  # No videos, but JSON loaded
    
    def test_empty_json_file(self, tmp_path):
        """Test handling of empty JSON files."""
        annotation_file = tmp_path / "empty.json"
        annotation_file.write_text("")
        
        config = {
            "name": "test_dataset",
            "path": str(tmp_path / "videos"),
            "annotation_file": str(annotation_file)
        }
        
        (tmp_path / "videos").mkdir()
        
        with pytest.raises(DataLoadError) as exc_info:
            loader = RobustActivityNetCaptionsLoader(config)
        
        assert "Invalid JSON format" in str(exc_info.value)
    
    def test_json_with_wrong_structure(self, tmp_path):
        """Test handling of JSON with unexpected structure."""
        # ActivityNet expects an array, provide an object
        annotation_file = tmp_path / "wrong_structure.json"
        annotation_file.write_text('{"not": "an array"}')
        
        config = {
            "name": "test_dataset",
            "path": str(tmp_path / "videos"),
            "annotation_file": str(annotation_file)
        }
        
        (tmp_path / "videos").mkdir()
        
        with pytest.raises(DataLoadError) as exc_info:
            loader = RobustActivityNetCaptionsLoader(config)
        
        assert "Expected JSON array" in str(exc_info.value)
    
    # Test missing files
    
    def test_missing_video_files(self, tmp_path):
        """Test handling of missing video files referenced in annotations."""
        # Create valid annotation but no video files
        annotations = [
            {
                "video_id": "missing_video_1",
                "video": "missing_video_1.mp4",
                "timestamps": [[0, 5], [5, 10]],
                "sentences": ["First", "Second"],
                "duration": 10.0
            },
            {
                "video_id": "missing_video_2",
                "video": "missing_video_2.mp4",
                "timestamps": [[0, 3]],
                "sentences": ["Only one"],
                "duration": 3.0
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        config = {
            "name": "test_dataset",
            "path": str(tmp_path / "videos"),
            "annotation_file": str(annotation_file)
        }
        
        (tmp_path / "videos").mkdir()
        
        # With skip_on_error=True, should initialize but with 0 samples
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        assert len(loader) == 0
        assert len(loader._validation_errors) == 2
        
        # Check error summary
        summary = loader.get_error_summary()
        assert summary['total_errors'] == 2
        assert 'missing_file' in summary['error_types']
    
    def test_corrupted_video_file(self, tmp_path):
        """Test handling of corrupted (empty) video files."""
        annotations = [
            {
                "video_id": "empty_video",
                "video": "empty_video.mp4",
                "timestamps": [[0, 5]],
                "sentences": ["Test"],
                "duration": 5.0
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        
        # Create empty video file (0 bytes)
        (video_dir / "empty_video.mp4").touch()
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        assert len(loader) == 0
        
        # Check that error was logged
        errors = loader.get_error_summary()
        assert 'empty_video' in errors['error_types']
    
    # Test malformed annotations
    
    def test_malformed_annotations(self, tmp_path):
        """Test handling of malformed annotation entries."""
        annotations = [
            # Valid annotation
            {
                "video_id": "good_video",
                "video": "good.mp4",
                "timestamps": [[0, 5]],
                "sentences": ["Good"],
                "duration": 5.0
            },
            # Missing video_id
            {
                "video": "no_id.mp4",
                "timestamps": [[0, 5]],
                "sentences": ["No ID"]
            },
            # Mismatched timestamps and sentences
            {
                "video_id": "mismatch",
                "video": "mismatch.mp4",
                "timestamps": [[0, 5], [5, 10]],
                "sentences": ["Only one sentence"],
                "duration": 10.0
            },
            # Not a dictionary
            "This is not a valid annotation",
            # Missing critical fields
            {
                "video_id": "incomplete",
                "video": "incomplete.mp4"
                # Missing timestamps and sentences
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "good.mp4").write_bytes(b"fake video content")
        (video_dir / "mismatch.mp4").write_bytes(b"fake video content")
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        
        # Should only load the valid video
        assert len(loader) == 2  # good_video and mismatch (despite the mismatch)
        
        # Check validation errors
        assert len(loader._validation_errors) > 0
        error_types = {e['type'] for e in loader._validation_errors}
        assert 'mismatched_timestamps_sentences' in error_types
    
    def test_invalid_timestamp_formats(self, tmp_path):
        """Test handling of invalid timestamp formats."""
        annotations = [
            {
                "video_id": "bad_timestamps",
                "video": "bad_timestamps.mp4",
                "timestamps": [
                    [0, 5],  # Valid
                    "not a list",  # Invalid format
                    [1],  # Wrong length
                    [-5, 10],  # Invalid range (negative)
                    [10, 5],  # Invalid range (end < start)
                    ["a", "b"]  # Non-numeric
                ],
                "sentences": ["One", "Two", "Three", "Four", "Five", "Six"],
                "duration": 20.0
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "bad_timestamps.mp4").write_bytes(b"fake video")
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        assert len(loader) == 1
        
        # Get the item and check that only valid timestamp was kept
        sample = loader.get_item(0)
        events = sample['annotations']['timed_events']
        assert len(events) == 1  # Only the first valid timestamp
        assert events[0]['timestamp_sec'] == [0.0, 5.0]
    
    # Test recovery mechanisms
    
    def test_alternative_video_extensions(self, tmp_path):
        """Test finding videos with alternative extensions."""
        annotations = [
            {
                "video_id": "test_video",
                "video": "test_video.mp4",  # Says mp4
                "timestamps": [[0, 5]],
                "sentences": ["Test"],
                "duration": 5.0
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        
        # Create video with different extension
        (video_dir / "test_video.mkv").write_bytes(b"fake video")
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = RobustActivityNetCaptionsLoader(config)
        assert len(loader) == 1
        
        # Should find and use the .mkv file
        sample = loader.get_item(0)
        assert "test_video.mkv" in sample['media_path']
    
    def test_skip_on_error_mode(self, tmp_path):
        """Test that skip_on_error mode allows training to continue."""
        # Create mix of valid and invalid annotations
        annotations = [
            {
                "video_id": "valid",
                "video": "valid.mp4",
                "timestamps": [[0, 5]],
                "sentences": ["Valid"],
                "duration": 5.0
            },
            {
                "video_id": "missing",
                "video": "missing.mp4",
                "timestamps": [[0, 5]],
                "sentences": ["Missing"],
                "duration": 5.0
            }
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "valid.mp4").write_bytes(b"fake video")
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        # With skip_on_error=False, accessing missing would fail
        loader_strict = RobustActivityNetCaptionsLoader(config, skip_on_error=False)
        assert len(loader_strict) == 1  # Only valid video
        
        # With skip_on_error=True, should handle gracefully
        loader_lenient = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        assert len(loader_lenient) == 1  # Still only valid video
        
        # safe_get_item should return None for invalid indices
        assert loader_lenient.safe_get_item(0) is not None
        assert loader_lenient.safe_get_item(999) is None  # Out of range
    
    def test_error_summary_reporting(self, tmp_path):
        """Test comprehensive error summary reporting."""
        # Create dataset with various issues
        annotations = [
            {"video_id": f"video_{i}", "video": f"video_{i}.mp4"}
            for i in range(5)
        ]
        
        annotation_file = tmp_path / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        
        # Only create some videos
        (video_dir / "video_0.mp4").write_bytes(b"valid")
        (video_dir / "video_1.mp4").touch()  # Empty file
        # video_2 missing
        # video_3 missing
        (video_dir / "video_4.txt").touch()  # Wrong extension
        
        config = {
            "name": "test_dataset",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = RobustActivityNetCaptionsLoader(config, skip_on_error=True)
        
        summary = loader.get_error_summary()
        assert summary['total_errors'] > 0
        assert 'error_types' in summary
        assert 'sample_errors' in summary
        
        # Should have various error types
        error_types = summary['error_types']
        assert any(t in error_types for t in ['missing_file', 'empty_video', 'incomplete_annotation'])