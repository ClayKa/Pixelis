# tests/dataloaders/test_activitynet_captions_loader.py

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from core.dataloaders.activitynet_captions_loader import ActivityNetCaptionsLoader


class TestActivityNetCaptionsLoader:
    """Test suite for ActivityNetCaptionsLoader."""

    @pytest.fixture
    def mock_annotations(self):
        """Create mock annotation data."""
        return [
            {
                "video_id": "v_test_video_1",
                "video": "v_test_video_1.mp4",
                "caption": "A person is doing something. They continue doing it. They finish.",
                "source": "ActivityNet_Captions",
                "duration": 120.5,
                "timestamps": [[0.0, 40.0], [35.0, 80.0], [75.0, 120.0]],
                "sentences": [
                    "A person is doing something.",
                    "They continue doing it.",
                    "They finish."
                ]
            },
            {
                "video_id": "v_test_video_2",
                "video": "v_test_video_2.mkv",
                "caption": "Another video caption.",
                "source": "ActivityNet_Captions", 
                "duration": 60.0,
                "timestamps": [[0.0, 30.0], [25.0, 60.0]],
                "sentences": [
                    "First part of video.",
                    "Second part of video."
                ]
            },
            {
                "video_id": "v_missing_video",
                "video": "v_missing_video.mp4",
                "caption": "This video file doesn't exist.",
                "source": "ActivityNet_Captions",
                "duration": 45.0,
                "timestamps": [[0.0, 45.0]],
                "sentences": ["Video that is missing."]
            }
        ]

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        # Create temporary directories and files
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        
        annotation_dir = tmp_path / "annotations"
        annotation_dir.mkdir()
        
        annotation_file = annotation_dir / "train.json"
        
        return {
            "name": "activitynet_captions_test",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }

    def test_build_index_success(self, mock_config, mock_annotations, tmp_path):
        """Test successful index building."""
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        # Create video files (only some exist)
        video_dir = Path(mock_config["path"])
        (video_dir / "v_test_video_1.mp4").touch()
        (video_dir / "v_test_video_2.mkv").touch()
        # v_missing_video.mp4 is not created
        
        # Initialize loader
        loader = ActivityNetCaptionsLoader(mock_config)
        
        # Check that index only contains videos that exist
        assert len(loader) == 2
        assert "v_test_video_1" in loader._index
        assert "v_test_video_2" in loader._index
        assert "v_missing_video" not in loader._index

    def test_build_index_missing_annotation_file(self, mock_config):
        """Test error when annotation file is missing."""
        # Don't create the annotation file
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            ActivityNetCaptionsLoader(mock_config)

    def test_build_index_missing_video_directory(self, mock_config, mock_annotations, tmp_path):
        """Test error when video directory is missing."""
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        # Set path to non-existent directory
        mock_config["path"] = "/non/existent/path"
        
        with pytest.raises(FileNotFoundError, match="Video directory not found"):
            ActivityNetCaptionsLoader(mock_config)

    def test_build_index_missing_config_keys(self):
        """Test error when required config keys are missing."""
        # Missing annotation_file
        config1 = {"name": "test", "path": "/some/path"}
        with pytest.raises(ValueError, match="requires 'annotation_file'"):
            ActivityNetCaptionsLoader(config1)
        
        # Missing path
        config2 = {"name": "test", "annotation_file": "/some/file.json"}
        # Need to mock the annotation file to exist
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('builtins.open', mock_open(read_data='[]')):
                with pytest.raises(ValueError, match="requires 'path'"):
                    ActivityNetCaptionsLoader(config2)

    def test_get_item_success(self, mock_config, mock_annotations, tmp_path):
        """Test successful retrieval of a sample."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        video1_path = video_dir / "v_test_video_1.mp4"
        video1_path.touch()
        
        # Initialize loader
        loader = ActivityNetCaptionsLoader(mock_config)
        
        # Get first item
        sample = loader.get_item(0)
        
        # Verify structure
        assert sample["source_dataset"] == "activitynet_captions_test"
        assert sample["sample_id"] == "v_test_video_1"
        assert sample["media_type"] == "video"
        assert sample["media_path"] == str(video1_path.resolve())
        
        # Verify annotations
        assert sample["annotations"]["duration_sec"] == 120.5
        assert sample["annotations"]["source"] == "ActivityNet_Captions"
        assert sample["annotations"]["full_caption"] == mock_annotations[0]["caption"]
        
        # Verify timed events
        events = sample["annotations"]["timed_events"]
        assert len(events) == 3
        assert events[0]["timestamp_sec"] == [0.0, 40.0]
        assert events[0]["description"] == "A person is doing something."
        assert events[1]["timestamp_sec"] == [35.0, 80.0]
        assert events[1]["description"] == "They continue doing it."
        assert events[2]["timestamp_sec"] == [75.0, 120.0]
        assert events[2]["description"] == "They finish."

    def test_get_item_with_mkv_video(self, mock_config, mock_annotations, tmp_path):
        """Test retrieval of sample with MKV video format."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        video1_path = video_dir / "v_test_video_1.mp4"
        video1_path.touch()
        video2_path = video_dir / "v_test_video_2.mkv"
        video2_path.touch()
        
        # Initialize loader
        loader = ActivityNetCaptionsLoader(mock_config)
        
        # Find index of video 2
        video2_index = loader._index.index("v_test_video_2")
        
        # Get the MKV video item
        sample = loader.get_item(video2_index)
        
        # Verify it found the MKV file
        assert sample["sample_id"] == "v_test_video_2"
        assert sample["media_path"] == str(video2_path.resolve())
        assert len(sample["annotations"]["timed_events"]) == 2

    def test_get_item_mismatched_timestamps_sentences(self, mock_config, tmp_path):
        """Test handling of mismatched timestamps and sentences."""
        # Create annotation with mismatched lengths
        bad_annotations = [{
            "video_id": "v_bad_video",
            "video": "v_bad_video.mp4",
            "duration": 100.0,
            "timestamps": [[0.0, 50.0], [40.0, 100.0]],  # 2 timestamps
            "sentences": ["Only one sentence"]  # 1 sentence
        }]
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(bad_annotations, f)
        
        video_dir = Path(mock_config["path"])
        (video_dir / "v_bad_video.mp4").touch()
        
        # Initialize loader
        with patch('core.dataloaders.activitynet_captions_loader.logger') as mock_logger:
            loader = ActivityNetCaptionsLoader(mock_config)
            sample = loader.get_item(0)
            
            # Should log a warning
            mock_logger.warning.assert_called()
            
            # Should handle gracefully by using minimum length
            assert len(sample["annotations"]["timed_events"]) == 1
            assert sample["annotations"]["timed_events"][0]["description"] == "Only one sentence"

    def test_get_item_file_not_found(self, mock_config, mock_annotations, tmp_path):
        """Test error when video file is not found during get_item."""
        # Setup annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        video_path = video_dir / "v_test_video_1.mp4"
        video_path.touch()
        
        # Initialize loader
        loader = ActivityNetCaptionsLoader(mock_config)
        
        # Delete the video file after initialization
        video_path.unlink()
        
        # Should raise FileNotFoundError when trying to get item
        with pytest.raises(FileNotFoundError, match="Media file not found"):
            loader.get_item(0)

    def test_len_method(self, mock_config, mock_annotations, tmp_path):
        """Test the __len__ method."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        (video_dir / "v_test_video_1.mp4").touch()
        (video_dir / "v_test_video_2.mkv").touch()
        
        loader = ActivityNetCaptionsLoader(mock_config)
        
        # Should return number of valid videos
        assert len(loader) == 2