# tests/dataloaders/test_didemo_loader.py

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from core.dataloaders.didemo_loader import DiDeMoLoader


class TestDiDeMoLoader:
    """Test suite for DiDeMoLoader."""

    @pytest.fixture
    def mock_annotations(self):
        """Create mock annotation data in DiDeMo format."""
        return [
            {
                "description": "camera zooms in on baby for the first time.",
                "reference_description": "Baby close-up",
                "times": [[1, 1], [1, 2], [1, 1], [1, 1]],  # Multiple annotators
                "video": "test_video_1.avi",
                "context": ["baby", "zoom"],
                "annotation_id": "30105",
                "train_times": [[1, 1]]  # Consensus timestamp
            },
            {
                "description": "the man in striped shirt looks away.",
                "reference_description": "",
                "times": [[0, 0], [3, 3], [0, 0], [0, 1]],
                "video": "test_video_2.mov",
                "context": [],
                "annotation_id": "30106",
                "train_times": [[0, 0]]
            },
            {
                "description": "a whale seen breaching the water.",
                "reference_description": "",
                "times": [[2, 2]],  # Single annotator
                "video": "missing_video.m4v",  # This video won't exist
                "context": [],
                "annotation_id": "30107",
                "train_times": []  # No train times
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
        
        annotation_file = annotation_dir / "didemo_train.json"
        
        return {
            "name": "didemo_test",
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
        (video_dir / "test_video_1.avi").touch()
        (video_dir / "test_video_2.mov").touch()
        # missing_video.m4v is not created
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        
        # Check that index only contains moments with existing videos
        assert len(loader) == 2
        assert loader._index[0]['video'] == "test_video_1.avi"
        assert loader._index[1]['video'] == "test_video_2.mov"

    def test_build_index_with_different_extensions(self, mock_config, mock_annotations, tmp_path):
        """Test that loader finds videos with different extensions."""
        # Modify annotations to have wrong extension
        mock_annotations[0]['video'] = "test_video_1.mp4"  # Says mp4
        
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        # Create video with different extension
        video_dir = Path(mock_config["path"])
        (video_dir / "test_video_1.avi").touch()  # Actually avi
        (video_dir / "test_video_2.mov").touch()
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        
        # Should still find the video with different extension
        assert len(loader) == 2
        assert loader._index[0]['video'] == "test_video_1.avi"  # Updated to actual extension

    def test_build_index_missing_annotation_file(self, mock_config):
        """Test error when annotation file is missing."""
        # Don't create the annotation file
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            DiDeMoLoader(mock_config)

    def test_build_index_missing_video_directory(self, mock_config, mock_annotations, tmp_path):
        """Test error when video directory is missing."""
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        # Set path to non-existent directory
        mock_config["path"] = "/non/existent/path"
        
        with pytest.raises(FileNotFoundError, match="Video directory not found"):
            DiDeMoLoader(mock_config)

    def test_build_index_missing_config_keys(self):
        """Test error when required config keys are missing."""
        # Missing annotation_file
        config1 = {"name": "test", "path": "/some/path"}
        with pytest.raises(ValueError, match="requires 'annotation_file'"):
            DiDeMoLoader(config1)
        
        # Missing path
        config2 = {"name": "test", "annotation_file": "/some/file.json"}
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('builtins.open', mock_open(read_data='[]')):
                with pytest.raises(ValueError, match="requires 'path'"):
                    DiDeMoLoader(config2)

    def test_get_item_with_train_times(self, mock_config, mock_annotations, tmp_path):
        """Test retrieval of a sample with train_times (consensus)."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        video1_path = video_dir / "test_video_1.avi"
        video1_path.touch()
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        
        # Get first item
        sample = loader.get_item(0)
        
        # Verify structure
        assert sample["source_dataset"] == "didemo_test"
        assert "test_video_1" in sample["sample_id"]
        assert sample["media_type"] == "video"
        assert sample["media_path"] == str(video1_path.resolve())
        
        # Verify moment annotation
        moment = sample["annotations"]["moment"]
        assert moment["segment_indices"] == [1, 1]
        assert moment["timestamp_sec"] == [5, 10]  # Segment 1 = seconds 5-10
        assert moment["description"] == "camera zooms in on baby for the first time."
        
        # Verify additional metadata
        assert sample["annotations"]["reference_description"] == "Baby close-up"
        assert sample["annotations"]["context"] == ["baby", "zoom"]
        assert sample["annotations"]["annotation_id"] == "30105"
        
        # Verify all annotations are included
        assert len(sample["annotations"]["all_annotations"]) == 4

    def test_get_item_without_train_times(self, mock_config, tmp_path):
        """Test retrieval when train_times is empty (uses first annotator)."""
        # Create annotation without train_times
        annotations = [{
            "description": "test moment",
            "times": [[2, 3], [2, 2]],  # Multiple annotators
            "video": "test.mp4",
            "annotation_id": "123",
            "train_times": []  # Empty
        }]
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = Path(mock_config["path"])
        (video_dir / "test.mp4").touch()
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        sample = loader.get_item(0)
        
        # Should use first annotator's timestamp
        moment = sample["annotations"]["moment"]
        assert moment["segment_indices"] == [2, 3]
        assert moment["timestamp_sec"] == [10, 20]  # Segments 2-3 = seconds 10-20

    def test_get_item_no_timestamps(self, mock_config, tmp_path):
        """Test handling of missing timestamps."""
        # Create annotation with no timestamps
        annotations = [{
            "description": "test moment",
            "video": "test.mp4",
            "annotation_id": "123",
            "times": [],
            "train_times": []
        }]
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = Path(mock_config["path"])
        (video_dir / "test.mp4").touch()
        
        # Initialize loader
        with patch('core.dataloaders.didemo_loader.logger') as mock_logger:
            loader = DiDeMoLoader(mock_config)
            sample = loader.get_item(0)
            
            # Should log a warning
            mock_logger.warning.assert_called()
            
            # Should use default timestamp
            moment = sample["annotations"]["moment"]
            assert moment["timestamp_sec"] == [0, 5]  # Default 5-second segment

    def test_get_item_file_not_found(self, mock_config, mock_annotations, tmp_path):
        """Test error when video file is not found during get_item."""
        # Setup annotation file
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        video_path = video_dir / "test_video_1.avi"
        video_path.touch()
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        
        # Delete the video file after initialization
        video_path.unlink()
        
        # Should raise FileNotFoundError when trying to get item
        with pytest.raises(FileNotFoundError, match="Media file not found"):
            loader.get_item(0)

    def test_timestamp_conversion(self, mock_config, tmp_path):
        """Test correct conversion of segment indices to seconds."""
        # Create annotations with various timestamps
        annotations = [
            {"description": "a", "video": "v1.mp4", "train_times": [[0, 0]]},  # 0-5 sec
            {"description": "b", "video": "v2.mp4", "train_times": [[1, 2]]},  # 5-15 sec
            {"description": "c", "video": "v3.mp4", "train_times": [[3, 5]]},  # 15-30 sec
        ]
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f)
        
        video_dir = Path(mock_config["path"])
        for ann in annotations:
            (video_dir / ann['video']).touch()
        
        # Initialize loader
        loader = DiDeMoLoader(mock_config)
        
        # Check timestamp conversions
        sample0 = loader.get_item(0)
        assert sample0["annotations"]["moment"]["timestamp_sec"] == [0, 5]
        
        sample1 = loader.get_item(1)
        assert sample1["annotations"]["moment"]["timestamp_sec"] == [5, 15]
        
        sample2 = loader.get_item(2)
        assert sample2["annotations"]["moment"]["timestamp_sec"] == [15, 30]

    def test_len_method(self, mock_config, mock_annotations, tmp_path):
        """Test the __len__ method."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        with open(annotation_file, 'w') as f:
            json.dump(mock_annotations, f)
        
        video_dir = Path(mock_config["path"])
        (video_dir / "test_video_1.avi").touch()
        (video_dir / "test_video_2.mov").touch()
        
        loader = DiDeMoLoader(mock_config)
        
        # Should return number of valid moments (with existing videos)
        assert len(loader) == 2