# tests/dataloaders/test_assembly101_loader.py

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from core.dataloaders.assembly101_loader import Assembly101Loader


class TestAssembly101Loader:
    """Test suite for Assembly101Loader."""

    @pytest.fixture
    def mock_csv_data(self):
        """Create mock CSV data in Assembly101 format."""
        data = {
            'id': ['0000000', '0000001', '0000002', '0000003'],
            'video': [
                'session1/C10095_rgb.mp4',
                'session1/C10115_rgb.mp4',
                'session2/C10118_rgb.mp4',
                'missing_session/C10119_rgb.mp4'  # This video won't exist
            ],
            'start_frame': ['000000135', '000000200', '000000050', '000000100'],
            'end_frame': ['000000168', '000000250', '000000100', '000000150'],
            'action_id': ['0010', '0003', '0019', '0182'],
            'verb_id': ['0018', '0000', '0000', '0004'],
            'noun_id': ['0027', '0002', '0005', '0026'],
            'action_cls': ['clap hand', 'pick up screwdriver', 'pick up finished toy', 'unscrew track'],
            'verb_cls': ['clap', 'pick up', 'pick up', 'unscrew'],
            'noun_cls': ['hand', 'screwdriver', 'finished toy', 'track'],
            'toy_id': ['b06b', 'c01a', '-', 'd12c'],
            'toy_name': ['-', 'truck', '-', 'train'],
            'is_shared': [0, 1, 0, 1],
            'is_RGB': [1, 1, 1, 1]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        # Create temporary directories and files
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        
        annotation_dir = tmp_path / "annotations"
        annotation_dir.mkdir()
        
        annotation_file = annotation_dir / "train.csv"
        
        return {
            "name": "assembly101_test",
            "path": str(video_dir),
            "annotation_file": str(annotation_file)
        }

    def test_build_index_success(self, mock_config, mock_csv_data, tmp_path):
        """Test successful index building."""
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        mock_csv_data.to_csv(annotation_file, index=False)
        
        # Create video directory structure with subdirectories
        video_dir = Path(mock_config["path"])
        
        # Create subdirectories and video files
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "C10095_rgb.mp4").touch()
        (session1_dir / "C10115_rgb.mp4").touch()
        
        session2_dir = video_dir / "session2"
        session2_dir.mkdir()
        (session2_dir / "C10118_rgb.mp4").touch()
        
        # missing_session directory is not created
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        
        # Check that index only contains segments with existing videos
        assert len(loader) == 3  # 3 out of 4 segments have valid videos
        
        # Check that the index contains the correct data
        assert loader._index[0]['video'] == 'session1/C10095_rgb.mp4'
        assert loader._index[0]['action_cls'] == 'clap hand'

    def test_build_index_missing_annotation_file(self, mock_config):
        """Test error when annotation file is missing."""
        # Don't create the annotation file
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            Assembly101Loader(mock_config)

    def test_build_index_missing_video_directory(self, mock_config, mock_csv_data, tmp_path):
        """Test error when video directory is missing."""
        # Create annotation file
        annotation_file = Path(mock_config["annotation_file"])
        mock_csv_data.to_csv(annotation_file, index=False)
        
        # Set path to non-existent directory
        mock_config["path"] = "/non/existent/path"
        
        with pytest.raises(FileNotFoundError, match="Video directory not found"):
            Assembly101Loader(mock_config)

    def test_build_index_missing_config_keys(self):
        """Test error when required config keys are missing."""
        # Missing annotation_file
        config1 = {"name": "test", "path": "/some/path"}
        with pytest.raises(ValueError, match="requires 'annotation_file'"):
            Assembly101Loader(config1)
        
        # Missing path
        config2 = {"name": "test", "annotation_file": "/some/file.csv"}
        # Need to mock the annotation file to exist
        with patch('pathlib.Path.is_file', return_value=True):
            mock_df = pd.DataFrame({'video': [], 'start_frame': [], 'end_frame': []})
            with patch('pandas.read_csv', return_value=mock_df):
                with pytest.raises(ValueError, match="requires 'path'"):
                    Assembly101Loader(config2)

    def test_get_item_success(self, mock_config, mock_csv_data, tmp_path):
        """Test successful retrieval of a sample."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        mock_csv_data.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        video1_path = session1_dir / "C10095_rgb.mp4"
        video1_path.touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        
        # Get first item
        sample = loader.get_item(0)
        
        # Verify structure
        assert sample["source_dataset"] == "assembly101_test"
        # Note: pandas may convert string '0000000' to int 0
        assert sample["sample_id"] in ["segment_0000000", "segment_0"]
        assert sample["media_type"] == "video"
        assert sample["media_path"] == str((video_dir / "session1/C10095_rgb.mp4").resolve())
        
        # Verify action segment annotations
        action_segment = sample["annotations"]["action_segment"]
        assert action_segment["start_frame"] == 135
        assert action_segment["end_frame"] == 168
        assert action_segment["duration_frames"] == 34  # 168 - 135 + 1
        assert action_segment["action"] == "clap hand"
        assert action_segment["verb"] == "clap"
        assert action_segment["noun"] == "hand"
        # pandas may convert string IDs to integers
        assert action_segment["action_id"] in ["0010", 10]
        assert action_segment["verb_id"] in ["0018", 18]
        assert action_segment["noun_id"] in ["0027", 27]
        assert action_segment["toy_id"] == "b06b"
        
        # Verify metadata
        assert sample["annotations"]["is_shared"] == False
        assert sample["annotations"]["is_RGB"] == True
        # pandas may convert string '0000000' to int 0
        assert sample["annotations"]["annotation_id"] in ["0000000", 0]

    def test_get_item_with_toy_info(self, mock_config, tmp_path):
        """Test retrieval of sample with toy information."""
        # Create data with toy info
        data = {
            'id': ['0000001'],
            'video': ['session1/C10115_rgb.mp4'],
            'start_frame': ['000000200'],
            'end_frame': ['000000250'],
            'action_cls': ['pick up screwdriver'],
            'verb_cls': ['pick up'],
            'noun_cls': ['screwdriver'],
            'toy_id': ['c01a'],
            'toy_name': ['truck'],
            'is_shared': [1],
            'is_RGB': [1]
        }
        df = pd.DataFrame(data)
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        df.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "C10115_rgb.mp4").touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        sample = loader.get_item(0)
        
        # Verify toy information
        action_segment = sample["annotations"]["action_segment"]
        assert action_segment["toy_id"] == "c01a"
        assert action_segment["toy_name"] == "truck"
        assert sample["annotations"]["is_shared"] == True

    def test_get_item_without_toy_info(self, mock_config, tmp_path):
        """Test retrieval of sample without toy information (marked as '-')."""
        # Create data without toy info
        data = {
            'video': ['session1/test.mp4'],
            'start_frame': [50],
            'end_frame': [100],
            'action_cls': ['some action'],
            'toy_id': ['-'],
            'toy_name': ['-']
        }
        df = pd.DataFrame(data)
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        df.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "test.mp4").touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        sample = loader.get_item(0)
        
        # Verify toy information is not included when marked as '-'
        action_segment = sample["annotations"]["action_segment"]
        assert "toy_id" not in action_segment
        assert "toy_name" not in action_segment

    def test_get_item_file_not_found(self, mock_config, mock_csv_data, tmp_path):
        """Test error when video file is not found during get_item."""
        # Setup annotation file
        annotation_file = Path(mock_config["annotation_file"])
        mock_csv_data.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        video_path = session1_dir / "C10095_rgb.mp4"
        video_path.touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        
        # Delete the video file after initialization
        video_path.unlink()
        
        # Should raise FileNotFoundError when trying to get item
        with pytest.raises(FileNotFoundError, match="Media file not found"):
            loader.get_item(0)

    def test_frame_number_conversion(self, mock_config, tmp_path):
        """Test conversion of string frame numbers with leading zeros."""
        # Create data with string frame numbers
        data = {
            'video': ['session1/test.mp4'],
            'start_frame': ['000000042'],  # String with leading zeros
            'end_frame': ['000000142']
        }
        df = pd.DataFrame(data)
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        df.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "test.mp4").touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        sample = loader.get_item(0)
        
        # Verify frame numbers are correctly converted to integers
        action_segment = sample["annotations"]["action_segment"]
        assert action_segment["start_frame"] == 42
        assert action_segment["end_frame"] == 142
        assert action_segment["duration_frames"] == 101

    def test_sample_id_generation(self, mock_config, tmp_path):
        """Test sample ID generation with and without annotation ID."""
        # Create data without id column
        data_no_id = {
            'video': ['session1/test.mp4'],
            'start_frame': [100],
            'end_frame': [200]
        }
        df = pd.DataFrame(data_no_id)
        
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        df.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "test.mp4").touch()
        
        # Initialize loader
        loader = Assembly101Loader(mock_config)
        sample = loader.get_item(0)
        
        # Sample ID should be generated from video name and frames
        assert sample["sample_id"] == "test_000100_000200"

    def test_len_method(self, mock_config, mock_csv_data, tmp_path):
        """Test the __len__ method."""
        # Setup files
        annotation_file = Path(mock_config["annotation_file"])
        mock_csv_data.to_csv(annotation_file, index=False)
        
        video_dir = Path(mock_config["path"])
        session1_dir = video_dir / "session1"
        session1_dir.mkdir()
        (session1_dir / "C10095_rgb.mp4").touch()
        (session1_dir / "C10115_rgb.mp4").touch()
        
        loader = Assembly101Loader(mock_config)
        
        # Should return number of valid segments (2 videos exist for 3 segments)
        assert len(loader) == 2