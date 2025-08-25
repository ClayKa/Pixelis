# tests/dataloaders/test_mot_loader.py

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
from collections import defaultdict

from core.dataloaders.mot_loader import MotLoader


class TestMotLoader:
    """Comprehensive test suite for MotLoader"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_mot_dataset(self, temp_dir):
        """Create a mock MOT dataset structure with test data"""
        # Create sequence directories
        seq1_dir = temp_dir / "MOT20-01"
        seq2_dir = temp_dir / "MOT20-02"
        seq1_dir.mkdir()
        seq2_dir.mkdir()
        
        # Create img1 directories with mock images
        for seq_dir in [seq1_dir, seq2_dir]:
            img_dir = seq_dir / "img1"
            img_dir.mkdir()
            
            # Create mock image files
            for i in range(1, 6):  # 5 frames each
                img_file = img_dir / f"{i:06d}.jpg"
                img_file.touch()
        
        # Create gt directories and files
        for seq_dir in [seq1_dir, seq2_dir]:
            gt_dir = seq_dir / "gt"
            gt_dir.mkdir()
            gt_file = gt_dir / "gt.txt"
            
            # Write mock ground truth data
            if seq_dir.name == "MOT20-01":
                gt_content = """1,1,100,200,50,100,1,1,1
1,2,200,300,60,120,1,1,1
2,1,105,205,50,100,1,1,1
2,2,205,305,60,120,1,1,1
3,1,110,210,50,100,1,1,1"""
            else:  # MOT20-02
                gt_content = """1,1,150,250,40,80,1,1,1
2,1,155,255,40,80,1,1,1
3,1,160,260,40,80,1,1,1"""
            
            gt_file.write_text(gt_content)
        
        # Create an invalid sequence (missing gt.txt)
        invalid_seq = temp_dir / "MOT20-03"
        invalid_seq.mkdir()
        invalid_img_dir = invalid_seq / "img1"
        invalid_img_dir.mkdir()
        (invalid_img_dir / "000001.jpg").touch()
        
        return temp_dir

    @pytest.fixture
    def valid_config(self, mock_mot_dataset):
        """Valid configuration for testing"""
        return {
            'name': 'test_mot',
            'path': str(mock_mot_dataset)
        }

    def test_init_success(self, valid_config):
        """Test successful initialization with valid config"""
        loader = MotLoader(valid_config)
        
        assert loader.source_name == 'test_mot'
        assert loader.base_path == Path(valid_config['path'])
        assert len(loader) == 2  # Should find 2 valid sequences

    def test_init_missing_required_config(self):
        """Test initialization fails with missing required config keys"""
        invalid_config = {'name': 'test_mot'}  # Missing 'path'
        
        with pytest.raises(ValueError, match="MotLoader config must include 'path'"):
            MotLoader(invalid_config)

    def test_init_invalid_paths(self):
        """Test initialization fails with invalid paths"""
        invalid_config = {
            'name': 'test_mot',
            'path': '/nonexistent/path'
        }
        
        with pytest.raises(FileNotFoundError, match="Base directory not found"):
            MotLoader(invalid_config)

    def test_init_path_not_directory(self, temp_dir):
        """Test initialization fails when path is not a directory"""
        file_path = temp_dir / "not_a_dir.txt"
        file_path.touch()
        
        invalid_config = {
            'name': 'test_mot',
            'path': str(file_path)
        }
        
        with pytest.raises(FileNotFoundError, match="Base path is not a directory"):
            MotLoader(invalid_config)

    def test_build_index_structure(self, valid_config):
        """Test that _build_index correctly identifies valid sequences"""
        loader = MotLoader(valid_config)
        
        # Should have exactly 2 valid sequences
        assert len(loader._index) == 2
        
        # Check that all items are Path objects
        for sequence_path in loader._index:
            assert isinstance(sequence_path, Path)
            assert sequence_path.is_dir()
        
        # Check sequence names
        sequence_names = [path.name for path in loader._index]
        assert "MOT20-01" in sequence_names
        assert "MOT20-02" in sequence_names
        assert "MOT20-03" not in sequence_names  # Invalid sequence should be excluded

    def test_get_item_basic_structure(self, valid_config):
        """Test basic structure of returned sample dictionary"""
        loader = MotLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check basic structure
        assert isinstance(sample, dict)
        assert 'source_dataset' in sample
        assert 'sample_id' in sample
        assert 'media_type' in sample
        assert 'media_path' in sample
        assert 'annotations' in sample
        
        # Check values
        assert sample['source_dataset'] == 'test_mot'
        assert sample['media_type'] == 'video'
        assert 'MOT20-0' in sample['sample_id']  # Either MOT20-01 or MOT20-02

    def test_get_item_mot_annotations(self, valid_config):
        """Test MOT-specific annotations in returned sample"""
        loader = MotLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check MOT annotations exist
        assert 'multi_object_tracking' in sample['annotations']
        mot_data = sample['annotations']['multi_object_tracking']
        
        # Check required fields
        assert 'sequence_id' in mot_data
        assert 'object_tracks' in mot_data
        assert 'num_objects' in mot_data
        assert 'num_frames' in mot_data
        assert 'track_statistics' in mot_data
        assert 'sequence_info' in mot_data
        
        # Check data types
        assert isinstance(mot_data['object_tracks'], dict)
        assert isinstance(mot_data['num_objects'], int)
        assert isinstance(mot_data['num_frames'], int)
        assert isinstance(mot_data['track_statistics'], dict)

    def test_get_item_object_tracks_structure(self, valid_config):
        """Test structure of parsed object tracks"""
        loader = MotLoader(valid_config)
        sample = loader.get_item(0)
        
        object_tracks = sample['annotations']['multi_object_tracking']['object_tracks']
        
        # Should have at least one track
        assert len(object_tracks) > 0
        
        # Check track structure
        for object_id, track in object_tracks.items():
            assert isinstance(object_id, int)
            assert isinstance(track, list)
            assert len(track) > 0
            
            # Check annotation structure
            for annotation in track:
                assert 'frame_id' in annotation
                assert 'bbox' in annotation
                assert 'confidence' in annotation
                assert 'class' in annotation
                assert 'visibility' in annotation
                
                # Check bbox format [left, top, width, height]
                bbox = annotation['bbox']
                assert isinstance(bbox, list)
                assert len(bbox) == 4
                assert all(isinstance(x, (int, float)) for x in bbox)

    def test_get_item_dataset_info(self, valid_config):
        """Test dataset_info section of returned sample"""
        loader = MotLoader(valid_config)
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        
        # Check required fields
        assert dataset_info['task_type'] == 'multi_object_tracking'
        assert dataset_info['source'] == 'MOT'
        assert dataset_info['suitable_for_tracking'] is True
        assert dataset_info['suitable_for_temporal_reasoning'] is True
        assert 'has_object_trajectories' in dataset_info
        assert 'has_frame_annotations' in dataset_info
        assert dataset_info['annotation_format'] == 'mot_format'

    def test_get_item_out_of_range(self, valid_config):
        """Test get_item raises IndexError for invalid indices"""
        loader = MotLoader(valid_config)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(999)

    def test_parse_ground_truth_success(self, valid_config, temp_dir):
        """Test successful parsing of ground truth file"""
        loader = MotLoader(valid_config)
        
        # Create a test gt.txt file
        gt_file = temp_dir / "test_gt.txt"
        gt_content = """1,1,100,200,50,100,1.0,1,1.0
2,1,105,205,50,100,0.9,1,0.8
1,2,200,300,60,120,1.0,1,1.0"""
        gt_file.write_text(gt_content)
        
        tracks = loader._parse_ground_truth(gt_file)
        
        # Should have 2 objects
        assert len(tracks) == 2
        assert 1 in tracks
        assert 2 in tracks
        
        # Check object 1 has 2 detections
        assert len(tracks[1]) == 2
        # Check object 2 has 1 detection
        assert len(tracks[2]) == 1
        
        # Check first detection of object 1
        detection = tracks[1][0]
        assert detection['frame_id'] == 1
        assert detection['bbox'] == [100.0, 200.0, 50.0, 100.0]
        assert detection['confidence'] == 1.0
        assert detection['class'] == 1
        assert detection['visibility'] == 1.0

    def test_parse_ground_truth_malformed_file(self, valid_config, temp_dir):
        """Test parsing handles malformed ground truth files gracefully"""
        loader = MotLoader(valid_config)
        
        # Create a malformed gt.txt file
        gt_file = temp_dir / "malformed_gt.txt"
        gt_file.write_text("invalid,data,format\n")
        
        tracks = loader._parse_ground_truth(gt_file)
        
        # Should return empty dict for malformed file
        assert tracks == {}

    def test_analyze_tracks_normal(self, valid_config):
        """Test track analysis with normal tracks"""
        loader = MotLoader(valid_config)
        
        # Create test tracks
        tracks = {
            1: [
                {'frame_id': 1, 'bbox': [100, 200, 50, 100], 'confidence': 1.0, 'class': 1, 'visibility': 1.0},
                {'frame_id': 2, 'bbox': [105, 205, 50, 100], 'confidence': 0.9, 'class': 1, 'visibility': 0.8}
            ],
            2: [
                {'frame_id': 1, 'bbox': [200, 300, 60, 120], 'confidence': 1.0, 'class': 1, 'visibility': 1.0}
            ]
        }
        
        stats = loader._analyze_tracks(tracks)
        
        assert stats['avg_track_length'] == 1.5  # (2+1)/2
        assert stats['min_track_length'] == 1
        assert stats['max_track_length'] == 2
        assert stats['total_detections'] == 3
        assert stats['unique_objects'] == 2
        assert stats['frame_span'] == 2  # Frames 1 and 2
        assert stats['objects_per_frame'] == 1.5  # 3 detections / 2 frames

    def test_analyze_tracks_empty(self, valid_config):
        """Test track analysis with empty tracks"""
        loader = MotLoader(valid_config)
        
        stats = loader._analyze_tracks({})
        
        assert stats['avg_track_length'] == 0.0
        assert stats['min_track_length'] == 0
        assert stats['max_track_length'] == 0
        assert stats['total_detections'] == 0
        assert stats['objects_per_frame'] == 0.0

    def test_get_samples_by_object_count(self, valid_config):
        """Test filtering samples by object count"""
        loader = MotLoader(valid_config)
        
        # Get samples with at least 1 object
        samples = loader.get_samples_by_object_count(min_objects=1)
        assert len(samples) > 0
        
        for sample in samples:
            num_objects = sample['annotations']['multi_object_tracking']['num_objects']
            assert num_objects >= 1
        
        # Test filtering with specific range
        samples = loader.get_samples_by_object_count(min_objects=2, max_objects=2)
        for sample in samples:
            num_objects = sample['annotations']['multi_object_tracking']['num_objects']
            assert num_objects == 2

    def test_get_samples_by_duration(self, valid_config):
        """Test filtering samples by sequence duration"""
        loader = MotLoader(valid_config)
        
        # Get samples with at least 1 frame
        samples = loader.get_samples_by_duration(min_frames=1)
        assert len(samples) > 0
        
        for sample in samples:
            num_frames = sample['annotations']['multi_object_tracking']['num_frames']
            assert num_frames >= 1
        
        # Test filtering with specific range
        samples = loader.get_samples_by_duration(min_frames=5, max_frames=5)
        for sample in samples:
            num_frames = sample['annotations']['multi_object_tracking']['num_frames']
            assert num_frames == 5

    def test_get_tracking_statistics(self, valid_config):
        """Test comprehensive tracking statistics"""
        loader = MotLoader(valid_config)
        
        stats = loader.get_tracking_statistics()
        
        # Check required fields
        assert 'total_sequences' in stats
        assert 'total_unique_objects' in stats
        assert 'total_detections' in stats
        assert 'total_frames' in stats
        assert 'avg_objects_per_sequence' in stats
        assert 'avg_frames_per_sequence' in stats
        assert 'avg_detections_per_frame' in stats
        assert 'sequence_length_distribution' in stats
        assert 'object_count_distribution' in stats
        
        # Check values make sense
        assert stats['total_sequences'] == len(loader)
        assert stats['total_unique_objects'] >= 0
        assert stats['total_detections'] >= 0
        assert stats['total_frames'] >= 0

    def test_get_sequence_by_name_success(self, valid_config):
        """Test getting sequence by name successfully"""
        loader = MotLoader(valid_config)
        
        # Get list of available sequences
        sequence_names = loader.list_sequence_names()
        assert len(sequence_names) == 2
        
        # Get a specific sequence
        sample = loader.get_sequence_by_name(sequence_names[0])
        
        assert sample['sample_id'] == sequence_names[0]
        assert 'multi_object_tracking' in sample['annotations']

    def test_get_sequence_by_name_not_found(self, valid_config):
        """Test getting sequence by name with invalid name"""
        loader = MotLoader(valid_config)
        
        with pytest.raises(ValueError, match="Sequence 'INVALID-SEQ' not found"):
            loader.get_sequence_by_name('INVALID-SEQ')

    def test_list_sequence_names(self, valid_config):
        """Test listing all sequence names"""
        loader = MotLoader(valid_config)
        
        names = loader.list_sequence_names()
        
        assert isinstance(names, list)
        assert len(names) == 2
        assert "MOT20-01" in names
        assert "MOT20-02" in names

    def test_len_method(self, valid_config):
        """Test __len__ method returns correct count"""
        loader = MotLoader(valid_config)
        
        assert len(loader) == 2
        assert len(loader) == len(loader._index)

    @patch('core.dataloaders.mot_loader.logger')
    def test_logging_calls(self, mock_logger, valid_config):
        """Test that appropriate logging calls are made"""
        loader = MotLoader(valid_config)
        
        # Check that info logs were called during initialization
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Scanning MOT dataset" in call for call in info_calls)
        assert any("Found" in call and "valid sequences" in call for call in info_calls)

    def test_missing_ground_truth_file(self, temp_dir):
        """Test handling of sequences with missing ground truth files"""
        # Create sequence with img1 but no gt directory
        seq_dir = temp_dir / "MOT20-MISSING-GT"
        seq_dir.mkdir()
        img_dir = seq_dir / "img1"
        img_dir.mkdir()
        (img_dir / "000001.jpg").touch()
        
        config = {
            'name': 'test_mot',
            'path': str(temp_dir)
        }
        
        loader = MotLoader(config)
        
        # Should not include the sequence with missing gt
        sequence_names = loader.list_sequence_names()
        assert "MOT20-MISSING-GT" not in sequence_names

    def test_empty_dataset_directory(self, temp_dir):
        """Test initialization with empty dataset directory"""
        config = {
            'name': 'test_mot',
            'path': str(temp_dir)
        }
        
        loader = MotLoader(config)
        
        # Should handle empty directory gracefully
        assert len(loader) == 0
        assert loader.get_tracking_statistics()['total_sequences'] == 0

    def test_parse_ground_truth_with_missing_values(self, valid_config, temp_dir):
        """Test parsing ground truth with missing/NaN values"""
        loader = MotLoader(valid_config)
        
        # Create gt.txt with missing values
        gt_file = temp_dir / "test_gt_missing.txt"
        gt_content = """1,1,100,200,50,100,1.0,,0.8
2,1,105,205,50,100,0.9,1,"""
        gt_file.write_text(gt_content)
        
        tracks = loader._parse_ground_truth(gt_file)
        
        # Should handle missing values gracefully
        assert len(tracks) == 1
        assert 1 in tracks
        assert len(tracks[1]) == 2
        
        # Check default values for missing data
        first_detection = tracks[1][0]
        assert first_detection['class'] == -1  # Default for missing class
        assert first_detection['visibility'] == 0.8
        
        second_detection = tracks[1][1]
        assert second_detection['class'] == 1
        assert second_detection['visibility'] == 1.0  # Default for missing visibility

    def test_track_temporal_ordering(self, valid_config, temp_dir):
        """Test that tracks are properly ordered by frame_id"""
        loader = MotLoader(valid_config)
        
        # Create gt.txt with out-of-order frames
        gt_file = temp_dir / "test_gt_unordered.txt"
        gt_content = """3,1,110,210,50,100,1.0,1,1.0
1,1,100,200,50,100,1.0,1,1.0
2,1,105,205,50,100,1.0,1,1.0"""
        gt_file.write_text(gt_content)
        
        tracks = loader._parse_ground_truth(gt_file)
        
        # Should have properly ordered track
        assert len(tracks) == 1
        assert 1 in tracks
        assert len(tracks[1]) == 3
        
        # Check temporal ordering
        frame_ids = [detection['frame_id'] for detection in tracks[1]]
        assert frame_ids == [1, 2, 3]  # Should be sorted

    def test_video_metadata_section(self, valid_config):
        """Test video_metadata section of returned sample"""
        loader = MotLoader(valid_config)
        sample = loader.get_item(0)
        
        video_metadata = sample['annotations']['video_metadata']
        
        # Check required fields
        assert 'frame_directory' in video_metadata
        assert 'total_frames' in video_metadata
        assert 'sequence_duration_frames' in video_metadata
        
        # Check that values make sense
        assert isinstance(video_metadata['total_frames'], int)
        assert video_metadata['total_frames'] > 0
        assert video_metadata['total_frames'] == video_metadata['sequence_duration_frames']