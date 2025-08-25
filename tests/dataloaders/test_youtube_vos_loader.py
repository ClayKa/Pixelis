# tests/dataloaders/test_youtube_vos_loader.py

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from collections import defaultdict

from core.dataloaders.youtube_vos_loader import YouTubeVOSLoader


class TestYouTubeVOSLoader:
    """Comprehensive test suite for YouTubeVOSLoader"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_vos_dataset(self, temp_dir):
        """Create a mock YouTube-VOS dataset structure with test data"""
        # Create frames directory structure
        frames_dir = temp_dir / "JPEGImages"
        frames_dir.mkdir()
        
        # Create video directories with mock frames
        video_names = ["video_001", "video_002", "video_003"]
        
        for video_name in video_names:
            video_dir = frames_dir / video_name
            video_dir.mkdir()
            
            # Create mock frame files
            for i in range(5):  # 5 frames each
                frame_file = video_dir / f"{i:06d}.jpg"
                frame_file.touch()
        
        # Create instances.json with complete structure
        instances_data = {
            "videos": [
                {
                    "id": 1,
                    "width": 1920,
                    "height": 1080,
                    "length": 5,
                    "file_names": [f"video_001/{i:06d}.jpg" for i in range(5)]
                },
                {
                    "id": 2,
                    "width": 1280,
                    "height": 720,
                    "length": 5,
                    "file_names": [f"video_002/{i:06d}.jpg" for i in range(5)]
                },
                {
                    "id": 3,
                    "width": 1920,
                    "height": 1080,
                    "length": 5,
                    "file_names": [f"video_003/{i:06d}.jpg" for i in range(5)]
                }
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"},
                {"id": 3, "name": "dog"}
            ],
            "annotations": [
                {
                    "id": 101,
                    "video_id": 1,
                    "category_id": 1,
                    "segmentations": [
                        {"size": [1080, 1920], "counts": "PCQR1g0O2N1O100O10O010000001O"},  # Frame 0
                        {"size": [1080, 1920], "counts": "PCQR1g0O2N1O100O10O010000001O"},  # Frame 1
                        None,  # Frame 2 - object not visible
                        {"size": [1080, 1920], "counts": "PCQR1g0O2N1O100O10O010000001O"},  # Frame 3
                        {"size": [1080, 1920], "counts": "PCQR1g0O2N1O100O10O010000001O"}   # Frame 4
                    ],
                    "bboxes": [
                        [100, 200, 50, 100],  # Frame 0
                        [105, 205, 50, 100],  # Frame 1
                        None,                  # Frame 2
                        [110, 210, 50, 100],  # Frame 3
                        [115, 215, 50, 100]   # Frame 4
                    ],
                    "areas": [5000, 5000, None, 5000, 5000],
                    "iscrowd": 0
                },
                {
                    "id": 102,
                    "video_id": 1,
                    "category_id": 2,
                    "segmentations": [
                        {"size": [1080, 1920], "counts": "ZZQR1g0O2N1O100O10O010000001O"},
                        {"size": [1080, 1920], "counts": "ZZQR1g0O2N1O100O10O010000001O"},
                        {"size": [1080, 1920], "counts": "ZZQR1g0O2N1O100O10O010000001O"},
                        None,
                        None
                    ],
                    "bboxes": [
                        [200, 300, 60, 120],
                        [205, 305, 60, 120],
                        [210, 310, 60, 120],
                        None,
                        None
                    ],
                    "areas": [7200, 7200, 7200, None, None],
                    "iscrowd": 0
                },
                {
                    "id": 201,
                    "video_id": 2,
                    "category_id": 3,
                    "segmentations": [
                        {"size": [720, 1280], "counts": "ABCD1g0O2N1O100O10O010000001O"},
                        None,
                        {"size": [720, 1280], "counts": "ABCD1g0O2N1O100O10O010000001O"},
                        {"size": [720, 1280], "counts": "ABCD1g0O2N1O100O10O010000001O"},
                        {"size": [720, 1280], "counts": "ABCD1g0O2N1O100O10O010000001O"}
                    ],
                    "bboxes": [
                        [50, 100, 30, 60],
                        None,
                        [55, 105, 30, 60],
                        [60, 110, 30, 60],
                        [65, 115, 30, 60]
                    ],
                    "areas": [1800, None, 1800, 1800, 1800],
                    "iscrowd": 0
                }
                # video_003 has no annotations (test case for validation)
            ]
        }
        
        # Write instances.json
        instances_file = temp_dir / "instances.json"
        instances_file.write_text(json.dumps(instances_data))
        
        return {
            "frames_dir": frames_dir,
            "instances_file": instances_file,
            "video_names": video_names,
            "instances_data": instances_data
        }

    @pytest.fixture
    def valid_config(self, mock_vos_dataset):
        """Valid configuration for testing"""
        return {
            'name': 'test_vos',
            'path': str(mock_vos_dataset["frames_dir"]),
            'annotation_file': str(mock_vos_dataset["instances_file"])
        }

    def test_init_success(self, valid_config):
        """Test successful initialization with valid config"""
        loader = YouTubeVOSLoader(valid_config)
        
        assert loader.source_name == 'test_vos'
        assert loader.frames_path == Path(valid_config['path'])
        assert loader.annotation_file == Path(valid_config['annotation_file'])
        assert len(loader) == 2  # Only video_001 and video_002 have annotations

    def test_init_missing_required_config(self):
        """Test initialization fails with missing required config keys"""
        invalid_configs = [
            {'name': 'test_vos', 'annotation_file': '/some/path'},  # Missing 'path'
            {'name': 'test_vos', 'path': '/some/path'}  # Missing 'annotation_file'
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="YouTubeVOSLoader config must include"):
                YouTubeVOSLoader(config)

    def test_init_invalid_paths(self):
        """Test initialization fails with invalid paths"""
        invalid_config = {
            'name': 'test_vos',
            'path': '/nonexistent/frames/path',
            'annotation_file': '/nonexistent/instances.json'
        }
        
        with pytest.raises(FileNotFoundError, match="Frames directory not found"):
            YouTubeVOSLoader(invalid_config)

    def test_init_invalid_annotation_file(self, temp_dir):
        """Test initialization fails with invalid annotation file"""
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        invalid_config = {
            'name': 'test_vos',
            'path': str(frames_dir),
            'annotation_file': '/nonexistent/instances.json'
        }
        
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            YouTubeVOSLoader(invalid_config)

    def test_build_index_structure(self, valid_config):
        """Test that _build_index correctly processes the JSON and validates videos"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Should have exactly 2 valid videos (video_003 has no annotations)
        assert len(loader._index) == 2
        
        # Check that all items are integers (video IDs)
        for video_id in loader._index:
            assert isinstance(video_id, int)
        
        # Check video IDs are correct
        assert set(loader._index) == {1, 2}

    def test_metadata_loading(self, valid_config):
        """Test that metadata is properly loaded from instances.json"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Check videos loaded
        assert len(loader._video_id_to_info) == 3
        assert 1 in loader._video_id_to_info
        assert loader._video_id_to_info[1]['width'] == 1920
        assert loader._video_id_to_info[1]['height'] == 1080
        
        # Check categories loaded
        assert len(loader._category_id_to_name) == 3
        assert loader._category_id_to_name[1] == 'person'
        assert loader._category_id_to_name[2] == 'car'
        assert loader._category_id_to_name[3] == 'dog'
        
        # Check annotations grouped by video
        assert len(loader._video_id_to_annotations[1]) == 2  # video_001 has 2 objects
        assert len(loader._video_id_to_annotations[2]) == 1  # video_002 has 1 object
        assert len(loader._video_id_to_annotations[3]) == 0  # video_003 has no annotations

    def test_get_item_basic_structure(self, valid_config):
        """Test basic structure of returned sample dictionary"""
        loader = YouTubeVOSLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check basic structure
        assert isinstance(sample, dict)
        assert 'source_dataset' in sample
        assert 'sample_id' in sample
        assert 'media_type' in sample
        assert 'media_path' in sample
        assert 'annotations' in sample
        
        # Check values
        assert sample['source_dataset'] == 'test_vos'
        assert sample['media_type'] == 'video'
        assert sample['sample_id'] in ['1', '2']

    def test_get_item_vos_annotations(self, valid_config):
        """Test VOS-specific annotations in returned sample"""
        loader = YouTubeVOSLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check VOS annotations exist
        assert 'video_object_segmentation' in sample['annotations']
        vos_data = sample['annotations']['video_object_segmentation']
        
        # Check required fields
        assert 'video_id' in vos_data
        assert 'video_name' in vos_data
        assert 'object_tracks_vos' in vos_data
        assert 'num_objects' in vos_data
        assert 'num_frames' in vos_data
        assert 'frame_files' in vos_data
        assert 'video_metadata' in vos_data
        assert 'tracking_statistics' in vos_data
        
        # Check data types
        assert isinstance(vos_data['object_tracks_vos'], dict)
        assert isinstance(vos_data['num_objects'], int)
        assert isinstance(vos_data['num_frames'], int)

    def test_get_item_object_tracks_structure(self, valid_config):
        """Test structure of object tracks"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Get video_001 which has 2 objects
        video_001_index = loader._index.index(1)
        sample = loader.get_item(video_001_index)
        
        tracks = sample['annotations']['video_object_segmentation']['object_tracks_vos']
        
        # Should have 2 objects
        assert len(tracks) == 2
        assert 101 in tracks  # Object ID 101
        assert 102 in tracks  # Object ID 102
        
        # Check track structure for object 101
        track_101 = tracks[101]
        assert 'category_id' in track_101
        assert 'category_name' in track_101
        assert 'trajectory' in track_101
        assert 'num_frames_visible' in track_101
        assert track_101['category_name'] == 'person'
        
        # Check trajectory structure
        trajectory = track_101['trajectory']
        assert isinstance(trajectory, list)
        assert len(trajectory) == 4  # 4 visible frames (skipping frame 2)
        
        for frame_data in trajectory:
            assert 'frame_index' in frame_data
            assert 'mask_rle' in frame_data
            assert 'bbox' in frame_data

    def test_get_item_dataset_info(self, valid_config):
        """Test dataset_info section of returned sample"""
        loader = YouTubeVOSLoader(valid_config)
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        
        # Check required fields
        assert dataset_info['task_type'] == 'video_object_segmentation'
        assert dataset_info['source'] == 'YouTube-VOS'
        assert dataset_info['suitable_for_tracking'] is True
        assert dataset_info['suitable_for_segmentation'] is True
        assert dataset_info['suitable_for_temporal_reasoning'] is True
        assert dataset_info['has_pixel_masks'] is True
        assert dataset_info['annotation_format'] == 'rle_masks'

    def test_get_item_out_of_range(self, valid_config):
        """Test get_item raises IndexError for invalid indices"""
        loader = YouTubeVOSLoader(valid_config)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(999)

    def test_calculate_tracking_statistics(self, valid_config):
        """Test tracking statistics calculation"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Test with mock tracks
        tracks = {
            1: {'num_frames_visible': 4},
            2: {'num_frames_visible': 3}
        }
        
        stats = loader._calculate_tracking_statistics(tracks, 5)
        
        assert stats['avg_track_length'] == 3.5  # (4+3)/2
        assert stats['min_track_length'] == 3
        assert stats['max_track_length'] == 4
        assert stats['total_annotations'] == 7
        assert stats['unique_objects'] == 2
        assert stats['density'] == 0.7  # 7/(5*2)

    def test_calculate_tracking_statistics_empty(self, valid_config):
        """Test tracking statistics with empty tracks"""
        loader = YouTubeVOSLoader(valid_config)
        
        stats = loader._calculate_tracking_statistics({}, 5)
        
        assert stats['avg_track_length'] == 0.0
        assert stats['min_track_length'] == 0
        assert stats['max_track_length'] == 0
        assert stats['total_annotations'] == 0
        assert stats['density'] == 0.0

    def test_decode_rle_success(self, valid_config):
        """Test RLE decoding with pycocotools"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Mock pycocotools mask_utils module
        with patch('builtins.__import__') as mock_import:
            mock_mask_utils = MagicMock()
            mock_mask_utils.decode.return_value = np.ones((100, 200), dtype=np.uint8)
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'pycocotools':
                    mock_module = MagicMock()
                    mock_module.mask = mock_mask_utils
                    return mock_module
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            rle = {"size": [100, 200], "counts": "PCQR1g0O2N1O100O10O010000001O"}
            mask = loader.decode_rle(rle, 100, 200)
            
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (100, 200)
            assert mask.dtype == np.uint8

    def test_decode_rle_no_pycocotools(self, valid_config):
        """Test RLE decoding without pycocotools"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Test with invalid RLE format
        rle = {"invalid": "format"}
        mask = loader.decode_rle(rle, 100, 200)
        
        # Should return empty mask for invalid format
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100, 200)
        assert np.all(mask == 0)

    def test_get_samples_by_object_count(self, valid_config):
        """Test filtering samples by object count"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Get samples with exactly 2 objects
        samples = loader.get_samples_by_object_count(min_objects=2, max_objects=2)
        
        # Should return video_001 which has 2 objects
        assert len(samples) == 1
        assert samples[0]['annotations']['video_object_segmentation']['num_objects'] == 2

    def test_get_samples_by_duration(self, valid_config):
        """Test filtering samples by duration"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Get samples with exactly 5 frames
        samples = loader.get_samples_by_duration(min_frames=5, max_frames=5)
        
        # Both videos have 5 frames
        assert len(samples) == 2
        for sample in samples:
            assert sample['annotations']['video_object_segmentation']['num_frames'] == 5

    def test_get_dataset_statistics(self, valid_config):
        """Test dataset statistics calculation"""
        loader = YouTubeVOSLoader(valid_config)
        
        stats = loader.get_dataset_statistics()
        
        # Check required fields
        assert 'total_videos' in stats
        assert 'total_unique_objects' in stats
        assert 'total_annotations' in stats
        assert 'total_frames' in stats
        assert 'category_distribution' in stats
        
        # Check values
        assert stats['total_videos'] == 2
        assert stats['total_unique_objects'] == 3  # 2 in video_001, 1 in video_002
        assert stats['total_frames'] == 10  # 5 frames * 2 videos
        assert stats['num_categories'] == 3

    def test_get_category_statistics(self, valid_config):
        """Test category distribution statistics"""
        loader = YouTubeVOSLoader(valid_config)
        
        category_stats = loader.get_category_statistics()
        
        assert isinstance(category_stats, dict)
        assert 'person' in category_stats
        assert 'car' in category_stats
        assert 'dog' in category_stats
        assert category_stats['person'] == 1
        assert category_stats['car'] == 1
        assert category_stats['dog'] == 1

    def test_get_video_by_name_success(self, valid_config):
        """Test getting video by name successfully"""
        loader = YouTubeVOSLoader(valid_config)
        
        sample = loader.get_video_by_name('video_001')
        
        assert sample['sample_id'] == '1'
        assert 'video_object_segmentation' in sample['annotations']

    def test_get_video_by_name_not_found(self, valid_config):
        """Test getting video by name with invalid name"""
        loader = YouTubeVOSLoader(valid_config)
        
        with pytest.raises(ValueError, match="Video 'invalid_video' not found"):
            loader.get_video_by_name('invalid_video')

    def test_list_video_names(self, valid_config):
        """Test listing all video names"""
        loader = YouTubeVOSLoader(valid_config)
        
        names = loader.list_video_names()
        
        assert isinstance(names, list)
        assert len(names) == 2
        assert 'video_001' in names
        assert 'video_002' in names
        assert 'video_003' not in names  # No annotations

    def test_list_categories(self, valid_config):
        """Test listing all categories"""
        loader = YouTubeVOSLoader(valid_config)
        
        categories = loader.list_categories()
        
        assert isinstance(categories, list)
        assert len(categories) == 3
        assert 'person' in categories
        assert 'car' in categories
        assert 'dog' in categories

    def test_len_method(self, valid_config):
        """Test __len__ method returns correct count"""
        loader = YouTubeVOSLoader(valid_config)
        
        assert len(loader) == 2
        assert len(loader) == len(loader._index)

    @patch('core.dataloaders.youtube_vos_loader.logger')
    def test_logging_calls(self, mock_logger, valid_config):
        """Test that appropriate logging calls are made"""
        loader = YouTubeVOSLoader(valid_config)
        
        # Check that info logs were called during initialization
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Loading annotations from" in call for call in info_calls)
        assert any("Found" in call and "valid videos" in call for call in info_calls)

    def test_missing_frame_directories(self, temp_dir):
        """Test handling of videos without frame directories"""
        # Create instances.json with video but no frame directory
        instances_data = {
            "videos": [{"id": 1, "file_names": ["missing_video/frame_000000.jpg"]}],
            "categories": [{"id": 1, "name": "test"}],
            "annotations": [
                {
                    "id": 1,
                    "video_id": 1,
                    "category_id": 1,
                    "segmentations": [{"size": [100, 100], "counts": "test"}]
                }
            ]
        }
        
        instances_file = temp_dir / "instances.json"
        instances_file.write_text(json.dumps(instances_data))
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        config = {
            'name': 'test_vos',
            'path': str(frames_dir),
            'annotation_file': str(instances_file)
        }
        
        loader = YouTubeVOSLoader(config)
        
        # Should have no valid videos (frame directory missing)
        assert len(loader) == 0

    def test_malformed_json_file(self, temp_dir):
        """Test handling of malformed instances.json"""
        instances_file = temp_dir / "instances.json"
        instances_file.write_text("invalid json content")
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        config = {
            'name': 'test_vos',
            'path': str(frames_dir),
            'annotation_file': str(instances_file)
        }
        
        with pytest.raises(json.JSONDecodeError):
            YouTubeVOSLoader(config)

    def test_video_metadata_section(self, valid_config):
        """Test video_metadata section of returned sample"""
        loader = YouTubeVOSLoader(valid_config)
        sample = loader.get_item(0)
        
        video_metadata = sample['annotations']['video_metadata']
        
        # Check required fields
        assert 'frame_directory' in video_metadata
        assert 'total_frames' in video_metadata
        assert 'resolution' in video_metadata
        
        # Check values
        assert isinstance(video_metadata['total_frames'], int)
        assert video_metadata['total_frames'] == 5
        assert isinstance(video_metadata['resolution'], list)
        assert len(video_metadata['resolution']) == 2

    def test_empty_dataset(self, temp_dir):
        """Test handling of empty dataset"""
        # Create empty instances.json
        instances_data = {
            "videos": [],
            "categories": [],
            "annotations": []
        }
        
        instances_file = temp_dir / "instances.json"
        instances_file.write_text(json.dumps(instances_data))
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        config = {
            'name': 'test_vos',
            'path': str(frames_dir),
            'annotation_file': str(instances_file)
        }
        
        loader = YouTubeVOSLoader(config)
        
        # Should handle empty dataset gracefully
        assert len(loader) == 0
        
        stats = loader.get_dataset_statistics()
        assert stats['total_videos'] == 0
        assert stats['total_unique_objects'] == 0