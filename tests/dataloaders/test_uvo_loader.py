# tests/dataloaders/test_uvo_loader.py

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
from collections import defaultdict

from core.dataloaders.uvo_loader import UvoLoader


class TestUvoLoader:
    """Comprehensive test suite for UvoLoader"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_uvo_dataset(self, temp_dir):
        """Create a mock UVO dataset structure with test data"""
        # Create video directory
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        
        # Create video files and frame directories
        video_ids = ["video_001", "video_002", "video_003"]
        
        for video_id in video_ids:
            # Create video file
            video_file = video_dir / f"{video_id}.mp4"
            video_file.touch()
            
            # Also create frame directory for some videos
            if video_id == "video_002":
                frame_dir = video_dir / video_id
                frame_dir.mkdir()
                for i in range(1, 6):  # 5 frames
                    (frame_dir / f"frame_{i:03d}.jpg").touch()
        
        # Create annotation directory structure
        annotation_dir = temp_dir / "annotations"
        annotation_dir.mkdir()
        
        # Create dense mask annotations
        dense_dir = annotation_dir / "uvo-dense"
        dense_dir.mkdir()
        
        dense_mask_data = {
            "annotations": [
                {
                    "id": 1,
                    "frame_id": 1,
                    "bbox": [100, 200, 50, 100],
                    "segmentation": [[100, 200, 150, 200, 150, 300, 100, 300]],
                    "area": 5000,
                    "category_id": 1
                },
                {
                    "id": 1,
                    "frame_id": 2,
                    "bbox": [105, 205, 50, 100],
                    "segmentation": [[105, 205, 155, 205, 155, 305, 105, 305]],
                    "area": 5000,
                    "category_id": 1
                }
            ],
            "info": {"description": "UVO dense annotations"}
        }
        
        (dense_dir / "video_001.json").write_text(json.dumps(dense_mask_data))
        
        # Create sparse mask annotations
        sparse_dir = annotation_dir / "uvo-sparse"
        sparse_dir.mkdir()
        
        sparse_mask_data = {
            "annotations": [
                {
                    "id": 1,
                    "frame_id": 1,
                    "bbox": [200, 300, 60, 120],
                    "segmentation": [[200, 300, 260, 300, 260, 420, 200, 420]],
                    "area": 7200,
                    "category_id": 2
                }
            ]
        }
        
        (sparse_dir / "video_002.json").write_text(json.dumps(sparse_mask_data))
        
        # Create expression annotations
        expression_dir = annotation_dir / "expressions"
        expression_dir.mkdir()
        
        expression_data = {
            "expressions": [
                {
                    "expression": "the red car moving left",
                    "object_id": 1,
                    "frames": [1, 2, 3]
                }
            ]
        }
        
        (expression_dir / "video_001.json").write_text(json.dumps(expression_data))
        
        # Create relationship annotations
        relationship_dir = annotation_dir / "rel_annotations"
        relationship_dir.mkdir()
        
        relationship_data = {
            "relationships": [
                {
                    "subject_id": 1,
                    "object_id": 2,
                    "predicate": "follows",
                    "frames": [1, 2]
                }
            ]
        }
        
        (relationship_dir / "video_001.json").write_text(json.dumps(relationship_data))
        
        # Create box annotations
        box_dir = annotation_dir / "box_annotations"
        box_dir.mkdir()
        
        box_data = {
            "boxes": [
                {
                    "frame_id": 1,
                    "object_id": 1,
                    "bbox": [100, 200, 50, 100],
                    "confidence": 0.95
                }
            ]
        }
        
        (box_dir / "video_002.json").write_text(json.dumps(box_data))
        
        # Create additional annotation type
        additional_dir = annotation_dir / "temporal_annotations"
        additional_dir.mkdir()
        
        additional_data = {
            "temporal_info": {
                "duration": 30.5,
                "fps": 25.0,
                "keyframes": [1, 15, 30]
            }
        }
        
        (additional_dir / "video_001.json").write_text(json.dumps(additional_data))
        
        return {
            "video_dir": video_dir,
            "annotation_dir": annotation_dir,
            "video_ids": video_ids
        }

    @pytest.fixture
    def valid_config(self, mock_uvo_dataset):
        """Valid configuration for testing"""
        return {
            'name': 'test_uvo',
            'path': str(mock_uvo_dataset["video_dir"]),
            'annotation_path': str(mock_uvo_dataset["annotation_dir"])
        }

    def test_init_success(self, valid_config):
        """Test successful initialization with valid config"""
        loader = UvoLoader(valid_config)
        
        assert loader.source_name == 'test_uvo'
        assert loader.video_path == Path(valid_config['path'])
        assert loader.annotation_path == Path(valid_config['annotation_path'])
        assert len(loader) == 3  # Should find 3 video IDs

    def test_init_missing_required_config(self):
        """Test initialization fails with missing required config keys"""
        invalid_configs = [
            {'name': 'test_uvo', 'annotation_path': '/some/path'},  # Missing 'path'
            {'name': 'test_uvo', 'path': '/some/path'}  # Missing 'annotation_path'
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="UvoLoader config must include"):
                UvoLoader(config)

    def test_init_invalid_paths(self):
        """Test initialization fails with invalid paths"""
        invalid_config = {
            'name': 'test_uvo',
            'path': '/nonexistent/video/path',
            'annotation_path': '/nonexistent/annotation/path'
        }
        
        with pytest.raises(FileNotFoundError, match="Video directory not found"):
            UvoLoader(invalid_config)

    def test_init_invalid_annotation_path(self, temp_dir):
        """Test initialization fails with invalid annotation path"""
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        
        invalid_config = {
            'name': 'test_uvo',
            'path': str(video_dir),
            'annotation_path': '/nonexistent/annotation/path'
        }
        
        with pytest.raises(FileNotFoundError, match="Annotation directory not found"):
            UvoLoader(invalid_config)

    def test_build_index_structure(self, valid_config):
        """Test that _build_index correctly identifies video IDs"""
        loader = UvoLoader(valid_config)
        
        # Should have exactly 3 video IDs
        assert len(loader._index) == 3
        
        # Check that all items are strings (video IDs)
        for video_id in loader._index:
            assert isinstance(video_id, str)
        
        # Check video IDs are correct
        expected_ids = ["video_001", "video_002", "video_003"]
        assert set(loader._index) == set(expected_ids)

    def test_annotation_indexing(self, valid_config):
        """Test that annotation files are properly indexed"""
        loader = UvoLoader(valid_config)
        
        # Check dense mask indexing
        assert "video_001" in loader._video_id_to_dense_mask_path
        
        # Check sparse mask indexing
        assert "video_002" in loader._video_id_to_sparse_mask_path
        
        # Check expression indexing
        assert "video_001" in loader._video_id_to_expression_path
        
        # Check relationship indexing
        assert "video_001" in loader._video_id_to_relationship_path
        
        # Check box annotation indexing
        assert "video_002" in loader._video_id_to_box_annotation_path
        
        # Check additional annotation indexing
        assert "video_001" in loader._video_id_to_additional_annotations
        assert "temporal_annotations" in loader._video_id_to_additional_annotations["video_001"]

    def test_get_item_basic_structure(self, valid_config):
        """Test basic structure of returned sample dictionary"""
        loader = UvoLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check basic structure
        assert isinstance(sample, dict)
        assert 'source_dataset' in sample
        assert 'sample_id' in sample
        assert 'media_type' in sample
        assert 'media_path' in sample
        assert 'annotations' in sample
        
        # Check values
        assert sample['source_dataset'] == 'test_uvo'
        assert sample['media_type'] == 'video'
        assert sample['sample_id'] in ["video_001", "video_002", "video_003"]

    def test_get_item_uvo_annotations(self, valid_config):
        """Test UVO-specific annotations in returned sample"""
        loader = UvoLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check UVO annotations exist
        assert 'unidentified_video_objects' in sample['annotations']
        uvo_data = sample['annotations']['unidentified_video_objects']
        
        # Check required fields
        assert 'video_id' in uvo_data
        assert 'object_tracks' in uvo_data
        assert 'mask_source' in uvo_data
        assert 'num_objects' in uvo_data
        assert 'tracking_statistics' in uvo_data
        assert 'annotation_coverage' in uvo_data
        
        # Check data types
        assert isinstance(uvo_data['object_tracks'], dict)
        assert isinstance(uvo_data['num_objects'], int)
        assert isinstance(uvo_data['tracking_statistics'], dict)
        assert uvo_data['mask_source'] in ['dense', 'sparse', 'none']

    def test_get_item_with_dense_masks(self, valid_config):
        """Test get_item for video with dense mask annotations"""
        loader = UvoLoader(valid_config)
        
        # Find video_001 which has dense masks
        video_001_index = loader._index.index("video_001")
        sample = loader.get_item(video_001_index)
        
        uvo_data = sample['annotations']['unidentified_video_objects']
        
        # Should have dense mask source
        assert uvo_data['mask_source'] == 'dense'
        
        # Should have object tracks
        assert len(uvo_data['object_tracks']) > 0
        
        # Should have expressions and relationships
        assert uvo_data['expressions'] is not None
        assert uvo_data['relationships'] is not None

    def test_get_item_with_sparse_masks(self, valid_config):
        """Test get_item for video with sparse mask annotations"""
        loader = UvoLoader(valid_config)
        
        # Find video_002 which has sparse masks
        video_002_index = loader._index.index("video_002")
        sample = loader.get_item(video_002_index)
        
        uvo_data = sample['annotations']['unidentified_video_objects']
        
        # Should have sparse mask source
        assert uvo_data['mask_source'] == 'sparse'
        
        # Should have object tracks
        assert len(uvo_data['object_tracks']) > 0
        
        # Should have box annotations
        assert uvo_data['box_annotations'] is not None

    def test_get_item_without_masks(self, valid_config):
        """Test get_item for video without mask annotations"""
        loader = UvoLoader(valid_config)
        
        # Find video_003 which has no mask annotations
        video_003_index = loader._index.index("video_003")
        sample = loader.get_item(video_003_index)
        
        uvo_data = sample['annotations']['unidentified_video_objects']
        
        # Should have no mask source
        assert uvo_data['mask_source'] == 'none'
        
        # Should have empty tracks
        assert len(uvo_data['object_tracks']) == 0

    def test_get_item_dataset_info(self, valid_config):
        """Test dataset_info section of returned sample"""
        loader = UvoLoader(valid_config)
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        
        # Check required fields
        assert dataset_info['task_type'] == 'unidentified_video_object_tracking'
        assert dataset_info['source'] == 'UVO'
        assert dataset_info['suitable_for_tracking'] is True
        assert dataset_info['suitable_for_object_discovery'] is True
        assert dataset_info['suitable_for_temporal_reasoning'] is True
        assert 'annotation_completeness' in dataset_info
        
        # Check completeness is between 0 and 1
        completeness = dataset_info['annotation_completeness']
        assert 0.0 <= completeness <= 1.0

    def test_get_item_out_of_range(self, valid_config):
        """Test get_item raises IndexError for invalid indices"""
        loader = UvoLoader(valid_config)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(999)

    def test_extract_video_id_strategies(self, valid_config):
        """Test different video ID extraction strategies"""
        loader = UvoLoader(valid_config)
        
        # Test filename stem extraction
        path1 = Path("/path/to/video_123.json")
        assert loader._extract_video_id(path1) == "video_123"
        
        # Test parent directory extraction
        path2 = Path("/path/video_456/masks.json")
        assert loader._extract_video_id(path2) == "video_456"
        
        # Test pattern matching
        path3 = Path("/path/to/123_masks.json")
        assert loader._extract_video_id(path3) == "123_masks"

    def test_parse_mask_file_json(self, valid_config, temp_dir):
        """Test parsing JSON mask files"""
        loader = UvoLoader(valid_config)
        
        # Create test JSON file
        mask_data = {
            "annotations": [
                {
                    "id": 1,
                    "frame_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
                    "area": 1200
                }
            ]
        }
        
        test_file = temp_dir / "test_mask.json"
        test_file.write_text(json.dumps(mask_data))
        
        result = loader._parse_mask_file(test_file)
        
        assert 'tracks' in result
        assert 1 in result['tracks']
        assert len(result['tracks'][1]) == 1
        assert result['tracks'][1][0]['frame_id'] == 1

    def test_parse_mask_file_directory(self, valid_config, temp_dir):
        """Test parsing directory-based mask files"""
        loader = UvoLoader(valid_config)
        
        # Create test directory with mask files
        mask_dir = temp_dir / "test_masks"
        mask_dir.mkdir()
        
        # Create mock PNG files with proper naming
        (mask_dir / "001-01.png").touch()
        (mask_dir / "002-01.png").touch()
        (mask_dir / "001-02.png").touch()
        
        result = loader._parse_mask_file(mask_dir)
        
        assert 'tracks' in result
        assert len(result['tracks']) == 2  # Objects 1 and 2
        assert 1 in result['tracks']
        assert 2 in result['tracks']

    def test_analyze_uvo_tracks_normal(self, valid_config):
        """Test track analysis with normal tracks"""
        loader = UvoLoader(valid_config)
        
        # Create test tracks
        tracks = {
            1: [
                {'frame_id': 1, 'bbox': [10, 20, 30, 40]},
                {'frame_id': 2, 'bbox': [15, 25, 30, 40]}
            ],
            2: [
                {'frame_id': 1, 'bbox': [100, 200, 50, 100]}
            ]
        }
        
        stats = loader._analyze_uvo_tracks(tracks)
        
        assert stats['avg_track_length'] == 1.5  # (2+1)/2
        assert stats['min_track_length'] == 1
        assert stats['max_track_length'] == 2
        assert stats['total_detections'] == 3
        assert stats['unique_objects'] == 2

    def test_analyze_uvo_tracks_empty(self, valid_config):
        """Test track analysis with empty tracks"""
        loader = UvoLoader(valid_config)
        
        stats = loader._analyze_uvo_tracks({})
        
        assert stats['avg_track_length'] == 0.0
        assert stats['min_track_length'] == 0
        assert stats['max_track_length'] == 0
        assert stats['total_detections'] == 0
        assert stats['unique_objects'] == 0

    def test_calculate_annotation_completeness(self, valid_config):
        """Test annotation completeness calculation"""
        loader = UvoLoader(valid_config)
        
        # Test with all annotations present
        completeness = loader._calculate_annotation_completeness(
            {'tracks': {1: []}},  # Has masks
            {'expressions': []},   # Has expressions
            {'relationships': []}, # Has relationships
            {'boxes': []},         # Has box annotations
            {'extra': 'data'}      # Has additional annotations
        )
        assert completeness == 1.0
        
        # Test with partial annotations
        completeness = loader._calculate_annotation_completeness(
            {'tracks': {1: []}},  # Has masks
            None,                 # No expressions
            None,                 # No relationships
            {'boxes': []},        # Has box annotations
            None                  # No additional annotations
        )
        assert completeness == 0.4  # 2/5

    def test_get_samples_by_annotation_completeness(self, valid_config):
        """Test filtering samples by annotation completeness"""
        loader = UvoLoader(valid_config)
        
        # Get samples with high completeness
        samples = loader.get_samples_by_annotation_completeness(min_completeness=0.5)
        
        # Should return some samples
        assert len(samples) > 0
        
        # All returned samples should meet the threshold
        for sample in samples:
            completeness = sample['annotations']['dataset_info']['annotation_completeness']
            assert completeness >= 0.5

    def test_get_samples_with_expressions(self, valid_config):
        """Test getting samples with expression annotations"""
        loader = UvoLoader(valid_config)
        
        samples = loader.get_samples_with_expressions()
        
        # Should find at least one sample with expressions (video_001)
        assert len(samples) >= 1
        
        # All returned samples should have expressions
        for sample in samples:
            uvo_data = sample['annotations']['unidentified_video_objects']
            assert uvo_data['expressions'] is not None

    def test_get_samples_with_relationships(self, valid_config):
        """Test getting samples with relationship annotations"""
        loader = UvoLoader(valid_config)
        
        samples = loader.get_samples_with_relationships()
        
        # Should find at least one sample with relationships (video_001)
        assert len(samples) >= 1
        
        # All returned samples should have relationships
        for sample in samples:
            uvo_data = sample['annotations']['unidentified_video_objects']
            assert uvo_data['relationships'] is not None

    def test_get_annotation_coverage_statistics(self, valid_config):
        """Test annotation coverage statistics"""
        loader = UvoLoader(valid_config)
        
        stats = loader.get_annotation_coverage_statistics()
        
        # Check required fields
        assert 'total_videos' in stats
        assert 'mask_coverage' in stats
        assert 'supplementary_coverage' in stats
        
        # Check values make sense
        assert stats['total_videos'] == len(loader)
        assert stats['mask_coverage']['total_with_masks'] >= 0
        assert stats['supplementary_coverage']['expressions'] >= 0

    def test_list_available_annotation_types(self, valid_config):
        """Test listing available annotation types"""
        loader = UvoLoader(valid_config)
        
        types = loader.list_available_annotation_types()
        
        # Should contain basic types
        assert 'masks' in types
        
        # Should contain discovered types
        expected_types = ['expressions', 'relationships', 'box_annotations', 'temporal_annotations']
        for expected_type in expected_types:
            if expected_type == 'temporal_annotations':
                # This is an additional annotation type
                assert 'temporal_annotations' in types
            else:
                assert expected_type in types

    def test_len_method(self, valid_config):
        """Test __len__ method returns correct count"""
        loader = UvoLoader(valid_config)
        
        assert len(loader) == 3
        assert len(loader) == len(loader._index)

    @patch('core.dataloaders.uvo_loader.logger')
    def test_logging_calls(self, mock_logger, valid_config):
        """Test that appropriate logging calls are made"""
        loader = UvoLoader(valid_config)
        
        # Check that info logs were called during initialization
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Scanning UVO dataset" in call for call in info_calls)
        assert any("video IDs from video directory" in call for call in info_calls)

    def test_missing_annotation_files(self, temp_dir):
        """Test handling of missing annotation files"""
        # Create minimal dataset with only videos
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        (video_dir / "video_001.mp4").touch()
        
        annotation_dir = temp_dir / "annotations"
        annotation_dir.mkdir()
        
        config = {
            'name': 'test_uvo',
            'path': str(video_dir),
            'annotation_path': str(annotation_dir)
        }
        
        loader = UvoLoader(config)
        
        # Should still work with no annotations
        assert len(loader) == 1
        
        sample = loader.get_item(0)
        uvo_data = sample['annotations']['unidentified_video_objects']
        
        # Should handle missing annotations gracefully
        assert uvo_data['mask_source'] == 'none'
        assert len(uvo_data['object_tracks']) == 0
        assert uvo_data['expressions'] is None

    def test_malformed_annotation_files(self, valid_config, temp_dir):
        """Test handling of malformed annotation files"""
        loader = UvoLoader(valid_config)
        
        # Create malformed JSON file
        bad_file = temp_dir / "bad_annotations.json"
        bad_file.write_text("invalid json content")
        
        # Should handle malformed files gracefully
        result = loader._parse_mask_file(bad_file)
        assert result == {'tracks': {}}

    def test_video_file_vs_directory_handling(self, valid_config):
        """Test handling of both video files and frame directories"""
        loader = UvoLoader(valid_config)
        
        # Test video file handling (video_001.mp4)
        video_001_index = loader._index.index("video_001")
        sample = loader.get_item(video_001_index)
        
        video_metadata = sample['annotations']['video_metadata']
        assert video_metadata['video_filename'] == 'video_001.mp4'
        assert video_metadata['frame_directory'] is None
        
        # Test frame directory handling (video_002/)
        video_002_index = loader._index.index("video_002")
        sample = loader.get_item(video_002_index)
        
        video_metadata = sample['annotations']['video_metadata']
        # Could be either video file or frame directory for video_002

    def test_find_video_file_strategies(self, valid_config, temp_dir):
        """Test different strategies for finding video files"""
        loader = UvoLoader(valid_config)
        
        # Test finding video file
        video_path = loader._find_video_file("video_001")
        assert video_path.exists()
        assert video_path.suffix == ".mp4"
        
        # Test finding frame directory
        video_path = loader._find_video_file("video_002")
        assert video_path.exists()
        # Could be either .mp4 file or directory

    def test_empty_dataset_directory(self, temp_dir):
        """Test initialization with empty dataset directories"""
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        
        annotation_dir = temp_dir / "annotations"
        annotation_dir.mkdir()
        
        config = {
            'name': 'test_uvo',
            'path': str(video_dir),
            'annotation_path': str(annotation_dir)
        }
        
        loader = UvoLoader(config)
        
        # Should handle empty directories gracefully
        assert len(loader) == 0
        
        stats = loader.get_annotation_coverage_statistics()
        assert stats['total_videos'] == 0