# tests/dataloaders/test_epic_kitchens_loader.py

import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil

from core.dataloaders.epic_kitchens_loader import EpicKitchensVisorLoader


class TestEpicKitchensVisorLoader:
    """Comprehensive test suite for EpicKitchensVisorLoader"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_epic_dataset(self, temp_dir):
        """Create a mock EPIC-KITCHENS dataset structure with test data"""
        # Create rgb_frames directory structure
        rgb_frames_dir = temp_dir / "rgb_frames"
        rgb_frames_dir.mkdir()
        
        # Create train and val splits
        train_dir = rgb_frames_dir / "train"
        val_dir = rgb_frames_dir / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        
        # Create video directories with mock frames
        video_ids = {
            'train': ['P01_01', 'P01_02'],
            'val': ['P02_01']
        }
        
        for split, ids in video_ids.items():
            split_dir = rgb_frames_dir / split
            for video_id in ids:
                video_dir = split_dir / video_id
                video_dir.mkdir()
                
                # Create mock frame files
                for i in range(1, 6):  # 5 frames each
                    frame_file = video_dir / f"frame_{i:05d}.jpg"
                    frame_file.touch()
        
        # Create annotations directory structure
        annotations_dir = temp_dir / "annotations"
        annotations_dir.mkdir()
        
        # Create sparse annotations
        sparse_dir = annotations_dir / "GroundTruth-SparseAnnotations"
        sparse_dir.mkdir()
        sparse_train = sparse_dir / "train"
        sparse_val = sparse_dir / "val"
        sparse_train.mkdir()
        sparse_val.mkdir()
        
        # Create sparse annotation files
        sparse_data = {
            "annotations": [
                {
                    "frame_id": 1,
                    "class_id": 10,
                    "bbox": [100, 200, 50, 100],
                    "segmentation": [[100, 200, 150, 200, 150, 300, 100, 300]]
                },
                {
                    "frame_id": 3,
                    "class_id": 20,
                    "bbox": [200, 300, 60, 120],
                    "segmentation": [[200, 300, 260, 300, 260, 420, 200, 420]]
                }
            ]
        }
        
        (sparse_train / "P01_01.json").write_text(json.dumps(sparse_data))
        (sparse_train / "P01_02.json").write_text(json.dumps(sparse_data))
        (sparse_val / "P02_01.json").write_text(json.dumps(sparse_data))
        
        # Create dense annotations
        dense_dir = annotations_dir / "Interpolations-DenseAnnotations"
        dense_dir.mkdir()
        dense_train = dense_dir / "train"
        dense_val = dense_dir / "val"
        dense_train.mkdir()
        dense_val.mkdir()
        
        # Create dense annotation files (interpolated between sparse)
        dense_data = {
            "annotations": [
                {
                    "frame_id": 1,
                    "class_id": 10,
                    "bbox": [100, 200, 50, 100],
                    "segmentation": [[100, 200, 150, 200, 150, 300, 100, 300]]
                },
                {
                    "frame_id": 2,
                    "class_id": 10,
                    "bbox": [105, 205, 50, 100],
                    "segmentation": [[105, 205, 155, 205, 155, 305, 105, 305]]
                },
                {
                    "frame_id": 3,
                    "class_id": 20,
                    "bbox": [200, 300, 60, 120],
                    "segmentation": [[200, 300, 260, 300, 260, 420, 200, 420]]
                }
            ]
        }
        
        (dense_train / "P01_01.json").write_text(json.dumps(dense_data))
        # P01_02 will only have sparse annotations (test partial coverage)
        (dense_val / "P02_01.json").write_text(json.dumps(dense_data))
        
        # Create class mapping CSV
        class_mapping_file = annotations_dir / "EPIC_100_noun_classes_v2.csv"
        class_data = pd.DataFrame({
            'noun_id': [10, 20, 30],
            'noun': ['knife', 'cutting_board', 'tomato']
        })
        class_data.to_csv(class_mapping_file, index=False)
        
        # Create frame mapping JSON
        frame_mapping_file = annotations_dir / "frame_mapping.json"
        frame_mapping = {
            "1": "frame_00001.jpg",
            "2": "frame_00002.jpg",
            "3": "frame_00003.jpg",
            "4": "frame_00004.jpg",
            "5": "frame_00005.jpg"
        }
        frame_mapping_file.write_text(json.dumps(frame_mapping))
        
        return {
            "rgb_frames_dir": rgb_frames_dir,
            "sparse_dir": sparse_dir,
            "dense_dir": dense_dir,
            "class_mapping_file": class_mapping_file,
            "frame_mapping_file": frame_mapping_file,
            "video_ids": video_ids
        }

    @pytest.fixture
    def valid_config(self, mock_epic_dataset):
        """Valid configuration for testing"""
        return {
            'name': 'test_epic',
            'image_path': str(mock_epic_dataset["rgb_frames_dir"]),
            'sparse_annotation_path': str(mock_epic_dataset["sparse_dir"]),
            'dense_annotation_path': str(mock_epic_dataset["dense_dir"]),
            'class_mapping_file': str(mock_epic_dataset["class_mapping_file"]),
            'frame_mapping_file': str(mock_epic_dataset["frame_mapping_file"])
        }

    def test_init_success(self, valid_config):
        """Test successful initialization with valid config"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        assert loader.source_name == 'test_epic'
        assert loader.image_path == Path(valid_config['image_path'])
        assert loader.sparse_annotation_path == Path(valid_config['sparse_annotation_path'])
        assert loader.dense_annotation_path == Path(valid_config['dense_annotation_path'])
        assert len(loader) == 3  # Should find 3 video IDs

    def test_init_missing_required_config(self):
        """Test initialization fails with missing required config keys"""
        required_keys = [
            'image_path', 
            'sparse_annotation_path', 
            'dense_annotation_path',
            'class_mapping_file', 
            'frame_mapping_file'
        ]
        
        for missing_key in required_keys:
            config = {
                'name': 'test_epic',
                'image_path': '/path',
                'sparse_annotation_path': '/path',
                'dense_annotation_path': '/path',
                'class_mapping_file': '/path',
                'frame_mapping_file': '/path'
            }
            del config[missing_key]
            
            with pytest.raises(ValueError, match=f"EpicKitchensVisorLoader config must include '{missing_key}'"):
                EpicKitchensVisorLoader(config)

    def test_init_invalid_paths(self):
        """Test initialization fails with invalid paths"""
        invalid_config = {
            'name': 'test_epic',
            'image_path': '/nonexistent/path',
            'sparse_annotation_path': '/nonexistent/path',
            'dense_annotation_path': '/nonexistent/path',
            'class_mapping_file': '/nonexistent/file.csv',
            'frame_mapping_file': '/nonexistent/file.json'
        }
        
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            EpicKitchensVisorLoader(invalid_config)

    def test_build_index_structure(self, valid_config):
        """Test that _build_index correctly identifies video IDs"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Should have exactly 3 video IDs
        assert len(loader._index) == 3
        
        # Check that all items are strings (video IDs)
        for video_id in loader._index:
            assert isinstance(video_id, str)
        
        # Check video IDs are correct
        expected_ids = ['P01_01', 'P01_02', 'P02_01']
        assert set(loader._index) == set(expected_ids)

    def test_metadata_loading(self, valid_config):
        """Test that metadata files are properly loaded"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Check class mapping loaded
        assert len(loader._class_id_to_name) == 3
        assert loader._class_id_to_name[10] == 'knife'
        assert loader._class_id_to_name[20] == 'cutting_board'
        assert loader._class_id_to_name[30] == 'tomato'
        
        # Check frame mapping loaded
        assert len(loader._frame_mapping) == 5
        assert loader._frame_mapping['1'] == 'frame_00001.jpg'
        assert loader._frame_mapping['5'] == 'frame_00005.jpg'

    def test_video_split_tracking(self, valid_config):
        """Test that video splits are properly tracked"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Check split assignments
        assert loader._video_id_to_split['P01_01'] == 'train'
        assert loader._video_id_to_split['P01_02'] == 'train'
        assert loader._video_id_to_split['P02_01'] == 'val'

    def test_get_item_basic_structure(self, valid_config):
        """Test basic structure of returned sample dictionary"""
        loader = EpicKitchensVisorLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check basic structure
        assert isinstance(sample, dict)
        assert 'source_dataset' in sample
        assert 'sample_id' in sample
        assert 'media_type' in sample
        assert 'media_path' in sample
        assert 'annotations' in sample
        
        # Check values
        assert sample['source_dataset'] == 'test_epic'
        assert sample['media_type'] == 'video'
        assert sample['sample_id'] in ['P01_01', 'P01_02', 'P02_01']

    def test_get_item_epic_annotations(self, valid_config):
        """Test EPIC-KITCHENS-specific annotations in returned sample"""
        loader = EpicKitchensVisorLoader(valid_config)
        sample = loader.get_item(0)
        
        # Check EPIC annotations exist
        assert 'epic_kitchens_visor' in sample['annotations']
        epic_data = sample['annotations']['epic_kitchens_visor']
        
        # Check required fields
        assert 'video_id' in epic_data
        assert 'split' in epic_data
        assert 'sparse_ground_truth' in epic_data
        assert 'dense_interpolations' in epic_data
        assert 'num_sparse_annotations' in epic_data
        assert 'num_dense_annotations' in epic_data
        assert 'has_sparse' in epic_data
        assert 'has_dense' in epic_data
        assert 'annotation_statistics' in epic_data

    def test_get_item_with_complete_annotations(self, valid_config):
        """Test get_item for video with both sparse and dense annotations"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Find P01_01 which has both annotations
        p01_01_index = loader._index.index('P01_01')
        sample = loader.get_item(p01_01_index)
        
        epic_data = sample['annotations']['epic_kitchens_visor']
        
        # Should have both annotation types
        assert epic_data['has_sparse'] is True
        assert epic_data['has_dense'] is True
        assert epic_data['sparse_ground_truth'] is not None
        assert epic_data['dense_interpolations'] is not None

    def test_get_item_with_partial_annotations(self, valid_config):
        """Test get_item for video with only sparse annotations"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Find P01_02 which has only sparse annotations
        p01_02_index = loader._index.index('P01_02')
        sample = loader.get_item(p01_02_index)
        
        epic_data = sample['annotations']['epic_kitchens_visor']
        
        # Should have only sparse annotations
        assert epic_data['has_sparse'] is True
        assert epic_data['has_dense'] is False
        assert epic_data['sparse_ground_truth'] is not None
        assert epic_data['dense_interpolations'] is None

    def test_get_item_dataset_info(self, valid_config):
        """Test dataset_info section of returned sample"""
        loader = EpicKitchensVisorLoader(valid_config)
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        
        # Check required fields
        assert dataset_info['task_type'] == 'egocentric_video_segmentation'
        assert dataset_info['source'] == 'EPIC-KITCHENS-VISOR'
        assert dataset_info['suitable_for_tracking'] is True
        assert dataset_info['suitable_for_segmentation'] is True
        assert dataset_info['suitable_for_temporal_reasoning'] is True
        assert 'has_sparse_annotations' in dataset_info
        assert 'has_dense_annotations' in dataset_info
        assert dataset_info['annotation_format'] == 'visor_format'

    def test_get_item_out_of_range(self, valid_config):
        """Test get_item raises IndexError for invalid indices"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(999)

    def test_annotation_translation(self, valid_config):
        """Test that annotations are properly translated using metadata"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Test translation method directly
        test_annotations = {
            "frame_id": 1,
            "class_id": 10,
            "nested": {
                "frame_ref": 2,
                "class_id": 20
            }
        }
        
        translated = loader._translate_annotations(test_annotations)
        
        # Check frame translation
        assert translated['frame_id'] == 'frame_00001.jpg'
        assert translated['nested']['frame_ref'] == 'frame_00002.jpg'
        
        # Check class translation
        assert translated['class_id'] == 'knife'
        assert translated['class_id_original'] == 10
        assert translated['nested']['class_id'] == 'cutting_board'

    def test_calculate_annotation_statistics(self, valid_config):
        """Test annotation statistics calculation"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Test with mock data
        sparse = [{"id": 1}, {"id": 2}]
        dense = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        stats = loader._calculate_annotation_statistics(sparse, dense)
        
        assert stats['sparse_count'] == 2
        assert stats['dense_count'] == 3
        assert stats['total_annotations'] == 5
        assert stats['annotation_density'] == 1.5  # 3/2

    def test_get_samples_by_split(self, valid_config):
        """Test filtering samples by split"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Get train samples
        train_samples = loader.get_samples_by_split('train')
        assert len(train_samples) == 2
        for sample in train_samples:
            assert sample['annotations']['epic_kitchens_visor']['split'] == 'train'
        
        # Get val samples
        val_samples = loader.get_samples_by_split('val')
        assert len(val_samples) == 1
        for sample in val_samples:
            assert sample['annotations']['epic_kitchens_visor']['split'] == 'val'

    def test_get_samples_with_both_annotations(self, valid_config):
        """Test getting samples with complete annotations"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        samples = loader.get_samples_with_both_annotations()
        
        # Should return P01_01 and P02_01 (both have sparse and dense)
        assert len(samples) == 2
        
        for sample in samples:
            epic_data = sample['annotations']['epic_kitchens_visor']
            assert epic_data['has_sparse'] is True
            assert epic_data['has_dense'] is True

    def test_get_annotation_coverage_statistics(self, valid_config):
        """Test annotation coverage statistics"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        stats = loader.get_annotation_coverage_statistics()
        
        # Check required fields
        assert 'total_videos' in stats
        assert 'split_distribution' in stats
        assert 'annotation_coverage' in stats
        assert 'metadata_statistics' in stats
        
        # Check values
        assert stats['total_videos'] == 3
        assert stats['split_distribution']['train'] == 2
        assert stats['split_distribution']['val'] == 1
        assert stats['annotation_coverage']['both_annotations'] == 2  # P01_01 and P02_01
        assert stats['annotation_coverage']['sparse_only'] == 1  # P01_02
        assert stats['metadata_statistics']['num_classes'] == 3
        assert stats['metadata_statistics']['num_frame_mappings'] == 5

    def test_get_class_distribution(self, valid_config):
        """Test class distribution calculation"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # This would need actual class counting logic
        # For now, just test that the method returns a dictionary
        distribution = loader.get_class_distribution()
        assert isinstance(distribution, dict)

    def test_list_available_videos(self, valid_config):
        """Test listing available video IDs"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        videos = loader.list_available_videos()
        
        assert isinstance(videos, list)
        assert len(videos) == 3
        assert set(videos) == {'P01_01', 'P01_02', 'P02_01'}

    def test_get_video_by_id_success(self, valid_config):
        """Test getting video by ID successfully"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        sample = loader.get_video_by_id('P01_01')
        
        assert sample['sample_id'] == 'P01_01'
        assert 'epic_kitchens_visor' in sample['annotations']

    def test_get_video_by_id_not_found(self, valid_config):
        """Test getting video by ID with invalid ID"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        with pytest.raises(ValueError, match="Video ID 'INVALID' not found"):
            loader.get_video_by_id('INVALID')

    def test_len_method(self, valid_config):
        """Test __len__ method returns correct count"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        assert len(loader) == 3
        assert len(loader) == len(loader._index)

    @patch('core.dataloaders.epic_kitchens_loader.logger')
    def test_logging_calls(self, mock_logger, valid_config):
        """Test that appropriate logging calls are made"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Check that info logs were called during initialization
        mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Building index for EPIC-KITCHENS" in call for call in info_calls)
        assert any("Found" in call and "valid videos" in call for call in info_calls)

    def test_missing_annotation_files(self, temp_dir):
        """Test handling of missing annotation files"""
        # Create minimal dataset with only frames
        rgb_frames_dir = temp_dir / "rgb_frames"
        rgb_frames_dir.mkdir()
        train_dir = rgb_frames_dir / "train"
        train_dir.mkdir()
        video_dir = train_dir / "P99_99"
        video_dir.mkdir()
        (video_dir / "frame_00001.jpg").touch()
        
        # Create empty annotation directories
        annotations_dir = temp_dir / "annotations"
        annotations_dir.mkdir()
        sparse_dir = annotations_dir / "GroundTruth-SparseAnnotations"
        sparse_dir.mkdir()
        (sparse_dir / "train").mkdir()
        dense_dir = annotations_dir / "Interpolations-DenseAnnotations"
        dense_dir.mkdir()
        (dense_dir / "train").mkdir()
        
        # Create metadata files
        class_file = annotations_dir / "EPIC_100_noun_classes_v2.csv"
        pd.DataFrame({'noun_id': [1], 'noun': ['test']}).to_csv(class_file, index=False)
        
        frame_file = annotations_dir / "frame_mapping.json"
        frame_file.write_text('{"1": "frame_00001.jpg"}')
        
        config = {
            'name': 'test_epic',
            'image_path': str(rgb_frames_dir),
            'sparse_annotation_path': str(sparse_dir),
            'dense_annotation_path': str(dense_dir),
            'class_mapping_file': str(class_file),
            'frame_mapping_file': str(frame_file)
        }
        
        loader = EpicKitchensVisorLoader(config)
        
        # Should have no videos (no annotations)
        assert len(loader) == 0

    def test_malformed_annotation_files(self, valid_config, temp_dir):
        """Test handling of malformed annotation files"""
        loader = EpicKitchensVisorLoader(valid_config)
        
        # Create malformed JSON file
        bad_file = temp_dir / "bad_annotations.json"
        bad_file.write_text("invalid json content")
        
        # Test loading malformed file
        result = loader._load_sparse_annotations("fake_video", "train")
        # Should handle gracefully and return None
        assert result is None

    def test_video_metadata_section(self, valid_config):
        """Test video_metadata section of returned sample"""
        loader = EpicKitchensVisorLoader(valid_config)
        sample = loader.get_item(0)
        
        video_metadata = sample['annotations']['video_metadata']
        
        # Check required fields
        assert 'frame_directory' in video_metadata
        assert 'split' in video_metadata
        assert 'num_frames' in video_metadata
        
        # Check values
        assert video_metadata['split'] in ['train', 'val']
        assert isinstance(video_metadata['num_frames'], int)
        assert video_metadata['num_frames'] >= 0

    def test_empty_dataset_directory(self, temp_dir):
        """Test initialization with empty dataset directories"""
        # Create empty structure
        rgb_frames_dir = temp_dir / "rgb_frames"
        rgb_frames_dir.mkdir()
        (rgb_frames_dir / "train").mkdir()
        (rgb_frames_dir / "val").mkdir()
        
        annotations_dir = temp_dir / "annotations"
        annotations_dir.mkdir()
        sparse_dir = annotations_dir / "GroundTruth-SparseAnnotations"
        sparse_dir.mkdir()
        (sparse_dir / "train").mkdir()
        (sparse_dir / "val").mkdir()
        dense_dir = annotations_dir / "Interpolations-DenseAnnotations"
        dense_dir.mkdir()
        (dense_dir / "train").mkdir()
        (dense_dir / "val").mkdir()
        
        # Create metadata files
        class_file = annotations_dir / "EPIC_100_noun_classes_v2.csv"
        pd.DataFrame({'noun_id': [], 'noun': []}).to_csv(class_file, index=False)
        
        frame_file = annotations_dir / "frame_mapping.json"
        frame_file.write_text('{}')
        
        config = {
            'name': 'test_epic',
            'image_path': str(rgb_frames_dir),
            'sparse_annotation_path': str(sparse_dir),
            'dense_annotation_path': str(dense_dir),
            'class_mapping_file': str(class_file),
            'frame_mapping_file': str(frame_file)
        }
        
        loader = EpicKitchensVisorLoader(config)
        
        # Should handle empty directories gracefully
        assert len(loader) == 0
        
        stats = loader.get_annotation_coverage_statistics()
        assert stats['total_videos'] == 0