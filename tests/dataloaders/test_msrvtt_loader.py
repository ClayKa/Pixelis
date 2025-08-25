# tests/dataloaders/test_msrvtt_loader.py

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from core.dataloaders.msrvtt_loader import MsrVttLoader


class TestMsrVttLoader:
    """Test suite for MsrVttLoader."""

    @pytest.fixture
    def mock_msrvtt_config(self):
        """Create a mock MSR-VTT configuration."""
        temp_dir = tempfile.mkdtemp()
        videos_dir = Path(temp_dir) / "videos"
        annotations_dir = Path(temp_dir) / "annotations"
        videos_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create mock video files
        video_files = ["video0.mp4", "video1.mp4", "video2.mp4"]
        for video_file in video_files:
            video_path = videos_dir / video_file
            video_path.write_text("mock video content")
        
        # Create category mapping file
        category_data = "music\t0\npeople\t1\ngaming\t2\nsports/actions\t3\n"
        category_file = annotations_dir / "category.txt"
        category_file.write_text(category_data)
        
        # Create main annotation file
        annotation_data = [
            {
                "video_id": "video0",
                "video": "video0.mp4",
                "caption": [
                    "a car is shown",
                    "a group is dancing",
                    "a man drives a vehicle through the countryside"
                ],
                "source": "MSR-VTT",
                "category": 0,
                "url": "https://www.youtube.com/watch?v=example1",
                "start time": 137.72,
                "end time": 149.44,
                "id": 0
            },
            {
                "video_id": "video1",
                "video": "video1.mp4",
                "caption": [
                    "a woman cooks in the kitchen",
                    "ingredients being added to pot"
                ],
                "source": "MSR-VTT",
                "category": 1,
                "url": "https://www.youtube.com/watch?v=example2",
                "start time": 20.5,
                "end time": 35.8,
                "id": 1
            },
            {
                "video_id": "video2",
                "video": "video2.mp4",
                "caption": [
                    "gameplay footage is shown",
                    "player navigating through level"
                ],
                "source": "MSR-VTT",
                "category": 2,
                "url": "https://www.youtube.com/watch?v=example3",
                "start time": 0.0,
                "end time": 12.3,
                "id": 2
            }
        ]
        
        annotation_file = annotations_dir / "msrvtt_train_9k.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create raw captions pickle file
        raw_captions_data = {
            "video0": ["additional caption 1", "additional caption 2"],
            "video1": ["cooking related caption"],
            "video2": []
        }
        
        raw_captions_file = annotations_dir / "raw-captions.pkl"
        with open(raw_captions_file, 'wb') as f:
            pickle.dump(raw_captions_data, f)
        
        return {
            'name': 'test_msrvtt',
            'path': str(videos_dir),
            'annotation_file': str(annotation_file),
            'category_file': str(category_file),
            'raw_captions_file': str(raw_captions_file)
        }

    @pytest.fixture
    def loader(self, mock_msrvtt_config):
        """Create a MsrVttLoader instance for testing."""
        return MsrVttLoader(mock_msrvtt_config)

    def test_init_success(self, mock_msrvtt_config):
        """Test successful initialization of MsrVttLoader."""
        loader = MsrVttLoader(mock_msrvtt_config)
        
        assert loader.videos_path == Path(mock_msrvtt_config['path'])
        assert loader.annotation_file == Path(mock_msrvtt_config['annotation_file'])
        assert loader.category_file == Path(mock_msrvtt_config['category_file'])
        assert loader.raw_captions_file == Path(mock_msrvtt_config['raw_captions_file'])
        assert len(loader._index) > 0
        assert len(loader._category_id_to_name) > 0

    def test_init_missing_required_config(self):
        """Test initialization failure with missing required config keys."""
        incomplete_config = {'path': '/some/path'}
        
        with pytest.raises(ValueError, match="MsrVttLoader config must include 'annotation_file'"):
            MsrVttLoader(incomplete_config)

    def test_init_invalid_paths(self):
        """Test initialization failure with invalid paths."""
        config = {
            'name': 'test_msrvtt',
            'path': '/nonexistent/videos',
            'annotation_file': '/nonexistent/annotation.json',
            'category_file': '/nonexistent/category.txt'
        }
        
        with pytest.raises(FileNotFoundError, match="Videos directory not found"):
            MsrVttLoader(config)

    def test_init_without_raw_captions(self, mock_msrvtt_config):
        """Test initialization without raw captions file."""
        config = mock_msrvtt_config.copy()
        del config['raw_captions_file']
        
        loader = MsrVttLoader(config)
        assert loader.raw_captions_file is None
        assert len(loader._index) > 0

    def test_build_index_structure(self, loader):
        """Test that _build_index creates proper index structure."""
        assert isinstance(loader._index, list)
        assert len(loader._index) > 0
        
        # Test enriched video info
        for video_info in loader._index:
            assert 'video_id' in video_info
            assert 'video' in video_info
            assert 'caption' in video_info
            assert 'category' in video_info
            assert 'category_name' in video_info
            assert 'raw_captions' in video_info
            assert 'duration' in video_info

    def test_category_mapping_loading(self, loader):
        """Test category mapping loading."""
        assert len(loader._category_id_to_name) == 4
        assert loader._category_id_to_name[0] == 'music'
        assert loader._category_id_to_name[1] == 'people'
        assert loader._category_id_to_name[2] == 'gaming'
        assert loader._category_id_to_name[3] == 'sports/actions'

    def test_raw_captions_loading(self, loader):
        """Test raw captions loading."""
        assert len(loader._video_id_to_raw_captions) == 3
        assert 'video0' in loader._video_id_to_raw_captions
        assert len(loader._video_id_to_raw_captions['video0']) == 2
        assert 'video1' in loader._video_id_to_raw_captions
        assert len(loader._video_id_to_raw_captions['video1']) == 1

    def test_get_item_basic_structure(self, loader):
        """Test basic structure of get_item output."""
        sample = loader.get_item(0)
        
        # Test base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'video'
        assert 'annotations' in sample
        
        # Test MSR-VTT-specific annotations
        annotations = sample['annotations']
        assert 'msrvtt_video_captioning' in annotations
        assert 'video_metadata' in annotations
        assert 'dataset_info' in annotations

    def test_get_item_video_captioning_annotations(self, loader):
        """Test MSR-VTT video captioning annotation processing."""
        sample = loader.get_item(0)
        
        msrvtt_annotations = sample['annotations']['msrvtt_video_captioning']
        
        # Test required fields
        assert 'video_id' in msrvtt_annotations
        assert 'primary_captions' in msrvtt_annotations
        assert 'raw_captions' in msrvtt_annotations
        assert 'all_captions' in msrvtt_annotations
        assert 'num_captions' in msrvtt_annotations
        assert 'caption_statistics' in msrvtt_annotations
        assert 'temporal_info' in msrvtt_annotations
        assert 'category_info' in msrvtt_annotations
        assert 'source_info' in msrvtt_annotations
        
        # Test data types and values
        assert isinstance(msrvtt_annotations['video_id'], str)
        assert isinstance(msrvtt_annotations['primary_captions'], list)
        assert isinstance(msrvtt_annotations['raw_captions'], list)
        assert isinstance(msrvtt_annotations['all_captions'], list)
        assert isinstance(msrvtt_annotations['num_captions'], int)
        assert msrvtt_annotations['num_captions'] > 0

    def test_get_item_temporal_info(self, loader):
        """Test temporal information processing."""
        sample = loader.get_item(0)
        
        temporal_info = sample['annotations']['msrvtt_video_captioning']['temporal_info']
        assert 'start_time' in temporal_info
        assert 'end_time' in temporal_info
        assert 'duration' in temporal_info
        
        assert isinstance(temporal_info['start_time'], float)
        assert isinstance(temporal_info['end_time'], float)
        assert isinstance(temporal_info['duration'], float)
        
        # Test duration calculation
        expected_duration = temporal_info['end_time'] - temporal_info['start_time']
        assert abs(temporal_info['duration'] - expected_duration) < 1e-6

    def test_get_item_category_info(self, loader):
        """Test category information processing."""
        sample = loader.get_item(0)
        
        category_info = sample['annotations']['msrvtt_video_captioning']['category_info']
        assert 'category_id' in category_info
        assert 'category_name' in category_info
        
        assert isinstance(category_info['category_id'], int)
        assert isinstance(category_info['category_name'], str)
        assert category_info['category_name'] == 'music'  # video0 has category 0 (music)

    def test_get_item_dataset_info(self, loader):
        """Test dataset_info field completeness."""
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        assert dataset_info['task_type'] == 'video_captioning'
        assert dataset_info['source'] == 'MSR-VTT'
        assert dataset_info['suitable_for_select_frame'] == True
        assert dataset_info['suitable_for_temporal_reasoning'] == True
        assert dataset_info['has_multiple_captions'] == True
        assert dataset_info['has_temporal_bounds'] == True
        assert dataset_info['has_category_labels'] == True
        assert isinstance(dataset_info['num_categories'], int)
        assert dataset_info['num_categories'] > 0
        assert dataset_info['video_format'] == 'mp4'

    def test_get_item_out_of_range(self, loader):
        """Test get_item with out-of-range index."""
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(len(loader))

    def test_caption_statistics(self, loader):
        """Test caption statistics calculation."""
        sample = loader.get_item(0)
        
        caption_stats = sample['annotations']['msrvtt_video_captioning']['caption_statistics']
        assert 'avg_length' in caption_stats
        assert 'min_length' in caption_stats
        assert 'max_length' in caption_stats
        assert 'total_words' in caption_stats
        assert 'unique_words' in caption_stats
        assert 'avg_words_per_caption' in caption_stats
        assert 'vocabulary_diversity' in caption_stats
        
        assert isinstance(caption_stats['avg_length'], float)
        assert isinstance(caption_stats['min_length'], int)
        assert isinstance(caption_stats['max_length'], int)
        assert isinstance(caption_stats['total_words'], int)
        assert isinstance(caption_stats['unique_words'], int)
        assert isinstance(caption_stats['avg_words_per_caption'], float)
        assert isinstance(caption_stats['vocabulary_diversity'], float)

    def test_analyze_captions_empty(self, loader):
        """Test caption analysis with empty caption list."""
        stats = loader._analyze_captions([])
        
        assert stats['avg_length'] == 0.0
        assert stats['min_length'] == 0
        assert stats['max_length'] == 0
        assert stats['total_words'] == 0
        assert stats['unique_words'] == 0
        assert stats['avg_words_per_caption'] == 0.0

    def test_analyze_captions_normal(self, loader):
        """Test caption analysis with normal captions."""
        captions = ["a cat sits", "the dog runs quickly", "animals playing"]
        stats = loader._analyze_captions(captions)
        
        assert stats['min_length'] == len("a cat sits")
        assert stats['max_length'] == len("the dog runs quickly")
        assert stats['total_words'] == 9  # "a cat sits the dog runs quickly animals playing" = 9 words
        assert stats['unique_words'] == 9  # All words are unique in this case
        assert stats['avg_words_per_caption'] > 0

    def test_get_samples_by_category(self, loader):
        """Test filtering samples by category."""
        # Test with existing category
        music_samples = loader.get_samples_by_category("music")
        assert isinstance(music_samples, list)
        assert len(music_samples) > 0
        
        for sample in music_samples:
            category_info = sample['annotations']['msrvtt_video_captioning']['category_info']
            assert category_info['category_name'] == 'music'
        
        # Test with non-existing category
        nonexistent_samples = loader.get_samples_by_category("nonexistent_category")
        assert isinstance(nonexistent_samples, list)
        assert len(nonexistent_samples) == 0

    def test_get_category_statistics(self, loader):
        """Test category statistics generation."""
        stats = loader.get_category_statistics()
        
        assert 'total_categories' in stats
        assert 'total_videos' in stats
        assert 'category_distribution' in stats
        assert 'most_common_categories' in stats
        assert 'available_categories' in stats
        
        assert isinstance(stats['total_categories'], int)
        assert isinstance(stats['total_videos'], int)
        assert isinstance(stats['category_distribution'], dict)
        assert isinstance(stats['most_common_categories'], list)
        assert isinstance(stats['available_categories'], list)
        
        assert stats['total_categories'] == 4
        assert stats['total_videos'] == len(loader._index)

    def test_get_samples_by_duration(self, loader):
        """Test filtering samples by duration."""
        # Test minimum duration filter
        short_videos = loader.get_samples_by_duration(min_duration=0.0, max_duration=15.0)
        assert isinstance(short_videos, list)
        
        for sample in short_videos:
            duration = sample['annotations']['video_metadata']['duration_seconds']
            assert 0.0 <= duration <= 15.0
        
        # Test with high minimum (should return fewer or no results)
        long_videos = loader.get_samples_by_duration(min_duration=100.0)
        assert isinstance(long_videos, list)
        assert len(long_videos) <= len(loader._index)

    def test_get_caption_diversity_statistics(self, loader):
        """Test caption diversity statistics generation."""
        stats = loader.get_caption_diversity_statistics()
        
        assert 'total_videos_with_captions' in stats
        assert 'total_captions' in stats
        assert 'total_unique_words' in stats
        assert 'avg_captions_per_video' in stats
        assert 'caption_length_statistics' in stats
        assert 'word_statistics' in stats
        
        # Test caption length statistics
        length_stats = stats['caption_length_statistics']
        assert 'avg_length' in length_stats
        assert 'min_avg_length' in length_stats
        assert 'max_avg_length' in length_stats
        
        # Test word statistics
        word_stats = stats['word_statistics']
        assert 'avg_words_per_caption' in word_stats
        assert 'vocabulary_diversity' in word_stats
        
        assert isinstance(stats['total_videos_with_captions'], int)
        assert isinstance(stats['total_captions'], int)
        assert isinstance(stats['total_unique_words'], int)
        assert stats['total_videos_with_captions'] > 0
        assert stats['total_captions'] > 0

    def test_get_temporal_distribution_statistics(self, loader):
        """Test temporal distribution statistics."""
        stats = loader.get_temporal_distribution_statistics()
        
        assert 'total_videos' in stats
        assert 'videos_with_duration' in stats
        assert 'duration_statistics' in stats
        assert 'temporal_bounds' in stats
        
        # Test duration statistics
        duration_stats = stats['duration_statistics']
        assert 'avg_duration' in duration_stats
        assert 'min_duration' in duration_stats
        assert 'max_duration' in duration_stats
        assert 'median_duration' in duration_stats
        
        # Test temporal bounds
        temporal_bounds = stats['temporal_bounds']
        assert 'avg_start_time' in temporal_bounds
        assert 'avg_end_time' in temporal_bounds
        
        assert isinstance(stats['total_videos'], int)
        assert isinstance(stats['videos_with_duration'], int)
        assert stats['total_videos'] == len(loader._index)

    def test_missing_video_files(self, mock_msrvtt_config):
        """Test behavior when some video files are missing."""
        # Remove one video file
        videos_dir = Path(mock_msrvtt_config['path'])
        missing_video = videos_dir / "video1.mp4"
        missing_video.unlink()
        
        loader = MsrVttLoader(mock_msrvtt_config)
        
        # Should only include videos that exist on disk
        assert len(loader._index) == 2  # One less due to missing video
        
        # Check that missing video is not in index
        video_ids = [v['video_id'] for v in loader._index]
        assert 'video1' not in video_ids

    @patch('core.dataloaders.msrvtt_loader.logger')
    def test_logging_calls(self, mock_logger, mock_msrvtt_config):
        """Test that appropriate logging calls are made."""
        MsrVttLoader(mock_msrvtt_config)
        
        # Verify logging calls were made during initialization
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Should log loading annotations, categories, raw captions, etc.
        assert any("Loading MSR-VTT annotations" in call for call in log_calls)
        assert any("categories" in call for call in log_calls)
        assert any("raw captions" in call for call in log_calls)

    def test_len_method(self, loader):
        """Test __len__ method."""
        length = len(loader)
        assert isinstance(length, int)
        assert length > 0
        assert length == len(loader._index)

    def test_malformed_category_file(self, mock_msrvtt_config):
        """Test handling of malformed category file."""
        # Create malformed category file
        category_file = Path(mock_msrvtt_config['category_file'])
        category_file.write_text("invalid_format_line\n")
        
        # Should still initialize but with empty category mapping
        loader = MsrVttLoader(mock_msrvtt_config)
        assert len(loader._category_id_to_name) == 0

    def test_missing_raw_captions_file(self, mock_msrvtt_config):
        """Test handling of missing raw captions file."""
        # Remove raw captions file
        raw_captions_file = Path(mock_msrvtt_config['raw_captions_file'])
        raw_captions_file.unlink()
        
        with pytest.raises(FileNotFoundError, match="Raw captions file not found"):
            MsrVttLoader(mock_msrvtt_config)

    def test_malformed_raw_captions_file(self, mock_msrvtt_config):
        """Test handling of malformed raw captions file."""
        # Create malformed pickle file
        raw_captions_file = Path(mock_msrvtt_config['raw_captions_file'])
        with open(raw_captions_file, 'wb') as f:
            pickle.dump("invalid_data_structure", f)
        
        # Should handle gracefully with warning
        with patch('core.dataloaders.msrvtt_loader.logger') as mock_logger:
            loader = MsrVttLoader(mock_msrvtt_config)
            assert len(loader._video_id_to_raw_captions) == 0
            mock_logger.warning.assert_called()

    def test_edge_case_empty_annotations(self, mock_msrvtt_config):
        """Test handling of empty annotation file."""
        # Create empty annotation file
        annotation_file = Path(mock_msrvtt_config['annotation_file'])
        with open(annotation_file, 'w') as f:
            json.dump([], f)
        
        loader = MsrVttLoader(mock_msrvtt_config)
        assert len(loader._index) == 0

    def test_video_with_missing_fields(self, mock_msrvtt_config):
        """Test handling of video entries with missing fields."""
        # Create annotation with missing fields
        incomplete_annotation = [
            {
                "video_id": "video0",
                "video": "video0.mp4",
                "caption": ["single caption"],
                # Missing: category, start time, end time, etc.
                "id": 0
            }
        ]
        
        annotation_file = Path(mock_msrvtt_config['annotation_file'])
        with open(annotation_file, 'w') as f:
            json.dump(incomplete_annotation, f)
        
        loader = MsrVttLoader(mock_msrvtt_config)
        assert len(loader._index) == 1
        
        # Should handle missing fields gracefully
        sample = loader.get_item(0)
        category_info = sample['annotations']['msrvtt_video_captioning']['category_info']
        assert category_info['category_id'] == -1  # Default for missing category
        assert category_info['category_name'] == 'category_-1'  # Default name format