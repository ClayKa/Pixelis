# tests/dataloaders/test_starqa_loader.py

import json
import pickle
import tempfile
import csv
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np

from core.dataloaders.starqa_loader import StarqaLoader


class TestStarqaLoader:
    """Test suite for StarqaLoader."""

    @pytest.fixture
    def mock_starqa_config(self):
        """Create a mock STARQA configuration."""
        temp_dir = tempfile.mkdtemp()
        videos_dir = Path(temp_dir) / "videos"
        annotations_dir = Path(temp_dir) / "annotations"
        videos_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create mock video files
        video_files = ["TJZ0P.mp4", "5UFCJ.mp4", "TEST1.mp4"]
        for video_file in video_files:
            video_path = videos_dir / video_file
            video_path.write_text("mock video content")
        
        # Create main annotation file
        qa_data = [
            {
                "question_id": "Interaction_T1_4",
                "video_id": "TJZ0P",
                "start": 7.7,
                "end": 15.7,
                "question": "Which object was eaten by the person?",
                "answer": "The sandwich.",
                "question_program": [
                    {"function": "Situations", "value_input": []},
                    {"function": "Actions", "value_input": []},
                    {"function": "Filter_Actions_with_Verb", "value_input": ["eat"]},
                    {"function": "Unique", "value_input": []},
                    {"function": "Query_Objs", "value_input": []}
                ],
                "choices": [
                    {"choice_id": 0, "choice": "The sandwich.", "choice_program": [{"function": "Equal", "value_input": ["sandwich"]}]},
                    {"choice_id": 1, "choice": "The medicine.", "choice_program": [{"function": "Equal", "value_input": ["medicine"]}]},
                    {"choice_id": 2, "choice": "The blanket.", "choice_program": [{"function": "Equal", "value_input": ["blanket"]}]},
                    {"choice_id": 3, "choice": "The box.", "choice_program": [{"function": "Equal", "value_input": ["box"]}]}
                ],
                "situations": {
                    "000245": {
                        "rel_pairs": [["o000", "o006"], ["o000", "o017"]],
                        "rel_labels": ["r002", "r009"],
                        "actions": ["a051"],
                        "bbox": [[98.48, 176.53, 244.68, 269.99], [234.38, 156.84, 292.24, 207.01]],
                        "bbox_labels": ["o006", "o017"]
                    }
                }
            },
            {
                "question_id": "Interaction_T2_6132",
                "video_id": "5UFCJ",
                "start": 20.2,
                "end": 30.0,
                "question": "What did the person do with the medicine?",
                "answer": "Ate.",
                "question_program": [
                    {"function": "Situations", "value_input": []},
                    {"function": "Filter_Situations_with_Obj", "value_input": ["medicine"]},
                    {"function": "Actions", "value_input": []},
                    {"function": "Filter_Actions_with_Obj", "value_input": ["medicine"]},
                    {"function": "Unique", "value_input": []},
                    {"function": "Query_Verbs", "value_input": []}
                ],
                "choices": [
                    {"choice_id": 0, "choice": "Put down.", "choice_program": [{"function": "Equal", "value_input": ["put"]}]},
                    {"choice_id": 1, "choice": "Lied on.", "choice_program": [{"function": "Equal", "value_input": ["lie"]}]},
                    {"choice_id": 2, "choice": "Sat on.", "choice_program": [{"function": "Equal", "value_input": ["sit"]}]},
                    {"choice_id": 3, "choice": "Ate.", "choice_program": [{"function": "Equal", "value_input": ["eat"]}]}
                ],
                "situations": {
                    "000661": {
                        "rel_pairs": [["o000", "o005"], ["o000", "o010"]],
                        "rel_labels": ["r002", "r009"],
                        "actions": ["a100"],
                        "bbox": [[1.0, 2.5, 145.0, 169.5], [156.06, 77.97, 167.25, 89.16]],
                        "bbox_labels": ["o005", "o010"]
                    }
                }
            },
            {
                "question_id": "Test_Q1",
                "video_id": "TEST1",
                "start": 5.0,
                "end": 12.0,
                "question": "What color is the object?",
                "answer": "Blue.",
                "question_program": [
                    {"function": "Situations", "value_input": []},
                    {"function": "Query_Color", "value_input": []}
                ],
                "choices": [
                    {"choice_id": 0, "choice": "Blue.", "choice_program": [{"function": "Equal", "value_input": ["blue"]}]},
                    {"choice_id": 1, "choice": "Red.", "choice_program": [{"function": "Equal", "value_input": ["red"]}]}
                ],
                "situations": {}
            }
        ]
        
        annotation_file = annotations_dir / "STAR_train.json"
        with open(annotation_file, 'w') as f:
            json.dump(qa_data, f)
        
        # Create person bbox pickle file
        person_bbox_data = {
            "TJZ0P.mp4/000245.png": {
                "bbox": np.array([[98.48, 176.53, 244.68, 269.99]], dtype=np.float32),
                "bbox_score": np.array([0.99], dtype=np.float32),
                "bbox_size": (480, 270),
                "bbox_mode": "xyxy",
                "keypoints": np.array([[[100, 200, 0.9], [150, 180, 0.8]]], dtype=np.float32)
            },
            "5UFCJ.mp4/000661.png": {
                "bbox": np.array([[1.0, 2.5, 145.0, 169.5]], dtype=np.float32),
                "bbox_score": np.array([0.95], dtype=np.float32),
                "bbox_size": (480, 270),
                "bbox_mode": "xyxy",
                "keypoints": None
            }
        }
        
        person_bbox_file = annotations_dir / "person_bbox.pkl"
        with open(person_bbox_file, 'wb') as f:
            pickle.dump(person_bbox_data, f)
        
        # Create object bbox and relationship pickle file
        object_relationship_data = {
            "TJZ0P.mp4/000245.png": [
                {
                    "class": "sandwich",
                    "bbox": [98.48, 176.53, 244.68, 269.99],
                    "attention_relationship": None,
                    "spatial_relationship": "on_table",
                    "contacting_relationship": "holding",
                    "metadata": {"tag": "TJZ0P.mp4/sandwich/000245", "set": "train"},
                    "visible": True
                }
            ],
            "5UFCJ.mp4/000661.png": [
                {
                    "class": "medicine",
                    "bbox": [1.0, 2.5, 145.0, 169.5],
                    "attention_relationship": "looking_at",
                    "spatial_relationship": "in_hand",
                    "contacting_relationship": "touching",
                    "metadata": {"tag": "5UFCJ.mp4/medicine/000661", "set": "train"},
                    "visible": True
                }
            ]
        }
        
        object_relationship_file = annotations_dir / "object_bbox_and_relationship.pkl"
        with open(object_relationship_file, 'wb') as f:
            pickle.dump(object_relationship_data, f)
        
        # Create keyframes CSV file
        keyframes_data = [
            {"video_id": "TJZ0P", "keyframe_id": "000245"},
            {"video_id": "TJZ0P", "keyframe_id": "000250"},
            {"video_id": "5UFCJ", "keyframe_id": "000661"}
        ]
        
        keyframes_file = annotations_dir / "Video_Keyframe_IDs.csv"
        with open(keyframes_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "keyframe_id"])
            writer.writeheader()
            writer.writerows(keyframes_data)
        
        # Create segments CSV file
        segments_data = [
            {"video_id": "TJZ0P", "start_time": "0.0", "end_time": "30.0", "segment_type": "action"},
            {"video_id": "5UFCJ", "start_time": "10.0", "end_time": "40.0", "segment_type": "interaction"},
            {"video_id": "TEST1", "start_time": "0.0", "end_time": "20.0", "segment_type": "observation"}
        ]
        
        segments_file = annotations_dir / "Video_Segments.csv"
        with open(segments_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "start_time", "end_time", "segment_type"])
            writer.writeheader()
            writer.writerows(segments_data)
        
        return {
            'name': 'test_starqa',
            'path': str(videos_dir),
            'annotation_file': str(annotation_file),
            'supplementary_paths': {
                'person_bbox': str(person_bbox_file),
                'object_bbox_and_relationship': str(object_relationship_file),
                'video_keyframe_ids': str(keyframes_file),
                'video_segments': str(segments_file)
            }
        }

    @pytest.fixture
    def loader(self, mock_starqa_config):
        """Create a StarqaLoader instance for testing."""
        return StarqaLoader(mock_starqa_config)

    def test_init_success(self, mock_starqa_config):
        """Test successful initialization of StarqaLoader."""
        loader = StarqaLoader(mock_starqa_config)
        
        assert loader.videos_path == Path(mock_starqa_config['path'])
        assert loader.annotation_file == Path(mock_starqa_config['annotation_file'])
        assert len(loader._index) > 0
        assert len(loader._person_bbox_data) > 0
        assert len(loader._object_relationship_data) > 0

    def test_init_missing_required_config(self):
        """Test initialization failure with missing required config keys."""
        incomplete_config = {'path': '/some/path'}
        
        with pytest.raises(ValueError, match="StarqaLoader config must include 'annotation_file'"):
            StarqaLoader(incomplete_config)

    def test_init_invalid_paths(self):
        """Test initialization failure with invalid paths."""
        config = {
            'name': 'test_starqa',
            'path': '/nonexistent/videos',
            'annotation_file': '/nonexistent/annotation.json'
        }
        
        with pytest.raises(FileNotFoundError, match="Videos directory not found"):
            StarqaLoader(config)

    def test_init_without_supplementary_paths(self, mock_starqa_config):
        """Test initialization without supplementary paths."""
        config = mock_starqa_config.copy()
        del config['supplementary_paths']
        
        loader = StarqaLoader(config)
        assert len(loader._index) > 0
        assert len(loader._person_bbox_data) == 0
        assert len(loader._object_relationship_data) == 0

    def test_build_index_structure(self, loader):
        """Test that _build_index creates proper index structure."""
        assert isinstance(loader._index, list)
        assert len(loader._index) > 0
        
        # Test QA sample structure
        for qa_sample in loader._index:
            assert 'question_id' in qa_sample
            assert 'video_id' in qa_sample
            assert 'question' in qa_sample
            assert 'answer' in qa_sample
            assert 'start' in qa_sample
            assert 'end' in qa_sample
            assert 'question_program' in qa_sample
            assert 'choices' in qa_sample

    def test_supplementary_data_loading(self, loader):
        """Test supplementary data loading."""
        # Test person bbox data
        assert len(loader._person_bbox_data) > 0
        assert any("TJZ0P.mp4/" in key for key in loader._person_bbox_data.keys())
        
        # Test object relationship data
        assert len(loader._object_relationship_data) > 0
        assert any("TJZ0P.mp4/" in key for key in loader._object_relationship_data.keys())
        
        # Test keyframes data
        assert len(loader._video_keyframes) > 0
        assert "TJZ0P" in loader._video_keyframes
        
        # Test segments data
        assert len(loader._video_segments) > 0
        assert "TJZ0P" in loader._video_segments

    def test_get_item_basic_structure(self, loader):
        """Test basic structure of get_item output."""
        sample = loader.get_item(0)
        
        # Test base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'video'
        assert 'annotations' in sample
        
        # Test STARQA-specific annotations
        annotations = sample['annotations']
        assert 'starqa_situated_video_qa' in annotations
        assert 'video_metadata' in annotations
        assert 'dataset_info' in annotations

    def test_get_item_starqa_annotations(self, loader):
        """Test STARQA-specific annotation processing."""
        sample = loader.get_item(0)
        
        starqa_annotations = sample['annotations']['starqa_situated_video_qa']
        
        # Test required fields
        assert 'question_id' in starqa_annotations
        assert 'video_id' in starqa_annotations
        assert 'question' in starqa_annotations
        assert 'answer' in starqa_annotations
        assert 'temporal_bounds' in starqa_annotations
        assert 'question_structure' in starqa_annotations
        assert 'choices' in starqa_annotations
        assert 'num_choices' in starqa_annotations
        assert 'situations' in starqa_annotations
        assert 'num_situations' in starqa_annotations
        
        # Test temporal bounds
        temporal_bounds = starqa_annotations['temporal_bounds']
        assert 'start_time' in temporal_bounds
        assert 'end_time' in temporal_bounds
        assert 'duration' in temporal_bounds
        assert isinstance(temporal_bounds['duration'], float)
        assert temporal_bounds['duration'] >= 0
        
        # Test question structure
        question_structure = starqa_annotations['question_structure']
        assert 'question_program' in question_structure
        assert 'program_length' in question_structure
        assert 'functions_used' in question_structure
        assert isinstance(question_structure['functions_used'], list)

    def test_get_item_supplementary_annotations(self, loader):
        """Test supplementary annotation processing."""
        sample = loader.get_item(0)
        
        starqa_annotations = sample['annotations']['starqa_situated_video_qa']
        
        # Should have supplementary annotations for this video
        if 'supplementary_annotations' in starqa_annotations:
            supp_annotations = starqa_annotations['supplementary_annotations']
            
            # Test person annotations
            if 'person_annotations' in supp_annotations:
                person_annotations = supp_annotations['person_annotations']
                assert isinstance(person_annotations, list)
                
                for annotation in person_annotations:
                    assert 'frame_id' in annotation
                    assert 'bboxes' in annotation
                    assert 'bbox_scores' in annotation
                    assert 'bbox_size' in annotation
                    assert 'bbox_mode' in annotation
            
            # Test object/relationship annotations
            if 'object_relationship_annotations' in supp_annotations:
                obj_annotations = supp_annotations['object_relationship_annotations']
                assert isinstance(obj_annotations, list)
                
                for annotation in obj_annotations:
                    assert 'frame_id' in annotation
                    assert 'objects' in annotation
                    
                    for obj in annotation['objects']:
                        assert 'class' in obj
                        assert 'bbox' in obj

    def test_get_item_dataset_info(self, loader):
        """Test dataset_info field completeness."""
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        assert dataset_info['task_type'] == 'situated_video_qa'
        assert dataset_info['source'] == 'STARQA'
        assert dataset_info['suitable_for_select_frame'] == True
        assert dataset_info['suitable_for_temporal_reasoning'] == True
        assert dataset_info['suitable_for_spatial_reasoning'] == True
        assert isinstance(dataset_info['has_structured_programs'], bool)
        assert isinstance(dataset_info['has_multiple_choice'], bool)
        assert isinstance(dataset_info['has_situation_grounding'], bool)
        assert dataset_info['video_format'] == 'mp4'
        assert 'supplementary_data_available' in dataset_info

    def test_get_item_out_of_range(self, loader):
        """Test get_item with out-of-range index."""
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(len(loader))

    def test_get_available_supplementary_data(self, loader):
        """Test checking available supplementary data."""
        available_data = loader._get_available_supplementary_data("TJZ0P")
        
        assert isinstance(available_data, dict)
        assert 'person_bboxes' in available_data
        assert 'object_relationships' in available_data
        assert 'keyframes' in available_data
        assert 'segments' in available_data
        
        # TJZ0P should have all types of data
        assert available_data['person_bboxes'] == True
        assert available_data['object_relationships'] == True
        assert available_data['keyframes'] == True
        assert available_data['segments'] == True

    def test_get_samples_by_question_type(self, loader):
        """Test filtering samples by question type."""
        # Test with existing function
        verb_samples = loader.get_samples_by_question_type("Filter_Actions_with_Verb")
        assert isinstance(verb_samples, list)
        assert len(verb_samples) > 0
        
        for sample in verb_samples:
            starqa_data = sample['annotations']['starqa_situated_video_qa']
            functions_used = starqa_data['question_structure']['functions_used']
            assert "Filter_Actions_with_Verb" in functions_used
        
        # Test with non-existing function
        nonexistent_samples = loader.get_samples_by_question_type("NonexistentFunction")
        assert isinstance(nonexistent_samples, list)
        assert len(nonexistent_samples) == 0

    def test_get_samples_by_duration(self, loader):
        """Test filtering samples by duration."""
        # Test short duration filter
        short_samples = loader.get_samples_by_duration(min_duration=0.0, max_duration=10.0)
        assert isinstance(short_samples, list)
        
        for sample in short_samples:
            temporal_bounds = sample['annotations']['starqa_situated_video_qa']['temporal_bounds']
            assert temporal_bounds['duration'] <= 10.0
        
        # Test with high minimum (should return fewer or no results)
        long_samples = loader.get_samples_by_duration(min_duration=20.0)
        assert isinstance(long_samples, list)

    def test_get_question_type_statistics(self, loader):
        """Test question type statistics generation."""
        stats = loader.get_question_type_statistics()
        
        assert 'total_samples' in stats
        assert 'function_usage' in stats
        assert 'most_common_functions' in stats
        assert 'program_length_statistics' in stats
        assert 'choice_statistics' in stats
        assert 'situation_statistics' in stats
        
        # Test program length statistics
        prog_stats = stats['program_length_statistics']
        assert 'avg_length' in prog_stats
        assert 'min_length' in prog_stats
        assert 'max_length' in prog_stats
        assert 'median_length' in prog_stats
        
        # Test choice statistics
        choice_stats = stats['choice_statistics']
        assert 'avg_choices' in choice_stats
        assert 'min_choices' in choice_stats
        assert 'max_choices' in choice_stats
        
        # Test situation statistics
        situation_stats = stats['situation_statistics']
        assert 'avg_situations' in situation_stats
        assert 'samples_with_situations' in situation_stats
        assert 'max_situations' in situation_stats
        
        assert isinstance(stats['total_samples'], int)
        assert stats['total_samples'] > 0
        assert isinstance(stats['function_usage'], dict)
        assert isinstance(stats['most_common_functions'], list)

    def test_get_supplementary_data_coverage(self, loader):
        """Test supplementary data coverage statistics."""
        coverage = loader.get_supplementary_data_coverage()
        
        assert 'total_unique_videos' in coverage
        assert 'person_bbox_coverage' in coverage
        assert 'object_relationship_coverage' in coverage
        assert 'keyframe_coverage' in coverage
        assert 'segment_coverage' in coverage
        
        # Test person bbox coverage
        person_cov = coverage['person_bbox_coverage']
        assert 'videos_covered' in person_cov
        assert 'coverage_percentage' in person_cov
        assert 'total_keyframes' in person_cov
        
        assert isinstance(coverage['total_unique_videos'], int)
        assert coverage['total_unique_videos'] > 0
        assert isinstance(person_cov['coverage_percentage'], (int, float))
        assert 0 <= person_cov['coverage_percentage'] <= 100

    def test_get_temporal_distribution_statistics(self, loader):
        """Test temporal distribution statistics."""
        stats = loader.get_temporal_distribution_statistics()
        
        assert 'total_samples' in stats
        assert 'samples_with_temporal_bounds' in stats
        assert 'duration_statistics' in stats
        assert 'start_time_statistics' in stats
        assert 'end_time_statistics' in stats
        
        # Test duration statistics
        duration_stats = stats['duration_statistics']
        assert 'avg_duration' in duration_stats
        assert 'min_duration' in duration_stats
        assert 'max_duration' in duration_stats
        assert 'median_duration' in duration_stats
        
        assert isinstance(stats['total_samples'], int)
        assert isinstance(stats['samples_with_temporal_bounds'], int)
        assert stats['total_samples'] > 0

    def test_missing_video_files(self, mock_starqa_config):
        """Test behavior when some video files are missing."""
        # Remove one video file
        videos_dir = Path(mock_starqa_config['path'])
        missing_video = videos_dir / "5UFCJ.mp4"
        missing_video.unlink()
        
        loader = StarqaLoader(mock_starqa_config)
        
        # Should only include QA samples for videos that exist on disk
        video_ids = [qa['video_id'] for qa in loader._index]
        assert '5UFCJ' not in video_ids
        assert 'TJZ0P' in video_ids

    @patch('core.dataloaders.starqa_loader.logger')
    def test_logging_calls(self, mock_logger, mock_starqa_config):
        """Test that appropriate logging calls are made."""
        StarqaLoader(mock_starqa_config)
        
        # Verify logging calls were made during initialization
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Should log loading QA samples and supplementary data
        assert any("Loading STARQA annotations" in call for call in log_calls)
        assert any("QA samples" in call for call in log_calls)
        assert any("person bbox data" in call for call in log_calls)

    def test_len_method(self, loader):
        """Test __len__ method."""
        length = len(loader)
        assert isinstance(length, int)
        assert length > 0
        assert length == len(loader._index)

    def test_malformed_supplementary_files(self, mock_starqa_config):
        """Test handling of malformed supplementary files."""
        # Create truly malformed pickle file (corrupted binary data)
        person_bbox_file = Path(mock_starqa_config['supplementary_paths']['person_bbox'])
        with open(person_bbox_file, 'wb') as f:
            f.write(b'corrupted_pickle_data_not_valid')
        
        # Should handle gracefully with warning
        with patch('core.dataloaders.starqa_loader.logger') as mock_logger:
            loader = StarqaLoader(mock_starqa_config)
            assert len(loader._person_bbox_data) == 0
            mock_logger.warning.assert_called()

    def test_edge_case_empty_annotations(self, mock_starqa_config):
        """Test handling of empty annotation file."""
        # Create empty annotation file
        annotation_file = Path(mock_starqa_config['annotation_file'])
        with open(annotation_file, 'w') as f:
            json.dump([], f)
        
        loader = StarqaLoader(mock_starqa_config)
        assert len(loader._index) == 0

    def test_qa_sample_with_missing_fields(self, mock_starqa_config):
        """Test handling of QA samples with missing fields."""
        # Create QA sample with missing fields
        incomplete_qa = [
            {
                "question_id": "Incomplete_Q1",
                "video_id": "TJZ0P",
                "question": "What happened?",
                "answer": "Something.",
                # Missing: start, end, question_program, choices, situations
            }
        ]
        
        annotation_file = Path(mock_starqa_config['annotation_file'])
        with open(annotation_file, 'w') as f:
            json.dump(incomplete_qa, f)
        
        loader = StarqaLoader(mock_starqa_config)
        assert len(loader._index) == 1
        
        # Should handle missing fields gracefully
        sample = loader.get_item(0)
        starqa_data = sample['annotations']['starqa_situated_video_qa']
        assert starqa_data['temporal_bounds']['start_time'] == 0.0  # Default
        assert starqa_data['temporal_bounds']['end_time'] == 0.0   # Default
        assert starqa_data['question_structure']['program_length'] == 0  # Empty program
        assert starqa_data['num_choices'] == 0  # No choices
        assert starqa_data['num_situations'] == 0  # No situations

    def test_person_annotations_processing(self, loader):
        """Test person annotation processing."""
        person_annotations = loader._get_person_annotations("TJZ0P")
        
        assert isinstance(person_annotations, list)
        assert len(person_annotations) > 0
        
        for annotation in person_annotations:
            assert 'frame_id' in annotation
            assert 'bboxes' in annotation
            assert 'bbox_scores' in annotation
            assert 'bbox_size' in annotation
            assert 'bbox_mode' in annotation
            assert 'has_keypoints' in annotation
            
            # Check data types
            assert isinstance(annotation['bboxes'], list)
            assert isinstance(annotation['bbox_scores'], list)
            assert isinstance(annotation['bbox_size'], tuple)
            assert isinstance(annotation['bbox_mode'], str)
            assert isinstance(annotation['has_keypoints'], bool)

    def test_object_relationship_annotations_processing(self, loader):
        """Test object and relationship annotation processing."""
        object_annotations = loader._get_object_relationship_annotations("TJZ0P")
        
        assert isinstance(object_annotations, list)
        assert len(object_annotations) > 0
        
        for annotation in object_annotations:
            assert 'frame_id' in annotation
            assert 'objects' in annotation
            assert isinstance(annotation['objects'], list)
            
            for obj in annotation['objects']:
                assert 'class' in obj
                assert 'bbox' in obj
                assert 'attention_relationship' in obj
                assert 'spatial_relationship' in obj
                assert 'contacting_relationship' in obj
                assert 'metadata' in obj
                assert 'visible' in obj
                
                assert isinstance(obj['class'], str)
                assert isinstance(obj['visible'], bool)

    def test_csv_file_loading(self, loader):
        """Test CSV file loading for keyframes and segments."""
        # Test keyframes loading
        assert "TJZ0P" in loader._video_keyframes
        assert "5UFCJ" in loader._video_keyframes
        assert isinstance(loader._video_keyframes["TJZ0P"], list)
        assert len(loader._video_keyframes["TJZ0P"]) > 0
        
        # Test segments loading
        assert "TJZ0P" in loader._video_segments
        assert "5UFCJ" in loader._video_segments
        assert isinstance(loader._video_segments["TJZ0P"], dict)
        
        # Check segment data structure
        tjz0p_segment = loader._video_segments["TJZ0P"]
        assert 'start_time' in tjz0p_segment
        assert 'end_time' in tjz0p_segment
        assert 'segment_type' in tjz0p_segment

    def test_malformed_csv_files(self, mock_starqa_config):
        """Test handling of malformed CSV files."""
        # Create malformed keyframes CSV
        keyframes_file = Path(mock_starqa_config['supplementary_paths']['video_keyframe_ids'])
        keyframes_file.write_text("invalid,csv,format\nno,proper,headers\n")
        
        # Should handle gracefully
        with patch('core.dataloaders.starqa_loader.logger') as mock_logger:
            loader = StarqaLoader(mock_starqa_config)
            assert len(loader._video_keyframes) == 0
            # May or may not log a warning depending on CSV format