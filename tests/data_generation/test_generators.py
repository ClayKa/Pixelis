"""
Unit Tests for Task Generators
===============================
Comprehensive test suite for all data generation components.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data_generation.task_generators import (
    GeometricComparisonTaskGenerator,
    TargetedOCRTaskGenerator,
    SpatioTemporalTaskGenerator,
    ZoomInTaskGenerator,
    SelectFrameTaskGenerator
)


class TestGeometricComparisonTaskGenerator:
    """Test suite for GeometricComparisonTaskGenerator."""
    
    @pytest.fixture
    def mock_coco_path(self):
        """Path to mock COCO data."""
        return Path(__file__).parent.parent / "fixtures" / "mock_data" / "mock_coco.json"
    
    @pytest.fixture
    def generator(self, mock_coco_path):
        """Initialize generator with mock data."""
        return GeometricComparisonTaskGenerator(
            data_source_path=mock_coco_path.parent,
            annotation_file=str(mock_coco_path)  # Pass as string
        )
    
    def test_generate_single_sample_successfully(self, generator):
        """Test successful generation of a single sample."""
        # Generate one sample
        samples = generator.generate(num_samples=1)
        
        # Assert we got exactly one sample
        assert len(samples) == 1
        
        # Get the sample
        sample = samples[0]
        
        # Assert schema is correct
        assert 'question' in sample
        assert 'trajectory' in sample
        assert 'final_answer' in sample
        assert 'provenance' in sample
        
        # Assert provenance is correct
        assert sample['provenance']['source'] == 'mock_coco'
        assert 'image_id' in sample['provenance']
        
        # Assert trajectory contains expected actions
        assert isinstance(sample['trajectory'], list)
        assert len(sample['trajectory']) > 0
        
        # Check for expected visual operations
        actions = [action['action'] for action in sample['trajectory'] if 'action' in action]
        assert 'SEGMENT_OBJECT_AT' in actions
        assert 'GET_PROPERTIES' in actions
        
        # Assert final answer format
        assert isinstance(sample['final_answer'], str)
        assert 'larger' in sample['final_answer'].lower()
    
    def test_handles_annotations_with_missing_area(self):
        """Test handling of malformed annotations missing area field."""
        # Create temporary malformed annotation file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            malformed_data = {
                "images": [{"id": 1, "file_name": "test.jpg"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [10, 10, 20, 20]},  # Missing area
                    {"id": 2, "image_id": 1, "bbox": [30, 30, 40, 40], "area": 1600}
                ]
            }
            json.dump(malformed_data, f)
            temp_path = f.name
        
        try:
            # Initialize generator with malformed data
            generator = GeometricComparisonTaskGenerator(
                data_source_path=Path(temp_path).parent,
                annotation_file=temp_path
            )
            
            # Should not crash
            samples = generator.generate(num_samples=10)
            
            # Should gracefully handle the error
            assert isinstance(samples, list)
            # May generate fewer samples due to invalid data
            assert len(samples) <= 10
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_handles_image_with_single_object(self):
        """Test handling of images with only one object."""
        # Create annotation with single object
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            single_obj_data = {
                "images": [{"id": 1, "file_name": "single.jpg"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [10, 10, 20, 20], "area": 400}
                ]
            }
            json.dump(single_obj_data, f)
            temp_path = f.name
        
        try:
            generator = GeometricComparisonTaskGenerator(
                data_source_path=Path(temp_path).parent,
                annotation_file=temp_path
            )
            
            # Should handle gracefully
            samples = generator.generate(num_samples=1)
            
            # Should return empty list or skip this image
            assert len(samples) == 0
            
        finally:
            Path(temp_path).unlink()
    
    def test_multiple_samples_generation(self, generator):
        """Test generation of multiple samples."""
        samples = generator.generate(num_samples=5)
        
        # Should generate some samples (may be less than 5 due to limited mock data)
        assert len(samples) > 0
        assert len(samples) <= 5
        
        # All samples should be valid
        for sample in samples:
            assert generator.validate_sample(sample)


class TestTargetedOCRTaskGenerator:
    """Test suite for TargetedOCRTaskGenerator."""
    
    @pytest.fixture
    def mock_ocr_path(self):
        """Path to mock OCR data."""
        return Path(__file__).parent.parent / "fixtures" / "mock_data" / "mock_infographics_vqa.jsonl"
    
    @pytest.fixture
    def generator(self, mock_ocr_path):
        """Initialize generator with mock data."""
        return TargetedOCRTaskGenerator(data_source_path=mock_ocr_path)
    
    def test_generate_ocr_sample_successfully(self, generator):
        """Test successful OCR sample generation."""
        samples = generator.generate(num_samples=1)
        
        assert len(samples) == 1
        sample = samples[0]
        
        # Check schema
        assert 'question' in sample
        assert 'trajectory' in sample
        assert 'final_answer' in sample
        assert 'provenance' in sample
        
        # Check trajectory contains READ_TEXT
        actions = [action['action'] for action in sample['trajectory'] if 'action' in action]
        assert 'READ_TEXT' in actions
        
        # Check final answer is text
        assert isinstance(sample['final_answer'], str)
    
    def test_handles_missing_text_field(self):
        """Test handling of data missing text field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": 1, "bbox": [10, 10, 20, 20]}\n')  # Missing text
            f.write('{"id": 2, "bbox": [30, 30, 40, 40], "text": "Valid"}\n')
            temp_path = f.name
        
        try:
            generator = TargetedOCRTaskGenerator(data_source_path=Path(temp_path))
            samples = generator.generate(num_samples=2)
            
            # Should handle gracefully
            assert len(samples) == 2
            # First sample should use default text
            assert samples[0]['final_answer'] == 'Sample text'
            assert samples[1]['final_answer'] == 'Valid'
            
        finally:
            Path(temp_path).unlink()


class TestSpatioTemporalTaskGenerator:
    """Test suite for SpatioTemporalTaskGenerator."""
    
    @pytest.fixture
    def mock_tracking_path(self):
        """Path to mock tracking data."""
        return Path(__file__).parent.parent / "fixtures" / "mock_data" / "mock_mot17_annotations.txt"
    
    @pytest.fixture
    def generator(self, mock_tracking_path):
        """Initialize generator with mock data."""
        return SpatioTemporalTaskGenerator(data_source_path=mock_tracking_path)
    
    def test_generate_tracking_sample_successfully(self, generator):
        """Test successful tracking sample generation."""
        samples = generator.generate(num_samples=1)
        
        assert len(samples) == 1
        sample = samples[0]
        
        # Check schema
        assert generator.validate_sample(sample)
        
        # Check trajectory contains TRACK_OBJECT
        actions = [action['action'] for action in sample['trajectory'] if 'action' in action]
        assert 'TRACK_OBJECT' in actions
        
        # Check answer mentions movement
        assert 'moved from' in sample['final_answer']
    
    def test_handles_single_frame_object(self):
        """Test handling of objects with only one frame."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('1,1,100.0,100.0,50.0,50.0\n')  # Only one frame for object 1
            f.write('1,2,200.0,200.0,60.0,60.0\n')
            f.write('2,2,202.0,202.0,60.0,60.0\n')
            temp_path = f.name
        
        try:
            generator = SpatioTemporalTaskGenerator(data_source_path=Path(temp_path))
            samples = generator.generate(num_samples=2)
            
            # Should only generate sample for object 2
            assert len(samples) == 1
            assert '2' in str(samples[0]['provenance']['object_id'])
            
        finally:
            Path(temp_path).unlink()
    
    def test_handles_malformed_tracking_data(self):
        """Test handling of malformed tracking data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('invalid,data,line\n')
            f.write('1,1,100.0,100.0,50.0,50.0\n')
            f.write('2,1,102.0,102.0,50.0,50.0\n')
            temp_path = f.name
        
        try:
            generator = SpatioTemporalTaskGenerator(data_source_path=Path(temp_path))
            samples = generator.generate(num_samples=1)
            
            # Should skip invalid line and still generate sample
            assert len(samples) == 1
            
        finally:
            Path(temp_path).unlink()


class TestZoomInTaskGenerator:
    """Test suite for ZoomInTaskGenerator."""
    
    @pytest.fixture
    def mock_image_dir(self, tmp_path):
        """Create mock image directory."""
        # Create some dummy image files
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()
        return tmp_path
    
    @pytest.fixture
    def generator(self, mock_image_dir):
        """Initialize generator with mock data."""
        return ZoomInTaskGenerator(data_source_path=mock_image_dir)
    
    def test_generate_zoom_sample_successfully(self, generator):
        """Test successful zoom sample generation."""
        samples = generator.generate(num_samples=1)
        
        assert len(samples) == 1
        sample = samples[0]
        
        # Check schema
        assert generator.validate_sample(sample)
        
        # Check trajectory contains ZOOM_IN
        actions = [action['action'] for action in sample['trajectory'] if 'action' in action]
        assert 'ZOOM_IN' in actions
        
        # Check parameters
        zoom_action = sample['trajectory'][0]
        assert 'scale' in zoom_action['parameters']
        assert zoom_action['parameters']['scale'] > 1
    
    def test_handles_empty_directory(self):
        """Test handling of empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ZoomInTaskGenerator(data_source_path=Path(temp_dir))
            samples = generator.generate(num_samples=1)
            
            # Should return empty list
            assert len(samples) == 0


class TestSelectFrameTaskGenerator:
    """Test suite for SelectFrameTaskGenerator."""
    
    @pytest.fixture
    def mock_video_data(self, tmp_path):
        """Create mock video data file."""
        video_file = tmp_path / "videos.json"
        video_data = [
            {"id": "video1", "key_frame": 15, "duration": 30},
            {"id": "video2", "key_frame": 25, "duration": 50}
        ]
        with open(video_file, 'w') as f:
            json.dump(video_data, f)
        return video_file
    
    @pytest.fixture
    def generator(self, mock_video_data):
        """Initialize generator with mock data."""
        return SelectFrameTaskGenerator(data_source_path=mock_video_data)
    
    def test_generate_frame_selection_successfully(self, generator):
        """Test successful frame selection sample generation."""
        samples = generator.generate(num_samples=1)
        
        assert len(samples) == 1
        sample = samples[0]
        
        # Check schema
        assert generator.validate_sample(sample)
        
        # Check trajectory contains SELECT_FRAME
        actions = [action['action'] for action in sample['trajectory'] if 'action' in action]
        assert 'SELECT_FRAME' in actions
        
        # Check answer mentions frame number
        assert 'Frame' in sample['final_answer']
        assert 'relevant' in sample['final_answer']
    
    def test_handles_missing_video_fields(self):
        """Test handling of videos with missing fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            video_data = [
                {"duration": 30},  # Missing id and key_frame
                {"id": "video2", "key_frame": 25}
            ]
            json.dump(video_data, f)
            temp_path = f.name
        
        try:
            generator = SelectFrameTaskGenerator(data_source_path=Path(temp_path))
            samples = generator.generate(num_samples=2)
            
            # Should handle missing fields with defaults
            assert len(samples) == 2
            assert samples[0]['provenance']['video_id'] == 'unknown'
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])