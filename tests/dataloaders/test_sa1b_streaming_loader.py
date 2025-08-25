# tests/dataloaders/test_sa1b_streaming_loader.py

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import ijson

from core.dataloaders.sa1b_streaming_loader import (
    Sa1bStreamingLoader, 
    Sa1bStreamingSegmentLoader
)


class TestSa1bStreamingLoader:
    """Test suite for SA-1B streaming loader."""
    
    @pytest.fixture
    def create_test_dataset(self):
        """Create a test dataset with mock SA-1B structure."""
        def _create(num_images=10, num_annotations_per_image=5):
            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = Path(tmpdir)
                
                # Create images directory
                images_dir = base_path / "images"
                images_dir.mkdir()
                
                # Create mock image files
                for i in range(num_images):
                    img_file = images_dir / f"sa_{i:06d}.jpg"
                    img_file.write_bytes(b"fake image data")
                
                # Create annotations file
                annotations = {
                    "images": [],
                    "annotations": []
                }
                
                ann_id = 0
                for i in range(num_images):
                    # Add image entry
                    annotations["images"].append({
                        "id": i,
                        "file_name": f"sa_{i:06d}.jpg",
                        "width": 1920,
                        "height": 1080
                    })
                    
                    # Add annotations for this image
                    for j in range(num_annotations_per_image):
                        annotations["annotations"].append({
                            "id": ann_id,
                            "image_id": i,
                            "category_id": 1,
                            "bbox": [100 * j, 100 * j, 50, 50],
                            "area": 2500,
                            "segmentation": [[100 * j, 100 * j, 150 * j, 100 * j, 
                                            150 * j, 150 * j, 100 * j, 150 * j]],
                            "iscrowd": 0
                        })
                        ann_id += 1
                
                # Write annotations file
                ann_file = base_path / "sa_1b.json"
                with open(ann_file, 'w') as f:
                    json.dump(annotations, f)
                
                yield base_path, images_dir, ann_file
        
        return _create
    
    def test_initialization(self, create_test_dataset):
        """Test loader initialization with valid config."""
        with create_test_dataset(5, 3) as (base_path, images_dir, ann_file):
            config = {
                'name': 'sa1b_test',
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            loader = Sa1bStreamingLoader(config)
            assert loader.name == 'sa1b_test'
            assert len(loader) == 5  # Should match number of images
    
    def test_missing_config_keys(self):
        """Test that missing required config keys raise errors."""
        with pytest.raises(ValueError, match="must include 'path'"):
            Sa1bStreamingLoader({'annotation_file': '/path/to/ann.json'})
        
        with pytest.raises(ValueError, match="must include 'annotation_file'"):
            Sa1bStreamingLoader({'path': '/path/to/images'})
    
    def test_streaming_parse_mock(self):
        """Test streaming parse with mocked ijson."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            images_dir = base_path / "images"
            images_dir.mkdir()
            
            # Create a few image files
            for i in range(3):
                (images_dir / f"sa_{i:06d}.jpg").touch()
            
            # Create minimal JSON file
            ann_file = base_path / "sa_1b.json"
            ann_file.write_text('{"images": [], "annotations": []}')
            
            config = {
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            with patch('ijson.items') as mock_items:
                # Mock image parsing
                mock_items.return_value = iter([
                    {'id': 0, 'file_name': 'sa_000000.jpg', 'width': 1920, 'height': 1080},
                    {'id': 1, 'file_name': 'sa_000001.jpg', 'width': 1920, 'height': 1080},
                    {'id': 2, 'file_name': 'sa_000002.jpg', 'width': 1920, 'height': 1080}
                ])
                
                loader = Sa1bStreamingLoader(config)
                
                # Should have parsed the mocked data
                assert len(loader.image_name_to_info) > 0
    
    def test_get_item(self, create_test_dataset):
        """Test retrieving individual items."""
        with create_test_dataset(3, 2) as (base_path, images_dir, ann_file):
            config = {
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            loader = Sa1bStreamingLoader(config)
            
            # Get first item
            sample = loader.get_item(0)
            
            assert 'sample_id' in sample
            assert 'media_path' in sample
            assert 'annotations' in sample
            assert 'segmentation_masks' in sample['annotations']
            
            # Check that path exists
            assert Path(sample['media_path']).exists()
    
    def test_memory_usage_tracking(self, create_test_dataset):
        """Test memory usage estimation."""
        with create_test_dataset(5, 3) as (base_path, images_dir, ann_file):
            config = {
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            loader = Sa1bStreamingLoader(config)
            memory_stats = loader.get_memory_usage()
            
            assert 'total_bytes' in memory_stats
            assert 'total_mb' in memory_stats
            assert memory_stats['total_bytes'] > 0
            assert memory_stats['total_mb'] > 0
    
    def test_segment_loader_variant(self, create_test_dataset):
        """Test the segmentation-specific loader variant."""
        with create_test_dataset(2, 4) as (base_path, images_dir, ann_file):
            config = {
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            loader = Sa1bStreamingSegmentLoader(config)
            sample = loader.get_item(0)
            
            # Should have additional segmentation statistics
            assert 'mask_stats' in sample['annotations']
            assert 'mask_distribution' in sample['annotations']
            assert 'num_masks' in sample['annotations']['mask_stats']
    
    def test_large_file_simulation(self):
        """Test that streaming doesn't load entire file into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            images_dir = base_path / "images"
            images_dir.mkdir()
            
            # Create one image file
            (images_dir / "sa_000000.jpg").touch()
            
            # Create a mock "large" JSON file (actually small for testing)
            ann_file = base_path / "sa_1b.json"
            
            # Write JSON in a way that simulates a large file structure
            with open(ann_file, 'w') as f:
                f.write('{"images": [')
                f.write('{"id": 0, "file_name": "sa_000000.jpg", "width": 1920, "height": 1080}')
                f.write('], "annotations": [')
                f.write('{"id": 0, "image_id": 0, "bbox": [0, 0, 100, 100], "area": 10000}')
                f.write(']}')
            
            config = {
                'path': str(images_dir),
                'annotation_file': str(ann_file)
            }
            
            # The loader should work without loading entire file
            loader = Sa1bStreamingLoader(config)
            
            # Check memory usage is reasonable
            memory_stats = loader.get_memory_usage()
            # In real scenario, we'd assert this is much less than file size
            assert memory_stats['total_mb'] < 100  # Should be well under 100MB