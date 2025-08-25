# tests/dataloaders/test_coco_segment_loader.py

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.dataloaders.coco_segment_loader import CocoSegmentLoader


class TestCocoSegmentLoader:
    """Test suite for CocoSegmentLoader class."""

    @pytest.fixture
    def temp_coco_dataset(self):
        """Create a temporary COCO dataset structure for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        
        # Create sample images (placeholder files)
        sample_images = [
            "000000000001.jpg",
            "000000000002.jpg", 
            "000000000003.jpg",
            "000000000004.jpg"
        ]
        
        for img_name in sample_images:
            img_path = images_dir / img_name
            img_path.write_text(f"fake image data for {img_name}")
        
        # Create COCO annotation structure
        coco_data = {
            "info": {
                "description": "COCO 2017 Dataset",
                "version": "1.0",
                "year": 2017,
                "contributor": "COCO Consortium"
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "000000000001.jpg",
                    "width": 640,
                    "height": 480,
                    "license": 1,
                    "coco_url": "http://images.cocodataset.org/train2017/000000000001.jpg",
                    "date_captured": "2013-11-14 17:02:52"
                },
                {
                    "id": 2,
                    "file_name": "000000000002.jpg",
                    "width": 512,
                    "height": 384,
                    "license": 2
                },
                {
                    "id": 3,
                    "file_name": "000000000003.jpg",
                    "width": 800,
                    "height": 600,
                    "license": 3
                },
                {
                    "id": 4,
                    "file_name": "000000000004.jpg",
                    "width": 1024,
                    "height": 768,
                    "license": 4
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person"
                },
                {
                    "id": 2,
                    "name": "bicycle",
                    "supercategory": "vehicle"
                },
                {
                    "id": 3,
                    "name": "car",
                    "supercategory": "vehicle"
                },
                {
                    "id": 16,
                    "name": "bird",
                    "supercategory": "animal"
                }
            ],
            "annotations": [
                # Image 1 annotations
                {
                    "id": 101,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[100, 200, 200, 200, 200, 300, 100, 300]],
                    "area": 10000,
                    "bbox": [100, 200, 100, 100],
                    "iscrowd": 0
                },
                {
                    "id": 102,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [[300, 100, 400, 100, 400, 150, 300, 150]],
                    "area": 5000,
                    "bbox": [300, 100, 100, 50],
                    "iscrowd": 0
                },
                # Image 2 annotations  
                {
                    "id": 103,
                    "image_id": 2,
                    "category_id": 3,
                    "segmentation": {"size": [384, 512], "counts": "fake_rle_data"},
                    "area": 15000,
                    "bbox": [50, 50, 200, 150],
                    "iscrowd": 0
                },
                # Image 3 - crowd annotation (should be filtered by default)
                {
                    "id": 104,
                    "image_id": 3,
                    "category_id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
                    "area": 1600,
                    "bbox": [10, 10, 40, 40],
                    "iscrowd": 1
                },
                # Image 3 - normal annotation
                {
                    "id": 105,
                    "image_id": 3,
                    "category_id": 16,
                    "segmentation": [[200, 300, 250, 300, 250, 350, 200, 350]],
                    "area": 2500,
                    "bbox": [200, 300, 50, 50],
                    "iscrowd": 0
                },
                # Small annotation (should be filtered with min_area)
                {
                    "id": 106,
                    "image_id": 4,
                    "category_id": 1,
                    "segmentation": [[5, 5, 10, 5, 10, 10, 5, 10]],
                    "area": 25,
                    "bbox": [5, 5, 5, 5],
                    "iscrowd": 0
                }
            ]
        }
        
        # Save annotation file
        annotation_file = temp_dir / "instances_train2017.json"
        with open(annotation_file, 'w') as f:
            json.dump(coco_data, f)
        
        yield {
            'temp_dir': temp_dir,
            'images_dir': images_dir,
            'annotation_file': annotation_file,
            'sample_images': sample_images,
            'coco_data': coco_data
        }
        
        # Cleanup is handled by tempfile

    def test_init_missing_required_config_keys(self):
        """Test initialization with missing required configuration keys."""
        # Test missing 'path'
        config = {'annotation_file': '/some/path.json'}
        with pytest.raises(ValueError, match="CocoSegmentLoader config must include 'path'"):
            CocoSegmentLoader(config)
        
        # Test missing 'annotation_file'
        config = {'path': '/some/path'}
        with pytest.raises(ValueError, match="CocoSegmentLoader config must include 'annotation_file'"):
            CocoSegmentLoader(config)

    def test_init_nonexistent_paths(self):
        """Test initialization with non-existent paths."""
        config = {
            'name': 'test_coco',
            'path': '/nonexistent/images',
            'annotation_file': '/nonexistent/annotations.json'
        }
        with pytest.raises(FileNotFoundError):
            CocoSegmentLoader(config)

    def test_build_index_success(self, temp_coco_dataset):
        """Test successful index building."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Should include all images that have annotations and exist on disk
        # All images have annotations that meet the minimum area threshold (default 0)
        assert len(loader._index) == 4
        assert 1 in loader._index
        assert 2 in loader._index  
        assert 3 in loader._index
        assert 4 in loader._index
        
        # Check lookups are populated
        assert len(loader._image_id_to_info) == 4  # All images from JSON
        assert len(loader._category_id_to_info) == 4  # All categories
        assert len(loader._image_id_to_annotations) == 4  # Images with annotations

    def test_build_index_with_filtering(self, temp_coco_dataset):
        """Test index building with filtering options."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file']),
            'min_area': 1000,  # Filter small annotations
            'include_crowd': True  # Include crowd annotations
        }
        
        loader = CocoSegmentLoader(config)
        
        # Should include more images due to including crowd annotations
        # and fewer annotations due to area filtering
        assert len(loader._index) == 3
        
        # Check filtering parameters
        assert loader.min_area == 1000
        assert loader.include_crowd is True

    def test_get_item_success(self, temp_coco_dataset):
        """Test successful item retrieval."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        sample = loader.get_item(0)
        
        # Verify base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'image'
        assert 'annotations' in sample
        
        # Verify COCO-specific annotations
        annotations = sample['annotations']
        assert 'coco_instance_segmentation' in annotations
        assert 'num_instances' in annotations
        assert 'total_segmented_area' in annotations
        assert 'coverage_ratio' in annotations
        assert 'category_distribution' in annotations
        assert 'unique_categories' in annotations
        assert 'image_metadata' in annotations
        assert 'dataset_info' in annotations
        
        # Check instance segmentation details
        instances = annotations['coco_instance_segmentation']
        assert len(instances) >= 1
        
        for instance in instances:
            assert 'annotation_id' in instance
            assert 'category_id' in instance
            assert 'category_name' in instance
            assert 'supercategory' in instance
            assert 'area_pixels' in instance
            assert 'bbox' in instance
            assert 'segmentation' in instance
            assert 'segmentation_type' in instance
            assert 'center_point' in instance
            assert 'geometric_properties' in instance
            
            # Verify center point calculation
            center = instance['center_point']
            assert len(center) == 2
            assert all(isinstance(coord, (int, float)) for coord in center)
            
            # Verify geometric properties
            geom = instance['geometric_properties']
            assert 'aspect_ratio' in geom
            assert 'relative_area' in geom
        
        # Verify dataset info
        dataset_info = annotations['dataset_info']
        assert dataset_info['task_type'] == 'coco_instance_segmentation'
        assert dataset_info['source'] == 'COCO2017'
        assert dataset_info['suitable_for_segment_object_at'] is True
        assert dataset_info['suitable_for_get_properties'] is True
        assert dataset_info['has_category_names'] is True

    def test_get_item_index_out_of_range(self, temp_coco_dataset):
        """Test get_item with invalid index."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        with pytest.raises(IndexError, match="Index 10 out of range"):
            loader.get_item(10)

    def test_calculate_center_point(self, temp_coco_dataset):
        """Test center point calculation."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Test normal bbox [x, y, width, height]
        bbox = [100, 200, 100, 100]  # From test data
        center = loader._calculate_center_point(bbox)
        assert center == [150.0, 250.0]  # [100 + 100/2, 200 + 100/2]
        
        # Test invalid bbox
        bbox = [10, 20, 0, 50]  # Zero width
        center = loader._calculate_center_point(bbox)
        assert center == [0.0, 0.0]

    def test_calculate_aspect_ratio(self, temp_coco_dataset):
        """Test aspect ratio calculation."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Test normal bbox
        bbox = [100, 200, 200, 100]  # width=200, height=100
        ratio = loader._calculate_aspect_ratio(bbox)
        assert ratio == 2.0  # 200/100
        
        # Test square bbox
        bbox = [0, 0, 100, 100]
        ratio = loader._calculate_aspect_ratio(bbox)
        assert ratio == 1.0
        
        # Test invalid bbox
        bbox = [10, 20, 50, 0]  # Zero height
        ratio = loader._calculate_aspect_ratio(bbox)
        assert ratio == 0.0

    def test_get_samples_by_category(self, temp_coco_dataset):
        """Test filtering samples by category."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Get samples with 'person' category
        person_samples = loader.get_samples_by_category('person')
        assert len(person_samples) >= 1
        
        for sample in person_samples:
            category_dist = sample['annotations']['category_distribution']
            assert 'person' in category_dist
        
        # Test non-existent category
        empty_samples = loader.get_samples_by_category('nonexistent')
        assert len(empty_samples) == 0

    def test_get_category_statistics(self, temp_coco_dataset):
        """Test category statistics calculation."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        stats = loader.get_category_statistics()
        
        # Verify statistics structure
        assert 'total_categories' in stats
        assert 'total_instances_estimated' in stats
        assert 'samples_analyzed' in stats
        assert 'category_distribution' in stats
        assert 'most_common_categories' in stats
        assert 'available_categories' in stats
        
        # Check values
        assert stats['total_categories'] == 4  # From test data
        assert stats['total_instances_estimated'] > 0

    def test_get_samples_by_complexity(self, temp_coco_dataset):
        """Test filtering samples by complexity (number of instances)."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Get samples with multiple instances
        complex_samples = loader.get_samples_by_complexity(min_instances=2)
        
        for sample in complex_samples:
            num_instances = sample['annotations']['num_instances']
            assert num_instances >= 2
        
        # Get samples with specific range
        range_samples = loader.get_samples_by_complexity(min_instances=1, max_instances=2)
        
        for sample in range_samples:
            num_instances = sample['annotations']['num_instances']
            assert 1 <= num_instances <= 2

    def test_get_supercategory_distribution(self, temp_coco_dataset):
        """Test supercategory distribution analysis."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        stats = loader.get_supercategory_distribution()
        
        # Verify structure
        assert 'supercategory_distribution' in stats
        assert 'num_supercategories' in stats
        assert 'samples_analyzed' in stats
        
        # Should include supercategories from test data
        supercats = stats['supercategory_distribution']
        expected_supercats = {'person', 'vehicle', 'animal'}
        assert any(supercat in supercats for supercat in expected_supercats)

    def test_get_geometric_analysis_statistics(self, temp_coco_dataset):
        """Test geometric analysis statistics."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        stats = loader.get_geometric_analysis_statistics()
        
        # Verify structure
        assert 'samples_analyzed' in stats
        assert 'total_instances_analyzed' in stats
        assert 'area_statistics' in stats
        assert 'aspect_ratio_statistics' in stats
        assert 'relative_area_statistics' in stats
        assert 'bbox_size_statistics' in stats
        
        # Check area statistics
        area_stats = stats['area_statistics']
        assert 'min_pixels' in area_stats
        assert 'max_pixels' in area_stats
        assert 'avg_pixels' in area_stats

    def test_segmentation_type_detection(self, temp_coco_dataset):
        """Test detection of different segmentation formats."""
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Check that we can handle both polygon and RLE formats
        found_polygon = False
        found_rle = False
        
        for i in range(len(loader)):
            sample = loader.get_item(i)
            instances = sample['annotations']['coco_instance_segmentation']
            
            for instance in instances:
                seg_type = instance['segmentation_type']
                if seg_type == 'polygon':
                    found_polygon = True
                elif seg_type == 'rle':
                    found_rle = True
        
        # Our test data includes both formats
        assert found_polygon
        assert found_rle

    def test_crowd_annotation_handling(self, temp_coco_dataset):
        """Test handling of crowd annotations."""
        # Test with crowd annotations excluded (default)
        config_no_crowd = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file']),
            'include_crowd': False
        }
        
        loader_no_crowd = CocoSegmentLoader(config_no_crowd)
        
        # Test with crowd annotations included
        config_with_crowd = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file']),
            'include_crowd': True
        }
        
        loader_with_crowd = CocoSegmentLoader(config_with_crowd)
        
        # Check that crowd annotations are properly handled
        for i in range(len(loader_with_crowd)):
            sample = loader_with_crowd.get_item(i)
            instances = sample['annotations']['coco_instance_segmentation']
            
            for instance in instances:
                # Should have is_crowd field
                assert 'is_crowd' in instance
                assert isinstance(instance['is_crowd'], bool)

    def test_empty_annotations(self, temp_coco_dataset):
        """Test handling of images with no valid annotations."""
        # Create config with high minimum area to filter out most annotations
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file']),
            'min_area': 50000  # Very high threshold
        }
        
        loader = CocoSegmentLoader(config)
        
        # Should have fewer images due to strict filtering
        assert len(loader) <= 3

    def test_malformed_annotation_file(self, temp_coco_dataset):
        """Test handling of malformed annotation file."""
        # Create invalid JSON file
        invalid_file = temp_coco_dataset['temp_dir'] / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write('{"invalid": json content')
        
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(invalid_file)
        }
        
        with pytest.raises(json.JSONDecodeError):
            CocoSegmentLoader(config)

    def test_missing_images_on_disk(self, temp_coco_dataset):
        """Test behavior when annotation references missing image files."""
        # Remove one image file
        missing_image = temp_coco_dataset['images_dir'] / "000000000001.jpg"
        missing_image.unlink()
        
        config = {
            'name': 'test_coco',
            'path': str(temp_coco_dataset['images_dir']),
            'annotation_file': str(temp_coco_dataset['annotation_file'])
        }
        
        loader = CocoSegmentLoader(config)
        
        # Should only include images that actually exist
        assert len(loader) == 3  # One less due to missing image