# tests/dataloaders/test_sa1b_segment_loader.py

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from core.dataloaders.sa1b_segment_loader import Sa1bSegmentLoader


class TestSa1bSegmentLoader:
    """Test suite for Sa1bSegmentLoader class."""

    @pytest.fixture
    def temp_dataset_structure(self):
        """Create a temporary dataset structure for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        images_dir = temp_dir / "images"
        annotations_dir = temp_dir / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create sample images (placeholder files)
        sample_images = [
            "sa_3624991.jpg",
            "sa_3624992.jpg", 
            "sa_3624993.jpg",
            "sa_3624994.jpg"
        ]
        
        for img_name in sample_images:
            img_path = images_dir / img_name
            # Create a simple placeholder file
            img_path.write_text(f"fake image data for {img_name}")
        
        # Create corresponding annotation files with segmentation data
        annotations = [
            # High-quality masks suitable for segmentation
            {
                "image": {"image_id": 3624991, "width": 1500, "height": 1500, "file_name": "sa_3624991.jpg"},
                "annotations": [
                    {
                        "bbox": [277.0, 1150.0, 229.0, 263.0], 
                        "area": 41898, 
                        "segmentation": {"size": [1500, 1500], "counts": "fake_rle_data_1"},
                        "predicted_iou": 0.95, 
                        "point_coords": [[460.6875, 1322.8125]], 
                        "crop_box": [0.0, 933.0, 567.0, 567.0], 
                        "id": 456617540, 
                        "stability_score": 0.99
                    },
                    {
                        "bbox": [553.0, 785.0, 51.0, 27.0], 
                        "area": 1029, 
                        "segmentation": {"size": [1500, 1500], "counts": "fake_rle_data_2"},
                        "predicted_iou": 0.90, 
                        "point_coords": [[559.0625, 799.1875]], 
                        "crop_box": [311.0, 622.0, 567.0, 567.0], 
                        "id": 456617541, 
                        "stability_score": 0.97
                    }
                ]
            },
            # Medium quality masks
            {
                "image": {"image_id": 3624992, "width": 1500, "height": 1500, "file_name": "sa_3624992.jpg"},
                "annotations": [
                    {
                        "bbox": [100.0, 200.0, 300.0, 400.0], 
                        "area": 80000, 
                        "segmentation": {"size": [1500, 1500], "counts": "fake_rle_data_3"},
                        "predicted_iou": 0.75, 
                        "point_coords": [[250.0, 400.0]], 
                        "crop_box": [0.0, 0.0, 567.0, 567.0], 
                        "id": 456617542, 
                        "stability_score": 0.80
                    }
                ]
            },
            # Low quality masks (should be filtered out by default)
            {
                "image": {"image_id": 3624993, "width": 1500, "height": 1500, "file_name": "sa_3624993.jpg"},
                "annotations": [
                    {
                        "bbox": [10.0, 20.0, 5.0, 5.0], 
                        "area": 25,  # Too small
                        "segmentation": {"size": [1500, 1500], "counts": "fake_rle_data_4"},
                        "predicted_iou": 0.30,  # Too low
                        "point_coords": [[12.5, 22.5]], 
                        "crop_box": [0.0, 0.0, 100.0, 100.0], 
                        "id": 456617543, 
                        "stability_score": 0.20  # Too low
                    }
                ]
            },
            # No usable masks after filtering
            {
                "image": {"image_id": 3624994, "width": 1500, "height": 1500, "file_name": "sa_3624994.jpg"},
                "annotations": []
            }
        ]
        
        for i, ann_data in enumerate(annotations):
            ann_name = f"sa_{ann_data['image']['image_id']}.json"
            ann_path = annotations_dir / ann_name
            with open(ann_path, 'w') as f:
                json.dump(ann_data, f)
        
        yield {
            'temp_dir': temp_dir,
            'images_dir': images_dir,
            'annotations_dir': annotations_dir,
            'sample_images': sample_images
        }
        
        # Cleanup is handled by tempfile

    def test_init_missing_required_config_keys(self):
        """Test initialization with missing required configuration keys."""
        # Test missing 'path'
        config = {'annotations_path': '/some/path'}
        with pytest.raises(ValueError, match="Sa1bSegmentLoader config must include 'path'"):
            Sa1bSegmentLoader(config)
        
        # Test missing 'annotations_path'
        config = {'path': '/some/path'}
        with pytest.raises(ValueError, match="Sa1bSegmentLoader config must include 'annotations_path'"):
            Sa1bSegmentLoader(config)

    def test_init_nonexistent_paths(self):
        """Test initialization with non-existent paths."""
        config = {
            'name': 'test_sa1b_segment',
            'path': '/nonexistent/images',
            'annotations_path': '/nonexistent/annotations'
        }
        with pytest.raises(FileNotFoundError):
            Sa1bSegmentLoader(config)

    def test_build_index_success_with_filtering(self, temp_dataset_structure):
        """Test successful index building with quality filtering."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100,
            'min_stability_score': 0.5,
            'min_predicted_iou': 0.5
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Should only include samples with usable masks after filtering
        # sa_3624991: 2 good masks, sa_3624992: 1 medium mask, sa_3624993: filtered out, sa_3624994: no masks
        assert len(loader._index) == 2  # Only first two images have usable masks
        
        # Check that filtering parameters are stored
        assert loader.min_pixel_area == 100
        assert loader.min_stability_score == 0.5
        assert loader.min_predicted_iou == 0.5
        
        # Verify index structure
        for entry in loader._index:
            assert 'image_id' in entry
            assert 'image_path' in entry
            assert 'annotation_path' in entry
            assert 'annotations' in entry
            assert 'num_usable_annotations' in entry
            assert entry['num_usable_annotations'] > 0  # Should only have usable masks

    def test_build_index_strict_filtering(self, temp_dataset_structure):
        """Test index building with strict quality filtering."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 1000,  # Higher threshold
            'min_stability_score': 0.95,  # Very high threshold
            'min_predicted_iou': 0.95  # Very high threshold
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # With strict filtering, only the highest quality masks should remain
        # Only sa_3624991 has masks meeting all criteria
        assert len(loader._index) == 1

    def test_get_item_segmentation_focused(self, temp_dataset_structure):
        """Test get_item returns segmentation-focused annotations."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100,
            'min_stability_score': 0.5,
            'min_predicted_iou': 0.5
        }
        
        loader = Sa1bSegmentLoader(config)
        sample = loader.get_item(0)
        
        # Verify base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'image'
        assert 'annotations' in sample
        
        # Verify segmentation-focused annotations
        annotations = sample['annotations']
        assert 'instance_segmentation' in annotations
        assert 'num_instances' in annotations
        assert 'total_segmented_area' in annotations
        assert 'coverage_ratio' in annotations
        assert 'quality_statistics' in annotations
        assert 'image_metadata' in annotations
        assert 'dataset_info' in annotations
        
        # Check instance segmentation details
        instances = annotations['instance_segmentation']
        assert len(instances) > 0
        
        for instance in instances:
            assert 'instance_id' in instance
            assert 'area_pixels' in instance
            assert 'center_point' in instance
            assert 'quality_metrics' in instance
            assert 'geometric_properties' in instance
            assert 'segmentation_mask_rle' in instance
            
            # Verify center point calculation
            center = instance['center_point']
            assert len(center) == 2
            assert all(isinstance(coord, (int, float)) for coord in center)
            
            # Verify quality metrics
            quality = instance['quality_metrics']
            assert 'stability_score' in quality
            assert 'predicted_iou' in quality
            
            # Verify geometric properties
            geom = instance['geometric_properties']
            assert 'aspect_ratio' in geom
            assert 'relative_area' in geom
        
        # Verify dataset info for segmentation
        dataset_info = annotations['dataset_info']
        assert dataset_info['task_type'] == 'instance_segmentation_optimized'
        assert dataset_info['suitable_for_segment_object_at'] is True
        assert dataset_info['suitable_for_get_properties'] is True
        assert dataset_info['has_center_points'] is True
        assert dataset_info['has_quality_metrics'] is True

    def test_get_item_index_out_of_range(self, temp_dataset_structure):
        """Test get_item with invalid index."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        
        with pytest.raises(IndexError, match="Index 10 out of range"):
            loader.get_item(10)

    def test_calculate_aspect_ratio(self, temp_dataset_structure):
        """Test aspect ratio calculation."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Test normal bbox [x, y, width, height]
        bbox = [10.0, 20.0, 100.0, 50.0]
        aspect_ratio = loader._calculate_aspect_ratio(bbox)
        assert aspect_ratio == 2.0  # 100/50
        
        # Test square bbox
        bbox = [0.0, 0.0, 100.0, 100.0]
        aspect_ratio = loader._calculate_aspect_ratio(bbox)
        assert aspect_ratio == 1.0
        
        # Test invalid bbox
        bbox = [10.0, 20.0, 0.0, 50.0]  # Zero width
        aspect_ratio = loader._calculate_aspect_ratio(bbox)
        assert aspect_ratio == 0.0
        
        # Test empty bbox
        bbox = []
        aspect_ratio = loader._calculate_aspect_ratio(bbox)
        assert aspect_ratio == 0.0

    def test_get_high_quality_instances(self, temp_dataset_structure):
        """Test filtering for high-quality instances."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100,
            'min_stability_score': 0.5,
            'min_predicted_iou': 0.5
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Get high-quality instances with strict thresholds
        high_quality = loader.get_high_quality_instances(
            min_stability_score=0.95,
            min_predicted_iou=0.90
        )
        
        # Should return samples with only the highest quality masks
        assert len(high_quality) > 0
        
        for sample in high_quality:
            instances = sample['annotations']['instance_segmentation']
            for instance in instances:
                quality = instance['quality_metrics']
                assert quality['stability_score'] >= 0.95
                assert quality['predicted_iou'] >= 0.90

    def test_get_samples_by_area_range(self, temp_dataset_structure):
        """Test filtering samples by area range."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Get samples with large instances only
        large_samples = loader.get_samples_by_area_range(min_area=10000)
        
        # Should include samples with large masks
        assert len(large_samples) > 0
        
        for sample in large_samples:
            instances = sample['annotations']['instance_segmentation']
            for instance in instances:
                assert instance['area_pixels'] >= 10000
        
        # Get samples in specific range
        medium_samples = loader.get_samples_by_area_range(min_area=1000, max_area=50000)
        
        for sample in medium_samples:
            instances = sample['annotations']['instance_segmentation']
            for instance in instances:
                assert 1000 <= instance['area_pixels'] <= 50000

    def test_get_geometric_analysis_statistics(self, temp_dataset_structure):
        """Test geometric analysis statistics."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100
        }
        
        loader = Sa1bSegmentLoader(config)
        stats = loader.get_geometric_analysis_statistics()
        
        # Verify statistics structure
        assert 'total_samples_analyzed' in stats
        assert 'total_instances' in stats
        assert 'area_statistics' in stats
        assert 'aspect_ratio_statistics' in stats
        assert 'relative_area_statistics' in stats
        assert 'quality_statistics' in stats
        assert 'coverage_statistics' in stats
        assert 'filtering_impact' in stats
        
        # Check area statistics
        area_stats = stats['area_statistics']
        assert 'min_pixels' in area_stats
        assert 'max_pixels' in area_stats
        assert 'avg_pixels' in area_stats
        
        # Check aspect ratio statistics
        aspect_stats = stats['aspect_ratio_statistics']
        assert 'min' in aspect_stats
        assert 'max' in aspect_stats
        assert 'avg' in aspect_stats
        
        # Check filtering impact
        filtering = stats['filtering_impact']
        assert filtering['min_pixel_area_applied'] == 100

    def test_get_samples_suitable_for_geometric_comparison(self, temp_dataset_structure):
        """Test filtering samples suitable for geometric comparison."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 100
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Get samples suitable for geometric comparison
        comparison_samples = loader.get_samples_suitable_for_geometric_comparison(
            min_instances=2, max_instances=5
        )
        
        # Should only include samples with multiple instances
        for sample in comparison_samples:
            num_instances = sample['annotations']['num_instances']
            assert 2 <= num_instances <= 5

    def test_empty_directories(self, temp_dataset_structure):
        """Test loader behavior with empty directories."""
        # Create empty directories
        empty_images = temp_dataset_structure['temp_dir'] / "empty_images"
        empty_annotations = temp_dataset_structure['temp_dir'] / "empty_annotations"
        empty_images.mkdir()
        empty_annotations.mkdir()
        
        config = {
            'name': 'test_sa1b_segment',
            'path': str(empty_images),
            'annotations_path': str(empty_annotations)
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Should create loader with empty index
        assert len(loader._index) == 0
        assert len(loader) == 0
        
        # Statistics should handle empty dataset
        stats = loader.get_geometric_analysis_statistics()
        assert stats['error'] == 'No samples available'

    def test_malformed_annotation_file(self, temp_dataset_structure):
        """Test handling of corrupted annotation files."""
        # Corrupt one annotation file
        corrupted_path = temp_dataset_structure['annotations_dir'] / "sa_3624991.json"
        with open(corrupted_path, 'w') as f:
            f.write('{"invalid": json content')
        
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Should handle corrupted files gracefully
        # Only samples with valid annotations should be included
        assert len(loader._index) == 1  # Only sa_3624992 remains valid

    def test_default_filtering_parameters(self, temp_dataset_structure):
        """Test default filtering parameters."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Check default values
        assert loader.min_pixel_area == 100
        assert loader.min_stability_score == 0.5
        assert loader.min_predicted_iou == 0.5

    def test_custom_filtering_parameters(self, temp_dataset_structure):
        """Test custom filtering parameters."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir']),
            'min_pixel_area': 500,
            'min_stability_score': 0.8,
            'min_predicted_iou': 0.75
        }
        
        loader = Sa1bSegmentLoader(config)
        
        # Check custom values
        assert loader.min_pixel_area == 500
        assert loader.min_stability_score == 0.8
        assert loader.min_predicted_iou == 0.75

    def test_center_point_calculation(self, temp_dataset_structure):
        """Test center point pre-calculation for SEGMENT_OBJECT_AT."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        sample = loader.get_item(0)
        
        instances = sample['annotations']['instance_segmentation']
        for instance in instances:
            center_point = instance['center_point']
            bbox = instance['bbox']
            
            # Verify center point is correctly calculated from bbox
            if len(bbox) >= 4:
                expected_x = bbox[0] + bbox[2] / 2
                expected_y = bbox[1] + bbox[3] / 2
                assert abs(center_point[0] - expected_x) < 0.001
                assert abs(center_point[1] - expected_y) < 0.001

    def test_quality_statistics_calculation(self, temp_dataset_structure):
        """Test quality statistics calculation."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        sample = loader.get_item(0)
        
        quality_stats = sample['annotations']['quality_statistics']
        
        # Check statistics are properly calculated
        assert 'avg_stability_score' in quality_stats
        assert 'min_stability_score' in quality_stats
        assert 'max_stability_score' in quality_stats
        assert 'usable_mask_ratio' in quality_stats
        
        # Values should be reasonable
        assert 0.0 <= quality_stats['avg_stability_score'] <= 1.0
        assert 0.0 <= quality_stats['usable_mask_ratio'] <= 1.0

    def test_coverage_ratio_calculation(self, temp_dataset_structure):
        """Test coverage ratio calculation."""
        config = {
            'name': 'test_sa1b_segment',
            'path': str(temp_dataset_structure['images_dir']),
            'annotations_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = Sa1bSegmentLoader(config)
        sample = loader.get_item(0)
        
        annotations = sample['annotations']
        coverage_ratio = annotations['coverage_ratio']
        total_area = annotations['total_segmented_area']
        
        # Coverage ratio should be between 0 and 1
        assert 0.0 <= coverage_ratio <= 1.0
        
        # Should be consistent with image dimensions
        image_area = 1500 * 1500  # From test data
        expected_ratio = total_area / image_area
        assert abs(coverage_ratio - expected_ratio) < 0.001