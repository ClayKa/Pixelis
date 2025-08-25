# tests/dataloaders/test_lvis_segment_loader.py

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from PIL import Image

from core.dataloaders.lvis_segment_loader import LvisSegmentLoader


class TestLvisSegmentLoader:
    """Test suite for LvisSegmentLoader."""

    @pytest.fixture
    def mock_lvis_config(self):
        """Create a mock LVIS configuration."""
        temp_dir = tempfile.mkdtemp()
        images_dir = Path(temp_dir) / "images"
        images_dir.mkdir()
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        image_path = images_dir / "test_image.jpg"
        test_image.save(image_path)
        
        # Create test annotation file
        lvis_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "width": 640,
                    "height": 480,
                    "license": 1,
                    "date_captured": "2023-01-01 00:00:00"
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "frequency": "f",
                    "synonyms": ["human", "individual"]
                },
                {
                    "id": 2,
                    "name": "bicycle",
                    "supercategory": "vehicle",
                    "frequency": "c",
                    "synonyms": ["bike"]
                },
                {
                    "id": 3,
                    "name": "rare_object",
                    "supercategory": "object",
                    "frequency": "r",
                    "synonyms": ["uncommon_item"]
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 200, 300],
                    "area": 60000,
                    "segmentation": [[100, 150, 300, 150, 300, 450, 100, 450]],
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [50, 75, 100, 150],
                    "area": 15000,
                    "segmentation": [[50, 75, 150, 75, 150, 225, 50, 225]],
                    "iscrowd": 0
                }
            ]
        }
        
        annotation_file = Path(temp_dir) / "lvis_v1_train.json"
        with open(annotation_file, 'w') as f:
            json.dump(lvis_data, f)
        
        return {
            'name': 'test_lvis_segment',
            'path': str(images_dir),
            'annotation_file': str(annotation_file),
            'min_area': 1000,
            'include_crowd': False
        }

    @pytest.fixture
    def loader(self, mock_lvis_config):
        """Create a LvisSegmentLoader instance for testing."""
        return LvisSegmentLoader(mock_lvis_config)

    def test_init_success(self, mock_lvis_config):
        """Test successful initialization of LvisSegmentLoader."""
        loader = LvisSegmentLoader(mock_lvis_config)
        
        assert loader.images_path == Path(mock_lvis_config['path'])
        assert loader.annotation_file == Path(mock_lvis_config['annotation_file'])
        assert loader.min_area == 1000
        assert len(loader._index) > 0
        assert len(loader._image_id_to_info) > 0
        assert len(loader._category_id_to_info) > 0

    def test_init_missing_required_config(self):
        """Test initialization failure with missing required config keys."""
        incomplete_config = {'path': '/some/path'}
        
        with pytest.raises(ValueError, match="LvisSegmentLoader config must include 'annotation_file'"):
            LvisSegmentLoader(incomplete_config)

    def test_init_invalid_path(self):
        """Test initialization failure with invalid paths."""
        config = {
            'name': 'test_lvis',
            'path': '/nonexistent/path',
            'annotation_file': '/nonexistent/annotation.json'
        }
        
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            LvisSegmentLoader(config)

    def test_build_index_structure(self, loader):
        """Test that _build_index creates proper lookup structures."""
        assert isinstance(loader._index, list)
        assert len(loader._index) > 0
        assert all(isinstance(img_id, int) for img_id in loader._index)
        
        # Test image lookup structure
        for img_id in loader._index:
            assert img_id in loader._image_id_to_info
            img_info = loader._image_id_to_info[img_id]
            assert 'file_name' in img_info
            assert 'width' in img_info
            assert 'height' in img_info
        
        # Test annotation lookup structure
        for img_id in loader._index:
            if img_id in loader._image_id_to_annotations:
                annotations = loader._image_id_to_annotations[img_id]
                assert isinstance(annotations, list)
                for ann in annotations:
                    assert 'category_id' in ann
                    assert 'bbox' in ann
                    assert 'area' in ann

    def test_get_item_basic_structure(self, loader):
        """Test basic structure of get_item output."""
        sample = loader.get_item(0)
        
        # Test base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'image'
        assert 'annotations' in sample
        
        # Test LVIS-specific annotations
        annotations = sample['annotations']
        assert 'lvis_instance_segmentation' in annotations
        assert 'num_instances' in annotations
        assert 'total_segmented_area' in annotations
        assert 'coverage_ratio' in annotations
        assert 'category_distribution' in annotations
        assert 'frequency_distribution' in annotations
        assert 'unique_categories' in annotations
        assert 'has_rare_categories' in annotations
        assert 'has_common_categories' in annotations
        assert 'image_metadata' in annotations
        assert 'dataset_info' in annotations

    def test_get_item_lvis_annotations(self, loader):
        """Test LVIS-specific annotation processing."""
        sample = loader.get_item(0)
        
        lvis_annotations = sample['annotations']['lvis_instance_segmentation']
        assert isinstance(lvis_annotations, list)
        assert len(lvis_annotations) > 0
        
        for ann in lvis_annotations:
            # Test required fields
            assert 'annotation_id' in ann
            assert 'category_id' in ann
            assert 'category_name' in ann
            assert 'synset' in ann
            assert 'synonyms' in ann
            assert 'def' in ann
            assert 'frequency' in ann
            assert 'area_pixels' in ann
            assert 'bbox' in ann
            assert 'segmentation_rle' in ann
            assert 'segmentation_type' in ann
            assert 'center_point' in ann
            assert 'geometric_properties' in ann
            
            # Test data types
            assert isinstance(ann['annotation_id'], int)
            assert isinstance(ann['category_id'], int)
            assert isinstance(ann['category_name'], str)
            assert isinstance(ann['area_pixels'], (int, float))
            assert isinstance(ann['bbox'], list) and len(ann['bbox']) == 4
            assert isinstance(ann['center_point'], list) and len(ann['center_point']) == 2
            assert isinstance(ann['synonyms'], list)
            assert ann['frequency'] in ['f', 'c', 'r', 'unknown']
            assert ann['segmentation_type'] == 'rle'

    def test_get_item_geometric_properties(self, loader):
        """Test geometric properties calculation."""
        sample = loader.get_item(0)
        
        lvis_annotations = sample['annotations']['lvis_instance_segmentation']
        for ann in lvis_annotations:
            geom_props = ann['geometric_properties']
            
            assert 'width' in geom_props
            assert 'height' in geom_props
            assert 'aspect_ratio' in geom_props
            assert 'relative_area' in geom_props
            
            assert isinstance(geom_props['width'], (int, float))
            assert isinstance(geom_props['height'], (int, float))
            assert isinstance(geom_props['aspect_ratio'], float)
            assert isinstance(geom_props['relative_area'], float)
            
            # Test aspect ratio calculation
            if geom_props['height'] > 0:
                expected_ratio = geom_props['width'] / geom_props['height']
                assert abs(geom_props['aspect_ratio'] - expected_ratio) < 1e-6

    def test_get_item_dataset_info(self, loader):
        """Test dataset_info field completeness."""
        sample = loader.get_item(0)
        
        dataset_info = sample['annotations']['dataset_info']
        assert dataset_info['task_type'] == 'lvis_instance_segmentation'
        assert dataset_info['source'] == 'LVIS-v1'
        assert dataset_info['suitable_for_segment_object_at'] == True
        assert dataset_info['suitable_for_get_properties'] == True
        assert dataset_info['has_category_names'] == True
        assert dataset_info['has_synonyms'] == True
        assert dataset_info['has_definitions'] == True
        assert dataset_info['has_frequency_info'] == True
        assert dataset_info['long_tail_vocabulary'] == True
        assert isinstance(dataset_info['num_categories'], int)
        assert dataset_info['num_categories'] > 0
        assert 'filtering_applied' in dataset_info

    def test_get_item_out_of_range(self, loader):
        """Test get_item with out-of-range index."""
        with pytest.raises(IndexError, match="Index .* out of range"):
            loader.get_item(len(loader))

    def test_calculate_center_point(self, loader):
        """Test center point calculation."""
        # Test normal bbox
        bbox = [10, 20, 30, 40]  # [x, y, width, height]
        center = loader._calculate_center_point(bbox)
        assert center == [25.0, 40.0]  # [x+w/2, y+h/2]
        
        # Test empty/invalid bbox
        empty_bbox = []
        center = loader._calculate_center_point(empty_bbox)
        assert center == [0.0, 0.0]
        
        # Test zero dimensions
        zero_bbox = [10, 20, 0, 40]
        center = loader._calculate_center_point(zero_bbox)
        assert center == [0.0, 0.0]

    def test_calculate_aspect_ratio(self, loader):
        """Test aspect ratio calculation."""
        # Test normal bbox
        bbox = [0, 0, 30, 20]  # width=30, height=20
        ratio = loader._calculate_aspect_ratio(bbox)
        assert ratio == 1.5
        
        # Test empty/invalid bbox
        empty_bbox = []
        ratio = loader._calculate_aspect_ratio(empty_bbox)
        assert ratio == 0.0
        
        # Test zero height
        zero_height_bbox = [0, 0, 30, 0]
        ratio = loader._calculate_aspect_ratio(zero_height_bbox)
        assert ratio == 0.0

    def test_get_samples_by_category(self, loader):
        """Test filtering samples by category."""
        # Test with existing category
        person_samples = loader.get_samples_by_category("person")
        assert isinstance(person_samples, list)
        
        for sample in person_samples:
            category_dist = sample['annotations']['category_distribution']
            assert "person" in category_dist
            assert category_dist["person"] > 0
        
        # Test with non-existing category
        nonexistent_samples = loader.get_samples_by_category("nonexistent_category")
        assert isinstance(nonexistent_samples, list)
        assert len(nonexistent_samples) == 0

    def test_get_frequency_statistics(self, loader):
        """Test frequency statistics generation."""
        stats = loader.get_frequency_statistics()
        
        assert 'instance_frequency_distribution' in stats
        assert 'category_frequency_distribution' in stats
        assert 'total_instances_analyzed' in stats
        assert 'samples_analyzed' in stats
        assert 'frequent_categories' in stats
        assert 'common_categories' in stats
        assert 'rare_categories' in stats
        assert 'num_rare_categories' in stats
        assert 'num_common_categories' in stats
        assert 'num_frequent_categories' in stats
        
        assert isinstance(stats['instance_frequency_distribution'], dict)
        assert isinstance(stats['category_frequency_distribution'], dict)
        assert isinstance(stats['total_instances_analyzed'], int)
        assert isinstance(stats['samples_analyzed'], int)
        assert isinstance(stats['frequent_categories'], list)
        assert isinstance(stats['common_categories'], list)
        assert isinstance(stats['rare_categories'], list)

    def test_get_samples_by_frequency(self, loader):
        """Test filtering samples by frequency class."""
        # Test frequent samples
        frequent_samples = loader.get_samples_by_frequency('f')
        assert isinstance(frequent_samples, list)
        
        for sample in frequent_samples:
            freq_dist = sample['annotations']['frequency_distribution']
            assert freq_dist.get('f', 0) > 0
        
        # Test rare samples
        rare_samples = loader.get_rare_category_samples()
        assert isinstance(rare_samples, list)
        
        # Test with non-existing frequency
        nonexistent_samples = loader.get_samples_by_frequency('x')
        assert isinstance(nonexistent_samples, list)
        assert len(nonexistent_samples) == 0

    def test_get_vocabulary_diversity_statistics(self, loader):
        """Test vocabulary diversity analysis."""
        vocab_stats = loader.get_vocabulary_diversity_statistics()
        
        assert 'total_categories' in vocab_stats
        assert 'categories_with_synonyms' in vocab_stats
        assert 'categories_with_definitions' in vocab_stats
        assert 'total_synonym_count' in vocab_stats
        assert 'avg_synonyms_per_category' in vocab_stats
        assert 'synonym_coverage_ratio' in vocab_stats
        assert 'definition_coverage_ratio' in vocab_stats
        
        # Test data types
        assert isinstance(vocab_stats['total_categories'], int)
        assert isinstance(vocab_stats['categories_with_synonyms'], int)
        assert isinstance(vocab_stats['categories_with_definitions'], int)
        assert isinstance(vocab_stats['total_synonym_count'], int)
        assert isinstance(vocab_stats['avg_synonyms_per_category'], float)
        assert isinstance(vocab_stats['synonym_coverage_ratio'], float)
        assert isinstance(vocab_stats['definition_coverage_ratio'], float)

    def test_get_samples_with_high_diversity(self, loader):
        """Test filtering samples with high category diversity."""
        diverse_samples = loader.get_samples_with_high_diversity(min_categories=2)
        assert isinstance(diverse_samples, list)
        
        for sample in diverse_samples:
            unique_categories = sample['annotations']['unique_categories']
            assert unique_categories >= 2
        
        # Test with high minimum (should return empty list or fewer samples)
        very_diverse_samples = loader.get_samples_with_high_diversity(min_categories=10)
        assert isinstance(very_diverse_samples, list)
        assert len(very_diverse_samples) <= len(diverse_samples)

    def test_get_geometric_analysis_statistics(self, loader):
        """Test geometric analysis statistics."""
        geom_stats = loader.get_geometric_analysis_statistics()
        
        assert 'samples_analyzed' in geom_stats
        assert 'total_instances_analyzed' in geom_stats
        assert 'area_statistics' in geom_stats
        assert 'aspect_ratio_statistics' in geom_stats
        assert 'relative_area_statistics' in geom_stats
        assert 'frequency_based_statistics' in geom_stats
        
        # Test area statistics
        area_stats = geom_stats['area_statistics']
        for key in ['min_pixels', 'max_pixels', 'avg_pixels', 'median_pixels']:
            assert key in area_stats
            assert isinstance(area_stats[key], (int, float))
        
        # Test aspect ratio statistics
        aspect_stats = geom_stats['aspect_ratio_statistics']
        for key in ['min', 'max', 'avg']:
            assert key in aspect_stats
            assert isinstance(aspect_stats[key], float)
        
        # Test frequency-based statistics
        freq_stats = geom_stats['frequency_based_statistics']
        assert isinstance(freq_stats, dict)

    def test_filtering_configuration(self, mock_lvis_config):
        """Test different filtering configurations."""
        # Test with different min_area
        config_high_area = mock_lvis_config.copy()
        config_high_area['min_area'] = 50000
        loader_high_area = LvisSegmentLoader(config_high_area)
        
        # Should have fewer samples due to higher area threshold
        sample = loader_high_area.get_item(0)
        for ann in sample['annotations']['lvis_instance_segmentation']:
            assert ann['area_pixels'] >= 50000
        
        # Test with different min_area
        assert loader_high_area.min_area == 50000

    @patch('core.dataloaders.lvis_segment_loader.logger')
    def test_logging_calls(self, mock_logger, mock_lvis_config):
        """Test that appropriate logging calls are made."""
        LvisSegmentLoader(mock_lvis_config)
        
        # Verify logging calls were made during initialization
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Should log loading annotations, image entries, categories, etc.
        assert any("Loading LVIS annotations" in call for call in log_calls)
        assert any("image entries" in call for call in log_calls)
        assert any("categories" in call for call in log_calls)

    def test_len_method(self, loader):
        """Test __len__ method."""
        length = len(loader)
        assert isinstance(length, int)
        assert length > 0
        assert length == len(loader._index)

    def test_synonym_processing(self, loader):
        """Test synonym processing in annotations."""
        sample = loader.get_item(0)
        
        lvis_annotations = sample['annotations']['lvis_instance_segmentation']
        for ann in lvis_annotations:
            assert 'synonyms' in ann
            assert isinstance(ann['synonyms'], list)
            # Each synonym should be a string
            for synonym in ann['synonyms']:
                assert isinstance(synonym, str)

    def test_edge_case_empty_directory(self, mock_lvis_config):
        """Test handling of empty image directory."""
        # Create empty images directory
        empty_config = mock_lvis_config.copy()
        empty_dir = Path(mock_lvis_config['path']).parent / "empty_images"
        empty_dir.mkdir(exist_ok=True)
        empty_config['path'] = str(empty_dir)
        
        # Should still initialize but with empty index
        loader = LvisSegmentLoader(empty_config)
        assert len(loader._index) == 0

    def test_malformed_annotation_handling(self, mock_lvis_config):
        """Test handling of malformed annotation file."""
        # Create malformed annotation file
        malformed_config = mock_lvis_config.copy()
        malformed_file = Path(mock_lvis_config['annotation_file']).parent / "malformed.json"
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": json}')  # Invalid JSON
        malformed_config['annotation_file'] = str(malformed_file)
        
        with pytest.raises(json.JSONDecodeError):
            LvisSegmentLoader(malformed_config)