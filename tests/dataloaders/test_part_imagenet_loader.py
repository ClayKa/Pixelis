# tests/dataloaders/test_part_imagenet_loader.py

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from core.dataloaders.part_imagenet_loader import PartImageNetLoader


class TestPartImageNetLoader:
    """Test suite for PartImageNetLoader class."""

    @pytest.fixture
    def temp_dataset_structure(self):
        """Create a temporary dataset structure for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        images_dir = temp_dir / "images"
        annotations_dir = temp_dir / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create sample images (placeholder JPEG files)
        sample_images = [
            "n01440764_10029.JPEG",
            "n01440764_10043.JPEG", 
            "n01443537_1062.JPEG",
            "n01484850_5097.JPEG"
        ]
        
        for img_name in sample_images:
            img_path = images_dir / img_name
            # Create a simple 10x10 RGB image
            img = Image.new('RGB', (10, 10), color='red')
            img.save(img_path, 'JPEG')
        
        # Create corresponding annotation masks
        for img_name in sample_images:
            mask_name = img_name.replace('.JPEG', '.png')
            mask_path = annotations_dir / mask_name
            
            # Create binary mask: background=158, object=class-specific value
            mask_array = np.full((10, 10), 158, dtype=np.uint8)  # Background
            
            # Object pixels (3x3 square in center)
            if 'n01440764' in mask_name:
                object_value = 82  # Fish class
            elif 'n01443537' in mask_name:
                object_value = 100  # Another class
            elif 'n01484850' in mask_name:
                object_value = 124  # Another class
            else:
                object_value = 50
                
            mask_array[3:6, 3:6] = object_value  # 3x3 object in center
            
            mask_img = Image.fromarray(mask_array)
            mask_img.save(mask_path)
        
        # Create metadata file
        metadata = {
            "n01440764": "fish",
            "n01443537": "salamander", 
            "n01484850": "newt"
        }
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        yield {
            'temp_dir': temp_dir,
            'images_dir': images_dir,
            'annotations_dir': annotations_dir,
            'metadata_path': metadata_path,
            'sample_images': sample_images
        }
        
        # Cleanup is handled by tempfile

    def test_init_missing_required_config_keys(self):
        """Test initialization with missing required configuration keys."""
        # Test missing 'path'
        config = {'annotation_path': '/some/path'}
        with pytest.raises(ValueError, match="PartImageNetLoader config must include 'path'"):
            PartImageNetLoader(config)
        
        # Test missing 'annotation_path'
        config = {'path': '/some/path'}
        with pytest.raises(ValueError, match="PartImageNetLoader config must include 'annotation_path'"):
            PartImageNetLoader(config)

    def test_init_nonexistent_paths(self):
        """Test initialization with non-existent paths."""
        config = {
            'path': '/nonexistent/images',
            'annotation_path': '/nonexistent/annotations'
        }
        with pytest.raises(FileNotFoundError):
            PartImageNetLoader(config)

    def test_build_index_success(self, temp_dataset_structure):
        """Test successful index building with matching images and annotations."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Check that index contains correct pairs
        assert len(loader._index) == 4  # All 4 sample images have matching annotations
        
        # Verify structure of index entries
        for image_path, annotation_path in loader._index:
            assert isinstance(image_path, Path)
            assert isinstance(annotation_path, Path)
            assert image_path.exists()
            assert annotation_path.exists()
            assert image_path.suffix == '.JPEG'
            assert annotation_path.suffix == '.png'
            assert image_path.stem == annotation_path.stem

    def test_build_index_missing_annotations(self, temp_dataset_structure):
        """Test index building when some images don't have corresponding annotations."""
        # Remove one annotation file
        missing_annotation = temp_dataset_structure['annotations_dir'] / "n01440764_10043.png"
        missing_annotation.unlink()
        
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Should only match 3 out of 4 images
        assert len(loader._index) == 3
        
        # Verify the missing image is not in the index
        indexed_stems = {path[0].stem for path in loader._index}
        assert 'n01440764_10043' not in indexed_stems

    def test_metadata_loading(self, temp_dataset_structure):
        """Test loading of optional metadata file."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir']),
            'metadata_file': str(temp_dataset_structure['metadata_path'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Check metadata was loaded
        assert loader.part_id_to_label['n01440764'] == 'fish'
        assert loader.part_id_to_label['n01443537'] == 'salamander'
        assert loader.part_id_to_label['n01484850'] == 'newt'

    def test_get_item_success(self, temp_dataset_structure):
        """Test successful item retrieval with mask parsing."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir']),
            'metadata_file': str(temp_dataset_structure['metadata_path'])
        }
        
        loader = PartImageNetLoader(config)
        sample = loader.get_item(0)
        
        # Verify base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'image'
        assert 'annotations' in sample
        
        # Verify annotations structure
        annotations = sample['annotations']
        assert 'part_level_segmentation' in annotations
        assert 'num_parts' in annotations
        assert 'mask_info' in annotations
        assert 'dataset_info' in annotations
        
        # Check part segmentation
        parts = annotations['part_level_segmentation']
        assert len(parts) == 1  # Single object per image
        assert annotations['num_parts'] == 1
        
        part = parts[0]
        assert 'annotation_id' in part
        assert 'class_id' in part
        assert 'part_label' in part
        assert 'pixel_value' in part
        assert 'bbox' in part
        assert 'area' in part
        assert 'segmentation_mask' in part
        assert 'mask_shape' in part
        
        # Verify mask info
        mask_info = annotations['mask_info']
        assert 'unique_values' in mask_info
        assert 'background_value' in mask_info
        assert 'object_value' in mask_info
        assert len(mask_info['unique_values']) == 2
        
        # Verify dataset info
        dataset_info = annotations['dataset_info']
        assert dataset_info['task_type'] == 'binary_segmentation'
        assert dataset_info['source'] == 'PartImageNet'
        assert dataset_info['has_hierarchical_parts'] is False

    def test_get_item_index_out_of_range(self, temp_dataset_structure):
        """Test get_item with invalid index."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        with pytest.raises(IndexError, match="Index 10 out of range"):
            loader.get_item(10)

    def test_calculate_bbox_from_mask(self, temp_dataset_structure):
        """Test bounding box calculation from binary mask."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Test with a known mask (3x3 object at center)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:6, 3:6] = 1  # 3x3 object at position (3,3)
        
        bbox = loader._calculate_bbox_from_mask(mask)
        assert bbox == [3.0, 3.0, 3.0, 3.0]  # [x, y, width, height]
        
        # Test with empty mask
        empty_mask = np.zeros((10, 10), dtype=np.uint8)
        bbox = loader._calculate_bbox_from_mask(empty_mask)
        assert bbox == [0.0, 0.0, 0.0, 0.0]

    def test_get_samples_by_class(self, temp_dataset_structure):
        """Test retrieval of samples by class ID."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Get samples for fish class
        fish_samples = loader.get_samples_by_class('n01440764')
        assert len(fish_samples) == 2  # Two fish samples in dataset
        
        for sample in fish_samples:
            class_id = sample['annotations']['dataset_info']['class_id']
            assert class_id == 'n01440764'
        
        # Test with non-existent class
        empty_samples = loader.get_samples_by_class('n99999999')
        assert len(empty_samples) == 0

    def test_get_class_statistics(self, temp_dataset_structure):
        """Test class statistics calculation."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        stats = loader.get_class_statistics()
        
        assert 'total_samples' in stats
        assert 'total_classes' in stats
        assert 'class_distribution' in stats
        assert 'samples_per_class' in stats
        assert 'top_classes' in stats
        
        assert stats['total_samples'] == 4
        assert stats['total_classes'] == 3
        
        # Check class distribution
        assert stats['class_distribution']['n01440764'] == 2  # Two fish samples
        assert stats['class_distribution']['n01443537'] == 1
        assert stats['class_distribution']['n01484850'] == 1
        
        # Check samples per class stats
        per_class = stats['samples_per_class']
        assert per_class['min'] == 1
        assert per_class['max'] == 2
        assert per_class['avg'] == 4/3

    def test_get_mask_statistics(self, temp_dataset_structure):
        """Test mask statistics analysis."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        stats = loader.get_mask_statistics(sample_size=4)
        
        assert 'samples_analyzed' in stats
        assert 'unique_value_distribution' in stats
        assert 'background_values' in stats
        assert 'object_values' in stats
        assert 'object_ratio_stats' in stats
        
        assert stats['samples_analyzed'] == 4
        
        # Check unique value distribution (should be 2 for all samples)
        unique_dist = stats['unique_value_distribution']
        assert unique_dist['min'] == 2
        assert unique_dist['max'] == 2
        assert unique_dist['avg'] == 2
        
        # Check background values (should all be 158)
        bg_values = stats['background_values']
        assert 158 in bg_values['unique']
        assert bg_values['most_common'] == 158
        
        # Check object ratios (3x3 object in 10x10 image = 9/100 = 0.09)
        obj_ratios = stats['object_ratio_stats']
        assert abs(obj_ratios['avg'] - 0.09) < 0.001  # Allow for small floating point differences

    def test_malformed_annotation_file(self, temp_dataset_structure):
        """Test handling of corrupted annotation files."""
        # Corrupt one annotation file
        corrupted_path = temp_dataset_structure['annotations_dir'] / "n01440764_10029.png"
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid png file')
        
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Should still build index (doesn't validate files during indexing)
        assert len(loader._index) == 4
        
        # But get_item should handle the error gracefully
        sample = loader.get_item(0)  # This should correspond to the corrupted file
        
        # Check that it returns empty annotations
        annotations = sample['annotations']
        assert annotations['part_level_segmentation'] == []
        assert annotations['num_parts'] == 0
        assert 'error' in annotations['mask_info']

    def test_empty_directories(self, temp_dataset_structure):
        """Test loader behavior with empty directories."""
        # Create empty directories
        empty_images = temp_dataset_structure['temp_dir'] / "empty_images"
        empty_annotations = temp_dataset_structure['temp_dir'] / "empty_annotations"
        empty_images.mkdir()
        empty_annotations.mkdir()
        
        config = {
            'name': 'test_part_imagenet',
            'path': str(empty_images),
            'annotation_path': str(empty_annotations)
        }
        
        loader = PartImageNetLoader(config)
        
        # Should create loader with empty index
        assert len(loader._index) == 0
        assert len(loader) == 0
        
        # Statistics should handle empty dataset
        stats = loader.get_class_statistics()
        assert stats['total_samples'] == 0
        assert stats['total_classes'] == 0

    def test_mask_with_single_value(self, temp_dataset_structure):
        """Test handling of masks with only one unique value."""
        # Create a mask with only background pixels
        single_value_path = temp_dataset_structure['annotations_dir'] / "n01440764_10029.png"
        mask_array = np.full((10, 10), 158, dtype=np.uint8)  # All background
        mask_img = Image.fromarray(mask_array)
        mask_img.save(single_value_path)
        
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        sample = loader.get_item(0)
        
        # Should handle single-value masks gracefully
        annotations = sample['annotations']
        assert annotations['part_level_segmentation'] == []
        assert annotations['num_parts'] == 0
        assert len(annotations['mask_info']['unique_values']) == 1

    def test_bbox_edge_cases(self, temp_dataset_structure):
        """Test bounding box calculation edge cases."""
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Test single pixel object
        single_pixel = np.zeros((10, 10), dtype=np.uint8)
        single_pixel[5, 5] = 1
        bbox = loader._calculate_bbox_from_mask(single_pixel)
        assert bbox == [5.0, 5.0, 1.0, 1.0]
        
        # Test full image object
        full_mask = np.ones((10, 10), dtype=np.uint8)
        bbox = loader._calculate_bbox_from_mask(full_mask)
        assert bbox == [0.0, 0.0, 10.0, 10.0]

    def test_metadata_file_errors(self, temp_dataset_structure):
        """Test handling of metadata file loading errors."""
        # Test with invalid JSON file
        invalid_metadata = temp_dataset_structure['temp_dir'] / "invalid.json"
        with open(invalid_metadata, 'w') as f:
            f.write("invalid json content {")
        
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir']),
            'metadata_file': str(invalid_metadata)
        }
        
        # Should create loader without crashing
        loader = PartImageNetLoader(config)
        
        # part_id_to_label should be empty
        assert loader.part_id_to_label == {}
        
        # Test with non-existent metadata file
        config = {
            'name': 'test_part_imagenet',
            'path': str(temp_dataset_structure['images_dir']),
            'annotation_path': str(temp_dataset_structure['annotations_dir']),
            'metadata_file': '/nonexistent/metadata.json'
        }
        loader = PartImageNetLoader(config)
        assert loader.part_id_to_label == {}