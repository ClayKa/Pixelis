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
        # Test missing 'image_path'
        config = {'mask_path': '/some/path'}
        with pytest.raises(ValueError, match="PartImageNetLoader config must include 'image_path'"):
            PartImageNetLoader(config)
        
        # Test missing 'mask_path'
        config = {'image_path': '/some/path'}
        with pytest.raises(ValueError, match="PartImageNetLoader config must include 'mask_path'"):
            PartImageNetLoader(config)

    def test_init_nonexistent_paths(self):
        """Test initialization with non-existent paths."""
        config = {
            'image_path': '/nonexistent/images',
            'mask_path': '/nonexistent/annotations'
        }
        with pytest.raises(FileNotFoundError):
            PartImageNetLoader(config)

    def test_build_index_success(self, temp_dataset_structure):
        """Test successful index building with matching images and annotations."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
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
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Should only match 3 out of 4 images
        assert len(loader._index) == 3
        
        # Verify the missing image is not in the index
        indexed_stems = {path[0].stem for path in loader._index}
        assert 'n01440764_10043' not in indexed_stems

    def test_metadata_loading(self, temp_dataset_structure):
        """Test that loader works without metadata file (revised implementation)."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        # Should initialize successfully without metadata file
        loader = PartImageNetLoader(config)
        
        # The loader no longer uses metadata files in the revised implementation
        assert len(loader._index) > 0  # Should have successfully built index

    def test_get_item_success(self, temp_dataset_structure):
        """Test successful item retrieval with mask parsing."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        sample = loader.get_item(0)
        
        # Verify base structure
        assert 'sample_id' in sample
        assert 'media_path' in sample
        assert 'media_type' in sample
        assert sample['media_type'] == 'image'
        assert 'annotations' in sample
        
        # Verify revised annotations structure
        annotations = sample['annotations']
        assert 'segmentation_mask_path' in annotations
        assert 'available_instance_ids' in annotations
        assert 'instance_details' in annotations
        assert 'num_instances' in annotations
        assert 'mask_shape' in annotations
        
        # Check that mask path is provided
        assert annotations['segmentation_mask_path'] is not None
        assert Path(annotations['segmentation_mask_path']).exists()
        
        # Check instance IDs (should have non-zero values from the mask)
        instance_ids = annotations['available_instance_ids']
        assert len(instance_ids) > 0  # Should have at least one instance
        assert all(isinstance(id, int) for id in instance_ids)
        assert all(id != 0 for id in instance_ids)  # No background (0) values
        
        # Check instance details
        instance_details = annotations['instance_details']
        assert len(instance_details) == len(instance_ids)
        assert annotations['num_instances'] == len(instance_ids)
        
        for detail in instance_details:
            assert 'instance_id' in detail
            assert 'bbox' in detail
            assert 'area' in detail
            assert 'pixel_ratio' in detail
            assert detail['instance_id'] in instance_ids

    def test_get_item_index_out_of_range(self, temp_dataset_structure):
        """Test get_item with invalid index."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        with pytest.raises(IndexError, match="Index 10 out of range"):
            loader.get_item(10)

    def test_calculate_bbox_from_mask(self, temp_dataset_structure):
        """Test bounding box calculation from binary mask."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
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
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Get samples for fish class
        fish_samples = loader.get_samples_by_class('n01440764')
        assert len(fish_samples) == 2  # Two fish samples in dataset
        
        for sample in fish_samples:
            # Check that sample ID starts with the class ID
            assert sample['sample_id'].startswith('n01440764')
        
        # Test with non-existent class
        empty_samples = loader.get_samples_by_class('n99999999')
        assert len(empty_samples) == 0

    def test_get_class_statistics(self, temp_dataset_structure):
        """Test class statistics calculation."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
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
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        stats = loader.get_mask_statistics(sample_size=4)
        
        assert 'samples_analyzed' in stats
        assert 'instance_count_distribution' in stats
        assert 'instance_ids' in stats
        assert 'instance_area_ratios' in stats
        
        assert stats['samples_analyzed'] == 4
        
        # Check instance count distribution (should be 1 instance per sample)
        instance_dist = stats['instance_count_distribution']
        assert instance_dist['min'] == 1
        assert instance_dist['max'] == 1
        assert instance_dist['avg'] == 1
        
        # Check instance IDs
        instance_ids_info = stats['instance_ids']
        assert instance_ids_info['unique_count'] >= 1  # At least one unique instance ID
        
        # Check instance area ratios (3x3 object in 10x10 image = 9/100 = 0.09)
        area_ratios = stats['instance_area_ratios']
        assert abs(area_ratios['avg'] - 0.09) < 0.001  # Allow for small floating point differences

    def test_malformed_annotation_file(self, temp_dataset_structure):
        """Test handling of corrupted annotation files."""
        # Corrupt one annotation file
        corrupted_path = temp_dataset_structure['annotations_dir'] / "n01440764_10029.png"
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid png file')
        
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        
        # Should still build index (doesn't validate files during indexing)
        assert len(loader._index) == 4
        
        # But get_item should handle the error gracefully
        sample = loader.get_item(0)  # This should correspond to the corrupted file
        
        # Check that it returns empty annotations with error
        annotations = sample['annotations']
        assert annotations['available_instance_ids'] == []
        assert annotations['num_instances'] == 0
        assert 'error' in annotations

    def test_empty_directories(self, temp_dataset_structure):
        """Test loader behavior with empty directories."""
        # Create empty directories
        empty_images = temp_dataset_structure['temp_dir'] / "empty_images"
        empty_annotations = temp_dataset_structure['temp_dir'] / "empty_annotations"
        empty_images.mkdir()
        empty_annotations.mkdir()
        
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(empty_images),
            'mask_path': str(empty_annotations)
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
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        loader = PartImageNetLoader(config)
        sample = loader.get_item(0)
        
        # Should handle single-value masks gracefully (no non-zero instances)
        annotations = sample['annotations']
        assert annotations['available_instance_ids'] == []  # No non-zero values
        assert annotations['num_instances'] == 0
        assert annotations['instance_details'] == []

    def test_bbox_edge_cases(self, temp_dataset_structure):
        """Test bounding box calculation edge cases."""
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
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
        """Test that loader no longer uses metadata files."""
        # The revised implementation doesn't use metadata files
        config = {
            'name': 'test_part_imagenet',
            'image_path': str(temp_dataset_structure['images_dir']),
            'mask_path': str(temp_dataset_structure['annotations_dir'])
        }
        
        # Should create loader successfully without any metadata
        loader = PartImageNetLoader(config)
        
        # Loader should work normally
        assert len(loader._index) > 0
        
        # Test that loader still works without metadata
        sample = loader.get_item(0)
        assert 'available_instance_ids' in sample['annotations']