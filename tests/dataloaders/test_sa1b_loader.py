# tests/dataloaders/test_sa1b_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from core.dataloaders.sa1b_loader import Sa1bLoader


class TestSa1bLoader:
    """Test suite for Sa1bLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_sa1b',
            'path': '/fake/images/path',
            'annotations_path': '/fake/annotations/path'
        }

    @pytest.fixture
    def sample_annotation_data(self):
        """Create sample SA-1B annotation data."""
        return [
            {
                "image": {
                    "image_id": 1062875,
                    "width": 2247,
                    "height": 1500,
                    "file_name": "sa_1062875.jpg"
                },
                "annotations": [
                    {
                        "bbox": [100.0, 150.0, 50.0, 75.0],
                        "area": 3750,
                        "segmentation": {
                            "size": [1500, 2247],
                            "counts": "abc123def456ghi789"
                        },
                        "predicted_iou": 0.9447848200798035,
                        "point_coords": [[125.0, 187.5]],
                        "crop_box": [0.0, 0.0, 2247.0, 1500.0],
                        "id": 953410117,
                        "stability_score": 0.9823110103607178
                    },
                    {
                        "bbox": [200.0, 250.0, 30.0, 40.0],
                        "area": 1200,
                        "segmentation": {
                            "size": [1500, 2247],
                            "counts": "xyz789abc123def456"
                        },
                        "predicted_iou": 0.8873583078384399,
                        "point_coords": [[215.0, 270.0]],
                        "crop_box": [0.0, 0.0, 2247.0, 1500.0],
                        "id": 953410118,
                        "stability_score": 0.9583781361579895
                    }
                ]
            },
            {
                "image": {
                    "image_id": 1062876,
                    "width": 1920,
                    "height": 1080,
                    "file_name": "sa_1062876.jpg"
                },
                "annotations": [
                    {
                        "bbox": [300.0, 400.0, 100.0, 150.0],
                        "area": 15000,
                        "segmentation": {
                            "size": [1080, 1920],
                            "counts": "mask_data_example_here"
                        },
                        "predicted_iou": 0.9124428033828735,
                        "point_coords": [[350.0, 475.0]],
                        "crop_box": [0.0, 0.0, 1920.0, 1080.0],
                        "id": 953410119,
                        "stability_score": 0.9742140769958496
                    }
                ]
            },
            {
                "image": {
                    "image_id": 1062877,
                    "width": 800,
                    "height": 600,
                    "file_name": "sa_1062877.jpg"
                },
                "annotations": []
            }
        ]

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'path'
        config_no_path = sample_config.copy()
        del config_no_path['path']
        
        with pytest.raises(ValueError, match="Sa1bLoader config must include 'path'"):
            Sa1bLoader(config_no_path)
        
        # Test missing 'annotations_path'
        config_no_annotations = sample_config.copy()
        del config_no_annotations['annotations_path']
        
        with pytest.raises(ValueError, match="Sa1bLoader config must include 'annotations_path'"):
            Sa1bLoader(config_no_annotations)

    def test_init_nonexistent_paths(self, sample_config):
        """Test that initialization fails when paths don't exist."""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            Sa1bLoader(sample_config)

    def test_build_index_success(self, sample_config, sample_annotation_data):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image and annotation files
            for i, ann_data in enumerate(sample_annotation_data):
                image_info = ann_data['image']
                image_name = image_info['file_name']
                image_id = image_info['image_id']
                
                # Create image file
                image_path = images_dir / image_name
                img = Image.new('RGB', (image_info['width'], image_info['height']), color='white')
                img.save(image_path)
                
                # Create annotation file
                ann_path = annotations_dir / f"sa_{image_id}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 3
            
            # Check sample data
            expected_ids = {1062875, 1062876, 1062877}
            found_ids = {entry['image_id'] for entry in loader._index}
            assert found_ids == expected_ids

    def test_build_index_missing_annotations(self, sample_config, sample_annotation_data):
        """Test handling of missing annotation files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image files
            for ann_data in sample_annotation_data:
                image_info = ann_data['image']
                image_name = image_info['file_name']
                
                image_path = images_dir / image_name
                img = Image.new('RGB', (100, 100), color='white')
                img.save(image_path)
            
            # Create only some annotation files (first two)
            for i, ann_data in enumerate(sample_annotation_data[:2]):
                image_info = ann_data['image']
                image_id = image_info['image_id']
                
                ann_path = annotations_dir / f"sa_{image_id}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            # Third annotation file is missing
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Should only have 2 matched samples
            assert len(loader) == 2
            expected_ids = {1062875, 1062876}
            found_ids = {entry['image_id'] for entry in loader._index}
            assert found_ids == expected_ids

    def test_get_item_success(self, sample_config, sample_annotation_data):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create files for first sample
            ann_data = sample_annotation_data[0]
            image_info = ann_data['image']
            image_name = image_info['file_name']
            image_id = image_info['image_id']
            
            # Create image file
            image_path = images_dir / image_name
            img = Image.new('RGB', (image_info['width'], image_info['height']), color='blue')
            img.save(image_path)
            
            # Create annotation file
            ann_path = annotations_dir / f"sa_{image_id}.json"
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Test get_item
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_sa1b'
            assert sample['sample_id'] == '1062875'
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 2247
            assert sample['height'] == 1500
            
            # Verify annotations structure
            annotations = sample['annotations']
            assert 'instance_segmentation' in annotations
            assert 'num_masks' in annotations
            assert 'total_area' in annotations
            assert 'image_metadata' in annotations
            assert 'dataset_info' in annotations
            
            # Check instance segmentation
            masks = annotations['instance_segmentation']
            assert len(masks) == 2
            assert annotations['num_masks'] == 2
            assert annotations['total_area'] == 4950  # 3750 + 1200
            
            # Check first mask
            first_mask = masks[0]
            assert first_mask['annotation_id'] == 953410117
            assert first_mask['bbox'] == [100.0, 150.0, 50.0, 75.0]
            assert first_mask['area'] == 3750
            assert 'segmentation_rle' in first_mask
            assert first_mask['predicted_iou'] == 0.9447848200798035
            assert first_mask['stability_score'] == 0.9823110103607178
            
            # Check dataset info
            dataset_info = annotations['dataset_info']
            assert dataset_info['task_type'] == 'instance_segmentation'
            assert dataset_info['suitable_for_zoom'] == True
            assert dataset_info['source'] == 'SA-1B'
            assert dataset_info['mask_format'] == 'rle'
            assert dataset_info['has_point_prompts'] == True
            assert isinstance(dataset_info['avg_stability_score'], float)
            assert isinstance(dataset_info['avg_predicted_iou'], float)

    def test_get_item_index_out_of_range(self, sample_config, sample_annotation_data):
        """Test get_item with invalid index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create one sample
            ann_data = sample_annotation_data[0]
            image_info = ann_data['image']
            image_name = image_info['file_name']
            image_id = image_info['image_id']
            
            image_path = images_dir / image_name
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            ann_path = annotations_dir / f"sa_{image_id}.json"
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Test invalid index
            with pytest.raises(IndexError, match="Index 5 out of range"):
                loader.get_item(5)

    def test_statistics_calculation_methods(self, sample_config, sample_annotation_data):
        """Test statistics calculation methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create files for first sample
            ann_data = sample_annotation_data[0]
            image_info = ann_data['image']
            image_name = image_info['file_name']
            image_id = image_info['image_id']
            
            image_path = images_dir / image_name
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            ann_path = annotations_dir / f"sa_{image_id}.json"
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Test calculation methods
            annotations = ann_data['annotations']
            avg_stability = loader._calculate_avg_stability_score(annotations)
            avg_iou = loader._calculate_avg_predicted_iou(annotations)
            
            expected_stability = (0.9823110103607178 + 0.9583781361579895) / 2
            expected_iou = (0.9447848200798035 + 0.8873583078384399) / 2
            
            assert abs(avg_stability - expected_stability) < 0.0001
            assert abs(avg_iou - expected_iou) < 0.0001

    def test_utility_methods(self, sample_config, sample_annotation_data):
        """Test utility methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create test files
            for ann_data in sample_annotation_data:
                image_info = ann_data['image']
                image_name = image_info['file_name']
                image_id = image_info['image_id']
                
                image_path = images_dir / image_name
                img = Image.new('RGB', (100, 100), color='white')
                img.save(image_path)
                
                ann_path = annotations_dir / f"sa_{image_id}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Test get_sample_by_image_id
            sample = loader.get_sample_by_image_id(1062875)
            assert sample is not None
            assert sample['sample_id'] == '1062875'
            
            # Test with non-existent ID
            sample = loader.get_sample_by_image_id(999999)
            assert sample is None
            
            # Test get_high_quality_samples
            high_quality = loader.get_high_quality_samples(min_stability_score=0.95, min_predicted_iou=0.9)
            assert len(high_quality) >= 1  # At least one sample should meet criteria
            
            # Test get_samples_by_mask_count
            multi_mask_samples = loader.get_samples_by_mask_count(min_masks=2)
            assert len(multi_mask_samples) == 1  # Only first sample has 2 masks
            
            single_mask_samples = loader.get_samples_by_mask_count(min_masks=1, max_masks=1)
            assert len(single_mask_samples) == 1  # Second sample has 1 mask
            
            no_mask_samples = loader.get_samples_by_mask_count(min_masks=0, max_masks=0)
            assert len(no_mask_samples) == 1  # Third sample has 0 masks

    def test_get_dataset_statistics(self, sample_config, sample_annotation_data):
        """Test dataset statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create test files
            for ann_data in sample_annotation_data:
                image_info = ann_data['image']
                image_name = image_info['file_name']
                image_id = image_info['image_id']
                
                image_path = images_dir / image_name
                img = Image.new('RGB', (image_info['width'], image_info['height']), color='white')
                img.save(image_path)
                
                ann_path = annotations_dir / f"sa_{image_id}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Get statistics
            stats = loader.get_dataset_statistics()
            
            # Check basic stats
            assert stats['total_images'] == 3
            assert stats['total_masks'] > 0
            assert stats['avg_masks_per_image'] > 0
            assert stats['avg_stability_score'] > 0
            assert stats['avg_predicted_iou'] > 0
            assert 'image_resolution_stats' in stats
            assert 'mask_count_distribution' in stats
            assert stats['sample_size_used'] == 3

    def test_malformed_annotation_file(self, sample_config):
        """Test handling of malformed annotation files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image file
            image_path = images_dir / "sa_test.jpg"
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            # Create malformed annotation file
            ann_path = annotations_dir / "sa_test.json"
            ann_path.write_text("invalid json content")
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader (should skip malformed file)
            loader = Sa1bLoader(config)
            
            # Should have no samples due to malformed annotation
            assert len(loader) == 0

    def test_empty_directories(self, sample_config):
        """Test handling of empty directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Update config with empty directories
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Should handle empty directories gracefully
            assert len(loader) == 0
            stats = loader.get_dataset_statistics()
            assert stats['total_images'] == 0
            assert stats['total_masks'] == 0

    def test_image_without_annotation(self, sample_config):
        """Test handling of images without matching annotation files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image file but no corresponding annotation
            image_path = images_dir / "sa_orphan.jpg"
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = Sa1bLoader(config)
            
            # Should have no samples due to missing annotation
            assert len(loader) == 0