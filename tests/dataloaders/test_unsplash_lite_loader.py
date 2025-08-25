# tests/dataloaders/test_unsplash_lite_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from core.dataloaders.unsplash_lite_loader import UnsplashLiteLoader


class TestUnsplashLiteLoader:
    """Test suite for UnsplashLiteLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_unsplash_lite',
            'path': '/fake/images/path'
        }

    @pytest.fixture
    def sample_annotation_data(self):
        """Create sample annotation data in Unsplash format."""
        return [
            {
                "image_id": "test_image_1",
                "image_filename": "test_image_1.jpg",
                "annotator_model": "google/gemini-2.5-flash",
                "source": "Unsplash-Lite-25k",
                "annotations": [
                    {
                        "box": [100, 150, 200, 250],
                        "desc": "A beautiful sunset over the mountains with vibrant colors."
                    },
                    {
                        "box": [300, 400, 450, 550],
                        "desc": "Rocky cliffs with intricate textures and patterns."
                    }
                ]
            },
            {
                "image_id": "test_image_2", 
                "image_filename": "test_image_2.jpg",
                "annotator_model": "google/gemini-2.5-flash",
                "source": "Unsplash-Lite-25k",
                "annotations": [
                    {
                        "box": [50, 60, 150, 160],
                        "desc": "Green foliage with morning dew drops."
                    }
                ]
            },
            {
                "image_id": "test_image_3",
                "image_filename": "test_image_3.jpg", 
                "annotator_model": "google/gemini-2.5-flash",
                "source": "Unsplash-Lite-25k",
                "annotations": []
            }
        ]

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'path'
        config_no_path = sample_config.copy()
        del config_no_path['path']
        
        with pytest.raises(ValueError, match="UnsplashLiteLoader config must include 'path'"):
            UnsplashLiteLoader(config_no_path)

    def test_init_nonexistent_path(self, sample_config):
        """Test that initialization fails when image path doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            UnsplashLiteLoader(sample_config)

    def test_build_index_simple_mode(self, sample_config):
        """Test successful index building in simple mode (no annotations)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy image files with different extensions
            image_files = ['test1.jpg', 'test2.png', 'test3.jpeg', 'not_image.txt']
            for filename in image_files:
                file_path = temp_path / filename
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    # Create actual image file
                    img = Image.new('RGB', (100, 80), color='white')
                    img.save(file_path)
                else:
                    # Create text file
                    file_path.write_text("not an image")
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Verify index was built correctly (only image files)
            assert len(loader) == 3
            assert not loader.use_annotations
            
            # Check that all indexed files are actual image files
            for image_path in loader._index:
                assert image_path.suffix.lower() in ['.jpg', '.png', '.jpeg']
                assert image_path.is_file()

    def test_build_index_enhanced_mode(self, sample_config, sample_annotation_data):
        """Test successful index building in enhanced mode (with annotations)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image files
            for i, ann_data in enumerate(sample_annotation_data, 1):
                image_path = images_dir / f"test_image_{i}.jpg"
                img = Image.new('RGB', (200, 150), color='white')
                img.save(image_path)
                
                # Create corresponding annotation file
                ann_path = annotations_dir / f"test_image_{i}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Create one extra image without annotations
            extra_image = images_dir / "no_annotations.jpg"
            img = Image.new('RGB', (100, 100), color='red')
            img.save(extra_image)
            
            # Update config with annotations
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Verify enhanced mode is enabled
            assert loader.use_annotations
            # Should only include images with annotations (3 out of 4)
            assert len(loader) == 3

    def test_get_item_simple_mode(self, sample_config):
        """Test successful sample retrieval in simple mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy image file
            image_path = temp_path / "test_image.jpg"
            img = Image.new('RGB', (300, 200), color='blue')
            img.save(image_path)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Test get_item
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_unsplash_lite'
            assert sample['sample_id'] == 'test_image'
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 300
            assert sample['height'] == 200
            
            # Verify annotations structure for simple mode
            assert 'dataset_info' in sample['annotations']
            dataset_info = sample['annotations']['dataset_info']
            assert dataset_info['task_type'] == 'high_resolution_image_collection'
            assert dataset_info['suitable_for_zoom'] == True
            assert dataset_info['source'] == 'Unsplash-Lite-25k'
            assert dataset_info['has_annotations'] == False

    def test_get_item_enhanced_mode(self, sample_config, sample_annotation_data):
        """Test successful sample retrieval in enhanced mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image and annotation files
            ann_data = sample_annotation_data[0]  # Use first annotation data
            image_path = images_dir / "test_image_1.jpg"
            img = Image.new('RGB', (500, 400), color='green')
            img.save(image_path)
            
            ann_path = annotations_dir / "test_image_1.json"
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Test get_item
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_unsplash_lite'
            assert sample['sample_id'] == 'test_image_1'
            assert sample['media_type'] == 'image'
            assert sample['width'] == 500
            assert sample['height'] == 400
            
            # Verify enhanced annotations
            annotations = sample['annotations']
            assert annotations['dataset_info']['has_annotations'] == True
            assert annotations['annotator_model'] == 'google/gemini-2.5-flash'
            assert annotations['num_annotations'] == 2
            assert len(annotations['bounding_boxes']) == 2
            assert len(annotations['regions']) == 2
            
            # Check processed regions
            region = annotations['regions'][0]
            assert 'region_id' in region
            assert 'description' in region
            assert 'bbox' in region
            assert region['bbox']['width'] == 100  # 200 - 100
            assert region['bbox']['height'] == 100  # 250 - 150

    def test_get_item_index_out_of_range(self, sample_config):
        """Test get_item with invalid index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create single image
            image_path = temp_path / "test.jpg"
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Test invalid index
            with pytest.raises(IndexError, match="Index 5 out of range"):
                loader.get_item(5)

    def test_annotation_processing(self, sample_config, sample_annotation_data):
        """Test annotation processing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create test files
            image_path = images_dir / "test_image_1.jpg"
            img = Image.new('RGB', (300, 300), color='white')
            img.save(image_path)
            
            ann_path = annotations_dir / "test_image_1.json"
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data[0], f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Test annotation processing directly
            raw_annotations = sample_annotation_data[0]['annotations']
            processed = loader._process_annotations(raw_annotations)
            
            assert len(processed) == 2
            assert processed[0]['region_id'] == 0
            assert processed[0]['description'] == "A beautiful sunset over the mountains with vibrant colors."
            assert processed[0]['bbox']['x'] == 100
            assert processed[0]['bbox']['y'] == 150
            assert processed[0]['bbox']['width'] == 100
            assert processed[0]['bbox']['height'] == 100

    def test_utility_methods(self, sample_config, sample_annotation_data):
        """Test utility methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create test files
            for i, ann_data in enumerate(sample_annotation_data, 1):
                image_path = images_dir / f"test_image_{i}.jpg"
                img = Image.new('RGB', (200, 200), color='white')
                img.save(image_path)
                
                ann_path = annotations_dir / f"test_image_{i}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Test get_image_by_id
            sample = loader.get_image_by_id('test_image_1')
            assert sample is not None
            assert sample['sample_id'] == 'test_image_1'
            
            # Test with non-existent ID
            sample = loader.get_image_by_id('nonexistent')
            assert sample is None
            
            # Test get_annotated_images
            annotated = loader.get_annotated_images()
            assert len(annotated) == 2  # test_image_3 has empty annotations
            
            # Test get_images_by_annotation_count
            multi_ann = loader.get_images_by_annotation_count(min_count=2)
            assert len(multi_ann) == 1  # Only test_image_1 has 2+ annotations
            
            single_ann = loader.get_images_by_annotation_count(min_count=1)
            assert len(single_ann) == 2  # test_image_1 and test_image_2

    def test_get_dataset_statistics_simple_mode(self, sample_config):
        """Test dataset statistics in simple mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create image files with different extensions
            for i, ext in enumerate(['jpg', 'png', 'jpeg'], 1):
                image_path = temp_path / f"test{i}.{ext}"
                img = Image.new('RGB', (100, 100), color='white')
                img.save(image_path)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Get statistics
            stats = loader.get_dataset_statistics()
            
            # Check basic stats
            assert stats['total_images'] == 3
            assert stats['annotations_enabled'] == False
            assert 'image_extensions' in stats
            assert stats['image_extensions']['.jpg'] == 1
            assert stats['image_extensions']['.png'] == 1
            assert stats['image_extensions']['.jpeg'] == 1
            assert isinstance(stats['avg_file_size_mb'], float)

    def test_get_dataset_statistics_enhanced_mode(self, sample_config, sample_annotation_data):
        """Test dataset statistics in enhanced mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create test files
            for i, ann_data in enumerate(sample_annotation_data, 1):
                image_path = images_dir / f"test_image_{i}.jpg"
                img = Image.new('RGB', (200, 200), color='white')
                img.save(image_path)
                
                ann_path = annotations_dir / f"test_image_{i}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(ann_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Get statistics
            stats = loader.get_dataset_statistics()
            
            # Check enhanced stats
            assert stats['total_images'] == 3
            assert stats['annotations_enabled'] == True
            assert stats['total_annotations'] > 0
            assert stats['avg_annotations_per_image'] > 0
            assert stats['annotation_coverage'] > 0
            assert 'image_extensions' in stats

    def test_nonexistent_annotations_path(self, sample_config):
        """Test handling of non-existent annotations path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create image file
            image_path = temp_path / "test.jpg"
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            # Update config with non-existent annotations path
            config = sample_config.copy()
            config['path'] = str(temp_path)
            config['annotations_path'] = str(temp_path / "nonexistent")
            
            # Initialize loader (should work but disable annotations)
            loader = UnsplashLiteLoader(config)
            
            assert not loader.use_annotations
            assert len(loader) == 1

    def test_empty_directory(self, sample_config):
        """Test handling of empty images directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config with empty directory
            config = sample_config.copy()
            config['path'] = str(temp_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Should handle empty directory gracefully
            assert len(loader) == 0
            stats = loader.get_dataset_statistics()
            assert stats['total_images'] == 0

    def test_malformed_annotation_file(self, sample_config):
        """Test handling of malformed annotation files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            annotations_dir = temp_path / "annotations"
            images_dir.mkdir()
            annotations_dir.mkdir()
            
            # Create image file
            image_path = images_dir / "test.jpg"
            img = Image.new('RGB', (100, 100), color='white')
            img.save(image_path)
            
            # Create malformed annotation file
            ann_path = annotations_dir / "test.json"
            ann_path.write_text("invalid json content")
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotations_path'] = str(annotations_dir)
            
            # Initialize loader
            loader = UnsplashLiteLoader(config)
            
            # Should load but with minimal annotations
            sample = loader.get_item(0)
            assert sample['annotations']['dataset_info']['has_annotations'] == True
            # But no actual annotation content due to parsing error
            assert sample['annotations'].get('num_annotations', 0) == 0