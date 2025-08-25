# tests/dataloaders/test_flickr30k_loader.py

import csv
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from core.dataloaders.flickr30k_loader import Flickr30kLoader


class TestFlickr30kLoader:
    """Test suite for Flickr30kLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_flickr30k',
            'path': '/fake/images/path',
            'annotation_file': '/fake/annotation_file.csv'
        }

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data in Flickr30k format."""
        return [
            {
                'raw': '["A man walking down the street.", "Person walking on sidewalk.", "A pedestrian in the city.", "Someone walking outside.", "A person strolling down the road."]',
                'sentids': '[0, 1, 2, 3, 4]',
                'split': 'train',
                'filename': '1000092795.jpg',
                'img_id': '0'
            },
            {
                'raw': '["Two dogs playing in the park.", "Dogs running around outdoors.", "Pets playing together.", "Two animals having fun.", "Dogs enjoying playtime."]',
                'sentids': '[5, 6, 7, 8, 9]',
                'split': 'train', 
                'filename': '1000268201.jpg',
                'img_id': '1'
            },
            {
                'raw': '["A red car on the highway.", "Vehicle driving on road.", "Automobile in motion.", "Car traveling fast.", "Red vehicle speeding."]',
                'sentids': '[10, 11, 12, 13, 14]',
                'split': 'val',
                'filename': '1000344755.jpg',
                'img_id': '2'
            }
        ]

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'path'
        config_no_path = sample_config.copy()
        del config_no_path['path']
        
        with pytest.raises(ValueError, match="Flickr30kLoader config must include 'path'"):
            Flickr30kLoader(config_no_path)
        
        # Test missing 'annotation_file'
        config_no_annotation = sample_config.copy()
        del config_no_annotation['annotation_file']
        
        with pytest.raises(ValueError, match="Flickr30kLoader config must include 'annotation_file'"):
            Flickr30kLoader(config_no_annotation)

    def test_init_nonexistent_paths(self, sample_config):
        """Test that initialization fails when paths don't exist."""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            Flickr30kLoader(sample_config)

    def test_build_index_success(self, sample_config, sample_csv_data):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for row in sample_csv_data:
                image_path = images_dir / row['filename']
                # Create a small dummy image
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(sample_csv_data)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 3
            expected_filenames = {'1000092795.jpg', '1000268201.jpg', '1000344755.jpg'}
            assert set(loader._index) == expected_filenames
            
            # Verify captions were grouped correctly
            assert len(loader._image_to_captions['1000092795.jpg']) == 5
            assert loader._image_to_captions['1000092795.jpg'][0] == "A man walking down the street."

    def test_build_index_with_split_filter(self, sample_config, sample_csv_data):
        """Test index building with split filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for row in sample_csv_data:
                image_path = images_dir / row['filename']
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(sample_csv_data)
            
            # Update config with real paths and train split filter
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            config['split'] = 'train'
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Verify only train split images are included
            assert len(loader) == 2  # Only train samples
            expected_filenames = {'1000092795.jpg', '1000268201.jpg'}
            assert set(loader._index) == expected_filenames

    def test_get_item_success(self, sample_config, sample_csv_data):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files with specific dimensions
            for row in sample_csv_data:
                image_path = images_dir / row['filename']
                img = Image.new('RGB', (200, 150), color='white')
                img.save(image_path)
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(sample_csv_data)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Test get_item for first sample
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_flickr30k'
            assert sample['sample_id'] in ['1000092795', '1000268201', '1000344755']  # Order not guaranteed
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 200
            assert sample['height'] == 150
            
            # Verify annotations structure
            assert 'captions' in sample['annotations']
            assert 'num_captions' in sample['annotations']
            assert 'flickr_img_id' in sample['annotations']
            assert 'split' in sample['annotations']
            assert 'sentence_ids' in sample['annotations']
            assert 'dataset_info' in sample['annotations']
            
            # Check captions
            captions = sample['annotations']['captions']
            assert len(captions) == 5
            assert sample['annotations']['num_captions'] == 5
            assert isinstance(captions[0], str)
            
            # Check metadata
            assert isinstance(sample['annotations']['flickr_img_id'], int)
            assert sample['annotations']['split'] in ['train', 'val']
            assert len(sample['annotations']['sentence_ids']) == 5
            
            # Check dataset info
            dataset_info = sample['annotations']['dataset_info']
            assert dataset_info['task_type'] == 'image_captioning'
            assert dataset_info['suitable_for_zoom'] == True
            assert dataset_info['caption_quality'] == 'high'
            assert isinstance(dataset_info['avg_caption_length'], float)

    def test_build_index_invalid_captions_format(self, sample_config):
        """Test handling of invalid captions format."""
        invalid_csv_data = [
            {
                'raw': 'invalid_format_not_a_list',  # Invalid format
                'sentids': '[0, 1, 2, 3, 4]',
                'split': 'train',
                'filename': '1000092795.jpg',
                'img_id': '0'
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / '1000092795.jpg'
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(invalid_csv_data)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader - should handle invalid format gracefully
            loader = Flickr30kLoader(config)
            
            # Should have no valid samples due to parsing error
            assert len(loader) == 0

    def test_build_index_missing_images(self, sample_config, sample_csv_data):
        """Test handling of missing image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create only some of the image files (first two)
            for row in sample_csv_data[:2]:
                image_path = images_dir / row['filename']
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            # Third image file is missing
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(sample_csv_data)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Should only have 2 valid samples (missing third image file)
            assert len(loader) == 2
            expected_filenames = {'1000092795.jpg', '1000268201.jpg'}
            assert set(loader._index) == expected_filenames

    def test_utility_methods(self, sample_config, sample_csv_data):
        """Test utility methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for row in sample_csv_data:
                image_path = images_dir / row['filename']
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create CSV annotation file
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                writer.writerows(sample_csv_data)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Test get_captions_for_image
            captions = loader.get_captions_for_image('1000092795.jpg')
            assert len(captions) == 5
            assert "A man walking down the street." in captions
            
            # Test get_images_by_split
            train_images = loader.get_images_by_split('train')
            val_images = loader.get_images_by_split('val')
            assert len(train_images) == 2
            assert len(val_images) == 1
            assert '1000344755.jpg' in val_images
            
            # Test get_dataset_statistics
            stats = loader.get_dataset_statistics()
            assert stats['total_images'] == 3
            assert stats['total_captions'] == 15  # 3 images × 5 captions each
            assert stats['avg_captions_per_image'] == 5.0
            assert stats['split_distribution']['train'] == 2
            assert stats['split_distribution']['val'] == 1
            assert isinstance(stats['avg_caption_length_words'], float)
            assert stats['filter_applied'] is None

    def test_empty_csv_file(self, sample_config):
        """Test handling of empty CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create empty CSV file (only headers)
            csv_file = temp_path / "annotations.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['raw', 'sentids', 'split', 'filename', 'img_id'])
                writer.writeheader()
                # No data rows
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(csv_file)
            
            # Initialize loader
            loader = Flickr30kLoader(config)
            
            # Should handle empty file gracefully
            assert len(loader) == 0
            stats = loader.get_dataset_statistics()
            assert stats['total_images'] == 0
            assert stats['total_captions'] == 0