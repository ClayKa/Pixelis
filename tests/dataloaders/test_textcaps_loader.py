# tests/dataloaders/test_textcaps_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from core.dataloaders.textcaps_loader import TextCapsLoader


class TestTextCapsLoader:
    """Test suite for TextCapsLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_textcaps',
            'annotation_file': '/fake/annotation_file.json',
            'ocr_file': '/fake/ocr_file.json',
            'image_path': '/fake/images/'
        }

    @pytest.fixture
    def sample_annotation_data(self):
        """Create sample TextCaps annotation data."""
        return {
            "dataset_type": "train",
            "dataset_name": "textcaps",
            "dataset_version": 0.1,
            "data": [
                {
                    "image_id": "011e7e629fb9ae7b",
                    "image_classes": ["Personal care", "Perfume"],
                    "flickr_original_url": "https://example.com/image1.jpg",
                    "flickr_300k_url": "https://example.com/image1_small.jpg",
                    "image_width": 1024,
                    "image_height": 702,
                    "set_name": "train",
                    "image_name": "011e7e629fb9ae7b",
                    "image_path": "train/011e7e629fb9ae7b.jpg",
                    "caption_id": 100000000,
                    "caption_str": "Five Listerine Zero mouthwash bottles on a store shelf",
                    "caption_tokens": ["<s>", "five", "listerine", "zero", "mouthwash", "bottles", "on", "a", "store", "shelf", "</s>"],
                    "reference_strs": [
                        "Five Listerine Zero mouthwash bottles on a store shelf",
                        "Many bottles of Listerine Zero are placed together.",
                        "Listerine Zero mouthwash is less intense with zero alcohol.",
                        "Several bottles of Listerine Zero sit on display on a shelf",
                        "Listerine zero bottles which is less intense and zero alcohol."
                    ],
                    "reference_tokens": [
                        ["<s>", "five", "listerine", "zero", "mouthwash", "bottles", "on", "a", "store", "shelf", "</s>"],
                        ["<s>", "many", "bottles", "of", "listerine", "zero", "are", "placed", "together", "</s>"],
                        ["<s>", "listerine", "zero", "mouthwash", "is", "less", "intense", "with", "zero", "alcohol", "</s>"],
                        ["<s>", "several", "bottles", "of", "listerine", "zero", "sit", "on", "display", "on", "a", "shelf", "</s>"],
                        ["<s>", "listerine", "zero", "bottles", "which", "is", "less", "intense", "and", "zero", "alcohol", "</s>"]
                    ]
                },
                {
                    "image_id": "09efcb22ca121f57",
                    "image_classes": ["Clock", "Watch", "Wall clock"],
                    "flickr_original_url": "https://example.com/image2.jpg",
                    "flickr_300k_url": "https://example.com/image2_small.jpg",
                    "image_width": 1024,
                    "image_height": 1024,
                    "set_name": "train",
                    "image_name": "09efcb22ca121f57",
                    "image_path": "train/09efcb22ca121f57.jpg",
                    "caption_id": 100000001,
                    "caption_str": "The time is now 10:22 according to this clock.",
                    "caption_tokens": ["<s>", "the", "time", "is", "now", "10", ":", "22", "according", "to", "this", "clock", "</s>"],
                    "reference_strs": [
                        "The time is now 10:22 according to this clock.",
                        "A large clock by Harris and Thompson displays a time of 10:22.",
                        "Large roman numeral clock by Angel Square London.",
                        "Clock with roman numeral numbers which say \"Angel Square\" on the front.",
                        "A clock says Angel Square London on its face."
                    ],
                    "reference_tokens": [
                        ["<s>", "the", "time", "is", "now", "10", ":", "22", "according", "to", "this", "clock", "</s>"],
                        ["<s>", "a", "large", "clock", "by", "harris", "and", "thompson", "displays", "a", "time", "of", "10", ":", "22", "</s>"],
                        ["<s>", "large", "roman", "numeral", "clock", "by", "angel", "square", "london", "</s>"],
                        ["<s>", "clock", "with", "roman", "numeral", "numbers", "which", "say", "\"", "angel", "square", "\"", "on", "the", "front", "</s>"],
                        ["<s>", "a", "clock", "says", "angel", "square", "london", "on", "its", "face", "</s>"]
                    ]
                }
            ]
        }

    @pytest.fixture
    def sample_ocr_data(self):
        """Create sample OCR data in TextCaps format."""
        return {
            "data": [
                {
                    "image_id": "011e7e629fb9ae7b",
                    "ocr_tokens": ["LESS", "INTENSE", "ZERO", "ALCOHOL", "LISTERINE", "ZERO", "MOUTHWASH"],
                    "ocr_info": [
                        {
                            "word": "LESS",
                            "bounding_box": {
                                "width": 0.053537655621767044,
                                "height": 0.03051217645406723,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.0947766974568367,
                                "top_left_y": 0.3009840250015259
                            }
                        },
                        {
                            "word": "INTENSE",
                            "bounding_box": {
                                "width": 0.1266782432794571,
                                "height": 0.044393330812454224,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0.3725101947784424,
                                "top_left_x": 0.48142001032829285,
                                "top_left_y": 0.3242570161819458
                            }
                        },
                        {
                            "word": "ZERO",
                            "bounding_box": {
                                "width": 0.05462116375565529,
                                "height": 0.03007006086409092,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.09750957787036896,
                                "top_left_y": 0.33078813552856445
                            }
                        },
                        {
                            "word": "ALCOHOL",
                            "bounding_box": {
                                "width": 0.09473343193531036,
                                "height": 0.03058902733027935,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0.3725101947784424,
                                "top_left_x": 0.689324140548706,
                                "top_left_y": 0.34880441427230835
                            }
                        },
                        {
                            "word": "LISTERINE",
                            "bounding_box": {
                                "width": 0.12661093473434448,
                                "height": 0.054334960877895355,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0.3725101947784424,
                                "top_left_x": 0.6051996350288391,
                                "top_left_y": 0.4055970311164856
                            }
                        },
                        {
                            "word": "ZERO",
                            "bounding_box": {
                                "width": 0.08325061202049255,
                                "height": 0.05052648484706879,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0.3725101947784424,
                                "top_left_x": 0.7371019721031189,
                                "top_left_y": 0.40215674042701721
                            }
                        },
                        {
                            "word": "MOUTHWASH",
                            "bounding_box": {
                                "width": 0.18949371576309204,
                                "height": 0.053604558110237122,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0.3725101947784424,
                                "top_left_x": 0.6109029650688171,
                                "top_left_y": 0.47251179814338684
                            }
                        }
                    ]
                },
                {
                    "image_id": "09efcb22ca121f57",
                    "ocr_tokens": ["10", "22", "ANGEL", "SQUARE", "LONDON"],
                    "ocr_info": [
                        {
                            "word": "10",
                            "bounding_box": {
                                "width": 0.04,
                                "height": 0.05,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.45,
                                "top_left_y": 0.25
                            }
                        },
                        {
                            "word": "22",
                            "bounding_box": {
                                "width": 0.04,
                                "height": 0.05,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.55,
                                "top_left_y": 0.45
                            }
                        },
                        {
                            "word": "ANGEL",
                            "bounding_box": {
                                "width": 0.08,
                                "height": 0.03,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.4,
                                "top_left_y": 0.8
                            }
                        },
                        {
                            "word": "SQUARE",
                            "bounding_box": {
                                "width": 0.08,
                                "height": 0.03,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.48,
                                "top_left_y": 0.8
                            }
                        },
                        {
                            "word": "LONDON",
                            "bounding_box": {
                                "width": 0.08,
                                "height": 0.03,
                                "rotation": 0,
                                "roll": 0,
                                "pitch": 0,
                                "yaw": 0,
                                "top_left_x": 0.56,
                                "top_left_y": 0.8
                            }
                        }
                    ]
                }
            ]
        }

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'annotation_file'
        config_no_annotation = sample_config.copy()
        del config_no_annotation['annotation_file']
        
        with pytest.raises(ValueError, match="TextCapsLoader config must include 'annotation_file'"):
            TextCapsLoader(config_no_annotation)
        
        # Test missing 'ocr_file'
        config_no_ocr = sample_config.copy()
        del config_no_ocr['ocr_file']
        
        with pytest.raises(ValueError, match="TextCapsLoader config must include 'ocr_file'"):
            TextCapsLoader(config_no_ocr)
        
        # Test missing 'image_path'
        config_no_images = sample_config.copy()
        del config_no_images['image_path']
        
        with pytest.raises(ValueError, match="TextCapsLoader config must include 'image_path'"):
            TextCapsLoader(config_no_images)

    def test_init_nonexistent_paths(self, sample_config):
        """Test that initialization fails when paths don't exist."""
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            TextCapsLoader(sample_config)

    def test_build_index_success(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for sample in sample_annotation_data['data']:
                image_id = sample['image_id']
                image_path = images_dir / f"{image_id}.jpg"
                # Create a small dummy image
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            # Create OCR file
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 2
            assert len(loader._image_id_to_ocr) == 2
            
            # Check that specific image IDs exist
            expected_ids = {'011e7e629fb9ae7b', '09efcb22ca121f57'}
            found_ids = {sample['image_id'] for sample in loader._index}
            assert found_ids == expected_ids

    def test_build_index_invalid_annotation_format(self, sample_config):
        """Test handling of invalid annotation file format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create invalid annotation file (missing 'data' key)
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump({"invalid": "format"}, f)
            
            # Create valid OCR file
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump({"data": []}, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Should raise ValueError for invalid format
            with pytest.raises(ValueError, match="Invalid annotation file format"):
                TextCapsLoader(config)

    def test_build_index_missing_images(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test handling of missing image files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create only one image file (first one)
            first_sample = sample_annotation_data['data'][0]
            image_path = images_dir / f"{first_sample['image_id']}.jpg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            # Second image file is missing
            
            # Create annotation file
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            # Create OCR file
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Should only have 1 valid sample
            assert len(loader) == 1
            assert loader._index[0]['image_id'] == first_sample['image_id']

    def test_get_item_success(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for sample in sample_annotation_data['data']:
                image_id = sample['image_id']
                image_path = images_dir / f"{image_id}.jpg"
                img = Image.new('RGB', (200, 150), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            # Create OCR file
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Test get_item for first sample
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_textcaps'
            assert sample['sample_id'] == '011e7e629fb9ae7b'
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 200
            assert sample['height'] == 150
            
            # Verify annotations structure
            assert 'captions' in sample['annotations']
            assert 'primary_caption' in sample['annotations']
            assert 'num_captions' in sample['annotations']
            assert 'ocr_tokens' in sample['annotations']
            assert 'num_ocr_tokens' in sample['annotations']
            assert 'image_classes' in sample['annotations']
            assert 'dataset_info' in sample['annotations']
            
            # Check captions
            captions = sample['annotations']['captions']
            assert len(captions) == 5
            assert sample['annotations']['num_captions'] == 5
            assert "Five Listerine Zero mouthwash bottles on a store shelf" in captions
            assert sample['annotations']['primary_caption'] == "Five Listerine Zero mouthwash bottles on a store shelf"
            
            # Check OCR tokens
            ocr_tokens = sample['annotations']['ocr_tokens']
            assert len(ocr_tokens) == 7
            assert sample['annotations']['num_ocr_tokens'] == 7
            assert ocr_tokens[0]['text'] == 'LESS'
            assert 'bbox' in ocr_tokens[0]
            assert 'confidence' in ocr_tokens[0]
            
            # Check image classes
            assert sample['annotations']['image_classes'] == ["Personal care", "Perfume"]
            
            # Check dataset info
            dataset_info = sample['annotations']['dataset_info']
            assert dataset_info['task_type'] == 'image_text_captioning'
            assert dataset_info['suitable_for_zoom'] == True
            assert dataset_info['text_grounded'] == True
            assert dataset_info['ocr_coverage'] == True
            assert isinstance(dataset_info['avg_caption_length'], float)

    def test_get_item_index_out_of_range(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test get_item with invalid index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for sample in sample_annotation_data['data']:
                image_id = sample['image_id']
                image_path = images_dir / f"{image_id}.jpg"
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create files
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Test invalid index
            with pytest.raises(IndexError, match="Index 10 out of range"):
                loader.get_item(10)

    def test_utility_methods(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test utility methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for sample in sample_annotation_data['data']:
                image_id = sample['image_id']
                image_path = images_dir / f"{image_id}.jpg"
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create files
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Test get_captions_for_image
            captions = loader.get_captions_for_image('011e7e629fb9ae7b')
            assert len(captions) == 5
            assert "Five Listerine Zero mouthwash bottles on a store shelf" in captions
            
            # Test with non-existent image ID
            with pytest.raises(ValueError, match="Image ID 'nonexistent' not found"):
                loader.get_captions_for_image('nonexistent')
            
            # Test get_ocr_tokens_for_image
            ocr_tokens = loader.get_ocr_tokens_for_image('011e7e629fb9ae7b')
            assert len(ocr_tokens) == 7
            assert ocr_tokens[0]['text'] == 'LESS'
            assert 'bbox' in ocr_tokens[0]
            
            # Test with non-existent OCR data
            with pytest.raises(ValueError, match="OCR data for image ID 'nonexistent' not found"):
                loader.get_ocr_tokens_for_image('nonexistent')
            
            # Test get_images_by_class
            clock_images = loader.get_images_by_class('Clock')
            assert len(clock_images) == 1
            assert clock_images[0]['image_id'] == '09efcb22ca121f57'
            
            personal_care_images = loader.get_images_by_class('Personal care')
            assert len(personal_care_images) == 1
            assert personal_care_images[0]['image_id'] == '011e7e629fb9ae7b'
            
            nonexistent_images = loader.get_images_by_class('NonexistentClass')
            assert len(nonexistent_images) == 0

    def test_get_dataset_statistics(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test dataset statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for sample in sample_annotation_data['data']:
                image_id = sample['image_id']
                image_path = images_dir / f"{image_id}.jpg"
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create files
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Get statistics
            stats = loader.get_dataset_statistics()
            
            # Check basic stats
            assert stats['total_images'] == 2
            assert stats['total_captions'] > 0
            assert stats['avg_captions_per_image'] == 5.0  # Both images have 5 captions
            assert stats['avg_caption_length_words'] > 0
            assert stats['total_ocr_tokens'] > 0
            assert stats['avg_ocr_tokens_per_image'] > 0
            assert 'image_classes_distribution' in stats
            assert stats['ocr_coverage'] == 1.0  # All images have OCR
            assert stats['sample_size_used'] == 2

    def test_empty_annotation_file(self, sample_config):
        """Test handling of empty annotation file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create empty annotation file
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump({"data": []}, f)
            
            # Create empty OCR file
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump({"data": []}, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Should handle empty file gracefully
            assert len(loader) == 0
            stats = loader.get_dataset_statistics()
            assert stats['total_images'] == 0
            assert stats['total_captions'] == 0

    def test_calculate_avg_caption_length(self, sample_config, sample_annotation_data, sample_ocr_data):
        """Test average caption length calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "011e7e629fb9ae7b.jpg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create files
            annotation_file = temp_path / "annotations.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotation_data, f)
            
            ocr_file = temp_path / "ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            config['image_path'] = str(images_dir)
            
            # Initialize loader
            loader = TextCapsLoader(config)
            
            # Test the calculation method directly
            test_captions = ["Short caption", "This is a longer caption with more words"]
            avg_length = loader._calculate_avg_caption_length(test_captions)
            
            # First caption: 2 words, second caption: 8 words, average: 5.0
            assert avg_length == 5.0
            
            # Test with empty list
            empty_avg = loader._calculate_avg_caption_length([])
            assert empty_avg == 0.0