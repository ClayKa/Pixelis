# tests/dataloaders/test_infographics_vqa_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from core.dataloaders.infographics_vqa_loader import InfographicsVqaLoader


class TestInfographicsVqaLoader:
    """Test suite for InfographicsVqaLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_infographics_vqa',
            'image_path': '/fake/images/path',
            'annotation_file': '/fake/annotation_file.json',
            'ocr_file': '/fake/ocr_file.json'
        }

    @pytest.fixture
    def sample_qa_data(self):
        """Create sample QA data in InfographicsVQA format."""
        return {
            "dataset_name": "InfographicsVQA",
            "dataset_version": "1.0",
            "dataset_split": "train",
            "data": [
                {
                    "questionId": 65718,
                    "question": "Which type of fonts offer better readability in printed works?",
                    "image_local_name": "20471.jpeg",
                    "image_url": "http://example.com/image.jpg",
                    "ocr_output_file": "20471.json",
                    "answers": ["serif fonts"],
                    "data_split": "train"
                },
                {
                    "questionId": 65719,
                    "question": "Which fonts are suited for the web?",
                    "image_local_name": "20471.jpeg", 
                    "image_url": "http://example.com/image.jpg",
                    "ocr_output_file": "20471.json",
                    "answers": ["sans serif", "sans serif fonts"],
                    "data_split": "train"
                },
                {
                    "questionId": 65720,
                    "question": "How many people use mobile devices?",
                    "image_local_name": "20472.jpeg",
                    "image_url": "http://example.com/image2.jpg", 
                    "ocr_output_file": "20472.json",
                    "answers": ["75%"],
                    "data_split": "train"
                }
            ]
        }

    @pytest.fixture
    def sample_ocr_data(self):
        """Create sample OCR data in AWS Textract format."""
        return {
            "20471": {
                "LINE": [
                    {
                        "BlockType": "LINE",
                        "Confidence": 99.75,
                        "Text": "Serif vs Sans Serif",
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.234,
                                "Height": 0.0106,
                                "Left": 0.104,
                                "Top": 0.018
                            }
                        }
                    },
                    {
                        "BlockType": "LINE",
                        "Confidence": 98.50,
                        "Text": "Better for print readability",
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.298,
                                "Height": 0.023,
                                "Left": 0.502,
                                "Top": 0.070
                            }
                        }
                    }
                ]
            },
            "20472": {
                "LINE": [
                    {
                        "BlockType": "LINE",
                        "Confidence": 95.30,
                        "Text": "Mobile Usage Statistics",
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.425,
                                "Height": 0.021,
                                "Left": 0.474,
                                "Top": 0.094
                            }
                        }
                    }
                ]
            }
        }

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'image_path'
        config_no_image = sample_config.copy()
        del config_no_image['image_path']
        
        with pytest.raises(ValueError, match="InfographicsVqaLoader config must include 'image_path'"):
            InfographicsVqaLoader(config_no_image)
        
        # Test missing 'annotation_file'
        config_no_annotation = sample_config.copy()
        del config_no_annotation['annotation_file']
        
        with pytest.raises(ValueError, match="InfographicsVqaLoader config must include 'annotation_file'"):
            InfographicsVqaLoader(config_no_annotation)
            
        # Test missing 'ocr_file'
        config_no_ocr = sample_config.copy()
        del config_no_ocr['ocr_file']
        
        with pytest.raises(ValueError, match="InfographicsVqaLoader config must include 'ocr_file'"):
            InfographicsVqaLoader(config_no_ocr)

    def test_init_nonexistent_paths(self, sample_config):
        """Test that initialization fails when paths don't exist."""
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            InfographicsVqaLoader(sample_config)

    def test_build_index_success(self, sample_config, sample_qa_data, sample_ocr_data):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files
            for filename in ["20471.jpeg", "20472.jpeg"]:
                image_path = images_dir / filename
                # Create a small dummy image
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_qa_data, f)
                
            # Create OCR file
            ocr_file = temp_path / "ocr_data.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 3  # 3 QA samples
            assert loader._index == sample_qa_data['data']
            
            # Verify OCR data was loaded
            assert '20471.jpeg' in loader._image_id_to_ocr
            assert '20472.jpeg' in loader._image_id_to_ocr

    def test_build_index_missing_data_field(self, sample_config, sample_ocr_data):
        """Test that build_index fails when annotation file is missing 'data' field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create annotation file without 'data' field
            annotation_file = temp_path / "qa_data.json"
            invalid_qa_data = {"dataset_name": "InfographicsVQA"}  # Missing 'data' field
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_qa_data, f)
                
            # Create OCR file
            ocr_file = temp_path / "ocr_data.json" 
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            with pytest.raises(ValueError, match="missing 'data' field"):
                InfographicsVqaLoader(config)

    def test_get_item_success(self, sample_config, sample_qa_data, sample_ocr_data):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image files with specific dimensions
            for filename in ["20471.jpeg", "20472.jpeg"]:
                image_path = images_dir / filename
                img = Image.new('RGB', (200, 150), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_qa_data, f)
                
            # Create OCR file
            ocr_file = temp_path / "ocr_data.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Test get_item for first sample
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_infographics_vqa'
            assert sample['sample_id'] == '65718'
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 200
            assert sample['height'] == 150
            
            # Verify annotations structure
            assert 'question' in sample['annotations']
            assert 'answers' in sample['annotations']
            assert 'ocr_tokens' in sample['annotations']
            
            # Check question and answers
            assert sample['annotations']['question'] == "Which type of fonts offer better readability in printed works?"
            assert sample['annotations']['answers'] == ["serif fonts"]
            
            # Check OCR tokens
            ocr_tokens = sample['annotations']['ocr_tokens']
            assert len(ocr_tokens) == 2  # Two lines from sample OCR data
            assert ocr_tokens[0]['text'] == "Serif vs Sans Serif"
            assert 'bbox' in ocr_tokens[0]
            assert 'confidence' in ocr_tokens[0]

    def test_get_item_missing_image_field(self, sample_config, sample_ocr_data):
        """Test that get_item fails when sample is missing image_local_name field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images" 
            images_dir.mkdir()
            
            # Create QA data with missing image_local_name
            invalid_qa_data = {
                "data": [
                    {
                        "questionId": 65718,
                        "question": "Test question?",
                        # Missing 'image_local_name' field
                        "answers": ["test answer"],
                        "data_split": "train"
                    }
                ]
            }
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_qa_data, f)
                
            # Create OCR file
            ocr_file = temp_path / "ocr_data.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(sample_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Test get_item should fail
            with pytest.raises(ValueError, match="missing 'image_local_name' field"):
                loader.get_item(0)

    def test_extract_ocr_tokens_empty_data(self, sample_config, sample_qa_data):
        """Test OCR token extraction with empty OCR data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "20471.jpeg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_qa_data, f)
                
            # Create empty OCR file
            ocr_file = temp_path / "ocr_data.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            
            # Update config
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Test get_item - should work with empty OCR data
            sample = loader.get_item(0)
            assert sample['annotations']['ocr_tokens'] == []

    def test_extract_ocr_tokens_word_format(self, sample_config, sample_qa_data):
        """Test OCR token extraction with WORD format data."""
        word_ocr_data = {
            "20471": {
                "WORD": [
                    {
                        "BlockType": "WORD",
                        "Confidence": 99.5,
                        "Text": "Hello",
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.1,
                                "Height": 0.05,
                                "Left": 0.2,
                                "Top": 0.3
                            }
                        }
                    },
                    {
                        "BlockType": "WORD", 
                        "Confidence": 98.2,
                        "Text": "World",
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.12,
                                "Height": 0.05,
                                "Left": 0.32,
                                "Top": 0.3
                            }
                        }
                    }
                ]
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "20471.jpeg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_qa_data, f)
                
            # Create OCR file with WORD format
            ocr_file = temp_path / "ocr_data.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(word_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Test get_item
            sample = loader.get_item(0)
            ocr_tokens = sample['annotations']['ocr_tokens']
            
            assert len(ocr_tokens) == 2
            assert ocr_tokens[0]['text'] == "Hello"
            assert ocr_tokens[1]['text'] == "World"
            assert ocr_tokens[0]['confidence'] == 99.5

    def test_load_individual_ocr_files(self, sample_config, sample_qa_data):
        """Test loading individual OCR files when main OCR file doesn't have the data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            ocr_dir = temp_path / "ocr"
            ocr_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "20471.jpeg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create annotation file
            annotation_file = temp_path / "qa_data.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_qa_data, f)
                
            # Create empty main OCR file
            ocr_file = ocr_dir / "main_ocr.json"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
                
            # Create individual OCR file
            individual_ocr_file = ocr_dir / "20471.json"
            individual_ocr_data = {
                "LINE": [
                    {
                        "Text": "Individual OCR",
                        "Confidence": 95.0,
                        "Geometry": {
                            "BoundingBox": {
                                "Width": 0.3,
                                "Height": 0.02,
                                "Left": 0.1,
                                "Top": 0.1
                            }
                        }
                    }
                ]
            }
            with open(individual_ocr_file, 'w', encoding='utf-8') as f:
                json.dump(individual_ocr_data, f)
            
            # Update config
            config = sample_config.copy()
            config['image_path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            config['ocr_file'] = str(ocr_file)
            
            # Initialize loader
            loader = InfographicsVqaLoader(config)
            
            # Test get_item - should load from individual OCR file
            sample = loader.get_item(0)
            ocr_tokens = sample['annotations']['ocr_tokens']
            
            assert len(ocr_tokens) == 1
            assert ocr_tokens[0]['text'] == "Individual OCR"