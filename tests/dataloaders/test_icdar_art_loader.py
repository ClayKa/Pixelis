# tests/dataloaders/test_icdar_art_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from core.dataloaders.icdar_art_loader import IcdarArTLoader


class TestIcdarArTLoader:
    """Test suite for IcdarArTLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_icdar_art',
            'path': '/fake/images/path',
            'annotation_file': '/fake/annotations/train_labels.json'
        }

    @pytest.fixture
    def sample_annotations(self):
        """Create sample annotation data in ICDAR ArT format."""
        return {
            "gt_1": [
                {
                    "transcription": "Hello",
                    "points": [[10, 20], [50, 20], [50, 40], [10, 40]],
                    "language": "English",
                    "illegibility": False
                },
                {
                    "transcription": "World",
                    "points": [[60, 20], [100, 20], [100, 40], [60, 40]],
                    "language": "English", 
                    "illegibility": False
                }
            ],
            "gt_2": [
                {
                    "transcription": "测试",
                    "points": [[15, 25], [45, 25], [45, 45], [15, 45]],
                    "language": "Chinese",
                    "illegibility": False
                },
                {
                    "transcription": "illegible_text",
                    "points": [[50, 50], [80, 50], [80, 70], [50, 70]],
                    "language": "English",
                    "illegibility": True  # This should be filtered out
                }
            ]
        }

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'path'
        config_no_path = sample_config.copy()
        del config_no_path['path']
        
        with pytest.raises(ValueError, match="IcdarArTLoader config must include 'path'"):
            IcdarArTLoader(config_no_path)
        
        # Test missing 'annotation_file'
        config_no_annotation = sample_config.copy()
        del config_no_annotation['annotation_file']
        
        with pytest.raises(ValueError, match="IcdarArTLoader config must include 'annotation_file'"):
            IcdarArTLoader(config_no_annotation)

    def test_init_nonexistent_paths(self, sample_config):
        """Test that initialization fails when paths don't exist."""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            IcdarArTLoader(sample_config)

    def test_build_index_success(self, sample_config, sample_annotations):
        """Test successful index building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            annotations_dir = temp_path / "annotations"
            annotations_dir.mkdir()
            
            # Create dummy image files
            for filename in ["gt_1.jpg", "gt_2.jpg"]:
                image_path = images_dir / filename
                # Create a small dummy image
                img = Image.new('RGB', (100, 80), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = annotations_dir / "train_labels.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotations, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            
            # Initialize loader
            loader = IcdarArTLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 2
            assert set(loader._index) == {"gt_1", "gt_2"}
            assert loader._annotations_map == sample_annotations

    def test_get_item_success(self, sample_config, sample_annotations):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images" 
            images_dir.mkdir()
            annotations_dir = temp_path / "annotations"
            annotations_dir.mkdir()
            
            # Create dummy image files with specific dimensions
            for filename in ["gt_1.jpg", "gt_2.jpg"]:
                image_path = images_dir / filename
                img = Image.new('RGB', (200, 150), color='white')
                img.save(image_path)
            
            # Create annotation file
            annotation_file = annotations_dir / "train_labels.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotations, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            
            # Initialize loader
            loader = IcdarArTLoader(config)
            
            # Test get_item for first sample
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_icdar_art'
            assert sample['sample_id'] in ["gt_1", "gt_2"]  # Order not guaranteed
            assert sample['media_type'] == 'image'
            assert Path(sample['media_path']).exists()
            assert sample['width'] == 200
            assert sample['height'] == 150
            
            # Verify annotations structure
            assert 'scene_text' in sample['annotations']
            scene_text = sample['annotations']['scene_text']
            
            # Check that we have the expected number of non-illegible annotations
            if sample['sample_id'] == "gt_1":
                assert len(scene_text) == 2  # Both annotations are legible
                assert scene_text[0]['text'] == "Hello"
                assert scene_text[1]['text'] == "World"
            elif sample['sample_id'] == "gt_2":
                assert len(scene_text) == 1  # Only one legible annotation
                assert scene_text[0]['text'] == "测试"
            
            # Verify annotation format
            for annotation in scene_text:
                assert 'text' in annotation
                assert 'bbox_polygon' in annotation
                assert 'language' in annotation
                assert isinstance(annotation['bbox_polygon'], list)
                assert len(annotation['bbox_polygon']) >= 3  # At least 3 points for a polygon

    def test_illegible_text_filtering(self, sample_config, sample_annotations):
        """Test that illegible text annotations are properly filtered out."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            annotations_dir = temp_path / "annotations"
            annotations_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "gt_2.jpg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create annotation file
            annotation_file = annotations_dir / "train_labels.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(sample_annotations, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            
            # Initialize loader
            loader = IcdarArTLoader(config)
            
            # Find the index for gt_2
            gt_2_index = loader._index.index("gt_2")
            sample = loader.get_item(gt_2_index)
            
            # Verify that only legible annotation is included
            scene_text = sample['annotations']['scene_text']
            assert len(scene_text) == 1
            assert scene_text[0]['text'] == "测试"
            
            # Verify that illegible annotation was filtered out
            texts = [ann['text'] for ann in scene_text]
            assert "illegible_text" not in texts

    def test_missing_language_field_handling(self, sample_config):
        """Test handling of annotations with missing language field."""
        # Create annotation data with missing language field
        annotations_no_lang = {
            "gt_1": [
                {
                    "transcription": "No Language",
                    "points": [[10, 20], [50, 20], [50, 40], [10, 40]],
                    "illegibility": False
                    # Note: no 'language' field
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary directory structure
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            annotations_dir = temp_path / "annotations"
            annotations_dir.mkdir()
            
            # Create dummy image file
            image_path = images_dir / "gt_1.jpg"
            img = Image.new('RGB', (100, 80), color='white')
            img.save(image_path)
            
            # Create annotation file
            annotation_file = annotations_dir / "train_labels.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations_no_lang, f)
            
            # Update config with real paths
            config = sample_config.copy()
            config['path'] = str(images_dir)
            config['annotation_file'] = str(annotation_file)
            
            # Initialize loader
            loader = IcdarArTLoader(config)
            
            # Get sample and verify language defaults to 'unknown'
            sample = loader.get_item(0)
            scene_text = sample['annotations']['scene_text']
            assert len(scene_text) == 1
            assert scene_text[0]['language'] == 'unknown'