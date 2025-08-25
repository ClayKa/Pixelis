# tests/dataloaders/test_hiertext_loader.py

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from core.dataloaders.hiertext_loader import HierTextLoader, HierTextStreamingLoader


class TestHierTextLoaderInitialization:
    """Test HierTextLoader initialization and configuration validation."""
    
    def test_missing_required_config_keys(self):
        """Test that missing required config keys raise ValueError."""
        # Test missing image_path
        config = {
            "name": "hiertext_test",
            "annotation_file": "/path/to/annotations.json"
        }
        with pytest.raises(ValueError, match="HierTextLoader config must include 'image_path'"):
            HierTextLoader(config)
        
        # Test missing annotation_file
        config = {
            "name": "hiertext_test",
            "image_path": "/path/to/images"
        }
        with pytest.raises(ValueError, match="HierTextLoader config must include 'annotation_file'"):
            HierTextLoader(config)
    
    def test_nonexistent_paths_raise_error(self):
        """Test that nonexistent paths raise FileNotFoundError."""
        config = {
            "name": "hiertext_test",
            "image_path": "/nonexistent/images",
            "annotation_file": "/nonexistent/annotations.json"
        }
        
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            HierTextLoader(config)


class TestHierTextLoaderWithMockData:
    """Test HierTextLoader with mock data."""
    
    @pytest.fixture
    def mock_hiertext_data(self, tmp_path):
        """Create mock HierText dataset structure."""
        # Create directory structure
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        
        # Create mock annotation data in HierText format
        annotation_data = {
            "info": {
                "date": "2022-03-16",
                "version": "v1.0"
            },
            "annotations": [
                {
                    "image_id": "test_image_001",
                    "paragraphs": [
                        {
                            "vertices": [[10, 10], [200, 10], [200, 50], [10, 50]],
                            "legible": True,
                            "lines": [
                                {
                                    "vertices": [[10, 10], [200, 10], [200, 30], [10, 30]],
                                    "text": "Hello World",
                                    "legible": True,
                                    "words": [
                                        {
                                            "vertices": [[10, 10], [80, 10], [80, 30], [10, 30]],
                                            "text": "Hello",
                                            "legible": True
                                        },
                                        {
                                            "vertices": [[90, 10], [200, 10], [200, 30], [90, 30]],
                                            "text": "World",
                                            "legible": True
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "image_id": "test_image_002",
                    "paragraphs": [
                        {
                            "vertices": [[20, 20], [300, 20], [300, 100], [20, 100]],
                            "legible": True,
                            "lines": [
                                {
                                    "vertices": [[20, 20], [300, 20], [300, 40], [20, 40]],
                                    "text": "First line of text",
                                    "legible": True,
                                    "words": [
                                        {
                                            "vertices": [[20, 20], [70, 20], [70, 40], [20, 40]],
                                            "text": "First",
                                            "legible": True
                                        },
                                        {
                                            "vertices": [[80, 20], [120, 20], [120, 40], [80, 40]],
                                            "text": "line",
                                            "legible": True
                                        },
                                        {
                                            "vertices": [[130, 20], [160, 20], [160, 40], [130, 40]],
                                            "text": "of",
                                            "legible": True
                                        },
                                        {
                                            "vertices": [[170, 20], [300, 20], [300, 40], [170, 40]],
                                            "text": "text",
                                            "legible": True
                                        }
                                    ]
                                },
                                {
                                    "vertices": [[20, 50], [250, 50], [250, 70], [20, 70]],
                                    "text": "Second line",
                                    "legible": True,
                                    "words": [
                                        {
                                            "vertices": [[20, 50], [100, 50], [100, 70], [20, 70]],
                                            "text": "Second",
                                            "legible": True
                                        },
                                        {
                                            "vertices": [[110, 50], [250, 50], [250, 70], [110, 70]],
                                            "text": "line",
                                            "legible": True
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "image_id": "test_image_003",
                    "paragraphs": []  # Empty paragraphs
                }
            ]
        }
        
        # Write annotation file
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create mock images
        (image_dir / "test_image_001.jpg").touch()
        (image_dir / "test_image_002.jpg").touch()
        (image_dir / "test_image_003.jpg").touch()
        
        return {
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file),
            "annotation_data": annotation_data
        }
    
    def test_successful_initialization(self, mock_hiertext_data):
        """Test successful initialization with valid paths and data."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Check that loader was initialized correctly
        assert loader.source_name == "hiertext_test"
        assert len(loader) == 3  # Three annotations
        assert len(loader._id_to_annotation) == 3  # Three cached annotations
    
    def test_build_index(self, mock_hiertext_data):
        """Test that _build_index correctly loads annotation data."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Check index content
        assert len(loader._index) == 3
        assert loader._index[0]["image_id"] == "test_image_001"
        assert loader._index[0]["has_paragraphs"] == True
        assert loader._index[2]["image_id"] == "test_image_003"
        assert loader._index[2]["has_paragraphs"] == False
    
    def test_get_item_with_hierarchical_text(self, mock_hiertext_data):
        """Test get_item returns properly formatted sample with hierarchical text."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Get first sample
        sample = loader.get_item(0)
        
        # Check basic structure
        assert sample["source_dataset"] == "hiertext_test"
        assert sample["sample_id"] == "test_image_001"
        assert sample["media_type"] == "image"
        assert sample["media_path"].endswith("test_image_001.jpg")
        
        # Check hierarchical annotations
        assert "hierarchical_text" in sample["annotations"]
        assert len(sample["annotations"]["hierarchical_text"]) == 1  # One paragraph
        
        # Check flat word list
        assert "flat_word_list" in sample["annotations"]
        assert len(sample["annotations"]["flat_word_list"]) == 2  # Two words
        assert sample["annotations"]["flat_word_list"][0]["text"] == "Hello"
        assert sample["annotations"]["flat_word_list"][1]["text"] == "World"
        
        # Check word metadata
        first_word = sample["annotations"]["flat_word_list"][0]
        assert first_word["paragraph_idx"] == 0
        assert first_word["line_idx"] == 0
        assert first_word["word_idx"] == 0
        assert first_word["legible"] == True
        assert "bbox" in first_word  # Should have computed bbox
        assert first_word["bbox"] == [10, 10, 80, 30]
        
        # Check summary statistics
        assert sample["annotations"]["num_paragraphs"] == 1
        assert sample["annotations"]["num_words"] == 2
        
        # Check full text extraction
        assert sample["annotations"]["full_text"] == "Hello World"
    
    def test_get_item_multi_line_paragraph(self, mock_hiertext_data):
        """Test get_item for sample with multiple lines."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Get second sample
        sample = loader.get_item(1)
        
        # Check basic structure
        assert sample["sample_id"] == "test_image_002"
        
        # Check flat word list
        assert len(sample["annotations"]["flat_word_list"]) == 6  # Total words
        
        # Check word from second line
        last_word = sample["annotations"]["flat_word_list"][-1]
        assert last_word["text"] == "line"
        assert last_word["line_idx"] == 1  # Second line
        assert last_word["line_text"] == "Second line"
        
        # Check full text extraction (should join lines with space)
        assert sample["annotations"]["full_text"] == "First line of text Second line"
        
        # Check statistics
        assert sample["annotations"]["num_words"] == 6
    
    def test_get_item_empty_paragraphs(self, mock_hiertext_data):
        """Test get_item for sample with no paragraphs."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Get third sample (empty)
        sample = loader.get_item(2)
        
        # Check basic structure
        assert sample["sample_id"] == "test_image_003"
        
        # Check empty annotations
        assert len(sample["annotations"]["hierarchical_text"]) == 0
        assert len(sample["annotations"]["flat_word_list"]) == 0
        assert sample["annotations"]["num_paragraphs"] == 0
        assert sample["annotations"]["num_words"] == 0
        assert sample["annotations"]["full_text"] == ""
    
    def test_vertices_to_bbox_conversion(self, mock_hiertext_data):
        """Test the vertices to bbox conversion utility."""
        config = {
            "name": "hiertext_test",
            "image_path": mock_hiertext_data["image_path"],
            "annotation_file": mock_hiertext_data["annotation_file"]
        }
        
        loader = HierTextLoader(config)
        
        # Test valid vertices
        vertices = [[10, 20], [100, 20], [100, 50], [10, 50]]
        bbox = loader._vertices_to_bbox(vertices)
        assert bbox == [10, 20, 100, 50]
        
        # Test invalid vertices
        assert loader._vertices_to_bbox([]) is None
        assert loader._vertices_to_bbox([[10]]) is None
        assert loader._vertices_to_bbox([[10, 20]]) is None  # Need at least 2 points


class TestHierTextLoaderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_image_id_in_annotation(self, tmp_path):
        """Test handling of annotation without image_id."""
        # Create minimal setup
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "info": {"date": "2022-03-16", "version": "v1.0"},
            "annotations": [
                {
                    # Missing 'image_id'
                    "paragraphs": []
                },
                {
                    "image_id": "valid_image",
                    "paragraphs": []
                }
            ]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        (image_dir / "valid_image.jpg").touch()
        
        config = {
            "name": "hiertext_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file)
        }
        
        loader = HierTextLoader(config)
        
        # Should only have one valid annotation
        assert len(loader) == 1
        assert loader._index[0]["image_id"] == "valid_image"
    
    def test_malformed_json_file(self, tmp_path):
        """Test handling of malformed JSON file."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        
        # Write malformed JSON
        with open(annotation_file, 'w') as f:
            f.write('{"info": {"version": "v1.0"}, "annotations": [}')  # Invalid JSON
        
        config = {
            "name": "hiertext_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file)
        }
        
        with pytest.raises(json.JSONDecodeError):
            HierTextLoader(config)
    
    def test_missing_annotations_field(self, tmp_path):
        """Test handling when JSON lacks 'annotations' field."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "info": {"date": "2022-03-16", "version": "v1.0"}
            # Missing 'annotations' field
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        config = {
            "name": "hiertext_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file)
        }
        
        with pytest.raises(ValueError, match="missing 'annotations' field"):
            HierTextLoader(config)


class TestHierTextStreamingLoader:
    """Test the streaming loader variant."""
    
    @patch('core.dataloaders.hiertext_loader.HierTextStreamingLoader._build_index')
    def test_streaming_loader_fallback_without_ijson(self, mock_build_index, tmp_path):
        """Test that streaming loader falls back when ijson is not available."""
        # Create minimal valid setup
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "info": {"date": "2022-03-16", "version": "v1.0"},
            "annotations": []
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        config = {
            "name": "hiertext_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file)
        }
        
        # Mock the parent's _build_index to be called
        mock_build_index.return_value = []
        
        # Create streaming loader (should fall back if ijson not installed)
        loader = HierTextStreamingLoader(config)
        
        # Verify it was called
        mock_build_index.assert_called_once()