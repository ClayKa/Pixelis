# tests/dataloaders/test_docvqa_loader.py

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any

from core.dataloaders.docvqa_loader import DocVqaLoader


class TestDocVqaLoaderInitialization:
    """Test DocVqaLoader initialization and configuration validation."""
    
    def test_missing_required_config_keys(self):
        """Test that missing required config keys raise ValueError."""
        # Test missing image_path
        config = {
            "name": "docvqa_test",
            "annotation_file": "/path/to/annotations.json",
            "ocr_path": "/path/to/ocr"
        }
        with pytest.raises(ValueError, match="DocVqaLoader config must include 'image_path'"):
            DocVqaLoader(config)
        
        # Test missing annotation_file
        config = {
            "name": "docvqa_test",
            "image_path": "/path/to/images",
            "ocr_path": "/path/to/ocr"
        }
        with pytest.raises(ValueError, match="DocVqaLoader config must include 'annotation_file'"):
            DocVqaLoader(config)
        
        # Test missing ocr_path
        config = {
            "name": "docvqa_test",
            "image_path": "/path/to/images",
            "annotation_file": "/path/to/annotations.json"
        }
        with pytest.raises(ValueError, match="DocVqaLoader config must include 'ocr_path'"):
            DocVqaLoader(config)
    
    def test_nonexistent_paths_raise_error(self):
        """Test that nonexistent paths raise FileNotFoundError."""
        config = {
            "name": "docvqa_test",
            "image_path": "/nonexistent/images",
            "annotation_file": "/nonexistent/annotations.json",
            "ocr_path": "/nonexistent/ocr"
        }
        
        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            DocVqaLoader(config)


class TestDocVqaLoaderWithMockData:
    """Test DocVqaLoader with mock data."""
    
    @pytest.fixture
    def mock_docvqa_data(self, tmp_path):
        """Create mock DocVQA dataset structure."""
        # Create directory structure
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        ocr_dir = tmp_path / "ocr"
        ocr_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        
        # Create mock annotation data
        annotation_data = {
            "dataset_name": "SP-DocVQA",
            "dataset_version": "1.0",
            "dataset_split": "test",
            "data": [
                {
                    "questionId": 1,
                    "question": "What is the date?",
                    "answers": ["January 1, 2024"],
                    "image": "documents/test_doc_1.png",
                    "ucsf_document_id": "test_doc",
                    "ucsf_document_page_no": "1",
                    "question_types": ["handwritten", "form"]
                },
                {
                    "questionId": 2,
                    "question": "What is the title?",
                    "answers": ["Test Document", "TEST DOCUMENT"],
                    "image": "documents/test_doc_2.png",
                    "question_types": ["layout"]
                }
            ]
        }
        
        # Write annotation file
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create mock images
        (image_dir / "test_doc_1.png").touch()
        (image_dir / "test_doc_2.png").touch()
        
        # Create mock OCR data
        ocr_data_1 = {
            "status": "Succeeded",
            "recognitionResults": [
                {
                    "page": 1,
                    "width": 1700,
                    "height": 2200,
                    "lines": [
                        {
                            "boundingBox": [100, 100, 300, 100, 300, 150, 100, 150],
                            "text": "January 1, 2024",
                            "words": [
                                {
                                    "boundingBox": [100, 100, 180, 100, 180, 150, 100, 150],
                                    "text": "January",
                                    "confidence": 0.99
                                },
                                {
                                    "boundingBox": [190, 100, 220, 100, 220, 150, 190, 150],
                                    "text": "1,",
                                    "confidence": 0.98
                                },
                                {
                                    "boundingBox": [230, 100, 300, 100, 300, 150, 230, 150],
                                    "text": "2024",
                                    "confidence": 0.99
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        ocr_data_2 = {
            "status": "Succeeded",
            "recognitionResults": [
                {
                    "page": 1,
                    "width": 1700,
                    "height": 2200,
                    "lines": [
                        {
                            "boundingBox": [200, 200, 500, 200, 500, 250, 200, 250],
                            "text": "Test Document",
                            "words": [
                                {
                                    "boundingBox": [200, 200, 300, 200, 300, 250, 200, 250],
                                    "text": "Test",
                                    "confidence": 0.95
                                },
                                {
                                    "boundingBox": [310, 200, 500, 200, 500, 250, 310, 250],
                                    "text": "Document",
                                    "confidence": 0.97
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Write OCR files
        with open(ocr_dir / "test_doc_1.json", 'w') as f:
            json.dump(ocr_data_1, f)
        
        with open(ocr_dir / "test_doc_2.json", 'w') as f:
            json.dump(ocr_data_2, f)
        
        return {
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file),
            "ocr_path": str(ocr_dir),
            "annotation_data": annotation_data,
            "ocr_data": {"test_doc_1": ocr_data_1, "test_doc_2": ocr_data_2}
        }
    
    def test_successful_initialization(self, mock_docvqa_data):
        """Test successful initialization with valid paths and data."""
        config = {
            "name": "docvqa_test",
            "image_path": mock_docvqa_data["image_path"],
            "annotation_file": mock_docvqa_data["annotation_file"],
            "ocr_path": mock_docvqa_data["ocr_path"]
        }
        
        loader = DocVqaLoader(config)
        
        # Check that loader was initialized correctly
        assert loader.source_name == "docvqa_test"
        assert len(loader) == 2  # Two QA samples
        assert len(loader._image_id_to_ocr) == 2  # Two OCR files loaded
    
    def test_build_index(self, mock_docvqa_data):
        """Test that _build_index correctly loads annotation data."""
        config = {
            "name": "docvqa_test",
            "image_path": mock_docvqa_data["image_path"],
            "annotation_file": mock_docvqa_data["annotation_file"],
            "ocr_path": mock_docvqa_data["ocr_path"]
        }
        
        loader = DocVqaLoader(config)
        
        # Check index content
        assert len(loader._index) == 2
        assert loader._index[0]["questionId"] == 1
        assert loader._index[0]["question"] == "What is the date?"
        assert loader._index[1]["questionId"] == 2
        assert loader._index[1]["question"] == "What is the title?"
    
    def test_get_item_with_ocr(self, mock_docvqa_data):
        """Test get_item returns properly formatted sample with OCR data."""
        config = {
            "name": "docvqa_test",
            "image_path": mock_docvqa_data["image_path"],
            "annotation_file": mock_docvqa_data["annotation_file"],
            "ocr_path": mock_docvqa_data["ocr_path"]
        }
        
        loader = DocVqaLoader(config)
        
        # Get first sample
        sample = loader.get_item(0)
        
        # Check basic structure
        assert sample["source_dataset"] == "docvqa_test"
        assert sample["sample_id"] == 1
        assert sample["media_type"] == "image"
        assert sample["media_path"].endswith("test_doc_1.png")
        
        # Check annotations
        assert sample["annotations"]["question"] == "What is the date?"
        assert sample["annotations"]["answers"] == ["January 1, 2024"]
        assert sample["annotations"]["question_types"] == ["handwritten", "form"]
        assert sample["annotations"]["document_id"] == "test_doc"
        assert sample["annotations"]["page_number"] == "1"
        
        # Check OCR tokens
        assert "ocr_tokens" in sample["annotations"]
        assert len(sample["annotations"]["ocr_tokens"]) == 3  # Three words
        assert sample["annotations"]["ocr_tokens"][0]["text"] == "January"
        assert sample["annotations"]["ocr_tokens"][0]["bbox"] == [100, 100, 180, 150]
    
    def test_get_item_second_sample(self, mock_docvqa_data):
        """Test get_item for second sample."""
        config = {
            "name": "docvqa_test",
            "image_path": mock_docvqa_data["image_path"],
            "annotation_file": mock_docvqa_data["annotation_file"],
            "ocr_path": mock_docvqa_data["ocr_path"]
        }
        
        loader = DocVqaLoader(config)
        
        # Get second sample
        sample = loader.get_item(1)
        
        # Check basic structure
        assert sample["sample_id"] == 2
        assert sample["media_path"].endswith("test_doc_2.png")
        
        # Check annotations
        assert sample["annotations"]["question"] == "What is the title?"
        assert sample["annotations"]["answers"] == ["Test Document", "TEST DOCUMENT"]
        assert sample["annotations"]["question_types"] == ["layout"]
        
        # Check that optional fields are not present
        assert "document_id" not in sample["annotations"]
        assert "page_number" not in sample["annotations"]
        
        # Check OCR tokens
        assert len(sample["annotations"]["ocr_tokens"]) == 2  # Two words
        assert sample["annotations"]["ocr_tokens"][0]["text"] == "Test"
        assert sample["annotations"]["ocr_tokens"][1]["text"] == "Document"
    
    def test_find_answer_bboxes(self, mock_docvqa_data):
        """Test that answer bounding boxes are correctly found in OCR tokens."""
        config = {
            "name": "docvqa_test",
            "image_path": mock_docvqa_data["image_path"],
            "annotation_file": mock_docvqa_data["annotation_file"],
            "ocr_path": mock_docvqa_data["ocr_path"]
        }
        
        loader = DocVqaLoader(config)
        
        # Test finding answer bbox for second sample
        sample = loader.get_item(1)
        
        # Check if answer bboxes were found
        if "answer_bboxes" in sample["annotations"]:
            answer_bboxes = sample["annotations"]["answer_bboxes"]
            assert len(answer_bboxes) > 0
            
            # Check first answer bbox
            first_bbox = answer_bboxes[0]
            assert first_bbox["answer"] == "Test Document"
            assert first_bbox["bbox"] == [200, 200, 500, 250]  # Combined bbox
            assert first_bbox["token_indices"] == [0, 1]


class TestDocVqaLoaderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_image_field_in_sample(self, tmp_path):
        """Test handling of sample without image field."""
        # Create minimal setup
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        ocr_dir = tmp_path / "ocr"
        ocr_dir.mkdir()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "data": [
                {
                    "questionId": 1,
                    "question": "Test question",
                    "answers": ["Test answer"]
                    # Missing 'image' field
                }
            ]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        config = {
            "name": "docvqa_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file),
            "ocr_path": str(ocr_dir)
        }
        
        loader = DocVqaLoader(config)
        
        with pytest.raises(ValueError, match="Sample at index 0 missing 'image' field"):
            loader.get_item(0)
    
    def test_missing_ocr_file(self, tmp_path):
        """Test handling when OCR file is missing."""
        # Create setup with missing OCR
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        ocr_dir = tmp_path / "ocr"
        ocr_dir.mkdir()
        
        # Create image file
        (image_dir / "test_doc.png").touch()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "data": [
                {
                    "questionId": 1,
                    "question": "Test question",
                    "answers": ["Test answer"],
                    "image": "documents/test_doc.png"
                }
            ]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        config = {
            "name": "docvqa_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file),
            "ocr_path": str(ocr_dir)
        }
        
        loader = DocVqaLoader(config)
        
        # Should work but with empty OCR tokens
        sample = loader.get_item(0)
        assert sample["annotations"]["ocr_tokens"] == []
    
    def test_malformed_ocr_data(self, tmp_path):
        """Test handling of malformed OCR data."""
        # Create setup
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        ocr_dir = tmp_path / "ocr"
        ocr_dir.mkdir()
        
        (image_dir / "test_doc.png").touch()
        
        annotation_file = tmp_path / "annotations.json"
        annotation_data = {
            "data": [
                {
                    "questionId": 1,
                    "question": "Test question",
                    "answers": ["Test answer"],
                    "image": "documents/test_doc.png"
                }
            ]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f)
        
        # Create malformed OCR file (missing recognitionResults)
        ocr_data = {"status": "Succeeded"}
        with open(ocr_dir / "test_doc.json", 'w') as f:
            json.dump(ocr_data, f)
        
        config = {
            "name": "docvqa_test",
            "image_path": str(image_dir),
            "annotation_file": str(annotation_file),
            "ocr_path": str(ocr_dir)
        }
        
        loader = DocVqaLoader(config)
        
        # Should work but with empty OCR tokens
        sample = loader.get_item(0)
        assert sample["annotations"]["ocr_tokens"] == []