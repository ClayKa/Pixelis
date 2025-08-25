# tests/dataloaders/test_base_loader.py

import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

from core.dataloaders.base_loader import BaseLoader


class TestBaseLoaderContract:
    """Test that BaseLoader correctly enforces its abstract contract."""
    
    def test_cannot_instantiate_base_loader_directly(self):
        """Test that BaseLoader cannot be instantiated directly."""
        config = {"name": "test_dataset"}
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseLoader"):
            BaseLoader(config)
    
    def test_concrete_class_without_build_index_fails(self):
        """Test that a concrete class missing _build_index cannot be instantiated."""
        
        class IncompleteLoader(BaseLoader):
            # Missing _build_index implementation
            def get_item(self, index: int) -> Dict[str, Any]:
                return {}
        
        config = {"name": "test_dataset"}
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteLoader"):
            IncompleteLoader(config)
    
    def test_concrete_class_without_get_item_fails(self):
        """Test that a concrete class missing get_item cannot be instantiated."""
        
        class IncompleteLoader(BaseLoader):
            def _build_index(self) -> List[Any]:
                return ["item1", "item2"]
            # Missing get_item implementation
        
        config = {"name": "test_dataset"}
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteLoader"):
            IncompleteLoader(config)


class TestBaseLoaderValidation:
    """Test BaseLoader validation and helper methods."""
    
    def test_config_must_contain_name_key(self):
        """Test that config must contain 'name' key."""
        
        class ValidLoader(BaseLoader):
            def _build_index(self) -> List[Any]:
                return ["item1", "item2"]
            
            def get_item(self, index: int) -> Dict[str, Any]:
                return {}
        
        # Config without 'name' key should raise ValueError
        config_without_name = {"path": "/some/path"}
        
        with pytest.raises(ValueError, match="Loader configuration must include a 'name' key"):
            ValidLoader(config_without_name)
    
    @patch('core.dataloaders.base_loader.logger')
    def test_successful_initialization_prints_message(self, mock_logger):
        """Test that successful initialization logs a status message."""
        
        class ValidLoader(BaseLoader):
            def _build_index(self) -> List[Any]:
                return ["item1", "item2", "item3"]
            
            def get_item(self, index: int) -> Dict[str, Any]:
                return {}
        
        config = {"name": "test_dataset"}
        loader = ValidLoader(config)
        
        # Check that initialization message was logged
        mock_logger.info.assert_called_once_with(
            "Loader for 'test_dataset' initialized successfully. Found 3 samples."
        )
        
        # Check that attributes are set correctly
        assert loader.config == config
        assert loader.source_name == "test_dataset"
        assert len(loader) == 3


class TestStandardizedBaseHelper:
    """Test the _get_standardized_base helper method."""
    
    def setup_method(self):
        """Set up a valid loader for testing."""
        
        class ValidLoader(BaseLoader):
            def _build_index(self) -> List[Any]:
                return ["item1"]
            
            def get_item(self, index: int) -> Dict[str, Any]:
                return {}
        
        config = {"name": "test_dataset"}
        with patch('core.dataloaders.base_loader.logger'):  # Suppress initialization log
            self.loader = ValidLoader(config)
    
    def test_standardized_base_with_valid_image_file(self):
        """Test _get_standardized_base with a valid image file."""
        
        # Create a temporary file to simulate an existing media file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
        try:
            # Call _get_standardized_base with valid parameters
            result = self.loader._get_standardized_base(
                sample_id="test_123",
                media_path=temp_path,
                media_type="image"
            )
            
            # Verify the structure and contents
            expected_structure = {
                "source_dataset": "test_dataset",
                "sample_id": "test_123",
                "media_type": "image",
                "media_path": str(temp_path.resolve()),
                "width": None,  # None because temp file is not a valid image
                "height": None,  # None because temp file is not a valid image
                "annotations": {}
            }
            
            assert result == expected_structure
            
        finally:
            # Clean up the temporary file
            temp_path.unlink(missing_ok=True)
    
    def test_standardized_base_with_valid_video_file(self):
        """Test _get_standardized_base with a valid video file."""
        
        # Create a temporary file to simulate an existing media file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
        try:
            # Call _get_standardized_base with valid parameters
            result = self.loader._get_standardized_base(
                sample_id="video_456",
                media_path=temp_path,
                media_type="video"
            )
            
            # Verify the structure and contents
            expected_structure = {
                "source_dataset": "test_dataset",
                "sample_id": "video_456",
                "media_type": "video",
                "media_path": str(temp_path.resolve()),
                "width": None,  # None for videos (not implemented yet)
                "height": None,  # None for videos (not implemented yet)
                "annotations": {}
            }
            
            assert result == expected_structure
            
        finally:
            # Clean up the temporary file
            temp_path.unlink(missing_ok=True)
    
    def test_standardized_base_with_real_image_file(self):
        """Test _get_standardized_base with a real image file to verify dimension extraction."""
        from PIL import Image
        
        # Create a real image file with known dimensions
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            # Create a simple 100x50 image
            img = Image.new('RGB', (100, 50), color='red')
            img.save(temp_path)
            
        try:
            # Call _get_standardized_base with the real image
            result = self.loader._get_standardized_base(
                sample_id="test_real_img",
                media_path=temp_path,
                media_type="image"
            )
            
            # Verify the structure and contents including dimensions
            assert result["source_dataset"] == "test_dataset"
            assert result["sample_id"] == "test_real_img"
            assert result["media_type"] == "image"
            assert result["media_path"] == str(temp_path.resolve())
            assert result["width"] == 100  # Actual width from the image
            assert result["height"] == 50  # Actual height from the image
            assert result["annotations"] == {}
            
        finally:
            # Clean up the temporary file
            temp_path.unlink(missing_ok=True)
    
    def test_standardized_base_with_nonexistent_file(self):
        """Test _get_standardized_base raises FileNotFoundError for nonexistent file."""
        
        nonexistent_path = Path("/nonexistent/path/image.jpg")
        
        with pytest.raises(FileNotFoundError, match="Media file not found for sample_id 'test_123'"):
            self.loader._get_standardized_base(
                sample_id="test_123",
                media_path=nonexistent_path,
                media_type="image"
            )
    
    def test_standardized_base_with_invalid_media_type(self):
        """Test _get_standardized_base raises ValueError for invalid media_type."""
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
        try:
            with pytest.raises(ValueError, match="media_type must be 'image' or 'video', but got 'text'"):
                self.loader._get_standardized_base(
                    sample_id="test_123",
                    media_path=temp_path,
                    media_type="text"
                )
        finally:
            # Clean up the temporary file
            temp_path.unlink(missing_ok=True)


class TestConcreteLoaderImplementation:
    """Test a complete concrete implementation of BaseLoader."""
    
    @patch('builtins.print')
    def test_complete_concrete_loader_works(self, mock_print):
        """Test that a complete concrete loader implementation works correctly."""
        
        class TestLoader(BaseLoader):
            def _build_index(self) -> List[Any]:
                return [
                    {"id": "sample_1", "filename": "image1.jpg"},
                    {"id": "sample_2", "filename": "image2.jpg"},
                    {"id": "sample_3", "filename": "image3.jpg"}
                ]
            
            def get_item(self, index: int) -> Dict[str, Any]:
                item = self._index[index]
                
                # Create a temporary file to simulate media file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                try:
                    # Use the helper to create standardized base
                    base_sample = self._get_standardized_base(
                        sample_id=item["id"],
                        media_path=temp_path,
                        media_type="image"
                    )
                    
                    # Add custom annotations
                    base_sample["annotations"]["custom_field"] = f"Custom data for {item['id']}"
                    
                    return base_sample
                    
                finally:
                    # Clean up
                    temp_path.unlink(missing_ok=True)
        
        # Test initialization
        config = {"name": "test_complete_loader"}
        loader = TestLoader(config)
        
        # Test length
        assert len(loader) == 3
        
        # Test get_item
        sample = loader.get_item(0)
        
        # Verify structure
        assert sample["source_dataset"] == "test_complete_loader"
        assert sample["sample_id"] == "sample_1"
        assert sample["media_type"] == "image"
        assert "media_path" in sample
        assert sample["annotations"]["custom_field"] == "Custom data for sample_1"


class TestBaseLoaderInit:
    """Test the initialization file for the dataloaders module."""
    
    def test_base_loader_import(self):
        """Test that BaseLoader can be imported from the module."""
        from core.dataloaders import BaseLoader as ImportedBaseLoader
        
        # Verify it's the same class
        assert ImportedBaseLoader is BaseLoader