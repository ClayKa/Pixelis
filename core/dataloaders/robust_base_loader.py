# core/dataloaders/robust_base_loader.py

import json
import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors with detailed context."""
    
    def __init__(self, message: str, sample_id: Optional[Any] = None, 
                 file_path: Optional[Path] = None, original_error: Optional[Exception] = None):
        self.sample_id = sample_id
        self.file_path = file_path
        self.original_error = original_error
        
        # Build detailed error message
        details = [message]
        if sample_id is not None:
            details.append(f"Sample ID: {sample_id}")
        if file_path is not None:
            details.append(f"File: {file_path}")
        if original_error is not None:
            details.append(f"Original error: {type(original_error).__name__}: {str(original_error)}")
        
        super().__init__(" | ".join(details))


class RobustBaseLoader(ABC):
    """
    Enhanced base loader with comprehensive error handling and recovery mechanisms.
    
    Key features:
    - Graceful handling of corrupted files
    - Detailed error reporting with context
    - Optional skip-on-error mode for training resilience
    - Validation during index building to catch issues early
    - Recovery mechanisms for common file format issues
    """
    
    def __init__(self, config: Dict[str, Any], skip_on_error: bool = False):
        """
        Initialize the robust loader.
        
        Args:
            config: Configuration dictionary for the datasource
            skip_on_error: If True, skip corrupted samples instead of raising errors
        """
        if 'name' not in config:
            raise ValueError("Loader configuration must include a 'name' key.")
        
        self.config = config
        self.source_name = self.config['name']
        self.skip_on_error = skip_on_error
        self._corrupted_samples = []  # Track corrupted samples
        self._validation_errors = []  # Track validation errors during indexing
        
        # Build index with comprehensive error tracking
        try:
            self._index = self._build_index()
            
            # Report any issues found during indexing
            if self._validation_errors:
                logger.warning(
                    f"Found {len(self._validation_errors)} validation errors during indexing. "
                    f"First 5 errors: {self._validation_errors[:5]}"
                )
            
            if self._corrupted_samples:
                logger.warning(
                    f"Skipped {len(self._corrupted_samples)} corrupted samples. "
                    f"First 5: {self._corrupted_samples[:5]}"
                )
            
            logger.info(
                f"Loader for '{self.source_name}' initialized successfully. "
                f"Found {len(self)} valid samples (skipped {len(self._corrupted_samples)} corrupted)."
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize loader for '{self.source_name}': {e}")
            raise DataLoadError(
                f"Loader initialization failed for dataset '{self.source_name}'",
                file_path=Path(config.get('path', 'unknown')),
                original_error=e
            )
    
    @abstractmethod
    def _build_index(self) -> List[Any]:
        """Build index with error handling. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self._index)
    
    @abstractmethod
    def get_item(self, index: int) -> Dict[str, Any]:
        """Get a single item with error handling. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def safe_get_item(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Safely get an item, returning None if there's an error.
        
        This method is useful for training loops that need to continue
        even when individual samples are corrupted.
        """
        try:
            return self.get_item(index)
        except Exception as e:
            logger.error(f"Failed to load sample at index {index}: {e}")
            if self.skip_on_error:
                return None
            raise
    
    def _load_json_safe(self, file_path: Path, required_fields: Optional[List[str]] = None) -> Dict:
        """
        Safely load and validate a JSON file with comprehensive error handling.
        
        Args:
            file_path: Path to the JSON file
            required_fields: Optional list of required top-level fields
            
        Returns:
            Parsed JSON data as dictionary
            
        Raises:
            DataLoadError: If file cannot be loaded or validated
        """
        if not file_path.exists():
            raise DataLoadError(
                f"JSON file not found",
                file_path=file_path
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        except json.JSONDecodeError as e:
            # Try to provide helpful context about where the error occurred
            raise DataLoadError(
                f"Invalid JSON format at line {e.lineno}, column {e.colno}",
                file_path=file_path,
                original_error=e
            )
        except UnicodeDecodeError as e:
            # Try different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    data = json.load(f)
                logger.warning(f"Had to use latin-1 encoding for {file_path}")
            except Exception:
                raise DataLoadError(
                    f"Unable to decode file with UTF-8 or latin-1 encoding",
                    file_path=file_path,
                    original_error=e
                )
        except MemoryError as e:
            raise DataLoadError(
                f"File too large to load into memory. Consider using streaming parser.",
                file_path=file_path,
                original_error=e
            )
        except Exception as e:
            raise DataLoadError(
                f"Unexpected error loading JSON file",
                file_path=file_path,
                original_error=e
            )
        
        # Validate required fields if specified
        if required_fields:
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise DataLoadError(
                    f"JSON missing required fields: {missing_fields}",
                    file_path=file_path
                )
        
        return data
    
    def _load_csv_safe(self, file_path: Path, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Safely load and validate a CSV file.
        
        Args:
            file_path: Path to the CSV file
            required_columns: Optional list of required column names
            
        Returns:
            Parsed CSV data as DataFrame
            
        Raises:
            DataLoadError: If file cannot be loaded or validated
        """
        if not file_path.exists():
            raise DataLoadError(
                f"CSV file not found",
                file_path=file_path
            )
        
        try:
            # Try to read with different error handling strategies
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                # Try with different encoding
                df = pd.read_csv(file_path, encoding='latin-1')
                logger.warning(f"Had to use latin-1 encoding for {file_path}")
            
        except pd.errors.EmptyDataError as e:
            raise DataLoadError(
                f"CSV file is empty",
                file_path=file_path,
                original_error=e
            )
        except pd.errors.ParserError as e:
            raise DataLoadError(
                f"CSV parsing error - file may be corrupted or have inconsistent formatting",
                file_path=file_path,
                original_error=e
            )
        except MemoryError as e:
            # Try reading in chunks
            logger.warning(f"CSV too large for memory, attempting chunked reading for {file_path}")
            try:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            except Exception:
                raise DataLoadError(
                    f"File too large to load even with chunking",
                    file_path=file_path,
                    original_error=e
                )
        except Exception as e:
            raise DataLoadError(
                f"Unexpected error loading CSV file",
                file_path=file_path,
                original_error=e
            )
        
        # Validate required columns if specified
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataLoadError(
                    f"CSV missing required columns: {missing_columns}. Available columns: {list(df.columns)}",
                    file_path=file_path
                )
        
        return df
    
    def _validate_media_file(self, media_path: Path, media_type: str, 
                           sample_id: Optional[Any] = None) -> bool:
        """
        Validate that a media file exists and is accessible.
        
        Args:
            media_path: Path to the media file
            media_type: Type of media ('image' or 'video')
            sample_id: Optional sample identifier for error reporting
            
        Returns:
            True if file is valid, False otherwise
        """
        if not media_path.exists():
            self._validation_errors.append({
                'type': 'missing_file',
                'path': str(media_path),
                'sample_id': sample_id
            })
            return False
        
        if not media_path.is_file():
            self._validation_errors.append({
                'type': 'not_a_file',
                'path': str(media_path),
                'sample_id': sample_id
            })
            return False
        
        # For images, try to open and validate
        if media_type == "image":
            try:
                with Image.open(media_path) as img:
                    img.verify()  # Verify it's a valid image
                return True
            except Exception as e:
                self._validation_errors.append({
                    'type': 'corrupted_image',
                    'path': str(media_path),
                    'sample_id': sample_id,
                    'error': str(e)
                })
                return False
        
        # For videos, check file size and extension
        elif media_type == "video":
            if media_path.stat().st_size == 0:
                self._validation_errors.append({
                    'type': 'empty_video',
                    'path': str(media_path),
                    'sample_id': sample_id
                })
                return False
            
            valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
            if media_path.suffix.lower() not in valid_extensions:
                self._validation_errors.append({
                    'type': 'invalid_video_extension',
                    'path': str(media_path),
                    'sample_id': sample_id,
                    'extension': media_path.suffix
                })
                return False
            
            return True
        
        return True
    
    def _get_standardized_base_safe(self, sample_id: Any, media_path: Path, 
                                   media_type: str) -> Dict[str, Any]:
        """
        Safe version of _get_standardized_base with enhanced error handling.
        
        Args:
            sample_id: Unique identifier for the sample
            media_path: Path to the media file
            media_type: Type of media ('image' or 'video')
            
        Returns:
            Standardized base dictionary
            
        Raises:
            DataLoadError: If media file is invalid or inaccessible
        """
        if not media_path.is_file():
            raise DataLoadError(
                f"Media file not found or inaccessible",
                sample_id=sample_id,
                file_path=media_path
            )
        
        if media_type not in ["image", "video"]:
            raise ValueError(f"media_type must be 'image' or 'video', but got '{media_type}'.")
        
        width, height = None, None
        
        # Extract dimensions based on media type with error handling
        if media_type == "image":
            try:
                with Image.open(media_path) as img:
                    width, height = img.size
                    # Also check if image is corrupted
                    img.verify()
            except Exception as e:
                if self.skip_on_error:
                    logger.warning(
                        f"Could not read image dimensions for '{media_path.name}' "
                        f"in source '{self.source_name}'. Error: {e}"
                    )
                else:
                    raise DataLoadError(
                        f"Failed to read or verify image file",
                        sample_id=sample_id,
                        file_path=media_path,
                        original_error=e
                    )
        
        return {
            "source_dataset": self.source_name,
            "sample_id": sample_id,
            "media_type": media_type,
            "media_path": str(media_path.resolve()),
            "width": width,
            "height": height,
            "annotations": {}
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered during loading.
        
        Returns:
            Dictionary with error statistics and details
        """
        error_types = {}
        for error in self._validation_errors:
            error_type = error.get('type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self._validation_errors),
            'corrupted_samples': len(self._corrupted_samples),
            'error_types': error_types,
            'sample_errors': self._validation_errors[:10],  # First 10 errors as examples
            'corrupted_sample_ids': self._corrupted_samples[:10]  # First 10 corrupted
        }