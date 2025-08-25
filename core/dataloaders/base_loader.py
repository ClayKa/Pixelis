# core/dataloaders/base_loader.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

class BaseLoader(ABC):
    """
    [The Constitution]
    An abstract base class that defines the unified interface for all dataset loaders.

    Its core responsibility is to "adapt" a unique, heterogeneous raw dataset into an
    iterable-like object that provides standardized samples for downstream consumers.
    This class enforces a contract for all concrete loader implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the loader by storing the config and triggering the index build.
        This constructor should be called by all subclasses via `super().__init__(config)`.

        Args:
            config (Dict[str, Any]): The configuration dictionary for this specific
                                     datasource, as loaded from the project's manifest YAML file.
                                     It must contain a 'name' key and all necessary path information.
        """
        if 'name' not in config:
            raise ValueError("Loader configuration must include a 'name' key.")
        
        self.config = config
        self.source_name = self.config['name']
        
        # _build_index() is the core implementation method in each subclass.
        # It is responsible for populating `self._index` with a lightweight list
        # of "pointers" to the individual samples within the dataset.
        self._index: List[Any] = self._build_index()
        logger.info(f"Loader for '{self.source_name}' initialized successfully. Found {len(self)} samples.")

    @abstractmethod
    def _build_index(self) -> List[Any]:
        """
        [Subclass Must Implement]
        Scans all necessary source files and builds a lightweight index of all available samples.

        This method is called only once during the loader's initialization. It is designed
        to handle all one-time, potentially expensive setup operations, such as loading a large
        central annotation file (e.g., a multi-gigabyte JSON) into memory or scanning a directory
        containing thousands of files.

        The returned index should be a list where each element contains the minimum information
        required to uniquely identify and later retrieve a single sample. This could be an
        image ID, a filename, a row index in a DataFrame, or a custom object.

        Returns:
            List[Any]: A list of unique "pointers" to every sample in the dataset.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns the total number of samples found in the dataset.
        This method is called to get the size of the dataset.
        """
        return len(self._index)

    @abstractmethod
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        [Subclass Must Implement]
        Retrieves, parses, and formats a single, standardized data sample dictionary
        identified by its integer index.

        This is the core method for "translating" a raw, dataset-specific format into
        the unified, standardized dictionary format that the rest of our system understands.
        Heavy I/O operations (like reading an image file from disk or parsing a small,
        specific annotation file) should occur here in a "lazy-loading" fashion.

        Args:
            index (int): The integer index of the sample in the `self._index` list.

        Returns:
            Dict[str, Any]: A standardized sample dictionary that must conform to the
                            project's defined "Output Contract".
        """
        raise NotImplementedError

    def _get_standardized_base(self, sample_id: Any, media_path: Path, media_type: str) -> Dict[str, Any]:
        """
        A concrete helper utility for subclasses to create the base structure of the
        standardized sample dictionary.

        This enforces consistency across all loaders for the common, top-level fields.
        It also performs a critical file existence check.

        Args:
            sample_id (Any): The unique identifier of the sample within its source dataset.
            media_path (Path): A `pathlib.Path` object pointing to the primary media file (image/video).
            media_type (str): The type of the media, must be either "image" or "video".

        Returns:
            Dict[str, Any]: A dictionary with the basic standardized structure.
        """
        if not media_path.is_file():
            # This is a crucial robustness check.
            raise FileNotFoundError(
                f"Media file not found for sample_id '{sample_id}' in source '{self.source_name}'. "
                f"Checked path: {media_path}"
            )
        
        if media_type not in ["image", "video"]:
            raise ValueError(f"media_type must be 'image' or 'video', but got '{media_type}'.")
        
        width, height = None, None
        
        # Extract dimensions based on media type
        if media_type == "image":
            try:
                with Image.open(media_path) as img:
                    width, height = img.size
            except Exception as e:
                # Log a warning if dimensions cannot be read, but don't crash
                logger.warning(
                    f"Could not read dimensions for image '{media_path.name}' "
                    f"in source '{self.source_name}'. Error: {e}"
                )
        elif media_type == "video":
            # Placeholder for future video dimension extraction logic (e.g., using OpenCV)
            # For now, we can leave them as None.
            pass
            
        return {
            "source_dataset": self.source_name,
            "sample_id": sample_id,
            "media_type": media_type,
            "media_path": str(media_path.resolve()),  # Always use absolute paths for consistency
            "width": width,   # NEW: image/video width
            "height": height, # NEW: image/video height
            "annotations": {}  # Initialize an empty dict for annotations
        }