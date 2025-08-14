# core.modules.operations

## Classes

### class `BaseOperation`

Abstract base class for all visual operations.

All specific operations must inherit from this class and implement
the run method to ensure a consistent interface.

#### Methods

##### `__init__(self)`

Initialize the base operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `__str__(self) -> str`

Human-readable string representation.

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with their default values.

Override in subclasses to specify optional parameters.

Returns:
    Dictionary mapping parameter names to default values

##### `get_required_params(self) -> List[str]`

Get list of required parameters for this operation.

Override in subclasses to specify required parameters.

Returns:
    List of required parameter names

##### `postprocess(self, result: Any) -> Any`

Postprocess the operation result.

Override in subclasses if postprocessing is needed.

Args:
    result: Raw operation result
    
Returns:
    Postprocessed result

##### `preprocess(self, **kwargs) -> Dict[str, Any]`

Preprocess inputs before execution.

Override in subclasses if preprocessing is needed.

Args:
    **kwargs: Raw input arguments
    
Returns:
    Preprocessed arguments

##### `run(self, **kwargs) -> Any`

Execute the operation.

Args:
    **kwargs: Operation-specific arguments
    
Returns:
    Operation result (format depends on specific operation)

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments for the operation.

Override in subclasses to provide specific validation.

Args:
    **kwargs: Operation-specific arguments
    
Returns:
    True if inputs are valid, False otherwise

---

### class `GetPropertiesOperation`

Extracts visual properties of an object or region in an image.

This operation analyzes a specified object or region and returns
various visual properties such as color, texture, shape, size,
and spatial relationships.

#### Methods

##### `__init__(self)`

Initialize the get properties operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with defaults.

##### `get_required_params(self) -> List[str]`

Get list of required parameters.

##### `run(self, **kwargs) -> Dict[str, Any]`

Execute the property extraction operation.

Args:
    image: Image tensor (CHW or BCHW format)
    mask: Optional binary mask defining the object
    bbox: Optional [x1, y1, x2, y2] bounding box
    properties: List of specific properties to extract
    
Returns:
    Dictionary containing various object properties:
        - color: Dominant colors and color statistics
        - texture: Texture descriptors
        - shape: Shape characteristics
        - size: Size metrics (area, dimensions)
        - position: Spatial location and relationships
        - appearance: General appearance features

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments.

Required:
    - image: Image tensor or array
    - Either 'mask' or 'bbox' to specify the object/region
    
Returns:
    True if inputs are valid, False otherwise

---

### class `ReadTextOperation`

Reads and extracts text from a specified region in an image.

This operation performs OCR on either the entire image or a specified
region, returning the extracted text along with confidence scores and
bounding boxes for detected text regions.

#### Methods

##### `__init__(self)`

Initialize the read text operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with defaults.

##### `get_required_params(self) -> List[str]`

Get list of required parameters.

##### `postprocess(self, result: Any) -> Any`

Postprocess the operation result.

Clean up text and format output.

Args:
    result: Raw operation result
    
Returns:
    Postprocessed result

##### `preprocess(self, **kwargs) -> Dict[str, Any]`

Preprocess inputs.

Converts image to appropriate format and validates region bounds.

Args:
    **kwargs: Raw input arguments
    
Returns:
    Preprocessed arguments

##### `run(self, **kwargs) -> Dict[str, Any]`

Execute the OCR operation.

Args:
    image: Image tensor (CHW or BCHW format)
    region: Optional [x1, y1, x2, y2] bounding box for specific region
    language: Optional language code (default: 'en')
    return_confidence: Whether to return confidence scores
    return_boxes: Whether to return text bounding boxes
    
Returns:
    Dictionary containing:
        - text: Extracted text string
        - lines: List of detected text lines
        - words: List of detected words with positions
        - confidence: Overall confidence score
        - boxes: Bounding boxes for each text element (if requested)
        - language: Detected or specified language

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments.

Required:
    - image: Image tensor or array
    
Optional:
    - region: [x1, y1, x2, y2] bounding box for specific region
    - language: Language code for OCR
    
Returns:
    True if inputs are valid, False otherwise

---

### class `SegmentObjectOperation`

Segments an object at a specified pixel location in an image.

This operation takes a point (x, y) in pixel coordinates and returns
a segmentation mask for the object at that location, along with
metadata about the segmented object.

#### Methods

##### `__init__(self)`

Initialize the segment object operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with defaults.

##### `get_required_params(self) -> List[str]`

Get list of required parameters.

##### `preprocess(self, **kwargs) -> Dict[str, Any]`

Preprocess inputs.

Converts image to appropriate format and validates point coordinates.

Args:
    **kwargs: Raw input arguments
    
Returns:
    Preprocessed arguments

##### `run(self, **kwargs) -> Dict[str, Any]`

Execute the segmentation operation.

Args:
    image: Image tensor (CHW or BCHW format)
    point: (x, y) tuple specifying the pixel location
    threshold: Optional confidence threshold for segmentation
    return_scores: Whether to return confidence scores
    
Returns:
    Dictionary containing:
        - mask: Binary segmentation mask (same spatial size as input)
        - bbox: Bounding box of the segmented object [x1, y1, x2, y2]
        - area: Area of the segmented region in pixels
        - confidence: Confidence score of the segmentation
        - object_id: Unique identifier for the segmented object

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments.

Required:
    - image: Image tensor or array
    - point: (x, y) tuple specifying the pixel location
    
Returns:
    True if inputs are valid, False otherwise

---

### class `TrackObjectOperation`

Tracks an object across multiple frames in a video sequence.

This operation performs object tracking given an initial object specification
(mask or bounding box) and tracks it through subsequent frames, returning
trajectories and tracking confidence.

#### Methods

##### `__init__(self)`

Initialize the track object operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `get_active_tracks(self) -> List[str]`

Get list of active track IDs.

Returns:
    List of active track IDs

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with defaults.

##### `get_required_params(self) -> List[str]`

Get list of required parameters.

##### `reset_all_tracks(self)`

Reset all tracking states.

##### `reset_track(self, track_id: str) -> bool`

Reset a specific track.

Args:
    track_id: Track to reset
    
Returns:
    True if track was reset, False if not found

##### `run(self, **kwargs) -> Dict[str, Any]`

Execute the tracking operation.

Args:
    For initialization:
        frames: List of video frames or single frame
        init_mask: Initial object mask
        init_bbox: Initial bounding box [x1, y1, x2, y2]
        track_id: Optional ID for this track
        
    For update:
        frame: New frame to process
        track_id: ID of track to update
        
    Common:
        max_frames: Maximum number of frames to process
        confidence_threshold: Minimum confidence to continue tracking
        
Returns:
    Dictionary containing:
        - track_id: Unique identifier for this track
        - trajectory: List of positions/boxes across frames
        - masks: List of segmentation masks (if available)
        - confidences: Tracking confidence for each frame
        - status: 'active', 'lost', or 'completed'
        - statistics: Motion statistics (velocity, direction, etc.)

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments.

Required for initialization:
    - frames: List of frame tensors or single frame
    - init_mask or init_bbox: Initial object specification
    
Required for update:
    - frame: New frame to process
    - track_id: ID of track to update
    
Returns:
    True if inputs are valid, False otherwise

---

### class `ZoomInOperation`

Zooms into a specific region of an image with optional enhancement.

This operation crops and potentially upscales a region of interest,
allowing for detailed examination of specific image areas.

#### Methods

##### `__init__(self)`

Initialize the zoom in operation.

##### `__repr__(self) -> str`

String representation of the operation.

##### `create_zoom_sequence(self, image: torch.Tensor, center: Tuple[int, int], zoom_levels: List[float], **kwargs) -> List[Dict[str, Any]]`

Create a sequence of zoomed images at different levels.

Args:
    image: Source image
    center: Center point for zoom
    zoom_levels: List of zoom factors
    **kwargs: Additional parameters for zoom operation
    
Returns:
    List of zoom results

##### `get_optional_params(self) -> Dict[str, Any]`

Get dictionary of optional parameters with defaults.

##### `get_required_params(self) -> List[str]`

Get list of required parameters.

##### `preprocess(self, **kwargs) -> Dict[str, Any]`

Preprocess inputs and calculate zoom region.

Args:
    **kwargs: Raw input arguments
    
Returns:
    Preprocessed arguments with calculated region

##### `run(self, **kwargs) -> Dict[str, Any]`

Execute the zoom operation.

Args:
    image: Image tensor (CHW or BCHW format)
    center: (x, y) center point for zoom
    zoom_factor: Zoom level (e.g., 2.0 for 2x zoom)
    region: Alternative [x1, y1, x2, y2] region specification
    target_size: Optional target output size (width, height)
    enhance: Whether to apply super-resolution enhancement
    maintain_aspect: Whether to maintain aspect ratio
    
Returns:
    Dictionary containing:
        - zoomed_image: The zoomed/cropped image
        - region: The actual region that was zoomed [x1, y1, x2, y2]
        - zoom_level: Effective zoom level applied
        - original_size: Original image dimensions
        - output_size: Output image dimensions
        - metadata: Additional zoom metadata

##### `validate_inputs(self, **kwargs) -> bool`

Validate input arguments.

Required:
    - image: Image tensor or array
    - Either 'center' + 'zoom_factor' or 'region'
    
Returns:
    True if inputs are valid, False otherwise

---

## Functions

### `list_operations_by_category()`

List all operations organized by category.

Returns:
    Dictionary mapping categories to operation names

---

