# Pixelis Dataloaders

This directory contains dataloader implementations for various visual question answering and document understanding datasets.

## Architecture

All dataloaders inherit from the `BaseLoader` abstract class, which provides:
- Standardized sample format across all datasets
- Lazy loading of data
- Automatic dimension extraction for images
- Consistent error handling

## Available Dataloaders

### DocVqaLoader

The `DocVqaLoader` handles the Single Page Document VQA (SP-DocVQA) dataset, which contains questions about document images with corresponding OCR data.

**Features:**
- Loads QA annotations from JSON files
- Pre-loads and caches OCR data for efficient access
- Extracts word-level bounding boxes from OCR results
- Attempts to locate answer text within OCR tokens
- Handles missing OCR files gracefully

**Configuration:**
```yaml
datasources:
  docvqa_train:
    name: "docvqa_train"
    image_path: "/path/to/spdocvqa_images"
    annotation_file: "/path/to/train_v1.0_withQT.json"
    ocr_path: "/path/to/spdocvqa_ocr"
```

**Sample Output Format:**
```python
{
    "source_dataset": "docvqa_train",
    "sample_id": 123,
    "media_type": "image",
    "media_path": "/absolute/path/to/image.png",
    "width": 1700,
    "height": 2200,
    "annotations": {
        "question": "What is the date?",
        "answers": ["January 1, 2024"],
        "question_types": ["handwritten", "form"],
        "document_id": "doc_123",
        "page_number": "1",
        "ocr_tokens": [
            {
                "text": "January",
                "bbox": [100, 100, 180, 150],
                "page": 1,
                "confidence": 0.99
            },
            # ... more tokens
        ],
        "answer_bboxes": [
            {
                "answer": "January 1, 2024",
                "bbox": [100, 100, 300, 150],
                "token_indices": [0, 1, 2]
            }
        ]
    }
}
```

### HierTextLoader

The `HierTextLoader` handles the HierText dataset, which provides hierarchical text annotations with paragraph→line→word structure for scene text understanding.

**Features:**
- Loads a single large JSON file containing all annotations (~1GB)
- Maintains hierarchical text structure (paragraphs, lines, words)
- Provides both hierarchical and flattened word lists
- Converts polygon vertices to axis-aligned bounding boxes
- Tracks legibility information at multiple levels
- Memory-efficient streaming variant available (`HierTextStreamingLoader`)

**Configuration:**
```yaml
datasources:
  hiertext_train:
    name: "hiertext_train"
    image_path: "/path/to/HierText/images"
    annotation_file: "/path/to/HierText/annotations/train.jsonl"
```

**Sample Output Format:**
```python
{
    "source_dataset": "hiertext_train",
    "sample_id": "0006289e4f292bcd",
    "media_type": "image",
    "media_path": "/absolute/path/to/0006289e4f292bcd.jpg",
    "width": 1920,
    "height": 1080,
    "annotations": {
        "hierarchical_text": [
            {
                "vertices": [[10, 10], [200, 10], [200, 50], [10, 50]],
                "legible": true,
                "lines": [
                    {
                        "vertices": [...],
                        "text": "Hello World",
                        "legible": true,
                        "words": [
                            {
                                "vertices": [...],
                                "text": "Hello",
                                "legible": true
                            },
                            ...
                        ]
                    }
                ]
            }
        ],
        "flat_word_list": [
            {
                "text": "Hello",
                "vertices": [[10, 10], [80, 10], [80, 30], [10, 30]],
                "bbox": [10, 10, 80, 30],
                "legible": true,
                "paragraph_idx": 0,
                "line_idx": 0,
                "word_idx": 0,
                "line_text": "Hello World"
            },
            ...
        ],
        "num_paragraphs": 1,
        "num_words": 2,
        "full_text": "Hello World"
    }
}
```

**Performance Considerations:**
- The annotation file is very large (~1GB with 44M+ lines)
- Initial loading may take significant time and memory
- For production use, consider `HierTextStreamingLoader` with `ijson` library:
  ```python
  pip install ijson
  from core.dataloaders import HierTextStreamingLoader
  loader = HierTextStreamingLoader(config)
  ```
- The streaming variant uses incremental JSON parsing to reduce memory usage

## Creating a New Dataloader

To create a new dataloader:

1. Create a new file in this directory (e.g., `mydataset_loader.py`)
2. Import and inherit from `BaseLoader`
3. Implement the required abstract methods:
   - `_build_index()`: Load and index your dataset
   - `get_item(index)`: Return a standardized sample

Example:
```python
from .base_loader import BaseLoader

class MyDatasetLoader(BaseLoader):
    def _build_index(self):
        # Load your dataset metadata
        # Return a list of sample identifiers
        pass
    
    def get_item(self, index):
        # Get sample from index
        sample_info = self._index[index]
        
        # Create standardized base
        sample = self._get_standardized_base(
            sample_id=sample_info['id'],
            media_path=Path(sample_info['image_path']),
            media_type="image"
        )
        
        # Add your dataset-specific annotations
        sample['annotations']['your_field'] = sample_info['your_data']
        
        return sample
```

4. Add your loader to `__init__.py`
5. Write comprehensive tests in `tests/dataloaders/`

## Testing

Run tests for all dataloaders:
```bash
python -m pytest tests/dataloaders/ -v
```

Run tests for a specific dataloader:
```bash
python -m pytest tests/dataloaders/test_docvqa_loader.py -v
```

## Performance Considerations

- **Pre-loading**: The DocVQA loader pre-loads all OCR data during initialization for faster access. Consider this pattern for frequently accessed auxiliary data.
- **Lazy Loading**: Image files are not loaded until specifically requested, keeping memory usage low.
- **Caching**: Consider implementing an LRU cache for frequently accessed samples in production scenarios.