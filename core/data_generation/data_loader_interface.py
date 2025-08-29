"""
Data Loader Abstraction Layer with SOLID Principles
Provides a unified interface for all dataset loaders following Interface Segregation Principle
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ======================== DATA STRUCTURES ========================

class DatasetType(Enum):
    """Types of datasets supported"""
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    VIDEO_UNDERSTANDING = "video_understanding"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VQA = "visual_question_answering"
    IMAGE_CAPTIONING = "image_captioning"
    VIDEO_TRACKING = "video_tracking"


@dataclass
class DataSample:
    """Unified data sample structure"""
    sample_id: str
    data_type: DatasetType
    
    # Core data fields
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    text: Optional[str] = None
    
    # Annotations
    labels: Optional[List[str]] = None
    bboxes: Optional[List[List[float]]] = None
    masks: Optional[np.ndarray] = None
    keypoints: Optional[List[List[float]]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.sample_id,
            "type": self.data_type.value,
            "image": str(self.image_path) if self.image_path else None,
            "video": str(self.video_path) if self.video_path else None,
            "text": self.text,
            "labels": self.labels,
            "bboxes": self.bboxes,
            "metadata": self.metadata
        }


# ======================== ABSTRACT INTERFACES ========================

class DataLoaderInterface(ABC):
    """Abstract interface for all data loaders"""
    
    @abstractmethod
    def load_sample(self, index: int) -> DataSample:
        """Load a single sample by index"""
        pass
    
    @abstractmethod
    def get_total_samples(self) -> int:
        """Get total number of samples"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        pass
    
    @abstractmethod
    def get_dataset_type(self) -> DatasetType:
        """Get the type of dataset"""
        pass


class FilterableDataLoader(DataLoaderInterface):
    """Interface for data loaders that support filtering"""
    
    @abstractmethod
    def filter_by_criteria(self, criteria: Dict[str, Any]) -> List[int]:
        """Filter samples by criteria, return matching indices"""
        pass
    
    @abstractmethod
    def get_samples_with_property(self, property_name: str, property_value: Any) -> List[int]:
        """Get samples with specific property"""
        pass


class StreamableDataLoader(DataLoaderInterface):
    """Interface for data loaders that support streaming"""
    
    @abstractmethod
    def stream_samples(self, batch_size: int = 1) -> Iterator[List[DataSample]]:
        """Stream samples in batches"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        pass


# ======================== BASE IMPLEMENTATION ========================

class BaseDataLoader(DataLoaderInterface):
    """Base implementation with common functionality"""
    
    def __init__(
        self,
        dataset_path: Path,
        dataset_type: DatasetType,
        cache_size: Optional[int] = 100,
        random_seed: Optional[int] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        self.cache_size = cache_size
        self.random_seed = random_seed
        
        # Initialize cache
        self._cache = {}
        self._cache_order = []
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load metadata
        self._metadata = self._load_metadata()
        self._total_samples = self._count_samples()
        
        logger.info(f"Initialized {self.__class__.__name__} with {self._total_samples} samples")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_path = self.dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"path": str(self.dataset_path)}
    
    @abstractmethod
    def _count_samples(self) -> int:
        """Count total samples in dataset"""
        pass
    
    @abstractmethod
    def _load_sample_impl(self, index: int) -> DataSample:
        """Implementation-specific sample loading"""
        pass
    
    def load_sample(self, index: int) -> DataSample:
        """Load sample with caching"""
        if index < 0 or index >= self._total_samples:
            raise IndexError(f"Index {index} out of range [0, {self._total_samples})")
        
        # Check cache
        if index in self._cache:
            # Move to end (LRU)
            self._cache_order.remove(index)
            self._cache_order.append(index)
            return self._cache[index]
        
        # Load sample
        sample = self._load_sample_impl(index)
        
        # Update cache
        if self.cache_size and len(self._cache) >= self.cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        if self.cache_size:
            self._cache[index] = sample
            self._cache_order.append(index)
        
        return sample
    
    def get_total_samples(self) -> int:
        """Get total number of samples"""
        return self._total_samples
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self._metadata
    
    def get_dataset_type(self) -> DatasetType:
        """Get dataset type"""
        return self.dataset_type
    
    def clear_cache(self):
        """Clear the sample cache"""
        self._cache.clear()
        self._cache_order.clear()


# ======================== CONCRETE IMPLEMENTATIONS ========================

class COCODataLoader(BaseDataLoader, FilterableDataLoader):
    """Data loader for COCO-format datasets"""
    
    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        dataset_type: DatasetType = DatasetType.INSTANCE_SEGMENTATION,
        **kwargs
    ):
        self.images_path = Path(images_path)
        self.annotations_path = Path(annotations_path)
        
        # Load annotations
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create indices
        self._build_indices()
        
        super().__init__(
            dataset_path=images_path.parent,
            dataset_type=dataset_type,
            **kwargs
        )
    
    def _build_indices(self):
        """Build indices for fast lookups"""
        self.image_id_to_index = {
            img['id']: idx 
            for idx, img in enumerate(self.coco_data['images'])
        }
        
        self.category_id_to_name = {
            cat['id']: cat['name']
            for cat in self.coco_data.get('categories', [])
        }
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
    
    def _count_samples(self) -> int:
        """Count total samples"""
        return len(self.coco_data['images'])
    
    def _load_sample_impl(self, index: int) -> DataSample:
        """Load a COCO sample"""
        image_info = self.coco_data['images'][index]
        image_id = image_info['id']
        
        # Get annotations for this image
        annotations = self.image_annotations.get(image_id, [])
        
        # Extract bboxes and labels
        bboxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            if 'bbox' in ann:
                bboxes.append(ann['bbox'])
            
            if 'category_id' in ann:
                cat_name = self.category_id_to_name.get(ann['category_id'], 'unknown')
                labels.append(cat_name)
            
            if 'segmentation' in ann:
                # Simplified - in production, decode RLE or polygon
                masks.append(ann['segmentation'])
        
        return DataSample(
            sample_id=str(image_id),
            data_type=self.dataset_type,
            image_path=self.images_path / image_info['file_name'],
            labels=labels,
            bboxes=bboxes,
            masks=np.array(masks) if masks else None,
            metadata={
                'width': image_info['width'],
                'height': image_info['height'],
                'coco_url': image_info.get('coco_url'),
                'date_captured': image_info.get('date_captured')
            }
        )
    
    def filter_by_criteria(self, criteria: Dict[str, Any]) -> List[int]:
        """Filter samples by criteria"""
        matching_indices = []
        
        for idx, image_info in enumerate(self.coco_data['images']):
            image_id = image_info['id']
            annotations = self.image_annotations.get(image_id, [])
            
            # Check criteria
            if 'min_objects' in criteria:
                if len(annotations) < criteria['min_objects']:
                    continue
            
            if 'max_objects' in criteria:
                if len(annotations) > criteria['max_objects']:
                    continue
            
            if 'required_categories' in criteria:
                image_categories = {
                    self.category_id_to_name.get(ann['category_id'])
                    for ann in annotations
                    if 'category_id' in ann
                }
                required = set(criteria['required_categories'])
                if not required.issubset(image_categories):
                    continue
            
            if 'min_image_size' in criteria:
                min_size = criteria['min_image_size']
                if image_info['width'] < min_size[0] or image_info['height'] < min_size[1]:
                    continue
            
            matching_indices.append(idx)
        
        return matching_indices
    
    def get_samples_with_property(self, property_name: str, property_value: Any) -> List[int]:
        """Get samples with specific property"""
        if property_name == 'category':
            # Find images with specific category
            category_id = None
            for cat_id, cat_name in self.category_id_to_name.items():
                if cat_name == property_value:
                    category_id = cat_id
                    break
            
            if category_id is None:
                return []
            
            matching_indices = []
            for idx, image_info in enumerate(self.coco_data['images']):
                image_id = image_info['id']
                annotations = self.image_annotations.get(image_id, [])
                
                for ann in annotations:
                    if ann.get('category_id') == category_id:
                        matching_indices.append(idx)
                        break
            
            return matching_indices
        
        return []


class VideoDataLoader(BaseDataLoader, StreamableDataLoader):
    """Data loader for video datasets"""
    
    def __init__(
        self,
        videos_path: Path,
        annotations_path: Optional[Path] = None,
        frame_sampling_rate: int = 1,
        max_frames: int = 300,
        **kwargs
    ):
        self.videos_path = Path(videos_path)
        self.annotations_path = annotations_path
        self.frame_sampling_rate = frame_sampling_rate
        self.max_frames = max_frames
        
        # Load video metadata
        self.video_files = sorted(list(self.videos_path.glob("*.mp4")))
        
        # Load annotations if available
        self.annotations = {}
        if annotations_path and annotations_path.exists():
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        
        super().__init__(
            dataset_path=videos_path,
            dataset_type=DatasetType.VIDEO_UNDERSTANDING,
            **kwargs
        )
    
    def _count_samples(self) -> int:
        """Count video files"""
        return len(self.video_files)
    
    def _load_sample_impl(self, index: int) -> DataSample:
        """Load video sample metadata"""
        video_file = self.video_files[index]
        video_id = video_file.stem
        
        # Get annotations if available
        video_annotations = self.annotations.get(video_id, {})
        
        return DataSample(
            sample_id=video_id,
            data_type=DatasetType.VIDEO_UNDERSTANDING,
            video_path=video_file,
            text=video_annotations.get('caption'),
            labels=video_annotations.get('labels'),
            metadata={
                'duration': video_annotations.get('duration'),
                'fps': video_annotations.get('fps'),
                'frame_count': video_annotations.get('frame_count'),
                'frame_sampling_rate': self.frame_sampling_rate,
                'max_frames': self.max_frames
            }
        )
    
    def stream_samples(self, batch_size: int = 1) -> Iterator[List[DataSample]]:
        """Stream video samples in batches"""
        batch = []
        
        for idx in range(self._total_samples):
            sample = self.load_sample(idx)
            batch.append(sample)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining samples
        if batch:
            yield batch
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        cache_size_mb = sum(
            self._estimate_sample_size(sample) 
            for sample in self._cache.values()
        ) / (1024 * 1024)
        
        return {
            'cache_size_mb': cache_size_mb,
            'cached_samples': len(self._cache),
            'total_samples': self._total_samples
        }
    
    def _estimate_sample_size(self, sample: DataSample) -> float:
        """Estimate memory size of a sample in bytes"""
        # Simplified estimation
        size = 0
        
        if sample.metadata.get('frame_count'):
            # Estimate based on video metadata
            frames = min(sample.metadata['frame_count'], self.max_frames)
            # Assume 3 bytes per pixel, 640x480 resolution
            size += frames * 640 * 480 * 3
        
        return size


class DocumentDataLoader(BaseDataLoader, FilterableDataLoader):
    """Data loader for document understanding datasets"""
    
    def __init__(
        self,
        documents_path: Path,
        ocr_path: Optional[Path] = None,
        annotations_path: Optional[Path] = None,
        **kwargs
    ):
        self.documents_path = Path(documents_path)
        self.ocr_path = ocr_path
        self.annotations_path = annotations_path
        
        # Load document list
        self.document_files = sorted(list(self.documents_path.glob("*.png")))
        self.document_files.extend(sorted(list(self.documents_path.glob("*.jpg"))))
        self.document_files.extend(sorted(list(self.documents_path.glob("*.pdf"))))
        
        # Load OCR data if available
        self.ocr_data = {}
        if ocr_path and ocr_path.exists():
            for ocr_file in ocr_path.glob("*.json"):
                doc_id = ocr_file.stem
                with open(ocr_file, 'r') as f:
                    self.ocr_data[doc_id] = json.load(f)
        
        # Load annotations
        self.annotations = {}
        if annotations_path and annotations_path.exists():
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        
        super().__init__(
            dataset_path=documents_path,
            dataset_type=DatasetType.DOCUMENT_UNDERSTANDING,
            **kwargs
        )
    
    def _count_samples(self) -> int:
        """Count document files"""
        return len(self.document_files)
    
    def _load_sample_impl(self, index: int) -> DataSample:
        """Load document sample"""
        doc_file = self.document_files[index]
        doc_id = doc_file.stem
        
        # Get OCR data
        ocr = self.ocr_data.get(doc_id, {})
        
        # Get annotations
        doc_annotations = self.annotations.get(doc_id, {})
        
        # Extract text from OCR
        text_content = ""
        text_bboxes = []
        
        if 'text_regions' in ocr:
            for region in ocr['text_regions']:
                text_content += region.get('text', '') + " "
                if 'bbox' in region:
                    text_bboxes.append(region['bbox'])
        
        return DataSample(
            sample_id=doc_id,
            data_type=DatasetType.DOCUMENT_UNDERSTANDING,
            image_path=doc_file,
            text=text_content.strip(),
            bboxes=text_bboxes if text_bboxes else None,
            labels=doc_annotations.get('labels'),
            metadata={
                'document_type': doc_annotations.get('type', 'unknown'),
                'language': ocr.get('language', 'en'),
                'num_words': len(text_content.split()),
                'num_regions': len(text_bboxes),
                'questions': doc_annotations.get('questions', [])
            }
        )
    
    def filter_by_criteria(self, criteria: Dict[str, Any]) -> List[int]:
        """Filter documents by criteria"""
        matching_indices = []
        
        for idx in range(self._total_samples):
            sample = self.load_sample(idx)
            
            # Check criteria
            if 'document_type' in criteria:
                if sample.metadata.get('document_type') != criteria['document_type']:
                    continue
            
            if 'min_words' in criteria:
                if sample.metadata.get('num_words', 0) < criteria['min_words']:
                    continue
            
            if 'language' in criteria:
                if sample.metadata.get('language') != criteria['language']:
                    continue
            
            if 'has_questions' in criteria and criteria['has_questions']:
                if not sample.metadata.get('questions'):
                    continue
            
            matching_indices.append(idx)
        
        return matching_indices
    
    def get_samples_with_property(self, property_name: str, property_value: Any) -> List[int]:
        """Get documents with specific property"""
        criteria = {property_name: property_value}
        return self.filter_by_criteria(criteria)


# ======================== PYTORCH DATASET WRAPPER ========================

class UnifiedPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for unified data loaders"""
    
    def __init__(
        self,
        data_loader: DataLoaderInterface,
        transform=None,
        indices: Optional[List[int]] = None
    ):
        self.data_loader = data_loader
        self.transform = transform
        
        # Use subset of indices if provided
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(data_loader.get_total_samples()))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        actual_idx = self.indices[idx]
        sample = self.data_loader.load_sample(actual_idx)
        
        # Convert to dict
        item = sample.to_dict()
        
        # Load and transform image if present
        if sample.image_path and sample.image_path.exists():
            image = Image.open(sample.image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Convert to tensor by default
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            item['image'] = image
        
        return item


# ======================== DATA LOADER FACTORY ========================

class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    _loaders = {
        'coco': COCODataLoader,
        'video': VideoDataLoader,
        'document': DocumentDataLoader
    }
    
    @classmethod
    def register(cls, name: str, loader_class: type):
        """Register a new loader class"""
        cls._loaders[name] = loader_class
    
    @classmethod
    def create(cls, loader_type: str, **kwargs) -> DataLoaderInterface:
        """Create a data loader instance"""
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        loader_class = cls._loaders[loader_type]
        return loader_class(**kwargs)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> DataLoaderInterface:
        """Create loader from configuration dict"""
        loader_type = config.pop('type')
        
        # Convert string paths to Path objects
        for key in ['dataset_path', 'images_path', 'annotations_path', 
                    'videos_path', 'documents_path', 'ocr_path']:
            if key in config and config[key]:
                config[key] = Path(config[key])
        
        return cls.create(loader_type, **config)


# ======================== USAGE EXAMPLES ========================

def example_usage():
    """Examples of using the data loader abstraction"""
    
    # Example 1: COCO dataset loader
    coco_loader = COCODataLoader(
        images_path=Path("datasets/coco/images"),
        annotations_path=Path("datasets/coco/annotations.json"),
        cache_size=50
    )
    
    # Load a sample
    sample = coco_loader.load_sample(0)
    print(f"COCO sample: {sample.sample_id}, labels: {sample.labels}")
    
    # Filter samples
    indices = coco_loader.filter_by_criteria({
        'min_objects': 3,
        'required_categories': ['person', 'car']
    })
    print(f"Found {len(indices)} matching COCO samples")
    
    # Example 2: Video dataset loader
    video_loader = VideoDataLoader(
        videos_path=Path("datasets/videos"),
        frame_sampling_rate=5,
        max_frames=100
    )
    
    # Stream samples
    for batch in video_loader.stream_samples(batch_size=4):
        print(f"Video batch: {[s.sample_id for s in batch]}")
        break
    
    # Example 3: Document loader
    doc_loader = DocumentDataLoader(
        documents_path=Path("datasets/documents"),
        ocr_path=Path("datasets/documents/ocr")
    )
    
    # Filter documents
    indices = doc_loader.filter_by_criteria({
        'min_words': 100,
        'language': 'en'
    })
    
    # Example 4: PyTorch integration
    pytorch_dataset = UnifiedPyTorchDataset(
        data_loader=coco_loader,
        indices=indices[:100]  # Use first 100 filtered samples
    )
    
    from torch.utils.data import DataLoader
    torch_loader = DataLoader(
        pytorch_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    # Example 5: Factory usage
    loader = DataLoaderFactory.create(
        'coco',
        images_path=Path("datasets/coco/images"),
        annotations_path=Path("datasets/coco/annotations.json")
    )
    
    # From config
    config = {
        'type': 'video',
        'videos_path': 'datasets/videos',
        'frame_sampling_rate': 10
    }
    loader = DataLoaderFactory.create_from_config(config)


if __name__ == "__main__":
    print("Data Loader Abstraction Layer - Examples")
    print("=" * 50)
    # Uncomment to run examples with actual data
    # example_usage()