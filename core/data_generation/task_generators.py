"""
Task Generators for Data Synthesis
===================================
This module contains various task generators for creating specialized
training data for different visual reasoning capabilities.
"""

import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseTaskGenerator(ABC):
    """Abstract base class for all task generators."""
    
    def __init__(self, data_source_path: Path, **kwargs):
        """
        Initialize the task generator.
        
        Args:
            data_source_path: Path to the source dataset
            **kwargs: Additional configuration parameters
        """
        self.data_source_path = data_source_path
        self.config = kwargs
        self.data = self._load_data()
        
    @abstractmethod
    def _load_data(self) -> Any:
        """Load and parse the source data."""
        pass
        
    @abstractmethod
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate task samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of generated task samples
        """
        pass
        
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a generated sample.
        
        Args:
            sample: The sample to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['question', 'trajectory', 'final_answer', 'provenance']
        return all(key in sample for key in required_keys)


class GeometricComparisonTaskGenerator(BaseTaskGenerator):
    """Generator for geometric comparison tasks using object segmentation data."""
    
    def _load_data(self) -> Dict[str, Any]:
        """Load COCO-style annotations."""
        try:
            annotation_file = self.config.get('annotation_file')
            if annotation_file is None:
                annotation_file = self.data_source_path / 'annotations.json'
            
            # Convert to Path if string
            if isinstance(annotation_file, str):
                annotation_file = Path(annotation_file)
                
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            return {'images': [], 'annotations': []}
            
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate geometric comparison tasks."""
        samples = []
        
        if not self.data.get('images') or not self.data.get('annotations'):
            logger.warning("No valid data to generate samples from")
            return samples
            
        # Group annotations by image
        annotations_by_image = {}
        for ann in self.data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id:
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
                
        # Generate samples
        attempts = 0
        max_attempts = num_samples * 10  # Allow multiple attempts
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Find images with at least 2 objects
            valid_images = [img_id for img_id, anns in annotations_by_image.items() 
                          if len(anns) >= 2]
                          
            if not valid_images:
                logger.warning("No images with multiple objects found")
                break
                
            # Select random image
            img_id = random.choice(valid_images)
            annotations = annotations_by_image[img_id]
            
            # Handle missing area gracefully
            valid_annotations = []
            for ann in annotations:
                if 'area' in ann and ann['area'] is not None:
                    valid_annotations.append(ann)
                else:
                    logger.debug(f"Skipping annotation without area: {ann.get('id')}")
                    
            if len(valid_annotations) < 2:
                continue  # Try another image
                
            # Select two objects for comparison
            obj1, obj2 = random.sample(valid_annotations, 2)
            
            # Generate the task
            sample = {
                'question': f"Which object is larger: the object at position {obj1.get('bbox', [0,0,0,0])[:2]} or the object at position {obj2.get('bbox', [0,0,0,0])[:2]}?",
                'trajectory': [
                    {
                        'action': 'SEGMENT_OBJECT_AT',
                        'parameters': {'x': obj1.get('bbox', [0,0,0,0])[0], 'y': obj1.get('bbox', [0,0,0,0])[1]},
                        'observation': f"Found object with area {obj1.get('area', 0)}"
                    },
                    {
                        'action': 'GET_PROPERTIES',
                        'parameters': {'object_id': 1},
                        'observation': f"Object 1 properties: area={obj1.get('area', 0)}"
                    },
                    {
                        'action': 'SEGMENT_OBJECT_AT',
                        'parameters': {'x': obj2.get('bbox', [0,0,0,0])[0], 'y': obj2.get('bbox', [0,0,0,0])[1]},
                        'observation': f"Found object with area {obj2.get('area', 0)}"
                    },
                    {
                        'action': 'GET_PROPERTIES',
                        'parameters': {'object_id': 2},
                        'observation': f"Object 2 properties: area={obj2.get('area', 0)}"
                    }
                ],
                'final_answer': f"The {'first' if obj1.get('area', 0) > obj2.get('area', 0) else 'second'} object is larger.",
                'provenance': {
                    'source': 'mock_coco',
                    'image_id': img_id
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
                
        return samples


class TargetedOCRTaskGenerator(BaseTaskGenerator):
    """Generator for targeted OCR tasks."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load infographics VQA style data."""
        data = []
        try:
            with open(self.data_source_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load OCR data: {e}")
        return data
        
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate targeted OCR tasks."""
        samples = []
        
        if not self.data:
            logger.warning("No OCR data available")
            return samples
            
        for i in range(min(num_samples, len(self.data))):
            source_item = self.data[i % len(self.data)]
            
            # Handle missing fields gracefully
            bbox = source_item.get('bbox', [100, 100, 200, 200])
            text = source_item.get('text', 'Sample text')
            
            sample = {
                'question': f"What text is located in the region {bbox}?",
                'trajectory': [
                    {
                        'action': 'READ_TEXT',
                        'parameters': {'region': bbox},
                        'observation': f"Text detected: '{text}'"
                    }
                ],
                'final_answer': text,
                'provenance': {
                    'source': 'infographics_vqa',
                    'sample_id': i
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
                
        return samples


class SpatioTemporalTaskGenerator(BaseTaskGenerator):
    """Generator for spatio-temporal analysis tasks."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load MOT17-style tracking data."""
        data = []
        try:
            with open(self.data_source_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        data.append({
                            'frame': int(parts[0]),
                            'object_id': int(parts[1]),
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'w': float(parts[4]),
                            'h': float(parts[5])
                        })
        except Exception as e:
            logger.error(f"Failed to load tracking data: {e}")
        return data
        
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate spatio-temporal analysis tasks."""
        samples = []
        
        if not self.data:
            logger.warning("No tracking data available")
            return samples
            
        # Group by object ID
        tracks_by_id = {}
        for item in self.data:
            obj_id = item['object_id']
            if obj_id not in tracks_by_id:
                tracks_by_id[obj_id] = []
            tracks_by_id[obj_id].append(item)
            
        # Generate samples for objects with tracks
        valid_objects = [obj_id for obj_id, tracks in tracks_by_id.items() 
                        if len(tracks) >= 2]
                        
        for i in range(min(num_samples, len(valid_objects))):
            obj_id = valid_objects[i % len(valid_objects)]
            tracks = sorted(tracks_by_id[obj_id], key=lambda x: x['frame'])
            
            start_frame = tracks[0]['frame']
            end_frame = tracks[-1]['frame']
            
            sample = {
                'question': f"Track object {obj_id} from frame {start_frame} to frame {end_frame}",
                'trajectory': [
                    {
                        'action': 'TRACK_OBJECT',
                        'parameters': {
                            'object_id': obj_id,
                            'frames': [start_frame, end_frame]
                        },
                        'observation': f"Tracked object {obj_id} across {len(tracks)} frames"
                    }
                ],
                'final_answer': f"Object {obj_id} moved from ({tracks[0]['x']}, {tracks[0]['y']}) to ({tracks[-1]['x']}, {tracks[-1]['y']})",
                'provenance': {
                    'source': 'mot17',
                    'object_id': obj_id
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
                
        return samples


class ZoomInTaskGenerator(BaseTaskGenerator):
    """Generator for zoom-in tasks (Pixel-Reasoner baseline)."""
    
    def _load_data(self) -> List[str]:
        """Load list of high-resolution images."""
        images = []
        try:
            if self.data_source_path.is_dir():
                images = [str(f) for f in self.data_source_path.glob('*.jpg')]
                images.extend([str(f) for f in self.data_source_path.glob('*.png')])
        except Exception as e:
            logger.error(f"Failed to load image list: {e}")
        return images
        
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate zoom-in tasks."""
        samples = []
        
        if not self.data:
            logger.warning("No images available for zoom-in tasks")
            return samples
            
        for i in range(min(num_samples, len(self.data))):
            image_path = self.data[i % len(self.data)]
            
            sample = {
                'question': "Zoom in on the central region of the image for more detail",
                'trajectory': [
                    {
                        'action': 'ZOOM_IN',
                        'parameters': {'scale': 2.0, 'center': [0.5, 0.5]},
                        'observation': "Zoomed in on central region, details are now clearer"
                    }
                ],
                'final_answer': "Successfully zoomed in on the central region",
                'provenance': {
                    'source': 'sa1b_subset',
                    'image': image_path
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
                
        return samples


class SelectFrameTaskGenerator(BaseTaskGenerator):
    """Generator for select frame tasks (Pixel-Reasoner baseline)."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load annotated video data."""
        videos = []
        try:
            if self.data_source_path.is_file():
                with open(self.data_source_path, 'r') as f:
                    videos = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load video data: {e}")
        return videos
        
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate select frame tasks."""
        samples = []
        
        if not self.data:
            logger.warning("No video data available")
            return samples
            
        for i in range(min(num_samples, len(self.data))):
            video = self.data[i % len(self.data)]
            
            sample = {
                'question': f"Select the most relevant frame from video {video.get('id', 'unknown')}",
                'trajectory': [
                    {
                        'action': 'SELECT_FRAME',
                        'parameters': {'video_id': video.get('id', 0), 'criteria': 'relevance'},
                        'observation': f"Selected frame {video.get('key_frame', 15)}"
                    }
                ],
                'final_answer': f"Frame {video.get('key_frame', 15)} is most relevant",
                'provenance': {
                    'source': 'starqa_subset',
                    'video_id': video.get('id', 'unknown')
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
                
        return samples