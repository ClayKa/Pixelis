# core/dataloaders/loader_factory.py
"""
Factory for creating DataLoader instances based on configuration.
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

# Import all available loaders with correct class names
from .sa1b_loader import Sa1bLoader
from .docvqa_loader import DocVqaLoader
from .infographics_vqa_loader import InfographicsVqaLoader
from .hiertext_loader import HierTextLoader
from .activitynet_captions_loader import ActivityNetCaptionsLoader
from .didemo_loader import DiDeMoLoader
from .assembly101_loader import Assembly101Loader
from .starqa_loader import StarqaLoader as StarQALoader
from .msrvtt_loader import MsrVttLoader
from .coco_segment_loader import CocoSegmentLoader
from .lvis_segment_loader import LvisSegmentLoader
from .part_imagenet_loader import PartImageNetLoader
from .mind2web_loader import Mind2WebLoader
from .textcaps_loader import TextCapsLoader
from .unsplash_lite_loader import UnsplashLiteLoader
from .flickr30k_loader import Flickr30kLoader
from .icdar_art_loader import IcdarArTLoader
from .mot_loader import MotLoader
from .uvo_loader import UvoLoader
from .epic_kitchens_loader import EpicKitchensVisorLoader
from .youtube_vos_loader import YouTubeVOSLoader

logger = logging.getLogger(__name__)


# Registry mapping datasource types to loader classes
DATALOADER_REGISTRY = {
    # SA-1B datasets (unified loader for both zoom and segmentation)
    'sa1b': Sa1bLoader,
    'sa1b_zoom': Sa1bLoader,
    'sa1b_segment': Sa1bLoader,
    'InstanceSegmentation': Sa1bLoader,  # For SA1B and LVIS
    
    # VQA datasets
    'docvqa': DocVqaLoader,
    'infographics_vqa': InfographicsVqaLoader,
    'DocumentVQA': DocVqaLoader,  # Type name from manifest
    
    # Text datasets
    'hiertext': HierTextLoader,
    'HierarchicalText': HierTextLoader,
    'textcaps': TextCapsLoader,
    'ImageTextCaptioning': TextCapsLoader,
    'icdar_art': IcdarArTLoader,
    'ArbitraryText': IcdarArTLoader,
    
    # Video datasets
    'activitynet_captions': ActivityNetCaptionsLoader,
    'DenseVideoCaptioning': ActivityNetCaptionsLoader,
    'didemo': DiDeMoLoader,
    'VideoMomentRetrieval': DiDeMoLoader,
    'assembly101': Assembly101Loader,
    'TimedActionVideo': Assembly101Loader,
    'starqa': StarQALoader,
    'VideoQA': StarQALoader,
    'msrvtt': MsrVttLoader,
    'VideoCaptioning': MsrVttLoader,
    'mot': MotLoader,
    'mot20': MotLoader,
    'MultiObjectTracking': MotLoader,
    'uvo': UvoLoader,
    'VideoObjectSegmentation': UvoLoader,
    'epic_kitchens': EpicKitchensVisorLoader,
    'epic_kitchens_visor': EpicKitchensVisorLoader,
    'EgocentricVideo': EpicKitchensVisorLoader,
    'youtube_vos': YouTubeVOSLoader,
    'vis2022': YouTubeVOSLoader,  # VIS2022 uses similar format
    'VideoInstanceSegmentation': YouTubeVOSLoader,
    
    # Image segmentation datasets
    'coco': CocoSegmentLoader,
    'coco2017': CocoSegmentLoader,
    'ObjectDetection': CocoSegmentLoader,
    'lvis': LvisSegmentLoader,
    'lvis_v1': LvisSegmentLoader,
    'part_imagenet': PartImageNetLoader,
    'PartSegmentation': PartImageNetLoader,
    
    # Web and other datasets
    'mind2web': Mind2WebLoader,
    'WebAutomation': Mind2WebLoader,
    'unsplash': UnsplashLiteLoader,
    'unsplash_lite': UnsplashLiteLoader,
    'HighResolutionImages': UnsplashLiteLoader,
    'flickr30k': Flickr30kLoader,
    'ImageCaptioning': Flickr30kLoader,
}


def create_dataloader(datasource_name: str, datasource_config: Dict[str, Any]) -> Any:
    """
    Factory function to create a DataLoader instance based on configuration.
    
    Args:
        datasource_name: Name of the datasource
        datasource_config: Configuration dictionary for the datasource
        
    Returns:
        Initialized DataLoader instance
        
    Raises:
        ValueError: If loader type is not found in registry
    """
    loader_type = datasource_config.get('type', 'unknown')
    
    # Check if loader type exists in registry
    if loader_type not in DATALOADER_REGISTRY:
        # Try to find a partial match
        for key in DATALOADER_REGISTRY:
            if key in loader_type or loader_type in key:
                loader_type = key
                break
        else:
            raise ValueError(f"Unknown loader type: {loader_type} for datasource: {datasource_name}")
    
    # Get the loader class
    LoaderClass = DATALOADER_REGISTRY[loader_type]
    
    # Prepare config for the loader
    loader_config = {
        'source_name': datasource_name,
        **datasource_config
    }
    
    # Handle path configurations
    if 'path' in loader_config:
        loader_config['path'] = str(Path(loader_config['path']))
    if 'annotation_path' in loader_config:
        loader_config['annotation_path'] = str(Path(loader_config['annotation_path']))
    if 'annotations_path' in loader_config:
        loader_config['annotations_path'] = str(Path(loader_config['annotations_path']))
    
    try:
        # Instantiate the loader
        loader = LoaderClass(loader_config)
        logger.info(f"Successfully created {LoaderClass.__name__} for '{datasource_name}'")
        return loader
    except Exception as e:
        logger.error(f"Failed to create loader for '{datasource_name}': {e}")
        raise


def get_available_loaders() -> Dict[str, type]:
    """
    Get a dictionary of all available loader types and their classes.
    
    Returns:
        Dictionary mapping loader type names to loader classes
    """
    return DATALOADER_REGISTRY.copy()