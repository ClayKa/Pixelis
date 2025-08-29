# core/data_generation/specialized_generator.py
"""
SpecializedGenerator Class
==========================
Task-specific implementation of the BaseGenerator for handling
different data generation tasks with their unique requirements.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from tqdm import tqdm

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)

# Import available dataloaders
try:
    from ..dataloaders import (
        BaseLoader, DocVqaLoader, HierTextLoader, ActivityNetCaptionsLoader, 
        DiDeMoLoader, Assembly101Loader
    )
    from ..dataloaders.sa1b_loader import Sa1bLoader
    from ..dataloaders.flickr30k_loader import Flickr30kLoader
    from ..dataloaders.starqa_loader import StarqaLoader
    from ..dataloaders.infographics_vqa_loader import InfographicsVqaLoader
    from ..dataloaders.part_imagenet_loader import PartImageNetLoader
    from ..dataloaders.unsplash_lite_loader import UnsplashLiteLoader
    from ..dataloaders.textcaps_loader import TextCapsLoader
    from ..dataloaders.mind2web_loader import Mind2WebLoader
    from ..dataloaders.msrvtt_loader import MsrVttLoader
    from ..dataloaders.uvo_loader import UvoLoader
    from ..dataloaders.mot_loader import MotLoader
    DATALOADERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import dataloaders: {e}. Using mock loaders.")
    DATALOADERS_AVAILABLE = False


class SpecializedGenerator(BaseGenerator):
    """
    Specialized generator that extends BaseGenerator with task-specific logic.
    
    This class handles:
    - Task-specific input/output formatting
    - Batch generation with progress tracking
    - Response parsing and validation
    - Sample creation for different task types
    """
    
    def __init__(self, task_name: str, loaders: Dict[str, Any] = None, 
                 config_path: str = "configs/data_generation_manifest.yaml"):
        """
        Initialize the SpecializedGenerator for a specific task.
        
        Args:
            task_name: Name of the task to generate data for
            loaders: Dictionary of data loaders for source datasets
            config_path: Path to the data generation manifest
        """
        super().__init__(config_path)
        
        self.task_name = task_name
        self.loaders = loaders or {}
        self.task_config = self.get_task_config(task_name)
        
        if not self.task_config:
            raise ValueError(f"Task '{task_name}' not found in configuration")
        
        self.generator_class = self.task_config.get('task_generator_class', '')
        self.target_samples = self.task_config.get('target_sample_count', 100)
        self.source_datasets = self.task_config.get('source_datasets', [])
        
        # Statistics tracking
        self.generation_stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'api_errors': 0,
            'parse_errors': 0
        }
        
        logger.info(f"SpecializedGenerator initialized for task: {task_name}")
    
    def generate_dataset(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset for the task.
        
        Args:
            num_samples: Number of samples to generate (uses config default if None)
            
        Returns:
            List of generated samples
        """
        num_samples = num_samples or self.target_samples
        generated_samples = []
        
        # Create sampling plan based on source weights
        sampling_plan = self._create_sampling_plan(num_samples)
        
        logger.info(f"Generating {num_samples} samples for {self.task_name}")
        logger.info(f"Sampling plan: {sampling_plan}")
        
        # Generate samples with progress tracking
        with tqdm(total=num_samples, desc=f"Generating {self.task_name}") as pbar:
            for source_name, count in sampling_plan.items():
                source_samples = self._generate_from_source(source_name, count, pbar)
                generated_samples.extend(source_samples)
        
        # Shuffle to mix samples from different sources
        random.shuffle(generated_samples)
        
        # Log generation statistics
        self._log_statistics()
        
        return generated_samples[:num_samples]  # Ensure exact count
    
    def _create_sampling_plan(self, num_samples: int) -> Dict[str, int]:
        """
        Create a sampling plan based on source dataset weights.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            Dictionary mapping source names to sample counts
        """
        plan = {}
        total_weight = 0.0
        
        # Calculate total weight
        for source_config in self.source_datasets:
            if isinstance(source_config, dict):
                weight = source_config.get('weight', 0)
                total_weight += weight
            elif isinstance(source_config, str):
                # Old format: just dataset names with equal weight
                total_weight += 1.0
        
        if total_weight == 0:
            logger.warning(f"No valid source datasets for {self.task_name}")
            return plan
        
        # Allocate samples based on weights
        allocated = 0
        for source_config in self.source_datasets:
            if isinstance(source_config, dict):
                source_name = source_config.get('name')
                weight = source_config.get('weight', 0)
            else:
                # Old format
                source_name = source_config
                weight = 1.0
            
            if source_name and weight > 0:
                count = int(num_samples * weight / total_weight)
                plan[source_name] = count
                allocated += count
        
        # Distribute remaining samples
        if allocated < num_samples and plan:
            # Add remaining to the highest weighted source
            max_source = max(plan.keys(), key=lambda x: plan[x])
            plan[max_source] += (num_samples - allocated)
        
        return plan
    
    def _generate_from_source(self, source_name: str, count: int, 
                             pbar: tqdm) -> List[Dict[str, Any]]:
        """
        Generate samples from a specific source dataset.
        
        Args:
            source_name: Name of the source dataset
            count: Number of samples to generate
            pbar: Progress bar to update
            
        Returns:
            List of generated samples
        """
        samples = []
        loader = self.loaders.get(source_name)
        
        if not loader:
            logger.warning(f"No loader available for source: {source_name}")
            pbar.update(count)
            self.generation_stats['failed_samples'] += count
            return samples
        
        for i in range(count):
            self.generation_stats['total_attempts'] += 1
            
            try:
                # Get raw sample from loader
                sample_idx = i % len(loader) if hasattr(loader, '__len__') else i
                raw_sample = self._get_raw_sample(loader, sample_idx)
                
                # Extract context for prompt
                context = self._extract_context(raw_sample, source_name)
                
                # Generate using API
                api_response = self.generate(self.task_name, context)
                
                if api_response:
                    # Process response into sample
                    sample = self.process_response(api_response, self.task_name)
                    
                    if sample:
                        # Add metadata
                        sample['provenance'] = {
                            'source_dataset': source_name,
                            'sample_id': raw_sample.get('sample_id', f'{source_name}_{i}'),
                            'task_type': self.task_name,
                            'generator_class': self.generator_class
                        }
                        
                        samples.append(sample)
                        self.generation_stats['successful_samples'] += 1
                    else:
                        self.generation_stats['parse_errors'] += 1
                        self.generation_stats['failed_samples'] += 1
                else:
                    self.generation_stats['api_errors'] += 1
                    self.generation_stats['failed_samples'] += 1
                    
            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                self.generation_stats['failed_samples'] += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Success': self.generation_stats['successful_samples'],
                'Failed': self.generation_stats['failed_samples']
            })
        
        return samples
    
    def _get_raw_sample(self, loader: Any, index: int) -> Dict[str, Any]:
        """
        Get a raw sample from a data loader.
        
        Args:
            loader: Data loader instance
            index: Sample index
            
        Returns:
            Raw sample dictionary
        """
        if hasattr(loader, 'get_item'):
            return loader.get_item(index)
        elif hasattr(loader, '__getitem__'):
            return loader[index]
        else:
            # Mock sample for testing
            return {
                'sample_id': f'sample_{index}',
                'media_path': f'/path/to/media_{index}',
                'annotations': {}
            }
    
    def _extract_context(self, raw_sample: Dict[str, Any], 
                        source_name: str) -> Dict[str, Any]:
        """
        Extract context from raw sample for prompt formatting.
        
        Args:
            raw_sample: Raw sample from loader
            source_name: Name of the source dataset
            
        Returns:
            Context dictionary for prompt variables
        """
        # Base context that all tasks can use
        context = {
            'source_dataset': source_name,
            'media_path': raw_sample.get('media_path', ''),
            'sample_id': raw_sample.get('sample_id', ''),
            'task_type': self.task_name,
            'context_block': ''  # Will be filled based on task type
        }
        
        # Task-specific context extraction
        if 'detail_perception' in self.task_name:
            context['context_block'] = self._create_detail_perception_context(raw_sample)
            context['action_name'] = 'ZOOM-IN'
            # Add specific variables for detail perception template
            self._add_detail_perception_variables(context, raw_sample, source_name)
        elif 'temporal_localization' in self.task_name:
            context['context_block'] = self._create_temporal_context(raw_sample)
            context['action_name'] = 'SELECT-FRAME'
        elif 'geometric_reasoning' in self.task_name:
            context['context_block'] = self._create_geometric_context(raw_sample)
            context['action_name'] = 'SEGMENT_OBJECT_AT+GET_PROPERTIES'
        elif 'contextual_reading' in self.task_name:
            context['context_block'] = self._create_reading_context(raw_sample)
            context['action_name'] = 'READ-TEXT'
        elif 'tracking' in self.task_name:
            context['context_block'] = self._create_tracking_context(raw_sample)
            context['action_name'] = 'TRACK-OBJECT'
        else:
            # Generic context
            context['context_block'] = self._create_generic_context(raw_sample)
            context['action_name'] = 'GENERIC'
        
        return context
    
    def _add_detail_perception_variables(self, context: Dict[str, Any], 
                                       raw_sample: Dict[str, Any], 
                                       source_name: str) -> None:
        """Add specific variables needed for detail perception prompt template."""
        # Generate mock bounding boxes and descriptions for each difficulty level
        import random
        
        # Easy task variables (presence/count)
        context['easy_source_dataset'] = source_name
        context['easy_bbox'] = f"[{random.randint(10, 100)}, {random.randint(10, 100)}, {random.randint(200, 400)}, {random.randint(200, 400)}]"
        context['easy_detail_description'] = random.choice([
            "A small red button", "A tiny logo", "A small animal", "A coin", "A small text label"
        ])
        
        # Medium task variables (reading text)
        context['medium_source_dataset'] = source_name
        context['medium_bbox'] = f"[{random.randint(50, 150)}, {random.randint(50, 150)}, {random.randint(250, 450)}, {random.randint(250, 450)}]"
        context['medium_text_content'] = random.choice([
            "License plate number", "Street sign text", "Product label", "Price tag", "Serial number"
        ])
        
        # Hard task variables (fine attributes)
        context['hard_source_dataset'] = source_name
        context['hard_bbox'] = f"[{random.randint(20, 120)}, {random.randint(20, 120)}, {random.randint(220, 420)}, {random.randint(220, 420)}]"
        context['hard_detail_description'] = random.choice([
            "Surface texture pattern", "Material composition", "Wear condition", "Color gradient", "Structural details"
        ])
    
    def _create_detail_perception_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create context for detail perception tasks."""
        annotations = raw_sample.get('annotations', {})
        description = annotations.get('description', 'Image with fine details')
        
        return f"""
**Task Type**: Detail Perception Analysis
**Visual Operation**: ZOOM-IN
**Image Description**: {description}
**Source**: {raw_sample.get('source_dataset', 'unknown')}

Generate a Chain-of-Thought-Action sample that uses ZOOM-IN to examine specific details."""
    
    def _create_temporal_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create context for temporal localization tasks."""
        annotations = raw_sample.get('annotations', {})
        description = annotations.get('summary', 'Video with temporal events')
        duration = annotations.get('duration', 'unknown')
        
        return f"""
**Task Type**: Temporal Localization
**Visual Operation**: SELECT-FRAME
**Video Description**: {description}
**Duration**: {duration}

Generate a Chain-of-Thought-Action sample that uses SELECT-FRAME to identify key moments."""
    
    def _create_geometric_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create context for geometric reasoning tasks."""
        annotations = raw_sample.get('annotations', {})
        objects = annotations.get('objects', [])
        
        return f"""
**Task Type**: Geometric Reasoning
**Visual Operations**: SEGMENT_OBJECT_AT, GET_PROPERTIES
**Scene**: Image with {len(objects)} objects

Generate a Chain-of-Thought-Action sample that analyzes object properties and spatial relationships."""
    
    def _create_reading_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create context for contextual reading tasks."""
        annotations = raw_sample.get('annotations', {})
        doc_type = annotations.get('doc_type', 'document')
        
        return f"""
**Task Type**: Contextual Reading
**Visual Operation**: READ-TEXT
**Document Type**: {doc_type}

Generate a Chain-of-Thought-Action sample that uses READ-TEXT to extract information."""
    
    def _create_tracking_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create context for tracking tasks."""
        annotations = raw_sample.get('annotations', {})
        
        return f"""
**Task Type**: Spatio-Temporal Tracking
**Visual Operation**: TRACK-OBJECT
**Scene**: Video with moving objects

Generate a Chain-of-Thought-Action sample that uses TRACK-OBJECT to follow objects."""
    
    def _create_generic_context(self, raw_sample: Dict[str, Any]) -> str:
        """Create generic context as fallback."""
        return f"""
**Task Type**: {self.task_name}
**Sample ID**: {raw_sample.get('sample_id', 'unknown')}

Generate a Chain-of-Thought-Action sample for this visual reasoning task."""
    
    def process_response(self, response: str, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Process the API response into a structured sample.
        
        Args:
            response: Raw API response text
            task_name: Name of the task
            
        Returns:
            Processed sample dictionary or None if parsing failed
        """
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            # Parse JSON
            sample = json.loads(cleaned)
            
            # Validate required fields
            required_fields = ['question', 'trajectory', 'final_answer']
            for field in required_fields:
                if field not in sample:
                    logger.warning(f"Missing required field: {field}")
                    return None
            
            # Ensure trajectory is a list
            if not isinstance(sample['trajectory'], list):
                logger.warning("Trajectory must be a list")
                return None
            
            return sample
            
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error processing response: {e}")
            return None
    
    def _log_statistics(self) -> None:
        """Log generation statistics."""
        stats = self.generation_stats
        total = stats['total_attempts']
        
        if total > 0:
            success_rate = (stats['successful_samples'] / total) * 100
            logger.info(f"Generation Statistics for {self.task_name}:")
            logger.info(f"  Total Attempts: {total}")
            logger.info(f"  Successful: {stats['successful_samples']} ({success_rate:.1f}%)")
            logger.info(f"  Failed: {stats['failed_samples']}")
            logger.info(f"  API Errors: {stats['api_errors']}")
            logger.info(f"  Parse Errors: {stats['parse_errors']}")
            
            # Also log API stats from base class
            api_stats = self.get_api_stats()
            logger.info(f"  API Requests: {api_stats['request_count']}")
            logger.info(f"  API Model: {api_stats['api_model']}")
    
    def save_dataset(self, samples: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save generated samples to a JSONL file.
        
        Args:
            samples: List of generated samples
            output_path: Path to save the dataset
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')
        
        logger.info(f"Saved {len(samples)} samples to {output_file}")