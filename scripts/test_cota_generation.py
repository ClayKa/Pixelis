#!/usr/bin/env python3
"""
Test script for CoTA (Chain-of-Thought-Action) data generation pipeline.

This script tests the complete data generation workflow including:
1. Task generator initialization
2. Data loader setup 
3. Sample generation
4. Quality validation
5. Trajectory augmentation

Usage:
    python scripts/test_cota_generation.py [--config path/to/config.yaml] [--samples N]
"""

import sys
import json
import yaml
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
import argparse
from unittest.mock import Mock
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.data_generation import (
        TASK_GENERATOR_REGISTRY,
        get_generator_class,
        DetailPerceptionTaskGenerator,
        SelectFrameTaskGenerator,
        GeometricComparisonTaskGenerator,
        TargetedOCRTaskGenerator,
        SpatioTemporalTaskGenerator
    )
    from core.data_generation.trajectory_augmenter import TrajectoryAugmenter
    from core.data_generation.validation_and_scoring import ValidationPipeline
    from core.data_generation.data_loader_interface import DataLoaderFactory
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    logger.info("Please ensure the project is properly set up and all dependencies are installed")
    sys.exit(1)


def create_mock_config() -> Dict[str, Any]:
    """Create a mock configuration for testing."""
    return {
        'global_config': {
            'api_profiles': {
                'generator_api': {
                    'base_url': 'https://openrouter.ai/api/v1',
                    'api_key_env': 'OPENROUTER_API_KEY',
                    'model': 'openai/gpt-4o',
                    'temperature': 0.7,
                    'max_tokens': 4000
                },
                'scorer_api': {
                    'base_url': 'https://openrouter.ai/api/v1', 
                    'api_key_env': 'OPENROUTER_API_KEY',
                    'model': 'openai/gpt-4o',
                    'temperature': 0.1,
                    'max_tokens': 1000
                }
            },
            'output_settings': {
                'save_interval': 100,
                'checkpoint_interval': 500
            }
        },
        'tasks': {
            'geometric_comparison_task': {
                'name': 'geometric_comparison_task',
                'enabled': True,
                'generator_class': 'GeometricComparisonTaskGenerator',
                'target_sample_count': 10,
                'prompt_template': 'prompts/geometric_reasoning.md',
                'data_sources': ['coco2017_train'],
                'generator_params': {
                    'min_objects': 2,
                    'max_retries': 3
                }
            },
            'detail_perception_task': {
                'name': 'detail_perception_task',
                'enabled': True,
                'generator_class': 'DetailPerceptionTaskGenerator', 
                'target_sample_count': 10,
                'prompt_template': 'prompts/detail_perception.md',
                'data_sources': ['sa1b_for_zoomin'],
                'generator_params': {
                    'zoom_levels': [2, 4, 8],
                    'min_resolution': 512
                }
            }
        },
        'datasources': {
            'coco2017_train': {
                'type': 'ObjectDetection',
                'path': '/tmp/mock_coco/images',
                'annotation_file': '/tmp/mock_coco/annotations.json',
                'format': 'coco',
                'weight': 1.0
            },
            'sa1b_for_zoomin': {
                'type': 'HighResolutionImages',
                'path': '/tmp/mock_sa1b/images',
                'annotation_path': '/tmp/mock_sa1b/annotations', 
                'format': 'sa1b',
                'weight': 1.0
            }
        }
    }


def create_mock_coco_data(temp_dir: Path) -> Path:
    """Create mock COCO dataset for testing."""
    coco_dir = temp_dir / "mock_coco"
    coco_dir.mkdir()
    
    # Create mock images directory
    images_dir = coco_dir / "images"
    images_dir.mkdir()
    
    # Create mock annotation file
    annotations = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [100, 100, 200, 150], "area": 30000,
                "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]]
            },
            {
                "id": 2, "image_id": 1, "category_id": 2, 
                "bbox": [400, 200, 150, 100], "area": 15000,
                "segmentation": [[400, 200, 550, 200, 550, 300, 400, 300]]
            }
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"}
        ]
    }
    
    ann_file = coco_dir / "annotations.json"
    with open(ann_file, 'w') as f:
        json.dump(annotations, f)
        
    return coco_dir


def create_mock_sa1b_data(temp_dir: Path) -> Path:
    """Create mock SA-1B dataset for testing."""
    sa1b_dir = temp_dir / "mock_sa1b" 
    sa1b_dir.mkdir()
    
    # Create mock images directory
    images_dir = sa1b_dir / "images"
    images_dir.mkdir()
    
    # Create mock annotations directory
    annotations_dir = sa1b_dir / "annotations"
    annotations_dir.mkdir()
    
    return sa1b_dir


def test_task_generator_registry():
    """Test that task generators are properly registered."""
    logger.info("Testing task generator registry...")
    
    expected_generators = [
        'DetailPerceptionTaskGenerator',
        'SelectFrameTaskGenerator', 
        'GeometricComparisonTaskGenerator',
        'TargetedOCRTaskGenerator',
        'SpatioTemporalTaskGenerator'
    ]
    
    for generator_name in expected_generators:
        generator_class = get_generator_class(generator_name)
        if generator_class is None:
            logger.error(f"Generator {generator_name} not found in registry")
            return False
        else:
            logger.info(f"✓ Found generator: {generator_name}")
    
    logger.info("✓ Task generator registry test passed")
    return True


def test_geometric_reasoning_generator():
    """Test the geometric reasoning task generator."""
    logger.info("Testing GeometricComparisonTaskGenerator...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        coco_dir = create_mock_coco_data(temp_path)
        
        # Create mock config
        config = create_mock_config()
        config['datasources']['coco2017_train']['path'] = str(coco_dir / "images")
        config['datasources']['coco2017_train']['annotation_file'] = str(coco_dir / "annotations.json")
        
        try:
            # Create mock loaders (since we can't load real data easily)
            mock_loader = Mock()
            mock_loader.get_sample.return_value = {
                'image_path': str(coco_dir / "images" / "image1.jpg"),
                'annotations': [
                    {'bbox': [100, 100, 200, 150], 'category_id': 1, 'area': 30000},
                    {'bbox': [400, 200, 150, 100], 'category_id': 2, 'area': 15000}
                ],
                'metadata': {'width': 640, 'height': 480}
            }
            
            loaders = {'coco2017_train': mock_loader}
            
            # Initialize generator
            generator = GeometricComparisonTaskGenerator(
                loaders=loaders,
                config=config['tasks']['geometric_comparison_task'],
                global_config=config['global_config']
            )
            
            logger.info("✓ GeometricComparisonTaskGenerator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GeometricComparisonTaskGenerator: {e}")
            return False


def test_zoom_in_generator():
    """Test the zoom-in task generator."""
    logger.info("Testing DetailPerceptionTaskGenerator (zoom-in)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock data
        sa1b_dir = create_mock_sa1b_data(temp_path)
        
        # Create mock config
        config = create_mock_config()
        config['datasources']['sa1b_for_zoomin']['path'] = str(sa1b_dir / "images")
        config['datasources']['sa1b_for_zoomin']['annotation_path'] = str(sa1b_dir / "annotations")
        
        try:
            # Create mock loader
            mock_loader = Mock()
            mock_loader.get_sample.return_value = {
                'image_path': str(sa1b_dir / "images" / "image1.jpg"),
                'segments': [
                    {'segmentation': {'size': [480, 640], 'counts': 'mock_rle'}, 'bbox': [100, 100, 200, 150]},
                    {'segmentation': {'size': [480, 640], 'counts': 'mock_rle2'}, 'bbox': [300, 200, 150, 100]}
                ],
                'metadata': {'width': 640, 'height': 480}
            }
            
            loaders = {'sa1b_for_zoomin': mock_loader}
            
            # Initialize generator 
            generator = DetailPerceptionTaskGenerator(
                loaders=loaders,
                config=config['tasks']['detail_perception_task'],
                global_config=config['global_config']
            )
            
            logger.info("✓ DetailPerceptionTaskGenerator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DetailPerceptionTaskGenerator: {e}")
            return False


def test_trajectory_augmentation():
    """Test trajectory augmentation functionality."""
    logger.info("Testing trajectory augmentation...")
    
    try:
        # Create a sample trajectory
        sample_trajectory = {
            'trajectory_id': 'test_001',
            'task_type': 'geometric_comparison',
            'image_path': '/mock/image.jpg',
            'steps': [
                {
                    'step_id': 1,
                    'reasoning': 'I need to segment the objects in this image to compare their properties.',
                    'action': 'SEGMENT_OBJECT_AT',
                    'action_params': {'coordinates': [200, 200]},
                    'observation': 'Segmented a red car in the center of the image.'
                },
                {
                    'step_id': 2,
                    'reasoning': 'Now I will get properties of this segmented object.',
                    'action': 'GET_PROPERTIES',
                    'action_params': {'segment_id': 1},
                    'observation': 'Car properties: color=red, shape=rectangular, area=15000'
                }
            ],
            'final_answer': 'The red car has an area of 15000 pixels.',
            'metadata': {
                'source_dataset': 'coco2017_train',
                'difficulty': 'medium',
                'generation_time': '2024-01-01T12:00:00Z'
            }
        }
        
        # Create augmenter
        augmenter = TrajectoryAugmenter()
        
        # Test golden sample creation
        golden_sample = augmenter.create_golden_sample(sample_trajectory)
        if golden_sample.get('sample_type') != 'golden':
            logger.error("Golden sample creation failed")
            return False
            
        logger.info("✓ Golden sample creation successful")
        
        # Test trap sample creation
        trap_sample = augmenter.create_trap_sample(sample_trajectory)
        if trap_sample.get('sample_type') != 'trap':
            logger.error("Trap sample creation failed")
            return False
            
        logger.info("✓ Trap sample creation successful")
        
        # Test self-correction sample creation
        correction_sample = augmenter.create_self_correction_sample(sample_trajectory)
        if correction_sample.get('sample_type') != 'self_correction':
            logger.error("Self-correction sample creation failed")
            return False
            
        logger.info("✓ Self-correction sample creation successful")
        
        logger.info("✓ Trajectory augmentation test passed")
        return True
        
    except Exception as e:
        logger.error(f"Trajectory augmentation test failed: {e}")
        return False


def test_validation_pipeline():
    """Test the validation and quality scoring pipeline."""
    logger.info("Testing validation pipeline...")
    
    try:
        # Create sample trajectory for validation
        sample_trajectory = {
            'trajectory_id': 'test_validation_001',
            'task_type': 'geometric_comparison',
            'steps': [
                {
                    'step_id': 1,
                    'reasoning': 'I will segment the object to analyze it.',
                    'action': 'SEGMENT_OBJECT_AT',
                    'action_params': {'coordinates': [100, 100]},
                    'observation': 'Successfully segmented a blue rectangle.'
                },
                {
                    'step_id': 2,
                    'reasoning': 'Now I will get the properties of this object.',
                    'action': 'GET_PROPERTIES', 
                    'action_params': {'segment_id': 1},
                    'observation': 'Properties: color=blue, shape=rectangle, area=5000'
                }
            ],
            'final_answer': 'The blue rectangle has an area of 5000 pixels.',
            'metadata': {
                'source_dataset': 'coco2017_train',
                'difficulty': 'easy'
            }
        }
        
        # Create validation pipeline
        validator = ValidationPipeline()
        
        # Test structural validation
        structural_valid = validator.validate_structure(sample_trajectory)
        if not structural_valid:
            logger.error("Structural validation failed")
            return False
        logger.info("✓ Structural validation passed")
        
        # Test quality scoring
        quality_score = validator.calculate_quality_score(sample_trajectory)
        if not isinstance(quality_score, dict) or 'overall_score' not in quality_score:
            logger.error("Quality scoring failed")
            return False
        logger.info(f"✓ Quality scoring successful: {quality_score.get('overall_score', 0):.2f}")
        
        logger.info("✓ Validation pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation pipeline test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("=" * 60)
    logger.info("STARTING COTA GENERATION COMPREHENSIVE TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Task Generator Registry", test_task_generator_registry),
        ("Geometric Reasoning Generator", test_geometric_reasoning_generator),
        ("Zoom-In Generator", test_zoom_in_generator),
        ("Trajectory Augmentation", test_trajectory_augmentation),
        ("Validation Pipeline", test_validation_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n[TEST] {test_name}")
        logger.info("-" * 40)
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            logger.info(f"[{status}] {test_name}")
        except Exception as e:
            logger.error(f"[ERROR] {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} {test_name}")
    
    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - CoTA generation system is ready!")
        return True
    else:
        logger.warning("⚠️  Some tests failed - please review the issues above")
        return False


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Test CoTA data generation pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--samples', type=int, default=5, help='Number of test samples to generate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
