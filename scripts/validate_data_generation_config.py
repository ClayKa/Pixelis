#!/usr/bin/env python3
"""
Validation Script for Data Generation Configuration
Ensures all components are properly aligned and ready for production.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class DataGenerationValidator:
    """Validates the complete data generation pipeline configuration"""
    
    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        self.validation_results = {
            "datasets": {},
            "generators": {},
            "loaders": {},
            "overall": True
        }
        self.errors = []
        self.warnings = []
    
    def _load_manifest(self) -> Dict:
        """Load the manifest file"""
        with open(self.manifest_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 60)
        logger.info("STARTING DATA GENERATION VALIDATION")
        logger.info("=" * 60)
        
        # 1. Validate dataset paths
        self._validate_dataset_paths()
        
        # 2. Validate task generators
        self._validate_task_generators()
        
        # 3. Validate data loader compatibility
        self._validate_data_loaders()
        
        # 4. Validate configuration consistency
        self._validate_configuration()
        
        # 5. Generate report
        self._generate_report()
        
        return self.validation_results["overall"]
    
    def _validate_dataset_paths(self):
        """Validate all dataset paths exist"""
        logger.info("\n1. VALIDATING DATASET PATHS")
        logger.info("-" * 40)
        
        datasources = self.manifest.get('datasources', {})
        
        for ds_name, ds_config in datasources.items():
            paths_valid = True
            missing_paths = []
            
            # Check all path fields
            path_fields = ['path', 'image_path', 'video_path', 'annotation_file',
                          'annotation_path', 'segments_file', 'keyframes_file',
                          'ocr_file', 'ocr_path', 'frame_path', 'data_path']
            
            for field in path_fields:
                if field in ds_config:
                    path = Path(ds_config[field])
                    if not path.exists():
                        paths_valid = False
                        missing_paths.append(f"{field}: {path}")
                        self.errors.append(f"Dataset {ds_name}: Missing {field}")
            
            self.validation_results["datasets"][ds_name] = {
                "valid": paths_valid,
                "missing": missing_paths
            }
            
            if paths_valid:
                logger.info(f"  ✓ {ds_name}: All paths valid")
            else:
                logger.error(f"  ✗ {ds_name}: Missing paths: {missing_paths}")
                self.validation_results["overall"] = False
    
    def _validate_task_generators(self):
        """Validate task generators exist and are properly configured"""
        logger.info("\n2. VALIDATING TASK GENERATORS")
        logger.info("-" * 40)
        
        try:
            from core.data_generation import TASK_GENERATOR_REGISTRY
        except ImportError as e:
            logger.error(f"Failed to import generator registry: {e}")
            self.validation_results["overall"] = False
            return
        
        tasks = self.manifest.get('tasks', {})
        
        for task_name, task_config in tasks.items():
            if not task_config.get('enabled', False):
                continue
            
            generator_class = task_config.get('task_generator_class')
            
            if generator_class in TASK_GENERATOR_REGISTRY:
                logger.info(f"  ✓ {task_name}: Generator '{generator_class}' found")
                self.validation_results["generators"][task_name] = True
                
                # Check source datasets
                for ds in task_config.get('source_datasets', []):
                    ds_name = ds.get('name')
                    if ds_name not in self.manifest.get('datasources', {}):
                        self.warnings.append(f"Task {task_name}: References undefined dataset '{ds_name}'")
            else:
                logger.error(f"  ✗ {task_name}: Generator '{generator_class}' not found")
                self.errors.append(f"Missing generator: {generator_class}")
                self.validation_results["generators"][task_name] = False
                self.validation_results["overall"] = False
    
    def _validate_data_loaders(self):
        """Validate data loader compatibility"""
        logger.info("\n3. VALIDATING DATA LOADERS")
        logger.info("-" * 40)
        
        # Check if data loaders can handle the dataset types
        dataset_types = set()
        for ds_config in self.manifest.get('datasources', {}).values():
            dataset_types.add(ds_config.get('type', 'unknown'))
        
        logger.info(f"  Dataset types found: {', '.join(sorted(dataset_types))}")
        
        # Map dataset types to expected loaders
        type_loader_map = {
            'InstanceSegmentation': 'COCODataLoader',
            'ObjectDetection': 'COCODataLoader',
            'ImageCaptioning': 'ImageCaptionDataLoader',
            'VideoQA': 'VideoDataLoader',
            'DocumentVQA': 'DocumentDataLoader',
            'MultiObjectTracking': 'MOTDataLoader',
            'VideoObjectSegmentation': 'VideoSegmentationDataLoader'
        }
        
        for ds_type in dataset_types:
            expected_loader = type_loader_map.get(ds_type, 'CustomDataLoader')
            logger.info(f"  {ds_type} -> {expected_loader}")
            self.validation_results["loaders"][ds_type] = expected_loader
    
    def _validate_configuration(self):
        """Validate overall configuration consistency"""
        logger.info("\n4. VALIDATING CONFIGURATION CONSISTENCY")
        logger.info("-" * 40)
        
        # Check total sample count
        tasks = self.manifest.get('tasks', {})
        total_samples = sum(
            task.get('target_sample_count', 0)
            for task in tasks.values()
            if task.get('enabled', False)
        )
        logger.info(f"  Total target samples: {total_samples:,}")
        
        # Check augmentation proportions
        aug_config = self.manifest.get('trajectory_augmentation', {})
        proportions = aug_config.get('proportions', {})
        total_prop = sum(proportions.values())
        
        if abs(total_prop - 1.0) > 0.01:
            self.warnings.append(f"Augmentation proportions sum to {total_prop}, not 1.0")
            logger.warning(f"  ⚠ Augmentation proportions: {total_prop}")
        else:
            logger.info(f"  ✓ Augmentation proportions: {total_prop}")
        
        # Check API configuration
        api_config = self.manifest.get('global_config', {}).get('api_profiles', {})
        for api_name, api_settings in api_config.items():
            env_var = api_settings.get('api_key_env_variable')
            if env_var and not os.environ.get(env_var):
                self.warnings.append(f"API key environment variable '{env_var}' not set")
                logger.warning(f"  ⚠ {api_name}: Environment variable '{env_var}' not set")
            else:
                logger.info(f"  ✓ {api_name}: Configuration valid")
    
    def _generate_report(self):
        """Generate final validation report"""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Dataset summary
        valid_datasets = sum(1 for d in self.validation_results["datasets"].values() if d.get("valid", False))
        total_datasets = len(self.validation_results["datasets"])
        logger.info(f"Datasets: {valid_datasets}/{total_datasets} valid")
        
        # Generator summary
        valid_generators = sum(1 for v in self.validation_results["generators"].values() if v)
        total_generators = len(self.validation_results["generators"])
        logger.info(f"Generators: {valid_generators}/{total_generators} valid")
        
        # Errors and warnings
        if self.errors:
            logger.error(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        # Final status
        if self.validation_results["overall"]:
            logger.info("\n✅ VALIDATION PASSED - Configuration is ready for production")
        else:
            logger.error("\n❌ VALIDATION FAILED - Please fix errors before proceeding")
        
        # Save detailed report
        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                "validation_results": self.validation_results,
                "errors": self.errors,
                "warnings": self.warnings
            }, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_path}")


def suggest_fixes(validator: DataGenerationValidator):
    """Suggest fixes for common issues"""
    logger.info("\n" + "=" * 60)
    logger.info("SUGGESTED FIXES")
    logger.info("=" * 60)
    
    suggestions = []
    
    # Check for missing datasets
    for ds_name, result in validator.validation_results["datasets"].items():
        if not result.get("valid", False):
            suggestions.append(f"Dataset '{ds_name}': Download or update paths in manifest")
    
    # Check for missing generators
    for task_name, valid in validator.validation_results["generators"].items():
        if not valid:
            suggestions.append(f"Task '{task_name}': Update task_generator_class to match registry")
    
    # Environment variables
    if any("Environment variable" in w for w in validator.warnings):
        suggestions.append("Set API key environment variables or use .env file")
    
    if suggestions:
        logger.info("Recommended actions:")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"  {i}. {suggestion}")
    else:
        logger.info("No immediate fixes required")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data generation configuration")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data_generation_manifest_fixed.yaml"),
        help="Path to manifest file"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Suggest fixes for issues"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check manifest exists
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        sys.exit(1)
    
    # Run validation
    validator = DataGenerationValidator(args.manifest)
    is_valid = validator.validate_all()
    
    # Suggest fixes if requested
    if args.fix:
        suggest_fixes(validator)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()