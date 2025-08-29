#!/usr/bin/env python3
"""
Configuration Update Script for CoTA Data Generation
Updates all data paths in the configuration files to match local dataset structure
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigurationUpdater:
    """Updates data generation configuration with correct paths"""
    
    def __init__(self, base_path: Path, dry_run: bool = False):
        """
        Initialize the configuration updater
        
        Args:
            base_path: Base path to the datasets directory
            dry_run: If True, only show what would be changed without modifying files
        """
        self.base_path = Path(base_path).resolve()
        self.dry_run = dry_run
        self.found_datasets = {}
        self.missing_datasets = []
        
        logger.info(f"Initializing ConfigurationUpdater with base path: {self.base_path}")
        logger.info(f"Dry run mode: {self.dry_run}")
    
    def scan_datasets(self) -> Dict[str, Path]:
        """
        Scan the base path for available datasets
        
        Returns:
            Dictionary mapping dataset names to their paths
        """
        logger.info("Scanning for datasets...")
        
        # Expected dataset structure
        expected_structure = {
            # ZOOM-IN datasets
            "SA1B4zoomin": "ZOOM-IN/SA1B4zoomin",
            "Flickr30k": "ZOOM-IN/Flickr30k",
            "Mind2Web": "ZOOM-IN/Mind2Web",
            "TextCaps": "ZOOM-IN/TextCaps",
            "Unsplash-lite-25k": "ZOOM-IN/Unsplash-lite-25k",
            
            # SELECT-FRAME datasets
            "STARQA": "SELECT-FRAME/STARQA",
            "DiDeMo": "SELECT-FRAME/DiDeMo",
            "MSRVTT": "SELECT-FRAME/MSRVTT",
            "ActivityNetCaptions": "SELECT-FRAME/ActivityNetCaptions",
            "Assembly101": "SELECT-FRAME/Assembly101",
            
            # SEGMENT_OBJECT_AT+GET_PROPERTIES datasets
            "COCO2017+LVIS": "SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017+LVIS",
            "PartImageNet": "SEGMENT_OBJECT_AT+GET_PROPERTIES/PartImageNet",
            "SA1B4segment": "SEGMENT_OBJECT_AT+GET_PROPERTIES/SA1B4segment",
            
            # READ_TEXT datasets
            "InfographicsVQA": "READ_TEXT/InfographicsVQA",
            "DocVQA": "READ_TEXT/DocVQA",
            "HierText": "READ_TEXT/HierText",
            "ICDAR2019ArT": "READ_TEXT/ICDAR2019ArT",
            
            # TRACK_OBJECT datasets
            "MOT20": "TRACK_OBJECT/MOT20",
            "UVO": "TRACK_OBJECT/UVO",
            "EPIC-KITCHENS": "TRACK_OBJECT/EPIC-KITCHENS",
            "VIS2022": "TRACK_OBJECT/VIS2022"
        }
        
        for dataset_name, relative_path in expected_structure.items():
            full_path = self.base_path / relative_path
            
            if full_path.exists():
                self.found_datasets[dataset_name] = full_path
                logger.info(f"✓ Found: {dataset_name} at {full_path}")
            else:
                self.missing_datasets.append((dataset_name, full_path))
                logger.warning(f"✗ Missing: {dataset_name} expected at {full_path}")
        
        logger.info(f"Found {len(self.found_datasets)} datasets, {len(self.missing_datasets)} missing")
        return self.found_datasets
    
    def update_manifest_paths(self, manifest_path: Path) -> Dict[str, Any]:
        """
        Update paths in the data generation manifest
        
        Args:
            manifest_path: Path to the manifest YAML file
            
        Returns:
            Updated manifest dictionary
        """
        logger.info(f"Updating manifest: {manifest_path}")
        
        # Load existing manifest
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Update datasource paths
        datasources = manifest.get('datasources', {})
        updated_count = 0
        
        for ds_name, ds_config in datasources.items():
            logger.debug(f"Processing datasource: {ds_name}")
            
            # Map datasource to expected dataset
            dataset_mapping = {
                'sa1b_for_zoomin': 'SA1B4zoomin',
                'flickr30k': 'Flickr30k',
                'mind2web_train': 'Mind2Web',
                'textcaps_train': 'TextCaps',
                'unsplash_lite': 'Unsplash-lite-25k',
                'starqa_train': 'STARQA',
                'didemo_train': 'DiDeMo',
                'msrvtt_train': 'MSRVTT',
                'activitynet_captions_train': 'ActivityNetCaptions',
                'assembly101_train': 'Assembly101',
                'coco2017_train': 'COCO2017+LVIS',
                'lvis_v1_train': 'COCO2017+LVIS',
                'part_imagenet_train': 'PartImageNet',
                'sa1b_for_segmentation': 'SA1B4segment',
                'infographics_vqa_train': 'InfographicsVQA',
                'docvqa_train': 'DocVQA',
                'hiertext_train': 'HierText',
                'icdar_2019_art_train': 'ICDAR2019ArT',
                'mot20_train': 'MOT20',
                'uvo_dense_train': 'UVO',
                'epic_kitchens_visor_train': 'EPIC-KITCHENS',
                'youtube_vos_2022_train': 'VIS2022'
            }
            
            if ds_name in dataset_mapping:
                dataset_name = dataset_mapping[ds_name]
                
                if dataset_name in self.found_datasets:
                    base_dataset_path = self.found_datasets[dataset_name]
                    
                    # Update various path fields
                    path_fields = ['path', 'image_path', 'video_path', 'annotation_file', 
                                   'annotation_path', 'mask_path', 'ocr_path', 'ocr_file',
                                   'raw_captions_file', 'category_file', 'action_metadata_file',
                                   'sparse_annotation_path', 'dense_annotation_path',
                                   'class_mapping_file', 'frame_mapping_file']
                    
                    for field in path_fields:
                        if field in ds_config and '# <-- CHANGEME' in str(ds_config.get(field, '')):
                            old_path = ds_config[field]
                            
                            # Construct new path based on field type
                            new_path = self._construct_path(base_dataset_path, field, ds_name)
                            
                            if new_path:
                                ds_config[field] = str(new_path)
                                updated_count += 1
                                logger.info(f"  Updated {ds_name}.{field}: {new_path}")
        
        logger.info(f"Updated {updated_count} paths in manifest")
        
        # Save updated manifest
        if not self.dry_run:
            backup_path = manifest_path.with_suffix('.yaml.bak')
            manifest_path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved updated manifest: {manifest_path}")
        else:
            logger.info("DRY RUN: Would save updated manifest")
            print("\nUpdated manifest preview:")
            print(yaml.dump(manifest, default_flow_style=False, sort_keys=False)[:1000])
        
        return manifest
    
    def _construct_path(self, base_path: Path, field: str, datasource: str) -> Optional[Path]:
        """
        Construct appropriate path based on field type and dataset structure
        
        Args:
            base_path: Base dataset path
            field: Field name (e.g., 'image_path', 'annotation_file')
            datasource: Datasource name
            
        Returns:
            Constructed path or None if cannot determine
        """
        # Common directory names
        common_dirs = {
            'image_path': ['images', 'imgs', 'Image', 'train_images', 'rgb_frames'],
            'video_path': ['videos', 'video', 'train'],
            'annotation_file': ['annotations', 'annotation', 'qas', 'gt'],
            'annotation_path': ['annotations', 'annotation'],
            'mask_path': ['masks', 'Segmentation', 'segmentation'],
            'ocr_path': ['ocr', 'OCR'],
            'ocr_file': ['ocr', 'OCR']
        }
        
        # Common file patterns
        file_patterns = {
            'annotation_file': ['*.json', '*.csv', '*.jsonl', '*.txt'],
            'ocr_file': ['*.json'],
            'raw_captions_file': ['raw*.json', 'caption*.json'],
            'category_file': ['category*.json', 'categories*.json', '*.csv'],
            'action_metadata_file': ['action*.csv', 'metadata*.csv'],
            'class_mapping_file': ['*class*.csv', '*mapping*.csv'],
            'frame_mapping_file': ['*frame*.json', '*mapping*.json']
        }
        
        # Try to find appropriate subdirectory
        if field in common_dirs:
            for dir_name in common_dirs[field]:
                potential_path = base_path / dir_name
                if potential_path.exists():
                    return potential_path
        
        # For file fields, try to find matching files
        if field in file_patterns:
            for pattern in file_patterns[field]:
                matches = list(base_path.rglob(pattern))
                if matches:
                    # Return the first match
                    return matches[0]
        
        # Default fallback
        if 'file' in field:
            # It's a file, return base_path with placeholder
            return base_path / f"PLACEHOLDER_{field}"
        else:
            # It's a directory
            return base_path / field.replace('_path', '').replace('_', '')
    
    def create_dataset_symlinks(self, target_dir: Path):
        """
        Create symbolic links for easier dataset access
        
        Args:
            target_dir: Directory where symlinks will be created
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating dataset symlinks in: {target_dir}")
        
        for dataset_name, dataset_path in self.found_datasets.items():
            symlink_path = target_dir / dataset_name
            
            if symlink_path.exists():
                if symlink_path.is_symlink():
                    logger.info(f"  Symlink exists: {dataset_name}")
                else:
                    logger.warning(f"  Path exists but is not a symlink: {symlink_path}")
            else:
                if not self.dry_run:
                    try:
                        symlink_path.symlink_to(dataset_path)
                        logger.info(f"  Created symlink: {dataset_name} -> {dataset_path}")
                    except OSError as e:
                        logger.error(f"  Failed to create symlink for {dataset_name}: {e}")
                else:
                    logger.info(f"  DRY RUN: Would create symlink {dataset_name} -> {dataset_path}")
    
    def generate_dataset_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a report of dataset availability
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report content as string
        """
        report = []
        report.append("=" * 80)
        report.append("DATASET AVAILABILITY REPORT")
        report.append("=" * 80)
        report.append(f"Base Path: {self.base_path}")
        report.append(f"Total Expected: {len(self.found_datasets) + len(self.missing_datasets)}")
        report.append(f"Found: {len(self.found_datasets)}")
        report.append(f"Missing: {len(self.missing_datasets)}")
        report.append("")
        
        if self.found_datasets:
            report.append("AVAILABLE DATASETS:")
            report.append("-" * 40)
            for name, path in sorted(self.found_datasets.items()):
                # Check size
                try:
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_gb = size / (1024**3)
                    report.append(f"✓ {name:30} {size_gb:.2f} GB")
                except:
                    report.append(f"✓ {name:30} [size unknown]")
        
        if self.missing_datasets:
            report.append("")
            report.append("MISSING DATASETS:")
            report.append("-" * 40)
            for name, expected_path in sorted(self.missing_datasets):
                report.append(f"✗ {name:30} Expected at: {expected_path}")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if self.missing_datasets:
            report.append("1. Download missing datasets to their expected locations")
            report.append("2. Or update the configuration to point to actual locations")
            report.append("3. Use mock data for testing with missing datasets")
        else:
            report.append("✓ All datasets are available!")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text
    
    def validate_manifest(self, manifest_path: Path) -> List[str]:
        """
        Validate that all paths in manifest exist
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            List of validation errors
        """
        logger.info(f"Validating manifest: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        errors = []
        datasources = manifest.get('datasources', {})
        
        for ds_name, ds_config in datasources.items():
            for field, value in ds_config.items():
                if 'path' in field or 'file' in field:
                    if isinstance(value, str) and not value.startswith('# <--'):
                        path = Path(value)
                        if not path.exists():
                            errors.append(f"{ds_name}.{field}: Path does not exist: {value}")
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors")
        else:
            logger.info("✓ All paths in manifest are valid")
        
        return errors


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Update CoTA data generation configuration with correct paths"
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Base directory containing all datasets (default: datasets)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data_generation_manifest.yaml"),
        help="Path to data generation manifest (default: configs/data_generation_manifest.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--create-symlinks",
        type=Path,
        help="Create symbolic links in specified directory"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Generate dataset availability report"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate manifest after updating"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if datasets directory exists
    if not args.datasets_dir.exists():
        logger.error(f"Datasets directory does not exist: {args.datasets_dir}")
        logger.info("Please create the directory or specify correct path with --datasets-dir")
        sys.exit(1)
    
    # Initialize updater
    updater = ConfigurationUpdater(args.datasets_dir, dry_run=args.dry_run)
    
    # Scan for datasets
    updater.scan_datasets()
    
    # Generate report if requested
    if args.report:
        report = updater.generate_dataset_report(args.report)
        print("\n" + report)
    
    # Update manifest if it exists
    if args.manifest.exists():
        updater.update_manifest_paths(args.manifest)
        
        # Validate if requested
        if args.validate:
            errors = updater.validate_manifest(args.manifest)
            if errors:
                print("\nValidation Errors:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("\n✓ Manifest validation passed!")
    else:
        logger.warning(f"Manifest file not found: {args.manifest}")
        logger.info("Please create the manifest file first or specify correct path with --manifest")
    
    # Create symlinks if requested
    if args.create_symlinks:
        updater.create_dataset_symlinks(args.create_symlinks)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Datasets found: {len(updater.found_datasets)}")
    print(f"Datasets missing: {len(updater.missing_datasets)}")
    
    if updater.missing_datasets:
        print("\nMissing datasets (first 5):")
        for name, _ in updater.missing_datasets[:5]:
            print(f"  - {name}")
        
        print("\nTo proceed with missing datasets:")
        print("  1. Download them to expected locations")
        print("  2. Or use --dry-run to test without modifying files")
        print("  3. Or use mock data for testing")


if __name__ == "__main__":
    main()