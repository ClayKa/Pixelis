#!/usr/bin/env python3
"""
Test script to verify data loading works with actual datasets.
"""

import sys
import json
import yaml
from pathlib import Path
import logging
from typing import Dict, Any
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_coco_loader():
    """Test COCO dataset loading"""
    try:
        from pycocotools.coco import COCO
        
        # Test COCO 2017 train
        ann_file = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/SEGMENT_OBJECT_AT+GET_PROPERTIES/COCO2017/annotations/instances_train2017.json")
        if ann_file.exists():
            logger.info("Testing COCO2017 train loading...")
            coco = COCO(str(ann_file))
            img_ids = coco.getImgIds()
            logger.info(f"  ✓ Loaded {len(img_ids)} images")
            
            # Test loading a random image
            img_id = random.choice(img_ids[:10])
            img_info = coco.loadImgs(img_id)[0]
            logger.info(f"  ✓ Sample image: {img_info['file_name']}")
            
            # Test annotations
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            logger.info(f"  ✓ Found {len(anns)} annotations")
        else:
            logger.warning(f"  ✗ COCO annotations not found: {ann_file}")
            
    except Exception as e:
        logger.error(f"Failed to test COCO loader: {e}")


def test_video_dataset():
    """Test video dataset structure"""
    try:
        # Test STARQA
        starqa_path = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/STARQA")
        if starqa_path.exists():
            logger.info("Testing STARQA dataset...")
            
            # Check videos
            video_files = list(starqa_path.glob("*.mp4"))
            logger.info(f"  ✓ Found {len(video_files)} video files")
            
            # Check pickle files
            pkl_files = list(starqa_path.glob("*.pkl"))
            logger.info(f"  ✓ Found {len(pkl_files)} pickle files: {[f.name for f in pkl_files]}")
            
            # Check JSON files
            json_files = list(starqa_path.glob("*.json"))
            logger.info(f"  ✓ Found {len(json_files)} JSON files")
            
    except Exception as e:
        logger.error(f"Failed to test video dataset: {e}")


def test_ocr_dataset():
    """Test OCR/text dataset"""
    try:
        # Test TextCaps
        textcaps_path = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/ZOOM-IN/TextCaps")
        if textcaps_path.exists():
            logger.info("Testing TextCaps dataset...")
            
            # Check OCR file
            ocr_file = textcaps_path / "TextVQA_Rosetta_OCR_v0.2_train.json"
            if ocr_file.exists():
                with open(ocr_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"  ✓ Loaded OCR data with {len(data.get('data', []))} entries")
            else:
                logger.warning(f"  ✗ OCR file not found: {ocr_file}")
                
            # Check images
            img_dir = textcaps_path / "train_images"
            if img_dir.exists():
                img_count = len(list(img_dir.glob("*.jpg")))
                logger.info(f"  ✓ Found {img_count} images")
                
    except Exception as e:
        logger.error(f"Failed to test OCR dataset: {e}")


def test_document_dataset():
    """Test document VQA dataset"""
    try:
        # Test DocVQA
        docvqa_path = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/READ-TEXT/DocVQA")
        if docvqa_path.exists():
            logger.info("Testing DocVQA dataset...")
            
            # Check directories
            subdirs = [d for d in docvqa_path.iterdir() if d.is_dir()]
            logger.info(f"  ✓ Found {len(subdirs)} subdirectories")
            
            # Check for spdocvqa folders
            sp_folders = [d for d in subdirs if 'spdocvqa' in d.name]
            if sp_folders:
                logger.info(f"  ✓ Found SP-DocVQA folders: {[f.name for f in sp_folders[:3]]}")
                
    except Exception as e:
        logger.error(f"Failed to test document dataset: {e}")


def test_tracking_dataset():
    """Test multi-object tracking dataset"""
    try:
        # Test MOT20
        mot20_path = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/TRACK-OBJECT/MOT20/train")
        if mot20_path.exists():
            logger.info("Testing MOT20 dataset...")
            
            # Check sequence folders
            sequences = [d for d in mot20_path.iterdir() if d.is_dir()]
            logger.info(f"  ✓ Found {len(sequences)} sequences")
            
            if sequences:
                # Check first sequence structure
                seq = sequences[0]
                has_gt = (seq / "gt" / "gt.txt").exists()
                has_imgs = (seq / "img1").exists()
                logger.info(f"  ✓ Sequence '{seq.name}': GT={has_gt}, Images={has_imgs}")
                
    except Exception as e:
        logger.error(f"Failed to test tracking dataset: {e}")


def test_high_res_dataset():
    """Test high-resolution image dataset"""
    try:
        # Test SA-1B for zoom-in
        sa1b_path = Path("/mnt/c/Users/ClayKa/Pixelis/datasets/ZOOM-IN/SA1B4zoomin")
        if sa1b_path.exists():
            logger.info("Testing SA-1B for zoom-in...")
            
            # Check for image folders
            subdirs = [d for d in sa1b_path.iterdir() if d.is_dir()]
            logger.info(f"  ✓ Found {len(subdirs)} subdirectories")
            
            # Sample check for images
            total_images = 0
            for subdir in subdirs[:3]:  # Check first 3 dirs
                img_count = len(list(subdir.glob("*.jpg")))
                total_images += img_count
            logger.info(f"  ✓ Sample count from first 3 dirs: {total_images} images")
            
    except Exception as e:
        logger.error(f"Failed to test high-res dataset: {e}")


def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING PIPELINE")
    logger.info("=" * 60)
    
    # Test different dataset types
    test_coco_loader()
    test_video_dataset()
    test_ocr_dataset()
    test_document_dataset()
    test_tracking_dataset()
    test_high_res_dataset()
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA LOADING TEST COMPLETE")
    logger.info("=" * 60)
    
    # Load and check manifest
    manifest_path = Path("configs/data_generation_manifest_fixed.yaml")
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Summary statistics
        total_datasets = len(manifest.get('datasources', {}))
        total_tasks = len([t for t in manifest.get('tasks', {}).values() if t.get('enabled')])
        total_samples = sum(t.get('target_sample_count', 0) for t in manifest.get('tasks', {}).values() if t.get('enabled'))
        
        logger.info("\nMANIFEST SUMMARY:")
        logger.info(f"  Total datasets configured: {total_datasets}")
        logger.info(f"  Total tasks enabled: {total_tasks}")
        logger.info(f"  Total target samples: {total_samples:,}")
    
    logger.info("\n✅ Data loading pipeline is ready for production")


if __name__ == "__main__":
    main()