#!/usr/bin/env python
"""
Simple test of DocVQA loader with validation dataset (smaller).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataloaders import DocVqaLoader


def main():
    """Test DocVQA loader with validation set."""
    
    # Use validation set which is smaller
    config = {
        "name": "docvqa_val",
        "image_path": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_images",
        "annotation_file": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_qas/val_v1.0_withQT.json",
        "ocr_path": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_ocr"
    }
    
    try:
        print("Initializing DocVQA loader with validation set...")
        loader = DocVqaLoader(config)
        print(f"✓ Successfully loaded {len(loader)} samples")
        
        if len(loader) > 0:
            print("\nTesting sample access...")
            sample = loader.get_item(0)
            
            print(f"✓ Sample ID: {sample['sample_id']}")
            print(f"✓ Question: {sample['annotations']['question'][:100]}...")
            print(f"✓ Answers: {sample['annotations']['answers']}")
            
            if 'ocr_tokens' in sample['annotations']:
                print(f"✓ OCR tokens available: {len(sample['annotations']['ocr_tokens'])} tokens")
            
            print("\nDataLoader is working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()