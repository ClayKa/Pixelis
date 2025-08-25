#!/usr/bin/env python
"""
Example script demonstrating how to use the DocVQA dataloader.
"""

import json
from pathlib import Path
from core.dataloaders import DocVqaLoader


def main():
    """Demonstrate DocVQA loader usage."""
    
    # Define configuration for the DocVQA dataset
    config = {
        "name": "docvqa_train",
        "image_path": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_images",
        "annotation_file": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_qas/train_v1.0_withQT.json",
        "ocr_path": "/mnt/c/Users/ClayKa/Pixelis/datasets/READ_TEXT/DocVQA/spdocvqa_ocr"
    }
    
    # Create the loader instance
    print("Initializing DocVQA loader...")
    loader = DocVqaLoader(config)
    print(f"Successfully loaded {len(loader)} samples")
    
    # Demonstrate accessing samples
    if len(loader) > 0:
        print("\n" + "="*60)
        print("SAMPLE 1:")
        print("="*60)
        
        # Get first sample
        sample = loader.get_item(0)
        
        # Display sample information
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Source Dataset: {sample['source_dataset']}")
        print(f"Media Type: {sample['media_type']}")
        print(f"Image Path: {Path(sample['media_path']).name}")
        
        if sample['width'] and sample['height']:
            print(f"Image Dimensions: {sample['width']}x{sample['height']}")
        
        # Display annotations
        annotations = sample['annotations']
        print(f"\nQuestion: {annotations['question']}")
        print(f"Answers: {annotations['answers']}")
        
        if 'question_types' in annotations:
            print(f"Question Types: {annotations['question_types']}")
        
        if 'document_id' in annotations:
            print(f"Document ID: {annotations['document_id']}")
        
        if 'page_number' in annotations:
            print(f"Page Number: {annotations['page_number']}")
        
        # Display OCR information
        if 'ocr_tokens' in annotations and annotations['ocr_tokens']:
            print(f"\nOCR Tokens Found: {len(annotations['ocr_tokens'])}")
            
            # Show first 5 OCR tokens as examples
            print("First 5 OCR tokens:")
            for i, token in enumerate(annotations['ocr_tokens'][:5], 1):
                print(f"  {i}. '{token['text']}' at bbox {token['bbox']}")
        
        # Display answer bounding boxes if found
        if 'answer_bboxes' in annotations and annotations['answer_bboxes']:
            print(f"\nAnswer Bounding Boxes Found: {len(annotations['answer_bboxes'])}")
            for answer_bbox in annotations['answer_bboxes']:
                print(f"  Answer: '{answer_bbox['answer']}'")
                print(f"  Bbox: {answer_bbox['bbox']}")
                print(f"  Token indices: {answer_bbox['token_indices']}")
    
    # Show a few more samples
    if len(loader) > 5:
        print("\n" + "="*60)
        print("ADDITIONAL SAMPLES (showing questions only):")
        print("="*60)
        
        for i in range(1, min(6, len(loader))):
            sample = loader.get_item(i)
            print(f"{i}. {sample['annotations']['question']}")
            print(f"   Answers: {sample['annotations']['answers']}")
    
    print("\n" + "="*60)
    print("DocVQA loader demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    main()