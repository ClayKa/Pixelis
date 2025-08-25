#!/usr/bin/env python3
"""
Example usage of the DiDeMoLoader.

This script demonstrates how to use the DiDeMoLoader to load
and iterate through the DiDeMo (Distinct Describable Moments) dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataloaders.didemo_loader import DiDeMoLoader


def main():
    """Main function to demonstrate loader usage."""
    
    # Configuration for the loader
    # Update these paths to match your actual dataset location
    config = {
        "name": "didemo_train",
        "type": "VideoMomentRetrieval",
        "path": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/DiDeMo/videos",
        "annotation_file": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/DiDeMo/annotations/didemo_train.json"
    }
    
    print("Initializing DiDeMoLoader...")
    print()
    
    try:
        # Initialize the loader
        loader = DiDeMoLoader(config)
        
        print(f"Successfully loaded dataset with {len(loader)} moment samples")
        print()
        
        # Load and display information about the first few samples
        num_samples_to_show = min(3, len(loader))
        
        for i in range(num_samples_to_show):
            print(f"Sample {i + 1}:")
            print("-" * 60)
            
            sample = loader.get_item(i)
            
            print(f"  Sample ID: {sample['sample_id']}")
            print(f"  Video Path: {sample['media_path']}")
            print(f"  Media Type: {sample['media_type']}")
            
            annotations = sample['annotations']
            moment = annotations['moment']
            
            print(f"  Description: {moment['description']}")
            print(f"  Timestamp (seconds): [{moment['timestamp_sec'][0]}, {moment['timestamp_sec'][1]}]")
            print(f"  Segment indices: {moment['segment_indices']}")
            
            # Show reference description if available
            if 'reference_description' in annotations and annotations['reference_description']:
                print(f"  Reference: {annotations['reference_description']}")
            
            # Show context if available
            if 'context' in annotations and annotations['context']:
                print(f"  Context: {annotations['context']}")
            
            # Show all annotator timestamps if available
            if 'all_annotations' in annotations:
                print(f"  Number of annotators: {len(annotations['all_annotations'])}")
                if len(annotations['all_annotations']) > 1:
                    print("  All annotator timestamps:")
                    for j, ann in enumerate(annotations['all_annotations'][:3]):
                        print(f"    Annotator {j+1}: {ann['timestamp_sec']} seconds")
            
            print()
        
        # Example: Find all moments in a specific time range
        print("Example: Finding moments in first 10 seconds of videos...")
        early_moments = []
        for idx in range(min(100, len(loader))):  # Check first 100 samples
            sample = loader.get_item(idx)
            moment = sample['annotations']['moment']
            if moment['timestamp_sec'][0] < 10:  # Start time < 10 seconds
                early_moments.append({
                    'description': moment['description'],
                    'time': moment['timestamp_sec']
                })
        
        print(f"Found {len(early_moments)} moments in first 10 seconds")
        if early_moments:
            print(f"Example: '{early_moments[0]['description']}' at {early_moments[0]['time']}")
        print()
        
        print("Dataset loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is downloaded and paths are correct.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()