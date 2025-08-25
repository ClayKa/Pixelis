#!/usr/bin/env python3
"""
Example usage of the ActivityNetCaptionsLoader.

This script demonstrates how to use the ActivityNetCaptionsLoader to load
and iterate through the ActivityNet Captions dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataloaders.activitynet_captions_loader import ActivityNetCaptionsLoader


def main():
    """Main function to demonstrate loader usage."""
    
    # Configuration for the loader
    # Update these paths to match your actual dataset location
    config = {
        "name": "activitynet_captions_train",
        "type": "DenseVideoCaptioning",
        "path": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/ActivityNetCaptions/videos",
        "annotation_file": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/ActivityNetCaptions/annotations/activitynet_captions_train.json"
    }
    
    print("Initializing ActivityNetCaptionsLoader...")
    
    try:
        # Initialize the loader
        loader = ActivityNetCaptionsLoader(config)
        
        print(f"Successfully loaded dataset with {len(loader)} samples")
        print()
        
        # Load and display information about the first few samples
        num_samples_to_show = min(3, len(loader))
        
        for i in range(num_samples_to_show):
            print(f"Sample {i + 1}:")
            print("-" * 50)
            
            sample = loader.get_item(i)
            
            print(f"  Video ID: {sample['sample_id']}")
            print(f"  Video Path: {sample['media_path']}")
            print(f"  Media Type: {sample['media_type']}")
            
            annotations = sample['annotations']
            print(f"  Duration: {annotations.get('duration_sec', 'N/A')} seconds")
            
            # Display timed events
            events = annotations.get('timed_events', [])
            print(f"  Number of events: {len(events)}")
            
            for j, event in enumerate(events[:2]):  # Show first 2 events
                timestamp = event['timestamp_sec']
                description = event['description']
                print(f"    Event {j + 1}: [{timestamp[0]:.1f}s - {timestamp[1]:.1f}s]")
                print(f"      Description: {description}")
            
            if len(events) > 2:
                print(f"    ... and {len(events) - 2} more events")
            
            print()
        
        # Example: Iterate through all samples (commented out for performance)
        # for idx in range(len(loader)):
        #     sample = loader.get_item(idx)
        #     # Process sample...
        
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