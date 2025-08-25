#!/usr/bin/env python3
"""
Example usage of the Assembly101Loader.

This script demonstrates how to use the Assembly101Loader to load
and iterate through the Assembly101 dataset for action segmentation in videos.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dataloaders.assembly101_loader import Assembly101Loader


def main():
    """Main function to demonstrate loader usage."""
    
    # Configuration for the loader
    # Update these paths to match your actual dataset location
    config = {
        "name": "assembly101_train",
        "type": "TimedActionVideo",
        "path": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/Assembly101/videos",
        "annotation_file": "/mnt/c/Users/ClayKa/Pixelis/datasets/SELECT-FRAME/Assembly101/annotations/train.csv"
    }
    
    print("Initializing Assembly101Loader...")
    print()
    
    try:
        # Initialize the loader
        loader = Assembly101Loader(config)
        
        print(f"Successfully loaded dataset with {len(loader)} action segments")
        print()
        
        # Load and display information about the first few samples
        num_samples_to_show = min(3, len(loader))
        
        for i in range(num_samples_to_show):
            print(f"Sample {i + 1}:")
            print("-" * 70)
            
            sample = loader.get_item(i)
            
            print(f"  Sample ID: {sample['sample_id']}")
            print(f"  Video Path: {sample['media_path']}")
            print(f"  Media Type: {sample['media_type']}")
            
            annotations = sample['annotations']
            action_segment = annotations['action_segment']
            
            # Display action segment information
            print(f"  Frame Range: [{action_segment['start_frame']}, {action_segment['end_frame']}]")
            print(f"  Duration: {action_segment['duration_frames']} frames")
            
            # Display action information if available
            if 'action' in action_segment:
                print(f"  Action: {action_segment['action']}")
            
            if 'verb' in action_segment and 'noun' in action_segment:
                print(f"  Verb-Noun: {action_segment['verb']} - {action_segment['noun']}")
            
            # Display action IDs if available
            if 'action_id' in action_segment:
                print(f"  Action ID: {action_segment['action_id']}")
            
            # Display toy information if available
            if 'toy_id' in action_segment:
                print(f"  Toy ID: {action_segment['toy_id']}")
            
            if 'toy_name' in action_segment:
                print(f"  Toy Name: {action_segment['toy_name']}")
            
            # Display metadata
            if 'is_RGB' in annotations:
                print(f"  Is RGB: {annotations['is_RGB']}")
            
            if 'is_shared' in annotations:
                print(f"  Is Shared: {annotations['is_shared']}")
            
            print()
        
        # Example: Find all "pick up" actions
        print("Example: Finding all 'pick up' actions...")
        pickup_actions = []
        
        # Sample first 100 segments for demonstration
        for idx in range(min(100, len(loader))):
            sample = loader.get_item(idx)
            action_segment = sample['annotations']['action_segment']
            
            if 'verb' in action_segment and action_segment['verb'] == 'pick up':
                pickup_actions.append({
                    'action': action_segment.get('action', 'N/A'),
                    'noun': action_segment.get('noun', 'N/A'),
                    'frames': [action_segment['start_frame'], action_segment['end_frame']]
                })
        
        print(f"Found {len(pickup_actions)} 'pick up' actions in first 100 segments")
        if pickup_actions:
            print(f"Example: '{pickup_actions[0]['action']}' at frames {pickup_actions[0]['frames']}")
        print()
        
        # Example: Calculate average action duration
        print("Example: Calculating average action duration...")
        total_duration = 0
        count = min(100, len(loader))
        
        for idx in range(count):
            sample = loader.get_item(idx)
            action_segment = sample['annotations']['action_segment']
            total_duration += action_segment['duration_frames']
        
        avg_duration = total_duration / count if count > 0 else 0
        print(f"Average action duration (first {count} segments): {avg_duration:.2f} frames")
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