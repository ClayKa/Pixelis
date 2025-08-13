#!/usr/bin/env python3
"""
Test script for CoTA data preprocessing with difficulty scoring
"""

import json
import tempfile
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess_data import CoTAPreprocessor, DifficultyScorer
from scripts.generate_cota_data import CoTADataGenerator

def generate_test_data():
    """Generate small test dataset"""
    # Create minimal test annotations
    test_annotations = [
        {
            "image_path": "test_image_1.jpg",
            "source_dataset": "test_dataset",
            "original_id": "test_001",
            "annotations": [
                {"category": "cat", "bbox": [10, 10, 50, 50]},
                {"category": "dog", "bbox": [100, 100, 50, 50]},
                {"category": "cat", "bbox": [200, 200, 50, 50]}
            ],
            "text_annotations": [
                {"bbox": [50, 50, 100, 30], "text": "Hello World", "description": "top region"}
            ]
        }
    ]
    
    # Generate samples
    generator = CoTADataGenerator()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_annotations, f)
        annotations_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    # Generate 20 test samples
    stats = generator.generate_dataset(
        test_annotations,
        num_samples=20,
        output_path=output_file
    )
    
    return output_file

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("=" * 60)
    print("Testing CoTA Data Preprocessing")
    print("=" * 60)
    
    # Generate test data
    print("\n1. Generating test data...")
    test_data_file = generate_test_data()
    print(f"   Generated test data at: {test_data_file}")
    
    # Create preprocessor
    print("\n2. Initializing preprocessor...")
    preprocessor = CoTAPreprocessor()
    
    # Process the data
    print("\n3. Processing samples...")
    processed_samples = preprocessor.process_dataset(test_data_file)
    print(f"   Processed {len(processed_samples)} samples")
    
    # Display sample results
    print("\n4. Sample difficulty scores:")
    for i, sample in enumerate(processed_samples[:5]):
        print(f"   Sample {i+1}:")
        print(f"     - Task type: {sample.task_type}")
        print(f"     - Sample type: {sample.sample_type}")
        print(f"     - Difficulty score: {sample.difficulty_score:.3f}")
        print(f"     - Category: {sample.difficulty_category}")
        print(f"     - Trajectory length: {sample.difficulty_metrics.trajectory_length}")
    
    # Display distribution
    print("\n5. Difficulty distribution:")
    for category, count in preprocessor.scorer.category_counts.items():
        percentage = count / len(processed_samples) * 100
        print(f"   {category}: {count} samples ({percentage:.1f}%)")
    
    # Test difficulty scorer directly
    print("\n6. Testing difficulty scorer components:")
    scorer = DifficultyScorer()
    
    # Test trajectory with various complexities
    test_trajectories = [
        # Simple trajectory
        [
            {"action": "THINK", "thought": "Looking at image", "parameters": {}},
            {"action": "READ_TEXT", "parameters": {"bbox": [0, 0, 100, 100]}}
        ],
        # Complex trajectory
        [
            {"action": "THINK", "thought": "Analyzing scene", "parameters": {}},
            {"action": "SEGMENT_OBJECT_AT", "parameters": {"coordinates": [50, 50]}},
            {"action": "GET_PROPERTIES", "parameters": {"mask": "mask_1"}},
            {"action": "TRACK_OBJECT", "parameters": {"object_id": 1}},
            {"action": "THINK", "thought": "Comparing results", "parameters": {}},
        ],
        # Self-correction trajectory
        [
            {"action": "THINK", "thought": "Starting analysis", "parameters": {}},
            {"action": "SEGMENT_OBJECT_AT", "parameters": {"coordinates": [10, 10]}},
            {"action": "THINK", "thought": "That doesn't seem right, let me try again", "parameters": {}},
            {"action": "SEGMENT_OBJECT_AT", "parameters": {"coordinates": [50, 50]}},
        ]
    ]
    
    for i, traj in enumerate(test_trajectories):
        complexity = scorer.calculate_trajectory_complexity(traj)
        sophistication, _, _ = scorer.calculate_operation_sophistication(traj)
        reasoning, _ = scorer.calculate_reasoning_depth(traj)
        
        print(f"\n   Trajectory {i+1}:")
        print(f"     - Length: {len(traj)}")
        print(f"     - Complexity score: {complexity:.3f}")
        print(f"     - Sophistication score: {sophistication:.3f}")
        print(f"     - Reasoning depth: {reasoning:.3f}")
    
    # Save test results
    print("\n7. Saving processed data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "processed"
        preprocessor.save_processed_data(output_dir, split_by_category=True)
        
        # Check saved files
        saved_files = list(output_dir.glob("*.json"))
        print(f"   Saved {len(saved_files)} files:")
        for file in saved_files:
            print(f"     - {file.name}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    # Clean up
    Path(test_data_file).unlink()

if __name__ == "__main__":
    test_preprocessing()