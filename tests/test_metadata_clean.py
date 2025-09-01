#!/usr/bin/env python3
"""Test script to verify metadata fields have been removed."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator

def test_metadata_clean():
    """Test that unwanted metadata fields are not generated."""
    
    # Fields that should NOT be in metadata
    unwanted_fields = [
        'generator_class',
        'api_provider', 
        'generator_version',
        'generation_timestamp'
    ]
    
    # Fields that SHOULD be in metadata
    expected_fields = [
        'task_name',
        'llm_model_used',
        'temperature'
    ]
    
    print("Testing metadata generation...")
    print("-" * 60)
    
    # Create a mock generator
    config = {
        'task_name': 'detail_perception',
        'num_samples': 1,
        'prompt_template': 'prompts/detail_perception.md'  # Full path to template
    }
    
    global_config = {
        'api_profiles': {
            'generator_api': {
                'model': 'test-model',
                'temperature': 0.8,
                'provider': 'test-provider'  # This should NOT appear in metadata
            }
        }
    }
    
    # Initialize generator (will use mock mode)
    generator = DetailPerceptionTaskGenerator(
        loaders={},
        config=config, 
        global_config=global_config
    )
    
    # Generate a mock sample
    samples = generator.generate(1)
    
    if samples:
        sample = samples[0]
        metadata = sample.get('metadata', {})
        
        print("Generated metadata keys:")
        for key in sorted(metadata.keys()):
            print(f"  ✓ {key}: {metadata[key]}")
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS:")
        print("=" * 60)
        
        # Check for unwanted fields
        found_unwanted = []
        for field in unwanted_fields:
            if field in metadata:
                found_unwanted.append(field)
                print(f"  ❌ FAIL: Found unwanted field '{field}'")
        
        # Check for expected fields
        missing_expected = []
        for field in expected_fields:
            if field not in metadata:
                missing_expected.append(field)
                print(f"  ❌ FAIL: Missing expected field '{field}'")
            else:
                print(f"  ✅ PASS: Found expected field '{field}'")
        
        print("\n" + "=" * 60)
        
        if not found_unwanted and not missing_expected:
            print("✅ SUCCESS: Metadata is clean!")
            print("   - No unwanted fields found")
            print("   - All expected fields present")
            return True
        else:
            print("❌ FAILURE: Metadata issues found!")
            if found_unwanted:
                print(f"   - Unwanted fields present: {', '.join(found_unwanted)}")
            if missing_expected:
                print(f"   - Expected fields missing: {', '.join(missing_expected)}")
            return False
    else:
        print("❌ No samples generated")
        return False

if __name__ == "__main__":
    success = test_metadata_clean()
    sys.exit(0 if success else 1)