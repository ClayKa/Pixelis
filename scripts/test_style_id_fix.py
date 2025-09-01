#!/usr/bin/env python3
"""
Test script to verify that the style_id fix is working correctly.
This tests that diverse style_ids are being assigned to generated samples.
"""

import sys
import json
import logging
from pathlib import Path
from collections import Counter
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_style_parsing():
    """Test that style cookbook parsing is working."""
    logger.info("="*60)
    logger.info("Testing Style Cookbook Parsing")
    logger.info("="*60)
    
    # Create a minimal config
    config = {
        'name': 'detail_perception_test',
        'generator_params': {},
        'prompt_template': '/dev/null',  # Will use fallback styles
        'generator_config': {'validation_strictness': 'ultra_lenient'}
    }
    
    # Initialize generator
    generator = DetailPerceptionTaskGenerator(
        loaders={},
        config=config,
        global_config={}
    )
    
    # Check that styles were parsed or fallback was used
    logger.info(f"Number of styles in cookbook: {len(generator.style_cookbook)}")
    
    if generator.style_cookbook:
        logger.info("Sample styles:")
        for i, style in enumerate(generator.style_cookbook[:3]):
            logger.info(f"  Style {i+1}:")
            logger.info(f"    ID: {style.get('style_id', 'MISSING')}")
            logger.info(f"    Name: {style.get('name', 'MISSING')}")
            logger.info(f"    Desc: {style.get('desc', 'N/A')[:50]}...")
        
        # Check that all styles have style_id
        missing_ids = [s.get('name', 'Unknown') for s in generator.style_cookbook if 'style_id' not in s]
        if missing_ids:
            logger.warning(f"Styles missing style_id: {missing_ids}")
        else:
            logger.info("✓ All styles have style_id")
    else:
        logger.error("No styles found in cookbook!")
        return False
    
    return True


def test_style_id_distribution():
    """Test that different style_ids are being selected."""
    logger.info("\n" + "="*60)
    logger.info("Testing Style ID Distribution")
    logger.info("="*60)
    
    # Create generator
    config = {
        'name': 'detail_perception_test',
        'generator_params': {},
        'prompt_template': '/dev/null',
        'generator_config': {'validation_strictness': 'ultra_lenient'}
    }
    
    generator = DetailPerceptionTaskGenerator(
        loaders={},
        config=config,
        global_config={}
    )
    
    # Simulate multiple context generations
    style_ids = []
    style_names = []
    
    for i in range(50):
        try:
            placeholders, metadata = generator._build_context_placeholders()
            
            # Check metadata contains style_id
            if 'style_id' in metadata:
                style_ids.append(metadata['style_id'])
                style_names.append(metadata.get('style_used', 'Unknown'))
            else:
                logger.warning(f"Sample {i+1}: No style_id in metadata!")
        except Exception as e:
            logger.error(f"Error generating sample {i+1}: {e}")
    
    # Analyze distribution
    if style_ids:
        style_counter = Counter(style_ids)
        unique_styles = len(style_counter)
        
        logger.info(f"\nGenerated {len(style_ids)} samples")
        logger.info(f"Unique style_ids used: {unique_styles}")
        logger.info(f"Style ID distribution:")
        
        for style_id, count in sorted(style_counter.items()):
            percentage = (count / len(style_ids)) * 100
            # Find the name for this style_id
            style_name = "Unknown"
            for i, sid in enumerate(style_ids):
                if sid == style_id:
                    style_name = style_names[i]
                    break
            logger.info(f"  Style {style_id} ({style_name}): {count} times ({percentage:.1f}%)")
        
        # Check if we have good diversity
        if unique_styles == 1:
            logger.error("✗ PROBLEM: All samples have the same style_id!")
            return False
        elif unique_styles < 3:
            logger.warning(f"⚠ Limited diversity: Only {unique_styles} different styles used")
            return False
        else:
            logger.info(f"✓ Good diversity: {unique_styles} different styles used")
            return True
    else:
        logger.error("No style_ids collected!")
        return False


def test_metadata_structure():
    """Test that metadata is correctly structured with style information."""
    logger.info("\n" + "="*60)
    logger.info("Testing Metadata Structure")
    logger.info("="*60)
    
    # Create generator
    config = {
        'name': 'detail_perception_test',
        'generator_params': {},
        'prompt_template': '/dev/null',
        'generator_config': {'validation_strictness': 'ultra_lenient'}
    }
    
    generator = DetailPerceptionTaskGenerator(
        loaders={},
        config=config,
        global_config={}
    )
    
    # Generate a sample
    try:
        placeholders, metadata = generator._build_context_placeholders()
        
        logger.info("Metadata fields:")
        for key, value in metadata.items():
            if key == 'generation_timestamp':
                logger.info(f"  {key}: <timestamp>")
            else:
                logger.info(f"  {key}: {value}")
        
        # Check required fields
        required_fields = ['style_id', 'style_used', 'difficulty']
        missing_fields = [f for f in required_fields if f not in metadata]
        
        if missing_fields:
            logger.error(f"✗ Missing required fields: {missing_fields}")
            return False
        else:
            logger.info("✓ All required fields present")
            
            # Verify style_id is not 0 (unless it's actually style 0)
            if metadata['style_id'] == 0:
                logger.warning("Note: style_id is 0 - verify this is intentional")
            
            return True
            
    except Exception as e:
        logger.error(f"Error generating metadata: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Testing Style ID Fix")
    logger.info("="*60)
    
    results = []
    
    # Test 1: Style parsing
    results.append(("Style Parsing", test_style_parsing()))
    
    # Test 2: Style ID distribution
    results.append(("Style ID Distribution", test_style_id_distribution()))
    
    # Test 3: Metadata structure
    results.append(("Metadata Structure", test_metadata_structure()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED - Style ID fix is working!")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED - Style ID issue may persist")
        return 1


if __name__ == "__main__":
    sys.exit(main())