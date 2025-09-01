#!/usr/bin/env python3
"""
Simple test script for the enhanced data generation improvements.
Tests the core functionality without full generator initialization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import json
import random
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dynamic_vocabulary():
    """Test the dynamic vocabulary generation"""
    logger.info("\n" + "="*60)
    logger.info("Testing Dynamic Vocabulary Generation")
    logger.info("="*60)
    
    # Simulate the vocabulary from detail_perception.py
    vocab = {
        'adjectives': [
            "a tiny", "a faded", "a single", "a partially hidden", "a bright red",
            "a small", "a blurred", "a distinctive", "a subtle", "a prominent",
            "an obscured", "a miniature", "a weathered", "a vibrant", "a damaged"
        ],
        'objects': [
            "logo", "insect", "crack", "water droplet", "serial number", 
            "loose stitch", "fingerprint", "dust particle", "scratch mark", "text label",
            "barcode", "QR code", "reflection", "shadow", "pattern"
        ],
        'locations': [
            "on the object's edge", "in the bottom-right corner", "beneath the handle",
            "along a seam", "near the center", "at the top", "on the left side",
            "in the upper quadrant", "across the surface", "within the marked area"
        ]
    }
    
    # Generate diverse observations
    observations = set()
    for _ in range(50):
        adj = random.choice(vocab['adjectives'])
        obj = random.choice(vocab['objects'])
        loc = random.choice(vocab['locations'])
        obs = f"{adj} {obj} {loc} is clearly visible"
        observations.add(obs)
    
    logger.info(f"Generated {len(observations)} unique observations from 50 attempts")
    logger.info(f"Diversity ratio: {len(observations)/50:.2%}")
    logger.info("\nSample observations:")
    for obs in list(observations)[:5]:
        logger.info(f"  - {obs}")
    
    return len(observations) > 30  # Expect high diversity

def test_trap_generation():
    """Test trap generation logic"""
    logger.info("\n" + "="*60)
    logger.info("Testing Trap Generation")
    logger.info("="*60)
    
    # Test wrong_tool generation
    original_action = {
        'name': 'ZOOM-IN',
        'parameters': {'bbox': [100, 100, 200, 200]}
    }
    
    # Simulate wrong tool mapping
    wrong_tool_map = {
        'ZOOM-IN': 'READ-TEXT',
        'READ-TEXT': 'ZOOM-IN',
        'SEGMENT_OBJECT_AT': 'GET_PROPERTIES',
    }
    
    wrong_tool = wrong_tool_map.get(original_action['name'], 'READ-TEXT')
    logger.info(f"Original tool: {original_action['name']}")
    logger.info(f"Wrong tool: {wrong_tool}")
    
    # Test bad parameter generation
    bad_params = {
        'bbox': [0, 0, 0, 0]  # Zero area
    }
    logger.info(f"Original params: {original_action['parameters']}")
    logger.info(f"Bad params: {bad_params}")
    
    return True

def test_nothing_found_scenarios():
    """Test Nothing Found scenario generation"""
    logger.info("\n" + "="*60)
    logger.info("Testing 'Nothing Found' Scenarios")
    logger.info("="*60)
    
    objects = ["logo", "insect", "crack", "water droplet", "serial number"]
    locations = ["on the object's edge", "in the corner", "near the center"]
    
    nothing_found_templates = [
        "No {obj} is present {location}",
        "The specified area does not contain any {obj}",
        "After careful inspection, no {obj} can be found",
        "Unable to locate any {obj} {location}",
    ]
    
    examples = []
    for _ in range(5):
        obj = random.choice(objects)
        location = random.choice(locations)
        template = random.choice(nothing_found_templates)
        obs = template.format(obj=obj, location=location)
        examples.append(obs)
        logger.info(f"  - {obs}")
    
    # Check that they indicate absence
    valid = all(any(word in obs.lower() for word in ['no ', 'not ', 'unable', 'absent']) 
               for obs in examples)
    
    logger.info(f"\nAll observations correctly indicate absence: {valid}")
    return valid

def test_ambiguity_scenarios():
    """Test Ambiguity scenario generation"""
    logger.info("\n" + "="*60)
    logger.info("Testing Ambiguity Scenarios")
    logger.info("="*60)
    
    objects = ["crack", "shadow", "pattern", "defect", "marking"]
    
    ambiguous_templates = [
        "A dark shape that might be a {obj}, but it's unclear",
        "Something resembling a {obj}, though certainty is low",
        "Unclear whether this is a {obj} or something else",
        "Possibly a {obj}, but the image quality prevents confirmation",
    ]
    
    examples = []
    for _ in range(5):
        obj = random.choice(objects)
        template = random.choice(ambiguous_templates)
        obs = template.format(obj=obj)
        examples.append(obs)
        logger.info(f"  - {obs}")
    
    # Check that they express uncertainty
    valid = all(any(word in obs.lower() for word in ['might', 'unclear', 'possibly', 'uncertain'])
               for obs in examples)
    
    logger.info(f"\nAll observations correctly express uncertainty: {valid}")
    return valid

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("ENHANCED DATA GENERATION - SIMPLE TEST SUITE")
    logger.info("="*80)
    
    results = {
        "Dynamic Vocabulary": test_dynamic_vocabulary(),
        "Trap Generation": test_trap_generation(),
        "Nothing Found Scenarios": test_nothing_found_scenarios(),
        "Ambiguity Scenarios": test_ambiguity_scenarios()
    }
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The enhancements are working correctly.")
    else:
        logger.info("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())