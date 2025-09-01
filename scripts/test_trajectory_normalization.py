#!/usr/bin/env python3
"""
Test script for trajectory normalization functionality.
Tests various trajectory formats to ensure they are properly normalized.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generation.base_generator import BaseTaskGenerator
from core.data_generation.detail_perception import DetailPerceptionTaskGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestGenerator(BaseTaskGenerator):
    """Test implementation for accessing normalization method."""
    
    def _build_context_placeholders(self, sample):
        return {}, {}
    
    def _validate_and_process_response(self, llm_response, context):
        return llm_response


def create_test_trajectories():
    """Create various trajectory formats to test normalization."""
    return [
        # Format 1: Standard format with 'type' field
        {
            "name": "Standard format",
            "trajectory": [
                {"type": "thought", "content": "I need to examine the image"},
                {"type": "action", "name": "ZOOM-IN", "parameters": {"bbox": [0, 0, 100, 100]}},
                {"type": "thought", "content": "I can see the details now"}
            ]
        },
        
        # Format 2: Using 'step' field
        {
            "name": "Step field format",
            "trajectory": [
                {"step": "THOUGHT", "text": "Looking at the image"},
                {"step": "ACTION", "action": "SEGMENT_OBJECT_AT", "params": {"x": 50, "y": 50}},
                {"step": "thought", "description": "Found the object"}
            ]
        },
        
        # Format 3: Direct THOUGHT/ACTION fields
        {
            "name": "Direct fields format",
            "trajectory": [
                {"THOUGHT": "Analyzing the scene"},
                {"ACTION": {"name": "READ_TEXT", "parameters": {"region": [10, 10, 90, 90]}}},
                {"thought": "Text has been read"}
            ]
        },
        
        # Format 4: Nested action structure
        {
            "name": "Nested action format",
            "trajectory": [
                {"type": "thought", "message": "Starting analysis"},
                {"action": {"name": "TRACK_OBJECT", "parameters": {"object_id": 1}}},
                {"type": "thought", "content": "Tracking complete"}
            ]
        },
        
        # Format 5: Alternative parameter names
        {
            "name": "Alternative parameter names",
            "trajectory": [
                {"type": "thought", "text": "Beginning task"},
                {"type": "action", "operation": "GET_PROPERTIES", "args": {"object_id": 2}},
                {"type": "action", "command": "SELECT_FRAME", "arguments": {"frame_number": 5}}
            ]
        },
        
        # Format 6: Mixed formats with observations
        {
            "name": "Mixed with observations",
            "trajectory": [
                {"thought": "Let me zoom in", "type": "thought"},
                {
                    "action": "ZOOM_IN",
                    "parameters": {"scale": 2.0},
                    "observation": "Zoomed image shows more detail"
                },
                {"step_type": "thought", "content": "I can see clearly now"}
            ]
        },
        
        # Format 7: Malformed/incomplete steps
        {
            "name": "Malformed steps",
            "trajectory": [
                {"type": "thought"},  # Missing content
                {"random_field": "some_value"},  # No recognizable fields
                {"type": "action"},  # Missing action name
                {"type": "thought", "content": "Valid thought"},
                "not_a_dict",  # Non-dict entry
                {"action": "SEGMENT_OBJECT_AT", "params": {"x": 30, "y": 40}}  # Valid
            ]
        },
        
        # Format 8: Various observation field names
        {
            "name": "Various observation names",
            "trajectory": [
                {
                    "type": "action",
                    "name": "READ_TEXT",
                    "parameters": {"region": [0, 0, 50, 50]},
                    "result": "Text: Hello World"
                },
                {
                    "type": "action",
                    "name": "GET_PROPERTIES",
                    "parameters": {},
                    "output": "Color: red, Size: large"
                },
                {
                    "type": "action",
                    "name": "TRACK_OBJECT",
                    "parameters": {"id": 1},
                    "response": "Object tracked successfully"
                }
            ]
        }
    ]


def test_normalization():
    """Test the trajectory normalization functionality."""
    logger.info("="*60)
    logger.info("Testing Trajectory Normalization")
    logger.info("="*60)
    
    # Create test generator instance with dummy prompt template
    test_gen = TestGenerator(
        loaders={},
        config={
            'name': 'test', 
            'generator_params': {},
            'prompt_template': '/dev/null'  # Dummy path to avoid directory error
        },
        global_config={}
    )
    
    test_cases = create_test_trajectories()
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}: {test_case['name']}")
        logger.info("-" * 40)
        
        trajectory = test_case['trajectory']
        logger.info(f"Original trajectory ({len(trajectory)} steps):")
        for j, step in enumerate(trajectory):
            if isinstance(step, dict):
                logger.info(f"  Step {j+1}: {json.dumps(step, indent=2)[:200]}...")
            else:
                logger.info(f"  Step {j+1}: {step} (non-dict)")
        
        # Normalize the trajectory
        try:
            normalized = test_gen._normalize_trajectory(trajectory)
            logger.info(f"\nNormalized trajectory ({len(normalized)} steps):")
            
            for j, step in enumerate(normalized):
                step_str = f"  Step {j+1}: type='{step.get('type')}'"
                if step.get('type') == 'thought':
                    content = step.get('content', '<missing>')
                    step_str += f", content='{content[:50]}...'" if len(str(content)) > 50 else f", content='{content}'"
                elif step.get('type') == 'action':
                    step_str += f", name='{step.get('name')}'"
                    if 'parameters' in step:
                        step_str += f", params={step['parameters']}"
                    if 'observation' in step:
                        obs = step['observation']
                        step_str += f", obs='{obs[:30]}...'" if len(str(obs)) > 30 else f", obs='{obs}'"
                logger.info(step_str)
            
            # Validation checks
            logger.info("\nValidation:")
            valid = True
            
            # Check all steps have type
            for step in normalized:
                if 'type' not in step:
                    logger.warning(f"  ✗ Step missing 'type': {step}")
                    valid = False
            
            # Check thoughts have content
            thought_steps = [s for s in normalized if s.get('type') == 'thought']
            for step in thought_steps:
                if 'content' not in step:
                    logger.warning(f"  ✗ Thought missing 'content': {step}")
                    valid = False
            
            # Check actions have name
            action_steps = [s for s in normalized if s.get('type') == 'action']
            for step in action_steps:
                if 'name' not in step:
                    logger.warning(f"  ✗ Action missing 'name': {step}")
                    valid = False
            
            if valid:
                logger.info("  ✓ All normalized steps are valid")
            
        except Exception as e:
            logger.error(f"  ✗ Normalization failed: {e}", exc_info=True)
    
    logger.info("\n" + "="*60)
    logger.info("Normalization testing complete!")
    logger.info("="*60)


def test_with_real_generator():
    """Test normalization with a real generator implementation."""
    logger.info("\n" + "="*60)
    logger.info("Testing with DetailPerceptionTaskGenerator")
    logger.info("="*60)
    
    # Create a detail perception generator
    generator = DetailPerceptionTaskGenerator(
        loaders={},
        config={
            'name': 'detail_perception',
            'generator_params': {},
            'generator_config': {'validation_strictness': 'ultra_lenient'}
        },
        global_config={}
    )
    
    # Test sample with mixed format
    test_response = {
        "question": "What details can you see?",
        "trajectory": [
            {"THOUGHT": "I need to zoom in to see details"},
            {"ACTION": "ZOOM-IN", "parameters": {"bbox": [10, 10, 90, 90]}},
            {"step": "thought", "text": "Now I can see the fine details"},
            {"type": "action", "name": "GET_PROPERTIES", "params": {"object_id": 1}}
        ],
        "final_answer": "I can see a red car with scratches"
    }
    
    context = {"difficulty": "Easy", "expected_observation": "red car scratches"}
    
    logger.info("Testing validation with mixed format trajectory...")
    result = generator._validate_and_process_response(test_response, context)
    
    if result:
        logger.info("✓ Validation successful!")
        logger.info(f"Normalized trajectory has {len(result['trajectory'])} steps:")
        for i, step in enumerate(result['trajectory']):
            logger.info(f"  Step {i+1}: {step}")
    else:
        logger.error("✗ Validation failed")


def main():
    """Main test execution."""
    test_normalization()
    test_with_real_generator()
    
    logger.info("\n" + "="*60)
    logger.info("✓ ALL TESTS COMPLETED")
    logger.info("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())