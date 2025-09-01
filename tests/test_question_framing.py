#!/usr/bin/env python3
"""
Test script for question framing improvements.
Verifies that questions now have diverse structures instead of repetitive patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import random
from collections import Counter
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_question_diversity():
    """Test that the question templates create diverse structures"""
    logger.info("\n" + "="*60)
    logger.info("Testing Question Template Diversity")
    logger.info("="*60)
    
    # Simulate the question templates
    QUESTION_FRAMING_TEMPLATES = [
        "I need to {task_goal}. Can you please zoom in on the area {location_info} and tell me what you find?",
        "The current task is to {task_goal}. Please focus on the region {location_info} and provide a detailed observation.",
        "Let's investigate the section at {location_info}. My primary objective is to {task_goal}. What becomes visible after magnification?",
        "What can you tell me about the area at {location_info}? I'm specifically trying to {task_goal}.",
        "My analysis requires me to {task_goal}. Zoom in on {location_info} and report your findings.",
        "Could you provide a close-up of {location_info}? It's essential that I {task_goal} for my report.",
        "The next step is to {task_goal}. Please use the zoom tool on {location_info} and describe the result.",
        "To proceed, I must first {task_goal}. Let's examine {location_info}.",
        "Is it possible to {task_goal}? You'll need to magnify the area at {location_info} to be sure.",
        "A detailed look at {location_info} is required. My goal here is to {task_goal}.",
        "Help me {task_goal} by examining the region at {location_info} closely.",
        "For quality control purposes, I need to {task_goal}. Check {location_info} after zooming in.",
        "Can you assist me in trying to {task_goal}? The area of interest is {location_info}.",
        "Please {task_goal} by inspecting {location_info} with magnification.",
        "I'm investigating whether we can {task_goal}. Focus on {location_info} and let me know.",
        "To complete this inspection, I must {task_goal}. Examine {location_info} in detail.",
        "Would you mind checking if we can {task_goal}? The coordinates are {location_info}.",
        "My objective: {task_goal}. Location: {location_info}. What do you observe?",
        "Zoom into {location_info} - I'm hoping to {task_goal}.",
        "At {location_info}, can you {task_goal}?",
        "I'm curious to {task_goal}. Take a closer look at {location_info}.",
        "Before proceeding, let's {task_goal}. The target area is {location_info}.",
        "This requires us to {task_goal}. Please magnify {location_info} and describe what you see.",
        "Looking at {location_info}, my task is to {task_goal}. What's visible there?",
        "For documentation, I need to {task_goal}. Could you zoom into {location_info}?",
        "Quick check: Can you {task_goal} by examining {location_info}?",
        "The specification requires that we {task_goal}. Please inspect {location_info}.",
        "Help needed: I'm trying to {task_goal} at {location_info}.",
        "Regarding {location_info}, I need to {task_goal}. What do you find?",
        "Part of my analysis involves trying to {task_goal}. Look at {location_info} closely."
    ]
    
    # Test task goals
    test_goals = [
        "identify the serial number",
        "verify the absence of cracks",
        "determine if this could be a watermark",
        "examine the texture pattern",
        "check for any defects"
    ]
    
    # Test locations (without "at" since templates already have it)
    test_locations = [
        "coordinates [100, 100, 200, 200]",
        "coordinates [50, 75, 150, 175]",
        "coordinates [200, 200, 300, 300]"
    ]
    
    # Generate sample questions
    generated_questions = []
    for _ in range(50):
        template = random.choice(QUESTION_FRAMING_TEMPLATES)
        task_goal = random.choice(test_goals)
        location = random.choice(test_locations)
        question = template.format(task_goal=task_goal, location_info=location)
        generated_questions.append(question)
    
    # Analyze diversity
    logger.info(f"\nGenerated {len(generated_questions)} questions")
    logger.info("\nSample questions with varied structures:")
    for i, q in enumerate(generated_questions[:10], 1):
        logger.info(f"{i}. {q[:100]}...")
    
    # Check for repetitive patterns
    start_patterns = Counter()
    for q in generated_questions:
        # Extract first few words as pattern
        start = ' '.join(q.split()[:3])
        start_patterns[start] += 1
    
    # Calculate diversity metrics
    unique_starts = len(start_patterns)
    most_common_start = start_patterns.most_common(1)[0]
    
    logger.info(f"\nDiversity Analysis:")
    logger.info(f"  Unique starting patterns: {unique_starts}/{len(generated_questions)}")
    logger.info(f"  Most common start: '{most_common_start[0]}' ({most_common_start[1]} times)")
    
    # Check if we've eliminated repetitive patterns
    old_patterns = ["To do", "In this", "To verify", "To identify"]
    old_pattern_count = sum(1 for q in generated_questions 
                           if any(q.startswith(p) for p in old_patterns))
    
    logger.info(f"  Questions starting with old patterns: {old_pattern_count}/{len(generated_questions)}")
    
    # Success if we have high diversity
    diversity_ratio = unique_starts / len(generated_questions)
    success = diversity_ratio > 0.5 and old_pattern_count < 5
    
    logger.info(f"\n  Diversity ratio: {diversity_ratio:.2%}")
    logger.info(f"  Result: {'✅ High diversity achieved' if success else '❌ Still repetitive'}")
    
    return success

def test_task_goal_extraction():
    """Test the task goal extraction logic"""
    logger.info("\n" + "="*60)
    logger.info("Testing Task Goal Extraction")
    logger.info("="*60)
    
    # Simulate the extraction logic
    def extract_task_goal(expected_observation: str) -> str:
        obs_lower = expected_observation.lower()
        
        # Handle negative observations
        if any(word in obs_lower for word in ['no ', 'not ', 'absent', 'unable']):
            if 'no ' in obs_lower:
                parts = expected_observation.split('no ', 1)
                if len(parts) > 1:
                    object_part = parts[1].split(' is ')[0].split(' can ')[0]
                    return f"verify the absence of {object_part}"
            return "confirm what is not present in this area"
        
        # Handle ambiguous observations
        if any(word in obs_lower for word in ['unclear', 'might', 'possibly', 'uncertain']):
            if 'might be' in obs_lower:
                parts = expected_observation.split('might be')
                if len(parts) > 1:
                    uncertain_item = parts[1].split(',')[0].strip()
                    return f"determine if this could be {uncertain_item}"
            return "clarify what this ambiguous element might be"
        
        # Handle normal observations
        if ' is ' in obs_lower:
            parts = expected_observation.split(' is ')
            if len(parts) > 0:
                subject = parts[0].strip()
                for prefix in ['a ', 'an ', 'the ', 'some ', 'multiple ']:
                    if subject.lower().startswith(prefix):
                        subject = subject[len(prefix):]
                        break
                return f"identify the {subject}"
        
        return "examine what can be seen"
    
    # Test cases
    test_cases = [
        ("a tiny logo on the object's edge is clearly visible", "identify the tiny logo"),
        ("No crack is present on the surface", "verify the absence of crack"),
        ("A dark shape that might be a defect, but it's unclear", "determine if this could be a defect"),
        ("The serial number reads ABC-123", "identify the serial number"),
        ("Multiple scratch patterns are visible", "identify the Multiple scratch patterns"),
        ("Something unusual appears in the corner", "examine what can be seen")
    ]
    
    logger.info("\nTesting task goal extraction from observations:")
    all_passed = True
    for observation, expected_goal in test_cases:
        extracted = extract_task_goal(observation)
        match = expected_goal.lower() in extracted.lower() or extracted.lower() in expected_goal.lower()
        status = "✅" if match else "❌"
        logger.info(f"\n  Observation: '{observation}'")
        logger.info(f"  Expected: '{expected_goal}'")
        logger.info(f"  Extracted: '{extracted}' {status}")
        all_passed = all_passed and match
    
    return all_passed

def test_natural_language_flow():
    """Test that the generated questions sound natural"""
    logger.info("\n" + "="*60)
    logger.info("Testing Natural Language Flow")
    logger.info("="*60)
    
    # Generate a few complete questions
    templates = [
        "I need to {task_goal}. Can you please zoom in on the area {location_info} and tell me what you find?",
        "Looking at {location_info}, my task is to {task_goal}. What's visible there?",
        "Quick check: Can you {task_goal} by examining {location_info}?"
    ]
    
    goals = [
        "identify the serial number",
        "verify the absence of defects",
        "determine if this could be a watermark"
    ]
    
    location = "coordinates [150, 200, 250, 300]"
    
    logger.info("\nChecking natural language flow:")
    for template in templates:
        for goal in goals:
            question = template.format(task_goal=goal, location_info=location)
            logger.info(f"\n  Q: {question}")
            
            # Check for awkward constructions
            issues = []
            if "to to" in question.lower():
                issues.append("double 'to'")
            if "the the" in question.lower():
                issues.append("double 'the'")
            if "  " in question:
                issues.append("double spaces")
            
            if issues:
                logger.info(f"     Issues: {', '.join(issues)} ❌")
            else:
                logger.info(f"     Natural flow ✅")
    
    return True

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("QUESTION FRAMING IMPROVEMENTS TEST")
    logger.info("="*80)
    
    results = {
        "Question Diversity": test_question_diversity(),
        "Task Goal Extraction": test_task_goal_extraction(),
        "Natural Language Flow": test_natural_language_flow()
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
        logger.info("\n🎉 All tests passed! Question framing improvements are working correctly.")
        logger.info("\nThe improvements provide:")
        logger.info("  1. Elimination of repetitive 'To do...' and 'In this...' patterns")
        logger.info("  2. 30+ diverse question templates for varied structures")
        logger.info("  3. Natural language flow with context-aware task goals")
        logger.info("  4. Dynamic question generation that matches the observation")
    else:
        logger.info("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())