#!/usr/bin/env python3
"""
Test to verify that generation actually uses random selection of templates and styles
"""

import json
import sys
import os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parse the already generated file to see what was actually produced
generated_file = "data_outputs/specialized/detail_perception_task.jsonl"

print("="*60)
print("ANALYZING ACTUAL GENERATION RANDOMNESS")
print("="*60)

if not os.path.exists(generated_file):
    print(f"Error: {generated_file} not found!")
    sys.exit(1)

questions = []
with open(generated_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        questions.append(data['question'])

print(f"\nTotal samples generated: {len(questions)}")
print("\n" + "="*60)
print("ANALYZING QUESTION PATTERNS")
print("="*60)

# Analyze question beginnings (first 30 characters)
beginnings = Counter()
for q in questions:
    beginning = q[:30]
    beginnings[beginning] += 1

print("\nQuestion beginnings (first 30 chars):")
for beginning, count in beginnings.most_common():
    print(f"  '{beginning}...' : {count} times")
    
# Check for exact duplicates
exact_duplicates = [q for q, count in Counter(questions).items() if count > 1]
if exact_duplicates:
    print(f"\n⚠️ Found {len(exact_duplicates)} exact duplicate questions:")
    for dup in exact_duplicates:
        print(f"  - {dup[:80]}...")
else:
    print("\n✅ No exact duplicate questions found!")

# Analyze patterns
print("\n" + "="*60)
print("PATTERN ANALYSIS")
print("="*60)

# Check for common patterns
patterns = {
    "Hey": 0,
    "Can you": 0,
    "I need": 0,
    "Please": 0,
    "The": 0,
    "I'm": 0,
    "In this": 0,
    "Looking": 0,
    "This": 0,
    "Is": 0,
    "What": 0,
    "Could": 0,
    "My": 0,
    "Help": 0,
    "Let": 0,
    "To": 0,
    "For": 0,
    "Quick": 0,
    "Before": 0,
    "Would": 0,
    "At": 0,
    "Part": 0,
    "Regarding": 0,
    "A ": 0,
    "I've": 0,
    "During": 0,
    "It's": 0,
    "While": 0
}

for q in questions:
    for pattern in patterns:
        if q.startswith(pattern):
            patterns[pattern] += 1
            break  # Count each question only once

print("\nStarting word/phrase distribution:")
sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
for pattern, count in sorted_patterns:
    if count > 0:
        percentage = (count / len(questions)) * 100
        print(f"  '{pattern}...': {count} ({percentage:.1f}%)")

# Calculate diversity metrics
unique_beginnings = len(beginnings)
diversity_ratio = unique_beginnings / len(questions) * 100

print("\n" + "="*60)
print("DIVERSITY METRICS")
print("="*60)
print(f"Unique question beginnings: {unique_beginnings}/{len(questions)} ({diversity_ratio:.1f}%)")
print(f"Most common beginning appears: {beginnings.most_common(1)[0][1]} times")

# Check if templates are being used
print("\n" + "="*60)
print("TEMPLATE USAGE CHECK")
print("="*60)

# Load the templates
from core.data_generation.detail_perception import DetailPerceptionTaskGenerator
templates = DetailPerceptionTaskGenerator.QUESTION_FRAMING_TEMPLATES

# Check if questions match template patterns (approximately)
template_matches = 0
for q in questions:
    # Check if the question seems to follow any template structure
    # This is approximate since the LLM fills in the placeholders
    for template in templates:
        # Extract the static parts of the template (before placeholders)
        template_start = template.split('{')[0].strip()
        if template_start and q.startswith(template_start[:20]):
            template_matches += 1
            break

print(f"Questions that appear to follow a template pattern: {template_matches}/{len(questions)}")
print(f"Questions that don't match any template: {len(questions) - template_matches}")

if template_matches < len(questions) * 0.5:
    print("⚠️ WARNING: Less than 50% of questions follow template patterns!")
    print("   The LLM might be ignoring the templates and generating its own questions.")
else:
    print("✅ Most questions appear to follow template patterns.")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if diversity_ratio > 70:
    print("✅ High diversity: Questions show good variation")
elif diversity_ratio > 50:
    print("⚠️ Moderate diversity: Some repetition in question patterns")
else:
    print("❌ Low diversity: Too much repetition in question patterns")

print(f"\nRecommendation:")
if template_matches < len(questions) * 0.5:
    print("- The LLM is not following the templates properly")
    print("- Need to strengthen the prompt to use the provided templates")
else:
    print("- Templates are being used effectively")
    print("- Good randomization is occurring")