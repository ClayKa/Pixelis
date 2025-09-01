#!/usr/bin/env python3
"""
Test why templates aren't being used by the LLM
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read the prompt to see what the LLM actually receives
with open('prompts/detail_perception.md', 'r') as f:
    prompt = f.read()

print("="*60)
print("ANALYZING WHY TEMPLATES AREN'T BEING USED")
print("="*60)

# Check if the placeholders are in the prompt
placeholders = [
    '{style_name}',
    '{style_description}', 
    '{example_question}',
    '{example_answer}'
]

print("\n1. CHECKING IF PLACEHOLDERS EXIST IN PROMPT:")
for p in placeholders:
    if p in prompt:
        print(f"  ✅ {p} - Found in prompt")
        # Find where it appears
        lines = prompt.split('\n')
        for i, line in enumerate(lines):
            if p in line:
                print(f"     Line {i+1}: {line.strip()[:80]}...")
                break
    else:
        print(f"  ❌ {p} - NOT found in prompt")

# Check what the LLM is being instructed to do
print("\n2. CHECKING INSTRUCTIONS ABOUT STYLE GUIDELINE:")
if "STYLE GUIDELINE FOR THIS SAMPLE" in prompt:
    print("  ✅ Style guideline section exists")
    
    # Find the section
    start = prompt.find("**## STYLE GUIDELINE FOR THIS SAMPLE:**")
    if start > 0:
        section = prompt[start:start+500]
        print("\n  Content of style guideline section:")
        print("  " + "-"*50)
        for line in section.split('\n')[:10]:
            print(f"  {line}")
else:
    print("  ❌ Style guideline section NOT found")

# Check if there's instruction to use the example
print("\n3. CHECKING INSTRUCTIONS TO USE EXAMPLE:")
important_instructions = [
    "Use the example question pattern",
    "DO NOT copy it verbatim",
    "create your own unique variation",
    "follow the style guideline"
]

for instruction in important_instructions:
    if instruction in prompt:
        print(f"  ✅ Found: '{instruction}'")
    else:
        print(f"  ❌ Missing: '{instruction}'")

# The real problem
print("\n" + "="*60)
print("THE CORE PROBLEM:")
print("="*60)
print("\nThe LLM receives:")
print("1. ✅ 40 hardcoded style examples with full questions in the cookbook")
print("2. ✅ The style guideline section with placeholders")
print("3. ✅ Instructions to use the example pattern")
print("\nBUT the LLM is likely:")
print("- Following the hardcoded examples more strongly (they appear first)")
print("- The 40 detailed examples (lines 25-506) override the later instructions")
print("- The cookbook examples are more concrete than the abstract template")
print("\nSOLUTION:")
print("Either:")
print("1. Remove/reduce the hardcoded cookbook examples")
print("2. Make the style guideline instructions MUCH stronger")
print("3. Move the style guideline BEFORE the cookbook examples")
print("="*60)