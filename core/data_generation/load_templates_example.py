#!/usr/bin/env python3
"""
Example of how to load templates from the YAML file instead of hardcoding them.
You can integrate this into detail_perception.py if you prefer external configuration.
"""

import yaml
import os
from typing import List

def load_question_templates(yaml_path: str = None) -> List[str]:
    """
    Load question templates from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file. If None, uses default location.
        
    Returns:
        List of question template strings
    """
    if yaml_path is None:
        # Default path relative to this file
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            'question_templates.yaml'
        )
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('templates', [])

# Example usage in DetailPerceptionTaskGenerator:
# Instead of hardcoding QUESTION_FRAMING_TEMPLATES, you could do:
#
# class DetailPerceptionTaskGenerator(BaseDataGenerator):
#     def __init__(self, ...):
#         super().__init__(...)
#         # Load templates from YAML file
#         self.QUESTION_FRAMING_TEMPLATES = load_question_templates()
#
# Or keep both hardcoded and external templates:
#         self.QUESTION_FRAMING_TEMPLATES = [
#             # ... existing hardcoded templates ...
#         ] + load_question_templates()  # Add external templates

if __name__ == "__main__":
    # Test loading templates
    templates = load_question_templates()
    print(f"Loaded {len(templates)} templates from YAML file")
    print("\nFirst 3 templates:")
    for i, template in enumerate(templates[:3], 1):
        print(f"{i}. {template}")