#!/usr/bin/env python3
"""
Fix prompt templates by escaping braces in JSON examples.
This prevents the template formatter from treating JSON content as placeholders.
"""

import re
from pathlib import Path

def fix_prompt_template(file_path: Path):
    """Fix a single prompt template file."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find JSON blocks (between ```json and ```)
    json_pattern = r'(```json\n)(.*?)(```)'
    
    def escape_json_block(match):
        prefix = match.group(1)
        json_content = match.group(2)
        suffix = match.group(3)
        
        # Check if already escaped (has {{ or }})
        if '{{' in json_content or '}}' in json_content:
            # Already escaped, don't double-escape
            return match.group(0)
        
        # Escape single braces (but not already escaped ones)
        # Replace { with {{ and } with }}
        escaped = json_content.replace('{', '{{').replace('}', '}}')
        
        return prefix + escaped + suffix
    
    # Apply escaping to all JSON blocks
    fixed_content = re.sub(json_pattern, escape_json_block, content, flags=re.DOTALL)
    
    # Also escape inline JSON objects (like Object A: { "name": ... })
    # Pattern for inline JSON-like structures
    inline_pattern = r'(Object [AB]|Spatial Zone):\s*(\{[^}]+\})'
    
    def escape_inline_json(match):
        prefix = match.group(1)
        json_obj = match.group(2)
        # Escape the braces
        escaped = json_obj.replace('{', '{{').replace('}', '}}')
        return f"{prefix}: {escaped}"
    
    fixed_content = re.sub(inline_pattern, escape_inline_json, fixed_content)
    
    # Write back if changed
    if fixed_content != content:
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        print(f"  ✓ Fixed {file_path}")
        return True
    else:
        print(f"  - No changes needed for {file_path}")
        return False

def main():
    """Fix all prompt templates in the prompts directory."""
    prompts_dir = Path('prompts')
    
    if not prompts_dir.exists():
        print(f"Error: {prompts_dir} directory not found")
        return
    
    fixed_count = 0
    for prompt_file in prompts_dir.glob('*.md'):
        if fix_prompt_template(prompt_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} prompt files")

if __name__ == "__main__":
    main()