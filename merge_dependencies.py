#!/usr/bin/env python3
"""
Merge dependencies from multiple requirements files, resolving version conflicts.
"""

import re
from pathlib import Path
from typing import Dict, Set, Tuple, List
from packaging import version
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement

def parse_requirement_line(line: str) -> Tuple[str, str, str]:
    """Parse a requirement line and return (package_name, operator, version)."""
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None, None, None
    
    # Handle git+ dependencies
    if 'git+' in line:
        if '@' in line:
            package = line.split('@')[0].strip()
            return package, 'git', line
        return line, 'git', line
    
    # Handle regular packages
    try:
        req = Requirement(line)
        package_name = req.name
        if req.specifier:
            # Get the main version spec
            specs = list(req.specifier)
            if specs:
                return package_name, str(req.specifier), line
        return package_name, '', line
    except:
        # Fallback parsing for special cases
        if '==' in line:
            parts = line.split('==')
            return parts[0].strip(), f'=={parts[1].strip()}', line
        elif '>=' in line:
            parts = line.split('>=')
            return parts[0].strip(), f'>={parts[1].strip()}', line
        elif '<=' in line:
            parts = line.split('<=')
            return parts[0].strip(), f'<={parts[1].strip()}', line
        elif '>' in line:
            parts = line.split('>')
            return parts[0].strip(), f'>{parts[1].strip()}', line
        elif '<' in line:
            parts = line.split('<')
            return parts[0].strip(), f'<{parts[1].strip()}', line
        else:
            return line.strip(), '', line

def read_requirements_file(filepath: Path) -> Dict[str, List[str]]:
    """Read a requirements file and return a dict of package -> [version_specs]."""
    requirements = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            package, spec, full_line = parse_requirement_line(line)
            if package:
                if package not in requirements:
                    requirements[package] = []
                requirements[package].append((spec, full_line, str(filepath)))
    
    return requirements

def merge_version_specs(package: str, specs: List[Tuple[str, str, str]]) -> str:
    """Merge multiple version specifications for a package."""
    
    # Handle git dependencies
    git_specs = [s for s in specs if s[0] == 'git']
    if git_specs:
        # Prefer git dependencies with specific branches/tags
        for spec in git_specs:
            if '@main' in spec[1] or '@master' in spec[1]:
                return spec[1]
        return git_specs[0][1]  # Return first git spec
    
    # Extract all version constraints
    constraints = []
    pinned_versions = []
    
    for spec, full_line, source in specs:
        if not spec:
            continue
        if spec.startswith('=='):
            pinned_versions.append((spec[2:], source))
        else:
            constraints.append(spec)
    
    # If we have pinned versions, choose the latest
    if pinned_versions:
        # Sort by version and take the latest
        pinned_versions.sort(key=lambda x: version.parse(x[0]), reverse=True)
        latest_version = pinned_versions[0][0]
        
        # Check if this version satisfies all constraints
        if constraints:
            try:
                spec_set = SpecifierSet(','.join(constraints))
                if latest_version in spec_set:
                    return f"=={latest_version}"
            except:
                pass
        
        # If conflicts, prefer the latest pinned version
        print(f"  Warning: Version conflict for {package}. Using latest pinned version: {latest_version}")
        return f"=={latest_version}"
    
    # If only constraints, combine them
    if constraints:
        # Try to find a common range
        try:
            combined = ','.join(constraints)
            spec_set = SpecifierSet(combined)
            # Return the most restrictive constraint
            return combined.replace(',', ', ')
        except:
            # If incompatible, return the first constraint
            print(f"  Warning: Incompatible constraints for {package}: {constraints}")
            return constraints[0]
    
    # No version specification
    return ''

def main():
    """Main function to merge all requirements files."""
    
    # Define requirements files to merge
    requirements_files = [
        Path("reference/Pixel-Reasoner/curiosity_driven_rl/requirements.txt"),
        Path("reference/Pixel-Reasoner/instruction_tuning/install/requirements.txt"),
        Path("reference/Reason-RFT/requirements_rl.txt"),
        Path("reference/Reason-RFT/requirements_sft.txt"),
        Path("reference/TTRL/verl/requirements.txt"),
    ]
    
    # Collect all requirements
    all_requirements = {}
    
    print("Reading requirements files...")
    for filepath in requirements_files:
        if filepath.exists():
            print(f"  Reading: {filepath}")
            file_reqs = read_requirements_file(filepath)
            
            for package, specs in file_reqs.items():
                if package not in all_requirements:
                    all_requirements[package] = []
                all_requirements[package].extend(specs)
        else:
            print(f"  Warning: File not found: {filepath}")
    
    print(f"\nFound {len(all_requirements)} unique packages")
    
    # Merge requirements
    print("\nResolving version conflicts...")
    merged_requirements = {}
    
    for package, specs in sorted(all_requirements.items()):
        # Skip duplicate psutil entries and other problematic ones
        if package == 'psutil' and len(specs) > 1:
            # Take the latest version
            versions = [s[1].split('==')[1] if '==' in s[1] else '' for s in specs if s[0]]
            versions = [v for v in versions if v]
            if versions:
                latest = max(versions, key=lambda x: version.parse(x))
                merged_requirements[package] = f"psutil=={latest}"
                continue
        
        if len(specs) > 1:
            # Multiple specifications found
            sources = list(set([s[2] for s in specs]))
            print(f"  {package}: {len(specs)} specifications from {len(sources)} files")
        
        merged_spec = merge_version_specs(package, specs)
        
        if merged_spec:
            if merged_spec.startswith('git+') or merged_spec.endswith('.git'):
                merged_requirements[package] = merged_spec
            else:
                merged_requirements[package] = f"{package}{merged_spec}"
        else:
            merged_requirements[package] = package
    
    # Write merged requirements
    output_file = Path("requirements.txt")
    
    print(f"\nWriting merged requirements to {output_file}...")
    
    # Organize requirements by category
    core_packages = []
    ml_packages = []
    utils_packages = []
    dev_packages = []
    cuda_packages = []
    
    for package, spec in sorted(merged_requirements.items()):
        if any(x in package for x in ['nvidia', 'cuda', 'cublas', 'cudnn', 'nccl', 'nvtx']):
            cuda_packages.append(spec)
        elif any(x in package for x in ['torch', 'transformers', 'accelerate', 'deepspeed', 
                                        'peft', 'datasets', 'huggingface', 'safetensors',
                                        'vllm', 'flash', 'xformers', 'triton', 'trl']):
            ml_packages.append(spec)
        elif any(x in package for x in ['pytest', 'black', 'isort', 'flake8', 'ruff']):
            dev_packages.append(spec)
        else:
            utils_packages.append(spec)
    
    with open(output_file, 'w') as f:
        f.write("# Pixelis Project - Merged Requirements\n")
        f.write("# Generated from Pixel-Reasoner, Reason-RFT, and TTRL dependencies\n\n")
        
        f.write("# Core ML/DL Packages\n")
        for pkg in sorted(ml_packages):
            f.write(f"{pkg}\n")
        
        f.write("\n# CUDA/GPU Packages\n")
        for pkg in sorted(cuda_packages):
            f.write(f"{pkg}\n")
        
        f.write("\n# Utilities and Dependencies\n")
        for pkg in sorted(utils_packages):
            f.write(f"{pkg}\n")
        
        f.write("\n# Development Tools\n")
        for pkg in sorted(dev_packages):
            f.write(f"{pkg}\n")
    
    print(f"Successfully merged {len(merged_requirements)} packages into {output_file}")
    
    # Report conflicts
    conflicts = []
    for package, specs in all_requirements.items():
        if len(specs) > 1:
            unique_specs = set([s[0] for s in specs if s[0] and s[0] != 'git'])
            if len(unique_specs) > 1:
                conflicts.append((package, unique_specs))
    
    if conflicts:
        print(f"\n{len(conflicts)} packages had version conflicts that were resolved:")
        for package, specs in conflicts[:10]:  # Show first 10
            print(f"  - {package}: {specs}")

if __name__ == "__main__":
    main()