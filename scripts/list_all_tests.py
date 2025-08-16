#!/usr/bin/env python3
"""
Script to discover and list all tests in the Pixelis project.
Generates a comprehensive markdown report of all test functions.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestFunction:
    """Information about a single test function"""
    name: str
    line_number: int
    docstring: str = ""
    test_class: str = ""


@dataclass
class TestFile:
    """Information about a test file"""
    path: Path
    module_name: str
    test_classes: Dict[str, List[TestFunction]]
    standalone_tests: List[TestFunction]
    total_tests: int


class TestDiscoverer:
    """Discovers all test functions in the project"""
    
    def __init__(self, test_dir: Path = Path("tests")):
        self.test_dir = test_dir
        self.test_files: List[TestFile] = []
        
    def discover_all_tests(self) -> List[TestFile]:
        """Discover all test files and their contents"""
        test_files = sorted(self.test_dir.rglob("test_*.py"))
        
        for test_file in test_files:
            file_info = self.parse_test_file(test_file)
            if file_info:
                self.test_files.append(file_info)
                
        return self.test_files
    
    def parse_test_file(self, file_path: Path) -> TestFile:
        """Parse a single test file to extract test information"""
        try:
            # Ensure file_path is absolute
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path
                
            with open(file_path, 'r') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract module docstring
            module_name = self.get_module_name(file_path)
            test_classes = {}
            standalone_tests = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a test class
                    if node.name.startswith('Test'):
                        test_methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                                docstring = ast.get_docstring(item) or ""
                                test_methods.append(TestFunction(
                                    name=item.name,
                                    line_number=item.lineno,
                                    docstring=docstring,
                                    test_class=node.name
                                ))
                        if test_methods:
                            test_classes[node.name] = test_methods
                            
                elif isinstance(node, ast.FunctionDef):
                    # Check for standalone test functions
                    if node.name.startswith('test_') and node.col_offset == 0:
                        # This is a module-level test function
                        docstring = ast.get_docstring(node) or ""
                        standalone_tests.append(TestFunction(
                            name=node.name,
                            line_number=node.lineno,
                            docstring=docstring,
                            test_class=""
                        ))
            
            total_tests = sum(len(tests) for tests in test_classes.values()) + len(standalone_tests)
            
            return TestFile(
                path=file_path,
                module_name=module_name,
                test_classes=test_classes,
                standalone_tests=standalone_tests,
                total_tests=total_tests
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name"""
        try:
            # Try to make path relative to project root
            if file_path.is_absolute():
                project_root = Path.cwd()
                relative_path = file_path.relative_to(project_root)
            else:
                relative_path = file_path
            module_parts = relative_path.with_suffix('').parts
            return '.'.join(module_parts)
        except ValueError:
            # If path operations fail, just use the file name
            return file_path.stem
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report of all tests"""
        report = []
        
        # Header
        report.append("# Pixelis Test Suite Documentation")
        report.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Summary statistics
        total_files = len(self.test_files)
        total_tests = sum(tf.total_tests for tf in self.test_files)
        total_classes = sum(len(tf.test_classes) for tf in self.test_files)
        total_standalone = sum(len(tf.standalone_tests) for tf in self.test_files)
        
        report.append("## Summary Statistics\n")
        report.append(f"- **Total Test Files**: {total_files}")
        report.append(f"- **Total Test Functions**: {total_tests}")
        report.append(f"- **Total Test Classes**: {total_classes}")
        report.append(f"- **Total Standalone Tests**: {total_standalone}")
        report.append("")
        
        # Table of contents
        report.append("## Table of Contents\n")
        for i, test_file in enumerate(self.test_files, 1):
            # Ensure path is absolute before making it relative
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = abs_path.relative_to(Path.cwd())
            anchor = str(rel_path).replace('/', '-').replace('.', '-').lower()
            report.append(f"{i}. [{rel_path}](#{anchor}) ({test_file.total_tests} tests)")
        report.append("")
        
        # Detailed test listing by file
        report.append("## Detailed Test Listing\n")
        
        for test_file in self.test_files:
            # Ensure path is absolute before making it relative
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = abs_path.relative_to(Path.cwd())
            anchor = str(rel_path).replace('/', '-').replace('.', '-').lower()
            
            report.append(f"### {rel_path} <a id='{anchor}'></a>\n")
            report.append(f"**Module**: `{test_file.module_name}`")
            report.append(f"**Total Tests**: {test_file.total_tests}\n")
            
            # List test classes and their methods
            if test_file.test_classes:
                report.append("#### Test Classes:\n")
                for class_name, methods in sorted(test_file.test_classes.items()):
                    report.append(f"**`{class_name}`** ({len(methods)} tests)")
                    report.append("")
                    for method in methods:
                        line_ref = f"{rel_path}:{method.line_number}"
                        report.append(f"- `{method.name}` (line {method.line_number})")
                        if method.docstring:
                            # Indent docstring
                            doc_lines = method.docstring.split('\n')
                            report.append(f"  > {doc_lines[0]}")
                    report.append("")
            
            # List standalone tests
            if test_file.standalone_tests:
                report.append("#### Standalone Tests:\n")
                for test in test_file.standalone_tests:
                    line_ref = f"{rel_path}:{test.line_number}"
                    report.append(f"- `{test.name}` (line {test.line_number})")
                    if test.docstring:
                        doc_lines = test.docstring.split('\n')
                        report.append(f"  > {doc_lines[0]}")
                report.append("")
            
            report.append("---\n")
        
        # Test organization by category
        report.append("## Test Organization by Category\n")
        
        categories = {
            "Unit Tests": [],
            "Integration Tests": [],
            "Engine Tests": [],
            "Module Tests": [],
            "Training Tests": [],
            "Protocol Tests": []
        }
        
        for test_file in self.test_files:
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = str(abs_path.relative_to(Path.cwd()))
            
            if "unit" in rel_path:
                categories["Unit Tests"].append((rel_path, test_file.total_tests))
            elif "integration" in rel_path:
                categories["Integration Tests"].append((rel_path, test_file.total_tests))
            elif "engine" in rel_path:
                categories["Engine Tests"].append((rel_path, test_file.total_tests))
            elif "modules" in rel_path:
                categories["Module Tests"].append((rel_path, test_file.total_tests))
            elif "rft" in rel_path or "sft" in rel_path:
                categories["Training Tests"].append((rel_path, test_file.total_tests))
            elif "protocol" in rel_path or "experimental" in rel_path:
                categories["Protocol Tests"].append((rel_path, test_file.total_tests))
        
        for category, files in categories.items():
            if files:
                total = sum(count for _, count in files)
                report.append(f"### {category} ({total} tests)\n")
                for file_path, count in files:
                    report.append(f"- `{file_path}` ({count} tests)")
                report.append("")
        
        # PyTest collection command for each file
        report.append("## PyTest Commands\n")
        report.append("### Run All Tests")
        report.append("```bash")
        report.append("pytest tests/")
        report.append("```\n")
        
        report.append("### Run Individual Test Files")
        report.append("```bash")
        for test_file in self.test_files:
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = abs_path.relative_to(Path.cwd())
            report.append(f"pytest {rel_path}")
        report.append("```\n")
        
        report.append("### Run Specific Test Classes")
        report.append("```bash")
        for test_file in self.test_files:
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = abs_path.relative_to(Path.cwd())
            for class_name in test_file.test_classes:
                report.append(f"pytest {rel_path}::{class_name}")
        report.append("```\n")
        
        report.append("### Run Specific Test Functions")
        report.append("```bash")
        # Show a few examples
        example_count = 0
        for test_file in self.test_files:
            if example_count >= 5:
                break
            abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
            rel_path = abs_path.relative_to(Path.cwd())
            for class_name, methods in test_file.test_classes.items():
                if example_count >= 5:
                    break
                if methods:
                    report.append(f"pytest {rel_path}::{class_name}::{methods[0].name}")
                    example_count += 1
        report.append("# ... (use same pattern for other tests)")
        report.append("```\n")
        
        return '\n'.join(report)


def main():
    """Main entry point"""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Discover tests
    print("Discovering all tests in the Pixelis project...")
    discoverer = TestDiscoverer()
    test_files = discoverer.discover_all_tests()
    
    # Generate report
    print(f"Found {len(test_files)} test files")
    print("Generating markdown report...")
    
    report = discoverer.generate_markdown_report()
    
    # Save report
    output_file = Path("docs/TEST_CATALOG.md")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Test catalog saved to: {output_file}")
    
    # Print summary
    total_tests = sum(tf.total_tests for tf in test_files)
    print(f"\nSummary:")
    print(f"  - Total test files: {len(test_files)}")
    print(f"  - Total test functions: {total_tests}")
    
    # Also print the test count by file
    print("\nTests per file:")
    for test_file in test_files:
        abs_path = test_file.path if test_file.path.is_absolute() else Path.cwd() / test_file.path
        rel_path = abs_path.relative_to(Path.cwd())
        print(f"  {rel_path}: {test_file.total_tests} tests")


if __name__ == "__main__":
    main()