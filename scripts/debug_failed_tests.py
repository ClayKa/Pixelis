#!/usr/bin/env python3
"""
Interactive Test Debugger for Pixelis Project
Helps you step through failed tests and understand why they failed.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse
from datetime import datetime
import tempfile


@dataclass
class FailedTest:
    """Information about a failed test"""
    file_path: str
    test_name: str
    failure_type: str  # 'failed', 'error', 'timeout', 'crashed'
    error_message: str
    full_path: str  # Full pytest path like tests/file.py::Class::method


class TestDebugger:
    """Interactive debugger for failed tests"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.failed_tests: List[FailedTest] = []
        
    def load_test_results(self, json_file: str = "test_results.json") -> bool:
        """Load test results from JSON file created by run_all_tests.py"""
        if not Path(json_file).exists():
            print(f"❌ Test results file '{json_file}' not found.")
            print("   Run 'python scripts/run_all_tests.py' first to generate results.")
            return False
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Extract failed tests
        for file_path, file_data in data.get('files', {}).items():
            if file_data.get('timeout'):
                # All tests in this file timed out
                for i in range(file_data.get('total_tests', 0)):
                    self.failed_tests.append(FailedTest(
                        file_path=file_path,
                        test_name=f"test_{i}",
                        failure_type='timeout',
                        error_message=f"File timed out after 30 seconds",
                        full_path=f"{file_path}::test_{i}"
                    ))
            elif file_data.get('crashed'):
                # All tests in this file crashed
                for i in range(file_data.get('total_tests', 0)):
                    self.failed_tests.append(FailedTest(
                        file_path=file_path,
                        test_name=f"test_{i}",
                        failure_type='crashed',
                        error_message="File crashed (likely FAISS or memory issue)",
                        full_path=f"{file_path}::test_{i}"
                    ))
            else:
                # Check individual test results
                for test in file_data.get('tests', []):
                    if test['status'] in ['failed', 'error']:
                        self.failed_tests.append(FailedTest(
                            file_path=file_path,
                            test_name=test['name'],
                            failure_type=test['status'],
                            error_message=test.get('error_message', 'No error message'),
                            full_path=f"{file_path}::{test['name']}"
                        ))
        
        return True
    
    def discover_failed_tests_live(self) -> List[FailedTest]:
        """Run pytest to discover failed tests in real-time"""
        print("🔍 Discovering failed tests by running pytest...")
        
        cmd = [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q", "--no-header"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            
            # Parse output for failed tests
            for line in output.split('\n'):
                if 'FAILED' in line or 'ERROR' in line:
                    # Extract test path
                    parts = line.split(' ')
                    if len(parts) > 0 and '::' in parts[0]:
                        test_path = parts[0]
                        status = 'error' if 'ERROR' in line else 'failed'
                        
                        # Split path
                        if '::' in test_path:
                            file_part, test_part = test_path.split('::', 1)
                            self.failed_tests.append(FailedTest(
                                file_path=file_part,
                                test_name=test_part,
                                failure_type=status,
                                error_message="Run individual test for details",
                                full_path=test_path
                            ))
            
        except subprocess.TimeoutExpired:
            print("⏱️  Test discovery timed out")
        except Exception as e:
            print(f"❌ Error discovering tests: {e}")
        
        return self.failed_tests
    
    def run_single_test(self, test: FailedTest, capture_output: bool = False) -> Dict:
        """Run a single test and capture detailed output"""
        print(f"\n{'='*60}")
        print(f"Running: {test.full_path}")
        print(f"{'='*60}")
        
        # Construct pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test.full_path,
            "-vv",  # Very verbose
            "--tb=short",  # Short traceback
            "--no-header",
            "--no-summary",
            "-s"  # Don't capture stdout
        ]
        
        if capture_output:
            # Capture to file for analysis
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
                temp_file = f.name
            
            cmd.extend(["--capture=no", f"--junit-xml={temp_file}"])
        
        # Run test
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout + result.stderr
            
            # Parse output for error details
            error_details = self._extract_error_details(output)
            
            return {
                'output': output,
                'return_code': result.returncode,
                'error_details': error_details,
                'passed': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'output': "Test timed out after 10 seconds",
                'return_code': -1,
                'error_details': {'type': 'Timeout', 'message': 'Test exceeded time limit'},
                'passed': False
            }
        except Exception as e:
            return {
                'output': str(e),
                'return_code': -1,
                'error_details': {'type': 'Exception', 'message': str(e)},
                'passed': False
            }
    
    def _extract_error_details(self, output: str) -> Dict:
        """Extract structured error information from pytest output"""
        details = {
            'assertion_errors': [],
            'exceptions': [],
            'import_errors': [],
            'attribute_errors': []
        }
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'AssertionError' in line:
                # Get context
                context = '\n'.join(lines[max(0, i-2):min(len(lines), i+3)])
                details['assertion_errors'].append(context)
            elif 'ImportError' in line or 'ModuleNotFoundError' in line:
                details['import_errors'].append(line)
            elif 'AttributeError' in line:
                details['attribute_errors'].append(line)
            elif 'Exception' in line or 'Error' in line:
                if 'AssertionError' not in line:
                    details['exceptions'].append(line)
        
        return details
    
    def debug_interactive(self):
        """Interactive debugging session"""
        if not self.failed_tests:
            print("No failed tests to debug!")
            return
        
        print(f"\n🔍 Found {len(self.failed_tests)} failed tests")
        print("="*60)
        
        # Group by failure type
        by_type = {}
        for test in self.failed_tests:
            by_type.setdefault(test.failure_type, []).append(test)
        
        print("\nFailures by type:")
        for failure_type, tests in by_type.items():
            print(f"  {failure_type}: {len(tests)} tests")
        
        while True:
            print("\n" + "="*60)
            print("OPTIONS:")
            print("  1. List all failed tests")
            print("  2. Debug specific test")
            print("  3. Run all failed tests with details")
            print("  4. Show common failure patterns")
            print("  5. Generate fix suggestions")
            print("  6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self._list_failed_tests()
            elif choice == '2':
                self._debug_specific_test()
            elif choice == '3':
                self._run_all_failed()
            elif choice == '4':
                self._show_failure_patterns()
            elif choice == '5':
                self._generate_fix_suggestions()
            elif choice == '6':
                break
            else:
                print("Invalid option")
    
    def _list_failed_tests(self):
        """List all failed tests with numbers"""
        print("\n" + "="*60)
        print("FAILED TESTS:")
        print("="*60)
        
        for i, test in enumerate(self.failed_tests, 1):
            status_icon = {
                'failed': '❌',
                'error': '⚠️',
                'timeout': '⏱️',
                'crashed': '💥'
            }.get(test.failure_type, '?')
            
            print(f"{i:3}. {status_icon} {test.file_path}::{test.test_name}")
            print(f"     Type: {test.failure_type}")
            if len(test.error_message) > 60:
                print(f"     Error: {test.error_message[:60]}...")
            else:
                print(f"     Error: {test.error_message}")
    
    def _debug_specific_test(self):
        """Debug a specific test interactively"""
        self._list_failed_tests()
        
        try:
            num = int(input("\nEnter test number to debug: "))
            if 1 <= num <= len(self.failed_tests):
                test = self.failed_tests[num - 1]
                
                print(f"\n🔍 Debugging: {test.full_path}")
                
                # Run the test
                result = self.run_single_test(test)
                
                # Show output
                print("\n" + "="*60)
                print("TEST OUTPUT:")
                print("="*60)
                print(result['output'][:2000])  # First 2000 chars
                
                if not result['passed']:
                    print("\n" + "="*60)
                    print("ERROR ANALYSIS:")
                    print("="*60)
                    
                    details = result['error_details']
                    if details['assertion_errors']:
                        print("\n❌ Assertion Errors:")
                        for err in details['assertion_errors']:
                            print(f"  {err}")
                    
                    if details['import_errors']:
                        print("\n📦 Import Errors:")
                        for err in details['import_errors']:
                            print(f"  {err}")
                    
                    if details['attribute_errors']:
                        print("\n🔤 Attribute Errors:")
                        for err in details['attribute_errors']:
                            print(f"  {err}")
                    
                    # Suggest fixes
                    self._suggest_fix_for_test(test, details)
            else:
                print("Invalid test number")
        except ValueError:
            print("Invalid input")
    
    def _run_all_failed(self):
        """Run all failed tests with details"""
        print(f"\n🏃 Running {len(self.failed_tests)} failed tests...")
        
        passed = 0
        still_failing = []
        
        for test in self.failed_tests:
            result = self.run_single_test(test, capture_output=True)
            if result['passed']:
                passed += 1
                print(f"  ✅ {test.test_name} - NOW PASSING!")
            else:
                still_failing.append(test)
                print(f"  ❌ {test.test_name} - Still failing")
        
        print(f"\n📊 Results: {passed}/{len(self.failed_tests)} now passing")
        
        if still_failing:
            print(f"\n{len(still_failing)} tests still failing")
    
    def _show_failure_patterns(self):
        """Analyze and show common failure patterns"""
        print("\n" + "="*60)
        print("COMMON FAILURE PATTERNS:")
        print("="*60)
        
        patterns = {
            'VotingResult': [],
            'multiprocessing': [],
            'import': [],
            'timeout': [],
            'faiss': [],
            'attribute': []
        }
        
        for test in self.failed_tests:
            msg = test.error_message.lower()
            if 'votingresult' in msg:
                patterns['VotingResult'].append(test)
            elif 'multiprocessing' in msg or 'pickle' in msg:
                patterns['multiprocessing'].append(test)
            elif 'import' in msg or 'module' in msg:
                patterns['import'].append(test)
            elif test.failure_type == 'timeout':
                patterns['timeout'].append(test)
            elif 'faiss' in msg or 'segfault' in msg:
                patterns['faiss'].append(test)
            elif 'attribute' in msg:
                patterns['attribute'].append(test)
        
        for pattern, tests in patterns.items():
            if tests:
                print(f"\n{pattern.upper()} Issues: {len(tests)} tests")
                for test in tests[:3]:  # Show first 3
                    print(f"  - {test.file_path}::{test.test_name}")
    
    def _suggest_fix_for_test(self, test: FailedTest, error_details: Dict):
        """Generate fix suggestions for a specific test"""
        print("\n" + "="*60)
        print("💡 SUGGESTED FIXES:")
        print("="*60)
        
        suggestions = []
        
        # Check error patterns
        error_msg = test.error_message.lower()
        
        if 'votingresult' in error_msg and 'unexpected keyword' in error_msg:
            suggestions.append(
                "Fix VotingResult dataclass:\n"
                "  - Check VotingResult definition in core/data_structures.py\n"
                "  - The test is passing 'votes' but the dataclass might expect different fields\n"
                "  - Solution: Update either the test mock or the dataclass definition"
            )
        
        if 'pickle' in error_msg or 'local object' in error_msg:
            suggestions.append(
                "Fix multiprocessing pickling issue:\n"
                "  - Move local functions to module level\n"
                "  - Use functools.partial instead of lambdas\n"
                "  - Define worker functions outside of test methods"
            )
        
        if 'module' in error_msg and 'core.models' in error_msg:
            suggestions.append(
                "Module not implemented:\n"
                "  - This module is part of Phase 1 implementation\n"
                "  - Either skip these tests or create stub module"
            )
        
        if test.failure_type == 'timeout':
            suggestions.append(
                "Fix timeout issues:\n"
                "  - Check for blocking I/O operations\n"
                "  - Add timeout parameters to network calls\n"
                "  - Mock external dependencies like WandB"
            )
        
        if 'faiss' in error_msg or test.failure_type == 'crashed':
            suggestions.append(
                "Fix FAISS crashes:\n"
                "  - Ensure FAISS is properly installed: pip install faiss-cpu\n"
                "  - Check tensor dimensions match FAISS index requirements\n"
                "  - Use try-except blocks around FAISS operations"
            )
        
        if error_details.get('attribute_errors'):
            suggestions.append(
                "Fix attribute errors:\n"
                "  - Check dataclass field definitions\n"
                "  - Verify mock objects match actual implementations\n"
                "  - Ensure all required attributes are initialized"
            )
        
        if not suggestions:
            suggestions.append(
                "General debugging steps:\n"
                "  1. Run test in isolation: pytest -xvs {}\n"
                "  2. Add print statements or use pdb debugger\n"
                "  3. Check test fixtures and setup methods\n"
                "  4. Verify all imports and dependencies".format(test.full_path)
            )
        
        for suggestion in suggestions:
            print(f"\n{suggestion}")
    
    def _generate_fix_suggestions(self):
        """Generate comprehensive fix suggestions"""
        print("\n" + "="*60)
        print("🔧 COMPREHENSIVE FIX GUIDE:")
        print("="*60)
        
        fixes = """
1. FIX VotingResult ISSUES (4 tests):
   Location: core/data_structures.py
   Problem: Test expects 'votes' parameter
   Fix: Add 'votes' field to VotingResult dataclass or update test mocks

2. FIX MULTIPROCESSING ISSUES (3 tests):
   Location: tests/engine/test_ipc.py
   Problem: Local functions can't be pickled
   Fix: Move worker functions to module level:
   ```python
   # Instead of:
   def test_something(self):
       def worker():  # Can't pickle this
           pass
   
   # Use:
   def worker():  # Module level
       pass
   
   def test_something(self):
       p = Process(target=worker)
   ```

3. FIX MODULE IMPORT ISSUES (12 tests):
   Location: tests/modules/test_model_init.py
   Problem: core.models module doesn't exist
   Fix: Either skip tests or create stub:
   ```python
   # core/models/__init__.py
   # Stub for Phase 1 implementation
   ```

4. FIX TIMEOUT ISSUES (26 tests):
   Locations: test_artifact_manager.py, test_experimental_protocol.py
   Problem: WandB initialization blocks
   Fix: Mock WandB in tests:
   ```python
   @patch('wandb.init')
   def test_something(self, mock_wandb):
       mock_wandb.return_value = Mock()
   ```

5. FIX FAISS CRASHES (20 tests):
   Location: test_experience_buffer.py
   Problem: FAISS segmentation fault
   Fix: 
   - Install: pip install faiss-cpu
   - Check tensor shapes before FAISS operations
   - Add error handling around FAISS calls

6. FIX SHARED MEMORY CLEANUP (multiple tests):
   Problem: Shared memory not being released
   Fix: Add cleanup in teardown:
   ```python
   def teardown_method(self):
       # Clean up any shared memory
       manager.cleanup_all()
   ```
"""
        print(fixes)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Debug failed Pixelis tests")
    parser.add_argument('--json', default='test_results.json', 
                       help='Test results JSON file')
    parser.add_argument('--live', action='store_true',
                       help='Discover failed tests by running pytest')
    parser.add_argument('--test', help='Debug specific test path')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    debugger = TestDebugger()
    
    if args.live:
        # Discover tests by running pytest
        debugger.discover_failed_tests_live()
    else:
        # Load from JSON
        if not debugger.load_test_results(args.json):
            return 1
    
    if args.test:
        # Debug specific test
        test = FailedTest(
            file_path=args.test.split('::')[0],
            test_name='::'.join(args.test.split('::')[1:]) if '::' in args.test else 'test',
            failure_type='unknown',
            error_message='',
            full_path=args.test
        )
        result = debugger.run_single_test(test)
        print(result['output'])
    else:
        # Interactive mode
        debugger.debug_interactive()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())