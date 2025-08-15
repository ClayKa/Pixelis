#!/usr/bin/env python3
"""
Comprehensive Test Runner for Pixelis Project
Handles timeouts, crashes, and provides detailed failure reporting
"""

import subprocess
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import multiprocessing as mp
import signal
import os

# Set environment variables to avoid issues
os.environ["PIXELIS_OFFLINE_MODE"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    status: str  # 'passed', 'failed', 'error', 'skipped', 'timeout', 'crashed'
    duration: float
    error_message: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class TestFileResult:
    """Result of running tests in a single file"""
    file_path: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    timeout: bool
    crashed: bool
    duration: float
    test_results: List[TestResult]
    raw_output: str


class TestRunner:
    """Comprehensive test runner with timeout and crash handling"""
    
    def __init__(self, timeout_per_file: int = 30, verbose: bool = True):
        self.timeout_per_file = timeout_per_file
        self.verbose = verbose
        self.results: Dict[str, TestFileResult] = {}
        
    def find_test_files(self) -> List[Path]:
        """Find all test files in the tests directory"""
        test_dir = Path("tests")
        test_files = list(test_dir.rglob("test_*.py"))
        return sorted(test_files)
    
    def count_tests_in_file(self, file_path: Path) -> int:
        """Count number of test functions in a file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                return len(re.findall(r'def test_\w+', content))
        except Exception:
            return 0
    
    def run_test_file(self, file_path: Path) -> TestFileResult:
        """Run tests in a single file with timeout protection"""
        start_time = time.time()
        test_count = self.count_tests_in_file(file_path)
        
        # Skip files with known import issues
        if "test_model_init" in str(file_path):
            return TestFileResult(
                file_path=str(file_path),
                total_tests=test_count,
                passed=0,
                failed=0,
                errors=test_count,
                skipped=0,
                timeout=False,
                crashed=False,
                duration=0,
                test_results=[
                    TestResult(
                        name=f"test_{i}",
                        status="error",
                        duration=0,
                        error_message="Module 'core.models' not implemented yet"
                    ) for i in range(test_count)
                ],
                raw_output="Skipped: Module not implemented"
            )
        
        # Run pytest with timeout
        cmd = [
            sys.executable, "-m", "pytest",
            str(file_path),
            "--no-cov",
            "-v",
            "--tb=short"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_per_file
            )
            
            duration = time.time() - start_time
            output = result.stdout + result.stderr
            
            # Parse results
            passed, failed, errors, skipped = self._parse_pytest_output(output)
            test_results = self._parse_test_details(output, file_path)
            
            # Check for crashes
            crashed = "SEGFAULT" in output or "Fatal Python error" in output or "Aborted" in output
            
            return TestFileResult(
                file_path=str(file_path),
                total_tests=test_count,
                passed=passed,
                failed=failed,
                errors=errors,
                skipped=skipped,
                timeout=False,
                crashed=crashed,
                duration=duration,
                test_results=test_results,
                raw_output=output[:1000]  # Keep first 1000 chars
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestFileResult(
                file_path=str(file_path),
                total_tests=test_count,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                timeout=True,
                crashed=False,
                duration=duration,
                test_results=[
                    TestResult(
                        name=f"test_{i}",
                        status="timeout",
                        duration=0,
                        error_message=f"Test file timed out after {self.timeout_per_file}s"
                    ) for i in range(test_count)
                ],
                raw_output=f"Timeout after {self.timeout_per_file} seconds"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestFileResult(
                file_path=str(file_path),
                total_tests=test_count,
                passed=0,
                failed=0,
                errors=test_count,
                skipped=0,
                timeout=False,
                crashed=True,
                duration=duration,
                test_results=[
                    TestResult(
                        name=f"test_{i}",
                        status="crashed",
                        duration=0,
                        error_message=str(e)
                    ) for i in range(test_count)
                ],
                raw_output=f"Exception: {str(e)}"
            )
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """Parse pytest output for summary statistics"""
        passed = failed = errors = skipped = 0
        
        # Look for summary line - it's usually like "5 passed in 0.95s"
        for line in output.split('\n'):
            # Check for the summary line with "=" characters
            if '=' in line and ('passed' in line or 'failed' in line or 'error' in line):
                passed_match = re.search(r'(\d+) passed', line)
                failed_match = re.search(r'(\d+) failed', line)
                error_match = re.search(r'(\d+) error', line)
                skipped_match = re.search(r'(\d+) skipped', line)
                
                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                    
                if any([passed_match, failed_match, error_match, skipped_match]):
                    break
        
        return passed, failed, errors, skipped
    
    def _parse_test_details(self, output: str, file_path: Path) -> List[TestResult]:
        """Parse individual test results from output"""
        results = []
        
        # Parse each test result line
        for line in output.split('\n'):
            if '::test_' in line:
                test_match = re.search(r'::(\w+).*?(PASSED|FAILED|ERROR|SKIPPED)', line)
                if test_match:
                    test_name = test_match.group(1)
                    status = test_match.group(2).lower()
                    results.append(TestResult(
                        name=test_name,
                        status=status,
                        duration=0,
                        file_path=str(file_path)
                    ))
        
        return results
    
    def run_all_tests(self) -> Dict[str, TestFileResult]:
        """Run all tests and collect results"""
        test_files = self.find_test_files()
        total_files = len(test_files)
        
        print(f"\n{'='*60}")
        print(f"PIXELIS TEST RUNNER")
        print(f"{'='*60}")
        print(f"Found {total_files} test files")
        print(f"Timeout per file: {self.timeout_per_file} seconds")
        print(f"{'='*60}\n")
        
        for i, test_file in enumerate(test_files, 1):
            try:
                rel_path = test_file.relative_to(Path.cwd())
            except ValueError:
                # If test_file is already relative or has issues, just use it as is
                rel_path = test_file
            print(f"[{i}/{total_files}] Running {rel_path}...", end=" ", flush=True)
            
            result = self.run_test_file(test_file)
            self.results[str(rel_path)] = result
            
            # Print status
            if result.timeout:
                print("⏱️  TIMEOUT")
            elif result.crashed:
                print("💥 CRASHED")
            elif result.errors > 0:
                print(f"⚠️  ERRORS ({result.errors})")
            elif result.failed > 0:
                print(f"❌ FAILED ({result.failed})")
            elif result.passed == result.total_tests:
                print(f"✅ PASSED ({result.passed})")
            else:
                print(f"⚪ PARTIAL ({result.passed}/{result.total_tests})")
        
        return self.results
    
    def generate_report(self) -> None:
        """Generate comprehensive test report"""
        # Calculate totals
        total_tests = sum(r.total_tests for r in self.results.values())
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        total_timeout = sum(1 for r in self.results.values() if r.timeout)
        total_crashed = sum(1 for r in self.results.values() if r.crashed)
        
        # Calculate tests that couldn't run
        timeout_tests = sum(r.total_tests for r in self.results.values() if r.timeout)
        crashed_tests = sum(r.total_tests for r in self.results.values() if r.crashed)
        
        print(f"\n{'='*60}")
        print(f"TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Test Functions: {total_tests}")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   ⚠️  Errors: {total_errors}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print(f"   ⏱️  Timed out: {timeout_tests} tests in {total_timeout} files")
        print(f"   💥 Crashed: {crashed_tests} tests in {total_crashed} files")
        
        # Calculate failure rate
        tests_with_results = total_passed + total_failed + total_errors + total_skipped
        if tests_with_results > 0:
            failure_rate = (total_failed + total_errors) / tests_with_results
            print(f"\n📈 FAILURE RATE: {failure_rate:.2%} ({total_failed + total_errors}/{tests_with_results})")
        
        overall_failure_rate = (total_tests - total_passed) / total_tests if total_tests > 0 else 0
        print(f"📉 OVERALL FAILURE RATE: {overall_failure_rate:.2%} ({total_tests - total_passed}/{total_tests})")
        
        # List all failed tests
        print(f"\n{'='*60}")
        print(f"FAILED TESTS DETAILED LIST")
        print(f"{'='*60}")
        
        failed_count = 0
        for file_path, result in self.results.items():
            if result.failed > 0 or result.errors > 0 or result.timeout or result.crashed:
                print(f"\n📁 {file_path}:")
                
                if result.timeout:
                    print(f"   ⏱️  ALL TESTS TIMED OUT ({result.total_tests} tests)")
                    failed_count += result.total_tests
                elif result.crashed:
                    print(f"   💥 CRASHED - ALL TESTS AFFECTED ({result.total_tests} tests)")
                    failed_count += result.total_tests
                else:
                    # List individual failed tests
                    for test in result.test_results:
                        if test.status in ['failed', 'error']:
                            print(f"   ❌ {test.name}: {test.status}")
                            if test.error_message:
                                print(f"      └── {test.error_message[:100]}")
                            failed_count += 1
        
        print(f"\n{'='*60}")
        print(f"FINAL METRICS")
        print(f"{'='*60}")
        print(f"Total Failed/Error/Timeout Tests: {failed_count}")
        print(f"Total Tests: {total_tests}")
        print(f"Final Failure Rate: {failed_count}/{total_tests} = {(failed_count/total_tests)*100:.1f}%")
        
        # Save results to JSON
        self.save_results_to_json()
        
    def save_results_to_json(self) -> None:
        """Save detailed results to JSON file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": sum(r.total_tests for r in self.results.values()),
                "passed": sum(r.passed for r in self.results.values()),
                "failed": sum(r.failed for r in self.results.values()),
                "errors": sum(r.errors for r in self.results.values()),
                "skipped": sum(r.skipped for r in self.results.values()),
                "timeout": sum(r.total_tests for r in self.results.values() if r.timeout),
                "crashed": sum(r.total_tests for r in self.results.values() if r.crashed),
            },
            "files": {}
        }
        
        for file_path, result in self.results.items():
            output["files"][file_path] = {
                "total_tests": result.total_tests,
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "skipped": result.skipped,
                "timeout": result.timeout,
                "crashed": result.crashed,
                "duration": result.duration,
                "tests": [asdict(t) for t in result.test_results]
            }
        
        with open("test_results.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Detailed results saved to test_results.json")


def main():
    """Main entry point"""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create runner with reasonable timeout
    runner = TestRunner(timeout_per_file=30, verbose=True)
    
    # Run all tests
    runner.run_all_tests()
    
    # Generate report
    runner.generate_report()
    
    # Return exit code based on failures
    total_tests = sum(r.total_tests for r in runner.results.values())
    total_passed = sum(r.passed for r in runner.results.values())
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())