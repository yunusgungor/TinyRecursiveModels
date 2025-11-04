#!/usr/bin/env python3
"""
Test runner for Gmail Dataset Creator comprehensive test suite.

This script runs all unit tests, integration tests, and performance tests
for the Gmail Dataset Creator project.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Gmail Dataset Creator - Comprehensive Test Suite")
    print("=" * 60)
    
    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    # Change to project root for proper imports
    os.chdir(project_root)
    
    # Test files to run
    test_files = [
        "gmail_dataset_creator/tests/test_core_components_unit.py",
        "gmail_dataset_creator/tests/test_integration_performance.py"
    ]
    
    # Check if existing test files should be included
    existing_tests = [
        "gmail_dataset_creator/tests/test_email_processor.py",
        "gmail_dataset_creator/tests/test_gemini_classifier.py", 
        "gmail_dataset_creator/tests/test_gmail_client.py"
    ]
    
    for test_file in existing_tests:
        if os.path.exists(test_file):
            print(f"Found existing test file: {test_file}")
            # Note: These may fail due to missing dependencies, so we'll run them separately
    
    print(f"\nRunning comprehensive test suite...")
    print(f"Test files: {len(test_files)} files")
    
    # Run tests with pytest
    cmd = [
        sys.executable, "-m", "pytest",
        *test_files,
        "-v",
        "--tb=short",
        "--durations=10"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ All tests passed successfully!")
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_specific_test_category(category: str):
    """Run tests for a specific category."""
    test_categories = {
        "unit": ["gmail_dataset_creator/tests/test_core_components_unit.py"],
        "integration": ["gmail_dataset_creator/tests/test_integration_performance.py"],
        "existing": [
            "gmail_dataset_creator/tests/test_email_processor.py",
            "gmail_dataset_creator/tests/test_gemini_classifier.py",
            "gmail_dataset_creator/tests/test_gmail_client.py"
        ]
    }
    
    if category not in test_categories:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {', '.join(test_categories.keys())}")
        return 1
    
    test_files = test_categories[category]
    
    # Filter existing files
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"No test files found for category: {category}")
        return 1
    
    print(f"Running {category} tests...")
    print(f"Files: {existing_files}")
    
    cmd = [sys.executable, "-m", "pytest", *existing_files, "-v"]
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Error running {category} tests: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        category = sys.argv[1]
        return run_specific_test_category(category)
    else:
        return run_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)