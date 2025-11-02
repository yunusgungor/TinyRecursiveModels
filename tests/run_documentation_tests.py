#!/usr/bin/env python3
"""
Documentation Test Runner

This script runs all tests related to MacBook training documentation,
including configuration validation, example script testing, and
troubleshooting solution verification.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_module(module_name, description):
    """Run a specific test module and return results."""
    print(f"\\n{'='*60}")
    print(f"Running {description}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        module = __import__(f"tests.{module_name}", fromlist=[module_name])
        
        if hasattr(module, f"run_{module_name.replace('test_', '')}_tests"):
            # Use custom test runner if available
            test_function = getattr(module, f"run_{module_name.replace('test_', '')}_tests")
            success = test_function()
        else:
            # Fall back to unittest discovery
            import unittest
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            success = result.wasSuccessful()
        
        elapsed = time.time() - start_time
        
        if success:
            print(f"\\n✅ {description} completed successfully in {elapsed:.1f}s")
        else:
            print(f"\\n❌ {description} failed after {elapsed:.1f}s")
        
        return success
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\\n💥 {description} crashed after {elapsed:.1f}s: {e}")
        return False


def check_prerequisites():
    """Check that all prerequisites are available."""
    print("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required modules
    required_modules = [
        'torch',
        'yaml', 
        'psutil',
        'unittest',
        'tempfile',
        'pathlib'
    ]
    
    missing_modules = []
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError:
            print(f"❌ {module_name} (missing)")
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"\\nMissing required modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    
    # Check project structure
    required_dirs = [
        'examples/macbook_training/configs',
        'examples/macbook_training/scripts',
        'docs',
        'macbook_optimization'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            return False
    
    return True


def run_all_documentation_tests():
    """Run all documentation-related tests."""
    print("MacBook TRM Documentation Test Suite")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\\n❌ Prerequisites not met. Please fix the issues above.")
        return False
    
    # Define test modules to run
    test_modules = [
        ('test_example_configurations', 'Example Configuration Tests'),
        ('test_documentation_examples', 'Documentation Example Tests'),
        ('test_troubleshooting_solutions', 'Troubleshooting Solution Tests')
    ]
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each test module
    for module_name, description in test_modules:
        success = run_test_module(module_name, description)
        results[description] = success
    
    # Print summary
    total_elapsed = time.time() - total_start_time
    
    print("\\n" + "="*60)
    print("DOCUMENTATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for description, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")
    
    print(f"\\nResults: {passed}/{total} test suites passed")
    print(f"Total time: {total_elapsed:.1f}s")
    
    if passed == total:
        print("\\n🎉 All documentation tests passed!")
        return True
    else:
        print(f"\\n💥 {total - passed} test suite(s) failed!")
        return False


def run_specific_test(test_name):
    """Run a specific test module."""
    test_mapping = {
        'configs': ('test_example_configurations', 'Example Configuration Tests'),
        'examples': ('test_documentation_examples', 'Documentation Example Tests'),
        'troubleshooting': ('test_troubleshooting_solutions', 'Troubleshooting Solution Tests')
    }
    
    if test_name not in test_mapping:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_mapping.keys())}")
        return False
    
    module_name, description = test_mapping[test_name]
    return run_test_module(module_name, description)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MacBook TRM documentation tests")
    parser.add_argument('--test', type=str, help="Run specific test (configs, examples, troubleshooting)")
    parser.add_argument('--check-only', action='store_true', help="Only check prerequisites")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.check_only:
        success = check_prerequisites()
        sys.exit(0 if success else 1)
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_documentation_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()