#!/usr/bin/env python3
"""
MacBook Email Training Setup Validation Script

This script validates that the MacBook email training setup is correct and all
components are working properly. It tests configurations, datasets, and dependencies.

Requirements: 1.5, 2.1, 5.4
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are available."""
    required_packages = [
        'torch',
        'numpy', 
        'yaml',
        'psutil'
    ]
    
    optional_packages = [
        'wandb',
        'sklearn'
    ]
    
    results = []
    all_required_available = True
    
    for package in required_packages:
        try:
            __import__(package)
            results.append(f"✓ {package}")
        except ImportError:
            results.append(f"✗ {package} (REQUIRED)")
            all_required_available = False
    
    for package in optional_packages:
        try:
            __import__(package)
            results.append(f"✓ {package} (optional)")
        except ImportError:
            results.append(f"- {package} (optional, not installed)")
    
    return all_required_available, results

def check_project_structure() -> Tuple[bool, List[str]]:
    """Check if project structure is correct."""
    required_paths = [
        "train_email_classifier_macbook.py",
        "macbook_optimization/",
        "models/",
        "examples/macbook_training/configs/",
        "docs/",
        "tests/"
    ]
    
    results = []
    all_present = True
    
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            results.append(f"✓ {path}")
        else:
            results.append(f"✗ {path}")
            all_present = False
    
    return all_present, results

def check_configurations() -> Tuple[bool, List[str]]:
    """Check if example configurations are valid."""
    config_dirs = [
        "examples/macbook_training/configs/macbook_8gb",
        "examples/macbook_training/configs/macbook_16gb",
        "examples/macbook_training/configs/macbook_32gb"
    ]
    
    results = []
    all_valid = True
    
    for config_dir in config_dirs:
        config_path = project_root / config_dir / "email_classification.yaml"
        
        if not config_path.exists():
            results.append(f"✗ {config_dir}/email_classification.yaml (missing)")
            all_valid = False
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['model', 'training', 'email', 'hardware', 'targets']
            missing_sections = [s for s in required_sections if s not in config]
            
            if missing_sections:
                results.append(f"✗ {config_dir} (missing sections: {missing_sections})")
                all_valid = False
            else:
                results.append(f"✓ {config_dir}")
        
        except Exception as e:
            results.append(f"✗ {config_dir} (error: {e})")
            all_valid = False
    
    return all_valid, results

def check_sample_datasets() -> Tuple[bool, List[str]]:
    """Check if sample datasets are valid."""
    dataset_files = [
        "examples/macbook_training/datasets/sample_emails_small.json",
        "examples/macbook_training/datasets/sample_emails_medium.json"
    ]
    
    results = []
    all_valid = True
    
    for dataset_file in dataset_files:
        dataset_path = project_root / dataset_file
        
        if not dataset_path.exists():
            results.append(f"✗ {dataset_file} (missing)")
            all_valid = False
            continue
        
        try:
            with open(dataset_path, 'r') as f:
                emails = json.load(f)
            
            if not isinstance(emails, list) or len(emails) == 0:
                results.append(f"✗ {dataset_file} (invalid format)")
                all_valid = False
                continue
            
            # Check first email has required fields
            required_fields = ['id', 'subject', 'body', 'category', 'sender']
            missing_fields = [f for f in required_fields if f not in emails[0]]
            
            if missing_fields:
                results.append(f"✗ {dataset_file} (missing fields: {missing_fields})")
                all_valid = False
            else:
                results.append(f"✓ {dataset_file} ({len(emails)} emails)")
        
        except Exception as e:
            results.append(f"✗ {dataset_file} (error: {e})")
            all_valid = False
    
    return all_valid, results

def check_documentation() -> Tuple[bool, List[str]]:
    """Check if documentation files exist."""
    doc_files = [
        "docs/macbook_email_training_setup_guide.md",
        "docs/macbook_email_training_troubleshooting_guide.md", 
        "docs/macbook_email_training_performance_optimization_guide.md"
    ]
    
    results = []
    all_present = True
    
    for doc_file in doc_files:
        doc_path = project_root / doc_file
        
        if doc_path.exists():
            # Check file size to ensure it's not empty
            size_kb = doc_path.stat().st_size / 1024
            results.append(f"✓ {doc_file} ({size_kb:.1f}KB)")
        else:
            results.append(f"✗ {doc_file}")
            all_present = False
    
    return all_present, results

def test_hardware_detection() -> Tuple[bool, str]:
    """Test hardware detection functionality."""
    try:
        # Try to import and run hardware detection
        from macbook_optimization.hardware_detection import HardwareDetector
        
        detector = HardwareDetector()
        specs = detector.get_hardware_specs()
        
        memory_gb = specs.memory.total_memory / (1024**3)
        cpu_cores = specs.cpu.cores
        
        return True, f"Detected: {cpu_cores} cores, {memory_gb:.1f}GB memory"
    
    except Exception as e:
        return False, f"Hardware detection failed: {e}"

def test_config_validator() -> Tuple[bool, str]:
    """Test configuration validator functionality."""
    try:
        from examples.macbook_training.config_validator import ConfigValidator, HardwareSpecs, MacBookModel
        
        validator = ConfigValidator()
        
        # Test with sample configuration
        test_config = {
            'model': {
                'vocab_size': 5000,
                'hidden_size': 512,
                'num_layers': 2,
                'max_sequence_length': 512
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-4,
                'max_steps': 5000
            },
            'hardware': {
                'memory_limit_mb': 12000
            }
        }
        
        hardware_specs = HardwareSpecs(16.0, 8, MacBookModel.MACBOOK_16GB)
        result = validator.validate_config(test_config, hardware_specs)
        
        if result.is_valid:
            return True, f"Config validation works (estimated {result.estimated_memory_usage:.0f}MB)"
        else:
            return False, f"Config validation failed: {result.errors}"
    
    except Exception as e:
        return False, f"Config validator test failed: {e}"

def test_training_script() -> Tuple[bool, str]:
    """Test main training script functionality."""
    try:
        script_path = project_root / "train_email_classifier_macbook.py"
        
        if not script_path.exists():
            return False, "Main training script not found"
        
        # Test hardware detection command
        result = subprocess.run([
            sys.executable, str(script_path), '--detect-hardware'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return True, "Training script hardware detection works"
        else:
            return False, f"Training script failed: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return False, "Training script timed out"
    except Exception as e:
        return False, f"Training script test failed: {e}"

def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation of the setup."""
    print("🔍 MacBook Email Training Setup Validation")
    print("=" * 50)
    
    validation_results = {}
    
    # Check Python version
    print("\n📋 Checking Python version...")
    python_ok, python_msg = check_python_version()
    validation_results['python'] = {'status': python_ok, 'message': python_msg}
    print(f"   {python_msg}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok, deps_msgs = check_dependencies()
    validation_results['dependencies'] = {'status': deps_ok, 'messages': deps_msgs}
    for msg in deps_msgs:
        print(f"   {msg}")
    
    # Check project structure
    print("\n📁 Checking project structure...")
    struct_ok, struct_msgs = check_project_structure()
    validation_results['structure'] = {'status': struct_ok, 'messages': struct_msgs}
    for msg in struct_msgs:
        print(f"   {msg}")
    
    # Check configurations
    print("\n⚙️  Checking configurations...")
    config_ok, config_msgs = check_configurations()
    validation_results['configurations'] = {'status': config_ok, 'messages': config_msgs}
    for msg in config_msgs:
        print(f"   {msg}")
    
    # Check sample datasets
    print("\n📊 Checking sample datasets...")
    dataset_ok, dataset_msgs = check_sample_datasets()
    validation_results['datasets'] = {'status': dataset_ok, 'messages': dataset_msgs}
    for msg in dataset_msgs:
        print(f"   {msg}")
    
    # Check documentation
    print("\n📚 Checking documentation...")
    docs_ok, docs_msgs = check_documentation()
    validation_results['documentation'] = {'status': docs_ok, 'messages': docs_msgs}
    for msg in docs_msgs:
        print(f"   {msg}")
    
    # Test hardware detection
    print("\n🖥️  Testing hardware detection...")
    hw_ok, hw_msg = test_hardware_detection()
    validation_results['hardware_detection'] = {'status': hw_ok, 'message': hw_msg}
    print(f"   {hw_msg}")
    
    # Test config validator
    print("\n✅ Testing config validator...")
    validator_ok, validator_msg = test_config_validator()
    validation_results['config_validator'] = {'status': validator_ok, 'message': validator_msg}
    print(f"   {validator_msg}")
    
    # Test training script
    print("\n🚀 Testing training script...")
    script_ok, script_msg = test_training_script()
    validation_results['training_script'] = {'status': script_ok, 'message': script_msg}
    print(f"   {script_msg}")
    
    # Overall status
    all_critical_ok = all([
        python_ok, deps_ok, struct_ok, config_ok, 
        dataset_ok, docs_ok, hw_ok, validator_ok, script_ok
    ])
    
    validation_results['overall'] = {'status': all_critical_ok}
    
    print("\n" + "=" * 50)
    if all_critical_ok:
        print("🎉 All validation checks passed! Setup is ready for training.")
        print("\nNext steps:")
        print("1. Review the setup guide: docs/macbook_email_training_setup_guide.md")
        print("2. Prepare your email dataset or use sample data")
        print("3. Run training: python train_email_classifier_macbook.py --train")
    else:
        print("❌ Some validation checks failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("1. Check the troubleshooting guide: docs/macbook_email_training_troubleshooting_guide.md")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify project structure is complete")
    
    return validation_results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MacBook email training setup")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    if args.quiet:
        # Redirect stdout to suppress output during validation
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        results = run_comprehensive_validation()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Print only final status
        if results['overall']['status']:
            print("✅ Setup validation passed")
        else:
            print("❌ Setup validation failed")
    else:
        results = run_comprehensive_validation()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results['overall']['status'] else 1)

if __name__ == "__main__":
    main()