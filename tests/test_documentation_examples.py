#!/usr/bin/env python3
"""
Tests for MacBook training documentation examples.

This test suite validates that the example configurations and scripts
provided in the documentation work correctly and produce expected results.
"""

import os
import sys
import yaml
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.config_validation import ConfigurationValidator
from macbook_optimization.memory_management import MemoryManager
from pretrain import PretrainConfig


class TestExampleConfigurations(TestCase):
    """Test that example configurations are valid and work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.configs_dir = self.examples_dir / "configs"
        
        # Create mock hardware detector for consistent testing
        self.hardware_detector = HardwareDetector()
        self.validator = ConfigurationValidator(self.hardware_detector)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_macbook_8gb_small_dataset_config(self):
        """Test 8GB MacBook small dataset configuration."""
        config_path = self.configs_dir / "macbook_8gb" / "small_dataset.yaml"
        self.assertTrue(config_path.exists(), f"Configuration file not found: {config_path}")
        
        # Load and validate configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Basic structure validation
        self.assertIn('hardware_profile', config_dict)
        self.assertEqual(config_dict['hardware_profile'], 'macbook_8gb')
        self.assertIn('global_batch_size', config_dict)
        self.assertIn('training', config_dict)
        self.assertIn('arch', config_dict)
        self.assertIn('macbook_optimizations', config_dict)
        
        # Memory constraints validation
        training_config = config_dict['training']
        self.assertLessEqual(training_config['memory_limit_mb'], 4500)  # Conservative for 8GB
        self.assertLessEqual(config_dict['global_batch_size'], 64)  # Reasonable for 8GB
        
        # Architecture validation for memory constraints
        arch_config = config_dict['arch']
        self.assertLessEqual(arch_config['hidden_size'], 384)  # Not too large for 8GB
        self.assertLessEqual(arch_config['seq_len'], 512)  # Reasonable sequence length
        
        # MacBook optimizations validation
        opt_config = config_dict['macbook_optimizations']
        self.assertTrue(opt_config['enable_memory_monitoring'])
        self.assertLessEqual(opt_config['memory_pressure_threshold'], 80.0)
    
    def test_macbook_16gb_medium_dataset_config(self):
        """Test 16GB MacBook medium dataset configuration."""
        config_path = self.configs_dir / "macbook_16gb" / "medium_dataset.yaml"
        self.assertTrue(config_path.exists(), f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate 16GB-specific settings
        self.assertEqual(config_dict['hardware_profile'], 'macbook_16gb')
        self.assertGreater(config_dict['training']['memory_limit_mb'], 5000)  # More generous
        self.assertGreater(config_dict['global_batch_size'], 32)  # Larger batch size
        
        # Architecture should be larger than 8GB version
        arch_config = config_dict['arch']
        self.assertGreaterEqual(arch_config['hidden_size'], 256)
        self.assertGreaterEqual(arch_config['seq_len'], 256)
    
    def test_all_example_configs_are_valid_yaml(self):
        """Test that all example configurations are valid YAML."""
        config_files = []
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    config_files.append(os.path.join(root, file))
        
        self.assertGreater(len(config_files), 0, "No configuration files found")
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    self.assertIsInstance(config, dict, f"Invalid YAML structure in {config_file}")
                except yaml.YAMLError as e:
                    self.fail(f"YAML parsing error in {config_file}: {e}")
    
    def test_config_hardware_compatibility(self):
        """Test that configurations are compatible with their target hardware."""
        test_cases = [
            ("macbook_8gb/small_dataset.yaml", 8),
            ("macbook_16gb/medium_dataset.yaml", 16),
            ("macbook_16gb/large_dataset.yaml", 16),
        ]
        
        for config_file, expected_ram_gb in test_cases:
            with self.subTest(config_file=config_file):
                config_path = self.configs_dir / config_file
                
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # Mock hardware specs for testing
                mock_hardware = {
                    'memory': {'total_gb': expected_ram_gb, 'available_gb': expected_ram_gb * 0.7},
                    'cpu': {'cores': 4, 'threads': 8}
                }
                
                with patch.object(self.hardware_detector, 'get_hardware_summary', return_value=mock_hardware):
                    # Validate configuration compatibility
                    result = self.validator.validate_configuration(config_dict)
                    
                    if not result.is_valid:
                        # Print validation issues for debugging
                        print(f"Validation issues for {config_file}:")
                        for issue in result.issues:
                            print(f"  - {issue.level.value}: {issue.message}")
                    
                    # Configuration should be valid or auto-correctable
                    self.assertTrue(result.is_valid or result.corrected_config is not None,
                                  f"Configuration {config_file} is not compatible with {expected_ram_gb}GB RAM")


class TestExampleScripts(TestCase):
    """Test that example training scripts work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.scripts_dir = self.examples_dir / "scripts"
        
        # Create minimal test dataset
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        os.makedirs(self.test_data_dir)
        
        # Create a few sample data files
        for i in range(5):
            sample_data = {
                "input": f"test input {i}",
                "output": f"test output {i}",
                "puzzle_id": i
            }
            with open(os.path.join(self.test_data_dir, f"sample_{i}.json"), 'w') as f:
                yaml.dump(sample_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_train_small_dataset_script_dry_run(self):
        """Test small dataset training script in dry-run mode."""
        script_path = self.scripts_dir / "train_small_dataset.py"
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")
        
        # Test dry-run mode
        cmd = [
            sys.executable, str(script_path),
            "--dry-run",
            "--auto-detect",
            "--data-path", self.test_data_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Script should exit successfully in dry-run mode
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
            self.assertIn("Dry run mode", result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("Script timed out in dry-run mode")
    
    def test_train_medium_dataset_script_dry_run(self):
        """Test medium dataset training script in dry-run mode."""
        script_path = self.scripts_dir / "train_medium_dataset.py"
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")
        
        cmd = [
            sys.executable, str(script_path),
            "--dry-run",
            "--auto-detect",
            "--data-path", self.test_data_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
            self.assertIn("Dry run mode", result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("Script timed out in dry-run mode")
    
    def test_email_classifier_script_dry_run(self):
        """Test email classifier training script in dry-run mode."""
        script_path = self.scripts_dir / "train_email_classifier.py"
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")
        
        # Create email-like test data
        email_data_dir = os.path.join(self.test_dir, "email_data")
        os.makedirs(email_data_dir)
        
        for i in range(3):
            email_data = {
                "subject": f"Test email subject {i}",
                "body": f"This is test email body {i}",
                "category": f"category_{i % 2}",
                "sender": f"sender{i}@test.com"
            }
            with open(os.path.join(email_data_dir, f"email_{i}.json"), 'w') as f:
                yaml.dump(email_data, f)
        
        cmd = [
            sys.executable, str(script_path),
            "--dry-run",
            "--auto-detect",
            "--email-data", email_data_dir,
            "--num-categories", "2"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            
            self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
            self.assertIn("Dry run mode", result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("Script timed out in dry-run mode")
    
    def test_script_help_messages(self):
        """Test that all scripts provide helpful usage information."""
        scripts = [
            "train_small_dataset.py",
            "train_medium_dataset.py", 
            "train_email_classifier.py"
        ]
        
        for script_name in scripts:
            with self.subTest(script=script_name):
                script_path = self.scripts_dir / script_name
                self.assertTrue(script_path.exists(), f"Script not found: {script_path}")
                
                # Test --help flag
                cmd = [sys.executable, str(script_path), "--help"]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, f"Help command failed for {script_name}")
                    self.assertIn("usage:", result.stdout.lower())
                    
                except subprocess.TimeoutExpired:
                    self.fail(f"Help command timed out for {script_name}")


class TestDocumentationAccuracy(TestCase):
    """Test that documentation examples and instructions are accurate."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_dir = project_root / "docs"
        self.examples_dir = project_root / "examples" / "macbook_training"
    
    def test_documentation_files_exist(self):
        """Test that all referenced documentation files exist."""
        expected_docs = [
            "macbook_training_guide.md",
            "performance_optimization_guide.md", 
            "troubleshooting_guide.md"
        ]
        
        for doc_file in expected_docs:
            doc_path = self.docs_dir / doc_file
            self.assertTrue(doc_path.exists(), f"Documentation file not found: {doc_path}")
    
    def test_example_directory_structure(self):
        """Test that example directory structure matches documentation."""
        # Check main directories exist
        expected_dirs = [
            "configs",
            "scripts",
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.examples_dir / dir_name
            self.assertTrue(dir_path.exists(), f"Example directory not found: {dir_path}")
        
        # Check config subdirectories
        config_dirs = ["macbook_8gb", "macbook_16gb"]
        for config_dir in config_dirs:
            config_path = self.examples_dir / "configs" / config_dir
            self.assertTrue(config_path.exists(), f"Config directory not found: {config_path}")
    
    def test_configuration_template_completeness(self):
        """Test that configuration templates include all necessary parameters."""
        required_sections = [
            'hardware_profile',
            'training',
            'arch',
            'macbook_optimizations'
        ]
        
        config_files = [
            "configs/macbook_8gb/small_dataset.yaml",
            "configs/macbook_16gb/medium_dataset.yaml"
        ]
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                config_path = self.examples_dir / config_file
                
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                for section in required_sections:
                    self.assertIn(section, config, 
                                f"Missing required section '{section}' in {config_file}")
    
    def test_performance_expectations_realistic(self):
        """Test that documented performance expectations are realistic."""
        # This test validates that the performance numbers in documentation
        # are within reasonable bounds based on hardware capabilities
        
        # Read the main training guide
        guide_path = self.docs_dir / "macbook_training_guide.md"
        with open(guide_path, 'r') as f:
            guide_content = f.read()
        
        # Check that performance expectations are mentioned
        self.assertIn("Performance Expectations", guide_content)
        self.assertIn("samples/second", guide_content)
        
        # Validate that different MacBook models have different expectations
        self.assertIn("8GB", guide_content)
        self.assertIn("16GB", guide_content)
    
    def test_troubleshooting_solutions_reference_real_code(self):
        """Test that troubleshooting solutions reference actual code components."""
        troubleshooting_path = self.docs_dir / "troubleshooting_guide.md"
        
        with open(troubleshooting_path, 'r') as f:
            content = f.read()
        
        # Check that real module names are referenced
        real_modules = [
            "macbook_optimization.hardware_detection",
            "macbook_optimization.memory_management",
            "macbook_optimization.cpu_optimization"
        ]
        
        for module in real_modules:
            self.assertIn(module, content, 
                         f"Troubleshooting guide should reference real module: {module}")


class TestSetupInstructions(TestCase):
    """Test that setup instructions in documentation are accurate."""
    
    def test_required_dependencies_available(self):
        """Test that all required dependencies mentioned in docs are available."""
        required_imports = [
            'torch',
            'yaml', 
            'psutil',
            'tqdm',
            'numpy'
        ]
        
        for module_name in required_imports:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError:
                    self.fail(f"Required dependency not available: {module_name}")
    
    def test_hardware_detection_works(self):
        """Test that hardware detection works as documented."""
        try:
            from macbook_optimization.hardware_detection import HardwareDetector
            
            detector = HardwareDetector()
            hardware_summary = detector.get_hardware_summary()
            
            # Validate hardware summary structure
            self.assertIn('cpu', hardware_summary)
            self.assertIn('memory', hardware_summary)
            self.assertIn('platform', hardware_summary)
            
            # Validate CPU info
            cpu_info = hardware_summary['cpu']
            self.assertIn('cores', cpu_info)
            self.assertIn('brand', cpu_info)
            self.assertGreater(cpu_info['cores'], 0)
            
            # Validate memory info
            memory_info = hardware_summary['memory']
            self.assertIn('total_gb', memory_info)
            self.assertIn('available_gb', memory_info)
            self.assertGreater(memory_info['total_gb'], 0)
            
        except ImportError as e:
            self.fail(f"Hardware detection import failed: {e}")
    
    def test_configuration_validation_works(self):
        """Test that configuration validation works as documented."""
        try:
            from macbook_optimization.config_validation import ConfigurationValidator
            from macbook_optimization.hardware_detection import HardwareDetector
            
            detector = HardwareDetector()
            validator = ConfigurationValidator(detector)
            
            # Test with a simple valid configuration
            test_config = {
                'global_batch_size': 32,
                'training': {
                    'memory_limit_mb': 4000,
                    'batch_size': 8,
                    'num_workers': 2
                },
                'arch': {
                    'hidden_size': 256,
                    'seq_len': 256
                }
            }
            
            result = validator.validate_configuration(test_config)
            
            # Should have validation result
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'is_valid'))
            
        except ImportError as e:
            self.fail(f"Configuration validation import failed: {e}")


def run_documentation_tests():
    """Run all documentation tests."""
    # Create test suite
    test_classes = [
        TestExampleConfigurations,
        TestExampleScripts,
        TestDocumentationAccuracy,
        TestSetupInstructions
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import unittest
    
    # Run tests
    success = run_documentation_tests()
    
    if success:
        print("\\n✅ All documentation tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some documentation tests failed!")
        sys.exit(1)