#!/usr/bin/env python3
"""
Tests for example configuration files.

This test suite validates that all example configurations are syntactically
correct, semantically valid, and compatible with their target hardware.
"""

import os
import sys
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.config_validation import ConfigurationValidator
from macbook_optimization.training_config_adapter import TrainingConfigAdapter
from pretrain import PretrainConfig


class TestConfigurationSyntax(TestCase):
    """Test that all configuration files have correct YAML syntax."""
    
    def setUp(self):
        """Set up test environment."""
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.configs_dir = self.examples_dir / "configs"
    
    def test_all_yaml_files_parse_correctly(self):
        """Test that all YAML configuration files parse without errors."""
        yaml_files = []
        
        # Find all YAML files in configs directory
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    yaml_files.append(os.path.join(root, file))
        
        self.assertGreater(len(yaml_files), 0, "No YAML configuration files found")
        
        for yaml_file in yaml_files:
            with self.subTest(yaml_file=yaml_file):
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    self.assertIsInstance(config, dict, 
                                        f"Configuration should be a dictionary: {yaml_file}")
                    self.assertGreater(len(config), 0, 
                                     f"Configuration should not be empty: {yaml_file}")
                    
                except yaml.YAMLError as e:
                    self.fail(f"YAML parsing error in {yaml_file}: {e}")
                except Exception as e:
                    self.fail(f"Unexpected error parsing {yaml_file}: {e}")
    
    def test_required_sections_present(self):
        """Test that all configurations have required sections."""
        required_sections = [
            'hardware_profile',
            'training', 
            'arch',
            'macbook_optimizations'
        ]
        
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                for section in required_sections:
                    self.assertIn(section, config, 
                                f"Missing required section '{section}' in {config_file}")
    
    def test_hardware_profile_consistency(self):
        """Test that hardware_profile matches directory structure."""
        config_mappings = {
            'macbook_8gb': ['macbook_8gb'],
            'macbook_16gb': ['macbook_16gb'],
            'macbook_32gb': ['macbook_32gb']
        }
        
        for expected_profile, directories in config_mappings.items():
            for directory in directories:
                config_dir = self.configs_dir / directory
                if config_dir.exists():
                    for config_file in config_dir.glob('*.yaml'):
                        with self.subTest(config_file=str(config_file)):
                            with open(config_file, 'r') as f:
                                config = yaml.safe_load(f)
                            
                            actual_profile = config.get('hardware_profile')
                            self.assertEqual(actual_profile, expected_profile,
                                           f"Hardware profile mismatch in {config_file}")
    
    def _get_all_config_files(self):
        """Get list of all configuration files."""
        config_files = []
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    config_files.append(os.path.join(root, file))
        return config_files


class TestConfigurationSemantics(TestCase):
    """Test that configurations are semantically valid."""
    
    def setUp(self):
        """Set up test environment."""
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.configs_dir = self.examples_dir / "configs"
        self.hardware_detector = HardwareDetector()
        self.validator = ConfigurationValidator(self.hardware_detector)
    
    def test_memory_limits_reasonable(self):
        """Test that memory limits are reasonable for target hardware."""
        memory_expectations = {
            'macbook_8gb': (2000, 5000),    # 2-5GB for 8GB system
            'macbook_16gb': (4000, 10000),  # 4-10GB for 16GB system
            'macbook_32gb': (8000, 20000)   # 8-20GB for 32GB system
        }
        
        config_files = self._get_config_files_by_hardware()
        
        for hardware_profile, files in config_files.items():
            if hardware_profile in memory_expectations:
                min_memory, max_memory = memory_expectations[hardware_profile]
                
                for config_file in files:
                    with self.subTest(config_file=config_file):
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        memory_limit = config['training']['memory_limit_mb']
                        self.assertGreaterEqual(memory_limit, min_memory,
                                              f"Memory limit too low in {config_file}")
                        self.assertLessEqual(memory_limit, max_memory,
                                           f"Memory limit too high in {config_file}")
    
    def test_batch_sizes_appropriate(self):
        """Test that batch sizes are appropriate for memory constraints."""
        config_files = self._get_config_files_by_hardware()
        
        for hardware_profile, files in config_files.items():
            for config_file in files:
                with self.subTest(config_file=config_file):
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    global_batch_size = config.get('global_batch_size', 32)
                    physical_batch_size = config['arch'].get('batch_size', 8)
                    
                    # Global batch size should be reasonable
                    self.assertGreater(global_batch_size, 0)
                    self.assertLessEqual(global_batch_size, 512)  # Not too large
                    
                    # Physical batch size should be smaller than global
                    self.assertGreater(physical_batch_size, 0)
                    self.assertLessEqual(physical_batch_size, global_batch_size)
    
    def test_architecture_parameters_valid(self):
        """Test that architecture parameters are valid."""
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                arch = config['arch']
                
                # Test required architecture parameters
                required_arch_params = ['hidden_size', 'num_heads', 'seq_len']
                for param in required_arch_params:
                    self.assertIn(param, arch, f"Missing architecture parameter: {param}")
                    self.assertGreater(arch[param], 0, f"Invalid {param} value")
                
                # Test parameter relationships
                if 'hidden_size' in arch and 'num_heads' in arch:
                    self.assertEqual(arch['hidden_size'] % arch['num_heads'], 0,
                                   "hidden_size must be divisible by num_heads")
                
                # Test reasonable ranges
                self.assertLessEqual(arch['hidden_size'], 1024, "hidden_size too large")
                self.assertLessEqual(arch['num_heads'], 32, "num_heads too large")
                self.assertLessEqual(arch['seq_len'], 2048, "seq_len too large")
    
    def test_macbook_optimizations_valid(self):
        """Test that MacBook optimization settings are valid."""
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                opt = config['macbook_optimizations']
                
                # Test boolean flags
                boolean_flags = [
                    'enable_memory_monitoring',
                    'use_mkl',
                    'optimize_tensor_operations',
                    'enable_vectorization'
                ]
                
                for flag in boolean_flags:
                    if flag in opt:
                        self.assertIsInstance(opt[flag], bool, 
                                            f"{flag} should be boolean")
                
                # Test numeric thresholds
                if 'memory_pressure_threshold' in opt:
                    threshold = opt['memory_pressure_threshold']
                    self.assertGreater(threshold, 0)
                    self.assertLessEqual(threshold, 100)
                
                if 'thermal_throttle_threshold' in opt:
                    temp_threshold = opt['thermal_throttle_threshold']
                    self.assertGreater(temp_threshold, 50)  # Reasonable minimum
                    self.assertLessEqual(temp_threshold, 100)  # Reasonable maximum
    
    def _get_all_config_files(self):
        """Get list of all configuration files."""
        config_files = []
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    config_files.append(os.path.join(root, file))
        return config_files
    
    def _get_config_files_by_hardware(self):
        """Get configuration files grouped by hardware profile."""
        config_files = {}
        
        for config_file in self._get_all_config_files():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            hardware_profile = config.get('hardware_profile', 'unknown')
            if hardware_profile not in config_files:
                config_files[hardware_profile] = []
            config_files[hardware_profile].append(config_file)
        
        return config_files


class TestConfigurationCompatibility(TestCase):
    """Test that configurations are compatible with target hardware."""
    
    def setUp(self):
        """Set up test environment."""
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.configs_dir = self.examples_dir / "configs"
        self.hardware_detector = HardwareDetector()
        self.validator = ConfigurationValidator(self.hardware_detector)
    
    def test_8gb_configurations_compatible(self):
        """Test that 8GB configurations are compatible with 8GB hardware."""
        mock_hardware = {
            'memory': {'total_gb': 8, 'available_gb': 5.6},
            'cpu': {'cores': 4, 'threads': 8, 'brand': 'Intel'},
            'platform': {'system': 'Darwin', 'machine': 'x86_64'}
        }
        
        config_dir = self.configs_dir / "macbook_8gb"
        if config_dir.exists():
            for config_file in config_dir.glob('*.yaml'):
                with self.subTest(config_file=str(config_file)):
                    self._test_config_compatibility(config_file, mock_hardware)
    
    def test_16gb_configurations_compatible(self):
        """Test that 16GB configurations are compatible with 16GB hardware."""
        mock_hardware = {
            'memory': {'total_gb': 16, 'available_gb': 11.2},
            'cpu': {'cores': 8, 'threads': 16, 'brand': 'Intel'},
            'platform': {'system': 'Darwin', 'machine': 'x86_64'}
        }
        
        config_dir = self.configs_dir / "macbook_16gb"
        if config_dir.exists():
            for config_file in config_dir.glob('*.yaml'):
                with self.subTest(config_file=str(config_file)):
                    self._test_config_compatibility(config_file, mock_hardware)
    
    def test_configurations_create_valid_pretrain_config(self):
        """Test that configurations can create valid PretrainConfig objects."""
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                # Remove MacBook-specific sections that aren't part of PretrainConfig
                pretrain_dict = {k: v for k, v in config_dict.items() 
                               if k not in ['hardware_profile', 'macbook_optimizations']}
                
                # Add required fields that might be missing
                if 'data_paths' not in pretrain_dict:
                    pretrain_dict['data_paths'] = ['data/test']
                
                try:
                    # This should not raise an exception
                    config = PretrainConfig(**pretrain_dict)
                    self.assertIsInstance(config, PretrainConfig)
                    
                except Exception as e:
                    self.fail(f"Failed to create PretrainConfig from {config_file}: {e}")
    
    def _test_config_compatibility(self, config_file, mock_hardware):
        """Test configuration compatibility with mock hardware."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        with patch.object(self.hardware_detector, 'get_hardware_summary', 
                         return_value=mock_hardware):
            result = self.validator.validate_configuration(config_dict, auto_correct=True)
            
            # Configuration should be valid or auto-correctable
            is_compatible = result.is_valid or (result.corrected_config is not None)
            
            if not is_compatible:
                print(f"Compatibility issues for {config_file}:")
                for issue in result.issues:
                    print(f"  - {issue.level.value}: {issue.message}")
            
            self.assertTrue(is_compatible, 
                          f"Configuration {config_file} is not compatible with target hardware")
    
    def _get_all_config_files(self):
        """Get list of all configuration files."""
        config_files = []
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    config_files.append(os.path.join(root, file))
        return config_files


class TestConfigurationCompleteness(TestCase):
    """Test that configurations are complete and include all necessary settings."""
    
    def setUp(self):
        """Set up test environment."""
        self.examples_dir = project_root / "examples" / "macbook_training"
        self.configs_dir = self.examples_dir / "configs"
    
    def test_training_section_complete(self):
        """Test that training sections include all necessary parameters."""
        required_training_params = [
            'memory_limit_mb',
            'num_workers',
            'checkpoint_interval'
        ]
        
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                training = config['training']
                
                for param in required_training_params:
                    self.assertIn(param, training, 
                                f"Missing training parameter '{param}' in {config_file}")
    
    def test_arch_section_complete(self):
        """Test that architecture sections include all necessary parameters."""
        required_arch_params = [
            'hidden_size',
            'num_heads', 
            'seq_len',
            'batch_size'
        ]
        
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                arch = config['arch']
                
                for param in required_arch_params:
                    self.assertIn(param, arch, 
                                f"Missing architecture parameter '{param}' in {config_file}")
    
    def test_macbook_optimizations_complete(self):
        """Test that MacBook optimization sections are complete."""
        required_opt_params = [
            'enable_memory_monitoring',
            'use_mkl',
            'streaming_threshold_mb'
        ]
        
        config_files = self._get_all_config_files()
        
        for config_file in config_files:
            with self.subTest(config_file=config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                opt = config['macbook_optimizations']
                
                for param in required_opt_params:
                    self.assertIn(param, opt, 
                                f"Missing optimization parameter '{param}' in {config_file}")
    
    def test_dataset_size_configurations_exist(self):
        """Test that configurations exist for different dataset sizes."""
        expected_configs = [
            ('macbook_8gb', 'small_dataset.yaml'),
            ('macbook_8gb', 'medium_dataset.yaml'),
            ('macbook_16gb', 'small_dataset.yaml'),
            ('macbook_16gb', 'medium_dataset.yaml'),
            ('macbook_16gb', 'large_dataset.yaml')
        ]
        
        for hardware_profile, config_name in expected_configs:
            config_path = self.configs_dir / hardware_profile / config_name
            self.assertTrue(config_path.exists(), 
                          f"Missing expected configuration: {config_path}")
    
    def _get_all_config_files(self):
        """Get list of all configuration files."""
        config_files = []
        for root, dirs, files in os.walk(self.configs_dir):
            for file in files:
                if file.endswith('.yaml'):
                    config_files.append(os.path.join(root, file))
        return config_files


def run_configuration_tests():
    """Run all configuration tests."""
    test_classes = [
        TestConfigurationSyntax,
        TestConfigurationSemantics,
        TestConfigurationCompatibility,
        TestConfigurationCompleteness
    ]
    
    import unittest
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_configuration_tests()
    
    if success:
        print("\\n✅ All configuration tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some configuration tests failed!")
        sys.exit(1)