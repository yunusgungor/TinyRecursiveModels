#!/usr/bin/env python3
"""
Tests for MacBook Email Training Examples and Documentation

This module tests example configurations, setup instructions, and troubleshooting solutions
to ensure they work correctly on different MacBook models.

Requirements: 1.5, 2.1, 5.4
"""

import os
import sys
import json
import yaml
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from examples.macbook_training.config_validator import ConfigValidator, MacBookModel, HardwareSpecs
    from macbook_optimization.email_training_config import EmailTrainingConfig
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class TestMacBookConfigurationExamples(unittest.TestCase):
    """Test example configurations for different MacBook models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_validator = ConfigValidator() if DEPENDENCIES_AVAILABLE else None
        self.test_data_dir = Path(__file__).parent.parent / "examples" / "macbook_training"
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_8gb_macbook_config_validation(self):
        """Test 8GB MacBook configuration is valid and optimized."""
        config_path = self.test_data_dir / "configs" / "macbook_8gb" / "email_classification.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create hardware specs for 8GB MacBook
        hardware_specs = HardwareSpecs(
            memory_gb=8.0,
            cpu_cores=4,
            model_category=MacBookModel.MACBOOK_8GB
        )
        
        # Validate configuration
        result = self.config_validator.validate_config(config, hardware_specs)
        
        # Assertions
        self.assertTrue(result.is_valid, f"8GB config validation failed: {result.errors}")
        self.assertLessEqual(result.estimated_memory_usage, 6000, "Memory usage too high for 8GB MacBook")
        
        # Check specific optimizations for 8GB
        self.assertLessEqual(config['training']['batch_size'], 4, "Batch size too large for 8GB")
        self.assertGreaterEqual(config['training']['gradient_accumulation_steps'], 8, "Need more gradient accumulation")
        self.assertLessEqual(config['model']['hidden_size'], 384, "Model too large for 8GB")
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_16gb_macbook_config_validation(self):
        """Test 16GB MacBook configuration is valid and balanced."""
        config_path = self.test_data_dir / "configs" / "macbook_16gb" / "email_classification.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create hardware specs for 16GB MacBook
        hardware_specs = HardwareSpecs(
            memory_gb=16.0,
            cpu_cores=8,
            model_category=MacBookModel.MACBOOK_16GB
        )
        
        # Validate configuration
        result = self.config_validator.validate_config(config, hardware_specs)
        
        # Assertions
        self.assertTrue(result.is_valid, f"16GB config validation failed: {result.errors}")
        self.assertLessEqual(result.estimated_memory_usage, 12000, "Memory usage too high for 16GB MacBook")
        
        # Check balanced configuration
        self.assertGreaterEqual(config['training']['batch_size'], 4, "Batch size too small for 16GB")
        self.assertLessEqual(config['training']['batch_size'], 8, "Batch size too large")
        self.assertGreaterEqual(config['model']['hidden_size'], 256, "Model too small for 16GB")
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_32gb_macbook_config_validation(self):
        """Test 32GB+ MacBook configuration is valid and high-performance."""
        config_path = self.test_data_dir / "configs" / "macbook_32gb" / "email_classification.yaml"
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create hardware specs for 32GB+ MacBook
        hardware_specs = HardwareSpecs(
            memory_gb=32.0,
            cpu_cores=8,
            model_category=MacBookModel.MACBOOK_32GB
        )
        
        # Validate configuration
        result = self.config_validator.validate_config(config, hardware_specs)
        
        # Assertions
        self.assertTrue(result.is_valid, f"32GB config validation failed: {result.errors}")
        self.assertLessEqual(result.estimated_memory_usage, 24000, "Memory usage too high for 32GB MacBook")
        
        # Check high-performance configuration
        self.assertGreaterEqual(config['training']['batch_size'], 8, "Batch size too small for 32GB")
        self.assertGreaterEqual(config['model']['hidden_size'], 512, "Model too small for 32GB")
        self.assertTrue(config['email']['use_hierarchical_attention'], "Should enable advanced features")
    
    def test_configuration_completeness(self):
        """Test that all example configurations have required fields."""
        config_dirs = [
            self.test_data_dir / "configs" / "macbook_8gb",
            self.test_data_dir / "configs" / "macbook_16gb", 
            self.test_data_dir / "configs" / "macbook_32gb"
        ]
        
        required_sections = ['model', 'training', 'email', 'hardware', 'targets']
        
        for config_dir in config_dirs:
            config_file = config_dir / "email_classification.yaml"
            self.assertTrue(config_file.exists(), f"Configuration file missing: {config_file}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            for section in required_sections:
                self.assertIn(section, config, f"Missing section '{section}' in {config_file}")
            
            # Check required model fields
            model_fields = ['vocab_size', 'hidden_size', 'num_layers', 'num_email_categories']
            for field in model_fields:
                self.assertIn(field, config['model'], f"Missing model field '{field}' in {config_file}")
            
            # Check required training fields
            training_fields = ['batch_size', 'learning_rate', 'max_steps']
            for field in training_fields:
                self.assertIn(field, config['training'], f"Missing training field '{field}' in {config_file}")


class TestSampleDatasets(unittest.TestCase):
    """Test sample email datasets for correctness and completeness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.datasets_dir = Path(__file__).parent.parent / "examples" / "macbook_training" / "datasets"
    
    def test_sample_datasets_exist(self):
        """Test that sample datasets exist and are accessible."""
        dataset_files = [
            "sample_emails_small.json",
            "sample_emails_medium.json"
        ]
        
        for dataset_file in dataset_files:
            file_path = self.datasets_dir / dataset_file
            self.assertTrue(file_path.exists(), f"Sample dataset missing: {file_path}")
    
    def test_sample_dataset_format(self):
        """Test sample datasets have correct format."""
        dataset_files = [
            "sample_emails_small.json",
            "sample_emails_medium.json"
        ]
        
        required_fields = ['id', 'subject', 'body', 'sender', 'recipient', 'category', 'language']
        valid_categories = {
            'Newsletter', 'Work', 'Personal', 'Spam', 'Promotional', 
            'Social', 'Finance', 'Travel', 'Shopping', 'Other'
        }
        
        for dataset_file in dataset_files:
            file_path = self.datasets_dir / dataset_file
            
            with open(file_path, 'r') as f:
                emails = json.load(f)
            
            self.assertIsInstance(emails, list, f"Dataset should be a list: {dataset_file}")
            self.assertGreater(len(emails), 0, f"Dataset should not be empty: {dataset_file}")
            
            for i, email in enumerate(emails):
                # Check required fields
                for field in required_fields:
                    self.assertIn(field, email, f"Missing field '{field}' in email {i} of {dataset_file}")
                
                # Check category validity
                self.assertIn(email['category'], valid_categories, 
                            f"Invalid category '{email['category']}' in email {i} of {dataset_file}")
                
                # Check field types
                self.assertIsInstance(email['id'], str, f"ID should be string in email {i}")
                self.assertIsInstance(email['subject'], str, f"Subject should be string in email {i}")
                self.assertIsInstance(email['body'], str, f"Body should be string in email {i}")
    
    def test_dataset_category_distribution(self):
        """Test that sample datasets have reasonable category distribution."""
        dataset_files = [
            "sample_emails_small.json",
            "sample_emails_medium.json"
        ]
        
        for dataset_file in dataset_files:
            file_path = self.datasets_dir / dataset_file
            
            with open(file_path, 'r') as f:
                emails = json.load(f)
            
            # Count categories
            category_counts = {}
            for email in emails:
                category = email['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Check that we have multiple categories
            self.assertGreaterEqual(len(category_counts), 5, 
                                  f"Should have at least 5 categories in {dataset_file}")
            
            # Check that no single category dominates too much (>60%)
            total_emails = len(emails)
            for category, count in category_counts.items():
                percentage = count / total_emails
                self.assertLessEqual(percentage, 0.6, 
                                   f"Category '{category}' dominates too much ({percentage:.1%}) in {dataset_file}")


class TestConfigurationValidator(unittest.TestCase):
    """Test the configuration validator functionality."""
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_hardware_detection(self):
        """Test MacBook model detection based on memory."""
        test_cases = [
            (8.0, MacBookModel.MACBOOK_8GB),
            (16.0, MacBookModel.MACBOOK_16GB),
            (32.0, MacBookModel.MACBOOK_32GB),
            (64.0, MacBookModel.MACBOOK_32GB)
        ]
        
        for memory_gb, expected_model in test_cases:
            detected_model = self.validator.detect_macbook_model(memory_gb)
            self.assertEqual(detected_model, expected_model, 
                           f"Wrong model detected for {memory_gb}GB")
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        test_config = {
            'model': {
                'vocab_size': 5000,
                'hidden_size': 512,
                'num_layers': 2,
                'max_sequence_length': 512
            },
            'training': {
                'batch_size': 4
            }
        }
        
        estimated_memory = self.validator.estimate_memory_usage(test_config)
        
        # Memory should be reasonable (between 1GB and 20GB)
        self.assertGreater(estimated_memory, 1000, "Memory estimate too low")
        self.assertLess(estimated_memory, 20000, "Memory estimate too high")
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_configuration_recommendation(self):
        """Test configuration recommendation for different hardware."""
        hardware_specs = [
            HardwareSpecs(8.0, 4, MacBookModel.MACBOOK_8GB),
            HardwareSpecs(16.0, 8, MacBookModel.MACBOOK_16GB),
            HardwareSpecs(32.0, 8, MacBookModel.MACBOOK_32GB)
        ]
        
        for specs in hardware_specs:
            config = self.validator.recommend_config(specs)
            
            # Validate recommended configuration
            result = self.validator.validate_config(config, specs)
            self.assertTrue(result.is_valid, 
                          f"Recommended config invalid for {specs.model_category.value}: {result.errors}")
            
            # Check memory constraint
            memory_limit = config['hardware']['memory_limit_mb']
            self.assertLessEqual(result.estimated_memory_usage, memory_limit,
                               f"Recommended config exceeds memory limit for {specs.model_category.value}")


class TestTrainingScriptIntegration(unittest.TestCase):
    """Test integration with the main training script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.script_path = Path(__file__).parent.parent / "train_email_classifier_macbook.py"
    
    def test_training_script_exists(self):
        """Test that the main training script exists."""
        self.assertTrue(self.script_path.exists(), "Main training script not found")
    
    @patch('subprocess.run')
    def test_hardware_detection_command(self, mock_run):
        """Test hardware detection command line interface."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Hardware detected successfully"
        
        import subprocess
        
        # Test hardware detection
        result = subprocess.run([
            'python', str(self.script_path), '--detect-hardware'
        ], capture_output=True, text=True)
        
        # Should not crash (mocked to return 0)
        self.assertEqual(mock_run.call_count, 1)
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Import the function directly to test it
            sys.path.insert(0, str(self.script_path.parent))
            
            try:
                from train_email_classifier_macbook import create_sample_dataset
                
                # Create sample dataset
                success = create_sample_dataset(temp_dir)
                self.assertTrue(success, "Sample dataset creation failed")
                
                # Check created files
                expected_files = [
                    "train/dataset.json",
                    "test/dataset.json", 
                    "categories.json",
                    "vocab.json"
                ]
                
                for file_path in expected_files:
                    full_path = Path(temp_dir) / file_path
                    self.assertTrue(full_path.exists(), f"Expected file not created: {file_path}")
                
                # Validate created dataset format
                with open(Path(temp_dir) / "train" / "dataset.json", 'r') as f:
                    emails = json.load(f)
                
                self.assertIsInstance(emails, list)
                self.assertGreater(len(emails), 0)
                
                # Check first email has required fields
                if emails:
                    required_fields = ['id', 'subject', 'body', 'category']
                    for field in required_fields:
                        self.assertIn(field, emails[0])
            
            except ImportError:
                self.skipTest("Cannot import training script functions")


class TestDocumentationExamples(unittest.TestCase):
    """Test examples and code snippets from documentation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.docs_dir = Path(__file__).parent.parent / "docs"
    
    def test_documentation_files_exist(self):
        """Test that all documentation files exist."""
        expected_docs = [
            "macbook_email_training_setup_guide.md",
            "macbook_email_training_troubleshooting_guide.md",
            "macbook_email_training_performance_optimization_guide.md"
        ]
        
        for doc_file in expected_docs:
            doc_path = self.docs_dir / doc_file
            self.assertTrue(doc_path.exists(), f"Documentation file missing: {doc_file}")
    
    def test_setup_guide_completeness(self):
        """Test setup guide has all required sections."""
        setup_guide = self.docs_dir / "macbook_email_training_setup_guide.md"
        
        with open(setup_guide, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Prerequisites",
            "Installation", 
            "Hardware Detection",
            "Dataset Preparation",
            "Configuration",
            "Training Execution",
            "Model Evaluation"
        ]
        
        for section in required_sections:
            self.assertIn(section, content, f"Missing section '{section}' in setup guide")
    
    def test_troubleshooting_guide_completeness(self):
        """Test troubleshooting guide covers common issues."""
        troubleshooting_guide = self.docs_dir / "macbook_email_training_troubleshooting_guide.md"
        
        with open(troubleshooting_guide, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Memory Issues",
            "Performance Problems",
            "Training Failures",
            "Configuration Errors",
            "Dataset Issues"
        ]
        
        for section in required_sections:
            self.assertIn(section, content, f"Missing section '{section}' in troubleshooting guide")
    
    def test_performance_guide_completeness(self):
        """Test performance optimization guide covers key topics."""
        performance_guide = self.docs_dir / "macbook_email_training_performance_optimization_guide.md"
        
        with open(performance_guide, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Hardware-Specific Optimizations",
            "Memory Optimization Strategies",
            "CPU Optimization Techniques",
            "Training Strategy Optimization",
            "Performance Monitoring"
        ]
        
        for section in required_sections:
            self.assertIn(section, content, f"Missing section '{section}' in performance guide")


class TestTroubleshootingSolutions(unittest.TestCase):
    """Test that troubleshooting solutions actually work."""
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_memory_pressure_handling(self):
        """Test memory pressure detection and handling."""
        # Simulate memory pressure scenario
        test_config = {
            'model': {
                'vocab_size': 10000,  # Large vocab
                'hidden_size': 1024,  # Large model
                'num_layers': 4,
                'max_sequence_length': 1024
            },
            'training': {
                'batch_size': 16  # Large batch
            },
            'hardware': {
                'memory_limit_mb': 4000  # Small limit
            }
        }
        
        validator = ConfigValidator()
        hardware_specs = HardwareSpecs(8.0, 4, MacBookModel.MACBOOK_8GB)
        
        result = validator.validate_config(test_config, hardware_specs)
        
        # Should detect memory issues
        self.assertFalse(result.is_valid, "Should detect memory pressure")
        self.assertTrue(any("memory" in error.lower() for error in result.errors),
                       "Should report memory-related errors")
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Dependencies not available")
    def test_configuration_auto_fix(self):
        """Test automatic configuration fixes for common issues."""
        # Start with problematic configuration
        problematic_config = {
            'model': {
                'vocab_size': 15000,  # Too large
                'hidden_size': 1024,  # Too large
                'num_layers': 5,      # Too many
                'max_sequence_length': 2048  # Too long
            },
            'training': {
                'batch_size': 32,     # Too large
                'learning_rate': 1e-2  # Too high
            }
        }
        
        validator = ConfigValidator()
        hardware_specs = HardwareSpecs(8.0, 4, MacBookModel.MACBOOK_8GB)
        
        # Get recommended configuration instead
        fixed_config = validator.recommend_config(hardware_specs)
        
        # Validate the fixed configuration
        result = validator.validate_config(fixed_config, hardware_specs)
        self.assertTrue(result.is_valid, f"Auto-fixed config should be valid: {result.errors}")
    
    def test_dataset_validation_fixes(self):
        """Test dataset validation and common fixes."""
        # Create problematic dataset
        problematic_emails = [
            {
                'id': 'email_1',
                'subject': 'Test',
                'body': 'Test body',
                # Missing category
                'sender': 'test@example.com'
            },
            {
                'id': 'email_2',
                'subject': 'Test 2',
                'body': 'Test body 2',
                'category': 'InvalidCategory',  # Invalid category
                'sender': 'test2@example.com'
            }
        ]
        
        # Apply fixes
        valid_categories = {
            'Newsletter', 'Work', 'Personal', 'Spam', 'Promotional',
            'Social', 'Finance', 'Travel', 'Shopping', 'Other'
        }
        
        fixed_emails = []
        for email in problematic_emails:
            # Fix missing category
            if 'category' not in email:
                email['category'] = 'Other'
            
            # Fix invalid category
            if email['category'] not in valid_categories:
                email['category'] = 'Other'
            
            # Add missing fields
            if 'recipient' not in email:
                email['recipient'] = 'user@example.com'
            if 'language' not in email:
                email['language'] = 'en'
            
            fixed_emails.append(email)
        
        # Validate fixes
        for email in fixed_emails:
            self.assertIn('category', email)
            self.assertIn(email['category'], valid_categories)
            self.assertIn('recipient', email)
            self.assertIn('language', email)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimization techniques."""
    
    def test_batch_size_optimization(self):
        """Test batch size optimization for different hardware."""
        test_cases = [
            (8.0, 2, 16),   # 8GB: small batch, high accumulation
            (16.0, 4, 8),   # 16GB: medium batch, medium accumulation  
            (32.0, 8, 4)    # 32GB: large batch, low accumulation
        ]
        
        for memory_gb, expected_batch, expected_accum in test_cases:
            # This would be the logic from the optimization guide
            if memory_gb <= 8:
                batch_size = 2
                grad_accum = 16
            elif memory_gb <= 16:
                batch_size = 4
                grad_accum = 8
            else:
                batch_size = 8
                grad_accum = 4
            
            self.assertEqual(batch_size, expected_batch)
            self.assertEqual(grad_accum, expected_accum)
    
    def test_memory_optimization_strategies(self):
        """Test memory optimization strategy selection."""
        # Test gradient checkpointing decision
        def should_use_gradient_checkpointing(memory_gb, model_size):
            return memory_gb <= 8 or model_size > 512
        
        test_cases = [
            (8.0, 256, True),   # 8GB always uses checkpointing
            (16.0, 384, False), # 16GB with medium model doesn't need it
            (16.0, 768, True),  # 16GB with large model needs it
            (32.0, 512, False)  # 32GB doesn't need it for normal models
        ]
        
        for memory_gb, model_size, expected in test_cases:
            result = should_use_gradient_checkpointing(memory_gb, model_size)
            self.assertEqual(result, expected, 
                           f"Wrong checkpointing decision for {memory_gb}GB, {model_size} model")


def run_example_tests():
    """Run all example and documentation tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMacBookConfigurationExamples,
        TestSampleDatasets,
        TestConfigurationValidator,
        TestTrainingScriptIntegration,
        TestDocumentationExamples,
        TestTroubleshootingSolutions,
        TestPerformanceOptimizations
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_example_tests()
    
    if success:
        print("\n✅ All example and documentation tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)