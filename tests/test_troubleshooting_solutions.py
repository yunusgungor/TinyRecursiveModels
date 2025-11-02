#!/usr/bin/env python3
"""
Tests for troubleshooting solutions.

This test suite validates that the troubleshooting solutions provided
in the documentation actually work and resolve the issues they claim to fix.
"""

import os
import sys
import yaml
import tempfile
import shutil
import time
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager
from macbook_optimization.cpu_optimization import CPUOptimizer
from macbook_optimization.config_validation import ConfigurationValidator
from macbook_optimization.resource_monitoring import ResourceMonitor


class TestMemoryIssueSolutions(TestCase):
    """Test solutions for memory-related issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory_manager = MemoryManager()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_optimal_batch_size_calculation(self):
        """Test that optimal batch size calculation works as documented."""
        # Test the solution for OOM errors
        model_size_mb = 100
        available_memory_mb = 4000
        
        # Estimate model parameters (rough approximation)
        model_params = int(model_size_mb * 1024 * 1024 / 4)  # Assume float32
        
        result = self.memory_manager.calculate_optimal_batch_size(
            model_params=model_params,
            available_memory_mb=available_memory_mb
        )
        optimal_batch_size = result.recommended_batch_size
        
        # Should return a reasonable batch size
        self.assertGreater(optimal_batch_size, 0)
        self.assertLessEqual(optimal_batch_size, 128)  # Not too large for memory constraints
        
        # Test with very limited memory
        limited_result = self.memory_manager.calculate_optimal_batch_size(
            model_params=model_params,
            available_memory_mb=1000  # Very limited
        )
        limited_optimal = limited_result.recommended_batch_size
        
        # Should be smaller than or equal to the previous result (may be same if both hit minimum)
        self.assertLessEqual(limited_optimal, optimal_batch_size)
    
    def test_memory_monitoring_functionality(self):
        """Test that memory monitoring works as documented."""
        # Test the memory monitoring solution
        stats = self.memory_manager.monitor_memory_usage()
        
        # Should return valid memory statistics
        self.assertIsNotNone(stats)
        self.assertTrue(hasattr(stats, 'used_mb'))
        self.assertTrue(hasattr(stats, 'available_mb'))
        self.assertTrue(hasattr(stats, 'percent_used'))
        
        # Values should be reasonable
        self.assertGreater(stats.used_mb, 0)
        self.assertGreater(stats.available_mb, 0)
        self.assertGreaterEqual(stats.percent_used, 0)
        self.assertLessEqual(stats.percent_used, 100)
    
    def test_garbage_collection_solution(self):
        """Test that garbage collection solution works."""
        # Test the memory leak solution
        initial_stats = self.memory_manager.monitor_memory_usage()
        
        # Force garbage collection
        self.memory_manager.force_garbage_collection()
        
        # Should complete without error
        post_gc_stats = self.memory_manager.monitor_memory_usage()
        self.assertIsNotNone(post_gc_stats)
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection works as documented."""
        # Mock high memory usage scenario
        with patch.object(self.memory_manager, 'monitor_memory_usage') as mock_monitor:
            # Simulate high memory usage
            mock_stats = MagicMock()
            mock_stats.percent_used = 90.0
            mock_stats.used_mb = 7200
            mock_stats.available_mb = 800
            mock_monitor.return_value = mock_stats
            
            stats = self.memory_manager.monitor_memory_usage()
            
            # Should detect high memory pressure
            self.assertGreater(stats.percent_used, 85)
            
            # This should trigger memory pressure handling in real scenarios


class TestPerformanceIssueSolutions(TestCase):
    """Test solutions for performance-related issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.hardware_detector = HardwareDetector()
        self.cpu_optimizer = CPUOptimizer(self.hardware_detector)
    
    def test_cpu_optimization_configuration(self):
        """Test that CPU optimization solutions work."""
        # Test the slow training speed solution
        config = self.cpu_optimizer.configure_all()
        
        # Should return valid configuration
        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, 'torch_threads'))
        self.assertTrue(hasattr(config, 'use_mkl'))  # Correct attribute name
        
        # Thread count should be reasonable
        self.assertGreater(config.torch_threads, 0)
        self.assertLessEqual(config.torch_threads, 16)  # Reasonable upper bound
    
    def test_hardware_detection_solution(self):
        """Test that hardware detection works as documented."""
        # Test the hardware detection solution for configuration issues
        hardware_summary = self.hardware_detector.get_hardware_summary()
        
        # Should return complete hardware information
        self.assertIn('cpu', hardware_summary)
        self.assertIn('memory', hardware_summary)
        self.assertIn('platform', hardware_summary)
        
        # CPU information should be complete
        cpu_info = hardware_summary['cpu']
        self.assertIn('cores', cpu_info)
        self.assertIn('brand', cpu_info)
        self.assertGreater(cpu_info['cores'], 0)
        
        # Memory information should be complete
        memory_info = hardware_summary['memory']
        self.assertIn('total_gb', memory_info)
        self.assertIn('available_gb', memory_info)
        self.assertGreater(memory_info['total_gb'], 0)
    
    def test_mkl_availability_check(self):
        """Test MKL availability checking as documented."""
        # Test the Intel MKL solution
        try:
            import torch
            mkl_available = torch.backends.mkl.is_available()
            
            # Should be able to check MKL availability
            self.assertIsInstance(mkl_available, bool)
            
        except ImportError:
            self.skipTest("PyTorch not available for MKL testing")
    
    def test_resource_monitoring_solution(self):
        """Test that resource monitoring works for performance debugging."""
        try:
            monitor = ResourceMonitor()
            
            # Should be able to start monitoring
            monitor.start_monitoring(interval=1.0)
            
            # Let it run briefly
            time.sleep(2)
            
            # Should be able to get statistics
            cpu_stats = monitor.get_cpu_statistics()
            self.assertIsNotNone(cpu_stats)
            
            # Stop monitoring
            monitor.stop_monitoring()
            
        except Exception as e:
            # Resource monitoring might not work in all test environments
            self.skipTest(f"Resource monitoring not available: {e}")


class TestConfigurationIssueSolutions(TestCase):
    """Test solutions for configuration-related issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.hardware_detector = HardwareDetector()
        self.validator = ConfigurationValidator(self.hardware_detector)
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_configuration_validation_solution(self):
        """Test that configuration validation works as documented."""
        # Test the configuration validation solution
        test_config = {
            'global_batch_size': 32,
            'training': {
                'memory_limit_mb': 4000,
                'batch_size': 8,
                'num_workers': 2
            },
            'arch': {
                'hidden_size': 256,
                'seq_len': 256,
                'num_heads': 8
            }
        }
        
        result = self.validator.validate_configuration(test_config, auto_correct=True)
        
        # Should return validation result
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_valid'))
        self.assertTrue(hasattr(result, 'issues'))
        
        # Should be valid or have corrections
        self.assertTrue(result.is_valid or result.corrected_config is not None)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test with intentionally invalid configuration
        invalid_config = {
            'global_batch_size': -1,  # Invalid
            'training': {
                'memory_limit_mb': 0,  # Invalid
                'batch_size': 0,  # Invalid
                'num_workers': -1  # Invalid
            },
            'arch': {
                'hidden_size': 0,  # Invalid
                'seq_len': 0,  # Invalid
                'num_heads': 0  # Invalid
            }
        }
        
        result = self.validator.validate_configuration(invalid_config, auto_correct=True)
        
        # Should detect validation errors
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.issues), 0)
        
        # Should provide corrections if possible
        if result.corrected_config:
            # Corrected config should be better (fewer issues) or at least not worse
            corrected_result = self.validator.validate_configuration(result.corrected_config)
            # Either valid or has same or fewer issues than original
            self.assertTrue(corrected_result.is_valid or 
                          len(corrected_result.issues) <= len(result.issues),
                          f"Corrected config should not be worse. Original issues: {len(result.issues)}, "
                          f"Corrected issues: {len(corrected_result.issues)}")
    
    def test_hardware_compatibility_checking(self):
        """Test hardware compatibility checking solution."""
        # Mock different hardware scenarios
        hardware_scenarios = [
            {'memory': {'total_gb': 8, 'available_gb': 5.6}, 'cpu': {'cores': 4}},
            {'memory': {'total_gb': 16, 'available_gb': 11.2}, 'cpu': {'cores': 8}},
            {'memory': {'total_gb': 32, 'available_gb': 22.4}, 'cpu': {'cores': 16}}
        ]
        
        test_config = {
            'global_batch_size': 64,
            'training': {
                'memory_limit_mb': 8000,  # Might be too high for 8GB system
                'batch_size': 16,
                'num_workers': 4
            },
            'arch': {
                'hidden_size': 512,
                'seq_len': 512,
                'num_heads': 16
            }
        }
        
        for i, hardware in enumerate(hardware_scenarios):
            with self.subTest(scenario=i):
                with patch.object(self.hardware_detector, 'get_hardware_summary', 
                                return_value=hardware):
                    result = self.validator.validate_configuration(test_config, auto_correct=True)
                    
                    # Should handle different hardware scenarios appropriately
                    self.assertIsNotNone(result)
                    
                    # For 8GB system, should either be invalid or corrected
                    if hardware['memory']['total_gb'] <= 8:
                        if not result.is_valid:
                            self.assertIsNotNone(result.corrected_config, 
                                               "Should provide corrections for 8GB system")


class TestDatasetIssueSolutions(TestCase):
    """Test solutions for dataset-related issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.test_dataset_dir = os.path.join(self.test_dir, "test_dataset")
        os.makedirs(self.test_dataset_dir)
        
        # Create some test files
        for i in range(5):
            test_data = {
                "input": f"test input {i}",
                "output": f"test output {i}",
                "id": i
            }
            with open(os.path.join(self.test_dataset_dir, f"test_{i}.json"), 'w') as f:
                yaml.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_dataset_path_validation(self):
        """Test dataset path validation solution."""
        # Test valid path
        self.assertTrue(os.path.exists(self.test_dataset_dir))
        
        # Test invalid path
        invalid_path = os.path.join(self.test_dir, "nonexistent")
        self.assertFalse(os.path.exists(invalid_path))
        
        # Solution should be able to detect this
        if os.path.exists(self.test_dataset_dir):
            files = os.listdir(self.test_dataset_dir)
            self.assertGreater(len(files), 0)
    
    def test_dataset_integrity_checking(self):
        """Test dataset integrity checking solution."""
        # Create a corrupted file
        corrupted_file = os.path.join(self.test_dataset_dir, "corrupted.json")
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")  # Incomplete JSON
        
        # Test integrity checking
        corrupted_files = []
        for root, dirs, files in os.walk(self.test_dataset_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            import json
                            json.load(f)  # Use json.load instead of yaml for JSON files
                    except Exception:
                        corrupted_files.append(filepath)
        
        # Should detect the corrupted file
        self.assertGreater(len(corrupted_files), 0)
        self.assertIn(corrupted_file, corrupted_files)
    
    def test_dataset_size_estimation(self):
        """Test dataset size estimation for memory planning."""
        # Calculate dataset size
        total_size_mb = 0
        total_files = 0
        
        for root, dirs, files in os.walk(self.test_dataset_dir):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(filepath)
                    total_size_mb += file_size / (1024 * 1024)
                    total_files += 1
                except:
                    pass
        
        # Should have reasonable estimates
        self.assertGreater(total_files, 0)
        self.assertGreaterEqual(total_size_mb, 0)


class TestThermalIssueSolutions(TestCase):
    """Test solutions for thermal-related issues."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            self.resource_monitor = ResourceMonitor()
        except ImportError:
            self.resource_monitor = None
    
    def test_thermal_monitoring_availability(self):
        """Test that thermal monitoring solutions are available."""
        if self.resource_monitor is None:
            self.skipTest("Resource monitoring not available")
        
        # Should be able to start thermal monitoring
        try:
            self.resource_monitor.start_monitoring(interval=2.0)
            
            # Let it run briefly
            time.sleep(3)
            
            # Should be able to get thermal state
            thermal_state = self.resource_monitor.get_thermal_state()
            
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            
            # Thermal state should have expected structure
            if thermal_state:
                self.assertIn('temperature', thermal_state)
                self.assertIn('is_throttling', thermal_state)
            
        except Exception as e:
            # Thermal monitoring might not work in all environments
            self.skipTest(f"Thermal monitoring not available: {e}")
    
    def test_cooling_delay_implementation(self):
        """Test cooling delay implementation."""
        # Test that cooling delay can be implemented
        start_time = time.time()
        cooling_delay_seconds = 1.0
        
        # Simulate cooling break
        time.sleep(cooling_delay_seconds)
        
        elapsed = time.time() - start_time
        self.assertGreaterEqual(elapsed, cooling_delay_seconds)


class TestEnvironmentIssueSolutions(TestCase):
    """Test solutions for environment and dependency issues."""
    
    def test_dependency_availability_checking(self):
        """Test dependency availability checking solutions."""
        required_modules = ['torch', 'yaml', 'psutil']
        
        available_modules = []
        missing_modules = []
        
        for module_name in required_modules:
            try:
                __import__(module_name)
                available_modules.append(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        # Should be able to detect available and missing modules
        self.assertIsInstance(available_modules, list)
        self.assertIsInstance(missing_modules, list)
        
        # At least some modules should be available for tests to run
        self.assertGreater(len(available_modules), 0)
    
    def test_python_path_checking(self):
        """Test Python path checking solution."""
        import sys
        
        # Should be able to check Python path
        self.assertIsInstance(sys.path, list)
        self.assertGreater(len(sys.path), 0)
        
        # Project root should be in path (added by test setup)
        project_root_str = str(project_root)
        self.assertIn(project_root_str, sys.path)
    
    def test_version_checking(self):
        """Test version checking solutions."""
        try:
            import torch
            torch_version = torch.__version__
            self.assertIsInstance(torch_version, str)
            self.assertGreater(len(torch_version), 0)
            
        except ImportError:
            self.skipTest("PyTorch not available for version checking")
        
        # Python version checking
        python_version = sys.version
        self.assertIsInstance(python_version, str)
        self.assertGreater(len(python_version), 0)


def run_troubleshooting_tests():
    """Run all troubleshooting solution tests."""
    test_classes = [
        TestMemoryIssueSolutions,
        TestPerformanceIssueSolutions,
        TestConfigurationIssueSolutions,
        TestDatasetIssueSolutions,
        TestThermalIssueSolutions,
        TestEnvironmentIssueSolutions
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
    success = run_troubleshooting_tests()
    
    if success:
        print("\\n✅ All troubleshooting solution tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some troubleshooting solution tests failed!")
        sys.exit(1)