"""
Unit tests for configuration management module.

Tests configuration detection, validation, and management
for MacBook-specific TRM training optimization.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from macbook_optimization.config_management import (
    MacBookTrainingConfig,
    MacBookConfigManager
)
from macbook_optimization.hardware_detection import CPUSpecs, MemorySpecs, PlatformCapabilities


class TestMacBookTrainingConfig:
    """Test MacBookTrainingConfig dataclass."""
    
    def test_config_creation(self):
        """Test MacBookTrainingConfig creation."""
        config = MacBookTrainingConfig(
            batch_size=16,
            gradient_accumulation_steps=4,
            num_workers=4,
            pin_memory=False,
            memory_limit_mb=6000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        assert config.batch_size == 16
        assert config.gradient_accumulation_steps == 4
        assert config.memory_limit_mb == 6000
        assert config.use_mkl is True
        assert config.learning_rate == 1e-4


class TestMacBookConfigManager:
    """Test MacBookConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for config files
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = MacBookConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        assert self.config_manager.config_dir.exists()
        assert isinstance(self.config_manager.hardware_detector, type(self.config_manager.hardware_detector))
    
    @patch('macbook_optimization.config_management.HardwareDetector')
    def test_detect_optimal_config(self, mock_detector_class):
        """Test optimal configuration detection."""
        # Mock hardware detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        # Mock hardware specs
        mock_detector.detect_cpu_specs.return_value = CPUSpecs(
            cores=4, threads=8, architecture="x86_64",
            features=["avx", "avx2"], base_frequency=2.4,
            max_frequency=3.8, brand="Intel", model="Intel"
        )
        mock_detector.detect_memory_specs.return_value = MemorySpecs(
            total_memory=8589934592,  # 8GB
            available_memory=6442450944,  # 6GB available
            memory_type="LPDDR3",
            memory_speed=2133
        )
        mock_detector.detect_platform_capabilities.return_value = PlatformCapabilities(
            has_mkl=True, has_accelerate=True, torch_version="1.12.0",
            python_version="3.9.0", macos_version="12.6.0",
            optimal_dtype="float32", supports_avx=True, supports_avx2=True
        )
        mock_detector.get_optimal_worker_count.return_value = 4
        mock_detector.get_hardware_summary.return_value = {"cpu": {"cores": 4}}
        
        # Create new config manager with mocked detector
        config_manager = MacBookConfigManager(config_dir=self.temp_dir)
        config_manager.hardware_detector = mock_detector
        
        config = config_manager.detect_optimal_config()
        
        assert isinstance(config, MacBookTrainingConfig)
        assert config.batch_size >= 1
        assert config.gradient_accumulation_steps >= 1
        assert config.num_workers == 4
        assert config.use_mkl is True
        assert config.memory_limit_mb > 0
    
    def test_save_and_load_config(self):
        """Test configuration saving and loading."""
        # Create test config
        config = MacBookTrainingConfig(
            batch_size=16,
            gradient_accumulation_steps=4,
            num_workers=4,
            pin_memory=False,
            memory_limit_mb=6000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={"test": "data"}
        )
        
        # Save config
        config_file = self.config_manager.save_config(config, "test_config")
        assert config_file.exists()
        
        # Load config
        loaded_config = self.config_manager.load_config("test_config")
        assert loaded_config is not None
        assert loaded_config.batch_size == 16
        assert loaded_config.memory_limit_mb == 6000
        assert loaded_config.hardware_summary == {"test": "data"}
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration."""
        config = self.config_manager.load_config("nonexistent")
        assert config is None
    
    def test_load_invalid_config(self):
        """Test loading invalid configuration file."""
        # Create invalid JSON file
        invalid_file = self.config_manager.config_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        config = self.config_manager.load_config("invalid")
        assert config is None
    
    def test_list_configs(self):
        """Test listing available configurations."""
        # Initially no configs
        configs = self.config_manager.list_configs()
        assert len(configs) == 0
        
        # Create test configs
        config = MacBookTrainingConfig(
            batch_size=16, gradient_accumulation_steps=4, num_workers=4,
            pin_memory=False, memory_limit_mb=6000, enable_memory_monitoring=True,
            dynamic_batch_sizing=True, memory_pressure_threshold=75.0,
            use_mkl=True, torch_threads=4, enable_cpu_optimization=True,
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            max_steps=10000, checkpoint_interval=500, max_checkpoints_to_keep=3,
            monitoring_interval=2.0, enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        self.config_manager.save_config(config, "config1")
        self.config_manager.save_config(config, "config2")
        
        configs = self.config_manager.list_configs()
        assert len(configs) == 2
        assert "config1" in configs
        assert "config2" in configs
    
    @patch('macbook_optimization.config_management.HardwareDetector')
    def test_validate_config(self, mock_detector_class):
        """Test configuration validation."""
        # Mock hardware detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        mock_detector.detect_memory_specs.return_value = MemorySpecs(
            total_memory=8589934592,  # 8GB
            available_memory=6442450944,  # 6GB available
            memory_type="LPDDR3",
            memory_speed=2133
        )
        mock_detector.detect_cpu_specs.return_value = CPUSpecs(
            cores=4, threads=8, architecture="x86_64",
            features=[], base_frequency=2.4,
            max_frequency=3.8, brand="Intel", model="Intel"
        )
        
        config_manager = MacBookConfigManager(config_dir=self.temp_dir)
        config_manager.hardware_detector = mock_detector
        
        # Test valid config
        valid_config = MacBookTrainingConfig(
            batch_size=8, gradient_accumulation_steps=4, num_workers=4,
            pin_memory=False, memory_limit_mb=5000, enable_memory_monitoring=True,
            dynamic_batch_sizing=True, memory_pressure_threshold=75.0,
            use_mkl=True, torch_threads=4, enable_cpu_optimization=True,
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            max_steps=10000, checkpoint_interval=500, max_checkpoints_to_keep=3,
            monitoring_interval=2.0, enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        result = config_manager.validate_config(valid_config)
        assert result["valid"] is True
        assert len(result["warnings"]) == 0
        
        # Test config with issues
        problematic_config = MacBookTrainingConfig(
            batch_size=64, gradient_accumulation_steps=4, num_workers=8,  # Too many workers
            pin_memory=False, memory_limit_mb=8000, enable_memory_monitoring=True,  # Too much memory
            dynamic_batch_sizing=True, memory_pressure_threshold=75.0,
            use_mkl=True, torch_threads=16, enable_cpu_optimization=True,  # Too many threads
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            max_steps=10000, checkpoint_interval=500, max_checkpoints_to_keep=3,
            monitoring_interval=2.0, enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        result = config_manager.validate_config(problematic_config)
        assert result["valid"] is False
        assert len(result["warnings"]) > 0
        assert len(result["suggestions"]) > 0
    
    def test_create_config_template(self):
        """Test configuration template creation."""
        # Test 8GB MacBook template
        config_8gb = self.config_manager.create_config_template(memory_gb=8, cpu_cores=4)
        assert isinstance(config_8gb, MacBookTrainingConfig)
        assert config_8gb.batch_size >= 1
        assert config_8gb.num_workers <= 4
        assert config_8gb.checkpoint_interval == 500  # More frequent for 8GB
        
        # Test 16GB MacBook template
        config_16gb = self.config_manager.create_config_template(memory_gb=16, cpu_cores=8)
        assert isinstance(config_16gb, MacBookTrainingConfig)
        assert config_16gb.checkpoint_interval == 1000  # Less frequent for 16GB
        assert config_16gb.memory_limit_mb > config_8gb.memory_limit_mb
    
    def test_get_config_summary(self):
        """Test configuration summary generation."""
        config = MacBookTrainingConfig(
            batch_size=16, gradient_accumulation_steps=4, num_workers=4,
            pin_memory=False, memory_limit_mb=6000, enable_memory_monitoring=True,
            dynamic_batch_sizing=True, memory_pressure_threshold=75.0,
            use_mkl=True, torch_threads=4, enable_cpu_optimization=True,
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            max_steps=10000, checkpoint_interval=500, max_checkpoints_to_keep=3,
            monitoring_interval=2.0, enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        summary = self.config_manager.get_config_summary(config)
        
        assert "training" in summary
        assert "hardware" in summary
        assert "monitoring" in summary
        assert "checkpointing" in summary
        
        assert summary["training"]["batch_size"] == 16
        assert summary["training"]["effective_batch_size"] == 64  # 16 * 4
        assert summary["hardware"]["memory_limit_mb"] == 6000
        assert summary["monitoring"]["memory_monitoring"] is True
    
    def test_memory_based_batch_size_calculation(self):
        """Test batch size calculation based on memory constraints."""
        with patch.object(self.config_manager.hardware_detector, 'detect_memory_specs') as mock_memory, \
             patch.object(self.config_manager.hardware_detector, 'detect_cpu_specs') as mock_cpu, \
             patch.object(self.config_manager.hardware_detector, 'detect_platform_capabilities') as mock_platform, \
             patch.object(self.config_manager.hardware_detector, 'get_optimal_worker_count') as mock_workers, \
             patch.object(self.config_manager.hardware_detector, 'get_hardware_summary') as mock_summary:
            
            # Test low memory scenario (4GB available)
            mock_memory.return_value = MemorySpecs(
                total_memory=8589934592,
                available_memory=4294967296,  # 4GB available
                memory_type="LPDDR3",
                memory_speed=2133
            )
            mock_cpu.return_value = CPUSpecs(4, 8, "x86_64", [], 2.4, 3.8, "Intel", "Intel")
            mock_platform.return_value = PlatformCapabilities(
                True, True, "1.12.0", "3.9.0", "12.6.0", "float32", True, True
            )
            mock_workers.return_value = 4
            mock_summary.return_value = {}
            
            config_low_mem = self.config_manager.detect_optimal_config()
            
            # Test high memory scenario (12GB available)
            mock_memory.return_value = MemorySpecs(
                total_memory=17179869184,  # 16GB total
                available_memory=12884901888,  # 12GB available
                memory_type="LPDDR4",
                memory_speed=3200
            )
            
            config_high_mem = self.config_manager.detect_optimal_config()
            
            # High memory config should allow larger batch size
            assert config_high_mem.batch_size >= config_low_mem.batch_size
            assert config_high_mem.memory_limit_mb > config_low_mem.memory_limit_mb