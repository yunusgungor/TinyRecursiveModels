"""
Tests for training configuration adapter module.

This module tests the TrainingConfigAdapter class and related functionality
for adapting TRM training configurations to MacBook hardware constraints.
"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

from macbook_optimization.training_config_adapter import (
    TrainingConfigAdapter, HardwareSpecs, TrainingParams, ConfigurationResult
)
from macbook_optimization.hardware_detection import CPUSpecs, MemorySpecs, PlatformCapabilities
from macbook_optimization.memory_management import BatchSizeRecommendation


@pytest.fixture
def mock_hardware_specs():
    """Create mock hardware specifications for testing."""
    cpu_specs = CPUSpecs(
        cores=4,
        threads=8,
        architecture="x86_64",
        features=["avx", "avx2", "sse4_2"],
        base_frequency=2.4,
        max_frequency=3.8,
        brand="Intel Core i5",
        model="Intel Core i5-8259U"
    )
    
    memory_specs = MemorySpecs(
        total_memory=8 * 1024**3,  # 8GB
        available_memory=6 * 1024**3,  # 6GB available
        memory_type="LPDDR3",
        memory_speed=2133
    )
    
    platform_caps = PlatformCapabilities(
        has_mkl=True,
        has_accelerate=True,
        torch_version="2.0.0",
        python_version="3.9.0",
        macos_version="12.0",
        optimal_dtype="float32",
        supports_avx=True,
        supports_avx2=True
    )
    
    return HardwareSpecs(
        cpu=cpu_specs,
        memory=memory_specs,
        platform=platform_caps,
        optimal_workers=2,
        hardware_summary={
            "cpu": {"cores": 4, "brand": "Intel Core i5"},
            "memory": {"available_gb": 6.0},
            "platform": {"has_mkl": True}
        }
    )


@pytest.fixture
def base_config():
    """Create base training configuration for testing."""
    return {
        "global_batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "epochs": 10,
        "model_params": 7_000_000,
        "seq_len": 512,
        "model_dim": 512,
        "num_layers": 6,
        "device": "cuda",  # Will be adapted to CPU
        "num_workers": 8,   # Will be adapted
        "pin_memory": True  # Will be adapted
    }


@pytest.fixture
def training_config_adapter():
    """Create TrainingConfigAdapter with mocked dependencies."""
    with patch('macbook_optimization.training_config_adapter.HardwareDetector') as mock_detector, \
         patch('macbook_optimization.training_config_adapter.MemoryManager') as mock_memory, \
         patch('macbook_optimization.training_config_adapter.CPUOptimizer') as mock_cpu, \
         patch('macbook_optimization.training_config_adapter.MacBookConfigManager') as mock_config:
        
        adapter = TrainingConfigAdapter()
        return adapter


class TestTrainingConfigAdapter:
    """Test cases for TrainingConfigAdapter class."""
    
    def test_initialization(self):
        """Test TrainingConfigAdapter initialization."""
        with patch('macbook_optimization.training_config_adapter.HardwareDetector') as mock_detector:
            adapter = TrainingConfigAdapter()
            assert adapter.hardware_detector is not None
            assert adapter.memory_manager is not None
            assert adapter.cpu_optimizer is not None
            assert adapter.config_manager is not None
    
    def test_get_hardware_specs_caching(self, training_config_adapter, mock_hardware_specs):
        """Test hardware specs caching."""
        with patch.object(training_config_adapter, 'hardware_detector') as mock_detector:
            mock_detector.detect_cpu_specs.return_value = mock_hardware_specs.cpu
            mock_detector.detect_memory_specs.return_value = mock_hardware_specs.memory
            mock_detector.detect_platform_capabilities.return_value = mock_hardware_specs.platform
            mock_detector.get_optimal_worker_count.return_value = mock_hardware_specs.optimal_workers
            mock_detector.get_hardware_summary.return_value = mock_hardware_specs.hardware_summary
            
            # First call should detect hardware
            specs1 = training_config_adapter.get_hardware_specs()
            
            # Second call should use cached specs
            specs2 = training_config_adapter.get_hardware_specs()
            
            assert specs1 == specs2
            mock_detector.detect_cpu_specs.assert_called_once()
    
    def test_adapt_model_config_batch_size(self, training_config_adapter, base_config, mock_hardware_specs):
        """Test batch size adaptation in model config."""
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=mock_hardware_specs), \
             patch.object(training_config_adapter.memory_manager, 'calculate_optimal_batch_size') as mock_calc:
            
            mock_calc.return_value = BatchSizeRecommendation(
                recommended_batch_size=16,
                max_safe_batch_size=24,
                memory_utilization_percent=70.0,
                reasoning="Memory constrained",
                warnings=[]
            )
            
            adapted_config = training_config_adapter.adapt_model_config(base_config, mock_hardware_specs)
            
            assert adapted_config['global_batch_size'] == 16
            assert 'gradient_accumulation_steps' in adapted_config
            assert adapted_config['gradient_accumulation_steps'] == 4  # 64 / 16
    
    def test_adapt_model_config_learning_rate_scaling(self, training_config_adapter, base_config, mock_hardware_specs):
        """Test learning rate scaling for smaller batch sizes."""
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=mock_hardware_specs), \
             patch.object(training_config_adapter.memory_manager, 'calculate_optimal_batch_size') as mock_calc:
            
            mock_calc.return_value = BatchSizeRecommendation(
                recommended_batch_size=16,
                max_safe_batch_size=24,
                memory_utilization_percent=70.0,
                reasoning="Memory constrained",
                warnings=[]
            )
            
            adapted_config = training_config_adapter.adapt_model_config(base_config, mock_hardware_specs)
            
            # Check learning rate scaling
            original_lr = base_config['lr']
            effective_batch_size = 16 * 4  # batch_size * gradient_accumulation_steps
            expected_lr_scale = math.sqrt(effective_batch_size / 64)
            expected_lr = original_lr * expected_lr_scale
            
            assert abs(adapted_config['lr'] - expected_lr) < 1e-6
    
    def test_adapt_model_config_memory_constraints(self, training_config_adapter, base_config):
        """Test model adaptation for severe memory constraints."""
        # Create low-memory hardware specs
        low_memory_specs = HardwareSpecs(
            cpu=CPUSpecs(cores=2, threads=4, architecture="x86_64", features=[], 
                        base_frequency=2.0, max_frequency=3.0, brand="Intel", model="Intel"),
            memory=MemorySpecs(total_memory=4*1024**3, available_memory=3*1024**3, 
                              memory_type="LPDDR3", memory_speed=1600),
            platform=PlatformCapabilities(has_mkl=False, has_accelerate=False, torch_version="2.0.0",
                                        python_version="3.9.0", macos_version="12.0", optimal_dtype="float32",
                                        supports_avx=False, supports_avx2=False),
            optimal_workers=1,
            hardware_summary={"memory": {"available_gb": 3.0}}
        )
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=low_memory_specs), \
             patch.object(training_config_adapter.memory_manager, 'calculate_optimal_batch_size') as mock_calc:
            
            mock_calc.return_value = BatchSizeRecommendation(
                recommended_batch_size=4,
                max_safe_batch_size=8,
                memory_utilization_percent=80.0,
                reasoning="Severe memory constraints",
                warnings=["Very limited memory"]
            )
            
            adapted_config = training_config_adapter.adapt_model_config(base_config, low_memory_specs)
            
            # Check model complexity reduction
            assert adapted_config['model_dim'] < base_config['model_dim']
            assert adapted_config['num_layers'] < base_config['num_layers']
    
    def test_adapt_model_config_cpu_settings(self, training_config_adapter, base_config, mock_hardware_specs):
        """Test CPU-specific settings adaptation."""
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=mock_hardware_specs), \
             patch.object(training_config_adapter.memory_manager, 'calculate_optimal_batch_size') as mock_calc:
            
            mock_calc.return_value = BatchSizeRecommendation(
                recommended_batch_size=16,
                max_safe_batch_size=24,
                memory_utilization_percent=70.0,
                reasoning="Standard adaptation",
                warnings=[]
            )
            
            adapted_config = training_config_adapter.adapt_model_config(base_config, mock_hardware_specs)
            
            # Check CPU-specific settings
            assert adapted_config['device'] == 'cpu'
            assert adapted_config['num_workers'] == mock_hardware_specs.optimal_workers
            assert adapted_config['pin_memory'] == False  # Disabled for memory constraints
            
            # Check MacBook optimization settings
            assert 'macbook_optimization' in adapted_config
            macbook_opts = adapted_config['macbook_optimization']
            assert macbook_opts['enable_memory_monitoring'] == True
            assert macbook_opts['enable_cpu_optimization'] == True
            assert macbook_opts['use_mkl'] == mock_hardware_specs.platform.has_mkl
    
    def test_calculate_training_parameters(self, training_config_adapter, mock_hardware_specs):
        """Test training parameter calculation."""
        dataset_size = 10000
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=mock_hardware_specs), \
             patch.object(training_config_adapter.memory_manager, 'calculate_optimal_batch_size') as mock_calc, \
             patch.object(training_config_adapter.cpu_optimizer, 'create_optimization_config') as mock_cpu_config:
            
            mock_calc.return_value = BatchSizeRecommendation(
                recommended_batch_size=8,
                max_safe_batch_size=16,
                memory_utilization_percent=60.0,
                reasoning="Conservative sizing",
                warnings=[]
            )
            
            mock_cpu_config.return_value = Mock(torch_threads=4)
            
            params = training_config_adapter.calculate_training_parameters(dataset_size, mock_hardware_specs)
            
            assert isinstance(params, TrainingParams)
            assert params.batch_size == 8
            assert params.gradient_accumulation_steps == 8  # 64 / 8
            assert params.effective_batch_size == 64
            assert params.num_workers == mock_hardware_specs.optimal_workers
            assert params.torch_threads == 4
            assert params.use_mkl == mock_hardware_specs.platform.has_mkl
    
    def test_create_hardware_appropriate_config(self, training_config_adapter, base_config, mock_hardware_specs):
        """Test complete hardware-appropriate configuration creation."""
        dataset_size = 10000
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=mock_hardware_specs), \
             patch.object(training_config_adapter, 'adapt_model_config') as mock_adapt, \
             patch.object(training_config_adapter, 'calculate_training_parameters') as mock_calc_params:
            
            mock_training_params = TrainingParams(
                batch_size=8, gradient_accumulation_steps=8, effective_batch_size=64,
                learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
                num_workers=2, pin_memory=False, torch_threads=4,
                memory_limit_mb=4000, enable_memory_monitoring=True, dynamic_batch_sizing=True,
                max_sequence_length=512, model_complexity_factor=1.0,
                use_mkl=True, enable_cpu_optimization=True, enable_mixed_precision=False,
                checkpoint_interval=500, max_checkpoints_to_keep=3
            )
            
            mock_adapt.return_value = base_config.copy()
            mock_calc_params.return_value = mock_training_params
            
            result = training_config_adapter.create_hardware_appropriate_config(base_config, dataset_size)
            
            assert isinstance(result, ConfigurationResult)
            assert result.adapted_config is not None
            assert result.training_params == mock_training_params
            assert result.hardware_specs == mock_hardware_specs
            assert isinstance(result.validation_warnings, list)
            assert isinstance(result.performance_estimates, dict)
            assert isinstance(result.reasoning, str)
    
    def test_estimate_training_speed(self, training_config_adapter, mock_hardware_specs):
        """Test training speed estimation."""
        training_params = TrainingParams(
            batch_size=8, gradient_accumulation_steps=8, effective_batch_size=64,
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            num_workers=2, pin_memory=False, torch_threads=4,
            memory_limit_mb=4000, enable_memory_monitoring=True, dynamic_batch_sizing=True,
            max_sequence_length=512, model_complexity_factor=1.0,
            use_mkl=True, enable_cpu_optimization=True, enable_mixed_precision=False,
            checkpoint_interval=500, max_checkpoints_to_keep=3
        )
        
        speed = training_config_adapter._estimate_training_speed(training_params, mock_hardware_specs)
        
        assert speed > 0
        assert isinstance(speed, float)
    
    def test_generate_configuration_reasoning(self, training_config_adapter, mock_hardware_specs):
        """Test configuration reasoning generation."""
        training_params = TrainingParams(
            batch_size=8, gradient_accumulation_steps=8, effective_batch_size=64,
            learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
            num_workers=2, pin_memory=False, torch_threads=4,
            memory_limit_mb=4000, enable_memory_monitoring=True, dynamic_batch_sizing=True,
            max_sequence_length=512, model_complexity_factor=0.8,  # Reduced complexity
            use_mkl=True, enable_cpu_optimization=True, enable_mixed_precision=False,
            checkpoint_interval=500, max_checkpoints_to_keep=3
        )
        
        warnings = ["Test warning"]
        
        reasoning = training_config_adapter._generate_configuration_reasoning(
            training_params, mock_hardware_specs, warnings
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "Intel Core i5" in reasoning
        assert "batch size" in reasoning.lower()
        assert "learning rate" in reasoning.lower()
    
    def test_get_configuration_templates(self, training_config_adapter):
        """Test configuration template retrieval."""
        with patch.object(training_config_adapter.config_manager, 'create_config_template') as mock_template:
            mock_template.return_value = Mock()
            
            templates = training_config_adapter.get_configuration_templates()
            
            assert isinstance(templates, dict)
            assert 'macbook_8gb' in templates
            assert 'macbook_16gb' in templates
            assert 'macbook_32gb' in templates
            assert 'macbook_conservative' in templates
            
            for template_name, template_data in templates.items():
                assert 'description' in template_data
                assert 'config' in template_data
                assert 'recommended_for' in template_data
    
    def test_recommend_configuration_template(self, training_config_adapter):
        """Test configuration template recommendation."""
        # Test 8GB system
        hardware_8gb = HardwareSpecs(
            cpu=Mock(cores=4), memory=Mock(total_memory=8*1024**3),
            platform=Mock(), optimal_workers=2, hardware_summary={}
        )
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=hardware_8gb):
            recommendation = training_config_adapter.recommend_configuration_template()
            assert recommendation == 'macbook_8gb'
        
        # Test 16GB system
        hardware_16gb = HardwareSpecs(
            cpu=Mock(cores=4), memory=Mock(total_memory=16*1024**3),
            platform=Mock(), optimal_workers=2, hardware_summary={}
        )
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=hardware_16gb):
            recommendation = training_config_adapter.recommend_configuration_template()
            assert recommendation == 'macbook_16gb'
        
        # Test high-end system
        hardware_32gb = HardwareSpecs(
            cpu=Mock(cores=8), memory=Mock(total_memory=32*1024**3),
            platform=Mock(), optimal_workers=4, hardware_summary={}
        )
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=hardware_32gb):
            recommendation = training_config_adapter.recommend_configuration_template()
            assert recommendation == 'macbook_32gb'
        
        # Test conservative system
        hardware_low = HardwareSpecs(
            cpu=Mock(cores=2), memory=Mock(total_memory=4*1024**3),
            platform=Mock(), optimal_workers=1, hardware_summary={}
        )
        
        with patch.object(training_config_adapter, 'get_hardware_specs', return_value=hardware_low):
            recommendation = training_config_adapter.recommend_configuration_template()
            assert recommendation == 'macbook_conservative'
    
    def test_export_configuration_yaml(self, training_config_adapter):
        """Test configuration export in YAML format."""
        config_result = ConfigurationResult(
            adapted_config={"batch_size": 8, "lr": 1e-4},
            training_params=Mock(),
            hardware_specs=Mock(),
            validation_warnings=[],
            performance_estimates={},
            reasoning="Test reasoning"
        )
        
        yaml_output = training_config_adapter.export_configuration(config_result, 'yaml')
        
        assert isinstance(yaml_output, str)
        assert 'batch_size: 8' in yaml_output
        assert 'lr: 1.0e-04' in yaml_output or 'lr: 0.0001' in yaml_output
    
    def test_export_configuration_json(self, training_config_adapter):
        """Test configuration export in JSON format."""
        config_result = ConfigurationResult(
            adapted_config={"batch_size": 8, "lr": 1e-4},
            training_params=Mock(),
            hardware_specs=Mock(),
            validation_warnings=[],
            performance_estimates={},
            reasoning="Test reasoning"
        )
        
        json_output = training_config_adapter.export_configuration(config_result, 'json')
        
        assert isinstance(json_output, str)
        assert '"batch_size": 8' in json_output
        assert '"lr": 0.0001' in json_output or '"lr": 1e-4' in json_output
    
    def test_export_configuration_python(self, training_config_adapter):
        """Test configuration export in Python format."""
        config_result = ConfigurationResult(
            adapted_config={"batch_size": 8, "lr": 1e-4},
            training_params=Mock(),
            hardware_specs=Mock(),
            validation_warnings=[],
            performance_estimates={},
            reasoning="Test reasoning"
        )
        
        python_output = training_config_adapter.export_configuration(config_result, 'python')
        
        assert isinstance(python_output, str)
        assert 'config = ' in python_output
        assert "'batch_size': 8" in python_output
        assert "Test reasoning" in python_output
    
    def test_export_configuration_invalid_format(self, training_config_adapter):
        """Test configuration export with invalid format."""
        config_result = ConfigurationResult(
            adapted_config={},
            training_params=Mock(),
            hardware_specs=Mock(),
            validation_warnings=[],
            performance_estimates={},
            reasoning=""
        )
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            training_config_adapter.export_configuration(config_result, 'invalid')


class TestHardwareSpecs:
    """Test cases for HardwareSpecs dataclass."""
    
    def test_hardware_specs_creation(self, mock_hardware_specs):
        """Test HardwareSpecs creation and attributes."""
        assert mock_hardware_specs.cpu.cores == 4
        assert mock_hardware_specs.memory.available_memory == 6 * 1024**3
        assert mock_hardware_specs.platform.has_mkl == True
        assert mock_hardware_specs.optimal_workers == 2
        assert isinstance(mock_hardware_specs.hardware_summary, dict)


class TestTrainingParams:
    """Test cases for TrainingParams dataclass."""
    
    def test_training_params_creation(self):
        """Test TrainingParams creation and validation."""
        params = TrainingParams(
            batch_size=8,
            gradient_accumulation_steps=8,
            effective_batch_size=64,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            num_workers=2,
            pin_memory=False,
            torch_threads=4,
            memory_limit_mb=4000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            max_sequence_length=512,
            model_complexity_factor=1.0,
            use_mkl=True,
            enable_cpu_optimization=True,
            enable_mixed_precision=False,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3
        )
        
        assert params.batch_size == 8
        assert params.effective_batch_size == 64
        assert params.learning_rate == 1e-4
        assert params.enable_memory_monitoring == True
        assert params.use_mkl == True


class TestConfigurationResult:
    """Test cases for ConfigurationResult dataclass."""
    
    def test_configuration_result_creation(self):
        """Test ConfigurationResult creation and attributes."""
        result = ConfigurationResult(
            adapted_config={"batch_size": 8},
            training_params=Mock(),
            hardware_specs=Mock(),
            validation_warnings=["Warning 1"],
            performance_estimates={"speed": 10.0},
            reasoning="Test reasoning"
        )
        
        assert result.adapted_config["batch_size"] == 8
        assert len(result.validation_warnings) == 1
        assert result.performance_estimates["speed"] == 10.0
        assert result.reasoning == "Test reasoning"


if __name__ == "__main__":
    pytest.main([__file__])