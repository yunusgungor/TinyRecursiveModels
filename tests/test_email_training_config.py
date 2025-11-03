"""
Tests for email training configuration system.

This module tests the EmailTrainingConfig and EmailTrainingConfigAdapter
classes for email classification training on MacBook hardware.
"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

from macbook_optimization.email_training_config import (
    EmailTrainingConfig, EmailTrainingParams, EmailTrainingConfigAdapter
)
from macbook_optimization.hardware_detection import CPUSpecs, MemorySpecs, PlatformCapabilities
from macbook_optimization.training_config_adapter import HardwareSpecs


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
        hardware_summary={"total_memory_gb": 8, "available_memory_gb": 6}
    )


@pytest.fixture
def low_memory_hardware_specs():
    """Create low memory hardware specs for testing constraints."""
    cpu_specs = CPUSpecs(
        cores=2,
        threads=4,
        architecture="x86_64",
        features=["sse4_2"],
        base_frequency=1.8,
        max_frequency=2.8,
        brand="Intel Core i3",
        model="Intel Core i3-7100U"
    )
    
    memory_specs = MemorySpecs(
        total_memory=4 * 1024**3,  # 4GB
        available_memory=2.5 * 1024**3,  # 2.5GB available
        memory_type="DDR3",
        memory_speed=1600
    )
    
    platform_caps = PlatformCapabilities(
        has_mkl=False,
        has_accelerate=False,
        torch_version="2.0.0",
        python_version="3.9.0",
        macos_version="11.0",
        optimal_dtype="float32",
        supports_avx=False,
        supports_avx2=False
    )
    
    return HardwareSpecs(
        cpu=cpu_specs,
        memory=memory_specs,
        platform=platform_caps,
        optimal_workers=1,
        hardware_summary={"total_memory_gb": 4, "available_memory_gb": 2.5}
    )


class TestEmailTrainingConfig:
    """Test EmailTrainingConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creating EmailTrainingConfig with default values."""
        config = EmailTrainingConfig()
        
        assert config.model_name == "EmailTRM"
        assert config.vocab_size == 5000
        assert config.num_email_categories == 10
        assert config.batch_size == 8
        assert config.max_sequence_length == 512
        assert config.target_accuracy == 0.95
        assert config.use_email_structure is True
        assert config.enable_subject_prioritization is True
    
    def test_custom_config_creation(self):
        """Test creating EmailTrainingConfig with custom values."""
        config = EmailTrainingConfig(
            vocab_size=8000,
            batch_size=4,
            learning_rate=2e-4,
            max_sequence_length=256,
            subject_attention_weight=3.0
        )
        
        assert config.vocab_size == 8000
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.max_sequence_length == 256
        assert config.subject_attention_weight == 3.0
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        config = EmailTrainingConfig(
            num_email_categories=10,
            target_accuracy=0.95,
            min_category_accuracy=0.90
        )
        
        assert config.num_email_categories == 10
        assert config.target_accuracy >= config.min_category_accuracy


class TestEmailTrainingConfigAdapter:
    """Test EmailTrainingConfigAdapter class."""
    
    def test_adapter_initialization(self):
        """Test EmailTrainingConfigAdapter initialization."""
        adapter = EmailTrainingConfigAdapter()
        
        assert adapter.email_model_params == 7_000_000
        assert adapter.email_categories == 10
        assert adapter.default_email_vocab_size == 5000
        assert adapter.hardware_detector is not None
        assert adapter.memory_manager is not None
    
    def test_adapt_email_config_normal_memory(self, mock_hardware_specs):
        """Test email config adaptation with normal memory."""
        adapter = EmailTrainingConfigAdapter()
        base_config = EmailTrainingConfig()
        dataset_size = 10000
        
        adapted_config = adapter.adapt_email_config(
            base_config, dataset_size, mock_hardware_specs
        )
        
        # Check basic adaptations
        assert 'batch_size' in adapted_config
        assert 'gradient_accumulation_steps' in adapted_config
        assert 'learning_rate' in adapted_config
        assert 'vocab_size' in adapted_config
        assert 'max_sequence_length' in adapted_config
        
        # Check email-specific features are enabled
        assert adapted_config['enable_subject_prioritization'] is True
        assert adapted_config['use_hierarchical_attention'] is True
        assert adapted_config['enable_content_features'] is True
        
        # Check memory limits
        assert adapted_config['memory_limit_mb'] > 0
        assert adapted_config['num_workers'] <= 4
    
    def test_adapt_email_config_low_memory(self, low_memory_hardware_specs):
        """Test email config adaptation with low memory constraints."""
        adapter = EmailTrainingConfigAdapter()
        base_config = EmailTrainingConfig()
        dataset_size = 5000
        
        adapted_config = adapter.adapt_email_config(
            base_config, dataset_size, low_memory_hardware_specs
        )
        
        # Check memory-constrained adaptations
        assert adapted_config['batch_size'] <= base_config.batch_size
        assert adapted_config['max_sequence_length'] <= base_config.max_sequence_length
        assert adapted_config['vocab_size'] <= base_config.vocab_size
        
        # Check features are disabled due to memory constraints
        assert adapted_config['enable_subject_prioritization'] is False  # < 4GB memory
        assert adapted_config['use_hierarchical_attention'] is False  # < 6GB memory
        assert adapted_config['enable_content_features'] is False  # < 8GB memory
    
    def test_calculate_email_memory_requirements(self, mock_hardware_specs):
        """Test email memory requirements calculation."""
        adapter = EmailTrainingConfigAdapter()
        config = EmailTrainingConfig()
        dataset_size = 10000
        
        memory_req = adapter._calculate_email_memory_requirements(
            config, dataset_size, mock_hardware_specs
        )
        
        # Check all memory components are calculated
        assert 'model_memory_mb' in memory_req
        assert 'vocab_memory_mb' in memory_req
        assert 'batch_memory_mb' in memory_req
        assert 'email_features_mb' in memory_req
        assert 'gradient_memory_mb' in memory_req
        assert 'optimizer_memory_mb' in memory_req
        assert 'total_memory_mb' in memory_req
        
        # Check memory values are reasonable
        assert memory_req['model_memory_mb'] > 0
        assert memory_req['total_memory_mb'] > memory_req['model_memory_mb']
        
        # Check email features add overhead
        if config.enable_subject_prioritization or config.use_hierarchical_attention:
            assert memory_req['email_features_mb'] > 0
    
    def test_calculate_email_batch_size(self, mock_hardware_specs):
        """Test email batch size calculation."""
        adapter = EmailTrainingConfigAdapter()
        config = EmailTrainingConfig()
        
        memory_req = adapter._calculate_email_memory_requirements(
            config, 10000, mock_hardware_specs
        )
        
        batch_size = adapter._calculate_email_batch_size(
            config, memory_req, mock_hardware_specs
        )
        
        # Check batch size is reasonable
        assert batch_size >= 1
        assert batch_size <= 16  # Max for email training
        assert isinstance(batch_size, int)
    
    def test_calculate_email_training_parameters(self, mock_hardware_specs):
        """Test email training parameters calculation."""
        adapter = EmailTrainingConfigAdapter()
        config = EmailTrainingConfig()
        dataset_size = 10000
        
        params = adapter.calculate_email_training_parameters(
            config, dataset_size, mock_hardware_specs
        )
        
        # Check EmailTrainingParams fields
        assert isinstance(params, EmailTrainingParams)
        assert params.batch_size >= 1
        assert params.gradient_accumulation_steps >= 1
        assert params.learning_rate > 0
        assert params.email_vocab_size == config.vocab_size
        assert params.num_email_categories == config.num_email_categories
        
        # Check email-specific parameters
        assert params.subject_attention_weight == config.subject_attention_weight
        assert isinstance(params.email_augmentation_enabled, bool)
        assert isinstance(params.use_hierarchical_attention, bool)
        assert params.email_streaming_threshold_mb > 0
    
    def test_create_email_hardware_config(self, mock_hardware_specs):
        """Test complete email hardware configuration creation."""
        adapter = EmailTrainingConfigAdapter()
        base_config = EmailTrainingConfig()
        dataset_size = 10000
        
        with patch.object(adapter, 'get_hardware_specs', return_value=mock_hardware_specs):
            result = adapter.create_email_hardware_config(base_config, dataset_size)
        
        # Check ConfigurationResult structure
        assert hasattr(result, 'adapted_config')
        assert hasattr(result, 'training_params')
        assert hasattr(result, 'hardware_specs')
        assert hasattr(result, 'validation_warnings')
        assert hasattr(result, 'performance_estimates')
        assert hasattr(result, 'reasoning')
        
        # Check performance estimates
        assert 'estimated_samples_per_second' in result.performance_estimates
        assert 'estimated_epoch_time_minutes' in result.performance_estimates
        assert 'estimated_accuracy_target_epochs' in result.performance_estimates
        
        # Check reasoning is generated
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
    
    def test_estimate_email_training_speed(self, mock_hardware_specs):
        """Test email training speed estimation."""
        adapter = EmailTrainingConfigAdapter()
        params = EmailTrainingParams(
            batch_size=4,
            gradient_accumulation_steps=8,
            effective_batch_size=32,
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
            max_checkpoints_to_keep=3,
            email_vocab_size=5000,
            num_email_categories=10,
            subject_attention_weight=2.0,
            email_augmentation_enabled=True,
            category_balancing_enabled=True,
            use_hierarchical_attention=True,
            enable_subject_prioritization=True,
            email_streaming_threshold_mb=200.0,
            email_cache_threshold_mb=100.0
        )
        
        speed = adapter._estimate_email_training_speed(params, mock_hardware_specs)
        
        # Check speed is reasonable
        assert speed > 0
        assert speed < 100  # Reasonable upper bound for email training
    
    def test_estimate_epochs_to_target(self):
        """Test epochs to target estimation."""
        adapter = EmailTrainingConfigAdapter()
        
        params = EmailTrainingParams(
            batch_size=4,
            gradient_accumulation_steps=8,
            effective_batch_size=32,
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
            max_checkpoints_to_keep=3,
            email_vocab_size=5000,
            num_email_categories=10,
            subject_attention_weight=2.0,
            email_augmentation_enabled=True,
            category_balancing_enabled=True,
            use_hierarchical_attention=True,
            enable_subject_prioritization=True,
            email_streaming_threshold_mb=200.0,
            email_cache_threshold_mb=100.0
        )
        
        # Test different dataset sizes
        epochs_large = adapter._estimate_epochs_to_target(100000, params)
        epochs_medium = adapter._estimate_epochs_to_target(20000, params)
        epochs_small = adapter._estimate_epochs_to_target(5000, params)
        
        # Larger datasets should need fewer epochs
        assert epochs_large <= epochs_medium <= epochs_small
        
        # All should be in reasonable range
        assert 2 <= epochs_large <= 15
        assert 2 <= epochs_medium <= 15
        assert 2 <= epochs_small <= 15
    
    def test_get_email_configuration_templates(self):
        """Test email configuration templates."""
        adapter = EmailTrainingConfigAdapter()
        templates = adapter.get_email_configuration_templates()
        
        # Check all expected templates exist
        expected_templates = ['email_macbook_8gb', 'email_macbook_16gb', 'email_macbook_32gb']
        for template_name in expected_templates:
            assert template_name in templates
            
            template = templates[template_name]
            assert 'description' in template
            assert 'config' in template
            assert 'recommended_for' in template
            
            # Check config is EmailTrainingConfig
            assert isinstance(template['config'], EmailTrainingConfig)
    
    def test_recommend_email_template(self, mock_hardware_specs, low_memory_hardware_specs):
        """Test email template recommendation."""
        adapter = EmailTrainingConfigAdapter()
        
        # Test with normal memory
        recommendation_normal = adapter.recommend_email_template(10000, mock_hardware_specs)
        assert recommendation_normal in ['email_macbook_8gb', 'email_macbook_16gb']
        
        # Test with low memory
        recommendation_low = adapter.recommend_email_template(10000, low_memory_hardware_specs)
        assert recommendation_low == 'email_macbook_8gb'
        
        # Test with large dataset
        recommendation_large = adapter.recommend_email_template(200000, mock_hardware_specs)
        assert recommendation_large in ['email_macbook_8gb', 'email_macbook_16gb']
    
    def test_configuration_reasoning_generation(self, mock_hardware_specs):
        """Test configuration reasoning generation."""
        adapter = EmailTrainingConfigAdapter()
        config = EmailTrainingConfig()
        
        params = adapter.calculate_email_training_parameters(
            config, 10000, mock_hardware_specs
        )
        
        reasoning = adapter._generate_email_configuration_reasoning(
            config, params, mock_hardware_specs, []
        )
        
        # Check reasoning contains key information
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert 'batch size' in reasoning.lower()
        assert 'memory' in reasoning.lower()
        
        # Test with warnings
        warnings = ["Test warning"]
        reasoning_with_warnings = adapter._generate_email_configuration_reasoning(
            config, params, mock_hardware_specs, warnings
        )
        
        assert 'warning' in reasoning_with_warnings.lower()


@pytest.mark.integration
class TestEmailTrainingConfigIntegration:
    """Integration tests for email training configuration."""
    
    def test_end_to_end_config_adaptation(self):
        """Test complete end-to-end configuration adaptation."""
        adapter = EmailTrainingConfigAdapter()
        base_config = EmailTrainingConfig(
            vocab_size=8000,
            batch_size=16,
            learning_rate=1e-4,
            max_sequence_length=512
        )
        dataset_size = 25000
        
        # This should work with real hardware detection
        result = adapter.create_email_hardware_config(base_config, dataset_size)
        
        # Verify the result is complete and valid
        assert result.adapted_config is not None
        assert result.training_params is not None
        assert result.hardware_specs is not None
        assert isinstance(result.validation_warnings, list)
        assert isinstance(result.performance_estimates, dict)
        assert isinstance(result.reasoning, str)
        
        # Check that adaptations are reasonable
        adapted_batch = result.adapted_config['batch_size']
        assert 1 <= adapted_batch <= 16
        
        # Check email-specific parameters are preserved
        assert result.training_params.num_email_categories == 10
        assert result.training_params.email_vocab_size > 0
    
    def test_memory_constraint_handling(self):
        """Test handling of various memory constraint scenarios."""
        adapter = EmailTrainingConfigAdapter()
        base_config = EmailTrainingConfig()
        
        # Test with different dataset sizes
        for dataset_size in [1000, 10000, 100000]:
            result = adapter.create_email_hardware_config(base_config, dataset_size)
            
            # Should always produce valid configuration
            assert result.adapted_config is not None
            assert result.training_params.batch_size >= 1
            assert result.training_params.memory_limit_mb > 0
            
            # Larger datasets might trigger streaming
            if dataset_size > 50000:
                assert result.training_params.email_streaming_threshold_mb > 0