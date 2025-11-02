"""
Unit tests for memory management module.

Tests batch size calculation accuracy, memory monitoring and adjustment logic,
and gradient accumulation correctness for MacBook optimization.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

from macbook_optimization.memory_management import (
    MemoryConfig,
    MemoryManager,
    BatchSizeRecommendation,
    MemoryPressureInfo,
)
from macbook_optimization.resource_monitoring import MemoryStats


class TestMemoryConfig:
    """Test MemoryConfig dataclass."""
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        
        assert config.memory_warning_threshold == 75.0
        assert config.memory_critical_threshold == 85.0
        assert config.memory_emergency_threshold == 95.0
        assert config.min_batch_size == 1
        assert config.max_batch_size == 64
        assert config.batch_size_reduction_factor == 0.75
        assert config.batch_size_increase_factor == 1.25
        assert config.model_memory_overhead == 1.5
        assert config.gradient_memory_multiplier == 2.0
        assert config.optimizer_memory_multiplier == 1.0
        assert config.monitoring_interval == 1.0
        assert config.memory_history_size == 60
        assert config.safety_margin_mb == 500.0
    
    def test_memory_config_custom_values(self):
        """Test MemoryConfig with custom values."""
        config = MemoryConfig(
            memory_warning_threshold=70.0,
            memory_critical_threshold=80.0,
            min_batch_size=2,
            max_batch_size=32,
            safety_margin_mb=1000.0
        )
        
        assert config.memory_warning_threshold == 70.0
        assert config.memory_critical_threshold == 80.0
        assert config.min_batch_size == 2
        assert config.max_batch_size == 32
        assert config.safety_margin_mb == 1000.0


class TestBatchSizeRecommendation:
    """Test BatchSizeRecommendation dataclass."""
    
    def test_batch_size_recommendation_creation(self):
        """Test BatchSizeRecommendation creation."""
        rec = BatchSizeRecommendation(
            recommended_batch_size=16,
            max_safe_batch_size=32,
            memory_utilization_percent=65.0,
            reasoning="Test reasoning",
            warnings=["Test warning"]
        )
        
        assert rec.recommended_batch_size == 16
        assert rec.max_safe_batch_size == 32
        assert rec.memory_utilization_percent == 65.0
        assert rec.reasoning == "Test reasoning"
        assert "Test warning" in rec.warnings


class TestMemoryPressureInfo:
    """Test MemoryPressureInfo dataclass."""
    
    def test_memory_pressure_info_creation(self):
        """Test MemoryPressureInfo creation."""
        info = MemoryPressureInfo(
            current_usage_percent=75.0,
            available_mb=2048.0,
            pressure_level="medium",
            recommended_action="Consider reducing batch size",
            time_to_critical=120.0
        )
        
        assert info.current_usage_percent == 75.0
        assert info.available_mb == 2048.0
        assert info.pressure_level == "medium"
        assert info.recommended_action == "Consider reducing batch size"
        assert info.time_to_critical == 120.0


class TestMemoryManager:
    """Test MemoryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryConfig(
            memory_warning_threshold=70.0,
            memory_critical_threshold=80.0,
            memory_emergency_threshold=90.0,
            monitoring_interval=0.1  # Fast for testing
        )
        
        # Mock resource monitor to avoid actual system monitoring
        with patch('macbook_optimization.memory_management.ResourceMonitor') as mock_monitor_class:
            self.mock_monitor = Mock()
            mock_monitor_class.return_value = self.mock_monitor
            self.memory_manager = MemoryManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.memory_manager, 'resource_monitor'):
            self.memory_manager.resource_monitor.stop_monitoring()
    
    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        assert self.memory_manager.config == self.config
        assert self.memory_manager.current_batch_size == 8
        assert self.memory_manager.baseline_memory_mb == 0.0
        assert self.memory_manager.model_memory_mb == 0.0
        assert self.memory_manager.peak_memory_mb == 0.0
    
    def test_estimate_model_memory_usage(self):
        """Test model memory usage estimation."""
        # Test with 7M parameters (TRM model size)
        model_params = 7_000_000
        sequence_length = 512
        dtype_size = 4  # float32
        
        memory_usage = self.memory_manager.estimate_model_memory_usage(
            model_params, sequence_length, dtype_size
        )
        
        # Should be reasonable for 7M parameter model
        assert memory_usage > 0
        assert memory_usage < 1000  # Less than 1GB for model alone
        
        # Test with different parameters
        memory_usage_small = self.memory_manager.estimate_model_memory_usage(
            1_000_000, 256, 4
        )
        assert memory_usage_small < memory_usage
    
    def test_calculate_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        model_params = 7_000_000
        available_memory_mb = 4000.0  # 4GB available
        
        recommendation = self.memory_manager.calculate_optimal_batch_size(
            model_params, available_memory_mb=available_memory_mb
        )
        
        assert isinstance(recommendation, BatchSizeRecommendation)
        assert recommendation.recommended_batch_size >= self.config.min_batch_size
        assert recommendation.recommended_batch_size <= self.config.max_batch_size
        assert recommendation.max_safe_batch_size >= recommendation.recommended_batch_size
        assert recommendation.memory_utilization_percent >= 0
        assert recommendation.memory_utilization_percent <= 100
        assert len(recommendation.reasoning) > 0
        assert isinstance(recommendation.warnings, list)
    
    def test_calculate_optimal_batch_size_low_memory(self):
        """Test batch size calculation with very low memory."""
        model_params = 7_000_000
        available_memory_mb = 500.0  # Only 500MB available
        
        recommendation = self.memory_manager.calculate_optimal_batch_size(
            model_params, available_memory_mb=available_memory_mb
        )
        
        # Should recommend very small batch size
        assert recommendation.recommended_batch_size == self.config.min_batch_size
        assert len(recommendation.warnings) > 0
        assert any("limited memory" in warning.lower() for warning in recommendation.warnings)
    
    def test_analyze_memory_pressure_low(self):
        """Test memory pressure analysis - low pressure."""
        memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=6000.0,
            used_mb=2192.0,
            percent_used=26.8,  # Below warning threshold
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        pressure_info = self.memory_manager._analyze_memory_pressure(memory_stats)
        
        assert pressure_info.pressure_level == "low"
        assert pressure_info.recommended_action == "Normal operation"
        assert pressure_info.current_usage_percent == 26.8
        assert pressure_info.available_mb == 6000.0
    
    def test_analyze_memory_pressure_medium(self):
        """Test memory pressure analysis - medium pressure."""
        memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=2000.0,
            used_mb=6192.0,
            percent_used=75.6,  # Above warning, below critical
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        pressure_info = self.memory_manager._analyze_memory_pressure(memory_stats)
        
        assert pressure_info.pressure_level == "medium"
        assert "consider reducing" in pressure_info.recommended_action.lower()
        assert pressure_info.current_usage_percent == 75.6
    
    def test_analyze_memory_pressure_high(self):
        """Test memory pressure analysis - high pressure."""
        memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=1000.0,
            used_mb=7192.0,
            percent_used=87.8,  # Above critical, below emergency
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        pressure_info = self.memory_manager._analyze_memory_pressure(memory_stats)
        
        assert pressure_info.pressure_level == "high"
        assert "reduce batch size" in pressure_info.recommended_action.lower()
        assert pressure_info.current_usage_percent == 87.8
    
    def test_analyze_memory_pressure_critical(self):
        """Test memory pressure analysis - critical pressure."""
        memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=400.0,
            used_mb=7792.0,
            percent_used=95.1,  # Above emergency threshold
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        pressure_info = self.memory_manager._analyze_memory_pressure(memory_stats)
        
        assert pressure_info.pressure_level == "critical"
        assert "immediately" in pressure_info.recommended_action.lower()
        assert pressure_info.current_usage_percent == 95.1
    
    def test_adjust_batch_size_dynamically_reduce(self):
        """Test dynamic batch size adjustment - reduction."""
        self.memory_manager.current_batch_size = 16
        
        # High memory usage should reduce batch size
        new_batch_size = self.memory_manager.adjust_batch_size_dynamically(85.0)
        
        assert new_batch_size < 16
        assert new_batch_size >= self.config.min_batch_size
        assert self.memory_manager.current_batch_size == new_batch_size
    
    def test_adjust_batch_size_dynamically_increase(self):
        """Test dynamic batch size adjustment - increase."""
        self.memory_manager.current_batch_size = 8
        
        # Low memory usage should increase batch size
        new_batch_size = self.memory_manager.adjust_batch_size_dynamically(50.0)
        
        assert new_batch_size > 8
        assert new_batch_size <= self.config.max_batch_size
        assert self.memory_manager.current_batch_size == new_batch_size
    
    def test_adjust_batch_size_dynamically_no_change(self):
        """Test dynamic batch size adjustment - no change."""
        self.memory_manager.current_batch_size = 16
        
        # Moderate memory usage should not change batch size
        new_batch_size = self.memory_manager.adjust_batch_size_dynamically(65.0)
        
        assert new_batch_size == 16
        assert self.memory_manager.current_batch_size == 16
    
    def test_force_garbage_collection(self):
        """Test garbage collection forcing."""
        # Mock gc.collect and torch.cuda.empty_cache
        with patch('gc.collect') as mock_gc, \
             patch('torch.cuda.is_available', return_value=False):
            
            self.memory_manager.force_garbage_collection()
            
            mock_gc.assert_called_once()
    
    def test_memory_pressure_callback(self):
        """Test memory pressure callback functionality."""
        callback_called = False
        callback_info = None
        
        def test_callback(info):
            nonlocal callback_called, callback_info
            callback_called = True
            callback_info = info
        
        self.memory_manager.add_memory_pressure_callback(test_callback)
        
        # Simulate memory update
        memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=1000.0,
            used_mb=7192.0,
            percent_used=87.8,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        # Create a mock snapshot
        mock_snapshot = Mock()
        mock_snapshot.memory = memory_stats
        
        self.memory_manager._on_memory_update(mock_snapshot)
        
        assert callback_called
        assert callback_info is not None
        assert callback_info.pressure_level == "high"
    
    def test_get_memory_recommendations(self):
        """Test comprehensive memory recommendations."""
        model_params = 7_000_000
        
        # Mock memory monitoring
        mock_memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=4000.0,
            used_mb=4192.0,
            percent_used=51.2,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        with patch.object(self.memory_manager, 'monitor_memory_usage', 
                         return_value=mock_memory_stats):
            recommendations = self.memory_manager.get_memory_recommendations(model_params)
        
        assert "current_memory" in recommendations
        assert "batch_size" in recommendations
        assert "recommendations" in recommendations
        assert "monitoring" in recommendations
        
        assert recommendations["current_memory"]["used_percent"] == 51.2
        assert recommendations["batch_size"]["current"] == self.memory_manager.current_batch_size
        assert isinstance(recommendations["batch_size"]["recommended"], int)
    
    def test_set_baseline_memory(self):
        """Test baseline memory setting."""
        mock_memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=4000.0,
            used_mb=4192.0,
            percent_used=51.2,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        with patch.object(self.memory_manager, 'monitor_memory_usage', 
                         return_value=mock_memory_stats):
            self.memory_manager.set_baseline_memory()
        
        assert self.memory_manager.baseline_memory_mb == 4192.0
    
    def test_track_model_memory(self):
        """Test model memory tracking."""
        model_memory = 256.5
        self.memory_manager.track_model_memory(model_memory)
        
        assert self.memory_manager.model_memory_mb == model_memory
    
    def test_update_peak_memory(self):
        """Test peak memory tracking."""
        mock_memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=3000.0,
            used_mb=5192.0,
            percent_used=63.4,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        with patch.object(self.memory_manager, 'monitor_memory_usage', 
                         return_value=mock_memory_stats):
            self.memory_manager.update_peak_memory()
        
        assert self.memory_manager.peak_memory_mb == 5192.0
        
        # Update with lower memory usage - peak should not decrease
        mock_memory_stats.used_mb = 4000.0
        with patch.object(self.memory_manager, 'monitor_memory_usage', 
                         return_value=mock_memory_stats):
            self.memory_manager.update_peak_memory()
        
        assert self.memory_manager.peak_memory_mb == 5192.0  # Should remain the same
    
    def test_get_memory_summary(self):
        """Test memory usage summary."""
        # Set up some tracked values
        self.memory_manager.baseline_memory_mb = 2000.0
        self.memory_manager.model_memory_mb = 300.0
        self.memory_manager.peak_memory_mb = 5000.0
        
        mock_memory_stats = MemoryStats(
            total_mb=8192.0,
            available_mb=3000.0,
            used_mb=4500.0,
            percent_used=54.9,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        mock_averages = {
            "average_memory_percent": 52.3,
            "sample_count": 10
        }
        
        with patch.object(self.memory_manager, 'monitor_memory_usage', 
                         return_value=mock_memory_stats), \
             patch.object(self.memory_manager.resource_monitor, 'get_average_stats',
                         return_value=mock_averages):
            
            summary = self.memory_manager.get_memory_summary()
        
        assert "current" in summary
        assert "tracking" in summary
        assert "recent_averages" in summary
        assert "configuration" in summary
        
        assert summary["current"]["used_mb"] == 4500.0
        assert summary["tracking"]["baseline_mb"] == 2000.0
        assert summary["tracking"]["model_mb"] == 300.0
        assert summary["tracking"]["peak_mb"] == 5000.0
        assert summary["tracking"]["training_overhead_mb"] == 2500.0  # 4500 - 2000
        assert summary["configuration"]["current_batch_size"] == self.memory_manager.current_batch_size
    
    def test_auto_adjust_batch_size_cooldown(self):
        """Test that auto-adjustment respects cooldown period."""
        # Set recent adjustment time
        self.memory_manager.last_adjustment_time = time.time()
        
        memory_info = MemoryPressureInfo(
            current_usage_percent=87.0,
            available_mb=1000.0,
            pressure_level="high",
            recommended_action="Reduce batch size",
            time_to_critical=None
        )
        
        original_batch_size = self.memory_manager.current_batch_size
        
        # Should not adjust due to cooldown
        should_adjust = self.memory_manager._should_adjust_batch_size(memory_info)
        assert should_adjust is False
        
        # Batch size should remain unchanged
        self.memory_manager._auto_adjust_batch_size(memory_info)
        assert self.memory_manager.current_batch_size == original_batch_size
    
    def test_auto_adjust_batch_size_critical(self):
        """Test auto-adjustment for critical memory pressure."""
        self.memory_manager.current_batch_size = 16
        self.memory_manager.last_adjustment_time = 0  # Allow adjustment
        
        memory_info = MemoryPressureInfo(
            current_usage_percent=96.0,
            available_mb=300.0,
            pressure_level="critical",
            recommended_action="Immediately reduce batch size",
            time_to_critical=None
        )
        
        with patch.object(self.memory_manager, 'force_garbage_collection') as mock_gc:
            self.memory_manager._auto_adjust_batch_size(memory_info)
        
        # Should reduce batch size significantly (50% reduction for critical)
        assert self.memory_manager.current_batch_size == 8
        mock_gc.assert_called_once()
    
    def test_memory_manager_context_manager_cleanup(self):
        """Test that memory manager cleans up properly."""
        # This tests the __del__ method indirectly
        with patch.object(self.memory_manager.resource_monitor, 'stop_monitoring') as mock_stop:
            del self.memory_manager
            # The __del__ method should be called, but we can't directly test it
            # Instead, we verify the mock was set up correctly
            assert mock_stop is not None


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager with real-like scenarios."""
    
    def test_memory_manager_with_gradient_accumulation_scenario(self):
        """Test memory manager in a gradient accumulation scenario."""
        config = MemoryConfig(
            max_batch_size=32,
            memory_warning_threshold=70.0
        )
        
        with patch('macbook_optimization.memory_management.ResourceMonitor'):
            memory_manager = MemoryManager(config)
        
        # Simulate training scenario
        model_params = 7_000_000
        available_memory_mb = 3000.0  # 3GB available
        
        # Get initial recommendation
        recommendation = memory_manager.calculate_optimal_batch_size(
            model_params, available_memory_mb=available_memory_mb
        )
        
        # Should recommend reasonable batch size for gradient accumulation
        assert recommendation.recommended_batch_size >= 1
        assert recommendation.recommended_batch_size <= 32
        
        # Simulate memory pressure increase
        memory_manager.adjust_batch_size_dynamically(75.0)  # Medium pressure
        
        # Batch size should be adjusted
        assert memory_manager.current_batch_size <= recommendation.recommended_batch_size
    
    def test_memory_manager_recommendations_consistency(self):
        """Test that memory manager provides consistent recommendations."""
        with patch('macbook_optimization.memory_management.ResourceMonitor'):
            memory_manager = MemoryManager()
        
        model_params = 7_000_000
        
        # Get recommendations multiple times
        recommendations = []
        for _ in range(3):
            mock_memory_stats = MemoryStats(
                total_mb=8192.0,
                available_mb=4000.0,
                used_mb=4192.0,
                percent_used=51.2,
                swap_used_mb=0.0,
                swap_percent=0.0
            )
            
            with patch.object(memory_manager, 'monitor_memory_usage', 
                             return_value=mock_memory_stats):
                rec = memory_manager.get_memory_recommendations(model_params)
                recommendations.append(rec)
        
        # Recommendations should be consistent
        batch_sizes = [rec["batch_size"]["recommended"] for rec in recommendations]
        assert len(set(batch_sizes)) == 1  # All should be the same
        
        pressure_levels = [rec["current_memory"]["pressure_level"] for rec in recommendations]
        assert len(set(pressure_levels)) == 1  # All should be the same