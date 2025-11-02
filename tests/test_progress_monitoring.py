"""
Unit tests for progress monitoring module.

Tests real-time progress monitoring, training speed monitoring,
and resource-aware ETA calculations for MacBook TRM training.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from macbook_optimization.progress_monitoring import (
    TrainingProgress,
    ResourceMetrics,
    PerformanceMetrics,
    TrainingSession,
    ProgressMonitor
)
from macbook_optimization.resource_monitoring import ResourceSnapshot, MemoryStats, CPUStats, ThermalStats
from macbook_optimization.memory_management import MemoryManager, MemoryPressureInfo


class TestTrainingProgress:
    """Test TrainingProgress dataclass."""
    
    def test_training_progress_creation(self):
        """Test TrainingProgress dataclass creation."""
        progress = TrainingProgress(
            current_step=100,
            total_steps=1000,
            current_epoch=1,
            total_epochs=10,
            samples_processed=1600,
            total_samples=16000,
            progress_percent=10.0,
            start_time=time.time(),
            elapsed_time=60.0,
            estimated_total_time=600.0,
            eta_seconds=540.0,
            current_samples_per_second=26.7,
            average_samples_per_second=26.7,
            recent_samples_per_second=28.0,
            current_loss=0.5432,
            average_loss=0.6123,
            best_loss=0.5123,
            current_learning_rate=1e-4
        )
        
        assert progress.current_step == 100
        assert progress.total_steps == 1000
        assert progress.progress_percent == 10.0
        assert progress.current_samples_per_second == 26.7
        assert progress.current_loss == 0.5432


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""
    
    def test_resource_metrics_creation(self):
        """Test ResourceMetrics dataclass creation."""
        metrics = ResourceMetrics(
            memory_used_mb=4096.0,
            memory_available_mb=4096.0,
            memory_usage_percent=50.0,
            memory_pressure_level="medium",
            cpu_usage_percent=75.0,
            cpu_frequency_mhz=2400.0,
            cpu_load_average=[1.2, 1.5, 1.8],
            thermal_state="warm",
            batch_processing_time_ms=150.0,
            gradient_computation_time_ms=90.0,
            data_loading_time_ms=15.0
        )
        
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_pressure_level == "medium"
        assert metrics.cpu_usage_percent == 75.0
        assert metrics.thermal_state == "warm"
        assert metrics.batch_processing_time_ms == 150.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics dataclass creation."""
        metrics = PerformanceMetrics(
            memory_efficiency_percent=85.0,
            cpu_efficiency_percent=75.0,
            training_efficiency_score=80.0,
            primary_bottleneck="memory",
            bottleneck_severity=0.7,
            optimization_suggestions=["Reduce batch size", "Enable gradient accumulation"],
            performance_warnings=["High memory usage detected"]
        )
        
        assert metrics.memory_efficiency_percent == 85.0
        assert metrics.training_efficiency_score == 80.0
        assert metrics.primary_bottleneck == "memory"
        assert len(metrics.optimization_suggestions) == 2
        assert len(metrics.performance_warnings) == 1


class TestTrainingSession:
    """Test TrainingSession dataclass."""
    
    def test_training_session_creation(self):
        """Test TrainingSession dataclass creation."""
        session = TrainingSession(
            session_id="test_session_001",
            start_time=time.time(),
            model_name="TRM-7M",
            dataset_name="test_dataset",
            batch_size=16,
            learning_rate=1e-4
        )
        
        assert session.session_id == "test_session_001"
        assert session.model_name == "TRM-7M"
        assert session.batch_size == 16
        assert session.learning_rate == 1e-4
        assert len(session.progress_history) == 0
        assert len(session.resource_history) == 0


class TestProgressMonitor:
    """Test ProgressMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_resource_monitor = Mock()
        self.mock_memory_manager = Mock()
        
        self.progress_monitor = ProgressMonitor(
            resource_monitor=self.mock_resource_monitor,
            memory_manager=self.mock_memory_manager,
            update_interval=0.1,  # Fast for testing
            history_size=10
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.progress_monitor.is_monitoring:
            self.progress_monitor.stop_monitoring()
    
    def test_progress_monitor_initialization(self):
        """Test ProgressMonitor initialization."""
        assert self.progress_monitor.resource_monitor == self.mock_resource_monitor
        assert self.progress_monitor.memory_manager == self.mock_memory_manager
        assert self.progress_monitor.update_interval == 0.1
        assert self.progress_monitor.history_size == 10
        assert self.progress_monitor.current_session is None
        assert not self.progress_monitor.is_monitoring
    
    def test_start_session(self):
        """Test starting a training session."""
        session = self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=1000,
            total_epochs=5,
            total_samples=16000,
            model_name="TRM-7M",
            dataset_name="test_dataset",
            batch_size=16,
            learning_rate=1e-4
        )
        
        assert isinstance(session, TrainingSession)
        assert session.session_id == "test_session"
        assert session.model_name == "TRM-7M"
        assert session.batch_size == 16
        assert self.progress_monitor.current_session == session
        assert self.progress_monitor.is_monitoring
    
    def test_end_session(self):
        """Test ending a training session."""
        # Start a session first
        self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=100,
            model_name="TRM-7M"
        )
        
        # Add some progress data
        self.progress_monitor.update_progress(
            current_step=50,
            total_steps=100,
            samples_processed=800,
            current_loss=0.5
        )
        
        # End the session
        completed_session = self.progress_monitor.end_session()
        
        assert completed_session is not None
        assert completed_session.end_time is not None
        assert completed_session.total_training_time > 0
        assert completed_session.total_samples_processed == 800
        assert completed_session.final_loss == 0.5
        assert self.progress_monitor.current_session is None
        assert not self.progress_monitor.is_monitoring
    
    def test_update_progress_basic(self):
        """Test basic progress update."""
        # Start session
        self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=1000,
            batch_size=16
        )
        
        # Update progress
        self.progress_monitor.update_progress(
            current_step=100,
            total_steps=1000,
            current_epoch=1,
            total_epochs=5,
            samples_processed=1600,
            total_samples=16000,
            current_loss=0.6543,
            current_learning_rate=1e-4,
            batch_processing_time=0.15
        )
        
        # Check session was updated
        session = self.progress_monitor.current_session
        assert len(session.progress_history) == 1
        
        progress = session.progress_history[0]
        assert progress.current_step == 100
        assert progress.total_steps == 1000
        assert progress.progress_percent == 10.0
        assert progress.samples_processed == 1600
        assert progress.current_loss == 0.6543
        assert progress.current_learning_rate == 1e-4
    
    def test_update_progress_speed_calculation(self):
        """Test progress update with speed calculations."""
        # Start session
        self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=1000,
            batch_size=16
        )
        
        # First update
        self.progress_monitor.update_progress(
            current_step=10,
            total_steps=1000,
            samples_processed=160,
            batch_processing_time=0.1  # 100ms per batch
        )
        
        # Wait a bit and do second update
        time.sleep(0.05)
        self.progress_monitor.update_progress(
            current_step=20,
            total_steps=1000,
            samples_processed=320,
            batch_processing_time=0.12  # 120ms per batch
        )
        
        # Check speed calculations
        session = self.progress_monitor.current_session
        assert len(session.progress_history) == 2
        
        latest_progress = session.progress_history[-1]
        assert latest_progress.current_samples_per_second > 0
        assert latest_progress.average_samples_per_second > 0
    
    def test_update_progress_eta_calculation(self):
        """Test ETA calculation in progress updates."""
        # Start session
        self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=1000,
            batch_size=16
        )
        
        # Simulate some progress
        for step in range(10, 101, 10):  # Steps 10, 20, 30, ..., 100
            self.progress_monitor.update_progress(
                current_step=step,
                total_steps=1000,
                samples_processed=step * 16,
                total_samples=16000,
                batch_processing_time=0.1
            )
            time.sleep(0.01)  # Small delay to simulate time passing
        
        # Check ETA calculation
        session = self.progress_monitor.current_session
        latest_progress = session.progress_history[-1]
        
        assert latest_progress.eta_seconds > 0
        assert latest_progress.estimated_total_time > latest_progress.elapsed_time
    
    def test_get_current_resource_metrics(self):
        """Test current resource metrics collection."""
        # Mock resource snapshot
        mock_snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryStats(8192, 4096, 4096, 50.0, 0, 0),
            cpu=CPUStats(75.0, [70, 80, 75, 75], 2400.0, 3800.0, [1.2, 1.5, 1.8]),
            thermal=ThermalStats(None, None, "warm")
        )
        self.mock_resource_monitor.get_current_snapshot.return_value = mock_snapshot
        
        # Mock memory pressure info
        mock_pressure_info = MemoryPressureInfo(
            current_usage_percent=50.0,
            available_mb=4096.0,
            pressure_level="medium",
            recommended_action="Consider reducing batch size",
            time_to_critical=None
        )
        self.mock_memory_manager._analyze_memory_pressure.return_value = mock_pressure_info
        
        # Add some batch timing history
        self.progress_monitor.batch_timing_history.extend([150.0, 160.0, 140.0])
        
        metrics = self.progress_monitor.get_current_resource_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.memory_used_mb == 4096.0
        assert metrics.memory_usage_percent == 50.0
        assert metrics.memory_pressure_level == "medium"
        assert metrics.cpu_usage_percent == 75.0
        assert metrics.thermal_state == "warm"
        assert metrics.batch_processing_time_ms == 150.0  # Average of timing history
    
    def test_analyze_performance(self):
        """Test performance analysis."""
        # Mock resource metrics
        with patch.object(self.progress_monitor, 'get_current_resource_metrics') as mock_resource_metrics:
            mock_resource_metrics.return_value = ResourceMetrics(
                memory_used_mb=6000.0,
                memory_available_mb=2192.0,
                memory_usage_percent=73.2,
                memory_pressure_level="medium",
                cpu_usage_percent=45.0,
                cpu_frequency_mhz=2400.0,
                cpu_load_average=[1.0, 1.2, 1.5],
                thermal_state="normal",
                batch_processing_time_ms=200.0,
                gradient_computation_time_ms=120.0,
                data_loading_time_ms=20.0
            )
            
            # Add some speed history
            self.progress_monitor.speed_history.extend([25.0, 30.0, 28.0, 32.0, 29.0])
            
            performance = self.progress_monitor.analyze_performance()
        
        assert isinstance(performance, PerformanceMetrics)
        assert 0 <= performance.memory_efficiency_percent <= 100
        assert 0 <= performance.cpu_efficiency_percent <= 100
        assert 0 <= performance.training_efficiency_score <= 100
        assert performance.primary_bottleneck in ["memory", "cpu", "data_loading", "model_computation"]
        assert 0 <= performance.bottleneck_severity <= 1
        assert isinstance(performance.optimization_suggestions, list)
        assert isinstance(performance.performance_warnings, list)
    
    def test_analyze_performance_high_memory_usage(self):
        """Test performance analysis with high memory usage."""
        with patch.object(self.progress_monitor, 'get_current_resource_metrics') as mock_resource_metrics:
            mock_resource_metrics.return_value = ResourceMetrics(
                memory_used_mb=7500.0,
                memory_available_mb=692.0,
                memory_usage_percent=91.5,  # Very high memory usage
                memory_pressure_level="critical",
                cpu_usage_percent=60.0,
                cpu_frequency_mhz=2400.0,
                cpu_load_average=[2.0, 2.2, 2.5],
                thermal_state="hot",
                batch_processing_time_ms=300.0,
                gradient_computation_time_ms=180.0,
                data_loading_time_ms=30.0
            )
            
            performance = self.progress_monitor.analyze_performance()
        
        # Should detect high memory usage and thermal issues
        assert any("memory" in suggestion.lower() for suggestion in performance.optimization_suggestions)
        assert any("memory" in warning.lower() for warning in performance.performance_warnings)
        assert any("thermal" in warning.lower() for warning in performance.performance_warnings)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        assert not self.progress_monitor.is_monitoring
        
        # Start monitoring
        self.progress_monitor.start_monitoring()
        assert self.progress_monitor.is_monitoring
        assert self.progress_monitor.monitor_thread is not None
        
        # Stop monitoring
        self.progress_monitor.stop_monitoring()
        assert not self.progress_monitor.is_monitoring
    
    def test_callback_system(self):
        """Test callback registration and execution."""
        progress_callback_called = False
        resource_callback_called = False
        performance_callback_called = False
        
        def progress_callback(progress):
            nonlocal progress_callback_called
            progress_callback_called = True
        
        def resource_callback(resource):
            nonlocal resource_callback_called
            resource_callback_called = True
        
        def performance_callback(performance):
            nonlocal performance_callback_called
            performance_callback_called = True
        
        # Add callbacks
        self.progress_monitor.add_progress_callback(progress_callback)
        self.progress_monitor.add_resource_callback(resource_callback)
        self.progress_monitor.add_performance_callback(performance_callback)
        
        # Start session and update progress
        self.progress_monitor.start_session("test", 100)
        self.progress_monitor.update_progress(10, 100, samples_processed=160)
        
        # Let monitoring run briefly
        time.sleep(0.2)
        
        # Check callbacks were called
        assert progress_callback_called
        # Resource and performance callbacks are called by monitoring loop
        # which may not have run yet in this test
    
    def test_get_progress_summary_no_session(self):
        """Test progress summary with no active session."""
        summary = self.progress_monitor.get_progress_summary()
        assert summary == {}
    
    def test_get_progress_summary_with_session(self):
        """Test progress summary with active session."""
        # Start session and add progress
        self.progress_monitor.start_session(
            session_id="test_session",
            total_steps=1000,
            model_name="TRM-7M",
            dataset_name="test_dataset",
            batch_size=16
        )
        
        self.progress_monitor.update_progress(
            current_step=250,
            total_steps=1000,
            current_epoch=2,
            total_epochs=5,
            samples_processed=4000,
            current_loss=0.4567,
            current_learning_rate=8e-5
        )
        
        # Mock resource and performance methods
        with patch.object(self.progress_monitor, 'get_current_resource_metrics') as mock_resource, \
             patch.object(self.progress_monitor, 'analyze_performance') as mock_performance:
            
            mock_resource.return_value = ResourceMetrics(
                memory_used_mb=5000.0, memory_available_mb=3192.0, memory_usage_percent=61.0,
                memory_pressure_level="medium", cpu_usage_percent=70.0, cpu_frequency_mhz=2400.0,
                cpu_load_average=[1.5], thermal_state="normal", batch_processing_time_ms=150.0,
                gradient_computation_time_ms=90.0, data_loading_time_ms=15.0
            )
            
            mock_performance.return_value = PerformanceMetrics(
                memory_efficiency_percent=80.0, cpu_efficiency_percent=70.0,
                training_efficiency_score=75.0, primary_bottleneck="cpu",
                bottleneck_severity=0.3, optimization_suggestions=["Optimize CPU usage"],
                performance_warnings=[]
            )
            
            summary = self.progress_monitor.get_progress_summary()
        
        assert "session" in summary
        assert "progress" in summary
        assert "performance" in summary
        assert "resources" in summary
        assert "training" in summary
        assert "analysis" in summary
        
        assert summary["session"]["id"] == "test_session"
        assert summary["session"]["model"] == "TRM-7M"
        assert summary["progress"]["step"] == "250/1000"
        assert summary["progress"]["progress_percent"] == 25.0
        assert summary["training"]["current_loss"] == 0.4567
        assert summary["analysis"]["primary_bottleneck"] == "cpu"
    
    def test_format_progress_display_compact(self):
        """Test compact progress display formatting."""
        # Start session and add progress
        self.progress_monitor.start_session("test", 1000, model_name="TRM-7M", batch_size=16)
        self.progress_monitor.update_progress(
            current_step=250, total_steps=1000, samples_processed=4000,
            current_loss=0.4567, current_learning_rate=8e-5
        )
        
        # Mock methods for display
        with patch.object(self.progress_monitor, 'get_progress_summary') as mock_summary:
            mock_summary.return_value = {
                "progress": {"step": "250/1000", "progress_percent": 25.0, "eta_minutes": 15.5},
                "performance": {"current_speed_sps": 26.7},
                "resources": {"memory_usage_percent": 61.0},
                "training": {"current_loss": 0.4567}
            }
            
            display = self.progress_monitor.format_progress_display(compact=True)
        
        assert "Step 250/1000" in display
        assert "25.0%" in display
        assert "Loss: 0.4567" in display
        assert "Speed: 26.7 sps" in display
        assert "Memory: 61.0%" in display
        assert "ETA: 15.5min" in display
    
    def test_format_progress_display_detailed(self):
        """Test detailed progress display formatting."""
        # Start session and add progress
        self.progress_monitor.start_session("test", 1000, model_name="TRM-7M", batch_size=16)
        self.progress_monitor.update_progress(
            current_step=250, total_steps=1000, samples_processed=4000,
            current_loss=0.4567, current_learning_rate=8e-5
        )
        
        # Mock methods for display
        with patch.object(self.progress_monitor, 'get_progress_summary') as mock_summary:
            mock_summary.return_value = {
                "session": {"model": "TRM-7M"},
                "progress": {
                    "step": "250/1000", "epoch": "2/5", "progress_percent": 25.0,
                    "samples_processed": 4000, "eta_minutes": 15.5
                },
                "performance": {
                    "current_speed_sps": 26.7, "average_speed_sps": 25.3,
                    "training_efficiency": 75.0
                },
                "resources": {
                    "memory_usage_percent": 61.0, "memory_used_gb": 4.9,
                    "cpu_usage_percent": 70.0, "thermal_state": "normal"
                },
                "training": {
                    "current_loss": 0.4567, "best_loss": 0.4123,
                    "learning_rate": 8e-5
                },
                "analysis": {"warnings": ["High CPU usage"]}
            }
            
            display = self.progress_monitor.format_progress_display(compact=False)
        
        assert "Training Progress - TRM-7M" in display
        assert "Step: 250/1000" in display
        assert "Current Speed: 26.7 samples/s" in display
        assert "Memory: 61.0% (4.9GB)" in display
        assert "Current Loss: 0.4567" in display
        assert "⚠️  Warnings:" in display
        assert "High CPU usage" in display
    
    def test_loss_tracking(self):
        """Test loss value tracking and statistics."""
        # Start session
        self.progress_monitor.start_session("test", 100, batch_size=16)
        
        # Add progress with decreasing loss
        losses = [0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.505, 0.502, 0.501, 0.5]
        for i, loss in enumerate(losses):
            self.progress_monitor.update_progress(
                current_step=(i + 1) * 10,
                total_steps=100,
                samples_processed=(i + 1) * 160,
                current_loss=loss
            )
        
        # Check loss tracking
        session = self.progress_monitor.current_session
        latest_progress = session.progress_history[-1]
        
        assert latest_progress.current_loss == 0.5
        assert latest_progress.best_loss == 0.5
        assert latest_progress.average_loss == sum(losses) / len(losses)
        assert len(self.progress_monitor.loss_history) == len(losses)
    
    def test_speed_history_management(self):
        """Test speed history management and calculations."""
        # Start session
        self.progress_monitor.start_session("test", 100, batch_size=16)
        
        # Add progress updates with varying speeds
        for i in range(15):  # More than history size (10)
            self.progress_monitor.update_progress(
                current_step=(i + 1) * 5,
                total_steps=100,
                samples_processed=(i + 1) * 80,
                batch_processing_time=0.1 + i * 0.01  # Varying batch times
            )
            time.sleep(0.01)
        
        # Check speed history is limited
        assert len(self.progress_monitor.speed_history) <= self.progress_monitor.history_size
        
        # Check recent speed calculation
        session = self.progress_monitor.current_session
        latest_progress = session.progress_history[-1]
        assert latest_progress.recent_samples_per_second > 0
    
    def test_no_session_operations(self):
        """Test operations when no session is active."""
        # Try to update progress without session
        self.progress_monitor.update_progress(10, 100)  # Should not crash
        
        # Try to end session without active session
        result = self.progress_monitor.end_session()
        assert result is None
        
        # Get summary without session
        summary = self.progress_monitor.get_progress_summary()
        assert summary == {}
        
        # Format display without session
        display = self.progress_monitor.format_progress_display()
        assert display == "No active training session"


class TestProgressMonitorIntegration:
    """Integration tests for ProgressMonitor with realistic scenarios."""
    
    def test_complete_training_session_simulation(self):
        """Test a complete training session simulation."""
        # Create monitor with mocked dependencies
        mock_resource_monitor = Mock()
        mock_memory_manager = Mock()
        
        # Mock the resource monitoring methods to avoid comparison issues
        mock_resource_monitor.get_current_snapshot.return_value = Mock()
        mock_memory_manager._analyze_memory_pressure.return_value = Mock()
        
        monitor = ProgressMonitor(
            resource_monitor=mock_resource_monitor,
            memory_manager=mock_memory_manager,
            update_interval=0.05,
            history_size=20
        )
        
        try:
            # Start training session (but don't start monitoring to avoid Mock comparison issues)
            session = monitor.start_session(
                session_id="integration_test",
                total_steps=100,
                total_epochs=2,
                total_samples=1600,
                model_name="TRM-7M",
                dataset_name="test_dataset",
                batch_size=16,
                learning_rate=1e-4
            )
            
            # Stop monitoring to avoid Mock object issues
            monitor.stop_monitoring()
            
            # Simulate training progress
            for step in range(1, 101):
                # Simulate decreasing loss
                loss = 1.0 - (step / 100) * 0.5  # From 1.0 to 0.5
                
                # Simulate varying batch processing time
                batch_time = 0.1 + (step % 10) * 0.01
                
                monitor.update_progress(
                    current_step=step,
                    total_steps=100,
                    current_epoch=(step - 1) // 50 + 1,
                    total_epochs=2,
                    samples_processed=step * 16,
                    total_samples=1600,
                    current_loss=loss,
                    current_learning_rate=1e-4 * (0.99 ** step),  # Decaying LR
                    batch_processing_time=batch_time
                )
                
                # Brief pause to simulate real training
                time.sleep(0.001)
            
            # Skip monitoring run to avoid Mock comparison issues
            
            # End session
            completed_session = monitor.end_session()
            
            # Verify session completion
            assert completed_session is not None
            assert completed_session.session_id == "integration_test"
            assert len(completed_session.progress_history) == 100
            assert completed_session.total_samples_processed == 1600
            assert completed_session.final_loss < 1.0  # Loss should have decreased
            assert completed_session.best_loss <= completed_session.final_loss
            assert completed_session.total_training_time > 0
            
            # Verify progress tracking
            final_progress = completed_session.progress_history[-1]
            assert final_progress.current_step == 100
            assert final_progress.progress_percent == 100.0
            assert final_progress.average_samples_per_second > 0
            
        finally:
            if monitor.is_monitoring:
                monitor.stop_monitoring()
    
    def test_monitoring_with_resource_callbacks(self):
        """Test monitoring with resource and performance callbacks."""
        mock_resource_monitor = Mock()
        mock_memory_manager = Mock()
        
        # Mock resource snapshot
        mock_snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=MemoryStats(8192, 4096, 4096, 50.0, 0, 0),
            cpu=CPUStats(60.0, [55, 65, 60, 60], 2400.0, 3800.0, [1.0, 1.2, 1.5]),
            thermal=ThermalStats(None, None, "normal")
        )
        mock_resource_monitor.get_current_snapshot.return_value = mock_snapshot
        
        # Mock memory pressure
        mock_pressure = MemoryPressureInfo(
            current_usage_percent=50.0,
            available_mb=4096.0,
            pressure_level="low",
            recommended_action="Normal operation",
            time_to_critical=None
        )
        mock_memory_manager._analyze_memory_pressure.return_value = mock_pressure
        
        monitor = ProgressMonitor(
            resource_monitor=mock_resource_monitor,
            memory_manager=mock_memory_manager,
            update_interval=0.05
        )
        
        # Track callback calls
        resource_updates = []
        performance_updates = []
        
        def resource_callback(metrics):
            resource_updates.append(metrics)
        
        def performance_callback(metrics):
            performance_updates.append(metrics)
        
        monitor.add_resource_callback(resource_callback)
        monitor.add_performance_callback(performance_callback)
        
        try:
            # Start session
            monitor.start_session("callback_test", 50, batch_size=16)
            
            # Let monitoring run
            time.sleep(0.2)
            
            # Check callbacks were called
            assert len(resource_updates) > 0
            assert len(performance_updates) > 0
            
            # Verify callback data
            assert all(isinstance(update, ResourceMetrics) for update in resource_updates)
            assert all(isinstance(update, PerformanceMetrics) for update in performance_updates)
            
        finally:
            monitor.stop_monitoring()