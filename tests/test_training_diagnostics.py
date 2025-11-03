"""
Unit tests for training diagnostics module.

Tests comprehensive logging for all training components, diagnostic tools
for troubleshooting training issues, and automated error reporting with
recovery suggestions.
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from macbook_optimization.training_diagnostics import (
    LogLevel,
    DiagnosticCategory,
    DiagnosticEntry,
    PerformanceMetrics,
    SystemDiagnostics,
    TrainingDiagnosticsConfig,
    TrainingDiagnosticsSystem
)


class TestLogLevel:
    """Test LogLevel enum."""
    
    def test_log_level_values(self):
        """Test LogLevel enum values."""
        assert LogLevel.TRACE.value == "TRACE"
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
        assert LogLevel.PERFORMANCE.value == "PERFORMANCE"
        assert LogLevel.DIAGNOSTIC.value == "DIAGNOSTIC"


class TestDiagnosticCategory:
    """Test DiagnosticCategory enum."""
    
    def test_diagnostic_category_values(self):
        """Test DiagnosticCategory enum values."""
        assert DiagnosticCategory.SYSTEM.value == "system"
        assert DiagnosticCategory.TRAINING.value == "training"
        assert DiagnosticCategory.MODEL.value == "model"
        assert DiagnosticCategory.DATA.value == "data"
        assert DiagnosticCategory.MEMORY.value == "memory"
        assert DiagnosticCategory.HARDWARE.value == "hardware"
        assert DiagnosticCategory.PERFORMANCE.value == "performance"
        assert DiagnosticCategory.ERROR.value == "error"


class TestDiagnosticEntry:
    """Test DiagnosticEntry dataclass."""
    
    def test_diagnostic_entry_creation(self):
        """Test DiagnosticEntry creation."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.TRAINING,
            level=LogLevel.INFO,
            component="training_loop",
            message="Training step completed",
            details={"step": 100, "loss": 0.5},
            context={"epoch": 1, "batch_size": 8},
            suggestions=["Continue training"]
        )
        
        assert entry.category == DiagnosticCategory.TRAINING
        assert entry.level == LogLevel.INFO
        assert entry.component == "training_loop"
        assert entry.message == "Training step completed"
        assert entry.details["step"] == 100
        assert entry.context["epoch"] == 1
        assert "Continue training" in entry.suggestions


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            samples_per_second=15.5,
            batch_processing_time_ms=200.0,
            model_forward_time_ms=150.0,
            model_backward_time_ms=180.0,
            optimizer_step_time_ms=50.0,
            memory_usage_mb=2048.0,
            memory_usage_percent=65.0,
            peak_memory_mb=2200.0,
            cpu_usage_percent=75.0,
            cpu_frequency_mhz=2800.0,
            current_step=1000,
            current_epoch=5,
            learning_rate=1e-4,
            loss_value=0.45,
            accuracy=0.87
        )
        
        assert metrics.samples_per_second == 15.5
        assert metrics.batch_processing_time_ms == 200.0
        assert metrics.memory_usage_mb == 2048.0
        assert metrics.cpu_usage_percent == 75.0
        assert metrics.current_step == 1000
        assert metrics.learning_rate == 1e-4
        assert metrics.load_average == [0.0, 0.0, 0.0]  # Default from __post_init__
    
    def test_performance_metrics_post_init(self):
        """Test PerformanceMetrics __post_init__ method."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            load_average=None
        )
        
        # Should set default load average
        assert metrics.load_average == [0.0, 0.0, 0.0]


class TestSystemDiagnostics:
    """Test SystemDiagnostics dataclass."""
    
    def test_system_diagnostics_creation(self):
        """Test SystemDiagnostics creation."""
        diagnostics = SystemDiagnostics(
            timestamp=datetime.now(),
            platform_info={"system": "Darwin", "machine": "x86_64"},
            python_version="3.9.7",
            pytorch_version="1.12.0",
            cpu_info={"cores": 8, "frequency": 2800},
            memory_info={"total_gb": 16, "available_gb": 8},
            disk_info={"total_gb": 512, "free_gb": 256},
            environment_variables={"PYTHONPATH": "/usr/local/lib"},
            installed_packages=["torch==1.12.0", "numpy==1.21.0"],
            macos_version="12.6",
            hardware_model="MacBook Pro",
            thermal_state="normal"
        )
        
        assert diagnostics.platform_info["system"] == "Darwin"
        assert diagnostics.python_version == "3.9.7"
        assert diagnostics.pytorch_version == "1.12.0"
        assert diagnostics.cpu_info["cores"] == 8
        assert diagnostics.memory_info["total_gb"] == 16
        assert diagnostics.macos_version == "12.6"
        assert diagnostics.thermal_state == "normal"


class TestTrainingDiagnosticsConfig:
    """Test TrainingDiagnosticsConfig dataclass."""
    
    def test_diagnostics_config_defaults(self):
        """Test TrainingDiagnosticsConfig default values."""
        config = TrainingDiagnosticsConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.log_to_file is True
        assert config.log_to_console is True
        assert config.log_file_path == "training_diagnostics.log"
        assert config.max_log_file_size_mb == 100.0
        assert config.log_file_backup_count == 5
        assert config.enable_performance_monitoring is True
        assert config.performance_log_interval_steps == 50
        assert config.detailed_timing is False
        assert config.collect_system_diagnostics is True
        assert config.system_diagnostics_interval_minutes == 30.0
        assert config.enable_error_reporting is True
        assert config.error_report_path == "error_reports"
        assert config.max_error_reports == 100
        assert config.enable_memory_profiling is True
        assert config.memory_snapshot_interval_steps == 100
        assert config.enable_hardware_monitoring is True
        assert config.hardware_check_interval_seconds == 10.0
        assert config.enable_automated_suggestions is True
        assert config.suggestion_confidence_threshold == 0.7


class TestTrainingDiagnosticsSystem:
    """Test TrainingDiagnosticsSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TrainingDiagnosticsConfig(
            log_to_file=False,  # Disable file logging for tests
            log_to_console=False,  # Disable console logging for tests
            system_diagnostics_interval_minutes=0.01,  # Fast for testing
            hardware_check_interval_seconds=0.1
        )
        
        # Create mocks
        self.mock_resource_monitor = Mock()
        self.mock_memory_manager = Mock()
        self.mock_hardware_manager = Mock()
        
        # Create diagnostics system
        self.diagnostics_system = TrainingDiagnosticsSystem(
            config=self.config,
            resource_monitor=self.mock_resource_monitor,
            memory_manager=self.mock_memory_manager,
            hardware_manager=self.mock_hardware_manager
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.diagnostics_system.monitoring_active:
            self.diagnostics_system.stop_monitoring()
    
    def test_diagnostics_system_initialization(self):
        """Test TrainingDiagnosticsSystem initialization."""
        assert self.diagnostics_system.config == self.config
        assert self.diagnostics_system.resource_monitor == self.mock_resource_monitor
        assert self.diagnostics_system.memory_manager == self.mock_memory_manager
        assert self.diagnostics_system.hardware_manager == self.mock_hardware_manager
        
        assert len(self.diagnostics_system.diagnostic_entries) == 0
        assert len(self.diagnostics_system.performance_history) == 0
        assert self.diagnostics_system.monitoring_active is False
        assert len(self.diagnostics_system.step_timings) == 0
        assert self.diagnostics_system.last_performance_log == 0
        assert len(self.diagnostics_system.error_patterns) == 0
        assert len(self.diagnostics_system.suggestion_cache) == 0
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping diagnostic monitoring."""
        # Start monitoring
        self.diagnostics_system.start_monitoring()
        
        assert self.diagnostics_system.monitoring_active is True
        assert self.diagnostics_system.monitoring_thread is not None
        
        # Stop monitoring
        self.diagnostics_system.stop_monitoring()
        
        assert self.diagnostics_system.monitoring_active is False
    
    def test_log_diagnostic(self):
        """Test logging diagnostic entries."""
        self.diagnostics_system.log_diagnostic(
            category=DiagnosticCategory.TRAINING,
            level=LogLevel.INFO,
            component="test_component",
            message="Test diagnostic message",
            details={"test_key": "test_value"},
            context={"step": 100},
            suggestions=["Test suggestion"]
        )
        
        # Process the queued entry
        self.diagnostics_system._process_diagnostic_queue()
        
        assert len(self.diagnostics_system.diagnostic_entries) == 1
        entry = self.diagnostics_system.diagnostic_entries[0]
        
        assert entry.category == DiagnosticCategory.TRAINING
        assert entry.level == LogLevel.INFO
        assert entry.component == "test_component"
        assert entry.message == "Test diagnostic message"
        assert entry.details["test_key"] == "test_value"
        assert entry.context["step"] == 100
        assert "Test suggestion" in entry.suggestions
    
    def test_log_performance_metrics(self):
        """Test logging performance metrics."""
        # Mock memory and resource stats
        mock_memory_stats = Mock()
        mock_memory_stats.used_mb = 2048.0
        mock_memory_stats.percent_used = 65.0
        
        mock_resource_snapshot = Mock()
        mock_resource_snapshot.cpu.percent_total = 75.0
        mock_resource_snapshot.cpu.frequency_current = 2800.0
        mock_resource_snapshot.cpu.load_average = [1.5, 1.3, 1.2]
        
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        self.mock_resource_monitor.get_current_snapshot.return_value = mock_resource_snapshot
        
        # Mock peak memory
        self.mock_memory_manager.peak_memory_mb = 2200.0
        
        # Log performance metrics
        timings = {
            "samples_per_second": 15.5,
            "batch_time": 200.0,
            "forward_time": 150.0,
            "backward_time": 180.0,
            "optimizer_time": 50.0
        }
        
        training_metrics = {
            "learning_rate": 1e-4,
            "loss": 0.45,
            "accuracy": 0.87
        }
        
        self.diagnostics_system.log_performance_metrics(
            step=100,
            epoch=5,
            timings=timings,
            training_metrics=training_metrics
        )
        
        # Should log performance
        assert len(self.diagnostics_system.performance_history) == 1
        metrics = self.diagnostics_system.performance_history[0]
        
        assert metrics.samples_per_second == 15.5
        assert metrics.batch_processing_time_ms == 200.0
        assert metrics.memory_usage_mb == 2048.0
        assert metrics.cpu_usage_percent == 75.0
        assert metrics.current_step == 100
        assert metrics.learning_rate == 1e-4
        
        # Should also create diagnostic entry
        self.diagnostics_system._process_diagnostic_queue()
        assert len(self.diagnostics_system.diagnostic_entries) == 1
        
        entry = self.diagnostics_system.diagnostic_entries[0]
        assert entry.category == DiagnosticCategory.PERFORMANCE
        assert entry.level == LogLevel.PERFORMANCE
    
    def test_log_performance_metrics_interval_check(self):
        """Test performance metrics logging respects interval."""
        # Set last performance log to recent step
        self.diagnostics_system.last_performance_log = 90
        
        # Try to log at step 95 (within interval of 50)
        self.diagnostics_system.log_performance_metrics(
            step=95,
            epoch=1,
            timings={},
            training_metrics={}
        )
        
        # Should not log due to interval
        assert len(self.diagnostics_system.performance_history) == 0
        
        # Try to log at step 150 (outside interval)
        self.diagnostics_system.log_performance_metrics(
            step=150,
            epoch=1,
            timings={},
            training_metrics={}
        )
        
        # Should log this time
        assert len(self.diagnostics_system.performance_history) == 1
    
    def test_analyze_performance_issues(self):
        """Test performance issue analysis."""
        # Create metrics with issues
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            memory_usage_percent=90.0,  # High memory usage
            cpu_usage_percent=95.0,     # High CPU usage
            samples_per_second=0.5      # Slow training
        )
        
        # Analyze performance issues
        self.diagnostics_system._analyze_performance_issues(metrics)
        
        # Process diagnostic queue
        self.diagnostics_system._process_diagnostic_queue()
        
        # Should create diagnostic entries for issues
        assert len(self.diagnostics_system.diagnostic_entries) >= 2  # At least memory and CPU issues
        
        # Check for specific issue types
        issue_messages = [entry.message for entry in self.diagnostics_system.diagnostic_entries]
        assert any("memory usage" in msg.lower() for msg in issue_messages)
        assert any("cpu usage" in msg.lower() for msg in issue_messages)
        assert any("training speed" in msg.lower() for msg in issue_messages)
    
    def test_collect_system_diagnostics(self):
        """Test system diagnostics collection."""
        with patch('platform.system', return_value='Darwin'), \
             patch('platform.release', return_value='21.6.0'), \
             patch('platform.python_version', return_value='3.9.7'), \
             patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock memory info
            mock_memory_obj = Mock()
            mock_memory_obj.total = 16 * 1024**3  # 16GB
            mock_memory_obj.available = 8 * 1024**3  # 8GB
            mock_memory_obj.percent = 50.0
            mock_memory.return_value = mock_memory_obj
            
            # Mock disk info
            mock_disk_obj = Mock()
            mock_disk_obj.total = 512 * 1024**3  # 512GB
            mock_disk_obj.free = 256 * 1024**3   # 256GB
            mock_disk_obj.used = 256 * 1024**3   # 256GB
            mock_disk.return_value = mock_disk_obj
            
            # Mock thermal state
            mock_thermal_stats = Mock()
            mock_thermal_stats.thermal_state = "normal"
            self.mock_resource_monitor.get_thermal_stats.return_value = mock_thermal_stats
            
            diagnostics = self.diagnostics_system._collect_system_diagnostics()
            
            assert isinstance(diagnostics, SystemDiagnostics)
            assert diagnostics.platform_info["system"] == "Darwin"
            assert diagnostics.python_version == "3.9.7"
            assert diagnostics.cpu_info["logical_cores"] == 8
            assert diagnostics.memory_info["total_gb"] == 16.0
            assert diagnostics.disk_info["total_gb"] == 512.0
            assert diagnostics.thermal_state == "normal"
    
    def test_generate_memory_suggestions(self):
        """Test memory-related suggestion generation."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.MEMORY,
            level=LogLevel.WARNING,
            component="memory_monitor",
            message="High memory usage detected",
            details={"memory_percent": 90.0},
            context={},
            suggestions=[]
        )
        
        suggestions = self.diagnostics_system._generate_memory_suggestions(entry)
        
        assert len(suggestions) > 0
        assert any("batch size" in suggestion.lower() for suggestion in suggestions)
        assert any("gradient accumulation" in suggestion.lower() for suggestion in suggestions)
    
    def test_generate_performance_suggestions(self):
        """Test performance-related suggestion generation."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.PERFORMANCE,
            level=LogLevel.WARNING,
            component="performance_monitor",
            message="Slow training detected",
            details={"issue_type": "slow_training"},
            context={},
            suggestions=[]
        )
        
        suggestions = self.diagnostics_system._generate_performance_suggestions(entry)
        
        assert len(suggestions) > 0
        assert any("worker" in suggestion.lower() for suggestion in suggestions)
        assert any("augmentation" in suggestion.lower() for suggestion in suggestions)
    
    def test_generate_hardware_suggestions(self):
        """Test hardware-related suggestion generation."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.HARDWARE,
            level=LogLevel.WARNING,
            component="hardware_monitor",
            message="Thermal throttling detected",
            details={},
            context={},
            suggestions=[]
        )
        
        suggestions = self.diagnostics_system._generate_hardware_suggestions(entry)
        
        assert len(suggestions) > 0
        assert any("thermal" in suggestion.lower() or "cool" in suggestion.lower() for suggestion in suggestions)
        assert any("ventilation" in suggestion.lower() for suggestion in suggestions)
    
    def test_generate_error_suggestions(self):
        """Test error-related suggestion generation."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.ERROR,
            level=LogLevel.ERROR,
            component="data_loader",
            message="JSON parsing error in email data",
            details={},
            context={},
            suggestions=[]
        )
        
        suggestions = self.diagnostics_system._generate_error_suggestions(entry)
        
        assert len(suggestions) > 0
        assert any("validate" in suggestion.lower() or "json" in suggestion.lower() for suggestion in suggestions)
        assert any("corrupted" in suggestion.lower() for suggestion in suggestions)
    
    def test_suggestion_caching(self):
        """Test that suggestions are cached for repeated entries."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.MEMORY,
            level=LogLevel.WARNING,
            component="memory_monitor",
            message="High memory usage",
            details={},
            context={},
            suggestions=[]
        )
        
        # Generate suggestions first time
        suggestions1 = self.diagnostics_system._generate_suggestions(entry)
        
        # Generate suggestions second time (should use cache)
        suggestions2 = self.diagnostics_system._generate_suggestions(entry)
        
        # Should be identical (from cache)
        assert suggestions1 == suggestions2
        
        # Check that cache was used
        cache_key = f"{entry.category.value}_{entry.component}_{hash(entry.message)}"
        assert cache_key in self.diagnostics_system.suggestion_cache
    
    def test_create_diagnostic_report(self):
        """Test comprehensive diagnostic report creation."""
        # Add some test data
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=DiagnosticCategory.TRAINING,
            level=LogLevel.INFO,
            component="training_loop",
            message="Training progress",
            details={"step": 100},
            context={},
            suggestions=[]
        )
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            samples_per_second=10.0,
            memory_usage_percent=60.0,
            cpu_usage_percent=70.0,
            current_step=100
        )
        
        self.diagnostics_system.diagnostic_entries = [entry]
        self.diagnostics_system.performance_history = [metrics]
        
        # Mock system diagnostics
        self.diagnostics_system.system_diagnostics = SystemDiagnostics(
            timestamp=datetime.now(),
            platform_info={"system": "Darwin"},
            python_version="3.9.7",
            pytorch_version="1.12.0",
            cpu_info={},
            memory_info={},
            disk_info={},
            environment_variables={},
            installed_packages=[],
            macos_version="12.6",
            hardware_model="MacBook Pro",
            thermal_state="normal"
        )
        
        # Mock hardware manager
        self.mock_hardware_manager.get_constraint_status.return_value = {
            "active_violations": {},
            "mitigation_state": {}
        }
        
        # Mock memory manager
        self.mock_memory_manager.get_memory_summary.return_value = {
            "current": {"percent_used": 60.0}
        }
        
        report = self.diagnostics_system.create_diagnostic_report()
        
        assert "report_timestamp" in report
        assert "report_id" in report
        assert "summary" in report
        assert "system_diagnostics" in report
        assert "performance_history" in report
        assert "recent_diagnostics" in report
        assert "hardware_status" in report
        assert "memory_status" in report
        assert "recommendations" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_entries"] == 1
        assert summary["performance_entries"] == 1
        
        # Check performance history
        assert len(report["performance_history"]) == 1
        assert "performance_summary" in report
    
    def test_save_diagnostic_report(self):
        """Test saving diagnostic report to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            self.diagnostics_system.config.error_report_path = temp_dir
            
            # Create simple report
            report = {
                "report_id": "test_report",
                "timestamp": datetime.now().isoformat(),
                "data": "test_data"
            }
            
            # Save report
            report_path = self.diagnostics_system.save_diagnostic_report(report)
            
            # Check file was created
            assert Path(report_path).exists()
            
            # Check file content
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            
            assert saved_report["report_id"] == "test_report"
            assert saved_report["data"] == "test_data"
    
    def test_get_diagnostic_summary(self):
        """Test diagnostic summary retrieval."""
        # Add some test data
        self.diagnostics_system.diagnostic_entries = [
            DiagnosticEntry(
                timestamp=datetime.now(),
                category=DiagnosticCategory.ERROR,
                level=LogLevel.ERROR,
                component="test",
                message="Test error",
                details={},
                context={},
                suggestions=[]
            )
        ]
        
        self.diagnostics_system.performance_history = [
            PerformanceMetrics(timestamp=datetime.now())
        ]
        
        # Mock system diagnostics
        self.diagnostics_system.system_diagnostics = SystemDiagnostics(
            timestamp=datetime.now(),
            platform_info={},
            python_version="3.9.7",
            pytorch_version=None,
            cpu_info={},
            memory_info={},
            disk_info={},
            environment_variables={},
            installed_packages=[],
            macos_version=None,
            hardware_model=None,
            thermal_state="normal"
        )
        
        # Mock hardware manager
        self.mock_hardware_manager.active_violations = {}
        
        summary = self.diagnostics_system.get_diagnostic_summary()
        
        assert summary["system_status"] == "healthy"  # No active violations
        assert summary["total_diagnostics"] == 1
        assert summary["recent_errors"] == 1  # One error in last 100 entries
        assert summary["monitoring_active"] is False
        assert summary["performance_samples"] == 1
    
    def test_automated_suggestion_generation(self):
        """Test automated suggestion generation for warning/error entries."""
        # Enable automated suggestions
        self.diagnostics_system.config.enable_automated_suggestions = True
        
        # Log warning entry
        self.diagnostics_system.log_diagnostic(
            category=DiagnosticCategory.MEMORY,
            level=LogLevel.WARNING,
            component="memory_monitor",
            message="High memory usage detected",
            details={"memory_percent": 90.0}
        )
        
        # Process diagnostic queue
        self.diagnostics_system._process_diagnostic_queue()
        
        # Should have generated suggestions
        assert len(self.diagnostics_system.diagnostic_entries) == 1
        entry = self.diagnostics_system.diagnostic_entries[0]
        assert len(entry.suggestions) > 0
    
    def test_hardware_monitoring_integration(self):
        """Test integration with hardware constraint manager."""
        # Mock hardware status with violations
        mock_constraint_status = {
            "active_violations": {
                "thermal_throttling": {
                    "severity": "warning",
                    "description": "CPU temperature high"
                }
            }
        }
        
        self.mock_hardware_manager.get_constraint_status.return_value = mock_constraint_status
        
        # Check hardware status
        self.diagnostics_system._check_hardware_status()
        
        # Process diagnostic queue
        self.diagnostics_system._process_diagnostic_queue()
        
        # Should create diagnostic entry for hardware violation
        assert len(self.diagnostics_system.diagnostic_entries) == 1
        entry = self.diagnostics_system.diagnostic_entries[0]
        assert entry.category == DiagnosticCategory.HARDWARE
        assert entry.level == LogLevel.WARNING
        assert "constraint violation" in entry.message.lower()


class TestTrainingDiagnosticsSystemIntegration:
    """Integration tests for TrainingDiagnosticsSystem with realistic scenarios."""
    
    def test_complete_training_monitoring_scenario(self):
        """Test complete training monitoring scenario."""
        config = TrainingDiagnosticsConfig(
            log_to_file=False,
            log_to_console=False,
            enable_performance_monitoring=True,
            performance_log_interval_steps=10,  # Log every 10 steps
            enable_automated_suggestions=True
        )
        
        mock_memory_manager = Mock()
        mock_resource_monitor = Mock()
        
        diagnostics_system = TrainingDiagnosticsSystem(
            config=config,
            resource_monitor=mock_resource_monitor,
            memory_manager=mock_memory_manager
        )
        
        # Simulate training progress with performance logging
        for step in range(0, 100, 10):
            # Mock resource stats
            mock_memory_stats = Mock()
            mock_memory_stats.used_mb = 2000.0 + step * 10  # Increasing memory usage
            mock_memory_stats.percent_used = 50.0 + step * 0.3
            
            mock_resource_snapshot = Mock()
            mock_resource_snapshot.cpu.percent_total = 70.0 + step * 0.2
            mock_resource_snapshot.cpu.frequency_current = 2800.0
            mock_resource_snapshot.cpu.load_average = [1.5, 1.3, 1.2]
            
            mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
            mock_resource_monitor.get_current_snapshot.return_value = mock_resource_snapshot
            
            # Log performance metrics
            timings = {
                "samples_per_second": 15.0 - step * 0.1,  # Decreasing performance
                "batch_time": 200.0 + step * 2,
                "forward_time": 150.0,
                "backward_time": 180.0
            }
            
            training_metrics = {
                "learning_rate": 1e-4,
                "loss": 1.0 - step * 0.01,  # Decreasing loss
                "accuracy": 0.5 + step * 0.005  # Increasing accuracy
            }
            
            diagnostics_system.log_performance_metrics(
                step=step,
                epoch=step // 50,
                timings=timings,
                training_metrics=training_metrics
            )
        
        # Should have logged performance for each interval
        assert len(diagnostics_system.performance_history) == 10
        
        # Check performance progression
        first_metrics = diagnostics_system.performance_history[0]
        last_metrics = diagnostics_system.performance_history[-1]
        
        assert first_metrics.samples_per_second > last_metrics.samples_per_second  # Performance degraded
        assert first_metrics.memory_usage_mb < last_metrics.memory_usage_mb  # Memory usage increased
        
        # Process diagnostic queue
        diagnostics_system._process_diagnostic_queue()
        
        # Should have created diagnostic entries
        assert len(diagnostics_system.diagnostic_entries) > 0
        
        # Should have performance entries
        performance_entries = [e for e in diagnostics_system.diagnostic_entries 
                             if e.category == DiagnosticCategory.PERFORMANCE]
        assert len(performance_entries) == 10
    
    def test_error_detection_and_suggestion_scenario(self):
        """Test error detection and automated suggestion scenario."""
        config = TrainingDiagnosticsConfig(
            log_to_file=False,
            log_to_console=False,
            enable_automated_suggestions=True,
            suggestion_confidence_threshold=0.5
        )
        
        diagnostics_system = TrainingDiagnosticsSystem(config=config)
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                "category": DiagnosticCategory.MEMORY,
                "level": LogLevel.ERROR,
                "message": "CUDA out of memory error",
                "details": {"memory_usage": 95.0}
            },
            {
                "category": DiagnosticCategory.DATA,
                "level": LogLevel.WARNING,
                "message": "JSON parsing error in email data",
                "details": {"file_path": "corrupted.json"}
            },
            {
                "category": DiagnosticCategory.HARDWARE,
                "level": LogLevel.WARNING,
                "message": "Thermal throttling detected",
                "details": {"cpu_temp": 85.0}
            },
            {
                "category": DiagnosticCategory.PERFORMANCE,
                "level": LogLevel.WARNING,
                "message": "Slow training speed detected",
                "details": {"issue_type": "slow_training", "samples_per_sec": 0.5}
            }
        ]
        
        # Log all error scenarios
        for scenario in error_scenarios:
            diagnostics_system.log_diagnostic(
                category=scenario["category"],
                level=scenario["level"],
                component="test_component",
                message=scenario["message"],
                details=scenario["details"]
            )
        
        # Process diagnostic queue
        diagnostics_system._process_diagnostic_queue()
        
        # Should have logged all errors
        assert len(diagnostics_system.diagnostic_entries) == 4
        
        # All entries should have suggestions (due to warning/error levels)
        for entry in diagnostics_system.diagnostic_entries:
            assert len(entry.suggestions) > 0
        
        # Check specific suggestions
        memory_entry = next(e for e in diagnostics_system.diagnostic_entries 
                          if e.category == DiagnosticCategory.MEMORY)
        assert any("batch size" in suggestion.lower() for suggestion in memory_entry.suggestions)
        
        data_entry = next(e for e in diagnostics_system.diagnostic_entries 
                        if e.category == DiagnosticCategory.DATA)
        assert any("validate" in suggestion.lower() for suggestion in data_entry.suggestions)
    
    def test_comprehensive_diagnostic_report_scenario(self):
        """Test comprehensive diagnostic report generation scenario."""
        config = TrainingDiagnosticsConfig(
            log_to_file=False,
            collect_system_diagnostics=True
        )
        
        mock_hardware_manager = Mock()
        mock_memory_manager = Mock()
        
        diagnostics_system = TrainingDiagnosticsSystem(
            config=config,
            hardware_manager=mock_hardware_manager,
            memory_manager=mock_memory_manager
        )
        
        # Add various diagnostic entries
        diagnostics_system.diagnostic_entries = [
            DiagnosticEntry(
                timestamp=datetime.now() - timedelta(minutes=30),
                category=DiagnosticCategory.TRAINING,
                level=LogLevel.INFO,
                component="training_loop",
                message="Training started",
                details={},
                context={},
                suggestions=[]
            ),
            DiagnosticEntry(
                timestamp=datetime.now() - timedelta(minutes=15),
                category=DiagnosticCategory.ERROR,
                level=LogLevel.ERROR,
                component="data_loader",
                message="Data loading error",
                details={},
                context={},
                suggestions=["Check data format"]
            ),
            DiagnosticEntry(
                timestamp=datetime.now() - timedelta(minutes=5),
                category=DiagnosticCategory.PERFORMANCE,
                level=LogLevel.WARNING,
                component="performance_monitor",
                message="Slow training detected",
                details={"issue_type": "slow_training"},
                context={},
                suggestions=["Increase workers"]
            )
        ]
        
        # Add performance history
        diagnostics_system.performance_history = [
            PerformanceMetrics(
                timestamp=datetime.now() - timedelta(minutes=20),
                samples_per_second=10.0,
                memory_usage_percent=60.0,
                cpu_usage_percent=70.0
            ),
            PerformanceMetrics(
                timestamp=datetime.now() - timedelta(minutes=10),
                samples_per_second=8.0,
                memory_usage_percent=75.0,
                cpu_usage_percent=80.0
            )
        ]
        
        # Mock system diagnostics
        with patch.object(diagnostics_system, '_collect_system_diagnostics') as mock_collect:
            mock_collect.return_value = SystemDiagnostics(
                timestamp=datetime.now(),
                platform_info={"system": "Darwin"},
                python_version="3.9.7",
                pytorch_version="1.12.0",
                cpu_info={"cores": 8},
                memory_info={"total_gb": 16},
                disk_info={"total_gb": 512},
                environment_variables={},
                installed_packages=[],
                macos_version="12.6",
                hardware_model="MacBook Pro",
                thermal_state="normal"
            )
            
            diagnostics_system.system_diagnostics = mock_collect.return_value
        
        # Mock hardware and memory status
        mock_hardware_manager.get_constraint_status.return_value = {
            "active_violations": {},
            "mitigation_state": {"thermal_throttling_active": False}
        }
        
        mock_memory_manager.get_memory_summary.return_value = {
            "current": {"percent_used": 65.0, "used_mb": 4000.0},
            "tracking": {"peak_mb": 4500.0}
        }
        
        # Create comprehensive report
        report = diagnostics_system.create_diagnostic_report(
            include_system_info=True,
            include_performance_history=True,
            include_recent_errors=True,
            last_n_entries=50
        )
        
        # Verify report structure
        assert "report_timestamp" in report
        assert "system_diagnostics" in report
        assert "performance_history" in report
        assert "performance_summary" in report
        assert "recent_diagnostics" in report
        assert "error_analysis" in report
        assert "hardware_status" in report
        assert "memory_status" in report
        assert "recommendations" in report
        
        # Check system diagnostics
        system_diag = report["system_diagnostics"]
        assert system_diag["platform_info"]["system"] == "Darwin"
        assert system_diag["macos_version"] == "12.6"
        
        # Check performance summary
        perf_summary = report["performance_summary"]
        assert perf_summary["total_samples"] == 2
        assert perf_summary["average_samples_per_second"] == 9.0  # (10 + 8) / 2
        
        # Check error analysis
        error_analysis = report["error_analysis"]
        assert error_analysis["total_errors"] == 1  # One error entry
        assert error_analysis["recent_errors_count"] >= 0
        
        # Check recommendations
        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
    
    def test_diagnostics_with_file_logging(self):
        """Test diagnostics system with file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "test_diagnostics.log"
            
            config = TrainingDiagnosticsConfig(
                log_to_file=True,
                log_to_console=False,
                log_file_path=str(log_path),
                max_log_file_size_mb=1.0,  # Small for testing
                log_file_backup_count=2
            )
            
            diagnostics_system = TrainingDiagnosticsSystem(config=config)
            
            # Log some diagnostics
            for i in range(10):
                diagnostics_system.log_diagnostic(
                    category=DiagnosticCategory.TRAINING,
                    level=LogLevel.INFO,
                    component="test",
                    message=f"Test message {i}",
                    details={"iteration": i}
                )
            
            # Process diagnostic queue
            diagnostics_system._process_diagnostic_queue()
            
            # Should have created diagnostic entries
            assert len(diagnostics_system.diagnostic_entries) == 10
            
            # Log file should exist (though content testing would require more complex setup)
            # This mainly tests that the logging setup doesn't crash