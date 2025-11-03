"""
Unit tests for hardware constraint manager module.

Tests thermal throttling detection and response, CPU overload protection
with automatic adjustment, and disk space monitoring and cleanup mechanisms.
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from macbook_optimization.hardware_constraint_manager import (
    ConstraintViolationType,
    ConstraintSeverity,
    ConstraintViolation,
    ThermalConstraints,
    CPUConstraints,
    DiskConstraints,
    PowerConstraints,
    HardwareConstraintConfig,
    HardwareConstraintManager
)
from macbook_optimization.resource_monitoring import ThermalStats, CPUStats


class TestConstraintEnums:
    """Test constraint-related enums."""
    
    def test_constraint_violation_type_values(self):
        """Test ConstraintViolationType enum values."""
        assert ConstraintViolationType.THERMAL_THROTTLING.value == "thermal_throttling"
        assert ConstraintViolationType.CPU_OVERLOAD.value == "cpu_overload"
        assert ConstraintViolationType.MEMORY_PRESSURE.value == "memory_pressure"
        assert ConstraintViolationType.DISK_SPACE_LOW.value == "disk_space_low"
        assert ConstraintViolationType.POWER_THROTTLING.value == "power_throttling"
    
    def test_constraint_severity_values(self):
        """Test ConstraintSeverity enum values."""
        assert ConstraintSeverity.WARNING.value == "warning"
        assert ConstraintSeverity.CRITICAL.value == "critical"
        assert ConstraintSeverity.EMERGENCY.value == "emergency"


class TestConstraintViolation:
    """Test ConstraintViolation dataclass."""
    
    def test_constraint_violation_creation(self):
        """Test ConstraintViolation creation."""
        violation = ConstraintViolation(
            violation_id="thermal_001",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.THERMAL_THROTTLING,
            severity=ConstraintSeverity.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            description="CPU temperature exceeded warning threshold",
            mitigation_actions=["reduce_cpu_usage"]
        )
        
        assert violation.violation_id == "thermal_001"
        assert violation.violation_type == ConstraintViolationType.THERMAL_THROTTLING
        assert violation.severity == ConstraintSeverity.WARNING
        assert violation.current_value == 75.0
        assert violation.threshold_value == 70.0
        assert violation.description == "CPU temperature exceeded warning threshold"
        assert violation.mitigation_actions == ["reduce_cpu_usage"]
        assert violation.resolved is False
        assert violation.resolution_time is None


class TestThermalConstraints:
    """Test ThermalConstraints dataclass."""
    
    def test_thermal_constraints_defaults(self):
        """Test ThermalConstraints default values."""
        constraints = ThermalConstraints()
        
        assert constraints.cpu_temp_warning_celsius == 70.0
        assert constraints.cpu_temp_critical_celsius == 85.0
        assert constraints.cpu_temp_emergency_celsius == 95.0
        assert constraints.thermal_warning_duration_seconds == 30.0
        assert constraints.thermal_critical_duration_seconds == 10.0
        assert constraints.enable_thermal_throttling is True
        assert constraints.thermal_cooldown_seconds == 60.0
        assert constraints.reduce_cpu_usage_on_thermal is True
        assert constraints.enable_fan_monitoring is True
        assert constraints.fan_speed_threshold_rpm == 4000
    
    def test_thermal_constraints_custom_values(self):
        """Test ThermalConstraints with custom values."""
        constraints = ThermalConstraints(
            cpu_temp_warning_celsius=65.0,
            cpu_temp_critical_celsius=80.0,
            thermal_cooldown_seconds=30.0,
            enable_thermal_throttling=False
        )
        
        assert constraints.cpu_temp_warning_celsius == 65.0
        assert constraints.cpu_temp_critical_celsius == 80.0
        assert constraints.thermal_cooldown_seconds == 30.0
        assert constraints.enable_thermal_throttling is False


class TestCPUConstraints:
    """Test CPUConstraints dataclass."""
    
    def test_cpu_constraints_defaults(self):
        """Test CPUConstraints default values."""
        constraints = CPUConstraints()
        
        assert constraints.cpu_usage_warning_percent == 80.0
        assert constraints.cpu_usage_critical_percent == 90.0
        assert constraints.cpu_usage_emergency_percent == 95.0
        assert constraints.load_average_warning == 2.0
        assert constraints.load_average_critical == 4.0
        assert constraints.enable_frequency_monitoring is True
        assert constraints.min_frequency_percent == 50.0
        assert constraints.reduce_worker_threads is True
        assert constraints.min_worker_threads == 1
        assert constraints.cpu_cooldown_seconds == 30.0


class TestDiskConstraints:
    """Test DiskConstraints dataclass."""
    
    def test_disk_constraints_defaults(self):
        """Test DiskConstraints default values."""
        constraints = DiskConstraints()
        
        assert constraints.free_space_warning_mb == 2000.0
        assert constraints.free_space_critical_mb == 1000.0
        assert constraints.free_space_emergency_mb == 500.0
        assert constraints.enable_auto_cleanup is True
        assert constraints.cleanup_temp_files is True
        assert constraints.cleanup_old_logs is True
        assert constraints.cleanup_old_checkpoints is True
        assert constraints.monitor_paths == ["/tmp", ".", "checkpoints", "logs"]
    
    def test_disk_constraints_post_init(self):
        """Test DiskConstraints __post_init__ method."""
        constraints = DiskConstraints(monitor_paths=None)
        
        # Should set default monitor paths
        assert constraints.monitor_paths == ["/tmp", ".", "checkpoints", "logs"]


class TestPowerConstraints:
    """Test PowerConstraints dataclass."""
    
    def test_power_constraints_defaults(self):
        """Test PowerConstraints default values."""
        constraints = PowerConstraints()
        
        assert constraints.enable_power_monitoring is True
        assert constraints.battery_warning_percent == 20.0
        assert constraints.battery_critical_percent == 10.0
        assert constraints.enable_power_saving is True
        assert constraints.reduce_performance_on_battery is True
        assert constraints.require_ac_power is False


class TestHardwareConstraintConfig:
    """Test HardwareConstraintConfig dataclass."""
    
    def test_hardware_constraint_config_defaults(self):
        """Test HardwareConstraintConfig default values."""
        config = HardwareConstraintConfig()
        
        assert isinstance(config.thermal, ThermalConstraints)
        assert isinstance(config.cpu, CPUConstraints)
        assert isinstance(config.disk, DiskConstraints)
        assert isinstance(config.power, PowerConstraints)
        assert config.monitoring_interval_seconds == 5.0
        assert config.violation_history_size == 1000
        assert config.enable_automatic_mitigation is True
        assert config.enable_notifications is True
        assert config.notification_cooldown_seconds == 60.0
    
    def test_hardware_constraint_config_post_init(self):
        """Test HardwareConstraintConfig __post_init__ method."""
        config = HardwareConstraintConfig(
            thermal=None,
            cpu=None,
            disk=None,
            power=None
        )
        
        # Should create default constraint objects
        assert isinstance(config.thermal, ThermalConstraints)
        assert isinstance(config.cpu, CPUConstraints)
        assert isinstance(config.disk, DiskConstraints)
        assert isinstance(config.power, PowerConstraints)


class TestHardwareConstraintManager:
    """Test HardwareConstraintManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HardwareConstraintConfig(
            monitoring_interval_seconds=0.1,  # Fast for testing
            notification_cooldown_seconds=0.1
        )
        
        # Create mocks
        self.mock_resource_monitor = Mock()
        self.mock_memory_manager = Mock()
        self.mock_logger = Mock()
        
        # Create constraint manager
        self.constraint_manager = HardwareConstraintManager(
            config=self.config,
            resource_monitor=self.mock_resource_monitor,
            memory_manager=self.mock_memory_manager,
            logger=self.mock_logger
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.constraint_manager.monitoring_active:
            self.constraint_manager.stop_monitoring()
    
    def test_constraint_manager_initialization(self):
        """Test HardwareConstraintManager initialization."""
        assert self.constraint_manager.config == self.config
        assert self.constraint_manager.resource_monitor == self.mock_resource_monitor
        assert self.constraint_manager.memory_manager == self.mock_memory_manager
        assert self.constraint_manager.logger == self.mock_logger
        
        assert len(self.constraint_manager.violation_history) == 0
        assert len(self.constraint_manager.active_violations) == 0
        assert len(self.constraint_manager.last_notification_time) == 0
        assert self.constraint_manager.monitoring_active is False
        assert self.constraint_manager.thermal_throttling_active is False
        assert self.constraint_manager.cpu_throttling_active is False
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.constraint_manager.start_monitoring()
        
        assert self.constraint_manager.monitoring_active is True
        assert self.constraint_manager.monitoring_thread is not None
        
        # Stop monitoring
        self.constraint_manager.stop_monitoring()
        
        assert self.constraint_manager.monitoring_active is False
    
    def test_check_thermal_constraints_normal(self):
        """Test thermal constraint checking with normal temperature."""
        thermal_stats = ThermalStats(
            cpu_temperature=60.0,
            fan_speed=2000,
            thermal_state="normal"
        )
        
        self.mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        
        # Should not create violations
        initial_violations = len(self.constraint_manager.active_violations)
        self.constraint_manager._check_thermal_constraints()
        
        assert len(self.constraint_manager.active_violations) == initial_violations
    
    def test_check_thermal_constraints_warning(self):
        """Test thermal constraint checking with warning temperature."""
        thermal_stats = ThermalStats(
            cpu_temperature=75.0,  # Above warning threshold (70.0)
            fan_speed=3500,
            thermal_state="warm"
        )
        
        self.mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        
        self.constraint_manager._check_thermal_constraints()
        
        # Should create thermal violation
        assert ConstraintViolationType.THERMAL_THROTTLING in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.THERMAL_THROTTLING]
        assert violation.severity == ConstraintSeverity.WARNING
        assert violation.current_value == 75.0
    
    def test_check_thermal_constraints_critical(self):
        """Test thermal constraint checking with critical temperature."""
        thermal_stats = ThermalStats(
            cpu_temperature=87.0,  # Above critical threshold (85.0)
            fan_speed=4500,
            thermal_state="hot"
        )
        
        self.mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        
        self.constraint_manager._check_thermal_constraints()
        
        # Should create critical thermal violation
        assert ConstraintViolationType.THERMAL_THROTTLING in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.THERMAL_THROTTLING]
        assert violation.severity == ConstraintSeverity.CRITICAL
        assert violation.current_value == 87.0
    
    def test_check_thermal_constraints_emergency(self):
        """Test thermal constraint checking with emergency temperature."""
        thermal_stats = ThermalStats(
            cpu_temperature=97.0,  # Above emergency threshold (95.0)
            fan_speed=5000,
            thermal_state="hot"
        )
        
        self.mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        
        self.constraint_manager._check_thermal_constraints()
        
        # Should create emergency thermal violation
        assert ConstraintViolationType.THERMAL_THROTTLING in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.THERMAL_THROTTLING]
        assert violation.severity == ConstraintSeverity.EMERGENCY
        assert violation.current_value == 97.0
    
    def test_check_cpu_constraints_normal(self):
        """Test CPU constraint checking with normal usage."""
        cpu_stats = CPUStats(
            percent_total=50.0,  # Below warning threshold
            percent_per_core=[45.0, 55.0, 48.0, 52.0],
            frequency_current=2800.0,
            frequency_max=3200.0,
            load_average=[1.2, 1.1, 1.0]
        )
        
        self.mock_resource_monitor.get_cpu_stats.return_value = cpu_stats
        
        # Should not create violations
        initial_violations = len(self.constraint_manager.active_violations)
        self.constraint_manager._check_cpu_constraints()
        
        assert len(self.constraint_manager.active_violations) == initial_violations
    
    def test_check_cpu_constraints_warning(self):
        """Test CPU constraint checking with warning usage."""
        cpu_stats = CPUStats(
            percent_total=85.0,  # Above warning threshold (80.0)
            percent_per_core=[80.0, 90.0, 85.0, 85.0],
            frequency_current=3000.0,
            frequency_max=3200.0,
            load_average=[2.5, 2.3, 2.1]  # Above warning (2.0)
        )
        
        self.mock_resource_monitor.get_cpu_stats.return_value = cpu_stats
        
        self.constraint_manager._check_cpu_constraints()
        
        # Should create CPU violation
        assert ConstraintViolationType.CPU_OVERLOAD in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.CPU_OVERLOAD]
        assert violation.severity == ConstraintSeverity.WARNING
        assert violation.current_value == 85.0
    
    def test_check_cpu_constraints_critical(self):
        """Test CPU constraint checking with critical usage."""
        cpu_stats = CPUStats(
            percent_total=92.0,  # Above critical threshold (90.0)
            percent_per_core=[90.0, 95.0, 90.0, 93.0],
            frequency_current=2900.0,
            frequency_max=3200.0,
            load_average=[4.5, 4.2, 4.0]  # Above critical (4.0)
        )
        
        self.mock_resource_monitor.get_cpu_stats.return_value = cpu_stats
        
        self.constraint_manager._check_cpu_constraints()
        
        # Should create critical CPU violation
        assert ConstraintViolationType.CPU_OVERLOAD in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.CPU_OVERLOAD]
        assert violation.severity == ConstraintSeverity.CRITICAL
        assert violation.current_value == 92.0
    
    def test_check_cpu_constraints_frequency_throttling(self):
        """Test CPU constraint checking with frequency throttling."""
        cpu_stats = CPUStats(
            percent_total=70.0,  # Normal usage
            percent_per_core=[65.0, 75.0, 68.0, 72.0],
            frequency_current=1600.0,  # 50% of max (3200.0)
            frequency_max=3200.0,
            load_average=[1.8, 1.7, 1.6]
        )
        
        self.mock_resource_monitor.get_cpu_stats.return_value = cpu_stats
        
        self.constraint_manager._check_cpu_constraints()
        
        # Should create CPU violation due to frequency throttling
        assert ConstraintViolationType.CPU_OVERLOAD in self.constraint_manager.active_violations
        violation = self.constraint_manager.active_violations[ConstraintViolationType.CPU_OVERLOAD]
        assert violation.severity == ConstraintSeverity.WARNING
    
    def test_check_disk_constraints_sufficient_space(self):
        """Test disk constraint checking with sufficient space."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock os.statvfs to return sufficient space
            mock_statvfs = Mock()
            mock_statvfs.f_frsize = 4096
            mock_statvfs.f_bavail = 1000000  # ~4GB free
            
            with patch('os.statvfs', return_value=mock_statvfs), \
                 patch('os.path.exists', return_value=True):
                
                # Update config to monitor temp directory
                self.constraint_manager.config.disk.monitor_paths = [temp_dir]
                
                # Should not create violations
                initial_violations = len(self.constraint_manager.active_violations)
                self.constraint_manager._check_disk_constraints()
                
                assert len(self.constraint_manager.active_violations) == initial_violations
    
    def test_check_disk_constraints_low_space_warning(self):
        """Test disk constraint checking with low space warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock os.statvfs to return low space (1.5GB free)
            mock_statvfs = Mock()
            mock_statvfs.f_frsize = 4096
            mock_statvfs.f_bavail = 400000  # ~1.6GB free (below 2GB warning)
            
            with patch('os.statvfs', return_value=mock_statvfs), \
                 patch('os.path.exists', return_value=True):
                
                # Update config to monitor temp directory
                self.constraint_manager.config.disk.monitor_paths = [temp_dir]
                
                self.constraint_manager._check_disk_constraints()
                
                # Should create disk space violation
                assert ConstraintViolationType.DISK_SPACE_LOW in self.constraint_manager.active_violations
                violation = self.constraint_manager.active_violations[ConstraintViolationType.DISK_SPACE_LOW]
                assert violation.severity == ConstraintSeverity.WARNING
    
    def test_check_disk_constraints_low_space_critical(self):
        """Test disk constraint checking with critically low space."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock os.statvfs to return critically low space (800MB free)
            mock_statvfs = Mock()
            mock_statvfs.f_frsize = 4096
            mock_statvfs.f_bavail = 200000  # ~800MB free (below 1GB critical)
            
            with patch('os.statvfs', return_value=mock_statvfs), \
                 patch('os.path.exists', return_value=True):
                
                # Update config to monitor temp directory
                self.constraint_manager.config.disk.monitor_paths = [temp_dir]
                
                self.constraint_manager._check_disk_constraints()
                
                # Should create critical disk space violation
                assert ConstraintViolationType.DISK_SPACE_LOW in self.constraint_manager.active_violations
                violation = self.constraint_manager.active_violations[ConstraintViolationType.DISK_SPACE_LOW]
                assert violation.severity == ConstraintSeverity.CRITICAL
    
    def test_check_disk_constraints_emergency_space(self):
        """Test disk constraint checking with emergency low space."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock os.statvfs to return emergency low space (400MB free)
            mock_statvfs = Mock()
            mock_statvfs.f_frsize = 4096
            mock_statvfs.f_bavail = 100000  # ~400MB free (below 500MB emergency)
            
            with patch('os.statvfs', return_value=mock_statvfs), \
                 patch('os.path.exists', return_value=True):
                
                # Update config to monitor temp directory
                self.constraint_manager.config.disk.monitor_paths = [temp_dir]
                
                self.constraint_manager._check_disk_constraints()
                
                # Should create emergency disk space violation
                assert ConstraintViolationType.DISK_SPACE_LOW in self.constraint_manager.active_violations
                violation = self.constraint_manager.active_violations[ConstraintViolationType.DISK_SPACE_LOW]
                assert violation.severity == ConstraintSeverity.EMERGENCY
    
    def test_check_power_constraints_sufficient_battery(self):
        """Test power constraint checking with sufficient battery."""
        # Mock battery info with sufficient charge
        mock_battery = Mock()
        mock_battery.percent = 50.0  # Above warning threshold (20.0)
        mock_battery.power_plugged = False
        
        with patch('psutil.sensors_battery', return_value=mock_battery):
            # Should not create violations
            initial_violations = len(self.constraint_manager.active_violations)
            self.constraint_manager._check_power_constraints()
            
            assert len(self.constraint_manager.active_violations) == initial_violations
    
    def test_check_power_constraints_low_battery_warning(self):
        """Test power constraint checking with low battery warning."""
        # Mock battery info with low charge
        mock_battery = Mock()
        mock_battery.percent = 15.0  # Below warning threshold (20.0)
        mock_battery.power_plugged = False
        
        with patch('psutil.sensors_battery', return_value=mock_battery):
            self.constraint_manager._check_power_constraints()
            
            # Should create power violation
            assert ConstraintViolationType.POWER_THROTTLING in self.constraint_manager.active_violations
            violation = self.constraint_manager.active_violations[ConstraintViolationType.POWER_THROTTLING]
            assert violation.severity == ConstraintSeverity.WARNING
            assert violation.current_value == 15.0
    
    def test_check_power_constraints_critical_battery(self):
        """Test power constraint checking with critical battery."""
        # Mock battery info with critical charge
        mock_battery = Mock()
        mock_battery.percent = 8.0  # Below critical threshold (10.0)
        mock_battery.power_plugged = False
        
        with patch('psutil.sensors_battery', return_value=mock_battery):
            self.constraint_manager._check_power_constraints()
            
            # Should create critical power violation
            assert ConstraintViolationType.POWER_THROTTLING in self.constraint_manager.active_violations
            violation = self.constraint_manager.active_violations[ConstraintViolationType.POWER_THROTTLING]
            assert violation.severity == ConstraintSeverity.CRITICAL
            assert violation.current_value == 8.0
    
    def test_check_power_constraints_ac_required(self):
        """Test power constraint checking when AC power is required."""
        # Enable AC power requirement
        self.constraint_manager.config.power.require_ac_power = True
        
        # Mock battery info without AC power
        mock_battery = Mock()
        mock_battery.percent = 80.0  # High battery
        mock_battery.power_plugged = False  # No AC power
        
        with patch('psutil.sensors_battery', return_value=mock_battery):
            self.constraint_manager._check_power_constraints()
            
            # Should create critical power violation due to missing AC
            assert ConstraintViolationType.POWER_THROTTLING in self.constraint_manager.active_violations
            violation = self.constraint_manager.active_violations[ConstraintViolationType.POWER_THROTTLING]
            assert violation.severity == ConstraintSeverity.CRITICAL
    
    def test_mitigation_thermal_violation(self):
        """Test thermal violation mitigation."""
        violation = ConstraintViolation(
            violation_id="thermal_test",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.THERMAL_THROTTLING,
            severity=ConstraintSeverity.CRITICAL,
            current_value=87.0,
            threshold_value=85.0,
            description="Critical thermal violation",
            mitigation_actions=[]
        )
        
        # Mock the mitigation methods
        with patch.object(self.constraint_manager, '_reduce_cpu_usage', return_value=True) as mock_reduce_cpu, \
             patch('time.sleep') as mock_sleep:
            
            self.constraint_manager._mitigate_thermal_violation(violation)
            
            # Should attempt CPU usage reduction
            mock_reduce_cpu.assert_called_once()
            
            # Should perform thermal cooldown for critical violation
            mock_sleep.assert_called_once_with(self.config.thermal.thermal_cooldown_seconds)
            
            # Should record mitigation actions
            assert len(violation.mitigation_actions) > 0
            assert "reduced_cpu_usage" in violation.mitigation_actions
    
    def test_mitigation_cpu_violation(self):
        """Test CPU violation mitigation."""
        violation = ConstraintViolation(
            violation_id="cpu_test",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.CPU_OVERLOAD,
            severity=ConstraintSeverity.CRITICAL,
            current_value=92.0,
            threshold_value=90.0,
            description="Critical CPU overload",
            mitigation_actions=[]
        )
        
        # Mock the mitigation methods
        with patch.object(self.constraint_manager, '_reduce_worker_threads', return_value=True) as mock_reduce_threads, \
             patch('time.sleep') as mock_sleep:
            
            self.constraint_manager._mitigate_cpu_violation(violation)
            
            # Should attempt worker thread reduction
            mock_reduce_threads.assert_called_once()
            
            # Should perform CPU cooldown for critical violation
            mock_sleep.assert_called_once_with(self.config.cpu.cpu_cooldown_seconds)
            
            # Should record mitigation actions
            assert len(violation.mitigation_actions) > 0
            assert "reduced_worker_threads" in violation.mitigation_actions
    
    def test_mitigation_disk_violation(self):
        """Test disk violation mitigation."""
        violation = ConstraintViolation(
            violation_id="disk_test",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.DISK_SPACE_LOW,
            severity=ConstraintSeverity.CRITICAL,
            current_value=800.0,
            threshold_value=1000.0,
            description="Critical disk space low",
            mitigation_actions=[]
        )
        
        # Mock the cleanup methods
        with patch.object(self.constraint_manager, '_cleanup_temp_files', return_value=100.0) as mock_cleanup_temp, \
             patch.object(self.constraint_manager, '_cleanup_old_logs', return_value=50.0) as mock_cleanup_logs, \
             patch.object(self.constraint_manager, '_cleanup_old_checkpoints', return_value=200.0) as mock_cleanup_checkpoints:
            
            self.constraint_manager._mitigate_disk_violation(violation, "/test/path")
            
            # Should attempt all cleanup methods
            mock_cleanup_temp.assert_called_once_with("/test/path")
            mock_cleanup_logs.assert_called_once_with("/test/path")
            mock_cleanup_checkpoints.assert_called_once_with("/test/path")
            
            # Should record mitigation actions
            assert len(violation.mitigation_actions) == 3
            assert any("cleaned_temp_files" in action for action in violation.mitigation_actions)
            assert any("cleaned_logs" in action for action in violation.mitigation_actions)
            assert any("cleaned_checkpoints" in action for action in violation.mitigation_actions)
    
    def test_reduce_worker_threads(self):
        """Test worker thread reduction."""
        # Set initial worker count
        self.constraint_manager.original_worker_count = 8
        self.constraint_manager.current_worker_count = 8
        
        success = self.constraint_manager._reduce_worker_threads()
        
        assert success is True
        assert self.constraint_manager.current_worker_count == 4  # Half of original
        assert self.constraint_manager.current_worker_count >= self.config.cpu.min_worker_threads
    
    def test_reduce_worker_threads_minimum_reached(self):
        """Test worker thread reduction when minimum is reached."""
        # Set worker count to minimum
        self.constraint_manager.original_worker_count = 2
        self.constraint_manager.current_worker_count = 1
        
        success = self.constraint_manager._reduce_worker_threads()
        
        assert success is True
        assert self.constraint_manager.current_worker_count == 1  # Should remain at minimum
    
    def test_resolve_violation(self):
        """Test violation resolution."""
        # Create active violation
        violation = ConstraintViolation(
            violation_id="test_violation",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.THERMAL_THROTTLING,
            severity=ConstraintSeverity.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            description="Test violation",
            mitigation_actions=["test_action"]
        )
        
        self.constraint_manager.active_violations[ConstraintViolationType.THERMAL_THROTTLING] = violation
        self.constraint_manager.thermal_throttling_active = True
        
        # Resolve violation
        self.constraint_manager._resolve_violation(ConstraintViolationType.THERMAL_THROTTLING)
        
        # Should move to history and reset state
        assert ConstraintViolationType.THERMAL_THROTTLING not in self.constraint_manager.active_violations
        assert len(self.constraint_manager.violation_history) == 1
        assert self.constraint_manager.violation_history[0].resolved is True
        assert self.constraint_manager.violation_history[0].resolution_time is not None
        assert self.constraint_manager.thermal_throttling_active is False
    
    def test_violation_callback(self):
        """Test violation callback functionality."""
        callback_called = False
        callback_violation = None
        
        def test_callback(violation):
            nonlocal callback_called, callback_violation
            callback_called = True
            callback_violation = violation
        
        self.constraint_manager.add_violation_callback(test_callback)
        
        # Create thermal stats that will trigger violation
        thermal_stats = ThermalStats(
            cpu_temperature=87.0,  # Above critical threshold
            fan_speed=4500,
            thermal_state="hot"
        )
        
        self.mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        
        # Trigger violation
        self.constraint_manager._check_thermal_constraints()
        
        # Callback should be called
        assert callback_called
        assert callback_violation is not None
        assert callback_violation.violation_type == ConstraintViolationType.THERMAL_THROTTLING
    
    def test_get_constraint_status(self):
        """Test constraint status retrieval."""
        # Add active violation
        violation = ConstraintViolation(
            violation_id="test_violation",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.CPU_OVERLOAD,
            severity=ConstraintSeverity.WARNING,
            current_value=85.0,
            threshold_value=80.0,
            description="CPU overload test",
            mitigation_actions=["reduce_threads"]
        )
        
        self.constraint_manager.active_violations[ConstraintViolationType.CPU_OVERLOAD] = violation
        self.constraint_manager.cpu_throttling_active = True
        self.constraint_manager.current_worker_count = 4
        self.constraint_manager.original_worker_count = 8
        
        status = self.constraint_manager.get_constraint_status()
        
        assert "active_violations" in status
        assert "mitigation_state" in status
        assert "monitoring_status" in status
        assert "violation_history_count" in status
        
        # Check active violations
        assert ConstraintViolationType.CPU_OVERLOAD.value in status["active_violations"]
        cpu_violation = status["active_violations"][ConstraintViolationType.CPU_OVERLOAD.value]
        assert cpu_violation["severity"] == "warning"
        assert cpu_violation["description"] == "CPU overload test"
        assert "reduce_threads" in cpu_violation["mitigation_actions"]
        
        # Check mitigation state
        mitigation_state = status["mitigation_state"]
        assert mitigation_state["cpu_throttling_active"] is True
        assert mitigation_state["current_worker_count"] == 4
        assert mitigation_state["original_worker_count"] == 8
    
    def test_get_hardware_summary(self):
        """Test hardware summary retrieval."""
        # Mock current snapshot
        mock_snapshot = Mock()
        mock_snapshot.cpu.percent_total = 75.0
        mock_snapshot.memory.percent_used = 60.0
        mock_snapshot.thermal.thermal_state = "warm"
        mock_snapshot.thermal.cpu_temperature = 72.0
        
        self.mock_resource_monitor.get_current_snapshot.return_value = mock_snapshot
        
        # Mock disk free space
        with patch.object(self.constraint_manager, '_get_disk_free_space', return_value=5000.0):
            summary = self.constraint_manager.get_hardware_summary()
        
        assert "current_status" in summary
        assert "constraints" in summary
        assert "active_violations" in summary
        assert "total_violations" in summary
        
        # Check current status
        current_status = summary["current_status"]
        assert current_status["cpu_percent"] == 75.0
        assert current_status["memory_percent"] == 60.0
        assert current_status["thermal_state"] == "warm"
        assert current_status["disk_free_gb"] == 5.0  # 5000MB / 1024
        
        # Check constraints
        constraints = summary["constraints"]
        assert "thermal" in constraints
        assert "cpu" in constraints
        assert "disk" in constraints
        
        thermal_constraints = constraints["thermal"]
        assert thermal_constraints["current_temp"] == 72.0
        assert thermal_constraints["warning_temp"] == self.config.thermal.cpu_temp_warning_celsius
    
    def test_notification_cooldown(self):
        """Test that notifications respect cooldown period."""
        # Create violation
        violation = ConstraintViolation(
            violation_id="test_violation",
            timestamp=datetime.now(),
            violation_type=ConstraintViolationType.THERMAL_THROTTLING,
            severity=ConstraintSeverity.WARNING,
            current_value=75.0,
            threshold_value=70.0,
            description="Test violation",
            mitigation_actions=[]
        )
        
        # First notification should go through
        self.constraint_manager._notify_violation(violation)
        
        # Check that notification time was recorded
        assert ConstraintViolationType.THERMAL_THROTTLING in self.constraint_manager.last_notification_time
        
        # Immediate second notification should be blocked by cooldown
        # (This is tested by checking the internal logic, as the actual notification
        # is just logging which is mocked)
        last_time = self.constraint_manager.last_notification_time[ConstraintViolationType.THERMAL_THROTTLING]
        current_time = time.time()
        
        # Should be within cooldown period
        assert current_time - last_time < self.config.notification_cooldown_seconds + 1.0


class TestHardwareConstraintManagerIntegration:
    """Integration tests for HardwareConstraintManager with realistic scenarios."""
    
    def test_thermal_throttling_scenario(self):
        """Test complete thermal throttling detection and mitigation scenario."""
        config = HardwareConstraintConfig(
            monitoring_interval_seconds=0.1,
            enable_automatic_mitigation=True
        )
        config.thermal.cpu_temp_critical_celsius = 85.0
        config.thermal.thermal_cooldown_seconds = 0.1  # Fast for testing
        
        mock_resource_monitor = Mock()
        constraint_manager = HardwareConstraintManager(
            config=config,
            resource_monitor=mock_resource_monitor
        )
        
        # Simulate thermal throttling scenario
        critical_thermal_stats = ThermalStats(
            cpu_temperature=87.0,  # Above critical threshold
            fan_speed=4800,
            thermal_state="hot"
        )
        
        mock_resource_monitor.get_thermal_stats.return_value = critical_thermal_stats
        
        # Mock mitigation methods
        with patch.object(constraint_manager, '_reduce_cpu_usage', return_value=True) as mock_reduce_cpu, \
             patch('time.sleep') as mock_sleep:
            
            # Check thermal constraints
            constraint_manager._check_thermal_constraints()
            
            # Should detect violation
            assert ConstraintViolationType.THERMAL_THROTTLING in constraint_manager.active_violations
            violation = constraint_manager.active_violations[ConstraintViolationType.THERMAL_THROTTLING]
            assert violation.severity == ConstraintSeverity.CRITICAL
            
            # Should attempt mitigation
            mock_reduce_cpu.assert_called_once()
            mock_sleep.assert_called_once()
            
            # Should activate thermal throttling
            assert constraint_manager.thermal_throttling_active is True
        
        # Simulate temperature returning to normal
        normal_thermal_stats = ThermalStats(
            cpu_temperature=65.0,  # Below warning threshold
            fan_speed=2500,
            thermal_state="normal"
        )
        
        mock_resource_monitor.get_thermal_stats.return_value = normal_thermal_stats
        
        # Check thermal constraints again
        constraint_manager._check_thermal_constraints()
        
        # Should resolve violation
        assert ConstraintViolationType.THERMAL_THROTTLING not in constraint_manager.active_violations
        assert len(constraint_manager.violation_history) == 1
        assert constraint_manager.violation_history[0].resolved is True
        assert constraint_manager.thermal_throttling_active is False
    
    def test_cpu_overload_scenario(self):
        """Test CPU overload detection and mitigation scenario."""
        config = HardwareConstraintConfig(
            monitoring_interval_seconds=0.1,
            enable_automatic_mitigation=True
        )
        config.cpu.cpu_usage_critical_percent = 90.0
        config.cpu.cpu_cooldown_seconds = 0.1  # Fast for testing
        
        mock_resource_monitor = Mock()
        constraint_manager = HardwareConstraintManager(
            config=config,
            resource_monitor=mock_resource_monitor
        )
        
        # Simulate CPU overload scenario
        overload_cpu_stats = CPUStats(
            percent_total=93.0,  # Above critical threshold
            percent_per_core=[90.0, 95.0, 92.0, 95.0],
            frequency_current=2800.0,
            frequency_max=3200.0,
            load_average=[4.5, 4.2, 4.0]
        )
        
        mock_resource_monitor.get_cpu_stats.return_value = overload_cpu_stats
        
        # Mock mitigation methods
        with patch.object(constraint_manager, '_reduce_worker_threads', return_value=True) as mock_reduce_threads, \
             patch('time.sleep') as mock_sleep:
            
            # Check CPU constraints
            constraint_manager._check_cpu_constraints()
            
            # Should detect violation
            assert ConstraintViolationType.CPU_OVERLOAD in constraint_manager.active_violations
            violation = constraint_manager.active_violations[ConstraintViolationType.CPU_OVERLOAD]
            assert violation.severity == ConstraintSeverity.CRITICAL
            
            # Should attempt mitigation
            mock_reduce_threads.assert_called_once()
            mock_sleep.assert_called_once()
            
            # Should activate CPU throttling
            assert constraint_manager.cpu_throttling_active is True
    
    def test_disk_space_cleanup_scenario(self):
        """Test disk space monitoring and cleanup scenario."""
        config = HardwareConstraintConfig(
            enable_automatic_mitigation=True
        )
        config.disk.free_space_critical_mb = 1000.0
        config.disk.cleanup_temp_files = True
        config.disk.cleanup_old_logs = True
        
        constraint_manager = HardwareConstraintManager(config=config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock low disk space
            mock_statvfs = Mock()
            mock_statvfs.f_frsize = 4096
            mock_statvfs.f_bavail = 200000  # ~800MB free (below critical)
            
            with patch('os.statvfs', return_value=mock_statvfs), \
                 patch('os.path.exists', return_value=True), \
                 patch.object(constraint_manager, '_cleanup_temp_files', return_value=150.0) as mock_cleanup_temp, \
                 patch.object(constraint_manager, '_cleanup_old_logs', return_value=100.0) as mock_cleanup_logs, \
                 patch.object(constraint_manager, '_cleanup_old_checkpoints', return_value=200.0) as mock_cleanup_checkpoints:
                
                # Update config to monitor temp directory
                constraint_manager.config.disk.monitor_paths = [temp_dir]
                
                # Check disk constraints
                constraint_manager._check_disk_constraints()
                
                # Should detect violation
                assert ConstraintViolationType.DISK_SPACE_LOW in constraint_manager.active_violations
                violation = constraint_manager.active_violations[ConstraintViolationType.DISK_SPACE_LOW]
                assert violation.severity == ConstraintSeverity.CRITICAL
                
                # Should attempt cleanup
                mock_cleanup_temp.assert_called_once()
                mock_cleanup_logs.assert_called_once()
                mock_cleanup_checkpoints.assert_called_once()
                
                # Should record cleanup actions
                assert len(violation.mitigation_actions) == 3
    
    def test_multiple_constraint_violations_scenario(self):
        """Test handling multiple simultaneous constraint violations."""
        config = HardwareConstraintConfig(
            monitoring_interval_seconds=0.1,
            enable_automatic_mitigation=True
        )
        
        mock_resource_monitor = Mock()
        constraint_manager = HardwareConstraintManager(
            config=config,
            resource_monitor=mock_resource_monitor
        )
        
        # Simulate multiple violations
        # 1. Thermal violation
        thermal_stats = ThermalStats(
            cpu_temperature=87.0,  # Critical
            fan_speed=4800,
            thermal_state="hot"
        )
        
        # 2. CPU overload
        cpu_stats = CPUStats(
            percent_total=92.0,  # Critical
            percent_per_core=[90.0, 95.0, 90.0, 93.0],
            frequency_current=2900.0,
            frequency_max=3200.0,
            load_average=[4.5, 4.2, 4.0]
        )
        
        # 3. Low battery
        mock_battery = Mock()
        mock_battery.percent = 8.0  # Critical
        mock_battery.power_plugged = False
        
        mock_resource_monitor.get_thermal_stats.return_value = thermal_stats
        mock_resource_monitor.get_cpu_stats.return_value = cpu_stats
        
        with patch('psutil.sensors_battery', return_value=mock_battery), \
             patch.object(constraint_manager, '_reduce_cpu_usage', return_value=True), \
             patch.object(constraint_manager, '_reduce_worker_threads', return_value=True), \
             patch.object(constraint_manager, '_enable_power_saving', return_value=True), \
             patch('time.sleep'):
            
            # Check all constraints
            constraint_manager._check_thermal_constraints()
            constraint_manager._check_cpu_constraints()
            constraint_manager._check_power_constraints()
            
            # Should detect all violations
            assert len(constraint_manager.active_violations) == 3
            assert ConstraintViolationType.THERMAL_THROTTLING in constraint_manager.active_violations
            assert ConstraintViolationType.CPU_OVERLOAD in constraint_manager.active_violations
            assert ConstraintViolationType.POWER_THROTTLING in constraint_manager.active_violations
            
            # All should be critical severity
            for violation in constraint_manager.active_violations.values():
                assert violation.severity == ConstraintSeverity.CRITICAL
            
            # Should activate multiple throttling modes
            assert constraint_manager.thermal_throttling_active is True
            assert constraint_manager.cpu_throttling_active is True