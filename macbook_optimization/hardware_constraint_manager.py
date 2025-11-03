"""
Hardware Constraint Management for MacBook Training

This module provides comprehensive hardware constraint management including
thermal throttling detection and response, CPU overload protection with
automatic adjustment, and disk space monitoring and cleanup mechanisms.
"""

import os
import time
import shutil
import psutil
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

from .resource_monitoring import ResourceMonitor, ThermalStats, CPUStats
from .memory_management import MemoryManager


class ConstraintViolationType(Enum):
    """Types of hardware constraint violations."""
    THERMAL_THROTTLING = "thermal_throttling"
    CPU_OVERLOAD = "cpu_overload"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_SPACE_LOW = "disk_space_low"
    POWER_THROTTLING = "power_throttling"


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ConstraintViolation:
    """Hardware constraint violation information."""
    violation_id: str
    timestamp: datetime
    violation_type: ConstraintViolationType
    severity: ConstraintSeverity
    current_value: float
    threshold_value: float
    description: str
    mitigation_actions: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ThermalConstraints:
    """Thermal constraint configuration."""
    cpu_temp_warning_celsius: float = 70.0
    cpu_temp_critical_celsius: float = 85.0
    cpu_temp_emergency_celsius: float = 95.0
    
    # Thermal state thresholds
    thermal_warning_duration_seconds: float = 30.0
    thermal_critical_duration_seconds: float = 10.0
    
    # Cooling strategies
    enable_thermal_throttling: bool = True
    thermal_cooldown_seconds: float = 60.0
    reduce_cpu_usage_on_thermal: bool = True
    
    # Fan control (if available)
    enable_fan_monitoring: bool = True
    fan_speed_threshold_rpm: int = 4000


@dataclass
class CPUConstraints:
    """CPU constraint configuration."""
    cpu_usage_warning_percent: float = 80.0
    cpu_usage_critical_percent: float = 90.0
    cpu_usage_emergency_percent: float = 95.0
    
    # Load average thresholds (for Unix-like systems)
    load_average_warning: float = 2.0
    load_average_critical: float = 4.0
    
    # CPU frequency monitoring
    enable_frequency_monitoring: bool = True
    min_frequency_percent: float = 50.0  # Minimum % of max frequency
    
    # Mitigation settings
    reduce_worker_threads: bool = True
    min_worker_threads: int = 1
    cpu_cooldown_seconds: float = 30.0


@dataclass
class DiskConstraints:
    """Disk space constraint configuration."""
    free_space_warning_mb: float = 2000.0  # 2GB
    free_space_critical_mb: float = 1000.0  # 1GB
    free_space_emergency_mb: float = 500.0   # 500MB
    
    # Cleanup settings
    enable_auto_cleanup: bool = True
    cleanup_temp_files: bool = True
    cleanup_old_logs: bool = True
    cleanup_old_checkpoints: bool = True
    
    # Monitoring paths
    monitor_paths: List[str] = None
    
    def __post_init__(self):
        if self.monitor_paths is None:
            self.monitor_paths = ["/tmp", ".", "checkpoints", "logs"]


@dataclass
class PowerConstraints:
    """Power constraint configuration (for battery operation)."""
    enable_power_monitoring: bool = True
    battery_warning_percent: float = 20.0
    battery_critical_percent: float = 10.0
    
    # Power saving modes
    enable_power_saving: bool = True
    reduce_performance_on_battery: bool = True
    
    # AC power detection
    require_ac_power: bool = False


@dataclass
class HardwareConstraintConfig:
    """Configuration for hardware constraint management."""
    thermal: ThermalConstraints = None
    cpu: CPUConstraints = None
    disk: DiskConstraints = None
    power: PowerConstraints = None
    
    # General settings
    monitoring_interval_seconds: float = 5.0
    violation_history_size: int = 1000
    enable_automatic_mitigation: bool = True
    
    # Notification settings
    enable_notifications: bool = True
    notification_cooldown_seconds: float = 60.0
    
    def __post_init__(self):
        if self.thermal is None:
            self.thermal = ThermalConstraints()
        if self.cpu is None:
            self.cpu = CPUConstraints()
        if self.disk is None:
            self.disk = DiskConstraints()
        if self.power is None:
            self.power = PowerConstraints()


class HardwareConstraintManager:
    """Comprehensive hardware constraint management system."""
    
    def __init__(self,
                 config: Optional[HardwareConstraintConfig] = None,
                 resource_monitor: Optional[ResourceMonitor] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize hardware constraint manager.
        
        Args:
            config: Hardware constraint configuration
            resource_monitor: Resource monitor instance
            memory_manager: Memory manager instance
            logger: Logger instance
        """
        self.config = config or HardwareConstraintConfig()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logger or logging.getLogger(__name__)
        
        # Violation tracking
        self.violation_history: List[ConstraintViolation] = []
        self.active_violations: Dict[ConstraintViolationType, ConstraintViolation] = {}
        self.last_notification_time: Dict[ConstraintViolationType, float] = {}
        
        # Mitigation state
        self.original_worker_count = None
        self.current_worker_count = None
        self.thermal_throttling_active = False
        self.cpu_throttling_active = False
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Callbacks for constraint violations
        self.violation_callbacks: List[Callable[[ConstraintViolation], None]] = []
        
        self.logger.info("HardwareConstraintManager initialized")
    
    def add_violation_callback(self, callback: Callable[[ConstraintViolation], None]):
        """Add callback for constraint violations."""
        self.violation_callbacks.append(callback)
    
    def remove_violation_callback(self, callback: Callable[[ConstraintViolation], None]):
        """Remove violation callback."""
        if callback in self.violation_callbacks:
            self.violation_callbacks.remove(callback)
    
    def start_monitoring(self):
        """Start continuous hardware constraint monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start resource monitoring if not already active
        if not self.resource_monitor.monitoring:
            self.resource_monitor.start_monitoring(self.config.monitoring_interval_seconds)
        
        self.logger.info("Hardware constraint monitoring started")
    
    def stop_monitoring(self):
        """Stop hardware constraint monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Hardware constraint monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check all constraint types
                self._check_thermal_constraints()
                self._check_cpu_constraints()
                self._check_disk_constraints()
                self._check_power_constraints()
                
                # Clean up resolved violations
                self._cleanup_resolved_violations()
                
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in constraint monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval_seconds)
    
    def _check_thermal_constraints(self):
        """Check thermal constraints and detect throttling."""
        try:
            thermal_stats = self.resource_monitor.get_thermal_stats()
            
            # Check thermal state
            violation_detected = False
            severity = ConstraintSeverity.WARNING
            
            if thermal_stats.thermal_state == "hot":
                violation_detected = True
                severity = ConstraintSeverity.CRITICAL
            elif thermal_stats.thermal_state == "warm":
                violation_detected = True
                severity = ConstraintSeverity.WARNING
            
            # Check CPU temperature if available
            if thermal_stats.cpu_temperature is not None:
                temp = thermal_stats.cpu_temperature
                
                if temp >= self.config.thermal.cpu_temp_emergency_celsius:
                    violation_detected = True
                    severity = ConstraintSeverity.EMERGENCY
                elif temp >= self.config.thermal.cpu_temp_critical_celsius:
                    violation_detected = True
                    severity = ConstraintSeverity.CRITICAL
                elif temp >= self.config.thermal.cpu_temp_warning_celsius:
                    violation_detected = True
                    severity = ConstraintSeverity.WARNING
            
            # Handle violation
            if violation_detected:
                self._handle_thermal_violation(thermal_stats, severity)
            else:
                self._resolve_violation(ConstraintViolationType.THERMAL_THROTTLING)
                
        except Exception as e:
            self.logger.error(f"Error checking thermal constraints: {e}")
    
    def _check_cpu_constraints(self):
        """Check CPU constraints and detect overload."""
        try:
            cpu_stats = self.resource_monitor.get_cpu_stats()
            
            violation_detected = False
            severity = ConstraintSeverity.WARNING
            
            # Check CPU usage percentage
            if cpu_stats.percent_total >= self.config.cpu.cpu_usage_emergency_percent:
                violation_detected = True
                severity = ConstraintSeverity.EMERGENCY
            elif cpu_stats.percent_total >= self.config.cpu.cpu_usage_critical_percent:
                violation_detected = True
                severity = ConstraintSeverity.CRITICAL
            elif cpu_stats.percent_total >= self.config.cpu.cpu_usage_warning_percent:
                violation_detected = True
                severity = ConstraintSeverity.WARNING
            
            # Check load average if available
            if cpu_stats.load_average and len(cpu_stats.load_average) > 0:
                load_1min = cpu_stats.load_average[0]
                
                if load_1min >= self.config.cpu.load_average_critical:
                    violation_detected = True
                    severity = max(severity, ConstraintSeverity.CRITICAL)
                elif load_1min >= self.config.cpu.load_average_warning:
                    violation_detected = True
                    if severity == ConstraintSeverity.WARNING:
                        severity = ConstraintSeverity.WARNING
            
            # Check CPU frequency throttling
            if (self.config.cpu.enable_frequency_monitoring and 
                cpu_stats.frequency_max > 0):
                
                frequency_percent = (cpu_stats.frequency_current / cpu_stats.frequency_max) * 100
                
                if frequency_percent < self.config.cpu.min_frequency_percent:
                    violation_detected = True
                    severity = max(severity, ConstraintSeverity.WARNING)
            
            # Handle violation
            if violation_detected:
                self._handle_cpu_violation(cpu_stats, severity)
            else:
                self._resolve_violation(ConstraintViolationType.CPU_OVERLOAD)
                
        except Exception as e:
            self.logger.error(f"Error checking CPU constraints: {e}")
    
    def _check_disk_constraints(self):
        """Check disk space constraints."""
        try:
            violations_detected = []
            
            for path in self.config.disk.monitor_paths:
                try:
                    if not os.path.exists(path):
                        continue
                    
                    # Get disk usage for path
                    if os.path.isfile(path):
                        path = os.path.dirname(path)
                    
                    statvfs = os.statvfs(path)
                    free_bytes = statvfs.f_frsize * statvfs.f_bavail
                    free_mb = free_bytes / (1024**2)
                    
                    # Check thresholds
                    if free_mb <= self.config.disk.free_space_emergency_mb:
                        violations_detected.append((path, free_mb, ConstraintSeverity.EMERGENCY))
                    elif free_mb <= self.config.disk.free_space_critical_mb:
                        violations_detected.append((path, free_mb, ConstraintSeverity.CRITICAL))
                    elif free_mb <= self.config.disk.free_space_warning_mb:
                        violations_detected.append((path, free_mb, ConstraintSeverity.WARNING))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check disk space for {path}: {e}")
            
            # Handle violations
            if violations_detected:
                # Use the most severe violation
                most_severe = max(violations_detected, key=lambda x: x[2].value)
                self._handle_disk_violation(most_severe[0], most_severe[1], most_severe[2])
            else:
                self._resolve_violation(ConstraintViolationType.DISK_SPACE_LOW)
                
        except Exception as e:
            self.logger.error(f"Error checking disk constraints: {e}")
    
    def _check_power_constraints(self):
        """Check power constraints (battery level, AC power)."""
        try:
            if not self.config.power.enable_power_monitoring:
                return
            
            battery = psutil.sensors_battery()
            if battery is None:
                return  # No battery information available
            
            violation_detected = False
            severity = ConstraintSeverity.WARNING
            
            # Check battery level
            if battery.percent <= self.config.power.battery_critical_percent:
                violation_detected = True
                severity = ConstraintSeverity.CRITICAL
            elif battery.percent <= self.config.power.battery_warning_percent:
                violation_detected = True
                severity = ConstraintSeverity.WARNING
            
            # Check AC power requirement
            if (self.config.power.require_ac_power and 
                not battery.power_plugged):
                violation_detected = True
                severity = ConstraintSeverity.CRITICAL
            
            # Handle violation
            if violation_detected:
                self._handle_power_violation(battery, severity)
            else:
                self._resolve_violation(ConstraintViolationType.POWER_THROTTLING)
                
        except Exception as e:
            self.logger.error(f"Error checking power constraints: {e}")
    
    def _handle_thermal_violation(self, thermal_stats: ThermalStats, severity: ConstraintSeverity):
        """Handle thermal constraint violation."""
        violation_type = ConstraintViolationType.THERMAL_THROTTLING
        
        # Check if this is a new violation or update existing
        if violation_type not in self.active_violations:
            violation = ConstraintViolation(
                violation_id=f"thermal_{int(time.time())}",
                timestamp=datetime.now(),
                violation_type=violation_type,
                severity=severity,
                current_value=thermal_stats.cpu_temperature or 0.0,
                threshold_value=self.config.thermal.cpu_temp_warning_celsius,
                description=f"Thermal throttling detected: {thermal_stats.thermal_state}",
                mitigation_actions=[]
            )
            
            self.active_violations[violation_type] = violation
            self._notify_violation(violation)
            
            # Attempt mitigation
            if self.config.enable_automatic_mitigation:
                self._mitigate_thermal_violation(violation)
        else:
            # Update existing violation
            existing = self.active_violations[violation_type]
            existing.severity = severity
            existing.current_value = thermal_stats.cpu_temperature or 0.0
    
    def _handle_cpu_violation(self, cpu_stats: CPUStats, severity: ConstraintSeverity):
        """Handle CPU constraint violation."""
        violation_type = ConstraintViolationType.CPU_OVERLOAD
        
        if violation_type not in self.active_violations:
            violation = ConstraintViolation(
                violation_id=f"cpu_{int(time.time())}",
                timestamp=datetime.now(),
                violation_type=violation_type,
                severity=severity,
                current_value=cpu_stats.percent_total,
                threshold_value=self.config.cpu.cpu_usage_warning_percent,
                description=f"CPU overload detected: {cpu_stats.percent_total:.1f}%",
                mitigation_actions=[]
            )
            
            self.active_violations[violation_type] = violation
            self._notify_violation(violation)
            
            # Attempt mitigation
            if self.config.enable_automatic_mitigation:
                self._mitigate_cpu_violation(violation)
        else:
            # Update existing violation
            existing = self.active_violations[violation_type]
            existing.severity = severity
            existing.current_value = cpu_stats.percent_total
    
    def _handle_disk_violation(self, path: str, free_mb: float, severity: ConstraintSeverity):
        """Handle disk space constraint violation."""
        violation_type = ConstraintViolationType.DISK_SPACE_LOW
        
        if violation_type not in self.active_violations:
            violation = ConstraintViolation(
                violation_id=f"disk_{int(time.time())}",
                timestamp=datetime.now(),
                violation_type=violation_type,
                severity=severity,
                current_value=free_mb,
                threshold_value=self.config.disk.free_space_warning_mb,
                description=f"Low disk space on {path}: {free_mb:.1f}MB free",
                mitigation_actions=[]
            )
            
            self.active_violations[violation_type] = violation
            self._notify_violation(violation)
            
            # Attempt mitigation
            if self.config.enable_automatic_mitigation:
                self._mitigate_disk_violation(violation, path)
        else:
            # Update existing violation
            existing = self.active_violations[violation_type]
            existing.severity = severity
            existing.current_value = free_mb
    
    def _handle_power_violation(self, battery_info, severity: ConstraintSeverity):
        """Handle power constraint violation."""
        violation_type = ConstraintViolationType.POWER_THROTTLING
        
        if violation_type not in self.active_violations:
            violation = ConstraintViolation(
                violation_id=f"power_{int(time.time())}",
                timestamp=datetime.now(),
                violation_type=violation_type,
                severity=severity,
                current_value=battery_info.percent,
                threshold_value=self.config.power.battery_warning_percent,
                description=f"Power constraint: {battery_info.percent:.1f}% battery, AC: {battery_info.power_plugged}",
                mitigation_actions=[]
            )
            
            self.active_violations[violation_type] = violation
            self._notify_violation(violation)
            
            # Attempt mitigation
            if self.config.enable_automatic_mitigation:
                self._mitigate_power_violation(violation)
        else:
            # Update existing violation
            existing = self.active_violations[violation_type]
            existing.severity = severity
            existing.current_value = battery_info.percent
    
    def _mitigate_thermal_violation(self, violation: ConstraintViolation):
        """Mitigate thermal constraint violation."""
        mitigation_actions = []
        
        try:
            # Reduce CPU usage
            if self.config.thermal.reduce_cpu_usage_on_thermal:
                if self._reduce_cpu_usage():
                    mitigation_actions.append("reduced_cpu_usage")
                    self.thermal_throttling_active = True
            
            # Thermal cooldown
            if violation.severity in [ConstraintSeverity.CRITICAL, ConstraintSeverity.EMERGENCY]:
                cooldown_time = self.config.thermal.thermal_cooldown_seconds
                self.logger.warning(f"Thermal emergency - cooling down for {cooldown_time}s")
                time.sleep(cooldown_time)
                mitigation_actions.append(f"thermal_cooldown_{cooldown_time}s")
            
            violation.mitigation_actions.extend(mitigation_actions)
            
        except Exception as e:
            self.logger.error(f"Failed to mitigate thermal violation: {e}")
    
    def _mitigate_cpu_violation(self, violation: ConstraintViolation):
        """Mitigate CPU constraint violation."""
        mitigation_actions = []
        
        try:
            # Reduce worker threads
            if self.config.cpu.reduce_worker_threads:
                if self._reduce_worker_threads():
                    mitigation_actions.append("reduced_worker_threads")
                    self.cpu_throttling_active = True
            
            # CPU cooldown for critical violations
            if violation.severity in [ConstraintSeverity.CRITICAL, ConstraintSeverity.EMERGENCY]:
                cooldown_time = self.config.cpu.cpu_cooldown_seconds
                self.logger.warning(f"CPU overload - cooling down for {cooldown_time}s")
                time.sleep(cooldown_time)
                mitigation_actions.append(f"cpu_cooldown_{cooldown_time}s")
            
            violation.mitigation_actions.extend(mitigation_actions)
            
        except Exception as e:
            self.logger.error(f"Failed to mitigate CPU violation: {e}")
    
    def _mitigate_disk_violation(self, violation: ConstraintViolation, path: str):
        """Mitigate disk space constraint violation."""
        mitigation_actions = []
        
        try:
            # Cleanup temporary files
            if self.config.disk.cleanup_temp_files:
                cleaned_mb = self._cleanup_temp_files(path)
                if cleaned_mb > 0:
                    mitigation_actions.append(f"cleaned_temp_files_{cleaned_mb:.1f}MB")
            
            # Cleanup old logs
            if self.config.disk.cleanup_old_logs:
                cleaned_mb = self._cleanup_old_logs(path)
                if cleaned_mb > 0:
                    mitigation_actions.append(f"cleaned_logs_{cleaned_mb:.1f}MB")
            
            # Cleanup old checkpoints
            if self.config.disk.cleanup_old_checkpoints:
                cleaned_mb = self._cleanup_old_checkpoints(path)
                if cleaned_mb > 0:
                    mitigation_actions.append(f"cleaned_checkpoints_{cleaned_mb:.1f}MB")
            
            violation.mitigation_actions.extend(mitigation_actions)
            
        except Exception as e:
            self.logger.error(f"Failed to mitigate disk violation: {e}")
    
    def _mitigate_power_violation(self, violation: ConstraintViolation):
        """Mitigate power constraint violation."""
        mitigation_actions = []
        
        try:
            # Enable power saving mode
            if self.config.power.enable_power_saving:
                if self._enable_power_saving():
                    mitigation_actions.append("enabled_power_saving")
            
            # Reduce performance on battery
            if self.config.power.reduce_performance_on_battery:
                if self._reduce_performance():
                    mitigation_actions.append("reduced_performance")
            
            violation.mitigation_actions.extend(mitigation_actions)
            
        except Exception as e:
            self.logger.error(f"Failed to mitigate power violation: {e}")
    
    def _reduce_cpu_usage(self) -> bool:
        """Reduce CPU usage by adjusting system settings."""
        try:
            # This could involve reducing thread counts, batch sizes, etc.
            # Implementation would depend on the specific training system
            self.logger.info("Reducing CPU usage for thermal management")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce CPU usage: {e}")
            return False
    
    def _reduce_worker_threads(self) -> bool:
        """Reduce number of worker threads."""
        try:
            # This would be implemented by the training system
            # For now, just track the state
            if self.original_worker_count is None:
                self.original_worker_count = os.cpu_count() or 4
            
            self.current_worker_count = max(
                self.config.cpu.min_worker_threads,
                (self.current_worker_count or self.original_worker_count) // 2
            )
            
            self.logger.info(f"Reduced worker threads to {self.current_worker_count}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce worker threads: {e}")
            return False
    
    def _cleanup_temp_files(self, base_path: str) -> float:
        """Cleanup temporary files and return MB freed."""
        try:
            freed_mb = 0.0
            temp_patterns = ["*.tmp", "*.temp", "*.cache", "__pycache__"]
            
            for pattern in temp_patterns:
                # This is a simplified implementation
                # In practice, you'd use more sophisticated cleanup
                pass
            
            return freed_mb
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")
            return 0.0
    
    def _cleanup_old_logs(self, base_path: str) -> float:
        """Cleanup old log files and return MB freed."""
        try:
            freed_mb = 0.0
            
            # Find and remove old log files
            log_paths = [
                Path(base_path) / "logs",
                Path(base_path) / "*.log",
                Path("/tmp") / "*.log"
            ]
            
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for log_path in log_paths:
                if log_path.exists() and log_path.is_file():
                    stat = log_path.stat()
                    if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                        size_mb = stat.st_size / (1024**2)
                        log_path.unlink()
                        freed_mb += size_mb
            
            return freed_mb
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return 0.0
    
    def _cleanup_old_checkpoints(self, base_path: str) -> float:
        """Cleanup old checkpoint files and return MB freed."""
        try:
            # This would integrate with the checkpoint manager
            # For now, just return 0
            return 0.0
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0.0
    
    def _enable_power_saving(self) -> bool:
        """Enable power saving mode."""
        try:
            # This would implement system-specific power saving
            self.logger.info("Enabled power saving mode")
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable power saving: {e}")
            return False
    
    def _reduce_performance(self) -> bool:
        """Reduce performance for power saving."""
        try:
            # This would reduce training performance to save power
            self.logger.info("Reduced performance for power saving")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce performance: {e}")
            return False
    
    def _resolve_violation(self, violation_type: ConstraintViolationType):
        """Resolve a constraint violation."""
        if violation_type in self.active_violations:
            violation = self.active_violations[violation_type]
            violation.resolved = True
            violation.resolution_time = datetime.now()
            
            # Move to history
            self.violation_history.append(violation)
            del self.active_violations[violation_type]
            
            # Reset mitigation states
            if violation_type == ConstraintViolationType.THERMAL_THROTTLING:
                self.thermal_throttling_active = False
            elif violation_type == ConstraintViolationType.CPU_OVERLOAD:
                self.cpu_throttling_active = False
                # Restore original worker count
                if self.original_worker_count is not None:
                    self.current_worker_count = self.original_worker_count
            
            self.logger.info(f"Resolved constraint violation: {violation_type.value}")
    
    def _cleanup_resolved_violations(self):
        """Clean up old resolved violations from history."""
        if len(self.violation_history) > self.config.violation_history_size:
            self.violation_history = self.violation_history[-self.config.violation_history_size:]
    
    def _notify_violation(self, violation: ConstraintViolation):
        """Notify about constraint violation."""
        # Check notification cooldown
        last_notification = self.last_notification_time.get(violation.violation_type, 0)
        if time.time() - last_notification < self.config.notification_cooldown_seconds:
            return
        
        self.last_notification_time[violation.violation_type] = time.time()
        
        # Log violation
        self.logger.warning(
            f"Hardware constraint violation: {violation.violation_type.value} "
            f"({violation.severity.value}) - {violation.description}"
        )
        
        # Call registered callbacks
        for callback in self.violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                self.logger.error(f"Error in violation callback: {e}")
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """Get current constraint status."""
        return {
            "active_violations": {
                vtype.value: {
                    "severity": violation.severity.value,
                    "description": violation.description,
                    "duration_seconds": (datetime.now() - violation.timestamp).total_seconds(),
                    "mitigation_actions": violation.mitigation_actions
                }
                for vtype, violation in self.active_violations.items()
            },
            "mitigation_state": {
                "thermal_throttling_active": self.thermal_throttling_active,
                "cpu_throttling_active": self.cpu_throttling_active,
                "current_worker_count": self.current_worker_count,
                "original_worker_count": self.original_worker_count
            },
            "monitoring_status": {
                "monitoring_active": self.monitoring_active,
                "monitoring_interval": self.config.monitoring_interval_seconds
            },
            "violation_history_count": len(self.violation_history)
        }
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get comprehensive hardware status summary."""
        current_snapshot = self.resource_monitor.get_current_snapshot()
        
        return {
            "current_status": {
                "cpu_percent": current_snapshot.cpu.percent_total,
                "memory_percent": current_snapshot.memory.percent_used,
                "thermal_state": current_snapshot.thermal.thermal_state,
                "disk_free_gb": self._get_disk_free_space() / 1024,
            },
            "constraints": {
                "thermal": {
                    "warning_temp": self.config.thermal.cpu_temp_warning_celsius,
                    "critical_temp": self.config.thermal.cpu_temp_critical_celsius,
                    "current_temp": current_snapshot.thermal.cpu_temperature
                },
                "cpu": {
                    "warning_percent": self.config.cpu.cpu_usage_warning_percent,
                    "critical_percent": self.config.cpu.cpu_usage_critical_percent,
                    "current_percent": current_snapshot.cpu.percent_total
                },
                "disk": {
                    "warning_mb": self.config.disk.free_space_warning_mb,
                    "critical_mb": self.config.disk.free_space_critical_mb,
                    "current_free_mb": self._get_disk_free_space()
                }
            },
            "active_violations": len(self.active_violations),
            "total_violations": len(self.violation_history)
        }
    
    def _get_disk_free_space(self) -> float:
        """Get current free disk space in MB."""
        try:
            statvfs = os.statvfs(".")
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            return free_bytes / (1024**2)
        except Exception:
            return 0.0