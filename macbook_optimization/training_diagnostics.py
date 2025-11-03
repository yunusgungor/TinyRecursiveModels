"""
Comprehensive Logging and Diagnostics System for MacBook Email Training

This module provides detailed logging for all training components, diagnostic
tools for troubleshooting training issues, and automated error reporting
with recovery suggestions.
"""

import os
import json
import time
import logging
import traceback
import platform
import subprocess
import psutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .resource_monitoring import ResourceMonitor
from .memory_management import MemoryManager
from .hardware_constraint_manager import HardwareConstraintManager


class LogLevel(Enum):
    """Enhanced log levels for training diagnostics."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"
    DIAGNOSTIC = "DIAGNOSTIC"


class DiagnosticCategory(Enum):
    """Categories for diagnostic information."""
    SYSTEM = "system"
    TRAINING = "training"
    MODEL = "model"
    DATA = "data"
    MEMORY = "memory"
    HARDWARE = "hardware"
    PERFORMANCE = "performance"
    ERROR = "error"


@dataclass
class DiagnosticEntry:
    """Single diagnostic entry."""
    timestamp: datetime
    category: DiagnosticCategory
    level: LogLevel
    component: str
    message: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    suggestions: List[str]


@dataclass
class PerformanceMetrics:
    """Performance metrics for diagnostics."""
    timestamp: datetime
    
    # Training metrics
    samples_per_second: float = 0.0
    batch_processing_time_ms: float = 0.0
    model_forward_time_ms: float = 0.0
    model_backward_time_ms: float = 0.0
    optimizer_step_time_ms: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    peak_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0
    
    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_frequency_mhz: float = 0.0
    load_average: List[float] = None
    
    # Training state
    current_step: int = 0
    current_epoch: int = 0
    learning_rate: float = 0.0
    loss_value: float = 0.0
    accuracy: float = 0.0
    
    def __post_init__(self):
        if self.load_average is None:
            self.load_average = [0.0, 0.0, 0.0]


@dataclass
class SystemDiagnostics:
    """System diagnostic information."""
    timestamp: datetime
    
    # System info
    platform_info: Dict[str, str]
    python_version: str
    pytorch_version: Optional[str]
    
    # Hardware info
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    disk_info: Dict[str, Any]
    
    # Environment info
    environment_variables: Dict[str, str]
    installed_packages: List[str]
    
    # MacBook specific
    macos_version: Optional[str]
    hardware_model: Optional[str]
    thermal_state: str


@dataclass
class TrainingDiagnosticsConfig:
    """Configuration for training diagnostics system."""
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_path: str = "training_diagnostics.log"
    max_log_file_size_mb: float = 100.0
    log_file_backup_count: int = 5
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_log_interval_steps: int = 50
    detailed_timing: bool = False
    
    # Diagnostic collection
    collect_system_diagnostics: bool = True
    system_diagnostics_interval_minutes: float = 30.0
    
    # Error reporting
    enable_error_reporting: bool = True
    error_report_path: str = "error_reports"
    max_error_reports: int = 100
    
    # Memory diagnostics
    enable_memory_profiling: bool = True
    memory_snapshot_interval_steps: int = 100
    
    # Hardware monitoring
    enable_hardware_monitoring: bool = True
    hardware_check_interval_seconds: float = 10.0
    
    # Diagnostic suggestions
    enable_automated_suggestions: bool = True
    suggestion_confidence_threshold: float = 0.7


class TrainingDiagnosticsSystem:
    """Comprehensive diagnostics and logging system."""
    
    def __init__(self,
                 config: Optional[TrainingDiagnosticsConfig] = None,
                 resource_monitor: Optional[ResourceMonitor] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 hardware_manager: Optional[HardwareConstraintManager] = None):
        """
        Initialize training diagnostics system.
        
        Args:
            config: Diagnostics configuration
            resource_monitor: Resource monitor instance
            memory_manager: Memory manager instance
            hardware_manager: Hardware constraint manager instance
        """
        self.config = config or TrainingDiagnosticsConfig()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.memory_manager = memory_manager or MemoryManager()
        self.hardware_manager = hardware_manager
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Diagnostic storage
        self.diagnostic_entries: List[DiagnosticEntry] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.system_diagnostics: Optional[SystemDiagnostics] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.diagnostic_queue = queue.Queue()
        
        # Performance tracking
        self.step_timings: Dict[str, List[float]] = {}
        self.last_performance_log = 0
        self.last_system_diagnostic = 0
        
        # Error tracking
        self.error_patterns: Dict[str, int] = {}
        self.suggestion_cache: Dict[str, List[str]] = {}
        
        # Initialize system diagnostics
        if self.config.collect_system_diagnostics:
            self.system_diagnostics = self._collect_system_diagnostics()
        
        self.logger.info("TrainingDiagnosticsSystem initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger("training_diagnostics")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level.value))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.log_to_file:
            from logging.handlers import RotatingFileHandler
            
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=int(self.config.max_log_file_size_mb * 1024 * 1024),
                backupCount=self.config.log_file_backup_count
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def start_monitoring(self):
        """Start diagnostic monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start resource monitoring
        if not self.resource_monitor.monitoring:
            self.resource_monitor.start_monitoring()
        
        self.logger.info("Diagnostic monitoring started")
    
    def stop_monitoring(self):
        """Stop diagnostic monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Diagnostic monitoring stopped")
    
    def _monitoring_loop(self):
        """Main diagnostic monitoring loop."""
        while self.monitoring_active:
            try:
                # Process queued diagnostics
                self._process_diagnostic_queue()
                
                # Collect system diagnostics periodically
                if self._should_collect_system_diagnostics():
                    self.system_diagnostics = self._collect_system_diagnostics()
                    self.last_system_diagnostic = time.time()
                
                # Hardware monitoring
                if (self.config.enable_hardware_monitoring and 
                    self.hardware_manager):
                    self._check_hardware_status()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in diagnostic monitoring loop: {e}")
                time.sleep(1.0)
    
    def _process_diagnostic_queue(self):
        """Process queued diagnostic entries."""
        while not self.diagnostic_queue.empty():
            try:
                entry = self.diagnostic_queue.get_nowait()
                self.diagnostic_entries.append(entry)
                
                # Log the entry
                log_level = getattr(logging, entry.level.value)
                self.logger.log(log_level, f"[{entry.category.value}:{entry.component}] {entry.message}")
                
                # Generate suggestions if enabled
                if (self.config.enable_automated_suggestions and 
                    entry.level in [LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]):
                    suggestions = self._generate_suggestions(entry)
                    entry.suggestions.extend(suggestions)
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing diagnostic entry: {e}")
    
    def _should_collect_system_diagnostics(self) -> bool:
        """Check if system diagnostics should be collected."""
        if not self.config.collect_system_diagnostics:
            return False
        
        interval_seconds = self.config.system_diagnostics_interval_minutes * 60
        return time.time() - self.last_system_diagnostic > interval_seconds
    
    def _collect_system_diagnostics(self) -> SystemDiagnostics:
        """Collect comprehensive system diagnostic information."""
        try:
            # Platform information
            platform_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0]
            }
            
            # Python version
            python_version = platform.python_version()
            
            # PyTorch version
            pytorch_version = None
            if TORCH_AVAILABLE:
                pytorch_version = torch.__version__
            
            # CPU information
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent
            }
            
            # Disk information
            disk = psutil.disk_usage('.')
            disk_info = {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_percent": (disk.used / disk.total) * 100
            }
            
            # Environment variables (filtered)
            env_vars = {
                k: v for k, v in os.environ.items()
                if k.startswith(('PYTHON', 'TORCH', 'CUDA', 'MKL', 'OMP'))
            }
            
            # Installed packages (simplified)
            installed_packages = []
            try:
                import pkg_resources
                installed_packages = [
                    f"{pkg.project_name}=={pkg.version}"
                    for pkg in pkg_resources.working_set
                    if pkg.project_name.lower() in ['torch', 'numpy', 'scipy', 'pandas']
                ]
            except Exception:
                pass
            
            # macOS specific information
            macos_version = None
            hardware_model = None
            if platform.system() == "Darwin":
                try:
                    macos_version = platform.mac_ver()[0]
                    
                    # Get hardware model
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if "Model Name" in line:
                                hardware_model = line.split(':')[1].strip()
                                break
                except Exception:
                    pass
            
            # Thermal state
            thermal_state = "unknown"
            if self.resource_monitor:
                thermal_stats = self.resource_monitor.get_thermal_stats()
                thermal_state = thermal_stats.thermal_state
            
            return SystemDiagnostics(
                timestamp=datetime.now(),
                platform_info=platform_info,
                python_version=python_version,
                pytorch_version=pytorch_version,
                cpu_info=cpu_info,
                memory_info=memory_info,
                disk_info=disk_info,
                environment_variables=env_vars,
                installed_packages=installed_packages,
                macos_version=macos_version,
                hardware_model=hardware_model,
                thermal_state=thermal_state
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system diagnostics: {e}")
            return SystemDiagnostics(
                timestamp=datetime.now(),
                platform_info={},
                python_version="unknown",
                pytorch_version=None,
                cpu_info={},
                memory_info={},
                disk_info={},
                environment_variables={},
                installed_packages=[],
                macos_version=None,
                hardware_model=None,
                thermal_state="unknown"
            )
    
    def _check_hardware_status(self):
        """Check hardware status and log issues."""
        try:
            constraint_status = self.hardware_manager.get_constraint_status()
            
            # Log active violations
            for violation_type, violation_info in constraint_status["active_violations"].items():
                self.log_diagnostic(
                    category=DiagnosticCategory.HARDWARE,
                    level=LogLevel.WARNING,
                    component="hardware_monitor",
                    message=f"Hardware constraint violation: {violation_type}",
                    details=violation_info,
                    context={"constraint_status": constraint_status}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking hardware status: {e}")
    
    def log_diagnostic(self,
                      category: DiagnosticCategory,
                      level: LogLevel,
                      component: str,
                      message: str,
                      details: Optional[Dict[str, Any]] = None,
                      context: Optional[Dict[str, Any]] = None,
                      suggestions: Optional[List[str]] = None):
        """
        Log a diagnostic entry.
        
        Args:
            category: Diagnostic category
            level: Log level
            component: Component name
            message: Diagnostic message
            details: Additional details
            context: Context information
            suggestions: Recovery suggestions
        """
        entry = DiagnosticEntry(
            timestamp=datetime.now(),
            category=category,
            level=level,
            component=component,
            message=message,
            details=details or {},
            context=context or {},
            suggestions=suggestions or []
        )
        
        # Queue for processing
        self.diagnostic_queue.put(entry)
    
    def log_performance_metrics(self,
                               step: int,
                               epoch: int,
                               timings: Dict[str, float],
                               training_metrics: Dict[str, float]):
        """
        Log performance metrics.
        
        Args:
            step: Current training step
            epoch: Current epoch
            timings: Timing measurements in milliseconds
            training_metrics: Training metrics (loss, accuracy, etc.)
        """
        if not self.config.enable_performance_monitoring:
            return
        
        # Check if we should log performance
        if step - self.last_performance_log < self.config.performance_log_interval_steps:
            return
        
        try:
            # Get current resource stats
            memory_stats = self.memory_manager.monitor_memory_usage()
            resource_snapshot = self.resource_monitor.get_current_snapshot()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                samples_per_second=timings.get("samples_per_second", 0.0),
                batch_processing_time_ms=timings.get("batch_time", 0.0),
                model_forward_time_ms=timings.get("forward_time", 0.0),
                model_backward_time_ms=timings.get("backward_time", 0.0),
                optimizer_step_time_ms=timings.get("optimizer_time", 0.0),
                memory_usage_mb=memory_stats.used_mb,
                memory_usage_percent=memory_stats.percent_used,
                peak_memory_mb=getattr(self.memory_manager, 'peak_memory_mb', 0.0),
                cpu_usage_percent=resource_snapshot.cpu.percent_total,
                cpu_frequency_mhz=resource_snapshot.cpu.frequency_current,
                load_average=resource_snapshot.cpu.load_average,
                current_step=step,
                current_epoch=epoch,
                learning_rate=training_metrics.get("learning_rate", 0.0),
                loss_value=training_metrics.get("loss", 0.0),
                accuracy=training_metrics.get("accuracy", 0.0)
            )
            
            # Add to history
            self.performance_history.append(metrics)
            
            # Log performance entry
            self.log_diagnostic(
                category=DiagnosticCategory.PERFORMANCE,
                level=LogLevel.PERFORMANCE,
                component="performance_monitor",
                message=f"Step {step} performance metrics",
                details=asdict(metrics),
                context={"step": step, "epoch": epoch}
            )
            
            # Check for performance issues
            self._analyze_performance_issues(metrics)
            
            self.last_performance_log = step
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")
    
    def _analyze_performance_issues(self, metrics: PerformanceMetrics):
        """Analyze performance metrics for potential issues."""
        issues = []
        
        # Check memory usage
        if metrics.memory_usage_percent > 85:
            issues.append({
                "type": "high_memory_usage",
                "severity": "warning" if metrics.memory_usage_percent < 95 else "critical",
                "message": f"High memory usage: {metrics.memory_usage_percent:.1f}%"
            })
        
        # Check CPU usage
        if metrics.cpu_usage_percent > 90:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": f"High CPU usage: {metrics.cpu_usage_percent:.1f}%"
            })
        
        # Check training speed
        if metrics.samples_per_second < 1.0:
            issues.append({
                "type": "slow_training",
                "severity": "warning",
                "message": f"Slow training speed: {metrics.samples_per_second:.2f} samples/sec"
            })
        
        # Log issues
        for issue in issues:
            level = LogLevel.WARNING if issue["severity"] == "warning" else LogLevel.CRITICAL
            self.log_diagnostic(
                category=DiagnosticCategory.PERFORMANCE,
                level=level,
                component="performance_analyzer",
                message=issue["message"],
                details={"issue_type": issue["type"], "metrics": asdict(metrics)}
            )
    
    def _generate_suggestions(self, entry: DiagnosticEntry) -> List[str]:
        """Generate automated suggestions for diagnostic entries."""
        suggestions = []
        
        try:
            # Cache key for suggestions
            cache_key = f"{entry.category.value}_{entry.component}_{hash(entry.message)}"
            
            if cache_key in self.suggestion_cache:
                return self.suggestion_cache[cache_key]
            
            # Generate suggestions based on category and details
            if entry.category == DiagnosticCategory.MEMORY:
                suggestions.extend(self._generate_memory_suggestions(entry))
            elif entry.category == DiagnosticCategory.PERFORMANCE:
                suggestions.extend(self._generate_performance_suggestions(entry))
            elif entry.category == DiagnosticCategory.HARDWARE:
                suggestions.extend(self._generate_hardware_suggestions(entry))
            elif entry.category == DiagnosticCategory.ERROR:
                suggestions.extend(self._generate_error_suggestions(entry))
            
            # Cache suggestions
            self.suggestion_cache[cache_key] = suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to generate suggestions: {e}")
        
        return suggestions
    
    def _generate_memory_suggestions(self, entry: DiagnosticEntry) -> List[str]:
        """Generate memory-related suggestions."""
        suggestions = []
        
        if "high_memory_usage" in entry.message.lower():
            suggestions.extend([
                "Reduce batch size to lower memory usage",
                "Enable gradient accumulation to maintain effective batch size",
                "Clear unused variables and call torch.cuda.empty_cache()",
                "Consider using mixed precision training (fp16)"
            ])
        
        if "out of memory" in entry.message.lower():
            suggestions.extend([
                "Immediately reduce batch size by 50%",
                "Enable dynamic batch sizing",
                "Check for memory leaks in data loading",
                "Restart training with smaller model or sequence length"
            ])
        
        return suggestions
    
    def _generate_performance_suggestions(self, entry: DiagnosticEntry) -> List[str]:
        """Generate performance-related suggestions."""
        suggestions = []
        
        if "slow_training" in entry.details.get("issue_type", ""):
            suggestions.extend([
                "Increase number of data loading workers",
                "Use faster data augmentation techniques",
                "Consider using compiled model (torch.compile)",
                "Profile code to identify bottlenecks"
            ])
        
        if "high_cpu_usage" in entry.details.get("issue_type", ""):
            suggestions.extend([
                "Reduce number of data loading workers",
                "Optimize data preprocessing pipeline",
                "Enable CPU affinity for better thread management"
            ])
        
        return suggestions
    
    def _generate_hardware_suggestions(self, entry: DiagnosticEntry) -> List[str]:
        """Generate hardware-related suggestions."""
        suggestions = []
        
        if "thermal" in entry.message.lower():
            suggestions.extend([
                "Reduce training intensity to cool down system",
                "Ensure adequate ventilation around MacBook",
                "Close unnecessary applications to reduce heat",
                "Consider training in shorter sessions with breaks"
            ])
        
        if "disk_space" in entry.message.lower():
            suggestions.extend([
                "Clean up temporary files and old checkpoints",
                "Move large datasets to external storage",
                "Enable automatic checkpoint cleanup",
                "Compress or archive old training logs"
            ])
        
        return suggestions
    
    def _generate_error_suggestions(self, entry: DiagnosticEntry) -> List[str]:
        """Generate error-related suggestions."""
        suggestions = []
        
        error_message = entry.message.lower()
        
        if "json" in error_message or "parsing" in error_message:
            suggestions.extend([
                "Validate dataset format and encoding",
                "Check for corrupted data files",
                "Enable robust data parsing with error handling",
                "Skip corrupted samples and continue training"
            ])
        
        if "model" in error_message or "forward" in error_message:
            suggestions.extend([
                "Check model configuration parameters",
                "Verify input tensor shapes and types",
                "Load from last known good checkpoint",
                "Reduce model complexity temporarily"
            ])
        
        return suggestions
    
    def create_diagnostic_report(self, 
                               include_system_info: bool = True,
                               include_performance_history: bool = True,
                               include_recent_errors: bool = True,
                               last_n_entries: int = 100) -> Dict[str, Any]:
        """
        Create comprehensive diagnostic report.
        
        Args:
            include_system_info: Include system diagnostic information
            include_performance_history: Include performance metrics history
            include_recent_errors: Include recent error entries
            last_n_entries: Number of recent entries to include
            
        Returns:
            Comprehensive diagnostic report
        """
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "report_id": f"diagnostic_report_{int(time.time())}",
            "summary": self._create_diagnostic_summary()
        }
        
        # System information
        if include_system_info and self.system_diagnostics:
            report["system_diagnostics"] = asdict(self.system_diagnostics)
        
        # Performance history
        if include_performance_history:
            recent_performance = self.performance_history[-50:]  # Last 50 entries
            report["performance_history"] = [asdict(p) for p in recent_performance]
            report["performance_summary"] = self._summarize_performance_history(recent_performance)
        
        # Recent diagnostic entries
        recent_entries = self.diagnostic_entries[-last_n_entries:]
        report["recent_diagnostics"] = [asdict(e) for e in recent_entries]
        
        # Error analysis
        if include_recent_errors:
            error_entries = [e for e in recent_entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
            report["error_analysis"] = self._analyze_error_patterns(error_entries)
        
        # Hardware status
        if self.hardware_manager:
            report["hardware_status"] = self.hardware_manager.get_constraint_status()
        
        # Memory status
        report["memory_status"] = self.memory_manager.get_memory_summary()
        
        # Recommendations
        report["recommendations"] = self._generate_report_recommendations(report)
        
        return report
    
    def _create_diagnostic_summary(self) -> Dict[str, Any]:
        """Create summary of diagnostic information."""
        total_entries = len(self.diagnostic_entries)
        
        # Count by level
        level_counts = {}
        for entry in self.diagnostic_entries:
            level_counts[entry.level.value] = level_counts.get(entry.level.value, 0) + 1
        
        # Count by category
        category_counts = {}
        for entry in self.diagnostic_entries:
            category_counts[entry.category.value] = category_counts.get(entry.category.value, 0) + 1
        
        # Recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_entries = [e for e in self.diagnostic_entries if e.timestamp > one_hour_ago]
        
        return {
            "total_entries": total_entries,
            "entries_by_level": level_counts,
            "entries_by_category": category_counts,
            "recent_activity_count": len(recent_entries),
            "monitoring_active": self.monitoring_active,
            "performance_entries": len(self.performance_history)
        }
    
    def _summarize_performance_history(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Summarize performance history."""
        if not performance_history:
            return {}
        
        # Calculate averages
        avg_samples_per_sec = sum(p.samples_per_second for p in performance_history) / len(performance_history)
        avg_memory_percent = sum(p.memory_usage_percent for p in performance_history) / len(performance_history)
        avg_cpu_percent = sum(p.cpu_usage_percent for p in performance_history) / len(performance_history)
        
        # Find peaks
        peak_memory = max(p.memory_usage_mb for p in performance_history)
        peak_cpu = max(p.cpu_usage_percent for p in performance_history)
        
        return {
            "average_samples_per_second": round(avg_samples_per_sec, 2),
            "average_memory_percent": round(avg_memory_percent, 2),
            "average_cpu_percent": round(avg_cpu_percent, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "total_samples": len(performance_history)
        }
    
    def _analyze_error_patterns(self, error_entries: List[DiagnosticEntry]) -> Dict[str, Any]:
        """Analyze patterns in error entries."""
        if not error_entries:
            return {"message": "No errors found"}
        
        # Count error types
        error_types = {}
        for entry in error_entries:
            error_type = entry.details.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Find most common errors
        most_common = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recent error trend
        recent_errors = [e for e in error_entries if e.timestamp > datetime.now() - timedelta(hours=1)]
        
        return {
            "total_errors": len(error_entries),
            "error_types": error_types,
            "most_common_errors": most_common,
            "recent_errors_count": len(recent_errors),
            "error_rate_per_hour": len(recent_errors)
        }
    
    def _generate_report_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic report."""
        recommendations = []
        
        # Memory recommendations
        memory_status = report.get("memory_status", {})
        current_memory = memory_status.get("current", {})
        if current_memory.get("percent_used", 0) > 80:
            recommendations.append("Consider reducing batch size due to high memory usage")
        
        # Performance recommendations
        perf_summary = report.get("performance_summary", {})
        if perf_summary.get("average_samples_per_second", 0) < 2.0:
            recommendations.append("Training speed is slow - consider optimizing data loading or model")
        
        # Error recommendations
        error_analysis = report.get("error_analysis", {})
        if error_analysis.get("recent_errors_count", 0) > 5:
            recommendations.append("High error rate detected - review recent error patterns")
        
        # Hardware recommendations
        hardware_status = report.get("hardware_status", {})
        if hardware_status.get("active_violations"):
            recommendations.append("Hardware constraint violations detected - check thermal and resource usage")
        
        return recommendations
    
    def save_diagnostic_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save diagnostic report to file.
        
        Args:
            report: Diagnostic report dictionary
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"
        
        report_path = Path(self.config.error_report_path) / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Diagnostic report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save diagnostic report: {e}")
            raise
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get quick diagnostic summary."""
        return {
            "system_status": "healthy" if not self.hardware_manager or not self.hardware_manager.active_violations else "issues_detected",
            "total_diagnostics": len(self.diagnostic_entries),
            "recent_errors": len([e for e in self.diagnostic_entries[-100:] if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]),
            "monitoring_active": self.monitoring_active,
            "last_system_check": self.system_diagnostics.timestamp.isoformat() if self.system_diagnostics else None,
            "performance_samples": len(self.performance_history)
        }