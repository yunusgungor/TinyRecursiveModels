"""
Real-time Resource Monitoring for Email Training

This module provides specialized resource monitoring for email classification training
with real-time dashboards, alerts, and performance tracking optimized for MacBook
hardware and EmailTRM models.
"""

import time
import threading
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import json

from .resource_monitoring import ResourceMonitor, ResourceSnapshot, MemoryStats, CPUStats, ThermalStats
from .email_memory_manager import EmailMemoryManager, EmailMemoryMetrics

logger = logging.getLogger(__name__)


@dataclass
class EmailTrainingSnapshot:
    """Extended resource snapshot for email training."""
    # Base resource info
    timestamp: float
    memory: MemoryStats
    cpu: CPUStats
    thermal: ThermalStats
    
    # Email-specific metrics
    email_metrics: Optional[EmailMemoryMetrics] = None
    training_step: Optional[int] = None
    current_accuracy: Optional[float] = None
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Performance indicators
    emails_processed_total: int = 0
    batch_processing_time_ms: float = 0.0
    model_forward_time_ms: float = 0.0
    
    # Resource efficiency
    memory_efficiency: float = 0.0
    cpu_efficiency: float = 0.0
    thermal_efficiency: float = 1.0  # 1.0 = no throttling


@dataclass
class ResourceAlert:
    """Resource monitoring alert."""
    timestamp: float
    severity: str  # "info", "warning", "critical"
    category: str  # "memory", "cpu", "thermal", "performance"
    message: str
    current_value: float
    threshold_value: float
    suggested_action: str


@dataclass
class TrainingPerformanceMetrics:
    """Training performance metrics over time."""
    # Throughput metrics
    average_emails_per_second: float
    peak_emails_per_second: float
    training_efficiency: float  # Actual vs theoretical max throughput
    
    # Resource utilization
    average_memory_usage: float
    peak_memory_usage: float
    average_cpu_usage: float
    peak_cpu_usage: float
    
    # Stability metrics
    memory_pressure_events: int
    thermal_throttling_events: int
    performance_degradation_events: int
    
    # Training quality
    convergence_rate: float  # Accuracy improvement per step
    training_stability: float  # Variance in loss/accuracy
    resource_stability: float  # Variance in resource usage


class EmailResourceMonitor:
    """
    Specialized resource monitor for email classification training.
    
    Provides real-time monitoring, alerting, and performance tracking
    specifically optimized for EmailTRM training on MacBook hardware.
    """
    
    def __init__(self, 
                 email_memory_manager: Optional[EmailMemoryManager] = None,
                 history_size: int = 1000,
                 alert_cooldown_seconds: float = 30.0):
        """
        Initialize email resource monitor.
        
        Args:
            email_memory_manager: Email memory manager instance
            history_size: Number of snapshots to keep in history
            alert_cooldown_seconds: Minimum time between similar alerts
        """
        self.email_memory_manager = email_memory_manager
        self.history_size = history_size
        self.alert_cooldown_seconds = alert_cooldown_seconds
        
        # Initialize base resource monitor
        self.base_monitor = ResourceMonitor(history_size=history_size)
        
        # Email training history
        self.training_history: deque = deque(maxlen=history_size)
        
        # Alert management
        self.alerts: List[ResourceAlert] = []
        self.last_alert_times: Dict[str, float] = {}
        
        # Performance tracking
        self.training_start_time: Optional[float] = None
        self.total_emails_processed = 0
        self.total_training_steps = 0
        
        # Callbacks
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        self.performance_callbacks: List[Callable[[EmailTrainingSnapshot], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info("EmailResourceMonitor initialized")
    
    def start_training_monitoring(self, 
                                training_id: str,
                                monitoring_interval: float = 1.0) -> None:
        """
        Start monitoring for email training session.
        
        Args:
            training_id: Unique training session identifier
            monitoring_interval: Monitoring interval in seconds
        """
        logger.info(f"Starting email training monitoring for session: {training_id}")
        
        self.training_start_time = time.time()
        self.total_emails_processed = 0
        self.total_training_steps = 0
        
        # Start base monitoring
        self.base_monitor.start_monitoring(monitoring_interval)
        
        # Start email-specific monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._email_monitoring_loop,
            args=(monitoring_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Email training monitoring started")
    
    def stop_training_monitoring(self) -> TrainingPerformanceMetrics:
        """
        Stop monitoring and return performance summary.
        
        Returns:
            Training performance metrics
        """
        logger.info("Stopping email training monitoring")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.base_monitor.stop_monitoring()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        logger.info("Email training monitoring stopped")
        return performance_metrics
    
    def _email_monitoring_loop(self, interval: float) -> None:
        """Email-specific monitoring loop."""
        while self.monitoring_active:
            try:
                # Get base resource snapshot
                base_snapshot = self.base_monitor.get_current_snapshot()
                
                # Get email-specific metrics
                email_metrics = None
                if self.email_memory_manager:
                    email_metrics = self.email_memory_manager.get_email_memory_metrics()
                
                # Create email training snapshot
                training_snapshot = EmailTrainingSnapshot(
                    timestamp=base_snapshot.timestamp,
                    memory=base_snapshot.memory,
                    cpu=base_snapshot.cpu,
                    thermal=base_snapshot.thermal,
                    email_metrics=email_metrics,
                    emails_processed_total=self.total_emails_processed,
                    memory_efficiency=email_metrics.memory_efficiency if email_metrics else 0.0,
                    cpu_efficiency=base_snapshot.cpu.percent_total / 100.0,
                    thermal_efficiency=1.0 if base_snapshot.thermal.thermal_state == "normal" else 0.5
                )
                
                # Add to history
                self.training_history.append(training_snapshot)
                
                # Check for alerts
                self._check_resource_alerts(training_snapshot)
                
                # Call performance callbacks
                for callback in self.performance_callbacks:
                    try:
                        callback(training_snapshot)
                    except Exception as e:
                        logger.error(f"Error in performance callback: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in email monitoring loop: {e}")
                time.sleep(interval)
    
    def _check_resource_alerts(self, snapshot: EmailTrainingSnapshot) -> None:
        """Check for resource alerts and trigger notifications."""
        current_time = snapshot.timestamp
        
        # Memory alerts
        if snapshot.memory.percent_used > 85:
            self._create_alert(
                "critical", "memory", 
                f"Critical memory usage: {snapshot.memory.percent_used:.1f}%",
                snapshot.memory.percent_used, 85.0,
                "Reduce batch size or sequence length immediately",
                current_time
            )
        elif snapshot.memory.percent_used > 75:
            self._create_alert(
                "warning", "memory",
                f"High memory usage: {snapshot.memory.percent_used:.1f}%",
                snapshot.memory.percent_used, 75.0,
                "Consider reducing batch size or enabling dynamic batching",
                current_time
            )
        
        # CPU alerts
        if snapshot.cpu.percent_total > 95:
            self._create_alert(
                "warning", "cpu",
                f"Very high CPU usage: {snapshot.cpu.percent_total:.1f}%",
                snapshot.cpu.percent_total, 95.0,
                "System may be overloaded - consider reducing training intensity",
                current_time
            )
        
        # Thermal alerts
        if snapshot.thermal.thermal_state == "hot":
            self._create_alert(
                "warning", "thermal",
                "System thermal throttling detected",
                1.0, 0.5,
                "Reduce training intensity or improve cooling",
                current_time
            )
        
        # Performance alerts
        if snapshot.email_metrics:
            if snapshot.email_metrics.emails_per_second < 1.0:
                self._create_alert(
                    "warning", "performance",
                    f"Low email processing speed: {snapshot.email_metrics.emails_per_second:.2f} emails/sec",
                    snapshot.email_metrics.emails_per_second, 1.0,
                    "Check for preprocessing bottlenecks or memory pressure",
                    current_time
                )
            
            if snapshot.email_metrics.memory_efficiency < 0.3:
                self._create_alert(
                    "info", "performance",
                    f"Low memory efficiency: {snapshot.email_metrics.memory_efficiency:.1%}",
                    snapshot.email_metrics.memory_efficiency, 0.3,
                    "Consider increasing batch size or sequence length",
                    current_time
                )
    
    def _create_alert(self, 
                     severity: str, 
                     category: str, 
                     message: str,
                     current_value: float,
                     threshold_value: float,
                     suggested_action: str,
                     timestamp: float) -> None:
        """Create and process resource alert."""
        alert_key = f"{category}_{severity}"
        
        # Check cooldown
        if alert_key in self.last_alert_times:
            if timestamp - self.last_alert_times[alert_key] < self.alert_cooldown_seconds:
                return
        
        # Create alert
        alert = ResourceAlert(
            timestamp=timestamp,
            severity=severity,
            category=category,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            suggested_action=suggested_action
        )
        
        # Add to alerts list
        self.alerts.append(alert)
        if len(self.alerts) > 100:  # Keep only recent alerts
            self.alerts.pop(0)
        
        # Update cooldown
        self.last_alert_times[alert_key] = timestamp
        
        # Log alert
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        logger.log(log_level, f"Resource Alert [{severity.upper()}]: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def update_training_progress(self, 
                               step: int,
                               accuracy: Optional[float] = None,
                               loss: Optional[float] = None,
                               learning_rate: Optional[float] = None,
                               emails_processed: int = 0) -> None:
        """
        Update training progress metrics.
        
        Args:
            step: Current training step
            accuracy: Current accuracy
            loss: Current loss
            learning_rate: Current learning rate
            emails_processed: Number of emails processed in this update
        """
        self.total_training_steps = step
        self.total_emails_processed += emails_processed
        
        # Update latest snapshot with training info
        if self.training_history:
            latest_snapshot = self.training_history[-1]
            latest_snapshot.training_step = step
            latest_snapshot.current_accuracy = accuracy
            latest_snapshot.current_loss = loss
            latest_snapshot.learning_rate = learning_rate
            latest_snapshot.emails_processed_total = self.total_emails_processed
    
    def _calculate_performance_metrics(self) -> TrainingPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.training_history:
            return TrainingPerformanceMetrics(
                average_emails_per_second=0.0,
                peak_emails_per_second=0.0,
                training_efficiency=0.0,
                average_memory_usage=0.0,
                peak_memory_usage=0.0,
                average_cpu_usage=0.0,
                peak_cpu_usage=0.0,
                memory_pressure_events=0,
                thermal_throttling_events=0,
                performance_degradation_events=0,
                convergence_rate=0.0,
                training_stability=0.0,
                resource_stability=0.0
            )
        
        snapshots = list(self.training_history)
        
        # Calculate throughput metrics
        email_speeds = [s.email_metrics.emails_per_second for s in snapshots 
                       if s.email_metrics and s.email_metrics.emails_per_second > 0]
        
        average_emails_per_second = sum(email_speeds) / len(email_speeds) if email_speeds else 0.0
        peak_emails_per_second = max(email_speeds) if email_speeds else 0.0
        
        # Training efficiency (simplified)
        theoretical_max_speed = 50.0  # emails per second (estimate)
        training_efficiency = average_emails_per_second / theoretical_max_speed if theoretical_max_speed > 0 else 0.0
        
        # Resource utilization
        memory_usages = [s.memory.percent_used for s in snapshots]
        cpu_usages = [s.cpu.percent_total for s in snapshots]
        
        average_memory_usage = sum(memory_usages) / len(memory_usages)
        peak_memory_usage = max(memory_usages)
        average_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        peak_cpu_usage = max(cpu_usages)
        
        # Count events
        memory_pressure_events = sum(1 for s in snapshots if s.memory.percent_used > 80)
        thermal_throttling_events = sum(1 for s in snapshots if s.thermal.thermal_state == "hot")
        performance_degradation_events = sum(1 for s in snapshots 
                                           if s.email_metrics and s.email_metrics.emails_per_second < 1.0)
        
        # Training quality metrics (simplified)
        accuracies = [s.current_accuracy for s in snapshots if s.current_accuracy is not None]
        if len(accuracies) > 1:
            convergence_rate = (accuracies[-1] - accuracies[0]) / len(accuracies)
            training_stability = 1.0 - (max(accuracies) - min(accuracies))  # Simplified stability measure
        else:
            convergence_rate = 0.0
            training_stability = 1.0
        
        # Resource stability
        memory_variance = sum((x - average_memory_usage) ** 2 for x in memory_usages) / len(memory_usages)
        cpu_variance = sum((x - average_cpu_usage) ** 2 for x in cpu_usages) / len(cpu_usages)
        resource_stability = 1.0 / (1.0 + (memory_variance + cpu_variance) / 1000.0)  # Normalized stability
        
        return TrainingPerformanceMetrics(
            average_emails_per_second=average_emails_per_second,
            peak_emails_per_second=peak_emails_per_second,
            training_efficiency=training_efficiency,
            average_memory_usage=average_memory_usage,
            peak_memory_usage=peak_memory_usage,
            average_cpu_usage=average_cpu_usage,
            peak_cpu_usage=peak_cpu_usage,
            memory_pressure_events=memory_pressure_events,
            thermal_throttling_events=thermal_throttling_events,
            performance_degradation_events=performance_degradation_events,
            convergence_rate=convergence_rate,
            training_stability=training_stability,
            resource_stability=resource_stability
        )
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """
        Get real-time dashboard data for monitoring UI.
        
        Returns:
            Dashboard data dictionary
        """
        if not self.training_history:
            return {"status": "no_data", "message": "No monitoring data available"}
        
        latest_snapshot = self.training_history[-1]
        recent_snapshots = list(self.training_history)[-60:]  # Last 60 samples (1 minute at 1s interval)
        
        # Current status
        current_status = {
            "timestamp": latest_snapshot.timestamp,
            "memory_usage_percent": latest_snapshot.memory.percent_used,
            "memory_available_gb": latest_snapshot.memory.available_mb / 1024,
            "cpu_usage_percent": latest_snapshot.cpu.percent_total,
            "thermal_state": latest_snapshot.thermal.thermal_state,
            "training_step": latest_snapshot.training_step,
            "current_accuracy": latest_snapshot.current_accuracy,
            "current_loss": latest_snapshot.current_loss,
            "emails_processed": latest_snapshot.emails_processed_total
        }
        
        # Performance metrics
        if latest_snapshot.email_metrics:
            current_status.update({
                "emails_per_second": latest_snapshot.email_metrics.emails_per_second,
                "cache_hit_rate": latest_snapshot.email_metrics.cache_hit_rate,
                "memory_efficiency": latest_snapshot.email_metrics.memory_efficiency
            })
        
        # Historical data for charts
        historical_data = {
            "timestamps": [s.timestamp for s in recent_snapshots],
            "memory_usage": [s.memory.percent_used for s in recent_snapshots],
            "cpu_usage": [s.cpu.percent_total for s in recent_snapshots],
            "emails_per_second": [s.email_metrics.emails_per_second if s.email_metrics else 0 
                                 for s in recent_snapshots],
            "accuracy": [s.current_accuracy if s.current_accuracy else 0 for s in recent_snapshots],
            "loss": [s.current_loss if s.current_loss else 0 for s in recent_snapshots]
        }
        
        # Recent alerts
        recent_alerts = [
            {
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "category": alert.category,
                "message": alert.message,
                "suggested_action": alert.suggested_action
            }
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        # Training progress
        training_progress = {
            "total_steps": self.total_training_steps,
            "total_emails_processed": self.total_emails_processed,
            "training_time_minutes": (time.time() - self.training_start_time) / 60 if self.training_start_time else 0,
            "estimated_completion": self._estimate_completion_time()
        }
        
        return {
            "status": "active",
            "current_status": current_status,
            "historical_data": historical_data,
            "recent_alerts": recent_alerts,
            "training_progress": training_progress,
            "performance_summary": self._get_performance_summary()
        }
    
    def _estimate_completion_time(self) -> Optional[float]:
        """Estimate training completion time in minutes."""
        if not self.training_start_time or self.total_training_steps == 0:
            return None
        
        elapsed_time = time.time() - self.training_start_time
        steps_per_second = self.total_training_steps / elapsed_time
        
        # Assume 10000 total steps (this should be configurable)
        remaining_steps = max(0, 10000 - self.total_training_steps)
        estimated_remaining_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
        
        return estimated_remaining_seconds / 60  # Convert to minutes
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        if len(self.training_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.training_history)[-300:]  # Last 5 minutes
        
        # Calculate averages
        avg_memory = sum(s.memory.percent_used for s in recent_snapshots) / len(recent_snapshots)
        avg_cpu = sum(s.cpu.percent_total for s in recent_snapshots) / len(recent_snapshots)
        
        email_speeds = [s.email_metrics.emails_per_second for s in recent_snapshots 
                       if s.email_metrics and s.email_metrics.emails_per_second > 0]
        avg_email_speed = sum(email_speeds) / len(email_speeds) if email_speeds else 0
        
        # Performance rating (0-100)
        memory_score = max(0, 100 - abs(avg_memory - 70))  # Optimal around 70%
        cpu_score = max(0, 100 - max(0, avg_cpu - 80))     # Penalty above 80%
        speed_score = min(100, avg_email_speed * 10)       # 10 emails/sec = 100 score
        
        overall_score = (memory_score + cpu_score + speed_score) / 3
        
        return {
            "status": "active",
            "overall_performance_score": overall_score,
            "memory_efficiency_score": memory_score,
            "cpu_efficiency_score": cpu_score,
            "processing_speed_score": speed_score,
            "recent_averages": {
                "memory_usage": avg_memory,
                "cpu_usage": avg_cpu,
                "emails_per_second": avg_email_speed
            }
        }
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[EmailTrainingSnapshot], None]) -> None:
        """Add callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def export_monitoring_data(self, filepath: str) -> None:
        """
        Export monitoring data to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            export_data = {
                "training_session": {
                    "start_time": self.training_start_time,
                    "total_emails_processed": self.total_emails_processed,
                    "total_training_steps": self.total_training_steps
                },
                "performance_metrics": asdict(self._calculate_performance_metrics()),
                "alerts": [asdict(alert) for alert in self.alerts],
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "memory_percent": s.memory.percent_used,
                        "cpu_percent": s.cpu.percent_total,
                        "thermal_state": s.thermal.thermal_state,
                        "training_step": s.training_step,
                        "accuracy": s.current_accuracy,
                        "loss": s.current_loss,
                        "emails_per_second": s.email_metrics.emails_per_second if s.email_metrics else 0
                    }
                    for s in list(self.training_history)
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Monitoring data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        performance_metrics = self._calculate_performance_metrics()
        
        return {
            "session_info": {
                "start_time": self.training_start_time,
                "duration_minutes": (time.time() - self.training_start_time) / 60 if self.training_start_time else 0,
                "total_emails_processed": self.total_emails_processed,
                "total_training_steps": self.total_training_steps,
                "monitoring_active": self.monitoring_active
            },
            "performance_metrics": asdict(performance_metrics),
            "alert_summary": {
                "total_alerts": len(self.alerts),
                "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
                "warning_alerts": len([a for a in self.alerts if a.severity == "warning"]),
                "info_alerts": len([a for a in self.alerts if a.severity == "info"])
            },
            "current_status": self.get_real_time_dashboard_data()["current_status"] if self.training_history else {},
            "recommendations": self._get_monitoring_recommendations(performance_metrics)
        }
    
    def _get_monitoring_recommendations(self, metrics: TrainingPerformanceMetrics) -> List[str]:
        """Get monitoring-based recommendations."""
        recommendations = []
        
        if metrics.average_memory_usage > 85:
            recommendations.append("Consider reducing batch size due to high memory usage")
        
        if metrics.thermal_throttling_events > 10:
            recommendations.append("Frequent thermal throttling detected - improve cooling or reduce intensity")
        
        if metrics.training_efficiency < 0.3:
            recommendations.append("Low training efficiency - check for bottlenecks")
        
        if metrics.memory_pressure_events > 20:
            recommendations.append("Frequent memory pressure - enable dynamic batch sizing")
        
        if metrics.average_emails_per_second < 2:
            recommendations.append("Low email processing speed - optimize preprocessing pipeline")
        
        return recommendations