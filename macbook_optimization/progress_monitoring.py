"""
Progress monitoring module for MacBook TRM training.

This module provides real-time progress monitoring with MacBook-specific metrics,
training speed monitoring, and resource-aware ETA calculations.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
import math

from .resource_monitoring import ResourceMonitor, ResourceSnapshot
from .memory_management import MemoryManager


@dataclass
class TrainingProgress:
    """Training progress information."""
    current_step: int
    total_steps: int
    current_epoch: int
    total_epochs: int
    samples_processed: int
    total_samples: int
    progress_percent: float
    
    # Time tracking
    start_time: float
    elapsed_time: float
    estimated_total_time: float
    eta_seconds: float
    
    # Speed metrics
    current_samples_per_second: float
    average_samples_per_second: float
    recent_samples_per_second: float  # Last 10 measurements
    
    # Loss tracking
    current_loss: Optional[float] = None
    average_loss: Optional[float] = None
    best_loss: Optional[float] = None
    
    # Learning rate
    current_learning_rate: Optional[float] = None


@dataclass
class ResourceMetrics:
    """Resource usage metrics for training."""
    # Memory metrics
    memory_used_mb: float
    memory_available_mb: float
    memory_usage_percent: float
    memory_pressure_level: str  # "low", "medium", "high", "critical"
    
    # CPU metrics
    cpu_usage_percent: float
    cpu_frequency_mhz: float
    cpu_load_average: List[float]
    
    # Thermal metrics
    thermal_state: str  # "normal", "warm", "hot"
    
    # Training-specific metrics
    batch_processing_time_ms: float
    gradient_computation_time_ms: float
    data_loading_time_ms: float


@dataclass
class PerformanceMetrics:
    """Performance analysis metrics."""
    # Efficiency metrics
    memory_efficiency_percent: float  # How well memory is utilized
    cpu_efficiency_percent: float     # How well CPU is utilized
    training_efficiency_score: float  # Overall training efficiency (0-100)
    
    # Bottleneck analysis
    primary_bottleneck: str  # "memory", "cpu", "data_loading", "model_computation"
    bottleneck_severity: float  # 0-1 scale
    
    # Optimization suggestions
    optimization_suggestions: List[str]
    performance_warnings: List[str]


@dataclass
class TrainingSession:
    """Complete training session information."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Configuration
    model_name: str = ""
    dataset_name: str = ""
    batch_size: int = 0
    learning_rate: float = 0.0
    
    # Progress tracking
    progress_history: List[TrainingProgress] = field(default_factory=list)
    resource_history: List[ResourceMetrics] = field(default_factory=list)
    performance_history: List[PerformanceMetrics] = field(default_factory=list)
    
    # Summary statistics
    total_samples_processed: int = 0
    total_training_time: float = 0.0
    average_samples_per_second: float = 0.0
    peak_memory_usage_mb: float = 0.0
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None


class ProgressMonitor:
    """Real-time progress monitoring for MacBook TRM training."""
    
    def __init__(self, resource_monitor: ResourceMonitor, 
                 memory_manager: MemoryManager,
                 update_interval: float = 2.0,
                 history_size: int = 100):
        """
        Initialize progress monitor.
        
        Args:
            resource_monitor: Resource monitoring instance
            memory_manager: Memory management instance
            update_interval: Update interval in seconds
            history_size: Number of historical measurements to keep
        """
        self.resource_monitor = resource_monitor
        self.memory_manager = memory_manager
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Progress tracking
        self.step_history: deque = deque(maxlen=history_size)
        self.loss_history: deque = deque(maxlen=history_size)
        self.speed_history: deque = deque(maxlen=history_size)
        self.resource_history: deque = deque(maxlen=history_size)
        
        # Timing tracking
        self.last_step_time = 0.0
        self.last_step_count = 0
        self.batch_timing_history: deque = deque(maxlen=50)
        
        # Callbacks
        self.progress_callbacks: List[Callable[[TrainingProgress], None]] = []
        self.resource_callbacks: List[Callable[[ResourceMetrics], None]] = []
        self.performance_callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
    def start_session(self, session_id: str, total_steps: int, total_epochs: int = 1,
                     total_samples: int = 0, model_name: str = "", 
                     dataset_name: str = "", batch_size: int = 0,
                     learning_rate: float = 0.0) -> TrainingSession:
        """
        Start a new training session.
        
        Args:
            session_id: Unique session identifier
            total_steps: Total training steps
            total_epochs: Total training epochs
            total_samples: Total samples in dataset
            model_name: Name of the model being trained
            dataset_name: Name of the dataset
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Training session object
        """
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=time.time(),
            model_name=model_name,
            dataset_name=dataset_name,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Reset tracking
        self.step_history.clear()
        self.loss_history.clear()
        self.speed_history.clear()
        self.resource_history.clear()
        self.batch_timing_history.clear()
        
        self.last_step_time = time.time()
        self.last_step_count = 0
        
        # Start monitoring
        self.start_monitoring()
        
        print(f"Started training session: {session_id}")
        print(f"Model: {model_name}, Dataset: {dataset_name}")
        print(f"Total steps: {total_steps}, Batch size: {batch_size}")
        
        return self.current_session
    
    def end_session(self) -> Optional[TrainingSession]:
        """
        End the current training session.
        
        Returns:
            Completed training session or None
        """
        if self.current_session is None:
            return None
        
        self.stop_monitoring()
        
        self.current_session.end_time = time.time()
        self.current_session.total_training_time = (
            self.current_session.end_time - self.current_session.start_time
        )
        
        # Calculate final statistics
        if self.current_session.progress_history:
            final_progress = self.current_session.progress_history[-1]
            self.current_session.total_samples_processed = final_progress.samples_processed
            self.current_session.average_samples_per_second = final_progress.average_samples_per_second
            self.current_session.final_loss = final_progress.current_loss
        
        if self.current_session.resource_history:
            self.current_session.peak_memory_usage_mb = max(
                r.memory_used_mb for r in self.current_session.resource_history
            )
        
        if self.loss_history:
            self.current_session.best_loss = min(self.loss_history)
        
        completed_session = self.current_session
        self.current_session = None
        
        print(f"Training session completed: {completed_session.session_id}")
        print(f"Total time: {completed_session.total_training_time/60:.1f} minutes")
        print(f"Samples processed: {completed_session.total_samples_processed:,}")
        print(f"Average speed: {completed_session.average_samples_per_second:.1f} samples/s")
        
        return completed_session
    
    def update_progress(self, current_step: int, total_steps: int, 
                       current_epoch: int = 0, total_epochs: int = 1,
                       samples_processed: int = 0, total_samples: int = 0,
                       current_loss: Optional[float] = None,
                       current_learning_rate: Optional[float] = None,
                       batch_processing_time: Optional[float] = None):
        """
        Update training progress.
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
            current_epoch: Current epoch
            total_epochs: Total epochs
            samples_processed: Total samples processed
            total_samples: Total samples in dataset
            current_loss: Current loss value
            current_learning_rate: Current learning rate
            batch_processing_time: Time to process current batch (seconds)
        """
        if self.current_session is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.current_session.start_time
        
        # Calculate progress percentage
        progress_percent = (current_step / total_steps) * 100 if total_steps > 0 else 0
        
        # Calculate speed metrics
        time_since_last = current_time - self.last_step_time
        steps_since_last = current_step - self.last_step_count
        
        current_sps = 0.0
        if time_since_last > 0 and steps_since_last > 0:
            # Estimate samples per second based on batch processing
            if batch_processing_time and batch_processing_time > 0:
                current_sps = self.current_session.batch_size / batch_processing_time
            else:
                # Fallback: estimate from step timing
                steps_per_second = steps_since_last / time_since_last
                current_sps = steps_per_second * self.current_session.batch_size
        
        # Update speed history
        if current_sps > 0:
            self.speed_history.append(current_sps)
        
        # Calculate average speeds
        average_sps = samples_processed / elapsed_time if elapsed_time > 0 else 0
        recent_sps = sum(list(self.speed_history)[-10:]) / min(10, len(self.speed_history)) if self.speed_history else 0
        
        # Calculate ETA
        if current_step > 0 and average_sps > 0:
            remaining_samples = total_samples - samples_processed if total_samples > 0 else 0
            remaining_steps = total_steps - current_step
            
            if remaining_samples > 0:
                eta_seconds = remaining_samples / average_sps
            else:
                # Fallback to step-based estimation
                time_per_step = elapsed_time / current_step
                eta_seconds = remaining_steps * time_per_step
            
            estimated_total_time = elapsed_time + eta_seconds
        else:
            eta_seconds = 0
            estimated_total_time = 0
        
        # Update loss tracking
        if current_loss is not None:
            self.loss_history.append(current_loss)
        
        # Calculate loss statistics
        average_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else None
        best_loss = min(self.loss_history) if self.loss_history else None
        
        # Create progress object
        progress = TrainingProgress(
            current_step=current_step,
            total_steps=total_steps,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            samples_processed=samples_processed,
            total_samples=total_samples,
            progress_percent=progress_percent,
            start_time=self.current_session.start_time,
            elapsed_time=elapsed_time,
            estimated_total_time=estimated_total_time,
            eta_seconds=eta_seconds,
            current_samples_per_second=current_sps,
            average_samples_per_second=average_sps,
            recent_samples_per_second=recent_sps,
            current_loss=current_loss,
            average_loss=average_loss,
            best_loss=best_loss,
            current_learning_rate=current_learning_rate
        )
        
        # Store in session history
        self.current_session.progress_history.append(progress)
        
        # Update tracking state
        self.last_step_time = current_time
        self.last_step_count = current_step
        
        # Record batch timing if provided
        if batch_processing_time:
            self.batch_timing_history.append(batch_processing_time * 1000)  # Convert to ms
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                print(f"Error in progress callback: {e}")
    
    def get_current_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        # Get resource snapshot
        snapshot = self.resource_monitor.get_current_snapshot()
        
        # Get memory pressure info
        memory_info = self.memory_manager._analyze_memory_pressure(snapshot.memory)
        
        # Calculate timing metrics
        avg_batch_time = (
            sum(self.batch_timing_history) / len(self.batch_timing_history)
            if self.batch_timing_history else 0.0
        )
        
        # Estimate component times (simplified)
        gradient_time = avg_batch_time * 0.6  # Rough estimate
        data_loading_time = avg_batch_time * 0.1  # Rough estimate
        
        return ResourceMetrics(
            memory_used_mb=snapshot.memory.used_mb,
            memory_available_mb=snapshot.memory.available_mb,
            memory_usage_percent=snapshot.memory.percent_used,
            memory_pressure_level=memory_info.pressure_level,
            cpu_usage_percent=snapshot.cpu.percent_total,
            cpu_frequency_mhz=snapshot.cpu.frequency_current,
            cpu_load_average=snapshot.cpu.load_average,
            thermal_state=snapshot.thermal.thermal_state,
            batch_processing_time_ms=avg_batch_time,
            gradient_computation_time_ms=gradient_time,
            data_loading_time_ms=data_loading_time
        )
    
    def analyze_performance(self) -> PerformanceMetrics:
        """Analyze current training performance and identify bottlenecks."""
        resource_metrics = self.get_current_resource_metrics()
        
        # Calculate efficiency metrics
        memory_efficiency = min(100, resource_metrics.memory_usage_percent)
        cpu_efficiency = min(100, resource_metrics.cpu_usage_percent)
        
        # Calculate overall training efficiency
        # This is a simplified metric combining various factors
        speed_factor = 1.0
        if self.speed_history:
            recent_speed = sum(list(self.speed_history)[-5:]) / min(5, len(self.speed_history))
            # Normalize against a reasonable baseline (e.g., 100 samples/s for MacBook)
            speed_factor = min(1.0, recent_speed / 100.0)
        
        training_efficiency = (
            (memory_efficiency / 100) * 0.3 +
            (cpu_efficiency / 100) * 0.4 +
            speed_factor * 0.3
        ) * 100
        
        # Identify primary bottleneck
        bottlenecks = {
            "memory": resource_metrics.memory_usage_percent,
            "cpu": 100 - resource_metrics.cpu_usage_percent,  # Low CPU usage is a bottleneck
            "data_loading": resource_metrics.data_loading_time_ms / max(1, resource_metrics.batch_processing_time_ms) * 100,
            "model_computation": resource_metrics.gradient_computation_time_ms / max(1, resource_metrics.batch_processing_time_ms) * 100
        }
        
        primary_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])[0]
        bottleneck_severity = bottlenecks[primary_bottleneck] / 100
        
        # Generate optimization suggestions
        suggestions = []
        warnings = []
        
        if resource_metrics.memory_usage_percent > 85:
            suggestions.append("Reduce batch size to lower memory usage")
            warnings.append("High memory usage detected - risk of OOM errors")
        elif resource_metrics.memory_usage_percent < 50:
            suggestions.append("Consider increasing batch size to better utilize memory")
        
        if resource_metrics.cpu_usage_percent < 30:
            suggestions.append("CPU utilization is low - check for data loading bottlenecks")
        elif resource_metrics.cpu_usage_percent > 90:
            warnings.append("Very high CPU usage - may cause thermal throttling")
        
        if resource_metrics.thermal_state == "hot":
            warnings.append("High thermal state detected - performance may be throttled")
            suggestions.append("Consider reducing batch size or adding cooling breaks")
        
        if primary_bottleneck == "data_loading":
            suggestions.append("Optimize data loading with more workers or prefetching")
        elif primary_bottleneck == "memory":
            suggestions.append("Optimize memory usage with gradient accumulation or smaller batches")
        
        return PerformanceMetrics(
            memory_efficiency_percent=memory_efficiency,
            cpu_efficiency_percent=cpu_efficiency,
            training_efficiency_score=training_efficiency,
            primary_bottleneck=primary_bottleneck,
            bottleneck_severity=bottleneck_severity,
            optimization_suggestions=suggestions,
            performance_warnings=warnings
        )
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Internal monitoring loop."""
        while self.is_monitoring:
            try:
                if self.current_session:
                    # Collect resource metrics
                    resource_metrics = self.get_current_resource_metrics()
                    self.current_session.resource_history.append(resource_metrics)
                    self.resource_history.append(resource_metrics)
                    
                    # Analyze performance
                    performance_metrics = self.analyze_performance()
                    self.current_session.performance_history.append(performance_metrics)
                    
                    # Call callbacks
                    for callback in self.resource_callbacks:
                        try:
                            callback(resource_metrics)
                        except Exception as e:
                            print(f"Error in resource callback: {e}")
                    
                    for callback in self.performance_callbacks:
                        try:
                            callback(performance_metrics)
                        except Exception as e:
                            print(f"Error in performance callback: {e}")
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def add_progress_callback(self, callback: Callable[[TrainingProgress], None]):
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def add_resource_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Add callback for resource updates."""
        self.resource_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        if not self.current_session or not self.current_session.progress_history:
            return {}
        
        latest_progress = self.current_session.progress_history[-1]
        latest_resources = self.get_current_resource_metrics()
        performance = self.analyze_performance()
        
        return {
            "session": {
                "id": self.current_session.session_id,
                "model": self.current_session.model_name,
                "dataset": self.current_session.dataset_name,
                "elapsed_time_minutes": latest_progress.elapsed_time / 60,
            },
            "progress": {
                "step": f"{latest_progress.current_step}/{latest_progress.total_steps}",
                "epoch": f"{latest_progress.current_epoch}/{latest_progress.total_epochs}",
                "progress_percent": latest_progress.progress_percent,
                "eta_minutes": latest_progress.eta_seconds / 60,
                "samples_processed": latest_progress.samples_processed,
            },
            "performance": {
                "current_speed_sps": latest_progress.current_samples_per_second,
                "average_speed_sps": latest_progress.average_samples_per_second,
                "recent_speed_sps": latest_progress.recent_samples_per_second,
                "training_efficiency": performance.training_efficiency_score,
            },
            "resources": {
                "memory_usage_percent": latest_resources.memory_usage_percent,
                "memory_used_gb": latest_resources.memory_used_mb / 1024,
                "cpu_usage_percent": latest_resources.cpu_usage_percent,
                "thermal_state": latest_resources.thermal_state,
                "memory_pressure": latest_resources.memory_pressure_level,
            },
            "training": {
                "current_loss": latest_progress.current_loss,
                "average_loss": latest_progress.average_loss,
                "best_loss": latest_progress.best_loss,
                "learning_rate": latest_progress.current_learning_rate,
            },
            "analysis": {
                "primary_bottleneck": performance.primary_bottleneck,
                "optimization_suggestions": performance.optimization_suggestions,
                "warnings": performance.performance_warnings,
            }
        }
    
    def format_progress_display(self, compact: bool = False) -> str:
        """
        Format progress information for display.
        
        Args:
            compact: If True, return compact single-line format
            
        Returns:
            Formatted progress string
        """
        summary = self.get_progress_summary()
        if not summary:
            return "No active training session"
        
        progress = summary["progress"]
        performance = summary["performance"]
        resources = summary["resources"]
        training = summary["training"]
        
        if compact:
            return (
                f"Step {progress['step']} ({progress['progress_percent']:.1f}%) | "
                f"Loss: {training['current_loss']:.4f} | "
                f"Speed: {performance['current_speed_sps']:.1f} sps | "
                f"Memory: {resources['memory_usage_percent']:.1f}% | "
                f"ETA: {progress['eta_minutes']:.1f}min"
            )
        else:
            lines = [
                f"Training Progress - {summary['session']['model']}",
                f"Step: {progress['step']} ({progress['progress_percent']:.1f}%)",
                f"Epoch: {progress['epoch']}",
                f"Samples: {progress['samples_processed']:,}",
                f"",
                f"Performance:",
                f"  Current Speed: {performance['current_speed_sps']:.1f} samples/s",
                f"  Average Speed: {performance['average_speed_sps']:.1f} samples/s",
                f"  Training Efficiency: {performance['training_efficiency']:.1f}%",
                f"",
                f"Resources:",
                f"  Memory: {resources['memory_usage_percent']:.1f}% ({resources['memory_used_gb']:.1f}GB)",
                f"  CPU: {resources['cpu_usage_percent']:.1f}%",
                f"  Thermal: {resources['thermal_state']}",
                f"",
                f"Training:",
                f"  Current Loss: {training['current_loss']:.4f}",
                f"  Best Loss: {training['best_loss']:.4f}",
                f"  Learning Rate: {training['learning_rate']:.2e}",
                f"",
                f"ETA: {progress['eta_minutes']:.1f} minutes"
            ]
            
            # Add warnings if any
            warnings = summary["analysis"]["warnings"]
            if warnings:
                lines.extend(["", "⚠️  Warnings:"] + [f"  - {w}" for w in warnings])
            
            return "\n".join(lines)