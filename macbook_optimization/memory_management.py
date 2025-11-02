"""
Memory management module for MacBook TRM training optimization.

This module provides dynamic memory management including batch size calculation,
memory pressure monitoring, and automatic adjustments for efficient training
on memory-constrained MacBook hardware.
"""

import gc
import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .resource_monitoring import ResourceMonitor, MemoryStats


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    # Memory thresholds (percentages)
    memory_warning_threshold: float = 75.0
    memory_critical_threshold: float = 85.0
    memory_emergency_threshold: float = 95.0
    
    # Batch size management
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_reduction_factor: float = 0.75
    batch_size_increase_factor: float = 1.25
    
    # Memory estimation parameters
    model_memory_overhead: float = 1.5  # Multiplier for model memory usage
    gradient_memory_multiplier: float = 2.0  # Memory for gradients
    optimizer_memory_multiplier: float = 1.0  # Memory for optimizer states
    
    # Monitoring settings
    monitoring_interval: float = 1.0  # seconds
    memory_history_size: int = 60  # Keep 60 samples
    
    # Safety margins
    safety_margin_mb: float = 500.0  # Reserve 500MB for system


@dataclass
class BatchSizeRecommendation:
    """Batch size recommendation with reasoning."""
    recommended_batch_size: int
    max_safe_batch_size: int
    memory_utilization_percent: float
    reasoning: str
    warnings: List[str]


@dataclass
class MemoryPressureInfo:
    """Information about current memory pressure."""
    current_usage_percent: float
    available_mb: float
    pressure_level: str  # "low", "medium", "high", "critical"
    recommended_action: str
    time_to_critical: Optional[float]  # seconds until critical if trend continues


class MemoryManager:
    """Dynamic memory management for MacBook TRM training."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Memory management configuration
        """
        self.config = config or MemoryConfig()
        self.resource_monitor = ResourceMonitor(history_size=self.config.memory_history_size)
        
        # State tracking
        self.current_batch_size = 8  # Default starting batch size
        self.last_adjustment_time = 0.0
        self.adjustment_cooldown = 5.0  # seconds between adjustments
        self.memory_pressure_callbacks: List[Callable[[MemoryPressureInfo], None]] = []
        
        # Memory usage tracking
        self.baseline_memory_mb = 0.0
        self.model_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        
        # Start monitoring
        self.resource_monitor.start_monitoring(self.config.monitoring_interval)
        self.resource_monitor.add_callback(self._on_memory_update)
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
    
    def _on_memory_update(self, snapshot):
        """Handle memory monitoring updates."""
        memory_info = self._analyze_memory_pressure(snapshot.memory)
        
        # Call registered callbacks
        for callback in self.memory_pressure_callbacks:
            try:
                callback(memory_info)
            except Exception as e:
                print(f"Error in memory pressure callback: {e}")
        
        # Auto-adjust batch size if needed
        if self._should_adjust_batch_size(memory_info):
            self._auto_adjust_batch_size(memory_info)
    
    def add_memory_pressure_callback(self, callback: Callable[[MemoryPressureInfo], None]):
        """Add callback for memory pressure updates."""
        self.memory_pressure_callbacks.append(callback)
    
    def remove_memory_pressure_callback(self, callback: Callable[[MemoryPressureInfo], None]):
        """Remove memory pressure callback."""
        if callback in self.memory_pressure_callbacks:
            self.memory_pressure_callbacks.remove(callback)
    
    def estimate_model_memory_usage(self, model_params: int, sequence_length: int = 512, 
                                  dtype_size: int = 4) -> float:
        """
        Estimate memory usage for a model.
        
        Args:
            model_params: Number of model parameters
            sequence_length: Input sequence length
            dtype_size: Size of data type in bytes (4 for float32, 2 for float16)
            
        Returns:
            Estimated memory usage in MB
        """
        # Model parameters memory
        model_memory = (model_params * dtype_size) / (1024**2)
        
        # Activation memory (rough estimate based on sequence length)
        activation_memory = (sequence_length * model_params * dtype_size * 0.1) / (1024**2)
        
        # Apply overhead multiplier
        total_memory = (model_memory + activation_memory) * self.config.model_memory_overhead
        
        return total_memory
    
    def calculate_optimal_batch_size(self, model_params: int, sequence_length: int = 512,
                                   available_memory_mb: Optional[float] = None) -> BatchSizeRecommendation:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            model_params: Number of model parameters
            sequence_length: Input sequence length
            available_memory_mb: Available memory in MB (auto-detected if None)
            
        Returns:
            Batch size recommendation with details
        """
        if available_memory_mb is None:
            memory_stats = self.resource_monitor.get_memory_stats()
            available_memory_mb = memory_stats.available_mb - self.config.safety_margin_mb
        
        # Estimate memory per sample
        base_memory_per_sample = self.estimate_model_memory_usage(
            model_params, sequence_length
        )
        
        # Account for gradients and optimizer states
        memory_per_sample = base_memory_per_sample * (
            1.0 + 
            self.config.gradient_memory_multiplier + 
            self.config.optimizer_memory_multiplier
        )
        
        # Calculate maximum safe batch size
        max_safe_batch_size = max(1, int(available_memory_mb / memory_per_sample))
        max_safe_batch_size = min(max_safe_batch_size, self.config.max_batch_size)
        
        # Recommend a conservative batch size (80% of max safe)
        recommended_batch_size = max(
            self.config.min_batch_size,
            int(max_safe_batch_size * 0.8)
        )
        
        # Calculate memory utilization
        estimated_usage = recommended_batch_size * memory_per_sample
        memory_utilization = (estimated_usage / available_memory_mb) * 100
        
        # Generate reasoning and warnings
        reasoning = f"Based on {model_params:,} parameters and {sequence_length} sequence length"
        warnings = []
        
        if memory_utilization > 70:
            warnings.append("High memory utilization - consider reducing batch size")
        if max_safe_batch_size < 4:
            warnings.append("Very limited memory - training may be slow")
        if available_memory_mb < 1000:
            warnings.append("Less than 1GB available - consider closing other applications")
        
        return BatchSizeRecommendation(
            recommended_batch_size=recommended_batch_size,
            max_safe_batch_size=max_safe_batch_size,
            memory_utilization_percent=memory_utilization,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _analyze_memory_pressure(self, memory_stats: MemoryStats) -> MemoryPressureInfo:
        """Analyze current memory pressure level."""
        usage_percent = memory_stats.percent_used
        available_mb = memory_stats.available_mb
        
        # Determine pressure level
        if usage_percent >= self.config.memory_emergency_threshold:
            pressure_level = "critical"
            recommended_action = "Immediately reduce batch size or stop training"
        elif usage_percent >= self.config.memory_critical_threshold:
            pressure_level = "high"
            recommended_action = "Reduce batch size and trigger garbage collection"
        elif usage_percent >= self.config.memory_warning_threshold:
            pressure_level = "medium"
            recommended_action = "Consider reducing batch size"
        else:
            pressure_level = "low"
            recommended_action = "Normal operation"
        
        # Estimate time to critical (if trend continues)
        time_to_critical = None
        history = self.resource_monitor.get_history(last_n=10)
        if len(history) >= 3:
            # Calculate memory usage trend
            recent_usage = [s.memory.percent_used for s in history[-3:]]
            if len(set(recent_usage)) > 1:  # Avoid division by zero
                usage_trend = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
                if usage_trend > 0:
                    remaining_percent = self.config.memory_emergency_threshold - usage_percent
                    time_to_critical = remaining_percent / usage_trend * self.config.monitoring_interval
        
        return MemoryPressureInfo(
            current_usage_percent=usage_percent,
            available_mb=available_mb,
            pressure_level=pressure_level,
            recommended_action=recommended_action,
            time_to_critical=time_to_critical
        )
    
    def _should_adjust_batch_size(self, memory_info: MemoryPressureInfo) -> bool:
        """Determine if batch size should be automatically adjusted."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return False
        
        # Adjust if memory pressure is high or critical
        return memory_info.pressure_level in ["high", "critical"]
    
    def _auto_adjust_batch_size(self, memory_info: MemoryPressureInfo):
        """Automatically adjust batch size based on memory pressure."""
        if memory_info.pressure_level == "critical":
            # Emergency reduction
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.5)
            )
        elif memory_info.pressure_level == "high":
            # Standard reduction
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.batch_size_reduction_factor)
            )
        else:
            return  # No adjustment needed
        
        if new_batch_size != self.current_batch_size:
            print(f"Auto-adjusting batch size: {self.current_batch_size} -> {new_batch_size} "
                  f"(Memory pressure: {memory_info.pressure_level})")
            self.current_batch_size = new_batch_size
            self.last_adjustment_time = time.time()
            
            # Trigger garbage collection
            self.force_garbage_collection()
    
    def adjust_batch_size_dynamically(self, current_usage_percent: float) -> int:
        """
        Dynamically adjust batch size based on current memory usage.
        
        Args:
            current_usage_percent: Current memory usage percentage
            
        Returns:
            New recommended batch size
        """
        if current_usage_percent >= self.config.memory_critical_threshold:
            # Reduce batch size significantly
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.6)
            )
        elif current_usage_percent >= self.config.memory_warning_threshold:
            # Reduce batch size moderately
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.batch_size_reduction_factor)
            )
        elif current_usage_percent < 60 and self.current_batch_size < self.config.max_batch_size:
            # Increase batch size if memory usage is low
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.batch_size_increase_factor)
            )
        else:
            new_batch_size = self.current_batch_size
        
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def monitor_memory_usage(self) -> MemoryStats:
        """Get current memory usage statistics."""
        return self.resource_monitor.get_memory_stats()
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force memory cleanup
        if TORCH_AVAILABLE and hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
            # Clear MKL cache if available
            pass
    
    def get_memory_recommendations(self, model_params: int) -> Dict:
        """
        Get comprehensive memory management recommendations.
        
        Args:
            model_params: Number of model parameters
            
        Returns:
            Dictionary with memory recommendations
        """
        memory_stats = self.monitor_memory_usage()
        batch_rec = self.calculate_optimal_batch_size(model_params)
        memory_info = self._analyze_memory_pressure(memory_stats)
        
        return {
            "current_memory": {
                "used_percent": memory_stats.percent_used,
                "available_gb": round(memory_stats.available_mb / 1024, 2),
                "pressure_level": memory_info.pressure_level,
            },
            "batch_size": {
                "current": self.current_batch_size,
                "recommended": batch_rec.recommended_batch_size,
                "max_safe": batch_rec.max_safe_batch_size,
                "utilization_percent": batch_rec.memory_utilization_percent,
            },
            "recommendations": {
                "action": memory_info.recommended_action,
                "warnings": batch_rec.warnings,
                "reasoning": batch_rec.reasoning,
            },
            "monitoring": {
                "time_to_critical": memory_info.time_to_critical,
                "adjustment_cooldown_remaining": max(0, 
                    self.adjustment_cooldown - (time.time() - self.last_adjustment_time)
                ),
            }
        }
    
    def set_baseline_memory(self):
        """Set current memory usage as baseline."""
        memory_stats = self.monitor_memory_usage()
        self.baseline_memory_mb = memory_stats.used_mb
    
    def track_model_memory(self, model_memory_mb: float):
        """Track model memory usage."""
        self.model_memory_mb = model_memory_mb
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        memory_stats = self.monitor_memory_usage()
        self.peak_memory_mb = max(self.peak_memory_mb, memory_stats.used_mb)
    
    def get_memory_summary(self) -> Dict:
        """Get comprehensive memory usage summary."""
        current_stats = self.monitor_memory_usage()
        averages = self.resource_monitor.get_average_stats(last_n=10)
        
        return {
            "current": {
                "used_mb": current_stats.used_mb,
                "available_mb": current_stats.available_mb,
                "percent_used": current_stats.percent_used,
            },
            "tracking": {
                "baseline_mb": self.baseline_memory_mb,
                "model_mb": self.model_memory_mb,
                "peak_mb": self.peak_memory_mb,
                "training_overhead_mb": current_stats.used_mb - self.baseline_memory_mb,
            },
            "recent_averages": averages,
            "configuration": {
                "current_batch_size": self.current_batch_size,
                "min_batch_size": self.config.min_batch_size,
                "max_batch_size": self.config.max_batch_size,
                "memory_thresholds": {
                    "warning": self.config.memory_warning_threshold,
                    "critical": self.config.memory_critical_threshold,
                    "emergency": self.config.memory_emergency_threshold,
                }
            }
        }