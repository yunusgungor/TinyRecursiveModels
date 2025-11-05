"""
Email-specific CPU Optimization for MacBook Training

This module provides specialized CPU optimization for email classification training,
including EmailTRM-specific optimizations, thermal management, and dynamic
performance scaling based on training workload.
"""

import os
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .cpu_optimization import CPUOptimizer, TensorOperationOptimizer, CPUOptimizationConfig
from .hardware_detection import HardwareDetector
from .resource_monitoring import ResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class EmailCPUConfig:
    """Email-specific CPU optimization configuration."""
    # Base CPU config
    base_config: CPUOptimizationConfig
    
    # Email-specific threading
    email_preprocessing_threads: int = 2
    tokenization_threads: int = 1
    batch_assembly_threads: int = 1
    
    # EmailTRM-specific optimizations
    recursive_reasoning_threads: int = 1  # Single-threaded for stability
    attention_computation_threads: int = 2
    classification_threads: int = 1
    
    # Dynamic scaling parameters
    enable_dynamic_scaling: bool = True
    cpu_usage_target: float = 80.0  # Target CPU usage percentage
    scaling_check_interval: float = 10.0  # Seconds between scaling checks
    
    # Thermal management
    enable_thermal_management: bool = True
    thermal_throttle_threshold: float = 85.0  # CPU usage to throttle at
    thermal_recovery_threshold: float = 70.0  # CPU usage to recover at
    
    # Performance tuning
    enable_email_specific_optimizations: bool = True
    optimize_for_sequence_processing: bool = True
    enable_batch_parallelization: bool = True


@dataclass
class EmailCPUMetrics:
    """Email training CPU performance metrics."""
    # Current utilization
    total_cpu_usage: float
    per_core_usage: List[float]
    email_processing_cpu: float
    model_training_cpu: float
    
    # Threading efficiency
    thread_utilization: Dict[str, float]
    thread_contention: float
    context_switches_per_second: float
    
    # Email-specific performance
    emails_processed_per_cpu_second: float
    batch_processing_efficiency: float
    recursive_reasoning_efficiency: float
    
    # Thermal status
    thermal_state: str
    cpu_frequency_mhz: float
    thermal_throttling_active: bool
    
    # Performance indicators
    cpu_efficiency_score: float  # 0-100
    bottleneck_indicators: List[str]
    optimization_opportunities: List[str]


class EmailCPUOptimizer:
    """
    Specialized CPU optimizer for email classification training.
    
    Provides email-specific CPU optimizations including dynamic scaling,
    thermal management, and EmailTRM-specific threading configurations.
    """
    
    def __init__(self, 
                 hardware_detector: Optional[HardwareDetector] = None,
                 enable_monitoring: bool = True):
        """
        Initialize email CPU optimizer.
        
        Args:
            hardware_detector: Hardware detector instance
            enable_monitoring: Enable CPU performance monitoring
        """
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.enable_monitoring = enable_monitoring
        
        # Initialize base optimizers
        self.base_optimizer = CPUOptimizer(self.hardware_detector)
        self.tensor_optimizer = TensorOperationOptimizer(self.base_optimizer)
        
        # Email-specific configuration
        self.email_config: Optional[EmailCPUConfig] = None
        self.is_configured = False
        
        # Performance monitoring
        self.resource_monitor: Optional[ResourceMonitor] = None
        if enable_monitoring:
            self.resource_monitor = ResourceMonitor(history_size=300)  # 5 minutes at 1s interval
        
        # Dynamic scaling state
        self.dynamic_scaling_active = False
        self.scaling_thread: Optional[threading.Thread] = None
        self.current_intensity_level = 1.0  # 0.0 to 1.0
        
        # Performance tracking
        self.cpu_metrics_history: List[EmailCPUMetrics] = []
        self.performance_callbacks: List[Callable[[EmailCPUMetrics], None]] = []
        
        logger.info("EmailCPUOptimizer initialized")
    
    def configure_email_cpu_optimization(self) -> EmailCPUConfig:
        """
        Configure CPU optimization specifically for email training.
        
        Returns:
            Email CPU configuration
        """
        logger.info("Configuring CPU optimization for email training...")
        
        # Get base CPU configuration
        base_config = self.base_optimizer.configure_all()
        
        # Detect hardware capabilities
        cpu_specs = self.hardware_detector.detect_cpu_specs()
        memory_specs = self.hardware_detector.detect_memory_specs()
        
        # Calculate email-specific thread allocation
        total_cores = cpu_specs.cores
        total_threads = cpu_specs.threads
        
        # Conservative threading for email processing
        email_preprocessing_threads = min(2, max(1, total_cores // 4))
        tokenization_threads = 1  # Keep simple to avoid contention
        batch_assembly_threads = 1
        
        # EmailTRM-specific threading
        recursive_reasoning_threads = 1  # Single-threaded for stability
        attention_computation_threads = min(2, max(1, total_cores // 2))
        classification_threads = 1
        
        # Dynamic scaling based on memory constraints
        memory_gb = memory_specs.total_memory / (1024**3)
        enable_dynamic_scaling = memory_gb >= 8  # Only enable on systems with sufficient memory
        
        # Thermal management based on CPU capabilities
        enable_thermal_management = True
        thermal_throttle_threshold = 85.0 if total_cores <= 4 else 90.0
        
        self.email_config = EmailCPUConfig(
            base_config=base_config,
            email_preprocessing_threads=email_preprocessing_threads,
            tokenization_threads=tokenization_threads,
            batch_assembly_threads=batch_assembly_threads,
            recursive_reasoning_threads=recursive_reasoning_threads,
            attention_computation_threads=attention_computation_threads,
            classification_threads=classification_threads,
            enable_dynamic_scaling=enable_dynamic_scaling,
            cpu_usage_target=75.0 if memory_gb < 16 else 80.0,
            enable_thermal_management=enable_thermal_management,
            thermal_throttle_threshold=thermal_throttle_threshold,
            enable_email_specific_optimizations=True,
            optimize_for_sequence_processing=True,
            enable_batch_parallelization=total_cores >= 4
        )
        
        # Apply email-specific optimizations
        self._apply_email_optimizations()
        
        self.is_configured = True
        logger.info("Email CPU optimization configured successfully")
        
        return self.email_config
    
    def _apply_email_optimizations(self) -> None:
        """Apply email-specific CPU optimizations."""
        if not self.email_config:
            return
        
        # Set email-specific environment variables
        email_env_vars = {
            'EMAIL_PREPROCESSING_THREADS': str(self.email_config.email_preprocessing_threads),
            'TOKENIZATION_THREADS': str(self.email_config.tokenization_threads),
            'BATCH_ASSEMBLY_THREADS': str(self.email_config.batch_assembly_threads),
            'RECURSIVE_REASONING_THREADS': str(self.email_config.recursive_reasoning_threads),
            'ATTENTION_THREADS': str(self.email_config.attention_computation_threads),
        }
        
        for var, value in email_env_vars.items():
            os.environ[var] = value
        
        # Configure PyTorch for email-specific workloads
        if TORCH_AVAILABLE and self.email_config.enable_email_specific_optimizations:
            # Optimize for sequence processing
            if self.email_config.optimize_for_sequence_processing:
                torch.backends.cudnn.benchmark = False  # Disable for variable sequence lengths
                
                # Enable optimizations for CPU inference
                if hasattr(torch.jit, 'set_fusion_strategy'):
                    torch.jit.set_fusion_strategy([('STATIC', 10), ('DYNAMIC', 10)])
            
            # Configure batch parallelization
            if self.email_config.enable_batch_parallelization:
                # Set intra-op parallelism for batch operations
                torch.set_num_interop_threads(max(1, self.email_config.base_config.torch_threads // 2))
        
        logger.info("Email-specific CPU optimizations applied")
    
    def start_dynamic_scaling(self) -> None:
        """Start dynamic CPU scaling based on workload."""
        if not self.email_config or not self.email_config.enable_dynamic_scaling:
            logger.info("Dynamic CPU scaling disabled")
            return
        
        if self.dynamic_scaling_active:
            logger.warning("Dynamic scaling already active")
            return
        
        logger.info("Starting dynamic CPU scaling")
        self.dynamic_scaling_active = True
        self.scaling_thread = threading.Thread(target=self._dynamic_scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        # Start resource monitoring if not already active
        if self.resource_monitor and not self.resource_monitor.monitoring:
            self.resource_monitor.start_monitoring(interval=1.0)
    
    def stop_dynamic_scaling(self) -> None:
        """Stop dynamic CPU scaling."""
        if not self.dynamic_scaling_active:
            return
        
        logger.info("Stopping dynamic CPU scaling")
        self.dynamic_scaling_active = False
        
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=2.0)
        
        # Reset intensity to normal
        self.current_intensity_level = 1.0
    
    def _dynamic_scaling_loop(self) -> None:
        """Dynamic scaling monitoring loop."""
        while self.dynamic_scaling_active:
            try:
                # Get current CPU metrics
                cpu_metrics = self.get_email_cpu_metrics()
                
                # Determine scaling action
                target_usage = self.email_config.cpu_usage_target
                current_usage = cpu_metrics.total_cpu_usage
                
                # Calculate desired intensity adjustment
                if current_usage > target_usage + 10:
                    # Reduce intensity
                    intensity_adjustment = -0.1
                elif current_usage < target_usage - 10:
                    # Increase intensity
                    intensity_adjustment = 0.1
                else:
                    intensity_adjustment = 0.0
                
                # Apply thermal throttling
                if cpu_metrics.thermal_throttling_active:
                    intensity_adjustment = min(intensity_adjustment, -0.2)
                
                # Update intensity level
                if intensity_adjustment != 0.0:
                    new_intensity = max(0.1, min(1.0, self.current_intensity_level + intensity_adjustment))
                    if abs(new_intensity - self.current_intensity_level) > 0.05:
                        self._apply_intensity_scaling(new_intensity)
                        self.current_intensity_level = new_intensity
                        logger.info(f"CPU intensity scaled to {new_intensity:.2f} "
                                   f"(CPU usage: {current_usage:.1f}%)")
                
                time.sleep(self.email_config.scaling_check_interval)
                
            except Exception as e:
                logger.error(f"Error in dynamic scaling loop: {e}")
                time.sleep(self.email_config.scaling_check_interval)
    
    def _apply_intensity_scaling(self, intensity: float) -> None:
        """
        Apply CPU intensity scaling.
        
        Args:
            intensity: Intensity level (0.0 to 1.0)
        """
        if not TORCH_AVAILABLE:
            return
        
        # Scale thread counts based on intensity
        base_threads = self.email_config.base_config.torch_threads
        scaled_threads = max(1, int(base_threads * intensity))
        
        # Apply thread scaling
        torch.set_num_threads(scaled_threads)
        
        # Scale other thread counts
        if intensity < 0.5:
            # Reduce email processing threads under high pressure
            os.environ['EMAIL_PREPROCESSING_THREADS'] = str(max(1, self.email_config.email_preprocessing_threads // 2))
            os.environ['ATTENTION_THREADS'] = str(max(1, self.email_config.attention_computation_threads // 2))
        else:
            # Restore normal thread counts
            os.environ['EMAIL_PREPROCESSING_THREADS'] = str(self.email_config.email_preprocessing_threads)
            os.environ['ATTENTION_THREADS'] = str(self.email_config.attention_computation_threads)
    
    def get_email_cpu_metrics(self) -> EmailCPUMetrics:
        """
        Get comprehensive email CPU metrics.
        
        Returns:
            Email CPU performance metrics
        """
        # Get base CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # Get process-specific CPU usage (simplified)
        current_process = psutil.Process()
        process_cpu = current_process.cpu_percent()
        
        # Estimate email processing vs model training CPU (simplified)
        email_processing_cpu = process_cpu * 0.3  # Rough estimate
        model_training_cpu = process_cpu * 0.7
        
        # Thread utilization (simplified - would need more detailed monitoring)
        thread_utilization = {
            "email_preprocessing": min(100.0, email_processing_cpu * 2),
            "tokenization": min(100.0, email_processing_cpu * 1.5),
            "model_training": min(100.0, model_training_cpu),
            "attention_computation": min(100.0, model_training_cpu * 0.6),
        }
        
        # Calculate thread contention (simplified)
        thread_contention = max(0.0, cpu_percent - 80.0) / 20.0  # 0-1 scale
        
        # Context switches (simplified estimate)
        context_switches_per_second = cpu_percent * 10  # Rough estimate
        
        # Email-specific performance metrics
        emails_per_cpu_second = max(0.0, (100.0 - cpu_percent) / 10.0)  # Inverse relationship
        batch_processing_efficiency = max(0.0, 1.0 - thread_contention)
        recursive_reasoning_efficiency = max(0.0, 1.0 - (cpu_percent - 70.0) / 30.0) if cpu_percent > 70 else 1.0
        
        # Thermal status
        thermal_state = "normal"
        thermal_throttling_active = False
        
        if self.resource_monitor:
            latest_snapshot = self.resource_monitor.get_current_snapshot()
            thermal_state = latest_snapshot.thermal.thermal_state
            thermal_throttling_active = thermal_state == "hot"
        
        # CPU efficiency score (0-100)
        efficiency_factors = [
            min(100.0, 100.0 - abs(cpu_percent - 75.0)),  # Optimal around 75%
            batch_processing_efficiency * 100,
            recursive_reasoning_efficiency * 100,
            100.0 if not thermal_throttling_active else 50.0
        ]
        cpu_efficiency_score = sum(efficiency_factors) / len(efficiency_factors)
        
        # Identify bottlenecks
        bottleneck_indicators = []
        if cpu_percent > 95:
            bottleneck_indicators.append("CPU overutilization")
        if thread_contention > 0.7:
            bottleneck_indicators.append("Thread contention")
        if thermal_throttling_active:
            bottleneck_indicators.append("Thermal throttling")
        if context_switches_per_second > 1000:
            bottleneck_indicators.append("High context switching")
        
        # Optimization opportunities
        optimization_opportunities = []
        if cpu_percent < 50:
            optimization_opportunities.append("Increase thread utilization")
        if batch_processing_efficiency < 0.7:
            optimization_opportunities.append("Optimize batch processing")
        if emails_per_cpu_second < 2:
            optimization_opportunities.append("Optimize email preprocessing")
        if not self.email_config or not self.email_config.enable_dynamic_scaling:
            optimization_opportunities.append("Enable dynamic CPU scaling")
        
        return EmailCPUMetrics(
            total_cpu_usage=cpu_percent,
            per_core_usage=cpu_per_core,
            email_processing_cpu=email_processing_cpu,
            model_training_cpu=model_training_cpu,
            thread_utilization=thread_utilization,
            thread_contention=thread_contention,
            context_switches_per_second=context_switches_per_second,
            emails_processed_per_cpu_second=emails_per_cpu_second,
            batch_processing_efficiency=batch_processing_efficiency,
            recursive_reasoning_efficiency=recursive_reasoning_efficiency,
            thermal_state=thermal_state,
            cpu_frequency_mhz=cpu_freq.current if cpu_freq else 0.0,
            thermal_throttling_active=thermal_throttling_active,
            cpu_efficiency_score=cpu_efficiency_score,
            bottleneck_indicators=bottleneck_indicators,
            optimization_opportunities=optimization_opportunities
        )
    
    def optimize_for_email_batch_size(self, batch_size: int) -> None:
        """
        Optimize CPU configuration for specific batch size.
        
        Args:
            batch_size: Training batch size
        """
        if not self.email_config:
            logger.warning("Email CPU config not initialized")
            return
        
        logger.info(f"Optimizing CPU configuration for batch size {batch_size}")
        
        # Adjust thread allocation based on batch size
        if batch_size <= 4:
            # Small batches - reduce parallelism to avoid overhead
            scale_factor = 0.7
        elif batch_size <= 16:
            # Medium batches - standard configuration
            scale_factor = 1.0
        else:
            # Large batches - increase parallelism
            scale_factor = 1.3
        
        # Apply scaling to thread counts
        if TORCH_AVAILABLE:
            base_threads = self.email_config.base_config.torch_threads
            scaled_threads = max(1, int(base_threads * scale_factor))
            torch.set_num_threads(scaled_threads)
            
            logger.info(f"Scaled PyTorch threads to {scaled_threads} for batch size {batch_size}")
        
        # Update environment variables
        scaled_preprocessing = max(1, int(self.email_config.email_preprocessing_threads * scale_factor))
        scaled_attention = max(1, int(self.email_config.attention_computation_threads * scale_factor))
        
        os.environ['EMAIL_PREPROCESSING_THREADS'] = str(scaled_preprocessing)
        os.environ['ATTENTION_THREADS'] = str(scaled_attention)
    
    def benchmark_email_cpu_performance(self, duration_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Benchmark CPU performance for email training workloads.
        
        Args:
            duration_seconds: Benchmark duration
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting {duration_seconds}s CPU benchmark for email training")
        
        benchmark_results = {
            "duration_seconds": duration_seconds,
            "cpu_metrics": [],
            "performance_summary": {},
            "optimization_recommendations": []
        }
        
        start_time = time.time()
        
        # Collect metrics during benchmark
        while time.time() - start_time < duration_seconds:
            metrics = self.get_email_cpu_metrics()
            benchmark_results["cpu_metrics"].append({
                "timestamp": time.time(),
                "cpu_usage": metrics.total_cpu_usage,
                "efficiency_score": metrics.cpu_efficiency_score,
                "emails_per_cpu_second": metrics.emails_processed_per_cpu_second,
                "thermal_throttling": metrics.thermal_throttling_active
            })
            
            time.sleep(1.0)
        
        # Calculate performance summary
        cpu_usages = [m["cpu_usage"] for m in benchmark_results["cpu_metrics"]]
        efficiency_scores = [m["efficiency_score"] for m in benchmark_results["cpu_metrics"]]
        email_speeds = [m["emails_per_cpu_second"] for m in benchmark_results["cpu_metrics"]]
        throttling_events = sum(1 for m in benchmark_results["cpu_metrics"] if m["thermal_throttling"])
        
        benchmark_results["performance_summary"] = {
            "average_cpu_usage": sum(cpu_usages) / len(cpu_usages),
            "peak_cpu_usage": max(cpu_usages),
            "average_efficiency_score": sum(efficiency_scores) / len(efficiency_scores),
            "average_email_processing_speed": sum(email_speeds) / len(email_speeds),
            "thermal_throttling_events": throttling_events,
            "performance_stability": 1.0 - (max(cpu_usages) - min(cpu_usages)) / 100.0
        }
        
        # Generate recommendations
        summary = benchmark_results["performance_summary"]
        recommendations = []
        
        if summary["average_cpu_usage"] < 60:
            recommendations.append("CPU underutilized - consider increasing thread counts or batch size")
        elif summary["average_cpu_usage"] > 90:
            recommendations.append("CPU overutilized - consider reducing thread counts or enabling dynamic scaling")
        
        if summary["average_efficiency_score"] < 70:
            recommendations.append("Low CPU efficiency - check for bottlenecks and optimize threading")
        
        if throttling_events > 5:
            recommendations.append("Frequent thermal throttling - improve cooling or reduce CPU intensity")
        
        if summary["performance_stability"] < 0.8:
            recommendations.append("Unstable CPU performance - enable dynamic scaling for better stability")
        
        benchmark_results["optimization_recommendations"] = recommendations
        
        logger.info("CPU benchmark completed")
        logger.info(f"Average CPU usage: {summary['average_cpu_usage']:.1f}%")
        logger.info(f"Average efficiency score: {summary['average_efficiency_score']:.1f}")
        
        return benchmark_results
    
    def add_performance_callback(self, callback: Callable[[EmailCPUMetrics], None]) -> None:
        """Add callback for CPU performance updates."""
        self.performance_callbacks.append(callback)
    
    def get_cpu_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive CPU optimization summary.
        
        Returns:
            CPU optimization summary
        """
        if not self.is_configured:
            return {"status": "not_configured", "message": "CPU optimization not configured"}
        
        current_metrics = self.get_email_cpu_metrics()
        
        return {
            "status": "configured",
            "configuration": {
                "email_preprocessing_threads": self.email_config.email_preprocessing_threads,
                "tokenization_threads": self.email_config.tokenization_threads,
                "recursive_reasoning_threads": self.email_config.recursive_reasoning_threads,
                "attention_computation_threads": self.email_config.attention_computation_threads,
                "dynamic_scaling_enabled": self.email_config.enable_dynamic_scaling,
                "thermal_management_enabled": self.email_config.enable_thermal_management,
                "current_intensity_level": self.current_intensity_level
            },
            "current_performance": {
                "cpu_usage_percent": current_metrics.total_cpu_usage,
                "efficiency_score": current_metrics.cpu_efficiency_score,
                "emails_per_cpu_second": current_metrics.emails_processed_per_cpu_second,
                "thermal_state": current_metrics.thermal_state,
                "thermal_throttling_active": current_metrics.thermal_throttling_active
            },
            "optimization_status": {
                "bottlenecks": current_metrics.bottleneck_indicators,
                "opportunities": current_metrics.optimization_opportunities,
                "dynamic_scaling_active": self.dynamic_scaling_active
            },
            "recommendations": self._get_cpu_recommendations(current_metrics)
        }
    
    def _get_cpu_recommendations(self, metrics: EmailCPUMetrics) -> List[str]:
        """Get CPU optimization recommendations."""
        recommendations = []
        
        if metrics.total_cpu_usage > 90:
            recommendations.append("High CPU usage - consider enabling dynamic scaling or reducing batch size")
        
        if metrics.thermal_throttling_active:
            recommendations.append("Thermal throttling active - improve cooling or reduce CPU intensity")
        
        if metrics.cpu_efficiency_score < 60:
            recommendations.append("Low CPU efficiency - optimize threading configuration")
        
        if metrics.thread_contention > 0.8:
            recommendations.append("High thread contention - reduce thread counts or optimize workload distribution")
        
        if metrics.emails_processed_per_cpu_second < 1:
            recommendations.append("Low email processing speed - optimize preprocessing pipeline")
        
        if not self.dynamic_scaling_active and self.email_config and self.email_config.enable_dynamic_scaling:
            recommendations.append("Dynamic scaling available but not active - consider enabling for better performance")
        
        return recommendations
    
    def cleanup(self) -> None:
        """Clean up CPU optimizer resources."""
        logger.info("Cleaning up EmailCPUOptimizer...")
        
        # Stop dynamic scaling
        self.stop_dynamic_scaling()
        
        # Stop resource monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        # Restore base CPU configuration
        if self.base_optimizer:
            self.base_optimizer.restore_environment()
        
        logger.info("EmailCPUOptimizer cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction