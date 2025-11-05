"""
MacBook Training Pipeline for Real Email Classification

This module provides a complete training pipeline that integrates hardware detection,
memory management, CPU optimization, and resource monitoring for training EmailTRM
models on MacBook hardware with optimal performance.
"""

import os
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .hardware_detection import HardwareDetector, CPUSpecs, MemorySpecs, PlatformCapabilities
from .memory_management import MemoryManager, MemoryConfig, BatchSizeRecommendation
from .cpu_optimization import CPUOptimizer, TensorOperationOptimizer, CPUOptimizationConfig
from .resource_monitoring import ResourceMonitor, ResourceSnapshot
from .email_training_config import EmailTrainingConfig
from .email_training_orchestrator import EmailTrainingOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class MacBookHardwareSpecs:
    """Complete MacBook hardware specifications."""
    cpu: CPUSpecs
    memory: MemorySpecs
    platform: PlatformCapabilities
    
    # Derived specifications
    memory_tier: str  # "8GB", "16GB", "32GB+"
    performance_tier: str  # "basic", "standard", "high"
    optimization_profile: str  # "memory_constrained", "balanced", "performance"


@dataclass
class MacBookTrainingConfig:
    """MacBook-optimized training configuration."""
    # Hardware-adapted parameters
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    memory_limit_mb: float
    
    # CPU optimization settings
    torch_threads: int
    mkl_threads: int
    omp_threads: int
    
    # Memory management settings
    enable_dynamic_batching: bool
    memory_monitoring_interval: float
    memory_pressure_threshold: float
    
    # Training optimization settings
    gradient_checkpointing: bool
    mixed_precision: bool
    cpu_optimization_level: str  # "basic", "standard", "aggressive"
    
    # Email-specific optimizations
    sequence_length_optimization: bool
    email_structure_caching: bool
    hierarchical_attention_optimization: bool


@dataclass
class MacBookTrainingResult:
    """Result of MacBook-optimized training."""
    success: bool
    training_time_seconds: float
    final_accuracy: float
    peak_memory_usage_mb: float
    average_cpu_usage: float
    
    # Hardware utilization
    hardware_specs: MacBookHardwareSpecs
    training_config: MacBookTrainingConfig
    
    # Performance metrics
    samples_per_second: float
    memory_efficiency: float  # Percentage of available memory used
    cpu_efficiency: float     # Percentage of available CPU used
    
    # Optimization effectiveness
    dynamic_batch_adjustments: int
    memory_pressure_events: int
    thermal_throttling_events: int
    
    # Training quality metrics
    convergence_speed: float  # Steps to reach target accuracy
    stability_score: float    # Training stability measure
    
    errors: List[str]
    warnings: List[str]


class MacBookTrainingPipeline:
    """
    Complete MacBook training pipeline with hardware optimization.
    
    This class integrates all MacBook optimization components to provide
    a seamless training experience that automatically adapts to hardware
    constraints and maximizes performance.
    """
    
    def __init__(self, output_dir: str = "macbook_training_output"):
        """
        Initialize MacBook training pipeline.
        
        Args:
            output_dir: Directory for training outputs and logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize hardware detection
        self.hardware_detector = HardwareDetector()
        self.hardware_specs = None
        
        # Initialize optimization components
        self.cpu_optimizer = None
        self.tensor_optimizer = None
        self.memory_manager = None
        self.resource_monitor = None
        
        # Training orchestrator
        self.orchestrator = None
        
        # Configuration state
        self.is_configured = False
        self.training_config = None
        
        logger.info("MacBookTrainingPipeline initialized")
    
    def detect_and_configure_hardware(self) -> MacBookHardwareSpecs:
        """
        Detect MacBook hardware and configure optimization components.
        
        Returns:
            Complete hardware specifications
        """
        logger.info("Detecting MacBook hardware specifications...")
        
        # Detect hardware components
        cpu_specs = self.hardware_detector.detect_cpu_specs()
        memory_specs = self.hardware_detector.detect_memory_specs()
        platform_caps = self.hardware_detector.detect_platform_capabilities()
        
        # Determine hardware tiers
        memory_gb = memory_specs.total_memory / (1024**3)
        if memory_gb <= 10:
            memory_tier = "8GB"
            performance_tier = "basic"
            optimization_profile = "memory_constrained"
        elif memory_gb <= 18:
            memory_tier = "16GB"
            performance_tier = "standard"
            optimization_profile = "balanced"
        else:
            memory_tier = "32GB+"
            performance_tier = "high"
            optimization_profile = "performance"
        
        self.hardware_specs = MacBookHardwareSpecs(
            cpu=cpu_specs,
            memory=memory_specs,
            platform=platform_caps,
            memory_tier=memory_tier,
            performance_tier=performance_tier,
            optimization_profile=optimization_profile
        )
        
        logger.info(f"Hardware detected: {memory_tier} MacBook, {performance_tier} performance tier")
        logger.info(f"CPU: {cpu_specs.cores} cores, {cpu_specs.brand}")
        logger.info(f"Memory: {memory_gb:.1f}GB total, {memory_specs.available_memory / (1024**3):.1f}GB available")
        logger.info(f"Optimization profile: {optimization_profile}")
        
        return self.hardware_specs
    
    def configure_optimization_components(self) -> MacBookTrainingConfig:
        """
        Configure all optimization components based on detected hardware.
        
        Returns:
            MacBook-optimized training configuration
        """
        if self.hardware_specs is None:
            self.detect_and_configure_hardware()
        
        logger.info("Configuring optimization components...")
        
        # Initialize CPU optimizer
        self.cpu_optimizer = CPUOptimizer(self.hardware_detector)
        cpu_config = self.cpu_optimizer.configure_all()
        
        # Initialize tensor operation optimizer
        self.tensor_optimizer = TensorOperationOptimizer(self.cpu_optimizer)
        self.tensor_optimizer.optimize_tensor_operations()
        
        # Initialize memory manager with hardware-specific configuration
        memory_config = self._create_memory_config()
        self.memory_manager = MemoryManager(memory_config)
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(history_size=100)
        self.resource_monitor.start_monitoring(interval=1.0)
        
        # Create MacBook training configuration
        self.training_config = MacBookTrainingConfig(
            # Hardware-adapted parameters
            batch_size=self._calculate_optimal_batch_size(),
            gradient_accumulation_steps=self._calculate_gradient_accumulation(),
            num_workers=self.hardware_detector.get_optimal_worker_count(),
            memory_limit_mb=self._calculate_memory_limit(),
            
            # CPU optimization settings
            torch_threads=cpu_config.torch_threads,
            mkl_threads=cpu_config.mkl_threads,
            omp_threads=cpu_config.omp_threads,
            
            # Memory management settings
            enable_dynamic_batching=True,
            memory_monitoring_interval=1.0,
            memory_pressure_threshold=75.0 if self.hardware_specs.optimization_profile == "memory_constrained" else 80.0,
            
            # Training optimization settings
            gradient_checkpointing=self.hardware_specs.optimization_profile in ["memory_constrained", "balanced"],
            mixed_precision=False,  # CPU training doesn't benefit much from mixed precision
            cpu_optimization_level=self._determine_cpu_optimization_level(),
            
            # Email-specific optimizations
            sequence_length_optimization=True,
            email_structure_caching=self.hardware_specs.memory_tier != "8GB",
            hierarchical_attention_optimization=True
        )
        
        self.is_configured = True
        logger.info("Optimization components configured successfully")
        
        return self.training_config
    
    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration based on hardware specs."""
        if self.hardware_specs.optimization_profile == "memory_constrained":
            return MemoryConfig(
                memory_warning_threshold=70.0,
                memory_critical_threshold=80.0,
                memory_emergency_threshold=90.0,
                min_batch_size=1,
                max_batch_size=16,
                batch_size_reduction_factor=0.6,
                safety_margin_mb=300.0
            )
        elif self.hardware_specs.optimization_profile == "balanced":
            return MemoryConfig(
                memory_warning_threshold=75.0,
                memory_critical_threshold=85.0,
                memory_emergency_threshold=95.0,
                min_batch_size=2,
                max_batch_size=32,
                batch_size_reduction_factor=0.75,
                safety_margin_mb=500.0
            )
        else:  # performance
            return MemoryConfig(
                memory_warning_threshold=80.0,
                memory_critical_threshold=90.0,
                memory_emergency_threshold=95.0,
                min_batch_size=4,
                max_batch_size=64,
                batch_size_reduction_factor=0.8,
                safety_margin_mb=1000.0
            )
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on hardware."""
        if self.hardware_specs.memory_tier == "8GB":
            return 4
        elif self.hardware_specs.memory_tier == "16GB":
            return 8
        else:  # 32GB+
            return 16
    
    def _calculate_gradient_accumulation(self) -> int:
        """Calculate gradient accumulation steps based on hardware."""
        if self.hardware_specs.memory_tier == "8GB":
            return 16  # Effective batch size of 64
        elif self.hardware_specs.memory_tier == "16GB":
            return 8   # Effective batch size of 64
        else:  # 32GB+
            return 4   # Effective batch size of 64
    
    def _calculate_memory_limit(self) -> float:
        """Calculate memory limit for training."""
        available_mb = self.hardware_specs.memory.available_memory / (1024**2)
        
        if self.hardware_specs.optimization_profile == "memory_constrained":
            return available_mb * 0.6  # Use 60% of available memory
        elif self.hardware_specs.optimization_profile == "balanced":
            return available_mb * 0.7  # Use 70% of available memory
        else:  # performance
            return available_mb * 0.8  # Use 80% of available memory
    
    def _determine_cpu_optimization_level(self) -> str:
        """Determine CPU optimization level based on hardware."""
        if self.hardware_specs.cpu.cores <= 4:
            return "basic"
        elif self.hardware_specs.cpu.cores <= 8:
            return "standard"
        else:
            return "aggressive"
    
    def create_email_training_config(self, 
                                   dataset_size: int,
                                   target_accuracy: float = 0.95) -> EmailTrainingConfig:
        """
        Create EmailTrainingConfig optimized for MacBook hardware.
        
        Args:
            dataset_size: Size of training dataset
            target_accuracy: Target accuracy to achieve
            
        Returns:
            Hardware-optimized EmailTrainingConfig
        """
        if not self.is_configured:
            self.configure_optimization_components()
        
        # Calculate training parameters based on dataset size and hardware
        max_steps = min(10000, dataset_size // self.training_config.batch_size * 3)
        max_epochs = max(1, min(10, max_steps // (dataset_size // self.training_config.batch_size)))
        
        # Adjust sequence length based on memory constraints
        if self.hardware_specs.memory_tier == "8GB":
            max_sequence_length = 256
        elif self.hardware_specs.memory_tier == "16GB":
            max_sequence_length = 384
        else:
            max_sequence_length = 512
        
        email_config = EmailTrainingConfig(
            # Model parameters
            model_name="EmailTRM",
            vocab_size=5000,
            hidden_size=384 if self.hardware_specs.memory_tier != "8GB" else 256,
            num_layers=2,
            num_email_categories=10,
            
            # Training parameters
            batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=1e-4,
            weight_decay=0.01,
            max_epochs=max_epochs,
            max_steps=max_steps,
            
            # Email-specific parameters
            max_sequence_length=max_sequence_length,
            use_email_structure=True,
            use_hierarchical_attention=True,
            enable_subject_prioritization=True,
            subject_attention_weight=2.0,
            pooling_strategy="weighted",
            email_augmentation_prob=0.3,
            
            # MacBook optimization parameters
            memory_limit_mb=self.training_config.memory_limit_mb,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=self.training_config.enable_dynamic_batching,
            use_cpu_optimization=True,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            num_workers=self.training_config.num_workers,
            
            # Performance targets
            target_accuracy=target_accuracy,
            min_category_accuracy=0.90,
            early_stopping_patience=5
        )
        
        logger.info(f"Created EmailTrainingConfig optimized for {self.hardware_specs.memory_tier} MacBook")
        logger.info(f"Batch size: {email_config.batch_size}, Gradient accumulation: {email_config.gradient_accumulation_steps}")
        logger.info(f"Sequence length: {email_config.max_sequence_length}, Hidden size: {email_config.hidden_size}")
        
        return email_config
    
    def setup_training_orchestrator(self) -> EmailTrainingOrchestrator:
        """
        Setup training orchestrator with MacBook optimizations.
        
        Returns:
            Configured EmailTrainingOrchestrator
        """
        if not self.is_configured:
            self.configure_optimization_components()
        
        # Create orchestrator with our optimized components
        self.orchestrator = EmailTrainingOrchestrator(
            output_dir=str(self.output_dir),
            enable_monitoring=True,
            enable_checkpointing=True
        )
        
        # Replace orchestrator components with our optimized ones
        self.orchestrator.hardware_detector = self.hardware_detector
        self.orchestrator.memory_manager = self.memory_manager
        self.orchestrator.resource_monitor = self.resource_monitor
        
        logger.info("Training orchestrator configured with MacBook optimizations")
        
        return self.orchestrator
    
    def execute_optimized_training(self,
                                 dataset_path: str,
                                 strategy: str = "multi_phase",
                                 target_accuracy: float = 0.95) -> MacBookTrainingResult:
        """
        Execute complete optimized training pipeline.
        
        Args:
            dataset_path: Path to email dataset
            strategy: Training strategy
            target_accuracy: Target accuracy to achieve
            
        Returns:
            MacBook training result with performance metrics
        """
        logger.info("Starting MacBook-optimized email classification training")
        
        start_time = time.time()
        
        # Ensure everything is configured
        if not self.is_configured:
            self.configure_optimization_components()
        
        if self.orchestrator is None:
            self.setup_training_orchestrator()
        
        # Validate dataset and get size
        dataset_validation = self.orchestrator.dataset_manager.validate_email_dataset(dataset_path)
        if not dataset_validation.get("valid", False):
            return MacBookTrainingResult(
                success=False,
                training_time_seconds=0.0,
                final_accuracy=0.0,
                peak_memory_usage_mb=0.0,
                average_cpu_usage=0.0,
                hardware_specs=self.hardware_specs,
                training_config=self.training_config,
                samples_per_second=0.0,
                memory_efficiency=0.0,
                cpu_efficiency=0.0,
                dynamic_batch_adjustments=0,
                memory_pressure_events=0,
                thermal_throttling_events=0,
                convergence_speed=0.0,
                stability_score=0.0,
                errors=[f"Dataset validation failed: {dataset_validation.get('error', 'Unknown error')}"],
                warnings=[]
            )
        
        dataset_size = dataset_validation["total_emails"]
        
        # Create optimized training configuration
        email_config = self.create_email_training_config(dataset_size, target_accuracy)
        
        # Set up monitoring callbacks
        performance_metrics = {
            "memory_pressure_events": 0,
            "thermal_throttling_events": 0,
            "dynamic_batch_adjustments": 0,
            "peak_memory_mb": 0.0,
            "cpu_usage_samples": []
        }
        
        def resource_callback(snapshot: ResourceSnapshot):
            performance_metrics["peak_memory_mb"] = max(
                performance_metrics["peak_memory_mb"], 
                snapshot.memory.used_mb
            )
            performance_metrics["cpu_usage_samples"].append(snapshot.cpu.percent_total)
            
            if snapshot.memory.percent_used > self.training_config.memory_pressure_threshold:
                performance_metrics["memory_pressure_events"] += 1
            
            if snapshot.thermal.thermal_state == "hot":
                performance_metrics["thermal_throttling_events"] += 1
        
        self.resource_monitor.add_callback(resource_callback)
        
        def memory_pressure_callback(memory_info):
            if memory_info.pressure_level in ["high", "critical"]:
                performance_metrics["dynamic_batch_adjustments"] += 1
        
        self.memory_manager.add_memory_pressure_callback(memory_pressure_callback)
        
        try:
            # Execute training
            training_result = self.orchestrator.execute_training_pipeline(
                dataset_path=dataset_path,
                config=email_config,
                strategy=strategy,
                total_steps=email_config.max_steps
            )
            
            # Calculate performance metrics
            training_time = time.time() - start_time
            samples_processed = training_result.samples_processed or (dataset_size * email_config.max_epochs)
            samples_per_second = samples_processed / training_time if training_time > 0 else 0.0
            
            # Memory efficiency
            available_memory_mb = self.hardware_specs.memory.available_memory / (1024**2)
            memory_efficiency = (performance_metrics["peak_memory_mb"] / available_memory_mb) * 100
            
            # CPU efficiency
            average_cpu_usage = sum(performance_metrics["cpu_usage_samples"]) / len(performance_metrics["cpu_usage_samples"]) if performance_metrics["cpu_usage_samples"] else 0.0
            cpu_efficiency = (average_cpu_usage / 100.0) * 100  # Already in percentage
            
            # Convergence speed (simplified)
            convergence_speed = training_result.total_steps / max(1, training_result.final_accuracy or 0.01)
            
            # Stability score (simplified - based on lack of errors and warnings)
            stability_score = max(0.0, 1.0 - (len(training_result.errors) * 0.3 + len(training_result.warnings) * 0.1))
            
            return MacBookTrainingResult(
                success=training_result.success,
                training_time_seconds=training_time,
                final_accuracy=training_result.final_accuracy or 0.0,
                peak_memory_usage_mb=performance_metrics["peak_memory_mb"],
                average_cpu_usage=average_cpu_usage,
                hardware_specs=self.hardware_specs,
                training_config=self.training_config,
                samples_per_second=samples_per_second,
                memory_efficiency=memory_efficiency,
                cpu_efficiency=cpu_efficiency,
                dynamic_batch_adjustments=performance_metrics["dynamic_batch_adjustments"],
                memory_pressure_events=performance_metrics["memory_pressure_events"],
                thermal_throttling_events=performance_metrics["thermal_throttling_events"],
                convergence_speed=convergence_speed,
                stability_score=stability_score,
                errors=training_result.errors,
                warnings=training_result.warnings
            )
        
        except Exception as e:
            logger.error(f"MacBook training pipeline failed: {e}")
            return MacBookTrainingResult(
                success=False,
                training_time_seconds=time.time() - start_time,
                final_accuracy=0.0,
                peak_memory_usage_mb=performance_metrics["peak_memory_mb"],
                average_cpu_usage=sum(performance_metrics["cpu_usage_samples"]) / len(performance_metrics["cpu_usage_samples"]) if performance_metrics["cpu_usage_samples"] else 0.0,
                hardware_specs=self.hardware_specs,
                training_config=self.training_config,
                samples_per_second=0.0,
                memory_efficiency=0.0,
                cpu_efficiency=0.0,
                dynamic_batch_adjustments=performance_metrics["dynamic_batch_adjustments"],
                memory_pressure_events=performance_metrics["memory_pressure_events"],
                thermal_throttling_events=performance_metrics["thermal_throttling_events"],
                convergence_speed=0.0,
                stability_score=0.0,
                errors=[str(e)],
                warnings=[]
            )
        
        finally:
            # Clean up callbacks
            if resource_callback in self.resource_monitor.callbacks:
                self.resource_monitor.remove_callback(resource_callback)
            if memory_pressure_callback in self.memory_manager.memory_pressure_callbacks:
                self.memory_manager.remove_memory_pressure_callback(memory_pressure_callback)
    
    def validate_hardware_optimization(self) -> Dict[str, Any]:
        """
        Validate that hardware optimization is working correctly.
        
        Returns:
            Validation results
        """
        if not self.is_configured:
            self.configure_optimization_components()
        
        logger.info("Validating hardware optimization...")
        
        validation_results = {
            "success": True,
            "hardware_detection": {},
            "cpu_optimization": {},
            "memory_management": {},
            "resource_monitoring": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test hardware detection
            hardware_summary = self.hardware_detector.get_hardware_summary()
            validation_results["hardware_detection"] = {
                "cpu_cores": hardware_summary["cpu"]["cores"],
                "memory_gb": hardware_summary["memory"]["total_gb"],
                "platform": hardware_summary["platform"]["os"],
                "optimization_features": {
                    "mkl_available": hardware_summary["platform"]["has_mkl"],
                    "accelerate_available": hardware_summary["platform"]["has_accelerate"],
                    "avx_support": hardware_summary["platform"]["supports_avx"]
                }
            }
            
            # Test CPU optimization
            cpu_summary = self.cpu_optimizer.get_configuration_summary()
            validation_results["cpu_optimization"] = {
                "status": cpu_summary["status"],
                "torch_threads": cpu_summary.get("threading", {}).get("torch_threads", 0),
                "mkl_enabled": cpu_summary.get("optimizations", {}).get("mkl_enabled", False),
                "environment_configured": len(cpu_summary.get("environment_vars", {})) > 0
            }
            
            # Test tensor operations
            if TORCH_AVAILABLE:
                tensor_benchmark = self.tensor_optimizer.benchmark_tensor_operations()
                validation_results["cpu_optimization"]["tensor_benchmark"] = tensor_benchmark
            
            # Test memory management
            memory_recommendations = self.memory_manager.get_memory_recommendations(1000000)  # 1M parameters
            validation_results["memory_management"] = {
                "current_usage_percent": memory_recommendations["current_memory"]["used_percent"],
                "recommended_batch_size": memory_recommendations["batch_size"]["recommended"],
                "pressure_level": memory_recommendations["current_memory"]["pressure_level"],
                "monitoring_active": self.memory_manager.resource_monitor.monitoring
            }
            
            # Test resource monitoring
            resource_summary = self.resource_monitor.get_resource_summary()
            validation_results["resource_monitoring"] = {
                "monitoring_active": self.resource_monitor.monitoring,
                "current_memory_percent": resource_summary["current"]["memory_used_percent"],
                "current_cpu_percent": resource_summary["current"]["cpu_percent"],
                "thermal_state": resource_summary["current"]["thermal_state"]
            }
            
            # Check for potential issues
            if validation_results["memory_management"]["current_usage_percent"] > 80:
                validation_results["warnings"].append("High memory usage detected before training")
            
            if not validation_results["cpu_optimization"]["mkl_enabled"] and self.hardware_specs.platform.has_mkl:
                validation_results["warnings"].append("MKL available but not enabled")
            
            if validation_results["resource_monitoring"]["thermal_state"] == "hot":
                validation_results["warnings"].append("System thermal state is hot - may affect performance")
            
            logger.info("Hardware optimization validation completed successfully")
            
        except Exception as e:
            validation_results["success"] = False
            validation_results["errors"].append(str(e))
            logger.error(f"Hardware optimization validation failed: {e}")
        
        return validation_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of MacBook optimization configuration.
        
        Returns:
            Optimization summary
        """
        if not self.is_configured:
            return {"status": "not_configured", "message": "Pipeline not configured yet"}
        
        return {
            "status": "configured",
            "hardware_specs": {
                "memory_tier": self.hardware_specs.memory_tier,
                "performance_tier": self.hardware_specs.performance_tier,
                "optimization_profile": self.hardware_specs.optimization_profile,
                "cpu_cores": self.hardware_specs.cpu.cores,
                "memory_gb": round(self.hardware_specs.memory.total_memory / (1024**3), 1)
            },
            "training_config": {
                "batch_size": self.training_config.batch_size,
                "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
                "memory_limit_mb": self.training_config.memory_limit_mb,
                "cpu_optimization_level": self.training_config.cpu_optimization_level,
                "dynamic_batching_enabled": self.training_config.enable_dynamic_batching,
                "gradient_checkpointing": self.training_config.gradient_checkpointing
            },
            "optimization_components": {
                "cpu_optimizer_configured": self.cpu_optimizer is not None,
                "memory_manager_active": self.memory_manager is not None,
                "resource_monitoring_active": self.resource_monitor is not None and self.resource_monitor.monitoring,
                "tensor_optimizer_configured": self.tensor_optimizer is not None
            }
        }
    
    def cleanup(self):
        """Clean up resources and restore environment."""
        logger.info("Cleaning up MacBook training pipeline...")
        
        # Stop resource monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        # Restore CPU environment
        if self.cpu_optimizer:
            self.cpu_optimizer.restore_environment()
        
        # Clean up memory manager
        if self.memory_manager:
            self.memory_manager.force_garbage_collection()
        
        logger.info("MacBook training pipeline cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction