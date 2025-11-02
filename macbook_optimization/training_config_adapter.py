"""
Training configuration adapter for MacBook TRM training optimization.

This module provides the TrainingConfigAdapter class that adapts TRM training
configurations for MacBook hardware constraints, including model parameter
adjustment and hardware-appropriate parameter calculation.
"""

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .hardware_detection import HardwareDetector, CPUSpecs, MemorySpecs, PlatformCapabilities
from .memory_management import MemoryManager, MemoryConfig
from .cpu_optimization import CPUOptimizer
from .config_management import MacBookTrainingConfig, MacBookConfigManager

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpecs:
    """Combined hardware specifications for configuration adaptation."""
    cpu: CPUSpecs
    memory: MemorySpecs
    platform: PlatformCapabilities
    optimal_workers: int
    hardware_summary: Dict[str, Any]


@dataclass
class TrainingParams:
    """Training parameters adapted for hardware constraints."""
    # Core training parameters
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    
    # Hardware-specific parameters
    num_workers: int
    pin_memory: bool
    torch_threads: int
    
    # Memory management
    memory_limit_mb: int
    enable_memory_monitoring: bool
    dynamic_batch_sizing: bool
    
    # Model parameters
    max_sequence_length: int
    model_complexity_factor: float  # 1.0 = full model, <1.0 = reduced
    
    # Optimization flags
    use_mkl: bool
    enable_cpu_optimization: bool
    enable_mixed_precision: bool
    
    # Checkpointing
    checkpoint_interval: int
    max_checkpoints_to_keep: int


@dataclass
class ConfigurationResult:
    """Result of configuration adaptation with validation info."""
    adapted_config: Dict[str, Any]
    training_params: TrainingParams
    hardware_specs: HardwareSpecs
    validation_warnings: List[str]
    performance_estimates: Dict[str, Any]
    reasoning: str


class TrainingConfigAdapter:
    """Adapter for TRM training configurations on MacBook hardware."""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        """
        Initialize training configuration adapter.
        
        Args:
            hardware_detector: Hardware detector instance (created if None)
        """
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.memory_manager = MemoryManager()
        self.cpu_optimizer = CPUOptimizer(self.hardware_detector)
        self.config_manager = MacBookConfigManager()
        
        # Cache hardware specs
        self._hardware_specs = None
        
    def get_hardware_specs(self) -> HardwareSpecs:
        """Get comprehensive hardware specifications."""
        if self._hardware_specs is None:
            cpu_specs = self.hardware_detector.detect_cpu_specs()
            memory_specs = self.hardware_detector.detect_memory_specs()
            platform_caps = self.hardware_detector.detect_platform_capabilities()
            optimal_workers = self.hardware_detector.get_optimal_worker_count()
            hardware_summary = self.hardware_detector.get_hardware_summary()
            
            self._hardware_specs = HardwareSpecs(
                cpu=cpu_specs,
                memory=memory_specs,
                platform=platform_caps,
                optimal_workers=optimal_workers,
                hardware_summary=hardware_summary
            )
            
        return self._hardware_specs
    
    def adapt_model_config(self, base_config: Dict[str, Any], 
                          hardware_specs: Optional[HardwareSpecs] = None) -> Dict[str, Any]:
        """
        Adapt model configuration for MacBook hardware constraints.
        
        Args:
            base_config: Base TRM training configuration
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Adapted configuration dictionary
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
            
        adapted_config = base_config.copy()
        
        # Adapt batch size based on memory constraints
        model_params = base_config.get('model_params', 7_000_000)  # 7M default
        sequence_length = base_config.get('seq_len', 512)
        
        batch_recommendation = self.memory_manager.calculate_optimal_batch_size(
            model_params=model_params,
            sequence_length=sequence_length
        )
        
        # Override batch size with hardware-appropriate value
        adapted_config['global_batch_size'] = batch_recommendation.recommended_batch_size
        
        # Calculate gradient accumulation to maintain effective batch size
        target_effective_batch = base_config.get('global_batch_size', 64)
        gradient_accumulation_steps = max(1, target_effective_batch // batch_recommendation.recommended_batch_size)
        
        # Add gradient accumulation to config if not present
        if 'gradient_accumulation_steps' not in adapted_config:
            adapted_config['gradient_accumulation_steps'] = gradient_accumulation_steps
        
        # Adjust learning rate for smaller batch sizes
        if 'lr' in adapted_config:
            original_lr = adapted_config['lr']
            effective_batch_size = batch_recommendation.recommended_batch_size * gradient_accumulation_steps
            original_batch_size = base_config.get('global_batch_size', 64)
            
            # Square root scaling for learning rate
            lr_scale = math.sqrt(effective_batch_size / original_batch_size)
            adapted_config['lr'] = original_lr * lr_scale
            
            logger.info(f"Adjusted learning rate: {original_lr} -> {adapted_config['lr']} "
                       f"(scale factor: {lr_scale:.3f})")
        
        # Adjust model complexity if memory is very constrained
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        if memory_gb < 6:  # Less than 6GB available
            complexity_factor = min(1.0, memory_gb / 6.0)
            
            # Reduce model dimensions if needed
            if 'model_dim' in adapted_config:
                original_dim = adapted_config['model_dim']
                adapted_config['model_dim'] = int(original_dim * complexity_factor)
                logger.warning(f"Reduced model dimension due to memory constraints: "
                             f"{original_dim} -> {adapted_config['model_dim']}")
            
            # Reduce number of layers if needed
            if 'num_layers' in adapted_config:
                original_layers = adapted_config['num_layers']
                adapted_config['num_layers'] = max(1, int(original_layers * complexity_factor))
                logger.warning(f"Reduced number of layers due to memory constraints: "
                             f"{original_layers} -> {adapted_config['num_layers']}")
        
        # Set CPU-specific optimizations
        adapted_config['device'] = 'cpu'  # Force CPU training
        adapted_config['num_workers'] = hardware_specs.optimal_workers
        adapted_config['pin_memory'] = False  # Usually better on memory-constrained systems
        
        # Add MacBook-specific training parameters
        adapted_config['macbook_optimization'] = {
            'enable_memory_monitoring': True,
            'enable_cpu_optimization': True,
            'use_mkl': hardware_specs.platform.has_mkl,
            'torch_threads': min(hardware_specs.cpu.cores, 4),
            'memory_limit_mb': int(hardware_specs.memory.available_memory / (1024**2) * 0.8),
            'dynamic_batch_sizing': True,
            'checkpoint_interval': 500 if memory_gb < 12 else 1000,
        }
        
        return adapted_config
    
    def calculate_training_parameters(self, dataset_size: int, 
                                    hardware_specs: Optional[HardwareSpecs] = None) -> TrainingParams:
        """
        Calculate training parameters based on dataset size and hardware.
        
        Args:
            dataset_size: Size of training dataset
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Calculated training parameters
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
        
        # Memory calculations
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        usable_memory_mb = max(1000, (hardware_specs.memory.available_memory / (1024**2)) - 2048)
        
        # Model memory estimation (7M parameters)
        model_params = 7_000_000
        sequence_length = 512
        
        batch_recommendation = self.memory_manager.calculate_optimal_batch_size(
            model_params=model_params,
            sequence_length=sequence_length,
            available_memory_mb=usable_memory_mb
        )
        
        batch_size = batch_recommendation.recommended_batch_size
        
        # Calculate gradient accumulation for effective batch size
        target_effective_batch = 64
        gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        # Learning rate scaling
        base_lr = 1e-4
        lr_scale = math.sqrt(effective_batch_size / 64)
        learning_rate = base_lr * lr_scale
        
        # Training steps calculation
        steps_per_epoch = max(1, dataset_size // effective_batch_size)
        warmup_steps = min(100, steps_per_epoch // 10)
        
        # Model complexity adjustment
        complexity_factor = 1.0
        if memory_gb < 6:
            complexity_factor = min(1.0, memory_gb / 6.0)
        
        # Sequence length adjustment for memory constraints
        max_seq_length = sequence_length
        if memory_gb < 8:
            max_seq_length = min(sequence_length, 256)
        
        # CPU optimization settings
        cpu_config = self.cpu_optimizer.create_optimization_config()
        
        return TrainingParams(
            # Core training parameters
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            effective_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            
            # Hardware-specific parameters
            num_workers=hardware_specs.optimal_workers,
            pin_memory=False,
            torch_threads=cpu_config.torch_threads,
            
            # Memory management
            memory_limit_mb=int(usable_memory_mb * 0.9),
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            
            # Model parameters
            max_sequence_length=max_seq_length,
            model_complexity_factor=complexity_factor,
            
            # Optimization flags
            use_mkl=hardware_specs.platform.has_mkl,
            enable_cpu_optimization=True,
            enable_mixed_precision=False,  # Not recommended for CPU
            
            # Checkpointing
            checkpoint_interval=500 if memory_gb < 12 else 1000,
            max_checkpoints_to_keep=3
        )
    
    def create_hardware_appropriate_config(self, base_config: Dict[str, Any],
                                         dataset_size: int) -> ConfigurationResult:
        """
        Create complete hardware-appropriate configuration.
        
        Args:
            base_config: Base training configuration
            dataset_size: Size of training dataset
            
        Returns:
            Complete configuration result with validation
        """
        hardware_specs = self.get_hardware_specs()
        
        # Adapt model configuration
        adapted_config = self.adapt_model_config(base_config, hardware_specs)
        
        # Calculate training parameters
        training_params = self.calculate_training_parameters(dataset_size, hardware_specs)
        
        # Update adapted config with calculated parameters
        adapted_config.update({
            'global_batch_size': training_params.batch_size,
            'lr': training_params.learning_rate,
            'weight_decay': training_params.weight_decay,
            'lr_warmup_steps': training_params.warmup_steps,
            'num_workers': training_params.num_workers,
            'pin_memory': training_params.pin_memory,
        })
        
        # Generate validation warnings
        warnings = []
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        if memory_gb < 8:
            warnings.append(f"Limited memory ({memory_gb:.1f}GB) may impact training performance")
        
        if training_params.batch_size < 4:
            warnings.append(f"Very small batch size ({training_params.batch_size}) may slow convergence")
        
        if training_params.model_complexity_factor < 1.0:
            warnings.append(f"Model complexity reduced to {training_params.model_complexity_factor:.2f} due to memory constraints")
        
        if hardware_specs.cpu.cores < 4:
            warnings.append(f"Limited CPU cores ({hardware_specs.cpu.cores}) may impact training speed")
        
        # Performance estimates
        estimated_samples_per_second = self._estimate_training_speed(training_params, hardware_specs)
        estimated_epoch_time = dataset_size / estimated_samples_per_second if estimated_samples_per_second > 0 else 0
        
        performance_estimates = {
            'estimated_samples_per_second': estimated_samples_per_second,
            'estimated_epoch_time_minutes': estimated_epoch_time / 60,
            'memory_utilization_percent': training_params.batch_size * 10 / training_params.memory_limit_mb * 100,
            'cpu_utilization_percent': min(100, training_params.torch_threads / hardware_specs.cpu.cores * 100)
        }
        
        # Generate reasoning
        reasoning = self._generate_configuration_reasoning(training_params, hardware_specs, warnings)
        
        return ConfigurationResult(
            adapted_config=adapted_config,
            training_params=training_params,
            hardware_specs=hardware_specs,
            validation_warnings=warnings,
            performance_estimates=performance_estimates,
            reasoning=reasoning
        )
    
    def _estimate_training_speed(self, params: TrainingParams, 
                               hardware_specs: HardwareSpecs) -> float:
        """
        Estimate training speed in samples per second.
        
        Args:
            params: Training parameters
            hardware_specs: Hardware specifications
            
        Returns:
            Estimated samples per second
        """
        # Base speed estimate for 7M parameter model on CPU
        base_speed = 10.0  # samples/second baseline
        
        # Adjust for CPU performance
        cpu_factor = min(2.0, hardware_specs.cpu.base_frequency / 2.4)  # Normalize to 2.4GHz
        core_factor = min(1.5, hardware_specs.cpu.cores / 4)  # Normalize to 4 cores
        
        # Adjust for memory constraints
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        memory_factor = min(1.2, memory_gb / 8)  # Normalize to 8GB
        
        # Adjust for batch size (larger batches are more efficient)
        batch_factor = min(1.3, params.batch_size / 8)  # Normalize to batch size 8
        
        # Adjust for optimizations
        opt_factor = 1.0
        if params.use_mkl:
            opt_factor *= 1.2
        if hardware_specs.platform.supports_avx2:
            opt_factor *= 1.1
        
        estimated_speed = base_speed * cpu_factor * core_factor * memory_factor * batch_factor * opt_factor
        
        return max(0.1, estimated_speed)  # Minimum 0.1 samples/second
    
    def _generate_configuration_reasoning(self, params: TrainingParams,
                                        hardware_specs: HardwareSpecs,
                                        warnings: List[str]) -> str:
        """Generate human-readable reasoning for configuration choices."""
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        reasoning_parts = [
            f"Configuration adapted for {hardware_specs.cpu.brand} with {hardware_specs.cpu.cores} cores and {memory_gb:.1f}GB available memory.",
            f"Batch size set to {params.batch_size} with {params.gradient_accumulation_steps} gradient accumulation steps for effective batch size of {params.effective_batch_size}.",
            f"Learning rate scaled to {params.learning_rate:.2e} based on effective batch size.",
        ]
        
        if params.model_complexity_factor < 1.0:
            reasoning_parts.append(f"Model complexity reduced by {(1-params.model_complexity_factor)*100:.1f}% due to memory constraints.")
        
        if params.use_mkl:
            reasoning_parts.append("Intel MKL optimization enabled for improved CPU performance.")
        
        if warnings:
            reasoning_parts.append(f"Note: {len(warnings)} performance warnings generated.")
        
        return " ".join(reasoning_parts)
    
    def get_configuration_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get pre-defined configuration templates for common MacBook configurations.
        
        Returns:
            Dictionary of configuration templates
        """
        templates = {}
        
        # Template for 8GB MacBook
        templates['macbook_8gb'] = {
            'description': 'Optimized for MacBook with 8GB RAM',
            'config': self.config_manager.create_config_template(memory_gb=8, cpu_cores=4),
            'recommended_for': ['MacBook Air 2017-2020', 'MacBook Pro 13" base model']
        }
        
        # Template for 16GB MacBook
        templates['macbook_16gb'] = {
            'description': 'Optimized for MacBook with 16GB RAM',
            'config': self.config_manager.create_config_template(memory_gb=16, cpu_cores=4),
            'recommended_for': ['MacBook Pro 13" upgraded', 'MacBook Pro 16" base model']
        }
        
        # Template for high-end MacBook
        templates['macbook_32gb'] = {
            'description': 'Optimized for MacBook with 32GB+ RAM',
            'config': self.config_manager.create_config_template(memory_gb=32, cpu_cores=8),
            'recommended_for': ['MacBook Pro 16" high-end', 'Mac Studio']
        }
        
        # Conservative template for older/slower MacBooks
        templates['macbook_conservative'] = {
            'description': 'Conservative settings for older or slower MacBooks',
            'config': self.config_manager.create_config_template(memory_gb=6, cpu_cores=2),
            'recommended_for': ['Older MacBook models', 'Systems with limited resources']
        }
        
        return templates
    
    def recommend_configuration_template(self, hardware_specs: Optional[HardwareSpecs] = None) -> str:
        """
        Recommend the best configuration template for current hardware.
        
        Args:
            hardware_specs: Hardware specifications (auto-detected if None)
            
        Returns:
            Recommended template name
        """
        if hardware_specs is None:
            hardware_specs = self.get_hardware_specs()
        
        memory_gb = hardware_specs.memory.total_memory / (1024**3)
        cpu_cores = hardware_specs.cpu.cores
        
        if memory_gb >= 24 and cpu_cores >= 6:
            return 'macbook_32gb'
        elif memory_gb >= 14 and cpu_cores >= 4:
            return 'macbook_16gb'
        elif memory_gb >= 7 and cpu_cores >= 4:
            return 'macbook_8gb'
        else:
            return 'macbook_conservative'
    
    def export_configuration(self, config_result: ConfigurationResult, 
                           format: str = 'yaml') -> str:
        """
        Export configuration to specified format.
        
        Args:
            config_result: Configuration result to export
            format: Export format ('yaml', 'json', 'python')
            
        Returns:
            Formatted configuration string
        """
        if format.lower() == 'yaml':
            import yaml
            return yaml.dump(config_result.adapted_config, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            import json
            return json.dumps(config_result.adapted_config, indent=2)
        elif format.lower() == 'python':
            return f"# MacBook TRM Training Configuration\n# {config_result.reasoning}\n\nconfig = {repr(config_result.adapted_config)}"
        else:
            raise ValueError(f"Unsupported export format: {format}")