"""
Configuration management module for MacBook optimization.

This module provides configuration management utilities specifically
designed for MacBook hardware constraints and TRM training optimization.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .hardware_detection import HardwareDetector, CPUSpecs, MemorySpecs, PlatformCapabilities


@dataclass
class MacBookTrainingConfig:
    """MacBook-specific training configuration."""
    # Hardware-adapted parameters
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    
    # Memory management
    memory_limit_mb: int
    enable_memory_monitoring: bool
    dynamic_batch_sizing: bool
    memory_pressure_threshold: float  # percentage
    
    # CPU optimization
    use_mkl: bool
    torch_threads: int
    enable_cpu_optimization: bool
    
    # Training parameters
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    
    # Checkpointing
    checkpoint_interval: int
    max_checkpoints_to_keep: int
    
    # Monitoring
    monitoring_interval: float
    enable_thermal_monitoring: bool
    
    # Hardware info (for reference)
    hardware_summary: Dict[str, Any]


class MacBookConfigManager:
    """Configuration manager for MacBook-specific TRM training."""
    
    def __init__(self, config_dir: str = ".macbook_config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.hardware_detector = HardwareDetector()
        
    def detect_optimal_config(self) -> MacBookTrainingConfig:
        """
        Detect optimal configuration based on current hardware.
        
        Returns:
            MacBookTrainingConfig optimized for current hardware
        """
        # Get hardware specifications
        cpu_specs = self.hardware_detector.detect_cpu_specs()
        memory_specs = self.hardware_detector.detect_memory_specs()
        platform_caps = self.hardware_detector.detect_platform_capabilities()
        hardware_summary = self.hardware_detector.get_hardware_summary()
        
        # Calculate memory-based parameters
        memory_gb = memory_specs.total_memory / (1024**3)
        available_memory_mb = memory_specs.available_memory / (1024**2)
        
        # Conservative batch size calculation for 7M parameter model
        # Rough estimate: 7M params * 4 bytes/param * 3 (forward, backward, optimizer) = ~84MB base
        # Add overhead for activations and intermediate tensors
        model_memory_mb = 200  # Conservative estimate for 7M param model
        
        # Calculate batch size based on available memory
        # Leave 2GB for system and other processes
        usable_memory_mb = max(1000, available_memory_mb - 2048)
        
        # Estimate memory per sample (depends on sequence length, but use conservative estimate)
        memory_per_sample_mb = 10  # Conservative estimate
        max_batch_size = max(1, int((usable_memory_mb - model_memory_mb) / memory_per_sample_mb))
        
        # Clamp batch size to reasonable range
        batch_size = min(max_batch_size, 32)  # Max 32 for stability
        batch_size = max(batch_size, 1)       # Min 1
        
        # Calculate gradient accumulation for effective larger batch size
        target_effective_batch_size = 64  # Target effective batch size
        gradient_accumulation_steps = max(1, target_effective_batch_size // batch_size)
        
        # CPU optimization parameters
        num_workers = self.hardware_detector.get_optimal_worker_count()
        torch_threads = min(cpu_specs.cores, 4)  # Conservative thread count
        
        # Memory management parameters
        memory_limit_mb = int(usable_memory_mb * 0.9)  # 90% of usable memory
        memory_pressure_threshold = 75.0  # Start adjusting at 75% usage
        
        # Training parameters (adjusted for smaller batch sizes)
        base_lr = 1e-4
        # Scale learning rate with effective batch size
        effective_batch_size = batch_size * gradient_accumulation_steps
        lr_scale = (effective_batch_size / 64) ** 0.5  # Square root scaling
        learning_rate = base_lr * lr_scale
        
        # Checkpointing (more frequent for smaller memory systems)
        checkpoint_interval = 500 if memory_gb < 12 else 1000
        
        return MacBookTrainingConfig(
            # Hardware-adapted parameters
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=num_workers,
            pin_memory=False,  # Usually better to disable on memory-constrained systems
            
            # Memory management
            memory_limit_mb=memory_limit_mb,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=memory_pressure_threshold,
            
            # CPU optimization
            use_mkl=platform_caps.has_mkl,
            torch_threads=torch_threads,
            enable_cpu_optimization=True,
            
            # Training parameters
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            
            # Checkpointing
            checkpoint_interval=checkpoint_interval,
            max_checkpoints_to_keep=3,
            
            # Monitoring
            monitoring_interval=2.0,  # Every 2 seconds
            enable_thermal_monitoring=True,
            
            # Hardware info
            hardware_summary=hardware_summary
        )
    
    def save_config(self, config: MacBookTrainingConfig, name: str = "default") -> Path:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            name: Configuration name
            
        Returns:
            Path to saved configuration file
        """
        config_file = self.config_dir / f"{name}.json"
        
        # Convert to dictionary and save
        config_dict = asdict(config)
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        return config_file
    
    def load_config(self, name: str = "default") -> Optional[MacBookTrainingConfig]:
        """
        Load configuration from file.
        
        Args:
            name: Configuration name
            
        Returns:
            Loaded configuration or None if not found
        """
        config_file = self.config_dir / f"{name}.json"
        
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            return MacBookTrainingConfig(**config_dict)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading configuration {name}: {e}")
            return None
    
    def list_configs(self) -> list[str]:
        """
        List available configuration names.
        
        Returns:
            List of configuration names
        """
        config_files = self.config_dir.glob("*.json")
        return [f.stem for f in config_files]
    
    def validate_config(self, config: MacBookTrainingConfig) -> Dict[str, Any]:
        """
        Validate configuration against current hardware.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with warnings and suggestions
        """
        warnings = []
        suggestions = []
        
        # Get current hardware specs
        memory_specs = self.hardware_detector.detect_memory_specs()
        cpu_specs = self.hardware_detector.detect_cpu_specs()
        
        # Check memory constraints
        available_memory_mb = memory_specs.available_memory / (1024**2)
        if config.memory_limit_mb > available_memory_mb * 0.9:
            warnings.append(f"Memory limit ({config.memory_limit_mb}MB) may be too high for available memory ({available_memory_mb:.0f}MB)")
            suggestions.append(f"Consider reducing memory_limit_mb to {int(available_memory_mb * 0.8)}MB")
        
        # Check batch size
        estimated_memory_usage = config.batch_size * 10 + 200  # Rough estimate
        if estimated_memory_usage > config.memory_limit_mb:
            warnings.append(f"Batch size ({config.batch_size}) may cause memory issues")
            max_safe_batch = max(1, (config.memory_limit_mb - 200) // 10)
            suggestions.append(f"Consider reducing batch_size to {max_safe_batch}")
        
        # Check worker count
        if config.num_workers > cpu_specs.cores:
            warnings.append(f"Number of workers ({config.num_workers}) exceeds CPU cores ({cpu_specs.cores})")
            suggestions.append(f"Consider reducing num_workers to {cpu_specs.cores}")
        
        # Check thread count
        if config.torch_threads > cpu_specs.threads:
            warnings.append(f"PyTorch threads ({config.torch_threads}) exceeds available threads ({cpu_specs.threads})")
            suggestions.append(f"Consider reducing torch_threads to {min(cpu_specs.cores, 4)}")
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def create_config_template(self, memory_gb: int, cpu_cores: int) -> MacBookTrainingConfig:
        """
        Create configuration template for specific hardware specs.
        
        Args:
            memory_gb: Total memory in GB
            cpu_cores: Number of CPU cores
            
        Returns:
            Configuration template
        """
        # Create mock hardware specs for template generation
        mock_memory_specs = type('MockMemory', (), {
            'total_memory': memory_gb * 1024**3,
            'available_memory': memory_gb * 1024**3 * 0.8  # 80% available
        })()
        
        mock_cpu_specs = type('MockCPU', (), {
            'cores': cpu_cores,
            'threads': cpu_cores * 2
        })()
        
        # Calculate parameters similar to detect_optimal_config
        available_memory_mb = mock_memory_specs.available_memory / (1024**2)
        usable_memory_mb = max(1000, available_memory_mb - 2048)
        
        batch_size = min(max(1, int((usable_memory_mb - 200) / 10)), 32)
        gradient_accumulation_steps = max(1, 64 // batch_size)
        
        return MacBookTrainingConfig(
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=min(cpu_cores, 4),
            pin_memory=False,
            memory_limit_mb=int(usable_memory_mb * 0.9),
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=min(cpu_cores, 4),
            enable_cpu_optimization=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500 if memory_gb < 12 else 1000,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={}
        )
    
    def get_config_summary(self, config: MacBookTrainingConfig) -> Dict[str, Any]:
        """
        Get human-readable configuration summary.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Configuration summary
        """
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        
        return {
            "training": {
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "learning_rate": config.learning_rate,
                "max_steps": config.max_steps,
            },
            "hardware": {
                "num_workers": config.num_workers,
                "torch_threads": config.torch_threads,
                "memory_limit_mb": config.memory_limit_mb,
                "use_mkl": config.use_mkl,
            },
            "monitoring": {
                "memory_monitoring": config.enable_memory_monitoring,
                "thermal_monitoring": config.enable_thermal_monitoring,
                "monitoring_interval": config.monitoring_interval,
            },
            "checkpointing": {
                "checkpoint_interval": config.checkpoint_interval,
                "max_checkpoints": config.max_checkpoints_to_keep,
            }
        }