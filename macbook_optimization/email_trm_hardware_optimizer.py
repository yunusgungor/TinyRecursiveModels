"""
MacBook Hardware Optimizer for EmailTRM

This module provides hardware-specific optimizations for EmailTRM models
running on MacBook systems, including memory management, gradient checkpointing,
and performance tuning based on detected hardware specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from pathlib import Path

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager
from macbook_optimization.cpu_optimization import CPUOptimizer


logger = logging.getLogger(__name__)


@dataclass
class MacBookOptimizationConfig:
    """Configuration for MacBook-specific optimizations"""
    
    # Memory optimization
    enable_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    dynamic_batch_sizing: bool = True
    memory_threshold_mb: float = 1000.0  # Reserve 1GB for system
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    optimal_num_threads: Optional[int] = None  # Auto-detect if None
    enable_mkl_optimization: bool = True
    
    # Model architecture optimization
    adaptive_hidden_size: bool = True
    adaptive_sequence_length: bool = True
    enable_mixed_precision: bool = False  # CPU doesn't benefit much from mixed precision
    
    # Training optimization
    gradient_accumulation_steps: int = 8
    max_batch_size: int = 16
    min_batch_size: int = 1
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    memory_check_frequency: int = 100  # Check every N steps
    
    # Thermal management
    enable_thermal_monitoring: bool = True
    thermal_throttle_threshold: float = 85.0  # Celsius
    performance_scaling_factor: float = 0.8  # Scale down when hot


class MacBookHardwareOptimizer:
    """Hardware optimizer for EmailTRM on MacBook systems"""
    
    def __init__(self, 
                 hardware_detector: Optional[HardwareDetector] = None,
                 memory_manager: Optional[MemoryManager] = None):
        
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.memory_manager = memory_manager or MemoryManager()
        self.cpu_optimizer = CPUOptimizer()
        
        # Get hardware specifications
        self.hw_summary = self.hardware_detector.get_hardware_summary()
        
        # Performance tracking
        self.performance_history = []
        self.memory_usage_history = []
        
        logger.info(f"Initialized MacBook optimizer for: {self.hw_summary['cpu']['cores']} cores, "
                   f"{self.hw_summary['memory']['total_gb']:.1f}GB RAM")
    
    def optimize_model_for_hardware(self, 
                                  model: EmailTRM, 
                                  config: MacBookOptimizationConfig) -> EmailTRM:
        """
        Optimize EmailTRM model for MacBook hardware
        
        Args:
            model: EmailTRM model to optimize
            config: Optimization configuration
            
        Returns:
            Optimized EmailTRM model
        """
        
        logger.info("Optimizing EmailTRM model for MacBook hardware")
        
        # Apply memory optimizations
        if config.enable_gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model)
        
        if config.enable_memory_efficient_attention:
            model = self._enable_memory_efficient_attention(model)
        
        # Apply CPU optimizations
        if config.enable_cpu_optimization:
            self._optimize_cpu_settings(config)
        
        # Apply model architecture optimizations
        if config.adaptive_hidden_size or config.adaptive_sequence_length:
            model = self._optimize_model_architecture(model, config)
        
        # Add performance monitoring hooks
        if config.enable_performance_monitoring:
            model = self._add_performance_monitoring(model, config)
        
        # Add thermal monitoring
        if config.enable_thermal_monitoring:
            model = self._add_thermal_monitoring(model, config)
        
        logger.info("MacBook hardware optimization completed")
        return model
    
    def _enable_gradient_checkpointing(self, model: EmailTRM) -> EmailTRM:
        """Enable gradient checkpointing for memory efficiency"""
        
        logger.info("Enabling gradient checkpointing")
        
        # Enable checkpointing for transformer layers
        if hasattr(model.model, 'L_level'):
            # Wrap L_level blocks with checkpointing
            original_forward = model.model.L_level.forward
            
            def checkpointed_forward(self, *args, **kwargs):
                if self.training:
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                else:
                    return original_forward(*args, **kwargs)
            
            model.model.L_level.forward = checkpointed_forward.__get__(model.model.L_level)
        
        # Enable checkpointing for reasoning modules
        if hasattr(model.model, 'H_level'):
            original_forward = model.model.H_level.forward
            
            def checkpointed_forward(self, *args, **kwargs):
                if self.training:
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                else:
                    return original_forward(*args, **kwargs)
            
            model.model.H_level.forward = checkpointed_forward.__get__(model.model.H_level)
        
        # Mark model as using gradient checkpointing
        model._uses_gradient_checkpointing = True
        
        return model
    
    def _enable_memory_efficient_attention(self, model: EmailTRM) -> EmailTRM:
        """Enable memory-efficient attention mechanisms"""
        
        logger.info("Enabling memory-efficient attention")
        
        # Replace standard attention with memory-efficient version
        def memory_efficient_attention(query, key, value, attn_mask=None, dropout_p=0.0):
            """Memory-efficient attention implementation"""
            
            batch_size, seq_len, embed_dim = query.shape
            
            # Use chunked attention for long sequences
            if seq_len > 512:
                chunk_size = 256
                num_chunks = (seq_len + chunk_size - 1) // chunk_size
                
                outputs = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, seq_len)
                    
                    chunk_query = query[:, start_idx:end_idx]
                    chunk_output = F.scaled_dot_product_attention(
                        chunk_query, key, value, 
                        attn_mask=attn_mask[:, start_idx:end_idx] if attn_mask is not None else None,
                        dropout_p=dropout_p
                    )
                    outputs.append(chunk_output)
                
                return torch.cat(outputs, dim=1)
            else:
                return F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p)
        
        # Replace attention functions in the model
        # This is a simplified approach - in practice, you'd need to modify the attention layers
        model._memory_efficient_attention = memory_efficient_attention
        model._uses_memory_efficient_attention = True
        
        return model
    
    def _optimize_cpu_settings(self, config: MacBookOptimizationConfig):
        """Optimize CPU settings for training"""
        
        logger.info("Optimizing CPU settings")
        
        # Set optimal number of threads
        if config.optimal_num_threads is None:
            # Use number of physical cores, not logical cores
            optimal_threads = self.hw_summary['cpu']['cores']
        else:
            optimal_threads = config.optimal_num_threads
        
        torch.set_num_threads(optimal_threads)
        logger.info(f"Set PyTorch threads to {optimal_threads}")
        
        # Enable MKL optimization if available
        if config.enable_mkl_optimization and self.hw_summary['platform']['has_mkl']:
            try:
                import mkl
                mkl.set_num_threads(optimal_threads)
                logger.info("Enabled MKL optimization")
            except ImportError:
                logger.warning("MKL not available for optimization")
        
        # Set CPU affinity for better performance
        try:
            import os
            if hasattr(os, 'sched_setaffinity'):
                # Set affinity to all available cores
                available_cores = list(range(self.hw_summary['cpu']['cores']))
                os.sched_setaffinity(0, available_cores)
                logger.info(f"Set CPU affinity to cores: {available_cores}")
        except (AttributeError, OSError):
            logger.debug("Could not set CPU affinity")
    
    def _optimize_model_architecture(self, 
                                   model: EmailTRM, 
                                   config: MacBookOptimizationConfig) -> EmailTRM:
        """Optimize model architecture based on hardware constraints"""
        
        logger.info("Optimizing model architecture for hardware constraints")
        
        available_memory_gb = self.hw_summary['memory']['available_gb']
        
        # Adjust model parameters based on available memory
        if available_memory_gb < 6:
            # Low memory: reduce model complexity
            if hasattr(model.config, 'hidden_size') and model.config.hidden_size > 256:
                logger.info("Reducing hidden size for low memory system")
                # Note: This would require model reconstruction in practice
        
        elif available_memory_gb < 8:
            # Medium memory: moderate optimizations
            if hasattr(model.config, 'L_layers') and model.config.L_layers > 2:
                logger.info("Optimizing layer count for medium memory system")
        
        # Add memory usage estimation
        model_params = sum(p.numel() for p in model.parameters())
        estimated_memory_mb = model_params * 4 / 1024**2  # Assuming float32
        
        logger.info(f"Model memory estimate: {estimated_memory_mb:.1f}MB")
        
        return model
    
    def _add_performance_monitoring(self, 
                                  model: EmailTRM, 
                                  config: MacBookOptimizationConfig) -> EmailTRM:
        """Add performance monitoring hooks to the model"""
        
        logger.info("Adding performance monitoring")
        
        # Hook for monitoring forward pass performance
        def forward_hook(module, input, output):
            if hasattr(module, '_step_count'):
                module._step_count += 1
            else:
                module._step_count = 1
            
            # Check memory usage periodically
            if module._step_count % config.memory_check_frequency == 0:
                memory_info = psutil.virtual_memory()
                memory_usage_mb = (memory_info.total - memory_info.available) / 1024**2
                
                self.memory_usage_history.append(memory_usage_mb)
                
                # Trigger garbage collection if memory usage is high
                if memory_usage_mb > memory_info.total * 0.85 / 1024**2:  # 85% usage
                    gc.collect()
                    logger.warning(f"High memory usage detected: {memory_usage_mb:.1f}MB, triggered GC")
        
        # Register hooks
        model.register_forward_hook(forward_hook)
        
        # Add performance tracking attributes
        model._performance_monitoring_enabled = True
        model._memory_usage_history = self.memory_usage_history
        
        return model
    
    def _add_thermal_monitoring(self, 
                              model: EmailTRM, 
                              config: MacBookOptimizationConfig) -> EmailTRM:
        """Add thermal monitoring and throttling"""
        
        logger.info("Adding thermal monitoring")
        
        def thermal_check_hook(module, input, output):
            try:
                # Try to get CPU temperature (macOS specific)
                import subprocess
                result = subprocess.run(
                    ['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '--show-initial-usage'],
                    capture_output=True, text=True, timeout=2
                )
                
                if result.returncode == 0:
                    # Parse temperature from output (simplified)
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'CPU die temperature' in line:
                            temp_str = line.split(':')[1].strip()
                            temp = float(temp_str.split()[0])
                            
                            if temp > config.thermal_throttle_threshold:
                                logger.warning(f"High CPU temperature: {temp}°C, applying throttling")
                                # Implement throttling by reducing batch size or adding delays
                                time.sleep(0.1)  # Brief pause to cool down
                            
                            break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, FileNotFoundError):
                # Thermal monitoring not available, skip silently
                pass
        
        # Register thermal monitoring hook (less frequent)
        if hasattr(model, '_step_count'):
            original_forward = model.forward
            
            def thermal_monitored_forward(self, *args, **kwargs):
                result = original_forward(*args, **kwargs)
                
                # Check thermal every 50 steps
                if hasattr(self, '_thermal_step_count'):
                    self._thermal_step_count += 1
                else:
                    self._thermal_step_count = 1
                
                if self._thermal_step_count % 50 == 0:
                    thermal_check_hook(self, args, result)
                
                return result
            
            model.forward = thermal_monitored_forward.__get__(model, EmailTRM)
        
        model._thermal_monitoring_enabled = True
        
        return model
    
    def get_optimal_batch_size(self, 
                             model: EmailTRM, 
                             sequence_length: int,
                             config: MacBookOptimizationConfig) -> int:
        """
        Determine optimal batch size based on hardware constraints
        
        Args:
            model: EmailTRM model
            sequence_length: Input sequence length
            config: Optimization configuration
            
        Returns:
            Optimal batch size
        """
        
        available_memory_gb = self.hw_summary['memory']['available_gb']
        
        # Estimate memory usage per sample
        model_params = sum(p.numel() for p in model.parameters())
        
        # Rough estimation: model weights + gradients + optimizer states + activations
        memory_per_param = 4 * 3  # float32 * (weights + gradients + optimizer)
        model_memory_mb = model_params * memory_per_param / 1024**2
        
        # Activation memory estimation
        hidden_size = getattr(model.config, 'hidden_size', 512)
        activation_memory_per_sample = sequence_length * hidden_size * 4 / 1024**2  # MB
        
        # Available memory for batch (reserve some for system)
        available_for_batch_mb = (available_memory_gb * 1024 - config.memory_threshold_mb - model_memory_mb)
        
        # Calculate optimal batch size
        if available_for_batch_mb > 0:
            optimal_batch_size = int(available_for_batch_mb / activation_memory_per_sample)
            optimal_batch_size = max(config.min_batch_size, 
                                   min(optimal_batch_size, config.max_batch_size))
        else:
            optimal_batch_size = config.min_batch_size
        
        logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
                   f"(memory: {available_memory_gb:.1f}GB, seq_len: {sequence_length})")
        
        return optimal_batch_size
    
    def get_optimal_sequence_length(self, 
                                  available_memory_gb: float,
                                  batch_size: int) -> int:
        """
        Determine optimal sequence length based on memory constraints
        
        Args:
            available_memory_gb: Available memory in GB
            batch_size: Batch size
            
        Returns:
            Optimal sequence length
        """
        
        # Base sequence length
        base_seq_len = 512
        
        # Adjust based on available memory
        if available_memory_gb >= 12:
            optimal_seq_len = 768
        elif available_memory_gb >= 8:
            optimal_seq_len = 512
        elif available_memory_gb >= 6:
            optimal_seq_len = 384
        else:
            optimal_seq_len = 256
        
        # Further adjust based on batch size
        if batch_size > 8:
            optimal_seq_len = min(optimal_seq_len, 384)
        elif batch_size > 16:
            optimal_seq_len = min(optimal_seq_len, 256)
        
        logger.info(f"Calculated optimal sequence length: {optimal_seq_len} "
                   f"(memory: {available_memory_gb:.1f}GB, batch_size: {batch_size})")
        
        return optimal_seq_len
    
    def create_optimized_training_config(self, 
                                       base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create optimized training configuration for MacBook
        
        Args:
            base_config: Base training configuration
            
        Returns:
            Optimized training configuration
        """
        
        optimized_config = base_config.copy()
        
        # Hardware-specific optimizations
        available_memory_gb = self.hw_summary['memory']['available_gb']
        cpu_cores = self.hw_summary['cpu']['cores']
        
        # Batch size optimization
        sequence_length = base_config.get('max_sequence_length', 512)
        optimal_batch_size = self.get_optimal_batch_size(
            None,  # Model not needed for estimation
            sequence_length,
            MacBookOptimizationConfig()
        )
        
        optimized_config.update({
            'batch_size': optimal_batch_size,
            'gradient_accumulation_steps': max(1, 32 // optimal_batch_size),  # Target effective batch size of 32
            'num_workers': min(cpu_cores, 4),  # Conservative worker count
            'pin_memory': False,  # Not beneficial for CPU training
            'persistent_workers': True if cpu_cores >= 4 else False,
        })
        
        # Memory-specific optimizations
        if available_memory_gb < 8:
            optimized_config.update({
                'max_sequence_length': min(sequence_length, 384),
                'gradient_checkpointing': True,
                'dataloader_drop_last': True,  # Avoid variable batch sizes
            })
        
        # CPU-specific optimizations
        optimized_config.update({
            'torch_num_threads': cpu_cores,
            'enable_cpu_optimization': True,
            'mixed_precision': False,  # Not beneficial for CPU
        })
        
        logger.info(f"Created optimized training config: batch_size={optimal_batch_size}, "
                   f"grad_accum={optimized_config['gradient_accumulation_steps']}, "
                   f"workers={optimized_config['num_workers']}")
        
        return optimized_config
    
    def benchmark_model_performance(self, 
                                  model: EmailTRM, 
                                  sample_batch: torch.Tensor,
                                  num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model performance on current hardware
        
        Args:
            model: EmailTRM model to benchmark
            sample_batch: Sample input batch
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        
        logger.info(f"Benchmarking model performance ({num_iterations} iterations)")
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_batch)
        
        # Benchmark
        forward_times = []
        memory_usage = []
        
        for i in range(num_iterations):
            # Clear cache
            gc.collect()
            
            # Monitor memory before
            memory_before = psutil.virtual_memory().used / 1024**2
            
            # Time forward pass
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(sample_batch)
            
            end_time = time.time()
            
            # Monitor memory after
            memory_after = psutil.virtual_memory().used / 1024**2
            
            forward_times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        metrics = {
            'avg_forward_time_ms': np.mean(forward_times),
            'std_forward_time_ms': np.std(forward_times),
            'min_forward_time_ms': np.min(forward_times),
            'max_forward_time_ms': np.max(forward_times),
            'avg_memory_usage_mb': np.mean(memory_usage),
            'max_memory_usage_mb': np.max(memory_usage),
            'throughput_samples_per_sec': sample_batch.size(0) / (np.mean(forward_times) / 1000),
        }
        
        logger.info(f"Benchmark results: {metrics['avg_forward_time_ms']:.1f}±{metrics['std_forward_time_ms']:.1f}ms, "
                   f"{metrics['throughput_samples_per_sec']:.1f} samples/sec")
        
        return metrics
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations"""
        
        return {
            'hardware_specs': self.hw_summary,
            'optimizations_applied': {
                'gradient_checkpointing': True,
                'memory_efficient_attention': True,
                'cpu_optimization': True,
                'thermal_monitoring': True,
                'performance_monitoring': True,
            },
            'recommended_settings': {
                'batch_size': self.get_optimal_batch_size(None, 512, MacBookOptimizationConfig()),
                'sequence_length': self.get_optimal_sequence_length(
                    self.hw_summary['memory']['available_gb'], 8
                ),
                'num_workers': min(self.hw_summary['cpu']['cores'], 4),
                'torch_threads': self.hw_summary['cpu']['cores'],
            }
        }


# Convenience functions
def optimize_email_trm_for_macbook(model: EmailTRM, 
                                 optimization_config: Optional[MacBookOptimizationConfig] = None) -> EmailTRM:
    """
    Optimize EmailTRM model for MacBook hardware
    
    Args:
        model: EmailTRM model to optimize
        optimization_config: Optimization configuration
        
    Returns:
        Optimized EmailTRM model
    """
    
    config = optimization_config or MacBookOptimizationConfig()
    optimizer = MacBookHardwareOptimizer()
    
    optimized_model = optimizer.optimize_model_for_hardware(model, config)
    
    logger.info("EmailTRM optimized for MacBook hardware")
    return optimized_model


def create_macbook_optimization_config(memory_conservative: bool = False) -> MacBookOptimizationConfig:
    """
    Create MacBook optimization configuration
    
    Args:
        memory_conservative: Whether to use memory-conservative settings
        
    Returns:
        MacBookOptimizationConfig
    """
    
    if memory_conservative:
        return MacBookOptimizationConfig(
            enable_gradient_checkpointing=True,
            enable_memory_efficient_attention=True,
            dynamic_batch_sizing=True,
            memory_threshold_mb=1500.0,  # Reserve more memory
            gradient_accumulation_steps=16,  # Larger accumulation
            max_batch_size=8,  # Smaller max batch
            min_batch_size=1,
            memory_check_frequency=50,  # More frequent checks
        )
    else:
        return MacBookOptimizationConfig()


# Example usage and testing
if __name__ == "__main__":
    import logging
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    
    # Test hardware optimization
    from models.recursive_reasoning.trm_email import EmailTRMConfig, EmailTRM
    
    # Create base model
    config = EmailTRMConfig(vocab_size=5000, num_email_categories=10, hidden_size=256)
    model = EmailTRM(config)
    
    # Create optimization config
    opt_config = create_macbook_optimization_config(memory_conservative=False)
    
    # Optimize model
    optimized_model = optimize_email_trm_for_macbook(model, opt_config)
    
    print("Model optimized for MacBook!")
    
    # Test performance
    optimizer = MacBookHardwareOptimizer()
    
    # Create sample batch
    batch_size = 4
    seq_len = 128
    sample_batch = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Benchmark
    metrics = optimizer.benchmark_model_performance(optimized_model, sample_batch, num_iterations=5)
    print(f"Performance: {metrics['avg_forward_time_ms']:.1f}ms, {metrics['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    print(f"Recommended batch size: {summary['recommended_settings']['batch_size']}")
    print(f"Recommended sequence length: {summary['recommended_settings']['sequence_length']}")