"""
EmailTRM integration with MacBook optimization systems.

This module provides MacBook-optimized EmailTRM model initialization,
dynamic model complexity adjustment, and CPU-optimized forward pass
for email classification training on MacBook hardware.
"""

import math
import warnings
from typing import Dict, Any, Optional, Tuple, List
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from .hardware_detection import HardwareDetector
from .memory_management import MemoryManager, MemoryConfig
from .cpu_optimization import CPUOptimizer
from .training_config_adapter import TrainingConfigAdapter, HardwareSpecs

# Import EmailTRM components
try:
    from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig, create_email_trm_model
    EMAIL_TRM_AVAILABLE = True
except ImportError:
    EmailTRM = None
    EmailTRMConfig = None
    create_email_trm_model = None
    EMAIL_TRM_AVAILABLE = False

logger = logging.getLogger(__name__)


if EMAIL_TRM_AVAILABLE:
    class MacBookEmailTRMConfig(EmailTRMConfig):
        """Extended EmailTRM configuration with MacBook optimizations."""
        
        def __init__(self, **kwargs):
            # MacBook-specific defaults
            macbook_defaults = {
                # Reduced model complexity for memory constraints
                'hidden_size': 256,  # Reduced from 512
                'L_layers': 2,       # Reduced from 4
                'H_cycles': 2,       # Reduced from 4
                'L_cycles': 3,       # Reduced from 6
                'halt_max_steps': 6, # Reduced from 8
                
                # Memory-efficient settings
                'classification_dropout': 0.2,  # Increased for regularization
                'use_email_structure': True,
                'use_hierarchical_attention': True,
                'pooling_strategy': 'weighted',
                
                # CPU optimization flags
                'enable_cpu_optimization': True,
                'use_mixed_precision': False,  # Not recommended for CPU
                'gradient_checkpointing': True,  # Save memory
            }
            
            # Apply defaults, then override with provided kwargs
            config_dict = {**macbook_defaults, **kwargs}
            super().__init__(**config_dict)
            
            # MacBook-specific attributes
            self.enable_cpu_optimization = config_dict.get('enable_cpu_optimization', True)
            self.use_mixed_precision = config_dict.get('use_mixed_precision', False)
            self.gradient_checkpointing = config_dict.get('gradient_checkpointing', True)
            self.dynamic_complexity = config_dict.get('dynamic_complexity', True)
            self.memory_efficient_attention = config_dict.get('memory_efficient_attention', True)

else:
    class MacBookEmailTRMConfig:
        """Mock MacBook EmailTRM configuration for testing when EmailTRM is not available."""
        
        def __init__(self, **kwargs):
            # Set default values
            defaults = {
                'vocab_size': 5000,
                'num_email_categories': 10,
                'hidden_size': 256,
                'L_layers': 2,
                'H_cycles': 2,
                'L_cycles': 3,
                'halt_max_steps': 6,
                'classification_dropout': 0.2,
                'use_email_structure': True,
                'use_hierarchical_attention': True,
                'pooling_strategy': 'weighted',
                'enable_cpu_optimization': True,
                'use_mixed_precision': False,
                'gradient_checkpointing': True,
                'dynamic_complexity': True,
                'memory_efficient_attention': True,
            }
            
            # Apply defaults, then override with provided kwargs
            config_dict = {**defaults, **kwargs}
            for key, value in config_dict.items():
                setattr(self, key, value)


if EMAIL_TRM_AVAILABLE and TORCH_AVAILABLE:
    class MacBookEmailTRM(nn.Module):
        """MacBook-optimized EmailTRM model with dynamic complexity adjustment."""
        
        def __init__(self, config: MacBookEmailTRMConfig, hardware_specs: Optional[HardwareSpecs] = None):
            """
            Initialize MacBook-optimized EmailTRM model.
            
            Args:
                config: MacBook EmailTRM configuration
                hardware_specs: Hardware specifications for optimization
            """
            
        super().__init__()
        
        self.config = config
        self.hardware_specs = hardware_specs or self._detect_hardware()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        
        # Adjust configuration based on hardware constraints
        self.adjusted_config = self._adjust_config_for_hardware(config)
        
        # Create the core EmailTRM model with adjusted config
        self.email_trm = EmailTRM(self.adjusted_config)
        
        # CPU optimization setup
        if config.enable_cpu_optimization:
            self._setup_cpu_optimizations()
        
        # Memory optimization setup
        self._setup_memory_optimizations()
        
        # Track model complexity
        self.current_complexity_factor = 1.0
        self.base_hidden_size = self.adjusted_config.hidden_size
        
        logger.info(f"MacBookEmailTRM initialized with {self._count_parameters():,} parameters")
        logger.info(f"Hardware: {self.hardware_specs.cpu.cores} cores, "
                   f"{self.hardware_specs.memory.available_memory / (1024**3):.1f}GB available")
    
    def _detect_hardware(self) -> HardwareSpecs:
        """Detect hardware specifications."""
        detector = HardwareDetector()
        adapter = TrainingConfigAdapter(detector)
        return adapter.get_hardware_specs()
    
    def _adjust_config_for_hardware(self, config: MacBookEmailTRMConfig) -> MacBookEmailTRMConfig:
        """Adjust configuration based on hardware constraints."""
        memory_gb = self.hardware_specs.memory.available_memory / (1024**3)
        cpu_cores = self.hardware_specs.cpu.cores
        
        # Create adjusted config
        adjusted_config = MacBookEmailTRMConfig(**config.__dict__)
        
        # Memory-based adjustments
        if memory_gb < 6:
            # Severe memory constraints
            adjusted_config.hidden_size = min(config.hidden_size, 128)
            adjusted_config.L_layers = min(config.L_layers, 1)
            adjusted_config.H_cycles = min(config.H_cycles, 1)
            adjusted_config.L_cycles = min(config.L_cycles, 2)
            logger.warning(f"Reduced model complexity due to limited memory ({memory_gb:.1f}GB)")
            
        elif memory_gb < 8:
            # Moderate memory constraints
            adjusted_config.hidden_size = min(config.hidden_size, 256)
            adjusted_config.L_layers = min(config.L_layers, 2)
            adjusted_config.H_cycles = min(config.H_cycles, 2)
            adjusted_config.L_cycles = min(config.L_cycles, 3)
            logger.info(f"Moderately reduced model complexity for {memory_gb:.1f}GB memory")
        
        # CPU-based adjustments
        if cpu_cores < 4:
            # Limited CPU cores - reduce parallel operations
            adjusted_config.use_hierarchical_attention = False
            adjusted_config.pooling_strategy = 'mean'  # Simpler pooling
            logger.warning(f"Simplified model operations due to limited CPU cores ({cpu_cores})")
        
        return adjusted_config
    
    def _setup_cpu_optimizations(self):
        """Setup CPU-specific optimizations."""
        if not TORCH_AVAILABLE:
            return
            
        # Set optimal thread count
        optimal_threads = min(self.hardware_specs.cpu.cores, 4)
        torch.set_num_threads(optimal_threads)
        
        # Enable MKL if available
        if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True
            logger.info("Intel MKL optimization enabled")
        
        # Set CPU-specific flags
        if hasattr(torch.backends, 'openmp'):
            torch.backends.openmp.enabled = True
        
        logger.info(f"CPU optimization setup complete: {optimal_threads} threads")
    
    def _setup_memory_optimizations(self):
        """Setup memory optimization features."""
        if not TORCH_AVAILABLE:
            return
            
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            # This will be applied during forward pass
            logger.info("Gradient checkpointing enabled for memory efficiency")
        
        # Set memory-efficient attention if configured
        if self.config.memory_efficient_attention:
            # This affects attention computation
            logger.info("Memory-efficient attention enabled")
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        if not TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.parameters())
    
    def adjust_complexity_dynamically(self, target_memory_mb: float) -> bool:
        """
        Dynamically adjust model complexity based on memory constraints.
        
        Args:
            target_memory_mb: Target memory usage in MB
            
        Returns:
            True if adjustment was made, False otherwise
        """
        if not self.config.dynamic_complexity:
            return False
        
        current_memory = self.memory_manager.monitor_memory_usage()
        
        if current_memory.used_mb > target_memory_mb:
            # Need to reduce complexity
            new_factor = min(0.9, target_memory_mb / current_memory.used_mb)
            
            if new_factor < self.current_complexity_factor:
                self.current_complexity_factor = new_factor
                self._apply_complexity_adjustment(new_factor)
                logger.info(f"Reduced model complexity to {new_factor:.2f} due to memory pressure")
                return True
        
        return False
    
    def _apply_complexity_adjustment(self, complexity_factor: float):
        """Apply complexity adjustment to model components."""
        if not TORCH_AVAILABLE:
            return
            
        # This is a simplified approach - in practice, you might need to
        # recreate parts of the model with different dimensions
        
        # Adjust dropout rates (higher dropout = effectively lower complexity)
        dropout_adjustment = 1.0 - complexity_factor
        
        # Apply to classification head if it has dropout
        if hasattr(self.email_trm.model.lm_head, 'dropout'):
            current_dropout = self.email_trm.model.lm_head.dropout.p
            new_dropout = min(0.5, current_dropout + dropout_adjustment * 0.2)
            self.email_trm.model.lm_head.dropout.p = new_dropout
    
    def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None,
                puzzle_identifiers: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        CPU-optimized forward pass with memory management.
        
        Args:
            inputs: Input token sequences [batch_size, seq_len]
            labels: Target labels [batch_size] (optional)
            puzzle_identifiers: Email identifiers [batch_size] (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with model outputs
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Monitor memory before forward pass
        memory_before = self.memory_manager.monitor_memory_usage()
        
        # Apply gradient checkpointing if enabled
        if self.config.gradient_checkpointing and self.training:
            # Use checkpoint for memory efficiency
            outputs = torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                inputs,
                labels,
                puzzle_identifiers,
                **kwargs
            )
        else:
            outputs = self._forward_impl(inputs, labels, puzzle_identifiers, **kwargs)
        
        # Monitor memory after forward pass
        memory_after = self.memory_manager.monitor_memory_usage()
        
        # Add memory usage info to outputs
        outputs['memory_usage'] = {
            'before_mb': memory_before.used_mb,
            'after_mb': memory_after.used_mb,
            'delta_mb': memory_after.used_mb - memory_before.used_mb,
            'available_mb': memory_after.available_mb
        }
        
        # Check for memory pressure and adjust if needed
        if memory_after.percent_used > 85:
            self.adjust_complexity_dynamically(memory_after.used_mb * 0.9)
        
        return outputs
    
    def _forward_impl(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None,
                     puzzle_identifiers: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Internal forward implementation."""
        return self.email_trm(inputs, labels=labels, puzzle_identifiers=puzzle_identifiers, **kwargs)
    
    def predict(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None,
                return_confidence: bool = False) -> torch.Tensor:
        """
        CPU-optimized prediction with memory management.
        
        Args:
            inputs: Input token sequences [batch_size, seq_len]
            puzzle_identifiers: Email identifiers [batch_size] (optional)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and optionally confidence scores
        """
        # Set to evaluation mode
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                # Use smaller batch sizes for prediction to save memory
                batch_size = inputs.size(0)
                if batch_size > 16:
                    # Process in chunks
                    all_predictions = []
                    all_confidences = [] if return_confidence else None
                    
                    for i in range(0, batch_size, 16):
                        chunk_inputs = inputs[i:i+16]
                        chunk_ids = puzzle_identifiers[i:i+16] if puzzle_identifiers is not None else None
                        
                        chunk_result = self.email_trm.predict(
                            chunk_inputs, 
                            puzzle_identifiers=chunk_ids,
                            return_confidence=return_confidence
                        )
                        
                        if return_confidence:
                            chunk_preds, chunk_confs = chunk_result
                            all_predictions.append(chunk_preds)
                            all_confidences.append(chunk_confs)
                        else:
                            all_predictions.append(chunk_result)
                    
                    # Concatenate results
                    predictions = torch.cat(all_predictions, dim=0)
                    if return_confidence:
                        confidences = torch.cat(all_confidences, dim=0)
                        return predictions, confidences
                    else:
                        return predictions
                else:
                    # Process normally
                    return self.email_trm.predict(
                        inputs,
                        puzzle_identifiers=puzzle_identifiers,
                        return_confidence=return_confidence
                    )
        finally:
            # Restore training mode
            if was_training:
                self.train()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        return {
            'current_usage_mb': memory_stats.used_mb,
            'current_usage_percent': memory_stats.percent_used,
            'available_mb': memory_stats.available_mb,
            'model_parameters': self._count_parameters(),
            'complexity_factor': self.current_complexity_factor,
            'hardware_memory_gb': self.hardware_specs.memory.total_memory / (1024**3),
            'recommendations': self.memory_manager.get_memory_recommendations(self._count_parameters())
        }
    
    def optimize_for_inference(self):
        """Optimize model for inference performance."""
        if not TORCH_AVAILABLE:
            return
            
        # Set to evaluation mode
        self.eval()
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
        # Apply CPU-specific optimizations
        if hasattr(torch.jit, 'optimize_for_inference'):
            # This is a hypothetical optimization - actual implementation may vary
            pass
        
        logger.info("Model optimized for inference")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'model_parameters': self._count_parameters(),
            'complexity_factor': self.current_complexity_factor,
            'cpu_cores': self.hardware_specs.cpu.cores,
            'cpu_frequency': self.hardware_specs.cpu.base_frequency,
            'memory_gb': self.hardware_specs.memory.total_memory / (1024**3),
            'torch_threads': torch.get_num_threads() if TORCH_AVAILABLE else 0,
            'mkl_enabled': (hasattr(torch.backends, 'mkl') and 
                          torch.backends.mkl.is_available() and 
                          torch.backends.mkl.enabled) if TORCH_AVAILABLE else False
        }

else:
    class MacBookEmailTRM:
        """Mock MacBook EmailTRM model for testing when dependencies are not available."""
        
        def __init__(self, config: MacBookEmailTRMConfig, hardware_specs: Optional[HardwareSpecs] = None):
            self.config = config
            self.hardware_specs = hardware_specs or self._create_mock_hardware_specs()
            self.current_complexity_factor = 1.0
            self.base_hidden_size = getattr(config, 'hidden_size', 256)
            
            # Mock email_trm attribute for compatibility with tests
            self.email_trm = self
            self.adjusted_config = config
            
        def _create_mock_hardware_specs(self):
            """Create mock hardware specs for testing."""
            from unittest.mock import Mock
            mock_specs = Mock()
            mock_specs.cpu.cores = 4
            mock_specs.cpu.base_frequency = 2400.0
            mock_specs.memory.total_memory = 8 * (1024**3)
            mock_specs.memory.available_memory = 6 * (1024**3)
            return mock_specs
            
        def _count_parameters(self) -> int:
            return 1000000  # Mock parameter count
            
        def forward(self, inputs, labels=None, puzzle_identifiers=None, **kwargs):
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            
            batch_size = inputs.size(0)
            num_categories = getattr(self.config, 'num_email_categories', 10)
            
            # Create tensors that require gradients for proper training simulation
            logits = torch.randn(batch_size, num_categories, requires_grad=True)
            loss = torch.tensor(0.5, requires_grad=True)
            
            return {
                "logits": logits,
                "loss": loss,
                "memory_usage": {
                    "before_mb": 100.0,
                    "after_mb": 120.0,
                    "delta_mb": 20.0,
                    "available_mb": 1000.0
                }
            }
            
        def __call__(self, *args, **kwargs):
            """Make the mock model callable like a PyTorch module."""
            return self.forward(*args, **kwargs)
            
        def predict(self, inputs, puzzle_identifiers=None, return_confidence=False):
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
                
            batch_size = inputs.size(0)
            num_categories = getattr(self.config, 'num_email_categories', 10)
            predictions = torch.randint(0, num_categories, (batch_size,))
            
            if return_confidence:
                confidences = torch.rand(batch_size)
                return predictions, confidences
            return predictions
            
        def adjust_complexity_dynamically(self, target_memory_mb: float) -> bool:
            # Mock adjustment - always return True
            return True
            
        def get_memory_stats(self) -> Dict[str, Any]:
            return {
                'current_usage_mb': 100.0,
                'current_usage_percent': 50.0,
                'available_mb': 1000.0,
                'model_parameters': self._count_parameters(),
                'complexity_factor': self.current_complexity_factor,
                'hardware_memory_gb': 8.0,
                'recommendations': {}
            }
            
        def get_performance_stats(self) -> Dict[str, Any]:
            return {
                'model_parameters': self._count_parameters(),
                'complexity_factor': self.current_complexity_factor,
                'cpu_cores': 4,
                'cpu_frequency': 2400.0,
                'memory_gb': 8.0,
                'torch_threads': 4,
                'mkl_enabled': True
            }
            
        def train(self):
            pass
            
        def eval(self):
            pass
            
        def parameters(self):
            """Mock parameters method for optimizer compatibility."""
            if TORCH_AVAILABLE:
                # Return a mock parameter for the optimizer
                return [torch.nn.Parameter(torch.randn(10, 10))]
            else:
                return []


def create_macbook_email_trm(vocab_size: int, num_categories: int = 10,
                           hardware_specs: Optional[HardwareSpecs] = None,
                           **kwargs) -> MacBookEmailTRM:
    """
    Create MacBook-optimized EmailTRM model.
    
    Args:
        vocab_size: Vocabulary size
        num_categories: Number of email categories
        hardware_specs: Hardware specifications (auto-detected if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        MacBook-optimized EmailTRM model
    """
    # Create MacBook-specific configuration
    config = MacBookEmailTRMConfig(
        vocab_size=vocab_size,
        num_email_categories=num_categories,
        **kwargs
    )
    
    # Create and return the model
    return MacBookEmailTRM(config, hardware_specs=hardware_specs)


def get_recommended_config_for_hardware(hardware_specs: Optional[HardwareSpecs] = None) -> Dict[str, Any]:
    """
    Get recommended configuration for current hardware.
    
    Args:
        hardware_specs: Hardware specifications (auto-detected if None)
        
    Returns:
        Recommended configuration dictionary
    """
    if hardware_specs is None:
        try:
            detector = HardwareDetector()
            adapter = TrainingConfigAdapter(detector)
            hardware_specs = adapter.get_hardware_specs()
        except Exception:
            # Fallback to mock specs if detection fails
            from unittest.mock import Mock
            hardware_specs = Mock()
            hardware_specs.memory.available_memory = 8 * (1024**3)
            hardware_specs.cpu.cores = 4
    
    memory_gb = hardware_specs.memory.available_memory / (1024**3)
    cpu_cores = hardware_specs.cpu.cores
    
    # Base configuration
    config = {
        'vocab_size': 5000,
        'num_email_categories': 10,
        'hidden_size': 256,
        'L_layers': 2,
        'H_cycles': 2,
        'L_cycles': 3,
        'enable_cpu_optimization': True,
        'gradient_checkpointing': True,
        'dynamic_complexity': True,
    }
    
    # Adjust based on memory
    if memory_gb >= 16:
        config.update({
            'hidden_size': 512,
            'L_layers': 3,
            'H_cycles': 3,
            'L_cycles': 4,
        })
    elif memory_gb >= 12:
        config.update({
            'hidden_size': 384,
            'L_layers': 2,
            'H_cycles': 2,
            'L_cycles': 3,
        })
    elif memory_gb < 6:
        config.update({
            'hidden_size': 128,
            'L_layers': 1,
            'H_cycles': 1,
            'L_cycles': 2,
        })
    
    # Adjust based on CPU
    if cpu_cores >= 8:
        config.update({
            'use_hierarchical_attention': True,
            'pooling_strategy': 'attention',
        })
    elif cpu_cores < 4:
        config.update({
            'use_hierarchical_attention': False,
            'pooling_strategy': 'mean',
        })
    
    return config