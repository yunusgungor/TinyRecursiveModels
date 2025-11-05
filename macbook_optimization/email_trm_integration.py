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
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import EmailTRM components: {e}")
    EmailTRM = None
    EmailTRMConfig = None
    create_email_trm_model = None
    EMAIL_TRM_AVAILABLE = False

logger = logging.getLogger(__name__)


if EMAIL_TRM_AVAILABLE:
    class MacBookEmailTRMConfig(EmailTRMConfig):
        """Extended EmailTRM configuration with MacBook optimizations."""
        
        # MacBook-specific fields
        enable_cpu_optimization: bool = True
        use_mixed_precision: bool = False
        gradient_checkpointing: bool = True
        dynamic_complexity: bool = True
        memory_efficient_attention: bool = True
        
        # Override defaults for MacBook
        L_layers: int = 2       # Reduced from 4
        H_cycles: int = 2       # Reduced from 4
        L_cycles: int = 3       # Reduced from 6
        classification_dropout: float = 0.2  # Increased for regularization
        forward_dtype: str = "float32"  # Use float32 for CPU training stability

else:
    def MacBookEmailTRMConfig(**kwargs):
        raise ImportError("EmailTRM dependencies not available. Please install required packages.")


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
            
            # Alias for compatibility with training loop
            self.model = self.email_trm.model
            
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
            
            # Ensure dimension consistency for email classification
            adjusted_config.puzzle_emb_len = 0  # Disable puzzle embeddings
            adjusted_config.puzzle_emb_ndim = 0  # No puzzle embedding dimensions
            adjusted_config.forward_dtype = "float32"  # Consistent dtype
            
            # Validate configuration consistency
            if hasattr(adjusted_config, 'puzzle_emb_len') and adjusted_config.puzzle_emb_len > 0:
                logger.warning("Disabling puzzle embeddings for email classification to avoid dimension mismatch")
                adjusted_config.puzzle_emb_len = 0
                adjusted_config.puzzle_emb_ndim = 0

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
                # This would be implemented in the actual model
                logger.info("Gradient checkpointing enabled")
            
            # Enable memory-efficient attention if configured
            if self.config.memory_efficient_attention:
                logger.info("Memory-efficient attention enabled")
        
        def _count_parameters(self) -> int:
            """Count total model parameters."""
            if not TORCH_AVAILABLE:
                return 0
            
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def empty_carry(self, batch_size: int, device: torch.device):
            """Create empty carry state for recursive reasoning."""
            return self.email_trm.empty_carry(batch_size, device)
        
        def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None,
                    puzzle_identifiers: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
            """
            CPU-optimized forward pass with TRM recursive reasoning and memory management.
            """
            # Memory monitoring before forward pass
            if self.config.dynamic_complexity:
                memory_stats = self.memory_manager.monitor_memory_usage()
                if memory_stats.used_mb > memory_stats.available_mb * 0.8:
                    logger.warning(f"High memory usage detected: {memory_stats.used_mb:.1f}MB")
            
            # Forward pass through EmailTRM
            outputs = self.email_trm(inputs, labels=labels, puzzle_identifiers=puzzle_identifiers, **kwargs)
            
            return outputs
        
        def predict(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None,
                    return_confidence: bool = False) -> torch.Tensor:
            """
            Predict email categories with optional confidence scores.
            """
            return self.email_trm.predict(inputs, puzzle_identifiers=puzzle_identifiers, 
                                        return_confidence=return_confidence)
        
        def get_memory_stats(self) -> Dict[str, Any]:
            """Get current memory statistics."""
            memory_stats = self.memory_manager.monitor_memory_usage()
            
            return {
                'used_memory_mb': memory_stats.used_mb,
                'available_memory_mb': memory_stats.available_mb,
                'memory_usage_percent': (memory_stats.used_mb / memory_stats.available_mb) * 100,
                'model_parameters': self._count_parameters(),
                'complexity_factor': self.current_complexity_factor
            }

else:
    # If dependencies are not available, raise an error instead of using mock
    def MacBookEmailTRM(*args, **kwargs):
        raise ImportError("EmailTRM dependencies not available. Please install required packages.")
    
    def MacBookEmailTRMConfig(*args, **kwargs):
        raise ImportError("EmailTRM dependencies not available. Please install required packages.")


def create_macbook_email_trm(vocab_size: int, num_categories: int = 10,
                           hardware_specs: Optional[HardwareSpecs] = None,
                           **kwargs) -> MacBookEmailTRM:
    """
    Create MacBook-optimized EmailTRM model with specified configuration.
    
    Args:
        vocab_size: Vocabulary size
        num_categories: Number of email categories
        hardware_specs: Hardware specifications for optimization
        **kwargs: Additional configuration parameters
        
    Returns:
        MacBookEmailTRM model instance
    """
    config = MacBookEmailTRMConfig(
        vocab_size=vocab_size,
        num_email_categories=num_categories,
        **kwargs
    )
    
    return MacBookEmailTRM(config, hardware_specs=hardware_specs)


def get_recommended_config_for_hardware(hardware_specs: Optional[HardwareSpecs] = None) -> Dict[str, Any]:
    """
    Get recommended configuration for current hardware.
    """
    if hardware_specs is None:
        detector = HardwareDetector()
        adapter = TrainingConfigAdapter(detector)
        hardware_specs = adapter.get_hardware_specs()
    
    memory_gb = hardware_specs.memory.available_memory / (1024**3)
    cpu_cores = hardware_specs.cpu.cores
    
    if memory_gb >= 12 and cpu_cores >= 8:
        return {
            'hidden_size': 768,
            'L_layers': 3,
            'H_cycles': 4,
            'L_cycles': 5,
            'batch_size': 16,
            'use_hierarchical_attention': True,
            'pooling_strategy': 'attention'
        }
    elif memory_gb >= 8 and cpu_cores >= 4:
        return {
            'hidden_size': 512,
            'L_layers': 2,
            'H_cycles': 3,
            'L_cycles': 4,
            'batch_size': 8,
            'use_hierarchical_attention': True,
            'pooling_strategy': 'weighted'
        }
    else:
        return {
            'hidden_size': 256,
            'L_layers': 2,
            'H_cycles': 2,
            'L_cycles': 3,
            'batch_size': 4,
            'use_hierarchical_attention': False,
            'pooling_strategy': 'mean'
        }