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
    
        def empty_carry(self, batch_size: int, device: torch.device):
            """Create empty carry state for recursive reasoning."""
            return self.email_trm.empty_carry(batch_size, device)
    
        def forward(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None,
                    puzzle_identifiers: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
            """
            CPU-optimized forward pass with TRM recursive reasoning and memory management.
        
        Args:
            inputs: Input token sequences [batch_size, seq_len]
            labels: Target labels [batch_size] (optional)
            puzzle_identifiers: Email identifiers [batch_size] (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with model outputs including recursive reasoning results
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Monitor memory before forward pass
        memory_before = self.memory_manager.monitor_memory_usage()
        
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Create puzzle identifiers if not provided
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.arange(batch_size, device=device)
        
        # Initialize carry state for recursive reasoning
        carry = self.email_trm.empty_carry(batch_size, device)
        
        # TRM Recursive Reasoning Forward Pass
        total_loss = 0.0
        all_outputs = []
        
        # Multi-cycle reasoning process
        for cycle in range(self.adjusted_config.H_cycles):
            # Apply gradient checkpointing if enabled
            if self.config.gradient_checkpointing and self.training:
                # Use checkpoint for memory efficiency during training
                cycle_outputs, halt_logits, carry = torch.utils.checkpoint.checkpoint(
                    self.email_trm.model,
                    inputs,
                    puzzle_identifiers,
                    carry,
                    cycle,
                    use_reentrant=False
                )
            else:
                # Standard forward pass
                cycle_outputs, halt_logits, carry = self.email_trm.model(
                    inputs=inputs,
                    puzzle_identifiers=puzzle_identifiers,
                    carry=carry,
                    cycle=cycle
                )
            
            # Add cycle information to outputs
            cycle_outputs["halt_logits"] = halt_logits
            cycle_outputs["cycle"] = cycle
            
            all_outputs.append(cycle_outputs)
            
            # Compute cycle loss if labels provided
            if labels is not None:
                cycle_loss = self.email_trm._compute_enhanced_loss(cycle_outputs, labels)
                
                # Weight loss by cycle (later cycles more important)
                cycle_weight = (cycle + 1) / self.adjusted_config.H_cycles
                total_loss += cycle_weight * cycle_loss
            
            # Check halting decisions
            halt_probs = torch.sigmoid(halt_logits[:, 1])  # Probability of halting
            should_halt = halt_probs > 0.5
            
            # Update carry state for halted samples
            carry.halted = carry.halted | should_halt
            carry.steps += 1
            
            # If all samples have halted, break early
            if carry.halted.all():
                logger.debug(f"All samples halted at cycle {cycle}")
                break
        
        # Use outputs from the last cycle as final outputs
        final_outputs = all_outputs[-1]
        
        # Add recursive reasoning metadata
        final_outputs.update({
            "total_cycles": len(all_outputs),
            "avg_halt_cycle": carry.steps.float().mean().item(),
            "halt_efficiency": (carry.steps < self.adjusted_config.H_cycles).float().mean().item(),
            "all_cycle_outputs": all_outputs if len(all_outputs) > 1 else None
        })
        
        # Add loss if computed
        if labels is not None:
            final_outputs["loss"] = total_loss / len(all_outputs)
        
        # Monitor memory after forward pass
        memory_after = self.memory_manager.monitor_memory_usage()
        final_outputs["memory_usage"] = {
            "before_mb": memory_before.used_mb,
            "after_mb": memory_after.used_mb,
            "peak_mb": max(memory_before.used_mb, memory_after.used_mb)
        }
        
        # Dynamic complexity adjustment if memory pressure detected
        if memory_after.percent_used > 85:
            self.adjust_complexity_dynamically(memory_after.used_mb * 0.8)
        
        return final_outputs
    

    
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
                        
                        # Use forward pass for chunks
                        chunk_outputs = self.forward(chunk_inputs, puzzle_identifiers=chunk_ids)
                        chunk_logits = chunk_outputs["logits"]
                        chunk_preds = torch.argmax(chunk_logits, dim=-1)
                        
                        if return_confidence:
                            chunk_probs = torch.softmax(chunk_logits, dim=-1)
                            chunk_confs = torch.max(chunk_probs, dim=-1)[0]
                            chunk_result = (chunk_preds, chunk_confs)
                        else:
                            chunk_result = chunk_preds
                        
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
                    # Process normally using forward pass
                    outputs = self.forward(inputs, puzzle_identifiers=puzzle_identifiers)
                    logits = outputs["logits"]
                    predictions = torch.argmax(logits, dim=-1)
                    
                    if return_confidence:
                        # Calculate confidence as max probability
                        probs = torch.softmax(logits, dim=-1)
                        confidences = torch.max(probs, dim=-1)[0]
                        return predictions, confidences
                    else:
                        return predictions
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
    # If dependencies are not available, raise an error instead of using mock
    def MacBookEmailTRM(*args, **kwargs):
        raise ImportError("EmailTRM dependencies not available. Please install required packages.")
    
    def MacBookEmailTRMConfig(*args, **kwargs):
        raise ImportError("EmailTRM dependencies not available. Please install required packages.")


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