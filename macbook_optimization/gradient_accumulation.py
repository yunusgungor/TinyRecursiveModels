"""
Gradient accumulation system for MacBook TRM training optimization.

This module provides gradient accumulation functionality to enable effective
larger batch sizes on memory-constrained MacBook hardware by accumulating
gradients over multiple smaller batches.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    # Target effective batch size
    target_batch_size: int = 32
    
    # Memory constraints
    max_micro_batch_size: int = 8
    min_micro_batch_size: int = 1
    
    # Gradient scaling
    gradient_clipping: Optional[float] = 1.0
    scale_gradients: bool = True
    
    # Optimization settings
    sync_batch_norm: bool = False  # Not applicable for single device
    accumulate_grad_batches: Optional[int] = None  # Auto-calculate if None


@dataclass
class AccumulationState:
    """State tracking for gradient accumulation."""
    current_step: int = 0
    accumulated_steps: int = 0
    effective_batch_size: int = 0
    micro_batch_size: int = 0
    accumulation_steps: int = 0
    total_samples_processed: int = 0
    gradient_scale: float = 1.0


class GradientAccumulator:
    """Gradient accumulation manager for memory-efficient training."""
    
    def __init__(self, config: Optional[GradientAccumulationConfig] = None,
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize gradient accumulator.
        
        Args:
            config: Gradient accumulation configuration
            memory_manager: Memory manager for dynamic adjustments
        """
        self.config = config or GradientAccumulationConfig()
        self.memory_manager = memory_manager
        
        # State tracking
        self.state = AccumulationState()
        self.optimizer = None
        self.scheduler = None
        
        # Calculate initial accumulation parameters
        self._calculate_accumulation_parameters()
        
    def _calculate_accumulation_parameters(self):
        """Calculate gradient accumulation parameters based on memory constraints."""
        # Determine micro batch size based on memory constraints
        if self.memory_manager:
            # Use memory manager's current batch size as micro batch size
            micro_batch_size = min(
                self.memory_manager.current_batch_size,
                self.config.max_micro_batch_size
            )
        else:
            micro_batch_size = self.config.max_micro_batch_size
        
        micro_batch_size = max(micro_batch_size, self.config.min_micro_batch_size)
        
        # Calculate accumulation steps
        if self.config.accumulate_grad_batches is not None:
            accumulation_steps = self.config.accumulate_grad_batches
        else:
            accumulation_steps = max(1, self.config.target_batch_size // micro_batch_size)
        
        # Calculate effective batch size
        effective_batch_size = micro_batch_size * accumulation_steps
        
        # Calculate gradient scale
        gradient_scale = 1.0 / accumulation_steps if self.config.scale_gradients else 1.0
        
        # Update state
        self.state.micro_batch_size = micro_batch_size
        self.state.accumulation_steps = accumulation_steps
        self.state.effective_batch_size = effective_batch_size
        self.state.gradient_scale = gradient_scale
        
    def setup_optimizer(self, optimizer: Any, 
                       scheduler: Optional[Any] = None):
        """
        Setup optimizer and scheduler for gradient accumulation.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated (not stepped)."""
        return (self.state.accumulated_steps + 1) % self.state.accumulation_steps != 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step (accumulation complete)."""
        return (self.state.accumulated_steps + 1) % self.state.accumulation_steps == 0
    
    def scale_loss(self, loss: Any) -> Any:
        """
        Scale loss for gradient accumulation.
        
        Args:
            loss: Original loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.config.scale_gradients:
            return loss * self.state.gradient_scale
        return loss
    
    def accumulate_gradients(self, model: Any, loss: Any, 
                           retain_graph: bool = False) -> Dict[str, Any]:
        """
        Accumulate gradients from loss.
        
        Args:
            model: PyTorch model
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary with accumulation info
        """
        # Scale loss for accumulation
        scaled_loss = self.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward(retain_graph=retain_graph)
        
        # Update accumulation state
        self.state.accumulated_steps += 1
        self.state.total_samples_processed += self.state.micro_batch_size
        
        # Check if we should step
        should_step = self.should_step()
        
        info = {
            "scaled_loss": scaled_loss.item(),
            "original_loss": loss.item(),
            "accumulated_steps": self.state.accumulated_steps,
            "should_step": should_step,
            "effective_batch_size": self.state.effective_batch_size,
            "gradient_scale": self.state.gradient_scale,
        }
        
        return info
    
    def step_optimizer(self) -> Dict[str, Any]:
        """
        Step optimizer and scheduler if accumulation is complete.
        
        Returns:
            Dictionary with step info
        """
        if not self.should_step():
            return {"stepped": False, "reason": "accumulation_incomplete"}
        
        step_info = {"stepped": True}
        
        # Gradient clipping if configured
        if self.config.gradient_clipping is not None and self.optimizer is not None:
            # Get model parameters from optimizer
            parameters = []
            for param_group in self.optimizer.param_groups:
                parameters.extend(param_group['params'])
            
            # Clip gradients
            if TORCH_AVAILABLE:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.gradient_clipping
                )
            else:
                grad_norm = 0.0
            step_info["grad_norm"] = grad_norm.item()
        
        # Step optimizer
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
            step_info["optimizer_stepped"] = True
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            step_info["scheduler_stepped"] = True
            step_info["learning_rate"] = self.scheduler.get_last_lr()[0]
        
        # Update state
        self.state.current_step += 1
        
        return step_info
    
    def zero_grad(self):
        """Zero gradients if optimizer is available."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
    
    def adjust_for_memory_pressure(self, memory_pressure_level: str):
        """
        Adjust accumulation parameters based on memory pressure.
        
        Args:
            memory_pressure_level: "low", "medium", "high", "critical"
        """
        if memory_pressure_level in ["high", "critical"]:
            # Reduce micro batch size and increase accumulation steps
            new_micro_batch_size = max(
                self.config.min_micro_batch_size,
                self.state.micro_batch_size // 2
            )
            
            if new_micro_batch_size != self.state.micro_batch_size:
                print(f"Adjusting micro batch size for memory pressure: "
                      f"{self.state.micro_batch_size} -> {new_micro_batch_size}")
                
                # Recalculate parameters
                old_effective_batch_size = self.state.effective_batch_size
                self.state.micro_batch_size = new_micro_batch_size
                self.state.accumulation_steps = max(1, old_effective_batch_size // new_micro_batch_size)
                self.state.effective_batch_size = new_micro_batch_size * self.state.accumulation_steps
                self.state.gradient_scale = 1.0 / self.state.accumulation_steps if self.config.scale_gradients else 1.0
        
        elif memory_pressure_level == "low" and self.state.micro_batch_size < self.config.max_micro_batch_size:
            # Increase micro batch size if memory allows
            new_micro_batch_size = min(
                self.config.max_micro_batch_size,
                self.state.micro_batch_size * 2
            )
            
            if new_micro_batch_size != self.state.micro_batch_size:
                print(f"Increasing micro batch size: "
                      f"{self.state.micro_batch_size} -> {new_micro_batch_size}")
                
                # Recalculate parameters
                old_effective_batch_size = self.state.effective_batch_size
                self.state.micro_batch_size = new_micro_batch_size
                self.state.accumulation_steps = max(1, old_effective_batch_size // new_micro_batch_size)
                self.state.effective_batch_size = new_micro_batch_size * self.state.accumulation_steps
                self.state.gradient_scale = 1.0 / self.state.accumulation_steps if self.config.scale_gradients else 1.0
    
    def calculate_optimal_accumulation(self, target_batch_size: int, 
                                     available_memory_mb: float,
                                     model_params: int) -> Dict[str, int]:
        """
        Calculate optimal accumulation parameters for given constraints.
        
        Args:
            target_batch_size: Desired effective batch size
            available_memory_mb: Available memory in MB
            model_params: Number of model parameters
            
        Returns:
            Dictionary with optimal parameters
        """
        # Estimate memory per sample (simplified)
        memory_per_sample_mb = (model_params * 4 * 3) / (1024**2)  # params + gradients + optimizer
        
        # Calculate maximum micro batch size that fits in memory
        max_micro_batch_size = max(1, int(available_memory_mb * 0.7 / memory_per_sample_mb))
        max_micro_batch_size = min(max_micro_batch_size, self.config.max_micro_batch_size)
        
        # Calculate accumulation steps
        accumulation_steps = max(1, target_batch_size // max_micro_batch_size)
        
        # Calculate actual effective batch size
        effective_batch_size = max_micro_batch_size * accumulation_steps
        
        return {
            "micro_batch_size": max_micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "memory_utilization_mb": max_micro_batch_size * memory_per_sample_mb,
        }
    
    def get_accumulation_info(self) -> Dict[str, Any]:
        """Get current accumulation state and configuration."""
        return {
            "config": {
                "target_batch_size": self.config.target_batch_size,
                "max_micro_batch_size": self.config.max_micro_batch_size,
                "gradient_clipping": self.config.gradient_clipping,
                "scale_gradients": self.config.scale_gradients,
            },
            "state": {
                "current_step": self.state.current_step,
                "accumulated_steps": self.state.accumulated_steps,
                "micro_batch_size": self.state.micro_batch_size,
                "accumulation_steps": self.state.accumulation_steps,
                "effective_batch_size": self.state.effective_batch_size,
                "gradient_scale": self.state.gradient_scale,
                "total_samples_processed": self.state.total_samples_processed,
            },
            "progress": {
                "accumulation_progress": self.state.accumulated_steps % self.state.accumulation_steps,
                "steps_until_optimizer_step": self.state.accumulation_steps - (self.state.accumulated_steps % self.state.accumulation_steps),
                "should_step_next": self.should_step(),
            }
        }
    
    def reset_accumulation(self):
        """Reset accumulation state."""
        self.state.accumulated_steps = 0
        if self.optimizer is not None:
            self.optimizer.zero_grad()
    
    def update_target_batch_size(self, new_target_batch_size: int):
        """
        Update target batch size and recalculate parameters.
        
        Args:
            new_target_batch_size: New target effective batch size
        """
        self.config.target_batch_size = new_target_batch_size
        self._calculate_accumulation_parameters()
        
        # Reset accumulation state to avoid inconsistencies
        self.reset_accumulation()


class TrainingLoopIntegration:
    """Helper class for integrating gradient accumulation into training loops."""
    
    def __init__(self, accumulator: GradientAccumulator):
        """
        Initialize training loop integration.
        
        Args:
            accumulator: Gradient accumulator instance
        """
        self.accumulator = accumulator
        
    def training_step(self, model: Any, batch: Any, 
                     loss_fn: callable) -> Dict[str, Any]:
        """
        Execute a training step with gradient accumulation.
        
        Args:
            model: PyTorch model
            batch: Training batch
            loss_fn: Loss function that takes (model, batch) and returns loss
            
        Returns:
            Dictionary with training step info
        """
        # Forward pass
        loss = loss_fn(model, batch)
        
        # Accumulate gradients
        accumulation_info = self.accumulator.accumulate_gradients(model, loss)
        
        # Step optimizer if accumulation is complete
        step_info = self.accumulator.step_optimizer()
        
        # Combine info
        training_info = {
            **accumulation_info,
            **step_info,
            "loss": loss.item(),
        }
        
        return training_info
    
    def get_effective_learning_rate(self) -> float:
        """Get effective learning rate considering accumulation."""
        if self.accumulator.scheduler is not None:
            base_lr = self.accumulator.scheduler.get_last_lr()[0]
        elif self.accumulator.optimizer is not None:
            base_lr = self.accumulator.optimizer.param_groups[0]['lr']
        else:
            return 0.0
        
        # Learning rate is effectively scaled by accumulation
        return base_lr
    
    def should_log_metrics(self) -> bool:
        """Check if metrics should be logged (after optimizer step)."""
        return self.accumulator.state.accumulated_steps % self.accumulator.state.accumulation_steps == 0
    
    def get_logging_info(self) -> Dict[str, Any]:
        """Get information for logging."""
        info = self.accumulator.get_accumulation_info()
        
        return {
            "effective_batch_size": info["state"]["effective_batch_size"],
            "micro_batch_size": info["state"]["micro_batch_size"],
            "accumulation_steps": info["state"]["accumulation_steps"],
            "gradient_scale": info["state"]["gradient_scale"],
            "optimizer_steps": info["state"]["current_step"],
            "samples_processed": info["state"]["total_samples_processed"],
            "effective_learning_rate": self.get_effective_learning_rate(),
        }