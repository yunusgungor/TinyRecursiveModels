"""
Multi-Phase Training Strategies for Email Classification

This module implements progressive training phases with adaptive learning rate
scheduling and phase transition logic based on performance metrics for
optimal convergence in email classification tasks.
"""

import math
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import _LRScheduler
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    optim = None
    _LRScheduler = object
    TORCH_AVAILABLE = False

from .email_training_config import EmailTrainingConfig

logger = logging.getLogger(__name__)


class PhaseTransitionCriteria(Enum):
    """Criteria for transitioning between training phases."""
    STEPS_COMPLETED = "steps_completed"
    ACCURACY_THRESHOLD = "accuracy_threshold"
    LOSS_PLATEAU = "loss_plateau"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    TIME_BASED = "time_based"
    MANUAL = "manual"


@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition logic."""
    criteria: PhaseTransitionCriteria
    
    # Threshold-based criteria
    accuracy_threshold: Optional[float] = None
    loss_threshold: Optional[float] = None
    
    # Plateau-based criteria
    plateau_patience: int = 5
    plateau_min_delta: float = 0.001
    
    # Time-based criteria
    max_phase_time_minutes: Optional[float] = None
    
    # Performance improvement criteria
    improvement_window: int = 10
    min_improvement_rate: float = 0.001


@dataclass
class PhaseMetrics:
    """Metrics for a training phase."""
    phase_name: str
    start_time: float
    end_time: Optional[float]
    
    # Training progress
    steps_completed: int
    target_steps: int
    
    # Performance metrics
    initial_accuracy: Optional[float]
    final_accuracy: Optional[float]
    best_accuracy: Optional[float]
    initial_loss: Optional[float]
    final_loss: Optional[float]
    best_loss: Optional[float]
    
    # Learning metrics
    learning_rate_history: List[float]
    convergence_rate: Optional[float]
    
    # Resource metrics
    average_memory_usage: float
    peak_memory_usage: float
    average_cpu_usage: float
    
    # Phase-specific metrics
    phase_success: bool = False
    transition_reason: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AdaptiveLearningRateScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler for multi-phase training.
    
    Implements multiple scheduling strategies with phase-aware adjustments
    and performance-based adaptations.
    """
    
    def __init__(self, 
                 optimizer,
                 phase_config: Dict[str, Any],
                 total_steps: int,
                 warmup_steps: int = 0,
                 scheduler_type: str = "cosine_with_warmup",
                 min_lr_ratio: float = 0.01,
                 performance_factor: float = 1.0):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            phase_config: Phase configuration
            total_steps: Total steps for this phase
            warmup_steps: Number of warmup steps
            scheduler_type: Type of scheduler
            min_lr_ratio: Minimum learning rate ratio
            performance_factor: Performance-based adjustment factor
        """
        self.phase_config = phase_config
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.min_lr_ratio = min_lr_ratio
        self.performance_factor = performance_factor
        
        # Performance tracking
        self.performance_history: List[float] = []
        self.last_performance_check = 0
        self.performance_check_interval = 100
        
        super().__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase - linear increase
            lr_scale = step / self.warmup_steps
        else:
            # Main phase - apply scheduling strategy
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            
            if self.scheduler_type == "cosine_with_warmup":
                lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            
            elif self.scheduler_type == "linear_decay":
                lr_scale = 1.0 - progress * (1 - self.min_lr_ratio)
            
            elif self.scheduler_type == "exponential_decay":
                decay_rate = -math.log(self.min_lr_ratio)
                lr_scale = math.exp(-decay_rate * progress)
            
            elif self.scheduler_type == "polynomial_decay":
                power = self.phase_config.get("polynomial_power", 2.0)
                lr_scale = (1 - progress) ** power
                lr_scale = max(lr_scale, self.min_lr_ratio)
            
            elif self.scheduler_type == "cosine_with_restarts":
                # Cosine annealing with restarts
                restart_period = self.total_steps // 3  # 3 restarts per phase
                cycle_progress = (step - self.warmup_steps) % restart_period / restart_period
                lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * cycle_progress))
            
            else:  # Default to cosine
                lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply performance-based adjustment
        lr_scale *= self.performance_factor
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]
    
    def update_performance_factor(self, current_performance: float):
        """Update performance factor based on recent performance."""
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        # Check if we should adjust performance factor
        if len(self.performance_history) >= 10:
            recent_trend = self._calculate_performance_trend()
            
            if recent_trend < -0.01:  # Performance declining
                self.performance_factor = max(0.1, self.performance_factor * 0.8)
                logger.info(f"Reducing learning rate due to performance decline: factor = {self.performance_factor:.3f}")
            elif recent_trend > 0.01:  # Performance improving
                self.performance_factor = min(2.0, self.performance_factor * 1.1)
                logger.debug(f"Increasing learning rate due to performance improvement: factor = {self.performance_factor:.3f}")
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend."""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_values = self.performance_history[-10:]
        
        # Simple linear regression to find trend
        n = len(recent_values)
        x_mean = (n - 1) / 2
        y_mean = sum(recent_values) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope


class PhaseTransitionManager:
    """
    Manages transitions between training phases based on performance metrics
    and predefined criteria.
    """
    
    def __init__(self, transition_configs: Dict[str, PhaseTransitionConfig]):
        """
        Initialize phase transition manager.
        
        Args:
            transition_configs: Dictionary mapping phase names to transition configs
        """
        self.transition_configs = transition_configs
        self.phase_metrics: Dict[str, PhaseMetrics] = {}
        self.current_phase: Optional[str] = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.plateau_counters: Dict[str, int] = {}
        
    def start_phase(self, phase_name: str, target_steps: int) -> PhaseMetrics:
        """
        Start a new training phase.
        
        Args:
            phase_name: Name of the phase
            target_steps: Target number of steps for this phase
            
        Returns:
            Phase metrics object
        """
        self.current_phase = phase_name
        
        metrics = PhaseMetrics(
            phase_name=phase_name,
            start_time=time.time(),
            end_time=None,
            steps_completed=0,
            target_steps=target_steps,
            initial_accuracy=None,
            final_accuracy=None,
            best_accuracy=None,
            initial_loss=None,
            final_loss=None,
            best_loss=None,
            learning_rate_history=[],
            convergence_rate=None,
            average_memory_usage=0.0,
            peak_memory_usage=0.0,
            average_cpu_usage=0.0
        )
        
        self.phase_metrics[phase_name] = metrics
        self.performance_history[phase_name] = []
        self.plateau_counters[phase_name] = 0
        
        logger.info(f"Started training phase: {phase_name} (target steps: {target_steps})")
        
        return metrics
    
    def update_phase_metrics(self, 
                           phase_name: str,
                           step: int,
                           accuracy: Optional[float] = None,
                           loss: Optional[float] = None,
                           learning_rate: Optional[float] = None,
                           memory_usage: Optional[float] = None,
                           cpu_usage: Optional[float] = None):
        """
        Update metrics for the current phase.
        
        Args:
            phase_name: Name of the phase
            step: Current step
            accuracy: Current accuracy
            loss: Current loss
            learning_rate: Current learning rate
            memory_usage: Current memory usage
            cpu_usage: Current CPU usage
        """
        if phase_name not in self.phase_metrics:
            logger.warning(f"Phase {phase_name} not found in metrics")
            return
        
        metrics = self.phase_metrics[phase_name]
        metrics.steps_completed = step
        
        # Update performance metrics
        if accuracy is not None:
            if metrics.initial_accuracy is None:
                metrics.initial_accuracy = accuracy
            metrics.final_accuracy = accuracy
            if metrics.best_accuracy is None or accuracy > metrics.best_accuracy:
                metrics.best_accuracy = accuracy
            
            # Track performance history
            self.performance_history[phase_name].append(accuracy)
        
        if loss is not None:
            if metrics.initial_loss is None:
                metrics.initial_loss = loss
            metrics.final_loss = loss
            if metrics.best_loss is None or loss < metrics.best_loss:
                metrics.best_loss = loss
        
        if learning_rate is not None:
            metrics.learning_rate_history.append(learning_rate)
        
        # Update resource metrics (running averages)
        if memory_usage is not None:
            if metrics.average_memory_usage == 0:
                metrics.average_memory_usage = memory_usage
            else:
                metrics.average_memory_usage = 0.9 * metrics.average_memory_usage + 0.1 * memory_usage
            
            if memory_usage > metrics.peak_memory_usage:
                metrics.peak_memory_usage = memory_usage
        
        if cpu_usage is not None:
            if metrics.average_cpu_usage == 0:
                metrics.average_cpu_usage = cpu_usage
            else:
                metrics.average_cpu_usage = 0.9 * metrics.average_cpu_usage + 0.1 * cpu_usage
    
    def should_transition_phase(self, phase_name: str) -> Tuple[bool, str]:
        """
        Check if the current phase should transition to the next phase.
        
        Args:
            phase_name: Name of the current phase
            
        Returns:
            Tuple of (should_transition, reason)
        """
        if phase_name not in self.transition_configs:
            return False, "no_transition_config"
        
        config = self.transition_configs[phase_name]
        metrics = self.phase_metrics.get(phase_name)
        
        if metrics is None:
            return False, "no_metrics"
        
        # Check steps completed
        if config.criteria == PhaseTransitionCriteria.STEPS_COMPLETED:
            if metrics.steps_completed >= metrics.target_steps:
                return True, "steps_completed"
        
        # Check accuracy threshold
        elif config.criteria == PhaseTransitionCriteria.ACCURACY_THRESHOLD:
            if (config.accuracy_threshold is not None and 
                metrics.final_accuracy is not None and 
                metrics.final_accuracy >= config.accuracy_threshold):
                return True, f"accuracy_threshold_reached_{metrics.final_accuracy:.4f}"
        
        # Check loss plateau
        elif config.criteria == PhaseTransitionCriteria.LOSS_PLATEAU:
            if self._is_loss_plateaued(phase_name, config):
                return True, f"loss_plateau_detected"
        
        # Check performance improvement
        elif config.criteria == PhaseTransitionCriteria.PERFORMANCE_IMPROVEMENT:
            if self._is_performance_stagnant(phase_name, config):
                return True, "performance_stagnant"
        
        # Check time-based criteria
        elif config.criteria == PhaseTransitionCriteria.TIME_BASED:
            if (config.max_phase_time_minutes is not None and
                metrics.start_time is not None):
                elapsed_minutes = (time.time() - metrics.start_time) / 60
                if elapsed_minutes >= config.max_phase_time_minutes:
                    return True, f"time_limit_reached_{elapsed_minutes:.1f}min"
        
        # Always check if steps are completed as a fallback
        if metrics.steps_completed >= metrics.target_steps:
            return True, "steps_completed_fallback"
        
        return False, "continue_phase"
    
    def _is_loss_plateaued(self, phase_name: str, config: PhaseTransitionConfig) -> bool:
        """Check if loss has plateaued."""
        if phase_name not in self.performance_history:
            return False
        
        history = self.performance_history[phase_name]
        if len(history) < config.plateau_patience + 1:
            return False
        
        # Check if recent losses show no improvement
        recent_losses = history[-config.plateau_patience:]
        best_recent = min(recent_losses)
        previous_best = min(history[:-config.plateau_patience]) if len(history) > config.plateau_patience else float('inf')
        
        improvement = previous_best - best_recent
        
        if improvement < config.plateau_min_delta:
            self.plateau_counters[phase_name] += 1
            return self.plateau_counters[phase_name] >= config.plateau_patience
        else:
            self.plateau_counters[phase_name] = 0
            return False
    
    def _is_performance_stagnant(self, phase_name: str, config: PhaseTransitionConfig) -> bool:
        """Check if performance improvement has stagnated."""
        if phase_name not in self.performance_history:
            return False
        
        history = self.performance_history[phase_name]
        if len(history) < config.improvement_window:
            return False
        
        # Calculate improvement rate over the window
        recent_window = history[-config.improvement_window:]
        
        if len(recent_window) < 2:
            return False
        
        # Simple linear regression to find improvement trend
        n = len(recent_window)
        x_mean = (n - 1) / 2
        y_mean = sum(recent_window) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent_window))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return True  # No variance means stagnant
        
        slope = numerator / denominator
        return slope < config.min_improvement_rate
    
    def end_phase(self, phase_name: str, success: bool = True, reason: str = "completed") -> PhaseMetrics:
        """
        End the current phase and finalize metrics.
        
        Args:
            phase_name: Name of the phase
            success: Whether the phase completed successfully
            reason: Reason for ending the phase
            
        Returns:
            Final phase metrics
        """
        if phase_name not in self.phase_metrics:
            logger.warning(f"Phase {phase_name} not found in metrics")
            return None
        
        metrics = self.phase_metrics[phase_name]
        metrics.end_time = time.time()
        metrics.phase_success = success
        metrics.transition_reason = reason
        
        # Calculate convergence rate
        if len(metrics.learning_rate_history) > 1:
            initial_lr = metrics.learning_rate_history[0]
            final_lr = metrics.learning_rate_history[-1]
            metrics.convergence_rate = (initial_lr - final_lr) / initial_lr if initial_lr > 0 else 0
        
        phase_duration = metrics.end_time - metrics.start_time
        
        logger.info(f"Ended training phase: {phase_name}")
        logger.info(f"  Duration: {phase_duration/60:.1f} minutes")
        logger.info(f"  Steps completed: {metrics.steps_completed}/{metrics.target_steps}")
        logger.info(f"  Success: {success}, Reason: {reason}")
        
        if metrics.final_accuracy is not None:
            logger.info(f"  Final accuracy: {metrics.final_accuracy:.4f}")
        if metrics.best_accuracy is not None:
            logger.info(f"  Best accuracy: {metrics.best_accuracy:.4f}")
        
        self.current_phase = None
        
        return metrics
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of all completed phases."""
        summary = {
            "total_phases": len(self.phase_metrics),
            "current_phase": self.current_phase,
            "phases": {}
        }
        
        for phase_name, metrics in self.phase_metrics.items():
            phase_duration = (metrics.end_time or time.time()) - metrics.start_time
            
            summary["phases"][phase_name] = {
                "duration_minutes": phase_duration / 60,
                "steps_completed": metrics.steps_completed,
                "target_steps": metrics.target_steps,
                "completion_rate": metrics.steps_completed / metrics.target_steps if metrics.target_steps > 0 else 0,
                "initial_accuracy": metrics.initial_accuracy,
                "final_accuracy": metrics.final_accuracy,
                "best_accuracy": metrics.best_accuracy,
                "accuracy_improvement": (
                    (metrics.final_accuracy - metrics.initial_accuracy) 
                    if metrics.initial_accuracy and metrics.final_accuracy else None
                ),
                "average_memory_usage": metrics.average_memory_usage,
                "peak_memory_usage": metrics.peak_memory_usage,
                "average_cpu_usage": metrics.average_cpu_usage,
                "phase_success": metrics.phase_success,
                "transition_reason": metrics.transition_reason,
                "warnings": metrics.warnings
            }
        
        return summary


class MultiPhaseTrainingStrategy:
    """
    Comprehensive multi-phase training strategy implementation.
    
    Coordinates phase transitions, learning rate scheduling, and performance
    monitoring for optimal email classification training convergence.
    """
    
    def __init__(self, 
                 phases: List[Dict[str, Any]],
                 transition_configs: Optional[Dict[str, PhaseTransitionConfig]] = None):
        """
        Initialize multi-phase training strategy.
        
        Args:
            phases: List of phase configurations
            transition_configs: Phase transition configurations
        """
        self.phases = phases
        self.transition_configs = transition_configs or {}
        
        # Initialize managers
        self.transition_manager = PhaseTransitionManager(self.transition_configs)
        self.schedulers: Dict[str, AdaptiveLearningRateScheduler] = {}
        
        # Strategy state
        self.current_phase_index = 0
        self.strategy_start_time = None
        self.total_steps_completed = 0
        
        logger.info(f"MultiPhaseTrainingStrategy initialized with {len(phases)} phases")
    
    def start_strategy(self) -> Dict[str, Any]:
        """
        Start the multi-phase training strategy.
        
        Returns:
            Strategy initialization info
        """
        self.strategy_start_time = time.time()
        self.current_phase_index = 0
        self.total_steps_completed = 0
        
        strategy_info = {
            "total_phases": len(self.phases),
            "phase_names": [phase["name"] for phase in self.phases],
            "total_target_steps": sum(phase["steps"] for phase in self.phases),
            "start_time": self.strategy_start_time
        }
        
        logger.info(f"Started multi-phase training strategy")
        logger.info(f"Total phases: {strategy_info['total_phases']}")
        logger.info(f"Total target steps: {strategy_info['total_target_steps']}")
        
        return strategy_info
    
    def get_current_phase_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration for the current phase."""
        if self.current_phase_index >= len(self.phases):
            return None
        
        return self.phases[self.current_phase_index]
    
    def start_current_phase(self, optimizer) -> Tuple[Dict[str, Any], AdaptiveLearningRateScheduler]:
        """
        Start the current phase and create its scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Tuple of (phase_config, scheduler)
        """
        phase_config = self.get_current_phase_config()
        if phase_config is None:
            raise RuntimeError("No more phases to execute")
        
        phase_name = phase_config["name"]
        
        # Start phase metrics tracking
        metrics = self.transition_manager.start_phase(phase_name, phase_config["steps"])
        
        # Create adaptive scheduler for this phase
        scheduler = AdaptiveLearningRateScheduler(
            optimizer=optimizer,
            phase_config=phase_config,
            total_steps=phase_config["steps"],
            warmup_steps=phase_config.get("warmup_steps", 0),
            scheduler_type=phase_config.get("scheduler_type", "cosine_with_warmup"),
            min_lr_ratio=phase_config.get("min_lr_ratio", 0.01),
            performance_factor=phase_config.get("performance_factor", 1.0)
        )
        
        self.schedulers[phase_name] = scheduler
        
        logger.info(f"Started phase: {phase_name}")
        logger.info(f"  Target steps: {phase_config['steps']}")
        logger.info(f"  Learning rate: {phase_config['learning_rate']}")
        logger.info(f"  Batch size: {phase_config['batch_size']}")
        
        return phase_config, scheduler
    
    def update_phase_progress(self,
                            step: int,
                            accuracy: Optional[float] = None,
                            loss: Optional[float] = None,
                            learning_rate: Optional[float] = None,
                            memory_usage: Optional[float] = None,
                            cpu_usage: Optional[float] = None) -> Dict[str, Any]:
        """
        Update progress for the current phase.
        
        Args:
            step: Current step within the phase
            accuracy: Current accuracy
            loss: Current loss
            learning_rate: Current learning rate
            memory_usage: Current memory usage
            cpu_usage: Current CPU usage
            
        Returns:
            Progress update info
        """
        phase_config = self.get_current_phase_config()
        if phase_config is None:
            return {"error": "No active phase"}
        
        phase_name = phase_config["name"]
        
        # Update phase metrics
        self.transition_manager.update_phase_metrics(
            phase_name=phase_name,
            step=step,
            accuracy=accuracy,
            loss=loss,
            learning_rate=learning_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        # Update scheduler performance factor if accuracy is available
        if phase_name in self.schedulers and accuracy is not None:
            self.schedulers[phase_name].update_performance_factor(accuracy)
        
        # Check for phase transition
        should_transition, reason = self.transition_manager.should_transition_phase(phase_name)
        
        progress_info = {
            "phase_name": phase_name,
            "phase_index": self.current_phase_index,
            "step": step,
            "target_steps": phase_config["steps"],
            "progress_percent": (step / phase_config["steps"]) * 100 if phase_config["steps"] > 0 else 0,
            "should_transition": should_transition,
            "transition_reason": reason,
            "accuracy": accuracy,
            "loss": loss,
            "learning_rate": learning_rate
        }
        
        return progress_info
    
    def transition_to_next_phase(self, optimizer, reason: str = "completed") -> Optional[Tuple[Dict[str, Any], AdaptiveLearningRateScheduler]]:
        """
        Transition to the next phase.
        
        Args:
            optimizer: PyTorch optimizer
            reason: Reason for transition
            
        Returns:
            Tuple of (next_phase_config, scheduler) or None if no more phases
        """
        # End current phase
        current_phase_config = self.get_current_phase_config()
        if current_phase_config:
            current_phase_name = current_phase_config["name"]
            self.transition_manager.end_phase(current_phase_name, success=True, reason=reason)
            
            # Update total steps completed
            metrics = self.transition_manager.phase_metrics[current_phase_name]
            self.total_steps_completed += metrics.steps_completed
        
        # Move to next phase
        self.current_phase_index += 1
        
        if self.current_phase_index >= len(self.phases):
            logger.info("All phases completed")
            return None
        
        # Start next phase
        return self.start_current_phase(optimizer)
    
    def is_strategy_complete(self) -> bool:
        """Check if the entire strategy is complete."""
        return self.current_phase_index >= len(self.phases)
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the training strategy."""
        phase_summary = self.transition_manager.get_phase_summary()
        
        strategy_duration = (time.time() - self.strategy_start_time) if self.strategy_start_time else 0
        
        summary = {
            "strategy_info": {
                "total_phases": len(self.phases),
                "completed_phases": self.current_phase_index,
                "current_phase_index": self.current_phase_index,
                "is_complete": self.is_strategy_complete(),
                "total_duration_minutes": strategy_duration / 60,
                "total_steps_completed": self.total_steps_completed
            },
            "phase_summary": phase_summary,
            "overall_performance": self._calculate_overall_performance()
        }
        
        return summary
    
    def _calculate_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall performance metrics across all phases."""
        all_metrics = list(self.transition_manager.phase_metrics.values())
        
        if not all_metrics:
            return {}
        
        # Calculate aggregate metrics
        completed_phases = [m for m in all_metrics if m.phase_success]
        
        if not completed_phases:
            return {"no_completed_phases": True}
        
        # Accuracy progression
        initial_accuracies = [m.initial_accuracy for m in completed_phases if m.initial_accuracy is not None]
        final_accuracies = [m.final_accuracy for m in completed_phases if m.final_accuracy is not None]
        best_accuracies = [m.best_accuracy for m in completed_phases if m.best_accuracy is not None]
        
        # Resource utilization
        avg_memory_usage = [m.average_memory_usage for m in completed_phases if m.average_memory_usage > 0]
        peak_memory_usage = [m.peak_memory_usage for m in completed_phases if m.peak_memory_usage > 0]
        avg_cpu_usage = [m.average_cpu_usage for m in completed_phases if m.average_cpu_usage > 0]
        
        performance = {
            "accuracy_metrics": {
                "initial_accuracy": min(initial_accuracies) if initial_accuracies else None,
                "final_accuracy": max(final_accuracies) if final_accuracies else None,
                "best_accuracy": max(best_accuracies) if best_accuracies else None,
                "total_improvement": (
                    max(final_accuracies) - min(initial_accuracies) 
                    if initial_accuracies and final_accuracies else None
                )
            },
            "resource_metrics": {
                "average_memory_usage": sum(avg_memory_usage) / len(avg_memory_usage) if avg_memory_usage else 0,
                "peak_memory_usage": max(peak_memory_usage) if peak_memory_usage else 0,
                "average_cpu_usage": sum(avg_cpu_usage) / len(avg_cpu_usage) if avg_cpu_usage else 0
            },
            "phase_success_rate": len(completed_phases) / len(all_metrics) if all_metrics else 0
        }
        
        return performance