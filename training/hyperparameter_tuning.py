"""
Hyperparameter Tuning and Training Strategies for Email Classification

This module provides advanced hyperparameter optimization, training strategies,
and performance monitoring for the email classification TRM model.
"""

import os
import json
import math
import time
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from omegaconf import DictConfig, OmegaConf


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search"""
    
    # Model architecture parameters
    hidden_size: List[int] = None  # [256, 512, 768]
    num_heads: List[int] = None  # [4, 8, 12]
    L_layers: List[int] = None  # [1, 2, 3]
    H_cycles: List[int] = None  # [2, 3, 4]
    L_cycles: List[int] = None  # [3, 4, 6]
    
    # Training parameters
    learning_rate: List[float] = None  # [1e-5, 5e-5, 1e-4, 5e-4]
    batch_size: List[int] = None  # [16, 32, 64]
    weight_decay: List[float] = None  # [0.01, 0.1, 0.2]
    
    # Email-specific parameters
    pooling_strategy: List[str] = None  # ["mean", "weighted", "attention"]
    classification_dropout: List[float] = None  # [0.1, 0.2, 0.3]
    subject_attention_weight: List[float] = None  # [1.5, 2.0, 2.5]
    
    # Scheduler parameters
    warmup_steps: List[int] = None  # [500, 1000, 2000]
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.hidden_size is None:
            self.hidden_size = [256, 512]
        if self.num_heads is None:
            self.num_heads = [8]
        if self.L_layers is None:
            self.L_layers = [2]
        if self.H_cycles is None:
            self.H_cycles = [2, 3]
        if self.L_cycles is None:
            self.L_cycles = [3, 4]
        if self.learning_rate is None:
            self.learning_rate = [5e-5, 1e-4, 2e-4]
        if self.batch_size is None:
            self.batch_size = [32, 64]
        if self.weight_decay is None:
            self.weight_decay = [0.1]
        if self.pooling_strategy is None:
            self.pooling_strategy = ["weighted", "attention"]
        if self.classification_dropout is None:
            self.classification_dropout = [0.1, 0.2]
        if self.subject_attention_weight is None:
            self.subject_attention_weight = [2.0]
        if self.warmup_steps is None:
            self.warmup_steps = [500, 1000]


class HyperparameterSearcher:
    """Hyperparameter search with different strategies"""
    
    def __init__(self, config: HyperparameterConfig, search_strategy: str = "random"):
        self.config = config
        self.search_strategy = search_strategy
        self.search_history = []
        
    def generate_hyperparameters(self, num_trials: int = 10) -> List[Dict[str, Any]]:
        """Generate hyperparameter combinations"""
        
        if self.search_strategy == "grid":
            return self._grid_search()
        elif self.search_strategy == "random":
            return self._random_search(num_trials)
        elif self.search_strategy == "bayesian":
            return self._bayesian_search(num_trials)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
    
    def _grid_search(self) -> List[Dict[str, Any]]:
        """Generate all combinations (grid search)"""
        import itertools
        
        param_names = []
        param_values = []
        
        for field_name, field_value in asdict(self.config).items():
            if field_value is not None:
                param_names.append(field_name)
                param_values.append(field_value)
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _random_search(self, num_trials: int) -> List[Dict[str, Any]]:
        """Generate random combinations"""
        combinations = []
        
        for _ in range(num_trials):
            param_dict = {}
            
            for field_name, field_value in asdict(self.config).items():
                if field_value is not None:
                    param_dict[field_name] = random.choice(field_value)
            
            combinations.append(param_dict)
        
        return combinations
    
    def _bayesian_search(self, num_trials: int) -> List[Dict[str, Any]]:
        """Bayesian optimization (simplified implementation)"""
        # For now, use random search with some heuristics
        # In a full implementation, you'd use libraries like Optuna or Hyperopt
        
        combinations = []
        
        # Start with some good baseline configurations
        baseline_configs = [
            {
                "hidden_size": 512,
                "learning_rate": 1e-4,
                "batch_size": 32,
                "H_cycles": 2,
                "L_cycles": 4,
                "pooling_strategy": "weighted"
            },
            {
                "hidden_size": 256,
                "learning_rate": 2e-4,
                "batch_size": 64,
                "H_cycles": 3,
                "L_cycles": 3,
                "pooling_strategy": "attention"
            }
        ]
        
        # Add baseline configs
        for config in baseline_configs[:min(len(baseline_configs), num_trials // 2)]:
            # Fill in missing parameters
            full_config = {}
            for field_name, field_value in asdict(self.config).items():
                if field_value is not None:
                    if field_name in config:
                        full_config[field_name] = config[field_name]
                    else:
                        full_config[field_name] = random.choice(field_value)
            combinations.append(full_config)
        
        # Fill remaining with random search
        remaining_trials = num_trials - len(combinations)
        combinations.extend(self._random_search(remaining_trials))
        
        return combinations
    
    def update_history(self, hyperparams: Dict[str, Any], performance: float):
        """Update search history with results"""
        self.search_history.append({
            "hyperparams": hyperparams,
            "performance": performance,
            "timestamp": time.time()
        })
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters from history"""
        if not self.search_history:
            return None
        
        best_result = max(self.search_history, key=lambda x: x["performance"])
        return best_result["hyperparams"]


class AdvancedLRScheduler(_LRScheduler):
    """Advanced learning rate scheduler with warmup and multiple phases"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_ratio: float = 0.01, scheduler_type: str = "cosine_with_restarts"):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.scheduler_type = scheduler_type
        
        super().__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current step"""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase
            lr_scale = step / self.warmup_steps
        else:
            # Main training phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.scheduler_type == "cosine":
                lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            
            elif self.scheduler_type == "cosine_with_restarts":
                # Cosine annealing with restarts
                restart_period = (self.total_steps - self.warmup_steps) // 3  # 3 restarts
                cycle_progress = (step - self.warmup_steps) % restart_period / restart_period
                lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * cycle_progress))
            
            elif self.scheduler_type == "polynomial":
                lr_scale = (1 - progress) ** 2
                lr_scale = max(lr_scale, self.min_lr_ratio)
            
            else:  # linear
                lr_scale = 1 - progress * (1 - self.min_lr_ratio)
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


class TrainingStrategy:
    """Advanced training strategies for email classification"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.strategy_name = config.get("training_strategy", "standard")
        
    def get_training_phases(self) -> List[Dict[str, Any]]:
        """Get training phases based on strategy"""
        
        if self.strategy_name == "curriculum":
            return self._curriculum_learning_phases()
        elif self.strategy_name == "progressive":
            return self._progressive_training_phases()
        elif self.strategy_name == "multi_stage":
            return self._multi_stage_phases()
        else:
            return self._standard_phases()
    
    def _standard_phases(self) -> List[Dict[str, Any]]:
        """Standard single-phase training"""
        return [{
            "name": "main_training",
            "steps": self.config.training.max_steps,
            "learning_rate": self.config.optimizer.lr,
            "batch_size": self.config.training.batch_size,
            "description": "Standard training phase"
        }]
    
    def _curriculum_learning_phases(self) -> List[Dict[str, Any]]:
        """Curriculum learning: start with easier examples"""
        total_steps = self.config.training.max_steps
        
        return [
            {
                "name": "easy_examples",
                "steps": total_steps // 4,
                "learning_rate": self.config.optimizer.lr * 0.5,
                "batch_size": self.config.training.batch_size // 2,
                "data_filter": "easy",  # Short emails, clear categories
                "description": "Training on easy examples (short, clear emails)"
            },
            {
                "name": "medium_examples", 
                "steps": total_steps // 2,
                "learning_rate": self.config.optimizer.lr,
                "batch_size": self.config.training.batch_size,
                "data_filter": "medium",  # Medium length emails
                "description": "Training on medium complexity examples"
            },
            {
                "name": "all_examples",
                "steps": total_steps // 4,
                "learning_rate": self.config.optimizer.lr * 0.8,
                "batch_size": self.config.training.batch_size,
                "data_filter": "all",  # All emails including complex ones
                "description": "Training on all examples including complex ones"
            }
        ]
    
    def _progressive_training_phases(self) -> List[Dict[str, Any]]:
        """Progressive training: gradually increase model complexity"""
        total_steps = self.config.training.max_steps
        
        return [
            {
                "name": "simple_model",
                "steps": total_steps // 3,
                "learning_rate": self.config.optimizer.lr,
                "batch_size": self.config.training.batch_size,
                "model_config": {"H_cycles": 1, "L_cycles": 2},
                "description": "Training with simplified model"
            },
            {
                "name": "medium_model",
                "steps": total_steps // 3,
                "learning_rate": self.config.optimizer.lr * 0.8,
                "batch_size": self.config.training.batch_size,
                "model_config": {"H_cycles": 2, "L_cycles": 3},
                "description": "Training with medium complexity model"
            },
            {
                "name": "full_model",
                "steps": total_steps // 3,
                "learning_rate": self.config.optimizer.lr * 0.6,
                "batch_size": self.config.training.batch_size,
                "model_config": {"H_cycles": 3, "L_cycles": 4},
                "description": "Training with full complexity model"
            }
        ]
    
    def _multi_stage_phases(self) -> List[Dict[str, Any]]:
        """Multi-stage training with different objectives"""
        total_steps = self.config.training.max_steps
        
        return [
            {
                "name": "pretraining",
                "steps": total_steps // 4,
                "learning_rate": self.config.optimizer.lr * 1.5,
                "batch_size": self.config.training.batch_size * 2,
                "objective": "representation_learning",
                "description": "Pretraining for good representations"
            },
            {
                "name": "classification_training",
                "steps": total_steps // 2,
                "learning_rate": self.config.optimizer.lr,
                "batch_size": self.config.training.batch_size,
                "objective": "classification",
                "description": "Main classification training"
            },
            {
                "name": "fine_tuning",
                "steps": total_steps // 4,
                "learning_rate": self.config.optimizer.lr * 0.1,
                "batch_size": self.config.training.batch_size // 2,
                "objective": "fine_tuning",
                "description": "Fine-tuning for best performance"
            }
        ]


class PerformanceTracker:
    """Track and analyze training performance"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        self.best_metrics = {}
        self.plateau_counter = 0
        self.last_improvement_step = 0
        
    def update(self, step: int, metrics: Dict[str, float]):
        """Update performance tracking"""
        
        # Add timestamp and step
        metrics_with_meta = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        self.metrics_history.append(metrics_with_meta)
        
        # Check for improvements
        main_metric = metrics.get("accuracy", metrics.get("f1", 0))
        
        if not self.best_metrics or main_metric > self.best_metrics.get("accuracy", 0):
            self.best_metrics = metrics_with_meta.copy()
            self.last_improvement_step = step
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Save metrics periodically
        if step % 1000 == 0:
            self.save_metrics()
    
    def is_plateaued(self, patience: int = 5) -> bool:
        """Check if training has plateaued"""
        return self.plateau_counter >= patience
    
    def get_early_stopping_recommendation(self, patience: int = 10, min_steps: int = 5000) -> bool:
        """Recommend early stopping"""
        if len(self.metrics_history) < min_steps:
            return False
        
        return self.is_plateaued(patience)
    
    def get_learning_rate_reduction_recommendation(self, patience: int = 3) -> bool:
        """Recommend learning rate reduction"""
        return self.is_plateaued(patience)
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.output_dir / "training_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump({
                "metrics_history": self.metrics_history,
                "best_metrics": self.best_metrics,
                "plateau_counter": self.plateau_counter,
                "last_improvement_step": self.last_improvement_step
            }, f, indent=2)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate training performance report"""
        
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        accuracies = [m.get("accuracy", 0) for m in self.metrics_history]
        losses = [m.get("loss", float('inf')) for m in self.metrics_history]
        
        report = {
            "total_steps": len(self.metrics_history),
            "best_accuracy": max(accuracies) if accuracies else 0,
            "final_accuracy": accuracies[-1] if accuracies else 0,
            "best_loss": min(losses) if losses else float('inf'),
            "final_loss": losses[-1] if losses else float('inf'),
            "improvement_steps": self.last_improvement_step,
            "plateau_duration": self.plateau_counter,
            "convergence_analysis": self._analyze_convergence()
        }
        
        return report
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence"""
        
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        # Look at recent trend
        recent_metrics = self.metrics_history[-10:]
        recent_accuracies = [m.get("accuracy", 0) for m in recent_metrics]
        
        # Calculate trend
        if len(recent_accuracies) >= 2:
            trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
            
            if trend > 0.001:
                status = "improving"
            elif trend < -0.001:
                status = "degrading"
            else:
                status = "stable"
        else:
            status = "unknown"
        
        return {
            "status": status,
            "trend_slope": trend if 'trend' in locals() else 0,
            "recent_variance": np.var(recent_accuracies) if recent_accuracies else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Test hyperparameter search
    hp_config = HyperparameterConfig()
    searcher = HyperparameterSearcher(hp_config, "random")
    
    combinations = searcher.generate_hyperparameters(5)
    print("Generated hyperparameter combinations:")
    for i, combo in enumerate(combinations):
        print(f"  {i+1}: {combo}")
    
    # Test training strategy
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        "training_strategy": "curriculum",
        "training": {"max_steps": 10000, "batch_size": 32},
        "optimizer": {"lr": 1e-4}
    })
    
    strategy = TrainingStrategy(config)
    phases = strategy.get_training_phases()
    
    print("\nTraining phases:")
    for phase in phases:
        print(f"  {phase['name']}: {phase['steps']} steps - {phase['description']}")
    
    # Test performance tracker
    tracker = PerformanceTracker("test_output")
    
    # Simulate some training metrics
    for step in range(0, 1000, 100):
        metrics = {
            "accuracy": 0.5 + 0.4 * (step / 1000) + np.random.normal(0, 0.02),
            "loss": 2.0 - 1.5 * (step / 1000) + np.random.normal(0, 0.1),
            "f1": 0.45 + 0.35 * (step / 1000) + np.random.normal(0, 0.02)
        }
        tracker.update(step, metrics)
    
    report = tracker.generate_report()
    print(f"\nPerformance report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\nHyperparameter tuning and training strategies module ready!")