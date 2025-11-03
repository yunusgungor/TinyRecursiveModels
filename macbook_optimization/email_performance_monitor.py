"""
Email performance monitoring system for MacBook training.

This module provides comprehensive performance monitoring for email classification
training, including per-category accuracy tracking, real-time progress monitoring
with MacBook resource usage, and early stopping based on accuracy targets.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    TORCH_AVAILABLE = False

from .resource_monitoring import ResourceMonitor, ResourceSnapshot
from .memory_management import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class CategoryPerformance:
    """Performance metrics for a single email category."""
    category_id: int
    category_name: str
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Counts
    total_samples: int = 0
    correct_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Recent performance (sliding window)
    recent_accuracy: float = 0.0
    recent_samples: int = 0


@dataclass
class EmailPerformanceSnapshot:
    """Snapshot of email classification performance at a point in time."""
    timestamp: float
    step: int
    epoch: int
    
    # Overall metrics
    overall_accuracy: float = 0.0
    overall_loss: float = 0.0
    overall_f1_macro: float = 0.0
    overall_f1_micro: float = 0.0
    
    # Per-category performance
    category_performance: Dict[int, CategoryPerformance] = field(default_factory=dict)
    
    # Training metrics
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    num_reasoning_cycles: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    samples_per_second: float = 0.0
    
    # Target progress
    target_accuracy_reached: bool = False
    min_category_accuracy_reached: bool = False
    
    # Early stopping metrics
    steps_without_improvement: int = 0
    best_accuracy_so_far: float = 0.0


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping mechanism."""
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.001
    monitor_metric: str = "overall_accuracy"  # "overall_accuracy", "f1_macro", "min_category_accuracy"
    mode: str = "max"  # "max" or "min"
    restore_best_weights: bool = True


class EmailPerformanceMonitor:
    """Comprehensive performance monitoring for email classification training."""
    
    def __init__(self, num_categories: int = 10, 
                 category_names: Optional[List[str]] = None,
                 target_accuracy: float = 0.95,
                 min_category_accuracy: float = 0.90,
                 early_stopping_config: Optional[EarlyStoppingConfig] = None):
        """
        Initialize email performance monitor.
        
        Args:
            num_categories: Number of email categories
            category_names: Names of email categories (optional)
            target_accuracy: Target overall accuracy (default 95%)
            min_category_accuracy: Minimum accuracy per category (default 90%)
            early_stopping_config: Early stopping configuration
        """
        self.num_categories = num_categories
        self.category_names = category_names or [f"Category_{i}" for i in range(num_categories)]
        self.target_accuracy = target_accuracy
        self.min_category_accuracy = min_category_accuracy
        
        # Early stopping configuration
        self.early_stopping_config = early_stopping_config or EarlyStoppingConfig()
        
        # Performance tracking
        self.performance_history: List[EmailPerformanceSnapshot] = []
        self.category_stats = {i: CategoryPerformance(i, name) for i, name in enumerate(self.category_names)}
        
        # Real-time monitoring
        self.resource_monitor = ResourceMonitor()
        self.memory_manager = MemoryManager()
        
        # Sliding window for recent performance (last 100 samples)
        self.recent_window_size = 100
        self.recent_predictions = deque(maxlen=self.recent_window_size)
        self.recent_labels = deque(maxlen=self.recent_window_size)
        self.recent_confidences = deque(maxlen=self.recent_window_size)
        
        # Early stopping state
        self.best_metric_value = float('-inf') if self.early_stopping_config.mode == 'max' else float('inf')
        self.steps_without_improvement = 0
        self.best_model_state = None
        
        # Callbacks
        self.performance_callbacks: List[Callable[[EmailPerformanceSnapshot], None]] = []
        self.target_reached_callbacks: List[Callable[[], None]] = []
        self.early_stopping_callbacks: List[Callable[[], None]] = []
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"EmailPerformanceMonitor initialized for {num_categories} categories")
        logger.info(f"Target accuracy: {target_accuracy:.1%}, Min category accuracy: {min_category_accuracy:.1%}")
    
    def add_performance_callback(self, callback: Callable[[EmailPerformanceSnapshot], None]):
        """Add callback for performance updates."""
        self.performance_callbacks.append(callback)
    
    def add_target_reached_callback(self, callback: Callable[[], None]):
        """Add callback for when accuracy target is reached."""
        self.target_reached_callbacks.append(callback)
    
    def add_early_stopping_callback(self, callback: Callable[[], None]):
        """Add callback for early stopping trigger."""
        self.early_stopping_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 1.0):
        """Start real-time resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.resource_monitor.start_monitoring(interval)
            logger.info("Real-time performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time resource monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.resource_monitor.stop_monitoring()
            logger.info("Real-time performance monitoring stopped")
    
    def update_performance(self, predictions: torch.Tensor, labels: torch.Tensor,
                          confidences: Optional[torch.Tensor] = None,
                          loss: float = 0.0, step: int = 0, epoch: int = 0,
                          learning_rate: float = 0.0, gradient_norm: float = 0.0,
                          num_reasoning_cycles: float = 0.0,
                          samples_per_second: float = 0.0) -> EmailPerformanceSnapshot:
        """
        Update performance metrics with new predictions.
        
        Args:
            predictions: Model predictions [batch_size]
            labels: True labels [batch_size]
            confidences: Prediction confidences [batch_size] (optional)
            loss: Current loss value
            step: Current training step
            epoch: Current epoch
            learning_rate: Current learning rate
            gradient_norm: Gradient norm
            num_reasoning_cycles: Average reasoning cycles
            samples_per_second: Training speed
            
        Returns:
            Performance snapshot
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Convert to numpy for easier processing
        predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        confidences_np = confidences.cpu().numpy() if confidences is not None and torch.is_tensor(confidences) else confidences
        
        # Update recent window
        self.recent_predictions.extend(predictions_np)
        self.recent_labels.extend(labels_np)
        if confidences_np is not None:
            self.recent_confidences.extend(confidences_np)
        
        # Update category statistics
        self._update_category_stats(predictions_np, labels_np, confidences_np)
        
        # Create performance snapshot
        snapshot = self._create_performance_snapshot(
            step, epoch, loss, learning_rate, gradient_norm,
            num_reasoning_cycles, samples_per_second
        )
        
        # Add to history
        self.performance_history.append(snapshot)
        
        # Check targets and early stopping
        self._check_targets_and_early_stopping(snapshot)
        
        # Call performance callbacks
        for callback in self.performance_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in performance callback: {e}")
        
        return snapshot
    
    def _update_category_stats(self, predictions: np.ndarray, labels: np.ndarray,
                              confidences: Optional[np.ndarray] = None):
        """Update per-category statistics."""
        for category_id in range(self.num_categories):
            category_stats = self.category_stats[category_id]
            
            # Find samples for this category
            category_mask = (labels == category_id)
            category_predictions = predictions[category_mask]
            category_labels = labels[category_mask]
            
            if len(category_labels) > 0:
                # Update counts
                category_stats.total_samples += len(category_labels)
                correct = (category_predictions == category_labels).sum()
                category_stats.correct_predictions += correct
                
                # Update accuracy
                category_stats.accuracy = category_stats.correct_predictions / category_stats.total_samples
                
                # Update precision, recall, F1
                # True positives: predicted as this category and actually this category
                tp = ((predictions == category_id) & (labels == category_id)).sum()
                # False positives: predicted as this category but not actually this category
                fp = ((predictions == category_id) & (labels != category_id)).sum()
                # False negatives: not predicted as this category but actually this category
                fn = ((predictions != category_id) & (labels == category_id)).sum()
                
                category_stats.true_positives += tp
                category_stats.false_positives += fp
                category_stats.false_negatives += fn
                
                # Calculate precision, recall, F1
                total_tp = category_stats.true_positives
                total_fp = category_stats.false_positives
                total_fn = category_stats.false_negatives
                
                if total_tp + total_fp > 0:
                    category_stats.precision = total_tp / (total_tp + total_fp)
                if total_tp + total_fn > 0:
                    category_stats.recall = total_tp / (total_tp + total_fn)
                if category_stats.precision + category_stats.recall > 0:
                    category_stats.f1_score = 2 * (category_stats.precision * category_stats.recall) / (category_stats.precision + category_stats.recall)
                
                # Update confidence metrics
                if confidences is not None:
                    category_confidences = confidences[category_mask]
                    if len(category_confidences) > 0:
                        category_stats.avg_confidence = np.mean(category_confidences)
                        category_stats.confidence_std = np.std(category_confidences)
                
                # Update recent performance (sliding window)
                recent_category_mask = np.array([l == category_id for l in self.recent_labels])
                if recent_category_mask.sum() > 0:
                    recent_category_predictions = np.array([p for p, l in zip(self.recent_predictions, self.recent_labels) if l == category_id])
                    recent_category_labels = np.array([l for l in self.recent_labels if l == category_id])
                    
                    category_stats.recent_samples = len(recent_category_labels)
                    if len(recent_category_labels) > 0:
                        category_stats.recent_accuracy = (recent_category_predictions == recent_category_labels).mean()
    
    def _create_performance_snapshot(self, step: int, epoch: int, loss: float,
                                   learning_rate: float, gradient_norm: float,
                                   num_reasoning_cycles: float, samples_per_second: float) -> EmailPerformanceSnapshot:
        """Create a performance snapshot."""
        # Calculate overall metrics
        if len(self.recent_predictions) > 0:
            recent_preds = np.array(list(self.recent_predictions))
            recent_labels = np.array(list(self.recent_labels))
            overall_accuracy = (recent_preds == recent_labels).mean()
        else:
            overall_accuracy = 0.0
        
        # Calculate F1 scores
        overall_f1_macro = np.mean([stats.f1_score for stats in self.category_stats.values()])
        
        # Micro F1 (same as accuracy for multi-class)
        overall_f1_micro = overall_accuracy
        
        # Get resource metrics
        memory_stats = self.memory_manager.monitor_memory_usage()
        resource_snapshot = self.resource_monitor.get_current_snapshot()
        
        cpu_usage = resource_snapshot.cpu.percent_total if resource_snapshot else 0.0
        
        # Check target achievement
        target_accuracy_reached = overall_accuracy >= self.target_accuracy
        min_category_accuracy_reached = all(
            stats.accuracy >= self.min_category_accuracy 
            for stats in self.category_stats.values()
            if stats.total_samples > 0
        )
        
        # Create snapshot
        snapshot = EmailPerformanceSnapshot(
            timestamp=time.time(),
            step=step,
            epoch=epoch,
            overall_accuracy=overall_accuracy,
            overall_loss=loss,
            overall_f1_macro=overall_f1_macro,
            overall_f1_micro=overall_f1_micro,
            category_performance={i: stats for i, stats in self.category_stats.items()},
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            num_reasoning_cycles=num_reasoning_cycles,
            memory_usage_mb=memory_stats.used_mb,
            memory_usage_percent=memory_stats.percent_used,
            cpu_usage_percent=cpu_usage,
            samples_per_second=samples_per_second,
            target_accuracy_reached=target_accuracy_reached,
            min_category_accuracy_reached=min_category_accuracy_reached,
            steps_without_improvement=self.steps_without_improvement,
            best_accuracy_so_far=self.best_metric_value if self.early_stopping_config.mode == 'max' else -self.best_metric_value
        )
        
        return snapshot
    
    def _check_targets_and_early_stopping(self, snapshot: EmailPerformanceSnapshot):
        """Check if targets are reached and handle early stopping."""
        # Check if accuracy targets are reached
        if snapshot.target_accuracy_reached and snapshot.min_category_accuracy_reached:
            logger.info(f"🎯 Accuracy targets reached! Overall: {snapshot.overall_accuracy:.1%}, "
                       f"Min category: {min(stats.accuracy for stats in snapshot.category_performance.values()):.1%}")
            
            # Call target reached callbacks
            for callback in self.target_reached_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in target reached callback: {e}")
        
        # Early stopping logic
        if self.early_stopping_config.enabled:
            current_metric = self._get_metric_value(snapshot, self.early_stopping_config.monitor_metric)
            
            improved = False
            if self.early_stopping_config.mode == 'max':
                if current_metric > self.best_metric_value + self.early_stopping_config.min_delta:
                    self.best_metric_value = current_metric
                    improved = True
            else:  # mode == 'min'
                if current_metric < self.best_metric_value - self.early_stopping_config.min_delta:
                    self.best_metric_value = current_metric
                    improved = True
            
            if improved:
                self.steps_without_improvement = 0
                # Store best model state if configured
                if self.early_stopping_config.restore_best_weights:
                    # This would need to be implemented with actual model state
                    pass
            else:
                self.steps_without_improvement += 1
            
            # Check if early stopping should trigger
            if self.steps_without_improvement >= self.early_stopping_config.patience:
                logger.info(f"🛑 Early stopping triggered after {self.steps_without_improvement} steps without improvement")
                
                # Call early stopping callbacks
                for callback in self.early_stopping_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in early stopping callback: {e}")
    
    def _get_metric_value(self, snapshot: EmailPerformanceSnapshot, metric_name: str) -> float:
        """Get metric value from snapshot."""
        if metric_name == "overall_accuracy":
            return snapshot.overall_accuracy
        elif metric_name == "f1_macro":
            return snapshot.overall_f1_macro
        elif metric_name == "min_category_accuracy":
            return min(stats.accuracy for stats in snapshot.category_performance.values() if stats.total_samples > 0)
        elif metric_name == "overall_loss":
            return snapshot.overall_loss
        else:
            logger.warning(f"Unknown metric: {metric_name}, using overall_accuracy")
            return snapshot.overall_accuracy
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return (self.early_stopping_config.enabled and 
                self.steps_without_improvement >= self.early_stopping_config.patience)
    
    def get_current_performance(self) -> Optional[EmailPerformanceSnapshot]:
        """Get the most recent performance snapshot."""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"status": "No performance data available"}
        
        latest = self.performance_history[-1]
        
        # Calculate trends (last 10 snapshots)
        recent_snapshots = self.performance_history[-10:]
        if len(recent_snapshots) > 1:
            accuracy_trend = recent_snapshots[-1].overall_accuracy - recent_snapshots[0].overall_accuracy
            loss_trend = recent_snapshots[-1].overall_loss - recent_snapshots[0].overall_loss
        else:
            accuracy_trend = 0.0
            loss_trend = 0.0
        
        # Category performance summary
        category_summary = {}
        for category_id, stats in latest.category_performance.items():
            category_summary[self.category_names[category_id]] = {
                "accuracy": stats.accuracy,
                "precision": stats.precision,
                "recall": stats.recall,
                "f1_score": stats.f1_score,
                "total_samples": stats.total_samples,
                "recent_accuracy": stats.recent_accuracy,
                "avg_confidence": stats.avg_confidence,
                "meets_target": stats.accuracy >= self.min_category_accuracy
            }
        
        return {
            "current_performance": {
                "overall_accuracy": latest.overall_accuracy,
                "overall_loss": latest.overall_loss,
                "f1_macro": latest.overall_f1_macro,
                "f1_micro": latest.overall_f1_micro,
                "step": latest.step,
                "epoch": latest.epoch,
            },
            "targets": {
                "target_accuracy": self.target_accuracy,
                "min_category_accuracy": self.min_category_accuracy,
                "target_accuracy_reached": latest.target_accuracy_reached,
                "min_category_accuracy_reached": latest.min_category_accuracy_reached,
            },
            "trends": {
                "accuracy_trend": accuracy_trend,
                "loss_trend": loss_trend,
            },
            "category_performance": category_summary,
            "resource_usage": {
                "memory_usage_mb": latest.memory_usage_mb,
                "memory_usage_percent": latest.memory_usage_percent,
                "cpu_usage_percent": latest.cpu_usage_percent,
                "samples_per_second": latest.samples_per_second,
            },
            "early_stopping": {
                "enabled": self.early_stopping_config.enabled,
                "steps_without_improvement": self.steps_without_improvement,
                "patience": self.early_stopping_config.patience,
                "should_stop": self.should_stop_early(),
            },
            "training_efficiency": {
                "learning_rate": latest.learning_rate,
                "gradient_norm": latest.gradient_norm,
                "num_reasoning_cycles": latest.num_reasoning_cycles,
            }
        }
    
    def export_performance_history(self, format: str = "dict") -> Any:
        """
        Export performance history in specified format.
        
        Args:
            format: Export format ("dict", "csv", "json")
            
        Returns:
            Performance history in requested format
        """
        if format == "dict":
            return [
                {
                    "timestamp": snapshot.timestamp,
                    "step": snapshot.step,
                    "epoch": snapshot.epoch,
                    "overall_accuracy": snapshot.overall_accuracy,
                    "overall_loss": snapshot.overall_loss,
                    "f1_macro": snapshot.overall_f1_macro,
                    "memory_usage_percent": snapshot.memory_usage_percent,
                    "cpu_usage_percent": snapshot.cpu_usage_percent,
                    "samples_per_second": snapshot.samples_per_second,
                    **{f"category_{i}_accuracy": stats.accuracy 
                       for i, stats in snapshot.category_performance.items()}
                }
                for snapshot in self.performance_history
            ]
        elif format == "csv":
            # This would require pandas or manual CSV formatting
            raise NotImplementedError("CSV export requires pandas")
        elif format == "json":
            import json
            return json.dumps(self.export_performance_history("dict"), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self.performance_history.clear()
        self.category_stats = {i: CategoryPerformance(i, name) for i, name in enumerate(self.category_names)}
        self.recent_predictions.clear()
        self.recent_labels.clear()
        self.recent_confidences.clear()
        self.best_metric_value = float('-inf') if self.early_stopping_config.mode == 'max' else float('inf')
        self.steps_without_improvement = 0
        logger.info("Performance statistics reset")


# Utility functions for creating monitoring configurations

def create_default_email_categories() -> List[str]:
    """Create default email category names."""
    return [
        "Newsletter",
        "Work", 
        "Personal",
        "Spam",
        "Promotional",
        "Social",
        "Finance",
        "Travel",
        "Shopping",
        "Other"
    ]


def create_performance_monitor_for_email_training(target_accuracy: float = 0.95,
                                                min_category_accuracy: float = 0.90,
                                                early_stopping_patience: int = 5) -> EmailPerformanceMonitor:
    """
    Create a performance monitor configured for email classification training.
    
    Args:
        target_accuracy: Target overall accuracy (default 95%)
        min_category_accuracy: Minimum accuracy per category (default 90%)
        early_stopping_patience: Early stopping patience (default 5)
        
    Returns:
        Configured EmailPerformanceMonitor
    """
    category_names = create_default_email_categories()
    
    early_stopping_config = EarlyStoppingConfig(
        enabled=True,
        patience=early_stopping_patience,
        min_delta=0.001,
        monitor_metric="overall_accuracy",
        mode="max",
        restore_best_weights=True
    )
    
    return EmailPerformanceMonitor(
        num_categories=len(category_names),
        category_names=category_names,
        target_accuracy=target_accuracy,
        min_category_accuracy=min_category_accuracy,
        early_stopping_config=early_stopping_config
    )