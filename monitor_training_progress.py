#!/usr/bin/env python3
"""
Monitor Training Progress and Metrics for Real Email Classification

This script implements task 4.3: Monitor training progress and metrics
- Track real-time accuracy, loss, and per-category performance
- Monitor memory usage, CPU utilization, and training speed
- Generate confusion matrices and detailed performance reports

Requirements: 4.1, 4.2, 4.3, 4.4
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macbook_optimization.email_training_orchestrator import EmailTrainingOrchestrator
from macbook_optimization.progress_monitoring import ProgressMonitor
from macbook_optimization.resource_monitoring import ResourceMonitor
from macbook_optimization.memory_management import MemoryManager
from macbook_optimization.email_training_loop import EmailTrainingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_progress_monitoring.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgressSnapshot:
    """Snapshot of training progress at a specific point in time."""
    
    timestamp: datetime
    step: int
    epoch: int
    
    # Performance metrics
    loss: float
    accuracy: float
    learning_rate: float
    
    # Per-category metrics
    category_accuracies: Dict[int, float]
    category_losses: Dict[int, float]
    
    # Resource metrics
    memory_usage_mb: float
    memory_usage_percent: float
    cpu_usage_percent: float
    
    # Training speed metrics
    samples_per_second: float
    steps_per_second: float
    estimated_time_remaining: float
    
    # Model-specific metrics
    gradient_norm: float
    num_reasoning_cycles: float
    
    # Hardware metrics
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None


@dataclass
class ConfusionMatrix:
    """Confusion matrix for email classification."""
    
    matrix: np.ndarray
    category_names: List[str]
    total_samples: int
    
    def get_precision(self, category_idx: int) -> float:
        """Get precision for a specific category."""
        if self.matrix[category_idx, category_idx] == 0:
            return 0.0
        return self.matrix[category_idx, category_idx] / np.sum(self.matrix[:, category_idx])
    
    def get_recall(self, category_idx: int) -> float:
        """Get recall for a specific category."""
        if np.sum(self.matrix[category_idx, :]) == 0:
            return 0.0
        return self.matrix[category_idx, category_idx] / np.sum(self.matrix[category_idx, :])
    
    def get_f1_score(self, category_idx: int) -> float:
        """Get F1 score for a specific category."""
        precision = self.get_precision(category_idx)
        recall = self.get_recall(category_idx)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_overall_accuracy(self) -> float:
        """Get overall accuracy."""
        if self.total_samples == 0:
            return 0.0
        return np.trace(self.matrix) / self.total_samples


class TrainingProgressMonitor:
    """
    Comprehensive training progress and metrics monitor.
    
    Implements task 4.3: Monitor training progress and metrics with:
    - Real-time accuracy, loss, and per-category performance tracking
    - Memory usage, CPU utilization, and training speed monitoring
    - Confusion matrix generation and detailed performance reporting
    """
    
    def __init__(self, 
                 output_dir: str = "training_monitoring_output",
                 update_interval: float = 10.0,
                 save_interval: float = 60.0):
        """
        Initialize training progress monitor.
        
        Args:
            output_dir: Directory for monitoring outputs
            update_interval: Interval between progress updates (seconds)
            save_interval: Interval between saving snapshots (seconds)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_interval = update_interval
        self.save_interval = save_interval
        
        # Initialize monitoring components
        self.resource_monitor = ResourceMonitor()
        self.memory_manager = MemoryManager()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.current_training_id: Optional[str] = None
        
        # Progress tracking
        self.progress_snapshots: List[TrainingProgressSnapshot] = []
        self.confusion_matrices: List[ConfusionMatrix] = []
        self.performance_reports: List[Dict[str, Any]] = []
        
        # Callbacks
        self.progress_callbacks: List[Callable[[TrainingProgressSnapshot], None]] = []
        self.milestone_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Training statistics
        self.training_start_time: Optional[datetime] = None
        self.last_snapshot_time: Optional[datetime] = None
        self.total_steps_completed: int = 0
        self.best_accuracy: float = 0.0
        self.best_loss: float = float('inf')
        
        logger.info(f"TrainingProgressMonitor initialized with output dir: {output_dir}")
    
    def start_monitoring(self, training_id: str, total_steps: int, total_epochs: int) -> None:
        """
        Start monitoring training progress.
        
        Args:
            training_id: Unique training identifier
            total_steps: Total training steps expected
            total_epochs: Total epochs expected
        """
        logger.info(f"Starting training progress monitoring for: {training_id}")
        
        self.current_training_id = training_id
        self.training_start_time = datetime.now()
        self.is_monitoring = True
        
        # Reset state
        self.progress_snapshots.clear()
        self.confusion_matrices.clear()
        self.performance_reports.clear()
        self.total_steps_completed = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(total_steps, total_epochs),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Training progress monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring and return final summary.
        
        Returns:
            Final monitoring summary
        """
        logger.info("Stopping training progress monitoring...")
        
        self.is_monitoring = False
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Generate final summary
        summary = self._generate_final_summary()
        
        # Save final results
        self._save_monitoring_results(summary)
        
        logger.info("Training progress monitoring stopped")
        
        return summary
    
    def update_progress(self, metrics: EmailTrainingMetrics, step: int, epoch: int) -> None:
        """
        Update training progress with new metrics.
        
        Args:
            metrics: Training metrics
            step: Current training step
            epoch: Current epoch
        """
        if not self.is_monitoring:
            return
        
        # Create progress snapshot
        snapshot = self._create_progress_snapshot(metrics, step, epoch)
        self.progress_snapshots.append(snapshot)
        
        # Update statistics
        self.total_steps_completed = step
        if metrics.accuracy > self.best_accuracy:
            self.best_accuracy = metrics.accuracy
        if metrics.loss < self.best_loss:
            self.best_loss = metrics.loss
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
        
        # Check for milestones
        self._check_milestones(snapshot)
        
        # Log progress
        if step % 100 == 0:  # Log every 100 steps
            self._log_progress(snapshot)
    
    def add_progress_callback(self, callback: Callable[[TrainingProgressSnapshot], None]) -> None:
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def add_milestone_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for milestone achievements."""
        self.milestone_callbacks.append(callback)
    
    def generate_confusion_matrix(self, 
                                predictions: np.ndarray, 
                                labels: np.ndarray,
                                category_names: List[str]) -> ConfusionMatrix:
        """
        Generate confusion matrix from predictions and labels.
        
        Args:
            predictions: Model predictions [N]
            labels: True labels [N]
            category_names: List of category names
            
        Returns:
            Confusion matrix
        """
        num_categories = len(category_names)
        matrix = np.zeros((num_categories, num_categories), dtype=int)
        
        for pred, label in zip(predictions, labels):
            if 0 <= pred < num_categories and 0 <= label < num_categories:
                matrix[label, pred] += 1
        
        confusion_matrix = ConfusionMatrix(
            matrix=matrix,
            category_names=category_names,
            total_samples=len(labels)
        )
        
        self.confusion_matrices.append(confusion_matrix)
        
        return confusion_matrix
    
    def generate_performance_report(self, 
                                  confusion_matrix: ConfusionMatrix,
                                  additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate detailed performance report.
        
        Args:
            confusion_matrix: Confusion matrix
            additional_metrics: Additional metrics to include
            
        Returns:
            Performance report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "training_id": self.current_training_id,
            "overall_accuracy": confusion_matrix.get_overall_accuracy(),
            "total_samples": confusion_matrix.total_samples,
            "category_performance": {}
        }
        
        # Per-category performance
        for i, category_name in enumerate(confusion_matrix.category_names):
            report["category_performance"][category_name] = {
                "precision": confusion_matrix.get_precision(i),
                "recall": confusion_matrix.get_recall(i),
                "f1_score": confusion_matrix.get_f1_score(i),
                "support": int(np.sum(confusion_matrix.matrix[i, :]))
            }
        
        # Macro averages
        precisions = [confusion_matrix.get_precision(i) for i in range(len(confusion_matrix.category_names))]
        recalls = [confusion_matrix.get_recall(i) for i in range(len(confusion_matrix.category_names))]
        f1_scores = [confusion_matrix.get_f1_score(i) for i in range(len(confusion_matrix.category_names))]
        
        report["macro_averages"] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores)
        }
        
        # Confusion matrix
        report["confusion_matrix"] = confusion_matrix.matrix.tolist()
        report["category_names"] = confusion_matrix.category_names
        
        # Additional metrics
        if additional_metrics:
            report["additional_metrics"] = additional_metrics
        
        self.performance_reports.append(report)
        
        return report
    
    def _create_progress_snapshot(self, 
                                metrics: EmailTrainingMetrics, 
                                step: int, 
                                epoch: int) -> TrainingProgressSnapshot:
        """Create progress snapshot from metrics."""
        
        # Get resource metrics
        memory_stats = self.memory_manager.monitor_memory_usage()
        resource_stats = self.resource_monitor.get_current_snapshot()
        
        # Calculate training speed
        current_time = datetime.now()
        if self.last_snapshot_time:
            time_diff = (current_time - self.last_snapshot_time).total_seconds()
            steps_per_second = 1.0 / max(time_diff, 0.001)  # Avoid division by zero
        else:
            steps_per_second = 0.0
        
        self.last_snapshot_time = current_time
        
        # Estimate time remaining
        if self.training_start_time and step > 0:
            elapsed_time = (current_time - self.training_start_time).total_seconds()
            avg_time_per_step = elapsed_time / step
            # Assuming total steps is available (would need to be passed in)
            estimated_remaining = avg_time_per_step * max(0, 10000 - step)  # Default 10000 steps
        else:
            estimated_remaining = 0.0
        
        return TrainingProgressSnapshot(
            timestamp=current_time,
            step=step,
            epoch=epoch,
            loss=metrics.loss,
            accuracy=metrics.accuracy,
            learning_rate=metrics.learning_rate,
            category_accuracies=metrics.category_accuracies.copy(),
            category_losses={},  # Would need to be calculated from metrics
            memory_usage_mb=memory_stats.used_mb,
            memory_usage_percent=memory_stats.percent_used,
            cpu_usage_percent=resource_stats.cpu.percent_total,
            samples_per_second=metrics.samples_per_second,
            steps_per_second=steps_per_second,
            estimated_time_remaining=estimated_remaining,
            gradient_norm=metrics.gradient_norm,
            num_reasoning_cycles=metrics.num_reasoning_cycles
        )
    
    def _monitoring_loop(self, total_steps: int, total_epochs: int) -> None:
        """Main monitoring loop running in separate thread."""
        
        logger.info("Monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Periodic monitoring tasks
                self._periodic_monitoring_tasks()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
        
        logger.info("Monitoring loop stopped")
    
    def _periodic_monitoring_tasks(self) -> None:
        """Perform periodic monitoring tasks."""
        
        # Save snapshots periodically
        current_time = datetime.now()
        if (not hasattr(self, '_last_save_time') or 
            (current_time - self._last_save_time).total_seconds() >= self.save_interval):
            
            self._save_progress_snapshots()
            self._last_save_time = current_time
        
        # Check for resource issues
        memory_stats = self.memory_manager.monitor_memory_usage()
        if memory_stats.percent_used > 90:
            logger.warning(f"High memory usage: {memory_stats.percent_used:.1f}%")
        
        # Check for performance issues
        if self.progress_snapshots:
            latest_snapshot = self.progress_snapshots[-1]
            if latest_snapshot.samples_per_second < 1.0:
                logger.warning(f"Low training speed: {latest_snapshot.samples_per_second:.2f} samples/sec")
    
    def _check_milestones(self, snapshot: TrainingProgressSnapshot) -> None:
        """Check for training milestones and trigger callbacks."""
        
        milestones = []
        
        # Accuracy milestones
        accuracy_milestones = [0.80, 0.85, 0.90, 0.95, 0.98]
        for milestone in accuracy_milestones:
            if (snapshot.accuracy >= milestone and 
                (not self.progress_snapshots or 
                 self.progress_snapshots[-2].accuracy < milestone)):
                milestones.append({
                    "type": "accuracy_milestone",
                    "value": milestone,
                    "step": snapshot.step,
                    "timestamp": snapshot.timestamp
                })
        
        # Step milestones
        step_milestones = [1000, 2500, 5000, 7500, 10000]
        for milestone in step_milestones:
            if snapshot.step >= milestone and snapshot.step - 1 < milestone:
                milestones.append({
                    "type": "step_milestone",
                    "value": milestone,
                    "accuracy": snapshot.accuracy,
                    "timestamp": snapshot.timestamp
                })
        
        # Call milestone callbacks
        for milestone in milestones:
            logger.info(f"Milestone reached: {milestone['type']} = {milestone['value']}")
            for callback in self.milestone_callbacks:
                try:
                    callback(milestone)
                except Exception as e:
                    logger.error(f"Milestone callback error: {e}")
    
    def _log_progress(self, snapshot: TrainingProgressSnapshot) -> None:
        """Log training progress."""
        
        elapsed_time = ""
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            elapsed_time = f", Elapsed: {elapsed}"
        
        logger.info(
            f"Step {snapshot.step}, Epoch {snapshot.epoch}: "
            f"Loss={snapshot.loss:.4f}, Acc={snapshot.accuracy:.4f}, "
            f"LR={snapshot.learning_rate:.2e}, "
            f"Mem={snapshot.memory_usage_percent:.1f}%, "
            f"Speed={snapshot.samples_per_second:.1f} samples/sec"
            f"{elapsed_time}"
        )
        
        # Log per-category accuracies periodically
        if snapshot.step % 500 == 0 and snapshot.category_accuracies:
            logger.info("Per-category accuracies:")
            for category, accuracy in snapshot.category_accuracies.items():
                logger.info(f"  Category {category}: {accuracy:.4f}")
    
    def _save_progress_snapshots(self) -> None:
        """Save progress snapshots to file."""
        
        if not self.progress_snapshots:
            return
        
        snapshots_file = self.output_dir / f"{self.current_training_id}_snapshots.json"
        
        try:
            # Convert snapshots to serializable format
            serializable_snapshots = []
            for snapshot in self.progress_snapshots:
                snapshot_dict = asdict(snapshot)
                snapshot_dict['timestamp'] = snapshot.timestamp.isoformat()
                serializable_snapshots.append(snapshot_dict)
            
            with open(snapshots_file, 'w') as f:
                json.dump(serializable_snapshots, f, indent=2)
            
            logger.debug(f"Saved {len(self.progress_snapshots)} progress snapshots")
            
        except Exception as e:
            logger.error(f"Failed to save progress snapshots: {e}")
    
    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final monitoring summary."""
        
        if not self.progress_snapshots:
            return {"error": "No progress snapshots available"}
        
        final_snapshot = self.progress_snapshots[-1]
        total_time = (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0
        
        summary = {
            "training_id": self.current_training_id,
            "monitoring_duration": total_time,
            "total_snapshots": len(self.progress_snapshots),
            "total_steps_completed": self.total_steps_completed,
            "final_metrics": {
                "accuracy": final_snapshot.accuracy,
                "loss": final_snapshot.loss,
                "learning_rate": final_snapshot.learning_rate,
                "category_accuracies": final_snapshot.category_accuracies
            },
            "best_metrics": {
                "accuracy": self.best_accuracy,
                "loss": self.best_loss
            },
            "resource_usage": {
                "peak_memory_mb": max(s.memory_usage_mb for s in self.progress_snapshots),
                "avg_memory_percent": np.mean([s.memory_usage_percent for s in self.progress_snapshots]),
                "avg_cpu_percent": np.mean([s.cpu_usage_percent for s in self.progress_snapshots])
            },
            "training_speed": {
                "avg_samples_per_second": np.mean([s.samples_per_second for s in self.progress_snapshots if s.samples_per_second > 0]),
                "avg_steps_per_second": np.mean([s.steps_per_second for s in self.progress_snapshots if s.steps_per_second > 0])
            },
            "model_metrics": {
                "avg_gradient_norm": np.mean([s.gradient_norm for s in self.progress_snapshots if s.gradient_norm > 0]),
                "avg_reasoning_cycles": np.mean([s.num_reasoning_cycles for s in self.progress_snapshots if s.num_reasoning_cycles > 0])
            },
            "confusion_matrices_generated": len(self.confusion_matrices),
            "performance_reports_generated": len(self.performance_reports)
        }
        
        return summary
    
    def _save_monitoring_results(self, summary: Dict[str, Any]) -> None:
        """Save final monitoring results."""
        
        results_file = self.output_dir / f"{self.current_training_id}_monitoring_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Monitoring results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring results: {e}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard display."""
        
        if not self.progress_snapshots:
            return {"error": "No data available"}
        
        latest = self.progress_snapshots[-1]
        
        # Get recent progress (last 10 snapshots)
        recent_snapshots = self.progress_snapshots[-10:]
        
        dashboard_data = {
            "current_metrics": {
                "step": latest.step,
                "epoch": latest.epoch,
                "accuracy": latest.accuracy,
                "loss": latest.loss,
                "learning_rate": latest.learning_rate,
                "memory_usage_percent": latest.memory_usage_percent,
                "cpu_usage_percent": latest.cpu_usage_percent,
                "samples_per_second": latest.samples_per_second
            },
            "progress_trend": {
                "accuracy_trend": [s.accuracy for s in recent_snapshots],
                "loss_trend": [s.loss for s in recent_snapshots],
                "memory_trend": [s.memory_usage_percent for s in recent_snapshots],
                "speed_trend": [s.samples_per_second for s in recent_snapshots]
            },
            "category_performance": latest.category_accuracies,
            "training_time": (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0,
            "estimated_time_remaining": latest.estimated_time_remaining
        }
        
        return dashboard_data


def create_monitoring_callbacks() -> Dict[str, Callable]:
    """Create example monitoring callbacks."""
    
    callbacks = {}
    
    def accuracy_alert_callback(snapshot: TrainingProgressSnapshot):
        """Alert when accuracy drops significantly."""
        if len(monitor.progress_snapshots) > 10:
            recent_accuracies = [s.accuracy for s in monitor.progress_snapshots[-10:]]
            if max(recent_accuracies) - min(recent_accuracies) > 0.1:
                logger.warning(f"Accuracy instability detected at step {snapshot.step}")
    
    def memory_alert_callback(snapshot: TrainingProgressSnapshot):
        """Alert when memory usage is high."""
        if snapshot.memory_usage_percent > 85:
            logger.warning(f"High memory usage: {snapshot.memory_usage_percent:.1f}% at step {snapshot.step}")
    
    def milestone_celebration_callback(milestone: Dict[str, Any]):
        """Celebrate milestone achievements."""
        logger.info(f"🎉 Milestone achieved: {milestone['type']} = {milestone['value']}")
    
    callbacks["accuracy_alert"] = accuracy_alert_callback
    callbacks["memory_alert"] = memory_alert_callback
    callbacks["milestone_celebration"] = milestone_celebration_callback
    
    return callbacks


def main():
    """Main function to demonstrate training progress monitoring."""
    
    logger.info("Starting training progress monitoring demonstration...")
    
    # Initialize monitor
    monitor = TrainingProgressMonitor(
        output_dir="training_monitoring_demo",
        update_interval=5.0,
        save_interval=30.0
    )
    
    # Add example callbacks
    callbacks = create_monitoring_callbacks()
    monitor.add_progress_callback(callbacks["accuracy_alert"])
    monitor.add_progress_callback(callbacks["memory_alert"])
    monitor.add_milestone_callback(callbacks["milestone_celebration"])
    
    # Simulate training monitoring
    training_id = f"demo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Start monitoring
        monitor.start_monitoring(training_id, total_steps=1000, total_epochs=5)
        
        # Simulate training progress
        logger.info("Simulating training progress...")
        
        for step in range(0, 1001, 50):
            # Simulate training metrics
            epoch = step // 200
            
            # Simulate improving metrics
            base_accuracy = min(0.95, 0.5 + (step / 1000) * 0.45)
            base_loss = max(0.1, 2.0 - (step / 1000) * 1.9)
            
            # Add some noise
            import random
            accuracy = base_accuracy + random.uniform(-0.02, 0.02)
            loss = base_loss + random.uniform(-0.1, 0.1)
            
            # Create mock metrics
            metrics = EmailTrainingMetrics(
                loss=loss,
                accuracy=accuracy,
                learning_rate=1e-4 * (0.95 ** (step // 100)),
                gradient_norm=random.uniform(0.5, 2.0),
                num_reasoning_cycles=random.uniform(2.0, 4.0),
                samples_per_second=random.uniform(8.0, 12.0),
                memory_usage_mb=random.uniform(2000, 4000),
                memory_usage_percent=random.uniform(40, 80),
                category_accuracies={
                    0: accuracy + random.uniform(-0.05, 0.05),
                    1: accuracy + random.uniform(-0.05, 0.05),
                    2: accuracy + random.uniform(-0.05, 0.05),
                    3: accuracy + random.uniform(-0.05, 0.05),
                    4: accuracy + random.uniform(-0.05, 0.05)
                }
            )
            
            # Update progress
            monitor.update_progress(metrics, step, epoch)
            
            # Simulate training delay
            time.sleep(0.1)
        
        # Generate confusion matrix
        logger.info("Generating confusion matrix...")
        predictions = np.random.randint(0, 5, 1000)
        labels = np.random.randint(0, 5, 1000)
        category_names = ["Newsletter", "Work", "Personal", "Spam", "Promotional"]
        
        confusion_matrix = monitor.generate_confusion_matrix(predictions, labels, category_names)
        
        # Generate performance report
        logger.info("Generating performance report...")
        performance_report = monitor.generate_performance_report(confusion_matrix)
        
        logger.info(f"Overall accuracy: {performance_report['overall_accuracy']:.4f}")
        logger.info("Per-category performance:")
        for category, metrics in performance_report["category_performance"].items():
            logger.info(f"  {category}: P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        # Get dashboard data
        dashboard_data = monitor.get_real_time_dashboard_data()
        logger.info("Real-time dashboard data:")
        logger.info(f"  Current accuracy: {dashboard_data['current_metrics']['accuracy']:.4f}")
        logger.info(f"  Memory usage: {dashboard_data['current_metrics']['memory_usage_percent']:.1f}%")
        logger.info(f"  Training speed: {dashboard_data['current_metrics']['samples_per_second']:.1f} samples/sec")
        
        # Stop monitoring
        final_summary = monitor.stop_monitoring()
        
        logger.info("✅ Training progress monitoring demonstration completed!")
        logger.info(f"Total snapshots: {final_summary['total_snapshots']}")
        logger.info(f"Final accuracy: {final_summary['final_metrics']['accuracy']:.4f}")
        logger.info(f"Best accuracy: {final_summary['best_metrics']['accuracy']:.4f}")
        logger.info(f"Average training speed: {final_summary['training_speed']['avg_samples_per_second']:.1f} samples/sec")
        
    except Exception as e:
        logger.error(f"Training progress monitoring demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()