"""
Email-specific training loop optimized for MacBook hardware.

This module provides a comprehensive training loop for email classification
with MacBook optimizations, gradient accumulation, memory pressure handling,
and email-specific loss functions and accuracy tracking.
"""

import time
import math
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager, MemoryPressureInfo
from .resource_monitoring import ResourceMonitor
from .email_trm_integration import MacBookEmailTRM
from .progress_monitoring import ProgressMonitor

logger = logging.getLogger(__name__)


@dataclass
class EmailTrainingConfig:
    """Configuration for email training loop."""
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    max_steps: Optional[int] = None
    
    # Email-specific parameters
    num_email_categories: int = 10
    target_accuracy: float = 0.95
    min_category_accuracy: float = 0.90
    
    # MacBook optimization parameters
    enable_memory_monitoring: bool = True
    dynamic_batch_sizing: bool = True
    memory_pressure_threshold: float = 85.0
    
    # Loss function weights
    classification_loss_weight: float = 1.0
    halt_loss_weight: float = 0.01
    contrastive_loss_weight: float = 0.1
    calibration_loss_weight: float = 0.05
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_interval: int = 500
    save_best_model: bool = True
    
    # Logging and monitoring
    log_interval: int = 50
    eval_interval: int = 200
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "exponential"
    warmup_steps: int = 100


@dataclass
class EmailTrainingMetrics:
    """Training metrics for email classification."""
    
    # Overall metrics
    loss: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    
    # Per-category metrics
    category_accuracies: Dict[int, float] = None
    category_f1_scores: Dict[int, float] = None
    category_precisions: Dict[int, float] = None
    category_recalls: Dict[int, float] = None
    
    # Training-specific metrics
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    num_reasoning_cycles: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Performance metrics
    samples_per_second: float = 0.0
    
    def __post_init__(self):
        if self.category_accuracies is None:
            self.category_accuracies = {}
        if self.category_f1_scores is None:
            self.category_f1_scores = {}
        if self.category_precisions is None:
            self.category_precisions = {}
        if self.category_recalls is None:
            self.category_recalls = {}


class EmailTrainingLoop:
    """MacBook-optimized training loop for email classification."""
    
    def __init__(self, model: MacBookEmailTRM, config: EmailTrainingConfig):
        """
        Initialize email training loop.
        
        Args:
            model: MacBook-optimized EmailTRM model
            config: Training configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        self.model = model
        self.config = config
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.resource_monitor = ResourceMonitor()
        self.progress_monitor = ProgressMonitor(self.resource_monitor, self.memory_manager)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_accuracy = 0.0
        self.best_f1_macro = 0.0
        self.early_stopping_counter = 0
        
        # Optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.lr_scheduler = None
        
        # Category tracking
        self.category_correct = torch.zeros(config.num_email_categories)
        self.category_total = torch.zeros(config.num_email_categories)
        
        # Memory pressure handling
        self.original_batch_size = None
        self.current_batch_size = None
        
        # Callbacks
        self.step_callbacks: List[Callable] = []
        self.epoch_callbacks: List[Callable] = []
        
        # Start monitoring
        if config.enable_memory_monitoring:
            self.resource_monitor.start_monitoring()
            self.memory_manager.add_memory_pressure_callback(self._handle_memory_pressure)
        
        logger.info("EmailTrainingLoop initialized")
    
    def add_step_callback(self, callback: Callable):
        """Add callback to be called after each training step."""
        self.step_callbacks.append(callback)
    
    def add_epoch_callback(self, callback: Callable):
        """Add callback to be called after each epoch."""
        self.epoch_callbacks.append(callback)
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create learning rate scheduler
        if self.config.use_lr_scheduler:
            if self.config.lr_scheduler_type == "cosine":
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.max_epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
            elif self.config.lr_scheduler_type == "linear":
                self.lr_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.01,
                    total_iters=self.config.max_epochs
                )
            elif self.config.lr_scheduler_type == "exponential":
                self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=0.95
                )
        
        logger.info(f"Optimizer and scheduler setup complete")
    
    def _handle_memory_pressure(self, memory_info: MemoryPressureInfo):
        """Handle memory pressure by adjusting batch size."""
        if not self.config.dynamic_batch_sizing:
            return
        
        if memory_info.pressure_level in ["high", "critical"]:
            if self.current_batch_size is not None and self.current_batch_size > 1:
                new_batch_size = max(1, int(self.current_batch_size * 0.75))
                logger.warning(f"Memory pressure detected ({memory_info.pressure_level}). "
                             f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
                
                # Force garbage collection
                self.memory_manager.force_garbage_collection()
    
    def _compute_email_loss(self, outputs: Dict[str, torch.Tensor], 
                           labels: torch.Tensor) -> torch.Tensor:
        """
        Compute email-specific loss with multiple components.
        
        Args:
            outputs: Model outputs dictionary
            labels: Target labels [batch_size]
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        
        # Main classification loss
        if "loss" in outputs:
            classification_loss = outputs["loss"]
        else:
            classification_loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
        
        total_loss += self.config.classification_loss_weight * classification_loss
        
        # Halt regularization loss
        if "halt_logits" in outputs:
            halt_probs = torch.sigmoid(outputs["halt_logits"])
            # Encourage efficient halting (not too early, not too late)
            halt_target = torch.full_like(halt_probs, 0.6)
            halt_loss = nn.MSELoss()(halt_probs, halt_target)
            total_loss += self.config.halt_loss_weight * halt_loss
        
        # Contrastive loss for category embeddings
        if "similarity_scores" in outputs:
            similarity_scores = outputs["similarity_scores"]
            target_similarities = torch.zeros_like(similarity_scores)
            target_similarities.scatter_(1, labels.unsqueeze(1), 1.0)
            contrastive_loss = nn.MSELoss()(similarity_scores, target_similarities)
            total_loss += self.config.contrastive_loss_weight * contrastive_loss
        
        # Confidence calibration loss
        if "calibrated_logits" in outputs:
            calibrated_loss = nn.CrossEntropyLoss()(outputs["calibrated_logits"], labels)
            calibration_loss = torch.abs(calibrated_loss - classification_loss)
            total_loss += self.config.calibration_loss_weight * calibration_loss
        
        return total_loss
    
    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], 
                        labels: torch.Tensor) -> EmailTrainingMetrics:
        """
        Compute comprehensive training metrics.
        
        Args:
            outputs: Model outputs dictionary
            labels: Target labels [batch_size]
            
        Returns:
            Training metrics
        """
        metrics = EmailTrainingMetrics()
        
        # Get predictions
        logits = outputs.get("calibrated_logits", outputs["logits"])
        predictions = torch.argmax(logits, dim=-1)
        
        # Overall accuracy
        correct = (predictions == labels).float()
        metrics.accuracy = correct.mean().item()
        
        # Loss
        metrics.loss = self._compute_email_loss(outputs, labels).item()
        
        # Per-category metrics
        for category in range(self.config.num_email_categories):
            category_mask = (labels == category)
            if category_mask.sum() > 0:
                category_correct = (predictions[category_mask] == labels[category_mask]).float()
                metrics.category_accuracies[category] = category_correct.mean().item()
                
                # Update running totals
                self.category_correct[category] += category_correct.sum()
                self.category_total[category] += category_mask.sum()
        
        # F1 scores (simplified calculation)
        # For proper F1, you'd need precision and recall per class
        metrics.f1_macro = metrics.accuracy  # Simplified
        metrics.f1_micro = metrics.accuracy  # Simplified
        
        # Training-specific metrics
        if self.optimizer:
            metrics.learning_rate = self.optimizer.param_groups[0]['lr']
        
        # Gradient norm (if available)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        metrics.gradient_norm = total_norm ** (1. / 2)
        
        # Reasoning cycles
        if "num_cycles" in outputs:
            metrics.num_reasoning_cycles = outputs["num_cycles"]
        
        # Memory metrics
        if "memory_usage" in outputs:
            memory_info = outputs["memory_usage"]
            metrics.memory_usage_mb = memory_info.get("after_mb", 0.0)
        
        memory_stats = self.memory_manager.monitor_memory_usage()
        metrics.memory_usage_percent = memory_stats.percent_used
        
        return metrics
    
    def _should_stop_early(self, val_metrics: EmailTrainingMetrics) -> bool:
        """Check if training should stop early."""
        if not hasattr(self, '_best_val_accuracy'):
            self._best_val_accuracy = 0.0
            self._steps_without_improvement = 0
        
        if val_metrics.accuracy > self._best_val_accuracy + self.config.early_stopping_min_delta:
            self._best_val_accuracy = val_metrics.accuracy
            self._steps_without_improvement = 0
            return False
        else:
            self._steps_without_improvement += 1
            return self._steps_without_improvement >= self.config.early_stopping_patience
    
    def _check_accuracy_target(self, metrics: EmailTrainingMetrics) -> bool:
        """Check if accuracy target has been reached."""
        # Check overall accuracy
        if metrics.accuracy < self.config.target_accuracy:
            return False
        
        # Check per-category accuracy
        for category, accuracy in metrics.category_accuracies.items():
            if accuracy < self.config.min_category_accuracy:
                return False
        
        return True
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> EmailTrainingMetrics:
        """
        Execute a single training step.
        
        Args:
            batch: Training batch dictionary
            
        Returns:
            Training metrics for this step
        """
        self.model.train()
        
        inputs = batch["input_ids"]
        labels = batch["labels"]
        puzzle_identifiers = batch.get("puzzle_identifiers")
        
        # Forward pass
        outputs = self.model(inputs, labels=labels, puzzle_identifiers=puzzle_identifiers)
        
        # Compute loss
        loss = self._compute_email_loss(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Learning rate scheduler step
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, labels)
        
        # Update step counter
        self.current_step += 1
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> EmailTrainingMetrics:
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        # Reset category counters
        category_correct = torch.zeros(self.config.num_email_categories)
        category_total = torch.zeros(self.config.num_email_categories)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"]
                labels = batch["labels"]
                puzzle_identifiers = batch.get("puzzle_identifiers")
                
                # Forward pass
                outputs = self.model(inputs, labels=labels, puzzle_identifiers=puzzle_identifiers)
                
                # Accumulate loss
                loss = self._compute_email_loss(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Get predictions
                logits = outputs.get("calibrated_logits", outputs["logits"])
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Per-category metrics
                for category in range(self.config.num_email_categories):
                    category_mask = (labels == category)
                    if category_mask.sum() > 0:
                        category_correct[category] += (predictions[category_mask] == labels[category_mask]).sum()
                        category_total[category] += category_mask.sum()
        
        # Compute final metrics
        metrics = EmailTrainingMetrics()
        metrics.loss = total_loss / total_samples
        
        # Overall accuracy
        correct_predictions = sum(p == l for p, l in zip(all_predictions, all_labels))
        metrics.accuracy = correct_predictions / len(all_labels)
        
        # Per-category accuracies
        for category in range(self.config.num_email_categories):
            if category_total[category] > 0:
                metrics.category_accuracies[category] = (category_correct[category] / category_total[category]).item()
            else:
                metrics.category_accuracies[category] = 0.0
        
        # Simplified F1 scores
        metrics.f1_macro = metrics.accuracy
        metrics.f1_micro = metrics.accuracy
        
        # Memory metrics
        memory_stats = self.memory_manager.monitor_memory_usage()
        metrics.memory_usage_percent = memory_stats.percent_used
        metrics.memory_usage_mb = memory_stats.used_mb
        
        return metrics
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Execute complete training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting email classification training")
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Initialize tracking
        if hasattr(train_dataloader, 'batch_size'):
            self.original_batch_size = train_dataloader.batch_size
        else:
            # Handle list-based dataloader (for testing)
            self.original_batch_size = len(train_dataloader[0]["input_ids"]) if train_dataloader else 4
        self.current_batch_size = self.original_batch_size
        
        training_history = {
            "train_metrics": [],
            "val_metrics": [],
            "best_accuracy": 0.0,
            "target_reached": False,
            "early_stopped": False
        }
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                self.model.train()
                epoch_metrics = []
                
                for batch_idx, batch in enumerate(train_dataloader):
                    step_start_time = time.time()
                    
                    # Training step
                    step_metrics = self.train_step(batch)
                    epoch_metrics.append(step_metrics)
                    
                    # Performance tracking
                    step_time = time.time() - step_start_time
                    step_metrics.samples_per_second = batch["input_ids"].size(0) / step_time
                    
                    # Logging
                    if self.current_step % self.config.log_interval == 0:
                        logger.info(
                            f"Epoch {epoch}, Step {self.current_step}: "
                            f"Loss={step_metrics.loss:.4f}, "
                            f"Acc={step_metrics.accuracy:.4f}, "
                            f"LR={step_metrics.learning_rate:.2e}, "
                            f"Mem={step_metrics.memory_usage_percent:.1f}%"
                        )
                    
                    # Step callbacks
                    for callback in self.step_callbacks:
                        callback(step_metrics)
                    
                    # Evaluation
                    if val_dataloader and self.current_step % self.config.eval_interval == 0:
                        val_metrics = self.evaluate(val_dataloader)
                        training_history["val_metrics"].append(val_metrics)
                        
                        logger.info(
                            f"Validation - Acc={val_metrics.accuracy:.4f}, "
                            f"Loss={val_metrics.loss:.4f}"
                        )
                        
                        # Check for early stopping
                        if self._should_stop_early(val_metrics):
                            logger.info("Early stopping triggered")
                            training_history["early_stopped"] = True
                            break
                        
                        # Check accuracy target
                        if self._check_accuracy_target(val_metrics):
                            logger.info(f"Target accuracy {self.config.target_accuracy} reached!")
                            training_history["target_reached"] = True
                            break
                    
                    # Max steps check
                    if self.config.max_steps and self.current_step >= self.config.max_steps:
                        logger.info(f"Maximum steps ({self.config.max_steps}) reached")
                        break
                
                # Epoch summary
                if epoch_metrics:
                    avg_metrics = self._average_metrics(epoch_metrics)
                    training_history["train_metrics"].append(avg_metrics)
                    
                    epoch_time = time.time() - epoch_start_time
                    logger.info(
                        f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                        f"Avg Loss={avg_metrics.loss:.4f}, "
                        f"Avg Acc={avg_metrics.accuracy:.4f}"
                    )
                
                # Epoch callbacks
                for callback in self.epoch_callbacks:
                    callback(epoch, avg_metrics if epoch_metrics else None)
                
                # Early stopping or target reached
                if training_history.get("early_stopped") or training_history.get("target_reached"):
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        
        finally:
            # Cleanup
            if self.config.enable_memory_monitoring:
                self.resource_monitor.stop_monitoring()
        
        # Final evaluation
        if val_dataloader:
            final_val_metrics = self.evaluate(val_dataloader)
            training_history["final_val_metrics"] = final_val_metrics
            training_history["best_accuracy"] = final_val_metrics.accuracy
        
        total_time = time.time() - start_time
        training_history["total_training_time"] = total_time
        
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return training_history
    
    def _average_metrics(self, metrics_list: List[EmailTrainingMetrics]) -> EmailTrainingMetrics:
        """Average a list of metrics."""
        if not metrics_list:
            return EmailTrainingMetrics()
        
        avg_metrics = EmailTrainingMetrics()
        
        # Simple averages
        avg_metrics.loss = sum(m.loss for m in metrics_list) / len(metrics_list)
        avg_metrics.accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list)
        avg_metrics.f1_macro = sum(m.f1_macro for m in metrics_list) / len(metrics_list)
        avg_metrics.f1_micro = sum(m.f1_micro for m in metrics_list) / len(metrics_list)
        avg_metrics.learning_rate = metrics_list[-1].learning_rate  # Use last LR
        avg_metrics.gradient_norm = sum(m.gradient_norm for m in metrics_list) / len(metrics_list)
        avg_metrics.num_reasoning_cycles = sum(m.num_reasoning_cycles for m in metrics_list) / len(metrics_list)
        avg_metrics.memory_usage_mb = sum(m.memory_usage_mb for m in metrics_list) / len(metrics_list)
        avg_metrics.memory_usage_percent = sum(m.memory_usage_percent for m in metrics_list) / len(metrics_list)
        avg_metrics.samples_per_second = sum(m.samples_per_second for m in metrics_list) / len(metrics_list)
        
        # Category metrics (average non-zero values)
        for category in range(self.config.num_email_categories):
            category_accs = [m.category_accuracies.get(category, 0) for m in metrics_list if category in m.category_accuracies]
            if category_accs:
                avg_metrics.category_accuracies[category] = sum(category_accs) / len(category_accs)
        
        return avg_metrics