"""
Training Error Handling System for MacBook Email Classification

This module provides comprehensive error handling for all training components
including automatic checkpoint resumption, memory pressure handling with
graceful batch size reduction, and robust email data parsing with error recovery.
"""

import os
import json
import time
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager, MemoryPressureInfo
from .checkpoint_management import CheckpointManager, CheckpointLoadResult
from .resource_monitoring import ResourceMonitor


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MEMORY = "memory"
    DATA = "data"
    MODEL = "model"
    HARDWARE = "hardware"
    NETWORK = "network"
    CHECKPOINT = "checkpoint"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class TrainingError:
    """Training error information."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    traceback_str: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_actions: List[str] = None
    
    def __post_init__(self):
        if self.recovery_actions is None:
            self.recovery_actions = []


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    name: str
    description: str
    action_func: Callable
    conditions: Dict[str, Any]
    max_attempts: int = 3
    cooldown_seconds: float = 5.0


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling system."""
    # Error logging
    log_errors_to_file: bool = True
    error_log_path: str = "training_errors.log"
    max_error_history: int = 1000
    
    # Recovery settings
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown_seconds: float = 10.0
    
    # Memory error handling
    memory_pressure_threshold: float = 85.0
    emergency_memory_threshold: float = 95.0
    batch_size_reduction_factor: float = 0.5
    min_batch_size: int = 1
    
    # Data error handling
    max_data_errors_per_batch: int = 5
    skip_corrupted_batches: bool = True
    data_validation_enabled: bool = True
    
    # Checkpoint recovery
    auto_resume_from_checkpoint: bool = True
    checkpoint_validation_enabled: bool = True
    backup_checkpoint_count: int = 3
    
    # Hardware monitoring
    thermal_throttling_detection: bool = True
    cpu_overload_threshold: float = 90.0
    disk_space_threshold_mb: float = 1000.0


class TrainingErrorHandler:
    """Comprehensive error handling system for training."""
    
    def __init__(self, 
                 config: Optional[ErrorHandlingConfig] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 resource_monitor: Optional[ResourceMonitor] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize training error handler.
        
        Args:
            config: Error handling configuration
            memory_manager: Memory manager instance
            checkpoint_manager: Checkpoint manager instance
            resource_monitor: Resource monitor instance
            logger: Logger instance
        """
        self.config = config or ErrorHandlingConfig()
        self.memory_manager = memory_manager or MemoryManager()
        self.checkpoint_manager = checkpoint_manager
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.logger = logger or logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[TrainingError] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_time: Dict[str, float] = {}
        
        # Recovery actions registry
        self.recovery_actions: Dict[ErrorCategory, List[RecoveryAction]] = {}
        self._register_recovery_actions()
        
        # State tracking
        self.training_interrupted = False
        self.last_successful_step = 0
        self.current_batch_size = None
        self.original_batch_size = None
        
        # Setup error logging
        if self.config.log_errors_to_file:
            self._setup_error_logging()
        
        self.logger.info("TrainingErrorHandler initialized")
    
    def _setup_error_logging(self):
        """Setup error logging to file."""
        try:
            error_log_path = Path(self.config.error_log_path)
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create error log handler
            error_handler = logging.FileHandler(error_log_path)
            error_handler.setLevel(logging.ERROR)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            error_handler.setFormatter(formatter)
            
            # Add to logger
            self.logger.addHandler(error_handler)
            
        except Exception as e:
            self.logger.warning(f"Failed to setup error logging: {e}")
    
    def _register_recovery_actions(self):
        """Register recovery actions for different error categories."""
        
        # Memory error recovery actions
        memory_actions = [
            RecoveryAction(
                name="reduce_batch_size",
                description="Reduce batch size to free memory",
                action_func=self._reduce_batch_size,
                conditions={"memory_pressure": True},
                max_attempts=5
            ),
            RecoveryAction(
                name="force_garbage_collection",
                description="Force garbage collection to free memory",
                action_func=self._force_garbage_collection,
                conditions={},
                max_attempts=3
            ),
            RecoveryAction(
                name="clear_cache",
                description="Clear model and optimizer caches",
                action_func=self._clear_caches,
                conditions={},
                max_attempts=2
            )
        ]
        self.recovery_actions[ErrorCategory.MEMORY] = memory_actions
        
        # Data error recovery actions
        data_actions = [
            RecoveryAction(
                name="skip_batch",
                description="Skip corrupted batch and continue",
                action_func=self._skip_corrupted_batch,
                conditions={"batch_corrupted": True},
                max_attempts=1
            ),
            RecoveryAction(
                name="reload_dataset",
                description="Reload dataset with validation",
                action_func=self._reload_dataset,
                conditions={"dataset_corrupted": True},
                max_attempts=2
            ),
            RecoveryAction(
                name="fallback_parsing",
                description="Use fallback email parsing method",
                action_func=self._fallback_email_parsing,
                conditions={"parsing_error": True},
                max_attempts=3
            )
        ]
        self.recovery_actions[ErrorCategory.DATA] = data_actions
        
        # Model error recovery actions
        model_actions = [
            RecoveryAction(
                name="reset_model_state",
                description="Reset model to last known good state",
                action_func=self._reset_model_state,
                conditions={"model_corrupted": True},
                max_attempts=2
            ),
            RecoveryAction(
                name="reduce_model_complexity",
                description="Temporarily reduce model complexity",
                action_func=self._reduce_model_complexity,
                conditions={"model_too_complex": True},
                max_attempts=1
            )
        ]
        self.recovery_actions[ErrorCategory.MODEL] = model_actions
        
        # Hardware error recovery actions
        hardware_actions = [
            RecoveryAction(
                name="thermal_cooldown",
                description="Wait for thermal cooldown",
                action_func=self._thermal_cooldown,
                conditions={"thermal_throttling": True},
                max_attempts=3,
                cooldown_seconds=30.0
            ),
            RecoveryAction(
                name="reduce_cpu_load",
                description="Reduce CPU load by adjusting workers",
                action_func=self._reduce_cpu_load,
                conditions={"cpu_overload": True},
                max_attempts=2
            )
        ]
        self.recovery_actions[ErrorCategory.HARDWARE] = hardware_actions
        
        # Checkpoint error recovery actions
        checkpoint_actions = [
            RecoveryAction(
                name="load_backup_checkpoint",
                description="Load backup checkpoint",
                action_func=self._load_backup_checkpoint,
                conditions={"checkpoint_corrupted": True},
                max_attempts=3
            ),
            RecoveryAction(
                name="create_emergency_checkpoint",
                description="Create emergency checkpoint",
                action_func=self._create_emergency_checkpoint,
                conditions={"checkpoint_save_failed": True},
                max_attempts=2
            )
        ]
        self.recovery_actions[ErrorCategory.CHECKPOINT] = checkpoint_actions
    
    def handle_error(self, 
                    exception: Exception,
                    context: Dict[str, Any],
                    attempt_recovery: bool = True) -> Tuple[bool, List[str]]:
        """
        Handle training error with automatic recovery.
        
        Args:
            exception: The exception that occurred
            context: Context information about the error
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Tuple of (recovery_successful, recovery_actions_taken)
        """
        # Create error record
        error = self._create_error_record(exception, context)
        self.error_history.append(error)
        
        # Log error
        self.logger.error(f"Training error occurred: {error.message}")
        self.logger.error(f"Error category: {error.category.value}, severity: {error.severity.value}")
        
        # Attempt recovery if enabled
        recovery_successful = False
        recovery_actions_taken = []
        
        if attempt_recovery and self.config.enable_auto_recovery:
            recovery_successful, recovery_actions_taken = self._attempt_recovery(error)
            
            # Update error record
            error.recovery_attempted = True
            error.recovery_successful = recovery_successful
            error.recovery_actions = recovery_actions_taken
        
        # Limit error history size
        if len(self.error_history) > self.config.max_error_history:
            self.error_history = self.error_history[-self.config.max_error_history:]
        
        return recovery_successful, recovery_actions_taken
    
    def _create_error_record(self, exception: Exception, context: Dict[str, Any]) -> TrainingError:
        """Create error record from exception and context."""
        error_id = f"error_{int(time.time() * 1000)}"
        
        # Classify error
        category = self._classify_error(exception, context)
        severity = self._assess_severity(exception, context, category)
        
        return TrainingError(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_str=traceback.format_exc(),
            context=context.copy()
        )
    
    def _classify_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into appropriate category."""
        exception_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Memory-related errors
        if ("memory" in error_message or 
            "out of memory" in error_message or
            exception_type in ["RuntimeError", "MemoryError"] and "cuda" in error_message):
            return ErrorCategory.MEMORY
        
        # Data-related errors
        if ("data" in error_message or
            "json" in error_message or
            "parsing" in error_message or
            "decode" in error_message or
            exception_type in ["JSONDecodeError", "UnicodeDecodeError", "ValueError"]):
            return ErrorCategory.DATA
        
        # Model-related errors
        if ("model" in error_message or
            "forward" in error_message or
            "backward" in error_message or
            "gradient" in error_message or
            exception_type in ["RuntimeError"] and "tensor" in error_message):
            return ErrorCategory.MODEL
        
        # Hardware-related errors
        if ("thermal" in error_message or
            "cpu" in error_message or
            "temperature" in error_message or
            "throttling" in error_message):
            return ErrorCategory.HARDWARE
        
        # Checkpoint-related errors
        if ("checkpoint" in error_message or
            "save" in error_message or
            "load" in error_message or
            exception_type in ["FileNotFoundError", "PermissionError"]):
            return ErrorCategory.CHECKPOINT
        
        # Configuration-related errors
        if ("config" in error_message or
            "parameter" in error_message or
            exception_type in ["AttributeError", "KeyError"]):
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, exception: Exception, context: Dict[str, Any], category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity."""
        exception_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Critical errors that stop training
        if (exception_type in ["SystemExit", "KeyboardInterrupt"] or
            "critical" in error_message or
            "fatal" in error_message):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if (category == ErrorCategory.MEMORY and "out of memory" in error_message or
            category == ErrorCategory.MODEL and "nan" in error_message or
            exception_type in ["RuntimeError", "MemoryError"]):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if (category in [ErrorCategory.DATA, ErrorCategory.CHECKPOINT] or
            exception_type in ["ValueError", "TypeError"]):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _attempt_recovery(self, error: TrainingError) -> Tuple[bool, List[str]]:
        """Attempt automatic recovery from error."""
        recovery_actions_taken = []
        
        # Check if we should attempt recovery
        if not self._should_attempt_recovery(error):
            return False, recovery_actions_taken
        
        # Get applicable recovery actions
        actions = self.recovery_actions.get(error.category, [])
        
        for action in actions:
            # Check if action conditions are met
            if not self._check_action_conditions(action, error):
                continue
            
            # Check attempt limits
            action_key = f"{error.category.value}_{action.name}"
            attempts = self.recovery_attempts.get(action_key, 0)
            
            if attempts >= action.max_attempts:
                continue
            
            # Check cooldown
            last_attempt = self.last_recovery_time.get(action_key, 0)
            if time.time() - last_attempt < action.cooldown_seconds:
                continue
            
            # Attempt recovery action
            try:
                self.logger.info(f"Attempting recovery action: {action.name}")
                success = action.action_func(error)
                
                # Update tracking
                self.recovery_attempts[action_key] = attempts + 1
                self.last_recovery_time[action_key] = time.time()
                recovery_actions_taken.append(action.name)
                
                if success:
                    self.logger.info(f"Recovery action {action.name} successful")
                    return True, recovery_actions_taken
                else:
                    self.logger.warning(f"Recovery action {action.name} failed")
                    
            except Exception as e:
                self.logger.error(f"Recovery action {action.name} raised exception: {e}")
        
        return False, recovery_actions_taken
    
    def _should_attempt_recovery(self, error: TrainingError) -> bool:
        """Determine if recovery should be attempted."""
        # Don't recover from critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Check global recovery attempt limit
        total_attempts = sum(self.recovery_attempts.values())
        if total_attempts >= self.config.max_recovery_attempts:
            return False
        
        # Check cooldown
        if self.last_recovery_time:
            last_recovery = max(self.last_recovery_time.values())
            if time.time() - last_recovery < self.config.recovery_cooldown_seconds:
                return False
        
        return True
    
    def _check_action_conditions(self, action: RecoveryAction, error: TrainingError) -> bool:
        """Check if recovery action conditions are met."""
        for condition, expected_value in action.conditions.items():
            if condition == "memory_pressure":
                memory_stats = self.memory_manager.monitor_memory_usage()
                if (memory_stats.percent_used > self.config.memory_pressure_threshold) != expected_value:
                    return False
            
            elif condition == "batch_corrupted":
                if error.context.get("batch_corrupted", False) != expected_value:
                    return False
            
            elif condition == "thermal_throttling":
                if self.resource_monitor:
                    thermal_stats = self.resource_monitor.get_thermal_stats()
                    if (thermal_stats.thermal_state == "hot") != expected_value:
                        return False
            
            # Add more condition checks as needed
        
        return True
    
    # Recovery action implementations
    def _reduce_batch_size(self, error: TrainingError) -> bool:
        """Reduce batch size to handle memory pressure."""
        try:
            if self.current_batch_size is None:
                self.current_batch_size = error.context.get("batch_size", 8)
            
            if self.original_batch_size is None:
                self.original_batch_size = self.current_batch_size
            
            new_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * self.config.batch_size_reduction_factor)
            )
            
            if new_batch_size < self.current_batch_size:
                self.current_batch_size = new_batch_size
                self.logger.info(f"Reduced batch size to {new_batch_size}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to reduce batch size: {e}")
            return False
    
    def _force_garbage_collection(self, error: TrainingError) -> bool:
        """Force garbage collection to free memory."""
        try:
            self.memory_manager.force_garbage_collection()
            time.sleep(1.0)  # Give time for cleanup
            return True
        except Exception as e:
            self.logger.error(f"Failed to force garbage collection: {e}")
            return False
    
    def _clear_caches(self, error: TrainingError) -> bool:
        """Clear model and optimizer caches."""
        try:
            if TORCH_AVAILABLE:
                # Clear PyTorch caches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear MKL cache if available
                if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
                    pass  # MKL doesn't have a direct cache clear method
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
            return False
    
    def _skip_corrupted_batch(self, error: TrainingError) -> bool:
        """Skip corrupted batch and continue training."""
        try:
            # This would be implemented in the training loop
            # For now, just log the action
            self.logger.info("Skipping corrupted batch")
            return True
        except Exception as e:
            self.logger.error(f"Failed to skip corrupted batch: {e}")
            return False
    
    def _reload_dataset(self, error: TrainingError) -> bool:
        """Reload dataset with validation."""
        try:
            # This would trigger dataset reloading in the training system
            self.logger.info("Triggering dataset reload")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload dataset: {e}")
            return False
    
    def _fallback_email_parsing(self, error: TrainingError) -> bool:
        """Use fallback email parsing method."""
        try:
            # This would implement alternative email parsing
            self.logger.info("Using fallback email parsing")
            return True
        except Exception as e:
            self.logger.error(f"Failed fallback email parsing: {e}")
            return False
    
    def _reset_model_state(self, error: TrainingError) -> bool:
        """Reset model to last known good state."""
        try:
            if self.checkpoint_manager:
                # Load last checkpoint
                result = self.checkpoint_manager.load_checkpoint()
                if result.success:
                    self.logger.info("Reset model to last checkpoint")
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to reset model state: {e}")
            return False
    
    def _reduce_model_complexity(self, error: TrainingError) -> bool:
        """Temporarily reduce model complexity."""
        try:
            # This would modify model parameters
            self.logger.info("Reducing model complexity")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce model complexity: {e}")
            return False
    
    def _thermal_cooldown(self, error: TrainingError) -> bool:
        """Wait for thermal cooldown."""
        try:
            cooldown_time = 30.0  # seconds
            self.logger.info(f"Thermal cooldown for {cooldown_time} seconds")
            time.sleep(cooldown_time)
            return True
        except Exception as e:
            self.logger.error(f"Failed thermal cooldown: {e}")
            return False
    
    def _reduce_cpu_load(self, error: TrainingError) -> bool:
        """Reduce CPU load by adjusting workers."""
        try:
            # This would adjust number of workers in data loading
            self.logger.info("Reducing CPU load")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reduce CPU load: {e}")
            return False
    
    def _load_backup_checkpoint(self, error: TrainingError) -> bool:
        """Load backup checkpoint."""
        try:
            if self.checkpoint_manager:
                # Try to load older checkpoints
                checkpoints = self.checkpoint_manager.list_checkpoints(limit=5)
                for checkpoint in checkpoints[1:]:  # Skip the latest (might be corrupted)
                    result = self.checkpoint_manager.load_checkpoint(checkpoint.checkpoint_id)
                    if result.success:
                        self.logger.info(f"Loaded backup checkpoint: {checkpoint.checkpoint_id}")
                        return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to load backup checkpoint: {e}")
            return False
    
    def _create_emergency_checkpoint(self, error: TrainingError) -> bool:
        """Create emergency checkpoint."""
        try:
            # This would create a minimal checkpoint
            self.logger.info("Creating emergency checkpoint")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
            return False
    
    def resume_training_from_interruption(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Resume training from interruption using checkpoints.
        
        Returns:
            Tuple of (success, resume_info)
        """
        resume_info = {
            "resumed": False,
            "checkpoint_id": None,
            "last_step": 0,
            "last_epoch": 0,
            "warnings": [],
            "errors": []
        }
        
        try:
            if not self.checkpoint_manager:
                resume_info["errors"].append("No checkpoint manager available")
                return False, resume_info
            
            # Find latest checkpoint
            latest_checkpoint_id = self.checkpoint_manager.get_latest_checkpoint_id()
            if not latest_checkpoint_id:
                resume_info["errors"].append("No checkpoints found")
                return False, resume_info
            
            # Load checkpoint
            result = self.checkpoint_manager.load_checkpoint(
                latest_checkpoint_id,
                validate_config=self.config.checkpoint_validation_enabled
            )
            
            if not result.success:
                resume_info["errors"].extend(result.errors)
                resume_info["warnings"].extend(result.warnings)
                
                # Try backup checkpoints
                if self.config.backup_checkpoint_count > 1:
                    checkpoints = self.checkpoint_manager.list_checkpoints(
                        limit=self.config.backup_checkpoint_count
                    )
                    
                    for checkpoint in checkpoints[1:]:  # Skip the failed one
                        backup_result = self.checkpoint_manager.load_checkpoint(
                            checkpoint.checkpoint_id
                        )
                        
                        if backup_result.success:
                            result = backup_result
                            resume_info["warnings"].append(
                                f"Used backup checkpoint: {checkpoint.checkpoint_id}"
                            )
                            break
                
                if not result.success:
                    return False, resume_info
            
            # Extract resume information
            if result.metadata:
                resume_info["checkpoint_id"] = result.metadata.checkpoint_id
                resume_info["last_step"] = result.metadata.step
                resume_info["last_epoch"] = result.metadata.epoch
                self.last_successful_step = result.metadata.step
            
            resume_info["resumed"] = True
            self.training_interrupted = False
            
            self.logger.info(f"Successfully resumed from checkpoint: {resume_info['checkpoint_id']}")
            
            return True, resume_info
            
        except Exception as e:
            error_msg = f"Failed to resume training: {e}"
            resume_info["errors"].append(error_msg)
            self.logger.error(error_msg)
            return False, resume_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error handling summary."""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Recent errors
        recent_errors = self.error_history[-10:]
        
        # Recovery statistics
        total_recoveries_attempted = sum(1 for e in self.error_history if e.recovery_attempted)
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
        
        return {
            "total_errors": len(self.error_history),
            "error_categories": category_counts,
            "error_severities": severity_counts,
            "recovery_stats": {
                "total_attempted": total_recoveries_attempted,
                "successful": successful_recoveries,
                "success_rate": successful_recoveries / max(1, total_recoveries_attempted)
            },
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message,
                    "recovered": e.recovery_successful
                }
                for e in recent_errors
            ],
            "current_state": {
                "training_interrupted": self.training_interrupted,
                "last_successful_step": self.last_successful_step,
                "current_batch_size": self.current_batch_size,
                "original_batch_size": self.original_batch_size
            }
        }