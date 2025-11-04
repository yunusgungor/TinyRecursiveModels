"""
Process state management for handling interruptions and resume capability.

This module provides checkpoint system, process state tracking, and recovery
logic for graceful handling of network interruptions and process resumption.
"""

import json
import pickle
import threading
import time
import signal
import os
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum
import logging


class ProcessState(Enum):
    """Process state enumeration."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class CheckpointData:
    """Structure for checkpoint data."""
    timestamp: str
    process_id: str
    stage: str
    state: ProcessState
    progress: Dict[str, Any]
    context_data: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None
    recovery_actions: Optional[List[str]] = None


@dataclass
class ProcessMetrics:
    """Process execution metrics."""
    start_time: str
    last_checkpoint_time: str
    total_runtime_seconds: float
    items_processed: int
    items_failed: int
    checkpoints_created: int
    recoveries_attempted: int
    current_stage: str
    estimated_completion: Optional[str] = None


class ProcessStateManager:
    """
    Manages process state, checkpoints, and recovery for long-running operations.
    
    Provides graceful handling of interruptions, automatic checkpointing,
    and resume capability for dataset creation processes.
    """
    
    def __init__(self, 
                 process_id: str,
                 checkpoint_dir: str = "./checkpoints",
                 checkpoint_interval: int = 30,
                 auto_save: bool = True):
        """
        Initialize process state manager.
        
        Args:
            process_id: Unique identifier for this process
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_interval: Interval in seconds between automatic checkpoints
            auto_save: Whether to automatically save checkpoints
        """
        self.process_id = process_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.auto_save = auto_save
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self._state = ProcessState.NOT_STARTED
        self._current_stage = ""
        self._progress = {}
        self._context_data = {}
        self._metrics = ProcessMetrics(
            start_time="",
            last_checkpoint_time="",
            total_runtime_seconds=0.0,
            items_processed=0,
            items_failed=0,
            checkpoints_created=0,
            recoveries_attempted=0,
            current_stage=""
        )
        
        # Threading and interruption handling
        self._lock = threading.Lock()
        self._checkpoint_thread = None
        self._stop_checkpoint_thread = False
        self._interruption_handlers = []
        self._recovery_callbacks = {}
        
        # Signal handlers for graceful shutdown
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        self._interrupted = False
        
        self.logger = logging.getLogger(__name__)
    
    def start_process(self, initial_stage: str = "initialization") -> None:
        """
        Start the process and initialize state tracking.
        
        Args:
            initial_stage: Initial processing stage name
        """
        with self._lock:
            self._state = ProcessState.INITIALIZING
            self._current_stage = initial_stage
            self._metrics.start_time = datetime.now().isoformat()
            self._metrics.current_stage = initial_stage
        
        # Setup signal handlers for graceful interruption
        self._setup_signal_handlers()
        
        # Start automatic checkpointing if enabled
        if self.auto_save:
            self._start_checkpoint_thread()
        
        # Create initial checkpoint
        self.create_checkpoint("Process started")
        
        with self._lock:
            self._state = ProcessState.RUNNING
        
        self.logger.info(f"Process {self.process_id} started in stage: {initial_stage}")
    
    def update_stage(self, stage: str, progress: Optional[Dict[str, Any]] = None) -> None:
        """
        Update current processing stage.
        
        Args:
            stage: New stage name
            progress: Optional progress data for this stage
        """
        with self._lock:
            self._current_stage = stage
            self._metrics.current_stage = stage
            if progress:
                self._progress.update(progress)
        
        self.logger.info(f"Process {self.process_id} moved to stage: {stage}")
        
        # Create checkpoint for stage transition
        if self.auto_save:
            self.create_checkpoint(f"Stage transition to {stage}")
    
    def update_progress(self, 
                       items_processed: Optional[int] = None,
                       items_failed: Optional[int] = None,
                       progress_data: Optional[Dict[str, Any]] = None,
                       context_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update process progress and context data.
        
        Args:
            items_processed: Number of items successfully processed
            items_failed: Number of items that failed processing
            progress_data: Additional progress information
            context_data: Additional context information
        """
        with self._lock:
            if items_processed is not None:
                self._metrics.items_processed = items_processed
            if items_failed is not None:
                self._metrics.items_failed = items_failed
            if progress_data:
                self._progress.update(progress_data)
            if context_data:
                self._context_data.update(context_data)
    
    def create_checkpoint(self, description: str = "") -> str:
        """
        Create a checkpoint with current process state.
        
        Args:
            description: Optional description for this checkpoint
            
        Returns:
            Checkpoint file path
        """
        with self._lock:
            checkpoint_data = CheckpointData(
                timestamp=datetime.now().isoformat(),
                process_id=self.process_id,
                stage=self._current_stage,
                state=self._state,
                progress=self._progress.copy(),
                context_data=self._context_data.copy()
            )
            
            self._metrics.checkpoints_created += 1
            self._metrics.last_checkpoint_time = checkpoint_data.timestamp
        
        # Save checkpoint to file
        checkpoint_file = self._get_checkpoint_file()
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Also save as JSON for human readability
            json_file = checkpoint_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(asdict(checkpoint_data), f, indent=2, default=str)
            
            self.logger.debug(f"Checkpoint created: {checkpoint_file} - {description}")
            return str(checkpoint_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_file: Optional[str] = None) -> Optional[CheckpointData]:
        """
        Load checkpoint data from file.
        
        Args:
            checkpoint_file: Specific checkpoint file to load (uses latest if None)
            
        Returns:
            Checkpoint data if found, None otherwise
        """
        if checkpoint_file is None:
            checkpoint_file = self._get_checkpoint_file()
        
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            self.logger.info(f"No checkpoint file found: {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def resume_from_checkpoint(self, checkpoint_file: Optional[str] = None) -> bool:
        """
        Resume process from a checkpoint.
        
        Args:
            checkpoint_file: Specific checkpoint file to resume from
            
        Returns:
            True if resume successful, False otherwise
        """
        checkpoint_data = self.load_checkpoint(checkpoint_file)
        if not checkpoint_data:
            return False
        
        try:
            with self._lock:
                self._state = ProcessState.RECOVERING
                self._current_stage = checkpoint_data.stage
                self._progress = checkpoint_data.progress.copy()
                self._context_data = checkpoint_data.context_data.copy()
                self._metrics.recoveries_attempted += 1
            
            self.logger.info(f"Resuming process {self.process_id} from checkpoint at stage: {checkpoint_data.stage}")
            
            # Call recovery callback for the current stage if available
            if checkpoint_data.stage in self._recovery_callbacks:
                callback = self._recovery_callbacks[checkpoint_data.stage]
                success = callback(checkpoint_data)
                
                if success:
                    with self._lock:
                        self._state = ProcessState.RUNNING
                    self.logger.info(f"Successfully resumed from checkpoint")
                    return True
                else:
                    self.logger.error(f"Recovery callback failed for stage: {checkpoint_data.stage}")
                    return False
            else:
                # No specific recovery callback, just restore state
                with self._lock:
                    self._state = ProcessState.RUNNING
                self.logger.info(f"Resumed from checkpoint (no recovery callback)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            with self._lock:
                self._state = ProcessState.FAILED
            return False
    
    def register_recovery_callback(self, stage: str, callback: Callable[[CheckpointData], bool]) -> None:
        """
        Register a recovery callback for a specific stage.
        
        Args:
            stage: Stage name to register callback for
            callback: Callback function that takes CheckpointData and returns success boolean
        """
        self._recovery_callbacks[stage] = callback
        self.logger.debug(f"Recovery callback registered for stage: {stage}")
    
    def register_interruption_handler(self, handler: Callable[[], None]) -> None:
        """
        Register a handler to be called when process is interrupted.
        
        Args:
            handler: Function to call on interruption
        """
        self._interruption_handlers.append(handler)
    
    def complete_process(self, final_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark process as completed and perform cleanup.
        
        Args:
            final_data: Optional final data to include in completion checkpoint
        """
        with self._lock:
            self._state = ProcessState.COMPLETED
            if final_data:
                self._context_data.update(final_data)
            
            # Calculate final metrics
            start_time = datetime.fromisoformat(self._metrics.start_time)
            end_time = datetime.now()
            self._metrics.total_runtime_seconds = (end_time - start_time).total_seconds()
        
        # Create final checkpoint
        self.create_checkpoint("Process completed successfully")
        
        # Stop checkpoint thread
        self._stop_checkpoint_thread = True
        if self._checkpoint_thread and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join(timeout=5)
        
        # Restore signal handlers
        self._restore_signal_handlers()
        
        self.logger.info(f"Process {self.process_id} completed successfully")
    
    def fail_process(self, error: Exception, recovery_actions: Optional[List[str]] = None) -> None:
        """
        Mark process as failed and create error checkpoint.
        
        Args:
            error: Exception that caused the failure
            recovery_actions: Optional list of suggested recovery actions
        """
        with self._lock:
            self._state = ProcessState.FAILED
            
            error_info = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': datetime.now().isoformat()
            }
        
        # Create failure checkpoint with error info
        checkpoint_data = CheckpointData(
            timestamp=datetime.now().isoformat(),
            process_id=self.process_id,
            stage=self._current_stage,
            state=self._state,
            progress=self._progress.copy(),
            context_data=self._context_data.copy(),
            error_info=error_info,
            recovery_actions=recovery_actions
        )
        
        # Save failure checkpoint
        failure_file = self._get_checkpoint_file(suffix="_failure")
        try:
            with open(failure_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Also save as JSON
            json_file = failure_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(asdict(checkpoint_data), f, indent=2, default=str)
                
        except Exception as save_error:
            self.logger.error(f"Failed to save failure checkpoint: {save_error}")
        
        # Stop checkpoint thread
        self._stop_checkpoint_thread = True
        
        self.logger.error(f"Process {self.process_id} failed: {error}")
    
    def pause_process(self) -> None:
        """Pause the process and create checkpoint."""
        with self._lock:
            self._state = ProcessState.PAUSED
        
        self.create_checkpoint("Process paused")
        self.logger.info(f"Process {self.process_id} paused")
    
    def resume_process(self) -> None:
        """Resume a paused process."""
        with self._lock:
            if self._state == ProcessState.PAUSED:
                self._state = ProcessState.RUNNING
                self.logger.info(f"Process {self.process_id} resumed")
            else:
                self.logger.warning(f"Cannot resume process in state: {self._state}")
    
    def get_state(self) -> ProcessState:
        """Get current process state."""
        with self._lock:
            return self._state
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress data."""
        with self._lock:
            return {
                'state': self._state.value,
                'stage': self._current_stage,
                'progress': self._progress.copy(),
                'metrics': asdict(self._metrics)
            }
    
    def is_interrupted(self) -> bool:
        """Check if process has been interrupted."""
        return self._interrupted
    
    @contextmanager
    def stage_context(self, stage_name: str, progress_data: Optional[Dict[str, Any]] = None):
        """
        Context manager for processing stages with automatic checkpointing.
        
        Args:
            stage_name: Name of the processing stage
            progress_data: Initial progress data for this stage
        """
        self.update_stage(stage_name, progress_data)
        
        try:
            yield self
        except Exception as e:
            self.logger.error(f"Error in stage {stage_name}: {e}")
            self.fail_process(e, [f"Retry stage: {stage_name}"])
            raise
        finally:
            # Create checkpoint at end of stage
            if self.auto_save and self._state == ProcessState.RUNNING:
                self.create_checkpoint(f"Completed stage: {stage_name}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful interruption."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._interrupted = True
            
            with self._lock:
                self._state = ProcessState.INTERRUPTED
            
            # Call interruption handlers
            for handler in self._interruption_handlers:
                try:
                    handler()
                except Exception as e:
                    self.logger.error(f"Error in interruption handler: {e}")
            
            # Create interruption checkpoint
            self.create_checkpoint("Process interrupted by signal")
            
            self.logger.info("Graceful shutdown completed")
        
        # Store original handlers
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)
    
    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
    
    def _start_checkpoint_thread(self) -> None:
        """Start background thread for automatic checkpointing."""
        def checkpoint_worker():
            while not self._stop_checkpoint_thread:
                time.sleep(self.checkpoint_interval)
                
                if not self._stop_checkpoint_thread and self._state == ProcessState.RUNNING:
                    try:
                        self.create_checkpoint("Automatic checkpoint")
                    except Exception as e:
                        self.logger.error(f"Automatic checkpoint failed: {e}")
        
        self._checkpoint_thread = threading.Thread(target=checkpoint_worker, daemon=True)
        self._checkpoint_thread.start()
        self.logger.debug(f"Automatic checkpointing started (interval: {self.checkpoint_interval}s)")
    
    def _get_checkpoint_file(self, suffix: str = "") -> Path:
        """Get checkpoint file path."""
        filename = f"{self.process_id}{suffix}.checkpoint"
        return self.checkpoint_dir / filename
    
    def cleanup_checkpoints(self, keep_latest: int = 5) -> None:
        """
        Clean up old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest checkpoints to keep
        """
        try:
            # Find all checkpoint files for this process
            pattern = f"{self.process_id}*.checkpoint"
            checkpoint_files = list(self.checkpoint_dir.glob(pattern))
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in checkpoint_files[keep_latest:]:
                old_file.unlink()
                # Also remove corresponding JSON file
                json_file = old_file.with_suffix('.json')
                if json_file.exists():
                    json_file.unlink()
            
            if len(checkpoint_files) > keep_latest:
                removed_count = len(checkpoint_files) - keep_latest
                self.logger.info(f"Cleaned up {removed_count} old checkpoint files")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")


class NetworkInterruptionHandler:
    """
    Specialized handler for network interruptions with retry logic.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        """
        Initialize network interruption handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def execute_with_retry(self, operation_func: Callable, operation_name: str, 
                          context_data: Optional[Dict[str, Any]] = None):
        """
        Execute an operation with automatic retry on network interruption.
        
        Args:
            operation_func: Function to execute
            operation_name: Name of the network operation
            context_data: Additional context data for logging
            
        Returns:
            Result of the operation function
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
                    self.logger.info(f"Retrying {operation_name} (attempt {attempt + 1}) after {delay:.1f}s delay")
                    time.sleep(delay)
                
                return operation_func()
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                is_network_error = (
                    isinstance(e, (ConnectionError, TimeoutError)) or
                    any(keyword in error_msg for keyword in [
                        'connection failed', 'connection refused', 'connection reset', 
                        'connection aborted', 'connection timeout', 'timeout',
                        'network unreachable', 'dns', 'socket', 'host unreachable'
                    ])
                )
                
                if is_network_error and attempt < self.max_retries:
                    self.logger.warning(f"Network error in {operation_name} (attempt {attempt + 1}): {e}")
                    continue
                else:
                    # Not a network error or max retries exceeded
                    self.logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts: {e}")
                    raise
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
    
    @contextmanager
    def handle_network_operation(self, operation_name: str, context_data: Optional[Dict[str, Any]] = None):
        """
        Simple context manager that just yields the attempt number.
        For more complex retry logic, use execute_with_retry instead.
        
        Args:
            operation_name: Name of the network operation
            context_data: Additional context data for logging
        """
        yield 0  # Just yield attempt 0 for simple context manager usage