"""
Comprehensive logging configuration for Gmail Dataset Creator.

Provides structured logging setup with API call tracking, progress monitoring,
detailed error logging, and process state management capabilities.
"""

import logging
import logging.handlers
import sys
import json
import time
import threading
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback


@dataclass
class APICallLog:
    """Structure for API call logging."""
    timestamp: str
    api_name: str
    method: str
    endpoint: str
    request_id: str
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    retry_attempt: int = 0
    rate_limited: bool = False


@dataclass
class ProcessingProgress:
    """Structure for processing progress tracking."""
    timestamp: str
    stage: str
    total_items: int
    processed_items: int
    failed_items: int
    current_item_id: Optional[str] = None
    estimated_completion: Optional[str] = None
    processing_rate: Optional[float] = None  # items per second


@dataclass
class ErrorContext:
    """Structure for detailed error context."""
    timestamp: str
    error_type: str
    error_message: str
    component: str
    operation: str
    context_data: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'api_call'):
            log_data['api_call'] = asdict(record.api_call)
        if hasattr(record, 'progress'):
            log_data['progress'] = asdict(record.progress)
        if hasattr(record, 'error_context'):
            log_data['error_context'] = asdict(record.error_context)
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'email_id'):
            log_data['email_id'] = record.email_id
        if hasattr(record, 'batch_id'):
            log_data['batch_id'] = record.batch_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class APICallLogger:
    """Logger specifically for API calls with timing and retry tracking."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._call_counter = 0
        self._lock = threading.Lock()
    
    def log_api_call(self, 
                     api_name: str, 
                     method: str, 
                     endpoint: str,
                     status_code: Optional[int] = None,
                     response_time_ms: Optional[float] = None,
                     error: Optional[str] = None,
                     retry_attempt: int = 0,
                     rate_limited: bool = False) -> str:
        """
        Log an API call with structured data.
        
        Args:
            api_name: Name of the API (e.g., 'Gmail', 'Gemini')
            method: HTTP method or API method name
            endpoint: API endpoint or operation name
            status_code: HTTP status code if applicable
            response_time_ms: Response time in milliseconds
            error: Error message if call failed
            retry_attempt: Retry attempt number (0 for first attempt)
            rate_limited: Whether the call was rate limited
            
        Returns:
            Unique request ID for this API call
        """
        with self._lock:
            self._call_counter += 1
            request_id = f"{api_name.lower()}_{self._call_counter}_{int(time.time())}"
        
        api_call = APICallLog(
            timestamp=datetime.now().isoformat(),
            api_name=api_name,
            method=method,
            endpoint=endpoint,
            request_id=request_id,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error=error,
            retry_attempt=retry_attempt,
            rate_limited=rate_limited
        )
        
        # Determine log level based on status
        if error:
            level = logging.ERROR
        elif rate_limited:
            level = logging.WARNING
        elif retry_attempt > 0:
            level = logging.WARNING
        else:
            level = logging.INFO
        
        # Log with structured data
        self.logger.log(
            level,
            f"API call {api_name}.{method} to {endpoint}",
            extra={'api_call': api_call, 'request_id': request_id}
        )
        
        return request_id
    
    @contextmanager
    def time_api_call(self, api_name: str, method: str, endpoint: str, retry_attempt: int = 0):
        """
        Context manager for timing API calls.
        
        Args:
            api_name: Name of the API
            method: HTTP method or API method name
            endpoint: API endpoint or operation name
            retry_attempt: Retry attempt number
            
        Yields:
            Request ID for this API call
        """
        start_time = time.time()
        request_id = None
        error = None
        rate_limited = False
        
        try:
            # Pre-log the API call attempt
            request_id = self.log_api_call(
                api_name=api_name,
                method=method,
                endpoint=endpoint,
                retry_attempt=retry_attempt
            )
            yield request_id
            
        except Exception as e:
            error = str(e)
            # Check if it's a rate limit error
            error_msg = str(e).lower()
            rate_limited = 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg
            raise
            
        finally:
            # Log the completion with timing
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            if request_id:  # Update the existing log entry
                self.log_api_call(
                    api_name=api_name,
                    method=method,
                    endpoint=endpoint,
                    response_time_ms=response_time_ms,
                    error=error,
                    retry_attempt=retry_attempt,
                    rate_limited=rate_limited
                )


class ProgressTracker:
    """Progress tracking with rate calculation and ETA estimation."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._start_time = None
        self._last_update = None
        self._processed_items = 0
        self._failed_items = 0
        self._total_items = 0
        self._current_stage = ""
        self._lock = threading.Lock()
    
    def start_stage(self, stage: str, total_items: int) -> None:
        """
        Start a new processing stage.
        
        Args:
            stage: Name of the processing stage
            total_items: Total number of items to process
        """
        with self._lock:
            self._start_time = time.time()
            self._last_update = self._start_time
            self._processed_items = 0
            self._failed_items = 0
            self._total_items = total_items
            self._current_stage = stage
        
        self.logger.info(
            f"Starting stage: {stage} ({total_items} items)",
            extra={'progress': self._create_progress_log()}
        )
    
    def update_progress(self, 
                       processed_items: Optional[int] = None,
                       failed_items: Optional[int] = None,
                       current_item_id: Optional[str] = None,
                       force_log: bool = False) -> None:
        """
        Update processing progress.
        
        Args:
            processed_items: Number of successfully processed items
            failed_items: Number of failed items
            current_item_id: ID of currently processing item
            force_log: Force logging even if not enough time has passed
        """
        with self._lock:
            if processed_items is not None:
                self._processed_items = processed_items
            if failed_items is not None:
                self._failed_items = failed_items
            
            current_time = time.time()
            
            # Only log progress updates every 5 seconds or if forced
            if not force_log and (current_time - self._last_update) < 5.0:
                return
            
            self._last_update = current_time
            progress_log = self._create_progress_log(current_item_id)
        
        self.logger.info(
            f"Progress update: {self._current_stage} - "
            f"{self._processed_items}/{self._total_items} processed "
            f"({self._failed_items} failed)",
            extra={'progress': progress_log, 'email_id': current_item_id}
        )
    
    def complete_stage(self) -> None:
        """Complete the current processing stage."""
        with self._lock:
            progress_log = self._create_progress_log()
        
        self.logger.info(
            f"Completed stage: {self._current_stage} - "
            f"{self._processed_items} processed, {self._failed_items} failed",
            extra={'progress': progress_log}
        )
    
    def _create_progress_log(self, current_item_id: Optional[str] = None) -> ProcessingProgress:
        """Create progress log structure."""
        current_time = time.time()
        
        # Calculate processing rate
        processing_rate = None
        estimated_completion = None
        
        if self._start_time and self._processed_items > 0:
            elapsed_time = current_time - self._start_time
            processing_rate = self._processed_items / elapsed_time
            
            # Estimate completion time
            remaining_items = self._total_items - self._processed_items
            if processing_rate > 0:
                eta_seconds = remaining_items / processing_rate
                estimated_completion = datetime.fromtimestamp(
                    current_time + eta_seconds
                ).isoformat()
        
        return ProcessingProgress(
            timestamp=datetime.fromtimestamp(current_time).isoformat(),
            stage=self._current_stage,
            total_items=self._total_items,
            processed_items=self._processed_items,
            failed_items=self._failed_items,
            current_item_id=current_item_id,
            estimated_completion=estimated_completion,
            processing_rate=processing_rate
        )


class ErrorLogger:
    """Enhanced error logging with context and recovery suggestions."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_error(self,
                  error: Exception,
                  component: str,
                  operation: str,
                  context_data: Optional[Dict[str, Any]] = None,
                  recovery_action: Optional[str] = None,
                  email_id: Optional[str] = None,
                  request_id: Optional[str] = None) -> None:
        """
        Log an error with detailed context information.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed when error occurred
            context_data: Additional context data
            recovery_action: Suggested recovery action
            email_id: Email ID if applicable
            request_id: Request ID if applicable
        """
        error_context = ErrorContext(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            component=component,
            operation=operation,
            context_data=context_data or {},
            stack_trace=traceback.format_exc(),
            recovery_action=recovery_action
        )
        
        extra_data = {'error_context': error_context}
        if email_id:
            extra_data['email_id'] = email_id
        if request_id:
            extra_data['request_id'] = request_id
        
        self.logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            extra=extra_data
        )
    
    def log_warning(self,
                   message: str,
                   component: str,
                   operation: str,
                   context_data: Optional[Dict[str, Any]] = None,
                   email_id: Optional[str] = None) -> None:
        """
        Log a warning with context information.
        
        Args:
            message: Warning message
            component: Component where warning occurred
            operation: Operation being performed
            context_data: Additional context data
            email_id: Email ID if applicable
        """
        extra_data = {}
        if context_data:
            extra_data['context'] = context_data
        if email_id:
            extra_data['email_id'] = email_id
        
        self.logger.warning(
            f"Warning in {component}.{operation}: {message}",
            extra=extra_data
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured_logging: bool = True,
    max_log_size_mb: int = 100,
    backup_count: int = 5
) -> None:
    """
    Setup comprehensive logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        structured_logging: Whether to use structured JSON logging
        max_log_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Choose formatter
    if structured_logging:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler with rotation if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_api_logger(name: str) -> APICallLogger:
    """
    Get an API call logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        API call logger instance
    """
    logger = get_logger(name)
    return APICallLogger(logger)


def get_progress_tracker(name: str) -> ProgressTracker:
    """
    Get a progress tracker instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Progress tracker instance
    """
    logger = get_logger(name)
    return ProgressTracker(logger)


def get_error_logger(name: str) -> ErrorLogger:
    """
    Get an error logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Error logger instance
    """
    logger = get_logger(name)
    return ErrorLogger(logger)