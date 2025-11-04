# Comprehensive Logging and Process Management

This document describes the comprehensive logging and error handling system implemented for the Gmail Dataset Creator, including process interruption handling and resume capability.

## Overview

The system provides:
- **Structured JSON logging** with API call tracking
- **Progress monitoring** with ETA estimation
- **Detailed error logging** with context and recovery suggestions
- **Process state management** with checkpointing
- **Graceful interruption handling** with resume capability
- **Network interruption recovery** with automatic retry

## Components

### 1. Enhanced Logging System (`utils/logging.py`)

#### Structured Logging
- JSON-formatted log output for machine parsing
- Configurable console and file output
- Log rotation with size limits
- External library log level management

#### API Call Logger
- Tracks all API calls with timing
- Request ID generation for correlation
- Rate limiting and retry attempt logging
- Success/failure status tracking

#### Progress Tracker
- Stage-based progress monitoring
- Processing rate calculation
- ETA estimation
- Configurable update intervals

#### Error Logger
- Detailed error context capture
- Stack trace logging
- Recovery action suggestions
- Component and operation tracking

### 2. Process State Management (`utils/process_state.py`)

#### ProcessStateManager
- Automatic checkpointing system
- Process state tracking (running, paused, interrupted, etc.)
- Stage-based execution with context managers
- Signal handling for graceful shutdown

#### Checkpoint System
- Binary and JSON checkpoint formats
- Resume from checkpoint capability
- Recovery callback registration
- Progress data preservation

#### Network Interruption Handler
- Automatic retry for network errors
- Exponential backoff with jitter
- Network vs. application error detection
- Configurable retry limits

## Usage Examples

### Basic Logging Setup
```python
from gmail_dataset_creator.utils.logging import setup_logging, get_api_logger

# Setup structured logging
setup_logging(
    level="INFO",
    log_file="./logs/app.log",
    structured_logging=True
)

# Use API logger
api_logger = get_api_logger(__name__)
with api_logger.time_api_call("Gmail", "list_messages", "gmail_api") as request_id:
    # Make API call
    response = gmail_client.list_messages()
```

### Process State Management
```python
from gmail_dataset_creator.utils.process_state import ProcessStateManager

# Initialize process manager
process_manager = ProcessStateManager(
    process_id="email_processing",
    checkpoint_dir="./checkpoints",
    checkpoint_interval=60
)

# Start process with automatic checkpointing
process_manager.start_process("initialization")

# Use stage context for automatic error handling
with process_manager.stage_context("email_fetching"):
    # Process emails with automatic progress tracking
    for email in emails:
        process_manager.update_progress(items_processed=i+1)
        # Process email...

# Complete process
process_manager.complete_process()
```

### Resume from Checkpoint
```python
# Try to resume from existing checkpoint
if process_manager.resume_from_checkpoint():
    print("Resumed from checkpoint")
else:
    print("Starting new process")
    process_manager.start_process()
```

## Integration with Existing Components

### Gmail API Client
- Enhanced with API call logging
- Network interruption handling
- Rate limit tracking and logging
- Detailed error context for troubleshooting

### Gemini Classifier
- Progress tracking for batch processing
- API call timing and error logging
- Checkpoint-based resume capability
- Enhanced error handling with recovery suggestions

### Main Application
- Process state management integration
- Comprehensive error logging
- Graceful interruption handling
- Resume capability for long-running operations

## Configuration

### Logging Configuration
```python
setup_logging(
    level="INFO",                    # Log level
    log_file="./logs/app.log",      # Optional log file
    structured_logging=True,         # JSON format
    max_log_size_mb=50,             # Log rotation size
    backup_count=3                   # Number of backup files
)
```

### Process Manager Configuration
```python
ProcessStateManager(
    process_id="unique_process_id",
    checkpoint_dir="./checkpoints",
    checkpoint_interval=60,          # Seconds between checkpoints
    auto_save=True                   # Enable automatic checkpointing
)
```

## Testing

Comprehensive test suite covers:
- Structured logging functionality
- API call tracking and timing
- Progress monitoring
- Error logging with context
- Process lifecycle management
- Checkpoint creation and loading
- Resume capability
- Network interruption handling
- Integration scenarios

Run tests with:
```bash
python -m pytest gmail_dataset_creator/tests/test_comprehensive_logging.py -v
```

## Demo

Interactive demo showing process interruption and resume:
```bash
python gmail_dataset_creator/examples/process_interruption_example.py
```

Press Ctrl+C to interrupt, then run again to see resume capability.

## Benefits

1. **Observability**: Detailed logging provides insight into system behavior
2. **Reliability**: Automatic checkpointing prevents data loss
3. **Resilience**: Network interruption handling improves robustness
4. **Debuggability**: Structured logs and error context aid troubleshooting
5. **User Experience**: Progress tracking and resume capability improve usability
6. **Monitoring**: API call tracking enables performance monitoring

## Requirements Satisfied

- **6.1**: Comprehensive API call logging with timestamps and response codes
- **6.2**: Detailed error logging with context information
- **6.3**: Progress tracking and status reporting
- **6.4**: Graceful handling of network interruptions
- **6.5**: Resume capability from last successful state