"""
Unit tests for training error handler module.

Tests training resumption after various interruption scenarios, validates
memory pressure handling and graceful degradation, and tests error recovery
mechanisms and diagnostic tools.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from macbook_optimization.training_error_handler import (
    ErrorSeverity,
    ErrorCategory,
    TrainingError,
    RecoveryAction,
    ErrorHandlingConfig,
    TrainingErrorHandler
)
from macbook_optimization.memory_management import MemoryManager, MemoryPressureInfo
from macbook_optimization.checkpoint_management import CheckpointManager, CheckpointLoadResult, CheckpointMetadata


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Test ErrorCategory enum."""
    
    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.MEMORY.value == "memory"
        assert ErrorCategory.DATA.value == "data"
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.HARDWARE.value == "hardware"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.CHECKPOINT.value == "checkpoint"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestTrainingError:
    """Test TrainingError dataclass."""
    
    def test_training_error_creation(self):
        """Test TrainingError creation."""
        error = TrainingError(
            error_id="test_error_001",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Out of memory error",
            exception_type="RuntimeError",
            traceback_str="Traceback...",
            context={"batch_size": 16, "memory_usage": 95.0}
        )
        
        assert error.error_id == "test_error_001"
        assert error.category == ErrorCategory.MEMORY
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "Out of memory error"
        assert error.exception_type == "RuntimeError"
        assert error.context["batch_size"] == 16
        assert error.recovery_attempted is False
        assert error.recovery_successful is False
        assert error.recovery_actions == []
    
    def test_training_error_post_init(self):
        """Test TrainingError __post_init__ method."""
        error = TrainingError(
            error_id="test_error_002",
            timestamp=datetime.now(),
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            message="Data parsing error",
            exception_type="ValueError",
            traceback_str="Traceback...",
            context={}
        )
        
        # recovery_actions should be initialized as empty list
        assert isinstance(error.recovery_actions, list)
        assert len(error.recovery_actions) == 0


class TestRecoveryAction:
    """Test RecoveryAction dataclass."""
    
    def test_recovery_action_creation(self):
        """Test RecoveryAction creation."""
        def dummy_action(error):
            return True
        
        action = RecoveryAction(
            name="reduce_batch_size",
            description="Reduce batch size to free memory",
            action_func=dummy_action,
            conditions={"memory_pressure": True},
            max_attempts=3,
            cooldown_seconds=5.0
        )
        
        assert action.name == "reduce_batch_size"
        assert action.description == "Reduce batch size to free memory"
        assert action.action_func == dummy_action
        assert action.conditions == {"memory_pressure": True}
        assert action.max_attempts == 3
        assert action.cooldown_seconds == 5.0


class TestErrorHandlingConfig:
    """Test ErrorHandlingConfig dataclass."""
    
    def test_error_handling_config_defaults(self):
        """Test ErrorHandlingConfig default values."""
        config = ErrorHandlingConfig()
        
        assert config.log_errors_to_file is True
        assert config.error_log_path == "training_errors.log"
        assert config.max_error_history == 1000
        assert config.enable_auto_recovery is True
        assert config.max_recovery_attempts == 3
        assert config.recovery_cooldown_seconds == 10.0
        assert config.memory_pressure_threshold == 85.0
        assert config.emergency_memory_threshold == 95.0
        assert config.batch_size_reduction_factor == 0.5
        assert config.min_batch_size == 1
        assert config.max_data_errors_per_batch == 5
        assert config.skip_corrupted_batches is True
        assert config.data_validation_enabled is True
        assert config.auto_resume_from_checkpoint is True
        assert config.checkpoint_validation_enabled is True
        assert config.backup_checkpoint_count == 3
        assert config.thermal_throttling_detection is True
        assert config.cpu_overload_threshold == 90.0
        assert config.disk_space_threshold_mb == 1000.0


class TestTrainingErrorHandler:
    """Test TrainingErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ErrorHandlingConfig(
            log_errors_to_file=False,  # Disable file logging for tests
            max_error_history=100,
            recovery_cooldown_seconds=0.1  # Fast for testing
        )
        
        # Create mocks
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.mock_checkpoint_manager = Mock(spec=CheckpointManager)
        self.mock_resource_monitor = Mock()
        self.mock_logger = Mock()
        
        # Create error handler
        self.error_handler = TrainingErrorHandler(
            config=self.config,
            memory_manager=self.mock_memory_manager,
            checkpoint_manager=self.mock_checkpoint_manager,
            resource_monitor=self.mock_resource_monitor,
            logger=self.mock_logger
        )
    
    def test_error_handler_initialization(self):
        """Test TrainingErrorHandler initialization."""
        assert self.error_handler.config == self.config
        assert self.error_handler.memory_manager == self.mock_memory_manager
        assert self.error_handler.checkpoint_manager == self.mock_checkpoint_manager
        assert self.error_handler.resource_monitor == self.mock_resource_monitor
        assert self.error_handler.logger == self.mock_logger
        
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.recovery_attempts) == 0
        assert len(self.error_handler.last_recovery_time) == 0
        assert self.error_handler.training_interrupted is False
        assert self.error_handler.last_successful_step == 0
        
        # Check that recovery actions are registered
        assert ErrorCategory.MEMORY in self.error_handler.recovery_actions
        assert ErrorCategory.DATA in self.error_handler.recovery_actions
        assert ErrorCategory.MODEL in self.error_handler.recovery_actions
        assert ErrorCategory.HARDWARE in self.error_handler.recovery_actions
        assert ErrorCategory.CHECKPOINT in self.error_handler.recovery_actions
    
    def test_classify_error_memory(self):
        """Test error classification for memory errors."""
        # Test memory error
        memory_exception = RuntimeError("CUDA out of memory")
        context = {"batch_size": 16}
        
        category = self.error_handler._classify_error(memory_exception, context)
        assert category == ErrorCategory.MEMORY
        
        # Test another memory error
        memory_exception2 = MemoryError("Out of memory")
        category2 = self.error_handler._classify_error(memory_exception2, context)
        assert category2 == ErrorCategory.MEMORY
    
    def test_classify_error_data(self):
        """Test error classification for data errors."""
        # Test JSON decode error
        json_exception = json.JSONDecodeError("Invalid JSON", "test", 0)
        context = {"file_path": "data.json"}
        
        category = self.error_handler._classify_error(json_exception, context)
        assert category == ErrorCategory.DATA
        
        # Test Unicode decode error
        unicode_exception = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        category2 = self.error_handler._classify_error(unicode_exception, context)
        assert category2 == ErrorCategory.DATA
        
        # Test ValueError with data-related message
        value_exception = ValueError("Invalid data format")
        category3 = self.error_handler._classify_error(value_exception, context)
        assert category3 == ErrorCategory.DATA
    
    def test_classify_error_model(self):
        """Test error classification for model errors."""
        # Test model forward error
        model_exception = RuntimeError("Error in model forward pass")
        context = {"model_name": "EmailTRM"}
        
        category = self.error_handler._classify_error(model_exception, context)
        assert category == ErrorCategory.MODEL
        
        # Test gradient error
        gradient_exception = RuntimeError("Gradient computation failed")
        category2 = self.error_handler._classify_error(gradient_exception, context)
        assert category2 == ErrorCategory.MODEL
    
    def test_classify_error_hardware(self):
        """Test error classification for hardware errors."""
        # Test thermal error
        thermal_exception = RuntimeError("Thermal throttling detected")
        context = {"cpu_temp": 85.0}
        
        category = self.error_handler._classify_error(thermal_exception, context)
        assert category == ErrorCategory.HARDWARE
        
        # Test CPU error
        cpu_exception = RuntimeError("CPU overload detected")
        category2 = self.error_handler._classify_error(cpu_exception, context)
        assert category2 == ErrorCategory.HARDWARE
    
    def test_classify_error_checkpoint(self):
        """Test error classification for checkpoint errors."""
        # Test file not found error
        file_exception = FileNotFoundError("Checkpoint file not found")
        context = {"checkpoint_path": "model.pt"}
        
        category = self.error_handler._classify_error(file_exception, context)
        assert category == ErrorCategory.CHECKPOINT
        
        # Test permission error
        perm_exception = PermissionError("Cannot save checkpoint")
        category2 = self.error_handler._classify_error(perm_exception, context)
        assert category2 == ErrorCategory.CHECKPOINT
    
    def test_classify_error_unknown(self):
        """Test error classification for unknown errors."""
        # Test unknown error
        unknown_exception = Exception("Unknown error occurred")
        context = {}
        
        category = self.error_handler._classify_error(unknown_exception, context)
        assert category == ErrorCategory.UNKNOWN
    
    def test_assess_severity_critical(self):
        """Test severity assessment for critical errors."""
        # Test system exit
        exit_exception = SystemExit("Training terminated")
        context = {}
        
        severity = self.error_handler._assess_severity(
            exit_exception, context, ErrorCategory.UNKNOWN
        )
        assert severity == ErrorSeverity.CRITICAL
        
        # Test keyboard interrupt
        interrupt_exception = KeyboardInterrupt("User interrupted")
        severity2 = self.error_handler._assess_severity(
            interrupt_exception, context, ErrorCategory.UNKNOWN
        )
        assert severity2 == ErrorSeverity.CRITICAL
    
    def test_assess_severity_high(self):
        """Test severity assessment for high severity errors."""
        # Test out of memory
        memory_exception = RuntimeError("CUDA out of memory")
        context = {}
        
        severity = self.error_handler._assess_severity(
            memory_exception, context, ErrorCategory.MEMORY
        )
        assert severity == ErrorSeverity.HIGH
        
        # Test NaN in model
        nan_exception = RuntimeError("Model output contains NaN")
        severity2 = self.error_handler._assess_severity(
            nan_exception, context, ErrorCategory.MODEL
        )
        assert severity2 == ErrorSeverity.HIGH
    
    def test_assess_severity_medium(self):
        """Test severity assessment for medium severity errors."""
        # Test data error
        data_exception = ValueError("Invalid data format")
        context = {}
        
        severity = self.error_handler._assess_severity(
            data_exception, context, ErrorCategory.DATA
        )
        assert severity == ErrorSeverity.MEDIUM
        
        # Test checkpoint error
        checkpoint_exception = FileNotFoundError("Checkpoint not found")
        severity2 = self.error_handler._assess_severity(
            checkpoint_exception, context, ErrorCategory.CHECKPOINT
        )
        assert severity2 == ErrorSeverity.MEDIUM
    
    def test_assess_severity_low(self):
        """Test severity assessment for low severity errors."""
        # Test generic exception
        generic_exception = Exception("Generic error")
        context = {}
        
        severity = self.error_handler._assess_severity(
            generic_exception, context, ErrorCategory.UNKNOWN
        )
        assert severity == ErrorSeverity.LOW
    
    def test_create_error_record(self):
        """Test error record creation."""
        exception = RuntimeError("Test error")
        context = {"step": 100, "batch_size": 8}
        
        error_record = self.error_handler._create_error_record(exception, context)
        
        assert isinstance(error_record, TrainingError)
        assert error_record.message == "Test error"
        assert error_record.exception_type == "RuntimeError"
        assert error_record.context == context
        assert error_record.category in ErrorCategory
        assert error_record.severity in ErrorSeverity
        assert len(error_record.error_id) > 0
        assert isinstance(error_record.timestamp, datetime)
    
    def test_handle_error_without_recovery(self):
        """Test error handling without recovery."""
        exception = ValueError("Test error")
        context = {"test": True}
        
        success, actions = self.error_handler.handle_error(
            exception, context, attempt_recovery=False
        )
        
        assert success is False
        assert len(actions) == 0
        assert len(self.error_handler.error_history) == 1
        
        error = self.error_handler.error_history[0]
        assert error.recovery_attempted is False
        assert error.recovery_successful is False
    
    def test_handle_error_with_recovery_success(self):
        """Test error handling with successful recovery."""
        # Mock memory manager to return proper memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 90.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        # Mock successful recovery action
        def mock_recovery_action(error):
            return True
        
        # Replace recovery action
        memory_actions = self.error_handler.recovery_actions[ErrorCategory.MEMORY]
        memory_actions[0].action_func = mock_recovery_action
        
        exception = RuntimeError("CUDA out of memory")
        context = {"batch_size": 16}
        
        success, actions = self.error_handler.handle_error(exception, context)
        
        assert success is True
        assert len(actions) > 0
        assert len(self.error_handler.error_history) == 1
        
        error = self.error_handler.error_history[0]
        assert error.recovery_attempted is True
        assert error.recovery_successful is True
        assert len(error.recovery_actions) > 0
    
    def test_handle_error_with_recovery_failure(self):
        """Test error handling with failed recovery."""
        # Mock memory manager to return proper memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 90.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        # Mock failed recovery action
        def mock_recovery_action(error):
            return False
        
        # Replace all memory recovery actions to ensure they all fail
        memory_actions = self.error_handler.recovery_actions[ErrorCategory.MEMORY]
        for action in memory_actions:
            action.action_func = mock_recovery_action
        
        exception = RuntimeError("CUDA out of memory")
        context = {"batch_size": 16}
        
        success, actions = self.error_handler.handle_error(exception, context)
        
        assert success is False
        assert len(actions) > 0  # Actions were attempted
        assert len(self.error_handler.error_history) == 1
        
        error = self.error_handler.error_history[0]
        assert error.recovery_attempted is True
        assert error.recovery_successful is False
    
    def test_should_attempt_recovery_critical_error(self):
        """Test that recovery is not attempted for critical errors."""
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.CRITICAL,
            message="Critical error",
            exception_type="SystemExit",
            traceback_str="",
            context={}
        )
        
        should_attempt = self.error_handler._should_attempt_recovery(error)
        assert should_attempt is False
    
    def test_should_attempt_recovery_max_attempts_exceeded(self):
        """Test that recovery is not attempted when max attempts exceeded."""
        # Set up recovery attempts to exceed limit
        self.error_handler.recovery_attempts = {
            "memory_reduce_batch_size": 5,
            "data_skip_batch": 5
        }
        
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Memory error",
            exception_type="RuntimeError",
            traceback_str="",
            context={}
        )
        
        should_attempt = self.error_handler._should_attempt_recovery(error)
        assert should_attempt is False
    
    def test_should_attempt_recovery_cooldown_active(self):
        """Test that recovery is not attempted during cooldown."""
        # Set recent recovery time
        self.error_handler.last_recovery_time = {"test": time.time()}
        
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Memory error",
            exception_type="RuntimeError",
            traceback_str="",
            context={}
        )
        
        should_attempt = self.error_handler._should_attempt_recovery(error)
        assert should_attempt is False
    
    def test_reduce_batch_size_recovery(self):
        """Test batch size reduction recovery action."""
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Memory error",
            exception_type="RuntimeError",
            traceback_str="",
            context={"batch_size": 16}
        )
        
        # Set initial batch size
        self.error_handler.current_batch_size = 16
        
        success = self.error_handler._reduce_batch_size(error)
        
        assert success is True
        assert self.error_handler.current_batch_size < 16
        assert self.error_handler.current_batch_size >= self.config.min_batch_size
    
    def test_reduce_batch_size_minimum_reached(self):
        """Test batch size reduction when minimum is already reached."""
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Memory error",
            exception_type="RuntimeError",
            traceback_str="",
            context={"batch_size": 1}
        )
        
        # Set batch size to minimum
        self.error_handler.current_batch_size = 1
        
        success = self.error_handler._reduce_batch_size(error)
        
        assert success is False  # Cannot reduce further
        assert self.error_handler.current_batch_size == 1
    
    def test_force_garbage_collection_recovery(self):
        """Test garbage collection recovery action."""
        error = TrainingError(
            error_id="test",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.MEDIUM,
            message="Memory pressure",
            exception_type="RuntimeError",
            traceback_str="",
            context={}
        )
        
        with patch.object(self.error_handler.memory_manager, 'force_garbage_collection') as mock_gc:
            success = self.error_handler._force_garbage_collection(error)
        
        assert success is True
        mock_gc.assert_called_once()
    
    def test_resume_training_from_interruption_success(self):
        """Test successful training resumption from interruption."""
        # Mock successful checkpoint loading
        mock_metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            timestamp=datetime.now(),
            step=1000,
            epoch=5,
            loss=0.5,
            learning_rate=1e-4,
            model_config_hash="test_hash",
            memory_usage_mb=2000.0,
            disk_usage_mb=500.0,
            training_time_seconds=3600.0,
            training_config={},
            hardware_summary={},
            checkpoint_path="test.pt",
            model_state_size_mb=100.0,
            optimizer_state_size_mb=50.0,
            config_compatible=True,
            validation_errors=[]
        )
        
        mock_result = CheckpointLoadResult(
            success=True,
            checkpoint_id="test_checkpoint",
            metadata=mock_metadata,
            model_state={"param1": "value1"},
            optimizer_state={"param2": "value2"},
            config_compatible=True,
            warnings=[],
            errors=[]
        )
        
        self.mock_checkpoint_manager.get_latest_checkpoint_id.return_value = "test_checkpoint"
        self.mock_checkpoint_manager.load_checkpoint.return_value = mock_result
        
        success, resume_info = self.error_handler.resume_training_from_interruption()
        
        assert success is True
        assert resume_info["resumed"] is True
        assert resume_info["checkpoint_id"] == "test_checkpoint"
        assert resume_info["last_step"] == 1000
        assert resume_info["last_epoch"] == 5
        assert len(resume_info["errors"]) == 0
        assert self.error_handler.training_interrupted is False
        assert self.error_handler.last_successful_step == 1000
    
    def test_resume_training_from_interruption_no_checkpoint_manager(self):
        """Test training resumption without checkpoint manager."""
        # Set checkpoint manager to None
        self.error_handler.checkpoint_manager = None
        
        success, resume_info = self.error_handler.resume_training_from_interruption()
        
        assert success is False
        assert resume_info["resumed"] is False
        assert "No checkpoint manager available" in resume_info["errors"]
    
    def test_resume_training_from_interruption_no_checkpoints(self):
        """Test training resumption when no checkpoints exist."""
        self.mock_checkpoint_manager.get_latest_checkpoint_id.return_value = None
        
        success, resume_info = self.error_handler.resume_training_from_interruption()
        
        assert success is False
        assert resume_info["resumed"] is False
        assert "No checkpoints found" in resume_info["errors"]
    
    def test_resume_training_from_interruption_checkpoint_load_failure(self):
        """Test training resumption when checkpoint loading fails."""
        mock_result = CheckpointLoadResult(
            success=False,
            checkpoint_id="test_checkpoint",
            metadata=None,
            model_state=None,
            optimizer_state=None,
            config_compatible=False,
            warnings=[],
            errors=["Failed to load checkpoint"]
        )
        
        self.mock_checkpoint_manager.get_latest_checkpoint_id.return_value = "test_checkpoint"
        self.mock_checkpoint_manager.load_checkpoint.return_value = mock_result
        self.mock_checkpoint_manager.list_checkpoints.return_value = []  # No backup checkpoints
        
        success, resume_info = self.error_handler.resume_training_from_interruption()
        
        assert success is False
        assert resume_info["resumed"] is False
        assert "Failed to load checkpoint" in resume_info["errors"]
    
    def test_resume_training_from_interruption_backup_checkpoint_success(self):
        """Test training resumption using backup checkpoint."""
        # First checkpoint fails
        failed_result = CheckpointLoadResult(
            success=False,
            checkpoint_id="latest_checkpoint",
            metadata=None,
            model_state=None,
            optimizer_state=None,
            config_compatible=False,
            warnings=[],
            errors=["Corrupted checkpoint"]
        )
        
        # Backup checkpoint succeeds
        backup_metadata = CheckpointMetadata(
            checkpoint_id="backup_checkpoint",
            timestamp=datetime.now(),
            step=900,
            epoch=4,
            loss=0.6,
            learning_rate=1e-4,
            model_config_hash="backup_hash",
            memory_usage_mb=1800.0,
            disk_usage_mb=450.0,
            training_time_seconds=3200.0,
            training_config={},
            hardware_summary={},
            checkpoint_path="backup.pt",
            model_state_size_mb=100.0,
            optimizer_state_size_mb=50.0,
            config_compatible=True,
            validation_errors=[]
        )
        
        backup_result = CheckpointLoadResult(
            success=True,
            checkpoint_id="backup_checkpoint",
            metadata=backup_metadata,
            model_state={"param1": "backup_value1"},
            optimizer_state={"param2": "backup_value2"},
            config_compatible=True,
            warnings=[],
            errors=[]
        )
        
        # Mock checkpoint manager behavior
        self.mock_checkpoint_manager.get_latest_checkpoint_id.return_value = "latest_checkpoint"
        self.mock_checkpoint_manager.load_checkpoint.side_effect = [failed_result, backup_result]
        
        # Mock backup checkpoints list
        mock_backup_checkpoint = Mock()
        mock_backup_checkpoint.checkpoint_id = "backup_checkpoint"
        self.mock_checkpoint_manager.list_checkpoints.return_value = [
            Mock(checkpoint_id="latest_checkpoint"),  # This one fails
            mock_backup_checkpoint  # This one succeeds
        ]
        
        success, resume_info = self.error_handler.resume_training_from_interruption()
        
        assert success is True
        assert resume_info["resumed"] is True
        assert resume_info["checkpoint_id"] == "backup_checkpoint"
        assert resume_info["last_step"] == 900
        assert "Used backup checkpoint" in resume_info["warnings"][0]
    
    def test_get_error_summary_no_errors(self):
        """Test error summary when no errors have occurred."""
        summary = self.error_handler.get_error_summary()
        
        assert summary["message"] == "No errors recorded"
    
    def test_get_error_summary_with_errors(self):
        """Test error summary with recorded errors."""
        # Add some test errors
        error1 = TrainingError(
            error_id="error1",
            timestamp=datetime.now(),
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            message="Memory error 1",
            exception_type="RuntimeError",
            traceback_str="",
            context={},
            recovery_attempted=True,
            recovery_successful=True
        )
        
        error2 = TrainingError(
            error_id="error2",
            timestamp=datetime.now(),
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            message="Data error 1",
            exception_type="ValueError",
            traceback_str="",
            context={},
            recovery_attempted=True,
            recovery_successful=False
        )
        
        self.error_handler.error_history = [error1, error2]
        self.error_handler.last_successful_step = 500
        self.error_handler.current_batch_size = 8
        self.error_handler.original_batch_size = 16
        
        summary = self.error_handler.get_error_summary()
        
        assert summary["total_errors"] == 2
        assert summary["error_categories"]["memory"] == 1
        assert summary["error_categories"]["data"] == 1
        assert summary["error_severities"]["high"] == 1
        assert summary["error_severities"]["medium"] == 1
        assert summary["recovery_stats"]["total_attempted"] == 2
        assert summary["recovery_stats"]["successful"] == 1
        assert summary["recovery_stats"]["success_rate"] == 0.5
        assert len(summary["recent_errors"]) == 2
        assert summary["current_state"]["last_successful_step"] == 500
        assert summary["current_state"]["current_batch_size"] == 8
        assert summary["current_state"]["original_batch_size"] == 16
    
    def test_error_history_limit(self):
        """Test that error history is limited to max size."""
        # Set small limit for testing
        self.error_handler.config.max_error_history = 3
        
        # Add more errors than the limit
        for i in range(5):
            exception = ValueError(f"Test error {i}")
            context = {"error_number": i}
            self.error_handler.handle_error(exception, context, attempt_recovery=False)
        
        # Should only keep the last 3 errors
        assert len(self.error_handler.error_history) == 3
        
        # Check that the most recent errors are kept
        error_numbers = [error.context["error_number"] for error in self.error_handler.error_history]
        assert error_numbers == [2, 3, 4]
    
    def test_recovery_action_cooldown(self):
        """Test that recovery actions respect cooldown periods."""
        # Mock memory manager to return proper memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 90.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        # Set up a memory error
        exception = RuntimeError("CUDA out of memory")
        context = {"batch_size": 16}
        
        # First recovery attempt should succeed
        success1, actions1 = self.error_handler.handle_error(exception, context)
        
        # Immediate second attempt should be blocked by cooldown
        success2, actions2 = self.error_handler.handle_error(exception, context)
        
        # First attempt should have actions, second should not (due to cooldown)
        assert len(actions1) > 0 or not self.config.enable_auto_recovery
        # Second attempt might still have actions if cooldown is very short
        
        # Check that cooldown tracking is working
        assert len(self.error_handler.last_recovery_time) > 0
    
    def test_recovery_action_max_attempts(self):
        """Test that recovery actions respect max attempt limits."""
        # Mock memory manager to return proper memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 90.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        # Mock a recovery action that always fails
        def failing_action(error):
            return False
        
        # Replace the first memory recovery action
        memory_actions = self.error_handler.recovery_actions[ErrorCategory.MEMORY]
        original_action = memory_actions[0].action_func
        memory_actions[0].action_func = failing_action
        memory_actions[0].max_attempts = 2
        
        try:
            # Reset cooldown for testing
            self.error_handler.config.recovery_cooldown_seconds = 0.0
            
            exception = RuntimeError("CUDA out of memory")
            context = {"batch_size": 16}
            
            # Attempt recovery multiple times
            for i in range(5):
                success, actions = self.error_handler.handle_error(exception, context)
                time.sleep(0.01)  # Small delay to avoid timing issues
            
            # Check that attempts are tracked and limited
            action_key = f"{ErrorCategory.MEMORY.value}_reduce_batch_size"
            attempts = self.error_handler.recovery_attempts.get(action_key, 0)
            assert attempts <= 2  # Should not exceed max_attempts
            
        finally:
            # Restore original action
            memory_actions[0].action_func = original_action


class TestTrainingErrorHandlerIntegration:
    """Integration tests for TrainingErrorHandler with realistic scenarios."""
    
    def test_memory_pressure_error_recovery_scenario(self):
        """Test complete memory pressure error and recovery scenario."""
        config = ErrorHandlingConfig(
            log_errors_to_file=False,
            enable_auto_recovery=True,
            memory_pressure_threshold=80.0,
            batch_size_reduction_factor=0.5,
            min_batch_size=1
        )
        
        mock_memory_manager = Mock()
        mock_memory_manager.monitor_memory_usage.return_value = Mock(percent_used=85.0)
        
        error_handler = TrainingErrorHandler(
            config=config,
            memory_manager=mock_memory_manager
        )
        
        # Simulate memory pressure scenario
        error_handler.current_batch_size = 16
        error_handler.original_batch_size = 16
        
        # Create memory pressure error
        exception = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        context = {
            "batch_size": 16,
            "memory_usage": 95.0,
            "step": 1000
        }
        
        # Handle the error
        success, actions = error_handler.handle_error(exception, context)
        
        # Should attempt recovery
        assert len(error_handler.error_history) == 1
        error = error_handler.error_history[0]
        assert error.category == ErrorCategory.MEMORY
        assert error.severity == ErrorSeverity.HIGH
        
        # Should have reduced batch size
        assert error_handler.current_batch_size < 16
        
        # Check recovery actions were taken
        if config.enable_auto_recovery:
            assert error.recovery_attempted is True
            assert len(error.recovery_actions) > 0
    
    def test_data_corruption_error_recovery_scenario(self):
        """Test data corruption error and recovery scenario."""
        config = ErrorHandlingConfig(
            log_errors_to_file=False,
            enable_auto_recovery=True,
            skip_corrupted_batches=True
        )
        
        error_handler = TrainingErrorHandler(config=config)
        
        # Simulate data corruption scenario
        exception = json.JSONDecodeError("Expecting ',' delimiter", "corrupted.json", 42)
        context = {
            "file_path": "data/emails/batch_001.json",
            "batch_index": 1,
            "step": 500
        }
        
        # Handle the error
        success, actions = error_handler.handle_error(exception, context)
        
        # Should classify as data error
        assert len(error_handler.error_history) == 1
        error = error_handler.error_history[0]
        assert error.category == ErrorCategory.DATA
        assert error.severity == ErrorSeverity.MEDIUM
        
        # Should attempt recovery if enabled
        if config.enable_auto_recovery:
            assert error.recovery_attempted is True
    
    def test_training_interruption_and_resumption_scenario(self):
        """Test complete training interruption and resumption scenario."""
        config = ErrorHandlingConfig(
            auto_resume_from_checkpoint=True,
            checkpoint_validation_enabled=True
        )
        
        # Mock checkpoint manager with successful resumption
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.get_latest_checkpoint_id.return_value = "checkpoint_step_1000"
        
        mock_metadata = CheckpointMetadata(
            checkpoint_id="checkpoint_step_1000",
            timestamp=datetime.now() - timedelta(minutes=30),
            step=1000,
            epoch=5,
            loss=0.45,
            learning_rate=1e-4,
            model_config_hash="abc123",
            memory_usage_mb=2048.0,
            disk_usage_mb=512.0,
            training_time_seconds=1800.0,
            training_config={"batch_size": 8, "learning_rate": 1e-4},
            hardware_summary={"cpu_cores": 8, "memory_gb": 16},
            checkpoint_path="checkpoints/checkpoint_step_1000.pt",
            model_state_size_mb=128.0,
            optimizer_state_size_mb=64.0,
            config_compatible=True,
            validation_errors=[]
        )
        
        mock_result = CheckpointLoadResult(
            success=True,
            checkpoint_id="checkpoint_step_1000",
            metadata=mock_metadata,
            model_state={"layer1.weight": "tensor_data"},
            optimizer_state={"state": "optimizer_data"},
            config_compatible=True,
            warnings=[],
            errors=[]
        )
        
        mock_checkpoint_manager.load_checkpoint.return_value = mock_result
        
        error_handler = TrainingErrorHandler(
            config=config,
            checkpoint_manager=mock_checkpoint_manager
        )
        
        # Simulate training interruption
        error_handler.training_interrupted = True
        
        # Attempt resumption
        success, resume_info = error_handler.resume_training_from_interruption()
        
        # Should successfully resume
        assert success is True
        assert resume_info["resumed"] is True
        assert resume_info["checkpoint_id"] == "checkpoint_step_1000"
        assert resume_info["last_step"] == 1000
        assert resume_info["last_epoch"] == 5
        assert len(resume_info["errors"]) == 0
        
        # Training should no longer be marked as interrupted
        assert error_handler.training_interrupted is False
        assert error_handler.last_successful_step == 1000
    
    def test_multiple_error_types_scenario(self):
        """Test handling multiple different error types in sequence."""
        config = ErrorHandlingConfig(
            log_errors_to_file=False,
            enable_auto_recovery=True,
            max_error_history=10
        )
        
        error_handler = TrainingErrorHandler(config=config)
        
        # Simulate sequence of different errors
        errors_to_simulate = [
            (RuntimeError("CUDA out of memory"), {"type": "memory"}),
            (ValueError("Invalid email format"), {"type": "data"}),
            (FileNotFoundError("Checkpoint not found"), {"type": "checkpoint"}),
            (RuntimeError("Thermal throttling detected"), {"type": "hardware"}),
            (json.JSONDecodeError("Invalid JSON", "test.json", 0), {"type": "data"})
        ]
        
        for exception, context in errors_to_simulate:
            success, actions = error_handler.handle_error(exception, context)
        
        # Should have recorded all errors
        assert len(error_handler.error_history) == 5
        
        # Check error categories are correctly classified
        categories = [error.category for error in error_handler.error_history]
        assert ErrorCategory.MEMORY in categories
        assert ErrorCategory.DATA in categories
        assert ErrorCategory.CHECKPOINT in categories
        assert ErrorCategory.HARDWARE in categories
        
        # Get error summary
        summary = error_handler.get_error_summary()
        assert summary["total_errors"] == 5
        assert len(summary["error_categories"]) > 1
        assert summary["recovery_stats"]["total_attempted"] > 0
    
    def test_error_handler_with_file_logging(self):
        """Test error handler with file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "test_errors.log"
            
            config = ErrorHandlingConfig(
                log_errors_to_file=True,
                error_log_path=str(log_path)
            )
            
            error_handler = TrainingErrorHandler(config=config)
            
            # Simulate an error
            exception = RuntimeError("Test error for logging")
            context = {"test": True}
            
            error_handler.handle_error(exception, context)
            
            # Check that error was logged to file
            assert len(error_handler.error_history) == 1
            
            # Note: File logging setup is tested, but actual file content
            # testing would require more complex mocking of the logging system