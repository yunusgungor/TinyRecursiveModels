"""
Comprehensive tests for logging and process state management functionality.

Tests the enhanced logging system, API call tracking, progress monitoring,
error handling, and process interruption/resume capabilities.
"""

import unittest
import tempfile
import shutil
import os
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from gmail_dataset_creator.utils.logging import (
    setup_logging, get_api_logger, get_progress_tracker, get_error_logger,
    APICallLogger, ProgressTracker, ErrorLogger, StructuredFormatter
)
from gmail_dataset_creator.utils.process_state import (
    ProcessStateManager, ProcessState, NetworkInterruptionHandler,
    CheckpointData
)
from gmail_dataset_creator.models import EmailData


class TestStructuredLogging(unittest.TestCase):
    """Test structured logging functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        setup_logging(
            level="DEBUG",
            log_file=self.log_file,
            structured_logging=True
        )
        
        logger = get_api_logger(__name__)
        request_id = logger.log_api_call(
            api_name="TestAPI",
            method="GET",
            endpoint="/test",
            status_code=200,
            response_time_ms=150.5
        )
        
        self.assertTrue(os.path.exists(self.log_file))
        self.assertIsNotNone(request_id)
        
        # Check log file content
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("TestAPI", log_content)
            self.assertIn("GET", log_content)
            self.assertIn("/test", log_content)
    
    def test_api_call_logger(self):
        """Test API call logging with timing."""
        setup_logging(level="DEBUG")
        api_logger = get_api_logger(__name__)
        
        # Test successful API call logging
        with api_logger.time_api_call("Gmail", "list_messages", "gmail_api") as request_id:
            time.sleep(0.1)  # Simulate API call
        
        self.assertIsNotNone(request_id)
        self.assertTrue(request_id.startswith("gmail_"))
    
    def test_progress_tracker(self):
        """Test progress tracking functionality."""
        setup_logging(level="DEBUG")
        progress_tracker = get_progress_tracker(__name__)
        
        # Start a stage
        progress_tracker.start_stage("test_processing", 100)
        
        # Update progress
        progress_tracker.update_progress(
            processed_items=50,
            failed_items=5,
            current_item_id="item_50",
            force_log=True
        )
        
        # Complete stage
        progress_tracker.complete_stage()
        
        # Test passes if no exceptions are raised
        self.assertTrue(True)
    
    def test_error_logger(self):
        """Test error logging with context."""
        setup_logging(level="DEBUG")
        error_logger = get_error_logger(__name__)
        
        test_error = ValueError("Test error message")
        
        error_logger.log_error(
            error=test_error,
            component="TestComponent",
            operation="test_operation",
            context_data={"test_key": "test_value"},
            recovery_action="Retry the operation",
            email_id="test_email_123"
        )
        
        # Test passes if no exceptions are raised
        self.assertTrue(True)


class TestProcessStateManager(unittest.TestCase):
    """Test process state management functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.process_manager = ProcessStateManager(
            process_id="test_process",
            checkpoint_dir=self.temp_dir,
            checkpoint_interval=1,  # 1 second for testing
            auto_save=False  # Disable auto-save for controlled testing
        )
    
    def tearDown(self):
        if hasattr(self, 'process_manager'):
            self.process_manager._stop_checkpoint_thread = True
        shutil.rmtree(self.temp_dir)
    
    def test_process_lifecycle(self):
        """Test complete process lifecycle."""
        # Start process
        self.process_manager.start_process("initialization")
        self.assertEqual(self.process_manager.get_state(), ProcessState.RUNNING)
        
        # Update stage
        self.process_manager.update_stage("processing", {"items": 100})
        
        # Update progress
        self.process_manager.update_progress(
            items_processed=50,
            items_failed=2,
            progress_data={"current_batch": 5}
        )
        
        # Create checkpoint
        checkpoint_file = self.process_manager.create_checkpoint("Test checkpoint")
        self.assertTrue(os.path.exists(checkpoint_file))
        
        # Complete process
        self.process_manager.complete_process({"final_count": 100})
        self.assertEqual(self.process_manager.get_state(), ProcessState.COMPLETED)
    
    def test_checkpoint_and_resume(self):
        """Test checkpoint creation and resume functionality."""
        # Start and run process
        self.process_manager.start_process("data_processing")
        self.process_manager.update_progress(
            items_processed=25,
            progress_data={"checkpoint_test": True}
        )
        
        # Create checkpoint
        checkpoint_file = self.process_manager.create_checkpoint("Mid-process checkpoint")
        
        # Load checkpoint
        checkpoint_data = self.process_manager.load_checkpoint(checkpoint_file)
        self.assertIsNotNone(checkpoint_data)
        self.assertEqual(checkpoint_data.stage, "data_processing")
        self.assertEqual(checkpoint_data.progress.get("checkpoint_test"), True)
        
        # Test resume
        recovery_called = False
        def recovery_callback(checkpoint_data):
            nonlocal recovery_called
            recovery_called = True
            return True
        
        self.process_manager.register_recovery_callback("data_processing", recovery_callback)
        
        # Simulate new process manager instance
        new_process_manager = ProcessStateManager(
            process_id="test_process",
            checkpoint_dir=self.temp_dir,
            auto_save=False
        )
        
        success = new_process_manager.resume_from_checkpoint(checkpoint_file)
        self.assertTrue(success)
    
    def test_stage_context_manager(self):
        """Test stage context manager functionality."""
        self.process_manager.start_process("initialization")
        
        # Test successful stage
        with self.process_manager.stage_context("test_stage", {"stage_data": "test"}):
            self.process_manager.update_progress(items_processed=10)
        
        # Test stage with exception
        with self.assertRaises(ValueError):
            with self.process_manager.stage_context("error_stage"):
                raise ValueError("Test error")
        
        self.assertEqual(self.process_manager.get_state(), ProcessState.FAILED)
    
    def test_interruption_handling(self):
        """Test interruption handling."""
        self.process_manager.start_process("long_running_task")
        
        # Register interruption handler
        handler_called = False
        def interruption_handler():
            nonlocal handler_called
            handler_called = True
        
        self.process_manager.register_interruption_handler(interruption_handler)
        
        # Simulate interruption (normally would be signal)
        self.process_manager._interrupted = True
        
        # Check if process detects interruption
        self.assertTrue(self.process_manager.is_interrupted())


class TestNetworkInterruptionHandler(unittest.TestCase):
    """Test network interruption handling."""
    
    def setUp(self):
        self.network_handler = NetworkInterruptionHandler(
            max_retries=2,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            backoff_factor=2.0
        )
    
    def test_successful_operation(self):
        """Test successful network operation."""
        call_count = 0
        
        with self.network_handler.handle_network_operation("test_operation") as attempt:
            call_count += 1
            # Simulate successful operation
            pass
        
        self.assertEqual(call_count, 1)
    
    def test_network_error_retry(self):
        """Test network error retry logic."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network connection failed")
            return "success"
        
        result = self.network_handler.execute_with_retry(failing_operation, "test_operation")
        
        self.assertEqual(call_count, 3)
        self.assertEqual(result, "success")
    
    def test_non_network_error_no_retry(self):
        """Test that non-network errors don't trigger retry."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a network error")
        
        with self.assertRaises(ValueError):
            self.network_handler.execute_with_retry(failing_operation, "test_operation")
        
        self.assertEqual(call_count, 1)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining logging and process management."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        setup_logging(level="DEBUG")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_email_processing_simulation(self):
        """Test simulated email processing with full logging and state management."""
        # Setup components
        process_manager = ProcessStateManager(
            process_id="email_processing_test",
            checkpoint_dir=self.temp_dir,
            auto_save=False
        )
        
        progress_tracker = get_progress_tracker(__name__)
        api_logger = get_api_logger(__name__)
        error_logger = get_error_logger(__name__)
        
        # Sample emails
        emails = [
            EmailData(id=f"email_{i}", subject=f"Test {i}", body=f"Body {i}",
                     sender=f"sender{i}@test.com", recipient="user@test.com",
                     timestamp="2024-01-01T00:00:00", raw_content=f"Raw content {i}")
            for i in range(5)
        ]
        
        try:
            # Start process
            process_manager.start_process("email_processing")
            progress_tracker.start_stage("email_processing", len(emails))
            
            # Process emails
            for i, email in enumerate(emails):
                # Log API call
                with api_logger.time_api_call("Gemini", "classify", "gemini_api") as request_id:
                    time.sleep(0.01)  # Simulate processing
                
                # Update progress
                process_manager.update_progress(
                    items_processed=i + 1,
                    context_data={"current_email": email.id}
                )
                
                progress_tracker.update_progress(
                    processed_items=i + 1,
                    current_item_id=email.id,
                    force_log=True
                )
                
                # Simulate occasional error
                if i == 2:
                    error = Exception("Simulated processing error")
                    error_logger.log_error(
                        error=error,
                        component="EmailProcessor",
                        operation="classify_email",
                        context_data={"email_id": email.id},
                        recovery_action="Skip and continue"
                    )
            
            # Complete processing
            progress_tracker.complete_stage()
            process_manager.complete_process({"total_processed": len(emails)})
            
            # Verify final state
            self.assertEqual(process_manager.get_state(), ProcessState.COMPLETED)
            
        finally:
            process_manager._stop_checkpoint_thread = True


if __name__ == '__main__':
    unittest.main()