"""
Example demonstrating process interruption handling and resume capability.

This example shows how the Gmail Dataset Creator handles interruptions
gracefully and can resume from checkpoints.
"""

import os
import sys
import time
import signal
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gmail_dataset_creator.utils.process_state import ProcessStateManager, ProcessState
from gmail_dataset_creator.utils.logging import setup_logging, get_progress_tracker, get_error_logger
from gmail_dataset_creator.models import EmailData, ClassificationResult


def simulate_email_processing_with_interruption():
    """
    Simulate email processing that can be interrupted and resumed.
    """
    # Setup logging
    setup_logging(level="INFO", structured_logging=True)
    
    # Initialize process state manager
    process_manager = ProcessStateManager(
        process_id="email_processing_demo",
        checkpoint_dir="./demo_checkpoints",
        checkpoint_interval=5,  # Checkpoint every 5 seconds for demo
        auto_save=True
    )
    
    # Get logging components
    progress_tracker = get_progress_tracker(__name__)
    error_logger = get_error_logger(__name__)
    
    # Sample email data for demonstration
    sample_emails = [
        EmailData(id=f"email_{i}", subject=f"Test Email {i}", 
                 body=f"This is test email {i}", sender=f"sender{i}@example.com",
                 recipient="user@example.com", timestamp="2024-01-01T00:00:00",
                 raw_content=f"Raw content for email {i}")
        for i in range(1, 21)  # 20 emails for demo
    ]
    
    def recovery_callback(checkpoint_data) -> bool:
        """Recovery callback for resuming processing."""
        print(f"Recovering from checkpoint at stage: {checkpoint_data.stage}")
        print(f"Progress: {checkpoint_data.progress}")
        return True
    
    # Register recovery callback
    process_manager.register_recovery_callback("email_processing", recovery_callback)
    
    # Try to resume from existing checkpoint
    if process_manager.resume_from_checkpoint():
        print("Resumed from existing checkpoint!")
        # Get current progress
        progress = process_manager.get_progress()
        processed_count = progress['progress'].get('processed_count', 0)
        print(f"Continuing from email {processed_count + 1}")
    else:
        print("Starting new process...")
        processed_count = 0
        process_manager.start_process("email_processing")
    
    # Start progress tracking
    progress_tracker.start_stage("email_processing", len(sample_emails))
    
    try:
        # Process emails with interruption handling
        with process_manager.stage_context("email_processing"):
            for i, email in enumerate(sample_emails[processed_count:], processed_count):
                # Check for interruption
                if process_manager.is_interrupted():
                    print("Process interrupted, creating final checkpoint...")
                    break
                
                # Simulate processing time
                print(f"Processing email {i + 1}: {email.subject}")
                time.sleep(1)  # Simulate processing delay
                
                # Update progress
                process_manager.update_progress(
                    items_processed=i + 1,
                    progress_data={'processed_count': i + 1, 'current_email_id': email.id}
                )
                
                progress_tracker.update_progress(
                    processed_items=i + 1,
                    current_item_id=email.id
                )
                
                # Simulate occasional errors
                if i == 10:  # Simulate error on 11th email
                    error = Exception("Simulated processing error")
                    error_logger.log_error(
                        error=error,
                        component="EmailProcessor",
                        operation="process_email",
                        context_data={'email_id': email.id, 'email_index': i},
                        recovery_action="Skip email and continue processing"
                    )
                    print(f"Error processing email {i + 1}, continuing...")
                    continue
        
        # Complete processing
        progress_tracker.complete_stage()
        process_manager.complete_process({'total_processed': len(sample_emails)})
        print("Email processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C)")
        process_manager.pause_process()
        print("Process state saved. You can resume later.")
        
    except Exception as e:
        error_logger.log_error(
            error=e,
            component="EmailProcessor",
            operation="simulate_processing",
            recovery_action="Resume from checkpoint"
        )
        process_manager.fail_process(e, ["Resume from checkpoint", "Check system resources"])
        print(f"Process failed: {e}")


if __name__ == "__main__":
    print("Gmail Dataset Creator - Process Interruption Demo")
    print("=" * 50)
    print("This demo shows process interruption handling and resume capability.")
    print("Press Ctrl+C to interrupt the process, then run again to resume.")
    print("=" * 50)
    
    simulate_email_processing_with_interruption()