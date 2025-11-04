"""
Main entry point for Gmail Dataset Creator.

This module provides the primary GmailDatasetCreator class that orchestrates
the entire dataset creation process from authentication to final export.
"""

import os
import logging
from typing import Optional, Tuple
from datetime import datetime

from .config.manager import ConfigManager, SystemConfig
from .models import DatasetStats
from .utils.logging import setup_logging


class GmailDatasetCreator:
    """
    Main orchestrator for Gmail dataset creation.
    
    This class coordinates all components of the system to create
    email classification datasets from Gmail data using Gemini API
    for intelligent categorization.
    """
    
    def __init__(self, config_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        Initialize Gmail Dataset Creator.
        
        Args:
            config_path: Path to configuration YAML file
            output_path: Override output path for datasets
        """
        self.config_manager = ConfigManager(config_path)
        self.config: Optional[SystemConfig] = None
        self.output_path_override = output_path
        self.logger = logging.getLogger(__name__)
        
        # Component instances (initialized during setup)
        self._auth_handler = None
        self._gmail_client = None
        self._email_processor = None
        self._gemini_classifier = None
        self._dataset_builder = None
        
        # Process state management
        self._process_manager = None
        try:
            from .utils.logging import get_progress_tracker, get_error_logger
            self.progress_tracker = get_progress_tracker(__name__)
            self.error_logger = get_error_logger(__name__)
        except ImportError:
            self.progress_tracker = None
            self.error_logger = None
        
        # Setup logging with structured format and file output
        log_file = os.path.join(output_path or "./logs", "gmail_dataset_creator.log") if output_path else None
        setup_logging(
            level="INFO",
            log_file=log_file,
            structured_logging=True,
            max_log_size_mb=50,
            backup_count=3
        )
        self.logger.info("Gmail Dataset Creator initialized")
    
    def setup(self) -> None:
        """
        Setup and initialize all system components.
        
        Loads configuration, validates settings, and initializes
        all necessary components for dataset creation.
        
        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing
        """
        self.logger.info("Setting up Gmail Dataset Creator...")
        
        # Load configuration
        self.config = self.config_manager.load_config()
        
        # Override output path if provided
        if self.output_path_override:
            self.config.dataset.output_path = self.output_path_override
        
        # Validate required files exist
        self._validate_setup()
        
        # Initialize components (will be implemented in subsequent tasks)
        self._initialize_components()
        
        self.logger.info("Setup completed successfully")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Gmail API.
        
        Handles OAuth2 authentication flow and token management.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self._auth_handler:
            raise RuntimeError("System not setup. Call setup() first.")
        
        self.logger.info("Starting Gmail API authentication...")
        
        try:
            success = self._auth_handler.authenticate()
            if success:
                self.logger.info("Gmail API authentication successful")
                return True
            else:
                self.logger.error("Gmail API authentication failed")
                return False
        except Exception as e:
            self.error_logger.log_error(
                error=e,
                component="AuthenticationHandler",
                operation="authenticate",
                context_data={'config_file': self.config.gmail_api.credentials_file},
                recovery_action="Check credentials file and network connectivity"
            )
            return False
    
    def create_dataset(
        self, 
        max_emails: Optional[int] = None,
        date_range: Optional[Tuple[str, str]] = None,
        resume_from_checkpoint: bool = True
    ) -> DatasetStats:
        """
        Create email classification dataset with comprehensive error handling and resume capability.
        
        Orchestrates the complete dataset creation process:
        1. Fetch emails from Gmail
        2. Process and clean email content
        3. Classify emails using Gemini API
        4. Build and export datasets
        
        Args:
            max_emails: Maximum number of emails to process (overrides config)
            date_range: Date range filter (start_date, end_date) in YYYY-MM-DD format
            resume_from_checkpoint: Whether to resume from previous checkpoint if available
            
        Returns:
            Dataset statistics and metadata
            
        Raises:
            RuntimeError: If system not setup or authentication failed
        """
        if not self.config:
            raise RuntimeError("System not setup. Call setup() first.")
        
        # Initialize process state manager
        process_id = f"dataset_creation_{int(datetime.now().timestamp())}"
        try:
            from .utils.process_state import ProcessStateManager
            self._process_manager = ProcessStateManager(
                process_id=process_id,
                checkpoint_dir=os.path.join(self.config.dataset.output_path, "checkpoints"),
                checkpoint_interval=60,  # Checkpoint every minute
                auto_save=True
            )
        except ImportError:
            self.logger.warning("ProcessStateManager not available, continuing without checkpoints")
            self._process_manager = None
        
        # Register interruption handler
        def cleanup_on_interruption():
            self.logger.info("Cleaning up resources due to interruption...")
            # Add any cleanup logic here
        
        self._process_manager.register_interruption_handler(cleanup_on_interruption)
        
        # Try to resume from checkpoint if requested
        if resume_from_checkpoint:
            checkpoint_data = self._process_manager.load_checkpoint()
            if checkpoint_data:
                self.logger.info(f"Found existing checkpoint from {checkpoint_data.timestamp}")
                if self._process_manager.resume_from_checkpoint():
                    self.logger.info("Successfully resumed from checkpoint")
                    # Continue with the process from where it left off
                    return self._continue_dataset_creation_from_checkpoint(checkpoint_data)
        
        # Start new process
        self.logger.info("Starting dataset creation process...")
        start_time = datetime.now()
        
        try:
            self._process_manager.start_process("initialization")
            
            # Override configuration if parameters provided
            if max_emails:
                self.config.dataset.max_emails_total = max_emails
            if date_range:
                self.config.filters.date_range = date_range
            
            # Store configuration in process context
            self._process_manager.update_progress(
                context_data={
                    'max_emails': self.config.dataset.max_emails_total,
                    'date_range': self.config.filters.date_range,
                    'output_path': self.config.dataset.output_path
                }
            )
            
            # Execute dataset creation stages with process management
            stats = self._execute_dataset_creation_stages()
            
            # Mark process as completed
            self._process_manager.complete_process({
                'final_stats': {
                    'total_emails': stats.total_emails,
                    'processing_time': stats.processing_time
                }
            })
            
            self.logger.info(f"Dataset creation completed in {stats.processing_time:.2f} seconds")
            return stats
            
        except KeyboardInterrupt:
            self.logger.warning("Dataset creation interrupted by user")
            self._process_manager.pause_process()
            raise
            
        except Exception as e:
            self.error_logger.log_error(
                error=e,
                component="GmailDatasetCreator",
                operation="create_dataset",
                context_data={
                    'process_id': process_id,
                    'max_emails': max_emails,
                    'date_range': date_range
                },
                recovery_action="Check logs and resume from checkpoint"
            )
            
            # Mark process as failed with recovery suggestions
            recovery_actions = [
                "Check network connectivity",
                "Verify API credentials",
                "Resume from checkpoint with: resume_from_checkpoint=True",
                "Reduce batch size or email count"
            ]
            self._process_manager.fail_process(e, recovery_actions)
            raise
    
    def _execute_dataset_creation_stages(self) -> DatasetStats:
        """
        Execute the main dataset creation stages with process state management.
        
        Returns:
            Dataset statistics
        """
        from datetime import datetime
        start_time = datetime.now()
        emails_processed = []
        
        # Stage 1: Email Fetching
        with self._process_manager.stage_context("email_fetching"):
            self.logger.info("Stage 1: Fetching emails from Gmail...")
            
            # Fetch emails from Gmail with simple query
            raw_emails = self._gmail_client.list_messages(
                max_results=self.config.dataset.max_emails_total,
                include_spam_trash=False
            )
            
            # Get full message content for each email
            message_ids = [msg['id'] for msg in raw_emails.get('messages', [])]
            if message_ids:
                raw_emails = self._gmail_client.get_messages_batch(message_ids, format='full')
            else:
                raw_emails = []
            
            self.logger.info(f"Fetched {len(raw_emails)} emails from Gmail")
            self._process_manager.update_progress(
                items_processed=len(raw_emails),
                progress_data={'stage_status': 'completed', 'emails_fetched': len(raw_emails)}
            )
        
        # Stage 2: Email Processing
        with self._process_manager.stage_context("email_processing"):
            self.logger.info("Stage 2: Processing and cleaning email content...")
            
            processed_emails = []
            for i, raw_email in enumerate(raw_emails):
                try:
                    email_data = self._email_processor.extract_content(raw_email)
                    if email_data:  # Only add if processing was successful
                        processed_emails.append(email_data)
                    
                    # Update progress periodically
                    if (i + 1) % 10 == 0:
                        self._process_manager.update_progress(
                            items_processed=i + 1,
                            progress_data={'processed_count': len(processed_emails)}
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process email {i}: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {len(processed_emails)} emails")
            self._process_manager.update_progress(
                items_processed=len(processed_emails),
                progress_data={'stage_status': 'completed', 'processed_emails': len(processed_emails)}
            )
        
        # Stage 3: Email Classification
        with self._process_manager.stage_context("email_classification"):
            self.logger.info("Stage 3: Classifying emails using Gemini API...")
            
            classified_emails = []
            for i, email_data in enumerate(processed_emails):
                try:
                    classification = self._gemini_classifier.classify_email(email_data)
                    
                    # Only include emails with sufficient confidence
                    if not classification.needs_review:
                        classified_emails.append((email_data, classification))
                    else:
                        self.logger.info(f"Email {email_data.id} flagged for manual review (confidence: {classification.confidence})")
                    
                    # Update progress periodically
                    if (i + 1) % 5 == 0:  # Less frequent updates due to API calls
                        self._process_manager.update_progress(
                            items_processed=i + 1,
                            progress_data={'classified_count': len(classified_emails)}
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Failed to classify email {email_data.id}: {e}")
                    continue
            
            self.logger.info(f"Successfully classified {len(classified_emails)} emails")
            self._process_manager.update_progress(
                items_processed=len(classified_emails),
                progress_data={'stage_status': 'completed', 'classified_emails': len(classified_emails)}
            )
        
        # Stage 4: Dataset Building
        with self._process_manager.stage_context("dataset_building"):
            self.logger.info("Stage 4: Building and exporting datasets...")
            
            # Add classified emails to dataset builder
            for email_data, classification in classified_emails:
                self._dataset_builder.add_email(email_data, classification)
            
            # Check category distribution and warn about imbalances
            distribution = self._dataset_builder.get_category_distribution()
            for category, count in distribution.items():
                if count < self.config.dataset.min_emails_per_category:
                    self.logger.warning(
                        f"Category '{category}' has only {count} emails "
                        f"(minimum recommended: {self.config.dataset.min_emails_per_category})"
                    )
            
            # Build and export datasets
            stats = self._dataset_builder.export_dataset()
            
            self.logger.info(f"Dataset export completed: {stats.total_emails} total emails")
            self._process_manager.update_progress(
                items_processed=stats.total_emails,
                progress_data={'stage_status': 'completed', 'final_stats': stats}
            )
        
        # Calculate final processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        stats.processing_time = processing_time
        
        return stats
    
    def _continue_dataset_creation_from_checkpoint(self, checkpoint_data: 'CheckpointData') -> DatasetStats:
        """
        Continue dataset creation from a checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data to resume from
            
        Returns:
            Dataset statistics
        """
        self.logger.info(f"Continuing dataset creation from stage: {checkpoint_data.stage}")
        
        # Restore context data
        context = checkpoint_data.context_data
        if 'max_emails' in context:
            self.config.dataset.max_emails_total = context['max_emails']
        if 'date_range' in context:
            self.config.filters.date_range = context['date_range']
        
        # Continue from the appropriate stage
        # This would contain logic to resume from the specific stage
        # For now, just execute all stages
        return self._execute_dataset_creation_stages()
    
    def _validate_setup(self) -> None:
        """
        Validate that all required files and settings are available.
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If configuration is invalid
        """
        # Check Gmail credentials file
        credentials_path = self.config.gmail_api.credentials_file
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Gmail credentials file not found: {credentials_path}. "
                "Please download credentials.json from Google Cloud Console."
            )
        
        # Check Gemini API key
        if not self.config.gemini_api.api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY environment variable or update config file."
            )
        
        # Ensure output directory exists
        output_dir = self.config.dataset.output_path
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Setup validation completed")
    
    def _initialize_components(self) -> None:
        """
        Initialize all system components.
        
        Initializes all components that have been implemented in previous tasks.
        """
        self.logger.info("Initializing system components...")
        
        # Task 2: Authentication Handler
        from .auth.authentication import AuthenticationHandler, AuthConfig
        auth_config = AuthConfig(
            credentials_file=self.config.gmail_api.credentials_file,
            token_file=self.config.gmail_api.token_file,
            scopes=self.config.gmail_api.scopes
        )
        self._auth_handler = AuthenticationHandler(auth_config)
        
        # Task 3: Gmail API Client
        from .gmail.client import GmailAPIClient, RateLimitConfig, BatchConfig
        rate_config = RateLimitConfig(
            requests_per_second=10.0,
            max_retries=3,
            base_delay=1.0
        )
        batch_config = BatchConfig(batch_size=100)
        
        # Get credentials from auth handler
        credentials = self._auth_handler.get_credentials()
        if not credentials:
            # Try to authenticate first
            if not self._auth_handler.authenticate():
                raise RuntimeError("Failed to authenticate with Gmail API")
            credentials = self._auth_handler.get_credentials()
        
        self._gmail_client = GmailAPIClient(
            credentials=credentials,
            rate_limit_config=rate_config,
            batch_config=batch_config
        )
        
        # Task 4: Email Processor
        from .processing.email_processor import EmailProcessor
        self._email_processor = EmailProcessor(
            anonymize_content=self.config.privacy.anonymize_senders,
            remove_sensitive=getattr(self.config.privacy, 'remove_sensitive_content', True)
        )
        
        # Task 5: Gemini Classifier
        from .processing.gemini_classifier import GeminiClassifier
        self._gemini_classifier = GeminiClassifier(
            config=self.config.gemini_api,
            confidence_threshold=getattr(self.config.privacy, 'min_confidence_threshold', 0.7)
        )
        
        # Task 6: Dataset Builder
        from .dataset.builder import DatasetBuilder
        self._dataset_builder = DatasetBuilder(
            output_path=self.config.dataset.output_path,
            train_ratio=self.config.dataset.train_ratio,
            min_emails_per_category=self.config.dataset.min_emails_per_category
        )
        
        self.logger.info("Component initialization completed")
    
    def get_status(self) -> dict:
        """
        Get current system status and configuration.
        
        Returns:
            Dictionary containing system status information
        """
        if not self.config:
            return {"status": "not_setup", "message": "System not initialized"}
        
        return {
            "status": "ready",
            "config": {
                "output_path": self.config.dataset.output_path,
                "max_emails": self.config.dataset.max_emails_total,
                "train_ratio": self.config.dataset.train_ratio,
                "date_range": self.config.filters.date_range,
            },
            "components": {
                "auth_handler": self._auth_handler is not None,
                "gmail_client": self._gmail_client is not None,
                "email_processor": self._email_processor is not None,
                "gemini_classifier": self._gemini_classifier is not None,
                "dataset_builder": self._dataset_builder is not None,
            }
        }


def main():
    """Console script entry point."""
    from .cli import main as cli_main
    cli_main()