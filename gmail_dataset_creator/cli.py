"""
Command-line interface for Gmail Dataset Creator.

Provides a comprehensive CLI with argument parsing, interactive mode,
and usage examples for creating email classification datasets.
"""

import argparse
import os
import sys
import yaml
from typing import Optional, Dict, Any
from datetime import datetime

from .main import GmailDatasetCreator
from .config.manager import ConfigManager


class GmailDatasetCreatorCLI:
    """
    Command-line interface for Gmail Dataset Creator.
    
    Provides both command-line argument parsing and interactive mode
    for configuring and running the dataset creation process.
    """
    
    def __init__(self):
        """Initialize CLI with argument parser."""
        self.parser = self._create_parser()
        self.creator: Optional[GmailDatasetCreator] = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            prog='gmail-dataset-creator',
            description='Create email classification datasets from Gmail data using Gemini API',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # Configuration options
        config_group = parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration YAML file'
        )
        config_group.add_argument(
            '--output', '-o',
            type=str,
            help='Output directory for generated datasets'
        )
        config_group.add_argument(
            '--credentials',
            type=str,
            help='Path to Gmail API credentials JSON file'
        )
        config_group.add_argument(
            '--gemini-key',
            type=str,
            help='Gemini API key (or set GEMINI_API_KEY environment variable)'
        )
        
        # Dataset options
        dataset_group = parser.add_argument_group('Dataset Options')
        dataset_group.add_argument(
            '--max-emails',
            type=int,
            help='Maximum number of emails to process'
        )
        dataset_group.add_argument(
            '--train-ratio',
            type=float,
            help='Train/test split ratio (0.0-1.0)'
        )
        dataset_group.add_argument(
            '--date-start',
            type=str,
            help='Start date for email filtering (YYYY-MM-DD)'
        )
        dataset_group.add_argument(
            '--date-end',
            type=str,
            help='End date for email filtering (YYYY-MM-DD)'
        )
        
        # Privacy and security options
        privacy_group = parser.add_argument_group('Privacy & Security')
        privacy_group.add_argument(
            '--anonymize-senders',
            action='store_true',
            help='Anonymize sender email addresses'
        )
        privacy_group.add_argument(
            '--exclude-personal',
            action='store_true',
            help='Exclude personal emails from dataset'
        )
        privacy_group.add_argument(
            '--confidence-threshold',
            type=float,
            help='Minimum confidence threshold for classifications (0.0-1.0)'
        )
        
        # Process control options
        process_group = parser.add_argument_group('Process Control')
        process_group.add_argument(
            '--resume',
            action='store_true',
            help='Resume from previous checkpoint if available'
        )
        process_group.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Run in interactive mode for configuration'
        )
        process_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Show configuration and exit without processing'
        )
        
        # Utility commands
        utility_group = parser.add_argument_group('Utilities')
        utility_group.add_argument(
            '--auth-only',
            action='store_true',
            help='Only perform authentication, then exit'
        )
        utility_group.add_argument(
            '--status',
            action='store_true',
            help='Show current system status and configuration'
        )
        utility_group.add_argument(
            '--generate-config',
            type=str,
            metavar='PATH',
            help='Generate sample configuration file at specified path'
        )
        
        # Logging options
        logging_group = parser.add_argument_group('Logging')
        logging_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        logging_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-error output'
        )
        logging_group.add_argument(
            '--log-file',
            type=str,
            help='Path to log file'
        )
        
        return parser
    
    def _get_usage_examples(self) -> str:
        """Get usage examples for help text."""
        return """
Examples:
  # Basic usage with interactive configuration
  gmail-dataset-creator --interactive
  
  # Create dataset with specific parameters
  gmail-dataset-creator --config config.yaml --max-emails 1000 --output ./my_dataset
  
  # Use date range filtering
  gmail-dataset-creator --date-start 2023-01-01 --date-end 2023-12-31 --max-emails 500
  
  # Generate sample configuration file
  gmail-dataset-creator --generate-config config.yaml
  
  # Check authentication status
  gmail-dataset-creator --auth-only --verbose
  
  # Resume interrupted process
  gmail-dataset-creator --resume --config config.yaml
  
  # Dry run to check configuration
  gmail-dataset-creator --config config.yaml --dry-run

Environment Variables:
  GEMINI_API_KEY          Gemini API key for email classification
  GMAIL_CREDENTIALS_PATH  Path to Gmail API credentials file
  OUTPUT_PATH             Default output directory for datasets
"""
    
    def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI application.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            parsed_args = self.parser.parse_args(args)
            
            # Handle utility commands first
            if parsed_args.generate_config:
                return self._generate_config(parsed_args.generate_config)
            
            # Setup logging based on arguments
            self._setup_logging(parsed_args)
            
            # Run interactive mode if requested
            if parsed_args.interactive:
                parsed_args = self._run_interactive_mode(parsed_args)
            
            # Initialize creator with configuration
            self._initialize_creator(parsed_args)
            
            # Handle status command
            if parsed_args.status:
                return self._show_status()
            
            # Handle dry run
            if parsed_args.dry_run:
                return self._run_dry_run()
            
            # Handle authentication only
            if parsed_args.auth_only:
                return self._run_auth_only()
            
            # Run main dataset creation process
            return self._run_dataset_creation(parsed_args)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _setup_logging(self, args: argparse.Namespace) -> None:
        """Setup logging based on command line arguments."""
        import logging
        
        level = logging.INFO
        if args.verbose:
            level = logging.DEBUG
        elif args.quiet:
            level = logging.ERROR
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                *([logging.FileHandler(args.log_file)] if args.log_file else [])
            ]
        )
    
    def _run_interactive_mode(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Run interactive configuration mode.
        
        Args:
            args: Current parsed arguments
            
        Returns:
            Updated arguments with interactive input
        """
        print("=== Gmail Dataset Creator - Interactive Configuration ===\n")
        
        # Configuration file
        if not args.config:
            config_path = input("Configuration file path (press Enter for default): ").strip()
            if config_path:
                args.config = config_path
        
        # Output directory
        if not args.output:
            output_path = input("Output directory (press Enter for './gmail_dataset'): ").strip()
            args.output = output_path or './gmail_dataset'
        
        # Gmail credentials
        if not args.credentials:
            creds_path = input("Gmail credentials file path (press Enter for 'credentials.json'): ").strip()
            args.credentials = creds_path or 'credentials.json'
        
        # Gemini API key
        if not args.gemini_key and not os.getenv('GEMINI_API_KEY'):
            api_key = input("Gemini API key (or set GEMINI_API_KEY env var): ").strip()
            if api_key:
                args.gemini_key = api_key
        
        # Dataset options
        if not args.max_emails:
            max_emails = input("Maximum number of emails to process (press Enter for 1000): ").strip()
            if max_emails:
                try:
                    args.max_emails = int(max_emails)
                except ValueError:
                    print("Invalid number, using default (1000)")
                    args.max_emails = 1000
            else:
                args.max_emails = 1000
        
        # Date range
        if not args.date_start:
            date_start = input("Start date for filtering (YYYY-MM-DD, press Enter to skip): ").strip()
            if date_start:
                args.date_start = date_start
        
        if not args.date_end:
            date_end = input("End date for filtering (YYYY-MM-DD, press Enter to skip): ").strip()
            if date_end:
                args.date_end = date_end
        
        # Privacy options
        if not args.anonymize_senders:
            anonymize = input("Anonymize sender addresses? (y/N): ").strip().lower()
            args.anonymize_senders = anonymize in ('y', 'yes')
        
        print("\n=== Configuration Complete ===\n")
        return args
    
    def _initialize_creator(self, args: argparse.Namespace) -> None:
        """Initialize GmailDatasetCreator with configuration."""
        # Set environment variables from arguments
        if args.gemini_key:
            os.environ['GEMINI_API_KEY'] = args.gemini_key
        if args.credentials:
            os.environ['GMAIL_CREDENTIALS_PATH'] = args.credentials
        if args.output:
            os.environ['OUTPUT_PATH'] = args.output
        if args.max_emails:
            os.environ['MAX_EMAILS'] = str(args.max_emails)
        if args.train_ratio:
            os.environ['TRAIN_RATIO'] = str(args.train_ratio)
        
        # Initialize creator
        self.creator = GmailDatasetCreator(
            config_path=args.config,
            output_path=args.output
        )
        
        # Setup the creator
        self.creator.setup()
        
        # Apply additional configuration overrides
        if args.date_start or args.date_end:
            date_range = (
                args.date_start or "1900-01-01",
                args.date_end or "2100-12-31"
            )
            self.creator.config.filters.date_range = date_range
        
        if args.anonymize_senders:
            self.creator.config.privacy.anonymize_senders = True
        
        if args.exclude_personal:
            self.creator.config.privacy.exclude_personal = True
        
        if args.confidence_threshold:
            self.creator.config.privacy.min_confidence_threshold = args.confidence_threshold
    
    def _show_status(self) -> int:
        """Show current system status."""
        if not self.creator:
            print("Error: Creator not initialized")
            return 1
        
        status = self.creator.get_status()
        
        print("=== Gmail Dataset Creator Status ===")
        print(f"Status: {status['status']}")
        
        if 'config' in status:
            config = status['config']
            print(f"\nConfiguration:")
            print(f"  Output Path: {config['output_path']}")
            print(f"  Max Emails: {config['max_emails']}")
            print(f"  Train Ratio: {config['train_ratio']}")
            if config['date_range']:
                print(f"  Date Range: {config['date_range'][0]} to {config['date_range'][1]}")
        
        if 'components' in status:
            components = status['components']
            print(f"\nComponents:")
            for component, initialized in components.items():
                status_icon = "✓" if initialized else "✗"
                print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        return 0
    
    def _run_dry_run(self) -> int:
        """Run dry run to show configuration without processing."""
        if not self.creator:
            print("Error: Creator not initialized")
            return 1
        
        print("=== Dry Run - Configuration Preview ===")
        
        config = self.creator.config
        print(f"Gmail API:")
        print(f"  Credentials: {config.gmail_api.credentials_file}")
        print(f"  Scopes: {', '.join(config.gmail_api.scopes)}")
        
        print(f"\nGemini API:")
        print(f"  Model: {config.gemini_api.model}")
        print(f"  Max Tokens: {config.gemini_api.max_tokens}")
        
        print(f"\nDataset:")
        print(f"  Output Path: {config.dataset.output_path}")
        print(f"  Max Emails: {config.dataset.max_emails_total}")
        print(f"  Train Ratio: {config.dataset.train_ratio}")
        print(f"  Min per Category: {config.dataset.min_emails_per_category}")
        
        print(f"\nFilters:")
        if config.filters.date_range:
            print(f"  Date Range: {config.filters.date_range[0]} to {config.filters.date_range[1]}")
        print(f"  Include Labels: {', '.join(config.filters.include_labels)}")
        print(f"  Exclude Labels: {', '.join(config.filters.exclude_labels)}")
        
        print(f"\nPrivacy:")
        print(f"  Anonymize Senders: {config.privacy.anonymize_senders}")
        print(f"  Exclude Personal: {config.privacy.exclude_personal}")
        print(f"  Confidence Threshold: {config.privacy.min_confidence_threshold}")
        
        print("\n=== Dry Run Complete ===")
        return 0
    
    def _run_auth_only(self) -> int:
        """Run authentication only."""
        if not self.creator:
            print("Error: Creator not initialized")
            return 1
        
        print("=== Gmail API Authentication ===")
        
        success = self.creator.authenticate()
        if success:
            print("✓ Authentication successful!")
            return 0
        else:
            print("✗ Authentication failed!")
            return 1
    
    def _run_dataset_creation(self, args: argparse.Namespace) -> int:
        """Run the main dataset creation process."""
        if not self.creator:
            print("Error: Creator not initialized")
            return 1
        
        print("=== Starting Dataset Creation ===")
        
        # Authenticate first
        print("Authenticating with Gmail API...")
        if not self.creator.authenticate():
            print("✗ Authentication failed!")
            return 1
        print("✓ Authentication successful!")
        
        # Create dataset
        try:
            stats = self.creator.create_dataset(
                max_emails=args.max_emails,
                date_range=(args.date_start, args.date_end) if args.date_start or args.date_end else None,
                resume_from_checkpoint=args.resume
            )
            
            # Display results
            print("\n=== Dataset Creation Complete ===")
            print(f"Total Emails Processed: {stats.total_emails}")
            print(f"Train Set Size: {stats.train_count}")
            print(f"Test Set Size: {stats.test_count}")
            print(f"Vocabulary Size: {stats.vocabulary_size}")
            print(f"Processing Time: {stats.processing_time:.2f} seconds")
            
            print(f"\nCategory Distribution:")
            for category, count in stats.categories_distribution.items():
                percentage = (count / stats.total_emails * 100) if stats.total_emails > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")
            
            print(f"\nDataset files saved to: {self.creator.config.dataset.output_path}")
            
            return 0
            
        except Exception as e:
            print(f"✗ Dataset creation failed: {e}")
            return 1
    
    def _generate_config(self, config_path: str) -> int:
        """Generate sample configuration file."""
        sample_config = {
            'gmail_api': {
                'credentials_file': 'credentials.json',
                'token_file': 'token.json',
                'scopes': ['https://www.googleapis.com/auth/gmail.readonly']
            },
            'gemini_api': {
                'api_key': '${GEMINI_API_KEY}',
                'model': 'gemini-pro',
                'max_tokens': 1000
            },
            'dataset': {
                'output_path': './gmail_dataset',
                'train_ratio': 0.8,
                'min_emails_per_category': 10,
                'max_emails_total': 1000
            },
            'filters': {
                'date_range': None,
                'exclude_labels': ['TRASH', 'SPAM'],
                'include_labels': ['INBOX'],
                'sender_filters': []
            },
            'privacy': {
                'anonymize_senders': True,
                'exclude_personal': False,
                'remove_attachments': True,
                'encrypt_tokens': True,
                'exclude_sensitive': True,
                'anonymize_recipients': True,
                'remove_sensitive_content': True,
                'exclude_keywords': [],
                'exclude_domains': [],
                'min_confidence_threshold': 0.7
            },
            'security': {
                'encryption_algorithm': 'fernet',
                'key_derivation_function': 'pbkdf2',
                'encryption_iterations': 100000,
                'salt_length': 32,
                'secure_export': True,
                'data_retention_days': 30,
                'secure_cleanup': True,
                'audit_logging': True
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
            print(f"✓ Sample configuration file generated: {config_path}")
            print("\nNext steps:")
            print("1. Set your Gemini API key: export GEMINI_API_KEY='your-api-key'")
            print("2. Download Gmail API credentials to 'credentials.json'")
            print("3. Edit the configuration file as needed")
            print(f"4. Run: gmail-dataset-creator --config {config_path}")
            
            return 0
            
        except Exception as e:
            print(f"✗ Failed to generate config file: {e}")
            return 1


def main():
    """Main entry point for CLI."""
    cli = GmailDatasetCreatorCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()