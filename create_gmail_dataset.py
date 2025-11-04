#!/usr/bin/env python3
"""
Command-line interface for Gmail Dataset Creator.

This script provides a simple CLI for creating email classification
datasets from Gmail data using the Gmail Dataset Creator system.
"""

import argparse
import sys
import os
from typing import Optional, Tuple

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmail_dataset_creator import GmailDatasetCreator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create email classification datasets from Gmail data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_gmail_dataset.py --config config.yaml
  python create_gmail_dataset.py --max-emails 500 --output ./my_dataset
  python create_gmail_dataset.py --date-range 2024-01-01 2024-12-31
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for generated datasets"
    )
    
    parser.add_argument(
        "--max-emails",
        type=int,
        help="Maximum number of emails to process"
    )
    
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Date range filter (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--authenticate-only",
        action="store_true",
        help="Only perform authentication, don't create dataset"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and configuration"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    try:
        # Initialize Gmail Dataset Creator
        creator = GmailDatasetCreator(
            config_path=args.config if os.path.exists(args.config) else None,
            output_path=args.output
        )
        
        # Setup system
        print("Setting up Gmail Dataset Creator...")
        creator.setup()
        
        # Show status if requested
        if args.status:
            status = creator.get_status()
            print(f"System Status: {status['status']}")
            print(f"Configuration: {status['config']}")
            print(f"Components: {status['components']}")
            return
        
        # Authenticate with Gmail
        print("Authenticating with Gmail API...")
        if not creator.authenticate():
            print("Authentication failed. Please check your credentials.")
            return 1
        
        # Exit if only authentication was requested
        if args.authenticate_only:
            print("Authentication completed successfully.")
            return 0
        
        # Create dataset
        print("Creating email classification dataset...")
        
        date_range: Optional[Tuple[str, str]] = None
        if args.date_range:
            date_range = (args.date_range[0], args.date_range[1])
        
        stats = creator.create_dataset(
            max_emails=args.max_emails,
            date_range=date_range
        )
        
        # Display results
        print("\nDataset Creation Complete!")
        print(f"Total emails processed: {stats.total_emails}")
        print(f"Training emails: {stats.train_count}")
        print(f"Test emails: {stats.test_count}")
        print(f"Vocabulary size: {stats.vocabulary_size}")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        
        if stats.categories_distribution:
            print("\nCategory distribution:")
            for category, count in stats.categories_distribution.items():
                percentage = (count / stats.total_emails) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files are available.")
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())