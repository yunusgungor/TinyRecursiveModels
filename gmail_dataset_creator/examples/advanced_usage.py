#!/usr/bin/env python3
"""
Advanced usage example for Gmail Dataset Creator.

This example demonstrates advanced features including:
- Custom configuration files
- Privacy settings
- Process interruption and resume
- Custom filtering options
"""

import os
import yaml
from gmail_dataset_creator import GmailDatasetCreator


def create_custom_config():
    """Create a custom configuration file."""
    config = {
        'gmail_api': {
            'credentials_file': './credentials.json',
            'token_file': './token.json',
            'scopes': ['https://www.googleapis.com/auth/gmail.readonly']
        },
        'gemini_api': {
            'api_key': '${GEMINI_API_KEY}',
            'model': 'gemini-pro',
            'max_tokens': 1000
        },
        'dataset': {
            'output_path': './advanced_email_dataset',
            'train_ratio': 0.85,  # 85% train, 15% test
            'min_emails_per_category': 20,
            'max_emails_total': 2000
        },
        'filters': {
            'date_range': ['2023-06-01', '2024-06-01'],
            'exclude_labels': ['TRASH', 'SPAM', 'DRAFT'],
            'include_labels': ['INBOX', 'SENT'],
            'sender_filters': []  # Can add specific senders
        },
        'privacy': {
            'anonymize_senders': True,
            'anonymize_recipients': True,
            'exclude_personal': False,
            'remove_attachments': True,
            'encrypt_tokens': True,
            'exclude_sensitive': True,
            'remove_sensitive_content': True,
            'exclude_keywords': ['password', 'ssn', 'credit card'],
            'exclude_domains': ['internal-company.com'],
            'min_confidence_threshold': 0.8
        },
        'security': {
            'encryption_algorithm': 'fernet',
            'secure_export': True,
            'data_retention_days': 30,
            'secure_cleanup': True,
            'audit_logging': True
        }
    }
    
    with open('advanced_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return 'advanced_config.yaml'


def main():
    """Advanced usage example."""
    print("=== Gmail Dataset Creator - Advanced Usage Example ===\n")
    
    # Create custom configuration
    print("Creating custom configuration...")
    config_path = create_custom_config()
    print(f"Configuration saved to: {config_path}")
    
    # Set environment variables
    os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key-here'
    
    # Initialize with custom configuration
    creator = GmailDatasetCreator(config_path=config_path)
    
    try:
        # Setup the system
        print("Setting up Gmail Dataset Creator with custom configuration...")
        creator.setup()
        
        # Show current status
        print("Current system status:")
        status = creator.get_status()
        print(f"  Status: {status['status']}")
        print(f"  Output path: {status['config']['output_path']}")
        print(f"  Max emails: {status['config']['max_emails']}")
        print(f"  Train ratio: {status['config']['train_ratio']}")
        
        # Authenticate
        print("\nAuthenticating with Gmail API...")
        if not creator.authenticate():
            print("Authentication failed! Please check your credentials.")
            return
        
        print("Authentication successful!")
        
        # Create dataset with resume capability
        print("Creating email dataset (with resume capability)...")
        try:
            stats = creator.create_dataset(
                resume_from_checkpoint=True  # Will resume if interrupted
            )
            
            # Display detailed results
            print("\n=== Dataset Creation Complete ===")
            print(f"Total emails processed: {stats.total_emails}")
            print(f"Train set size: {stats.train_count} ({stats.train_ratio:.1%})")
            print(f"Test set size: {stats.test_count}")
            print(f"Vocabulary size: {stats.vocabulary_size}")
            print(f"Processing time: {stats.processing_time:.2f} seconds")
            
            print("\nDetailed category distribution:")
            category_balance = stats.get_category_balance()
            for category, count in stats.categories_distribution.items():
                percentage = category_balance[category] * 100
                print(f"  {category:12}: {count:4d} emails ({percentage:5.1f}%)")
            
            print(f"\nDataset files saved to: {creator.config.dataset.output_path}")
            
            # Show file structure
            import os
            output_path = creator.config.dataset.output_path
            if os.path.exists(output_path):
                print(f"\nGenerated files:")
                for root, dirs, files in os.walk(output_path):
                    level = root.replace(output_path, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            print("You can resume later by running with --resume flag or resume_from_checkpoint=True")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Check the logs for more detailed error information.")


if __name__ == '__main__':
    main()