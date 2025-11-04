#!/usr/bin/env python3
"""
Basic usage example for Gmail Dataset Creator.

This example demonstrates the most common use case: creating an email
classification dataset from Gmail data with default settings.
"""

import os
from gmail_dataset_creator import GmailDatasetCreator


def main():
    """Basic usage example."""
    print("=== Gmail Dataset Creator - Basic Usage Example ===\n")
    
    # Set up environment variables (you can also use a config file)
    os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key-here'
    os.environ['GMAIL_CREDENTIALS_PATH'] = './credentials.json'
    
    # Initialize the creator
    creator = GmailDatasetCreator(output_path='./my_email_dataset')
    
    try:
        # Setup the system
        print("Setting up Gmail Dataset Creator...")
        creator.setup()
        
        # Authenticate with Gmail
        print("Authenticating with Gmail API...")
        if not creator.authenticate():
            print("Authentication failed! Please check your credentials.")
            return
        
        print("Authentication successful!")
        
        # Create the dataset
        print("Creating email dataset...")
        stats = creator.create_dataset(
            max_emails=500,  # Process up to 500 emails
            date_range=("2023-01-01", "2024-12-31")  # Filter by date range
        )
        
        # Display results
        print("\n=== Dataset Creation Complete ===")
        print(f"Total emails processed: {stats.total_emails}")
        print(f"Train set size: {stats.train_count}")
        print(f"Test set size: {stats.test_count}")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        
        print("\nCategory distribution:")
        for category, count in stats.categories_distribution.items():
            percentage = (count / stats.total_emails * 100) if stats.total_emails > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\nDataset files saved to: ./my_email_dataset")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()