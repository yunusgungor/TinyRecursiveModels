#!/usr/bin/env python3
"""
Example usage of the Gemini classifier for email classification.

This script demonstrates how to use the GeminiClassifier to classify emails
using Google's Gemini API with various configuration options.
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmail_dataset_creator.models import EmailData
from gmail_dataset_creator.config.manager import GeminiAPIConfig
from gmail_dataset_creator.processing.gemini_classifier import (
    GeminiClassifier, 
    BatchProcessingConfig, 
    RateLimitConfig
)


def create_sample_emails():
    """Create sample emails for testing classification."""
    sample_emails = [
        EmailData(
            id="email_001",
            subject="Weekly Team Meeting",
            body="Hi team, don't forget about our weekly standup meeting tomorrow at 10 AM. We'll discuss the project progress and upcoming deadlines.",
            sender="manager@company.com",
            recipient="team@company.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_002",
            subject="50% Off Everything - Limited Time!",
            body="Don't miss our biggest sale of the year! Get 50% off all items with code SAVE50. Shop now before it's too late!",
            sender="sales@retailstore.com",
            recipient="customer@example.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_003",
            subject="Your Monthly Newsletter",
            body="Here's what's new this month: product updates, industry insights, and upcoming events. Stay informed with our latest news.",
            sender="newsletter@techcompany.com",
            recipient="subscriber@example.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_004",
            subject="Flight Confirmation - Trip to Paris",
            body="Your flight booking is confirmed! Flight AA123 departing on March 15th at 8:30 AM. Please arrive at the airport 2 hours early.",
            sender="bookings@airline.com",
            recipient="traveler@example.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_005",
            subject="Happy Birthday!",
            body="Hey! Just wanted to wish you a very happy birthday. Hope you have a wonderful day filled with joy and celebration!",
            sender="friend@personal.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
    ]
    return sample_emails


def main():
    """Main function demonstrating Gemini classifier usage."""
    print("Gemini Email Classifier Example")
    print("=" * 40)
    
    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Configure the classifier
    gemini_config = GeminiAPIConfig(
        api_key=api_key,
        model="gemini-2.0-flash-001",  # Use the latest model
        max_tokens=500
    )
    
    batch_config = BatchProcessingConfig(
        batch_size=3,
        max_concurrent_batches=1,
        retry_failed_emails=True,
        save_progress=False  # Disable for this example
    )
    
    rate_limit_config = RateLimitConfig(
        requests_per_minute=10,  # Conservative rate limiting
        requests_per_hour=100,
        backoff_factor=2.0,
        jitter=True
    )
    
    # Initialize the classifier
    print("Initializing Gemini classifier...")
    try:
        classifier = GeminiClassifier(
            config=gemini_config,
            confidence_threshold=0.7,
            batch_config=batch_config,
            rate_limit_config=rate_limit_config
        )
        print("✓ Classifier initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize classifier: {e}")
        return
    
    # Create sample emails
    emails = create_sample_emails()
    print(f"\nCreated {len(emails)} sample emails for classification")
    
    # Classify individual emails
    print("\n" + "=" * 40)
    print("Individual Email Classification")
    print("=" * 40)
    
    for i, email in enumerate(emails[:2]):  # Classify first 2 emails individually
        print(f"\nClassifying email {i+1}: '{email.subject}'")
        try:
            result = classifier.classify_email(email)
            print(f"  Category: {result.category}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Needs Review: {result.needs_review}")
            print(f"  Reasoning: {result.reasoning}")
        except Exception as e:
            print(f"  ✗ Classification failed: {e}")
    
    # Batch classification
    print("\n" + "=" * 40)
    print("Batch Email Classification")
    print("=" * 40)
    
    print(f"\nClassifying {len(emails)} emails in batch...")
    try:
        def progress_callback(completed, total):
            print(f"  Progress: {completed}/{total} emails classified")
        
        results = classifier.classify_batch(
            emails, 
            progress_callback=progress_callback
        )
        
        print(f"\n✓ Batch classification completed!")
        print("\nResults Summary:")
        print("-" * 20)
        
        for email, result in zip(emails, results):
            status = "✓" if not result.needs_review else "⚠"
            print(f"{status} {email.subject[:30]:<30} → {result.category:<12} ({result.confidence:.2f})")
        
        # Show statistics
        stats = classifier.get_detailed_statistics()
        print(f"\nClassification Statistics:")
        print(f"  Total: {stats['total_classifications']}")
        print(f"  Successful: {stats['successful_classifications']}")
        print(f"  Failed: {stats['failed_classifications']}")
        print(f"  Low Confidence: {stats['low_confidence_classifications']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        
        if stats['api_errors']:
            print(f"  API Errors: {stats['api_errors']}")
        
    except Exception as e:
        print(f"✗ Batch classification failed: {e}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()