#!/usr/bin/env python3
"""
Production Email Dataset Preparation Script

This script prepares a production-ready email dataset for training the EmailTRM model.
It can either use the gmail_dataset_creator to generate real email data or expand
the existing test dataset to meet production requirements.

Requirements:
- At least 1000 emails per category (10 categories = 10,000 total emails)
- Balanced representation across all categories
- Quality validation and content verification
- Proper train/validation/test splits (70%/15%/15%)
"""

import os
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmailSample:
    """Email sample data structure."""
    id: str
    subject: str
    body: str
    sender: str
    recipient: str
    category: str
    language: str = "en"
    timestamp: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'subject': self.subject,
            'body': self.body,
            'sender': self.sender,
            'recipient': self.recipient,
            'category': self.category,
            'language': self.language,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailSample':
        return cls(
            id=data.get('id', ''),
            subject=data.get('subject', ''),
            body=data.get('body', ''),
            sender=data.get('sender', ''),
            recipient=data.get('recipient', ''),
            category=data.get('category', 'other'),
            language=data.get('language', 'en'),
            timestamp=data.get('timestamp')
        )

class ProductionDatasetPreparer:
    """Prepares production email dataset for training."""
    
    def __init__(self, output_path: str = "data/production-emails"):
        self.output_path = Path(output_path)
        self.categories = [
            "newsletter", "work", "personal", "spam", "promotional",
            "social", "finance", "travel", "shopping", "other"
        ]
        self.min_emails_per_category = 1000
        self.target_total_emails = len(self.categories) * self.min_emails_per_category
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self, method: str = "expand_existing") -> Dict[str, Any]:
        """
        Prepare production dataset using specified method.
        
        Args:
            method: "expand_existing", "gmail_creator", or "hybrid"
            
        Returns:
            Dataset preparation statistics
        """
        logger.info(f"Starting production dataset preparation using method: {method}")
        
        if method == "expand_existing":
            return self._expand_existing_dataset()
        elif method == "gmail_creator":
            return self._use_gmail_creator()
        elif method == "hybrid":
            return self._hybrid_approach()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _expand_existing_dataset(self) -> Dict[str, Any]:
        """Expand existing test dataset to production size."""
        logger.info("Expanding existing test dataset to production size...")
        
        # Load existing emails
        existing_emails = self._load_existing_emails()
        logger.info(f"Loaded {len(existing_emails)} existing emails")
        
        # Analyze category distribution
        category_counts = Counter(email.category for email in existing_emails)
        logger.info(f"Current category distribution: {dict(category_counts)}")
        
        # Generate additional emails for each category
        expanded_emails = []
        for category in self.categories:
            existing_category_emails = [e for e in existing_emails if e.category == category]
            needed = self.min_emails_per_category - len(existing_category_emails)
            
            if needed > 0:
                logger.info(f"Generating {needed} additional emails for category: {category}")
                generated = self._generate_emails_for_category(category, existing_category_emails, needed)
                expanded_emails.extend(generated)
            
            # Add existing emails
            expanded_emails.extend(existing_category_emails)
        
        # Shuffle and validate
        random.shuffle(expanded_emails)
        stats = self._validate_dataset(expanded_emails)
        
        # Create splits and save
        self._create_dataset_splits(expanded_emails)
        
        logger.info(f"Dataset expansion completed. Total emails: {len(expanded_emails)}")
        return stats
    
    def _load_existing_emails(self) -> List[EmailSample]:
        """Load existing email samples from test-emails directory."""
        emails = []
        test_emails_path = Path("data/test-emails")
        
        for split in ["train", "val", "test"]:
            split_path = test_emails_path / split
            if split_path.exists():
                for email_file in split_path.glob("*.json"):
                    try:
                        with open(email_file, 'r', encoding='utf-8') as f:
                            email_data = json.load(f)
                            email = EmailSample.from_dict(email_data)
                            emails.append(email)
                    except Exception as e:
                        logger.warning(f"Failed to load {email_file}: {e}")
        
        return emails
    
    def _generate_emails_for_category(self, category: str, existing_emails: List[EmailSample], count: int) -> List[EmailSample]:
        """Generate synthetic emails for a category based on existing patterns."""
        generated = []
        
        # Email templates and patterns for each category
        templates = self._get_email_templates(category)
        
        for i in range(count):
            # Use existing emails as templates if available
            if existing_emails:
                template_email = random.choice(existing_emails)
                base_subject = template_email.subject
                base_body = template_email.body
            else:
                # Use predefined templates
                template = random.choice(templates)
                base_subject = template['subject']
                base_body = template['body']
            
            # Generate variations
            new_email = self._create_email_variation(category, base_subject, base_body, i)
            generated.append(new_email)
        
        return generated
    
    def _get_email_templates(self, category: str) -> List[Dict[str, str]]:
        """Get email templates for each category."""
        templates = {
            "newsletter": [
                {"subject": "Weekly Tech Newsletter", "body": "Here are this week's top tech stories and updates from the industry."},
                {"subject": "Monthly Product Updates", "body": "Check out the latest features and improvements in our products."},
                {"subject": "Industry News Digest", "body": "Stay informed with the latest news and trends in your industry."},
            ],
            "work": [
                {"subject": "Project Status Update", "body": "Please find the latest project status report attached. We are on track for delivery."},
                {"subject": "Meeting Reminder", "body": "This is a reminder about our scheduled meeting tomorrow at 2 PM."},
                {"subject": "Quarterly Review", "body": "Time for our quarterly performance review. Please prepare your reports."},
            ],
            "personal": [
                {"subject": "Birthday Party Invitation", "body": "You're invited to my birthday party this Saturday! Hope to see you there."},
                {"subject": "Weekend Plans", "body": "Hey! What are your plans for this weekend? Want to hang out?"},
                {"subject": "Family Reunion", "body": "Our annual family reunion is coming up. Looking forward to seeing everyone."},
            ],
            "spam": [
                {"subject": "Congratulations! You've Won!", "body": "You have won a million dollars! Click here to claim your prize now!"},
                {"subject": "Urgent: Account Verification Required", "body": "Your account will be suspended unless you verify immediately."},
                {"subject": "Amazing Deal - Limited Time!", "body": "Don't miss this incredible offer! Act now before it's too late!"},
            ],
            "promotional": [
                {"subject": "50% Off Everything!", "body": "Limited time offer - get 50% off all items in our store this weekend only."},
                {"subject": "New Product Launch", "body": "Introducing our latest product with amazing features and benefits."},
                {"subject": "Exclusive Member Discount", "body": "As a valued member, enjoy this exclusive discount on selected items."},
            ],
            "social": [
                {"subject": "New Connection Request", "body": "Someone wants to connect with you on our social platform."},
                {"subject": "Event Invitation", "body": "You're invited to join our upcoming community event this Friday."},
                {"subject": "Photo Tagged", "body": "You've been tagged in a photo. Check it out and share with friends."},
            ],
            "finance": [
                {"subject": "Monthly Statement", "body": "Your monthly account statement is now available for review."},
                {"subject": "Payment Reminder", "body": "This is a friendly reminder that your payment is due soon."},
                {"subject": "Investment Update", "body": "Here's your quarterly investment portfolio performance update."},
            ],
            "travel": [
                {"subject": "Flight Confirmation", "body": "Your flight booking has been confirmed. Please check in online 24 hours before departure."},
                {"subject": "Hotel Reservation", "body": "Your hotel reservation is confirmed. We look forward to your stay."},
                {"subject": "Travel Insurance", "body": "Protect your trip with comprehensive travel insurance coverage."},
            ],
            "shopping": [
                {"subject": "Order Confirmation", "body": "Thank you for your order! Your items will be shipped within 2-3 business days."},
                {"subject": "Cart Reminder", "body": "You have items waiting in your shopping cart. Complete your purchase now."},
                {"subject": "Product Recommendation", "body": "Based on your recent purchases, we think you'll love these items."},
            ],
            "other": [
                {"subject": "System Notification", "body": "This is an automated system notification regarding your account."},
                {"subject": "Survey Request", "body": "We'd love to hear your feedback. Please take a moment to complete our survey."},
                {"subject": "Newsletter Subscription", "body": "Thank you for subscribing to our newsletter. Welcome to our community!"},
            ]
        }
        
        return templates.get(category, templates["other"])
    
    def _create_email_variation(self, category: str, base_subject: str, base_body: str, index: int) -> EmailSample:
        """Create a variation of an email."""
        # Add variation to subject and body
        subject_variations = [
            f"Re: {base_subject}",
            f"Fwd: {base_subject}",
            f"{base_subject} - Update",
            f"{base_subject} #{index + 1}",
            base_subject.replace("Weekly", "Monthly").replace("Daily", "Weekly"),
        ]
        
        body_variations = [
            f"{base_body}\n\nBest regards,\nThe Team",
            f"Hi there,\n\n{base_body}\n\nThanks!",
            f"{base_body}\n\nPlease let us know if you have any questions.",
            f"Dear valued customer,\n\n{base_body}",
            f"{base_body}\n\nThis is an automated message.",
        ]
        
        # Generate unique ID
        email_id = f"prod_{category}_{index + 1000:04d}"
        
        # Create email sample
        return EmailSample(
            id=email_id,
            subject=random.choice(subject_variations),
            body=random.choice(body_variations),
            sender=f"sender_{index}@{category}.com",
            recipient="user@example.com",
            category=category,
            language="en",
            timestamp=None
        )
    
    def _use_gmail_creator(self) -> Dict[str, Any]:
        """Use gmail_dataset_creator to generate real email data."""
        logger.info("Using gmail_dataset_creator to generate real email data...")
        
        try:
            from gmail_dataset_creator import GmailDatasetCreator
            
            # Initialize creator
            creator = GmailDatasetCreator(output_path=str(self.output_path))
            creator.setup()
            
            # Authenticate
            if not creator.authenticate():
                raise RuntimeError("Failed to authenticate with Gmail API")
            
            # Create dataset
            stats = creator.create_dataset(max_emails=self.target_total_emails)
            
            logger.info(f"Gmail dataset creation completed: {stats.total_emails} emails")
            return {
                'method': 'gmail_creator',
                'total_emails': stats.total_emails,
                'processing_time': stats.processing_time,
                'source': 'real_gmail_data'
            }
            
        except ImportError:
            logger.error("gmail_dataset_creator not available. Please install required dependencies.")
            raise
        except Exception as e:
            logger.error(f"Failed to use gmail_dataset_creator: {e}")
            raise
    
    def _hybrid_approach(self) -> Dict[str, Any]:
        """Use hybrid approach: gmail_creator + expanded existing data."""
        logger.info("Using hybrid approach: combining real and synthetic data...")
        
        # Try to use gmail_creator for some categories
        try:
            gmail_stats = self._use_gmail_creator()
            # If successful, supplement with expanded data if needed
            return gmail_stats
        except Exception as e:
            logger.warning(f"Gmail creator failed: {e}. Falling back to expanded dataset.")
            return self._expand_existing_dataset()
    
    def _validate_dataset(self, emails: List[EmailSample]) -> Dict[str, Any]:
        """Validate dataset quality and generate statistics."""
        logger.info("Validating dataset quality...")
        
        # Basic statistics
        total_emails = len(emails)
        category_counts = Counter(email.category for email in emails)
        language_counts = Counter(email.language for email in emails)
        
        # Quality checks
        validation_errors = 0
        for email in emails:
            if not email.subject or not email.body or not email.sender:
                validation_errors += 1
            if len(email.body) < 10:  # Minimum body length
                validation_errors += 1
        
        # Category balance check
        min_count = min(category_counts.values()) if category_counts else 0
        max_count = max(category_counts.values()) if category_counts else 0
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        stats = {
            'total_emails': total_emails,
            'category_distribution': dict(category_counts),
            'language_distribution': dict(language_counts),
            'validation_errors': validation_errors,
            'balance_ratio': balance_ratio,
            'quality_score': 1.0 - (validation_errors / total_emails) if total_emails > 0 else 0,
            'meets_requirements': all(count >= self.min_emails_per_category for count in category_counts.values())
        }
        
        logger.info(f"Dataset validation completed:")
        logger.info(f"  Total emails: {total_emails}")
        logger.info(f"  Category balance ratio: {balance_ratio:.3f}")
        logger.info(f"  Quality score: {stats['quality_score']:.3f}")
        logger.info(f"  Meets requirements: {stats['meets_requirements']}")
        
        return stats
    
    def _create_dataset_splits(self, emails: List[EmailSample]) -> None:
        """Create train/validation/test splits."""
        logger.info("Creating dataset splits (70%/15%/15%)...")
        
        # Shuffle emails
        random.shuffle(emails)
        
        # Calculate split sizes
        total = len(emails)
        train_size = int(total * 0.70)
        val_size = int(total * 0.15)
        test_size = total - train_size - val_size
        
        # Create splits
        train_emails = emails[:train_size]
        val_emails = emails[train_size:train_size + val_size]
        test_emails = emails[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_emails,
            'val': val_emails,
            'test': test_emails
        }
        
        for split_name, split_emails in splits.items():
            split_dir = self.output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save each email as separate JSON file
            for email in split_emails:
                filename = f"{email.id}.json"
                filepath = split_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(email.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(train_emails)} emails")
        logger.info(f"  Validation: {len(val_emails)} emails")
        logger.info(f"  Test: {len(test_emails)} emails")
        
        # Save dataset metadata
        metadata = {
            'total_emails': total,
            'splits': {
                'train': len(train_emails),
                'val': len(val_emails),
                'test': len(test_emails)
            },
            'categories': self.categories,
            'min_emails_per_category': self.min_emails_per_category,
            'creation_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
        }
        
        with open(self.output_path / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare production email dataset")
    parser.add_argument(
        '--method', 
        choices=['expand_existing', 'gmail_creator', 'hybrid'],
        default='expand_existing',
        help='Method to use for dataset preparation'
    )
    parser.add_argument(
        '--output-path',
        default='data/production-emails',
        help='Output path for production dataset'
    )
    parser.add_argument(
        '--min-per-category',
        type=int,
        default=1000,
        help='Minimum emails per category'
    )
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = ProductionDatasetPreparer(args.output_path)
    preparer.min_emails_per_category = args.min_per_category
    
    try:
        # Prepare dataset
        stats = preparer.prepare_dataset(args.method)
        
        logger.info("Production dataset preparation completed successfully!")
        logger.info(f"Statistics: {stats}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())