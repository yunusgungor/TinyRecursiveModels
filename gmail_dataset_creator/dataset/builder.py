"""
Dataset builder for creating train/test splits and managing email collections.

This module provides the DatasetBuilder class that handles:
- Email collection management
- Train/test splitting with configurable ratios
- Category distribution tracking and balance warnings
- Dataset export in JSONL format
"""

import json
import os
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import warnings

from ..models import EmailData, ClassificationResult, DatasetStats, CATEGORIES, validate_category


class DatasetBuilder:
    """
    Builds and manages email datasets with train/test splitting.
    
    This class maintains collections of emails, handles train/test splitting,
    tracks category distributions, and provides warnings for class imbalance.
    """
    
    def __init__(self, output_path: str, train_ratio: float = 0.8, min_emails_per_category: int = 10):
        """
        Initialize the dataset builder.
        
        Args:
            output_path: Directory path for dataset output
            train_ratio: Ratio of emails to use for training (0.0 to 1.0)
            min_emails_per_category: Minimum emails per category to avoid warnings
        """
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError("Train ratio must be between 0.0 and 1.0")
        
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.min_emails_per_category = min_emails_per_category
        
        # Email collections organized by category
        self.emails_by_category: Dict[str, List[Tuple[EmailData, ClassificationResult]]] = defaultdict(list)
        
        # Statistics tracking
        self.total_emails = 0
        self.processing_start_time = None
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "train").mkdir(exist_ok=True)
        (self.output_path / "test").mkdir(exist_ok=True)
    
    def add_email(self, email_data: EmailData, classification_result: ClassificationResult) -> None:
        """
        Add an email to the dataset collection.
        
        Args:
            email_data: The email data to add
            classification_result: The classification result for the email
            
        Raises:
            ValueError: If the category is not supported
        """
        if not validate_category(classification_result.category):
            raise ValueError(f"Unsupported category: {classification_result.category}")
        
        # Start timing on first email
        if self.processing_start_time is None:
            self.processing_start_time = datetime.now()
        
        self.emails_by_category[classification_result.category].append((email_data, classification_result))
        self.total_emails += 1
    
    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get the current distribution of emails by category.
        
        Returns:
            Dictionary mapping category names to email counts
        """
        return {category: len(emails) for category, emails in self.emails_by_category.items()}
    
    def check_category_balance(self) -> List[str]:
        """
        Check for category imbalance and return warnings.
        
        Returns:
            List of warning messages about category imbalance
        """
        warnings_list = []
        distribution = self.get_category_distribution()
        
        # Check for categories with too few emails
        for category, count in distribution.items():
            if count < self.min_emails_per_category:
                warnings_list.append(
                    f"Category '{category}' has only {count} emails "
                    f"(minimum recommended: {self.min_emails_per_category})"
                )
        
        # Check for missing categories
        for category in CATEGORIES.keys():
            if category not in distribution or distribution[category] == 0:
                warnings_list.append(f"Category '{category}' has no emails")
        
        # Check for severe imbalance (category has less than 5% of total)
        if self.total_emails > 0:
            for category, count in distribution.items():
                percentage = (count / self.total_emails) * 100
                if percentage < 5.0:
                    warnings_list.append(
                        f"Category '{category}' represents only {percentage:.1f}% of total emails"
                    )
        
        return warnings_list
    
    def create_train_test_split(self) -> Tuple[List[Tuple[EmailData, ClassificationResult]], 
                                             List[Tuple[EmailData, ClassificationResult]]]:
        """
        Create train/test split maintaining category distribution.
        
        Returns:
            Tuple of (train_emails, test_emails) lists
        """
        train_emails = []
        test_emails = []
        
        # Split each category separately to maintain distribution
        for category, emails in self.emails_by_category.items():
            if not emails:
                continue
            
            # Shuffle emails for random split
            shuffled_emails = emails.copy()
            random.shuffle(shuffled_emails)
            
            # Calculate split point
            train_count = int(len(emails) * self.train_ratio)
            
            # Ensure at least one email in test if we have more than one email
            if len(emails) > 1 and train_count == len(emails):
                train_count = len(emails) - 1
            
            # Split the emails
            train_emails.extend(shuffled_emails[:train_count])
            test_emails.extend(shuffled_emails[train_count:])
        
        # Final shuffle to mix categories
        random.shuffle(train_emails)
        random.shuffle(test_emails)
        
        return train_emails, test_emails
    
    def _email_to_dict(self, email_data: EmailData, classification_result: ClassificationResult) -> Dict:
        """
        Convert email data to dictionary format for export.
        
        Args:
            email_data: The email data
            classification_result: The classification result
            
        Returns:
            Dictionary representation of the email
        """
        return {
            "id": email_data.id,
            "subject": email_data.subject,
            "body": email_data.body,
            "sender": email_data.sender,
            "recipient": email_data.recipient,
            "category": classification_result.category,
            "language": "en",  # Default to English for now
            "timestamp": email_data.timestamp.isoformat() if email_data.timestamp else None
        }
    
    def export_dataset(self, filename: str = "dataset.json") -> DatasetStats:
        """
        Export the dataset to JSONL format files.
        
        Args:
            filename: Name of the dataset files to create
            
        Returns:
            DatasetStats object with information about the exported dataset
        """
        if self.total_emails == 0:
            raise ValueError("No emails to export")
        
        # Check for balance warnings
        balance_warnings = self.check_category_balance()
        if balance_warnings:
            for warning in balance_warnings:
                warnings.warn(f"Dataset balance warning: {warning}")
        
        # Create train/test split
        train_emails, test_emails = self.create_train_test_split()
        
        # Export train dataset
        train_path = self.output_path / "train" / filename
        with open(train_path, 'w', encoding='utf-8') as f:
            for email_data, classification_result in train_emails:
                email_dict = self._email_to_dict(email_data, classification_result)
                f.write(json.dumps(email_dict, ensure_ascii=False) + '\n')
        
        # Export test dataset
        test_path = self.output_path / "test" / filename
        with open(test_path, 'w', encoding='utf-8') as f:
            for email_data, classification_result in test_emails:
                email_dict = self._email_to_dict(email_data, classification_result)
                f.write(json.dumps(email_dict, ensure_ascii=False) + '\n')
        
        # Calculate processing time
        processing_time = 0.0
        if self.processing_start_time:
            processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        
        # Generate vocabulary and export metadata
        vocabulary = self.build_vocabulary()
        vocab_size = len(vocabulary)
        
        # Export vocabulary
        self._export_vocabulary(vocabulary)
        
        # Export category mappings
        self._export_category_mappings()
        
        # Export dataset metadata
        self._export_metadata(len(train_emails), len(test_emails), vocab_size, processing_time)
        
        # Create dataset statistics
        stats = DatasetStats(
            total_emails=self.total_emails,
            categories_distribution=self.get_category_distribution(),
            train_count=len(train_emails),
            test_count=len(test_emails),
            vocabulary_size=vocab_size,
            processing_time=processing_time
        )
        
        return stats
    
    def get_stats(self) -> DatasetStats:
        """
        Get current dataset statistics without exporting.
        
        Returns:
            DatasetStats object with current information
        """
        processing_time = 0.0
        if self.processing_start_time:
            processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        
        # Estimate train/test counts based on current ratio
        train_count = int(self.total_emails * self.train_ratio)
        test_count = self.total_emails - train_count
        
        return DatasetStats(
            total_emails=self.total_emails,
            categories_distribution=self.get_category_distribution(),
            train_count=train_count,
            test_count=test_count,
            vocabulary_size=0,  # Will be set by vocabulary generation
            processing_time=processing_time
        )
    
    def build_vocabulary(self) -> Dict[str, int]:
        """
        Build vocabulary from all email content.
        
        Extracts unique tokens from email subjects and bodies,
        creating a mapping from tokens to unique IDs.
        
        Returns:
            Dictionary mapping tokens to unique integer IDs
        """
        token_counter = Counter()
        
        # Process all emails to collect tokens
        for category_emails in self.emails_by_category.values():
            for email_data, _ in category_emails:
                # Tokenize subject and body
                subject_tokens = self._tokenize_text(email_data.subject)
                body_tokens = self._tokenize_text(email_data.body)
                
                # Count tokens
                token_counter.update(subject_tokens)
                token_counter.update(body_tokens)
        
        # Create vocabulary mapping (most frequent tokens get lower IDs)
        vocabulary = {}
        for idx, (token, _) in enumerate(token_counter.most_common()):
            vocabulary[token] = idx
        
        return vocabulary
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Convert to lowercase and split on whitespace and punctuation
        text = text.lower()
        # Keep alphanumeric characters and some punctuation
        text = re.sub(r'[^\w\s\-@.]', ' ', text)
        # Split on whitespace
        tokens = text.split()
        
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        
        return tokens
    
    def _export_vocabulary(self, vocabulary: Dict[str, int]) -> None:
        """
        Export vocabulary to JSON file.
        
        Args:
            vocabulary: Dictionary mapping tokens to IDs
        """
        vocab_path = self.output_path / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    
    def _export_category_mappings(self) -> None:
        """Export category mappings to JSON file."""
        categories_path = self.output_path / "categories.json"
        
        # Create comprehensive category information
        category_info = {
            "categories": CATEGORIES,
            "category_names": {v: k for k, v in CATEGORIES.items()},
            "total_categories": len(CATEGORIES),
            "description": "Email classification categories mapping"
        }
        
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(category_info, f, ensure_ascii=False, indent=2)
    
    def _export_metadata(self, train_count: int, test_count: int, vocab_size: int, processing_time: float) -> None:
        """
        Export dataset metadata to JSON file.
        
        Args:
            train_count: Number of training emails
            test_count: Number of test emails
            vocab_size: Size of vocabulary
            processing_time: Time taken to process dataset
        """
        metadata_path = self.output_path / "metadata.json"
        
        metadata = {
            "dataset_info": {
                "total_emails": self.total_emails,
                "train_count": train_count,
                "test_count": test_count,
                "train_ratio": self.train_ratio,
                "vocabulary_size": vocab_size
            },
            "category_distribution": self.get_category_distribution(),
            "category_balance": self._calculate_category_percentages(),
            "processing_info": {
                "processing_time_seconds": processing_time,
                "created_at": datetime.now().isoformat(),
                "min_emails_per_category": self.min_emails_per_category
            },
            "file_structure": {
                "train_data": "train/dataset.json",
                "test_data": "test/dataset.json",
                "vocabulary": "vocab.json",
                "categories": "categories.json",
                "metadata": "metadata.json"
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _calculate_category_percentages(self) -> Dict[str, float]:
        """
        Calculate percentage distribution of categories.
        
        Returns:
            Dictionary mapping category names to percentages
        """
        if self.total_emails == 0:
            return {}
        
        distribution = self.get_category_distribution()
        return {
            category: (count / self.total_emails) * 100
            for category, count in distribution.items()
        }