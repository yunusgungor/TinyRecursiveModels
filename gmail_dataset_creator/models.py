"""
Core data models and interfaces for the Gmail Dataset Creator.

This module defines the base data structures used throughout the system
for email data, classification results, and dataset statistics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List


@dataclass
class EmailData:
    """
    Core data structure for email information.
    
    Represents an email with all necessary metadata for processing
    and classification.
    """
    id: str
    subject: str
    body: str
    sender: str
    recipient: str
    timestamp: datetime
    raw_content: str
    
    def __post_init__(self):
        """Validate email data after initialization."""
        if not self.id:
            raise ValueError("Email ID cannot be empty")
        if not self.subject and not self.body:
            raise ValueError("Email must have either subject or body content")


@dataclass
class ClassificationResult:
    """
    Result of email classification by Gemini API.
    
    Contains the predicted category, confidence score, and additional
    metadata about the classification process.
    """
    category: str
    confidence: float
    reasoning: str
    needs_review: bool
    
    def __post_init__(self):
        """Validate classification result after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not self.category:
            raise ValueError("Category cannot be empty")


@dataclass
class DatasetStats:
    """
    Statistics about the generated dataset.
    
    Provides comprehensive information about the dataset creation process
    including distribution, counts, and performance metrics.
    """
    total_emails: int
    categories_distribution: Dict[str, int]
    train_count: int
    test_count: int
    vocabulary_size: int
    processing_time: float
    
    def __post_init__(self):
        """Validate dataset statistics after initialization."""
        if self.total_emails < 0:
            raise ValueError("Total emails cannot be negative")
        if self.train_count + self.test_count != self.total_emails:
            raise ValueError("Train and test counts must sum to total emails")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
    
    @property
    def train_ratio(self) -> float:
        """Calculate the train/test split ratio."""
        if self.total_emails == 0:
            return 0.0
        return self.train_count / self.total_emails
    
    def get_category_balance(self) -> Dict[str, float]:
        """Get the percentage distribution of categories."""
        if self.total_emails == 0:
            return {}
        return {
            category: count / self.total_emails 
            for category, count in self.categories_distribution.items()
        }


# Email categories mapping
CATEGORIES = {
    "newsletter": 0,
    "work": 1,
    "personal": 2,
    "spam": 3,
    "promotional": 4,
    "social": 5,
    "finance": 6,
    "travel": 7,
    "shopping": 8,
    "other": 9
}

# Reverse mapping for category lookup
CATEGORY_NAMES = {v: k for k, v in CATEGORIES.items()}


def validate_category(category: str) -> bool:
    """
    Validate if a category is supported.
    
    Args:
        category: Category name to validate
        
    Returns:
        True if category is valid, False otherwise
    """
    return category in CATEGORIES


def get_category_id(category: str) -> int:
    """
    Get numeric ID for a category.
    
    Args:
        category: Category name
        
    Returns:
        Numeric category ID
        
    Raises:
        ValueError: If category is not supported
    """
    if not validate_category(category):
        raise ValueError(f"Unsupported category: {category}")
    return CATEGORIES[category]


def get_category_name(category_id: int) -> str:
    """
    Get category name from numeric ID.
    
    Args:
        category_id: Numeric category ID
        
    Returns:
        Category name
        
    Raises:
        ValueError: If category ID is not supported
    """
    if category_id not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported category ID: {category_id}")
    return CATEGORY_NAMES[category_id]