"""
Gmail Dataset Creator - A system for creating email classification datasets from Gmail data.

This package provides tools to:
- Authenticate with Gmail API using OAuth2
- Fetch emails from Gmail accounts with configurable filters
- Classify emails using Gemini API
- Generate structured training/test datasets
"""

from .main import GmailDatasetCreator
from .models import EmailData, ClassificationResult, DatasetStats

__version__ = "1.0.0"
__all__ = ["GmailDatasetCreator", "EmailData", "ClassificationResult", "DatasetStats"]