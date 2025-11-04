"""Tests for Gmail API client functionality."""

import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Mock Google API modules to avoid import errors
sys.modules['google'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.errors'] = MagicMock()
sys.modules['googleapiclient.http'] = MagicMock()

from gmail_dataset_creator.gmail.client import (
    EmailFilter, 
    QueryBuilder,
    RateLimitConfig,
    BatchConfig,
    RateLimiter
)


class TestQueryBuilder(unittest.TestCase):
    """Test query builder functionality."""
    
    def test_build_date_query(self):
        """Test date range query building."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        query = QueryBuilder.build_date_query(start_date, end_date)
        expected = "after:2024/01/01 before:2024/01/31"
        
        self.assertEqual(query, expected)
    
    def test_build_sender_query_single(self):
        """Test single sender query building."""
        senders = ["test@example.com"]
        query = QueryBuilder.build_sender_query(senders)
        
        self.assertEqual(query, "from:test@example.com")
    
    def test_build_sender_query_multiple(self):
        """Test multiple sender query building."""
        senders = ["test1@example.com", "test2@example.com"]
        query = QueryBuilder.build_sender_query(senders)
        
        self.assertEqual(query, "(from:test1@example.com OR from:test2@example.com)")
    
    def test_build_sender_query_domain(self):
        """Test domain sender query building."""
        senders = ["@example.com"]
        query = QueryBuilder.build_sender_query(senders)
        
        self.assertEqual(query, "from:@example.com")
    
    def test_build_query_comprehensive(self):
        """Test comprehensive query building."""
        email_filter = EmailFilter(
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            sender_filters=["test@example.com"],
            query="is:unread",
            exclude_labels=["SPAM", "TRASH"]
        )
        
        query = QueryBuilder.build_query(email_filter)
        
        # Check that all parts are included
        self.assertIn("is:unread", query)
        self.assertIn("after:2024/01/01", query)
        self.assertIn("before:2024/01/31", query)
        self.assertIn("from:test@example.com", query)
        self.assertIn("-label:SPAM", query)
        self.assertIn("-label:TRASH", query)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RateLimitConfig(
            requests_per_second=2.0,
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0
        )
        self.rate_limiter = RateLimiter(self.config)
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        self.assertEqual(self.rate_limiter.config.requests_per_second, 2.0)
        self.assertEqual(self.rate_limiter.request_count, 0)
    
    def test_handle_rate_limit_error(self):
        """Test rate limit error handling."""
        delay = self.rate_limiter.handle_rate_limit_error(0)
        self.assertGreaterEqual(delay, 0.05)  # With jitter, should be at least half base_delay
        self.assertLessEqual(delay, 0.15)     # With jitter, should be at most 1.5x base_delay
    
    def test_max_retries_exceeded(self):
        """Test max retries exceeded exception."""
        with self.assertRaises(Exception) as context:
            self.rate_limiter.handle_rate_limit_error(3)
        
        self.assertIn("Max retries", str(context.exception))


class TestGmailAPIClientMocked(unittest.TestCase):
    """Test Gmail API client functionality with mocked dependencies."""
    
    def test_email_filter_creation(self):
        """Test email filter creation."""
        email_filter = EmailFilter(
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            sender_filters=["test@example.com"],
            max_results=50
        )
        
        self.assertIsNotNone(email_filter.date_range)
        self.assertEqual(email_filter.sender_filters, ["test@example.com"])
        self.assertEqual(email_filter.max_results, 50)
    
    def test_config_creation(self):
        """Test configuration objects creation."""
        rate_config = RateLimitConfig(
            requests_per_second=5.0,
            max_retries=3
        )
        
        batch_config = BatchConfig(
            batch_size=50,
            timeout_seconds=120
        )
        
        self.assertEqual(rate_config.requests_per_second, 5.0)
        self.assertEqual(rate_config.max_retries, 3)
        self.assertEqual(batch_config.batch_size, 50)
        self.assertEqual(batch_config.timeout_seconds, 120)


if __name__ == '__main__':
    unittest.main()