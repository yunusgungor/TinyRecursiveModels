"""
Tests for the Gemini classifier module.

This module contains unit tests for the GeminiClassifier class,
testing classification logic, error handling, and batch processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import tempfile
import os

from gmail_dataset_creator.models import EmailData, ClassificationResult
from gmail_dataset_creator.config.manager import GeminiAPIConfig
from gmail_dataset_creator.processing.gemini_classifier import (
    GeminiClassifier, 
    BatchProcessingConfig, 
    RateLimitConfig,
    ClassificationPrompt
)


class TestClassificationPrompt(unittest.TestCase):
    """Test the ClassificationPrompt class."""
    
    def test_create_classification_prompt(self):
        """Test prompt creation from email data."""
        email_data = EmailData(
            id="test_001",
            subject="Meeting Tomorrow",
            body="Don't forget about our meeting tomorrow at 2 PM.",
            sender="manager@company.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
        
        prompt = ClassificationPrompt.create_classification_prompt(email_data)
        
        self.assertIn("Meeting Tomorrow", prompt)
        self.assertIn("company.com", prompt)
        self.assertIn("Don't forget about our meeting", prompt)
    
    def test_create_classification_prompt_with_long_content(self):
        """Test prompt creation with content that needs truncation."""
        long_body = "A" * 2000  # Very long body
        long_subject = "B" * 300  # Very long subject
        
        email_data = EmailData(
            id="test_002",
            subject=long_subject,
            body=long_body,
            sender="test@example.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
        
        prompt = ClassificationPrompt.create_classification_prompt(email_data)
        
        # Check that content was truncated
        self.assertLess(len(prompt), 2000)  # Should be much shorter than original
        self.assertIn("example.com", prompt)  # Domain should still be there


class TestGeminiClassifier(unittest.TestCase):
    """Test the GeminiClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GeminiAPIConfig(
            api_key="test_api_key",
            model="gemini-pro",
            max_tokens=1000
        )
        
        self.batch_config = BatchProcessingConfig(
            batch_size=2,
            max_concurrent_batches=1,
            save_progress=False  # Disable for tests
        )
        
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100
        )
        
        # Create test email data
        self.test_email = EmailData(
            id="test_001",
            subject="Work Meeting",
            body="Please join the team meeting tomorrow.",
            sender="manager@company.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
    
    @patch('gmail_dataset_creator.processing.gemini_classifier.genai.Client')
    def test_classifier_initialization(self, mock_client_class):
        """Test classifier initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        classifier = GeminiClassifier(
            config=self.config,
            batch_config=self.batch_config,
            rate_limit_config=self.rate_limit_config
        )
        
        self.assertEqual(classifier.config, self.config)
        self.assertEqual(classifier.batch_config, self.batch_config)
        self.assertEqual(classifier.rate_limit_config, self.rate_limit_config)
        mock_client_class.assert_called_once_with(api_key="test_api_key")
    
    @patch('gmail_dataset_creator.processing.gemini_classifier.genai.Client')
    def test_classify_email_success(self, mock_client_class):
        """Test successful email classification."""
        # Mock API response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "category": "work",
            "confidence": 0.9,
            "reasoning": "Email contains work-related content about meetings",
            "needs_review": False
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        with patch.object(classifier, '_apply_rate_limiting'):
            result = classifier.classify_email(self.test_email)
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.category, "work")
        self.assertEqual(result.confidence, 0.9)
        self.assertFalse(result.needs_review)
        self.assertIn("work-related", result.reasoning)
    
    @patch('gmail_dataset_creator.processing.gemini_classifier.genai.Client')
    def test_classify_email_with_invalid_category(self, mock_client_class):
        """Test classification with invalid category response."""
        # Mock API response with invalid category
        mock_response = Mock()
        mock_response.text = json.dumps({
            "category": "invalid_category",
            "confidence": 0.8,
            "reasoning": "Some reasoning",
            "needs_review": False
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        with patch.object(classifier, '_apply_rate_limiting'):
            result = classifier.classify_email(self.test_email)
        
        # Should default to "other" for invalid category
        self.assertEqual(result.category, "other")
        self.assertTrue(result.needs_review)  # Should flag for review
        self.assertLess(result.confidence, 0.8)  # Confidence should be reduced
    
    @patch('gmail_dataset_creator.processing.gemini_classifier.genai.Client')
    def test_classify_email_api_error(self, mock_client_class):
        """Test classification with API error."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        with patch.object(classifier, '_apply_rate_limiting'):
            result = classifier.classify_email(self.test_email)
        
        # Should return fallback classification
        self.assertIsInstance(result, ClassificationResult)
        self.assertTrue(result.needs_review)
        self.assertLess(result.confidence, 0.5)  # Low confidence for fallback
    
    @patch('gmail_dataset_creator.processing.gemini_classifier.genai.Client')
    def test_classify_batch(self, mock_client_class):
        """Test batch classification."""
        # Create multiple test emails
        emails = [
            EmailData(
                id=f"test_{i:03d}",
                subject=f"Test Email {i}",
                body=f"This is test email number {i}",
                sender="test@example.com",
                recipient="user@example.com",
                timestamp=datetime.now(),
                raw_content=""
            )
            for i in range(3)
        ]
        
        # Mock API responses
        mock_response = Mock()
        mock_response.text = json.dumps({
            "category": "other",
            "confidence": 0.7,
            "reasoning": "Test email classification",
            "needs_review": False
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        with patch.object(classifier, '_apply_rate_limiting'):
            results = classifier.classify_batch(emails)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, ClassificationResult)
            self.assertEqual(result.category, "other")
    
    def test_rule_based_classification(self):
        """Test rule-based fallback classification."""
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        # Test work-related email
        work_email = EmailData(
            id="work_001",
            subject="Team Meeting Tomorrow",
            body="Please join our project meeting at 2 PM",
            sender="manager@company.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
        
        category = classifier._rule_based_classification(work_email)
        self.assertEqual(category, "work")
        
        # Test promotional email
        promo_email = EmailData(
            id="promo_001",
            subject="50% Sale Today Only!",
            body="Don't miss our amazing discount offer",
            sender="noreply@store.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
        
        category = classifier._rule_based_classification(promo_email)
        self.assertEqual(category, "promotional")
    
    def test_get_statistics(self):
        """Test statistics tracking."""
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        # Simulate some classifications
        classifier.total_classifications = 10
        classifier.successful_classifications = 8
        classifier.failed_classifications = 2
        classifier.low_confidence_classifications = 3
        
        stats = classifier.get_statistics()
        
        self.assertEqual(stats["total_classifications"], 10)
        self.assertEqual(stats["successful_classifications"], 8)
        self.assertEqual(stats["failed_classifications"], 2)
        self.assertEqual(stats["success_rate"], 0.8)
    
    def test_handle_api_error_content_policy(self):
        """Test handling of content policy violations."""
        classifier = GeminiClassifier(config=self.config, batch_config=self.batch_config)
        
        error = Exception("Content policy violation detected")
        result = classifier.handle_api_error(error, "test_001")
        
        self.assertEqual(result.category, "spam")
        self.assertTrue(result.needs_review)
        self.assertIn("content policy", result.reasoning.lower())


class TestBatchProcessingConfig(unittest.TestCase):
    """Test the BatchProcessingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BatchProcessingConfig()
        
        self.assertEqual(config.batch_size, 5)
        self.assertEqual(config.max_concurrent_batches, 2)
        self.assertTrue(config.retry_failed_emails)
        self.assertTrue(config.save_progress)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchProcessingConfig(
            batch_size=10,
            max_concurrent_batches=4,
            retry_failed_emails=False,
            save_progress=False
        )
        
        self.assertEqual(config.batch_size, 10)
        self.assertEqual(config.max_concurrent_batches, 4)
        self.assertFalse(config.retry_failed_emails)
        self.assertFalse(config.save_progress)


class TestRateLimitConfig(unittest.TestCase):
    """Test the RateLimitConfig class."""
    
    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        
        self.assertEqual(config.requests_per_minute, 60)
        self.assertEqual(config.requests_per_hour, 1000)
        self.assertEqual(config.backoff_factor, 2.0)
        self.assertTrue(config.jitter)
    
    def test_custom_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            backoff_factor=1.5,
            jitter=False
        )
        
        self.assertEqual(config.requests_per_minute, 30)
        self.assertEqual(config.requests_per_hour, 500)
        self.assertEqual(config.backoff_factor, 1.5)
        self.assertFalse(config.jitter)


if __name__ == '__main__':
    unittest.main()