"""
Tests for the dataset builder module.

This module contains unit tests for the DatasetBuilder class,
testing dataset creation, train/test splitting, and export functionality.
"""

import unittest
import tempfile
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Mock any missing modules
sys.modules['google'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()
sys.modules['google.auth.exceptions'] = MagicMock()
sys.modules['google_auth_oauthlib'] = MagicMock()
sys.modules['google_auth_oauthlib.flow'] = MagicMock()

from gmail_dataset_creator.dataset.builder import DatasetBuilder
from gmail_dataset_creator.models import EmailData, ClassificationResult, CATEGORIES


class TestDatasetBuilder(unittest.TestCase):
    """Test the DatasetBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_dataset")
        
        self.builder = DatasetBuilder(
            output_path=self.output_path,
            train_ratio=0.8,
            min_emails_per_category=5
        )
        
        # Create sample email data
        self.sample_emails = []
        self.sample_classifications = []
        
        categories = ["work", "personal", "promotional", "newsletter", "spam"]
        for i, category in enumerate(categories):
            for j in range(10):  # 10 emails per category
                email = EmailData(
                    id=f"{category}_{j:03d}",
                    subject=f"Test {category} email {j}",
                    body=f"This is a test {category} email body with content {j}",
                    sender=f"sender{j}@{category}.com",
                    recipient="user@example.com",
                    timestamp=datetime.now(),
                    raw_content=""
                )
                
                classification = ClassificationResult(
                    category=category,
                    confidence=0.8 + (j * 0.01),  # Varying confidence
                    reasoning=f"Classified as {category} based on content",
                    needs_review=False
                )
                
                self.sample_emails.append(email)
                self.sample_classifications.append(classification)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DatasetBuilder initialization."""
        self.assertEqual(self.builder.output_path, Path(self.output_path))
        self.assertEqual(self.builder.train_ratio, 0.8)
        self.assertEqual(self.builder.min_emails_per_category, 5)
        self.assertEqual(self.builder.total_emails, 0)
        
        # Check that directories were created
        self.assertTrue(Path(self.output_path).exists())
        self.assertTrue(Path(self.output_path, "train").exists())
        self.assertTrue(Path(self.output_path, "test").exists())
    
    def test_initialization_invalid_train_ratio(self):
        """Test initialization with invalid train ratio."""
        with self.assertRaises(ValueError):
            DatasetBuilder(self.output_path, train_ratio=1.5)
        
        with self.assertRaises(ValueError):
            DatasetBuilder(self.output_path, train_ratio=-0.1)
    
    def test_add_email(self):
        """Test adding emails to the dataset."""
        email = self.sample_emails[0]
        classification = self.sample_classifications[0]
        
        self.builder.add_email(email, classification)
        
        self.assertEqual(self.builder.total_emails, 1)
        self.assertIn(classification.category, self.builder.emails_by_category)
        self.assertEqual(len(self.builder.emails_by_category[classification.category]), 1)
        self.assertIsNotNone(self.builder.processing_start_time)
    
    def test_add_email_invalid_category(self):
        """Test adding email with invalid category."""
        email = self.sample_emails[0]
        invalid_classification = ClassificationResult(
            category="invalid_category",
            confidence=0.8,
            reasoning="Test",
            needs_review=False
        )
        
        with self.assertRaises(ValueError):
            self.builder.add_email(email, invalid_classification)
    
    def test_get_category_distribution(self):
        """Test getting category distribution."""
        # Add emails from different categories
        for i in range(5):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        distribution = self.builder.get_category_distribution()
        
        self.assertIsInstance(distribution, dict)
        self.assertEqual(sum(distribution.values()), 5)
        self.assertIn("work", distribution)
    
    def test_check_category_balance_sufficient(self):
        """Test category balance check with sufficient emails."""
        # Add enough emails per category
        for i in range(25):  # 5 emails per category (5 categories)
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        warnings = self.builder.check_category_balance()
        
        # Should have warnings for missing categories but not for insufficient emails
        self.assertTrue(any("has no emails" in warning for warning in warnings))
        self.assertFalse(any("has only" in warning for warning in warnings))
    
    def test_check_category_balance_insufficient(self):
        """Test category balance check with insufficient emails."""
        # Add only 2 emails (less than minimum of 5)
        for i in range(2):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        warnings = self.builder.check_category_balance()
        
        # Should have warnings for insufficient emails
        self.assertTrue(any("has only" in warning for warning in warnings))
    
    def test_check_category_balance_severe_imbalance(self):
        """Test category balance check with severe imbalance."""
        # Add many emails from one category and few from another
        for i in range(10):  # 10 work emails
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        # Add 1 personal email (will be less than 5% of total)
        personal_email = EmailData(
            id="personal_001",
            subject="Personal email",
            body="Personal content",
            sender="friend@example.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content=""
        )
        personal_classification = ClassificationResult(
            category="personal",
            confidence=0.9,
            reasoning="Personal email",
            needs_review=False
        )
        
        for i in range(100):  # Add many more work emails
            work_email = EmailData(
                id=f"work_extra_{i}",
                subject=f"Work email {i}",
                body=f"Work content {i}",
                sender=f"work{i}@company.com",
                recipient="user@example.com",
                timestamp=datetime.now(),
                raw_content=""
            )
            work_classification = ClassificationResult(
                category="work",
                confidence=0.8,
                reasoning="Work email",
                needs_review=False
            )
            self.builder.add_email(work_email, work_classification)
        
        self.builder.add_email(personal_email, personal_classification)
        
        warnings = self.builder.check_category_balance()
        
        # Should have warning about severe imbalance
        self.assertTrue(any("represents only" in warning for warning in warnings))
    
    def test_create_train_test_split(self):
        """Test creating train/test split."""
        # Add emails from multiple categories
        for i in range(20):  # 4 emails per category (5 categories)
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        train_emails, test_emails = self.builder.create_train_test_split()
        
        # Check split ratio (approximately 80/20)
        total_emails = len(train_emails) + len(test_emails)
        self.assertEqual(total_emails, 20)
        
        train_ratio = len(train_emails) / total_emails
        self.assertGreater(train_ratio, 0.7)  # Should be around 0.8
        self.assertLess(train_ratio, 0.9)
        
        # Check that we have both train and test emails
        self.assertGreater(len(train_emails), 0)
        self.assertGreater(len(test_emails), 0)
    
    def test_create_train_test_split_single_email_per_category(self):
        """Test train/test split with single email per category."""
        # Add one email per category
        for i in range(5):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        train_emails, test_emails = self.builder.create_train_test_split()
        
        # With single emails, some should go to test to ensure test set is not empty
        self.assertGreater(len(test_emails), 0)
    
    def test_email_to_dict(self):
        """Test converting email to dictionary format."""
        email = self.sample_emails[0]
        classification = self.sample_classifications[0]
        
        email_dict = self.builder._email_to_dict(email, classification)
        
        expected_keys = ["id", "subject", "body", "sender", "recipient", "category", "language", "timestamp"]
        for key in expected_keys:
            self.assertIn(key, email_dict)
        
        self.assertEqual(email_dict["id"], email.id)
        self.assertEqual(email_dict["subject"], email.subject)
        self.assertEqual(email_dict["category"], classification.category)
        self.assertEqual(email_dict["language"], "en")
    
    def test_build_vocabulary(self):
        """Test building vocabulary from email content."""
        # Add some emails
        for i in range(5):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        vocabulary = self.builder.build_vocabulary()
        
        self.assertIsInstance(vocabulary, dict)
        self.assertGreater(len(vocabulary), 0)
        
        # Check that common words from our test emails are in vocabulary
        self.assertIn("test", vocabulary)
        self.assertIn("email", vocabulary)
        
        # Check that vocabulary values are integers (token IDs)
        for token, token_id in vocabulary.items():
            self.assertIsInstance(token_id, int)
    
    def test_tokenize_text(self):
        """Test text tokenization."""
        text = "This is a Test Email with PUNCTUATION! And numbers 123."
        
        tokens = self.builder._tokenize_text(text)
        
        # Should be lowercase
        self.assertIn("test", tokens)
        self.assertNotIn("Test", tokens)
        
        # Should filter out stop words
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        
        # Should include meaningful words
        self.assertIn("email", tokens)
        self.assertIn("punctuation", tokens)
        self.assertIn("numbers", tokens)
        # Should include numbers (may have punctuation attached)
        self.assertTrue(any("123" in token for token in tokens))
    
    def test_tokenize_text_empty(self):
        """Test tokenizing empty text."""
        tokens = self.builder._tokenize_text("")
        self.assertEqual(tokens, [])
        
        tokens = self.builder._tokenize_text(None)
        self.assertEqual(tokens, [])
    
    def test_get_stats(self):
        """Test getting dataset statistics."""
        # Add some emails
        for i in range(10):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        stats = self.builder.get_stats()
        
        self.assertEqual(stats.total_emails, 10)
        self.assertEqual(stats.train_count, 8)  # 80% of 10
        self.assertEqual(stats.test_count, 2)   # 20% of 10
        self.assertIsInstance(stats.categories_distribution, dict)
        self.assertGreaterEqual(stats.processing_time, 0)
    
    def test_export_dataset_no_emails(self):
        """Test exporting dataset with no emails."""
        with self.assertRaises(ValueError):
            self.builder.export_dataset()
    
    def test_export_dataset_success(self):
        """Test successful dataset export."""
        # Add emails
        for i in range(10):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        with patch('warnings.warn') as mock_warn:
            stats = self.builder.export_dataset()
        
        # Check that files were created
        train_file = Path(self.output_path) / "train" / "dataset.json"
        test_file = Path(self.output_path) / "test" / "dataset.json"
        vocab_file = Path(self.output_path) / "vocab.json"
        categories_file = Path(self.output_path) / "categories.json"
        metadata_file = Path(self.output_path) / "metadata.json"
        
        self.assertTrue(train_file.exists())
        self.assertTrue(test_file.exists())
        self.assertTrue(vocab_file.exists())
        self.assertTrue(categories_file.exists())
        self.assertTrue(metadata_file.exists())
        
        # Check stats
        self.assertEqual(stats.total_emails, 10)
        self.assertGreater(stats.vocabulary_size, 0)
        self.assertGreaterEqual(stats.processing_time, 0)
        
        # Check that warnings were issued for imbalanced categories
        mock_warn.assert_called()
    
    def test_export_dataset_file_contents(self):
        """Test the contents of exported dataset files."""
        # Add a few emails
        for i in range(4):
            self.builder.add_email(self.sample_emails[i], self.sample_classifications[i])
        
        stats = self.builder.export_dataset()
        
        # Check train file contents
        train_file = Path(self.output_path) / "train" / "dataset.json"
        with open(train_file, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
        
        self.assertGreater(len(train_lines), 0)
        
        # Check that each line is valid JSON
        for line in train_lines:
            email_data = json.loads(line.strip())
            self.assertIn("id", email_data)
            self.assertIn("subject", email_data)
            self.assertIn("category", email_data)
        
        # Check vocabulary file
        vocab_file = Path(self.output_path) / "vocab.json"
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.assertIsInstance(vocab_data, dict)
        self.assertGreater(len(vocab_data), 0)
        
        # Check categories file
        categories_file = Path(self.output_path) / "categories.json"
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories_data = json.load(f)
        
        self.assertIn("categories", categories_data)
        self.assertIn("category_names", categories_data)
        self.assertEqual(categories_data["categories"], CATEGORIES)
        
        # Check metadata file
        metadata_file = Path(self.output_path) / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.assertIn("dataset_info", metadata)
        self.assertIn("category_distribution", metadata)
        self.assertIn("processing_info", metadata)
        self.assertEqual(metadata["dataset_info"]["total_emails"], 4)
    
    def test_calculate_category_percentages(self):
        """Test calculating category percentages."""
        # Add emails with known distribution
        categories = ["work", "personal", "promotional"]
        counts = [6, 3, 1]  # 60%, 30%, 10%
        
        for category, count in zip(categories, counts):
            for i in range(count):
                email = EmailData(
                    id=f"{category}_{i}",
                    subject=f"{category} email {i}",
                    body=f"{category} content {i}",
                    sender=f"sender@{category}.com",
                    recipient="user@example.com",
                    timestamp=datetime.now(),
                    raw_content=""
                )
                classification = ClassificationResult(
                    category=category,
                    confidence=0.8,
                    reasoning=f"{category} email",
                    needs_review=False
                )
                self.builder.add_email(email, classification)
        
        percentages = self.builder._calculate_category_percentages()
        
        self.assertAlmostEqual(percentages["work"], 60.0, places=1)
        self.assertAlmostEqual(percentages["personal"], 30.0, places=1)
        self.assertAlmostEqual(percentages["promotional"], 10.0, places=1)
    
    def test_calculate_category_percentages_no_emails(self):
        """Test calculating percentages with no emails."""
        percentages = self.builder._calculate_category_percentages()
        self.assertEqual(percentages, {})


if __name__ == '__main__':
    unittest.main()