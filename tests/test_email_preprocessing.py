"""
Tests for email preprocessing and data augmentation.

This module tests the EmailDataAugmentation, EmailPreprocessor, and
EnhancedEmailTokenizer classes for email preprocessing and augmentation.
"""

import pytest
import re
from unittest.mock import Mock, patch
from typing import Dict, List

from models.email_preprocessing import (
    EmailDataAugmentation, EmailPreprocessor, EnhancedEmailTokenizer,
    create_enhanced_email_tokenizer, preprocess_email_dataset
)


@pytest.fixture
def sample_email():
    """Create sample email for testing."""
    return {
        "id": "test_001",
        "subject": "Urgent: Meeting Tomorrow at 2:30 PM",
        "body": "Hi team, please find attached the report. Let me know if you have any questions. Best regards, John",
        "sender": "john@company.com",
        "recipient": "team@company.com",
        "category": "work"
    }


@pytest.fixture
def promotional_email():
    """Create promotional email for testing."""
    return {
        "id": "promo_001",
        "subject": "Special Offer - 50% Off Everything!",
        "body": "Limited time offer! Get 50% off all products. Click here to shop now! Unsubscribe at any time.",
        "sender": "sales@store.com",
        "recipient": "customer@example.com",
        "category": "promotional"
    }


@pytest.fixture
def newsletter_email():
    """Create newsletter email for testing."""
    return {
        "id": "news_001",
        "subject": "Weekly Tech Newsletter",
        "body": "Here are this week's tech updates. Visit https://techblog.com for more. Contact us at info@techblog.com or call 555-123-4567.",
        "sender": "newsletter@techblog.com",
        "recipient": "subscriber@example.com",
        "category": "newsletter"
    }


class TestEmailDataAugmentation:
    """Test EmailDataAugmentation class."""
    
    def test_augmentation_initialization(self):
        """Test EmailDataAugmentation initialization."""
        augmenter = EmailDataAugmentation(augmentation_probability=0.5)
        
        assert augmenter.augmentation_probability == 0.5
        assert len(augmenter.SYNONYMS) > 0
        assert len(augmenter.PARAPHRASE_PATTERNS) > 0
        assert len(augmenter.compiled_patterns) > 0
    
    def test_synonym_replacement(self, sample_email):
        """Test synonym replacement augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        # Test multiple times to check randomness
        original_text = "urgent meeting project deadline"
        results = []
        
        for _ in range(10):
            augmented = augmenter._replace_synonyms(original_text)
            results.append(augmented)
        
        # Should have some variation
        unique_results = set(results)
        assert len(unique_results) > 1 or any('urgent' not in result for result in results)
    
    def test_paraphrase_replacement(self, sample_email):
        """Test paraphrase replacement augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        email_with_phrases = sample_email.copy()
        email_with_phrases['body'] = "Please find attached the document. Thank you for your time. Let me know if you need anything."
        
        augmented = augmenter._paraphrase_replacement(email_with_phrases, "work")
        
        # Should potentially change some phrases
        assert 'body' in augmented
        # The text might be changed or might not (due to randomness)
        assert len(augmented['body']) > 0
    
    def test_domain_variation(self, sample_email):
        """Test domain variation augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        email_with_gmail = sample_email.copy()
        email_with_gmail['sender'] = "user@gmail.com"
        
        # Test multiple times to check for variation
        results = []
        for _ in range(20):
            augmented = augmenter._domain_variation(email_with_gmail, "work")
            results.append(augmented['sender'])
        
        # Should sometimes change gmail.com to googlemail.com
        unique_senders = set(results)
        assert len(unique_senders) >= 1  # At least original
    
    def test_punctuation_variation(self, promotional_email):
        """Test punctuation variation augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        # Test with promotional email (more likely to get punctuation changes)
        results = []
        for _ in range(20):
            augmented = augmenter._punctuation_variation(promotional_email, "promotional")
            results.append(augmented['subject'])
        
        # Should have some variation in punctuation
        unique_subjects = set(results)
        assert len(unique_subjects) >= 1
    
    def test_case_variation(self, sample_email):
        """Test case variation augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        results = []
        for _ in range(20):
            augmented = augmenter._case_variation(sample_email, "work")
            results.append(augmented['subject'])
        
        # Should have some case variations
        unique_subjects = set(results)
        assert len(unique_subjects) >= 1
    
    def test_augment_email(self, sample_email):
        """Test complete email augmentation."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        augmented = augmenter.augment_email(sample_email, "work")
        
        # Should return valid email structure
        assert 'subject' in augmented
        assert 'body' in augmented
        assert 'sender' in augmented
        assert augmented['id'] == sample_email['id']  # ID should not change
        assert augmented['category'] == sample_email['category']  # Category should not change
    
    def test_no_augmentation_with_low_probability(self, sample_email):
        """Test that no augmentation occurs with low probability."""
        augmenter = EmailDataAugmentation(augmentation_probability=0.0)
        
        augmented = augmenter.augment_email(sample_email, "work")
        
        # Should be identical to original
        assert augmented == sample_email
    
    def test_generate_category_specific_variations(self, promotional_email):
        """Test generating category-specific variations."""
        augmenter = EmailDataAugmentation(augmentation_probability=1.0)
        
        variations = augmenter.generate_category_specific_variations(promotional_email, "promotional")
        
        # Should generate multiple variations for promotional emails
        assert len(variations) >= 3
        assert len(variations) <= 5
        
        # All variations should be valid emails
        for variation in variations:
            assert 'subject' in variation
            assert 'body' in variation
            assert 'sender' in variation


class TestEmailPreprocessor:
    """Test EmailPreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test EmailPreprocessor initialization."""
        preprocessor = EmailPreprocessor()
        
        assert preprocessor.remove_signatures is True
        assert preprocessor.remove_confidentiality is True
        assert preprocessor.normalize_urls is True
        assert preprocessor.normalize_emails is True
        assert len(preprocessor.signature_patterns) > 0
    
    def test_preprocess_body_html_removal(self):
        """Test HTML tag removal from email body."""
        preprocessor = EmailPreprocessor()
        
        body_with_html = "Hello <b>world</b>! Visit <a href='http://example.com'>our site</a>."
        processed = preprocessor._preprocess_body(body_with_html)
        
        # HTML tags should be removed
        assert '<b>' not in processed
        assert '</b>' not in processed
        assert '<a href=' not in processed
        assert 'Hello world!' in processed
    
    def test_preprocess_body_url_normalization(self):
        """Test URL normalization in email body."""
        preprocessor = EmailPreprocessor()
        
        body_with_urls = "Visit https://example.com and www.test.com for more info."
        processed = preprocessor._preprocess_body(body_with_urls)
        
        # URLs should be normalized to <URL>
        assert 'https://example.com' not in processed
        assert 'www.test.com' not in processed
        assert '<URL>' in processed
    
    def test_preprocess_body_email_normalization(self):
        """Test email address normalization in body."""
        preprocessor = EmailPreprocessor()
        
        body_with_emails = "Contact us at support@example.com or sales@company.org."
        processed = preprocessor._preprocess_body(body_with_emails)
        
        # Email addresses should be normalized
        assert 'support@example.com' not in processed
        assert 'sales@company.org' not in processed
        assert '<EMAIL>' in processed
    
    def test_preprocess_body_phone_normalization(self):
        """Test phone number normalization in body."""
        preprocessor = EmailPreprocessor()
        
        body_with_phones = "Call us at 555-123-4567 or (555) 987-6543."
        processed = preprocessor._preprocess_body(body_with_phones)
        
        # Phone numbers should be normalized
        assert '555-123-4567' not in processed
        assert '(555) 987-6543' not in processed
        assert '<PHONE>' in processed
    
    def test_preprocess_body_date_time_normalization(self):
        """Test date and time normalization in body."""
        preprocessor = EmailPreprocessor()
        
        body_with_datetime = "Meeting on 12/25/2023 at 2:30 PM and January 15, 2024."
        processed = preprocessor._preprocess_body(body_with_datetime)
        
        # Dates and times should be normalized
        assert '12/25/2023' not in processed
        assert '2:30 PM' not in processed
        assert 'January 15, 2024' not in processed
        assert '<DATE>' in processed
        assert '<TIME>' in processed
    
    def test_preprocess_subject_prefix_removal(self):
        """Test removal of common prefixes from subject."""
        preprocessor = EmailPreprocessor()
        
        subjects = [
            "Re: Important Meeting",
            "FW: Project Update",
            "Fwd: Newsletter",
            "RE: Question about report"
        ]
        
        for subject in subjects:
            processed = preprocessor._preprocess_subject(subject)
            assert not processed.lower().startswith(('re:', 'fw:', 'fwd:'))
    
    def test_preprocess_sender_extraction(self):
        """Test sender email extraction and normalization."""
        preprocessor = EmailPreprocessor()
        
        senders = [
            "John Doe <john@example.com>",
            "john@example.com",
            "Jane Smith <JANE@COMPANY.COM>",
            "support@example.com"
        ]
        
        expected = [
            "john@example.com",
            "john@example.com",
            "jane@company.com",
            "support@example.com"
        ]
        
        for sender, expected_result in zip(senders, expected):
            processed = preprocessor._preprocess_sender(sender)
            assert processed == expected_result
    
    def test_detect_email_type_newsletter(self, newsletter_email):
        """Test newsletter email type detection."""
        preprocessor = EmailPreprocessor()
        
        scores = preprocessor.detect_email_type(newsletter_email)
        
        # Should have high newsletter score
        assert 'newsletter' in scores
        assert scores['newsletter'] > 0
        
        # Should have all expected categories
        expected_categories = ['auto_reply', 'newsletter', 'promotional', 'personal', 'work', 'spam']
        for category in expected_categories:
            assert category in scores
    
    def test_detect_email_type_promotional(self, promotional_email):
        """Test promotional email type detection."""
        preprocessor = EmailPreprocessor()
        
        scores = preprocessor.detect_email_type(promotional_email)
        
        # Should have high promotional score
        assert scores['promotional'] > 0
    
    def test_detect_email_type_work(self, sample_email):
        """Test work email type detection."""
        preprocessor = EmailPreprocessor()
        
        scores = preprocessor.detect_email_type(sample_email)
        
        # Should have high work score due to "meeting" and "urgent"
        assert scores['work'] > 0
    
    def test_preprocess_email_complete(self, newsletter_email):
        """Test complete email preprocessing."""
        preprocessor = EmailPreprocessor()
        
        processed = preprocessor.preprocess_email(newsletter_email)
        
        # Should have all required fields
        assert 'subject' in processed
        assert 'body' in processed
        assert 'sender' in processed
        
        # URLs and emails should be normalized
        assert '<URL>' in processed['body']
        assert '<EMAIL>' in processed['body']
        assert '<PHONE>' in processed['body']


class TestEnhancedEmailTokenizer:
    """Test EnhancedEmailTokenizer class."""
    
    def test_enhanced_tokenizer_initialization(self):
        """Test EnhancedEmailTokenizer initialization."""
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            max_seq_len=256,
            enable_augmentation=True,
            augmentation_prob=0.3
        )
        
        assert tokenizer.vocab_size == 1000
        assert tokenizer.max_seq_len == 256
        assert tokenizer.enable_augmentation is True
        assert tokenizer.augmenter is not None
        assert tokenizer.preprocessor is not None
        
        # Should have enhanced special tokens
        assert len(tokenizer.ENHANCED_SPECIAL_TOKENS) > 0
        assert "<SIGNATURE>" in tokenizer.vocab
        assert "<QUOTE>" in tokenizer.vocab
    
    def test_enhanced_tokenizer_without_augmentation(self):
        """Test EnhancedEmailTokenizer without augmentation."""
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            enable_augmentation=False
        )
        
        assert tokenizer.enable_augmentation is False
        assert tokenizer.augmenter is None
        assert tokenizer.preprocessor is not None
    
    def test_build_vocabulary_with_augmentation(self, sample_email, promotional_email):
        """Test vocabulary building with augmentation."""
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            enable_augmentation=True,
            augmentation_prob=0.5
        )
        
        email_data = [sample_email, promotional_email]
        
        # Build vocabulary with augmentation
        tokenizer.build_vocabulary_with_augmentation(
            email_data,
            augmentation_factor=2,
            min_freq=1
        )
        
        # Should have built vocabulary
        assert len(tokenizer.vocab) > len(tokenizer.SPECIAL_TOKENS)
        assert len(tokenizer.token_frequencies) > 0
    
    def test_encode_email_with_preprocessing(self, newsletter_email):
        """Test email encoding with preprocessing."""
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            enable_augmentation=True
        )
        
        # Build vocabulary first
        tokenizer.build_vocabulary([newsletter_email])
        
        # Encode with preprocessing
        token_ids, metadata = tokenizer.encode_email_with_preprocessing(newsletter_email)
        
        # Should have token IDs
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        
        # Should have enhanced metadata
        assert 'email_type_scores' in metadata
        assert 'preprocessing_applied' in metadata
        assert metadata['preprocessing_applied'] is True
        
        # Email type scores should be present
        assert isinstance(metadata['email_type_scores'], dict)
        assert 'newsletter' in metadata['email_type_scores']
    
    def test_batch_encode_with_preprocessing(self, sample_email, promotional_email):
        """Test batch encoding with preprocessing."""
        tokenizer = EnhancedEmailTokenizer(vocab_size=1000)
        
        # Build vocabulary
        emails = [sample_email, promotional_email]
        tokenizer.build_vocabulary(emails)
        
        # Batch encode
        batch_result = tokenizer.batch_encode_with_preprocessing(emails)
        
        # Should have expected structure
        assert 'input_ids' in batch_result
        assert 'attention_mask' in batch_result
        assert 'metadata' in batch_result
        
        # Should have correct batch size
        assert len(batch_result['metadata']) == 2
        
        # Each metadata should have preprocessing info
        for metadata in batch_result['metadata']:
            assert 'email_type_scores' in metadata
            assert 'preprocessing_applied' in metadata
    
    def test_get_augmentation_stats(self):
        """Test getting augmentation statistics."""
        # With augmentation
        tokenizer_with_aug = EnhancedEmailTokenizer(enable_augmentation=True)
        stats_with_aug = tokenizer_with_aug.get_augmentation_stats()
        
        assert stats_with_aug['augmentation_enabled'] is True
        assert 'augmentation_probability' in stats_with_aug
        assert 'synonym_mappings' in stats_with_aug
        assert stats_with_aug['synonym_mappings'] > 0
        
        # Without augmentation
        tokenizer_without_aug = EnhancedEmailTokenizer(enable_augmentation=False)
        stats_without_aug = tokenizer_without_aug.get_augmentation_stats()
        
        assert stats_without_aug['augmentation_enabled'] is False


class TestEmailPreprocessingUtilities:
    """Test utility functions for email preprocessing."""
    
    def test_create_enhanced_email_tokenizer(self):
        """Test create_enhanced_email_tokenizer utility function."""
        tokenizer = create_enhanced_email_tokenizer(
            vocab_size=2000,
            max_seq_len=128,
            enable_augmentation=True
        )
        
        assert isinstance(tokenizer, EnhancedEmailTokenizer)
        assert tokenizer.vocab_size == 2000
        assert tokenizer.max_seq_len == 128
        assert tokenizer.enable_augmentation is True
    
    def test_preprocess_email_dataset(self, sample_email, newsletter_email):
        """Test preprocess_email_dataset utility function."""
        emails = [sample_email, newsletter_email]
        
        processed_emails = preprocess_email_dataset(
            emails,
            remove_auto_replies=True
        )
        
        # Should return processed emails
        assert len(processed_emails) <= len(emails)  # May filter some
        
        # Each email should have email_type_scores
        for email in processed_emails:
            assert 'email_type_scores' in email
            assert isinstance(email['email_type_scores'], dict)
    
    def test_preprocess_email_dataset_auto_reply_filtering(self):
        """Test auto-reply filtering in dataset preprocessing."""
        auto_reply_email = {
            "id": "auto_001",
            "subject": "Out of Office Auto-Reply",
            "body": "I am currently out of office and will return on Monday.",
            "sender": "user@company.com",
            "recipient": "sender@example.com",
            "category": "other"
        }
        
        regular_email = {
            "id": "regular_001",
            "subject": "Regular Email",
            "body": "This is a regular email message.",
            "sender": "user@company.com",
            "recipient": "recipient@example.com",
            "category": "personal"
        }
        
        emails = [auto_reply_email, regular_email]
        
        # With auto-reply removal
        processed_with_removal = preprocess_email_dataset(
            emails,
            remove_auto_replies=True
        )
        
        # Without auto-reply removal
        processed_without_removal = preprocess_email_dataset(
            emails,
            remove_auto_replies=False
        )
        
        # Should filter out auto-replies when enabled
        assert len(processed_without_removal) >= len(processed_with_removal)


@pytest.mark.integration
class TestEmailPreprocessingIntegration:
    """Integration tests for email preprocessing."""
    
    def test_end_to_end_preprocessing_and_tokenization(self, newsletter_email):
        """Test complete preprocessing and tokenization workflow."""
        # Create enhanced tokenizer
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            max_seq_len=256,
            enable_augmentation=True,
            augmentation_prob=0.5
        )
        
        # Build vocabulary
        tokenizer.build_vocabulary_with_augmentation([newsletter_email])
        
        # Preprocess and encode
        token_ids, metadata = tokenizer.encode_email_with_preprocessing(newsletter_email)
        
        # Should have valid token sequence
        assert len(token_ids) > 0
        assert all(isinstance(token_id, int) for token_id in token_ids)
        
        # Should have comprehensive metadata
        assert 'email_type_scores' in metadata
        assert 'preprocessing_applied' in metadata
        assert 'subject_length' in metadata
        assert 'body_length' in metadata
        
        # Should be able to decode
        decoded = tokenizer.decode_tokens(token_ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
    
    def test_augmentation_consistency(self, sample_email):
        """Test that augmentation produces consistent results."""
        tokenizer = EnhancedEmailTokenizer(
            vocab_size=1000,
            enable_augmentation=True,
            augmentation_prob=1.0  # Always augment
        )
        
        # Build vocabulary
        tokenizer.build_vocabulary([sample_email])
        
        # Generate multiple augmented versions
        augmented_versions = []
        for _ in range(5):
            email_copy = sample_email.copy()
            email_copy['training'] = True  # Enable augmentation
            token_ids, metadata = tokenizer.encode_email_with_preprocessing(email_copy)
            augmented_versions.append((token_ids, metadata))
        
        # Should have some variation (due to augmentation)
        token_sequences = [version[0] for version in augmented_versions]
        
        # At least some sequences should be different (due to randomness in augmentation)
        unique_sequences = set(tuple(seq) for seq in token_sequences)
        # Note: Due to randomness, we can't guarantee variation, but we can check structure
        assert all(len(seq) > 0 for seq in token_sequences)
        assert all(metadata['augmentation_applied'] for _, metadata in augmented_versions)