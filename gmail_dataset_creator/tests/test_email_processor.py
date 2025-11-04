"""
Tests for the EmailProcessor class.

This module contains unit tests for email content extraction, cleaning,
and anonymization functionality.
"""

import unittest
from datetime import datetime
from unittest.mock import patch
import base64

from gmail_dataset_creator.processing.email_processor import EmailProcessor
from gmail_dataset_creator.models import EmailData


class TestEmailProcessor(unittest.TestCase):
    """Test cases for EmailProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = EmailProcessor(anonymize_content=True, remove_sensitive=True)
        self.processor_no_anon = EmailProcessor(anonymize_content=False, remove_sensitive=False)
    
    def test_extract_headers(self):
        """Test header extraction from payload."""
        payload = {
            'headers': [
                {'name': 'Subject', 'value': 'Test Subject'},
                {'name': 'From', 'value': 'sender@example.com'},
                {'name': 'To', 'value': 'recipient@example.com'},
                {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
            ]
        }
        
        headers = self.processor._extract_headers(payload)
        
        self.assertEqual(headers['Subject'], 'Test Subject')
        self.assertEqual(headers['From'], 'sender@example.com')
        self.assertEqual(headers['To'], 'recipient@example.com')
        self.assertEqual(headers['Date'], 'Mon, 1 Jan 2024 12:00:00 +0000')
    
    def test_extract_text_content(self):
        """Test extraction of plain text content."""
        # Create base64url encoded test content
        test_content = "This is a test email body."
        encoded_content = base64.urlsafe_b64encode(test_content.encode()).decode().rstrip('=')
        
        payload = {
            'mimeType': 'text/plain',
            'body': {
                'data': encoded_content
            }
        }
        
        result = self.processor_no_anon._extract_text_content(payload)
        self.assertEqual(result, test_content)
    
    def test_html_to_text_conversion(self):
        """Test HTML to text conversion."""
        html_content = """
        <html>
            <body>
                <h1>Test Email</h1>
                <p>This is a <strong>test</strong> email with <em>HTML</em> content.</p>
                <script>alert('malicious');</script>
                <style>body { color: red; }</style>
            </body>
        </html>
        """
        
        result = self.processor._html_to_text(html_content)
        
        # Should contain text content but not HTML tags or script/style content
        self.assertIn('Test Email', result)
        self.assertIn('test email with HTML content', result)
        self.assertNotIn('<h1>', result)
        self.assertNotIn('<script>', result)
        self.assertNotIn('alert', result)
        self.assertNotIn('color: red', result)
    
    def test_email_anonymization(self):
        """Test email address anonymization."""
        test_cases = [
            ('user@example.com', 'u***@***.com'),
            ('john.doe@company.org', 'j***@***.org'),
            ('a@test.co.uk', 'a@***.uk'),
        ]
        
        for original, expected_pattern in test_cases:
            result = self.processor._anonymize_email_address(original)
            # Check that it starts with first character and contains anonymization
            self.assertTrue(result.startswith(original[0]), f"Expected {result} to start with {original[0]}")
            self.assertIn('***', result)
            self.assertIn('@', result)
    
    def test_sensitive_pattern_removal(self):
        """Test removal of sensitive information patterns."""
        test_content = """
        Contact me at 555-123-4567 or (555) 987-6543.
        My credit card is 4532-1234-5678-9012.
        SSN: 123-45-6789
        IP: 192.168.1.1
        """
        
        result = self.processor._remove_sensitive_patterns(test_content)
        
        self.assertNotIn('555-123-4567', result)
        self.assertNotIn('4532-1234-5678-9012', result)
        self.assertNotIn('123-45-6789', result)
        self.assertNotIn('192.168.1.1', result)
        
        self.assertIn('[PHONE_REMOVED]', result)
        self.assertIn('[CREDIT_CARD_REMOVED]', result)
        self.assertIn('[SSN_REMOVED]', result)
        self.assertIn('[IP_REMOVED]', result)
    
    def test_content_validation_and_cleaning(self):
        """Test content validation and cleaning."""
        # Test with excessive whitespace
        messy_content = "  This   is    a   test   with   lots   of   whitespace  \n\n\n  "
        result = self.processor._validate_and_clean_content(messy_content)
        self.assertEqual(result, "This is a test with lots of whitespace")
        
        # Test with very long line (should be filtered out)
        long_line = "a" * 1500  # Very long line
        content_with_long_line = f"Normal line\n{long_line}\nAnother normal line"
        result = self.processor._validate_and_clean_content(content_with_long_line)
        self.assertNotIn(long_line, result)
        self.assertIn("Normal line", result)
        self.assertIn("Another normal line", result)
    
    def test_extract_content_integration(self):
        """Test full content extraction integration."""
        # Create a mock Gmail API message
        test_body = "This is a test email with user@example.com and phone 555-1234."
        encoded_body = base64.urlsafe_b64encode(test_body.encode()).decode().rstrip('=')
        
        raw_message = {
            'id': 'test_message_123',
            'internalDate': '1704067200000',  # Jan 1, 2024
            'payload': {
                'headers': [
                    {'name': 'Subject', 'value': 'Test Subject'},
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'}
                ],
                'mimeType': 'text/plain',
                'body': {
                    'data': encoded_body
                }
            }
        }
        
        result = self.processor.extract_content(raw_message)
        
        self.assertEqual(result.id, 'test_message_123')
        self.assertEqual(result.subject, 'Test Subject')
        self.assertIn('test email', result.body)
        self.assertNotIn('user@example.com', result.body)  # Should be anonymized
        self.assertNotIn('555-1234', result.body)  # Should be removed
        self.assertIn('***', result.sender)  # Should be anonymized
        self.assertIn('***', result.recipient)  # Should be anonymized
    
    def test_corrupted_email_handling(self):
        """Test handling of corrupted emails."""
        # Test with missing required fields
        corrupted_message = {
            'id': 'corrupted_123',
            'snippet': 'This is a snippet from corrupted email'
        }
        
        # This should not raise an exception but return a fallback EmailData
        try:
            result = self.processor.extract_content(corrupted_message)
            self.assertEqual(result.id, 'corrupted_123')
            self.assertEqual(result.subject, '[CORRUPTED_EMAIL]')
            self.assertIn('snippet from corrupted email', result.body)
        except ValueError:
            # If it raises ValueError, test the error handling
            error = Exception("Test error")
            result = self.processor.handle_corrupted_email(corrupted_message, error)
            self.assertIsNotNone(result)
            self.assertEqual(result.id, 'corrupted_123')
    
    def test_email_validation(self):
        """Test email data validation."""
        # Valid email
        valid_email = EmailData(
            id='test_123',
            subject='Test Subject',
            body='Test body',
            sender='sender@example.com',
            recipient='recipient@example.com',
            timestamp=datetime.now(),
            raw_content='raw content'
        )
        self.assertTrue(self.processor.validate_email(valid_email))
        
        # Test validation logic directly (EmailData constructor will raise error for empty ID)
        # So we test the validation method with a manually created object
        class MockEmailData:
            def __init__(self, id, subject, body, timestamp):
                self.id = id
                self.subject = subject
                self.body = body
                self.timestamp = timestamp
        
        # Invalid email - no ID
        invalid_email = MockEmailData('', 'Test Subject', 'Test body', datetime.now())
        self.assertFalse(self.processor.validate_email(invalid_email))
        
        # Invalid email - no subject or body
        invalid_email2 = MockEmailData('test_123', '', '', datetime.now())
        self.assertFalse(self.processor.validate_email(invalid_email2))
        
        # Invalid email - no timestamp
        invalid_email3 = MockEmailData('test_123', 'Subject', 'Body', None)
        self.assertFalse(self.processor.validate_email(invalid_email3))


if __name__ == '__main__':
    unittest.main()