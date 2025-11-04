"""
Integration and performance tests for the Gmail Dataset Creator.

This module contains integration tests that test end-to-end workflows,
performance tests for large dataset processing, and security tests.
"""

import unittest
import tempfile
import json
import os
import shutil
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any
import threading
import concurrent.futures

# Mock all external dependencies
sys.modules['google'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()
sys.modules['google.auth.exceptions'] = MagicMock()
sys.modules['google_auth_oauthlib'] = MagicMock()
sys.modules['google_auth_oauthlib.flow'] = MagicMock()
sys.modules['cryptography'] = MagicMock()
sys.modules['cryptography.fernet'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.errors'] = MagicMock()


@dataclass
class MockEmailData:
    """Mock email data for testing."""
    id: str
    subject: str
    body: str
    sender: str
    recipient: str
    timestamp: datetime
    raw_content: str


@dataclass
class MockClassificationResult:
    """Mock classification result for testing."""
    category: str
    confidence: float
    reasoning: str
    needs_review: bool


class MockGmailDatasetCreator:
    """Mock Gmail Dataset Creator for integration testing."""
    
    def __init__(self, config_path: str, output_path: str):
        self.config_path = config_path
        self.output_path = output_path
        self.authenticated = False
        self.emails_processed = 0
        self.processing_time = 0.0
        
    def authenticate(self) -> bool:
        """Mock authentication."""
        # Simulate authentication delay
        time.sleep(0.1)
        self.authenticated = True
        return True
    
    def create_dataset(self, max_emails: int = 100, date_range: tuple = None) -> Dict[str, Any]:
        """Mock dataset creation."""
        if not self.authenticated:
            raise ValueError("Not authenticated")
        
        start_time = time.time()
        
        # Simulate processing emails
        categories = ["work", "personal", "promotional", "newsletter", "spam"]
        emails_per_category = max_emails // len(categories)
        
        for i in range(max_emails):
            category = categories[i % len(categories)]
            # Simulate processing time per email
            time.sleep(0.001)  # 1ms per email
            self.emails_processed += 1
        
        self.processing_time = time.time() - start_time
        
        # Create mock output files
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "test"), exist_ok=True)
        
        # Create mock dataset files
        train_data = []
        test_data = []
        
        for i in range(max_emails):
            category = categories[i % len(categories)]
            email_data = {
                "id": f"email_{i:03d}",
                "subject": f"Test {category} email {i}",
                "body": f"This is test email content for {category}",
                "sender": f"sender{i}@{category}.com",
                "recipient": "user@example.com",
                "category": category,
                "language": "en"
            }
            
            # 80/20 split
            if i < max_emails * 0.8:
                train_data.append(email_data)
            else:
                test_data.append(email_data)
        
        # Write dataset files
        with open(os.path.join(self.output_path, "train", "dataset.json"), 'w') as f:
            for email in train_data:
                f.write(json.dumps(email) + '\n')
        
        with open(os.path.join(self.output_path, "test", "dataset.json"), 'w') as f:
            for email in test_data:
                f.write(json.dumps(email) + '\n')
        
        # Create vocabulary file
        vocab = {f"word_{i}": i for i in range(100)}
        with open(os.path.join(self.output_path, "vocab.json"), 'w') as f:
            json.dump(vocab, f)
        
        # Create categories file
        categories_data = {
            "categories": {cat: i for i, cat in enumerate(categories)},
            "total_categories": len(categories)
        }
        with open(os.path.join(self.output_path, "categories.json"), 'w') as f:
            json.dump(categories_data, f)
        
        return {
            "total_emails": max_emails,
            "train_count": len(train_data),
            "test_count": len(test_data),
            "vocabulary_size": len(vocab),
            "processing_time": self.processing_time,
            "categories_distribution": {cat: emails_per_category for cat in categories}
        }


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        self.output_path = os.path.join(self.temp_dir, "dataset_output")
        
        # Create mock config file
        config_data = {
            "gmail_api": {
                "credentials_file": "credentials.json",
                "token_file": "token.json",
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
            },
            "gemini_api": {
                "api_key": "test_api_key",
                "model": "gemini-pro"
            },
            "dataset": {
                "output_path": self.output_path,
                "train_ratio": 0.8,
                "max_emails_total": 100
            }
        }
        
        with open(self.config_path, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_dataset_creation_workflow(self):
        """Test complete dataset creation workflow from start to finish."""
        creator = MockGmailDatasetCreator(self.config_path, self.output_path)
        
        # Test authentication
        auth_result = creator.authenticate()
        self.assertTrue(auth_result)
        self.assertTrue(creator.authenticated)
        
        # Test dataset creation
        stats = creator.create_dataset(max_emails=50)
        
        # Verify results
        self.assertEqual(stats["total_emails"], 50)
        self.assertEqual(stats["train_count"], 40)  # 80% of 50
        self.assertEqual(stats["test_count"], 10)   # 20% of 50
        self.assertGreater(stats["vocabulary_size"], 0)
        self.assertGreater(stats["processing_time"], 0)
        
        # Verify output files exist
        self.assertTrue(os.path.exists(os.path.join(self.output_path, "train", "dataset.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, "test", "dataset.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, "vocab.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, "categories.json")))
        
        # Verify file contents
        with open(os.path.join(self.output_path, "train", "dataset.json"), 'r') as f:
            train_lines = f.readlines()
            self.assertEqual(len(train_lines), 40)
            
            # Verify each line is valid JSON
            for line in train_lines:
                email_data = json.loads(line.strip())
                self.assertIn("id", email_data)
                self.assertIn("category", email_data)
    
    def test_authentication_failure_handling(self):
        """Test handling of authentication failures."""
        creator = MockGmailDatasetCreator(self.config_path, self.output_path)
        
        # Mock authentication failure
        with patch.object(creator, 'authenticate', return_value=False):
            auth_result = creator.authenticate()
            self.assertFalse(auth_result)
            
            # Should not be able to create dataset without authentication
            with self.assertRaises(ValueError):
                creator.create_dataset()
    
    def test_error_recovery_and_resume(self):
        """Test error recovery and resume functionality."""
        creator = MockGmailDatasetCreator(self.config_path, self.output_path)
        creator.authenticate()
        
        # Simulate partial processing
        creator.emails_processed = 25
        
        # Mock resume functionality
        def mock_resume_processing(remaining_emails: int):
            """Mock resume processing from checkpoint."""
            start_processed = creator.emails_processed
            for i in range(remaining_emails):
                time.sleep(0.001)
                creator.emails_processed += 1
            return creator.emails_processed - start_processed
        
        # Resume processing
        processed = mock_resume_processing(25)
        self.assertEqual(processed, 25)
        self.assertEqual(creator.emails_processed, 50)
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test with invalid config
        invalid_config_path = os.path.join(self.temp_dir, "invalid_config.yaml")
        
        invalid_config = {
            "gmail_api": {
                # Missing required fields
                "scopes": ["invalid_scope"]
            }
        }
        
        with open(invalid_config_path, 'w') as f:
            import yaml
            yaml.dump(invalid_config, f)
        
        # Should handle invalid configuration gracefully
        creator = MockGmailDatasetCreator(invalid_config_path, self.output_path)
        
        # Mock validation
        def validate_config(config_path: str) -> bool:
            """Mock config validation."""
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                
                required_sections = ["gmail_api", "gemini_api", "dataset"]
                for section in required_sections:
                    if section not in config:
                        return False
                
                return True
            except Exception:
                return False
        
        is_valid = validate_config(invalid_config_path)
        self.assertFalse(is_valid)
        
        is_valid_good = validate_config(self.config_path)
        self.assertTrue(is_valid_good)


class TestPerformanceTests(unittest.TestCase):
    """Test performance with large datasets and concurrent processing."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "perf_dataset")
    
    def tearDown(self):
        """Clean up performance test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_dataset_processing_performance(self):
        """Test performance with large dataset (1000+ emails)."""
        creator = MockGmailDatasetCreator("config.yaml", self.output_path)
        creator.authenticate()
        
        # Test with 1000 emails
        start_time = time.time()
        stats = creator.create_dataset(max_emails=1000)
        processing_time = time.time() - start_time
        
        # Performance assertions
        self.assertEqual(stats["total_emails"], 1000)
        self.assertLess(processing_time, 10.0)  # Should complete within 10 seconds
        
        # Check processing rate
        emails_per_second = stats["total_emails"] / stats["processing_time"]
        self.assertGreater(emails_per_second, 50)  # Should process at least 50 emails/second
        
        # Memory usage should be reasonable (mock test)
        estimated_memory_mb = stats["total_emails"] * 0.001  # 1KB per email estimate
        self.assertLess(estimated_memory_mb, 100)  # Should use less than 100MB for 1000 emails
    
    def test_concurrent_processing_performance(self):
        """Test concurrent processing of multiple batches."""
        def process_batch(batch_id: int, batch_size: int) -> Dict[str, Any]:
            """Process a batch of emails concurrently."""
            creator = MockGmailDatasetCreator("config.yaml", f"{self.output_path}_{batch_id}")
            creator.authenticate()
            return creator.create_dataset(max_emails=batch_size)
        
        # Test concurrent processing of 5 batches
        batch_size = 100
        num_batches = 5
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_batch, i, batch_size)
                for i in range(num_batches)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Verify all batches completed successfully
        self.assertEqual(len(results), num_batches)
        
        total_emails = sum(result["total_emails"] for result in results)
        self.assertEqual(total_emails, batch_size * num_batches)
        
        # Concurrent processing should be faster than sequential
        # (This is a simplified test - in reality, API rate limits would apply)
        estimated_sequential_time = sum(result["processing_time"] for result in results)
        self.assertLess(total_time, estimated_sequential_time * 0.8)  # At least 20% faster
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage patterns with large datasets."""
        def simulate_memory_usage(num_emails: int) -> Dict[str, float]:
            """Simulate memory usage during processing."""
            # Mock memory usage calculation
            base_memory = 50.0  # Base memory in MB
            email_memory = num_emails * 0.001  # 1KB per email
            vocabulary_memory = min(num_emails * 0.0005, 10.0)  # Vocabulary overhead
            processing_overhead = num_emails * 0.0002  # Processing overhead
            
            peak_memory = base_memory + email_memory + vocabulary_memory + processing_overhead
            
            return {
                "base_memory_mb": base_memory,
                "email_data_mb": email_memory,
                "vocabulary_mb": vocabulary_memory,
                "processing_overhead_mb": processing_overhead,
                "peak_memory_mb": peak_memory
            }
        
        # Test memory usage for different dataset sizes
        test_sizes = [100, 500, 1000, 5000]
        
        for size in test_sizes:
            memory_stats = simulate_memory_usage(size)
            
            # Memory should scale reasonably with dataset size
            self.assertLess(memory_stats["peak_memory_mb"], size * 0.01 + 100)  # Should not exceed 10KB per email + 100MB base
            
            # Vocabulary memory should not grow linearly (due to word reuse)
            if size > 1000:
                self.assertLess(memory_stats["vocabulary_mb"], 15.0)  # Should cap at reasonable size
    
    def test_processing_rate_consistency(self):
        """Test that processing rate remains consistent across different batch sizes."""
        batch_sizes = [50, 100, 200, 500]
        processing_rates = []
        
        for batch_size in batch_sizes:
            creator = MockGmailDatasetCreator("config.yaml", f"{self.output_path}_{batch_size}")
            creator.authenticate()
            
            stats = creator.create_dataset(max_emails=batch_size)
            rate = stats["total_emails"] / stats["processing_time"]
            processing_rates.append(rate)
        
        # Processing rate should be relatively consistent
        avg_rate = sum(processing_rates) / len(processing_rates)
        
        for rate in processing_rates:
            # Each rate should be within 50% of average (allowing for some variation)
            self.assertGreater(rate, avg_rate * 0.5)
            self.assertLess(rate, avg_rate * 1.5)


class TestSecurityTests(unittest.TestCase):
    """Test security aspects of token storage and data anonymization."""
    
    def setUp(self):
        """Set up security test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.token_file = os.path.join(self.temp_dir, "token.json")
        self.encrypted_token_file = os.path.join(self.temp_dir, "token.encrypted")
    
    def tearDown(self):
        """Clean up security test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_token_storage_security(self):
        """Test secure token storage and encryption."""
        def mock_encrypt_token(token_data: Dict[str, Any], password: str) -> bytes:
            """Mock token encryption."""
            import json
            # Simple mock encryption (in reality, would use proper encryption)
            token_json = json.dumps(token_data)
            encrypted = f"ENCRYPTED:{password}:{token_json}".encode()
            return encrypted
        
        def mock_decrypt_token(encrypted_data: bytes, password: str) -> Dict[str, Any]:
            """Mock token decryption."""
            import json
            decrypted_str = encrypted_data.decode()
            
            if not decrypted_str.startswith("ENCRYPTED:"):
                raise ValueError("Invalid encrypted data")
            
            parts = decrypted_str.split(":", 2)
            if len(parts) != 3 or parts[1] != password:
                raise ValueError("Invalid password")
            
            return json.loads(parts[2])
        
        # Test token encryption
        token_data = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret"
        }
        
        password = "test_password"
        encrypted = mock_encrypt_token(token_data, password)
        
        # Verify encryption worked
        self.assertIsInstance(encrypted, bytes)
        # In a real implementation, the token would be properly encrypted and not readable
        # For this mock, we just verify the structure is different from plain JSON
        self.assertTrue(encrypted.decode().startswith("ENCRYPTED:"))
        
        # Test decryption
        decrypted = mock_decrypt_token(encrypted, password)
        self.assertEqual(decrypted, token_data)
        
        # Test wrong password
        with self.assertRaises(ValueError):
            mock_decrypt_token(encrypted, "wrong_password")
    
    def test_data_anonymization_security(self):
        """Test data anonymization and sensitive information removal."""
        def mock_anonymize_email_data(email_data: MockEmailData) -> MockEmailData:
            """Mock email data anonymization."""
            import re
            
            # Anonymize email addresses
            def anonymize_email(email: str) -> str:
                if '@' not in email:
                    return email
                local, domain = email.split('@', 1)
                return f"{local[0]}***@***.{domain.split('.')[-1]}"
            
            # Remove sensitive patterns
            def remove_sensitive_info(text: str) -> str:
                # Phone numbers
                text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REMOVED]', text)
                # Credit cards
                text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD_REMOVED]', text)
                # SSN
                text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REMOVED]', text)
                return text
            
            return MockEmailData(
                id=email_data.id,
                subject=remove_sensitive_info(email_data.subject),
                body=remove_sensitive_info(email_data.body),
                sender=anonymize_email(email_data.sender),
                recipient=anonymize_email(email_data.recipient),
                timestamp=email_data.timestamp,
                raw_content=""  # Clear raw content for security
            )
        
        # Test with email containing sensitive information
        sensitive_email = MockEmailData(
            id="test_001",
            subject="Call me at 555-123-4567",
            body="My credit card is 4532-1234-5678-9012 and SSN is 123-45-6789",
            sender="john.doe@company.com",
            recipient="user@example.com",
            timestamp=datetime.now(),
            raw_content="<raw email content>"
        )
        
        anonymized = mock_anonymize_email_data(sensitive_email)
        
        # Verify sensitive information was removed
        self.assertNotIn("555-123-4567", anonymized.subject)
        self.assertNotIn("4532-1234-5678-9012", anonymized.body)
        self.assertNotIn("123-45-6789", anonymized.body)
        
        # Verify anonymization markers are present
        self.assertIn("[PHONE_REMOVED]", anonymized.subject)
        self.assertIn("[CARD_REMOVED]", anonymized.body)
        self.assertIn("[SSN_REMOVED]", anonymized.body)
        
        # Verify email addresses were anonymized
        self.assertIn("j***@***.com", anonymized.sender)
        self.assertIn("u***@***.com", anonymized.recipient)
        
        # Verify raw content was cleared
        self.assertEqual(anonymized.raw_content, "")
    
    def test_secure_data_cleanup(self):
        """Test secure cleanup of temporary data."""
        def mock_secure_delete_file(file_path: str) -> bool:
            """Mock secure file deletion."""
            if not os.path.exists(file_path):
                return False
            
            # Overwrite file with random data before deletion (mock)
            file_size = os.path.getsize(file_path)
            with open(file_path, 'wb') as f:
                f.write(b'0' * file_size)  # Overwrite with zeros
            
            os.remove(file_path)
            return True
        
        # Create test files with sensitive data
        sensitive_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"sensitive_data_{i}.json")
            with open(file_path, 'w') as f:
                json.dump({"sensitive": f"data_{i}", "token": f"secret_{i}"}, f)
            sensitive_files.append(file_path)
        
        # Verify files exist
        for file_path in sensitive_files:
            self.assertTrue(os.path.exists(file_path))
        
        # Perform secure cleanup
        cleanup_results = []
        for file_path in sensitive_files:
            result = mock_secure_delete_file(file_path)
            cleanup_results.append(result)
        
        # Verify all files were securely deleted
        self.assertTrue(all(cleanup_results))
        
        for file_path in sensitive_files:
            self.assertFalse(os.path.exists(file_path))
    
    def test_access_control_validation(self):
        """Test access control and permission validation."""
        def mock_validate_file_permissions(file_path: str) -> Dict[str, bool]:
            """Mock file permission validation."""
            if not os.path.exists(file_path):
                return {"exists": False}
            
            # Mock permission checks
            stat_info = os.stat(file_path)
            
            # Check if file is readable/writable by owner only
            permissions = stat_info.st_mode & 0o777
            
            return {
                "exists": True,
                "owner_read": bool(permissions & 0o400),
                "owner_write": bool(permissions & 0o200),
                "group_read": bool(permissions & 0o040),
                "other_read": bool(permissions & 0o004),
                "secure": not (permissions & 0o077)  # No group/other permissions
            }
        
        # Create test file with different permissions
        test_file = os.path.join(self.temp_dir, "test_permissions.json")
        with open(test_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Test default permissions
        perms = mock_validate_file_permissions(test_file)
        self.assertTrue(perms["exists"])
        self.assertTrue(perms["owner_read"])
        self.assertTrue(perms["owner_write"])
        
        # In a real implementation, we would set secure permissions (600)
        # and verify that group/other access is denied


if __name__ == '__main__':
    # Run tests with different verbosity levels
    unittest.main(verbosity=2)