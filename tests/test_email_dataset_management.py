"""
Tests for email dataset management system.

This module tests the EmailDatasetManager and related classes for
email dataset loading, validation, and memory-efficient processing.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from macbook_optimization.email_dataset_management import (
    EmailDatasetConfig, EmailSample, EmailDataset, StreamingEmailDataset,
    EmailDatasetManager, EmailDatasetMetrics
)
from macbook_optimization.memory_management import MemoryManager
from models.email_tokenizer import EmailTokenizer


@pytest.fixture
def sample_emails():
    """Create sample email data for testing."""
    return [
        {
            "id": "email_001",
            "subject": "Weekly Newsletter - Tech Updates",
            "body": "Here are the latest tech updates from this week. Visit https://example.com for more info.",
            "sender": "newsletter@techblog.com",
            "recipient": "user@example.com",
            "category": "newsletter",
            "language": "en"
        },
        {
            "id": "email_002",
            "subject": "Meeting Tomorrow at 2:30 PM",
            "body": "Hi team, urgent reminder about our project meeting tomorrow at 2:30 PM in conference room A.",
            "sender": "manager@company.com",
            "recipient": "team@company.com",
            "category": "work",
            "language": "en"
        },
        {
            "id": "email_003",
            "subject": "Special Offer - 50% Off!",
            "body": "Limited time offer! Get 50% off all products. Click here to shop now!",
            "sender": "sales@store.com",
            "recipient": "customer@example.com",
            "category": "promotional",
            "language": "en"
        },
        {
            "id": "email_004",
            "subject": "Personal Note",
            "body": "Hey, how are you doing? Let's catch up soon!",
            "sender": "friend@gmail.com",
            "recipient": "user@example.com",
            "category": "personal",
            "language": "en"
        }
    ]


@pytest.fixture
def email_tokenizer():
    """Create email tokenizer for testing."""
    tokenizer = EmailTokenizer(vocab_size=1000, max_seq_len=256)
    # Build vocabulary with sample data
    sample_data = [
        {"subject": "test", "body": "test body", "sender": "test@example.com", "category": "test"}
    ]
    tokenizer.build_vocabulary(sample_data)
    return tokenizer


@pytest.fixture
def temp_email_dataset(sample_emails):
    """Create temporary email dataset files for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create train directory
    train_dir = os.path.join(temp_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Write email data to JSON file
    train_file = os.path.join(train_dir, "emails.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for email in sample_emails:
            f.write(json.dumps(email) + '\n')
    
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager for testing."""
    memory_manager = Mock(spec=MemoryManager)
    memory_manager.monitor_memory_usage.return_value = Mock(
        used_mb=2000,
        available_mb=4000,
        percent_used=50.0
    )
    return memory_manager


class TestEmailDatasetConfig:
    """Test EmailDatasetConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test creating EmailDatasetConfig with default values."""
        config = EmailDatasetConfig()
        
        assert config.email_json_path == "data/emails"
        assert len(config.categories) == 10
        assert "newsletter" in config.categories
        assert "work" in config.categories
        assert config.min_emails_per_category == 100
        assert config.validate_email_format is True
        assert config.enable_augmentation is True
    
    def test_custom_config_creation(self):
        """Test creating EmailDatasetConfig with custom values."""
        custom_categories = ["work", "personal", "spam"]
        config = EmailDatasetConfig(
            categories=custom_categories,
            min_emails_per_category=50,
            max_body_length=500,
            enable_augmentation=False
        )
        
        assert config.categories == custom_categories
        assert config.min_emails_per_category == 50
        assert config.max_body_length == 500
        assert config.enable_augmentation is False


class TestEmailSample:
    """Test EmailSample dataclass."""
    
    def test_email_sample_creation(self):
        """Test creating EmailSample."""
        sample = EmailSample(
            id="test_001",
            subject="Test Subject",
            body="Test body content",
            sender="test@example.com",
            recipient="user@example.com",
            category="test"
        )
        
        assert sample.id == "test_001"
        assert sample.subject == "Test Subject"
        assert sample.category == "test"
        assert sample.language == "en"  # Default
    
    def test_email_sample_to_dict(self):
        """Test converting EmailSample to dictionary."""
        sample = EmailSample(
            id="test_001",
            subject="Test Subject",
            body="Test body",
            sender="test@example.com",
            recipient="user@example.com",
            category="test"
        )
        
        sample_dict = sample.to_dict()
        
        assert isinstance(sample_dict, dict)
        assert sample_dict['id'] == "test_001"
        assert sample_dict['subject'] == "Test Subject"
        assert sample_dict['category'] == "test"
    
    def test_email_sample_from_dict(self):
        """Test creating EmailSample from dictionary."""
        data = {
            "id": "test_001",
            "subject": "Test Subject",
            "body": "Test body",
            "sender": "test@example.com",
            "recipient": "user@example.com",
            "category": "test",
            "language": "en"
        }
        
        sample = EmailSample.from_dict(data)
        
        assert sample.id == "test_001"
        assert sample.subject == "Test Subject"
        assert sample.category == "test"
        assert sample.language == "en"


class TestEmailDataset:
    """Test EmailDataset class."""
    
    def test_email_dataset_creation(self, sample_emails, email_tokenizer):
        """Test creating EmailDataset."""
        config = EmailDatasetConfig()
        email_samples = [EmailSample.from_dict(email) for email in sample_emails]
        
        dataset = EmailDataset(email_samples, email_tokenizer, config)
        
        assert len(dataset) == len(sample_emails)
        assert dataset.config == config
        assert dataset.tokenizer == email_tokenizer
        
        # Check category mapping
        assert len(dataset.category_to_id) == len(config.categories)
        assert "newsletter" in dataset.category_to_id
        assert "work" in dataset.category_to_id
    
    def test_email_dataset_getitem(self, sample_emails, email_tokenizer):
        """Test EmailDataset __getitem__ method."""
        config = EmailDatasetConfig()
        email_samples = [EmailSample.from_dict(email) for email in sample_emails]
        
        dataset = EmailDataset(email_samples, email_tokenizer, config)
        
        # Get first sample
        sample = dataset[0]
        
        # Check sample structure
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert 'email_id' in sample
        assert 'category' in sample
        assert 'metadata' in sample
        
        # Check data types and shapes
        assert len(sample['input_ids']) == email_tokenizer.max_seq_len
        assert len(sample['attention_mask']) == email_tokenizer.max_seq_len
        assert sample['email_id'] == "email_001"
        assert sample['category'] == "newsletter"
    
    def test_get_category_distribution(self, sample_emails, email_tokenizer):
        """Test category distribution calculation."""
        config = EmailDatasetConfig()
        email_samples = [EmailSample.from_dict(email) for email in sample_emails]
        
        dataset = EmailDataset(email_samples, email_tokenizer, config)
        distribution = dataset.get_category_distribution()
        
        # Check distribution
        assert isinstance(distribution, dict)
        assert distribution['newsletter'] == 1
        assert distribution['work'] == 1
        assert distribution['promotional'] == 1
        assert distribution['personal'] == 1
        
        # Check all categories are present
        for category in config.categories:
            assert category in distribution


class TestStreamingEmailDataset:
    """Test StreamingEmailDataset class."""
    
    def test_streaming_dataset_creation(self, temp_email_dataset, email_tokenizer, mock_memory_manager):
        """Test creating StreamingEmailDataset."""
        config = EmailDatasetConfig()
        email_files = [os.path.join(temp_email_dataset, "train", "emails.jsonl")]
        
        dataset = StreamingEmailDataset(
            email_files=email_files,
            tokenizer=email_tokenizer,
            config=config,
            memory_manager=mock_memory_manager
        )
        
        assert dataset.email_files == email_files
        assert dataset.tokenizer == email_tokenizer
        assert dataset.config == config
        assert len(dataset.file_info) == 1
    
    def test_streaming_dataset_iteration(self, temp_email_dataset, email_tokenizer, mock_memory_manager):
        """Test StreamingEmailDataset iteration."""
        config = EmailDatasetConfig()
        email_files = [os.path.join(temp_email_dataset, "train", "emails.jsonl")]
        
        dataset = StreamingEmailDataset(
            email_files=email_files,
            tokenizer=email_tokenizer,
            config=config,
            memory_manager=mock_memory_manager
        )
        
        # Iterate through dataset
        samples = list(dataset)
        
        # Should have samples
        assert len(samples) > 0
        
        # Check sample structure
        sample = samples[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert 'email_id' in sample
        assert 'category' in sample


class TestEmailDatasetManager:
    """Test EmailDatasetManager class."""
    
    def test_manager_initialization(self):
        """Test EmailDatasetManager initialization."""
        manager = EmailDatasetManager()
        
        assert manager.email_config is not None
        assert isinstance(manager.email_config, EmailDatasetConfig)
        assert manager.memory_manager is not None
        assert manager.tokenizer is None  # Not set until loading
        assert len(manager.email_metrics) == 0
    
    def test_manager_with_custom_config(self, mock_memory_manager):
        """Test EmailDatasetManager with custom configuration."""
        config = EmailDatasetConfig(
            email_streaming_threshold_mb=100.0,
            enable_augmentation=False
        )
        
        manager = EmailDatasetManager(config, mock_memory_manager)
        
        assert manager.email_config == config
        assert manager.memory_manager == mock_memory_manager
        assert manager.email_config.enable_augmentation is False
    
    def test_find_email_files(self, temp_email_dataset):
        """Test finding email files in dataset directory."""
        manager = EmailDatasetManager()
        
        # Test finding files in split directory
        email_files = manager._find_email_files(temp_email_dataset, "train")
        
        assert len(email_files) == 1
        assert email_files[0].endswith("emails.jsonl")
        assert os.path.exists(email_files[0])
    
    def test_analyze_email_dataset_requirements(self, temp_email_dataset):
        """Test analyzing email dataset requirements."""
        manager = EmailDatasetManager()
        email_files = manager._find_email_files(temp_email_dataset, "train")
        
        analysis = manager._analyze_email_dataset_requirements(email_files)
        
        # Check analysis results
        assert 'total_size_mb' in analysis
        assert 'total_emails' in analysis
        assert 'category_distribution' in analysis
        assert 'language_distribution' in analysis
        assert 'validation_errors' in analysis
        
        # Check values are reasonable
        assert analysis['total_size_mb'] > 0
        assert analysis['total_emails'] > 0
        assert isinstance(analysis['category_distribution'], dict)
    
    def test_validate_email_sample(self):
        """Test email sample validation."""
        manager = EmailDatasetManager()
        
        # Valid email
        valid_email = EmailSample(
            id="test_001",
            subject="Test Subject",
            body="This is a test email body with sufficient length.",
            sender="test@example.com",
            recipient="user@example.com",
            category="test"
        )
        
        assert manager._validate_email_sample(valid_email) is True
        
        # Invalid email - no subject
        invalid_email = EmailSample(
            id="test_002",
            subject="",
            body="Test body",
            sender="test@example.com",
            recipient="user@example.com",
            category="test"
        )
        
        assert manager._validate_email_sample(invalid_email) is False
        
        # Invalid email - body too short
        short_body_email = EmailSample(
            id="test_003",
            subject="Test",
            body="Short",
            sender="test@example.com",
            recipient="user@example.com",
            category="test"
        )
        
        assert manager._validate_email_sample(short_body_email) is False
    
    def test_load_email_dataset_memory_mode(self, temp_email_dataset, mock_memory_manager):
        """Test loading email dataset in memory mode."""
        config = EmailDatasetConfig(email_streaming_threshold_mb=1000.0)  # High threshold
        manager = EmailDatasetManager(config, mock_memory_manager)
        
        dataset = manager.load_email_dataset(temp_email_dataset, "train")
        
        # Should be regular EmailDataset (not streaming)
        assert isinstance(dataset, EmailDataset)
        assert len(dataset) > 0
        assert manager.tokenizer is not None
        assert len(manager.email_metrics) == 1
    
    def test_load_email_dataset_streaming_mode(self, temp_email_dataset, mock_memory_manager):
        """Test loading email dataset in streaming mode."""
        config = EmailDatasetConfig(email_streaming_threshold_mb=0.001)  # Very low threshold
        manager = EmailDatasetManager(config, mock_memory_manager)
        
        dataset = manager.load_email_dataset(temp_email_dataset, "train")
        
        # Should be StreamingEmailDataset
        assert isinstance(dataset, StreamingEmailDataset)
        assert manager.tokenizer is not None
        assert len(manager.email_metrics) == 1
    
    def test_create_email_dataloader(self, temp_email_dataset, mock_memory_manager):
        """Test creating email dataloader."""
        manager = EmailDatasetManager(memory_manager=mock_memory_manager)
        
        dataloader, creation_info = manager.create_email_dataloader(
            temp_email_dataset, batch_size=2, split="train"
        )
        
        # Check dataloader
        assert dataloader is not None
        assert hasattr(dataloader, '__iter__')
        
        # Check creation info
        assert 'original_batch_size' in creation_info
        assert 'optimized_batch_size' in creation_info
        assert 'dataset_type' in creation_info
        assert creation_info['original_batch_size'] == 2
        assert creation_info['optimized_batch_size'] <= 2  # May be reduced
    
    def test_validate_email_dataset(self, temp_email_dataset):
        """Test email dataset validation."""
        manager = EmailDatasetManager()
        
        validation_result = manager.validate_email_dataset(temp_email_dataset, "train")
        
        # Check validation result structure
        assert 'valid' in validation_result
        assert 'files_found' in validation_result
        assert 'total_emails' in validation_result
        assert 'category_distribution' in validation_result
        assert 'warnings' in validation_result
        assert 'recommendations' in validation_result
        
        # Should be valid
        assert validation_result['valid'] is True
        assert validation_result['files_found'] > 0
        assert validation_result['total_emails'] > 0
    
    def test_get_email_dataset_stats(self, temp_email_dataset, mock_memory_manager):
        """Test getting email dataset statistics."""
        manager = EmailDatasetManager(memory_manager=mock_memory_manager)
        
        # No datasets loaded yet
        stats = manager.get_email_dataset_stats()
        assert 'no_datasets_loaded' in stats
        
        # Load a dataset
        manager.load_email_dataset(temp_email_dataset, "train")
        
        # Get stats after loading
        stats = manager.get_email_dataset_stats()
        
        assert 'latest_dataset' in stats
        assert 'all_datasets_count' in stats
        assert 'total_emails_processed' in stats
        assert stats['all_datasets_count'] == 1
        assert stats['total_emails_processed'] > 0


class TestEmailDatasetUtilities:
    """Test utility functions for email dataset management."""
    
    def test_create_email_dataloader_utility(self, temp_email_dataset, mock_memory_manager):
        """Test create_email_dataloader utility function."""
        from macbook_optimization.email_dataset_management import create_email_dataloader
        
        dataloader, creation_info = create_email_dataloader(
            temp_email_dataset,
            batch_size=2,
            split="train",
            memory_manager=mock_memory_manager
        )
        
        assert dataloader is not None
        assert isinstance(creation_info, dict)
        assert 'optimized_batch_size' in creation_info
    
    def test_validate_email_dataset_format_utility(self, temp_email_dataset):
        """Test validate_email_dataset_format utility function."""
        from macbook_optimization.email_dataset_management import validate_email_dataset_format
        
        validation_result = validate_email_dataset_format(temp_email_dataset)
        
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        assert validation_result['valid'] is True


@pytest.mark.integration
class TestEmailDatasetManagerIntegration:
    """Integration tests for email dataset management."""
    
    def test_end_to_end_dataset_loading(self, temp_email_dataset):
        """Test complete end-to-end dataset loading workflow."""
        manager = EmailDatasetManager()
        
        # Validate dataset first
        validation = manager.validate_email_dataset(temp_email_dataset, "train")
        assert validation['valid'] is True
        
        # Load dataset
        dataset = manager.load_email_dataset(temp_email_dataset, "train")
        assert dataset is not None
        
        # Create dataloader
        dataloader, info = manager.create_email_dataloader(
            temp_email_dataset, batch_size=2, split="train"
        )
        assert dataloader is not None
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            if batch_count >= 2:  # Test a few batches
                break
        
        assert batch_count > 0
    
    def test_memory_constraint_handling(self):
        """Test handling of memory constraints in dataset loading."""
        # Create manager with very low memory thresholds
        config = EmailDatasetConfig(
            email_streaming_threshold_mb=0.001,
            email_cache_threshold_mb=0.0005
        )
        manager = EmailDatasetManager(config)
        
        # Should handle gracefully even with extreme constraints
        assert manager.email_config.email_streaming_threshold_mb == 0.001
        assert manager.email_config.email_cache_threshold_mb == 0.0005
    
    def test_large_dataset_simulation(self, temp_email_dataset):
        """Test behavior with simulated large dataset."""
        # Create config that forces streaming
        config = EmailDatasetConfig(email_streaming_threshold_mb=0.001)
        manager = EmailDatasetManager(config)
        
        # Load dataset (should use streaming)
        dataset = manager.load_email_dataset(temp_email_dataset, "train")
        
        # Should be streaming dataset
        assert isinstance(dataset, StreamingEmailDataset)
        
        # Should still be iterable
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 2:  # Just test a few samples
                break
        
        assert len(samples) > 0