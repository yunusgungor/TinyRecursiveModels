"""
Email Dataset Management for MacBook Training Optimization

This module provides email-specific dataset management capabilities,
extending the base DatasetManager with email JSON loading, validation,
and memory-efficient streaming for email classification training.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Union, Tuple, Callable
import numpy as np
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object
    IterableDataset = object
    TORCH_AVAILABLE = False

from .dataset_management import DatasetManager, DatasetManagementConfig, DatasetMetrics
from .memory_management import MemoryManager
from models.email_tokenizer import EmailTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EmailDatasetConfig:
    """Configuration for email dataset management."""
    
    # Email dataset parameters
    email_json_path: str = "data/emails"
    categories: List[str] = None  # Will default to 10 categories
    min_emails_per_category: int = 100
    max_emails_per_category: int = 10000
    
    # Email preprocessing
    max_subject_length: int = 100
    max_body_length: int = 1000
    min_body_length: int = 10
    
    # Validation parameters
    validate_email_format: bool = True
    validate_categories: bool = True
    remove_duplicates: bool = True
    
    # Memory management
    email_streaming_threshold_mb: float = 200.0
    email_cache_threshold_mb: float = 100.0
    chunk_size_emails: int = 1000  # Number of emails per chunk
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.3
    
    def __post_init__(self):
        """Set default categories if not provided."""
        if self.categories is None:
            self.categories = [
                "newsletter", "work", "personal", "spam", "promotional",
                "social", "finance", "travel", "shopping", "other"
            ]


@dataclass
class EmailSample:
    """Email sample data structure."""
    id: str
    subject: str
    body: str
    sender: str
    recipient: str
    category: str
    language: str = "en"
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'subject': self.subject,
            'body': self.body,
            'sender': self.sender,
            'recipient': self.recipient,
            'category': self.category,
            'language': self.language,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailSample':
        """Create from dictionary."""
        return cls(
            id=data.get('id', ''),
            subject=data.get('subject', ''),
            body=data.get('body', ''),
            sender=data.get('sender', ''),
            recipient=data.get('recipient', ''),
            category=data.get('category', 'other'),
            language=data.get('language', 'en'),
            timestamp=data.get('timestamp')
        )


@dataclass
class EmailDatasetMetrics:
    """Metrics for email dataset loading and processing."""
    total_emails: int
    category_distribution: Dict[str, int]
    avg_subject_length: float
    avg_body_length: float
    language_distribution: Dict[str, int]
    loading_time_seconds: float
    memory_usage_mb: float
    validation_errors: int
    duplicates_removed: int


class EmailDataset(Dataset):
    """Email dataset for training email classification models."""
    
    def __init__(self, emails: List[EmailSample], tokenizer: EmailTokenizer,
                 config: EmailDatasetConfig):
        """
        Initialize email dataset.
        
        Args:
            emails: List of email samples
            tokenizer: Email tokenizer
            config: Email dataset configuration
        """
        self.emails = emails
        self.tokenizer = tokenizer
        self.config = config
        
        # Create category mapping
        self.category_to_id = {cat: idx for idx, cat in enumerate(config.categories)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        
        # Validate categories
        self._validate_categories()
        
    def _validate_categories(self):
        """Validate that all emails have valid categories."""
        invalid_categories = set()
        for email in self.emails:
            if email.category not in self.category_to_id:
                invalid_categories.add(email.category)
        
        if invalid_categories:
            logger.warning(f"Found emails with invalid categories: {invalid_categories}")
            # Map invalid categories to 'other'
            for email in self.emails:
                if email.category not in self.category_to_id:
                    email.category = 'other'
    
    def __len__(self) -> int:
        return len(self.emails)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get email sample by index."""
        email = self.emails[idx]
        
        # Tokenize email
        token_ids, metadata = self.tokenizer.encode_email(email.to_dict())
        
        # Get category ID
        category_id = self.category_to_id.get(email.category, self.category_to_id['other'])
        
        # Pad sequence
        padded_ids = self.tokenizer.pad_sequence(token_ids)
        attention_mask = self.tokenizer.get_attention_mask(padded_ids)
        
        sample = {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long) if TORCH_AVAILABLE else np.array(padded_ids),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long) if TORCH_AVAILABLE else np.array(attention_mask),
            'labels': torch.tensor(category_id, dtype=torch.long) if TORCH_AVAILABLE else np.array(category_id),
            'email_id': email.id,
            'category': email.category,
            'metadata': metadata
        }
        
        return sample
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of categories in dataset."""
        distribution = {cat: 0 for cat in self.config.categories}
        for email in self.emails:
            if email.category in distribution:
                distribution[email.category] += 1
        return distribution


class StreamingEmailDataset(IterableDataset):
    """Memory-efficient streaming email dataset."""
    
    def __init__(self, email_files: List[str], tokenizer: EmailTokenizer,
                 config: EmailDatasetConfig, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize streaming email dataset.
        
        Args:
            email_files: List of email JSON files
            tokenizer: Email tokenizer
            config: Email dataset configuration
            memory_manager: Memory manager for monitoring
        """
        self.email_files = email_files
        self.tokenizer = tokenizer
        self.config = config
        self.memory_manager = memory_manager
        
        # Create category mapping
        self.category_to_id = {cat: idx for idx, cat in enumerate(config.categories)}
        
        # Analyze files
        self.file_info = self._analyze_email_files()
        
    def _analyze_email_files(self) -> List[Dict[str, Any]]:
        """Analyze email files to determine streaming strategy."""
        file_info = []
        
        for file_path in self.email_files:
            if not os.path.exists(file_path):
                logger.warning(f"Email file not found: {file_path}")
                continue
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Estimate number of emails (rough heuristic)
            estimated_emails = int(file_size_mb * 100)  # ~100 emails per MB
            
            info = {
                'file_path': file_path,
                'file_size_mb': file_size_mb,
                'estimated_emails': estimated_emails,
                'chunks_needed': max(1, int(file_size_mb / 10))  # 10MB chunks
            }
            file_info.append(info)
        
        return file_info
    
    def _load_email_chunk(self, file_path: str, start_line: int, chunk_size: int) -> List[EmailSample]:
        """Load a chunk of emails from file."""
        emails = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip to start line
                for _ in range(start_line):
                    f.readline()
                
                # Read chunk
                for _ in range(chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    
                    try:
                        email_data = json.loads(line.strip())
                        email = EmailSample.from_dict(email_data)
                        
                        # Validate email
                        if self._validate_email(email):
                            emails.append(email)
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {file_path} at line {start_line}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading email chunk from {file_path}: {e}")
        
        return emails
    
    def _validate_email(self, email: EmailSample) -> bool:
        """Validate email sample."""
        if not self.config.validate_email_format:
            return True
        
        # Check required fields
        if not email.subject or not email.body or not email.sender:
            return False
        
        # Check body length
        if len(email.body) < self.config.min_body_length:
            return False
        
        # Check category
        if self.config.validate_categories and email.category not in self.category_to_id:
            email.category = 'other'  # Map to default category
        
        return True
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over email dataset in chunks."""
        for file_info in self.file_info:
            file_path = file_info['file_path']
            estimated_emails = file_info['estimated_emails']
            
            # Process file in chunks
            chunk_size = self.config.chunk_size_emails
            start_line = 0
            
            while start_line < estimated_emails:
                # Monitor memory usage
                if self.memory_manager:
                    memory_stats = self.memory_manager.monitor_memory_usage()
                    if memory_stats.percent_used > 85:
                        logger.info(f"High memory usage ({memory_stats.percent_used:.1f}%) - forcing cleanup")
                        self.memory_manager.force_garbage_collection()
                
                # Load chunk
                emails = self._load_email_chunk(file_path, start_line, chunk_size)
                
                if not emails:
                    break
                
                # Process emails in chunk
                for email in emails:
                    # Tokenize email
                    token_ids, metadata = self.tokenizer.encode_email(email.to_dict())
                    
                    # Get category ID
                    category_id = self.category_to_id.get(email.category, self.category_to_id['other'])
                    
                    # Pad sequence
                    padded_ids = self.tokenizer.pad_sequence(token_ids)
                    attention_mask = self.tokenizer.get_attention_mask(padded_ids)
                    
                    sample = {
                        'input_ids': torch.tensor(padded_ids, dtype=torch.long) if TORCH_AVAILABLE else np.array(padded_ids),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long) if TORCH_AVAILABLE else np.array(attention_mask),
                        'labels': torch.tensor(category_id, dtype=torch.long) if TORCH_AVAILABLE else np.array(category_id),
                        'email_id': email.id,
                        'category': email.category,
                        'metadata': metadata
                    }
                    
                    yield sample
                
                start_line += chunk_size


class EmailDatasetManager(DatasetManager):
    """Email-specific dataset manager extending base DatasetManager."""
    
    def __init__(self, config: Optional[EmailDatasetConfig] = None,
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize email dataset manager.
        
        Args:
            config: Email dataset configuration
            memory_manager: Memory manager for monitoring
        """
        # Initialize base dataset manager
        base_config = DatasetManagementConfig(
            max_dataset_memory_mb=config.email_streaming_threshold_mb if config else 200.0,
            streaming_threshold_mb=config.email_streaming_threshold_mb if config else 200.0,
            cache_threshold_mb=config.email_cache_threshold_mb if config else 100.0
        )
        super().__init__(base_config, memory_manager)
        
        self.email_config = config or EmailDatasetConfig()
        self.tokenizer = None
        
        # Email-specific metrics
        self.email_metrics: List[EmailDatasetMetrics] = []
    
    def load_email_dataset(self, dataset_path: str, split: str = "train",
                          tokenizer: Optional[EmailTokenizer] = None) -> Union[EmailDataset, StreamingEmailDataset]:
        """
        Load email dataset with automatic strategy selection.
        
        Args:
            dataset_path: Path to email dataset directory
            split: Dataset split (train/val/test)
            tokenizer: Email tokenizer (created if None)
            
        Returns:
            Email dataset instance
        """
        start_time = time.time()
        
        # Create tokenizer if not provided
        if tokenizer is None:
            tokenizer = EmailTokenizer(vocab_size=5000, max_seq_len=512)
            logger.info("Created default email tokenizer")
        
        self.tokenizer = tokenizer
        
        # Find email files
        email_files = self._find_email_files(dataset_path, split)
        
        if not email_files:
            raise ValueError(f"No email files found in {dataset_path}/{split}")
        
        # Analyze dataset requirements
        analysis = self._analyze_email_dataset_requirements(email_files)
        
        # Determine loading strategy
        if analysis['total_size_mb'] > self.email_config.email_streaming_threshold_mb:
            logger.info(f"Using streaming dataset for {analysis['total_size_mb']:.1f}MB email data")
            dataset = StreamingEmailDataset(
                email_files=email_files,
                tokenizer=tokenizer,
                config=self.email_config,
                memory_manager=self.memory_manager
            )
            strategy = "streaming"
            memory_usage_mb = 50.0  # Estimated streaming overhead
        else:
            logger.info(f"Loading full email dataset ({analysis['total_size_mb']:.1f}MB)")
            emails = self._load_all_emails(email_files)
            dataset = EmailDataset(
                emails=emails,
                tokenizer=tokenizer,
                config=self.email_config
            )
            strategy = "memory"
            memory_usage_mb = analysis['total_size_mb'] * 1.2  # 20% overhead
        
        # Validate dataset is not empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty! No valid emails found in {dataset_path}/{split}")
        
        # Record metrics
        load_time = time.time() - start_time
        metrics = EmailDatasetMetrics(
            total_emails=analysis['total_emails'],
            category_distribution=analysis['category_distribution'],
            avg_subject_length=analysis['avg_subject_length'],
            avg_body_length=analysis['avg_body_length'],
            language_distribution=analysis['language_distribution'],
            loading_time_seconds=load_time,
            memory_usage_mb=memory_usage_mb,
            validation_errors=analysis['validation_errors'],
            duplicates_removed=analysis['duplicates_removed']
        )
        self.email_metrics.append(metrics)
        
        logger.info(f"Loaded email dataset: {metrics.total_emails} emails, "
                   f"strategy: {strategy}, time: {load_time:.2f}s")
        
        return dataset
    
    def _find_email_files(self, dataset_path: str, split: str) -> List[str]:
        """Find email JSON files in dataset directory."""
        email_files = []
        
        # Look for split-specific files
        split_dir = os.path.join(dataset_path, split)
        if os.path.exists(split_dir):
            for file_name in os.listdir(split_dir):
                if file_name.endswith('.json') or file_name.endswith('.jsonl'):
                    email_files.append(os.path.join(split_dir, file_name))
        
        # Look for general files with split prefix
        if os.path.exists(dataset_path):
            for file_name in os.listdir(dataset_path):
                if file_name.startswith(f"{split}_") and (file_name.endswith('.json') or file_name.endswith('.jsonl')):
                    email_files.append(os.path.join(dataset_path, file_name))
        
        return sorted(email_files)
    
    def _analyze_email_dataset_requirements(self, email_files: List[str]) -> Dict[str, Any]:
        """Analyze email dataset requirements."""
        total_size_mb = 0.0
        total_emails = 0
        category_distribution = {cat: 0 for cat in self.email_config.categories}
        language_distribution = {}
        subject_lengths = []
        body_lengths = []
        validation_errors = 0
        duplicates_removed = 0
        seen_ids = set()
        
        for file_path in email_files:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_mb += file_size_mb
            
            # Sample emails for analysis (first 1000 or 10% of file)
            sample_size = min(1000, max(100, int(file_size_mb * 10)))
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= sample_size:
                            break
                        
                        try:
                            email_data = json.loads(line.strip())
                            email = EmailSample.from_dict(email_data)
                            
                            # Check for duplicates
                            if email.id in seen_ids:
                                duplicates_removed += 1
                                continue
                            seen_ids.add(email.id)
                            
                            # Validate
                            if not self._validate_email_sample(email):
                                validation_errors += 1
                                continue
                            
                            total_emails += 1
                            
                            # Update distributions
                            if email.category in category_distribution:
                                category_distribution[email.category] += 1
                            else:
                                category_distribution['other'] += 1
                            
                            language_distribution[email.language] = language_distribution.get(email.language, 0) + 1
                            
                            # Update lengths
                            subject_lengths.append(len(email.subject))
                            body_lengths.append(len(email.body))
                        
                        except json.JSONDecodeError:
                            validation_errors += 1
                            continue
            
            except Exception as e:
                logger.error(f"Error analyzing email file {file_path}: {e}")
        
        return {
            'total_size_mb': total_size_mb,
            'total_emails': total_emails,
            'category_distribution': category_distribution,
            'language_distribution': language_distribution,
            'avg_subject_length': np.mean(subject_lengths) if subject_lengths else 0,
            'avg_body_length': np.mean(body_lengths) if body_lengths else 0,
            'validation_errors': validation_errors,
            'duplicates_removed': duplicates_removed
        }
    
    def _validate_email_sample(self, email: EmailSample) -> bool:
        """Validate email sample."""
        # Check required fields
        if not email.subject or not email.body or not email.sender:
            return False
        
        # Check lengths
        if len(email.body) < self.email_config.min_body_length:
            return False
        
        if len(email.subject) > self.email_config.max_subject_length:
            return False
        
        if len(email.body) > self.email_config.max_body_length:
            return False
        
        return True
    
    def _load_all_emails(self, email_files: List[str]) -> List[EmailSample]:
        """Load all emails from files."""
        emails = []
        seen_ids = set()
        
        for file_path in email_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            email_data = json.loads(line.strip())
                            email = EmailSample.from_dict(email_data)
                            
                            # Remove duplicates
                            if self.email_config.remove_duplicates:
                                if email.id in seen_ids:
                                    continue
                                seen_ids.add(email.id)
                            
                            # Validate
                            if self._validate_email_sample(email):
                                emails.append(email)
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {file_path} at line {line_num}")
                            continue
            
            except Exception as e:
                logger.error(f"Error loading emails from {file_path}: {e}")
        
        return emails
    
    def create_email_dataloader(self, dataset_path: str, batch_size: int,
                              split: str = "train", tokenizer: Optional[EmailTokenizer] = None,
                              **kwargs) -> Tuple[DataLoader, Dict[str, Any]]:
        """
        Create email dataloader with MacBook optimizations.
        
        Args:
            dataset_path: Path to email dataset
            batch_size: Batch size
            split: Dataset split
            tokenizer: Email tokenizer
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Tuple of (dataloader, creation_info)
        """
        # Load dataset
        dataset = self.load_email_dataset(dataset_path, split, tokenizer)
        
        # Optimize batch size for email data
        memory_stats = self.memory_manager.monitor_memory_usage()
        available_memory_mb = memory_stats.available_mb
        
        # Estimate memory per email sample
        estimated_memory_per_sample = 2.0  # MB (conservative estimate)
        max_safe_batch_size = max(1, int(available_memory_mb * 0.3 / estimated_memory_per_sample))
        
        optimized_batch_size = min(batch_size, max_safe_batch_size, 16)  # Cap at 16 for emails
        
        # DataLoader configuration
        dataloader_config = {
            'batch_size': optimized_batch_size,
            'shuffle': split == "train",
            'num_workers': 0,  # Disable multiprocessing to avoid collate issues
            'pin_memory': False,  # Better for MacBook
            'drop_last': split == "train"
        }
        dataloader_config.update(kwargs)
        
        # Create dataloader with custom collate function
        if isinstance(dataset, StreamingEmailDataset):
            # For streaming datasets, don't shuffle (handled internally)
            dataloader_config['shuffle'] = False
        
        # Add custom collate function to handle metadata properly
        dataloader_config['collate_fn'] = self._email_collate_fn
        
        dataloader = DataLoader(dataset, **dataloader_config)
        
        creation_info = {
            'original_batch_size': batch_size,
            'optimized_batch_size': optimized_batch_size,
            'dataset_type': type(dataset).__name__,
            'total_emails': len(dataset) if hasattr(dataset, '__len__') else 'streaming',
            'memory_usage_mb': estimated_memory_per_sample * optimized_batch_size,
            'dataloader_config': dataloader_config
        }
        
        return dataloader, creation_info
    
    def _email_collate_fn(self, batch):
        """Custom collate function for email batches to handle metadata properly."""
        try:
            # Separate the different fields
            input_ids = []
            attention_masks = []
            labels = []
            email_ids = []
            categories = []
            metadatas = []
            
            for sample in batch:
                input_ids.append(sample['input_ids'])
                attention_masks.append(sample['attention_mask'])
                labels.append(sample['labels'])
                email_ids.append(sample['email_id'])
                categories.append(sample['category'])
                metadatas.append(sample['metadata'])
            
            # Stack tensors
            if TORCH_AVAILABLE:
                import torch
                batched = {
                    'input_ids': torch.stack(input_ids),
                    'attention_mask': torch.stack(attention_masks),
                    'labels': torch.stack(labels),
                    'email_id': email_ids,  # Keep as list
                    'category': categories,  # Keep as list
                    'metadata': metadatas  # Keep as list of dicts
                }
            else:
                import numpy as np
                batched = {
                    'input_ids': np.stack(input_ids),
                    'attention_mask': np.stack(attention_masks),
                    'labels': np.stack(labels),
                    'email_id': email_ids,
                    'category': categories,
                    'metadata': metadatas
                }
            
            return batched
            
        except Exception as e:
            # Fallback: if there's still an issue, provide more detailed error info
            logger.error(f"Error in email collate function: {e}")
            logger.error(f"Batch size: {len(batch)}")
            if batch:
                sample = batch[0]
                logger.error(f"Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        logger.error(f"  {key} shape: {value.shape}")
                    elif hasattr(value, '__len__'):
                        logger.error(f"  {key} length: {len(value)}")
                    else:
                        logger.error(f"  {key} type: {type(value)}")
            raise
    
    def validate_email_dataset(self, dataset_path: str, split: str = "train") -> Dict[str, Any]:
        """
        Validate email dataset format and content.
        
        Args:
            dataset_path: Path to email dataset
            split: Dataset split
            
        Returns:
            Validation results
        """
        email_files = self._find_email_files(dataset_path, split)
        
        if not email_files:
            return {
                'valid': False,
                'error': f"No email files found in {dataset_path}/{split}",
                'files_found': 0
            }
        
        analysis = self._analyze_email_dataset_requirements(email_files)
        
        # Check category balance
        category_dist = analysis['category_distribution']
        min_samples = min(category_dist.values()) if category_dist.values() else 0
        max_samples = max(category_dist.values()) if category_dist.values() else 0
        balance_ratio = min_samples / max_samples if max_samples > 0 else 0
        
        # Validation checks
        validation_results = {
            'valid': True,
            'files_found': len(email_files),
            'total_emails': analysis['total_emails'],
            'total_size_mb': analysis['total_size_mb'],
            'category_distribution': category_dist,
            'balance_ratio': balance_ratio,
            'validation_errors': analysis['validation_errors'],
            'duplicates_removed': analysis['duplicates_removed'],
            'warnings': [],
            'recommendations': []
        }
        
        # Generate warnings and recommendations
        if analysis['validation_errors'] > analysis['total_emails'] * 0.1:
            validation_results['warnings'].append(f"High validation error rate: {analysis['validation_errors']} errors")
        
        if balance_ratio < 0.1:
            validation_results['warnings'].append(f"Severe category imbalance: ratio {balance_ratio:.3f}")
            validation_results['recommendations'].append("Consider category balancing or data augmentation")
        
        if analysis['total_size_mb'] > self.email_config.email_streaming_threshold_mb:
            validation_results['recommendations'].append("Large dataset - streaming mode will be used")
        
        if min_samples < self.email_config.min_emails_per_category:
            validation_results['warnings'].append(f"Some categories have < {self.email_config.min_emails_per_category} samples")
        
        return validation_results
    
    def get_email_dataset_stats(self) -> Dict[str, Any]:
        """Get email dataset loading statistics."""
        if not self.email_metrics:
            return {'no_datasets_loaded': True}
        
        latest_metrics = self.email_metrics[-1]
        
        return {
            'latest_dataset': {
                'total_emails': latest_metrics.total_emails,
                'category_distribution': latest_metrics.category_distribution,
                'avg_subject_length': latest_metrics.avg_subject_length,
                'avg_body_length': latest_metrics.avg_body_length,
                'language_distribution': latest_metrics.language_distribution,
                'loading_time_seconds': latest_metrics.loading_time_seconds,
                'memory_usage_mb': latest_metrics.memory_usage_mb
            },
            'all_datasets_count': len(self.email_metrics),
            'total_emails_processed': sum(m.total_emails for m in self.email_metrics),
            'avg_loading_time': np.mean([m.loading_time_seconds for m in self.email_metrics])
        }


# Utility functions for email dataset management
def create_email_dataloader(dataset_path: str, batch_size: int, split: str = "train",
                          tokenizer: Optional[EmailTokenizer] = None,
                          memory_manager: Optional[MemoryManager] = None,
                          **kwargs) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Convenience function to create email dataloader.
    
    Args:
        dataset_path: Path to email dataset
        batch_size: Batch size
        split: Dataset split
        tokenizer: Email tokenizer
        memory_manager: Memory manager
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (dataloader, creation_info)
    """
    email_manager = EmailDatasetManager(memory_manager=memory_manager)
    return email_manager.create_email_dataloader(
        dataset_path, batch_size, split, tokenizer, **kwargs
    )


def validate_email_dataset_format(dataset_path: str) -> Dict[str, Any]:
    """
    Validate email dataset format.
    
    Args:
        dataset_path: Path to email dataset
        
    Returns:
        Validation results
    """
    email_manager = EmailDatasetManager()
    return email_manager.validate_email_dataset(dataset_path)