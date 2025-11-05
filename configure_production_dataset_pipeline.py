#!/usr/bin/env python3
"""
Production Dataset Loading Pipeline Configuration

This script configures the EmailDatasetManager to work with production email datasets,
supporting both gmail_dataset_creator format and custom formats, with multilingual
support for English and Turkish emails.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Import the existing EmailDatasetManager
from macbook_optimization.email_dataset_management import (
    EmailDatasetManager, 
    EmailDatasetConfig, 
    EmailSample,
    EmailDataset,
    StreamingEmailDataset
)
from macbook_optimization.memory_management import MemoryManager
from models.email_tokenizer import EmailTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProductionDatasetConfig:
    """Configuration for production dataset loading pipeline."""
    
    # Dataset paths
    production_dataset_path: str = "data/production-emails"
    gmail_dataset_path: Optional[str] = None
    custom_dataset_path: Optional[str] = None
    
    # Language support
    supported_languages: List[str] = None
    default_language: str = "en"
    
    # Email processing
    max_subject_length: int = 200
    max_body_length: int = 10000
    min_body_length: int = 10
    
    # Memory management
    streaming_threshold_mb: float = 500.0  # Use streaming for datasets > 500MB
    cache_threshold_mb: float = 200.0
    batch_size: int = 32
    
    # Validation
    validate_categories: bool = True
    validate_content_quality: bool = True
    remove_duplicates: bool = True
    
    def __post_init__(self):
        """Set default supported languages if not provided."""
        if self.supported_languages is None:
            self.supported_languages = ["en", "tr"]  # English and Turkish

class ProductionEmailDatasetManager(EmailDatasetManager):
    """Extended EmailDatasetManager that can handle individual JSON files."""
    
    def _load_all_emails(self, email_files: List[str]) -> List[EmailSample]:
        """Load all emails from files, supporting both individual JSON files and JSONL."""
        emails = []
        seen_ids = set()
        
        for file_path in email_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # First, try to parse as a single JSON object
                try:
                    email_data = json.loads(content)
                    # If successful, it's a single JSON object
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
                    # If single JSON parsing fails, try JSONL format
                    try:
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if not line.strip():
                                continue
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
                        logger.error(f"Failed to parse {file_path} as JSONL: {e}")
                        
            except Exception as e:
                logger.error(f"Error loading emails from {file_path}: {e}")
        
        return emails

class ProductionDatasetPipeline:
    """Production dataset loading pipeline with multi-format support."""
    
    def __init__(self, config: Optional[ProductionDatasetConfig] = None):
        """Initialize the production dataset pipeline."""
        self.config = config or ProductionDatasetConfig()
        self.memory_manager = MemoryManager()
        
        # Create EmailDatasetConfig from ProductionDatasetConfig
        self.email_config = EmailDatasetConfig(
            email_json_path=self.config.production_dataset_path,
            categories=[
                "newsletter", "work", "personal", "spam", "promotional",
                "social", "finance", "travel", "shopping", "other"
            ],
            min_emails_per_category=100,  # Minimum for validation
            max_emails_per_category=10000,
            max_subject_length=self.config.max_subject_length,
            max_body_length=self.config.max_body_length,
            min_body_length=self.config.min_body_length,
            validate_email_format=True,
            validate_categories=self.config.validate_categories,
            remove_duplicates=self.config.remove_duplicates,
            email_streaming_threshold_mb=self.config.streaming_threshold_mb,
            email_cache_threshold_mb=self.config.cache_threshold_mb,
            chunk_size_emails=1000,
            enable_augmentation=False  # Disable for production
        )
        
        # Initialize custom EmailDatasetManager that can handle individual JSON files
        self.dataset_manager = ProductionEmailDatasetManager(
            config=self.email_config,
            memory_manager=self.memory_manager
        )
        
        # Initialize tokenizer with multilingual support
        self.tokenizer = self._create_multilingual_tokenizer()
        
        logger.info("Production dataset pipeline initialized")
    
    def _create_multilingual_tokenizer(self) -> EmailTokenizer:
        """Create email tokenizer with multilingual support."""
        logger.info(f"Creating multilingual tokenizer for languages: {self.config.supported_languages}")
        
        # Create tokenizer with expanded vocabulary for multilingual support
        tokenizer = EmailTokenizer(
            vocab_size=8000,  # Increased for multilingual support
            max_seq_len=512
        )
        
        # The EmailTokenizer already has built-in special tokens and multilingual support
        # We can extend it by adding language-specific tokens to the vocabulary later
        
        return tokenizer
    
    def detect_dataset_format(self, dataset_path: str) -> str:
        """
        Detect the format of the dataset (gmail_creator, custom, or standard).
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dataset format: 'gmail_creator', 'custom', or 'standard'
        """
        dataset_path = Path(dataset_path)
        
        # Check for gmail_dataset_creator format indicators
        if (dataset_path / "categories.json").exists():
            try:
                with open(dataset_path / "categories.json", 'r') as f:
                    categories_data = json.load(f)
                
                # Gmail dataset creator format has nested structure
                if isinstance(categories_data, dict) and "categories" in categories_data:
                    logger.info("Detected gmail_dataset_creator format")
                    return "gmail_creator"
            except Exception as e:
                logger.warning(f"Error reading categories.json: {e}")
        
        # Check for custom format indicators
        if (dataset_path / "dataset_metadata.json").exists():
            logger.info("Detected custom format with metadata")
            return "custom"
        
        # Check for standard split directories
        if all((dataset_path / split).exists() for split in ["train", "val", "test"]):
            logger.info("Detected standard format with train/val/test splits")
            return "standard"
        
        logger.warning("Unknown dataset format, assuming standard")
        return "standard"
    
    def load_production_dataset(self, dataset_path: str, split: str = "train") -> Union[EmailDataset, StreamingEmailDataset]:
        """
        Load production dataset with automatic format detection.
        
        Args:
            dataset_path: Path to the dataset directory
            split: Dataset split to load (train/val/test)
            
        Returns:
            Loaded email dataset
        """
        logger.info(f"Loading production dataset from {dataset_path}, split: {split}")
        
        # Detect dataset format
        format_type = self.detect_dataset_format(dataset_path)
        
        # Load dataset based on format
        if format_type == "gmail_creator":
            return self._load_gmail_creator_format(dataset_path, split)
        elif format_type == "custom":
            return self._load_custom_format(dataset_path, split)
        else:
            return self._load_standard_format(dataset_path, split)
    
    def _load_gmail_creator_format(self, dataset_path: str, split: str) -> Union[EmailDataset, StreamingEmailDataset]:
        """Load dataset in gmail_dataset_creator format."""
        logger.info("Loading gmail_dataset_creator format dataset")
        
        # Load categories mapping
        categories_path = Path(dataset_path) / "categories.json"
        with open(categories_path, 'r') as f:
            categories_data = json.load(f)
        
        # Extract categories from gmail_creator format
        if "categories" in categories_data:
            category_mapping = categories_data["categories"]
            self.email_config.categories = list(category_mapping.keys())
        
        # Use standard loading with updated categories
        return self.dataset_manager.load_email_dataset(dataset_path, split, self.tokenizer)
    
    def _load_custom_format(self, dataset_path: str, split: str) -> Union[EmailDataset, StreamingEmailDataset]:
        """Load dataset in custom format with metadata."""
        logger.info("Loading custom format dataset")
        
        # Load metadata
        metadata_path = Path(dataset_path) / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update configuration from metadata
        if "categories" in metadata:
            self.email_config.categories = metadata["categories"]
        if "min_emails_per_category" in metadata:
            self.email_config.min_emails_per_category = metadata["min_emails_per_category"]
        
        # Use standard loading
        return self.dataset_manager.load_email_dataset(dataset_path, split, self.tokenizer)
    
    def _load_standard_format(self, dataset_path: str, split: str) -> Union[EmailDataset, StreamingEmailDataset]:
        """Load dataset in standard format."""
        logger.info("Loading standard format dataset")
        return self.dataset_manager.load_email_dataset(dataset_path, split, self.tokenizer)
    
    def validate_multilingual_support(self, dataset: Union[EmailDataset, StreamingEmailDataset]) -> Dict[str, Any]:
        """
        Validate multilingual support in the dataset.
        
        Args:
            dataset: Loaded email dataset
            
        Returns:
            Validation results including language distribution
        """
        logger.info("Validating multilingual support...")
        
        language_counts = {}
        total_emails = 0
        unsupported_languages = set()
        
        # Sample emails to check language distribution
        if isinstance(dataset, EmailDataset):
            emails = dataset.emails[:1000]  # Sample first 1000 emails
        else:
            # For streaming datasets, we'll check what we can
            emails = []
            for i, sample in enumerate(dataset):
                if i >= 1000:  # Limit sampling
                    break
                # Extract language from metadata if available
                if 'metadata' in sample and 'language' in sample['metadata']:
                    lang = sample['metadata']['language']
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    if lang not in self.config.supported_languages:
                        unsupported_languages.add(lang)
                total_emails += 1
            
            return {
                'total_emails_sampled': total_emails,
                'language_distribution': language_counts,
                'supported_languages': self.config.supported_languages,
                'unsupported_languages': list(unsupported_languages),
                'multilingual_support_ok': len(unsupported_languages) == 0
            }
        
        # For regular datasets, check email language field
        for email in emails:
            lang = getattr(email, 'language', self.config.default_language)
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            if lang not in self.config.supported_languages:
                unsupported_languages.add(lang)
            
            total_emails += 1
        
        validation_results = {
            'total_emails_sampled': total_emails,
            'language_distribution': language_counts,
            'supported_languages': self.config.supported_languages,
            'unsupported_languages': list(unsupported_languages),
            'multilingual_support_ok': len(unsupported_languages) == 0
        }
        
        logger.info(f"Language distribution: {language_counts}")
        if unsupported_languages:
            logger.warning(f"Found unsupported languages: {unsupported_languages}")
        else:
            logger.info("All languages are supported")
        
        return validation_results
    
    def create_production_dataloader(self, dataset_path: str, split: str = "train", 
                                   batch_size: Optional[int] = None) -> tuple:
        """
        Create production dataloader with optimized settings.
        
        Args:
            dataset_path: Path to the dataset
            split: Dataset split
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Tuple of (dataloader, creation_info)
        """
        batch_size = batch_size or self.config.batch_size
        
        logger.info(f"Creating production dataloader for {split} split, batch_size: {batch_size}")
        
        # Create dataloader using the dataset manager
        dataloader, creation_info = self.dataset_manager.create_email_dataloader(
            dataset_path=dataset_path,
            batch_size=batch_size,
            split=split,
            tokenizer=self.tokenizer,
            shuffle=(split == "train"),
            num_workers=0,  # Disable multiprocessing for stability
            pin_memory=False,  # Better for MacBook
            drop_last=(split == "train")
        )
        
        logger.info(f"Created dataloader: {creation_info}")
        return dataloader, creation_info
    
    def validate_production_pipeline(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate the entire production pipeline.
        
        Args:
            dataset_path: Path to the production dataset
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Validating production dataset pipeline...")
        
        validation_results = {
            'dataset_path': dataset_path,
            'format_detection': None,
            'splits_validation': {},
            'multilingual_validation': {},
            'dataloader_creation': {},
            'overall_status': 'unknown'
        }
        
        try:
            # 1. Format detection
            format_type = self.detect_dataset_format(dataset_path)
            validation_results['format_detection'] = {
                'detected_format': format_type,
                'status': 'success'
            }
            
            # 2. Validate each split
            for split in ['train', 'val', 'test']:
                try:
                    dataset = self.load_production_dataset(dataset_path, split)
                    dataset_size = len(dataset) if hasattr(dataset, '__len__') else 'streaming'
                    
                    validation_results['splits_validation'][split] = {
                        'status': 'success',
                        'dataset_size': dataset_size,
                        'dataset_type': type(dataset).__name__
                    }
                    
                    # Validate multilingual support for train split
                    if split == 'train':
                        multilingual_results = self.validate_multilingual_support(dataset)
                        validation_results['multilingual_validation'] = multilingual_results
                    
                except Exception as e:
                    validation_results['splits_validation'][split] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # 3. Test dataloader creation
            try:
                dataloader, creation_info = self.create_production_dataloader(dataset_path, 'train')
                # Remove non-serializable items from creation_info
                serializable_info = creation_info.copy()
                if 'dataloader_config' in serializable_info and 'collate_fn' in serializable_info['dataloader_config']:
                    serializable_info['dataloader_config'] = serializable_info['dataloader_config'].copy()
                    serializable_info['dataloader_config']['collate_fn'] = '<custom_collate_function>'
                
                validation_results['dataloader_creation'] = {
                    'status': 'success',
                    'creation_info': serializable_info
                }
            except Exception as e:
                validation_results['dataloader_creation'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # 4. Overall status
            all_splits_ok = all(
                result.get('status') == 'success' 
                for result in validation_results['splits_validation'].values()
            )
            dataloader_ok = validation_results['dataloader_creation'].get('status') == 'success'
            multilingual_ok = validation_results['multilingual_validation'].get('multilingual_support_ok', True)
            
            if all_splits_ok and dataloader_ok and multilingual_ok:
                validation_results['overall_status'] = 'success'
            else:
                validation_results['overall_status'] = 'partial_success'
                
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
        
        logger.info(f"Pipeline validation completed: {validation_results['overall_status']}")
        return validation_results
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        return self.dataset_manager.get_email_dataset_stats()

def main():
    """Main entry point for testing the production dataset pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure and test production dataset pipeline")
    parser.add_argument(
        '--dataset-path',
        default='data/production-emails',
        help='Path to production dataset'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation, do not create dataloaders'
    )
    parser.add_argument(
        '--test-splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to test'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        config = ProductionDatasetConfig(
            production_dataset_path=args.dataset_path,
            supported_languages=['en', 'tr'],
            streaming_threshold_mb=500.0,
            batch_size=32
        )
        
        pipeline = ProductionDatasetPipeline(config)
        
        if args.validate_only:
            # Run validation only
            results = pipeline.validate_production_pipeline(args.dataset_path)
            print(f"Validation Results: {json.dumps(results, indent=2)}")
        else:
            # Test loading and dataloader creation for each split
            for split in args.test_splits:
                logger.info(f"Testing {split} split...")
                
                # Load dataset
                dataset = pipeline.load_production_dataset(args.dataset_path, split)
                logger.info(f"Loaded {split} dataset: {type(dataset).__name__}")
                
                # Create dataloader
                dataloader, info = pipeline.create_production_dataloader(args.dataset_path, split)
                logger.info(f"Created {split} dataloader: {info}")
                
                # Test first batch
                try:
                    first_batch = next(iter(dataloader))
                    logger.info(f"First batch shape: input_ids={first_batch['input_ids'].shape}, "
                              f"labels={first_batch['labels'].shape}")
                except Exception as e:
                    logger.error(f"Error getting first batch: {e}")
        
        logger.info("Production dataset pipeline configuration completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline configuration failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())