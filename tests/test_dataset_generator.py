"""
Unit tests for DatasetGenerator
"""

import pytest
import tempfile
import json
from pathlib import Path

from scraping.services.dataset_generator import DatasetGenerator
from scraping.utils.models import RawProductData


def test_generate_dataset():
    """Test dataset generation"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        generator = DatasetGenerator(temp_path)
        
        # Create test data
        products = [
            RawProductData(
                source='test', url='https://test.com/1', name='Product 1',
                price=100.0, description='Description 1', raw_category='test'
            )
        ]
        
        enhancements = [
            {
                'category': 'technology',
                'target_audience': ['adults'],
                'gift_occasions': ['birthday'],
                'emotional_tags': ['practical'],
                'age_range': [18, 65]
            }
        ]
        
        dataset = generator.generate_dataset(products, enhancements)
        
        assert 'gifts' in dataset
        assert 'metadata' in dataset
        assert len(dataset['gifts']) == 1
        assert dataset['gifts'][0]['category'] == 'technology'
        
        # Check file was created
        assert Path(temp_path).exists()
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_metadata_generation():
    """Test metadata generation"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        generator = DatasetGenerator(temp_path)
        
        gifts = [
            {
                'id': 'test_001',
                'category': 'technology',
                'price': 100.0,
                'source': 'test'
            },
            {
                'id': 'test_002',
                'category': 'books',
                'price': 50.0,
                'source': 'test'
            }
        ]
        
        metadata = generator._generate_metadata(gifts)
        
        assert metadata['total_gifts'] == 2
        assert 'technology' in metadata['categories']
        assert 'books' in metadata['categories']
        assert metadata['price_range']['min'] == 50.0
        assert metadata['price_range']['max'] == 100.0
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
