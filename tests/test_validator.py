"""
Unit tests for DataValidator
"""

import pytest
from scraping.utils.validator import DataValidator
from scraping.utils.models import RawProductData


def test_validate_product_success():
    """Test successful product validation"""
    validator = DataValidator()
    
    raw_data = {
        'source': 'test',
        'url': 'https://test.com/product',
        'name': 'Test Product',
        'price': 100.0,
        'description': 'This is a test product description',
        'rating': 4.5,
        'in_stock': True,
        'raw_category': 'test'
    }
    
    result = validator.validate_product(raw_data)
    assert result is not None
    assert result.name == 'Test Product'
    assert result.price == 100.0


def test_validate_product_failure():
    """Test product validation failure"""
    validator = DataValidator()
    
    # Invalid: name too short
    raw_data = {
        'source': 'test',
        'url': 'https://test.com/product',
        'name': 'AB',  # Too short
        'price': 100.0,
        'description': 'Description',
        'raw_category': 'test'
    }
    
    result = validator.validate_product(raw_data)
    assert result is None


def test_remove_duplicates():
    """Test duplicate removal"""
    validator = DataValidator()
    
    products = [
        RawProductData(
            source='test', url='https://test.com/1', name='Product A',
            price=100.0, description='Description A', raw_category='test'
        ),
        RawProductData(
            source='test', url='https://test.com/2', name='Product A',
            price=100.0, description='Description A', raw_category='test'
        ),
        RawProductData(
            source='test', url='https://test.com/3', name='Product B',
            price=200.0, description='Description B', raw_category='test'
        )
    ]
    
    unique = validator.remove_duplicates(products)
    assert len(unique) == 2  # Should remove one duplicate


def test_filter_by_price_range():
    """Test price range filtering"""
    validator = DataValidator()
    
    products = [
        RawProductData(
            source='test', url='https://test.com/1', name='Cheap Product',
            price=50.0, description='Description', raw_category='test'
        ),
        RawProductData(
            source='test', url='https://test.com/2', name='Expensive Product',
            price=500.0, description='Description', raw_category='test'
        )
    ]
    
    filtered = validator.filter_by_price_range(products, min_price=0, max_price=100)
    assert len(filtered) == 1
    assert filtered[0].price == 50.0
