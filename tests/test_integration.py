"""
Integration tests for scraping pipeline
"""

import pytest
import asyncio
from scraping.utils.validator import DataValidator
from scraping.utils.models import RawProductData


def test_scraper_validator_integration():
    """Test integration between scraper output and validator"""
    # Simulate scraper output
    scraped_data = [
        {
            'source': 'test',
            'url': 'https://test.com/product1',
            'name': 'Valid Product',
            'price': 150.0,
            'description': 'This is a valid product description',
            'rating': 4.5,
            'in_stock': True,
            'raw_category': 'test'
        },
        {
            'source': 'test',
            'url': 'https://test.com/product2',
            'name': 'AB',  # Invalid: too short
            'price': 100.0,
            'description': 'Description',
            'raw_category': 'test'
        }
    ]
    
    validator = DataValidator()
    validated = validator.validate_batch(scraped_data)
    
    # Should only validate the first product
    assert len(validated) == 1
    assert validated[0].name == 'Valid Product'


@pytest.mark.asyncio
async def test_end_to_end_pipeline_test_mode():
    """Test end-to-end pipeline in test mode"""
    # This would require mocking the scrapers and Gemini API
    # For now, just test that components can be initialized
    
    from scraping.config.config_manager import ConfigurationManager
    from scraping.utils.validator import DataValidator
    
    # Create minimal config
    import tempfile
    import yaml
    
    config_data = {
        'scraping': {
            'test_mode': True,
            'test_products_limit': 5,
            'websites': [],
            'rate_limit': {
                'requests_per_minute': 20,
                'delay_between_requests': [1, 2],
                'max_concurrent_requests': 5
            },
            'browser': {
                'headless': True,
                'user_agents': ['test-agent'],
                'viewport': {'width': 1920, 'height': 1080},
                'timeout': 30000
            }
        },
        'gemini': {
            'api_key_env': 'GEMINI_API_KEY',
            'model': 'gemini-1.5-flash',
            'max_requests_per_day': 100,
            'retry_attempts': 3,
            'retry_delay': 1,
            'timeout': 30,
            'enhancement_prompt': 'Test prompt'
        },
        'output': {
            'raw_data_path': 'data/test_raw',
            'processed_data_path': 'data/test_processed',
            'final_dataset_path': 'data/test_output.json'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/test.log',
            'error_file': 'logs/test_errors.log',
            'verbose': False
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_path)
        validator = DataValidator()
        
        assert config_manager.is_test_mode() == True
        assert validator is not None
        
    finally:
        import os
        os.unlink(temp_path)
