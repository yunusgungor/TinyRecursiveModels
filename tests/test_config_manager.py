"""
Unit tests for ConfigurationManager
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from scraping.config.config_manager import ConfigurationManager


def test_load_config():
    """Test loading configuration from YAML file"""
    # Create temporary config file
    config_data = {
        'scraping': {
            'test_mode': True,
            'websites': [
                {'name': 'test_site', 'enabled': True}
            ]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_path)
        assert config_manager.is_test_mode() == True
        assert len(config_manager.get_enabled_websites()) == 1
    finally:
        Path(temp_path).unlink()


def test_get_website_config():
    """Test getting specific website configuration"""
    config_data = {
        'scraping': {
            'websites': [
                {'name': 'ciceksepeti', 'enabled': True, 'max_products': 100}
            ]
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_path)
        website_config = config_manager.get_website_config('ciceksepeti')
        assert website_config is not None
        assert website_config['max_products'] == 100
    finally:
        Path(temp_path).unlink()


def test_update_config():
    """Test updating configuration values"""
    config_data = {'scraping': {'test_mode': False}}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigurationManager(temp_path)
        config_manager.update_config('scraping.test_mode', True)
        assert config_manager.is_test_mode() == True
    finally:
        Path(temp_path).unlink()
