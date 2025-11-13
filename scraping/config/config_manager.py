"""
Configuration Manager for Web Scraping Pipeline
Handles loading and accessing configuration from YAML file
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigurationManager:
    """Manages configuration for the scraping pipeline"""
    
    def __init__(self, config_path: str = "config/scraping_config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def get_website_config(self, website_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific website
        
        Args:
            website_name: Name of the website (e.g., 'ciceksepeti')
            
        Returns:
            Website configuration dictionary or None if not found
        """
        websites = self.config.get('scraping', {}).get('websites', [])
        for website in websites:
            if website.get('name') == website_name:
                return website
        return None
    
    def get_enabled_websites(self) -> List[Dict[str, Any]]:
        """
        Get list of enabled websites
        
        Returns:
            List of enabled website configurations
        """
        websites = self.config.get('scraping', {}).get('websites', [])
        return [w for w in websites if w.get('enabled', False)]
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """
        Get rate limiting configuration
        
        Returns:
            Rate limit configuration dictionary
        """
        return self.config.get('scraping', {}).get('rate_limit', {})
    
    def get_browser_config(self) -> Dict[str, Any]:
        """
        Get browser configuration
        
        Returns:
            Browser configuration dictionary
        """
        return self.config.get('scraping', {}).get('browser', {})

    def get_gemini_config(self) -> Dict[str, Any]:
        """
        Get Gemini API configuration
        
        Returns:
            Gemini configuration dictionary
        """
        return self.config.get('gemini', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output paths configuration
        
        Returns:
            Output configuration dictionary
        """
        return self.config.get('output', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration
        
        Returns:
            Logging configuration dictionary
        """
        return self.config.get('logging', {})
    
    def is_test_mode(self) -> bool:
        """
        Check if test mode is enabled
        
        Returns:
            True if test mode is enabled, False otherwise
        """
        return self.config.get('scraping', {}).get('test_mode', False)
    
    def get_test_products_limit(self) -> int:
        """
        Get test mode products limit
        
        Returns:
            Number of products to scrape in test mode
        """
        return self.config.get('scraping', {}).get('test_products_limit', 10)

    def get_all_config(self) -> Dict[str, Any]:
        """
        Get entire configuration
        
        Returns:
            Complete configuration dictionary
        """
        return self.config
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value
        
        Args:
            key_path: Dot-separated path to config key (e.g., 'scraping.test_mode')
            value: New value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file
        
        Args:
            output_path: Path to save config (defaults to original path)
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
