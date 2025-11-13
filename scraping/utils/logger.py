"""
Logging utilities for Web Scraping Pipeline
Provides centralized logging with rotating file handlers
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from pathlib import Path


class ScrapingLogger:
    """Centralized logging for scraping pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize logging system
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.logger = None
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration with rotating file handlers"""
        # Create logs directory if it doesn't exist
        log_file = self.config.get('file', 'logs/scraping.log')
        error_file = self.config.get('error_file', 'logs/scraping_errors.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        
        # Get log level
        level_str = self.config.get('level', 'INFO')
        level = getattr(logging, level_str.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Main log file handler (rotating)
        main_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(level)
        main_handler.setFormatter(formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file handler (rotating)
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Verbose mode: detailed console output
        if self.config.get('verbose', False):
            verbose_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(verbose_formatter)
        else:
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
        
        self.logger = root_logger

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(name)
        return logging.getLogger()
    
    @staticmethod
    def get_module_logger(module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module
        
        Args:
            module_name: Name of the module
            
        Returns:
            Logger instance for the module
        """
        return logging.getLogger(module_name)


def setup_logger(config: Dict[str, Any]) -> ScrapingLogger:
    """
    Convenience function to setup logger
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured ScrapingLogger instance
    """
    return ScrapingLogger(config)
