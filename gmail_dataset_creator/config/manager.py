"""
Configuration management for Gmail Dataset Creator.

Handles loading, validation, and management of system configuration
from YAML files and environment variables.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class GmailAPIConfig:
    """Configuration for Gmail API access."""
    credentials_file: str = "credentials.json"
    token_file: str = "token.json"
    scopes: List[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/gmail.readonly"])


@dataclass
class GeminiAPIConfig:
    """Configuration for Gemini API access."""
    api_key: str = ""
    model: str = "gemini-pro"
    max_tokens: int = 1000


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    output_path: str = "./gmail_dataset"
    train_ratio: float = 0.8
    min_emails_per_category: int = 10
    max_emails_total: int = 1000


@dataclass
class FilterConfig:
    """Configuration for email filtering."""
    date_range: Optional[Tuple[str, str]] = None
    exclude_labels: List[str] = field(default_factory=lambda: ["TRASH", "SPAM"])
    include_labels: List[str] = field(default_factory=lambda: ["INBOX"])
    sender_filters: List[str] = field(default_factory=list)


@dataclass
class PrivacyConfig:
    """Configuration for privacy and security settings."""
    anonymize_senders: bool = True
    exclude_personal: bool = False
    remove_attachments: bool = True
    encrypt_tokens: bool = True
    exclude_sensitive: bool = True
    anonymize_recipients: bool = True
    remove_sensitive_content: bool = True
    exclude_keywords: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    min_confidence_threshold: float = 0.7


@dataclass
class SecurityConfig:
    """Configuration for security measures."""
    encryption_algorithm: str = "fernet"
    key_derivation_function: str = "pbkdf2"
    encryption_iterations: int = 100000
    salt_length: int = 32
    secure_export: bool = True
    data_retention_days: int = 30
    secure_cleanup: bool = True
    audit_logging: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""
    gmail_api: GmailAPIConfig = field(default_factory=GmailAPIConfig)
    gemini_api: GeminiAPIConfig = field(default_factory=GeminiAPIConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


class ConfigManager:
    """
    Manages system configuration loading and validation.
    
    Supports loading configuration from YAML files and environment variables,
    with validation and default value handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self._config: Optional[SystemConfig] = None
    
    def load_config(self) -> SystemConfig:
        """
        Load configuration from file and environment variables.
        
        Returns:
            Complete system configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config_dict = {}
        
        # Load from YAML file if provided
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create configuration objects
        self._config = self._create_config_from_dict(config_dict)
        
        # Validate configuration
        self._validate_config(self._config)
        
        return self._config
    
    def _apply_env_overrides(self, config_dict: Dict) -> Dict:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'GEMINI_API_KEY': ['gemini_api', 'api_key'],
            'GMAIL_CREDENTIALS_PATH': ['gmail_api', 'credentials_file'],
            'OUTPUT_PATH': ['dataset', 'output_path'],
            'MAX_EMAILS': ['dataset', 'max_emails_total'],
            'TRAIN_RATIO': ['dataset', 'train_ratio'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the nested dictionary location
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = config_path[-1]
                if final_key in ['max_emails_total', 'min_emails_per_category', 'max_tokens']:
                    current[final_key] = int(value)
                elif final_key in ['train_ratio']:
                    current[final_key] = float(value)
                elif final_key in ['anonymize_senders', 'exclude_personal', 'remove_attachments', 'encrypt_tokens']:
                    current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    current[final_key] = value
        
        return config_dict
    
    def _create_config_from_dict(self, config_dict: Dict) -> SystemConfig:
        """Create configuration objects from dictionary."""
        gmail_config = GmailAPIConfig(
            **config_dict.get('gmail_api', {})
        )
        
        gemini_config = GeminiAPIConfig(
            **config_dict.get('gemini_api', {})
        )
        
        dataset_config = DatasetConfig(
            **config_dict.get('dataset', {})
        )
        
        filter_config = FilterConfig(
            **config_dict.get('filters', {})
        )
        
        privacy_config = PrivacyConfig(
            **config_dict.get('privacy', {})
        )
        
        security_config = SecurityConfig(
            **config_dict.get('security', {})
        )
        
        return SystemConfig(
            gmail_api=gmail_config,
            gemini_api=gemini_config,
            dataset=dataset_config,
            filters=filter_config,
            privacy=privacy_config,
            security=security_config
        )
    
    def _validate_config(self, config: SystemConfig) -> None:
        """
        Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate Gmail API configuration
        if not config.gmail_api.credentials_file:
            raise ValueError("Gmail credentials file path is required")
        
        # Validate Gemini API configuration
        if not config.gemini_api.api_key:
            raise ValueError("Gemini API key is required")
        
        # Validate dataset configuration
        if config.dataset.train_ratio <= 0 or config.dataset.train_ratio >= 1:
            raise ValueError("Train ratio must be between 0 and 1")
        
        if config.dataset.max_emails_total <= 0:
            raise ValueError("Max emails total must be positive")
        
        if config.dataset.min_emails_per_category < 0:
            raise ValueError("Min emails per category cannot be negative")
        
        # Validate date range format if provided
        if config.filters.date_range:
            try:
                start_date, end_date = config.filters.date_range
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid date range format: {e}")
    
    def get_config(self) -> SystemConfig:
        """
        Get current configuration, loading if necessary.
        
        Returns:
            Current system configuration
        """
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration file
        """
        if self._config is None:
            raise ValueError("No configuration loaded to save")
        
        config_dict = {
            'gmail_api': {
                'credentials_file': self._config.gmail_api.credentials_file,
                'token_file': self._config.gmail_api.token_file,
                'scopes': self._config.gmail_api.scopes,
            },
            'gemini_api': {
                'api_key': self._config.gemini_api.api_key,
                'model': self._config.gemini_api.model,
                'max_tokens': self._config.gemini_api.max_tokens,
            },
            'dataset': {
                'output_path': self._config.dataset.output_path,
                'train_ratio': self._config.dataset.train_ratio,
                'min_emails_per_category': self._config.dataset.min_emails_per_category,
                'max_emails_total': self._config.dataset.max_emails_total,
            },
            'filters': {
                'date_range': self._config.filters.date_range,
                'exclude_labels': self._config.filters.exclude_labels,
                'include_labels': self._config.filters.include_labels,
                'sender_filters': self._config.filters.sender_filters,
            },
            'privacy': {
                'anonymize_senders': self._config.privacy.anonymize_senders,
                'exclude_personal': self._config.privacy.exclude_personal,
                'remove_attachments': self._config.privacy.remove_attachments,
                'encrypt_tokens': self._config.privacy.encrypt_tokens,
                'exclude_sensitive': self._config.privacy.exclude_sensitive,
                'anonymize_recipients': self._config.privacy.anonymize_recipients,
                'remove_sensitive_content': self._config.privacy.remove_sensitive_content,
                'exclude_keywords': self._config.privacy.exclude_keywords,
                'exclude_domains': self._config.privacy.exclude_domains,
                'min_confidence_threshold': self._config.privacy.min_confidence_threshold,
            },
            'security': {
                'encryption_algorithm': self._config.security.encryption_algorithm,
                'key_derivation_function': self._config.security.key_derivation_function,
                'encryption_iterations': self._config.security.encryption_iterations,
                'salt_length': self._config.security.salt_length,
                'secure_export': self._config.security.secure_export,
                'data_retention_days': self._config.security.data_retention_days,
                'secure_cleanup': self._config.security.secure_cleanup,
                'audit_logging': self._config.security.audit_logging,
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)