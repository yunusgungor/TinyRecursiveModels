"""
Data anonymization utilities for email content.

This module provides comprehensive data anonymization capabilities including
email address anonymization, sensitive data replacement, and content sanitization.
"""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, replace

from ..models import EmailData
from .sensitive_data_detector import SensitiveDataDetector, SensitiveDataType


@dataclass
class AnonymizationSettings:
    """Configuration for data anonymization."""
    anonymize_emails: bool = True
    anonymize_phone_numbers: bool = True
    anonymize_credit_cards: bool = True
    anonymize_ssns: bool = True
    anonymize_ip_addresses: bool = True
    anonymize_urls: bool = False  # URLs might be needed for context
    preserve_domain_structure: bool = True
    use_consistent_replacements: bool = True
    hash_seed: str = "gmail_dataset_creator"


class DataAnonymizer:
    """
    Handles anonymization of sensitive data in email content.
    
    Provides consistent anonymization of various data types while preserving
    the overall structure and context of the email content.
    """
    
    def __init__(self, 
                 anonymize_emails: bool = True,
                 remove_sensitive: bool = True,
                 settings: Optional[AnonymizationSettings] = None):
        """
        Initialize data anonymizer.
        
        Args:
            anonymize_emails: Whether to anonymize email addresses
            remove_sensitive: Whether to remove sensitive data
            settings: Detailed anonymization settings
        """
        self.anonymize_emails = anonymize_emails
        self.remove_sensitive = remove_sensitive
        self.settings = settings or AnonymizationSettings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sensitive data detector
        self.sensitive_detector = SensitiveDataDetector()
        
        # Cache for consistent replacements
        self._replacement_cache: Dict[str, str] = {}
        
        # Compile anonymization patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for anonymization."""
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone patterns
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),
            re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,14}\b'),
            re.compile(r'\b\d{3}\s\d{3}\s\d{4}\b'),
        ]
        
        # Credit card patterns
        self.credit_card_patterns = [
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            re.compile(r'\b\d{13,19}\b'),  # Generic card number
        ]
        
        # SSN patterns
        self.ssn_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
        ]
        
        # IP address pattern
        self.ip_pattern = re.compile(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        )
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
    
    def anonymize_email_data(self, email_data: EmailData) -> EmailData:
        """
        Anonymize an EmailData object.
        
        Args:
            email_data: Email data to anonymize
            
        Returns:
            Anonymized email data
        """
        # Create a copy to avoid modifying the original
        anonymized_data = replace(email_data)
        
        # Anonymize subject
        if email_data.subject:
            anonymized_data.subject = self.anonymize_text(email_data.subject)
        
        # Anonymize body
        if email_data.body:
            anonymized_data.body = self.anonymize_text(email_data.body)
        
        # Anonymize sender and recipient
        if self.anonymize_emails:
            if email_data.sender:
                anonymized_data.sender = self.anonymize_email_address(email_data.sender)
            if email_data.recipient:
                anonymized_data.recipient = self.anonymize_email_address(email_data.recipient)
        
        return anonymized_data
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive data in text content.
        
        Args:
            text: Text to anonymize
            
        Returns:
            Anonymized text
        """
        if not text:
            return text
        
        anonymized_text = text
        
        # Detect sensitive data
        sensitive_matches = self.sensitive_detector.detect_sensitive_data(text)
        
        # Sort matches by position (reverse order to maintain positions)
        sensitive_matches.sort(key=lambda x: x.start_pos, reverse=True)
        
        # Apply anonymization based on data type
        for match in sensitive_matches:
            replacement = self._get_replacement_for_match(match)
            if replacement:
                anonymized_text = (
                    anonymized_text[:match.start_pos] + 
                    replacement + 
                    anonymized_text[match.end_pos:]
                )
        
        return anonymized_text
    
    def _get_replacement_for_match(self, match) -> Optional[str]:
        """
        Get appropriate replacement for a sensitive data match.
        
        Args:
            match: SensitiveMatch object
            
        Returns:
            Replacement string or None if no replacement needed
        """
        data_type = match.data_type
        original_text = match.matched_text
        
        # Check if we should anonymize this type
        if not self._should_anonymize_type(data_type):
            return None
        
        # Get consistent replacement if enabled
        if self.settings.use_consistent_replacements:
            cache_key = f"{data_type.value}:{original_text}"
            if cache_key in self._replacement_cache:
                return self._replacement_cache[cache_key]
        
        # Generate replacement based on type
        replacement = self._generate_replacement(data_type, original_text)
        
        # Cache the replacement
        if self.settings.use_consistent_replacements and replacement:
            cache_key = f"{data_type.value}:{original_text}"
            self._replacement_cache[cache_key] = replacement
        
        return replacement
    
    def _should_anonymize_type(self, data_type: SensitiveDataType) -> bool:
        """Check if a data type should be anonymized."""
        type_settings = {
            SensitiveDataType.EMAIL_ADDRESS: self.settings.anonymize_emails,
            SensitiveDataType.PHONE_NUMBER: self.settings.anonymize_phone_numbers,
            SensitiveDataType.CREDIT_CARD: self.settings.anonymize_credit_cards,
            SensitiveDataType.SSN: self.settings.anonymize_ssns,
            SensitiveDataType.IP_ADDRESS: self.settings.anonymize_ip_addresses,
            SensitiveDataType.URL: self.settings.anonymize_urls,
        }
        
        return type_settings.get(data_type, True)  # Default to True for safety
    
    def _generate_replacement(self, data_type: SensitiveDataType, original_text: str) -> str:
        """
        Generate appropriate replacement for sensitive data.
        
        Args:
            data_type: Type of sensitive data
            original_text: Original text to replace
            
        Returns:
            Replacement string
        """
        if data_type == SensitiveDataType.EMAIL_ADDRESS:
            return self._anonymize_email_preserving_structure(original_text)
        elif data_type == SensitiveDataType.PHONE_NUMBER:
            return self._anonymize_phone_number(original_text)
        elif data_type == SensitiveDataType.CREDIT_CARD:
            return "[CREDIT_CARD]"
        elif data_type == SensitiveDataType.SSN:
            return "[SSN]"
        elif data_type == SensitiveDataType.IP_ADDRESS:
            return self._anonymize_ip_address(original_text)
        elif data_type == SensitiveDataType.URL:
            return "[URL]"
        elif data_type == SensitiveDataType.BANK_ACCOUNT:
            return "[BANK_ACCOUNT]"
        elif data_type == SensitiveDataType.PASSPORT:
            return "[PASSPORT]"
        elif data_type == SensitiveDataType.DRIVER_LICENSE:
            return "[LICENSE]"
        else:
            return "[SENSITIVE_DATA]"
    
    def anonymize_email_address(self, email_address: str) -> str:
        """
        Anonymize a single email address.
        
        Args:
            email_address: Email address to anonymize
            
        Returns:
            Anonymized email address
        """
        if not email_address or not self.anonymize_emails:
            return email_address
        
        return self._anonymize_email_preserving_structure(email_address)
    
    def _anonymize_email_preserving_structure(self, email_address: str) -> str:
        """
        Anonymize email while preserving domain structure.
        
        Args:
            email_address: Email address to anonymize
            
        Returns:
            Anonymized email address
        """
        try:
            local, domain = email_address.split('@', 1)
            
            # Anonymize local part
            if len(local) > 2:
                anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
            elif len(local) == 2:
                anonymized_local = local[0] + '*'
            else:
                anonymized_local = '*'
            
            # Handle domain anonymization
            if self.settings.preserve_domain_structure:
                domain_parts = domain.split('.')
                if len(domain_parts) > 1:
                    # Keep TLD, anonymize domain name
                    tld = domain_parts[-1]
                    if len(domain_parts) > 2:
                        # Keep subdomain structure but anonymize
                        anonymized_parts = ['***'] * (len(domain_parts) - 1) + [tld]
                        anonymized_domain = '.'.join(anonymized_parts)
                    else:
                        anonymized_domain = '***.' + tld
                else:
                    anonymized_domain = '***'
            else:
                anonymized_domain = '***'
            
            return f"{anonymized_local}@{anonymized_domain}"
            
        except ValueError:
            # If splitting fails, return generic replacement
            return "[EMAIL]"
    
    def _anonymize_phone_number(self, phone_number: str) -> str:
        """
        Anonymize phone number while preserving format.
        
        Args:
            phone_number: Phone number to anonymize
            
        Returns:
            Anonymized phone number
        """
        # Extract digits
        digits = re.sub(r'\D', '', phone_number)
        
        if len(digits) >= 10:
            # Keep area code, anonymize rest
            if len(digits) == 10:
                return f"({digits[:3]}) ***-****"
            elif len(digits) == 11 and digits.startswith('1'):
                return f"+1 ({digits[1:4]}) ***-****"
            else:
                return f"+{digits[:2]} ***-***-****"
        else:
            return "[PHONE]"
    
    def _anonymize_ip_address(self, ip_address: str) -> str:
        """
        Anonymize IP address while preserving class.
        
        Args:
            ip_address: IP address to anonymize
            
        Returns:
            Anonymized IP address
        """
        parts = ip_address.split('.')
        if len(parts) == 4:
            # Keep first octet for class identification, anonymize rest
            return f"{parts[0]}.***.***.***"
        else:
            return "[IP_ADDRESS]"
    
    def _generate_consistent_hash(self, text: str, length: int = 8) -> str:
        """
        Generate consistent hash for replacement.
        
        Args:
            text: Text to hash
            length: Length of hash to return
            
        Returns:
            Consistent hash string
        """
        # Create hash with seed for consistency
        hash_input = f"{self.settings.hash_seed}:{text}"
        hash_object = hashlib.md5(hash_input.encode())
        hash_hex = hash_object.hexdigest()
        
        # Return first 'length' characters
        return hash_hex[:length]
    
    def get_anonymization_stats(self) -> Dict[str, int]:
        """
        Get statistics about anonymization operations.
        
        Returns:
            Dictionary with anonymization statistics
        """
        stats = {}
        
        # Count replacements by type
        for cache_key in self._replacement_cache:
            data_type = cache_key.split(':')[0]
            stats[data_type] = stats.get(data_type, 0) + 1
        
        return stats
    
    def clear_cache(self):
        """Clear the replacement cache."""
        self._replacement_cache.clear()
        self.logger.debug("Anonymization cache cleared")