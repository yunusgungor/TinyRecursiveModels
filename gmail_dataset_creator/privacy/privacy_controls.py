"""
Privacy controls for email dataset creation.

This module provides comprehensive privacy controls including options to exclude
personal or sensitive emails, data anonymization, and secure data cleanup.
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..models import EmailData
from .sensitive_data_detector import SensitiveDataDetector
from .data_anonymizer import DataAnonymizer


@dataclass
class PrivacySettings:
    """Configuration for privacy controls."""
    exclude_personal: bool = False
    exclude_sensitive: bool = True
    anonymize_senders: bool = True
    anonymize_recipients: bool = True
    remove_sensitive_content: bool = True
    exclude_keywords: List[str] = None
    exclude_domains: List[str] = None
    min_confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.exclude_keywords is None:
            self.exclude_keywords = []
        if self.exclude_domains is None:
            self.exclude_domains = []


class PrivacyController:
    """
    Main privacy controller that orchestrates privacy and security measures.
    
    Handles email filtering, content anonymization, and sensitive data detection
    to ensure user privacy is maintained during dataset creation.
    """
    
    def __init__(self, privacy_settings: PrivacySettings):
        """
        Initialize privacy controller.
        
        Args:
            privacy_settings: Privacy configuration settings
        """
        self.settings = privacy_settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize privacy components
        self.sensitive_detector = SensitiveDataDetector()
        self.anonymizer = DataAnonymizer(
            anonymize_emails=privacy_settings.anonymize_senders,
            remove_sensitive=privacy_settings.remove_sensitive_content
        )
        
        # Compile patterns for efficiency
        self._compile_exclusion_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'excluded_personal': 0,
            'excluded_sensitive': 0,
            'excluded_keywords': 0,
            'excluded_domains': 0,
            'anonymized_emails': 0
        }
    
    def _compile_exclusion_patterns(self):
        """Compile regex patterns for exclusion criteria."""
        # Personal email indicators
        self.personal_patterns = [
            re.compile(r'\b(love|dear|honey|sweetheart|darling)\b', re.IGNORECASE),
            re.compile(r'\b(family|mom|dad|mother|father|sister|brother)\b', re.IGNORECASE),
            re.compile(r'\b(birthday|anniversary|wedding|graduation)\b', re.IGNORECASE),
            re.compile(r'\b(personal|private|confidential)\b', re.IGNORECASE),
        ]
        
        # Keyword exclusion patterns
        if self.settings.exclude_keywords:
            self.keyword_patterns = [
                re.compile(re.escape(keyword), re.IGNORECASE) 
                for keyword in self.settings.exclude_keywords
            ]
        else:
            self.keyword_patterns = []
        
        # Domain exclusion patterns
        if self.settings.exclude_domains:
            domain_pattern = '|'.join(re.escape(domain) for domain in self.settings.exclude_domains)
            self.domain_pattern = re.compile(f'@({domain_pattern})', re.IGNORECASE)
        else:
            self.domain_pattern = None
    
    def should_exclude_email(self, email_data: EmailData) -> Tuple[bool, str]:
        """
        Determine if an email should be excluded based on privacy settings.
        
        Args:
            email_data: Email to evaluate
            
        Returns:
            Tuple of (should_exclude, reason)
        """
        self.stats['total_processed'] += 1
        
        # Check for personal content
        if self.settings.exclude_personal and self._is_personal_email(email_data):
            self.stats['excluded_personal'] += 1
            return True, "personal_content"
        
        # Check for sensitive content
        if self.settings.exclude_sensitive and self._contains_sensitive_data(email_data):
            self.stats['excluded_sensitive'] += 1
            return True, "sensitive_content"
        
        # Check for excluded keywords
        if self._contains_excluded_keywords(email_data):
            self.stats['excluded_keywords'] += 1
            return True, "excluded_keywords"
        
        # Check for excluded domains
        if self._from_excluded_domain(email_data):
            self.stats['excluded_domains'] += 1
            return True, "excluded_domain"
        
        return False, ""
    
    def _is_personal_email(self, email_data: EmailData) -> bool:
        """
        Check if email contains personal content.
        
        Args:
            email_data: Email to check
            
        Returns:
            True if email appears to be personal
        """
        # Combine subject and body for analysis
        content = f"{email_data.subject} {email_data.body}".lower()
        
        # Check against personal patterns
        for pattern in self.personal_patterns:
            if pattern.search(content):
                return True
        
        # Check for personal email domains (common personal email providers)
        personal_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'icloud.com', 'aol.com', 'protonmail.com'
        ]
        
        sender_domain = self._extract_domain(email_data.sender)
        recipient_domain = self._extract_domain(email_data.recipient)
        
        # If both sender and recipient are from personal domains, likely personal
        if (sender_domain in personal_domains and 
            recipient_domain in personal_domains):
            return True
        
        return False
    
    def _contains_sensitive_data(self, email_data: EmailData) -> bool:
        """
        Check if email contains sensitive data.
        
        Args:
            email_data: Email to check
            
        Returns:
            True if email contains sensitive information
        """
        # Use sensitive data detector
        sensitive_types = self.sensitive_detector.detect_sensitive_data(
            f"{email_data.subject} {email_data.body}"
        )
        
        # Consider email sensitive if it contains any sensitive data types
        return len(sensitive_types) > 0
    
    def _contains_excluded_keywords(self, email_data: EmailData) -> bool:
        """
        Check if email contains excluded keywords.
        
        Args:
            email_data: Email to check
            
        Returns:
            True if email contains excluded keywords
        """
        if not self.keyword_patterns:
            return False
        
        content = f"{email_data.subject} {email_data.body}"
        
        for pattern in self.keyword_patterns:
            if pattern.search(content):
                return True
        
        return False
    
    def _from_excluded_domain(self, email_data: EmailData) -> bool:
        """
        Check if email is from an excluded domain.
        
        Args:
            email_data: Email to check
            
        Returns:
            True if email is from excluded domain
        """
        if not self.domain_pattern:
            return False
        
        # Check both sender and recipient domains
        if (self.domain_pattern.search(email_data.sender) or 
            self.domain_pattern.search(email_data.recipient)):
            return True
        
        return False
    
    def _extract_domain(self, email_address: str) -> str:
        """
        Extract domain from email address.
        
        Args:
            email_address: Email address to extract domain from
            
        Returns:
            Domain part of email address
        """
        try:
            return email_address.split('@')[1].lower()
        except (IndexError, AttributeError):
            return ""
    
    def apply_privacy_controls(self, email_data: EmailData) -> Optional[EmailData]:
        """
        Apply privacy controls to an email.
        
        Args:
            email_data: Email to process
            
        Returns:
            Processed email data or None if email should be excluded
        """
        # Check if email should be excluded
        should_exclude, reason = self.should_exclude_email(email_data)
        if should_exclude:
            self.logger.debug(f"Excluding email {email_data.id}: {reason}")
            return None
        
        # Apply anonymization
        processed_email = self.anonymizer.anonymize_email_data(email_data)
        
        if processed_email != email_data:
            self.stats['anonymized_emails'] += 1
        
        return processed_email
    
    def get_privacy_stats(self) -> Dict[str, int]:
        """
        Get privacy processing statistics.
        
        Returns:
            Dictionary of privacy statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset privacy processing statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def validate_privacy_settings(self) -> List[str]:
        """
        Validate privacy settings and return any warnings.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if (self.settings.exclude_personal and 
            self.settings.exclude_sensitive and 
            len(self.settings.exclude_keywords) > 10):
            warnings.append("Very restrictive privacy settings may result in very small dataset")
        
        if self.settings.min_confidence_threshold > 0.9:
            warnings.append("High confidence threshold may exclude valid emails")
        
        if not self.settings.anonymize_senders and not self.settings.remove_sensitive_content:
            warnings.append("No anonymization enabled - personal data may be exposed")
        
        return warnings