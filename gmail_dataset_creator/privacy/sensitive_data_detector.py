"""
Sensitive data detection for email content.

This module provides comprehensive detection of sensitive information in email
content including PII, financial data, and other confidential information.
"""

import re
import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class SensitiveDataType(Enum):
    """Types of sensitive data that can be detected."""
    EMAIL_ADDRESS = "email_address"
    PHONE_NUMBER = "phone_number"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"
    URL = "url"
    BANK_ACCOUNT = "bank_account"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    MEDICAL_INFO = "medical_info"
    FINANCIAL_INFO = "financial_info"
    PERSONAL_ID = "personal_id"


@dataclass
class SensitiveMatch:
    """Represents a match of sensitive data."""
    data_type: SensitiveDataType
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float


class SensitiveDataDetector:
    """
    Detects various types of sensitive information in text content.
    
    Uses regex patterns and heuristics to identify PII, financial data,
    and other sensitive information that should be handled carefully.
    """
    
    def __init__(self):
        """Initialize the sensitive data detector."""
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for sensitive data detection."""
        
        # Email addresses (comprehensive pattern)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone numbers (various formats)
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # US format
            re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),  # (123) 456-7890
            re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,14}\b'),  # International
            re.compile(r'\b\d{3}\s\d{3}\s\d{4}\b'),  # Space separated
            re.compile(r'\b\d{10,15}\b'),  # Simple long numbers
        ]
        
        # Credit card numbers
        self.credit_card_patterns = [
            re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Visa
            re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # MasterCard
            re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),  # American Express
            re.compile(r'\b6011[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Discover
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Generic
        ]
        
        # Social Security Numbers
        self.ssn_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # XXX-XX-XXXX
            re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),  # XXX XX XXXX
            re.compile(r'\b\d{9}\b'),  # XXXXXXXXX (context-dependent)
        ]
        
        # IP addresses
        self.ip_pattern = re.compile(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        )
        
        # URLs
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        
        # Bank account numbers (basic patterns)
        self.bank_account_patterns = [
            re.compile(r'\b\d{8,17}\b'),  # Generic account numbers
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4,6}\b'),  # Formatted
        ]
        
        # Passport numbers (various formats)
        self.passport_patterns = [
            re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),  # US format
            re.compile(r'\b\d{9}\b'),  # Numeric passports
            re.compile(r'\b[A-Z]\d{8}\b'),  # Letter + 8 digits
        ]
        
        # Driver's license patterns (basic)
        self.license_patterns = [
            re.compile(r'\b[A-Z]\d{7,8}\b'),  # Letter + digits
            re.compile(r'\b\d{8,10}\b'),  # Numeric licenses
        ]
        
        # Medical information keywords
        self.medical_keywords = [
            'diagnosis', 'prescription', 'medication', 'doctor', 'physician',
            'hospital', 'clinic', 'patient', 'medical', 'health', 'treatment',
            'surgery', 'therapy', 'insurance', 'medicare', 'medicaid'
        ]
        
        # Financial information keywords
        self.financial_keywords = [
            'account', 'balance', 'loan', 'mortgage', 'credit', 'debit',
            'investment', 'portfolio', 'bank', 'routing', 'swift', 'iban',
            'salary', 'income', 'tax', 'irs', 'w2', '1099'
        ]
        
        # Personal ID keywords
        self.personal_id_keywords = [
            'ssn', 'social security', 'passport', 'license', 'id number',
            'employee id', 'student id', 'member id', 'customer id'
        ]
    
    def detect_sensitive_data(self, text: str) -> List[SensitiveMatch]:
        """
        Detect sensitive data in text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of sensitive data matches found
        """
        if not text:
            return []
        
        matches = []
        
        # Detect different types of sensitive data
        matches.extend(self._detect_emails(text))
        matches.extend(self._detect_phone_numbers(text))
        matches.extend(self._detect_credit_cards(text))
        matches.extend(self._detect_ssns(text))
        matches.extend(self._detect_ip_addresses(text))
        matches.extend(self._detect_urls(text))
        matches.extend(self._detect_bank_accounts(text))
        matches.extend(self._detect_passports(text))
        matches.extend(self._detect_licenses(text))
        matches.extend(self._detect_medical_info(text))
        matches.extend(self._detect_financial_info(text))
        matches.extend(self._detect_personal_ids(text))
        
        # Sort matches by position
        matches.sort(key=lambda x: x.start_pos)
        
        return matches
    
    def _detect_emails(self, text: str) -> List[SensitiveMatch]:
        """Detect email addresses in text."""
        matches = []
        for match in self.email_pattern.finditer(text):
            matches.append(SensitiveMatch(
                data_type=SensitiveDataType.EMAIL_ADDRESS,
                matched_text=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95
            ))
        return matches
    
    def _detect_phone_numbers(self, text: str) -> List[SensitiveMatch]:
        """Detect phone numbers in text."""
        matches = []
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                # Basic validation to reduce false positives
                phone_text = match.group()
                if self._is_likely_phone(phone_text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.PHONE_NUMBER,
                        matched_text=phone_text,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8
                    ))
        return matches
    
    def _detect_credit_cards(self, text: str) -> List[SensitiveMatch]:
        """Detect credit card numbers in text."""
        matches = []
        for pattern in self.credit_card_patterns:
            for match in pattern.finditer(text):
                cc_text = match.group()
                if self._is_likely_credit_card(cc_text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.CREDIT_CARD,
                        matched_text=cc_text,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9
                    ))
        return matches
    
    def _detect_ssns(self, text: str) -> List[SensitiveMatch]:
        """Detect Social Security Numbers in text."""
        matches = []
        for pattern in self.ssn_patterns:
            for match in pattern.finditer(text):
                ssn_text = match.group()
                if self._is_likely_ssn(ssn_text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.SSN,
                        matched_text=ssn_text,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85
                    ))
        return matches
    
    def _detect_ip_addresses(self, text: str) -> List[SensitiveMatch]:
        """Detect IP addresses in text."""
        matches = []
        for match in self.ip_pattern.finditer(text):
            ip_text = match.group()
            if self._is_valid_ip(ip_text):
                matches.append(SensitiveMatch(
                    data_type=SensitiveDataType.IP_ADDRESS,
                    matched_text=ip_text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                ))
        return matches
    
    def _detect_urls(self, text: str) -> List[SensitiveMatch]:
        """Detect URLs in text."""
        matches = []
        for match in self.url_pattern.finditer(text):
            matches.append(SensitiveMatch(
                data_type=SensitiveDataType.URL,
                matched_text=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95
            ))
        return matches
    
    def _detect_bank_accounts(self, text: str) -> List[SensitiveMatch]:
        """Detect bank account numbers in text."""
        matches = []
        # Look for bank account context
        bank_context = re.search(r'\b(account|routing|bank|aba)\b', text, re.IGNORECASE)
        if bank_context:
            for pattern in self.bank_account_patterns:
                for match in pattern.finditer(text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.BANK_ACCOUNT,
                        matched_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.7
                    ))
        return matches
    
    def _detect_passports(self, text: str) -> List[SensitiveMatch]:
        """Detect passport numbers in text."""
        matches = []
        # Look for passport context
        passport_context = re.search(r'\bpassport\b', text, re.IGNORECASE)
        if passport_context:
            for pattern in self.passport_patterns:
                for match in pattern.finditer(text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.PASSPORT,
                        matched_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8
                    ))
        return matches
    
    def _detect_licenses(self, text: str) -> List[SensitiveMatch]:
        """Detect driver's license numbers in text."""
        matches = []
        # Look for license context
        license_context = re.search(r'\b(license|dl|driver)\b', text, re.IGNORECASE)
        if license_context:
            for pattern in self.license_patterns:
                for match in pattern.finditer(text):
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.DRIVER_LICENSE,
                        matched_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.7
                    ))
        return matches
    
    def _detect_medical_info(self, text: str) -> List[SensitiveMatch]:
        """Detect medical information in text."""
        matches = []
        text_lower = text.lower()
        
        for keyword in self.medical_keywords:
            if keyword in text_lower:
                # Find all occurrences of the keyword
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.MEDICAL_INFO,
                        matched_text=text[pos:pos+len(keyword)],
                        start_pos=pos,
                        end_pos=pos+len(keyword),
                        confidence=0.6
                    ))
                    start = pos + 1
        
        return matches
    
    def _detect_financial_info(self, text: str) -> List[SensitiveMatch]:
        """Detect financial information in text."""
        matches = []
        text_lower = text.lower()
        
        for keyword in self.financial_keywords:
            if keyword in text_lower:
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.FINANCIAL_INFO,
                        matched_text=text[pos:pos+len(keyword)],
                        start_pos=pos,
                        end_pos=pos+len(keyword),
                        confidence=0.6
                    ))
                    start = pos + 1
        
        return matches
    
    def _detect_personal_ids(self, text: str) -> List[SensitiveMatch]:
        """Detect personal ID information in text."""
        matches = []
        text_lower = text.lower()
        
        for keyword in self.personal_id_keywords:
            if keyword in text_lower:
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    matches.append(SensitiveMatch(
                        data_type=SensitiveDataType.PERSONAL_ID,
                        matched_text=text[pos:pos+len(keyword)],
                        start_pos=pos,
                        end_pos=pos+len(keyword),
                        confidence=0.5
                    ))
                    start = pos + 1
        
        return matches
    
    def _is_likely_phone(self, phone_text: str) -> bool:
        """Check if text is likely a phone number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', phone_text)
        
        # Check length
        if len(digits) < 7 or len(digits) > 15:
            return False
        
        # Check for obvious non-phone patterns
        if digits == '0' * len(digits):  # All zeros
            return False
        if digits == '1' * len(digits):  # All ones
            return False
        
        return True
    
    def _is_likely_credit_card(self, cc_text: str) -> bool:
        """Check if text is likely a credit card number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', cc_text)
        
        # Check length (13-19 digits for most cards)
        if len(digits) < 13 or len(digits) > 19:
            return False
        
        # Basic Luhn algorithm check
        return self._luhn_check(digits)
    
    def _is_likely_ssn(self, ssn_text: str) -> bool:
        """Check if text is likely a Social Security Number."""
        # Remove non-digits
        digits = re.sub(r'\D', '', ssn_text)
        
        # Must be exactly 9 digits
        if len(digits) != 9:
            return False
        
        # Check for invalid SSN patterns
        if digits.startswith('000'):  # Invalid area number
            return False
        if digits[3:5] == '00':  # Invalid group number
            return False
        if digits[5:9] == '0000':  # Invalid serial number
            return False
        
        return True
    
    def _is_valid_ip(self, ip_text: str) -> bool:
        """Check if text is a valid IP address."""
        parts = ip_text.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            return True
        except ValueError:
            return False
    
    def _luhn_check(self, card_number: str) -> bool:
        """Perform Luhn algorithm check for credit card validation."""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d*2))
        return checksum % 10 == 0
    
    def get_sensitive_data_summary(self, matches: List[SensitiveMatch]) -> Dict[str, int]:
        """
        Get summary of detected sensitive data types.
        
        Args:
            matches: List of sensitive data matches
            
        Returns:
            Dictionary with counts of each sensitive data type
        """
        summary = {}
        for match in matches:
            data_type = match.data_type.value
            summary[data_type] = summary.get(data_type, 0) + 1
        
        return summary