"""Security utilities for encryption, sanitization, and validation"""

import re
import html
import hashlib
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from app.core.config import settings


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    def __init__(self, secret_key: str = None):
        """Initialize encryption service with secret key"""
        key = secret_key or settings.SECRET_KEY
        # Derive a proper Fernet key from the secret
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'trendyol_gift_salt',  # In production, use a random salt
            iterations=100000,
            backend=default_backend()
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.cipher = Fernet(derived_key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not data:
            return ""
        encrypted = self.cipher.encrypt(data.encode())
        return encrypted.decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted string data"""
        if not encrypted_data:
            return ""
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return decrypted.decode()
    
    def encrypt_dict(self, data: Dict[str, Any], fields: list[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        for field in fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], fields: list[str]) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in fields:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
        return decrypted_data


class InputSanitizer:
    """Service for sanitizing user inputs to prevent XSS and injection attacks"""
    
    # Patterns for detecting potential attacks
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
    ]
    
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
    ]
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML to prevent XSS attacks"""
        if not text:
            return ""
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove any remaining script tags or dangerous patterns
        for pattern in InputSanitizer.XSS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def sanitize_sql(text: str) -> str:
        """Sanitize input to prevent SQL injection"""
        if not text:
            return ""
        
        # Check for SQL injection patterns
        for pattern in InputSanitizer.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # If detected, escape or remove
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_input(text: str, allow_html: bool = False) -> str:
        """General input sanitization"""
        if not text:
            return ""
        
        # Trim whitespace
        sanitized = text.strip()
        
        # SQL injection prevention
        sanitized = InputSanitizer.sanitize_sql(sanitized)
        
        # XSS prevention (unless HTML is explicitly allowed)
        if not allow_html:
            sanitized = InputSanitizer.sanitize_html(sanitized)
        
        return sanitized
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any], fields: list[str] = None) -> Dict[str, Any]:
        """Sanitize specific fields in a dictionary"""
        sanitized_data = data.copy()
        
        # If no fields specified, sanitize all string fields
        if fields is None:
            fields = [k for k, v in data.items() if isinstance(v, str)]
        
        for field in fields:
            if field in sanitized_data and isinstance(sanitized_data[field], str):
                sanitized_data[field] = InputSanitizer.sanitize_input(sanitized_data[field])
        
        return sanitized_data


class SecurityValidator:
    """Validator for security-related checks"""
    
    @staticmethod
    def is_safe_url(url: str) -> bool:
        """Check if URL is safe (no javascript:, data:, etc.)"""
        if not url:
            return False
        
        url_lower = url.lower().strip()
        
        # Check for dangerous protocols
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
        
        # Must be http or https
        if not (url_lower.startswith('http://') or url_lower.startswith('https://')):
            return False
        
        return True
    
    @staticmethod
    def contains_xss(text: str) -> bool:
        """Check if text contains potential XSS"""
        if not text:
            return False
        
        for pattern in InputSanitizer.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def contains_sql_injection(text: str) -> bool:
        """Check if text contains potential SQL injection"""
        if not text:
            return False
        
        for pattern in InputSanitizer.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


# Global instances
encryption_service = EncryptionService()
input_sanitizer = InputSanitizer()
security_validator = SecurityValidator()
