"""
Helper utility functions for Gmail Dataset Creator.

Provides common utility functions for validation, formatting,
and data processing tasks.
"""

import re
import os
from typing import Optional
from datetime import datetime


def validate_email_format(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    if not filename:
        return "untitled"
    
    # Remove invalid characters for most filesystems
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty after sanitization
    if not sanitized:
        sanitized = "untitled"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parse date string in various formats.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # Common date formats to try
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


def ensure_directory_exists(path: str) -> None:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to specified maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text with suffix if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_domain_from_email(email: str) -> Optional[str]:
    """
    Extract domain from email address.
    
    Args:
        email: Email address
        
    Returns:
        Domain part of email or None if invalid
    """
    if not validate_email_format(email):
        return None
    
    try:
        return email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return None


def anonymize_email(email: str) -> str:
    """
    Anonymize email address while preserving domain.
    
    Args:
        email: Email address to anonymize
        
    Returns:
        Anonymized email address
    """
    if not validate_email_format(email):
        return "anonymous@unknown.com"
    
    try:
        local, domain = email.split('@')
        # Keep first and last character of local part if long enough
        if len(local) > 2:
            anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
        else:
            anonymized_local = '*' * len(local)
        
        return f"{anonymized_local}@{domain}"
    except (IndexError, AttributeError):
        return "anonymous@unknown.com"