"""
Email content processing and extraction module.

This module handles the extraction and preprocessing of email content from Gmail API
messages, including handling different email formats, encoding detection, and content
cleaning.
"""

import base64
import email
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import chardet
from bs4 import BeautifulSoup

from ..models import EmailData


class EmailProcessor:
    """
    Processes and extracts content from Gmail API messages.
    
    Handles different email formats (plain text, HTML, multipart), encoding detection,
    and content extraction from Gmail API message payloads.
    """
    
    def __init__(self, anonymize_content: bool = True, remove_sensitive: bool = True):
        """
        Initialize the email processor.
        
        Args:
            anonymize_content: Whether to anonymize sensitive information
            remove_sensitive: Whether to remove sensitive data patterns
        """
        self.encoding_fallbacks = ['utf-8', 'latin-1', 'ascii', 'cp1252']
        self.anonymize_content = anonymize_content
        self.remove_sensitive = remove_sensitive
        
        # Compile regex patterns for sensitive information
        self._compile_patterns()
    
    def extract_content(self, raw_message: Dict[str, Any]) -> EmailData:
        """
        Extract email content from Gmail API message.
        
        Args:
            raw_message: Gmail API message object containing payload and metadata
            
        Returns:
            EmailData object with extracted content
            
        Raises:
            ValueError: If message is invalid or cannot be processed
        """
        if not raw_message or 'id' not in raw_message:
            raise ValueError("Invalid message: missing required fields")
        
        try:
            message_id = raw_message['id']
            
            # Extract basic metadata
            timestamp = self._extract_timestamp(raw_message)
            
            # Extract content from payload
            payload = raw_message.get('payload', {})
            headers = self._extract_headers(payload)
            
            subject = headers.get('Subject', '')
            sender = headers.get('From', '')
            recipient = headers.get('To', '')
            
            # Extract body content
            body = self._extract_body_content(payload)
            
            # Apply content cleaning and anonymization
            if self.anonymize_content or self.remove_sensitive:
                subject = self._clean_and_anonymize_content(subject)
                body = self._clean_and_anonymize_content(body)
                sender = self._anonymize_email_address(sender)
                recipient = self._anonymize_email_address(recipient)
            
            # Store raw content for debugging/fallback
            raw_content = str(raw_message)
            
            email_data = EmailData(
                id=message_id,
                subject=subject,
                body=body,
                sender=sender,
                recipient=recipient,
                timestamp=timestamp,
                raw_content=raw_content
            )
            
            # Validate the extracted data
            if not self.validate_email(email_data):
                raise ValueError(f"Extracted email data failed validation for message {message_id}")
            
            return email_data
            
        except Exception as e:
            # Try to handle corrupted email
            recovered_email = self.handle_corrupted_email(raw_message, e)
            if recovered_email:
                return recovered_email
            else:
                raise ValueError(f"Failed to process email {raw_message.get('id', 'unknown')}: {str(e)}")
    
    def _extract_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract headers from message payload.
        
        Args:
            payload: Gmail API message payload
            
        Returns:
            Dictionary of header name-value pairs
        """
        headers = {}
        header_list = payload.get('headers', [])
        
        for header in header_list:
            name = header.get('name', '')
            value = header.get('value', '')
            if name and value:
                headers[name] = value
        
        return headers
    
    def _extract_timestamp(self, raw_message: Dict[str, Any]) -> datetime:
        """
        Extract timestamp from message.
        
        Args:
            raw_message: Gmail API message object
            
        Returns:
            Datetime object representing message timestamp
        """
        # Try to get internal date first (more reliable)
        internal_date = raw_message.get('internalDate')
        if internal_date:
            try:
                # Internal date is in milliseconds since epoch
                timestamp_ms = int(internal_date)
                return datetime.fromtimestamp(timestamp_ms / 1000)
            except (ValueError, TypeError):
                pass
        
        # Fallback to Date header
        payload = raw_message.get('payload', {})
        headers = self._extract_headers(payload)
        date_header = headers.get('Date', '')
        
        if date_header:
            try:
                # Parse RFC 2822 date format
                return email.utils.parsedate_to_datetime(date_header)
            except (ValueError, TypeError):
                pass
        
        # Last resort: use current time
        return datetime.now()
    
    def _extract_body_content(self, payload: Dict[str, Any]) -> str:
        """
        Extract body content from message payload.
        
        Args:
            payload: Gmail API message payload
            
        Returns:
            Extracted and cleaned body text
        """
        mime_type = payload.get('mimeType', '')
        
        # Handle different MIME types
        if mime_type.startswith('text/'):
            return self._extract_text_content(payload)
        elif mime_type.startswith('multipart/'):
            return self._extract_multipart_content(payload)
        else:
            # Try to extract from parts if available
            parts = payload.get('parts', [])
            if parts:
                return self._extract_multipart_content(payload)
            else:
                # Try to get body data directly
                return self._extract_body_data(payload)
    
    def _extract_text_content(self, payload: Dict[str, Any]) -> str:
        """
        Extract content from text/* MIME type.
        
        Args:
            payload: Message payload with text content
            
        Returns:
            Extracted text content
        """
        mime_type = payload.get('mimeType', '')
        body_data = self._extract_body_data(payload)
        
        if mime_type == 'text/html':
            return self._html_to_text(body_data)
        else:
            return body_data
    
    def _extract_multipart_content(self, payload: Dict[str, Any]) -> str:
        """
        Extract content from multipart message.
        
        Args:
            payload: Message payload with multipart content
            
        Returns:
            Extracted and combined text content
        """
        parts = payload.get('parts', [])
        if not parts:
            return ''
        
        text_parts = []
        html_parts = []
        
        for part in parts:
            mime_type = part.get('mimeType', '')
            
            if mime_type == 'text/plain':
                content = self._extract_body_data(part)
                if content:
                    text_parts.append(content)
            elif mime_type == 'text/html':
                content = self._extract_body_data(part)
                if content:
                    html_parts.append(self._html_to_text(content))
            elif mime_type.startswith('multipart/'):
                # Recursively handle nested multipart
                nested_content = self._extract_multipart_content(part)
                if nested_content:
                    text_parts.append(nested_content)
        
        # Prefer plain text over HTML
        if text_parts:
            return '\n\n'.join(text_parts)
        elif html_parts:
            return '\n\n'.join(html_parts)
        else:
            return ''
    
    def _extract_body_data(self, payload: Dict[str, Any]) -> str:
        """
        Extract raw body data from payload.
        
        Args:
            payload: Message payload or part
            
        Returns:
            Decoded body text
        """
        body = payload.get('body', {})
        data = body.get('data', '')
        
        if not data:
            return ''
        
        try:
            # Gmail API returns base64url-encoded data
            decoded_bytes = base64.urlsafe_b64decode(data + '==')  # Add padding
            return self._decode_bytes(decoded_bytes)
        except Exception:
            return ''
    
    def _decode_bytes(self, data_bytes: bytes) -> str:
        """
        Decode bytes to string with encoding detection.
        
        Args:
            data_bytes: Raw bytes to decode
            
        Returns:
            Decoded string
        """
        if not data_bytes:
            return ''
        
        # Try to detect encoding
        try:
            detected = chardet.detect(data_bytes)
            if detected and detected.get('encoding'):
                encoding = detected['encoding']
                confidence = detected.get('confidence', 0)
                
                # Use detected encoding if confidence is high enough
                if confidence > 0.7:
                    try:
                        return data_bytes.decode(encoding)
                    except (UnicodeDecodeError, LookupError):
                        pass
        except Exception:
            pass
        
        # Fallback to common encodings
        for encoding in self.encoding_fallbacks:
            try:
                return data_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: decode with errors ignored
        return data_bytes.decode('utf-8', errors='ignore')
    
    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML content to clean plain text.
        
        Args:
            html_content: HTML content string
            
        Returns:
            Clean plain text
        """
        if not html_content:
            return ''
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean up whitespace
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception:
            # Fallback: simple HTML tag removal
            return self._simple_html_strip(html_content)
    
    def _simple_html_strip(self, html_content: str) -> str:
        """
        Simple HTML tag removal fallback.
        
        Args:
            html_content: HTML content string
            
        Returns:
            Text with HTML tags removed
        """
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', html_content)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_email(self, email_data: EmailData) -> bool:
        """
        Validate extracted email data.
        
        Args:
            email_data: EmailData object to validate
            
        Returns:
            True if email data is valid, False otherwise
        """
        if not email_data.id:
            return False
        
        # Must have either subject or body content
        if not email_data.subject and not email_data.body:
            return False
        
        # Basic timestamp validation
        if not email_data.timestamp:
            return False
        
        return True
    
    def _compile_patterns(self):
        """Compile regex patterns for sensitive information detection."""
        # Email addresses
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone numbers (various formats)
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # US format
            re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'),  # (123) 456-7890
            re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,14}\b'),  # International
            re.compile(r'\b\d{3}\s\d{3}\s\d{4}\b'),  # Space separated
            re.compile(r'\b\d{3}-\d{4}\b'),  # Simple format like 555-1234
        ]
        
        # Credit card numbers (basic pattern)
        self.credit_card_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
        
        # Social Security Numbers
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        
        # URLs (for optional removal)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # IP addresses
        self.ip_pattern = re.compile(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        )
    
    def _clean_and_anonymize_content(self, content: str) -> str:
        """
        Clean and anonymize content by removing or replacing sensitive information.
        
        Args:
            content: Text content to clean
            
        Returns:
            Cleaned and anonymized content
        """
        if not content:
            return content
        
        cleaned_content = content
        
        if self.remove_sensitive:
            # Remove or replace sensitive patterns
            cleaned_content = self._remove_sensitive_patterns(cleaned_content)
        
        if self.anonymize_content:
            # Anonymize remaining identifiable information
            cleaned_content = self._anonymize_identifiers(cleaned_content)
        
        # Additional content validation and cleaning
        cleaned_content = self._validate_and_clean_content(cleaned_content)
        
        return cleaned_content
    
    def _remove_sensitive_patterns(self, content: str) -> str:
        """
        Remove sensitive information patterns from content.
        
        Args:
            content: Text content to process
            
        Returns:
            Content with sensitive patterns removed
        """
        # Remove credit card numbers
        content = self.credit_card_pattern.sub('[CREDIT_CARD_REMOVED]', content)
        
        # Remove SSNs
        content = self.ssn_pattern.sub('[SSN_REMOVED]', content)
        
        # Remove phone numbers
        for pattern in self.phone_patterns:
            content = pattern.sub('[PHONE_REMOVED]', content)
        
        # Remove IP addresses
        content = self.ip_pattern.sub('[IP_REMOVED]', content)
        
        return content
    
    def _anonymize_identifiers(self, content: str) -> str:
        """
        Anonymize identifiable information in content.
        
        Args:
            content: Text content to anonymize
            
        Returns:
            Content with identifiers anonymized
        """
        # Anonymize email addresses (but keep domain structure for context)
        content = self.email_pattern.sub(self._anonymize_email_match, content)
        
        # Optionally anonymize URLs (keep for context but remove identifying info)
        content = self.url_pattern.sub('[URL]', content)
        
        return content
    
    def _anonymize_email_match(self, match) -> str:
        """
        Anonymize an email address match while preserving structure.
        
        Args:
            match: Regex match object for email address
            
        Returns:
            Anonymized email address
        """
        email_addr = match.group(0)
        try:
            local, domain = email_addr.split('@', 1)
            # Keep first character and domain, anonymize the rest
            if len(local) > 1:
                anonymized_local = local[0] + '*' * (len(local) - 1)
            else:
                anonymized_local = local[0]  # Keep single character as is
            
            # Keep domain structure but anonymize specific domain
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                # Keep TLD, anonymize domain name
                tld = domain_parts[-1]
                anonymized_domain = '***.' + tld
            else:
                anonymized_domain = '***'
            
            return f"{anonymized_local}@{anonymized_domain}"
        except ValueError:
            return '[EMAIL]'
    
    def _anonymize_email_address(self, email_addr: str) -> str:
        """
        Anonymize a single email address.
        
        Args:
            email_addr: Email address to anonymize
            
        Returns:
            Anonymized email address
        """
        if not email_addr or not self.anonymize_content:
            return email_addr
        
        return self.email_pattern.sub(self._anonymize_email_match, email_addr)
    
    def _validate_and_clean_content(self, content: str) -> str:
        """
        Validate and perform final cleaning of content.
        
        Args:
            content: Content to validate and clean
            
        Returns:
            Validated and cleaned content
        """
        if not content:
            return content
        
        # First, filter out problematic lines before whitespace normalization
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            # Skip extremely long lines (likely encoded data)
            if len(line) > 1000:
                continue
            # Skip lines that are mostly non-alphabetic (likely encoded/spam)
            # Only apply this filter to very long lines to avoid filtering normal content
            if len(line) > 200:  # Increased threshold
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / len(line)
                if alpha_ratio < 0.3:
                    continue
            cleaned_lines.append(line)
        
        # Rejoin and then normalize whitespace
        content = '\n'.join(cleaned_lines)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Final length check
        if len(content) > 10000:  # Truncate very long emails
            content = content[:10000] + '... [TRUNCATED]'
        
        return content
    
    def handle_corrupted_email(self, raw_message: Dict[str, Any], error: Exception) -> Optional[EmailData]:
        """
        Handle corrupted or problematic emails with error recovery.
        
        Args:
            raw_message: Original Gmail API message
            error: Exception that occurred during processing
            
        Returns:
            EmailData with minimal content or None if unrecoverable
        """
        try:
            message_id = raw_message.get('id', 'unknown')
            
            # Try to extract minimal information
            snippet = raw_message.get('snippet', '')
            if snippet:
                # Use snippet as fallback content
                return EmailData(
                    id=message_id,
                    subject='[CORRUPTED_EMAIL]',
                    body=snippet,
                    sender='[UNKNOWN]',
                    recipient='[UNKNOWN]',
                    timestamp=datetime.now(),
                    raw_content=f"Error: {str(error)}"
                )
        except Exception:
            pass
        
        return None