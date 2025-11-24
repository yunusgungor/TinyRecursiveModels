"""Property-based tests for security features"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings

from app.core.security import (
    EncryptionService,
    InputSanitizer,
    SecurityValidator
)


class TestEncryptionProperties:
    """Property tests for encryption service"""
    
    @given(
        data=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_encryption_round_trip(self, data: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any string data, encrypting and then decrypting should produce the original value
        """
        encryption_service = EncryptionService()
        
        # Encrypt
        encrypted = encryption_service.encrypt(data)
        
        # Encrypted data should be different from original
        assert encrypted != data
        
        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        
        # Should match original
        assert decrypted == data
    
    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(min_size=1, max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False)
            ),
            min_size=1,
            max_size=10
        ),
        fields_to_encrypt=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=5
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dict_encryption_preserves_structure(self, data: dict, fields_to_encrypt: list):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any dictionary, encrypting specific fields should preserve dictionary structure
        and allow round-trip decryption
        """
        encryption_service = EncryptionService()
        
        # Only encrypt fields that exist in the dictionary
        valid_fields = [f for f in fields_to_encrypt if f in data]
        
        if not valid_fields:
            # Skip if no valid fields
            return
        
        # Encrypt
        encrypted_data = encryption_service.encrypt_dict(data, valid_fields)
        
        # Structure should be preserved
        assert set(encrypted_data.keys()) == set(data.keys())
        
        # Encrypted fields should be different
        for field in valid_fields:
            if isinstance(data[field], str) and data[field]:
                assert encrypted_data[field] != data[field]
        
        # Decrypt
        decrypted_data = encryption_service.decrypt_dict(encrypted_data, valid_fields)
        
        # Should match original for encrypted fields
        for field in valid_fields:
            assert str(decrypted_data[field]) == str(data[field])
    
    @given(
        data=st.text(min_size=0, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_encryption_handles_empty_strings(self, data: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any string including empty strings, encryption should handle gracefully
        """
        encryption_service = EncryptionService()
        
        # Encrypt
        encrypted = encryption_service.encrypt(data)
        
        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        
        # Should match original
        assert decrypted == data


class TestInputSanitizationProperties:
    """Property tests for input sanitization"""
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitized_input_contains_no_xss(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain XSS patterns
        """
        sanitized = InputSanitizer.sanitize_html(text)
        
        # Check that dangerous patterns are removed
        assert not SecurityValidator.contains_xss(sanitized)
        
        # Should not contain script tags
        assert '<script' not in sanitized.lower()
        assert 'javascript:' not in sanitized.lower()
        assert 'onerror=' not in sanitized.lower()
        assert 'onclick=' not in sanitized.lower()
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitized_input_contains_no_sql_injection(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain SQL injection patterns
        """
        sanitized = InputSanitizer.sanitize_sql(text)
        
        # Check that SQL injection patterns are removed or neutralized
        # Note: This is a best-effort check, not foolproof
        dangerous_keywords = ['UNION SELECT', 'DROP TABLE', 'DELETE FROM', 'INSERT INTO']
        
        for keyword in dangerous_keywords:
            # Should not contain these patterns in uppercase
            assert keyword not in sanitized.upper()
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitize_preserves_safe_text(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain dangerous patterns
        """
        sanitized = InputSanitizer.sanitize_input(text)
        
        # Sanitized text should not contain XSS or SQL injection
        assert not SecurityValidator.contains_xss(sanitized)
        
        # Should not be empty unless original was empty
        if text.strip():
            assert len(sanitized) > 0
    
    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            values=st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=10
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dict_sanitization_preserves_keys(self, data: dict):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any dictionary, sanitization should preserve all keys
        """
        sanitized = InputSanitizer.sanitize_dict(data)
        
        # All keys should be preserved
        assert set(sanitized.keys()) == set(data.keys())
        
        # All values should be sanitized (no XSS)
        for value in sanitized.values():
            if isinstance(value, str):
                assert not SecurityValidator.contains_xss(value)


class TestSecurityValidatorProperties:
    """Property tests for security validator"""
    
    @given(
        url=st.one_of(
            st.just("http://example.com"),
            st.just("https://example.com"),
            st.just("https://trendyol.com/product/123"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_safe_urls_are_accepted(self, url: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any safe HTTP/HTTPS URL, validator should accept it
        """
        assert SecurityValidator.is_safe_url(url)
    
    @given(
        url=st.one_of(
            st.just("javascript:alert('xss')"),
            st.just("data:text/html,<script>alert('xss')</script>"),
            st.just("vbscript:msgbox('xss')"),
            st.just("file:///etc/passwd"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dangerous_urls_are_rejected(self, url: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any dangerous URL (javascript:, data:, etc.), validator should reject it
        """
        assert not SecurityValidator.is_safe_url(url)
    
    @given(
        text=st.one_of(
            st.just("<script>alert('xss')</script>"),
            st.just("javascript:alert('xss')"),
            st.just("<img onerror='alert(1)' src='x'>"),
            st.just("<iframe src='evil.com'></iframe>"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_xss_patterns_are_detected(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any text containing XSS patterns, validator should detect them
        """
        assert SecurityValidator.contains_xss(text)
