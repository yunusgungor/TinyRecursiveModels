"""Unit tests for security features"""

import pytest
import time
from unittest.mock import Mock, patch
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.core.security import (
    EncryptionService,
    InputSanitizer,
    SecurityValidator,
    encryption_service,
    input_sanitizer,
    security_validator
)
from app.middleware.rate_limiter import RateLimiter, rate_limit_middleware
from app.middleware.session import SessionManager, session_middleware
from app.middleware.https_redirect import https_redirect_middleware


class TestEncryptionService:
    """Unit tests for encryption service"""
    
    def test_encrypt_decrypt_basic(self):
        """Test basic encryption and decryption"""
        service = EncryptionService()
        original = "sensitive data"
        
        encrypted = service.encrypt(original)
        assert encrypted != original
        
        decrypted = service.decrypt(encrypted)
        assert decrypted == original
    
    def test_encrypt_empty_string(self):
        """Test encryption of empty string"""
        service = EncryptionService()
        
        encrypted = service.encrypt("")
        assert encrypted == ""
        
        decrypted = service.decrypt("")
        assert decrypted == ""
    
    def test_encrypt_dict_specific_fields(self):
        """Test encrypting specific fields in dictionary"""
        service = EncryptionService()
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30
        }
        
        encrypted = service.encrypt_dict(data, ["name", "email"])
        
        # Encrypted fields should be different
        assert encrypted["name"] != data["name"]
        assert encrypted["email"] != data["email"]
        # Non-encrypted field should be same
        assert encrypted["age"] == data["age"]
        
        # Decrypt
        decrypted = service.decrypt_dict(encrypted, ["name", "email"])
        assert decrypted["name"] == data["name"]
        assert decrypted["email"] == data["email"]
    
    def test_global_encryption_service(self):
        """Test global encryption service instance"""
        original = "test data"
        encrypted = encryption_service.encrypt(original)
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == original


class TestInputSanitizer:
    """Unit tests for input sanitizer"""
    
    def test_sanitize_html_removes_script_tags(self):
        """Test that script tags are removed or escaped"""
        malicious = "<script>alert('xss')</script>Hello"
        sanitized = InputSanitizer.sanitize_html(malicious)
        
        # Script tags should be escaped or removed
        assert "<script>" not in sanitized.lower()
        # Should not contain executable script
        assert not SecurityValidator.contains_xss(sanitized)
    
    def test_sanitize_html_removes_javascript_protocol(self):
        """Test that javascript: protocol is removed"""
        malicious = "javascript:alert('xss')"
        sanitized = InputSanitizer.sanitize_html(malicious)
        
        assert "javascript:" not in sanitized.lower()
    
    def test_sanitize_html_removes_event_handlers(self):
        """Test that event handlers are removed"""
        malicious = "<img onerror='alert(1)' src='x'>"
        sanitized = InputSanitizer.sanitize_html(malicious)
        
        assert "onerror" not in sanitized.lower()
    
    def test_sanitize_sql_removes_union_select(self):
        """Test that SQL injection patterns are removed"""
        malicious = "1' UNION SELECT * FROM users--"
        sanitized = InputSanitizer.sanitize_sql(malicious)
        
        assert "UNION SELECT" not in sanitized.upper()
    
    def test_sanitize_sql_removes_drop_table(self):
        """Test that DROP TABLE is removed"""
        malicious = "'; DROP TABLE users; --"
        sanitized = InputSanitizer.sanitize_sql(malicious)
        
        assert "DROP TABLE" not in sanitized.upper()
    
    def test_sanitize_input_combines_both(self):
        """Test that sanitize_input applies both XSS and SQL sanitization"""
        malicious = "<script>alert('xss')</script> UNION SELECT"
        sanitized = InputSanitizer.sanitize_input(malicious)
        
        assert "<script" not in sanitized.lower()
        assert "UNION SELECT" not in sanitized.upper()
    
    def test_sanitize_dict_all_string_fields(self):
        """Test sanitizing all string fields in dictionary"""
        data = {
            "name": "<script>alert('xss')</script>",
            "comment": "Normal text",
            "age": 30
        }
        
        sanitized = InputSanitizer.sanitize_dict(data)
        
        assert "<script" not in sanitized["name"].lower()
        assert sanitized["comment"] == "Normal text"
        assert sanitized["age"] == 30
    
    def test_global_input_sanitizer(self):
        """Test global input sanitizer instance"""
        malicious = "<script>alert('xss')</script>"
        sanitized = input_sanitizer.sanitize_html(malicious)
        assert "<script" not in sanitized.lower()


class TestSecurityValidator:
    """Unit tests for security validator"""
    
    def test_is_safe_url_accepts_https(self):
        """Test that HTTPS URLs are accepted"""
        assert SecurityValidator.is_safe_url("https://example.com")
        assert SecurityValidator.is_safe_url("https://trendyol.com/product/123")
    
    def test_is_safe_url_accepts_http(self):
        """Test that HTTP URLs are accepted"""
        assert SecurityValidator.is_safe_url("http://example.com")
    
    def test_is_safe_url_rejects_javascript(self):
        """Test that javascript: URLs are rejected"""
        assert not SecurityValidator.is_safe_url("javascript:alert('xss')")
    
    def test_is_safe_url_rejects_data(self):
        """Test that data: URLs are rejected"""
        assert not SecurityValidator.is_safe_url("data:text/html,<script>alert('xss')</script>")
    
    def test_is_safe_url_rejects_file(self):
        """Test that file: URLs are rejected"""
        assert not SecurityValidator.is_safe_url("file:///etc/passwd")
    
    def test_contains_xss_detects_script_tags(self):
        """Test XSS detection for script tags"""
        assert SecurityValidator.contains_xss("<script>alert('xss')</script>")
    
    def test_contains_xss_detects_javascript_protocol(self):
        """Test XSS detection for javascript: protocol"""
        assert SecurityValidator.contains_xss("javascript:alert('xss')")
    
    def test_contains_xss_detects_event_handlers(self):
        """Test XSS detection for event handlers"""
        assert SecurityValidator.contains_xss("<img onerror='alert(1)' src='x'>")
    
    def test_contains_sql_injection_detects_union(self):
        """Test SQL injection detection for UNION"""
        assert SecurityValidator.contains_sql_injection("1' UNION SELECT * FROM users")
    
    def test_contains_sql_injection_detects_drop(self):
        """Test SQL injection detection for DROP"""
        assert SecurityValidator.contains_sql_injection("'; DROP TABLE users; --")
    
    def test_global_security_validator(self):
        """Test global security validator instance"""
        assert security_validator.is_safe_url("https://example.com")
        assert not security_validator.is_safe_url("javascript:alert('xss')")


class TestRateLimiter:
    """Unit tests for rate limiter"""
    
    def test_rate_limiter_allows_within_limit(self):
        """Test that requests within limit are allowed"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for i in range(5):
            is_allowed, remaining = limiter.is_allowed("client1")
            assert is_allowed
            assert remaining == 4 - i
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # First 3 requests should be allowed
        for _ in range(3):
            is_allowed, _ = limiter.is_allowed("client1")
            assert is_allowed
        
        # 4th request should be blocked
        is_allowed, remaining = limiter.is_allowed("client1")
        assert not is_allowed
        assert remaining == 0
    
    def test_rate_limiter_resets_after_window(self):
        """Test that rate limit resets after time window"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Use up the limit
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        # Should be blocked
        is_allowed, _ = limiter.is_allowed("client1")
        assert not is_allowed
        
        # Wait for window to pass
        time.sleep(1.1)
        
        # Should be allowed again
        is_allowed, _ = limiter.is_allowed("client1")
        assert is_allowed
    
    def test_rate_limiter_separate_clients(self):
        """Test that different clients have separate limits"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Client 1 uses up limit
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        # Client 2 should still be allowed
        is_allowed, _ = limiter.is_allowed("client2")
        assert is_allowed
    
    def test_get_reset_time(self):
        """Test getting reset time"""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        limiter.is_allowed("client1")
        reset_time = limiter.get_reset_time("client1")
        
        # Should be approximately 60 seconds
        assert 55 <= reset_time <= 60


class TestSessionManager:
    """Unit tests for session manager"""
    
    def test_create_session(self):
        """Test creating a new session"""
        manager = SessionManager(timeout_minutes=30)
        
        session_id = manager.create_session("client1")
        assert session_id is not None
        assert len(session_id) > 0
    
    def test_get_session(self):
        """Test getting an existing session"""
        manager = SessionManager(timeout_minutes=30)
        
        session_id = manager.create_session("client1")
        session = manager.get_session(session_id)
        
        assert session is not None
        assert session["client_id"] == "client1"
    
    def test_get_nonexistent_session(self):
        """Test getting a non-existent session"""
        manager = SessionManager(timeout_minutes=30)
        
        session = manager.get_session("nonexistent")
        assert session is None
    
    def test_session_expires(self):
        """Test that sessions expire after timeout"""
        manager = SessionManager(timeout_minutes=0.01)  # 0.6 seconds
        
        session_id = manager.create_session("client1")
        
        # Should exist initially
        session = manager.get_session(session_id)
        assert session is not None
        
        # Wait for expiration
        time.sleep(1)
        
        # Should be expired
        session = manager.get_session(session_id)
        assert session is None
    
    def test_update_activity(self):
        """Test updating session activity"""
        manager = SessionManager(timeout_minutes=30)
        
        session_id = manager.create_session("client1")
        
        # Update activity
        result = manager.update_activity(session_id)
        assert result is True
        
        # Session should still exist
        session = manager.get_session(session_id)
        assert session is not None
    
    def test_delete_session(self):
        """Test deleting a session"""
        manager = SessionManager(timeout_minutes=30)
        
        session_id = manager.create_session("client1")
        
        # Delete session
        result = manager.delete_session(session_id)
        assert result is True
        
        # Session should not exist
        session = manager.get_session(session_id)
        assert session is None
    
    def test_cleanup_expired_sessions(self):
        """Test cleaning up expired sessions"""
        manager = SessionManager(timeout_minutes=0.01)
        
        # Create multiple sessions
        session_id1 = manager.create_session("client1")
        session_id2 = manager.create_session("client2")
        
        # Wait for expiration
        time.sleep(1)
        
        # Cleanup
        manager.cleanup_expired_sessions()
        
        # Both should be gone
        assert manager.get_session(session_id1) is None
        assert manager.get_session(session_id2) is None


@pytest.mark.asyncio
class TestMiddlewares:
    """Unit tests for security middlewares"""
    
    async def test_rate_limit_middleware_allows_within_limit(self):
        """Test that rate limit middleware allows requests within limit"""
        # Create mock request and response
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/test"
        request.state = Mock()
        request.state.request_id = "test-id"
        
        async def call_next(req):
            response = Mock(spec=Response)
            response.headers = {}
            return response
        
        # Should allow request
        response = await rate_limit_middleware(request, call_next)
        assert response is not None
        assert "X-RateLimit-Limit" in response.headers
    
    async def test_session_middleware_creates_session(self):
        """Test that session middleware creates session"""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.cookies = {}
        request.headers = {}
        request.state = Mock()
        
        async def call_next(req):
            response = Mock(spec=Response)
            response.set_cookie = Mock()
            return response
        
        response = await session_middleware(request, call_next)
        assert hasattr(request.state, "session_id")
        assert response.set_cookie.called
