"""Unit tests for error handling and logging"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.core.exceptions import (
    BaseAPIException,
    ModelInferenceError,
    ModelLoadError,
    TrendyolAPIError,
    ValidationError
)
from app.services.alert_service import AlertService


class TestErrorMessages:
    """Test error message generation"""
    
    def test_model_inference_error_default_message(self):
        """Test ModelInferenceError has correct default message"""
        error = ModelInferenceError()
        assert error.message == "Model şu anda kullanılamıyor"
        assert error.error_code == "MODEL_INFERENCE_ERROR"
    
    def test_model_inference_error_custom_message(self):
        """Test ModelInferenceError with custom message"""
        custom_msg = "Custom error message"
        error = ModelInferenceError(message=custom_msg)
        assert error.message == custom_msg
        assert error.error_code == "MODEL_INFERENCE_ERROR"
    
    def test_trendyol_api_error_default_message(self):
        """Test TrendyolAPIError has correct default message"""
        error = TrendyolAPIError()
        assert error.message == "Ürün verileri şu anda alınamıyor"
        assert error.error_code == "TRENDYOL_API_ERROR"
    
    def test_validation_error_field_identification(self):
        """Test ValidationError identifies field correctly"""
        field_name = "age"
        error_msg = "Age must be between 18 and 100"
        error = ValidationError(message=error_msg, field=field_name)
        
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details["field"] == field_name
        assert error.message == error_msg


class TestAlertService:
    """Test email alert service"""
    
    @pytest.fixture
    def alert_service(self):
        """Create alert service instance"""
        return AlertService()
    
    @pytest.mark.asyncio
    async def test_alert_disabled_by_default(self, alert_service):
        """Test that alerts are disabled by default"""
        result = await alert_service.send_critical_error_alert(
            error_code="TEST_ERROR",
            error_message="Test message"
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_alert_with_incomplete_config(self, alert_service):
        """Test that alert fails with incomplete configuration"""
        alert_service.enabled = True
        alert_service.smtp_user = ""  # Missing config
        
        result = await alert_service.send_critical_error_alert(
            error_code="TEST_ERROR",
            error_message="Test message"
        )
        assert result is False
    
    @pytest.mark.asyncio
    @patch('smtplib.SMTP')
    async def test_alert_sends_email_successfully(self, mock_smtp, alert_service):
        """Test successful email sending"""
        # Configure alert service
        alert_service.enabled = True
        alert_service.smtp_user = "test@example.com"
        alert_service.smtp_password = "password"
        alert_service.alert_email_to = "admin@example.com"
        alert_service.alert_email_from = "alerts@example.com"
        
        # Mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        result = await alert_service.send_critical_error_alert(
            error_code="CRITICAL_ERROR",
            error_message="Critical error occurred",
            details={"key": "value"},
            request_id="test-123"
        )
        
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@example.com", "password")
        mock_server.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('smtplib.SMTP')
    async def test_alert_handles_smtp_failure(self, mock_smtp, alert_service):
        """Test alert handles SMTP failures gracefully"""
        # Configure alert service
        alert_service.enabled = True
        alert_service.smtp_user = "test@example.com"
        alert_service.smtp_password = "password"
        alert_service.alert_email_to = "admin@example.com"
        alert_service.alert_email_from = "alerts@example.com"
        
        # Mock SMTP to raise exception
        mock_smtp.side_effect = Exception("SMTP connection failed")
        
        result = await alert_service.send_critical_error_alert(
            error_code="TEST_ERROR",
            error_message="Test message"
        )
        
        assert result is False
    
    def test_email_body_formatting(self, alert_service):
        """Test email body is formatted correctly"""
        error_code = "TEST_ERROR"
        error_message = "Test error message"
        details = {"key1": "value1", "key2": "value2"}
        request_id = "req-123"
        
        body = alert_service._build_email_body(
            error_code, error_message, details, request_id
        )
        
        assert error_code in body
        assert error_message in body
        assert request_id in body
        assert "key1" in body
        assert "value1" in body
        assert "html" in body.lower()
    
    def test_details_formatting_with_none(self, alert_service):
        """Test details formatting handles None"""
        formatted = alert_service._format_details(None)
        assert formatted == "No additional details"
    
    def test_details_formatting_with_dict(self, alert_service):
        """Test details formatting with dictionary"""
        details = {"error": "test", "code": 500}
        formatted = alert_service._format_details(details)
        
        assert "error: test" in formatted
        assert "code: 500" in formatted


class TestLogRotation:
    """Test log rotation configuration"""
    
    def test_log_rotation_config_exists(self):
        """Test that log rotation is configured"""
        from app.core.config import settings
        
        assert hasattr(settings, 'LOG_MAX_SIZE_MB')
        assert hasattr(settings, 'LOG_BACKUP_COUNT')
        assert settings.LOG_MAX_SIZE_MB == 100
        assert settings.LOG_BACKUP_COUNT == 5
    
    def test_logging_setup_creates_rotating_handler(self):
        """Test that logging setup creates rotating file handler"""
        from app.core.logging import logger
        from logging.handlers import RotatingFileHandler
        
        # Check if logger has rotating file handler
        has_rotating_handler = any(
            isinstance(handler, RotatingFileHandler)
            for handler in logger.handlers
        )
        
        assert has_rotating_handler, "Logger should have RotatingFileHandler"
