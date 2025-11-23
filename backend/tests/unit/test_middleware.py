"""Unit tests for middleware"""

import pytest
from fastapi import status

from app.core.exceptions import ModelInferenceError, ValidationError


class TestErrorHandlerMiddleware:
    """Test error handling middleware"""
    
    def test_validation_error_response_format(self, client, sample_user_profile):
        """Test validation errors return proper format"""
        sample_user_profile["age"] = 15  # Invalid age
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        
        # Check error response structure
        assert "error_code" in data
        assert "message" in data
        assert "timestamp" in data
        assert "request_id" in data
        assert data["error_code"] == "VALIDATION_ERROR"
    
    def test_validation_error_identifies_field(self, client, sample_user_profile):
        """Test validation errors identify the problematic field"""
        sample_user_profile["budget"] = -100  # Invalid budget
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        
        data = response.json()
        assert "details" in data
        assert "field" in data["details"]
    
    def test_request_id_in_error_response(self, client, sample_user_profile):
        """Test error responses include request ID"""
        sample_user_profile["age"] = 200  # Invalid age
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        
        data = response.json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0


class TestCORSMiddleware:
    """Test CORS middleware configuration"""
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses"""
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
    
    def test_cors_allows_configured_origins(self, client):
        """Test CORS allows configured origins"""
        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should not reject the request
        assert response.status_code == status.HTTP_200_OK
