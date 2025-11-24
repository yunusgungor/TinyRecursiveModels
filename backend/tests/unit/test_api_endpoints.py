"""Unit tests for API endpoints"""

import pytest
from fastapi import status
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_trendyol_service(sample_gift_item):
    """Mock Trendyol API service for testing"""
    from app.models.schemas import GiftItem
    
    with patch('app.api.v1.recommendations.get_trendyol_service') as mock:
        service_instance = MagicMock()
        
        # Mock search_products to return raw product data
        service_instance.search_products = AsyncMock(return_value=[
            {
                "id": "12345",
                "name": "Premium Coffee Set",
                "category": "Kitchen & Dining",
                "price": 299.99,
                "rating": 4.5,
                "image_url": "https://cdn.trendyol.com/test.jpg",
                "trendyol_url": "https://www.trendyol.com/product/12345",
                "description": "Test description",
                "tags": ["coffee", "kitchen"],
                "age_suitability": (25, 65),
                "occasion_fit": ["birthday"],
                "in_stock": True
            }
        ])
        
        # Mock convert_to_gift_item to return GiftItem objects
        def convert_mock(product):
            return GiftItem(**sample_gift_item)
        
        service_instance.convert_to_gift_item = MagicMock(side_effect=convert_mock)
        
        mock.return_value = service_instance
        yield service_instance


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200 OK"""
        response = client.get("/api/health")
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_check_response_structure(self, client):
        """Test health endpoint returns correct structure"""
        response = client.get("/api/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "trendyol_api_status" in data
        assert "cache_status" in data
        assert "timestamp" in data
    
    def test_health_check_status_value(self, client):
        """Test health endpoint returns status (healthy or unhealthy)"""
        response = client.get("/api/health")
        data = response.json()
        # Status can be healthy or unhealthy depending on model availability
        assert data["status"] in ["healthy", "unhealthy"]


class TestRecommendationsEndpoint:
    """Test recommendations endpoint"""
    
    def test_recommendations_endpoint_exists(self, client, sample_user_profile, mock_trendyol_service, mock_model_service):
        """Test recommendations endpoint is accessible"""
        response = client.post(
            "/api/recommendations",
            json={
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
        )
        # Should return 200 even with empty implementation
        assert response.status_code == status.HTTP_200_OK
    
    def test_recommendations_response_structure(self, client, sample_user_profile, mock_trendyol_service, mock_model_service):
        """Test recommendations endpoint returns correct structure"""
        response = client.post(
            "/api/recommendations",
            json={
                "user_profile": sample_user_profile,
                "max_recommendations": 5
            }
        )
        data = response.json()
        
        assert "recommendations" in data
        assert "tool_results" in data
        assert "inference_time" in data
        assert "cache_hit" in data
    
    def test_recommendations_invalid_age(self, client, sample_user_profile):
        """Test recommendations endpoint validates age"""
        sample_user_profile["age"] = 15  # Invalid age
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommendations_invalid_budget(self, client, sample_user_profile):
        """Test recommendations endpoint validates budget"""
        sample_user_profile["budget"] = -100  # Invalid budget
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommendations_empty_hobbies(self, client, sample_user_profile):
        """Test recommendations endpoint validates hobbies"""
        sample_user_profile["hobbies"] = []  # Empty hobbies
        response = client.post(
            "/api/recommendations",
            json={"user_profile": sample_user_profile}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestToolsStatsEndpoint:
    """Test tools statistics endpoint"""
    
    def test_tools_stats_returns_200(self, client):
        """Test tools stats endpoint returns 200 OK"""
        response = client.get("/api/tools/stats")
        assert response.status_code == status.HTTP_200_OK
    
    def test_tools_stats_response_structure(self, client):
        """Test tools stats endpoint returns correct structure"""
        response = client.get("/api/tools/stats")
        data = response.json()
        
        assert "tool_usage" in data
        assert "success_rates" in data
        assert "average_execution_times" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200 OK"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
    
    def test_root_response_structure(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data
