"""
Integration tests for recommendation endpoint

Tests the complete flow from request to response including:
- Request validation
- Cache integration
- Model inference
- Tool execution
- Response formatting
- Error scenarios
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status

from app.models.schemas import (
    UserProfile,
    GiftItem,
    GiftRecommendation,
    RecommendationRequest
)


class TestRecommendationEndpoint:
    """Integration tests for POST /api/recommendations endpoint"""
    
    @pytest.mark.asyncio
    async def test_successful_recommendation_flow(self, client, sample_user_profile):
        """
        Test successful end-to-end recommendation flow
        
        Validates: Requirements 1.1, 2.3, 6.1
        """
        # Mock services
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service mock
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            # Create mock recommendations
            mock_gift = GiftItem(
                id="12345",
                name="Test Gift",
                category="Test Category",
                price=299.99,
                rating=4.5,
                image_url="https://cdn.trendyol.com/test.jpg",
                trendyol_url="https://www.trendyol.com/product/12345",
                description="Test description",
                tags=["test"],
                age_suitability=(18, 100),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Good match", "Within budget"],
                tool_insights={},
                rank=1
            )
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {})
            )
            mock_model.return_value = model_service
            
            # Setup cache service mock (cache miss)
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            # Setup Trendyol service mock
            trendyol_service = MagicMock()
            
            # Create mock Trendyol product
            mock_product = MagicMock()
            mock_product.id = "12345"
            mock_product.name = "Test Gift"
            mock_product.category = "Test Category"
            mock_product.price = 299.99
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/test.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/12345"
            mock_product.description = "Test description"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(
                return_value=[mock_product]
            )
            trendyol_service.convert_to_gift_item = MagicMock(
                return_value=mock_gift
            )
            mock_trendyol.return_value = trendyol_service
            
            # Make request
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "recommendations" in data
            assert "tool_results" in data
            assert "inference_time" in data
            assert "cache_hit" in data
            
            assert len(data["recommendations"]) == 1
            assert data["cache_hit"] is False
            assert data["inference_time"] > 0
            
            # Verify recommendation structure
            rec = data["recommendations"][0]
            assert rec["gift"]["id"] == "12345"
            assert rec["confidence_score"] == 0.85
            assert rec["rank"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_hit_flow(self, client, sample_user_profile):
        """
        Test recommendation flow with cache hit
        
        Validates: Requirements 6.1
        """
        with patch('app.api.v1.recommendations.get_cache_service') as mock_cache:
            # Setup cache service mock (cache hit)
            cache_service = AsyncMock()
            
            # Create cached recommendation
            mock_gift = GiftItem(
                id="cached123",
                name="Cached Gift",
                category="Test Category",
                price=199.99,
                rating=4.0,
                image_url="https://cdn.trendyol.com/cached.jpg",
                trendyol_url="https://www.trendyol.com/product/cached123",
                description="Cached description",
                tags=["cached"],
                age_suitability=(18, 100),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            cached_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.90,
                reasoning=["Cached match"],
                tool_insights={},
                rank=1
            )
            
            cache_service.get_recommendations = AsyncMock(
                return_value=[cached_recommendation]
            )
            mock_cache.return_value = cache_service
            
            # Make request
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["cache_hit"] is True
            assert len(data["recommendations"]) == 1
            assert data["recommendations"][0]["gift"]["id"] == "cached123"
    
    @pytest.mark.asyncio
    async def test_model_not_loaded_error(self, client, sample_user_profile):
        """
        Test error handling when model is not loaded
        
        Validates: Requirements 2.3
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache:
            
            # Setup model service mock (not loaded)
            model_service = MagicMock()
            model_service.is_loaded.return_value = False
            mock_model.return_value = model_service
            
            # Setup cache service mock (cache miss)
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            mock_cache.return_value = cache_service
            
            # Make request
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "Model şu anda kullanılamıyor" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_trendyol_api_error(self, client, sample_user_profile):
        """
        Test error handling when Trendyol API fails
        
        Validates: Requirements 1.1
        """
        from app.core.exceptions import TrendyolAPIError
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service mock
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            mock_model.return_value = model_service
            
            # Setup cache service mock (cache miss)
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            mock_cache.return_value = cache_service
            
            # Setup Trendyol service mock (API error)
            trendyol_service = MagicMock()
            trendyol_service.search_products = AsyncMock(
                side_effect=TrendyolAPIError("API error")
            )
            mock_trendyol.return_value = trendyol_service
            
            # Make request
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "Ürün verileri şu anda alınamıyor" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_no_gifts_found(self, client, sample_user_profile):
        """
        Test handling when no gifts are found from Trendyol
        
        Validates: Requirements 1.1
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service mock
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            mock_model.return_value = model_service
            
            # Setup cache service mock (cache miss)
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            mock_cache.return_value = cache_service
            
            # Setup Trendyol service mock (no products)
            trendyol_service = MagicMock()
            trendyol_service.search_products = AsyncMock(return_value=[])
            mock_trendyol.return_value = trendyol_service
            
            # Make request
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data["recommendations"]) == 0
            assert "error" in data["tool_results"]
    
    def test_invalid_user_profile(self, client):
        """
        Test validation error for invalid user profile
        
        Validates: Requirements 1.2, 1.3, 1.4, 1.5
        """
        # Test invalid age (too young)
        request_data = {
            "user_profile": {
                "age": 10,  # Invalid: < 18
                "hobbies": ["reading"],
                "relationship": "friend",
                "budget": 100.0,
                "occasion": "birthday",
                "personality_traits": []
            },
            "max_recommendations": 5,
            "use_cache": True
        }
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test invalid budget (negative)
        request_data["user_profile"]["age"] = 25
        request_data["user_profile"]["budget"] = -100.0
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test empty hobbies
        request_data["user_profile"]["budget"] = 100.0
        request_data["user_profile"]["hobbies"] = []
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, client, sample_user_profile):
        """
        Test recommendation flow with cache disabled
        
        Validates: Requirements 6.1
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service mock
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="12345",
                name="Test Gift",
                category="Test Category",
                price=299.99,
                rating=4.5,
                image_url="https://cdn.trendyol.com/test.jpg",
                trendyol_url="https://www.trendyol.com/product/12345",
                description="Test description",
                tags=["test"],
                age_suitability=(18, 100),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Good match"],
                tool_insights={},
                rank=1
            )
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {})
            )
            mock_model.return_value = model_service
            
            # Setup Trendyol service mock
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "12345"
            mock_product.name = "Test Gift"
            mock_product.category = "Test Category"
            mock_product.price = 299.99
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/test.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/12345"
            mock_product.description = "Test description"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(
                return_value=[mock_product]
            )
            trendyol_service.convert_to_gift_item = MagicMock(
                return_value=mock_gift
            )
            mock_trendyol.return_value = trendyol_service
            
            # Make request with cache disabled
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["cache_hit"] is False
            assert len(data["recommendations"]) == 1
