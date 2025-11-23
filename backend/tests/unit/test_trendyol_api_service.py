"""Unit tests for Trendyol API service"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import httpx

from app.services.trendyol_api import (
    TrendyolAPIService,
    TrendyolProduct,
    RateLimiter
)
from app.core.exceptions import TrendyolAPIError, RateLimitError


class TestRateLimiter:
    """Tests for rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Rate limiter should allow requests within the limit"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Should allow 5 requests
        for _ in range(5):
            await limiter.acquire()
        
        # Check remaining
        assert limiter.get_remaining_requests() == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_limit_exceeded(self):
        """Rate limiter should block when limit is exceeded"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Make 2 requests
        await limiter.acquire()
        await limiter.acquire()
        
        # Check that we're at the limit
        assert limiter.get_remaining_requests() == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_resets_after_window(self):
        """Rate limiter should reset after time window"""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Make 2 requests
        await limiter.acquire()
        await limiter.acquire()
        
        # Manually manipulate the request times to simulate time passing
        import time
        current_time = time.time()
        limiter.requests = [current_time - 2.0, current_time - 2.0]  # 2 seconds ago
        
        # Should be able to make requests again
        remaining = limiter.get_remaining_requests()
        assert remaining == 2


class TestTrendyolAPIService:
    """Tests for Trendyol API service"""
    
    @pytest.fixture
    def service(self):
        """Create a TrendyolAPIService instance"""
        return TrendyolAPIService(
            api_key="test_key",
            base_url="https://api.test.com",
            rate_limit=100
        )
    
    @pytest.fixture
    def mock_product_data(self):
        """Mock product data"""
        return {
            "id": "12345",
            "name": "Test Product",
            "category": {"name": "Electronics"},
            "price": {"sellingPrice": 299.99},
            "rating": {"averageRating": 4.5, "totalCount": 100},
            "images": [{"url": "https://cdn.trendyol.com/test.jpg"}],
            "url": "https://www.trendyol.com/product/12345",
            "description": "Test description",
            "brand": {"name": "TestBrand"},
            "inStock": True
        }
    
    @pytest.mark.asyncio
    async def test_search_products_success(self, service, mock_product_data):
        """Test successful product search"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"products": [mock_product_data]}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # Search products
            products = await service.search_products(
                category="Electronics",
                keywords=["laptop"],
                max_results=10
            )
            
            # Verify results
            assert len(products) == 1
            assert isinstance(products[0], TrendyolProduct)
            assert products[0].id == "12345"
            assert products[0].name == "Test Product"
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_search_products_uses_cache(self, service, mock_product_data):
        """Test that search uses cache for repeated requests"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"products": [mock_product_data]}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # First search
            products1 = await service.search_products(
                category="Electronics",
                keywords=["laptop"],
                max_results=10
            )
            
            # Second search (should use cache)
            products2 = await service.search_products(
                category="Electronics",
                keywords=["laptop"],
                max_results=10
            )
            
            # API should only be called once
            assert mock_get.call_count == 1
            
            # Results should be the same
            assert len(products1) == len(products2)
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_search_products_fallback_on_error(self, service, mock_product_data):
        """Test fallback to cached data when API fails"""
        # First, populate cache with successful request
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = {"products": [mock_product_data]}
        mock_response_success.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response_success
            
            # First search (populates cache)
            products1 = await service.search_products(
                category="Electronics",
                keywords=["laptop"],
                max_results=10
            )
            
            assert len(products1) == 1
        
        # Now simulate API failure
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")
            
            # Second search (should use stale cache as fallback)
            products2 = await service.search_products(
                category="Electronics",
                keywords=["laptop"],
                max_results=10
            )
            
            # Should still get results from cache
            assert len(products2) == 1
            assert products2[0].id == "12345"
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_search_products_raises_error_without_cache(self, service):
        """Test that error is raised when API fails and no cache exists"""
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")
            
            # Should raise TrendyolAPIError
            with pytest.raises(TrendyolAPIError):
                await service.search_products(
                    category="Electronics",
                    keywords=["laptop"],
                    max_results=10
                )
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_get_product_details_success(self, service, mock_product_data):
        """Test successful product details retrieval"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_product_data
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # Get product details
            product = await service.get_product_details("12345")
            
            # Verify result
            assert isinstance(product, TrendyolProduct)
            assert product.id == "12345"
            assert product.name == "Test Product"
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_get_product_details_uses_cache(self, service, mock_product_data):
        """Test that product details use cache"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_product_data
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # First call
            product1 = await service.get_product_details("12345")
            
            # Second call (should use cache)
            product2 = await service.get_product_details("12345")
            
            # API should only be called once
            assert mock_get.call_count == 1
            
            # Results should be the same
            assert product1.id == product2.id
        
        await service.close()
    
    @pytest.mark.asyncio
    async def test_get_product_details_fallback_on_error(self, service, mock_product_data):
        """Test fallback to cached data when API fails"""
        # First, populate cache
        mock_response_success = MagicMock()
        mock_response_success.json.return_value = mock_product_data
        mock_response_success.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response_success
            
            # First call (populates cache)
            product1 = await service.get_product_details("12345")
            assert product1.id == "12345"
        
        # Now simulate API failure
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")
            
            # Second call (should use stale cache as fallback)
            product2 = await service.get_product_details("12345")
            
            # Should still get result from cache
            assert product2.id == "12345"
        
        await service.close()
    
    def test_convert_to_gift_item_success(self, service, mock_product_data):
        """Test successful conversion to GiftItem"""
        product = TrendyolProduct(mock_product_data)
        gift_item = service.convert_to_gift_item(product)
        
        # Should produce a valid GiftItem
        assert gift_item is not None
        assert gift_item.id == "12345"
        assert gift_item.name == "Test Product"
        assert gift_item.category == "Electronics"
        assert gift_item.price == 299.99
        assert gift_item.rating == 4.5
    
    def test_convert_to_gift_item_with_invalid_url(self, service):
        """Test conversion fails with invalid URL"""
        invalid_data = {
            "id": "12345",
            "name": "Test Product",
            "category": {"name": "Electronics"},
            "price": {"sellingPrice": 299.99},
            "rating": {"averageRating": 4.5, "totalCount": 100},
            "images": [{"url": "invalid-url"}],
            "url": "https://www.trendyol.com/product/12345",
            "description": "Test description",
            "brand": {"name": "TestBrand"},
            "inStock": True
        }
        
        product = TrendyolProduct(invalid_data)
        gift_item = service.convert_to_gift_item(product)
        
        # Should return None due to invalid image URL
        assert gift_item is None
    
    def test_validate_url_valid_http(self, service):
        """Test URL validation with valid HTTP URL"""
        url = "http://example.com/path"
        result = service._validate_url(url)
        
        assert result is not None
        assert result.startswith("http://")
    
    def test_validate_url_valid_https(self, service):
        """Test URL validation with valid HTTPS URL"""
        url = "https://example.com/path"
        result = service._validate_url(url)
        
        assert result is not None
        assert result.startswith("https://")
    
    def test_validate_url_invalid_scheme(self, service):
        """Test URL validation with invalid scheme"""
        url = "ftp://example.com/path"
        result = service._validate_url(url)
        
        assert result is None
    
    def test_validate_url_empty(self, service):
        """Test URL validation with empty string"""
        result = service._validate_url("")
        
        assert result is None
    
    def test_validate_url_no_scheme(self, service):
        """Test URL validation with no scheme"""
        url = "example.com/path"
        result = service._validate_url(url)
        
        assert result is None
    
    def test_normalize_price_positive(self, service):
        """Test price normalization with positive value"""
        price = 123.456
        result = service._normalize_price(price)
        
        assert result == 123.46
    
    def test_normalize_price_negative(self, service):
        """Test price normalization with negative value"""
        price = -50.0
        result = service._normalize_price(price)
        
        # Should be clamped to 0
        assert result == 0.0
    
    def test_normalize_price_zero(self, service):
        """Test price normalization with zero"""
        price = 0.0
        result = service._normalize_price(price)
        
        assert result == 0.0
    
    def test_normalize_price_rounding(self, service):
        """Test price normalization rounds correctly"""
        price = 99.999
        result = service._normalize_price(price)
        
        assert result == 100.0
    
    def test_get_rate_limit_info(self, service):
        """Test getting rate limit information"""
        info = service.get_rate_limit_info()
        
        assert "max_requests" in info
        assert "window_seconds" in info
        assert "remaining_requests" in info
        assert info["max_requests"] == 100
        assert info["window_seconds"] == 60
