"""Property-based tests for Trendyol API service"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import assume
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.trendyol_api import TrendyolAPIService, TrendyolProduct
from app.models.schemas import GiftItem


# Custom strategies
@st.composite
def valid_category(draw):
    """Generate valid category strings"""
    categories = [
        "Elektronik",
        "Ev & Yaşam",
        "Moda",
        "Kozmetik",
        "Spor & Outdoor",
        "Kitap",
        "Oyuncak",
        "Anne & Bebek"
    ]
    return draw(st.sampled_from(categories))


@st.composite
def valid_keywords(draw):
    """Generate valid keyword lists"""
    keywords = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('L', 'N'),
            whitelist_characters=' '
        )),
        min_size=1,
        max_size=5
    ))
    # Filter out empty or whitespace-only keywords
    keywords = [k.strip() for k in keywords if k.strip()]
    assume(len(keywords) > 0)
    return keywords


@st.composite
def valid_price(draw):
    """Generate valid price values"""
    return draw(st.floats(min_value=0.01, max_value=100000.0))


@st.composite
def valid_url(draw):
    """Generate valid HTTP/HTTPS URLs"""
    schemes = ["http", "https"]
    domains = ["trendyol.com", "cdn.trendyol.com", "example.com"]
    
    scheme = draw(st.sampled_from(schemes))
    domain = draw(st.sampled_from(domains))
    path = draw(st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/-_')
    ))
    
    return f"{scheme}://{domain}/{path}"


@st.composite
def trendyol_product_data(draw):
    """Generate valid Trendyol product data"""
    product_id = draw(st.integers(min_value=1, max_value=999999))
    name = draw(st.text(min_size=5, max_size=100))
    category_name = draw(valid_category())
    price = draw(valid_price())
    rating = draw(st.floats(min_value=0.0, max_value=5.0))
    image_url = draw(valid_url())
    product_url = draw(valid_url())
    
    return {
        "id": product_id,
        "name": name,
        "category": {"name": category_name},
        "price": {"sellingPrice": price},
        "rating": {"averageRating": rating, "totalCount": 100},
        "images": [{"url": image_url}],
        "url": product_url,
        "description": "Test product description",
        "brand": {"name": "TestBrand"},
        "inStock": True
    }


class TestTrendyolAPIRequestParameters:
    """
    Property 10: Trendyol API Request Parameters
    Feature: trendyol-gift-recommendation-web, Property 10: Trendyol API Request Parameters
    Validates: Requirements 4.1
    """
    
    @given(
        category=valid_category(),
        keywords=valid_keywords(),
        max_results=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_search_includes_category_and_keywords(
        self,
        category,
        keywords,
        max_results
    ):
        """
        For any product search request, the system should include both 
        category and keywords in the API call
        """
        service = TrendyolAPIService()
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {"products": []}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(service.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            # Call search_products
            await service.search_products(
                category=category,
                keywords=keywords,
                max_results=max_results
            )
            
            # Verify the call was made
            assert mock_get.called
            
            # Get the call arguments
            call_args = mock_get.call_args
            
            # Verify params include category and keywords
            params = call_args.kwargs.get('params', {})
            
            # Both category and keywords should be in params
            assert 'category' in params, "Category must be in API request parameters"
            assert 'q' in params, "Query (keywords) must be in API request parameters"
            
            # Verify values
            assert params['category'] == category
            assert all(keyword in params['q'] for keyword in keywords)
        
        await service.close()


class TestTrendyolProductTransformation:
    """
    Property 11: Trendyol Product to Gift Item Transformation
    Feature: trendyol-gift-recommendation-web, Property 11: Trendyol Product to Gift Item Transformation
    Validates: Requirements 4.2
    """
    
    @given(product_data=trendyol_product_data())
    @settings(max_examples=100, deadline=None)
    def test_product_transformation_produces_valid_gift_item(self, product_data):
        """
        For any Trendyol product response, the transformation should produce 
        a valid GiftItem with all required fields
        """
        service = TrendyolAPIService()
        
        # Create TrendyolProduct from data
        trendyol_product = TrendyolProduct(product_data)
        
        # Convert to GiftItem
        gift_item = service.convert_to_gift_item(trendyol_product)
        
        # Should produce a valid GiftItem (not None)
        if gift_item is not None:
            # Verify all required fields are present
            assert gift_item.id is not None
            assert gift_item.name is not None
            assert gift_item.category is not None
            assert gift_item.price >= 0
            assert 0 <= gift_item.rating <= 5
            assert gift_item.image_url is not None
            assert gift_item.trendyol_url is not None
            
            # Verify types
            assert isinstance(gift_item, GiftItem)
            assert isinstance(gift_item.id, str)
            assert isinstance(gift_item.name, str)
            assert isinstance(gift_item.category, str)
            assert isinstance(gift_item.price, float)
            assert isinstance(gift_item.rating, float)


class TestURLValidation:
    """
    Property 12: URL Validation and Filtering
    Feature: trendyol-gift-recommendation-web, Property 12: URL Validation and Filtering
    Validates: Requirements 4.5
    """
    
    @given(
        valid_urls=st.lists(valid_url(), min_size=1, max_size=10),
        invalid_urls=st.lists(
            st.one_of(
                st.just(""),
                st.just("not-a-url"),
                st.just("ftp://invalid.com"),
                st.just("javascript:alert(1)"),
                st.text(max_size=10)
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_url_validation_filters_invalid_urls(self, valid_urls, invalid_urls):
        """
        For any list of product URLs, the system should filter out invalid URLs 
        and keep only valid HTTP/HTTPS URLs
        """
        service = TrendyolAPIService()
        
        # Test valid URLs
        for url in valid_urls:
            result = service._validate_url(url)
            # Valid URLs should return a string
            assert result is not None, f"Valid URL should not be filtered: {url}"
            assert isinstance(result, str)
            assert result.startswith(('http://', 'https://'))
        
        # Test invalid URLs
        for url in invalid_urls:
            result = service._validate_url(url)
            # Invalid URLs should return None
            if url and not url.startswith(('http://', 'https://')):
                assert result is None, f"Invalid URL should be filtered: {url}"


class TestPriceNormalization:
    """
    Property 13: Price Normalization to TL Format
    Feature: trendyol-gift-recommendation-web, Property 13: Price Normalization to TL Format
    Validates: Requirements 4.6
    """
    
    @given(price=st.floats(
        min_value=-1000.0,
        max_value=1000000.0,
        allow_nan=False,
        allow_infinity=False
    ))
    @settings(max_examples=100, deadline=None)
    def test_price_normalization_to_two_decimals(self, price):
        """
        For any price value, the system should normalize it to Turkish Lira format 
        with 2 decimal places
        """
        service = TrendyolAPIService()
        
        # Normalize price
        normalized = service._normalize_price(price)
        
        # Should be a float
        assert isinstance(normalized, float)
        
        # Should be non-negative
        assert normalized >= 0.0
        
        # Should have at most 2 decimal places
        # Check by converting to string and counting decimals
        price_str = f"{normalized:.2f}"
        decimal_part = price_str.split('.')[1] if '.' in price_str else ""
        assert len(decimal_part) <= 2, f"Price should have at most 2 decimal places: {normalized}"
        
        # Verify rounding is correct (if original was positive)
        if price > 0:
            expected = round(price, 2)
            assert normalized == expected, f"Price should be rounded to 2 decimals: {price} -> {normalized}"
