"""Trendyol API integration service"""

import asyncio
import time
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import logging
from datetime import datetime, timedelta

import httpx
from pydantic import HttpUrl, ValidationError

from app.models.schemas import GiftItem
from app.core.config import settings
from app.core.exceptions import TrendyolAPIError, RateLimitError


logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request
        
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.window_seconds
            ]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = self.window_seconds - (now - oldest_request)
                
                logger.warning(
                    f"Rate limit reached. Need to wait {wait_time:.2f} seconds"
                )
                
                # Wait until we can make a request
                await asyncio.sleep(wait_time)
                
                # Retry acquire
                return await self.acquire()
            
            # Add current request
            self.requests.append(now)
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window"""
        now = time.time()
        
        # Remove old requests
        self.requests = [
            req_time for req_time in self.requests
            if now - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(self.requests))


class TrendyolProduct:
    """Trendyol product data model"""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize from API response data"""
        self.id = str(data.get("id", ""))
        self.name = data.get("name", "")
        self.category = data.get("category", {}).get("name", "")
        self.price = float(data.get("price", {}).get("sellingPrice", 0))
        self.rating = float(data.get("rating", {}).get("averageRating", 0))
        self.image_url = data.get("images", [{}])[0].get("url", "") if data.get("images") else ""
        self.product_url = data.get("url", "")
        self.description = data.get("description", "")
        self.brand = data.get("brand", {}).get("name", "")
        self.in_stock = data.get("inStock", True)
        self.review_count = int(data.get("rating", {}).get("totalCount", 0))


class TrendyolAPIService:
    """Service for interacting with Trendyol API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit: Optional[int] = None
    ):
        """
        Initialize Trendyol API service
        
        Args:
            api_key: Trendyol API key
            base_url: Base URL for Trendyol API
            rate_limit: Maximum requests per minute
        """
        self.api_key = api_key or settings.TRENDYOL_API_KEY
        self.base_url = base_url or settings.TRENDYOL_API_BASE_URL
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit or settings.TRENDYOL_RATE_LIMIT,
            window_seconds=60
        )
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            headers={
                "User-Agent": "TrendyolGiftRecommendation/1.0",
                "Accept": "application/json"
            }
        )
        
        # Cache for fallback
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        logger.info("TrendyolAPIService initialized")
    
    async def search_products(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 50,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[TrendyolProduct]:
        """
        Search products on Trendyol
        
        Args:
            category: Product category
            keywords: Search keywords
            max_results: Maximum number of results
            min_price: Minimum price filter
            max_price: Maximum price filter
            
        Returns:
            List of TrendyolProduct objects
            
        Raises:
            TrendyolAPIError: If API request fails
        """
        try:
            # Acquire rate limit permission
            await self.rate_limiter.acquire()
            
            # Build search query
            query = " ".join(keywords)
            
            # Build request parameters
            params = {
                "q": query,
                "category": category,
                "size": max_results,
            }
            
            if min_price is not None:
                params["minPrice"] = min_price
            
            if max_price is not None:
                params["maxPrice"] = max_price
            
            # Create cache key
            cache_key = f"search:{category}:{query}:{min_price}:{max_price}"
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached search results for: {query}")
                return cached_result
            
            # Make API request
            url = f"{self.base_url}/products/search"
            
            logger.info(f"Searching Trendyol: category={category}, keywords={keywords}")
            
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse products
                products = []
                for item in data.get("products", [])[:max_results]:
                    try:
                        product = TrendyolProduct(item)
                        products.append(product)
                    except Exception as e:
                        logger.warning(f"Failed to parse product: {e}")
                        continue
                
                # Cache results
                self._set_cache(cache_key, products)
                
                logger.info(f"Found {len(products)} products")
                return products
                
            except httpx.HTTPStatusError as e:
                logger.error(f"Trendyol API HTTP error: {e}")
                
                # Try to use cached data as fallback
                cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
                if cached_result is not None:
                    logger.warning("Using stale cached data as fallback")
                    return cached_result
                
                raise TrendyolAPIError(f"Trendyol API error: {e.response.status_code}")
                
            except httpx.RequestError as e:
                logger.error(f"Trendyol API request error: {e}")
                
                # Try to use cached data as fallback
                cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
                if cached_result is not None:
                    logger.warning("Using stale cached data as fallback")
                    return cached_result
                
                raise TrendyolAPIError(f"Trendyol API request failed: {str(e)}")
        
        except RateLimitError:
            raise
        except TrendyolAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search_products: {e}")
            raise TrendyolAPIError(f"Unexpected error: {str(e)}")
    
    async def get_product_details(self, product_id: str) -> TrendyolProduct:
        """
        Get detailed product information
        
        Args:
            product_id: Product ID
            
        Returns:
            TrendyolProduct object
            
        Raises:
            TrendyolAPIError: If API request fails
        """
        try:
            # Acquire rate limit permission
            await self.rate_limiter.acquire()
            
            # Create cache key
            cache_key = f"product:{product_id}"
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached product details for: {product_id}")
                return cached_result
            
            # Make API request
            url = f"{self.base_url}/products/{product_id}"
            
            logger.info(f"Fetching product details: {product_id}")
            
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                
                data = response.json()
                product = TrendyolProduct(data)
                
                # Cache result
                self._set_cache(cache_key, product)
                
                return product
                
            except httpx.HTTPStatusError as e:
                logger.error(f"Trendyol API HTTP error: {e}")
                
                # Try to use cached data as fallback
                cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
                if cached_result is not None:
                    logger.warning("Using stale cached data as fallback")
                    return cached_result
                
                raise TrendyolAPIError(f"Trendyol API error: {e.response.status_code}")
                
            except httpx.RequestError as e:
                logger.error(f"Trendyol API request error: {e}")
                
                # Try to use cached data as fallback
                cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
                if cached_result is not None:
                    logger.warning("Using stale cached data as fallback")
                    return cached_result
                
                raise TrendyolAPIError(f"Trendyol API request failed: {str(e)}")
        
        except RateLimitError:
            raise
        except TrendyolAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_product_details: {e}")
            raise TrendyolAPIError(f"Unexpected error: {str(e)}")
    
    def convert_to_gift_item(self, product: TrendyolProduct) -> Optional[GiftItem]:
        """
        Convert Trendyol product to GiftItem format
        
        Args:
            product: TrendyolProduct object
            
        Returns:
            GiftItem object or None if conversion fails
        """
        try:
            # Validate and normalize image URL
            image_url = self._validate_url(product.image_url)
            if not image_url:
                logger.warning(f"Invalid image URL for product {product.id}")
                return None
            
            # Validate and normalize product URL
            product_url = self._validate_url(product.product_url)
            if not product_url:
                logger.warning(f"Invalid product URL for product {product.id}")
                return None
            
            # Normalize price
            normalized_price = self._normalize_price(product.price)
            
            # Create GiftItem
            gift_item = GiftItem(
                id=product.id,
                name=product.name,
                category=product.category,
                price=normalized_price,
                rating=min(product.rating, 5.0),  # Ensure rating is within bounds
                image_url=image_url,
                trendyol_url=product_url,
                description=product.description,
                tags=[product.brand, product.category] if product.brand else [product.category],
                age_suitability=(18, 100),  # Default age range
                occasion_fit=[],  # Will be filled by model
                in_stock=product.in_stock
            )
            
            return gift_item
            
        except ValidationError as e:
            logger.error(f"Failed to convert product {product.id} to GiftItem: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error converting product {product.id}: {e}")
            return None
    
    def _validate_url(self, url: str) -> Optional[str]:
        """
        Validate and filter URL
        
        Args:
            url: URL string to validate
            
        Returns:
            Validated URL string or None if invalid
        """
        if not url:
            return None
        
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ["http", "https"]:
                return None
            
            # Check if URL has a valid domain
            if not parsed.netloc:
                return None
            
            # Reconstruct URL to ensure it's properly formatted
            validated_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            if parsed.query:
                validated_url += f"?{parsed.query}"
            
            # Validate with Pydantic
            HttpUrl(validated_url)
            
            return validated_url
            
        except (ValueError, ValidationError):
            return None
    
    def _normalize_price(self, price: float) -> float:
        """
        Normalize price to TL format with 2 decimal places
        
        Args:
            price: Price value
            
        Returns:
            Normalized price
        """
        # Round to 2 decimal places
        normalized = round(price, 2)
        
        # Ensure non-negative
        return max(0.0, normalized)
    
    def _get_from_cache(
        self,
        key: str,
        ignore_ttl: bool = False
    ) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            ignore_ttl: Whether to ignore TTL check
            
        Returns:
            Cached value or None
        """
        if key not in self._cache:
            return None
        
        if not ignore_ttl:
            # Check TTL
            timestamp = self._cache_timestamps.get(key)
            if timestamp:
                ttl = timedelta(seconds=settings.CACHE_TTL_TRENDYOL_DATA)
                if datetime.now() - timestamp > ttl:
                    # Cache expired
                    del self._cache[key]
                    del self._cache_timestamps[key]
                    return None
        
        return self._cache[key]
    
    def _set_cache(self, key: str, value: Any) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    async def close(self) -> None:
        """Close HTTP client"""
        await self.client.aclose()
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "max_requests": self.rate_limiter.max_requests,
            "window_seconds": self.rate_limiter.window_seconds,
            "remaining_requests": self.rate_limiter.get_remaining_requests()
        }


# Singleton instance
_trendyol_service: Optional[TrendyolAPIService] = None


def get_trendyol_service() -> TrendyolAPIService:
    """Get or create Trendyol API service singleton"""
    global _trendyol_service
    
    if _trendyol_service is None:
        _trendyol_service = TrendyolAPIService()
    
    return _trendyol_service
