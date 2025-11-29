"""
Trendyol Scraping Service
Replaces the fake API service with real web scraping functionality
"""

import asyncio
import time
import re
import sys
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add scraping directory to path
SCRAPING_DIR = Path(__file__).parent.parent.parent.parent / "scraping"
sys.path.insert(0, str(SCRAPING_DIR))

try:
    from scrapers.trendyol_scraper import TrendyolScraper
    from utils.rate_limiter import RateLimiter as ScrapingRateLimiter
except ImportError as e:
    logging.error(f"Failed to import scraping modules: {e}")
    logging.error(f"Scraping directory: {SCRAPING_DIR}")
    raise

from app.models.schemas import GiftItem
from app.core.config import settings
from app.core.exceptions import TrendyolAPIError


logger = logging.getLogger(__name__)


class TrendyolProduct:
    """Trendyol product data model - compatible with original API service"""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize from scraped data"""
        self.id = str(data.get("id", data.get("url", "").split("/")[-1]))
        self.name = data.get("name", "")
        self.category = data.get("category", data.get("raw_category", ""))
        self.price = float(data.get("price", 0))
        self.rating = float(data.get("rating", 0))
        self.image_url = data.get("image_url", "")
        self.product_url = data.get("url", data.get("trendyol_url", ""))
        self.description = data.get("description", "")
        self.brand = data.get("brand", "")
        self.in_stock = data.get("in_stock", True)
        self.review_count = int(data.get("review_count", 0))


class TrendyolScrapingService:
    """
    Service for scraping product data from Trendyol
    Maintains same interface as TrendyolAPIService for compatibility
    """
    
    # Category mapping for search
    CATEGORY_MAPPING = {
        "elektronik": "elektronik",
        "ev_yasam": "ev-yasam",
        "ev": "ev-yasam",
        "kozmetik": "kozmetik",
        "giyim": "kadin-giyim",
        "kadin": "kadin-giyim",
        "erkek": "erkek-giyim",
        "cocuk": "anne-cocuk",
        "ayakkabi": "ayakkabi-canta",
        "supermarket": "supermarket",
        "mobilya": "mobilya",
        "spor": "spor-outdoor",
        "kitap": "kitap-kirtasiye",
    }
    
    def __init__(
        self,
        rate_limit: Optional[int] = None,
        cache_ttl: Optional[int] = None
    ):
        """
        Initialize Trendyol scraping service
        
        Args:
            rate_limit: Maximum requests per minute
            cache_ttl: Cache time-to-live in seconds
        """
        self.rate_limit = rate_limit or 20  # Lower than API to avoid detection
        self.cache_ttl = cache_ttl or settings.CACHE_TTL_TRENDYOL_DATA
        
        # Initialize rate limiter for scraping
        rate_limiter_config = {
            'requests_per_minute': self.rate_limit,
            'delay_between_requests': [2, 5],  # Random delay between requests
            'max_concurrent_requests': 3  # Lower for scraping
        }
        self.scraping_rate_limiter = ScrapingRateLimiter(rate_limiter_config)

        
        # Cache for scraped data
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Scraper instance (will be created when needed)
        self._scraper: Optional[TrendyolScraper] = None
        self._scraper_config: Dict[str, Any] = {
            "url": "https://www.trendyol.com",
            "categories": ["elektronik"],
            "max_products": 50,
            "browser": {
                "headless": True,
                "timeout": 30000,
                "viewport": {"width": 1920, "height": 1080},
                "user_agents": [
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                ]
            }
        }
        
        logger.info("TrendyolScrapingService initialized")
    
    async def _get_scraper(self, categories: List[str] = None) -> TrendyolScraper:
        """Get or create scraper instance"""
        if categories:
            self._scraper_config["categories"] = categories
        
        if self._scraper is None:
            self._scraper = TrendyolScraper(
                self._scraper_config,
                self.scraping_rate_limiter
            )
            await self._scraper.setup_browser()
        
        return self._scraper
    
    async def search_products(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 50,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[TrendyolProduct]:
        """
        Search products on Trendyol using scraping
        
        Args:
            category: Product category
            keywords: Search keywords
            max_results: Maximum number of results
            min_price: Minimum price filter
            max_price: Maximum price filter
            
        Returns:
            List of TrendyolProduct objects
            
        Raises:
            TrendyolAPIError: If scraping fails
        """
        try:
            # Create cache key
            keywords_str = "_".join(keywords) if keywords else "all"
            cache_key = f"search:{category}:{keywords_str}:{min_price}:{max_price}:{max_results}"
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached search results for: {category}")
                return cached_result
            
            # Map category to scraping category
            scraping_category = self.CATEGORY_MAPPING.get(
                category.lower().replace(" ", "_"),
                "elektronik"
            )
            
            logger.info(
                f"Scraping Trendyol: category={scraping_category}, "
                f"keywords={keywords}, max_results={max_results}"
            )
            
            # Get scraper
            scraper = await self._get_scraper(categories=[scraping_category])
            
            all_scraped_data = []
            
            if keywords:
                # Search for each keyword separately
                # Calculate limit per keyword to maintain diversity but respect total limit
                # We fetch a bit more to allow for filtering
                per_keyword_limit = max(15, int(max_results * 1.5 / len(keywords)))
                
                for keyword in keywords:
                    logger.info(f"Searching for keyword: {keyword}")
                    print(f"DEBUG: Searching for keyword: {keyword}")
                    keyword_data = await scraper.scrape_products(
                        max_products=per_keyword_limit, 
                        search_query=keyword
                    )
                    all_scraped_data.extend(keyword_data)
            else:
                # Fallback to category scraping if no keywords
                all_scraped_data = await scraper.scrape_products(max_products=max_results)
            
            if not all_scraped_data:
                logger.warning(f"No products found for category: {category}")
                return []
            
            # Convert to TrendyolProduct objects and dedup
            products = []
            seen_ids = set()
            
            for item in all_scraped_data:
                try:
                    # Apply price filters
                    price = item.get("price", 0)
                    if min_price is not None and price < min_price:
                        continue
                    if max_price is not None and price > max_price:
                        continue
                    
                    # Dedup check (using URL or name as ID proxy if ID not present)
                    product_id = item.get("id") or item.get("url")
                    if not product_id or product_id in seen_ids:
                        continue
                    seen_ids.add(product_id)
                    
                    product = TrendyolProduct(item)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to parse scraped product: {e}")
                    continue
            
            # Cache results
            self._set_cache(cache_key, products)
            
            logger.info(f"Scraped {len(products)} unique products")
            return products[:max_results]
            
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            
            # Try to use cached data as fallback
            cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
            if cached_result is not None:
                logger.warning("Using stale cached data as fallback")
                return cached_result
            
            raise TrendyolAPIError(f"Trendyol scraping failed: {str(e)}")
    
    async def get_product_details(self, product_id: str) -> TrendyolProduct:
        """
        Get detailed product information using scraping
        
        Args:
            product_id: Product ID or URL
            
        Returns:
            TrendyolProduct object
            
        Raises:
            TrendyolAPIError: If scraping fails
        """
        try:
            # Create cache key
            cache_key = f"product:{product_id}"
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached product details for: {product_id}")
                return cached_result
            
            # Construct product URL if not already a URL
            if not product_id.startswith("http"):
                product_url = f"https://www.trendyol.com/product/-p-{product_id}"
            else:
                product_url = product_id
            
            logger.info(f"Scraping product details: {product_url}")
            
            # Get scraper
            scraper = await self._get_scraper()
            
            # Create a temporary page
            page = await scraper.create_page()
            
            try:
                # Extract product details
                product_data = await scraper.extract_product_details(page, product_url)
                
                if not product_data:
                    raise TrendyolAPIError(f"Failed to scrape product: {product_id}")
                
                product = TrendyolProduct(product_data)
                
                # Cache result
                self._set_cache(cache_key, product)
                
                return product
                
            finally:
                await page.close()
            
        except Exception as e:
            logger.error(f"Error getting product details: {e}")
            
            # Try to use cached data as fallback
            cached_result = self._get_from_cache(cache_key, ignore_ttl=True)
            if cached_result is not None:
                logger.warning("Using stale cached data as fallback")
                return cached_result
            
            raise TrendyolAPIError(f"Failed to get product details: {str(e)}")
    
    def convert_to_gift_item(self, product: TrendyolProduct) -> Optional[GiftItem]:
        """
        Convert Trendyol product to GiftItem format
        
        Args:
            product: TrendyolProduct object
            
        Returns:
            GiftItem object or None if conversion fails
        """
        try:
            # Validate required fields
            if not product.name or not product.product_url:
                logger.warning(f"Product missing required fields: {product.id}")
                return None
            
            # Use placeholder image if none provided
            image_url = product.image_url or "https://via.placeholder.com/400x400?text=No+Image"
            
            # Create GiftItem
            gift_item = GiftItem(
                id=product.id,
                name=product.name,
                category=product.category or "Genel",
                price=max(0.0, round(product.price, 2)),
                rating=min(5.0, max(0.0, product.rating)),
                image_url=image_url,
                trendyol_url=product.product_url,
                description=product.description or product.name,
                tags=[product.brand, product.category] if product.brand else [product.category],
                age_suitability=(18, 100),  # Default age range
                occasion_fit=[],  # Will be filled by model
                in_stock=product.in_stock
            )
            
            return gift_item
            
        except Exception as e:
            logger.error(f"Failed to convert product {product.id} to GiftItem: {e}")
            return None
    
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
                ttl = timedelta(seconds=self.cache_ttl)
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
        """Close scraper and cleanup"""
        if self._scraper:
            await self._scraper.close_browser()
            self._scraper = None
        logger.info("TrendyolScrapingService closed")
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        return {
            "max_requests": self.rate_limit,
            "window_seconds": 60,
            "type": "scraping"
        }


# Singleton instance
_trendyol_scraping_service: Optional[TrendyolScrapingService] = None


def get_trendyol_scraping_service() -> TrendyolScrapingService:
    """Get or create Trendyol scraping service singleton"""
    global _trendyol_scraping_service
    
    if _trendyol_scraping_service is None:
        _trendyol_scraping_service = TrendyolScrapingService()
    
    return _trendyol_scraping_service


async def cleanup_trendyol_service():
    """Cleanup singleton service"""
    global _trendyol_scraping_service
    
    if _trendyol_scraping_service:
        await _trendyol_scraping_service.close()
        _trendyol_scraping_service = None
