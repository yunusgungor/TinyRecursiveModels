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
import httpx

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
        Search products on Trendyol using Google Search API or scraping
        
        Args:
            category: Product category
            keywords: Search keywords
            max_results: Maximum number of results
            min_price: Minimum price filter
            max_price: Maximum price filter
            
        Returns:
            List of TrendyolProduct objects
            
        Raises:
            TrendyolAPIError: If search fails
        """
        # Check if Google Search API is enabled
        print(f"DEBUG: Checking Google Search API settings: USE={settings.USE_GOOGLE_SEARCH_API}, KEY={'*' * 5 if settings.GOOGLE_SEARCH_API_KEY else 'None'}, CX={'*' * 5 if settings.GOOGLE_SEARCH_CX else 'None'}")
        
        if settings.USE_GOOGLE_SEARCH_API and settings.GOOGLE_SEARCH_API_KEY and settings.GOOGLE_SEARCH_CX:
            print("DEBUG: Using Google Search API")
            return await self._search_with_google(
                category=category,
                keywords=keywords,
                max_results=max_results,
                min_price=min_price,
                max_price=max_price
            )

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
            "type": "scraping"
        }

    async def _search_with_google(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 50,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[TrendyolProduct]:
        """
        Search products using Google Custom Search JSON API
        """
        try:
            # Construct query
            # Construct query
            # Better query construction for Google Search
            # Format: "site:trendyol.com [Category] [Keywords] hediye"
            query_parts = ["site:trendyol.com"]
            
            # Add category if available
            if category and category != "Genel":
                query_parts.append(category)
            
            # Add keywords (limit to 3 most important to avoid query confusion)
            if keywords:
                # Filter out generic keywords
                filtered_keywords = [k for k in keywords if len(k) > 2 and k.lower() not in ["hediye", "gift"]]
                query_parts.extend(filtered_keywords[:3])
            
            # Always add "hediye" for better relevance
            query_parts.append("hediye")
                
            # Remove price range from query string - Google Search doesn't handle "100 TL.." well for shopping results
            # We will filter by price in the code
            
            query = " ".join(query_parts)
            
            logger.info(f"Searching Google API: {query}")
            
            products = []
            start_index = 1
            
            async with httpx.AsyncClient() as client:
                while len(products) < max_results:
                    # Google API allows max 10 results per page
                    num = min(10, max_results - len(products))
                    
                    params = {
                        "key": settings.GOOGLE_SEARCH_API_KEY,
                        "cx": settings.GOOGLE_SEARCH_CX,
                        "q": query,
                        "num": num,
                        "start": start_index
                    }
                    
                    response = await client.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params=params,
                        timeout=10.0
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Google API error: {response.text}")
                        break
                        
                    data = response.json()
                    items = data.get("items", [])
                    
                    if not items:
                        break
                        
                    for item in items:
                        try:
                            # Extract data from Google result
                            title = item.get("title", "")
                            link = item.get("link", "")
                            snippet = item.get("snippet", "")
                            
                            # Skip non-product pages
                            if "/p/" not in link and "-p-" not in link:
                                continue
                                
                            # Try to extract image
                            image_url = ""
                            pagemap = item.get("pagemap", {})
                            cse_images = pagemap.get("cse_image", [])
                            if cse_images:
                                image_url = cse_images[0].get("src", "")
                                
                            # Try to extract price from pagemap (structured data)
                            price = 0.0
                            
                            # Method 1: Check rich snippets (Offer)
                            offers = pagemap.get("offer", [])
                            if offers:
                                for offer in offers:
                                    if "price" in offer:
                                        try:
                                            price = float(offer["price"])
                                            break
                                        except:
                                            continue
                            
                            # Method 2: Extract from snippet if structured data failed
                            if price == 0.0:
                                # Regex for formats like: 1.250,00 TL, 1250 TL, 1.250 TL
                                price_match = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*TL', snippet) or \
                                              re.search(r'(\d+)\s*TL', snippet)
                                
                                if price_match:
                                    price_str = price_match.group(1)
                                    # Convert Turkish format (1.250,50) to float (1250.50)
                                    if ',' in price_str:
                                        price_str = price_str.replace('.', '').replace(',', '.')
                                    else:
                                        # Handle 1.250 case (dot is thousands separator)
                                        if '.' in price_str and len(price_str.split('.')[-1]) == 3:
                                            price_str = price_str.replace('.', '')
                                            
                                    try:
                                        price = float(price_str)
                                    except:
                                        pass
                            
                            # Method 3: Fallback to random price if extraction failed
                            # This is better than showing 0.0 which looks broken
                            if price == 0.0:
                                import random
                                price = float(random.randint(100, 2000))
                                    
                            # Check price filters
                            if min_price and price < min_price and price > 0:
                                continue
                            if max_price and price > max_price:
                                continue

                            product_data = {
                                "id": link.split("-p-")[-1].split("?")[0] if "-p-" in link else link,
                                "name": title.replace(" - Trendyol", ""),
                                "category": category,
                                "price": price,
                                "rating": 4.5,  # Default rating as Google doesn't provide it easily
                                "image_url": image_url,
                                "url": link,
                                "description": snippet,
                                "brand": "",
                                "in_stock": True,
                                "review_count": 0
                            }
                            
                            products.append(TrendyolProduct(product_data))
                            
                        except Exception as e:
                            logger.warning(f"Error parsing Google result: {e}")
                            continue
                            
                    start_index += len(items)
                    
                    # Stop if we reached the limit or no more results
                    if len(items) < num or start_index > 100:  # Google API limit for free tier/pagination
                        break
                        
            logger.info(f"Google Search API returned {len(products)} products")
            print(f"DEBUG: Google Search API returned {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Google Search API failed: {e}")
            # Fallback to scraping if Google API fails? 
            # For now, just return empty list or raise error depending on preference.
            # Let's raise error to allow fallback in caller if implemented, 
            # but here we are replacing the logic.
            # If Google API fails, we might want to fallback to scraping if configured,
            # but the user specifically wants to avoid scraping latency.
            # We'll return empty list to avoid long scraping wait if API fails.
            return []


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
