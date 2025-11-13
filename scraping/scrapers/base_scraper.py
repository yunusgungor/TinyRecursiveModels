"""
Base Scraper Abstract Class
Provides common functionality for all website scrapers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import logging
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from ..utils.rate_limiter import RateLimiter
from ..utils.anti_bot import AntiBotHelper


class BaseScraper(ABC):
    """Base class for all website scrapers"""
    
    def __init__(self, config: Dict[str, Any], rate_limiter: RateLimiter):
        """
        Initialize base scraper
        
        Args:
            config: Website-specific configuration
            rate_limiter: Rate limiter instance
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Browser configuration
        self.browser_config = config.get('browser', {})
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Anti-bot helper
        user_agents = self.browser_config.get('user_agents', [])
        self.anti_bot = AntiBotHelper(user_agents)
        
        # Statistics
        self.stats = {
            'products_scraped': 0,
            'products_failed': 0,
            'pages_visited': 0
        }
        
        self.logger.info(f"{self.__class__.__name__} initialized")

    async def setup_browser(self) -> None:
        """Initialize Playwright browser"""
        self.logger.info("Setting up browser...")
        
        playwright = await async_playwright().start()
        
        # Launch browser
        self.browser = await playwright.chromium.launch(
            headless=self.browser_config.get('headless', True)
        )
        
        # Create context with anti-bot settings
        viewport = self.browser_config.get('viewport', {'width': 1920, 'height': 1080})
        user_agent = self.anti_bot.get_random_user_agent()
        
        self.context = await self.browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale='tr-TR',
            timezone_id='Europe/Istanbul'
        )
        
        # Set extra headers
        await self.context.set_extra_http_headers(
            self.anti_bot.get_browser_headers()
        )
        
        self.logger.info("Browser setup complete")
    
    async def close_browser(self) -> None:
        """Close browser and cleanup"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        self.logger.info("Browser closed")
    
    async def create_page(self) -> Page:
        """
        Create a new page with timeout settings
        
        Returns:
            Playwright Page object
        """
        if not self.context:
            await self.setup_browser()
        
        page = await self.context.new_page()
        timeout = self.browser_config.get('timeout', 30000)
        page.set_default_timeout(timeout)
        
        return page

    async def wait_random_delay(self) -> None:
        """Wait random delay using rate limiter"""
        async with self.rate_limiter:
            pass
    
    async def safe_goto(self, page: Page, url: str, wait_until: str = 'networkidle') -> bool:
        """
        Safely navigate to URL with error handling
        
        Args:
            page: Playwright page
            url: URL to navigate to
            wait_until: Wait condition ('load', 'domcontentloaded', 'networkidle')
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            await self.wait_random_delay()
            await page.goto(url, wait_until=wait_until)
            self.stats['pages_visited'] += 1
            
            # Check for CAPTCHA
            content = await page.content()
            if self.anti_bot.detect_captcha_keywords(content):
                self.logger.error(f"CAPTCHA detected on {url}")
                return False
            
            # Simulate human behavior
            await self.anti_bot.simulate_human_behavior(page)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to navigate to {url}: {e}")
            return False
    
    @abstractmethod
    async def scrape_products(self, max_products: int) -> List[Dict[str, Any]]:
        """
        Scrape products from the website
        
        Args:
            max_products: Maximum number of products to scrape
            
        Returns:
            List of scraped product dictionaries
        """
        pass
    
    @abstractmethod
    async def extract_product_details(self, page: Page, product_url: str) -> Dict[str, Any]:
        """
        Extract detailed product information from product page
        
        Args:
            page: Playwright page
            product_url: URL of the product
            
        Returns:
            Product data dictionary
        """
        pass

    def get_stats(self) -> Dict[str, int]:
        """
        Get scraping statistics
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.setup_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close_browser()
