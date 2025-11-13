"""
Scraping Orchestrator
Coordinates scraping from multiple websites
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config.config_manager import ConfigurationManager
from ..utils.rate_limiter import RateLimiter
from .ciceksepeti_scraper import CicekSepetiScraper
from .hepsiburada_scraper import HepsiburadaScraper
from .trendyol_scraper import TrendyolScraper


class ScrapingOrchestrator:
    """Orchestrates scraping from multiple websites"""
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize scraping orchestrator
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter
        rate_limit_config = config_manager.get_rate_limit_config()
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Scraper registry
        self.scrapers: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_products': 0,
            'products_by_source': {},
            'errors': []
        }
        
        self.logger.info("ScrapingOrchestrator initialized")

    def initialize_scrapers(self) -> None:
        """Initialize all enabled scrapers using factory pattern"""
        self.logger.info("Initializing scrapers...")
        
        enabled_websites = self.config_manager.get_enabled_websites()
        browser_config = self.config_manager.get_browser_config()
        
        for website_config in enabled_websites:
            website_name = website_config.get('name')
            
            # Merge browser config with website config
            full_config = {**website_config, 'browser': browser_config}
            
            # Factory pattern: create appropriate scraper
            scraper = self._create_scraper(website_name, full_config)
            
            if scraper:
                self.scrapers[website_name] = scraper
                self.logger.info(f"Initialized scraper for {website_name}")
            else:
                self.logger.warning(f"No scraper implementation for {website_name}")
        
        self.logger.info(f"Initialized {len(self.scrapers)} scrapers")
    
    def _create_scraper(self, website_name: str, config: Dict[str, Any]):
        """
        Factory method to create appropriate scraper
        
        Args:
            website_name: Name of the website
            config: Website configuration
            
        Returns:
            Scraper instance or None
        """
        scraper_map = {
            'ciceksepeti': CicekSepetiScraper,
            'hepsiburada': HepsiburadaScraper,
            'trendyol': TrendyolScraper
        }
        
        scraper_class = scraper_map.get(website_name)
        if scraper_class:
            return scraper_class(config, self.rate_limiter)
        
        return None

    async def scrape_all_websites(self) -> List[Dict[str, Any]]:
        """
        Scrape products from all enabled websites
        
        Returns:
            List of all scraped products
        """
        self.logger.info("Starting multi-website scraping...")
        self.stats['start_time'] = datetime.now()
        
        all_products = []
        
        # Check if test mode
        test_mode = self.config_manager.is_test_mode()
        if test_mode:
            max_products_per_site = self.config_manager.get_test_products_limit()
            self.logger.info(f"TEST MODE: Limiting to {max_products_per_site} products per site")
        
        # Scrape from each website
        for website_name, scraper in self.scrapers.items():
            try:
                self.logger.info(f"Starting scraping from {website_name}...")
                
                # Get max products for this website
                website_config = self.config_manager.get_website_config(website_name)
                max_products = website_config.get('max_products', 500)
                
                if test_mode:
                    max_products = min(max_products, max_products_per_site)
                
                # Scrape products
                async with scraper:
                    products = await scraper.scrape_products(max_products)
                    
                    all_products.extend(products)
                    self.stats['products_by_source'][website_name] = len(products)
                    
                    self.logger.info(
                        f"Completed {website_name}: {len(products)} products scraped"
                    )
                    
            except Exception as e:
                error_msg = f"Error scraping {website_name}: {e}"
                self.logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        self.stats['end_time'] = datetime.now()
        self.stats['total_products'] = len(all_products)
        
        self.logger.info(
            f"Multi-website scraping completed: {len(all_products)} total products"
        )
        
        return all_products

    async def scrape_website(self, website_name: str) -> List[Dict[str, Any]]:
        """
        Scrape products from a specific website
        
        Args:
            website_name: Name of the website to scrape
            
        Returns:
            List of scraped products
        """
        self.logger.info(f"Starting scraping from {website_name}...")
        
        if website_name not in self.scrapers:
            self.logger.error(f"No scraper found for {website_name}")
            return []
        
        scraper = self.scrapers[website_name]
        
        try:
            # Get max products
            website_config = self.config_manager.get_website_config(website_name)
            max_products = website_config.get('max_products', 500)
            
            if self.config_manager.is_test_mode():
                max_products = min(max_products, self.config_manager.get_test_products_limit())
            
            # Scrape
            async with scraper:
                products = await scraper.scrape_products(max_products)
                
                self.logger.info(
                    f"Completed {website_name}: {len(products)} products scraped"
                )
                
                return products
                
        except Exception as e:
            self.logger.error(f"Error scraping {website_name}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scraping statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Calculate duration
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            stats['duration_seconds'] = duration.total_seconds()
            stats['duration_formatted'] = str(duration)
        
        # Add rate limiter stats
        stats['rate_limiter'] = self.rate_limiter.get_stats()
        
        return stats

    def handle_captcha_detected(self, website_name: str, url: str) -> None:
        """
        Handle CAPTCHA detection
        
        Args:
            website_name: Name of the website where CAPTCHA was detected
            url: URL where CAPTCHA was detected
        """
        error_msg = f"CAPTCHA detected on {website_name} at {url}"
        self.logger.error(error_msg)
        self.stats['errors'].append(error_msg)
        
        # Log detailed message for user
        self.logger.warning(
            f"\n{'='*60}\n"
            f"CAPTCHA DETECTED!\n"
            f"Website: {website_name}\n"
            f"URL: {url}\n"
            f"Action: Scraping paused for this website\n"
            f"Recommendation: Try again later or adjust rate limits\n"
            f"{'='*60}\n"
        )
