"""
Çiçek Sepeti Scraper
Scrapes product data from ciceksepeti.com
"""

from typing import List, Dict, Any
from playwright.async_api import Page

from .base_scraper import BaseScraper


class CicekSepetiScraper(BaseScraper):
    """Scraper for ciceksepeti.com"""
    
    # Site-specific selectors (to be updated based on actual site structure)
    SELECTORS = {
        'product_list': '.product-item',  # Placeholder
        'product_link': 'a.product-link',  # Placeholder
        'product_name': 'h1.product-name',  # Placeholder
        'product_price': '.product-price',  # Placeholder
        'product_description': '.product-description',  # Placeholder
        'product_image': 'img.product-image',  # Placeholder
        'product_rating': '.product-rating',  # Placeholder
        'next_page': '.pagination-next',  # Placeholder
    }
    
    def __init__(self, config: Dict[str, Any], rate_limiter):
        """
        Initialize Çiçek Sepeti scraper
        
        Args:
            config: Website configuration
            rate_limiter: Rate limiter instance
        """
        super().__init__(config, rate_limiter)
        self.base_url = config.get('url', 'https://www.ciceksepeti.com')
        self.categories = config.get('categories', ['hediye'])
        self.max_products = config.get('max_products', 500)

    async def scrape_products(self, max_products: int) -> List[Dict[str, Any]]:
        """
        Scrape products from Çiçek Sepeti
        
        Args:
            max_products: Maximum number of products to scrape
            
        Returns:
            List of scraped product dictionaries
        """
        self.logger.info(f"Starting to scrape up to {max_products} products from Çiçek Sepeti")
        
        all_products = []
        
        for category in self.categories:
            if len(all_products) >= max_products:
                break
            
            self.logger.info(f"Scraping category: {category}")
            category_products = await self._scrape_category(category, max_products - len(all_products))
            all_products.extend(category_products)
        
        self.logger.info(f"Scraped {len(all_products)} products from Çiçek Sepeti")
        return all_products
    
    async def _scrape_category(self, category: str, max_products: int) -> List[Dict[str, Any]]:
        """
        Scrape products from a specific category
        
        Args:
            category: Category name
            max_products: Maximum products to scrape from this category
            
        Returns:
            List of product dictionaries
        """
        products = []
        page = await self.create_page()
        
        try:
            # Build category URL
            category_url = f"{self.base_url}/{category}"
            
            # Navigate to category page
            success = await self.safe_goto(page, category_url)
            if not success:
                self.logger.error(f"Failed to load category: {category}")
                return products
            
            # Extract product URLs from listing
            product_urls = await self._extract_product_urls(page, max_products)
            self.logger.info(f"Found {len(product_urls)} product URLs in {category}")
            
            # Scrape each product
            for url in product_urls:
                if len(products) >= max_products:
                    break
                
                try:
                    product_data = await self.extract_product_details(page, url)
                    if product_data:
                        products.append(product_data)
                        self.stats['products_scraped'] += 1
                        
                        if len(products) % 10 == 0:
                            self.logger.info(f"Scraped {len(products)} products so far...")
                except Exception as e:
                    self.logger.error(f"Failed to scrape product {url}: {e}")
                    self.stats['products_failed'] += 1
        
        finally:
            await page.close()
        
        return products

    async def _extract_product_urls(self, page: Page, max_urls: int) -> List[str]:
        """
        Extract product URLs from category listing page
        
        Args:
            page: Playwright page
            max_urls: Maximum number of URLs to extract
            
        Returns:
            List of product URLs
        """
        urls = []
        current_page = 1
        
        while len(urls) < max_urls:
            try:
                # Wait for product list to load
                await page.wait_for_selector(self.SELECTORS['product_list'], timeout=10000)
                
                # Extract product links
                links = await page.query_selector_all(self.SELECTORS['product_link'])
                
                for link in links:
                    if len(urls) >= max_urls:
                        break
                    
                    href = await link.get_attribute('href')
                    if href:
                        # Make absolute URL if relative
                        if href.startswith('/'):
                            href = self.base_url + href
                        urls.append(href)
                
                # Check for next page
                next_button = await page.query_selector(self.SELECTORS['next_page'])
                if not next_button or len(urls) >= max_urls:
                    break
                
                # Click next page
                await next_button.click()
                await page.wait_for_load_state('networkidle')
                current_page += 1
                self.logger.debug(f"Moved to page {current_page}")
                
            except Exception as e:
                self.logger.warning(f"Error extracting URLs on page {current_page}: {e}")
                break
        
        return urls[:max_urls]

    async def extract_product_details(self, page: Page, product_url: str) -> Dict[str, Any]:
        """
        Extract detailed product information from product page
        
        Args:
            page: Playwright page
            product_url: URL of the product
            
        Returns:
            Product data dictionary
        """
        try:
            # Navigate to product page
            success = await self.safe_goto(page, product_url)
            if not success:
                return None
            
            # Extract product name
            name_element = await page.query_selector(self.SELECTORS['product_name'])
            name = await name_element.inner_text() if name_element else "Unknown Product"
            name = name.strip()
            
            # Extract price
            price_element = await page.query_selector(self.SELECTORS['product_price'])
            price_text = await price_element.inner_text() if price_element else "0"
            price = self._parse_price(price_text)
            
            # Extract description
            desc_element = await page.query_selector(self.SELECTORS['product_description'])
            description = await desc_element.inner_text() if desc_element else name
            description = description.strip()
            
            # Extract image URL
            img_element = await page.query_selector(self.SELECTORS['product_image'])
            image_url = await img_element.get_attribute('src') if img_element else None
            
            # Extract rating
            rating_element = await page.query_selector(self.SELECTORS['product_rating'])
            rating = 0.0
            if rating_element:
                rating_text = await rating_element.inner_text()
                rating = self._parse_rating(rating_text)
            
            # Build product data
            product_data = {
                'source': 'ciceksepeti',
                'url': product_url,
                'name': name,
                'price': price,
                'description': description,
                'image_url': image_url,
                'rating': rating,
                'in_stock': True,  # Assume in stock if page loads
                'raw_category': self.categories[0] if self.categories else 'hediye'
            }
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error extracting product details from {product_url}: {e}")
            return None

    def _parse_price(self, price_text: str) -> float:
        """
        Parse price from text
        
        Args:
            price_text: Price text (e.g., "150,00 TL", "150.00 TL")
            
        Returns:
            Price as float
        """
        try:
            # Remove currency symbols and whitespace
            price_text = price_text.replace('TL', '').replace('₺', '').strip()
            # Replace Turkish decimal separator
            price_text = price_text.replace('.', '').replace(',', '.')
            return float(price_text)
        except:
            self.logger.warning(f"Could not parse price: {price_text}")
            return 0.0
    
    def _parse_rating(self, rating_text: str) -> float:
        """
        Parse rating from text
        
        Args:
            rating_text: Rating text (e.g., "4.5", "4,5")
            
        Returns:
            Rating as float (0-5)
        """
        try:
            rating_text = rating_text.replace(',', '.').strip()
            # Extract first number
            import re
            match = re.search(r'(\d+\.?\d*)', rating_text)
            if match:
                rating = float(match.group(1))
                return min(5.0, max(0.0, rating))
            return 0.0
        except:
            self.logger.warning(f"Could not parse rating: {rating_text}")
            return 0.0
