"""
Cimri.com Scraper
Scrapes product data from cimri.com - a price comparison website
This is especially valuable as it aggregates products from multiple stores
"""

from typing import List, Dict, Any
from playwright.async_api import Page
import re

from .base_scraper import BaseScraper


class CimriScraper(BaseScraper):
    """Scraper for cimri.com - Price comparison website"""
    
    # Site-specific selectors based on actual HTML structure
    SELECTORS = {
        # List page selectors
        'product_list': 'article',  # Each product is in an article tag
        'product_link': 'article a[href*="/hediyelik-esya/en-ucuz"]',  # Product detail link
        'product_name': 'article h3',  # Product name in h3
        'product_price': 'article p',  # Price in p tag
        'product_image': 'article img',  # Product image
        'store_names': 'article img[alt]',  # Store logos with alt text
        'next_page': 'a[href*="?page="]',  # Pagination link
        
        # Detail page selectors
        'detail_name': 'h1',
        'detail_description': 'div.product-description, p',
        'detail_prices': 'div.price-list',
        'detail_stores': 'div.store-list',
        'detail_rating': 'span.rating, div.rating',
        'detail_image': 'img.product-image, img[alt*=""]',
    }
    
    def __init__(self, config: Dict[str, Any], rate_limiter):
        """
        Initialize Cimri scraper
        
        Args:
            config: Website configuration
            rate_limiter: Rate limiter instance
        """
        super().__init__(config, rate_limiter)
        self.base_url = config.get('url', 'https://www.cimri.com')
        self.categories = config.get('categories', ['hediyelik-esya'])
        self.max_products = config.get('max_products', 500)
        
        self.logger.info(f"CimriScraper initialized for categories: {self.categories}")

    async def scrape_products(self, max_products: int) -> List[Dict[str, Any]]:
        """
        Scrape products from Cimri.com
        
        Args:
            max_products: Maximum number of products to scrape
            
        Returns:
            List of scraped product dictionaries
        """
        self.logger.info(f"Starting to scrape up to {max_products} products from Cimri.com")
        
        all_products = []
        
        for category in self.categories:
            if len(all_products) >= max_products:
                break
            
            self.logger.info(f"Scraping category: {category}")
            category_products = await self._scrape_category(category, max_products - len(all_products))
            all_products.extend(category_products)
        
        self.logger.info(f"Scraped {len(all_products)} products from Cimri.com")
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
        
        # Ensure browser is set up
        if not self.browser or not self.context:
            await self.setup_browser()
        
        page = await self.create_page()
        
        try:
            # Build category URL
            category_url = f"{self.base_url}/{category}"
            
            # Navigate to category page
            success = await self.safe_goto(page, category_url)
            if not success:
                self.logger.error(f"Failed to load category: {category}")
                return products
            
            # Wait for products to load
            await page.wait_for_selector(self.SELECTORS['product_list'], timeout=10000)
            
            # Extract products from current page
            page_num = 1
            while len(products) < max_products:
                self.logger.info(f"Scraping page {page_num} of {category}")
                
                # Get all product articles on current page
                product_elements = await page.query_selector_all(self.SELECTORS['product_list'])
                self.logger.info(f"Found {len(product_elements)} products on page {page_num}")
                
                for element in product_elements:
                    if len(products) >= max_products:
                        break
                    
                    try:
                        product_data = await self._extract_product_from_list(element)
                        if product_data:
                            products.append(product_data)
                            self.stats['products_scraped'] += 1
                            
                            if len(products) % 10 == 0:
                                self.logger.info(f"Scraped {len(products)} products so far...")
                    except Exception as e:
                        self.logger.error(f"Failed to extract product: {e}")
                        self.stats['products_failed'] += 1
                
                # Check if we need more products and if there's a next page
                if len(products) >= max_products:
                    break
                
                # Try to go to next page
                next_button = await page.query_selector(self.SELECTORS['next_page'])
                if not next_button:
                    self.logger.info("No more pages available")
                    break
                
                # Click next page
                try:
                    await next_button.click()
                    await page.wait_for_load_state('networkidle')
                    page_num += 1
                except Exception as e:
                    self.logger.warning(f"Could not navigate to next page: {e}")
                    break
        
        finally:
            await page.close()
        
        return products

    async def _extract_product_from_list(self, element) -> Dict[str, Any]:
        """
        Extract product data from a list item element
        
        Args:
            element: Playwright element handle for product article
            
        Returns:
            Product data dictionary
        """
        try:
            # Extract product name
            name_element = await element.query_selector('h3')
            name = await name_element.inner_text() if name_element else "Unknown Product"
            name = name.strip()
            
            # Extract product link
            link_element = await element.query_selector('a[href*="/hediyelik-esya/en-ucuz"]')
            product_url = await link_element.get_attribute('href') if link_element else None
            if product_url and not product_url.startswith('http'):
                product_url = self.base_url + product_url
            
            # Extract price (first p tag usually contains the lowest price)
            price_elements = await element.query_selector_all('p')
            price = 0.0
            for price_el in price_elements:
                price_text = await price_el.inner_text()
                parsed_price = self._parse_price(price_text)
                if parsed_price > 0:
                    price = parsed_price
                    break
            
            # Extract image
            img_element = await element.query_selector('img')
            image_url = None
            if img_element:
                image_url = await img_element.get_attribute('src')
                if not image_url:
                    image_url = await img_element.get_attribute('data-src')
            
            # Extract store names (from store logos)
            store_elements = await element.query_selector_all('img[alt]')
            stores = []
            for store_el in store_elements:
                store_name = await store_el.get_attribute('alt')
                if store_name and store_name not in ['', 'ürün yükleniyor']:
                    stores.append(store_name)
            
            # Build description from available stores
            description = f"{name}"
            if stores:
                description += f" - Mevcut mağazalar: {', '.join(stores[:3])}"
            
            # Build product data
            product_data = {
                'source': 'cimri',
                'url': product_url or f"{self.base_url}/hediyelik-esya",
                'name': name,
                'price': price,
                'description': description,
                'image_url': image_url,
                'rating': 0.0,  # Cimri doesn't show ratings on list page
                'in_stock': True,
                'raw_category': 'hediyelik-esya',
                'stores': stores,  # Extra field: available stores
                'store_count': len(stores)  # Extra field: number of stores
            }
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error extracting product from list: {e}")
            return None

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
            name_element = await page.query_selector(self.SELECTORS['detail_name'])
            name = await name_element.inner_text() if name_element else "Unknown Product"
            name = name.strip()
            
            # Extract description
            desc_element = await page.query_selector(self.SELECTORS['detail_description'])
            description = await desc_element.inner_text() if desc_element else name
            description = description.strip()
            
            # Extract lowest price
            price_elements = await page.query_selector_all('p')
            price = 0.0
            for price_el in price_elements:
                price_text = await price_el.inner_text()
                parsed_price = self._parse_price(price_text)
                if parsed_price > 0:
                    price = parsed_price
                    break
            
            # Extract image URL
            img_element = await page.query_selector(self.SELECTORS['detail_image'])
            image_url = None
            if img_element:
                image_url = await img_element.get_attribute('src')
                if not image_url:
                    image_url = await img_element.get_attribute('data-src')
            
            # Extract rating if available
            rating_element = await page.query_selector(self.SELECTORS['detail_rating'])
            rating = 0.0
            if rating_element:
                rating_text = await rating_element.inner_text()
                rating = self._parse_rating(rating_text)
            
            # Extract available stores
            store_elements = await page.query_selector_all('img[alt]')
            stores = []
            for store_el in store_elements:
                store_name = await store_el.get_attribute('alt')
                if store_name and store_name not in ['', 'ürün yükleniyor']:
                    stores.append(store_name)
            
            # Build product data
            product_data = {
                'source': 'cimri',
                'url': product_url,
                'name': name,
                'price': price,
                'description': description,
                'image_url': image_url,
                'rating': rating,
                'in_stock': True,
                'raw_category': 'hediyelik-esya',
                'stores': stores,
                'store_count': len(stores)
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
            # Extract first number
            import re
            match = re.search(r'(\d+\.?\d*)', price_text)
            if match:
                return float(match.group(1))
            return 0.0
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
