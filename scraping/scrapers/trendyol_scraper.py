"""
Trendyol Scraper
Scrapes product data from trendyol.com
"""

from typing import List, Dict, Any
from playwright.async_api import Page
import re
import asyncio

from .base_scraper import BaseScraper


class TrendyolScraper(BaseScraper):
    """Scraper for trendyol.com"""
    
    # Site-specific selectors (to be updated based on actual site structure)
    SELECTORS = {
        'product_list': '.p-card-wrppr',
        'product_link': '.p-card-wrppr a',
        'product_name': 'h1',  # Updated: class pr-new-br no longer used
        'product_price': 'div.product-price-container span.prc-dsc',
        'product_description': 'div.detail-border-container',
        'product_image': 'div.gallery-container img.ph-gl-img',
        'product_rating': 'div.pr-rnr-sm-p > span',
        'next_page': 'a.ty-page-i.ty-page-i-next',
        'product_card_alternatives': [
            '.p-card-wrppr',
            '.product-card',
            '.p-card-ch-item-wrapper',
            'div[class*="product-item"]'
        ]
    }
    
    # Category to search URL mapping
    CATEGORY_URLS = {
        'elektronik': 'https://www.trendyol.com/sr?wc=104024&sst=BEST_SELLER',
        'ev-yasam': 'https://www.trendyol.com/sr?wc=1354,94,1365,95,101514,104166,104216&sst=BEST_SELLER',
        'kozmetik': 'https://www.trendyol.com/sr?wc=86,1180,109363,143992,1346,143835,1347&sst=BEST_SELLER',
        'kadin-giyim': 'https://www.trendyol.com/sr?wc=82&wg=1&sst=BEST_SELLER',
        'erkek-giyim': 'https://www.trendyol.com/sr?wc=82&wg=2&sst=BEST_SELLER',
        'anne-cocuk': 'https://www.trendyol.com/sr?wc=83&sst=BEST_SELLER',
        'ayakkabi-canta': 'https://www.trendyol.com/sr?wc=1,2&sst=BEST_SELLER',
        'supermarket': 'https://www.trendyol.com/sr?wc=103799&sst=BEST_SELLER',
        'mobilya': 'https://www.trendyol.com/sr?wc=1119&sst=BEST_SELLER',
        'spor-outdoor': 'https://www.trendyol.com/sr?wc=73,1172,120,101484,119,101426,110,1181,144727,115,1174,66,111,101457,103687,104224,1020,65&sst=BEST_SELLER',
        'kitap-kirtasiye': 'https://www.trendyol.com/sr?wc=97,91,104125,105777,108934&sst=BEST_SELLER'
    }
    
    def __init__(self, config: Dict[str, Any], rate_limiter):
        """Initialize Trendyol scraper"""
        super().__init__(config, rate_limiter)
        self.base_url = config.get('url', 'https://www.trendyol.com')
        self.categories = config.get('categories', ['elektronik'])
        self.max_products = config.get('max_products', 500)
    
    async def scrape_products(self, max_products: int) -> List[Dict[str, Any]]:
        """Scrape products from Trendyol"""
        self.logger.info(f"Starting to scrape up to {max_products} products from Trendyol")
        
        all_products = []
        
        for category in self.categories:
            if len(all_products) >= max_products:
                break
            
            self.logger.info(f"Scraping category: {category}")
            category_products = await self._scrape_category(category, max_products - len(all_products))
            all_products.extend(category_products)
        
        self.logger.info(f"Scraped {len(all_products)} products from Trendyol")
        return all_products

    async def _scrape_category(self, category: str, max_products: int) -> List[Dict[str, Any]]:
        """Scrape products from a specific category"""
        products = []
        page = await self.create_page()
        
        try:
            # Get category URL from mapping
            category_url = self.CATEGORY_URLS.get(category)
            if not category_url:
                self.logger.error(f"Unknown category: {category}")
                return products
            
            success = await self.safe_goto(page, category_url)
            if not success:
                self.logger.error(f"Failed to load category: {category}")
                return products
            
            product_urls = await self._extract_product_urls(page, max_products)
            self.logger.info(f"Found {len(product_urls)} product URLs in {category}")
            
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
        """Extract product URLs from category listing page"""
        urls = []
        current_page = 1
        max_retries = 3  # Limit retry attempts
        retry_count = 0
        
        while len(urls) < max_urls and retry_count < max_retries:
            try:
                # Scroll down to trigger lazy loading
                for _ in range(5):
                    await page.keyboard.press('PageDown')
                    await asyncio.sleep(0.5)
                
                # Wait for products to load - try multiple selectors
                found_selector = None
                for selector in self.SELECTORS['product_card_alternatives']:
                    try:
                        if await page.query_selector(selector):
                            found_selector = selector
                            break
                    except:
                        continue
                
                if not found_selector:
                    # Try waiting for the primary one as a fallback
                    try:
                        await page.wait_for_selector(self.SELECTORS['product_list'], timeout=15000)
                        found_selector = self.SELECTORS['product_list']
                    except:
                        retry_count += 1
                        self.logger.warning(f"Timeout waiting for product list (attempt {retry_count}/{max_retries})")
                        if retry_count >= max_retries:
                            self.logger.error(f"Failed to find products after {max_retries} attempts. Category may be blocked or selectors outdated.")
                            break
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(2)
                        continue

                # Extract links using the found selector
                # We assume the link is an 'a' tag inside the card or the card itself is an 'a' tag
                cards = await page.query_selector_all(found_selector)
                
                for card in cards:
                    if len(urls) >= max_urls:
                        break
                    
                    # Try to find 'a' tag inside
                    link = await card.query_selector('a')
                    if not link:
                        # Maybe the card itself is the link
                        tag_name = await card.evaluate('el => el.tagName')
                        if tag_name == 'A':
                            link = card
                    
                    if link:
                        href = await link.get_attribute('href')
                        if href:
                            if href.startswith('/'):
                                href = self.base_url + href
                            urls.append(href)
                
                self.logger.info(f"Found {len(urls)} URLs using selector: {found_selector}")
                
                # If we still don't have enough URLs, try next page
                if len(urls) < max_urls:
                    next_button = await page.query_selector(self.SELECTORS['next_page'])
                    if not next_button:
                        break
                    
                    await next_button.click()
                    await page.wait_for_load_state('networkidle')
                    current_page += 1
                else:
                    break
                
            except Exception as e:
                self.logger.warning(f"Error extracting URLs on page {current_page}: {e}")
                break
        
        return urls[:max_urls]

    async def extract_product_details(self, page: Page, product_url: str) -> Dict[str, Any]:
        """Extract detailed product information from product page"""
        try:
            success = await self.safe_goto(page, product_url)
            if not success:
                return None
            
            # Wait a bit for dynamic content to load
            await asyncio.sleep(1)
            
            # Extract product name
            name_element = await page.query_selector(self.SELECTORS['product_name'])
            name = await name_element.inner_text() if name_element else "Unknown Product"
            name = name.strip()
            
            # Extract price - try multiple selectors
            price = 0.0
            price_selectors = [
                'span.prc-dsc',  # Discounted price
                'span.prc-slg',  # Regular price
                '.product-price span',  # Generic price
                '[data-price]'  # Data attribute
            ]
            
            for price_selector in price_selectors:
                try:
                    price_element = await page.query_selector(price_selector)
                    if price_element:
                        price_text = await price_element.inner_text()
                        price = self._parse_price(price_text)
                        if price > 0:
                            self.logger.debug(f"Found price {price} using selector: {price_selector}")
                            break
                except:
                    continue
            
            # If still no price, try to get from page text
            if price == 0:
                try:
                    page_text = await page.content()
                    # Look for price patterns in TL
                    import re
                    price_match = re.search(r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*TL', page_text)
                    if price_match:
                        price = self._parse_price(price_match.group(1))
                        self.logger.debug(f"Found price {price} from page content")
                except:
                    pass
            
            # Extract description
            desc_element = await page.query_selector(self.SELECTORS['product_description'])
            description = await desc_element.inner_text() if desc_element else name
            description = description.strip()
            
            # Extract image
            img_element = await page.query_selector(self.SELECTORS['product_image'])
            if not img_element:
                # Try alternative selectors
                img_element = await page.query_selector('img.ph-gl-img, .gallery img, .product-image img')
            image_url = await img_element.get_attribute('src') if img_element else None
            
            # Extract rating
            rating = 0.0
            rating_element = await page.query_selector(self.SELECTORS['product_rating'])
            if rating_element:
                rating_text = await rating_element.inner_text()
                rating = self._parse_rating(rating_text)
            
            product_data = {
                'source': 'trendyol',
                'url': product_url,
                'name': name,
                'price': price,
                'description': description,
                'image_url': image_url,
                'rating': rating,
                'in_stock': True,
                'raw_category': self.categories[0] if self.categories else 'genel'
            }
            
            self.logger.debug(f"Extracted product: {name[:50]}... Price: {price} TL")
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Error extracting product details from {product_url}: {e}")
            return None
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price from text"""
        try:
            price_text = price_text.replace('TL', '').replace('₺', '').strip()
            price_text = price_text.replace('.', '').replace(',', '.')
            return float(price_text)
        except:
            self.logger.warning(f"Could not parse price: {price_text}")
            return 0.0
    
    def _parse_rating(self, rating_text: str) -> float:
        """Parse rating from text"""
        try:
            rating_text = rating_text.replace(',', '.').strip()
            match = re.search(r'(\d+\.?\d*)', rating_text)
            if match:
                rating = float(match.group(1))
                return min(5.0, max(0.0, rating))
            return 0.0
        except:
            self.logger.warning(f"Could not parse rating: {rating_text}")
            return 0.0
