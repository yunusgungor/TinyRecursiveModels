"""
Cerebras Enhancement Service
Uses Cerebras Cloud API for ultra-fast AI inference
"""

import logging
import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List

from ..utils.models import RawProductData, GeminiEnhancement
from ..utils.cache_manager import CacheManager
from ..utils.similarity import ProductSimilarityDetector


class CerebrasEnhancementService:
    """Enhances product data using Cerebras Cloud API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Cerebras enhancement service
        
        Args:
            config: Cerebras configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.get('enabled', True)
        
        # Configuration
        self.base_url = config.get('base_url', 'https://api.cerebras.ai/v1')
        self.model_name = config.get('model', 'llama3.1-8b')
        self.api_key = config.get('api_key', '')
        self.timeout = config.get('timeout', 30)
        self.max_requests = config.get('max_requests_per_day', 10000)
        
        # Smart optimization features
        self.use_cache = config.get('use_cache', True)
        self.smart_batching = config.get('smart_batching', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.6)
        
        # Batch processing
        self.enable_true_batch = config.get('enable_true_batch', True)
        self.products_per_batch = config.get('products_per_batch', 10)
        
        # Initialize cache manager
        if self.use_cache:
            self.cache_manager = CacheManager()
            self.logger.info(f"Cerebras Cache manager enabled (Model: {self.model_name})")
        else:
            self.cache_manager = None
            
        # Initialize similarity detector
        if self.smart_batching:
            self.similarity_detector = ProductSimilarityDetector(self.similarity_threshold)
            self.logger.info("Smart batching enabled for Cerebras")
        else:
            self.similarity_detector = None
            
        self.request_count = 0
        
        if not self.api_key:
            self.logger.warning("Cerebras API key not found. Service will be disabled.")
            self.enabled = False
        else:
            self.logger.info(f"CerebrasEnhancementService initialized with model: {self.model_name}")

    async def _call_cerebras_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Cerebras Cloud API (OpenAI-compatible)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Cerebras API error: {response.status} - {error_text}")
                        return ""
        except asyncio.TimeoutError:
            self.logger.error(f"Cerebras API timeout after {self.timeout}s")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to call Cerebras API: {e}")
            return ""

    async def enhance_batch(self, products: list, batch_size: int = None) -> list:
        """
        Enhance a batch of products using Cerebras
        
        Args:
            products: List of RawProductData objects
            batch_size: Ignored, uses config batch settings
            
        Returns:
            List of enhancement dictionaries
        """
        if not self.enabled:
            self.logger.warning("Cerebras service is disabled. Using fallback.")
            return [self._get_fallback(p) for p in products]
        
        self.logger.info(f"Starting Cerebras batch enhancement of {len(products)} products...")
        
        # 1. Filter cached products
        uncached_products = []
        cached_enhancements = []
        
        if self.cache_manager:
            for product in products:
                cached = self.cache_manager.get_enhancement(
                    product.name, 
                    product.price, 
                    getattr(product, 'category', '')
                )
                if cached:
                    cached_enhancements.append(cached)
                else:
                    uncached_products.append(product)
            
            self.logger.info(
                f"Cache hits: {len(cached_enhancements)}/{len(products)} "
                f"(saved {len(cached_enhancements)} API calls)"
            )
        else:
            uncached_products = products
            
        if not uncached_products:
            self.logger.info("All products found in cache!")
            return cached_enhancements

        # 2. Smart batching (Similarity grouping)
        products_to_enhance = uncached_products
        product_groups = []
        
        if self.smart_batching and self.similarity_detector:
            product_groups = self.similarity_detector.group_similar_products(uncached_products)
            products_to_enhance = [
                self.similarity_detector.get_representative_product(group)
                for group in product_groups
            ]
            self.logger.info(
                f"Similarity grouping: {len(uncached_products)} → {len(products_to_enhance)} unique products "
                f"(saved {len(uncached_products) - len(products_to_enhance)} API calls)"
            )

        # 3. Process in batches
        enhanced = []
        batch_delay = self.config.get('batch_delay', 1)
        chunk_size = self.products_per_batch if self.enable_true_batch else 1
        
        for i in range(0, len(products_to_enhance), chunk_size):
            batch = products_to_enhance[i:i + chunk_size]
            
            if self.enable_true_batch and len(batch) > 1:
                batch_results = await self._enhance_product_batch(batch)
                enhanced.extend(batch_results)
            else:
                for product in batch:
                    result = await self.enhance_product(product)
                    if result:
                        enhanced.append(result)
            
            self.logger.info(
                f"Enhanced {len(enhanced)}/{len(products_to_enhance)} unique products "
                f"({(len(enhanced)/len(products_to_enhance)*100):.1f}%)"
            )
            
            if i + chunk_size < len(products_to_enhance):
                await asyncio.sleep(batch_delay)

        # 4. Expand groups if smart batching was used
        if self.smart_batching and product_groups:
            self.logger.info("Applying enhancements to similar products...")
            all_enhancements = []
            
            for i, group in enumerate(product_groups):
                if i < len(enhanced):
                    group_enhancements = self.similarity_detector.apply_enhancement_to_group(
                        enhanced[i], group
                    )
                    all_enhancements.extend(group_enhancements)
            
            enhanced = all_enhancements

        # 5. Save cache
        if self.cache_manager:
            self.cache_manager.save_all()
            
        all_enhancements = cached_enhancements + enhanced
        
        self.logger.info(f"✅ Cerebras batch enhancement complete: {len(all_enhancements)} products enhanced")
        self.logger.info(f"📊 API calls made: {self.request_count}")
        
        return all_enhancements

    async def enhance_product(self, product: RawProductData) -> Optional[Dict[str, Any]]:
        """Enhance single product"""
        if not self.enabled:
            return self._get_fallback(product)
        
        if self.request_count >= self.max_requests:
            self.logger.warning(f"Daily request limit reached ({self.max_requests})")
            return self._get_fallback(product)
        
        messages = [
            {
                "role": "system",
                "content": "You are a product categorization expert. Return only valid JSON."
            },
            {
                "role": "user",
                "content": self._build_prompt(product)
            }
        ]
        
        response_text = await self._call_cerebras_api(messages)
        enhancement = self._parse_response(response_text)
        
        self.request_count += 1
        
        if enhancement and self.cache_manager:
            self.cache_manager.set_enhancement(
                product.name, 
                product.price, 
                enhancement, 
                getattr(product, 'category', '')
            )
            
        return enhancement

    async def _enhance_product_batch(self, products: list) -> list:
        """Enhance multiple products in one API call"""
        if not self.enabled:
            return [self._get_fallback(p) for p in products]
        
        if self.request_count >= self.max_requests:
            self.logger.warning(f"Daily request limit reached ({self.max_requests})")
            return [self._get_fallback(p) for p in products]
        
        messages = [
            {
                "role": "system",
                "content": "You are a product categorization expert. Return only valid JSON arrays."
            },
            {
                "role": "user",
                "content": self._build_batch_prompt(products)
            }
        ]
        
        response_text = await self._call_cerebras_api(messages)
        enhancements = self._parse_batch_response(response_text, len(products))
        
        self.request_count += 1
        
        if self.cache_manager:
            for prod, enh in zip(products, enhancements):
                self.cache_manager.set_enhancement(
                    prod.name, 
                    prod.price, 
                    enh, 
                    getattr(prod, 'category', '')
                )
        
        self.logger.info(f"✅ Cerebras batch API call successful: {len(enhancements)} products enhanced")
        return enhancements

    def _build_prompt(self, product: RawProductData) -> str:
        """Build prompt for single product enhancement"""
        return f"""Analyze this product and return a JSON object.

Product: {product.name}
Price: {product.price} TL
Description: {product.description}

Return JSON with:
- category: main category (technology, books, cooking, art, wellness, fitness, outdoor, home, food, experience, gaming, fashion, gardening, beauty, health, kitchen)
- target_audience: list of target demographics
- gift_occasions: suitable occasions (birthday, mothers_day, fathers_day, christmas, valentines_day, graduation, wedding, anniversary, new_year, housewarming, etc.)
- emotional_tags: emotional attributes (relaxing, exciting, practical, luxury, wireless, portable, professional, smart, eco-friendly, digital, rechargeable, etc.)
- age_range: [min_age, max_age]

Return ONLY valid JSON, no markdown:"""

    def _build_batch_prompt(self, products: list) -> str:
        """Build prompt for batch product enhancement"""
        items = "\n".join([
            f"{i+1}. Name: {p.name}\n   Price: {p.price} TL\n   Description: {p.description}"
            for i, p in enumerate(products)
        ])
        
        return f"""Analyze these {len(products)} products and return a JSON ARRAY.

PRODUCTS:
{items}

For EACH product, return JSON with:
- category: main category (technology, books, cooking, art, wellness, fitness, outdoor, home, food, experience, gaming, fashion, gardening, beauty, health, kitchen)
- target_audience: list of target demographics
- gift_occasions: suitable occasions
- emotional_tags: emotional attributes
- age_range: [min_age, max_age]

Return ONLY a valid JSON array with {len(products)} objects, no markdown:
[{{"category": "...", "target_audience": [...], "gift_occasions": [...], "emotional_tags": [...], "age_range": [min, max]}}, ...]"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse single product response"""
        try:
            # Clean markdown code blocks
            response = response.replace('```json', '').replace('```', '').strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Validate with Pydantic model
                enhancement = GeminiEnhancement(**data)
                return enhancement.model_dump()
        except Exception as e:
            self.logger.warning(f"Failed to parse Cerebras response: {e}")
        
        return self._get_fallback()

    def _parse_batch_response(self, response: str, count: int) -> list:
        """Parse batch response into multiple enhancements"""
        try:
            # Clean markdown code blocks
            response = response.replace('```json', '').replace('```', '').strip()
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    enhancements = []
                    for item in data:
                        try:
                            enhancement = GeminiEnhancement(**item)
                            enhancements.append(enhancement.model_dump())
                        except:
                            enhancements.append(self._get_fallback())
                    
                    # Pad or trim to expected count
                    while len(enhancements) < count:
                        enhancements.append(self._get_fallback())
                    
                    return enhancements[:count]
        except Exception as e:
            self.logger.warning(f"Failed to parse Cerebras batch response: {e}")
        
        return [self._get_fallback() for _ in range(count)]

    def _get_fallback(self, product: RawProductData = None) -> Dict[str, Any]:
        """Get fallback enhancement when API fails"""
        return {
            "category": "general",
            "target_audience": ["anyone"],
            "gift_occasions": ["any"],
            "emotional_tags": ["practical"],
            "age_range": [18, 99]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics"""
        stats = {
            "provider": "cerebras",
            "model": self.model_name,
            "requests_made": self.request_count,
            "requests_remaining": self.max_requests - self.request_count,
            "max_requests_per_day": self.max_requests,
            "smart_batching_enabled": self.smart_batching,
            "cache_enabled": self.use_cache
        }
        
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats.update({"cache_stats": cache_stats})
        
        return stats
