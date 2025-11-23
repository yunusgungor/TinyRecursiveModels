"""
Ollama Enhancement Service
Uses local Ollama models (specifically gemma3:270m) to enhance product data
"""

import logging
import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List

from ..utils.models import RawProductData, GeminiEnhancement
from ..utils.cache_manager import CacheManager
from ..utils.similarity import ProductSimilarityDetector

class OllamaEnhancementService:
    """Enhances product data using local Ollama API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama enhancement service
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.get('enabled', True)
        
        # Configuration
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'gemma3:270m') # Default fallback
        self.timeout = config.get('timeout', 60)
        
        # Smart optimization features
        self.use_cache = config.get('use_cache', True)
        self.smart_batching = config.get('smart_batching', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.6)
        
        # Batch processing
        self.enable_true_batch = config.get('enable_true_batch', True)
        self.products_per_batch = config.get('products_per_batch', 5)
        
        # Initialize cache manager
        if self.use_cache:
            self.cache_manager = CacheManager()
            self.logger.info(f"Ollama Cache manager enabled (Model: {self.model_name})")
        else:
            self.cache_manager = None
            
        # Initialize similarity detector
        if self.smart_batching:
            self.similarity_detector = ProductSimilarityDetector(self.similarity_threshold)
            self.logger.info("Smart batching enabled for Ollama")
        else:
            self.similarity_detector = None
            
        self.request_count = 0
        self.logger.info(f"OllamaEnhancementService initialized with model: {self.model_name}")

    async def _call_ollama_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call Ollama API via HTTP"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3, # Low temperature for more deterministic JSON
                "num_ctx": 4096
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        self.logger.error(f"Ollama API error: {response.status} - {await response.text()}")
                        return ""
        except Exception as e:
            self.logger.error(f"Failed to call Ollama API: {e}")
            return ""

    async def enhance_batch(self, products: list, batch_size: int = None) -> list:
        """Enhance a batch of products using Ollama"""
        # Similar logic to Gemini service but using Ollama
        
        # 1. Filter cached
        uncached_products = []
        cached_enhancements = []
        
        if self.cache_manager:
            for product in products:
                cached = self.cache_manager.get_enhancement(
                    product.name, product.price, getattr(product, 'category', '')
                )
                if cached:
                    cached_enhancements.append(cached)
                else:
                    uncached_products.append(product)
        else:
            uncached_products = products
            
        if not uncached_products:
            return cached_enhancements

        # 2. Smart batching (Similarity)
        products_to_enhance = uncached_products
        product_groups = []
        
        if self.smart_batching and self.similarity_detector:
            product_groups = self.similarity_detector.group_similar_products(uncached_products)
            products_to_enhance = [
                self.similarity_detector.get_representative_product(group)
                for group in product_groups
            ]
            self.logger.info(f"Ollama: Reduced {len(uncached_products)} to {len(products_to_enhance)} unique products")

        # 3. Process
        enhanced = []
        batch_delay = self.config.get('batch_delay', 1)
        
        # Process in chunks
        chunk_size = self.products_per_batch if self.enable_true_batch else 1
        
        for i in range(0, len(products_to_enhance), chunk_size):
            batch = products_to_enhance[i:i + chunk_size]
            
            if self.enable_true_batch and len(batch) > 1:
                batch_results = await self._enhance_product_batch_ollama(batch)
                enhanced.extend(batch_results)
            else:
                for product in batch:
                    result = await self.enhance_product(product)
                    if result:
                        enhanced.append(result)
            
            if i + chunk_size < len(products_to_enhance):
                await asyncio.sleep(batch_delay)

        # 4. Expand groups
        if self.smart_batching and product_groups:
            all_enhancements = []
            # Map back representative enhancements to groups
            # Note: This logic assumes order is preserved, which it is in the simple loop above
            # But we need to be careful if we had parallel execution. 
            # Here we processed sequentially or in batches, so order matches products_to_enhance
            
            if len(enhanced) != len(products_to_enhance):
                self.logger.warning(f"Mismatch in enhancement count: {len(enhanced)} vs {len(products_to_enhance)}")
                # Fallback logic would go here
            
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
            
        return cached_enhancements + enhanced

    async def enhance_product(self, product: RawProductData) -> Optional[Dict[str, Any]]:
        """Enhance single product"""
        prompt = self._build_prompt(product)
        response_text = await self._call_ollama_api(prompt)
        enhancement = self._parse_response(response_text)
        
        if enhancement and self.cache_manager:
            self.cache_manager.set_enhancement(
                product.name, product.price, enhancement, getattr(product, 'category', '')
            )
            
        return enhancement

    async def _enhance_product_batch_ollama(self, products: list) -> list:
        """Enhance multiple products in one call"""
        prompt = self._build_batch_prompt(products)
        response_text = await self._call_ollama_api(prompt)
        enhancements = self._parse_batch_response(response_text, len(products))
        
        if self.cache_manager:
            for prod, enh in zip(products, enhancements):
                self.cache_manager.set_enhancement(
                    prod.name, prod.price, enh, getattr(prod, 'category', '')
                )
        
        return enhancements

    def _build_prompt(self, product: RawProductData) -> str:
        return f"""Analyze this product and return a JSON object.
Product: {product.name}
Price: {product.price}
Description: {product.description}

Return JSON with: category (string), target_audience (list), gift_occasions (list), emotional_tags (list), age_range (list of 2 numbers).
JSON ONLY:"""

    def _build_batch_prompt(self, products: list) -> str:
        items = "\n".join([f"{i+1}. {p.name} ({p.price} TL)" for i, p in enumerate(products)])
        return f"""Analyze these products and return a JSON ARRAY of objects.
{items}

For EACH, return JSON with: category, target_audience, gift_occasions, emotional_tags, age_range.
Return ONLY the JSON Array."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            # Clean markdown code blocks
            response = response.replace('```json', '').replace('```', '').strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(response[start:end])
        except:
            pass
        return self._get_fallback()

    def _parse_batch_response(self, response: str, count: int) -> list:
        try:
            response = response.replace('```json', '').replace('```', '').strip()
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                data = json.loads(response[start:end])
                if isinstance(data, list):
                    # Pad or trim
                    if len(data) < count:
                        data.extend([self._get_fallback()] * (count - len(data)))
                    return data[:count]
        except:
            pass
        return [self._get_fallback() for _ in range(count)]

    def _get_fallback(self) -> Dict[str, Any]:
        return {
            "category": "general",
            "target_audience": ["anyone"],
            "gift_occasions": ["any"],
            "emotional_tags": ["practical"],
            "age_range": [18, 99]
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model_name,
            "requests_made": self.request_count,
            "max_requests_per_day": "unlimited"
        }
