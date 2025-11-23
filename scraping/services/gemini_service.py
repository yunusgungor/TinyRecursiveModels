"""
Gemini Enhancement Service
Uses Google Gemini API to enhance product data with AI-generated metadata
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
import json

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..utils.models import RawProductData, GeminiEnhancement
from ..utils.cache_manager import CacheManager
from ..utils.similarity import ProductSimilarityDetector


class GeminiEnhancementService:
    """Enhances product data using Gemini API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini enhancement service
        
        Args:
            config: Gemini API configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        
        # Configuration
        self.max_requests = config.get('max_requests_per_day', 1000)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)
        self.timeout = config.get('timeout', 30)
        
        # Smart optimization features
        self.use_cache = config.get('use_cache', True)
        self.smart_batching = config.get('smart_batching', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
        # TRUE batch processing (multiple products per API call)
        self.enable_true_batch = config.get('enable_true_batch', True)
        self.products_per_batch = config.get('products_per_batch', 5)
        
        # Initialize cache manager
        if self.use_cache:
            self.cache_manager = CacheManager()
            self.logger.info("Cache manager enabled")
        else:
            self.cache_manager = None
        
        # Initialize similarity detector
        if self.smart_batching:
            self.similarity_detector = ProductSimilarityDetector(self.similarity_threshold)
            self.logger.info("Smart batching enabled")
        else:
            self.similarity_detector = None
        
        # Log optimization settings
        if self.enable_true_batch:
            self.logger.info(f"TRUE batch processing enabled: {self.products_per_batch} products/call")
        
        # Request tracking
        self.request_count = 0

        # Check if google-generativeai is installed
        if genai is None:
            self.logger.warning("google-generativeai package not installed. AI enhancement will be disabled.")
            self.enabled = False
            return
        
        # Get API key from environment
        api_key_env = config.get('api_key_env', 'GEMINI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            self.logger.warning(f"Gemini API key not found in environment variable: {api_key_env}. AI enhancement will be disabled.")
            self.enabled = False
            return
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            
            # Initialize model
            model_name = config.get('model', 'gemini-1.5-flash')
            self.model = genai.GenerativeModel(model_name)
            
            self.logger.info(f"GeminiEnhancementService initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}. AI enhancement will be disabled.")
            self.enabled = False

    async def enhance_product(self, product: RawProductData) -> Optional[Dict[str, Any]]:
        """
        Enhance a single product with AI-generated metadata
        
        Args:
            product: Raw product data
            
        Returns:
            Enhancement dictionary or None if failed
        """
        # If service is disabled, return fallback enhancement
        if not self.enabled:
            return self._get_fallback_enhancement(product)
        
        # Check cache first
        if self.cache_manager:
            cached_enhancement = self.cache_manager.get_enhancement(
                product.name, 
                product.price,
                getattr(product, 'category', '')
            )
            if cached_enhancement:
                self.logger.debug(f"Using cached enhancement for: {product.name[:30]}...")
                return cached_enhancement
        
        # Check request limit
        if self.request_count >= self.max_requests:
            self.logger.warning(
                f"Daily request limit reached ({self.max_requests}). "
                "Using fallback enhancement."
            )
            return self._get_fallback_enhancement(product)
        
        # Build prompt
        prompt = self._build_prompt(product)
        
        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await self._call_gemini_api(prompt)
                enhancement = self._parse_response(response)
                
                self.request_count += 1
                
                # Cache the result
                if self.cache_manager and enhancement:
                    self.cache_manager.set_enhancement(
                        product.name,
                        product.price,
                        enhancement,
                        getattr(product, 'category', '')
                    )
                
                if self.request_count % 100 == 0:
                    self.logger.info(
                        f"Enhanced {self.request_count} products "
                        f"({self.max_requests - self.request_count} remaining)"
                    )
                
                return enhancement
                
            except Exception as e:
                self.logger.error(
                    f"Gemini API error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        self.logger.error(f"Failed to enhance product after {self.retry_attempts} attempts")
        return self._get_fallback_enhancement(product)
    
    def _build_prompt(self, product: RawProductData) -> str:
        """
        Build enhancement prompt for Gemini
        
        Args:
            product: Raw product data
            
        Returns:
            Formatted prompt string
        """
        prompt_template = self.config.get('enhancement_prompt', '')
        
        return prompt_template.format(
            name=product.name,
            description=product.description,
            price=product.price
        )

    async def _call_gemini_api(self, prompt: str) -> str:
        """
        Call Gemini API asynchronously
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        # Run in thread pool to avoid blocking
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
        
        return response.text
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Gemini API response
        
        Args:
            response: Response text from Gemini
            
        Returns:
            Parsed enhancement dictionary
        """
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Validate with Pydantic model
                enhancement = GeminiEnhancement(**data)
                return enhancement.model_dump()
            
        except Exception as e:
            self.logger.warning(f"Could not parse Gemini response as JSON: {e}")
        
        # Fallback: return default enhancement
        return self._get_fallback_enhancement()
    
    def _get_fallback_enhancement(self, product: RawProductData = None) -> Dict[str, Any]:
        """
        Get fallback enhancement when API fails or is disabled
        Uses rule-based categorization and tag extraction
        
        Args:
            product: Optional product data for smart categorization
            
        Returns:
            Enhancement dictionary
        """
        category = "unknown"
        tags = []
        occasions = ["any"]
        age_range = [18, 65]
        
        if product:
            name_lower = product.name.lower()
            
            # Category detection based on keywords
            category_keywords = {
                "technology": ["laptop", "macbook", "iphone", "tablet", "phone", "bilgisayar", "telefon", "kulaklık", "headphone", "speaker", "hoparlör"],
                "home": ["ev", "home", "mutfak", "kitchen", "yatak", "bed", "koltuk", "sofa", "masa", "table"],
                "beauty": ["makyaj", "makeup", "parfüm", "perfume", "cilt", "skin", "saç", "hair", "fön", "epilasyon", "lazer"],
                "fitness": ["spor", "sport", "fitness", "yoga", "koşu", "running", "gym", "egzersiz", "dumbbell"],
                "kitchen": ["kahve", "coffee", "espresso", "blender", "mikser", "tost", "fırın", "oven", "tencere", "pot"],
                "health": ["sağlık", "health", "terazi", "baskül", "scale", "tansiyon", "termometre", "vitamin"],
                "gaming": ["gaming", "oyun", "game", "konsol", "console", "joystick", "klavye", "keyboard", "mouse"],
                "outdoor": ["kamp", "camp", "outdoor", "çadır", "tent", "sırt çantası", "backpack", "bisiklet", "bike"]
            }
            
            for cat, keywords in category_keywords.items():
                if any(keyword in name_lower for keyword in keywords):
                    category = cat
                    break
            
            # Tag extraction from product name
            tag_keywords = {
                "wireless": ["kablosuz", "wireless", "bluetooth"],
                "portable": ["taşınabilir", "portable", "mini", "kompakt"],
                "professional": ["professional", "profesyonel", "pro"],
                "smart": ["smart", "akıllı", "otomatik", "automatic"],
                "luxury": ["luxury", "lüks", "premium"],
                "eco-friendly": ["organik", "organic", "eco", "doğal", "natural"],
                "digital": ["dijital", "digital", "elektronik", "electronic"],
                "rechargeable": ["şarj edilebilir", "rechargeable", "pil", "battery"]
            }
            
            for tag, keywords in tag_keywords.items():
                if any(keyword in name_lower for keyword in keywords):
                    tags.append(tag)
            
            # Occasion detection
            if category in ["beauty", "health"]:
                occasions = ["birthday", "mothers_day", "valentines_day"]
            elif category == "technology":
                occasions = ["birthday", "graduation", "christmas"]
            elif category == "fitness":
                occasions = ["new_year", "birthday"]
            elif category == "home":
                occasions = ["housewarming", "wedding", "anniversary"]
            
            # Age range based on category
            if category == "gaming":
                age_range = [12, 35]
            elif category == "beauty":
                age_range = [16, 60]
            elif category == "fitness":
                age_range = [18, 55]
        
        return {
            "category": category,
            "target_audience": list(set(tags[:2])) if tags else [],  # Use first 2 unique tags as audience
            "gift_occasions": occasions,
            "emotional_tags": list(set(tags)),  # Remove duplicates
            "age_range": age_range
        }

    async def enhance_batch(self, products: list, batch_size: int = None) -> list:
        """
        Enhance a batch of products with AGGRESSIVE optimization
        
        Optimization strategies:
        1. Cache checking: Skip already enhanced products
        2. Similarity detection: Group similar products
        3. TRUE batch processing: Send multiple products in ONE API call
        
        Args:
            products: List of RawProductData objects
            batch_size: Number of batches to process in parallel (default from config)
            
        Returns:
            List of enhancement dictionaries
        """
        # Use config batch_size if not specified
        if batch_size is None:
            batch_size = self.config.get('batch_size', 1)
        
        batch_delay = self.config.get('batch_delay', 5)
        
        self.logger.info(f"Starting batch enhancement of {len(products)} products...")
        
        # Step 1: Filter out cached products
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
        
        # If all products are cached, return immediately
        if not uncached_products:
            self.logger.info("All products found in cache!")
            return cached_enhancements
        
        # Step 2: Smart batching - group similar products
        if self.smart_batching and self.similarity_detector:
            self.logger.info("Using smart batching to reduce API calls...")
            product_groups = self.similarity_detector.group_similar_products(uncached_products)
            
            # Process only representative products
            products_to_enhance = [
                self.similarity_detector.get_representative_product(group)
                for group in product_groups
            ]
            
            self.logger.info(
                f"Similarity grouping: {len(uncached_products)} → {len(products_to_enhance)} unique products "
                f"(saved {len(uncached_products) - len(products_to_enhance)} API calls)"
            )
        else:
            products_to_enhance = uncached_products
            product_groups = [[p] for p in uncached_products]
        
        # Step 3: TRUE batch processing - send multiple products per API call
        enhanced = []
        
        if self.enable_true_batch and self.products_per_batch > 1:
            self.logger.info(
                f"TRUE batch processing: {self.products_per_batch} products per API call"
            )
            
            # Process products in batches
            for i in range(0, len(products_to_enhance), self.products_per_batch):
                batch = products_to_enhance[i:i + self.products_per_batch]
                
                # Enhance multiple products in ONE API call
                batch_enhancements = await self._enhance_product_batch(batch)
                enhanced.extend(batch_enhancements)
                
                # Progress logging
                self.logger.info(
                    f"Enhanced {len(enhanced)}/{len(products_to_enhance)} unique products "
                    f"({(len(enhanced)/len(products_to_enhance)*100):.1f}%)"
                )
                
                # Wait between batches to avoid rate limits
                if i + self.products_per_batch < len(products_to_enhance):
                    self.logger.info(f"Waiting {batch_delay}s before next batch...")
                    await asyncio.sleep(batch_delay)
        else:
            # Fallback: Process one by one (old method)
            self.logger.info(f"Processing {batch_size} product(s) at a time with {batch_delay}s delay")
            
            for i in range(0, len(products_to_enhance), batch_size):
                batch = products_to_enhance[i:i + batch_size]
                
                # Process batch in parallel
                tasks = [self.enhance_product(p) for p in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out None and exceptions
                for result in results:
                    if result is not None and not isinstance(result, Exception):
                        enhanced.append(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Exception in batch processing: {result}")
                
                # Progress logging
                self.logger.info(
                    f"Enhanced {len(enhanced)}/{len(products_to_enhance)} unique products "
                    f"({(len(enhanced)/len(products_to_enhance)*100):.1f}%)"
                )
                
                # Wait between batches to avoid rate limits
                if i + batch_size < len(products_to_enhance):
                    self.logger.info(f"Waiting {batch_delay}s before next request...")
                    await asyncio.sleep(batch_delay)
        
        # Step 4: If smart batching was used, expand enhancements to all products in groups
        if self.smart_batching and self.similarity_detector and len(product_groups) > 0:
            self.logger.info("Applying enhancements to similar products...")
            all_enhancements = []
            
            for group, enhancement in zip(product_groups, enhanced):
                # Apply same enhancement to all products in the group
                group_enhancements = self.similarity_detector.apply_enhancement_to_group(
                    enhancement, group
                )
                all_enhancements.extend(group_enhancements)
            
            enhanced = all_enhancements
        
        # Combine with cached enhancements
        all_enhancements = cached_enhancements + enhanced
        
        # Save cache
        if self.cache_manager:
            self.cache_manager.save_all()
        
        self.logger.info(f"✅ Batch enhancement complete: {len(all_enhancements)} products enhanced")
        self.logger.info(f"📊 API calls made: {self.request_count} (saved ~{len(products) - self.request_count} calls)")
        
        return all_enhancements
    
    async def _enhance_product_batch(self, products: list) -> list:
        """
        Enhance multiple products in a SINGLE API call
        
        This is the KEY optimization for product enhancement
        
        Args:
            products: List of RawProductData objects
            
        Returns:
            List of enhancement dictionaries
        """
        if not products:
            return []
        
        # Check request limit
        if self.request_count >= self.max_requests:
            self.logger.warning(
                f"Daily request limit reached ({self.max_requests}). "
                "Using fallback enhancement."
            )
            return [self._get_fallback_enhancement(p) for p in products]
        
        # Build batch prompt
        prompt = self._build_batch_prompt(products)
        
        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await self._call_gemini_api(prompt)
                enhancements = self._parse_batch_response(response, len(products))
                
                self.request_count += 1
                
                # Cache all results
                if self.cache_manager:
                    for product, enhancement in zip(products, enhancements):
                        self.cache_manager.set_enhancement(
                            product.name,
                            product.price,
                            enhancement,
                            getattr(product, 'category', '')
                        )
                
                self.logger.info(f"✅ Batch API call successful: {len(enhancements)} products enhanced")
                return enhancements
                
            except Exception as e:
                self.logger.error(
                    f"Gemini API error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed - use fallback for all products
        self.logger.error(f"Failed to enhance batch after {self.retry_attempts} attempts")
        return [self._get_fallback_enhancement(p) for p in products]
    
    def _build_batch_prompt(self, products: list) -> str:
        """Build prompt for batch product enhancement"""
        
        # Build product list
        products_text = ""
        for idx, product in enumerate(products, 1):
            products_text += f"\n{idx}. Name: {product.name}\n"
            products_text += f"   Description: {product.description}\n"
            products_text += f"   Price: {product.price} TL\n"
        
        return f"""Analyze these {len(products)} products and provide structured information for EACH product.

PRODUCTS:
{products_text}

For EACH product, return JSON with:
- category: main category (technology, books, cooking, art, wellness, fitness, outdoor, home, food, experience, gaming, fashion, gardening, beauty, health, kitchen)
- target_audience: list of target demographics
- gift_occasions: suitable occasions (birthday, mothers_day, fathers_day, christmas, valentines_day, graduation, wedding, anniversary, new_year, housewarming, etc.)
- emotional_tags: emotional attributes (relaxing, exciting, practical, luxury, wireless, portable, professional, smart, eco-friendly, digital, rechargeable, etc.)
- age_range: [min_age, max_age]

Return ONLY a valid JSON array with {len(products)} objects:
[
  {{
    "category": "...",
    "target_audience": [...],
    "gift_occasions": [...],
    "emotional_tags": [...],
    "age_range": [min, max]
  }},
  ... ({len(products)} objects total)
]"""
    
    def _parse_batch_response(self, response: str, expected_count: int) -> list:
        """Parse batch response into multiple enhancements"""
        try:
            # Extract JSON array from response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                data_list = json.loads(json_str)
                
                if not isinstance(data_list, list):
                    self.logger.warning("Response is not a list")
                    return []
                
                # Validate each enhancement
                enhancements = []
                for data in data_list:
                    try:
                        enhancement = GeminiEnhancement(**data)
                        enhancements.append(enhancement.model_dump())
                    except Exception as e:
                        self.logger.warning(f"Failed to validate enhancement: {e}")
                        enhancements.append(self._get_fallback_enhancement())
                
                self.logger.info(f"Parsed {len(enhancements)}/{expected_count} enhancements from batch response")
                
                # Pad with fallbacks if needed
                while len(enhancements) < expected_count:
                    enhancements.append(self._get_fallback_enhancement())
                
                return enhancements[:expected_count]
                
        except Exception as e:
            self.logger.warning(f"Failed to parse batch response: {e}")
        
        # Fallback: return default enhancements for all products
        return [self._get_fallback_enhancement() for _ in range(expected_count)]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get enhancement statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'requests_made': self.request_count,
            'requests_remaining': self.max_requests - self.request_count,
            'max_requests_per_day': self.max_requests,
            'smart_batching_enabled': self.smart_batching,
            'cache_enabled': self.use_cache
        }
        
        # Add cache statistics if available
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats.update({
                'cache_stats': cache_stats
            })
        
        return stats
