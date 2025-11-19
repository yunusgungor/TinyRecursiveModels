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
        
        # Check request limit
        if self.request_count >= self.max_requests:
            self.logger.warning(
                f"Daily request limit reached ({self.max_requests}). "
                "Skipping enhancement."
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

    async def enhance_batch(self, products: list, batch_size: int = 10) -> list:
        """
        Enhance a batch of products with parallel processing
        
        Args:
            products: List of RawProductData objects
            batch_size: Number of products to process in parallel
            
        Returns:
            List of enhancement dictionaries
        """
        self.logger.info(f"Starting batch enhancement of {len(products)} products...")
        
        enhanced = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
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
                f"Enhanced {len(enhanced)}/{len(products)} products "
                f"({(len(enhanced)/len(products)*100):.1f}%)"
            )
        
        self.logger.info(f"Batch enhancement complete: {len(enhanced)} products enhanced")
        return enhanced
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get enhancement statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'requests_made': self.request_count,
            'requests_remaining': self.max_requests - self.request_count,
            'max_requests_per_day': self.max_requests
        }
