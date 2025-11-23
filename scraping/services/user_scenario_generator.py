"""
User Scenario Generator
Generates realistic user scenarios from gift catalog data using Gemini API
"""

import json
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import aiohttp

from ..utils.cache_manager import CacheManager


class UserScenarioGenerator:
    """Generates realistic user scenarios for gift recommendation testing"""
    
    def __init__(self, config: Dict[str, Any], gift_catalog_path: str, ollama_config: Dict[str, Any] = None):
        """
        Initialize user scenario generator
        
        Args:
            config: Gemini API configuration
            gift_catalog_path: Path to gift catalog JSON file
            ollama_config: Optional Ollama configuration
        """
        self.config = config
        self.ollama_config = ollama_config or {}
        self.use_ollama = self.ollama_config.get('enabled', False)
        
        self.gift_catalog_path = gift_catalog_path
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        
        # Load gift catalog
        self.gift_catalog = self._load_gift_catalog()
        
        # Extract dynamic keywords from catalog
        self._extract_dynamic_keywords()
        
        # Configuration
        self.max_requests = config.get('max_requests_per_day', 1000)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Optimization parameters
        self.batch_size = config.get('scenario_batch_size', 10)  # Generate N scenarios per API call
        self.ai_ratio = config.get('scenario_ai_ratio', 0.2)  # Use AI for X% of scenarios
        
        # Cache manager
        self.use_cache = config.get('scenario_use_cache', True)
        if self.use_cache:
            self.cache_manager = CacheManager()
            self.logger.info(f"Cache manager enabled for scenarios (batch_size={self.batch_size}, ai_ratio={self.ai_ratio})")
        else:
            self.cache_manager = None
        
        # Request tracking
        self.request_count = 0

        if self.use_ollama:
            self.model_name = self.ollama_config.get('model', 'gemma3:270m')
            self.base_url = self.ollama_config.get('base_url', 'http://localhost:11434')
            self.logger.info(f"UserScenarioGenerator initialized with Ollama model: {self.model_name}")
        else:
            # Check if google-generativeai is installed
            if genai is None:
                self.logger.warning("google-generativeai not installed. Using fallback generation.")
                self.enabled = False
                return
            
            # Get API key
            api_key_env = config.get('api_key_env', 'GEMINI_API_KEY')
            api_key = os.getenv(api_key_env)
            
            if not api_key:
                self.logger.warning(f"Gemini API key not found. Using fallback generation.")
                self.enabled = False
                return
            
            # Configure Gemini
            try:
                genai.configure(api_key=api_key)
                model_name = config.get('model', 'gemini-2.0-flash')
                self.model = genai.GenerativeModel(model_name)
                self.logger.info(f"UserScenarioGenerator initialized with model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini: {e}")
                self.enabled = False

    def _load_gift_catalog(self) -> Dict[str, Any]:
        """Load gift catalog from JSON file"""
        try:
            with open(self.gift_catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            self.logger.info(f"Loaded {len(catalog.get('gifts', []))} gifts from catalog")
            return catalog
        except Exception as e:
            self.logger.error(f"Failed to load gift catalog: {e}")
            return {"gifts": [], "metadata": {}}
    
    def _extract_dynamic_keywords(self) -> None:
        """
        Extract dynamic keywords from gift catalog tags and categories
        Categorizes tags into trend, tech, quality, and value keywords
        """
        gifts = self.gift_catalog.get('gifts', [])
        categories = self.gift_catalog.get('metadata', {}).get('categories', [])
        
        # Collect all tags from catalog
        all_tags = set()
        for gift in gifts:
            all_tags.update(gift.get('tags', []))
        
        # Convert to lowercase for matching
        all_tags_lower = {tag.lower() for tag in all_tags}
        categories_lower = {cat.lower() for cat in categories}
        
        # Define base patterns for categorization
        trend_patterns = ['trendy', 'modern', 'innovative', 'new', 'latest', 'smart', 
                         'digital', 'tech', 'advanced', 'contemporary']
        tech_patterns = ['technology', 'electronic', 'digital', 'smart', 'wireless',
                        'bluetooth', 'usb', 'gaming', 'computer', 'phone', 'gadget']
        quality_patterns = ['luxury', 'premium', 'quality', 'expensive', 'high-end',
                           'designer', 'exclusive', 'elegant', 'sophisticated']
        value_patterns = ['affordable', 'cheap', 'discount', 'deal', 'value', 
                         'budget', 'economical', 'bargain']
        
        # Match tags with patterns
        self.trend_keywords = []
        self.tech_keywords = []
        self.quality_keywords = []
        self.value_keywords = []
        
        for tag in all_tags_lower:
            # Check trend patterns
            if any(pattern in tag for pattern in trend_patterns):
                self.trend_keywords.append(tag)
            # Check tech patterns
            if any(pattern in tag for pattern in tech_patterns):
                self.tech_keywords.append(tag)
            # Check quality patterns
            if any(pattern in tag for pattern in quality_patterns):
                self.quality_keywords.append(tag)
            # Check value patterns
            if any(pattern in tag for pattern in value_patterns):
                self.value_keywords.append(tag)
        
        # Add relevant categories to tech keywords
        tech_categories = ['technology', 'electronics', 'gaming', 'digital']
        for cat in categories_lower:
            if any(tech_cat in cat for tech_cat in tech_categories):
                self.tech_keywords.append(cat)
        
        # Remove duplicates and ensure we have fallbacks
        self.trend_keywords = list(set(self.trend_keywords)) or ['modern', 'trendy']
        self.tech_keywords = list(set(self.tech_keywords)) or ['technology', 'electronic']
        self.quality_keywords = list(set(self.quality_keywords)) or ['premium', 'quality']
        self.value_keywords = list(set(self.value_keywords)) or ['affordable', 'value']
        
        self.logger.info(f"Extracted dynamic keywords:")
        self.logger.info(f"  Trend: {len(self.trend_keywords)} keywords")
        self.logger.info(f"  Tech: {len(self.tech_keywords)} keywords")
        self.logger.info(f"  Quality: {len(self.quality_keywords)} keywords")
        self.logger.info(f"  Value: {len(self.value_keywords)} keywords")

    async def generate_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """
        Generate user scenarios with optimized API usage
        
        Optimization strategies:
        1. Batch processing: Generate multiple scenarios per API call
        2. AI ratio: Use AI for only a percentage of scenarios
        3. Caching: Reuse similar scenario profiles
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of user scenario dictionaries
        """
        self.logger.info(f"Generating {num_scenarios} user scenarios...")
        self.logger.info(f"Optimization: batch_size={self.batch_size}, ai_ratio={self.ai_ratio*100:.0f}%")
        
        # Get delay from config
        batch_delay = self.config.get('batch_delay', 5)
        
        scenarios = []
        
        # Extract real data from catalog
        categories = self._extract_categories()
        tags = self._extract_tags()
        occasions = self._extract_occasions()
        price_ranges = self._extract_price_ranges()
        
        self.logger.info(f"Extracted from catalog: {len(categories)} categories, {len(tags)} tags, {len(occasions)} occasions")
        
        # Calculate how many scenarios to generate with AI vs fallback
        num_ai_scenarios = int(num_scenarios * self.ai_ratio)
        num_fallback_scenarios = num_scenarios - num_ai_scenarios
        
        self.logger.info(f"Strategy: {num_ai_scenarios} AI scenarios, {num_fallback_scenarios} fallback scenarios")
        
        # Generate AI scenarios in batches
        if self.enabled and num_ai_scenarios > 0 and self.request_count < self.max_requests:
            num_batches = (num_ai_scenarios + self.batch_size - 1) // self.batch_size
            self.logger.info(f"Generating {num_ai_scenarios} AI scenarios in {num_batches} batches...")
            
            for batch_idx in range(num_batches):
                # Calculate scenarios for this batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_ai_scenarios)
                batch_count = end_idx - start_idx
                
                # Generate batch
                batch_scenarios = await self._generate_ai_scenario_batch(
                    start_idx, batch_count, categories, tags, occasions, price_ranges
                )
                
                if batch_scenarios:
                    scenarios.extend(batch_scenarios)
                    self.logger.info(f"Batch {batch_idx + 1}/{num_batches}: Generated {len(batch_scenarios)} scenarios")
                
                # Wait between batches to avoid rate limits
                if batch_idx < num_batches - 1:
                    await asyncio.sleep(batch_delay)
        
        # Generate remaining scenarios with fallback
        self.logger.info(f"Generating {num_fallback_scenarios} fallback scenarios...")
        for i in range(num_fallback_scenarios):
            scenario = self._generate_fallback_scenario(
                len(scenarios) + i, categories, tags, occasions, price_ranges
            )
            if scenario:
                scenarios.append(scenario)
        
        self.logger.info(f"Scenario generation complete: {len(scenarios)} scenarios")
        self.logger.info(f"API calls made: {self.request_count} (saved ~{num_scenarios - self.request_count} calls)")
        
        return scenarios

    def _extract_categories(self) -> List[str]:
        """Extract unique categories from gift catalog"""
        gifts = self.gift_catalog.get('gifts', [])
        categories = list(set(gift.get('category', 'unknown') for gift in gifts))
        return [cat for cat in categories if cat != 'unknown']
    
    def _extract_tags(self) -> List[str]:
        """Extract unique tags from gift catalog"""
        gifts = self.gift_catalog.get('gifts', [])
        all_tags = []
        for gift in gifts:
            tags = gift.get('tags', [])
            all_tags.extend(tags)
        return list(set(all_tags))
    
    def _extract_occasions(self) -> List[str]:
        """Extract unique occasions from gift catalog"""
        gifts = self.gift_catalog.get('gifts', [])
        all_occasions = []
        for gift in gifts:
            occasions = gift.get('occasions', [])
            all_occasions.extend(occasions)
        return list(set(all_occasions))
    
    def _extract_price_ranges(self) -> Dict[str, tuple]:
        """Extract price ranges from gift catalog"""
        gifts = self.gift_catalog.get('gifts', [])
        prices = [gift.get('price', 0) for gift in gifts if gift.get('price', 0) > 0]
        
        if not prices:
            return {
                'low': (0, 100),
                'medium': (100, 200),
                'high': (200, 300),
                'premium': (300, 500)
            }
        
        min_price = min(prices)
        max_price = max(prices)
        range_size = (max_price - min_price) / 4
        
        return {
            'low': (min_price, min_price + range_size),
            'medium': (min_price + range_size, min_price + 2 * range_size),
            'high': (min_price + 2 * range_size, min_price + 3 * range_size),
            'premium': (min_price + 3 * range_size, max_price)
        }

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        self.logger.error(f"Ollama API error: {response.status}")
                        return ""
        except Exception as e:
            self.logger.error(f"Ollama connection error: {e}")
            return ""

    async def _generate_ai_scenario_batch(self, start_index: int, batch_count: int,
                                          categories: List[str], tags: List[str], 
                                          occasions: List[str], price_ranges: Dict[str, tuple]) -> List[Dict[str, Any]]:
        """Generate multiple scenarios in a single API call (Gemini or Ollama)"""
        
        # Build batch prompt
        prompt = self._build_batch_scenario_prompt(batch_count, categories, tags, occasions, price_ranges)
        
        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                if self.use_ollama:
                    response_text = await self._call_ollama(prompt)
                else:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt
                    )
                    response_text = response.text
                
                scenarios = self._parse_batch_scenario_response(response_text, start_index, batch_count)
                self.request_count += 1
                
                if scenarios:
                    self.logger.info(f"✅ Batch API call successful: {len(scenarios)} scenarios generated")
                    return scenarios
                
            except Exception as e:
                self.logger.error(f"AI Generation error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # Fallback
        self.logger.warning(f"Batch generation failed, using fallback for {batch_count} scenarios")
        fallback_scenarios = []
        for i in range(batch_count):
            scenario = self._generate_fallback_scenario(
                start_index + i, categories, tags, occasions, price_ranges
            )
            if scenario:
                fallback_scenarios.append(scenario)
        
        return fallback_scenarios

    async def _generate_ai_scenario(self, index: int, categories: List[str], 
                                   tags: List[str], occasions: List[str], 
                                   price_ranges: Dict[str, tuple]) -> Optional[Dict[str, Any]]:
        """Generate scenario using AI (Gemini or Ollama)"""
        
        # Build prompt
        prompt = self._build_scenario_prompt(categories, tags, occasions, price_ranges)
        
        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                if self.use_ollama:
                    response_text = await self._call_ollama(prompt)
                else:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt
                    )
                    response_text = response.text
                
                scenario = self._parse_scenario_response(response_text, index)
                self.request_count += 1
                return scenario
                
            except Exception as e:
                self.logger.error(f"AI Generation error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # Fallback
        return self._generate_fallback_scenario(index, categories, tags, occasions, price_ranges)

    def _build_scenario_prompt(self, categories: List[str], tags: List[str], 
                              occasions: List[str], price_ranges: Dict[str, tuple]) -> str:
        """Build prompt for scenario generation"""
        
        # Sample from available data
        sample_categories = random.sample(categories, min(10, len(categories)))
        sample_tags = random.sample(tags, min(15, len(tags)))
        sample_occasions = occasions if len(occasions) <= 10 else random.sample(occasions, 10)
        
        # Get price range info
        price_info = f"{price_ranges['low'][0]:.0f}-{price_ranges['premium'][1]:.0f}"
        
        return f"""Generate a realistic user profile for gift recommendation testing based on REAL scraped data.

REAL Available Categories: {', '.join(sample_categories)}
REAL Available Tags/Interests: {', '.join(sample_tags)}
REAL Available Occasions: {', '.join(sample_occasions)}
REAL Price Range: {price_info} TL

Create a diverse, realistic person with:
- Age (16-70)
- 2-4 hobbies/interests ONLY from the available tags above
- Relationship to gift recipient (friend, mother, father, sister, brother, colleague, spouse, uncle, aunt, cousin, grandparent)
- Budget in TL (within the real price range above)
- Occasion ONLY from the available occasions above
- 2-4 preferences that match the available tags (e.g., if "luxury" tag exists, use it)

Return ONLY valid JSON:
{{
  "age": <number>,
  "hobbies": [<list of 2-4 hobbies from available tags>],
  "relationship": "<relationship>",
  "budget": <number within real price range>,
  "occasion": "<occasion from available occasions>",
  "preferences": [<list of 2-4 preferences from available tags>]
}}"""

    def _build_batch_scenario_prompt(self, batch_count: int, categories: List[str], tags: List[str], 
                              occasions: List[str], price_ranges: Dict[str, tuple]) -> str:
        """Build prompt for batch scenario generation"""
        
        # Sample from available data (larger sample for batch)
        sample_categories = random.sample(categories, min(20, len(categories)))
        sample_tags = random.sample(tags, min(30, len(tags)))
        sample_occasions = occasions if len(occasions) <= 15 else random.sample(occasions, 15)
        
        # Get price range info
        price_info = f"{price_ranges['low'][0]:.0f}-{price_ranges['premium'][1]:.0f}"
        
        return f"""Generate {batch_count} REALISTIC and DIVERSE user profiles for gift recommendation testing based on REAL scraped data.

REAL Available Categories: {', '.join(sample_categories)}
REAL Available Tags/Interests: {', '.join(sample_tags)}
REAL Available Occasions: {', '.join(sample_occasions)}
REAL Price Range: {price_info} TL

Create {batch_count} different people with diverse ages, budgets, and needs.
For each person:
- Age (16-70)
- 2-4 hobbies/interests ONLY from the available tags above
- Relationship to gift recipient
- Budget in TL (within the real price range above)
- Occasion ONLY from the available occasions above
- 2-4 preferences that match the available tags

Return ONLY a valid JSON ARRAY of objects:
[
  {{
    "age": <number>,
    "hobbies": [<list of 2-4 hobbies>],
    "relationship": "<relationship>",
    "budget": <number>,
    "occasion": "<occasion>",
    "preferences": [<list of 2-4 preferences>]
  }},
  ...
]"""

    def _parse_batch_scenario_response(self, response: str, start_index: int, expected_count: int) -> List[Dict[str, Any]]:
        """Parse batch response into scenario list"""
        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                profiles = json.loads(json_str)
                
                if not isinstance(profiles, list):
                    self.logger.warning("Batch response is not a list")
                    return []
                
                scenarios = []
                for i, profile in enumerate(profiles):
                    # Build scenario
                    scenario = {
                        "id": f"scenario_{start_index + i:04d}",
                        "profile": profile,
                        "expected_categories": self._infer_categories(profile),
                        "expected_tools": self._infer_tools(profile)
                    }
                    scenarios.append(scenario)
                
                return scenarios
        except Exception as e:
            self.logger.warning(f"Failed to parse batch scenario response: {e}")
        
        return []


    def _parse_scenario_response(self, response: str, index: int) -> Optional[Dict[str, Any]]:
        """Parse Gemini response into scenario format"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                profile = json.loads(json_str)
                
                # Build scenario
                scenario = {
                    "id": f"scenario_{index:04d}",
                    "profile": profile,
                    "expected_categories": self._infer_categories(profile),
                    "expected_tools": self._infer_tools(profile)
                }
                
                return scenario
        except Exception as e:
            self.logger.warning(f"Failed to parse scenario response: {e}")
        
        return None

    def _generate_fallback_scenario(self, index: int, categories: List[str], 
                                   tags: List[str], occasions: List[str], 
                                   price_ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """Generate scenario using rule-based approach with REAL scraped data"""
        
        # Use REAL tags from scraped data as hobbies/preferences
        available_tags = tags if tags else [
            "technology", "photography", "gaming", "music", "art", "reading",
            "fitness", "sports", "outdoor", "wellness", "home_decor", "cooking"
        ]
        
        # Use REAL occasions from scraped data
        available_occasions = occasions if occasions else [
            "birthday", "mothers_day", "fathers_day", "christmas", "graduation", 
            "anniversary", "wedding", "new_year"
        ]
        
        # Use REAL categories from scraped data
        available_categories = categories if categories else ["electronics", "fashion", "home"]
        
        # Random profile generation
        age = random.randint(16, 70)
        num_hobbies = random.randint(2, 4)
        hobbies = random.sample(available_tags, min(num_hobbies, len(available_tags)))
        
        # Relationships
        relationships = ["friend", "mother", "father", "sister", "brother", "colleague", 
                        "spouse", "uncle", "aunt", "cousin", "grandparent"]
        
        # Select budget tier and generate budget within that range
        budget_tier = random.choice(['low', 'medium', 'high', 'premium'])
        price_range = price_ranges[budget_tier]
        budget = round(random.uniform(price_range[0], price_range[1]), 2)
        
        # Select preferences from available tags
        num_preferences = random.randint(2, 4)
        preferences = random.sample(available_tags, min(num_preferences, len(available_tags)))
        
        profile = {
            "age": age,
            "hobbies": hobbies,
            "relationship": random.choice(relationships),
            "budget": budget,
            "occasion": random.choice(available_occasions),
            "preferences": preferences
        }
        
        return {
            "id": f"scenario_{index+1:03d}",
            "profile": profile,
            "expected_categories": self._infer_categories_from_catalog(profile, available_categories),
            "expected_tools": self._infer_tools(profile)
        }

    def _infer_categories_from_catalog(self, profile: Dict[str, Any], 
                                      available_categories: List[str]) -> List[str]:
        """Infer expected gift categories from profile using REAL catalog data"""
        hobbies = profile.get('hobbies', [])
        preferences = profile.get('preferences', [])
        
        # Combine hobbies and preferences for matching
        interests = set(hobbies + preferences)
        
        # Match interests with actual catalog categories
        matched_categories = []
        
        for category in available_categories:
            category_lower = category.lower()
            for interest in interests:
                interest_lower = interest.lower()
                # Check if interest matches category or vice versa
                if interest_lower in category_lower or category_lower in interest_lower:
                    matched_categories.append(category)
                    break
        
        # If no matches found, return random categories from catalog
        if not matched_categories:
            num_cats = min(3, len(available_categories))
            matched_categories = random.sample(available_categories, num_cats)
        
        return list(set(matched_categories))[:3]  # Max 3 categories
    
    def _infer_categories(self, profile: Dict[str, Any]) -> List[str]:
        """Legacy method for backward compatibility"""
        hobbies = profile.get('hobbies', [])
        
        # Map hobbies to categories
        category_mapping = {
            'technology': ['technology', 'gaming'],
            'fitness': ['fitness', 'wellness', 'outdoor'],
            'art': ['art', 'craft'],
            'cooking': ['kitchen', 'food'],
            'reading': ['books'],
            'music': ['entertainment'],
            'gardening': ['gardening', 'outdoor'],
            'fashion': ['fashion', 'beauty']
        }
        
        categories = []
        for hobby in hobbies:
            hobby_lower = hobby.lower()
            for key, cats in category_mapping.items():
                if key in hobby_lower:
                    categories.extend(cats)
        
        # If no mapping found, use hobbies directly
        if not categories:
            categories = hobbies[:3]
        
        return list(set(categories))[:3]  # Max 3 categories

    def _infer_tools(self, profile: Dict[str, Any]) -> List[str]:
        """
        Infer expected tools based on profile characteristics
        
        Tool selection logic:
        - budget_optimizer: Low budget users (< 200 TL)
        - price_comparison: Budget-conscious users (< 300 TL) or value seekers
        - review_analysis: Quality-focused, luxury seekers, or high-budget users
        - trend_analysis: Tech-savvy, modern, trendy interests
        - inventory_check: Technology/electronics interests
        
        Uses dynamically extracted keywords from gift catalog
        """
        tools = []
        
        budget = profile.get('budget', 100)
        preferences = profile.get('preferences', [])
        hobbies = profile.get('hobbies', [])
        all_interests = [str(p).lower() for p in preferences + hobbies]
        
        # Budget-based tools
        if budget < 200:
            # Low budget: need optimization and price comparison
            tools.extend(['budget_optimizer', 'price_comparison'])
        elif budget < 300:
            # Medium budget: price comparison helpful
            tools.append('price_comparison')
        elif budget > 500:
            # High budget: quality matters more than price
            tools.append('review_analysis')
        
        # Trend-conscious users (using dynamic keywords from catalog)
        if any(keyword in interest for keyword in self.trend_keywords for interest in all_interests):
            tools.append('trend_analysis')
        
        # Technology/electronics: inventory important (using dynamic keywords from catalog)
        if any(keyword in interest for keyword in self.tech_keywords for interest in all_interests):
            tools.append('inventory_check')
        
        # Quality-focused users (using dynamic keywords from catalog)
        if any(keyword in interest for keyword in self.quality_keywords for interest in all_interests):
            tools.append('review_analysis')
        
        # Value seekers (using dynamic keywords from catalog)
        if any(keyword in interest for keyword in self.value_keywords for interest in all_interests):
            if 'price_comparison' not in tools:
                tools.append('price_comparison')
            if 'budget_optimizer' not in tools and budget < 300:
                tools.append('budget_optimizer')
        
        # Ensure at least one tool is selected (default to price_comparison)
        if not tools:
            tools.append('price_comparison')
        
        return list(set(tools))

    async def save_scenarios(self, scenarios: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save scenarios to JSON file, merging with existing scenarios
        
        Args:
            scenarios: List of new scenario dictionaries
            output_path: Path to save JSON file
        """
        try:
            # Load existing scenarios if file exists
            existing_scenarios = self._load_existing_scenarios(output_path)
            
            # Merge with existing, removing duplicates
            all_scenarios = self._merge_scenarios(existing_scenarios, scenarios)
            
            self.logger.info(f"Total scenarios after merge: {len(all_scenarios)} (added {len(all_scenarios) - len(existing_scenarios)} new)")
            
            # Generate metadata
            metadata = self._generate_metadata(all_scenarios)
            
            # Create output structure
            output = {
                "scenarios": all_scenarios,
                "metadata": metadata
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Scenarios saved to {output_path}")
            
            # Log file size
            file_size = os.path.getsize(output_path)
            file_size_kb = file_size / 1024
            self.logger.info(f"File size: {file_size_kb:.2f} KB")
            
        except Exception as e:
            self.logger.error(f"Failed to save scenarios: {e}")
            raise
    
    def _load_existing_scenarios(self, output_path: str) -> List[Dict[str, Any]]:
        """
        Load existing scenarios from file
        
        Args:
            output_path: Path to scenarios file
            
        Returns:
            List of existing scenarios or empty list
        """
        if not os.path.exists(output_path):
            self.logger.info("No existing scenarios found, starting fresh")
            return []
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            existing_scenarios = data.get('scenarios', [])
            self.logger.info(f"Loaded {len(existing_scenarios)} existing scenarios")
            return existing_scenarios
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing scenarios: {e}. Starting fresh.")
            return []
    
    def _merge_scenarios(self, existing_scenarios: List[Dict[str, Any]], 
                        new_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge new scenarios with existing ones, removing duplicates
        
        Args:
            existing_scenarios: List of existing scenarios
            new_scenarios: List of new scenarios
            
        Returns:
            Merged list without duplicates
        """
        # Create signatures for duplicate detection
        existing_signatures = set()
        for scenario in existing_scenarios:
            profile = scenario.get('profile', {})
            # Use age, relationship, occasion, and budget as signature
            signature = (
                profile.get('age', 0),
                profile.get('relationship', ''),
                profile.get('occasion', ''),
                round(profile.get('budget', 0), 2),
                tuple(sorted(profile.get('hobbies', []))),
                tuple(sorted(profile.get('preferences', [])))
            )
            existing_signatures.add(signature)
        
        # Filter out duplicates from new scenarios
        unique_new_scenarios = []
        duplicates_found = 0
        
        for scenario in new_scenarios:
            profile = scenario.get('profile', {})
            signature = (
                profile.get('age', 0),
                profile.get('relationship', ''),
                profile.get('occasion', ''),
                round(profile.get('budget', 0), 2),
                tuple(sorted(profile.get('hobbies', []))),
                tuple(sorted(profile.get('preferences', [])))
            )
            
            if signature not in existing_signatures:
                existing_signatures.add(signature)
                unique_new_scenarios.append(scenario)
            else:
                duplicates_found += 1
        
        if duplicates_found > 0:
            self.logger.info(f"Filtered out {duplicates_found} duplicate scenarios")
        
        # Merge lists
        all_scenarios = existing_scenarios + unique_new_scenarios
        
        # Re-index all scenarios to ensure unique IDs
        for idx, scenario in enumerate(all_scenarios):
            scenario['id'] = f"scenario_{idx:04d}"
        
        return all_scenarios

    def _generate_metadata(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata for scenarios (matching expanded_user_scenarios.json format)"""
        ages = [s['profile']['age'] for s in scenarios]
        budgets = [s['profile']['budget'] for s in scenarios]
        
        # Extract unique values for coverage
        all_hobbies = set()
        all_relationships = set()
        all_occasions = set()
        all_preferences = set()
        
        for s in scenarios:
            profile = s['profile']
            all_hobbies.update(profile.get('hobbies', []))
            all_relationships.add(profile.get('relationship', ''))
            all_occasions.add(profile.get('occasion', ''))
            all_preferences.update(profile.get('preferences', []))
        
        # Calculate age groups (16-25, 26-35, 36-45, 46-55, 56-70)
        age_groups = set()
        for age in ages:
            if age <= 25:
                age_groups.add('16-25')
            elif age <= 35:
                age_groups.add('26-35')
            elif age <= 45:
                age_groups.add('36-45')
            elif age <= 55:
                age_groups.add('46-55')
            else:
                age_groups.add('56-70')
        
        # Calculate budget tiers
        budget_tiers = set()
        for budget in budgets:
            if budget < 50:
                budget_tiers.add('low')
            elif budget < 150:
                budget_tiers.add('medium')
            elif budget < 250:
                budget_tiers.add('high')
            else:
                budget_tiers.add('premium')
        
        return {
            "total_scenarios": len(scenarios),
            "generation_method": "gemini_ai" if self.enabled else "systematic_combination",
            "coverage": {
                "age_groups": len(age_groups),
                "hobby_categories": len(all_hobbies),
                "preference_types": len(all_preferences),
                "relationships": len(all_relationships),
                "occasions": len(all_occasions),
                "budget_tiers": len(budget_tiers)
            },
            "version": "2.0",
            "created": datetime.now().strftime("%Y-%m-%d")
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'requests_made': self.request_count,
            'requests_remaining': self.max_requests - self.request_count,
            'max_requests_per_day': self.max_requests,
            'enabled': self.enabled
        }
