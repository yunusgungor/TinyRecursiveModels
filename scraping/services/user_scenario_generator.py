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


class UserScenarioGenerator:
    """Generates realistic user scenarios for gift recommendation testing"""
    
    def __init__(self, config: Dict[str, Any], gift_catalog_path: str):
        """
        Initialize user scenario generator
        
        Args:
            config: Gemini API configuration
            gift_catalog_path: Path to gift catalog JSON file
        """
        self.config = config
        self.gift_catalog_path = gift_catalog_path
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        
        # Load gift catalog
        self.gift_catalog = self._load_gift_catalog()
        
        # Configuration
        self.max_requests = config.get('max_requests_per_day', 1000)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Request tracking
        self.request_count = 0

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

    async def generate_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """
        Generate user scenarios
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of user scenario dictionaries
        """
        self.logger.info(f"Generating {num_scenarios} user scenarios...")
        
        # Get delay from config
        batch_delay = self.config.get('batch_delay', 5)
        
        scenarios = []
        
        # Extract real data from catalog
        categories = self._extract_categories()
        tags = self._extract_tags()
        occasions = self._extract_occasions()
        price_ranges = self._extract_price_ranges()
        
        self.logger.info(f"Extracted from catalog: {len(categories)} categories, {len(tags)} tags, {len(occasions)} occasions")
        
        for i in range(num_scenarios):
            if self.enabled and self.request_count < self.max_requests:
                scenario = await self._generate_ai_scenario(i, categories, tags, occasions, price_ranges)
                
                # Wait between AI requests to avoid rate limits
                if scenario and i < num_scenarios - 1:
                    await asyncio.sleep(batch_delay)
            else:
                scenario = self._generate_fallback_scenario(i, categories, tags, occasions, price_ranges)
            
            if scenario:
                scenarios.append(scenario)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i + 1}/{num_scenarios} scenarios")
        
        self.logger.info(f"Scenario generation complete: {len(scenarios)} scenarios")
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

    async def _generate_ai_scenario(self, index: int, categories: List[str], 
                                   tags: List[str], occasions: List[str], 
                                   price_ranges: Dict[str, tuple]) -> Optional[Dict[str, Any]]:
        """Generate scenario using Gemini API"""
        
        # Build prompt
        prompt = self._build_scenario_prompt(categories, tags, occasions, price_ranges)
        
        # Try with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                )
                
                scenario = self._parse_scenario_response(response.text, index)
                self.request_count += 1
                return scenario
                
            except Exception as e:
                self.logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # Fallback if all retries failed
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
        """Infer expected tools based on profile"""
        tools = []
        
        budget = profile.get('budget', 100)
        preferences = profile.get('preferences', [])
        
        # Budget-based tools
        if budget < 100:
            tools.append('budget_optimizer')
        
        # Always useful tools
        tools.extend(['review_analysis', 'price_comparison'])
        
        # Preference-based tools
        if any(p in ['trendy', 'tech-savvy'] for p in preferences):
            tools.append('trend_analysis')
        
        if any(p in ['luxury', 'premium'] for p in preferences):
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
