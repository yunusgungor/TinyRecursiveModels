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
        
        scenarios = []
        categories = self._extract_categories()
        
        for i in range(num_scenarios):
            if self.enabled and self.request_count < self.max_requests:
                scenario = await self._generate_ai_scenario(i, categories)
            else:
                scenario = self._generate_fallback_scenario(i, categories)
            
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

    async def _generate_ai_scenario(self, index: int, categories: List[str]) -> Optional[Dict[str, Any]]:
        """Generate scenario using Gemini API"""
        
        # Build prompt
        prompt = self._build_scenario_prompt(categories)
        
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
        return self._generate_fallback_scenario(index, categories)

    def _build_scenario_prompt(self, categories: List[str]) -> str:
        """Build prompt for scenario generation"""
        return f"""Generate a realistic user profile for gift recommendation testing.

Available gift categories: {', '.join(categories)}

Create a diverse, realistic person with:
- Age (16-70)
- 2-4 hobbies/interests from the available categories
- Relationship to gift recipient (friend, mother, father, sister, brother, colleague, partner, boss)
- Budget in TL (50-500)
- Occasion (birthday, mothers_day, fathers_day, christmas, graduation, anniversary, wedding, new_year, valentines_day)
- 2-4 preferences (trendy, practical, luxury, eco-friendly, tech-savvy, traditional, creative, sporty, etc.)

Return ONLY valid JSON:
{{
  "age": <number>,
  "hobbies": [<list of 2-4 hobbies>],
  "relationship": "<relationship>",
  "budget": <number>,
  "occasion": "<occasion>",
  "preferences": [<list of 2-4 preferences>]
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

    def _generate_fallback_scenario(self, index: int, categories: List[str]) -> Dict[str, Any]:
        """Generate scenario using rule-based approach"""
        
        # Expanded hobby categories (matching expanded_user_scenarios.json)
        expanded_hobbies = [
            "technology", "photography", "gaming", "music", "art", "reading",
            "fitness", "sports", "outdoor", "yoga", "meditation", "wellness",
            "home_decor", "gardening", "cooking", "food", "travel", 
            "entertainment", "learning", "business"
        ]
        
        # Random profile generation
        age = random.randint(16, 70)
        num_hobbies = random.randint(2, 4)
        hobbies = random.sample(expanded_hobbies, num_hobbies)
        
        # Expanded relationships (matching expanded_user_scenarios.json)
        relationships = ["friend", "mother", "father", "sister", "brother", "colleague", 
                        "spouse", "uncle", "aunt", "cousin", "grandparent"]
        
        # Expanded occasions
        occasions = ["birthday", "mothers_day", "fathers_day", "christmas", "graduation", 
                    "anniversary", "wedding", "new_year", "promotion", "appreciation"]
        
        # Expanded preferences
        preferences_pool = ["trendy", "practical", "luxury", "eco-friendly", "tech-savvy", 
                           "traditional", "creative", "sporty", "affordable", "quality",
                           "premium", "sophisticated", "sustainable", "natural", "modern",
                           "classic", "timeless", "unique", "artistic", "motivational",
                           "energetic", "active"]
        
        profile = {
            "age": age,
            "hobbies": hobbies,
            "relationship": random.choice(relationships),
            "budget": round(random.uniform(30, 300), 2),
            "occasion": random.choice(occasions),
            "preferences": random.sample(preferences_pool, random.randint(2, 4))
        }
        
        return {
            "id": f"scenario_{index+1:03d}",  # Format: scenario_001, scenario_002, etc.
            "profile": profile,
            "expected_categories": self._infer_categories(profile),
            "expected_tools": self._infer_tools(profile)
        }

    def _infer_categories(self, profile: Dict[str, Any]) -> List[str]:
        """Infer expected gift categories from profile"""
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
        Save scenarios to JSON file
        
        Args:
            scenarios: List of scenario dictionaries
            output_path: Path to save JSON file
        """
        try:
            # Generate metadata
            metadata = self._generate_metadata(scenarios)
            
            # Create output structure
            output = {
                "scenarios": scenarios,
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
