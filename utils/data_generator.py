"""
Data generation utilities for gift recommendation training
"""

import json
import random
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os


class GiftDataGenerator:
    """Generate synthetic gift recommendation data for training"""
    
    def __init__(self):
        self.hobbies = [
            "gardening", "cooking", "reading", "sports", "music", "art", 
            "technology", "travel", "photography", "crafts", "gaming", 
            "fitness", "fashion", "home_decor", "pets", "outdoor"
        ]
        
        self.relationships = [
            "mother", "father", "sister", "brother", "friend", "partner", 
            "colleague", "boss", "teacher", "neighbor", "cousin", "aunt", 
            "uncle", "grandparent", "child"
        ]
        
        self.occasions = [
            "birthday", "christmas", "anniversary", "graduation", "wedding", 
            "mothers_day", "fathers_day", "valentines", "new_year", "easter",
            "thanksgiving", "housewarming", "retirement", "promotion", "baby_shower"
        ]
        
        self.personality_traits = [
            "eco-conscious", "practical", "trendy", "traditional", "adventurous",
            "minimalist", "luxury-loving", "tech-savvy", "artistic", "sporty",
            "intellectual", "social", "introverted", "creative", "organized"
        ]
        
        self.gift_categories = [
            "books", "electronics", "clothing", "jewelry", "home_decor", 
            "kitchen", "sports", "beauty", "toys", "art_supplies", "tools",
            "gardening", "music", "games", "travel", "food", "experiences"
        ]
        
        self.gift_templates = self._create_gift_templates()
    
    def _create_gift_templates(self) -> Dict[str, List[Dict]]:
        """Create templates for different gift categories"""
        return {
            "books": [
                {"name": "Bestseller Novel", "price_range": (15, 30), "tags": ["educational", "entertainment"]},
                {"name": "Cookbook", "price_range": (20, 50), "tags": ["cooking", "educational"]},
                {"name": "Art Book", "price_range": (25, 80), "tags": ["art", "coffee_table"]},
                {"name": "Self-Help Book", "price_range": (12, 25), "tags": ["personal_development"]},
            ],
            "electronics": [
                {"name": "Wireless Earbuds", "price_range": (50, 200), "tags": ["music", "technology"]},
                {"name": "Smart Watch", "price_range": (100, 400), "tags": ["fitness", "technology"]},
                {"name": "Tablet", "price_range": (150, 600), "tags": ["technology", "entertainment"]},
                {"name": "Bluetooth Speaker", "price_range": (30, 150), "tags": ["music", "portable"]},
            ],
            "home_decor": [
                {"name": "Decorative Vase", "price_range": (25, 100), "tags": ["home", "decorative"]},
                {"name": "Wall Art", "price_range": (30, 200), "tags": ["art", "home"]},
                {"name": "Candle Set", "price_range": (20, 60), "tags": ["relaxation", "home"]},
                {"name": "Plant Pot", "price_range": (15, 50), "tags": ["gardening", "home"]},
            ],
            "kitchen": [
                {"name": "Coffee Maker", "price_range": (50, 300), "tags": ["cooking", "appliance"]},
                {"name": "Knife Set", "price_range": (40, 200), "tags": ["cooking", "tools"]},
                {"name": "Mixing Bowls", "price_range": (20, 80), "tags": ["cooking", "baking"]},
                {"name": "Spice Rack", "price_range": (25, 100), "tags": ["cooking", "organization"]},
            ],
            "gardening": [
                {"name": "Seed Set", "price_range": (15, 50), "tags": ["organic", "sustainable"]},
                {"name": "Garden Tools", "price_range": (30, 120), "tags": ["tools", "outdoor"]},
                {"name": "Planter Box", "price_range": (25, 100), "tags": ["containers", "outdoor"]},
                {"name": "Watering Can", "price_range": (20, 60), "tags": ["tools", "decorative"]},
            ],
            "sports": [
                {"name": "Yoga Mat", "price_range": (25, 100), "tags": ["fitness", "wellness"]},
                {"name": "Dumbbells", "price_range": (30, 150), "tags": ["fitness", "strength"]},
                {"name": "Running Shoes", "price_range": (60, 200), "tags": ["fitness", "running"]},
                {"name": "Water Bottle", "price_range": (15, 50), "tags": ["fitness", "hydration"]},
            ]
        }
    
    def generate_user_profile(self) -> Dict[str, Any]:
        """Generate a random user profile"""
        return {
            "age": random.randint(18, 80),
            "hobbies": random.sample(self.hobbies, random.randint(1, 4)),
            "relationship": random.choice(self.relationships),
            "budget": round(random.uniform(20, 500), 2),
            "occasion": random.choice(self.occasions),
            "personality_traits": random.sample(self.personality_traits, random.randint(1, 3)),
            "purchase_history": self._generate_purchase_history()
        }
    
    def _generate_purchase_history(self) -> List[str]:
        """Generate mock purchase history"""
        history_size = random.randint(0, 10)
        categories = random.sample(self.gift_categories, min(history_size, len(self.gift_categories)))
        return categories
    
    def generate_gift_item(self, category: str = None) -> Dict[str, Any]:
        """Generate a gift item"""
        if category is None:
            category = random.choice(list(self.gift_templates.keys()))
        
        if category not in self.gift_templates:
            category = random.choice(list(self.gift_templates.keys()))
        
        template = random.choice(self.gift_templates[category])
        
        # Generate unique ID
        gift_id = f"{category}_{random.randint(1000, 9999)}"
        
        # Generate price within range
        price_min, price_max = template["price_range"]
        price = round(random.uniform(price_min, price_max), 2)
        
        # Generate rating
        rating = round(random.uniform(3.5, 5.0), 1)
        
        # Age suitability
        age_ranges = [
            (18, 30), (25, 45), (30, 60), (40, 70), (18, 80), (25, 65)
        ]
        age_suitability = random.choice(age_ranges)
        
        # Occasion fit
        occasion_fit = random.sample(self.occasions, random.randint(2, 5))
        
        return {
            "id": gift_id,
            "name": template["name"],
            "category": category,
            "price": price,
            "rating": rating,
            "tags": template["tags"] + random.sample(["premium", "bestseller", "eco-friendly", "handmade"], random.randint(0, 2)),
            "description": f"High-quality {template['name'].lower()} perfect for {category} enthusiasts",
            "age_suitability": list(age_suitability),
            "occasion_fit": occasion_fit
        }
    
    def generate_gift_catalog(self, num_items: int = 100) -> List[Dict[str, Any]]:
        """Generate a complete gift catalog"""
        catalog = []
        
        # Ensure we have items from each category
        items_per_category = num_items // len(self.gift_templates)
        remaining_items = num_items % len(self.gift_templates)
        
        for category in self.gift_templates.keys():
            category_items = items_per_category
            if remaining_items > 0:
                category_items += 1
                remaining_items -= 1
            
            for _ in range(category_items):
                catalog.append(self.generate_gift_item(category))
        
        return catalog
    
    def generate_training_example(self) -> Dict[str, Any]:
        """Generate a single training example"""
        user_profile = self.generate_user_profile()
        
        # Generate candidate gifts
        num_candidates = random.randint(5, 20)
        candidate_gifts = [self.generate_gift_item() for _ in range(num_candidates)]
        
        # Generate ground truth recommendations with scores
        recommendations = []
        for gift in candidate_gifts:
            score = self._calculate_compatibility_score(user_profile, gift)
            recommendations.append({
                "gift_id": gift["id"],
                "gift": gift,
                "score": score
            })
        
        # Sort by score and take top recommendations
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        top_recommendations = recommendations[:5]
        
        return {
            "user_profile": user_profile,
            "candidate_gifts": candidate_gifts,
            "recommendations": top_recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_compatibility_score(self, user_profile: Dict, gift: Dict) -> float:
        """Calculate compatibility score between user and gift"""
        score = 0.0
        
        # Budget compatibility (30% weight)
        budget_score = self._budget_compatibility(user_profile["budget"], gift["price"])
        score += 0.3 * budget_score
        
        # Hobby alignment (25% weight)
        hobby_score = self._hobby_alignment(user_profile["hobbies"], gift["category"], gift["tags"])
        score += 0.25 * hobby_score
        
        # Occasion appropriateness (20% weight)
        occasion_score = self._occasion_appropriateness(user_profile["occasion"], gift["occasion_fit"])
        score += 0.2 * occasion_score
        
        # Age appropriateness (15% weight)
        age_score = self._age_appropriateness(user_profile["age"], gift["age_suitability"])
        score += 0.15 * age_score
        
        # Quality score (10% weight)
        quality_score = gift["rating"] / 5.0
        score += 0.1 * quality_score
        
        # Add some randomness
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _budget_compatibility(self, budget: float, price: float) -> float:
        """Calculate budget compatibility score"""
        if price <= budget * 0.8:
            return 1.0
        elif price <= budget:
            return 0.8
        elif price <= budget * 1.2:
            return 0.4
        else:
            return 0.0
    
    def _hobby_alignment(self, hobbies: List[str], category: str, tags: List[str]) -> float:
        """Calculate hobby alignment score"""
        if not hobbies:
            return 0.5
        
        matches = 0
        for hobby in hobbies:
            if (hobby.lower() in category.lower() or 
                any(hobby.lower() in tag.lower() for tag in tags)):
                matches += 1
        
        return min(1.0, matches / len(hobbies))
    
    def _occasion_appropriateness(self, occasion: str, occasion_fit: List[str]) -> float:
        """Calculate occasion appropriateness score"""
        if occasion.lower() in [occ.lower() for occ in occasion_fit]:
            return 1.0
        else:
            return 0.3
    
    def _age_appropriateness(self, age: int, age_suitability: List[int]) -> float:
        """Calculate age appropriateness score"""
        min_age, max_age = age_suitability
        if min_age <= age <= max_age:
            return 1.0
        elif abs(age - min_age) <= 5 or abs(age - max_age) <= 5:
            return 0.7
        else:
            return 0.3
    
    def generate_training_dataset(self, num_examples: int = 1000, 
                                output_dir: str = "data") -> str:
        """Generate a complete training dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_examples} training examples...")
        
        # Generate gift catalog
        catalog = self.generate_gift_catalog(200)
        catalog_path = os.path.join(output_dir, "gift_catalog.json")
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        
        print(f"Gift catalog saved to {catalog_path}")
        
        # Generate training examples
        training_data = []
        for i in range(num_examples):
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_examples} examples")
            
            example = self.generate_training_example()
            training_data.append(example)
        
        # Split into train/test
        split_idx = int(0.8 * len(training_data))
        train_data = training_data[:split_idx]
        test_data = training_data[split_idx:]
        
        # Save training data
        train_path = os.path.join(output_dir, "gift_recommendation_train.json")
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save test data
        test_path = os.path.join(output_dir, "gift_recommendation_test.json")
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to {train_path} ({len(train_data)} examples)")
        print(f"Test data saved to {test_path} ({len(test_data)} examples)")
        
        # Generate metadata
        metadata = {
            "total_examples": num_examples,
            "train_examples": len(train_data),
            "test_examples": len(test_data),
            "gift_catalog_size": len(catalog),
            "categories": list(self.gift_templates.keys()),
            "hobbies": self.hobbies,
            "relationships": self.relationships,
            "occasions": self.occasions,
            "generated_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset metadata saved to {metadata_path}")
        
        return output_dir


def main():
    """Generate sample dataset"""
    generator = GiftDataGenerator()
    
    # Generate dataset
    output_dir = generator.generate_training_dataset(
        num_examples=1000,
        output_dir="data"
    )
    
    print(f"\nDataset generation completed!")
    print(f"Files created in: {output_dir}")
    
    # Show sample data
    with open(os.path.join(output_dir, "gift_recommendation_train.json"), 'r') as f:
        sample_data = json.load(f)
    
    print(f"\nSample training example:")
    sample = sample_data[0]
    print(f"User: {sample['user_profile']['age']}y, {sample['user_profile']['relationship']}")
    print(f"Hobbies: {sample['user_profile']['hobbies']}")
    print(f"Budget: ${sample['user_profile']['budget']}")
    print(f"Occasion: {sample['user_profile']['occasion']}")
    print(f"Top recommendation: {sample['recommendations'][0]['gift']['name']} "
          f"(Score: {sample['recommendations'][0]['score']:.3f})")


if __name__ == "__main__":
    main()