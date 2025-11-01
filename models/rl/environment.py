"""
RL Environment for Gift Recommendation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json
import random


@dataclass
class UserProfile:
    age: int
    hobbies: List[str]
    relationship: str
    budget: float
    occasion: str
    personality_traits: List[str]
    purchase_history: List[str] = None
    
    def to_dict(self):
        return {
            'age': self.age,
            'hobbies': self.hobbies,
            'relationship': self.relationship,
            'budget': self.budget,
            'occasion': self.occasion,
            'personality_traits': self.personality_traits,
            'purchase_history': self.purchase_history or []
        }


@dataclass
class GiftItem:
    id: str
    name: str
    category: str
    price: float
    rating: float
    tags: List[str]
    description: str
    age_suitability: Tuple[int, int]
    occasion_fit: List[str]
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'price': self.price,
            'rating': self.rating,
            'tags': self.tags,
            'description': self.description,
            'age_suitability': self.age_suitability,
            'occasion_fit': self.occasion_fit
        }


@dataclass
class EnvironmentState:
    user_profile: UserProfile
    available_gifts: List[GiftItem]
    current_recommendations: List[GiftItem]
    interaction_history: List[Dict]
    step_count: int
    
    def to_tensor(self, device='cpu'):
        """Convert state to tensor representation for model input"""
        # User profile encoding
        user_features = [
            self.user_profile.age / 100.0,  # Normalize age
            self.user_profile.budget / 1000.0,  # Normalize budget
            len(self.user_profile.hobbies),
            len(self.user_profile.personality_traits),
            self.step_count / 10.0  # Normalize step count
        ]
        
        # Add categorical features (simplified encoding)
        hobby_encoding = self._encode_hobbies(self.user_profile.hobbies)
        relationship_encoding = self._encode_relationship(self.user_profile.relationship)
        occasion_encoding = self._encode_occasion(self.user_profile.occasion)
        
        state_vector = user_features + hobby_encoding + relationship_encoding + occasion_encoding
        
        return torch.tensor(state_vector, dtype=torch.float32, device=device)
    
    def _encode_hobbies(self, hobbies):
        """Encode hobbies as binary vector"""
        hobby_categories = ['gardening', 'cooking', 'reading', 'sports', 'music', 'art', 'technology', 'travel']
        encoding = [1.0 if hobby in hobbies else 0.0 for hobby in hobby_categories]
        return encoding
    
    def _encode_relationship(self, relationship):
        """Encode relationship as one-hot vector"""
        relationships = ['mother', 'father', 'friend', 'partner', 'sibling', 'colleague', 'other']
        encoding = [1.0 if rel == relationship else 0.0 for rel in relationships]
        return encoding
    
    def _encode_occasion(self, occasion):
        """Encode occasion as one-hot vector"""
        occasions = ['birthday', 'christmas', 'anniversary', 'graduation', 'wedding', 'other']
        encoding = [1.0 if occ == occasion else 0.0 for occ in occasions]
        return encoding


class GiftRecommendationEnvironment:
    """RL Environment for Gift Recommendation Task"""
    
    def __init__(self, gift_catalog_path: str, user_feedback_data_path: Optional[str] = None):
        self.gift_catalog = self._load_gift_catalog(gift_catalog_path)
        self.user_feedback_data = self._load_user_feedback(user_feedback_data_path) if user_feedback_data_path else {}
        
        # Environment parameters
        self.max_steps = 10
        self.max_recommendations_per_step = 5
        
        # Current state
        self.current_state: Optional[EnvironmentState] = None
        self.episode_rewards = []
        
    def _load_gift_catalog(self, path: str) -> List[GiftItem]:
        """Load gift catalog from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            gifts = []
            for item in data:
                gift = GiftItem(
                    id=item['id'],
                    name=item['name'],
                    category=item['category'],
                    price=item['price'],
                    rating=item['rating'],
                    tags=item['tags'],
                    description=item['description'],
                    age_suitability=tuple(item['age_suitability']),
                    occasion_fit=item['occasion_fit']
                )
                gifts.append(gift)
            
            return gifts
        except FileNotFoundError:
            # Create sample catalog if file doesn't exist
            return self._create_sample_catalog()
    
    def _create_sample_catalog(self) -> List[GiftItem]:
        """Create sample gift catalog for testing"""
        sample_gifts = [
            GiftItem("1", "Organic Seed Set", "gardening", 75.0, 4.8, 
                    ["organic", "sustainable", "educational"], 
                    "Premium organic vegetable seeds", (25, 65), ["birthday", "mothers_day"]),
            GiftItem("2", "Cooking Masterclass", "cooking", 120.0, 4.9,
                    ["educational", "experience", "skill"], 
                    "Online cooking course", (20, 60), ["birthday", "christmas"]),
            GiftItem("3", "Bestseller Book Set", "reading", 45.0, 4.5,
                    ["educational", "entertainment"], 
                    "Collection of bestseller novels", (18, 80), ["birthday", "christmas"]),
            GiftItem("4", "Yoga Mat Premium", "sports", 85.0, 4.7,
                    ["health", "fitness", "sustainable"], 
                    "Eco-friendly yoga mat", (20, 50), ["birthday", "new_year"]),
            GiftItem("5", "Smart Plant Monitor", "technology", 95.0, 4.6,
                    ["technology", "gardening", "smart"], 
                    "IoT device for plant care", (25, 55), ["birthday", "christmas"])
        ]
        return sample_gifts
    
    def _load_user_feedback(self, path: str) -> Dict:
        """Load historical user feedback data"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def reset(self, user_profile: UserProfile) -> EnvironmentState:
        """Reset environment with new user profile"""
        self.current_state = EnvironmentState(
            user_profile=user_profile,
            available_gifts=self.gift_catalog.copy(),
            current_recommendations=[],
            interaction_history=[],
            step_count=0
        )
        self.episode_rewards = []
        return self.current_state
    
    def step(self, action: Dict[str, Any]) -> Tuple[EnvironmentState, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Dictionary containing:
                - recommendations: List of recommended gift IDs
                - confidence_scores: Confidence scores for each recommendation
        
        Returns:
            next_state, reward, done, info
        """
        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Extract recommendations from action
        recommended_gift_ids = action.get('recommendations', [])
        confidence_scores = action.get('confidence_scores', [1.0] * len(recommended_gift_ids))
        
        # Get recommended gifts
        recommended_gifts = [
            gift for gift in self.gift_catalog 
            if gift.id in recommended_gift_ids
        ]
        
        # Update state
        self.current_state.current_recommendations = recommended_gifts
        self.current_state.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(recommended_gifts, confidence_scores)
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        done = (
            self.current_state.step_count >= self.max_steps or 
            reward > 0.9 or  # High satisfaction achieved
            len(recommended_gifts) == 0  # No valid recommendations
        )
        
        # Add interaction to history
        interaction = {
            'step': self.current_state.step_count,
            'recommendations': [gift.to_dict() for gift in recommended_gifts],
            'reward': reward,
            'confidence_scores': confidence_scores
        }
        self.current_state.interaction_history.append(interaction)
        
        # Info dictionary
        info = {
            'episode_reward': sum(self.episode_rewards),
            'step_count': self.current_state.step_count,
            'recommendation_count': len(recommended_gifts)
        }
        
        return self.current_state, reward, done, info
    
    def _calculate_reward(self, recommended_gifts: List[GiftItem], confidence_scores: List[float]) -> float:
        """Calculate reward based on recommendation quality"""
        if not recommended_gifts:
            return -0.5  # Penalty for no recommendations
        
        user_profile = self.current_state.user_profile
        total_reward = 0.0
        
        for i, gift in enumerate(recommended_gifts):
            gift_reward = 0.0
            confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
            
            # Budget compatibility (30% of reward)
            budget_score = self._calculate_budget_score(gift.price, user_profile.budget)
            gift_reward += 0.3 * budget_score
            
            # Hobby alignment (25% of reward)
            hobby_score = self._calculate_hobby_score(gift, user_profile.hobbies)
            gift_reward += 0.25 * hobby_score
            
            # Occasion appropriateness (20% of reward)
            occasion_score = self._calculate_occasion_score(gift, user_profile.occasion)
            gift_reward += 0.2 * occasion_score
            
            # Age appropriateness (15% of reward)
            age_score = self._calculate_age_score(gift, user_profile.age)
            gift_reward += 0.15 * age_score
            
            # Quality score (10% of reward)
            quality_score = gift.rating / 5.0
            gift_reward += 0.1 * quality_score
            
            # Apply confidence weighting
            gift_reward *= confidence
            
            total_reward += gift_reward
        
        # Average reward across recommendations
        avg_reward = total_reward / len(recommended_gifts)
        
        # Add diversity bonus
        diversity_bonus = self._calculate_diversity_bonus(recommended_gifts)
        avg_reward += 0.1 * diversity_bonus
        
        return min(1.0, max(-1.0, avg_reward))  # Clamp between -1 and 1
    
    def _calculate_budget_score(self, price: float, budget: float) -> float:
        """Calculate how well the price fits the budget"""
        if price <= budget * 0.8:  # Well within budget
            return 1.0
        elif price <= budget:  # Within budget
            return 0.8
        elif price <= budget * 1.2:  # Slightly over budget
            return 0.4
        else:  # Way over budget
            return 0.0
    
    def _calculate_hobby_score(self, gift: GiftItem, hobbies: List[str]) -> float:
        """Calculate how well the gift aligns with user hobbies"""
        if not hobbies:
            return 0.5  # Neutral if no hobbies specified
        
        # Check if gift category or tags match hobbies
        matches = 0
        for hobby in hobbies:
            if (hobby.lower() in gift.category.lower() or 
                any(hobby.lower() in tag.lower() for tag in gift.tags)):
                matches += 1
        
        return min(1.0, matches / len(hobbies))
    
    def _calculate_occasion_score(self, gift: GiftItem, occasion: str) -> float:
        """Calculate how appropriate the gift is for the occasion"""
        if occasion.lower() in [occ.lower() for occ in gift.occasion_fit]:
            return 1.0
        elif 'any' in gift.occasion_fit or len(gift.occasion_fit) == 0:
            return 0.7  # Generic gifts
        else:
            return 0.3  # Not specifically appropriate
    
    def _calculate_age_score(self, gift: GiftItem, age: int) -> float:
        """Calculate age appropriateness score"""
        min_age, max_age = gift.age_suitability
        if min_age <= age <= max_age:
            return 1.0
        elif abs(age - min_age) <= 5 or abs(age - max_age) <= 5:
            return 0.7  # Close to appropriate range
        else:
            return 0.3  # Not age appropriate
    
    def _calculate_diversity_bonus(self, gifts: List[GiftItem]) -> float:
        """Calculate bonus for diverse recommendations"""
        if len(gifts) <= 1:
            return 0.0
        
        categories = set(gift.category for gift in gifts)
        diversity_score = len(categories) / len(gifts)
        return diversity_score
    
    def get_state_tensor(self, device='cpu') -> torch.Tensor:
        """Get current state as tensor for model input"""
        if self.current_state is None:
            raise ValueError("Environment not initialized")
        return self.current_state.to_tensor(device)
    
    def render(self) -> str:
        """Render current state for debugging"""
        if self.current_state is None:
            return "Environment not initialized"
        
        output = f"Step: {self.current_state.step_count}\n"
        output += f"User: {self.current_state.user_profile.age}y, {self.current_state.user_profile.relationship}\n"
        output += f"Budget: ${self.current_state.user_profile.budget}\n"
        output += f"Hobbies: {', '.join(self.current_state.user_profile.hobbies)}\n"
        output += f"Occasion: {self.current_state.user_profile.occasion}\n"
        
        if self.current_state.current_recommendations:
            output += "\nCurrent Recommendations:\n"
            for gift in self.current_state.current_recommendations:
                output += f"  - {gift.name} (${gift.price}) - {gift.category}\n"
        
        if self.episode_rewards:
            output += f"\nEpisode Rewards: {self.episode_rewards}\n"
            output += f"Total Reward: {sum(self.episode_rewards):.3f}\n"
        
        return output


def create_sample_gift_catalog(output_path: str):
    """Create a sample gift catalog file for testing"""
    sample_data = [
        {
            "id": "1",
            "name": "Organic Seed Set",
            "category": "gardening",
            "price": 75.0,
            "rating": 4.8,
            "tags": ["organic", "sustainable", "educational"],
            "description": "Premium organic vegetable seeds for home gardening",
            "age_suitability": [25, 65],
            "occasion_fit": ["birthday", "mothers_day", "christmas"]
        },
        {
            "id": "2", 
            "name": "Cooking Masterclass",
            "category": "cooking",
            "price": 120.0,
            "rating": 4.9,
            "tags": ["educational", "experience", "skill"],
            "description": "Online cooking course with professional chef",
            "age_suitability": [20, 60],
            "occasion_fit": ["birthday", "christmas", "anniversary"]
        },
        {
            "id": "3",
            "name": "Bestseller Book Set",
            "category": "reading", 
            "price": 45.0,
            "rating": 4.5,
            "tags": ["educational", "entertainment", "literature"],
            "description": "Collection of current bestseller novels",
            "age_suitability": [18, 80],
            "occasion_fit": ["birthday", "christmas", "graduation"]
        },
        {
            "id": "4",
            "name": "Premium Yoga Mat",
            "category": "sports",
            "price": 85.0,
            "rating": 4.7,
            "tags": ["health", "fitness", "sustainable", "wellness"],
            "description": "Eco-friendly yoga mat with alignment guides",
            "age_suitability": [20, 50],
            "occasion_fit": ["birthday", "new_year", "wellness"]
        },
        {
            "id": "5",
            "name": "Smart Plant Monitor",
            "category": "technology",
            "price": 95.0,
            "rating": 4.6,
            "tags": ["technology", "gardening", "smart", "iot"],
            "description": "IoT device for monitoring plant health",
            "age_suitability": [25, 55],
            "occasion_fit": ["birthday", "christmas", "housewarming"]
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_gift_catalog("data/sample_gift_catalog.json")
    
    # Test environment
    env = GiftRecommendationEnvironment("data/sample_gift_catalog.json")
    
    # Test user profile
    user = UserProfile(
        age=35,
        hobbies=["gardening", "cooking"],
        relationship="mother",
        budget=100.0,
        occasion="birthday",
        personality_traits=["eco-conscious", "practical"]
    )
    
    # Reset environment
    state = env.reset(user)
    print("Initial State:")
    print(env.render())
    
    # Take a step
    action = {
        'recommendations': ["1", "2"],  # Organic seeds and cooking class
        'confidence_scores': [0.9, 0.8]
    }
    
    next_state, reward, done, info = env.step(action)
    print(f"\nAfter step:")
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print(env.render())