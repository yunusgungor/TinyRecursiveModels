"""
Enhanced User Profile Processing for Better Category Matching
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class CategoryMapping:
    """Enhanced category mapping with weights and context"""
    primary_categories: List[str]
    secondary_categories: List[str]
    weight: float
    age_preference: Tuple[int, int]
    occasion_boost: Dict[str, float]


class EnhancedUserProfiler:
    """Enhanced user profiler with better category matching"""
    
    def __init__(self):
        self.hobby_category_mapping = self._create_enhanced_hobby_mapping()
        self.preference_category_mapping = self._create_preference_mapping()
        self.occasion_category_mapping = self._create_occasion_mapping()
        
    def _create_enhanced_hobby_mapping(self) -> Dict[str, CategoryMapping]:
        """Create comprehensive hobby to category mapping"""
        return {
            "gardening": CategoryMapping(
                primary_categories=["gardening", "outdoor", "home"],
                secondary_categories=["wellness", "books", "tools"],
                weight=1.0,
                age_preference=(25, 70),
                occasion_boost={"mothers_day": 1.5, "fathers_day": 1.3, "birthday": 1.2}
            ),
            "cooking": CategoryMapping(
                primary_categories=["cooking", "food", "kitchen"],
                secondary_categories=["books", "experience", "home"],
                weight=1.0,
                age_preference=(20, 65),
                occasion_boost={"mothers_day": 1.4, "christmas": 1.3, "birthday": 1.2}
            ),
            "reading": CategoryMapping(
                primary_categories=["books", "education"],
                secondary_categories=["art", "wellness", "technology"],
                weight=0.9,
                age_preference=(15, 80),
                occasion_boost={"graduation": 1.5, "birthday": 1.2, "christmas": 1.1}
            ),
            "fitness": CategoryMapping(
                primary_categories=["fitness", "sports", "wellness"],
                secondary_categories=["technology", "outdoor", "clothing"],
                weight=1.0,
                age_preference=(18, 55),
                occasion_boost={"new_year": 1.5, "birthday": 1.3, "promotion": 1.2}
            ),
            "technology": CategoryMapping(
                primary_categories=["technology", "gadgets"],
                secondary_categories=["gaming", "fitness", "education"],
                weight=0.8,  # Lower weight as it's overused
                age_preference=(15, 50),
                occasion_boost={"graduation": 1.4, "birthday": 1.2, "christmas": 1.1}
            ),
            "art": CategoryMapping(
                primary_categories=["art", "craft", "creative"],
                secondary_categories=["books", "home", "experience"],
                weight=1.0,
                age_preference=(18, 65),
                occasion_boost={"birthday": 1.3, "christmas": 1.2, "appreciation": 1.1}
            ),
            "music": CategoryMapping(
                primary_categories=["music", "entertainment"],
                secondary_categories=["technology", "experience", "books"],
                weight=1.0,
                age_preference=(15, 60),
                occasion_boost={"birthday": 1.4, "graduation": 1.2, "christmas": 1.1}
            ),
            "wellness": CategoryMapping(
                primary_categories=["wellness", "health", "self-care"],
                secondary_categories=["books", "home", "experience"],
                weight=1.0,
                age_preference=(20, 65),
                occasion_boost={"mothers_day": 1.5, "birthday": 1.3, "anniversary": 1.2}
            ),
            "outdoor": CategoryMapping(
                primary_categories=["outdoor", "sports", "adventure"],
                secondary_categories=["fitness", "technology", "clothing"],
                weight=1.0,
                age_preference=(18, 55),
                occasion_boost={"birthday": 1.3, "fathers_day": 1.2, "graduation": 1.1}
            ),
            "gaming": CategoryMapping(
                primary_categories=["gaming", "technology", "entertainment"],
                secondary_categories=["books", "art", "music"],
                weight=0.9,
                age_preference=(15, 35),
                occasion_boost={"birthday": 1.4, "graduation": 1.2, "christmas": 1.1}
            ),
            "photography": CategoryMapping(
                primary_categories=["art", "technology", "creative"],
                secondary_categories=["books", "experience", "outdoor"],
                weight=1.0,
                age_preference=(20, 55),
                occasion_boost={"birthday": 1.3, "graduation": 1.2, "appreciation": 1.1}
            ),
            "design": CategoryMapping(
                primary_categories=["art", "creative", "technology"],
                secondary_categories=["books", "home", "experience"],
                weight=1.0,
                age_preference=(20, 50),
                occasion_boost={"birthday": 1.3, "appreciation": 1.2, "promotion": 1.1}
            ),
            "travel": CategoryMapping(
                primary_categories=["experience", "adventure", "books"],
                secondary_categories=["technology", "clothing", "outdoor"],
                weight=1.0,
                age_preference=(25, 65),
                occasion_boost={"anniversary": 1.4, "birthday": 1.2, "promotion": 1.1}
            ),
            "wine": CategoryMapping(
                primary_categories=["food", "experience", "luxury"],
                secondary_categories=["books", "home", "cooking"],
                weight=1.0,
                age_preference=(25, 70),
                occasion_boost={"anniversary": 1.5, "appreciation": 1.3, "fathers_day": 1.2}
            ),
            "business": CategoryMapping(
                primary_categories=["books", "experience", "technology"],
                secondary_categories=["fashion", "luxury", "education"],
                weight=1.0,
                age_preference=(25, 65),
                occasion_boost={"promotion": 1.5, "appreciation": 1.3, "birthday": 1.1}
            ),
            "environment": CategoryMapping(
                primary_categories=["gardening", "books", "wellness"],
                secondary_categories=["home", "outdoor", "experience"],
                weight=1.0,
                age_preference=(20, 60),
                occasion_boost={"anniversary": 1.3, "birthday": 1.2, "mothers_day": 1.1}
            ),
            "sustainability": CategoryMapping(
                primary_categories=["gardening", "wellness", "home"],
                secondary_categories=["books", "outdoor", "cooking"],
                weight=1.0,
                age_preference=(20, 55),
                occasion_boost={"anniversary": 1.3, "birthday": 1.2, "mothers_day": 1.1}
            ),
            "home_decor": CategoryMapping(
                primary_categories=["home", "art", "creative"],
                secondary_categories=["books", "wellness", "gardening"],
                weight=1.0,
                age_preference=(25, 65),
                occasion_boost={"mothers_day": 1.4, "anniversary": 1.3, "birthday": 1.2}
            ),
            "studying": CategoryMapping(
                primary_categories=["education", "books", "technology"],
                secondary_categories=["wellness", "fitness", "art"],
                weight=1.0,
                age_preference=(15, 30),
                occasion_boost={"graduation": 1.5, "birthday": 1.2, "new_year": 1.1}
            )
        }
    
    def _create_preference_mapping(self) -> Dict[str, List[str]]:
        """Map personality preferences to categories"""
        return {
            "trendy": ["technology", "fashion", "art", "music"],
            "practical": ["home", "cooking", "tools", "books"],
            "tech-savvy": ["technology", "gadgets", "gaming"],
            "relaxing": ["wellness", "books", "home", "art"],
            "self-care": ["wellness", "beauty", "books", "experience"],
            "affordable": ["books", "art", "home", "cooking"],
            "traditional": ["books", "cooking", "home", "gardening"],
            "quality": ["luxury", "books", "cooking", "art"],
            "active": ["fitness", "sports", "outdoor", "wellness"],
            "healthy": ["wellness", "fitness", "cooking", "books"],
            "motivational": ["books", "experience", "fitness", "art"],
            "creative": ["art", "craft", "books", "music"],
            "unique": ["art", "craft", "experience", "books"],
            "artistic": ["art", "craft", "books", "music"],
            "luxury": ["luxury", "fashion", "experience", "food"],
            "professional": ["books", "technology", "fashion", "experience"],
            "sophisticated": ["books", "art", "luxury", "experience"],
            "eco-friendly": ["gardening", "wellness", "books", "home"],
            "sustainable": ["gardening", "wellness", "home", "books"],
            "natural": ["gardening", "wellness", "outdoor", "home"]
        }
    
    def _create_occasion_mapping(self) -> Dict[str, List[str]]:
        """Map occasions to preferred categories"""
        return {
            "birthday": ["books", "art", "technology", "experience", "wellness"],
            "christmas": ["books", "art", "home", "cooking", "technology"],
            "mothers_day": ["wellness", "books", "home", "gardening", "cooking"],
            "fathers_day": ["books", "tools", "outdoor", "cooking", "technology"],
            "graduation": ["books", "technology", "experience", "art", "education"],
            "anniversary": ["experience", "books", "art", "luxury", "wellness"],
            "promotion": ["books", "experience", "luxury", "technology", "art"],
            "appreciation": ["books", "art", "experience", "luxury", "wellness"],
            "new_year": ["fitness", "books", "wellness", "experience", "art"],
            "wedding": ["home", "experience", "art", "books", "luxury"]
        }
    
    def calculate_category_scores(self, user_profile, available_categories: List[str]) -> Dict[str, float]:
        """
        Calculate enhanced category scores based on user profile
        
        Args:
            user_profile: UserProfile object
            available_categories: List of available gift categories
            
        Returns:
            Dictionary mapping categories to scores (0-1)
        """
        category_scores = {cat: 0.0 for cat in available_categories}
        
        # 1. Hobby-based scoring (40% weight)
        hobby_scores = self._calculate_hobby_scores(user_profile, available_categories)
        for cat, score in hobby_scores.items():
            category_scores[cat] += 0.4 * score
        
        # 2. Preference-based scoring (25% weight)
        preference_scores = self._calculate_preference_scores(user_profile, available_categories)
        for cat, score in preference_scores.items():
            category_scores[cat] += 0.25 * score
        
        # 3. Occasion-based scoring (20% weight)
        occasion_scores = self._calculate_occasion_scores(user_profile, available_categories)
        for cat, score in occasion_scores.items():
            category_scores[cat] += 0.2 * score
        
        # 4. Age appropriateness (10% weight)
        age_scores = self._calculate_age_scores(user_profile, available_categories)
        for cat, score in age_scores.items():
            category_scores[cat] += 0.1 * score
        
        # 5. Diversity penalty for overused categories (5% weight)
        diversity_penalty = self._calculate_diversity_penalty(available_categories)
        for cat, penalty in diversity_penalty.items():
            category_scores[cat] -= 0.05 * penalty
        
        # Normalize scores to 0-1 range
        max_score = max(category_scores.values()) if category_scores.values() else 1.0
        if max_score > 0:
            category_scores = {cat: score / max_score for cat, score in category_scores.items()}
        
        return category_scores
    
    def _calculate_hobby_scores(self, user_profile, available_categories: List[str]) -> Dict[str, float]:
        """Calculate scores based on user hobbies"""
        scores = {cat: 0.0 for cat in available_categories}
        
        if not user_profile.hobbies:
            return scores
        
        for hobby in user_profile.hobbies:
            if hobby in self.hobby_category_mapping:
                mapping = self.hobby_category_mapping[hobby]
                
                # Age-based weight adjustment
                age_weight = self._calculate_age_weight(user_profile.age, mapping.age_preference)
                
                # Occasion boost
                occasion_boost = mapping.occasion_boost.get(user_profile.occasion, 1.0)
                
                # Primary categories (full weight)
                for cat in mapping.primary_categories:
                    if cat in available_categories:
                        scores[cat] += mapping.weight * age_weight * occasion_boost
                
                # Secondary categories (half weight)
                for cat in mapping.secondary_categories:
                    if cat in available_categories:
                        scores[cat] += 0.5 * mapping.weight * age_weight * occasion_boost
        
        # Normalize by number of hobbies
        if user_profile.hobbies:
            scores = {cat: score / len(user_profile.hobbies) for cat, score in scores.items()}
        
        return scores
    
    def _calculate_preference_scores(self, user_profile, available_categories: List[str]) -> Dict[str, float]:
        """Calculate scores based on personality preferences"""
        scores = {cat: 0.0 for cat in available_categories}
        
        if not user_profile.personality_traits:
            return scores
        
        for preference in user_profile.personality_traits:
            if preference in self.preference_category_mapping:
                preferred_categories = self.preference_category_mapping[preference]
                for cat in preferred_categories:
                    if cat in available_categories:
                        scores[cat] += 1.0 / len(preferred_categories)
        
        # Normalize by number of preferences
        if user_profile.personality_traits:
            scores = {cat: score / len(user_profile.personality_traits) for cat, score in scores.items()}
        
        return scores
    
    def _calculate_occasion_scores(self, user_profile, available_categories: List[str]) -> Dict[str, float]:
        """Calculate scores based on occasion"""
        scores = {cat: 0.0 for cat in available_categories}
        
        if user_profile.occasion in self.occasion_category_mapping:
            preferred_categories = self.occasion_category_mapping[user_profile.occasion]
            for i, cat in enumerate(preferred_categories):
                if cat in available_categories:
                    # Higher score for earlier categories in the list
                    scores[cat] = (len(preferred_categories) - i) / len(preferred_categories)
        
        return scores
    
    def _calculate_age_scores(self, user_profile, available_categories: List[str]) -> Dict[str, float]:
        """Calculate age appropriateness scores"""
        scores = {cat: 0.5 for cat in available_categories}  # Default neutral score
        
        # Age-based category preferences
        age_preferences = {
            (15, 25): ["technology", "gaming", "music", "art", "books"],
            (25, 35): ["technology", "fitness", "art", "books", "experience"],
            (35, 50): ["books", "wellness", "home", "cooking", "art"],
            (50, 65): ["books", "gardening", "cooking", "wellness", "art"],
            (65, 100): ["books", "gardening", "wellness", "home", "art"]
        }
        
        user_age = user_profile.age
        for age_range, preferred_cats in age_preferences.items():
            if age_range[0] <= user_age <= age_range[1]:
                for cat in preferred_cats:
                    if cat in available_categories:
                        scores[cat] = 1.0
                break
        
        return scores
    
    def _calculate_age_weight(self, user_age: int, age_preference: Tuple[int, int]) -> float:
        """Calculate age-based weight for hobby mapping"""
        min_age, max_age = age_preference
        
        if min_age <= user_age <= max_age:
            return 1.0
        elif abs(user_age - min_age) <= 10 or abs(user_age - max_age) <= 10:
            return 0.7
        else:
            return 0.3
    
    def _calculate_diversity_penalty(self, available_categories: List[str]) -> Dict[str, float]:
        """Calculate penalty for overused categories"""
        penalties = {cat: 0.0 for cat in available_categories}
        
        # Penalize technology category as it's overused
        if "technology" in available_categories:
            penalties["technology"] = 0.3
        
        return penalties
    
    def get_top_categories(self, user_profile, available_categories: List[str], 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K categories for the user"""
        scores = self.calculate_category_scores(user_profile, available_categories)
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_categories[:top_k]


def create_enhanced_category_matcher():
    """Factory function to create enhanced category matcher"""
    return EnhancedUserProfiler()


if __name__ == "__main__":
    # Test the enhanced profiler
    from models.rl.environment import UserProfile
    
    profiler = EnhancedUserProfiler()
    
    # Test user profile
    user = UserProfile(
        age=35,
        hobbies=["gardening", "cooking", "wellness"],
        relationship="mother",
        budget=100.0,
        occasion="mothers_day",
        personality_traits=["eco-friendly", "practical", "relaxing"]
    )
    
    available_categories = ["technology", "gardening", "cooking", "books", "wellness", "art", "fitness"]
    
    scores = profiler.calculate_category_scores(user, available_categories)
    top_categories = profiler.get_top_categories(user, available_categories, top_k=3)
    
    print("Category Scores:")
    for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {score:.3f}")
    
    print(f"\nTop 3 Categories:")
    for cat, score in top_categories:
        print(f"  {cat}: {score:.3f}")