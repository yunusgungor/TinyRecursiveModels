"""
Enhanced Recommendation Engine with Better Category Matching
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import random

from .environment import UserProfile, GiftItem, EnvironmentState
from .enhanced_user_profiler import EnhancedUserProfiler


class EnhancedRecommendationEngine:
    """Enhanced recommendation engine with improved category matching"""
    
    def __init__(self, gift_catalog: List[GiftItem]):
        self.gift_catalog = gift_catalog
        self.user_profiler = EnhancedUserProfiler()
        
        # Create category index
        self.category_index = self._build_category_index()
        
        # Track recommendation history for diversity
        self.recommendation_history = []
        
    def _build_category_index(self) -> Dict[str, List[GiftItem]]:
        """Build index of gifts by category"""
        index = {}
        for gift in self.gift_catalog:
            if gift.category not in index:
                index[gift.category] = []
            index[gift.category].append(gift)
        return index
    
    def generate_recommendations(self, user_profile: UserProfile, 
                               max_recommendations: int = 5,
                               diversity_weight: float = 0.3) -> List[Tuple[GiftItem, float]]:
        """
        Generate enhanced recommendations with better category matching
        
        Args:
            user_profile: User profile
            max_recommendations: Maximum number of recommendations
            diversity_weight: Weight for diversity in recommendations
            
        Returns:
            List of (gift, confidence_score) tuples
        """
        available_categories = list(self.category_index.keys())
        
        # Get category scores from enhanced profiler
        category_scores = self.user_profiler.calculate_category_scores(
            user_profile, available_categories
        )
        
        # Generate candidate recommendations
        candidates = []
        
        for category, category_score in category_scores.items():
            if category_score > 0.1:  # Only consider categories with reasonable scores
                category_gifts = self.category_index.get(category, [])
                
                for gift in category_gifts:
                    # Calculate individual gift score
                    gift_score = self._calculate_gift_score(
                        gift, user_profile, category_score
                    )
                    
                    if gift_score > 0.2:  # Minimum threshold
                        candidates.append((gift, gift_score))
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity selection
        final_recommendations = self._apply_diversity_selection(
            candidates, max_recommendations, diversity_weight
        )
        
        # Update recommendation history
        self.recommendation_history.extend([gift for gift, _ in final_recommendations])
        
        return final_recommendations
    
    def _calculate_gift_score(self, gift: GiftItem, user_profile: UserProfile, 
                             category_score: float) -> float:
        """Calculate comprehensive score for a specific gift"""
        score = 0.0
        
        # 1. Category score (30%)
        score += 0.3 * category_score
        
        # 2. Budget compatibility (25%)
        budget_score = self._calculate_budget_compatibility(gift.price, user_profile.budget)
        score += 0.25 * budget_score
        
        # 3. Hobby alignment (20%)
        hobby_score = self._calculate_hobby_alignment(gift, user_profile.hobbies)
        score += 0.2 * hobby_score
        
        # 4. Occasion appropriateness (15%)
        occasion_score = self._calculate_occasion_appropriateness(gift, user_profile.occasion)
        score += 0.15 * occasion_score
        
        # 5. Age appropriateness (10%)
        age_score = self._calculate_age_appropriateness(gift, user_profile.age)
        score += 0.1 * age_score
        
        return min(1.0, max(0.0, score))
    
    def _calculate_budget_compatibility(self, price: float, budget: float) -> float:
        """Calculate budget compatibility score"""
        if price <= budget * 0.7:  # Well within budget
            return 1.0
        elif price <= budget * 0.9:  # Comfortably within budget
            return 0.9
        elif price <= budget:  # Within budget
            return 0.7
        elif price <= budget * 1.1:  # Slightly over budget
            return 0.4
        elif price <= budget * 1.2:  # Moderately over budget
            return 0.2
        else:  # Way over budget
            return 0.0
    
    def _calculate_hobby_alignment(self, gift: GiftItem, hobbies: List[str]) -> float:
        """Calculate how well gift aligns with user hobbies"""
        if not hobbies:
            return 0.5
        
        alignment_score = 0.0
        
        # Direct category match
        for hobby in hobbies:
            if hobby.lower() in gift.category.lower():
                alignment_score += 1.0
                break
        
        # Tag matches
        tag_matches = 0
        for hobby in hobbies:
            for tag in gift.tags:
                if hobby.lower() in tag.lower() or tag.lower() in hobby.lower():
                    tag_matches += 1
                    break
        
        if tag_matches > 0:
            alignment_score += min(1.0, tag_matches / len(hobbies))
        
        # Semantic matches (simplified)
        semantic_matches = self._calculate_semantic_matches(gift, hobbies)
        alignment_score += 0.5 * semantic_matches
        
        return min(1.0, alignment_score / 2.0)  # Normalize
    
    def _calculate_semantic_matches(self, gift: GiftItem, hobbies: List[str]) -> float:
        """Calculate semantic matches between gift and hobbies"""
        semantic_mapping = {
            "gardening": ["plant", "seed", "garden", "outdoor", "nature", "grow"],
            "cooking": ["kitchen", "recipe", "food", "chef", "culinary", "bake"],
            "reading": ["book", "novel", "literature", "story", "author", "read"],
            "fitness": ["exercise", "workout", "health", "gym", "sport", "active"],
            "technology": ["tech", "digital", "smart", "electronic", "gadget", "device"],
            "art": ["creative", "paint", "draw", "craft", "design", "artistic"],
            "music": ["sound", "audio", "instrument", "song", "melody", "rhythm"],
            "wellness": ["relax", "calm", "peace", "meditation", "spa", "therapy"]
        }
        
        matches = 0
        total_checks = 0
        
        for hobby in hobbies:
            if hobby in semantic_mapping:
                keywords = semantic_mapping[hobby]
                total_checks += len(keywords)
                
                for keyword in keywords:
                    if (keyword in gift.name.lower() or 
                        keyword in gift.description.lower() or
                        any(keyword in tag.lower() for tag in gift.tags)):
                        matches += 1
        
        return matches / total_checks if total_checks > 0 else 0.0
    
    def _calculate_occasion_appropriateness(self, gift: GiftItem, occasion: str) -> float:
        """Calculate occasion appropriateness score"""
        if not occasion:
            return 0.7  # Neutral score
        
        # Direct match
        if occasion.lower() in [occ.lower() for occ in gift.occasion_fit]:
            return 1.0
        
        # Occasion compatibility mapping
        occasion_compatibility = {
            "birthday": ["christmas", "graduation", "appreciation"],
            "mothers_day": ["birthday", "anniversary", "appreciation"],
            "fathers_day": ["birthday", "appreciation"],
            "graduation": ["birthday", "appreciation", "promotion"],
            "anniversary": ["birthday", "mothers_day", "appreciation"],
            "promotion": ["graduation", "appreciation", "birthday"],
            "christmas": ["birthday", "appreciation"]
        }
        
        if occasion in occasion_compatibility:
            compatible_occasions = occasion_compatibility[occasion]
            for occ in gift.occasion_fit:
                if occ.lower() in [c.lower() for c in compatible_occasions]:
                    return 0.7
        
        # Generic gifts
        if not gift.occasion_fit or "any" in gift.occasion_fit:
            return 0.5
        
        return 0.2  # Not appropriate
    
    def _calculate_age_appropriateness(self, gift: GiftItem, age: int) -> float:
        """Calculate age appropriateness score"""
        min_age, max_age = gift.age_suitability
        
        if min_age <= age <= max_age:
            return 1.0
        elif abs(age - min_age) <= 5 or abs(age - max_age) <= 5:
            return 0.8  # Close to appropriate range
        elif abs(age - min_age) <= 10 or abs(age - max_age) <= 10:
            return 0.6  # Somewhat appropriate
        else:
            return 0.3  # Not very appropriate
    
    def _apply_diversity_selection(self, candidates: List[Tuple[GiftItem, float]], 
                                  max_recommendations: int, 
                                  diversity_weight: float) -> List[Tuple[GiftItem, float]]:
        """Apply diversity selection to avoid recommending similar items"""
        if len(candidates) <= max_recommendations:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # Always select the top candidate
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining candidates with diversity consideration
        while len(selected) < max_recommendations and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for i, (candidate_gift, candidate_score) in enumerate(remaining):
                # Calculate diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(
                    candidate_gift, [gift for gift, _ in selected]
                )
                
                # Combined score
                combined_score = (1 - diversity_weight) * candidate_score + diversity_weight * diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (candidate_gift, candidate_score)
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        return selected
    
    def _calculate_diversity_bonus(self, candidate: GiftItem, selected_gifts: List[GiftItem]) -> float:
        """Calculate diversity bonus for a candidate gift"""
        if not selected_gifts:
            return 1.0
        
        # Category diversity
        selected_categories = set(gift.category for gift in selected_gifts)
        category_bonus = 1.0 if candidate.category not in selected_categories else 0.3
        
        # Price range diversity
        selected_prices = [gift.price for gift in selected_gifts]
        avg_price = sum(selected_prices) / len(selected_prices)
        price_diff = abs(candidate.price - avg_price) / avg_price
        price_bonus = min(1.0, price_diff)
        
        # Tag diversity
        selected_tags = set()
        for gift in selected_gifts:
            selected_tags.update(gift.tags)
        
        candidate_tags = set(candidate.tags)
        tag_overlap = len(candidate_tags.intersection(selected_tags))
        tag_bonus = 1.0 - (tag_overlap / len(candidate_tags)) if candidate_tags else 0.5
        
        # Combined diversity score
        diversity_score = (0.5 * category_bonus + 0.3 * price_bonus + 0.2 * tag_bonus)
        return diversity_score
    
    def get_category_distribution(self, recommendations: List[Tuple[GiftItem, float]]) -> Dict[str, int]:
        """Get category distribution of recommendations"""
        distribution = {}
        for gift, _ in recommendations:
            category = gift.category
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def explain_recommendations(self, user_profile: UserProfile, 
                               recommendations: List[Tuple[GiftItem, float]]) -> List[Dict[str, Any]]:
        """Generate explanations for recommendations"""
        explanations = []
        
        for gift, score in recommendations:
            explanation = {
                "gift": gift.name,
                "category": gift.category,
                "price": gift.price,
                "confidence": score,
                "reasons": []
            }
            
            # Budget reason
            if gift.price <= user_profile.budget * 0.8:
                explanation["reasons"].append("Well within your budget")
            elif gift.price <= user_profile.budget:
                explanation["reasons"].append("Fits your budget")
            
            # Hobby alignment
            for hobby in user_profile.hobbies:
                if (hobby.lower() in gift.category.lower() or 
                    any(hobby.lower() in tag.lower() for tag in gift.tags)):
                    explanation["reasons"].append(f"Matches your interest in {hobby}")
                    break
            
            # Occasion appropriateness
            if user_profile.occasion.lower() in [occ.lower() for occ in gift.occasion_fit]:
                explanation["reasons"].append(f"Perfect for {user_profile.occasion}")
            
            # Quality
            if gift.rating >= 4.5:
                explanation["reasons"].append("Highly rated product")
            elif gift.rating >= 4.0:
                explanation["reasons"].append("Well-rated product")
            
            explanations.append(explanation)
        
        return explanations


if __name__ == "__main__":
    # Test the enhanced recommendation engine
    from models.rl.environment import UserProfile, GiftItem
    
    # Sample gift catalog
    sample_gifts = [
        GiftItem("1", "Organic Seed Set", "gardening", 75.0, 4.8, 
                ["organic", "sustainable", "educational"], 
                "Premium organic vegetable seeds", (25, 65), ["birthday", "mothers_day"]),
        GiftItem("2", "Cooking Masterclass", "cooking", 120.0, 4.9,
                ["educational", "experience", "skill"], 
                "Online cooking course", (20, 60), ["birthday", "christmas"]),
        GiftItem("3", "Wellness Journal", "wellness", 35.0, 4.6,
                ["self-care", "mindfulness", "writing"], 
                "Guided wellness and gratitude journal", (18, 65), ["birthday", "mothers_day"]),
        GiftItem("4", "Smart Fitness Watch", "technology", 199.99, 4.3,
                ["fitness", "health", "smart"], 
                "Advanced fitness tracking watch", (18, 55), ["birthday", "graduation"]),
        GiftItem("5", "Herbal Tea Set", "wellness", 45.0, 4.7,
                ["relaxation", "natural", "health"], 
                "Premium herbal tea collection", (20, 70), ["mothers_day", "birthday"])
    ]
    
    # Test user profile
    user = UserProfile(
        age=35,
        hobbies=["gardening", "wellness", "cooking"],
        relationship="mother",
        budget=100.0,
        occasion="mothers_day",
        personality_traits=["eco-friendly", "practical", "relaxing"]
    )
    
    # Create recommendation engine
    engine = EnhancedRecommendationEngine(sample_gifts)
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(user, max_recommendations=3)
    
    print("Enhanced Recommendations:")
    for i, (gift, score) in enumerate(recommendations, 1):
        print(f"{i}. {gift.name} (${gift.price:.2f}) - {gift.category}")
        print(f"   Confidence: {score:.3f}")
        print(f"   Rating: {gift.rating}/5.0")
        print()
    
    # Get explanations
    explanations = engine.explain_recommendations(user, recommendations)
    print("Explanations:")
    for exp in explanations:
        print(f"• {exp['gift']}: {', '.join(exp['reasons'])}")