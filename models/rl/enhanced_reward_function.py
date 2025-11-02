"""
Enhanced Reward Function for Better Category Matching
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from .environment import UserProfile, GiftItem
from .enhanced_user_profiler import EnhancedUserProfiler


class EnhancedRewardFunction:
    """Enhanced reward function that better incentivizes category matching"""
    
    def __init__(self):
        self.user_profiler = EnhancedUserProfiler()
        
        # Reward weights
        self.weights = {
            "category_match": 0.35,      # Increased from 0.25
            "budget_compatibility": 0.25, # Slightly reduced from 0.30
            "hobby_alignment": 0.20,     # Same
            "occasion_appropriateness": 0.15, # Slightly reduced from 0.20
            "age_appropriateness": 0.05  # Reduced from 0.15
        }
        
        # Bonus weights
        self.bonus_weights = {
            "diversity_bonus": 0.10,
            "quality_bonus": 0.05,
            "tool_usage_bonus": 0.15,
            "perfect_match_bonus": 0.20
        }
        
        # Penalty weights
        self.penalty_weights = {
            "overused_category_penalty": 0.15,
            "poor_match_penalty": 0.10,
            "budget_violation_penalty": 0.25
        }
    
    def calculate_reward(self, user_profile: UserProfile, 
                        recommended_gifts: List[GiftItem],
                        confidence_scores: List[float],
                        tool_calls: List[Any] = None,
                        available_categories: List[str] = None) -> float:
        """
        Calculate enhanced reward for recommendations
        
        Args:
            user_profile: User profile
            recommended_gifts: List of recommended gifts
            confidence_scores: Confidence scores for recommendations
            tool_calls: List of tool calls made (optional)
            available_categories: Available gift categories (optional)
            
        Returns:
            Reward score between -1.0 and 1.0
        """
        if not recommended_gifts:
            return -0.8  # Heavy penalty for no recommendations
        
        # Calculate base reward components
        base_reward = self._calculate_base_reward(user_profile, recommended_gifts, confidence_scores)
        
        # Calculate bonuses
        bonuses = self._calculate_bonuses(user_profile, recommended_gifts, tool_calls, available_categories)
        
        # Calculate penalties
        penalties = self._calculate_penalties(user_profile, recommended_gifts, available_categories)
        
        # Combine all components
        total_reward = base_reward + bonuses - penalties
        
        # Clamp to valid range
        return max(-1.0, min(1.0, total_reward))
    
    def _calculate_base_reward(self, user_profile: UserProfile, 
                              recommended_gifts: List[GiftItem],
                              confidence_scores: List[float]) -> float:
        """Calculate base reward components"""
        total_reward = 0.0
        
        for i, gift in enumerate(recommended_gifts):
            confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
            gift_reward = 0.0
            
            # 1. Category matching (most important)
            category_score = self._calculate_enhanced_category_score(gift, user_profile)
            gift_reward += self.weights["category_match"] * category_score
            
            # 2. Budget compatibility
            budget_score = self._calculate_budget_score(gift.price, user_profile.budget)
            gift_reward += self.weights["budget_compatibility"] * budget_score
            
            # 3. Hobby alignment
            hobby_score = self._calculate_hobby_score(gift, user_profile.hobbies)
            gift_reward += self.weights["hobby_alignment"] * hobby_score
            
            # 4. Occasion appropriateness
            occasion_score = self._calculate_occasion_score(gift, user_profile.occasion)
            gift_reward += self.weights["occasion_appropriateness"] * occasion_score
            
            # 5. Age appropriateness
            age_score = self._calculate_age_score(gift, user_profile.age)
            gift_reward += self.weights["age_appropriateness"] * age_score
            
            # Apply confidence weighting
            gift_reward *= confidence
            total_reward += gift_reward
        
        # Average across recommendations
        return total_reward / len(recommended_gifts)
    
    def _calculate_enhanced_category_score(self, gift: GiftItem, user_profile: UserProfile) -> float:
        """Calculate enhanced category matching score"""
        available_categories = [gift.category]  # Simplified for single gift
        category_scores = self.user_profiler.calculate_category_scores(user_profile, available_categories)
        
        base_score = category_scores.get(gift.category, 0.0)
        
        # Boost score for perfect matches
        perfect_match_boost = 0.0
        for hobby in user_profile.hobbies:
            if hobby.lower() == gift.category.lower():
                perfect_match_boost = 0.3
                break
            elif hobby.lower() in gift.category.lower() or gift.category.lower() in hobby.lower():
                perfect_match_boost = 0.2
                break
        
        # Boost for preference alignment
        preference_boost = 0.0
        for preference in user_profile.personality_traits:
            if self._preference_matches_category(preference, gift.category):
                preference_boost += 0.1
        
        preference_boost = min(0.3, preference_boost)  # Cap at 0.3
        
        return min(1.0, base_score + perfect_match_boost + preference_boost)
    
    def _preference_matches_category(self, preference: str, category: str) -> bool:
        """Check if preference matches category"""
        preference_category_map = {
            "eco-friendly": ["gardening", "wellness", "outdoor"],
            "practical": ["home", "cooking", "tools", "books"],
            "relaxing": ["wellness", "books", "art", "home"],
            "trendy": ["technology", "fashion", "art"],
            "creative": ["art", "craft", "books", "music"],
            "active": ["fitness", "sports", "outdoor"],
            "luxury": ["luxury", "fashion", "experience"],
            "traditional": ["books", "cooking", "home", "gardening"]
        }
        
        if preference in preference_category_map:
            return category in preference_category_map[preference]
        
        return False
    
    def _calculate_budget_score(self, price: float, budget: float) -> float:
        """Calculate budget compatibility score with enhanced scoring"""
        ratio = price / budget if budget > 0 else float('inf')
        
        if ratio <= 0.6:  # Great value
            return 1.0
        elif ratio <= 0.8:  # Good value
            return 0.9
        elif ratio <= 0.95:  # Fair value
            return 0.8
        elif ratio <= 1.0:  # Within budget
            return 0.6
        elif ratio <= 1.1:  # Slightly over
            return 0.3
        elif ratio <= 1.2:  # Moderately over
            return 0.1
        else:  # Way over budget
            return 0.0
    
    def _calculate_hobby_score(self, gift: GiftItem, hobbies: List[str]) -> float:
        """Calculate hobby alignment score"""
        if not hobbies:
            return 0.5
        
        score = 0.0
        
        # Direct category match
        for hobby in hobbies:
            if hobby.lower() in gift.category.lower():
                score += 1.0
                break
        
        # Tag matches
        tag_matches = 0
        for hobby in hobbies:
            for tag in gift.tags:
                if (hobby.lower() in tag.lower() or 
                    tag.lower() in hobby.lower() or
                    self._are_semantically_related(hobby, tag)):
                    tag_matches += 1
                    break
        
        if tag_matches > 0:
            score += min(0.8, tag_matches / len(hobbies))
        
        # Name/description matches
        name_desc_text = (gift.name + " " + gift.description).lower()
        name_matches = sum(1 for hobby in hobbies if hobby.lower() in name_desc_text)
        if name_matches > 0:
            score += min(0.6, name_matches / len(hobbies))
        
        return min(1.0, score / 2.0)  # Normalize
    
    def _are_semantically_related(self, hobby: str, tag: str) -> bool:
        """Check if hobby and tag are semantically related"""
        semantic_relations = {
            "gardening": ["plant", "seed", "grow", "outdoor", "nature", "organic"],
            "cooking": ["kitchen", "food", "recipe", "culinary", "chef", "bake"],
            "wellness": ["health", "relax", "calm", "spa", "meditation", "mindful"],
            "fitness": ["exercise", "workout", "sport", "active", "gym", "health"],
            "technology": ["smart", "digital", "electronic", "tech", "gadget", "device"],
            "art": ["creative", "paint", "draw", "craft", "design", "artistic"],
            "reading": ["book", "literature", "story", "novel", "educational"]
        }
        
        hobby_lower = hobby.lower()
        tag_lower = tag.lower()
        
        if hobby_lower in semantic_relations:
            return any(related in tag_lower for related in semantic_relations[hobby_lower])
        
        return False
    
    def _calculate_occasion_score(self, gift: GiftItem, occasion: str) -> float:
        """Calculate occasion appropriateness score"""
        if not occasion:
            return 0.7
        
        occasion_lower = occasion.lower()
        
        # Direct match
        if occasion_lower in [occ.lower() for occ in gift.occasion_fit]:
            return 1.0
        
        # Compatible occasions
        occasion_compatibility = {
            "birthday": ["christmas", "graduation", "appreciation", "any"],
            "mothers_day": ["birthday", "anniversary", "appreciation", "any"],
            "fathers_day": ["birthday", "appreciation", "any"],
            "graduation": ["birthday", "appreciation", "promotion", "any"],
            "anniversary": ["birthday", "mothers_day", "appreciation", "any"],
            "promotion": ["graduation", "appreciation", "birthday", "any"],
            "christmas": ["birthday", "appreciation", "any"]
        }
        
        if occasion in occasion_compatibility:
            compatible = occasion_compatibility[occasion]
            for occ in gift.occasion_fit:
                if occ.lower() in [c.lower() for c in compatible]:
                    return 0.8
        
        # Generic gifts
        if not gift.occasion_fit or "any" in [occ.lower() for occ in gift.occasion_fit]:
            return 0.6
        
        return 0.2
    
    def _calculate_age_score(self, gift: GiftItem, age: int) -> float:
        """Calculate age appropriateness score"""
        min_age, max_age = gift.age_suitability
        
        if min_age <= age <= max_age:
            return 1.0
        
        # Calculate distance from age range
        if age < min_age:
            distance = min_age - age
        else:
            distance = age - max_age
        
        if distance <= 5:
            return 0.8
        elif distance <= 10:
            return 0.6
        elif distance <= 15:
            return 0.4
        else:
            return 0.2
    
    def _calculate_bonuses(self, user_profile: UserProfile, 
                          recommended_gifts: List[GiftItem],
                          tool_calls: List[Any] = None,
                          available_categories: List[str] = None) -> float:
        """Calculate bonus rewards"""
        total_bonus = 0.0
        
        # 1. Diversity bonus
        if len(recommended_gifts) > 1:
            categories = set(gift.category for gift in recommended_gifts)
            diversity_ratio = len(categories) / len(recommended_gifts)
            total_bonus += self.bonus_weights["diversity_bonus"] * diversity_ratio
        
        # 2. Quality bonus
        avg_rating = sum(gift.rating for gift in recommended_gifts) / len(recommended_gifts)
        if avg_rating >= 4.5:
            total_bonus += self.bonus_weights["quality_bonus"]
        elif avg_rating >= 4.0:
            total_bonus += self.bonus_weights["quality_bonus"] * 0.5
        
        # 3. Tool usage bonus
        if tool_calls:
            successful_tools = sum(1 for call in tool_calls if getattr(call, 'success', False))
            if successful_tools > 0:
                tool_bonus = min(self.bonus_weights["tool_usage_bonus"], 
                               successful_tools * 0.05)
                total_bonus += tool_bonus
        
        # 4. Perfect match bonus
        perfect_matches = 0
        for gift in recommended_gifts:
            for hobby in user_profile.hobbies:
                if hobby.lower() == gift.category.lower():
                    perfect_matches += 1
                    break
        
        if perfect_matches > 0:
            perfect_bonus = min(self.bonus_weights["perfect_match_bonus"],
                              perfect_matches * 0.1)
            total_bonus += perfect_bonus
        
        return total_bonus
    
    def _calculate_penalties(self, user_profile: UserProfile,
                           recommended_gifts: List[GiftItem],
                           available_categories: List[str] = None) -> float:
        """Calculate penalty deductions"""
        total_penalty = 0.0
        
        # 1. Overused category penalty (especially technology)
        category_counts = {}
        for gift in recommended_gifts:
            category_counts[gift.category] = category_counts.get(gift.category, 0) + 1
        
        for category, count in category_counts.items():
            if category == "technology" and count > 1:
                # Heavy penalty for multiple technology recommendations
                total_penalty += self.penalty_weights["overused_category_penalty"] * (count - 1) * 0.5
            elif count > 2:
                # Penalty for too many items from same category
                total_penalty += self.penalty_weights["overused_category_penalty"] * (count - 2) * 0.3
        
        # 2. Poor match penalty
        poor_matches = 0
        for gift in recommended_gifts:
            category_score = self._calculate_enhanced_category_score(gift, user_profile)
            if category_score < 0.3:
                poor_matches += 1
        
        if poor_matches > 0:
            total_penalty += self.penalty_weights["poor_match_penalty"] * poor_matches * 0.2
        
        # 3. Budget violation penalty
        over_budget_items = sum(1 for gift in recommended_gifts if gift.price > user_profile.budget)
        if over_budget_items > 0:
            total_penalty += self.penalty_weights["budget_violation_penalty"] * over_budget_items * 0.3
        
        return total_penalty
    
    def explain_reward(self, user_profile: UserProfile,
                      recommended_gifts: List[GiftItem],
                      confidence_scores: List[float],
                      tool_calls: List[Any] = None) -> Dict[str, Any]:
        """Generate detailed explanation of reward calculation"""
        explanation = {
            "total_reward": self.calculate_reward(user_profile, recommended_gifts, confidence_scores, tool_calls),
            "components": {},
            "bonuses": {},
            "penalties": {}
        }
        
        # Base components
        base_reward = self._calculate_base_reward(user_profile, recommended_gifts, confidence_scores)
        explanation["components"]["base_reward"] = base_reward
        
        # Individual gift scores
        gift_scores = []
        for i, gift in enumerate(recommended_gifts):
            confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
            gift_score = {
                "gift": gift.name,
                "category_score": self._calculate_enhanced_category_score(gift, user_profile),
                "budget_score": self._calculate_budget_score(gift.price, user_profile.budget),
                "hobby_score": self._calculate_hobby_score(gift, user_profile.hobbies),
                "occasion_score": self._calculate_occasion_score(gift, user_profile.occasion),
                "age_score": self._calculate_age_score(gift, user_profile.age),
                "confidence": confidence
            }
            gift_scores.append(gift_score)
        
        explanation["gift_scores"] = gift_scores
        
        # Bonuses
        bonuses = self._calculate_bonuses(user_profile, recommended_gifts, tool_calls)
        explanation["bonuses"]["total"] = bonuses
        
        # Penalties
        penalties = self._calculate_penalties(user_profile, recommended_gifts)
        explanation["penalties"]["total"] = penalties
        
        return explanation


if __name__ == "__main__":
    # Test the enhanced reward function
    from models.rl.environment import UserProfile, GiftItem
    
    # Sample gifts
    gifts = [
        GiftItem("1", "Organic Seed Set", "gardening", 75.0, 4.8, 
                ["organic", "sustainable"], "Premium seeds", (25, 65), ["mothers_day"]),
        GiftItem("2", "Smart Phone", "technology", 899.0, 4.2,
                ["smart", "communication"], "Latest smartphone", (18, 55), ["birthday"])
    ]
    
    # Test user
    user = UserProfile(
        age=35,
        hobbies=["gardening", "wellness"],
        relationship="mother",
        budget=100.0,
        occasion="mothers_day",
        personality_traits=["eco-friendly", "practical"]
    )
    
    # Test reward function
    reward_func = EnhancedRewardFunction()
    
    # Test with good match
    good_recommendations = [gifts[0]]  # Gardening gift
    good_reward = reward_func.calculate_reward(user, good_recommendations, [0.9])
    print(f"Good match reward: {good_reward:.3f}")
    
    # Test with poor match
    poor_recommendations = [gifts[1]]  # Technology gift
    poor_reward = reward_func.calculate_reward(user, poor_recommendations, [0.8])
    print(f"Poor match reward: {poor_reward:.3f}")
    
    # Get detailed explanation
    explanation = reward_func.explain_reward(user, good_recommendations, [0.9])
    print(f"\nDetailed explanation for good match:")
    print(f"Total reward: {explanation['total_reward']:.3f}")
    print(f"Base reward: {explanation['components']['base_reward']:.3f}")
    print(f"Bonuses: {explanation['bonuses']['total']:.3f}")
    print(f"Penalties: {explanation['penalties']['total']:.3f}")