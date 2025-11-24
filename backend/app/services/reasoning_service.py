"""
Reasoning service for generating human-readable explanations

This service generates dynamic, context-aware reasoning for gift recommendations,
tool selections, category matching, and confidence scores.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from app.models.schemas import UserProfile, GiftItem


logger = logging.getLogger(__name__)


class ReasoningService:
    """Service for generating human-readable reasoning from model outputs"""
    
    def __init__(self):
        """Initialize reasoning service with mappings"""
        self.hobby_category_map = self._load_hobby_category_map()
        self.occasion_category_map = self._load_occasion_category_map()
        logger.info("ReasoningService initialized")
    
    def _load_hobby_category_map(self) -> Dict[str, List[str]]:
        """
        Load hobby to category mappings
        
        Returns:
            Dict mapping hobbies to relevant gift categories
        """
        # Default mappings - can be loaded from file in production
        return {
            "cooking": ["Kitchen & Dining", "Appliances", "Books", "Food & Beverage"],
            "gardening": ["Garden & Outdoor", "Tools", "Books", "Home & Living"],
            "reading": ["Books", "Electronics", "Home & Living", "Stationery"],
            "sports": ["Sports & Outdoor", "Fitness", "Clothing", "Electronics"],
            "technology": ["Electronics", "Computers", "Gaming", "Smart Home"],
            "fitness": ["Sports & Outdoor", "Fitness", "Health & Beauty", "Clothing"],
            "art": ["Art & Craft", "Books", "Stationery", "Home & Living"],
            "music": ["Electronics", "Musical Instruments", "Books", "Entertainment"],
            "travel": ["Luggage & Bags", "Electronics", "Books", "Clothing"],
            "gaming": ["Gaming", "Electronics", "Entertainment", "Furniture"],
            "photography": ["Electronics", "Cameras", "Books", "Accessories"],
            "fashion": ["Clothing", "Accessories", "Jewelry", "Beauty"],
            "beauty": ["Health & Beauty", "Cosmetics", "Accessories", "Wellness"],
            "pets": ["Pet Supplies", "Books", "Home & Living", "Outdoor"],
        }
    
    def _load_occasion_category_map(self) -> Dict[str, List[str]]:
        """
        Load occasion to category mappings
        
        Returns:
            Dict mapping occasions to relevant gift categories
        """
        return {
            "birthday": ["Jewelry", "Electronics", "Books", "Clothing", "Toys", "Entertainment"],
            "christmas": ["Toys", "Electronics", "Home & Living", "Books", "Clothing", "Food & Beverage"],
            "anniversary": ["Jewelry", "Flowers", "Perfume", "Watches", "Home & Living", "Electronics"],
            "graduation": ["Books", "Electronics", "Watches", "Jewelry", "Stationery", "Clothing"],
            "wedding": ["Home & Living", "Kitchen & Dining", "Jewelry", "Appliances", "Decor"],
            "valentine": ["Jewelry", "Flowers", "Perfume", "Chocolate", "Accessories", "Clothing"],
            "mother_day": ["Jewelry", "Flowers", "Perfume", "Home & Living", "Beauty", "Clothing"],
            "father_day": ["Electronics", "Tools", "Watches", "Clothing", "Sports", "Books"],
            "new_year": ["Home & Living", "Electronics", "Clothing", "Accessories", "Books"],
        }
    
    def generate_gift_reasoning(
        self,
        gift: GiftItem,
        user_profile: UserProfile,
        model_output: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate dynamic, context-aware gift reasoning
        
        Args:
            gift: Gift item to generate reasoning for
            user_profile: User profile
            model_output: Model output dictionary
            tool_results: Tool execution results
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        try:
            # Validate inputs
            if not gift or not user_profile:
                logger.warning("Invalid inputs for gift reasoning generation")
                return ["Recommended based on your profile"]
            # 1. Hobby matching
            try:
                matching_hobbies = self._find_matching_hobbies(gift, user_profile.hobbies)
                if matching_hobbies:
                    if len(matching_hobbies) == 1:
                        reasoning.append(
                            f"Perfect match for your hobby: {matching_hobbies[0]}"
                        )
                    else:
                        reasoning.append(
                            f"Perfect match for your hobbies: {', '.join(matching_hobbies)}"
                        )
            except Exception as e:
                logger.warning(f"Error in hobby matching: {str(e)}")
            
            # 2. Budget optimization
            try:
                if user_profile.budget > 0:  # Avoid division by zero
                    budget_usage = (gift.price / user_profile.budget) * 100
                    if budget_usage < 70:
                        reasoning.append(
                            f"Great value: Only {budget_usage:.0f}% of your budget"
                        )
                    elif budget_usage > 95:
                        reasoning.append(
                            f"Premium choice: Uses {budget_usage:.0f}% of budget"
                        )
                    else:
                        reasoning.append(
                            f"Well-balanced: {budget_usage:.0f}% of your budget"
                        )
            except Exception as e:
                logger.warning(f"Error in budget calculation: {str(e)}")
            
            # 3. Tool insights integration
            try:
                if tool_results and isinstance(tool_results, dict):
                    if "review_analysis" in tool_results:
                        review_data = tool_results["review_analysis"]
                        if isinstance(review_data, dict):
                            avg_rating = review_data.get("average_rating", 0)
                            if avg_rating >= 4.0:
                                reasoning.append(f"Highly rated: {avg_rating}/5.0 stars")
                            elif avg_rating >= 3.5:
                                reasoning.append(f"Good reviews: {avg_rating}/5.0 stars")
                    
                    if "trend_analysis" in tool_results:
                        trend_data = tool_results["trend_analysis"]
                        if isinstance(trend_data, dict):
                            trending = trend_data.get("trending", [])
                            if trending and any(str(g) == gift.id or (hasattr(g, 'id') and g.id == gift.id) for g in trending):
                                reasoning.append("Currently trending in this category")
                    
                    if "inventory_check" in tool_results:
                        inventory_data = tool_results["inventory_check"]
                        if isinstance(inventory_data, dict):
                            available = inventory_data.get("available", [])
                            if available and any(str(g) == gift.id or (hasattr(g, 'id') and g.id == gift.id) for g in available):
                                reasoning.append("In stock and ready to ship")
                    
                    if "price_comparison" in tool_results:
                        price_data = tool_results["price_comparison"]
                        if isinstance(price_data, dict):
                            savings = price_data.get("savings_percentage", 0)
                            if savings > 10:
                                reasoning.append(f"Great deal: {savings:.0f}% savings compared to average")
            except Exception as e:
                logger.warning(f"Error integrating tool insights: {str(e)}")
            
            # 4. Age appropriateness
            try:
                if hasattr(gift, 'age_suitability') and gift.age_suitability:
                    age_min, age_max = gift.age_suitability
                    if age_min <= user_profile.age <= age_max:
                        reasoning.append(
                            f"Age-appropriate for {user_profile.age} years old"
                        )
            except Exception as e:
                logger.warning(f"Error checking age appropriateness: {str(e)}")
            
            # 5. Occasion fit
            try:
                if hasattr(gift, 'occasion_fit') and gift.occasion_fit and user_profile.occasion in gift.occasion_fit:
                    reasoning.append(
                        f"Perfect for {user_profile.occasion}"
                    )
            except Exception as e:
                logger.warning(f"Error checking occasion fit: {str(e)}")
            
            # 6. Relationship appropriateness
            try:
                relationship_categories = self._get_relationship_appropriate_categories(
                    user_profile.relationship
                )
                if gift.category in relationship_categories:
                    reasoning.append(
                        f"Appropriate gift for {user_profile.relationship}"
                    )
            except Exception as e:
                logger.warning(f"Error checking relationship appropriateness: {str(e)}")
            
            # 7. Personality trait matching
            try:
                if user_profile.personality_traits:
                    matching_traits = self._find_matching_personality_traits(
                        gift, user_profile.personality_traits
                    )
                    if matching_traits:
                        reasoning.append(
                            f"Matches personality: {', '.join(matching_traits)}"
                        )
            except Exception as e:
                logger.warning(f"Error matching personality traits: {str(e)}")
            
            # Ensure we have at least some reasoning
            if not reasoning:
                reasoning.append(f"Recommended based on your profile")
            
        except Exception as e:
            logger.error(f"Error generating gift reasoning: {str(e)}", exc_info=True)
            reasoning = [f"Recommended based on your profile"]
        
        # Ensure we always return at least one reasoning item
        if not reasoning:
            reasoning = [f"Recommended based on your profile"]
        
        return reasoning
    
    def _find_matching_hobbies(
        self,
        gift: GiftItem,
        hobbies: List[str]
    ) -> List[str]:
        """
        Find hobbies that match the gift
        
        Args:
            gift: Gift item
            hobbies: User hobbies
            
        Returns:
            List of matching hobbies
        """
        matching = []
        
        # Check direct tag matches
        gift_tags_lower = [tag.lower() for tag in gift.tags]
        for hobby in hobbies:
            hobby_lower = hobby.lower()
            if hobby_lower in gift_tags_lower:
                matching.append(hobby)
                continue
            
            # Check category matches
            if hobby_lower in self.hobby_category_map:
                relevant_categories = self.hobby_category_map[hobby_lower]
                if gift.category in relevant_categories:
                    matching.append(hobby)
        
        return matching
    
    def _get_relationship_appropriate_categories(
        self,
        relationship: str
    ) -> List[str]:
        """
        Get categories appropriate for relationship
        
        Args:
            relationship: Relationship type
            
        Returns:
            List of appropriate categories
        """
        relationship_map = {
            "mother": ["Jewelry", "Flowers", "Home & Living", "Beauty", "Kitchen & Dining"],
            "father": ["Electronics", "Tools", "Sports", "Watches", "Books"],
            "partner": ["Jewelry", "Perfume", "Watches", "Electronics", "Clothing"],
            "friend": ["Books", "Electronics", "Gaming", "Entertainment", "Accessories"],
            "sibling": ["Gaming", "Electronics", "Clothing", "Books", "Sports"],
            "colleague": ["Books", "Stationery", "Coffee", "Accessories", "Plants"],
            "child": ["Toys", "Books", "Gaming", "Sports", "Educational"],
        }
        
        return relationship_map.get(relationship.lower(), [])
    
    def _find_matching_personality_traits(
        self,
        gift: GiftItem,
        personality_traits: List[str]
    ) -> List[str]:
        """
        Find personality traits that match the gift
        
        Args:
            gift: Gift item
            personality_traits: User personality traits
            
        Returns:
            List of matching traits
        """
        matching = []
        
        trait_keywords = {
            "practical": ["practical", "useful", "functional", "everyday"],
            "eco-friendly": ["eco", "sustainable", "organic", "natural", "green"],
            "creative": ["creative", "artistic", "craft", "design", "handmade"],
            "tech-savvy": ["tech", "smart", "digital", "electronic", "gadget"],
            "traditional": ["classic", "traditional", "vintage", "timeless"],
            "luxury": ["luxury", "premium", "high-end", "designer"],
            "minimalist": ["minimal", "simple", "clean", "modern"],
            "adventurous": ["adventure", "outdoor", "travel", "exploration"],
        }
        
        gift_text = f"{gift.name} {gift.description} {' '.join(gift.tags)}".lower()
        
        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_keywords:
                keywords = trait_keywords[trait_lower]
                if any(keyword in gift_text for keyword in keywords):
                    matching.append(trait)
        
        return matching
    
    def explain_confidence_score(
        self,
        confidence: float,
        gift: GiftItem,
        user_profile: UserProfile,
        model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain confidence score
        
        Args:
            confidence: Confidence score (0.0-1.0)
            gift: Gift item
            user_profile: User profile
            model_output: Model output dictionary
            
        Returns:
            Dict with score, level, and factors
        """
        try:
            # Validate inputs
            if not gift or not user_profile:
                logger.warning("Invalid inputs for confidence score explanation")
                return {
                    "score": confidence,
                    "level": "medium",
                    "factors": {
                        "positive": ["Matched with profile"],
                        "negative": []
                    }
                }
            
            # Determine confidence level
            if confidence > 0.8:
                level = "high"
            elif confidence > 0.5:
                level = "medium"
            else:
                level = "low"
            
            positive_factors = []
            negative_factors = []
        except Exception as e:
            logger.error(f"Error initializing confidence explanation: {str(e)}")
            return {
                "score": confidence,
                "level": "medium",
                "factors": {
                    "positive": ["Matched with profile"],
                    "negative": []
                }
            }
        
        try:
            # 1. Category matching analysis
            try:
                category_scores = model_output.get("category_scores")
                if category_scores is not None:
                    import torch
                    if isinstance(category_scores, torch.Tensor):
                        max_category_score = category_scores.max().item()
                        if max_category_score > 0.7:
                            positive_factors.append(
                                f"Strong category match ({max_category_score:.2f})"
                            )
                        elif max_category_score < 0.3:
                            negative_factors.append(
                                f"Weak category match ({max_category_score:.2f})"
                            )
            except Exception as e:
                logger.warning(f"Error analyzing category scores: {str(e)}")
            
            # 2. Budget analysis
            try:
                if user_profile.budget > 0:
                    budget_usage = (gift.price / user_profile.budget) * 100
                    if 50 <= budget_usage <= 90:
                        positive_factors.append("Price within optimal budget range")
                    elif budget_usage > 100:
                        negative_factors.append("Price exceeds budget")
                    elif budget_usage < 30:
                        negative_factors.append("Price significantly below budget (may indicate lower quality)")
            except Exception as e:
                logger.warning(f"Error analyzing budget: {str(e)}")
            
            # 3. Rating analysis
            try:
                if hasattr(gift, 'rating') and gift.rating:
                    if gift.rating >= 4.5:
                        positive_factors.append(f"Excellent reviews ({gift.rating}/5.0)")
                    elif gift.rating >= 4.0:
                        positive_factors.append(f"Good reviews ({gift.rating}/5.0)")
                    elif gift.rating < 3.0:
                        negative_factors.append(f"Lower ratings ({gift.rating}/5.0)")
            except Exception as e:
                logger.warning(f"Error analyzing rating: {str(e)}")
            
            # 4. Hobby matching
            try:
                matching_hobbies = self._find_matching_hobbies(gift, user_profile.hobbies)
                if len(matching_hobbies) >= 2:
                    positive_factors.append(f"Matches multiple hobbies ({len(matching_hobbies)})")
                elif len(matching_hobbies) == 1:
                    positive_factors.append(f"Matches hobby: {matching_hobbies[0]}")
                elif len(matching_hobbies) == 0:
                    negative_factors.append("No direct hobby match")
            except Exception as e:
                logger.warning(f"Error matching hobbies: {str(e)}")
            
            # 5. Occasion fit
            try:
                if hasattr(gift, 'occasion_fit') and gift.occasion_fit:
                    if user_profile.occasion in gift.occasion_fit:
                        positive_factors.append(f"Perfect for {user_profile.occasion}")
                    else:
                        negative_factors.append(f"Not specifically tagged for {user_profile.occasion}")
            except Exception as e:
                logger.warning(f"Error checking occasion fit: {str(e)}")
            
            # 6. Age appropriateness
            try:
                if hasattr(gift, 'age_suitability') and gift.age_suitability:
                    age_min, age_max = gift.age_suitability
                    if age_min <= user_profile.age <= age_max:
                        positive_factors.append("Age-appropriate")
                    else:
                        negative_factors.append(f"Age range mismatch (suitable for {age_min}-{age_max})")
            except Exception as e:
                logger.warning(f"Error checking age appropriateness: {str(e)}")
            
            # Ensure we have at least some factors based on confidence level
            if level == "high" and not positive_factors:
                positive_factors.append("Strong overall match with profile")
            elif level == "low" and not negative_factors:
                negative_factors.append("Limited match with profile")
            elif level == "medium" and not positive_factors and not negative_factors:
                positive_factors.append("Moderate match with profile")
        
        except Exception as e:
            logger.error(f"Error explaining confidence score: {str(e)}", exc_info=True)
            # Determine level from confidence if not set
            try:
                if confidence > 0.8:
                    level = "high"
                elif confidence > 0.5:
                    level = "medium"
                else:
                    level = "low"
            except:
                level = "medium"
            
            if level == "high":
                positive_factors = ["Strong overall match"]
            elif level == "low":
                negative_factors = ["Limited match with profile"]
            else:
                positive_factors = ["Moderate match"]
        
        # Ensure we have at least some factors
        if not positive_factors and not negative_factors:
            positive_factors = ["Matched with profile"]
        
        return {
            "score": confidence,
            "level": level,
            "factors": {
                "positive": positive_factors,
                "negative": negative_factors
            }
        }
    
    def generate_tool_selection_reasoning(
        self,
        tool_selection_trace: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Generate human-readable tool selection reasoning
        
        Args:
            tool_selection_trace: Raw tool selection data from model
            user_profile: User profile
            
        Returns:
            Structured reasoning for each tool
        """
        reasoning = {}
        
        try:
            # Validate inputs
            if not tool_selection_trace or not isinstance(tool_selection_trace, dict):
                logger.warning("Invalid tool selection trace")
                return reasoning
            
            if not user_profile:
                logger.warning("Invalid user profile for tool selection reasoning")
                return reasoning
            
            for tool_name, tool_data in tool_selection_trace.items():
                try:
                    # Extract data with defaults
                    selected = tool_data.get("selected", False)
                    score = tool_data.get("score", 0.0)
                    confidence = tool_data.get("confidence", score)
                    priority = tool_data.get("priority", 0)
                    factors = tool_data.get("factors", {})
                    
                    # Generate human-readable reason
                    reason = self._generate_tool_reason(
                        tool_name, selected, score, user_profile, factors
                    )
                    
                    reasoning[tool_name] = {
                        "name": tool_name,
                        "selected": selected,
                        "score": score,
                        "reason": reason,
                        "confidence": confidence,
                        "priority": priority,
                        "factors": factors
                    }
                except Exception as e:
                    logger.warning(f"Error processing tool {tool_name}: {str(e)}")
                    # Add minimal reasoning for this tool
                    reasoning[tool_name] = {
                        "name": tool_name,
                        "selected": False,
                        "score": 0.0,
                        "reason": "Tool reasoning unavailable",
                        "confidence": 0.0,
                        "priority": 0,
                        "factors": {}
                    }
        
        except Exception as e:
            logger.error(f"Error generating tool selection reasoning: {str(e)}", exc_info=True)
        
        return reasoning
    
    def _generate_tool_reason(
        self,
        tool_name: str,
        selected: bool,
        score: float,
        user_profile: UserProfile,
        factors: Dict[str, float]
    ) -> str:
        """
        Generate reason for tool selection/rejection
        
        Args:
            tool_name: Name of the tool
            selected: Whether tool was selected
            score: Tool selection score
            user_profile: User profile
            factors: Contributing factors
            
        Returns:
            Human-readable reason string
        """
        if tool_name == "price_comparison":
            if selected:
                if user_profile.budget < 500:
                    return f"Budget-conscious selection (budget: {user_profile.budget:.0f} TL)"
                else:
                    return "Comparing prices to find best value"
            else:
                return f"Low priority for current budget level (score: {score:.2f})"
        
        elif tool_name == "review_analysis":
            if selected:
                return "Analyzing customer reviews for quality insights"
            else:
                return f"Not selected for this recommendation (score: {score:.2f})"
        
        elif tool_name == "inventory_check":
            if selected:
                return "Checking stock availability"
            else:
                return f"Inventory check not prioritized (score: {score:.2f})"
        
        elif tool_name == "trend_analyzer":
            if selected:
                if any(trait in ["modern", "trendy", "tech-savvy"] for trait in user_profile.personality_traits):
                    return "User prefers trendy items - analyzing current trends"
                else:
                    return "Analyzing trending items in category"
            else:
                return f"Trend analysis not needed (score: {score:.2f})"
        
        elif tool_name == "budget_optimizer":
            if selected:
                return f"Optimizing recommendations for {user_profile.budget:.0f} TL budget"
            else:
                return f"Budget optimization not required (score: {score:.2f})"
        
        else:
            # Generic reason
            if selected:
                return f"Selected based on user profile (score: {score:.2f})"
            else:
                return f"Not selected (score: {score:.2f})"
    
    def generate_category_reasoning(
        self,
        category_trace: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Generate human-readable category matching reasoning
        
        Args:
            category_trace: Raw category matching data from model
            user_profile: User profile
            
        Returns:
            Structured reasoning for each category
        """
        reasoning = {}
        
        try:
            # Validate inputs
            if not category_trace or not isinstance(category_trace, dict):
                logger.warning("Invalid category trace")
                return reasoning
            
            if not user_profile:
                logger.warning("Invalid user profile for category reasoning")
                return reasoning
            
            for category_name, category_data in category_trace.items():
                try:
                    score = category_data.get("score", 0.0)
                    reasons = category_data.get("reasons", [])
                    feature_contributions = category_data.get("feature_contributions", {})
                    
                    # Generate additional context-aware reasons if needed
                    if not reasons:
                        reasons = self._generate_category_reasons(
                            category_name, score, user_profile, feature_contributions
                        )
                    
                    reasoning[category_name] = {
                        "category_name": category_name,
                        "score": score,
                        "reasons": reasons,
                        "feature_contributions": feature_contributions
                    }
                except Exception as e:
                    logger.warning(f"Error processing category {category_name}: {str(e)}")
                    # Add minimal reasoning for this category
                    reasoning[category_name] = {
                        "category_name": category_name,
                        "score": 0.0,
                        "reasons": ["Category reasoning unavailable"],
                        "feature_contributions": {}
                    }
        
        except Exception as e:
            logger.error(f"Error generating category reasoning: {str(e)}", exc_info=True)
        
        return reasoning
    
    def _generate_category_reasons(
        self,
        category_name: str,
        score: float,
        user_profile: UserProfile,
        feature_contributions: Dict[str, float]
    ) -> List[str]:
        """
        Generate reasons for category matching
        
        Args:
            category_name: Category name
            score: Category score
            user_profile: User profile
            feature_contributions: Feature contribution scores
            
        Returns:
            List of reason strings
        """
        reasons = []
        
        # Check hobby matches
        for hobby in user_profile.hobbies:
            if hobby.lower() in self.hobby_category_map:
                relevant_categories = self.hobby_category_map[hobby.lower()]
                if category_name in relevant_categories:
                    reasons.append(f"Matches hobby: {hobby}")
        
        # Check occasion fit
        if user_profile.occasion.lower() in self.occasion_category_map:
            relevant_categories = self.occasion_category_map[user_profile.occasion.lower()]
            if category_name in relevant_categories:
                reasons.append(f"Suitable for {user_profile.occasion}")
        
        # Check relationship appropriateness
        relationship_categories = self._get_relationship_appropriate_categories(
            user_profile.relationship
        )
        if category_name in relationship_categories:
            reasons.append(f"Appropriate for {user_profile.relationship}")
        
        # Add score-based reason if no specific reasons found
        if not reasons:
            if score > 0.7:
                reasons.append(f"Strong match with profile (score: {score:.2f})")
            elif score < 0.3:
                reasons.append(f"Weak match with profile (score: {score:.2f})")
            else:
                reasons.append(f"Moderate match with profile (score: {score:.2f})")
        
        return reasons


# Singleton instance
_reasoning_service: Optional[ReasoningService] = None


def get_reasoning_service() -> ReasoningService:
    """Get or create reasoning service singleton"""
    global _reasoning_service
    
    if _reasoning_service is None:
        _reasoning_service = ReasoningService()
    
    return _reasoning_service
