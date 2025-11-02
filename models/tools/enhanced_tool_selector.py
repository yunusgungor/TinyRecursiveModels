"""
Enhanced Tool Selection Strategy for Better Tool Diversity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import random

from models.rl.environment import UserProfile, EnvironmentState


class ContextAwareToolSelector:
    """Context-aware tool selector that chooses appropriate tools based on user profile and scenario"""
    
    def __init__(self):
        self.tool_usage_history = []
        self.tool_effectiveness_scores = {
            "price_comparison": 0.7,
            "inventory_check": 0.6,
            "review_analysis": 0.8,
            "trend_analysis": 0.5,  # Reduced from overuse
            "budget_optimizer": 0.9
        }
        
        # Context-based tool selection rules
        self.context_rules = self._create_context_rules()
        
    def _create_context_rules(self) -> Dict[str, Dict[str, float]]:
        """Create context-based tool selection rules"""
        return {
            # Budget-based rules
            "budget_low": {  # Budget < 75
                "budget_optimizer": 0.9,
                "price_comparison": 0.8,
                "trend_analysis": 0.3,
                "review_analysis": 0.6,
                "inventory_check": 0.4
            },
            "budget_medium": {  # Budget 75-150
                "budget_optimizer": 0.6,
                "price_comparison": 0.7,
                "trend_analysis": 0.5,
                "review_analysis": 0.8,
                "inventory_check": 0.6
            },
            "budget_high": {  # Budget > 150
                "budget_optimizer": 0.3,
                "price_comparison": 0.5,
                "trend_analysis": 0.7,
                "review_analysis": 0.9,
                "inventory_check": 0.8
            },
            
            # Age-based rules
            "age_young": {  # Age < 30
                "trend_analysis": 0.9,
                "price_comparison": 0.8,
                "review_analysis": 0.7,
                "budget_optimizer": 0.6,
                "inventory_check": 0.5
            },
            "age_middle": {  # Age 30-50
                "review_analysis": 0.9,
                "price_comparison": 0.7,
                "inventory_check": 0.8,
                "budget_optimizer": 0.6,
                "trend_analysis": 0.4
            },
            "age_senior": {  # Age > 50
                "review_analysis": 0.9,
                "inventory_check": 0.8,
                "price_comparison": 0.6,
                "budget_optimizer": 0.5,
                "trend_analysis": 0.3
            },
            
            # Occasion-based rules
            "birthday": {
                "trend_analysis": 0.7,
                "review_analysis": 0.8,
                "price_comparison": 0.6,
                "budget_optimizer": 0.5,
                "inventory_check": 0.6
            },
            "christmas": {
                "price_comparison": 0.9,
                "inventory_check": 0.8,
                "budget_optimizer": 0.7,
                "review_analysis": 0.6,
                "trend_analysis": 0.5
            },
            "mothers_day": {
                "review_analysis": 0.9,
                "trend_analysis": 0.6,
                "price_comparison": 0.5,
                "budget_optimizer": 0.6,
                "inventory_check": 0.7
            },
            "graduation": {
                "trend_analysis": 0.8,
                "price_comparison": 0.7,
                "budget_optimizer": 0.8,
                "review_analysis": 0.6,
                "inventory_check": 0.5
            },
            
            # Hobby-based rules
            "technology": {
                "review_analysis": 0.9,
                "price_comparison": 0.8,
                "trend_analysis": 0.7,
                "inventory_check": 0.6,
                "budget_optimizer": 0.5
            },
            "gardening": {
                "review_analysis": 0.8,
                "inventory_check": 0.7,
                "trend_analysis": 0.4,
                "price_comparison": 0.6,
                "budget_optimizer": 0.6
            },
            "cooking": {
                "review_analysis": 0.9,
                "price_comparison": 0.7,
                "inventory_check": 0.6,
                "trend_analysis": 0.5,
                "budget_optimizer": 0.5
            },
            "fitness": {
                "trend_analysis": 0.8,
                "review_analysis": 0.7,
                "price_comparison": 0.6,
                "budget_optimizer": 0.6,
                "inventory_check": 0.5
            },
            "reading": {
                "review_analysis": 0.9,
                "price_comparison": 0.6,
                "inventory_check": 0.7,
                "trend_analysis": 0.4,
                "budget_optimizer": 0.5
            },
            
            # Personality-based rules
            "practical": {
                "price_comparison": 0.9,
                "review_analysis": 0.8,
                "budget_optimizer": 0.7,
                "inventory_check": 0.6,
                "trend_analysis": 0.3
            },
            "trendy": {
                "trend_analysis": 0.9,
                "review_analysis": 0.6,
                "price_comparison": 0.5,
                "inventory_check": 0.5,
                "budget_optimizer": 0.4
            },
            "luxury": {
                "review_analysis": 0.9,
                "inventory_check": 0.8,
                "price_comparison": 0.4,
                "trend_analysis": 0.6,
                "budget_optimizer": 0.3
            },
            "eco-friendly": {
                "review_analysis": 0.8,
                "inventory_check": 0.7,
                "price_comparison": 0.6,
                "trend_analysis": 0.5,
                "budget_optimizer": 0.6
            }
        }
    
    def select_tools(self, user_profile: UserProfile, 
                    max_tools: int = 2,
                    diversity_weight: float = 0.3) -> List[Tuple[str, float]]:
        """
        Select appropriate tools based on user context
        
        Args:
            user_profile: User profile
            max_tools: Maximum number of tools to select
            diversity_weight: Weight for diversity in tool selection
            
        Returns:
            List of (tool_name, confidence) tuples
        """
        # Calculate base scores for each tool
        tool_scores = self._calculate_base_tool_scores(user_profile)
        
        # Apply context-based adjustments
        context_adjusted_scores = self._apply_context_adjustments(tool_scores, user_profile)
        
        # Apply diversity penalty for overused tools
        diversity_adjusted_scores = self._apply_diversity_penalty(context_adjusted_scores)
        
        # Select tools using weighted sampling
        selected_tools = self._select_diverse_tools(
            diversity_adjusted_scores, max_tools, diversity_weight
        )
        
        # Update usage history
        for tool_name, _ in selected_tools:
            self.tool_usage_history.append(tool_name)
        
        return selected_tools
    
    def _calculate_base_tool_scores(self, user_profile: UserProfile) -> Dict[str, float]:
        """Calculate base scores for each tool"""
        scores = {}
        
        for tool_name, base_effectiveness in self.tool_effectiveness_scores.items():
            # Start with base effectiveness
            score = base_effectiveness
            
            # Add small random variation
            score += random.uniform(-0.1, 0.1)
            
            # Ensure score is in valid range
            scores[tool_name] = max(0.0, min(1.0, score))
        
        return scores
    
    def _apply_context_adjustments(self, base_scores: Dict[str, float], 
                                  user_profile: UserProfile) -> Dict[str, float]:
        """Apply context-based score adjustments"""
        adjusted_scores = base_scores.copy()
        
        # Budget context
        budget_context = self._get_budget_context(user_profile.budget)
        if budget_context in self.context_rules:
            for tool, weight in self.context_rules[budget_context].items():
                if tool in adjusted_scores:
                    adjusted_scores[tool] *= weight
        
        # Age context
        age_context = self._get_age_context(user_profile.age)
        if age_context in self.context_rules:
            for tool, weight in self.context_rules[age_context].items():
                if tool in adjusted_scores:
                    adjusted_scores[tool] *= weight
        
        # Occasion context
        if user_profile.occasion in self.context_rules:
            for tool, weight in self.context_rules[user_profile.occasion].items():
                if tool in adjusted_scores:
                    adjusted_scores[tool] *= weight
        
        # Hobby context
        for hobby in user_profile.hobbies:
            if hobby in self.context_rules:
                for tool, weight in self.context_rules[hobby].items():
                    if tool in adjusted_scores:
                        adjusted_scores[tool] *= weight
        
        # Personality context
        for trait in user_profile.personality_traits:
            if trait in self.context_rules:
                for tool, weight in self.context_rules[trait].items():
                    if tool in adjusted_scores:
                        adjusted_scores[tool] *= weight
        
        # Normalize scores
        max_score = max(adjusted_scores.values()) if adjusted_scores.values() else 1.0
        if max_score > 0:
            adjusted_scores = {tool: score / max_score for tool, score in adjusted_scores.items()}
        
        return adjusted_scores
    
    def _get_budget_context(self, budget: float) -> str:
        """Get budget context category"""
        if budget < 75:
            return "budget_low"
        elif budget < 150:
            return "budget_medium"
        else:
            return "budget_high"
    
    def _get_age_context(self, age: int) -> str:
        """Get age context category"""
        if age < 30:
            return "age_young"
        elif age < 50:
            return "age_middle"
        else:
            return "age_senior"
    
    def _apply_diversity_penalty(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply penalty for overused tools"""
        penalized_scores = scores.copy()
        
        # Count recent tool usage (last 10 calls)
        recent_usage = self.tool_usage_history[-10:] if len(self.tool_usage_history) >= 10 else self.tool_usage_history
        usage_counts = {}
        for tool in recent_usage:
            usage_counts[tool] = usage_counts.get(tool, 0) + 1
        
        # Apply penalties
        for tool, count in usage_counts.items():
            if tool in penalized_scores:
                # Penalty increases with usage frequency
                penalty = min(0.5, count * 0.1)  # Max 50% penalty
                penalized_scores[tool] *= (1.0 - penalty)
        
        # Special penalty for trend_analysis (overused tool)
        if "trend_analysis" in penalized_scores:
            penalized_scores["trend_analysis"] *= 0.6  # 40% penalty
        
        return penalized_scores
    
    def _select_diverse_tools(self, scores: Dict[str, float], 
                             max_tools: int, diversity_weight: float) -> List[Tuple[str, float]]:
        """Select diverse set of tools"""
        if not scores:
            return []
        
        selected = []
        remaining_tools = list(scores.keys())
        
        # Always select the highest scoring tool first
        best_tool = max(remaining_tools, key=lambda x: scores[x])
        selected.append((best_tool, scores[best_tool]))
        remaining_tools.remove(best_tool)
        
        # Select remaining tools with diversity consideration
        while len(selected) < max_tools and remaining_tools:
            best_candidate = None
            best_score = -1
            
            for tool in remaining_tools:
                # Base score
                base_score = scores[tool]
                
                # Diversity bonus (prefer tools from different categories)
                diversity_bonus = self._calculate_diversity_bonus(tool, [t for t, _ in selected])
                
                # Combined score
                combined_score = (1 - diversity_weight) * base_score + diversity_weight * diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = tool
            
            if best_candidate:
                selected.append((best_candidate, scores[best_candidate]))
                remaining_tools.remove(best_candidate)
        
        return selected
    
    def _calculate_diversity_bonus(self, candidate_tool: str, selected_tools: List[str]) -> float:
        """Calculate diversity bonus for a candidate tool"""
        if not selected_tools:
            return 1.0
        
        # Tool categories for diversity calculation
        tool_categories = {
            "price_comparison": "comparison",
            "inventory_check": "availability",
            "review_analysis": "quality",
            "trend_analysis": "market",
            "budget_optimizer": "financial"
        }
        
        candidate_category = tool_categories.get(candidate_tool, "other")
        selected_categories = set(tool_categories.get(tool, "other") for tool in selected_tools)
        
        # Bonus for different category
        if candidate_category not in selected_categories:
            return 1.0
        else:
            return 0.3  # Penalty for same category
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        if not self.tool_usage_history:
            return {"total_calls": 0, "tool_distribution": {}}
        
        total_calls = len(self.tool_usage_history)
        tool_counts = {}
        
        for tool in self.tool_usage_history:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        tool_distribution = {
            tool: {
                "count": count,
                "percentage": round((count / total_calls) * 100, 1)
            }
            for tool, count in tool_counts.items()
        }
        
        return {
            "total_calls": total_calls,
            "tool_distribution": tool_distribution,
            "most_used": max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None,
            "least_used": min(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None
        }
    
    def explain_tool_selection(self, user_profile: UserProfile, 
                              selected_tools: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Explain why specific tools were selected"""
        explanations = {}
        
        for tool_name, confidence in selected_tools:
            reasons = []
            
            # Budget-based reasons
            budget_context = self._get_budget_context(user_profile.budget)
            if budget_context == "budget_low" and tool_name in ["budget_optimizer", "price_comparison"]:
                reasons.append("Budget-conscious selection")
            elif budget_context == "budget_high" and tool_name in ["review_analysis", "inventory_check"]:
                reasons.append("Quality-focused for higher budget")
            
            # Age-based reasons
            age_context = self._get_age_context(user_profile.age)
            if age_context == "age_young" and tool_name == "trend_analysis":
                reasons.append("Trend-aware for younger demographic")
            elif age_context == "age_senior" and tool_name == "review_analysis":
                reasons.append("Quality-focused for mature buyers")
            
            # Hobby-based reasons
            for hobby in user_profile.hobbies:
                if hobby == "technology" and tool_name in ["review_analysis", "price_comparison"]:
                    reasons.append(f"Tech-savvy approach for {hobby} interest")
                elif hobby == "gardening" and tool_name == "review_analysis":
                    reasons.append(f"Quality assessment for {hobby} products")
            
            # Personality-based reasons
            for trait in user_profile.personality_traits:
                if trait == "practical" and tool_name in ["price_comparison", "budget_optimizer"]:
                    reasons.append(f"Practical approach matching {trait} preference")
                elif trait == "trendy" and tool_name == "trend_analysis":
                    reasons.append(f"Trend-focused for {trait} personality")
            
            # Diversity reasons
            if len(selected_tools) > 1:
                reasons.append("Selected for tool diversity")
            
            explanations[tool_name] = {
                "confidence": confidence,
                "reasons": reasons if reasons else ["General recommendation tool"]
            }
        
        return explanations


if __name__ == "__main__":
    # Test the enhanced tool selector
    from models.rl.environment import UserProfile
    
    selector = ContextAwareToolSelector()
    
    # Test different user profiles
    test_users = [
        UserProfile(
            age=25,
            hobbies=["technology", "gaming"],
            relationship="friend",
            budget=80.0,
            occasion="birthday",
            personality_traits=["trendy", "tech-savvy"]
        ),
        UserProfile(
            age=45,
            hobbies=["gardening", "cooking"],
            relationship="mother",
            budget=120.0,
            occasion="mothers_day",
            personality_traits=["practical", "eco-friendly"]
        ),
        UserProfile(
            age=65,
            hobbies=["reading", "gardening"],
            relationship="father",
            budget=200.0,
            occasion="fathers_day",
            personality_traits=["traditional", "quality"]
        )
    ]
    
    for i, user in enumerate(test_users, 1):
        print(f"\n👤 Test User {i}:")
        print(f"   Age: {user.age}, Budget: ${user.budget}")
        print(f"   Hobbies: {user.hobbies}")
        print(f"   Traits: {user.personality_traits}")
        
        # Select tools
        selected_tools = selector.select_tools(user, max_tools=2)
        
        print(f"   Selected Tools:")
        for tool, confidence in selected_tools:
            print(f"     • {tool}: {confidence:.3f}")
        
        # Get explanations
        explanations = selector.explain_tool_selection(user, selected_tools)
        print(f"   Explanations:")
        for tool, exp in explanations.items():
            print(f"     • {tool}: {', '.join(exp['reasons'])}")
    
    # Get usage statistics
    stats = selector.get_tool_usage_stats()
    print(f"\n📊 Tool Usage Statistics:")
    print(f"   Total Calls: {stats['total_calls']}")
    if stats['tool_distribution']:
        print(f"   Distribution:")
        for tool, data in stats['tool_distribution'].items():
            print(f"     • {tool}: {data['count']} calls ({data['percentage']}%)")