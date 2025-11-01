"""
Reward calculation utilities for RL training
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional


class RewardCalculator:
    """Calculate rewards for gift recommendation RL training"""
    
    def __init__(self):
        self.reward_history = []
    
    def calculate_reward(self, recommendations: List[Dict], 
                        user_profile: Dict, user_feedback: Dict) -> float:
        """Calculate reward based on recommendations and user feedback"""
        if not recommendations:
            return -0.5  # Penalty for no recommendations
        
        total_reward = 0.0
        
        for rec in recommendations:
            # Budget compatibility
            budget_score = self._budget_compatibility(
                rec.get('price', 0), user_profile.get('budget', 100)
            )
            
            # Hobby alignment
            hobby_score = self._hobby_alignment(
                rec.get('category', ''), rec.get('tags', []), 
                user_profile.get('hobbies', [])
            )
            
            # User satisfaction (from feedback)
            satisfaction_score = user_feedback.get('satisfaction', 0.5)
            
            # Combine scores
            rec_reward = (
                0.3 * budget_score + 
                0.3 * hobby_score + 
                0.4 * satisfaction_score
            )
            
            total_reward += rec_reward
        
        # Average reward
        avg_reward = total_reward / len(recommendations)
        
        # Store in history
        self.reward_history.append(avg_reward)
        
        return avg_reward
    
    def _budget_compatibility(self, price: float, budget: float) -> float:
        """Calculate budget compatibility score"""
        if price <= budget * 0.8:
            return 1.0
        elif price <= budget:
            return 0.8
        elif price <= budget * 1.2:
            return 0.4
        else:
            return 0.0
    
    def _hobby_alignment(self, category: str, tags: List[str], 
                        hobbies: List[str]) -> float:
        """Calculate hobby alignment score"""
        if not hobbies:
            return 0.5
        
        matches = 0
        for hobby in hobbies:
            if (hobby.lower() in category.lower() or 
                any(hobby.lower() in tag.lower() for tag in tags)):
                matches += 1
        
        return min(1.0, matches / len(hobbies))
    
    def get_average_reward(self, window: int = 100) -> float:
        """Get average reward over recent window"""
        if not self.reward_history:
            return 0.0
        
        recent_rewards = self.reward_history[-window:]
        return np.mean(recent_rewards)
    
    def reset_history(self):
        """Reset reward history"""
        self.reward_history.clear()