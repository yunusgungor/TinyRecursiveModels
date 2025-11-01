"""
Reinforcement Learning components for TRM
"""

from .environment import GiftRecommendationEnvironment
from .rl_trm import RLEnhancedTRM
# from .rewards import RewardCalculator  # Optional component
from .trainer import RLTrainer

__all__ = [
    'GiftRecommendationEnvironment',
    'RLEnhancedTRM', 
    'RLTrainer'
]