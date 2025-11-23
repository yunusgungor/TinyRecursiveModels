#!/usr/bin/env python3
"""
Training script for Integrated Enhanced TRM Model
All improvements built into the model architecture for end-to-end training
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import GiftRecommendationEnvironment, UserProfile, GiftItem
from models.rl.trainer import RLTrainer, TrainingConfig


class ToolResultEncoder(nn.Module):
    """Encode tool execution results into tensor format for model feedback"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoders for different tool result types
        self.price_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [num_in_budget, num_over_budget, avg_price]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.review_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [avg_rating, num_items]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.inventory_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [num_available, num_unavailable]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.trend_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [num_trending, avg_popularity]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.budget_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [num_optimized, total_value]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer to combine multiple tool results
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def encode_tool_result(self, tool_name: str, result: Dict, device: torch.device) -> torch.Tensor:
        """Encode a single tool result"""
        if tool_name == 'price_comparison':
            num_in_budget = len(result.get('in_budget', []))
            num_over_budget = len(result.get('over_budget', []))
            avg_price = result.get('average_price', 0.0)
            features = torch.tensor([num_in_budget, num_over_budget, avg_price], dtype=torch.float32, device=device)
            return self.price_encoder(features)
        
        elif tool_name == 'review_analysis':
            avg_rating = result.get('average_rating', 0.0)
            num_items = len(result.get('top_rated', []))
            features = torch.tensor([avg_rating, num_items], dtype=torch.float32, device=device)
            return self.review_encoder(features)
        
        elif tool_name == 'inventory_check':
            num_available = len(result.get('available', []))
            num_unavailable = len(result.get('unavailable', []))
            features = torch.tensor([num_available, num_unavailable], dtype=torch.float32, device=device)
            return self.inventory_encoder(features)
        
        elif tool_name == 'trend_analysis':
            num_trending = len(result.get('trending', []))
            avg_popularity = result.get('average_popularity', 0.0)
            features = torch.tensor([num_trending, avg_popularity], dtype=torch.float32, device=device)
            return self.trend_encoder(features)
        
        elif tool_name == 'budget_optimizer':
            num_optimized = len(result.get('optimized_gifts', []))
            total_value = result.get('total_value', 0.0)
            features = torch.tensor([num_optimized, total_value], dtype=torch.float32, device=device)
            return self.budget_encoder(features)
        
        else:
            return torch.zeros(self.hidden_dim, device=device)
    
    def forward(self, tool_results: Dict[str, Dict], device: torch.device) -> torch.Tensor:
        """Encode all tool results and fuse them"""
        # Initialize with zeros on correct device
        encoded_results = {
            'price_comparison': torch.zeros(self.hidden_dim, device=device),
            'review_analysis': torch.zeros(self.hidden_dim, device=device),
            'inventory_check': torch.zeros(self.hidden_dim, device=device),
            'trend_analysis': torch.zeros(self.hidden_dim, device=device),
            'budget_optimizer': torch.zeros(self.hidden_dim, device=device)
        }
        
        # Encode available results
        for tool_name, result in tool_results.items():
            if result:
                encoded_results[tool_name] = self.encode_tool_result(tool_name, result, device)
        
        # Concatenate and fuse
        concatenated = torch.cat([
            encoded_results['price_comparison'],
            encoded_results['review_analysis'],
            encoded_results['inventory_check'],
            encoded_results['trend_analysis'],
            encoded_results['budget_optimizer']
        ])
        
        return self.fusion(concatenated)


class IntegratedEnhancedTrainer:
    """Trainer for the integrated enhanced model with all improvements"""
    
    def __init__(self, config: dict, use_synthetic_data: bool = False, synthetic_ratio: float = 0.5):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_ratio = synthetic_ratio  # Ratio of synthetic data in batches (0.0-1.0)
        
        # Initialize model
        self.model = IntegratedEnhancedTRM(config).to(self.device)
        
        # Initialize tool result encoder
        self.tool_result_encoder = ToolResultEncoder(hidden_dim=config.get('hidden_dim', 128)).to(self.device)
        
        # Initialize environment with appropriate gift catalog
        gift_catalog_path = "data/fully_learned_synthetic_gifts.json" if use_synthetic_data else "data/gift_catalog.json"
        self.env = GiftRecommendationEnvironment(gift_catalog_path)
        
        # Curriculum learning settings - SLOWER PROGRESSION (v11 ORIGINAL)
        self.curriculum_stage = 0
        self.available_tools_by_stage = {
            0: ['price_comparison'],  # Stage 0: Only price comparison
            1: ['price_comparison', 'review_analysis'],  # Stage 1: Add reviews
            2: ['price_comparison', 'review_analysis', 'inventory_check'],  # Stage 2: Add inventory
            3: ['price_comparison', 'review_analysis', 'inventory_check', 'trend_analysis'],  # Stage 3: Add trend
            4: ['price_comparison', 'review_analysis', 'inventory_check', 'trend_analysis', 'budget_optimizer']  # Stage 4: All tools
        }
        # Track when each stage started for adaptive progression
        self.stage_start_epochs = {0: 0}
        self.min_epochs_per_stage = 20  # RESTORED: v11 original value
        
        # Tool exploration settings - MUCH MORE AGGRESSIVE
        self.force_exploration_prob = 0.5  # Increased from 30% to 50%
        self.exploration_bonus_threshold = 5000  # Force exploration if tool count < this
        self.tool_selection_counts = {
            'price_comparison': 0,
            'review_analysis': 0,
            'inventory_check': 0,
            'trend_analysis': 0,
            'budget_optimizer': 0
        }
        self.min_tool_usage_per_stage = 10000  # Minimum usage before advancing stage
        
        # Load and split scenarios for train/val
        self._load_and_split_scenarios()
        
        # Initialize optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.category_criterion = nn.CrossEntropyLoss()
        self.tool_criterion = nn.BCEWithLogitsLoss()
        
        # Training metrics
        self.training_metrics = {
            'total_loss': [],
            'category_loss': [],
            'tool_loss': [],
            'reward_loss': [],
            'category_accuracy': [],
            'tool_diversity': [],
            'recommendation_quality': []
        }
        
        # Early stopping - BALANCED patience
        self.best_eval_score = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 8)  # 8 evaluations = 40 epochs (eval every 5)
        
        # Scenario storage
        self.train_scenarios = []
        self.val_scenarios = []
        self.synthetic_train_scenarios = []
        self.synthetic_val_scenarios = []
        
        # Load scenarios BEFORE printing
        self._load_and_split_scenarios()
        
        # Load synthetic data if enabled
        if use_synthetic_data:
            self._load_synthetic_data()
        
        print(f"🚀 Integrated Enhanced Trainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 Training scenarios: {len(self.train_scenarios)}")
        print(f"📊 Validation scenarios: {len(self.val_scenarios)}")
        if use_synthetic_data:
            print(f"🤖 Synthetic training scenarios: {len(self.synthetic_train_scenarios)}")
            print(f"🤖 Synthetic validation scenarios: {len(self.synthetic_val_scenarios)}")
            print(f"📈 Synthetic data ratio: {synthetic_ratio:.1%}")
        
    def _load_and_split_scenarios(self):
        """Load scenarios and split into train/validation sets"""
        try:
            with open("data/user_scenarios.json", "r") as f:
                scenario_data = json.load(f)
            all_scenarios = scenario_data["scenarios"]
        except:
            try:
                with open("data/user_scenarios.json", "r") as f:
                    scenario_data = json.load(f)
                all_scenarios = scenario_data["scenarios"]
            except:
                all_scenarios = self._generate_fallback_scenarios()
        
        # Shuffle and split 80/20
        np.random.shuffle(all_scenarios)
        split_idx = int(len(all_scenarios) * 0.8)
        self.train_scenarios = all_scenarios[:split_idx]
        self.val_scenarios = all_scenarios[split_idx:]
        
        print(f"📊 Loaded {len(all_scenarios)} scenarios: {len(self.train_scenarios)} train, {len(self.val_scenarios)} val")
    
    def _load_synthetic_data(self):
        """Load synthetic scenarios and split into train/validation sets"""
        try:
            with open("data/fully_learned_synthetic_users.json", "r") as f:
                synthetic_data = json.load(f)
            all_synthetic_scenarios = synthetic_data["scenarios"]
            
            # Shuffle and split 80/20
            np.random.shuffle(all_synthetic_scenarios)
            split_idx = int(len(all_synthetic_scenarios) * 0.8)
            self.synthetic_train_scenarios = all_synthetic_scenarios[:split_idx]
            self.synthetic_val_scenarios = all_synthetic_scenarios[split_idx:]
            
            print(f"🤖 Sentetik veri yüklendi: {len(all_synthetic_scenarios)} senaryo")
            print(f"   Eğitim: {len(self.synthetic_train_scenarios)}, Doğrulama: {len(self.synthetic_val_scenarios)}")
        except Exception as e:
            print(f"⚠️ Sentetik veri yüklenemedi: {e}")
            self.synthetic_train_scenarios = []
            self.synthetic_val_scenarios = []
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components"""
        # Group parameters by component
        param_groups = [
            {
                'params': list(self.model.user_profile_encoder.parameters()) + 
                         list(self.model.hobby_embeddings.parameters()) +
                         list(self.model.preference_embeddings.parameters()) +
                         list(self.model.occasion_embeddings.parameters()),
                'lr': self.config.get('user_profile_lr', 1e-4),
                'name': 'user_profiling'
            },
            {
                'params': list(self.model.category_embeddings.parameters()) +
                         [p for layer in self.model.semantic_matcher for p in layer.parameters()] +
                         list(self.model.category_attention.parameters()) +
                         list(self.model.category_scorer.parameters()),
                'lr': self.config.get('category_matching_lr', 2e-4),  # INCREASED from 4e-5 to 2e-4 for better category learning
                'name': 'category_matching'
            },
            {
                'params': list(self.model.context_aware_tool_selector.parameters()) +
                         list(self.model.tool_diversity_head.parameters()) +
                         list(self.model.enhanced_tool_param_generator.parameters()),
                'lr': self.config.get('tool_selection_lr', 1.5e-4),
                'name': 'tool_selection'
            },
            {
                'params': [p for component in self.model.reward_components.values() for p in component.parameters()] +
                         list(self.model.reward_fusion.parameters()),
                'lr': self.config.get('reward_prediction_lr', 1e-4),
                'name': 'reward_prediction'
            },
            {
                'params': list(self.model.cross_modal_layers.parameters()) +
                         list(self.model.recommendation_head.parameters()),
                'lr': self.config.get('main_lr', 1e-4),
                'name': 'main_architecture'
            }
        ]
        
        # Add tool result encoder parameters
        param_groups.append({
            'params': self.tool_result_encoder.parameters(),
            'lr': self.config.get('tool_encoder_lr', 1e-4),
            'name': 'tool_result_encoder'
        })
        
        optimizer = optim.AdamW(param_groups, weight_decay=self.config.get('weight_decay', 0.02))  # Increased for better regularization
        
        # Add learning rate scheduler with AGGRESSIVE reduction when stuck
        # Use mode='max' since we're tracking improvement (higher is better)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True  # Faster adaptation
        )
        
        return optimizer
    
    def generate_training_batch(self, batch_size: int = 16, use_validation: bool = False) -> Tuple[List[UserProfile], List[List[GiftItem]], List[Dict]]:
        """Generate a training batch with diverse user profiles and scenarios with augmentation"""
        
        # Determine how many samples from synthetic vs real data
        if self.use_synthetic_data and (self.synthetic_train_scenarios or self.synthetic_val_scenarios):
            num_synthetic = int(batch_size * self.synthetic_ratio)
            num_real = batch_size - num_synthetic
        else:
            num_synthetic = 0
            num_real = batch_size
        
        # Use appropriate scenario sets
        real_scenarios = self.val_scenarios if use_validation else self.train_scenarios
        synthetic_scenarios = self.synthetic_val_scenarios if use_validation else self.synthetic_train_scenarios
        
        if not real_scenarios:
            real_scenarios = self._generate_fallback_scenarios()
        
        batch_users = []
        batch_gifts = []
        batch_targets = []
        
        # Generate real data samples
        for _ in range(num_real):
            # Sample a scenario
            scenario = np.random.choice(real_scenarios)
            
            # MINIMAL data augmentation - preserve category signals
            augmented_age = scenario["profile"]["age"] + np.random.randint(-2, 3)  # Minimal age variation
            augmented_age = max(18, min(75, augmented_age))  # Clamp to valid range
            
            augmented_budget = scenario["profile"]["budget"] * np.random.uniform(0.9, 1.1)  # Minimal budget variation
            augmented_budget = max(30.0, min(300.0, augmented_budget))  # Clamp to valid range
            
            # MINIMAL hobby manipulation - preserve category signals
            hobbies = scenario["profile"]["hobbies"].copy()
            if len(hobbies) > 3 and np.random.random() < 0.1:  # Very low probability
                num_to_drop = 1  # Only drop one hobby at a time
                hobbies = hobbies[:max(2, len(hobbies) - num_to_drop)]  # Keep at least 2
            # Don't shuffle - preserve order which may contain semantic info
            
            # MINIMAL preference manipulation
            preferences = scenario["profile"]["preferences"].copy()
            if np.random.random() < 0.1 and len(preferences) > 3:  # Very low probability
                num_to_drop = 1  # Only drop one preference at a time
                preferences = preferences[:max(2, len(preferences) - num_to_drop)]  # Keep at least 2
            # Don't shuffle - preserve order
            
            # Create user profile with augmentation
            user = UserProfile(
                age=int(augmented_age),
                hobbies=hobbies,
                relationship=scenario["profile"]["relationship"],
                budget=float(augmented_budget),
                occasion=scenario["profile"]["occasion"],
                personality_traits=preferences
            )
            
            # Get available gifts
            available_gifts = self.env.gift_catalog
            
            # Create target based on expected categories and tools
            target = {
                'expected_categories': scenario["expected_categories"],
                'expected_tools': scenario["expected_tools"],
                'user_profile': user
            }
            
            batch_users.append(user)
            batch_gifts.append(available_gifts)
            batch_targets.append(target)
        
        # Generate synthetic data samples
        for _ in range(num_synthetic):
            if not synthetic_scenarios:
                continue
                
            # Sample a synthetic scenario
            scenario = np.random.choice(synthetic_scenarios)
            
            # MINIMAL augmentation for synthetic data too
            augmented_age = scenario["profile"]["age"] + np.random.randint(-2, 3)
            augmented_age = max(18, min(75, augmented_age))
            
            augmented_budget = scenario["profile"]["budget"] * np.random.uniform(0.9, 1.1)
            augmented_budget = max(30.0, min(300.0, augmented_budget))
            
            hobbies = scenario["profile"]["hobbies"].copy()
            if len(hobbies) > 3 and np.random.random() < 0.1:
                hobbies = hobbies[:max(2, len(hobbies)-1)]
            # Don't shuffle
            
            preferences = scenario["profile"]["preferences"].copy()
            if np.random.random() < 0.1 and len(preferences) > 3:
                preferences = preferences[:max(2, len(preferences)-1)]
            # Don't shuffle
            
            # Create user profile from synthetic data
            user = UserProfile(
                age=int(augmented_age),
                hobbies=hobbies,
                relationship=scenario["profile"]["relationship"],
                budget=float(augmented_budget),
                occasion=scenario["profile"]["occasion"],
                personality_traits=preferences
            )
            
            # Get available gifts
            available_gifts = self.env.gift_catalog
            
            # Create target
            target = {
                'expected_categories': scenario["expected_categories"],
                'expected_tools': scenario["expected_tools"],
                'user_profile': user
            }
            
            batch_users.append(user)
            batch_gifts.append(available_gifts)
            batch_targets.append(target)
        
        return batch_users, batch_gifts, batch_targets
    
    def _generate_fallback_scenarios(self) -> List[Dict]:
        """Generate fallback scenarios if file not available"""
        return [
            {
                "profile": {
                    "age": 28,
                    "hobbies": ["technology", "fitness"],
                    "relationship": "friend",
                    "budget": 150.0,
                    "occasion": "birthday",
                    "preferences": ["trendy", "practical"]
                },
                "expected_categories": ["technology", "fitness"],
                "expected_tools": ["price_comparison", "review_analysis"]
            },
            {
                "profile": {
                    "age": 45,
                    "hobbies": ["gardening", "cooking"],
                    "relationship": "mother",
                    "budget": 120.0,
                    "occasion": "mothers_day",
                    "preferences": ["practical", "eco-friendly"]
                },
                "expected_categories": ["gardening", "cooking"],
                "expected_tools": ["review_analysis", "inventory_check"]
            }
        ]
    
    def compute_enhanced_loss(self, model_outputs: Dict[str, torch.Tensor], 
                            targets: List[Dict]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute enhanced loss with multiple components"""
        device = self.device
        batch_size = len(targets)
        
        # Extract model outputs
        category_scores = model_outputs['category_scores']
        tool_scores = model_outputs['tool_scores']
        predicted_rewards = model_outputs['predicted_rewards']
        action_probs = model_outputs['action_probs']
        
        # Initialize loss components
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}
        
        # 1. Category matching loss with label smoothing (v11 ORIGINAL)
        category_targets = []
        label_smoothing = 0.1  # RESTORED: Original v11 value
        
        for target in targets:
            expected_cats = target['expected_categories']
            # Create target vector with label smoothing (v11 original)
            target_vector = torch.full((len(self.model.gift_categories),), 
                                      label_smoothing / len(self.model.gift_categories), 
                                      device=device)
            for cat in expected_cats:
                if cat in self.model.gift_categories:
                    idx = self.model.gift_categories.index(cat)
                    target_vector[idx] = 1.0 - label_smoothing + (label_smoothing / len(self.model.gift_categories))
            category_targets.append(target_vector)
        
        if category_targets:
            category_target_tensor = torch.stack(category_targets)
            # Ensure category_scores has the right shape
            if category_scores.dim() == 3:
                category_scores = category_scores.squeeze(1)
            category_loss = nn.BCELoss()(category_scores, category_target_tensor)
            total_loss += self.config.get('category_loss_weight', 0.50) * category_loss  # INCREASED to 0.50 to prevent category matching collapse
            loss_components['category_loss'] = category_loss.item()
        
        # 2. Tool diversity loss (filtered by curriculum)
        tool_targets = []
        tool_names = list(self.model.tool_registry.list_tools())
        available_curriculum_tools = self.available_tools_by_stage[self.curriculum_stage]
        
        for target in targets:
            expected_tools = target['expected_tools']
            # Filter expected tools by curriculum stage
            filtered_expected_tools = [t for t in expected_tools if t in available_curriculum_tools]
            
            target_vector = torch.zeros(len(tool_names), device=device)
            for tool in filtered_expected_tools:
                if tool in tool_names:
                    idx = tool_names.index(tool)
                    target_vector[idx] = 1.0
            tool_targets.append(target_vector)
        
        if tool_targets:
            tool_target_tensor = torch.stack(tool_targets)
            # Ensure tool_scores has the right shape
            if tool_scores.dim() == 3:
                tool_scores = tool_scores.squeeze(1)
            tool_loss = nn.BCELoss()(tool_scores, tool_target_tensor)
            total_loss += self.config.get('tool_diversity_loss_weight', 0.30) * tool_loss  # BALANCED at 0.30 to give more weight to category learning
            loss_components['tool_loss'] = tool_loss.item()
        
        # 3. Enhanced reward prediction loss with tool execution feedback
        if predicted_rewards.numel() > 0:
            # Calculate target rewards based on category, tool matches, and tool execution
            target_rewards = []
            for target in targets:
                # Base reward from category match
                expected_cats = set(target['expected_categories'])
                base_reward = 0.5  # Base
                
                # Add bonus for budget appropriateness
                budget = target['user_profile'].budget
                if 50 <= budget <= 150:
                    base_reward += 0.15
                elif budget > 150:
                    base_reward += 0.20
                
                # Add bonus for hobby diversity
                num_hobbies = len(target['user_profile'].hobbies)
                base_reward += min(0.15, num_hobbies * 0.05)
                
                # Add tool execution reward (this is the key addition!)
                tool_exec_reward = target.get('tool_execution_reward', 0.0)
                base_reward += tool_exec_reward
                
                # Clamp to reasonable range
                base_reward = max(0.4, min(1.0, base_reward))
                target_rewards.append(base_reward)
            
            target_reward_tensor = torch.tensor(target_rewards, device=device)
            
            # Fix shape mismatch - predicted_rewards can be [batch, num_gifts] or [batch, 1] or [batch]
            # We need to reduce it to [batch] to match target
            
            # Debug: Print shapes to understand the issue
            # print(f"DEBUG: predicted_rewards shape: {predicted_rewards.shape}, target shape: {target_reward_tensor.shape}")
            
            if predicted_rewards.dim() == 3:
                # [batch, 1, 1] -> [batch]
                avg_predicted_reward = predicted_rewards.squeeze(-1).squeeze(-1)
            elif predicted_rewards.dim() == 2:
                # [batch, num_gifts] or [batch, 1] -> [batch]
                if predicted_rewards.size(-1) > 1:
                    # Multiple predictions per sample (e.g., per gift), take mean across gifts
                    avg_predicted_reward = predicted_rewards.mean(dim=-1)
                else:
                    # Single prediction per sample, just squeeze
                    avg_predicted_reward = predicted_rewards.squeeze(-1)
            elif predicted_rewards.dim() == 1:
                # Already [batch]
                avg_predicted_reward = predicted_rewards
            else:
                # Fallback: flatten to 1D
                avg_predicted_reward = predicted_rewards.view(-1)
            
            # Ensure both are 1D tensors [batch]
            if target_reward_tensor.dim() > 1:
                target_reward_tensor = target_reward_tensor.view(-1)
            if avg_predicted_reward.dim() > 1:
                avg_predicted_reward = avg_predicted_reward.view(-1)
            
            # Final safety check: ensure same size
            if avg_predicted_reward.size(0) != target_reward_tensor.size(0):
                # If sizes don't match, take first batch_size elements
                batch_size = target_reward_tensor.size(0)
                avg_predicted_reward = avg_predicted_reward[:batch_size]
            
            reward_loss = nn.MSELoss()(avg_predicted_reward, target_reward_tensor)
            total_loss += self.config.get('reward_loss_weight', 0.40) * reward_loss
            loss_components['reward_loss'] = reward_loss.item()
            
            # Track tool execution contribution
            avg_tool_reward = np.mean([t.get('tool_execution_reward', 0.0) for t in targets])
            loss_components['tool_execution_reward'] = avg_tool_reward
        
        # 4. Tool execution success loss (binary classification)
        tool_execution_loss = torch.tensor(0.0, device=device)
        tool_usage_penalty = torch.tensor(0.0, device=device)  # NEW: Penalty for not using tools
        tool_selection_penalty = torch.tensor(0.0, device=device)  # NEW: Penalty for not SELECTING tools
        
        # NEW: Check if tools were selected by the model (before execution)
        # This penalizes the model for not selecting tools in the first place
        if 'selected_tools' in model_outputs:
            selected_tools_list = model_outputs['selected_tools']
            for i, target in enumerate(targets):
                expected_tools = set(target['expected_tools'])
                # Check if model selected any tools
                if i < len(selected_tools_list):
                    selected = selected_tools_list[i] if isinstance(selected_tools_list[i], list) else []
                    # VERY HEAVY penalty if NO tools were selected
                    if len(selected) == 0 and len(expected_tools) > 0:
                        tool_selection_penalty += 1.0  # VERY strong penalty (doubled)
                    # Heavy penalty if only 1 tool selected
                    elif len(selected) == 1 and len(expected_tools) > 1:
                        tool_selection_penalty += 0.5  # Strong penalty
                    # Moderate penalty if too few tools selected
                    elif len(selected) < min(2, len(expected_tools)):
                        tool_selection_penalty += 0.3
            
            if len(targets) > 0:
                tool_selection_penalty = tool_selection_penalty / len(targets)
                total_loss += 0.50 * tool_selection_penalty  # VERY strong weight for selection penalty (v10)
                loss_components['tool_selection_penalty'] = tool_selection_penalty.item()
        
        if 'tool_execution_success' in model_outputs and isinstance(model_outputs['tool_execution_success'], list):
            for i, target in enumerate(targets):
                expected_tools = set(target['expected_tools'])
                # tool_execution_success is a list of dicts
                tool_success = model_outputs['tool_execution_success'][i] if i < len(model_outputs['tool_execution_success']) else {}
                
                if isinstance(tool_success, dict):
                    # Penalize if expected tools were not executed successfully
                    for expected_tool in expected_tools:
                        if expected_tool not in tool_success or not tool_success[expected_tool]:
                            tool_execution_loss += 0.1
                    
                    # Penalize if unexpected tools were executed
                    for executed_tool, success in tool_success.items():
                        if executed_tool not in expected_tools and success:
                            tool_execution_loss += 0.05
                    
                    # NEW: Heavy penalty if NO tools were used at all
                    if len(tool_success) == 0 and len(expected_tools) > 0:
                        tool_usage_penalty += 0.3
            
            if len(targets) > 0:
                tool_execution_loss = tool_execution_loss / len(targets)
                tool_usage_penalty = tool_usage_penalty / len(targets)
                total_loss += self.config.get('tool_execution_loss_weight', 0.20) * tool_execution_loss
                total_loss += 0.25 * tool_usage_penalty  # NEW: Strong penalty for not using tools
                loss_components['tool_execution_loss'] = tool_execution_loss.item()
                loss_components['tool_usage_penalty'] = tool_usage_penalty.item()
        
        # 5. Semantic matching loss
        semantic_loss = self._compute_semantic_matching_loss(category_scores, targets)
        total_loss += self.config.get('semantic_matching_loss_weight', 0.20) * semantic_loss
        loss_components['semantic_loss'] = semantic_loss.item()
        
        # 6. Regularization losses
        # L2 regularization for embeddings
        embedding_reg = (
            torch.norm(self.model.hobby_embeddings.weight) +
            torch.norm(self.model.category_embeddings.weight) +
            torch.norm(self.model.preference_embeddings.weight)
        )
        total_loss += self.config.get('embedding_reg_weight', 1e-5) * embedding_reg
        loss_components['embedding_reg'] = embedding_reg.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components
    
    def _calculate_target_reward(self, user_profile: UserProfile, category_scores: torch.Tensor) -> torch.Tensor:
        """Calculate target reward based on enhanced criteria"""
        device = self.device
        
        # Base reward from category matching
        base_reward = category_scores.max().item()
        
        # Budget factor
        budget_factor = min(1.0, user_profile.budget / 200.0)
        
        # Age appropriateness factor
        age_factor = 1.0 if 20 <= user_profile.age <= 60 else 0.8
        
        # Hobby diversity factor
        hobby_factor = min(1.0, len(user_profile.hobbies) / 3.0)
        
        # Combined target reward
        target_reward = base_reward * budget_factor * age_factor * hobby_factor
        target_reward = max(0.1, min(1.0, target_reward))  # Clamp between 0.1 and 1.0
        
        return torch.tensor(target_reward, device=device)
    
    def _compute_semantic_matching_loss(self, category_scores: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        """Compute semantic matching loss to encourage better understanding"""
        device = self.device
        
        # Encourage high scores for semantically related categories
        semantic_loss = torch.tensor(0.0, device=device)
        
        for i, target in enumerate(targets):
            user_hobbies = target['user_profile'].hobbies
            expected_categories = target['expected_categories']
            
            # Find semantically related categories
            related_categories = []
            for hobby in user_hobbies:
                if hobby in self.model.gift_categories:
                    related_categories.append(hobby)
            
            # Encourage high scores for related categories
            for cat in related_categories:
                if cat in self.model.gift_categories:
                    cat_idx = self.model.gift_categories.index(cat)
                    # Encourage high score for this category
                    semantic_loss += (1.0 - category_scores[i, cat_idx]) ** 2
        
        return semantic_loss / len(targets) if targets else torch.tensor(0.0, device=device)
    
    def evaluate_model(self, num_eval_episodes: int = 50) -> Dict[str, float]:
        """Evaluate the integrated enhanced model on validation set with tool execution"""
        self.model.eval()
        self.tool_result_encoder.eval()
        
        eval_metrics = {
            'category_match_rate': 0.0,
            'tool_match_rate': 0.0,
            'average_reward': 0.0,
            'tool_execution_success': 0.0,
            'recommendation_quality': 0.0
        }
        
        category_matches = 0
        tool_matches = 0
        total_rewards = []
        tool_execution_successes = []
        
        with torch.no_grad():
            for episode in range(num_eval_episodes):
                # Generate evaluation batch from validation set
                users, gifts, targets = self.generate_training_batch(batch_size=1, use_validation=True)
                user = users[0]
                target = targets[0]
                
                # Reset environment
                env_state = self.env.reset(user)
                
                # Forward pass
                carry = self.model.initial_carry({"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1, 1)})
                carry, model_outputs, selected_tools = self.model.forward_with_enhancements(
                    carry, env_state, self.env.gift_catalog
                )
                
                # Execute tools and measure success (with all curriculum tools available)
                tool_execution_reward = 0.0
                tool_results = {}
                expected_tools = set(target['expected_tools'])
                
                for tool_name in selected_tools:
                    try:
                        if tool_name == 'price_comparison':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'price_comparison', gifts=self.env.gift_catalog, budget=user.budget
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            if result and len(result.get('in_budget', [])) > 0:
                                tool_execution_reward += 0.3  # Increased from 0.2
                            # Negative reward if not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05  # Reduced from -0.1
                        
                        elif tool_name == 'review_analysis':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'review_analysis', gifts=self.env.gift_catalog
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            if result and result.get('average_rating', 0) > 3.5:  # Relaxed from 4.0
                                tool_execution_reward += 0.25  # Increased from 0.15
                            # Negative reward if not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05  # Reduced from -0.1
                        
                        elif tool_name == 'inventory_check':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'inventory_check', gifts=self.env.gift_catalog
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            if result and len(result.get('available', [])) > 0:
                                tool_execution_reward += 0.2  # Increased from 0.1
                            # Negative reward if not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05  # Reduced from -0.1
                        
                        elif tool_name == 'trend_analysis':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'trend_analysis', gifts=self.env.gift_catalog, user_age=user.age
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            if result and len(result.get('trending', [])) > 0:
                                tool_execution_reward += 0.25  # Increased from 0.15
                            # Negative reward if not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05  # Reduced from -0.1
                        
                        elif tool_name == 'budget_optimizer':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'budget_optimizer', gifts=self.env.gift_catalog, budget=user.budget
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            if result and len(result.get('optimized_gifts', [])) > 0:
                                tool_execution_reward += 0.25
                            # Negative reward if not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                    except:
                        tool_execution_reward -= 0.05  # Penalty for failed execution
                        continue
                
                # Bonus for successful combinations
                if len(tool_results) >= 2:
                    tool_execution_reward += 0.15 * (len(tool_results) - 1)  # Increased from 0.1
                
                tool_execution_successes.append(tool_execution_reward)
                
                # Check category matching
                category_scores = model_outputs['category_scores'][0]
                top_categories = torch.topk(category_scores, 3).indices
                predicted_categories = [self.model.gift_categories[idx] for idx in top_categories]
                
                expected_categories = set(target['expected_categories'])
                actual_categories = set(predicted_categories)
                if len(expected_categories.intersection(actual_categories)) > 0:
                    category_matches += 1
                
                # Check tool matching
                expected_tools = set(target['expected_tools'])
                actual_tools = set(selected_tools)
                if len(expected_tools.intersection(actual_tools)) > 0:
                    tool_matches += 1
                
                # Calculate reward (including tool execution)
                predicted_reward = model_outputs['predicted_rewards'].mean().item()
                total_rewards.append(predicted_reward + tool_execution_reward)
        
        eval_metrics['category_match_rate'] = category_matches / num_eval_episodes
        eval_metrics['tool_match_rate'] = tool_matches / num_eval_episodes
        eval_metrics['average_reward'] = np.mean(total_rewards)
        eval_metrics['tool_execution_success'] = np.mean(tool_execution_successes)
        eval_metrics['recommendation_quality'] = (
            eval_metrics['category_match_rate'] + 
            eval_metrics['average_reward'] +
            eval_metrics['tool_execution_success']
        ) / 3
        
        self.model.train()
        self.tool_result_encoder.train()
        return eval_metrics
    
    def train_epoch(self, epoch: int, num_batches: int = 100) -> Dict[str, float]:
        """Train one epoch with gradient accumulation and tool execution"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'category_loss': [],
            'tool_loss': [],
            'reward_loss': [],
            'semantic_loss': [],
            'tool_execution_reward': [],
            'tool_execution_loss': []
        }
        
        accumulation_steps = 2  # Accumulate gradients over 2 batches
        self.optimizer.zero_grad()
        
        for batch_idx in range(num_batches):
            # Generate training batch
            users, gifts, targets = self.generate_training_batch(
                batch_size=self.config.get('batch_size', 16),
                use_validation=False
            )
            
            # Forward pass with tool execution
            batch_outputs = []
            batch_tool_rewards = []
            
            for i, user in enumerate(users):
                env_state = self.env.reset(user)
                carry = self.model.initial_carry({
                    "inputs": torch.zeros(1, 10, device=self.device), 
                    "puzzle_identifiers": torch.zeros(1, 1, device=self.device)
                })
                
                # Forward pass with tool selection and execution
                # Use forward_with_tools for iterative tool usage
                if hasattr(self.model, 'forward_with_tools'):
                    carry, model_output, tool_calls_result = self.model.forward_with_tools(
                        carry, env_state, self.env.gift_catalog, max_tool_calls=2
                    )
                    # Extract selected tools from tool_calls
                    selected_tools = [tc.tool_name for tc in tool_calls_result] if tool_calls_result else []
                else:
                    # Fallback to forward_with_enhancements
                    carry, model_output, selected_tools = self.model.forward_with_enhancements(
                        carry, env_state, self.env.gift_catalog
                    )
                
                # Execute selected tools with parameters and sequential execution
                tool_results = {}
                tool_execution_reward = 0.0
                tool_execution_success = {}
                expected_tools = set(targets[i]['expected_tools'])
                
                # Filter tools based on curriculum stage
                available_tools = self.available_tools_by_stage[self.curriculum_stage]
                filtered_tools = [t for t in selected_tools if t in available_tools]
                
                # AGGRESSIVE EXPLORATION for underused tools (Stage 2+)
                if self.curriculum_stage >= 2:
                    # Find all underused tools (below threshold)
                    underused_tools = [tool for tool in available_tools 
                                      if self.tool_selection_counts.get(tool, 0) < self.exploration_bonus_threshold]
                    
                    if underused_tools:
                        # Force exploration with higher probability
                        if random.random() < self.force_exploration_prob:
                            # Pick the LEAST used tool
                            least_used_tool = min(underused_tools, key=lambda t: self.tool_selection_counts.get(t, 0))
                            
                            if least_used_tool not in filtered_tools:
                                filtered_tools.append(least_used_tool)
                                if batch_idx == 0 and i == 0:
                                    print(f"  🔍 EXPLORATION: Forcing tool '{least_used_tool}' (count: {self.tool_selection_counts[least_used_tool]})")
                        
                        # ADDITIONAL: Force multiple underused tools if count is very low
                        for tool in underused_tools:
                            if self.tool_selection_counts.get(tool, 0) < 1000 and random.random() < 0.3:
                                if tool not in filtered_tools:
                                    filtered_tools.append(tool)
                
                # Update tool selection counts
                for tool in filtered_tools:
                    self.tool_selection_counts[tool] += 1
                
                # DEBUG: Log selected tools
                if batch_idx == 0 and i == 0:
                    print(f"  🎯 Selected tools: {selected_tools}")
                    print(f"  ✅ Filtered tools: {filtered_tools}")
                    print(f"  📋 Expected tools: {list(expected_tools)}")
                    print(f"  🎓 Available tools (curriculum): {available_tools}")
                    
                    # Log tool scores if available
                    if 'tool_scores' in model_output:
                        tool_scores_tensor = model_output['tool_scores']
                        if tool_scores_tensor.dim() > 1:
                            tool_scores_tensor = tool_scores_tensor[0]
                        tool_names = list(self.model.tool_registry.list_tools())
                        print(f"  📊 Tool scores:")
                        for idx, (tool_name, score) in enumerate(zip(tool_names, tool_scores_tensor.tolist())):
                            print(f"     {tool_name}: {score:.3f}")
                
                # Negative reward for selecting unavailable tools (curriculum penalty)
                if len(selected_tools) > len(filtered_tools):
                    tool_execution_reward -= 0.05 * (len(selected_tools) - len(filtered_tools))
                
                # Sequential tool execution with context passing
                tool_context = {}
                
                for tool_name in filtered_tools:
                    try:
                        # Get tool parameters from model (now available!)
                        tool_params_dict = model_output.get('tool_params', {})
                        
                        # Execute tool with parameters and context
                        if tool_name == 'price_comparison':
                            # Use model-generated budget parameter if available, otherwise user budget
                            if tool_name in tool_params_dict and 'budget' in tool_params_dict[tool_name]:
                                budget = tool_params_dict[tool_name]['budget']
                            else:
                                budget = user.budget
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'price_comparison',
                                gifts=self.env.gift_catalog,
                                budget=budget
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            tool_context['price_info'] = result
                            
                            # Positive reward for finding gifts in budget (increased from 0.2 to 0.3)
                            if result and len(result.get('in_budget', [])) > 0:
                                tool_execution_reward += 0.3
                                tool_execution_success[tool_name] = True
                            else:
                                tool_execution_success[tool_name] = False
                            
                            # Negative reward if tool was not expected (reduced from -0.1 to -0.05)
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                        
                        elif tool_name == 'review_analysis':
                            # Can use price context if available
                            gifts_to_analyze = self.env.gift_catalog
                            if 'price_info' in tool_context:
                                # Focus on in-budget items
                                in_budget_items = tool_context['price_info'].get('in_budget', [])
                                if in_budget_items:
                                    # Extract IDs from in_budget items (they are GiftItem objects)
                                    in_budget_ids = [item.id if hasattr(item, 'id') else item['id'] for item in in_budget_items]
                                    gifts_to_analyze = [g for g in self.env.gift_catalog if g.id in in_budget_ids]
                            
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'review_analysis',
                                gifts=gifts_to_analyze
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            tool_context['review_info'] = result
                            
                            # Positive reward for finding highly rated items (relaxed from 4.0 to 3.5, increased from 0.15 to 0.25)
                            if result and result.get('average_rating', 0) > 3.5:
                                tool_execution_reward += 0.25
                                tool_execution_success[tool_name] = True
                            else:
                                tool_execution_success[tool_name] = False
                            
                            # Negative reward if tool was not expected (reduced from -0.1 to -0.05)
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                        
                        elif tool_name == 'inventory_check':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'inventory_check',
                                gifts=self.env.gift_catalog
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            tool_context['inventory_info'] = result
                            
                            # Positive reward for checking availability (increased from 0.1 to 0.2)
                            if result and len(result.get('available', [])) > 0:
                                tool_execution_reward += 0.2
                                tool_execution_success[tool_name] = True
                            else:
                                tool_execution_success[tool_name] = False
                            
                            # Negative reward if tool was not expected (reduced from -0.1 to -0.05)
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                        
                        elif tool_name == 'trend_analysis':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'trend_analysis',
                                gifts=self.env.gift_catalog,
                                user_age=user.age
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            tool_context['trend_info'] = result
                            
                            # Positive reward for finding trending items (increased from 0.15 to 0.25)
                            if result and len(result.get('trending', [])) > 0:
                                tool_execution_reward += 0.25
                                tool_execution_success[tool_name] = True
                            else:
                                tool_execution_success[tool_name] = False
                            
                            # Negative reward if tool was not expected (reduced from -0.1 to -0.05)
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                        
                        elif tool_name == 'budget_optimizer':
                            tool_call = self.model.tool_registry.call_tool_by_name(
                                'budget_optimizer',
                                gifts=self.env.gift_catalog,
                                budget=user.budget
                            )
                            result = tool_call.result if tool_call.success else None
                            tool_results[tool_name] = result
                            tool_context['budget_info'] = result
                            
                            # Positive reward for optimizing budget
                            if result and len(result.get('optimized_gifts', [])) > 0:
                                tool_execution_reward += 0.25
                                tool_execution_success[tool_name] = True
                            else:
                                tool_execution_success[tool_name] = False
                            
                            # Negative reward if tool was not expected
                            if tool_name not in expected_tools:
                                tool_execution_reward -= 0.05
                        
                    except Exception as e:
                        tool_execution_success[tool_name] = False
                        # Penalty for failed execution
                        tool_execution_reward -= 0.05
                        continue
                
                # Bonus reward for using correct tool combinations (increased from 0.1 to 0.15)
                if len(filtered_tools) >= 2:
                    successful_tools = [t for t, success in tool_execution_success.items() if success]
                    if len(successful_tools) >= 2:
                        # Bonus for successful multi-tool usage
                        tool_execution_reward += 0.15 * (len(successful_tools) - 1)
                
                # Encode tool results for future use
                # Note: Tool feedback integration with carry state is prepared for future enhancement
                if tool_results:
                    encoded_tool_results = self.tool_result_encoder(tool_results, self.device)
                    # Store encoded results in model output for potential future use
                    model_output['encoded_tool_results'] = encoded_tool_results
                
                # Add tool execution results to model output
                model_output['tool_results'] = tool_results
                model_output['tool_execution_reward'] = tool_execution_reward
                model_output['tool_execution_success'] = tool_execution_success
                # NEW: Add selected_tools to track what model selected (before filtering)
                model_output['selected_tools_raw'] = selected_tools  # Raw selection from model
                model_output['selected_tools'] = filtered_tools  # After curriculum filtering
                
                batch_outputs.append(model_output)
                batch_tool_rewards.append(tool_execution_reward)
            
            # Stack outputs
            stacked_outputs = {}
            for key in batch_outputs[0].keys():
                if isinstance(batch_outputs[0][key], torch.Tensor):
                    # Check if all outputs have this key as tensor with same shape
                    try:
                        tensors_to_stack = []
                        for output in batch_outputs:
                            if key in output and isinstance(output[key], torch.Tensor):
                                tensors_to_stack.append(output[key])
                        
                        if len(tensors_to_stack) == len(batch_outputs):
                            stacked_outputs[key] = torch.stack(tensors_to_stack)
                        else:
                            # Not all outputs have this tensor, store as list
                            stacked_outputs[key] = [output.get(key) for output in batch_outputs]
                    except RuntimeError:
                        # Tensors have different shapes, store as list
                        stacked_outputs[key] = [output.get(key) for output in batch_outputs]
                else:
                    stacked_outputs[key] = [output[key] for output in batch_outputs]
            
            # Add tool execution rewards to targets
            for i, target in enumerate(targets):
                target['tool_execution_reward'] = batch_tool_rewards[i]
            
            # Compute loss with tool execution feedback
            loss, loss_components = self.compute_enhanced_loss(stacked_outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for both model and tool encoder (CONSERVATIVE for stability)
                all_params = list(self.model.parameters()) + list(self.tool_result_encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)  # Very conservative to prevent instability
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Record losses (unscaled)
            for key, value in loss_components.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value * accumulation_steps)
            
            # Record tool execution rewards
            if batch_tool_rewards:
                epoch_losses['tool_execution_reward'].append(np.mean(batch_tool_rewards))
            
            # Print progress
            if batch_idx % 20 == 0:
                avg_tool_reward = np.mean(batch_tool_rewards) if batch_tool_rewards else 0.0
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, Tool Reward: {avg_tool_reward:.3f}")
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            if values:
                epoch_metrics[key] = np.mean(values)
        
        # Log tool selection counts (Stage 3 and 4)
        if self.curriculum_stage >= 3:
            print(f"\n📊 Tool Selection Counts (Epoch {epoch}):")
            for tool_name, count in sorted(self.tool_selection_counts.items(), key=lambda x: x[1], reverse=True):
                if tool_name in self.available_tools_by_stage[3]:
                    print(f"   {tool_name}: {count}")
        
        return epoch_metrics
    
    def train(self, num_epochs: int = 150, eval_frequency: int = 5, start_epoch: int = 0):
        """Main training loop with ADAPTIVE curriculum, early stopping, and LR scheduling"""
        # Calculate actual end epoch
        end_epoch = start_epoch + num_epochs
        print(f"🚀 Starting training for {num_epochs} epochs (epoch {start_epoch} to {end_epoch})")
        print(f"📊 Early stopping patience: {self.early_stopping_patience} epochs")
        print(f"📚 Curriculum learning enabled with {len(self.available_tools_by_stage)} stages")
        
        for epoch in range(start_epoch, end_epoch):
            # ADAPTIVE curriculum stage progression based on tool usage
            old_stage = self.curriculum_stage
            
            if epoch < 20:  # Stage 0: Learn price_comparison
                self.curriculum_stage = 0
            elif epoch < 60:  # Stage 1: EXTENDED - Learn price + review (was 45)
                self.curriculum_stage = 1
            elif epoch < 100:  # Stage 2: EXTENDED - Add inventory (was 75)
                self.curriculum_stage = 2
            else:
                # For Stage 3+, check if tools are being used enough before advancing
                current_stage_tools = self.available_tools_by_stage[self.curriculum_stage]
                min_usage = min([self.tool_selection_counts.get(tool, 0) for tool in current_stage_tools])
                
                # Only advance if minimum tool usage threshold is met OR enough epochs have passed
                epochs_in_stage = epoch - self.stage_start_epochs.get(self.curriculum_stage, epoch)
                
                stage_timeout = self.config.get('stage_timeout_epochs', 35)
                
                if self.curriculum_stage == 2 and (min_usage >= self.min_tool_usage_per_stage or epochs_in_stage >= stage_timeout):
                    self.curriculum_stage = 3
                elif self.curriculum_stage == 3 and (min_usage >= self.min_tool_usage_per_stage or epochs_in_stage >= stage_timeout + 5):
                    self.curriculum_stage = 4
                elif self.curriculum_stage < 2:
                    self.curriculum_stage = 2
            
            # Track stage changes
            if old_stage != self.curriculum_stage:
                self.stage_start_epochs[self.curriculum_stage] = epoch
                print(f"\n🎓 CURRICULUM STAGE ADVANCED: {old_stage} → {self.curriculum_stage}")
                print(f"   New tools available: {self.available_tools_by_stage[self.curriculum_stage]}")
            
            available_tools = self.available_tools_by_stage[self.curriculum_stage]
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs} - Curriculum Stage {self.curriculum_stage} - Tools: {available_tools}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch, num_batches=50)
            
            # Log training metrics
            print(f"Training - Total Loss: {train_metrics.get('total_loss', 0):.4f}, "
                  f"Category Loss: {train_metrics.get('category_loss', 0):.4f}, "
                  f"Tool Loss: {train_metrics.get('tool_loss', 0):.4f}, "
                  f"Tool Exec Loss: {train_metrics.get('tool_execution_loss', 0):.4f}, "
                  f"Tool Reward: {train_metrics.get('tool_execution_reward', 0):.3f}")
            
            # Evaluate periodically
            if (epoch + 1) % eval_frequency == 0:
                print("🔍 Evaluating model...")
                eval_metrics = self.evaluate_model(num_eval_episodes=30)
                
                print(f"Evaluation - Category Match: {eval_metrics['category_match_rate']:.1%}, "
                      f"Tool Match: {eval_metrics['tool_match_rate']:.1%}, "
                      f"Tool Exec Success: {eval_metrics['tool_execution_success']:.3f}, "
                      f"Avg Reward: {eval_metrics['average_reward']:.3f}, "
                      f"Quality: {eval_metrics['recommendation_quality']:.3f}")
                
                # Check for improvement with SMALLER threshold
                # Use a balanced score combining multiple metrics
                current_score = (
                    eval_metrics['recommendation_quality'] * 0.4 +  # Main quality metric
                    eval_metrics['category_match_rate'] * 0.3 +     # Category accuracy
                    eval_metrics['tool_match_rate'] * 0.2 +         # Tool selection accuracy
                    eval_metrics['tool_execution_success'] * 0.1    # Tool execution success
                )
                improvement_threshold = self.config.get('improvement_threshold', 0.001)  # Consider 0.1% improvement as progress
                
                # Update learning rate scheduler with balanced score (higher is better)
                self.scheduler.step(current_score)
                
                if current_score > self.best_eval_score + improvement_threshold:
                    self.best_eval_score = current_score
                    self.patience_counter = 0
                    self.save_model(f"integrated_enhanced_best.pt", epoch, eval_metrics)
                    print(f"💾 New best model saved! Score: {current_score:.3f}")
                else:
                    self.patience_counter += 1
                    print(f"⏳ No improvement for {self.patience_counter} evaluation(s)")
                
                # Early stopping check - MORE LENIENT
                # patience_counter counts evaluations, not epochs
                # We want to wait for early_stopping_patience evaluations before stopping
                if self.patience_counter >= self.early_stopping_patience:
                    # Don't stop if we're in early stages or just advanced curriculum
                    epochs_in_current_stage = epoch - self.stage_start_epochs.get(self.curriculum_stage, 0)
                    
                    if self.curriculum_stage < 4 or epochs_in_current_stage < 20:
                        print(f"⏰ Early stopping delayed - still in curriculum stage {self.curriculum_stage}")
                        self.patience_counter = self.patience_counter // 2  # Reset partially
                    else:
                        print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                        print(f"🏆 Best score achieved: {self.best_eval_score:.3f}")
                        break
            
            # Save checkpoint
            if (epoch + 1) % 25 == 0:
                self.save_model(f"integrated_enhanced_epoch_{epoch + 1}.pt", epoch, train_metrics)
        
        print(f"\n🎉 Training completed! Best score: {self.best_eval_score:.3f}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        print(f"📂 Loading model from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load with strict=True to ensure architecture matches
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # Load tool result encoder if available
        if 'tool_result_encoder_state_dict' in checkpoint:
            self.tool_result_encoder.load_state_dict(checkpoint['tool_result_encoder_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        if 'training_history' in checkpoint:
            self.best_eval_score = checkpoint['training_history'].get('best_score', 0.0)
            self.patience_counter = checkpoint['training_history'].get('patience_counter', 0)
            if 'curriculum_stage' in checkpoint['training_history']:
                self.curriculum_stage = checkpoint['training_history']['curriculum_stage']
        
        print(f"✅ Model loaded successfully from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_model(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with comprehensive information"""
        os.makedirs("checkpoints/integrated_enhanced", exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'tool_result_encoder_state_dict': self.tool_result_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_history': {
                'best_score': self.best_eval_score,
                'patience_counter': self.patience_counter,
                'training_scenarios': len(self.train_scenarios),
                'validation_scenarios': len(self.val_scenarios),
                'curriculum_stage': self.curriculum_stage
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.tool_result_encoder.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) + sum(p.numel() for p in self.tool_result_encoder.parameters() if p.requires_grad),
                'enhanced_components': [
                    'user_profiling', 'category_matching', 'tool_selection', 
                    'reward_prediction', 'cross_modal_fusion', 'tool_result_encoding'
                ],
                'training_date': datetime.now().isoformat(),
                'optimization_version': 'v3.0'
            }
        }
        
        filepath = os.path.join("checkpoints/integrated_enhanced", filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Integrated Enhanced TRM Model')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--use_synthetic_data', action='store_true',
                       help='Use synthetic data for training')
    parser.add_argument('--synthetic_ratio', type=float, default=0.5,
                       help='Ratio of synthetic data in batches (0.0-1.0, default: 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None - random)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility (only if seed is provided)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("🚀 INTEGRATED ENHANCED TRM TRAINING")
        print("=" * 60)
        print(f"🎲 Random seed: {args.seed}")
    else:
        print("🚀 INTEGRATED ENHANCED TRM TRAINING")
        print("=" * 60)
        print("🎲 Random seed: None (random initialization)")
    
    if args.use_synthetic_data:
        print(f"🤖 Sentetik veri modu aktif - Oran: {args.synthetic_ratio:.1%}")
    
    # Create enhanced configuration
    config = create_integrated_enhanced_config()
    
    # Add training-specific parameters (v11 EXACT RESTORE - PROVEN WORKING)
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'eval_frequency': 5,
        'early_stopping_patience': 8,  # 8 evaluations = 40 epochs with eval_frequency=5
        'user_profile_lr': 5e-5,  # ULTRA STABLE: Prevent overfitting
        'category_matching_lr': 8e-5,  # ULTRA STABLE: Slow and steady
        'tool_selection_lr': 5e-5,  # ULTRA STABLE: Minimal tool focus
        'reward_prediction_lr': 5e-5,  # ULTRA STABLE: Prevent overfitting
        'main_lr': 5e-5,  # ULTRA STABLE: Overall stability
        'weight_decay': 0.03,  # INCREASED: Stronger regularization against overfitting
        'category_loss_weight': 2.00,  # CRITICAL: Maximum priority on category learning
        'tool_diversity_loss_weight': 0.15,  # MINIMAL: Reduce tool pressure further
        'tool_execution_loss_weight': 0.10,  # MINIMAL: Reduce execution pressure
        'reward_loss_weight': 0.10,  # MINIMAL: Balance with category
        'semantic_matching_loss_weight': 0.15,  # INCREASED: Help category understanding
        'embedding_reg_weight': 1.5e-5,
        'tool_encoder_lr': 3.0e-4,
        'hidden_dim': 128,
        'improvement_threshold': 0.001,
        'stage_timeout_epochs': 35
    })
    
    # Initialize trainer with synthetic data support
    trainer = IntegratedEnhancedTrainer(
        config, 
        use_synthetic_data=args.use_synthetic_data,
        synthetic_ratio=args.synthetic_ratio
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_model(args.resume)
        print(f"🔄 Resuming training from epoch {start_epoch}")
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        eval_frequency=config['eval_frequency'],
        start_epoch=start_epoch
    )
    
    print("🎉 Integrated Enhanced TRM training completed!")


if __name__ == "__main__":
    main()