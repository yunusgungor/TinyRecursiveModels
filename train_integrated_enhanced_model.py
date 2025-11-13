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
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import GiftRecommendationEnvironment, UserProfile, GiftItem
from models.rl.trainer import RLTrainer, TrainingConfig


class IntegratedEnhancedTrainer:
    """Trainer for the integrated enhanced model with all improvements"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = IntegratedEnhancedTRM(config).to(self.device)
        
        # Initialize environment
        self.env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
        
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
        
        # Early stopping
        self.best_eval_score = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        
        # Scenario storage
        self.train_scenarios = []
        self.val_scenarios = []
        
        print(f"🚀 Integrated Enhanced Trainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 Training scenarios: {len(self.train_scenarios)}")
        print(f"📊 Validation scenarios: {len(self.val_scenarios)}")
        
    def _load_and_split_scenarios(self):
        """Load scenarios and split into train/validation sets"""
        try:
            with open("data/expanded_user_scenarios.json", "r") as f:
                scenario_data = json.load(f)
            all_scenarios = scenario_data["scenarios"]
        except:
            try:
                with open("data/realistic_user_scenarios.json", "r") as f:
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
                'lr': self.config.get('category_matching_lr', 4e-5),  # Much more reduced
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
        
        optimizer = optim.AdamW(param_groups, weight_decay=self.config.get('weight_decay', 0.01))
        
        # Add learning rate scheduler with more aggressive reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=3, verbose=True, min_lr=1e-7
        )
        
        return optimizer
    
    def generate_training_batch(self, batch_size: int = 16, use_validation: bool = False) -> Tuple[List[UserProfile], List[List[GiftItem]], List[Dict]]:
        """Generate a training batch with diverse user profiles and scenarios with augmentation"""
        
        # Use appropriate scenario set
        scenarios = self.val_scenarios if use_validation else self.train_scenarios
        
        if not scenarios:
            scenarios = self._generate_fallback_scenarios()
        
        batch_users = []
        batch_gifts = []
        batch_targets = []
        
        for _ in range(batch_size):
            # Sample a scenario
            scenario = np.random.choice(scenarios)
            
            # Enhanced data augmentation with more aggressive variations
            augmented_age = scenario["profile"]["age"] + np.random.randint(-7, 8)
            augmented_age = max(18, min(75, augmented_age))  # Clamp to valid range
            
            augmented_budget = scenario["profile"]["budget"] * np.random.uniform(0.7, 1.3)
            augmented_budget = max(30.0, min(300.0, augmented_budget))  # Clamp to valid range
            
            # More aggressive hobby manipulation
            hobbies = scenario["profile"]["hobbies"].copy()
            if len(hobbies) > 2 and np.random.random() < 0.4:  # Increased probability
                hobbies = hobbies[:max(1, len(hobbies)-1)]  # Drop at least one
            np.random.shuffle(hobbies)
            
            # More aggressive preference manipulation
            preferences = scenario["profile"]["preferences"].copy()
            if np.random.random() < 0.3 and len(preferences) > 2:  # Increased probability
                preferences = preferences[:max(1, len(preferences)-1)]  # Drop at least one
            np.random.shuffle(preferences)
            
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
        
        # 1. Category matching loss with label smoothing
        category_targets = []
        label_smoothing = 0.1  # Smooth labels to prevent overconfidence
        
        for target in targets:
            expected_cats = target['expected_categories']
            # Create target vector with label smoothing
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
            total_loss += self.config.get('category_loss_weight', 0.15) * category_loss  # Much more reduced
            loss_components['category_loss'] = category_loss.item()
        
        # 2. Tool diversity loss
        tool_targets = []
        tool_names = list(self.model.tool_registry.list_tools())
        for target in targets:
            expected_tools = target['expected_tools']
            target_vector = torch.zeros(len(tool_names), device=device)
            for tool in expected_tools:
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
            total_loss += self.config.get('tool_diversity_loss_weight', 0.25) * tool_loss  # Increased from 0.15
            loss_components['tool_loss'] = tool_loss.item()
        
        # 3. Enhanced reward prediction loss
        if predicted_rewards.numel() > 0:
            # Calculate target rewards based on category and tool matches
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
                
                # Clamp to reasonable range
                base_reward = max(0.4, min(0.9, base_reward))
                target_rewards.append(base_reward)
            
            target_reward_tensor = torch.tensor(target_rewards, device=device).unsqueeze(-1)
            
            # Expand to match predicted_rewards shape if needed
            if predicted_rewards.dim() > 1 and predicted_rewards.size(-1) > 1:
                avg_predicted_reward = predicted_rewards.mean(dim=-1, keepdim=True)
            else:
                avg_predicted_reward = predicted_rewards
            
            reward_loss = nn.MSELoss()(avg_predicted_reward, target_reward_tensor)
            total_loss += self.config.get('reward_loss_weight', 0.40) * reward_loss
            loss_components['reward_loss'] = reward_loss.item()
        
        # 4. Semantic matching loss
        semantic_loss = self._compute_semantic_matching_loss(category_scores, targets)
        total_loss += self.config.get('semantic_matching_loss_weight', 0.20) * semantic_loss
        loss_components['semantic_loss'] = semantic_loss.item()
        
        # 5. Regularization losses
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
        """Evaluate the integrated enhanced model on validation set"""
        self.model.eval()
        
        eval_metrics = {
            'category_match_rate': 0.0,
            'tool_match_rate': 0.0,
            'average_reward': 0.0,
            'recommendation_quality': 0.0
        }
        
        category_matches = 0
        tool_matches = 0
        total_rewards = []
        
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
                
                # Calculate reward
                predicted_reward = model_outputs['predicted_rewards'].mean().item()
                total_rewards.append(predicted_reward)
        
        eval_metrics['category_match_rate'] = category_matches / num_eval_episodes
        eval_metrics['tool_match_rate'] = tool_matches / num_eval_episodes
        eval_metrics['average_reward'] = np.mean(total_rewards)
        eval_metrics['recommendation_quality'] = (eval_metrics['category_match_rate'] + 
                                                eval_metrics['average_reward']) / 2
        
        self.model.train()
        return eval_metrics
    
    def train_epoch(self, epoch: int, num_batches: int = 100) -> Dict[str, float]:
        """Train one epoch with gradient accumulation"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'category_loss': [],
            'tool_loss': [],
            'reward_loss': [],
            'semantic_loss': []
        }
        
        accumulation_steps = 2  # Accumulate gradients over 2 batches
        self.optimizer.zero_grad()
        
        for batch_idx in range(num_batches):
            # Generate training batch
            users, gifts, targets = self.generate_training_batch(
                batch_size=self.config.get('batch_size', 16),
                use_validation=False
            )
            
            # Forward pass
            batch_outputs = []
            for i, user in enumerate(users):
                env_state = self.env.reset(user)
                carry = self.model.initial_carry({
                    "inputs": torch.zeros(1, 10, device=self.device), 
                    "puzzle_identifiers": torch.zeros(1, 1, device=self.device)
                })
                
                carry, model_output, selected_tools = self.model.forward_with_enhancements(
                    carry, env_state, self.env.gift_catalog
                )
                batch_outputs.append(model_output)
            
            # Stack outputs
            stacked_outputs = {}
            for key in batch_outputs[0].keys():
                if isinstance(batch_outputs[0][key], torch.Tensor):
                    stacked_outputs[key] = torch.stack([output[key] for output in batch_outputs])
                else:
                    stacked_outputs[key] = [output[key] for output in batch_outputs]
            
            # Compute loss
            loss, loss_components = self.compute_enhanced_loss(stacked_outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Record losses (unscaled)
            for key, value in loss_components.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value * accumulation_steps)
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Calculate epoch averages
        epoch_metrics = {}
        for key, values in epoch_losses.items():
            if values:
                epoch_metrics[key] = np.mean(values)
        
        return epoch_metrics
    
    def train(self, num_epochs: int = 100, eval_frequency: int = 10):
        """Main training loop with early stopping and LR scheduling"""
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"📊 Early stopping patience: {self.early_stopping_patience} epochs")
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch, num_batches=50)
            
            # Log training metrics
            print(f"Training - Total Loss: {train_metrics.get('total_loss', 0):.4f}, "
                  f"Category Loss: {train_metrics.get('category_loss', 0):.4f}, "
                  f"Tool Loss: {train_metrics.get('tool_loss', 0):.4f}")
            
            # Evaluate periodically
            if (epoch + 1) % eval_frequency == 0:
                print("🔍 Evaluating model...")
                eval_metrics = self.evaluate_model(num_eval_episodes=30)
                
                print(f"Evaluation - Category Match: {eval_metrics['category_match_rate']:.1%}, "
                      f"Tool Match: {eval_metrics['tool_match_rate']:.1%}, "
                      f"Avg Reward: {eval_metrics['average_reward']:.3f}, "
                      f"Quality: {eval_metrics['recommendation_quality']:.3f}")
                
                # Update learning rate scheduler
                self.scheduler.step(train_metrics.get('total_loss', 0))
                
                # Check for improvement
                current_score = eval_metrics['recommendation_quality']
                if current_score > self.best_eval_score:
                    self.best_eval_score = current_score
                    self.patience_counter = 0
                    self.save_model(f"integrated_enhanced_best.pt", epoch, eval_metrics)
                    print(f"💾 New best model saved! Score: {current_score:.3f}")
                else:
                    self.patience_counter += 1
                    print(f"⏳ No improvement for {self.patience_counter} evaluation(s)")
                
                # Early stopping check
                if self.patience_counter >= self.early_stopping_patience // eval_frequency:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    print(f"🏆 Best score achieved: {self.best_eval_score:.3f}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % 25 == 0:
                self.save_model(f"integrated_enhanced_epoch_{epoch + 1}.pt", epoch, train_metrics)
        
        print(f"\n🎉 Training completed! Best score: {self.best_eval_score:.3f}")
    
    def save_model(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with comprehensive information"""
        os.makedirs("checkpoints/integrated_enhanced", exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_history': {
                'best_score': self.best_eval_score,
                'patience_counter': self.patience_counter,
                'training_scenarios': len(self.train_scenarios),
                'validation_scenarios': len(self.val_scenarios)
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'enhanced_components': [
                    'user_profiling', 'category_matching', 'tool_selection', 
                    'reward_prediction', 'cross_modal_fusion'
                ],
                'training_date': datetime.now().isoformat(),
                'optimization_version': 'v2.0'
            }
        }
        
        filepath = os.path.join("checkpoints/integrated_enhanced", filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")


def main():
    """Main training function"""
    print("🚀 INTEGRATED ENHANCED TRM TRAINING")
    print("=" * 60)
    
    # Create enhanced configuration
    config = create_integrated_enhanced_config()
    
    # Add training-specific parameters (optimized v3 - aggressive)
    config.update({
        'batch_size': 16,
        'num_epochs': 150,
        'eval_frequency': 5,  # More frequent evaluation
        'early_stopping_patience': 25,  # Even more patience
        'user_profile_lr': 5e-5,  # Further reduced
        'category_matching_lr': 4e-5,  # Much more reduced (main problem)
        'tool_selection_lr': 8e-5,  # Reduced
        'reward_prediction_lr': 1.5e-4,  # INCREASED (needs to learn faster)
        'main_lr': 5e-5,  # Further reduced
        'weight_decay': 0.025,  # Much stronger regularization
        'category_loss_weight': 0.15,  # Much more reduced (learning too fast)
        'tool_diversity_loss_weight': 0.30,  # Keep same
        'reward_loss_weight': 0.40,  # INCREASED (main problem - reward too low)
        'semantic_matching_loss_weight': 0.15,  # Slightly reduced
        'embedding_reg_weight': 3e-5  # Even stronger regularization
    })
    
    # Initialize trainer
    trainer = IntegratedEnhancedTrainer(config)
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        eval_frequency=config['eval_frequency']
    )
    
    print("🎉 Integrated Enhanced TRM training completed!")


if __name__ == "__main__":
    main()