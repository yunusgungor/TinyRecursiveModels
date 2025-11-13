#!/usr/bin/env python3
"""
Fine-tune script to improve category diversity
Loads best model and trains with category diversity loss
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile
from train_integrated_enhanced_model import IntegratedEnhancedTrainer


class CategoryDiversityFineTuner:
    """Fine-tune model to improve category diversity"""
    
    def __init__(self, checkpoint_path: str):
        print("🔧 CATEGORY DIVERSITY FINE-TUNING")
        print("="*60)
        
        # Load checkpoint
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.config = checkpoint['config']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = IntegratedEnhancedTRM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        print(f"📊 Original metrics: {checkpoint['metrics']}")
        
        # Initialize environment
        self.env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
        
        # Load scenarios
        self._load_scenarios()
        
        # Create optimizer with very low learning rate (fine-tuning)
        self.optimizer = self._create_optimizer()
        
        # Loss functions
        self.category_criterion = nn.BCELoss()
        
        print(f"🚀 Fine-tuner initialized")
        print(f"📱 Device: {self.device}")
        print(f"📊 Training scenarios: {len(self.train_scenarios)}")
        
    def _load_scenarios(self):
        """Load and split scenarios"""
        import json
        
        try:
            with open("data/expanded_user_scenarios.json", "r") as f:
                scenario_data = json.load(f)
            all_scenarios = scenario_data["scenarios"]
        except:
            with open("data/realistic_user_scenarios.json", "r") as f:
                scenario_data = json.load(f)
            all_scenarios = scenario_data["scenarios"]
        
        # Use same split as training
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(all_scenarios)
        split_idx = int(len(all_scenarios) * 0.8)
        self.train_scenarios = all_scenarios[:split_idx]
        self.val_scenarios = all_scenarios[split_idx:]
        
        print(f"📊 Loaded {len(all_scenarios)} scenarios: {len(self.train_scenarios)} train, {len(self.val_scenarios)} val")
    
    def _create_optimizer(self):
        """Create optimizer with very low LR for fine-tuning"""
        # Only optimize category-related parameters
        category_params = (
            list(self.model.category_embeddings.parameters()) +
            [p for layer in self.model.semantic_matcher for p in layer.parameters()] +
            list(self.model.category_attention.parameters()) +
            list(self.model.category_scorer.parameters())
        )
        
        return optim.AdamW(category_params, lr=1e-5, weight_decay=0.01)
    
    def compute_category_diversity_loss(self, category_scores: torch.Tensor, 
                                       targets: List[Dict]) -> torch.Tensor:
        """Compute loss that encourages diverse category predictions"""
        device = self.device
        batch_size = category_scores.size(0)
        
        # 1. Standard category matching loss with label smoothing
        category_targets = []
        label_smoothing = 0.15  # Increased smoothing
        
        for target in targets:
            expected_cats = target['expected_categories']
            # Create soft target with more smoothing
            target_vector = torch.full((len(self.model.gift_categories),), 
                                      label_smoothing / len(self.model.gift_categories), 
                                      device=device)
            for cat in expected_cats:
                if cat in self.model.gift_categories:
                    idx = self.model.gift_categories.index(cat)
                    target_vector[idx] = 1.0 - label_smoothing + (label_smoothing / len(self.model.gift_categories))
            category_targets.append(target_vector)
        
        category_target_tensor = torch.stack(category_targets)
        if category_scores.dim() == 3:
            category_scores = category_scores.squeeze(1)
        
        matching_loss = self.category_criterion(category_scores, category_target_tensor)
        
        # 2. Diversity loss - encourage using different categories
        # Penalize if model always predicts same categories
        avg_category_scores = category_scores.mean(dim=0)  # Average across batch
        
        # We want a more uniform distribution across categories
        # Entropy-based diversity: higher entropy = more diverse
        epsilon = 1e-8
        entropy = -(avg_category_scores * torch.log(avg_category_scores + epsilon)).sum()
        max_entropy = np.log(len(self.model.gift_categories))
        
        # Diversity loss: penalize low entropy (low diversity)
        diversity_loss = (max_entropy - entropy) / max_entropy
        
        # 3. Top-k diversity: ensure top predictions vary
        top_k = 5
        top_indices = torch.topk(category_scores, top_k, dim=1).indices
        
        # Count how many times each category appears in top-k
        category_counts = torch.zeros(len(self.model.gift_categories), device=device)
        for batch_idx in range(batch_size):
            for idx in top_indices[batch_idx]:
                category_counts[idx] += 1
        
        # Penalize if some categories never appear in top-k
        min_count = category_counts.min()
        max_count = category_counts.max()
        topk_diversity_loss = (max_count - min_count) / (batch_size * top_k)
        
        # Combined loss
        total_loss = (
            0.6 * matching_loss +      # Main task
            0.25 * diversity_loss +     # Encourage diversity
            0.15 * topk_diversity_loss  # Ensure all categories used
        )
        
        return total_loss, {
            'matching_loss': matching_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'topk_diversity_loss': topk_diversity_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def generate_batch(self, batch_size: int = 16, use_validation: bool = False):
        """Generate training batch"""
        scenarios = self.val_scenarios if use_validation else self.train_scenarios
        
        batch_users = []
        batch_targets = []
        
        for _ in range(batch_size):
            scenario = np.random.choice(scenarios)
            
            # Data augmentation
            augmented_age = scenario["profile"]["age"] + np.random.randint(-5, 6)
            augmented_age = max(18, min(75, augmented_age))
            
            augmented_budget = scenario["profile"]["budget"] * np.random.uniform(0.8, 1.2)
            augmented_budget = max(30.0, min(300.0, augmented_budget))
            
            hobbies = scenario["profile"]["hobbies"].copy()
            np.random.shuffle(hobbies)
            
            user = UserProfile(
                age=int(augmented_age),
                hobbies=hobbies,
                relationship=scenario["profile"]["relationship"],
                budget=float(augmented_budget),
                occasion=scenario["profile"]["occasion"],
                personality_traits=scenario["profile"]["preferences"]
            )
            
            target = {
                'expected_categories': scenario["expected_categories"],
                'expected_tools': scenario["expected_tools"],
                'user_profile': user
            }
            
            batch_users.append(user)
            batch_targets.append(target)
        
        return batch_users, batch_targets
    
    def finetune_epoch(self, epoch: int, num_batches: int = 50):
        """Fine-tune one epoch"""
        self.model.train()
        
        epoch_losses = []
        
        for batch_idx in range(num_batches):
            users, targets = self.generate_batch(batch_size=16)
            
            # Forward pass
            batch_category_scores = []
            for user in users:
                env_state = self.env.reset(user)
                
                # Encode user and get category scores
                user_encoding = self.model.encode_user_profile(env_state.user_profile)
                category_scores = self.model.enhanced_category_matching(user_encoding)
                batch_category_scores.append(category_scores)
            
            stacked_scores = torch.cat(batch_category_scores, dim=0)
            
            # Compute diversity loss
            loss, loss_components = self.compute_category_diversity_loss(stacked_scores, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            epoch_losses.append(loss_components)
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Calculate averages
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        return avg_losses
    
    def evaluate(self, num_samples: int = 50):
        """Evaluate category diversity"""
        self.model.eval()
        
        category_predictions = []
        category_matches = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                users, targets = self.generate_batch(batch_size=1, use_validation=True)
                user = users[0]
                target = targets[0]
                
                env_state = self.env.reset(user)
                user_encoding = self.model.encode_user_profile(env_state.user_profile)
                category_scores = self.model.enhanced_category_matching(user_encoding)
                
                # Get top 3 categories
                top_indices = torch.topk(category_scores[0], 3).indices
                predicted_cats = [self.model.gift_categories[idx] for idx in top_indices]
                category_predictions.extend(predicted_cats)
                
                # Check match
                expected_cats = set(target['expected_categories'])
                actual_cats = set(predicted_cats)
                if len(expected_cats.intersection(actual_cats)) > 0:
                    category_matches += 1
        
        # Calculate diversity metrics
        unique_categories = len(set(category_predictions))
        category_match_rate = category_matches / num_samples
        
        # Category distribution
        from collections import Counter
        category_dist = Counter(category_predictions)
        
        return {
            'category_match_rate': category_match_rate,
            'unique_categories_used': unique_categories,
            'total_categories': len(self.model.gift_categories),
            'category_distribution': dict(category_dist.most_common(10))
        }
    
    def finetune(self, num_epochs: int = 10):
        """Main fine-tuning loop"""
        print(f"\n🚀 Starting fine-tuning for {num_epochs} epochs")
        print("🎯 Goal: Improve category diversity\n")
        
        # Initial evaluation
        print("📊 Initial evaluation:")
        initial_metrics = self.evaluate()
        print(f"  Category Match: {initial_metrics['category_match_rate']:.1%}")
        print(f"  Unique Categories: {initial_metrics['unique_categories_used']}/{initial_metrics['total_categories']}")
        print(f"  Top Categories: {list(initial_metrics['category_distribution'].keys())[:5]}\n")
        
        best_match_rate = initial_metrics['category_match_rate']
        
        for epoch in range(num_epochs):
            print(f"📚 Epoch {epoch + 1}/{num_epochs}")
            
            # Fine-tune
            train_losses = self.finetune_epoch(epoch, num_batches=50)
            print(f"Training - Total: {train_losses['total_loss']:.4f}, "
                  f"Matching: {train_losses['matching_loss']:.4f}, "
                  f"Diversity: {train_losses['diversity_loss']:.4f}")
            
            # Evaluate every 2 epochs
            if (epoch + 1) % 2 == 0:
                print("🔍 Evaluating...")
                eval_metrics = self.evaluate()
                print(f"Evaluation - Category Match: {eval_metrics['category_match_rate']:.1%}, "
                      f"Unique Categories: {eval_metrics['unique_categories_used']}/{eval_metrics['total_categories']}")
                
                # Save if improved
                if eval_metrics['category_match_rate'] > best_match_rate:
                    best_match_rate = eval_metrics['category_match_rate']
                    self.save_model(f"finetuned_best.pt", epoch, eval_metrics)
                    print(f"💾 New best model saved! Match rate: {best_match_rate:.1%}\n")
        
        # Final evaluation
        print("\n📊 Final evaluation:")
        final_metrics = self.evaluate(num_samples=100)
        print(f"  Category Match: {final_metrics['category_match_rate']:.1%}")
        print(f"  Unique Categories: {final_metrics['unique_categories_used']}/{final_metrics['total_categories']}")
        print(f"  Category Distribution:")
        for cat, count in list(final_metrics['category_distribution'].items())[:10]:
            print(f"    {cat}: {count}")
        
        print(f"\n🎉 Fine-tuning completed!")
        print(f"📈 Improvement: {initial_metrics['category_match_rate']:.1%} → {final_metrics['category_match_rate']:.1%}")
    
    def save_model(self, filename: str, epoch: int, metrics: Dict):
        """Save fine-tuned model"""
        os.makedirs("checkpoints/finetuned", exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'finetuning_info': {
                'method': 'category_diversity',
                'base_model': 'integrated_enhanced_best.pt',
                'improvements': 'Added diversity loss and label smoothing'
            }
        }
        
        filepath = os.path.join("checkpoints/finetuned", filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")


def main():
    """Main fine-tuning function"""
    checkpoint_path = "checkpoints/integrated_enhanced/integrated_enhanced_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Initialize fine-tuner
    finetuner = CategoryDiversityFineTuner(checkpoint_path)
    
    # Fine-tune
    finetuner.finetune(num_epochs=10)
    
    print("\n✅ Fine-tuning completed!")
    print("📂 Fine-tuned model saved to: checkpoints/finetuned/finetuned_best.pt")
    print("\n🧪 To test the fine-tuned model, update test_best_model.py to use:")
    print("   checkpoint_path = 'checkpoints/finetuned/finetuned_best.pt'")


if __name__ == "__main__":
    main()
