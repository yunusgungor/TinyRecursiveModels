#!/usr/bin/env python3
"""
Quick training test for Integrated Enhanced TRM Model
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_integrated_enhanced_model import IntegratedEnhancedTrainer, create_integrated_enhanced_config


def quick_training_test():
    """Run a quick training test to validate the training pipeline"""
    print("🚀 QUICK TRAINING TEST FOR INTEGRATED ENHANCED MODEL")
    print("=" * 70)
    
    # Create configuration for quick test
    config = create_integrated_enhanced_config()
    config.update({
        'batch_size': 4,  # Small batch for quick test
        'num_epochs': 5,  # Just 5 epochs for test
        'eval_frequency': 2,
        'user_profile_lr': 1e-3,  # Higher learning rate for quick test
        'category_matching_lr': 2e-3,
        'tool_selection_lr': 1.5e-3,
        'reward_prediction_lr': 1e-3,
        'main_lr': 5e-4,
        'weight_decay': 0.01,
        'category_loss_weight': 0.35,
        'tool_diversity_loss_weight': 0.15,
        'reward_loss_weight': 0.25,
        'semantic_matching_loss_weight': 0.20,
        'embedding_reg_weight': 1e-5
    })
    
    print(f"📋 Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize trainer
    print(f"\n🔧 Initializing trainer...")
    trainer = IntegratedEnhancedTrainer(config)
    
    # Test batch generation
    print(f"\n📦 Testing batch generation...")
    users, gifts, targets = trainer.generate_training_batch(batch_size=2)
    print(f"  ✅ Generated batch with {len(users)} users")
    print(f"  👤 Sample user: {users[0].age}y, hobbies: {users[0].hobbies}")
    
    # Test loss calculation
    print(f"\n💰 Testing loss calculation...")
    try:
        # Generate dummy model outputs
        batch_size = len(users)
        dummy_outputs = {
            'category_scores': torch.rand(batch_size, len(trainer.model.gift_categories)),
            'tool_scores': torch.rand(batch_size, len(trainer.model.tool_registry.list_tools())),
            'predicted_rewards': torch.rand(batch_size, 10),
            'action_probs': torch.rand(batch_size, config['action_space_size'])
        }
        
        loss, loss_components = trainer.compute_enhanced_loss(dummy_outputs, targets)
        print(f"  ✅ Loss calculation successful")
        print(f"  📊 Total loss: {loss.item():.4f}")
        print(f"  🔍 Components: {list(loss_components.keys())}")
        
    except Exception as e:
        print(f"  ❌ Loss calculation failed: {e}")
        return False
    
    # Test evaluation
    print(f"\n🔍 Testing evaluation...")
    try:
        eval_metrics = trainer.evaluate_model(num_eval_episodes=3)
        print(f"  ✅ Evaluation successful")
        print(f"  📊 Category match rate: {eval_metrics['category_match_rate']:.1%}")
        print(f"  🛠️ Tool match rate: {eval_metrics['tool_match_rate']:.1%}")
        print(f"  💰 Average reward: {eval_metrics['average_reward']:.3f}")
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        return False
    
    # Run quick training
    print(f"\n🏋️ Running quick training test...")
    try:
        # Train for just 2 epochs with few batches
        for epoch in range(2):
            print(f"\n📚 Quick Epoch {epoch + 1}/2")
            train_metrics = trainer.train_epoch(epoch, num_batches=3)
            
            print(f"  Training - Total Loss: {train_metrics.get('total_loss', 0):.4f}")
            print(f"  Category Loss: {train_metrics.get('category_loss', 0):.4f}")
            print(f"  Tool Loss: {train_metrics.get('tool_loss', 0):.4f}")
        
        print(f"  ✅ Quick training completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return False
    
    # Final evaluation
    print(f"\n🎯 Final evaluation after quick training...")
    try:
        final_metrics = trainer.evaluate_model(num_eval_episodes=5)
        print(f"  📊 Final Results:")
        print(f"    Category match rate: {final_metrics['category_match_rate']:.1%}")
        print(f"    Tool match rate: {final_metrics['tool_match_rate']:.1%}")
        print(f"    Average reward: {final_metrics['average_reward']:.3f}")
        print(f"    Overall quality: {final_metrics['recommendation_quality']:.3f}")
        
    except Exception as e:
        print(f"  ❌ Final evaluation failed: {e}")
        return False
    
    print(f"\n🎉 QUICK TRAINING TEST COMPLETED SUCCESSFULLY!")
    print(f"✅ All components working correctly")
    print(f"🚀 Ready for full training!")
    
    return True


def main():
    """Main function"""
    success = quick_training_test()
    
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. 🚀 Start full training:")
        print(f"   python train_integrated_enhanced_model.py")
        print(f"2. 📊 Monitor training progress")
        print(f"3. 🔍 Evaluate trained model")
        print(f"4. 🎉 Deploy enhanced model!")
    else:
        print(f"\n⚠️ Training test failed - check errors above")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)