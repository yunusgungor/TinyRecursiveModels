#!/usr/bin/env python3
"""
Start full training for Integrated Enhanced TRM Model
"""

import sys
import os
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_integrated_enhanced_model import IntegratedEnhancedTrainer, create_integrated_enhanced_config


def start_full_training():
    """Start full training with optimized configuration"""
    print("🚀 STARTING FULL TRAINING - INTEGRATED ENHANCED TRM")
    print("=" * 70)
    
    # Create optimized configuration
    config = create_integrated_enhanced_config()
    config.update({
        # Training parameters
        'batch_size': 8,  # Reasonable batch size
        'num_epochs': 50,  # Sufficient epochs for convergence
        'eval_frequency': 5,  # Evaluate every 5 epochs
        
        # Learning rates (optimized from quick test)
        'user_profile_lr': 5e-4,
        'category_matching_lr': 1e-3,  # Higher for category matching
        'tool_selection_lr': 8e-4,
        'reward_prediction_lr': 5e-4,
        'main_lr': 3e-4,
        'weight_decay': 0.01,
        
        # Loss weights (optimized)
        'category_loss_weight': 0.40,  # Increased for better category matching
        'tool_diversity_loss_weight': 0.20,  # Increased for better tool usage
        'reward_loss_weight': 0.20,
        'semantic_matching_loss_weight': 0.15,
        'embedding_reg_weight': 5e-6,
    })
    
    print(f"📋 Training Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Category loss weight: {config['category_loss_weight']}")
    print(f"  Tool diversity weight: {config['tool_diversity_loss_weight']}")
    
    # Initialize trainer
    print(f"\n🔧 Initializing trainer...")
    trainer = IntegratedEnhancedTrainer(config)
    
    # Pre-training evaluation
    print(f"\n🔍 Pre-training evaluation...")
    pre_metrics = trainer.evaluate_model(num_eval_episodes=10)
    print(f"  📊 Pre-training Results:")
    print(f"    Category match rate: {pre_metrics['category_match_rate']:.1%}")
    print(f"    Tool match rate: {pre_metrics['tool_match_rate']:.1%}")
    print(f"    Average reward: {pre_metrics['average_reward']:.3f}")
    print(f"    Overall quality: {pre_metrics['recommendation_quality']:.3f}")
    
    # Start training
    print(f"\n🏋️ Starting full training...")
    print(f"⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer.train(
            num_epochs=config['num_epochs'],
            eval_frequency=config['eval_frequency']
        )
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        
        # Final evaluation
        print(f"\n🎯 Final evaluation...")
        final_metrics = trainer.evaluate_model(num_eval_episodes=20)
        print(f"  📊 Final Results:")
        print(f"    Category match rate: {final_metrics['category_match_rate']:.1%}")
        print(f"    Tool match rate: {final_metrics['tool_match_rate']:.1%}")
        print(f"    Average reward: {final_metrics['average_reward']:.3f}")
        print(f"    Overall quality: {final_metrics['recommendation_quality']:.3f}")
        
        # Calculate improvement
        category_improvement = final_metrics['category_match_rate'] - pre_metrics['category_match_rate']
        tool_improvement = final_metrics['tool_match_rate'] - pre_metrics['tool_match_rate']
        reward_improvement = final_metrics['average_reward'] - pre_metrics['average_reward']
        
        print(f"\n📈 IMPROVEMENTS:")
        print(f"  Category matching: {category_improvement:+.1%}")
        print(f"  Tool matching: {tool_improvement:+.1%}")
        print(f"  Average reward: {reward_improvement:+.3f}")
        
        # Success assessment
        if final_metrics['recommendation_quality'] > 0.7:
            print(f"\n🌟 EXCELLENT: Model achieved excellent performance!")
        elif final_metrics['recommendation_quality'] > 0.5:
            print(f"\n✅ GOOD: Model achieved good performance!")
        else:
            print(f"\n⚠️ FAIR: Model shows improvement but could be better")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def main():
    """Main function"""
    success = start_full_training()
    
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. 🧪 Test the trained model:")
        print(f"   python test_trained_integrated_model.py")
        print(f"2. 📊 Compare with original performance")
        print(f"3. 🚀 Deploy to production!")
    else:
        print(f"\n⚠️ Training was not completed successfully")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)