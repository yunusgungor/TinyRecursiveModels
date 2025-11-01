#!/usr/bin/env python3
"""
Test script for RL and Tool integration
"""

import torch
import json
import os
import sys
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rl.rl_trm import RLEnhancedTRM
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.environment import (
    GiftRecommendationEnvironment, 
    UserProfile, 
    EnvironmentState,
    create_sample_gift_catalog
)
from models.rl.trainer import RLTrainer, TrainingConfig


def test_rl_model():
    """Test RL-enhanced TRM model"""
    print("="*50)
    print("TESTING RL-ENHANCED TRM")
    print("="*50)
    
    # Model configuration
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,  # No puzzle embedding for simplicity
        "puzzle_emb_len": 0,
        "hidden_size": 128,
        "H_cycles": 2,
        "L_cycles": 3,
        "H_layers": 2,
        "L_layers": 2,
        "num_heads": 4,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,
        "action_space_size": 20,
        "max_recommendations": 3,
        "value_head_hidden": 64,
        "policy_head_hidden": 64,
        "reward_prediction": True,
        "reward_head_hidden": 32,
        "forward_dtype": "float32"
    }
    
    # Create model
    model = RLEnhancedTRM(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test environment
    os.makedirs("data", exist_ok=True)
    create_sample_gift_catalog("data/test_catalog.json")
    env = GiftRecommendationEnvironment("data/test_catalog.json")
    
    # Test user profile
    user = UserProfile(
        age=35,
        hobbies=["gardening", "cooking"],
        relationship="mother",
        budget=100.0,
        occasion="birthday",
        personality_traits=["eco-conscious", "practical"]
    )
    
    # Reset environment
    state = env.reset(user)
    print(f"✓ Environment reset with {len(state.available_gifts)} available gifts")
    
    # Test model forward pass
    # Create proper batch for initial carry
    dummy_batch = {
        "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
    }
    carry = model.initial_carry(dummy_batch)
    
    with torch.no_grad():
        rl_output = model.forward_rl(carry, state, state.available_gifts)
        print(f"✓ Forward pass completed")
        print(f"  Action probabilities shape: {rl_output['action_probs'].shape}")
        print(f"  State value: {rl_output['state_value'].item():.3f}")
        
        # Test action selection
        action = model.select_action(rl_output["action_probs"], state.available_gifts)
        print(f"✓ Action selected: {len(action['recommendations'])} recommendations")
        print(f"  Recommendations: {action['recommendations']}")
        print(f"  Confidence scores: {[f'{s:.3f}' for s in action['confidence_scores']]}")
        
        # Test environment step
        next_state, reward, done, info = env.step({
            "recommendations": action["recommendations"],
            "confidence_scores": action["confidence_scores"]
        })
        print(f"✓ Environment step completed")
        print(f"  Reward: {reward:.3f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
    
    print("✓ RL model test completed successfully!\n")
    return True


def test_tool_enhanced_model():
    """Test Tool-enhanced TRM model"""
    print("="*50)
    print("TESTING TOOL-ENHANCED TRM")
    print("="*50)
    
    # Model configuration
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 128,
        "H_cycles": 2,
        "L_cycles": 3,
        "H_layers": 2,
        "L_layers": 2,
        "num_heads": 4,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,
        "action_space_size": 20,
        "max_recommendations": 3,
        "value_head_hidden": 64,
        "policy_head_hidden": 64,
        "reward_prediction": True,
        "reward_head_hidden": 32,
        # Tool parameters
        "max_tool_calls_per_step": 2,
        "tool_call_threshold": 0.3,
        "tool_result_encoding_dim": 64,
        "tool_selection_method": "confidence",
        "tool_fusion_method": "concatenate",
        "tool_usage_reward_weight": 0.1,
        "forward_dtype": "float32"
    }
    
    # Create model
    model = ToolEnhancedTRM(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Available tools: {model.tool_registry.list_tools()}")
    
    # Test environment
    env = GiftRecommendationEnvironment("data/test_catalog.json")
    
    # Test user profile
    user = UserProfile(
        age=45,
        hobbies=["gardening", "sustainability"],
        relationship="mother",
        budget=150.0,
        occasion="mothers_day",
        personality_traits=["eco-conscious", "quality-focused"]
    )
    
    # Reset environment
    state = env.reset(user)
    print(f"✓ Environment reset")
    
    # Test model forward pass with tools
    dummy_batch = {
        "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
    }
    carry = model.initial_carry(dummy_batch)
    
    with torch.no_grad():
        new_carry, rl_output, tool_calls = model.forward_with_tools(
            carry, state, state.available_gifts, max_tool_calls=2
        )
        
        print(f"✓ Forward pass with tools completed")
        print(f"  Tool calls made: {len(tool_calls)}")
        
        for i, call in enumerate(tool_calls):
            print(f"  Tool {i+1}: {call.tool_name}")
            print(f"    Success: {call.success}")
            print(f"    Execution time: {call.execution_time:.3f}s")
            if call.success and isinstance(call.result, dict):
                result_keys = list(call.result.keys())[:3]  # Show first 3 keys
                print(f"    Result keys: {result_keys}")
        
        # Test action selection
        action = model.select_action(rl_output["action_probs"], state.available_gifts)
        print(f"✓ Action selected with tool assistance")
        print(f"  Recommendations: {action['recommendations']}")
        
        # Test tool usage statistics
        tool_stats = model.get_tool_usage_stats()
        print(f"✓ Tool usage statistics:")
        print(f"  Total calls: {tool_stats.get('total_calls', 0)}")
        print(f"  Most used tool: {tool_stats.get('most_used_tool', 'None')}")
    
    print("✓ Tool-enhanced model test completed successfully!\n")
    return True


def test_training_integration():
    """Test training integration"""
    print("="*50)
    print("TESTING TRAINING INTEGRATION")
    print("="*50)
    
    # Small model for quick testing
    config = {
        "batch_size": 1,
        "seq_len": 30,
        "vocab_size": 500,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 64,
        "H_cycles": 1,
        "L_cycles": 2,
        "H_layers": 1,
        "L_layers": 1,
        "num_heads": 2,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.1,
        "action_space_size": 10,
        "max_recommendations": 2,
        "value_head_hidden": 32,
        "policy_head_hidden": 32,
        "reward_prediction": False,
        "forward_dtype": "float32"
    }
    
    # Training configuration
    training_config = TrainingConfig(
        num_episodes=5,  # Very small for testing
        max_steps_per_episode=3,
        batch_size=2,
        learning_rate=1e-3,
        eval_frequency=3,
        eval_episodes=2,
        log_frequency=1,
        save_frequency=10,
        checkpoint_dir="test_checkpoints",
        enable_tools=False
    )
    
    # Create components
    model = RLEnhancedTRM(config)
    env = GiftRecommendationEnvironment("data/test_catalog.json")
    trainer = RLTrainer(model, env, training_config)
    
    print(f"✓ Training setup completed")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training episodes: {training_config.num_episodes}")
    
    # Test experience collection
    experiences = trainer.collect_experience(2)
    print(f"✓ Experience collection test: {len(experiences)} experiences")
    
    # Test policy update
    if len(experiences) >= 2:
        loss_stats = trainer.update_policy(experiences)
        print(f"✓ Policy update test completed")
        print(f"  Policy loss: {loss_stats.get('policy_loss', 0):.4f}")
        print(f"  Value loss: {loss_stats.get('value_loss', 0):.4f}")
    
    # Test evaluation
    eval_stats = trainer.evaluate(num_episodes=2)
    print(f"✓ Evaluation test completed")
    print(f"  Eval reward: {eval_stats.get('eval_reward_mean', 0):.3f}")
    
    # Test checkpoint save/load
    os.makedirs("test_checkpoints", exist_ok=True)
    checkpoint_path = trainer.save_checkpoint("test_checkpoints/test_checkpoint.pt")
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Create new trainer and load checkpoint
    new_trainer = RLTrainer(model, env, training_config)
    new_trainer.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    
    print("✓ Training integration test completed successfully!\n")
    return True


def test_data_generation():
    """Test data generation utilities"""
    print("="*50)
    print("TESTING DATA GENERATION")
    print("="*50)
    
    from utils.data_generator import GiftDataGenerator
    
    generator = GiftDataGenerator()
    
    # Test user profile generation
    user_profile = generator.generate_user_profile()
    print(f"✓ User profile generated:")
    print(f"  Age: {user_profile['age']}")
    print(f"  Hobbies: {user_profile['hobbies']}")
    print(f"  Budget: ${user_profile['budget']}")
    print(f"  Occasion: {user_profile['occasion']}")
    
    # Test gift item generation
    gift = generator.generate_gift_item("gardening")
    print(f"✓ Gift item generated:")
    print(f"  Name: {gift['name']}")
    print(f"  Category: {gift['category']}")
    print(f"  Price: ${gift['price']}")
    print(f"  Rating: {gift['rating']}")
    
    # Test training example generation
    example = generator.generate_training_example()
    print(f"✓ Training example generated:")
    print(f"  Candidate gifts: {len(example['candidate_gifts'])}")
    print(f"  Recommendations: {len(example['recommendations'])}")
    print(f"  Top score: {example['recommendations'][0]['score']:.3f}")
    
    # Test small dataset generation
    output_dir = generator.generate_training_dataset(
        num_examples=10,
        output_dir="test_data"
    )
    print(f"✓ Small dataset generated in: {output_dir}")
    
    print("✓ Data generation test completed successfully!\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("STARTING COMPREHENSIVE TESTS")
    print("="*60)
    
    tests = [
        ("RL Model", test_rl_model),
        ("Tool-Enhanced Model", test_tool_enhanced_model),
        ("Training Integration", test_training_integration),
        ("Data Generation", test_data_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "ERROR"
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RL and Tool integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RL and Tool integration")
    parser.add_argument("--test", type=str, choices=["all", "rl", "tools", "training", "data"],
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "rl":
        success = test_rl_model()
    elif args.test == "tools":
        success = test_tool_enhanced_model()
    elif args.test == "training":
        success = test_training_integration()
    elif args.test == "data":
        success = test_data_generation()
    
    sys.exit(0 if success else 1)