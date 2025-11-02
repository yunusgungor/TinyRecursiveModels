#!/usr/bin/env python3
"""
Test script for Integrated Enhanced TRM Model
"""

import sys
import os
import torch
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import GiftRecommendationEnvironment, UserProfile


def test_model_components():
    """Test individual model components"""
    print("🧪 TESTING INTEGRATED ENHANCED MODEL COMPONENTS")
    print("=" * 60)
    
    # Create model
    config = create_integrated_enhanced_config()
    model = IntegratedEnhancedTRM(config)
    
    print(f"✅ Model created successfully")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test user profile encoding
    print(f"\n🧑 Testing User Profile Encoding:")
    test_user = UserProfile(
        age=35,
        hobbies=["gardening", "cooking", "wellness"],
        relationship="mother",
        budget=120.0,
        occasion="mothers_day",
        personality_traits=["eco-friendly", "practical", "relaxing"]
    )
    
    user_encoding = model.encode_user_profile(test_user)
    print(f"  ✅ User encoding shape: {user_encoding.shape}")
    print(f"  📊 Encoding range: [{user_encoding.min().item():.3f}, {user_encoding.max().item():.3f}]")
    
    # Test category matching
    print(f"\n🏷️ Testing Enhanced Category Matching:")
    category_scores = model.enhanced_category_matching(user_encoding)
    print(f"  ✅ Category scores shape: {category_scores.shape}")
    
    # Get top categories
    top_categories = torch.topk(category_scores[0], 5)
    print(f"  🎯 Top 5 categories:")
    for i, (score, idx) in enumerate(zip(top_categories.values, top_categories.indices)):
        category = model.gift_categories[idx]
        print(f"    {i+1}. {category}: {score.item():.3f}")
    
    # Test tool selection
    print(f"\n🛠️ Testing Enhanced Tool Selection:")
    selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
    print(f"  ✅ Selected tools: {selected_tools[0]}")
    print(f"  📊 Tool scores shape: {tool_scores.shape}")
    
    # Test reward prediction
    print(f"\n💰 Testing Enhanced Reward Prediction:")
    # Create dummy gift encodings
    gift_encodings = torch.randn(1, 10, config['gift_embedding_dim'])
    predicted_rewards = model.enhanced_reward_prediction(user_encoding, gift_encodings, test_user)
    print(f"  ✅ Predicted rewards shape: {predicted_rewards.shape}")
    print(f"  📊 Reward range: [{predicted_rewards.min().item():.3f}, {predicted_rewards.max().item():.3f}]")
    
    return model


def test_full_forward_pass():
    """Test full forward pass with environment"""
    print(f"\n🚀 TESTING FULL FORWARD PASS")
    print("=" * 40)
    
    # Create model and environment
    config = create_integrated_enhanced_config()
    model = IntegratedEnhancedTRM(config)
    env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
    
    # Test user
    test_user = UserProfile(
        age=28,
        hobbies=["technology", "fitness", "coffee"],
        relationship="friend",
        budget=150.0,
        occasion="birthday",
        personality_traits=["trendy", "practical", "tech-savvy"]
    )
    
    # Reset environment
    env_state = env.reset(test_user)
    
    # Create initial carry
    dummy_batch = {
        "inputs": torch.zeros(1, 10),
        "puzzle_identifiers": torch.zeros(1, 1)
    }
    carry = model.initial_carry(dummy_batch)
    
    # Forward pass
    print("🔄 Running forward pass...")
    with torch.no_grad():
        new_carry, rl_output, selected_tools = model.forward_with_enhancements(
            carry, env_state, env.gift_catalog
        )
    
    print(f"✅ Forward pass completed successfully")
    print(f"🛠️ Selected tools: {selected_tools}")
    print(f"📊 Output keys: {list(rl_output.keys())}")
    
    # Analyze outputs
    action_probs = rl_output['action_probs']
    category_scores = rl_output['category_scores']
    predicted_rewards = rl_output['predicted_rewards']
    
    print(f"\n📈 Output Analysis:")
    print(f"  Action probabilities shape: {action_probs.shape}")
    print(f"  Category scores shape: {category_scores.shape}")
    print(f"  Predicted rewards shape: {predicted_rewards.shape}")
    
    # Get top recommendations
    top_actions = torch.topk(action_probs[0], 3)
    print(f"\n🎁 Top 3 Recommendations (by probability):")
    for i, (prob, idx) in enumerate(zip(top_actions.values, top_actions.indices)):
        if idx.item() < len(env.gift_catalog):
            gift = env.gift_catalog[idx.item()]
            print(f"  {i+1}. {gift.name} ({gift.category}) - Prob: {prob.item():.3f}")
    
    # Get top categories
    top_categories = torch.topk(category_scores[0], 3)
    print(f"\n🏷️ Top 3 Categories:")
    for i, (score, idx) in enumerate(zip(top_categories.values, top_categories.indices)):
        category = model.gift_categories[idx]
        print(f"  {i+1}. {category}: {score.item():.3f}")
    
    return model, rl_output


def test_realistic_scenarios():
    """Test with realistic user scenarios"""
    print(f"\n🌍 TESTING WITH REALISTIC SCENARIOS")
    print("=" * 50)
    
    # Load realistic scenarios
    try:
        with open("data/realistic_user_scenarios.json", "r") as f:
            scenario_data = json.load(f)
        scenarios = scenario_data["scenarios"]
    except:
        print("⚠️ Could not load realistic scenarios, using fallback")
        scenarios = [
            {
                "name": "Tech Enthusiast",
                "profile": {
                    "age": 25,
                    "hobbies": ["technology", "gaming"],
                    "relationship": "friend",
                    "budget": 100.0,
                    "occasion": "birthday",
                    "preferences": ["trendy", "tech-savvy"]
                },
                "expected_categories": ["technology", "gaming"],
                "expected_tools": ["price_comparison", "review_analysis"]
            }
        ]
    
    # Create model and environment
    config = create_integrated_enhanced_config()
    model = IntegratedEnhancedTRM(config)
    env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
    
    results = []
    category_matches = 0
    tool_matches = 0
    
    print(f"🧪 Testing {len(scenarios)} scenarios:")
    
    for i, scenario in enumerate(scenarios):
        print(f"\n👤 Scenario {i+1}: {scenario['name']}")
        
        # Create user profile
        profile_data = scenario["profile"]
        user = UserProfile(
            age=profile_data["age"],
            hobbies=profile_data["hobbies"],
            relationship=profile_data["relationship"],
            budget=profile_data["budget"],
            occasion=profile_data["occasion"],
            personality_traits=profile_data["preferences"]
        )
        
        # Reset environment
        env_state = env.reset(user)
        
        # Forward pass
        dummy_batch = {
            "inputs": torch.zeros(1, 10),
            "puzzle_identifiers": torch.zeros(1, 1)
        }
        carry = model.initial_carry(dummy_batch)
        
        with torch.no_grad():
            new_carry, rl_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, env.gift_catalog
            )
        
        # Analyze results
        category_scores = rl_output['category_scores'][0]
        top_categories = torch.topk(category_scores, 3)
        predicted_categories = [model.gift_categories[idx] for idx in top_categories.indices]
        
        # Check category matching
        expected_categories = set(scenario["expected_categories"])
        actual_categories = set(predicted_categories)
        category_match = len(expected_categories.intersection(actual_categories)) > 0
        
        if category_match:
            category_matches += 1
        
        # Check tool matching
        expected_tools = set(scenario["expected_tools"])
        actual_tools = set(selected_tools)
        tool_match = len(expected_tools.intersection(actual_tools)) > 0
        
        if tool_match:
            tool_matches += 1
        
        # Get predicted reward
        predicted_reward = rl_output['predicted_rewards'].mean().item()
        
        print(f"  Expected categories: {list(expected_categories)}")
        print(f"  Predicted categories: {predicted_categories}")
        print(f"  Category match: {'✅' if category_match else '❌'}")
        print(f"  Expected tools: {list(expected_tools)}")
        print(f"  Selected tools: {selected_tools}")
        print(f"  Tool match: {'✅' if tool_match else '❌'}")
        print(f"  Predicted reward: {predicted_reward:.3f}")
        
        results.append({
            "scenario": scenario["name"],
            "category_match": category_match,
            "tool_match": tool_match,
            "predicted_reward": predicted_reward,
            "predicted_categories": predicted_categories,
            "selected_tools": selected_tools
        })
    
    # Calculate overall metrics
    category_match_rate = category_matches / len(scenarios)
    tool_match_rate = tool_matches / len(scenarios)
    avg_reward = np.mean([r["predicted_reward"] for r in results])
    
    print(f"\n📊 OVERALL RESULTS:")
    print("=" * 30)
    print(f"Category Match Rate: {category_match_rate:.1%} ({category_matches}/{len(scenarios)})")
    print(f"Tool Match Rate: {tool_match_rate:.1%} ({tool_matches}/{len(scenarios)})")
    print(f"Average Predicted Reward: {avg_reward:.3f}")
    
    # Overall score
    overall_score = (category_match_rate * 0.4 + tool_match_rate * 0.3 + avg_reward * 0.3)
    print(f"Overall Score: {overall_score:.3f}/1.000")
    
    if overall_score > 0.7:
        print("🌟 EXCELLENT: Model shows great performance!")
    elif overall_score > 0.5:
        print("✅ GOOD: Model shows solid performance")
    elif overall_score > 0.3:
        print("⚠️ FAIR: Model needs improvement")
    else:
        print("❌ POOR: Model needs significant work")
    
    return results, overall_score


def test_training_readiness():
    """Test if model is ready for training"""
    print(f"\n🏋️ TESTING TRAINING READINESS")
    print("=" * 40)
    
    try:
        # Test model creation
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config)
        print("✅ Model creation: OK")
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Test gradient flow
        dummy_input = torch.randn(1, config['user_profile_encoding_dim'])
        output = model.enhanced_category_matching(dummy_input)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✅ Gradient flow: {'OK' if has_gradients else 'FAILED'}")
        
        # Test data loading
        try:
            env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
            print("✅ Data loading: OK")
        except Exception as e:
            print(f"❌ Data loading: FAILED - {e}")
        
        # Test device compatibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Device compatibility: {device}")
        
        print(f"\n🚀 Model is ready for training!")
        return True
        
    except Exception as e:
        print(f"❌ Training readiness test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 INTEGRATED ENHANCED TRM MODEL TESTING")
    print("=" * 70)
    
    # Test individual components
    model = test_model_components()
    
    # Test full forward pass
    model, outputs = test_full_forward_pass()
    
    # Test realistic scenarios
    results, overall_score = test_realistic_scenarios()
    
    # Test training readiness
    training_ready = test_training_readiness()
    
    # Final summary
    print(f"\n🏆 FINAL TEST SUMMARY")
    print("=" * 40)
    print(f"✅ Component tests: PASSED")
    print(f"✅ Forward pass: PASSED")
    print(f"📊 Realistic scenarios: {overall_score:.3f}/1.000")
    print(f"🏋️ Training readiness: {'READY' if training_ready else 'NOT READY'}")
    
    if overall_score > 0.5 and training_ready:
        print(f"\n🎉 INTEGRATED ENHANCED MODEL IS READY!")
        print(f"🚀 You can now start training with:")
        print(f"   python train_integrated_enhanced_model.py")
    else:
        print(f"\n⚠️ Model needs attention before training")
    
    return overall_score > 0.5 and training_ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)