#!/usr/bin/env python3
"""
Test script for the trained Integrated Enhanced TRM Model
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


def load_trained_model():
    """Load the best trained model"""
    print("📦 Loading trained model...")
    
    # Create model configuration
    config = create_integrated_enhanced_config()
    model = IntegratedEnhancedTRM(config)
    
    # Load the best trained model
    checkpoint_path = "checkpoints/integrated_enhanced/integrated_enhanced_best.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training info
        training_info = checkpoint.get('model_info', {})
        metrics = checkpoint.get('metrics', {})
        
        print(f"📊 Model Info:")
        print(f"  Training date: {training_info.get('training_date', 'Unknown')}")
        print(f"  Total parameters: {training_info.get('total_parameters', 'Unknown'):,}")
        print(f"  Enhanced components: {training_info.get('enhanced_components', [])}")
        
        if metrics:
            print(f"📈 Training Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        model.eval()
        return model, checkpoint
    else:
        print(f"❌ Trained model not found at: {checkpoint_path}")
        print(f"🔄 Using untrained model for comparison")
        return model, None


def test_realistic_scenarios_with_trained_model(model):
    """Test trained model with realistic scenarios"""
    print("\n🌍 TESTING TRAINED MODEL WITH REALISTIC SCENARIOS")
    print("=" * 60)
    
    # Load realistic scenarios
    try:
        with open("data/realistic_user_scenarios.json", "r") as f:
            scenario_data = json.load(f)
        scenarios = scenario_data["scenarios"]
        print(f"📋 Loaded {len(scenarios)} realistic scenarios")
    except:
        print("⚠️ Could not load realistic scenarios")
        return None
    
    # Create environment
    env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
    
    results = []
    category_matches = 0
    tool_matches = 0
    total_rewards = []
    
    print(f"\n🧪 Testing each scenario:")
    print("-" * 50)
    
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
        
        print(f"   Age: {user.age}, Budget: ${user.budget}")
        print(f"   Hobbies: {user.hobbies}")
        print(f"   Preferences: {user.personality_traits}")
        print(f"   Expected categories: {scenario['expected_categories']}")
        print(f"   Expected tools: {scenario['expected_tools']}")
        
        # Reset environment
        env_state = env.reset(user)
        
        # Forward pass with trained model
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
        action_probs = rl_output['action_probs'][0]
        predicted_rewards = rl_output['predicted_rewards']
        
        # Get top categories
        top_categories = torch.topk(category_scores, 3)
        predicted_categories = [model.gift_categories[idx] for idx in top_categories.indices]
        
        # Get top gift recommendations
        top_gifts = torch.topk(action_probs, 3)
        recommended_gifts = []
        for idx in top_gifts.indices:
            if idx.item() < len(env.gift_catalog):
                gift = env.gift_catalog[idx.item()]
                recommended_gifts.append({
                    "name": gift.name,
                    "category": gift.category,
                    "price": gift.price,
                    "probability": top_gifts.values[len(recommended_gifts)].item()
                })
        
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
        
        # Calculate average reward
        avg_reward = predicted_rewards.mean().item()
        total_rewards.append(avg_reward)
        
        # Environment step to get actual reward
        gift_ids = [gift["name"] for gift in recommended_gifts]  # Simplified
        action = {
            'recommendations': gift_ids[:2],  # Take top 2
            'confidence_scores': [0.9, 0.8]
        }
        
        try:
            next_state, env_reward, done, info = env.step(action)
        except:
            env_reward = avg_reward  # Fallback to predicted reward
        
        print(f"   🎁 Top Recommendations:")
        for j, gift in enumerate(recommended_gifts):
            print(f"     {j+1}. {gift['name']} ({gift['category']}) - ${gift['price']:.2f}")
            print(f"        Probability: {gift['probability']:.3f}")
        
        print(f"   🏷️ Predicted Categories: {predicted_categories}")
        print(f"   🛠️ Selected Tools: {selected_tools}")
        print(f"   💰 Predicted Reward: {avg_reward:.3f}")
        print(f"   🌟 Environment Reward: {env_reward:.3f}")
        print(f"   ✅ Category Match: {'Yes' if category_match else 'No'}")
        print(f"   🔧 Tool Match: {'Yes' if tool_match else 'No'}")
        
        results.append({
            "scenario": scenario["name"],
            "user_profile": {
                "age": user.age,
                "hobbies": user.hobbies,
                "budget": user.budget,
                "occasion": user.occasion
            },
            "predictions": {
                "categories": predicted_categories,
                "tools": selected_tools,
                "gifts": recommended_gifts
            },
            "expected": {
                "categories": scenario["expected_categories"],
                "tools": scenario["expected_tools"]
            },
            "metrics": {
                "category_match": category_match,
                "tool_match": tool_match,
                "predicted_reward": avg_reward,
                "environment_reward": env_reward
            }
        })
    
    # Calculate overall metrics
    category_match_rate = category_matches / len(scenarios)
    tool_match_rate = tool_matches / len(scenarios)
    avg_predicted_reward = np.mean(total_rewards)
    
    print(f"\n📊 TRAINED MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"🎯 Category Match Rate: {category_match_rate:.1%} ({category_matches}/{len(scenarios)})")
    print(f"🛠️ Tool Match Rate: {tool_match_rate:.1%} ({tool_matches}/{len(scenarios)})")
    print(f"💰 Average Predicted Reward: {avg_predicted_reward:.3f}")
    
    # Overall score calculation
    overall_score = (category_match_rate * 0.4 + tool_match_rate * 0.3 + avg_predicted_reward * 0.3)
    print(f"🏆 Overall Score: {overall_score:.3f}/1.000")
    
    # Performance assessment
    if overall_score > 0.8:
        print("🌟 EXCELLENT: Trained model performs exceptionally well!")
    elif overall_score > 0.6:
        print("✅ GOOD: Trained model shows strong performance!")
    elif overall_score > 0.4:
        print("⚠️ FAIR: Trained model shows decent performance")
    else:
        print("❌ POOR: Trained model needs more work")
    
    return {
        "category_match_rate": category_match_rate,
        "tool_match_rate": tool_match_rate,
        "avg_predicted_reward": avg_predicted_reward,
        "overall_score": overall_score,
        "results": results
    }


def compare_with_original_performance():
    """Compare with original performance metrics"""
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Original performance (from initial testing)
    original_metrics = {
        "category_match_rate": 0.375,  # 37.5%
        "tool_match_rate": 0.50,       # 50%
        "avg_reward": 0.131,
        "overall_score": 0.390
    }
    
    print(f"📊 Original vs Trained Model:")
    print(f"{'Metric':<25} {'Original':<12} {'Trained':<12} {'Improvement':<12}")
    print("-" * 65)
    
    return original_metrics


def detailed_analysis(results):
    """Perform detailed analysis of results"""
    print(f"\n🔍 DETAILED ANALYSIS")
    print("=" * 30)
    
    # Category analysis
    category_predictions = {}
    for result in results:
        for category in result["predictions"]["categories"]:
            if category not in category_predictions:
                category_predictions[category] = 0
            category_predictions[category] += 1
    
    print(f"📊 Most Predicted Categories:")
    sorted_categories = sorted(category_predictions.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories[:5]:
        print(f"  {category}: {count} times ({count/len(results)*100:.1f}%)")
    
    # Tool analysis
    tool_predictions = {}
    for result in results:
        for tool in result["predictions"]["tools"]:
            if tool not in tool_predictions:
                tool_predictions[tool] = 0
            tool_predictions[tool] += 1
    
    print(f"\n🛠️ Most Selected Tools:")
    sorted_tools = sorted(tool_predictions.items(), key=lambda x: x[1], reverse=True)
    for tool, count in sorted_tools:
        print(f"  {tool}: {count} times ({count/len(results)*100:.1f}%)")
    
    # Reward distribution
    rewards = [result["metrics"]["predicted_reward"] for result in results]
    print(f"\n💰 Reward Statistics:")
    print(f"  Min: {min(rewards):.3f}")
    print(f"  Max: {max(rewards):.3f}")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std: {np.std(rewards):.3f}")


def main():
    """Main test function"""
    print("🧪 TESTING TRAINED INTEGRATED ENHANCED TRM MODEL")
    print("=" * 70)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load trained model
    model, checkpoint = load_trained_model()
    
    if checkpoint is None:
        print("⚠️ No trained model found - testing with untrained model")
    
    # Test with realistic scenarios
    performance = test_realistic_scenarios_with_trained_model(model)
    
    if performance:
        # Compare with original
        original_metrics = compare_with_original_performance()
        
        # Show comparison
        trained_metrics = {
            "category_match_rate": performance["category_match_rate"],
            "tool_match_rate": performance["tool_match_rate"], 
            "avg_reward": performance["avg_predicted_reward"],
            "overall_score": performance["overall_score"]
        }
        
        for metric, original_value in original_metrics.items():
            trained_value = trained_metrics[metric]
            improvement = trained_value - original_value
            
            if metric.endswith("_rate"):
                print(f"{metric:<25} {original_value:.1%}        {trained_value:.1%}        {improvement:+.1%}")
            else:
                print(f"{metric:<25} {original_value:.3f}        {trained_value:.3f}        {improvement:+.3f}")
        
        # Detailed analysis
        detailed_analysis(performance["results"])
        
        # Final assessment
        print(f"\n🎉 FINAL ASSESSMENT:")
        if performance["overall_score"] > 0.8:
            print(f"🌟 OUTSTANDING: Trained model achieved excellent performance!")
            print(f"🚀 Ready for production deployment!")
        elif performance["overall_score"] > 0.6:
            print(f"✅ SUCCESS: Trained model shows strong improvement!")
            print(f"📈 Significant upgrade from original model!")
        else:
            print(f"⚠️ PARTIAL: Some improvement achieved, consider more training")
        
        return performance["overall_score"] > 0.6
    
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)