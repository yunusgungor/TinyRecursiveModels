#!/usr/bin/env python3
"""
Test script for the best trained Integrated Enhanced TRM model
"""

import os
import torch
import json
import numpy as np
from typing import Dict, List

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import GiftRecommendationEnvironment, UserProfile


def load_best_model(checkpoint_path: str = "checkpoints/finetuned/finetuned_best.pt"):
    """Load the best trained model"""
    print(f"📂 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    model = IntegratedEnhancedTRM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Epoch: {checkpoint['epoch']}")
    print(f"📊 Metrics: {checkpoint['metrics']}")
    
    return model, checkpoint


def test_scenario(model: IntegratedEnhancedTRM, env: GiftRecommendationEnvironment, 
                 scenario: Dict, verbose: bool = True) -> Dict:
    """Test model on a single scenario"""
    
    # Create user profile
    user = UserProfile(
        age=scenario["profile"]["age"],
        hobbies=scenario["profile"]["hobbies"],
        relationship=scenario["profile"]["relationship"],
        budget=scenario["profile"]["budget"],
        occasion=scenario["profile"]["occasion"],
        personality_traits=scenario["profile"]["preferences"]
    )
    
    # Reset environment
    env_state = env.reset(user)
    
    # Forward pass
    with torch.no_grad():
        carry = model.initial_carry({
            "inputs": torch.zeros(1, 10),
            "puzzle_identifiers": torch.zeros(1, 1)
        })
        
        carry, model_outputs, selected_tools = model.forward_with_enhancements(
            carry, env_state, env.gift_catalog
        )
    
    # Analyze results
    category_scores = model_outputs['category_scores'][0]
    top_categories_idx = torch.topk(category_scores, 3).indices
    predicted_categories = [model.gift_categories[idx] for idx in top_categories_idx]
    
    tool_scores = model_outputs['tool_scores'][0]
    predicted_reward = model_outputs['predicted_rewards'].mean().item()
    
    # Check matches
    expected_categories = set(scenario["expected_categories"])
    actual_categories = set(predicted_categories)
    category_match = len(expected_categories.intersection(actual_categories)) > 0
    
    expected_tools = set(scenario["expected_tools"])
    actual_tools = set(selected_tools)
    tool_match = len(expected_tools.intersection(actual_tools)) > 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['id']}")
        print(f"User: {user.age}y, {user.relationship}, ${user.budget:.0f}, {user.occasion}")
        print(f"Hobbies: {', '.join(user.hobbies)}")
        print(f"Preferences: {', '.join(user.personality_traits)}")
        print(f"\nExpected Categories: {', '.join(expected_categories)}")
        print(f"Predicted Categories: {', '.join(predicted_categories)}")
        print(f"Category Match: {'✅' if category_match else '❌'}")
        print(f"\nExpected Tools: {', '.join(expected_tools)}")
        print(f"Selected Tools: {', '.join(selected_tools)}")
        print(f"Tool Match: {'✅' if tool_match else '❌'}")
        print(f"\nPredicted Reward: {predicted_reward:.3f}")
    
    return {
        'category_match': category_match,
        'tool_match': tool_match,
        'predicted_reward': predicted_reward,
        'predicted_categories': predicted_categories,
        'selected_tools': selected_tools
    }


def comprehensive_test(model: IntegratedEnhancedTRM, num_tests: int = 50):
    """Run comprehensive tests on the model"""
    print(f"\n{'='*60}")
    print(f"🧪 COMPREHENSIVE MODEL TEST")
    print(f"{'='*60}\n")
    
    # Load test scenarios
    try:
        with open("data/expanded_user_scenarios.json", "r") as f:
            scenario_data = json.load(f)
        scenarios = scenario_data["scenarios"]
    except:
        with open("data/realistic_user_scenarios.json", "r") as f:
            scenario_data = json.load(f)
        scenarios = scenario_data["scenarios"]
    
    # Initialize environment
    env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
    
    # Test on random scenarios
    test_scenarios = np.random.choice(scenarios, min(num_tests, len(scenarios)), replace=False)
    
    results = []
    for scenario in test_scenarios:
        result = test_scenario(model, env, scenario, verbose=False)
        results.append(result)
    
    # Calculate statistics
    category_match_rate = sum(r['category_match'] for r in results) / len(results)
    tool_match_rate = sum(r['tool_match'] for r in results) / len(results)
    avg_reward = np.mean([r['predicted_reward'] for r in results])
    quality_score = (category_match_rate + avg_reward) / 2
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS (n={len(results)})")
    print(f"{'='*60}")
    print(f"Category Match Rate: {category_match_rate:.1%}")
    print(f"Tool Match Rate: {tool_match_rate:.1%}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Quality Score: {quality_score:.3f}")
    print(f"{'='*60}\n")
    
    # Show some examples
    print(f"📝 Sample Test Cases:\n")
    for i, (scenario, result) in enumerate(zip(test_scenarios[:5], results[:5])):
        test_scenario(model, env, scenario, verbose=True)
    
    return {
        'category_match_rate': category_match_rate,
        'tool_match_rate': tool_match_rate,
        'average_reward': avg_reward,
        'quality_score': quality_score,
        'num_tests': len(results)
    }


def main():
    """Main test function"""
    print("🚀 INTEGRATED ENHANCED TRM - MODEL TESTING")
    print("="*60)
    
    # Load best model
    model, checkpoint = load_best_model()
    
    # Run comprehensive tests
    results = comprehensive_test(model, num_tests=50)
    
    # Save test results
    test_report = {
        'model_checkpoint': 'integrated_enhanced_best.pt',
        'model_epoch': checkpoint['epoch'],
        'training_metrics': checkpoint['metrics'],
        'test_results': results,
        'model_info': checkpoint.get('model_info', {})
    }
    
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/best_model_test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n✅ Test report saved to test_results/best_model_test_report.json")
    print(f"🎉 Testing completed!")


if __name__ == "__main__":
    main()
