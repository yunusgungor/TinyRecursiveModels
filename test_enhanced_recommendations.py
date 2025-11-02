#!/usr/bin/env python3
"""
Test script for enhanced recommendation system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from models.rl.environment import UserProfile, GiftItem, GiftRecommendationEnvironment
from models.rl.enhanced_recommendation_engine import EnhancedRecommendationEngine
from models.rl.enhanced_reward_function import EnhancedRewardFunction
from models.rl.enhanced_user_profiler import EnhancedUserProfiler


def load_test_data():
    """Load test data"""
    # Load realistic gift catalog
    with open("data/realistic_gift_catalog.json", "r") as f:
        gift_data = json.load(f)
    
    # Load user scenarios
    with open("data/realistic_user_scenarios.json", "r") as f:
        scenario_data = json.load(f)
    
    return gift_data, scenario_data


def create_gift_catalog(gift_data):
    """Create gift catalog from data"""
    gifts = []
    for gift in gift_data["gifts"]:
        gift_item = GiftItem(
            id=gift["id"],
            name=gift["name"],
            category=gift["category"],
            price=gift["price"],
            rating=gift["rating"],
            tags=gift["tags"],
            description=f"{gift['name']} - {gift['category']} item",
            age_suitability=tuple(gift["age_range"]),
            occasion_fit=gift["occasions"]
        )
        gifts.append(gift_item)
    return gifts


def test_enhanced_profiler():
    """Test the enhanced user profiler"""
    print("🧪 Testing Enhanced User Profiler")
    print("=" * 50)
    
    profiler = EnhancedUserProfiler()
    
    # Test user
    user = UserProfile(
        age=35,
        hobbies=["gardening", "wellness", "cooking"],
        relationship="mother",
        budget=100.0,
        occasion="mothers_day",
        personality_traits=["eco-friendly", "practical", "relaxing"]
    )
    
    available_categories = ["technology", "gardening", "cooking", "books", "wellness", "art", "fitness"]
    
    scores = profiler.calculate_category_scores(user, available_categories)
    top_categories = profiler.get_top_categories(user, available_categories, top_k=5)
    
    print("Category Scores:")
    for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {score:.3f}")
    
    print(f"\nTop 5 Categories:")
    for cat, score in top_categories:
        print(f"  {cat}: {score:.3f}")
    
    return scores


def test_enhanced_recommendation_engine():
    """Test the enhanced recommendation engine"""
    print("\n🎯 Testing Enhanced Recommendation Engine")
    print("=" * 50)
    
    # Load data
    gift_data, scenario_data = load_test_data()
    gift_catalog = create_gift_catalog(gift_data)
    
    # Create recommendation engine
    engine = EnhancedRecommendationEngine(gift_catalog)
    
    # Test with first scenario
    scenario = scenario_data["scenarios"][0]  # Sarah - Young Professional
    user = UserProfile(
        age=scenario["profile"]["age"],
        hobbies=scenario["profile"]["hobbies"],
        relationship=scenario["profile"]["relationship"],
        budget=scenario["profile"]["budget"],
        occasion=scenario["profile"]["occasion"],
        personality_traits=scenario["profile"]["preferences"]
    )
    
    print(f"Testing with: {scenario['name']}")
    print(f"Hobbies: {user.hobbies}")
    print(f"Expected categories: {scenario['expected_categories']}")
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(user, max_recommendations=3)
    
    print(f"\nRecommendations:")
    recommended_categories = []
    for i, (gift, score) in enumerate(recommendations, 1):
        print(f"{i}. {gift.name} (${gift.price:.2f}) - {gift.category}")
        print(f"   Confidence: {score:.3f}")
        recommended_categories.append(gift.category)
    
    # Check category match
    expected_categories = set(scenario["expected_categories"])
    actual_categories = set(recommended_categories)
    category_match = len(expected_categories.intersection(actual_categories)) > 0
    
    print(f"\nCategory Match: {'✅ Yes' if category_match else '❌ No'}")
    print(f"Expected: {expected_categories}")
    print(f"Actual: {actual_categories}")
    
    # Get explanations
    explanations = engine.explain_recommendations(user, recommendations)
    print(f"\nExplanations:")
    for exp in explanations:
        print(f"• {exp['gift']}: {', '.join(exp['reasons'])}")
    
    return recommendations, category_match


def test_enhanced_reward_function():
    """Test the enhanced reward function"""
    print("\n💰 Testing Enhanced Reward Function")
    print("=" * 50)
    
    # Load data
    gift_data, scenario_data = load_test_data()
    gift_catalog = create_gift_catalog(gift_data)
    
    reward_func = EnhancedRewardFunction()
    
    # Test with different scenarios
    results = []
    
    for scenario in scenario_data["scenarios"][:3]:  # Test first 3 scenarios
        user = UserProfile(
            age=scenario["profile"]["age"],
            hobbies=scenario["profile"]["hobbies"],
            relationship=scenario["profile"]["relationship"],
            budget=scenario["profile"]["budget"],
            occasion=scenario["profile"]["occasion"],
            personality_traits=scenario["profile"]["preferences"]
        )
        
        print(f"\nTesting: {scenario['name']}")
        
        # Find gifts that match expected categories
        good_gifts = []
        poor_gifts = []
        
        for gift in gift_catalog:
            if gift.category in scenario["expected_categories"]:
                good_gifts.append(gift)
            elif gift.category == "technology":  # Known overused category
                poor_gifts.append(gift)
        
        # Test good match
        if good_gifts:
            good_gift = good_gifts[0]
            good_reward = reward_func.calculate_reward(user, [good_gift], [0.9])
            print(f"  Good match ({good_gift.category}): {good_reward:.3f}")
        
        # Test poor match
        if poor_gifts:
            poor_gift = poor_gifts[0]
            poor_reward = reward_func.calculate_reward(user, [poor_gift], [0.8])
            print(f"  Poor match ({poor_gift.category}): {poor_reward:.3f}")
            
            # Get detailed explanation for poor match
            explanation = reward_func.explain_reward(user, [poor_gift], [0.8])
            print(f"    Category score: {explanation['gift_scores'][0]['category_score']:.3f}")
            print(f"    Hobby score: {explanation['gift_scores'][0]['hobby_score']:.3f}")
        
        results.append({
            "scenario": scenario["name"],
            "good_reward": good_reward if good_gifts else 0,
            "poor_reward": poor_reward if poor_gifts else 0
        })
    
    return results


def test_full_system():
    """Test the full enhanced system"""
    print("\n🚀 Testing Full Enhanced System")
    print("=" * 50)
    
    # Load data
    gift_data, scenario_data = load_test_data()
    
    # Create enhanced environment
    gift_catalog = []
    for gift in gift_data["gifts"]:
        gift_item = GiftItem(
            id=gift["id"],
            name=gift["name"],
            category=gift["category"],
            price=gift["price"],
            rating=gift["rating"],
            tags=gift["tags"],
            description=f"{gift['name']} - {gift['category']} item",
            age_suitability=tuple(gift["age_range"]),
            occasion_fit=gift["occasions"]
        )
        gift_catalog.append(gift_item)
    
    # Save catalog for environment
    catalog_data = []
    for gift in gift_catalog:
        catalog_data.append({
            "id": gift.id,
            "name": gift.name,
            "category": gift.category,
            "price": gift.price,
            "rating": gift.rating,
            "tags": gift.tags,
            "description": gift.description,
            "age_suitability": list(gift.age_suitability),
            "occasion_fit": gift.occasion_fit
        })
    
    with open("data/enhanced_test_catalog.json", "w") as f:
        json.dump(catalog_data, f, indent=2)
    
    # Create environment
    env = GiftRecommendationEnvironment("data/enhanced_test_catalog.json")
    
    # Create recommendation engine
    rec_engine = EnhancedRecommendationEngine(gift_catalog)
    
    # Test scenarios
    results = []
    category_matches = 0
    total_scenarios = 0
    
    for scenario in scenario_data["scenarios"]:
        user = UserProfile(
            age=scenario["profile"]["age"],
            hobbies=scenario["profile"]["hobbies"],
            relationship=scenario["profile"]["relationship"],
            budget=scenario["profile"]["budget"],
            occasion=scenario["profile"]["occasion"],
            personality_traits=scenario["profile"]["preferences"]
        )
        
        print(f"\n👤 {scenario['name']}")
        print(f"   Expected categories: {scenario['expected_categories']}")
        
        # Reset environment
        state = env.reset(user)
        
        # Generate enhanced recommendations
        recommendations = rec_engine.generate_recommendations(user, max_recommendations=3)
        
        if recommendations:
            # Extract gift IDs and scores
            gift_ids = [gift.id for gift, _ in recommendations]
            confidence_scores = [score for _, score in recommendations]
            
            # Step environment
            action = {
                'recommendations': gift_ids,
                'confidence_scores': confidence_scores
            }
            
            next_state, reward, done, info = env.step(action)
            
            # Check category match
            recommended_categories = [gift.category for gift, _ in recommendations]
            expected_categories = set(scenario["expected_categories"])
            actual_categories = set(recommended_categories)
            category_match = len(expected_categories.intersection(actual_categories)) > 0
            
            if category_match:
                category_matches += 1
            
            print(f"   Recommendations: {[f'{gift.name} ({gift.category})' for gift, _ in recommendations]}")
            print(f"   Reward: {reward:.3f}")
            print(f"   Category Match: {'✅' if category_match else '❌'}")
            
            results.append({
                "scenario": scenario["name"],
                "reward": reward,
                "category_match": category_match,
                "recommendations": [(gift.name, gift.category, score) for gift, score in recommendations]
            })
        
        total_scenarios += 1
    
    # Calculate metrics
    avg_reward = np.mean([r["reward"] for r in results])
    category_match_rate = category_matches / total_scenarios
    
    print(f"\n📊 Enhanced System Results:")
    print(f"   Average Reward: {avg_reward:.3f}")
    print(f"   Category Match Rate: {category_match_rate:.1%}")
    print(f"   Total Scenarios: {total_scenarios}")
    
    return results, avg_reward, category_match_rate


def main():
    """Main test function"""
    print("🧪 TESTING ENHANCED RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Test individual components
    profiler_scores = test_enhanced_profiler()
    recommendations, category_match = test_enhanced_recommendation_engine()
    reward_results = test_enhanced_reward_function()
    
    # Test full system
    system_results, avg_reward, category_match_rate = test_full_system()
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Enhanced Category Match Rate: {category_match_rate:.1%}")
    print(f"   Enhanced Average Reward: {avg_reward:.3f}")
    
    # Compare with expected improvements
    print(f"\n📈 EXPECTED IMPROVEMENTS:")
    print(f"   Original Category Match Rate: 37.5%")
    print(f"   Enhanced Category Match Rate: {category_match_rate:.1%}")
    print(f"   Improvement: {category_match_rate - 0.375:.1%}")
    
    print(f"\n   Original Average Reward: 0.131")
    print(f"   Enhanced Average Reward: {avg_reward:.3f}")
    print(f"   Improvement: {avg_reward - 0.131:.3f}")
    
    if category_match_rate > 0.5 and avg_reward > 0.2:
        print(f"\n✅ SUCCESS: Enhanced system shows significant improvement!")
    else:
        print(f"\n⚠️ PARTIAL: Some improvements achieved, but more work needed.")


if __name__ == "__main__":
    main()