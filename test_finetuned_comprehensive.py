#!/usr/bin/env python3
"""
Comprehensive test script for fine-tuned model with real-world scenarios
"""

import os
import torch
import json
import numpy as np
from typing import Dict, List
from collections import Counter

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile


class ComprehensiveModelTester:
    """Comprehensive testing for fine-tuned model"""
    
    def __init__(self, checkpoint_path: str):
        print("🧪 COMPREHENSIVE MODEL TESTING")
        print("="*60)
        
        # Load checkpoint
        print(f"📂 Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = checkpoint['config']
        self.model = IntegratedEnhancedTRM(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"📊 Training metrics: {checkpoint['metrics']}")
        
        # Initialize environment
        self.env = GiftRecommendationEnvironment("data/realistic_gift_catalog.json")
        
        print(f"🎁 Gift catalog: {len(self.env.gift_catalog)} items")
        print(f"📦 Categories: {len(self.model.gift_categories)}")
        print(f"🔧 Tools: {len(self.model.tool_registry.list_tools())}")
        print()
    
    def create_real_world_scenarios(self) -> List[Dict]:
        """Create realistic test scenarios"""
        scenarios = [
            {
                "name": "Tech-savvy friend's birthday",
                "profile": {
                    "age": 28,
                    "hobbies": ["technology", "gaming", "photography"],
                    "relationship": "friend",
                    "budget": 120.0,
                    "occasion": "birthday",
                    "preferences": ["trendy", "tech-savvy", "practical"]
                },
                "expected_categories": ["technology", "gaming"],
                "expected_tools": ["price_comparison", "review_analysis"]
            },
            {
                "name": "Gardening enthusiast mother",
                "profile": {
                    "age": 55,
                    "hobbies": ["gardening", "cooking", "reading"],
                    "relationship": "mother",
                    "budget": 80.0,
                    "occasion": "mothers_day",
                    "preferences": ["eco-friendly", "natural", "practical"]
                },
                "expected_categories": ["gardening", "cooking", "books"],
                "expected_tools": ["review_analysis", "inventory_check"]
            },
            {
                "name": "Wellness-focused sister",
                "profile": {
                    "age": 32,
                    "hobbies": ["wellness", "yoga", "fitness"],
                    "relationship": "sister",
                    "budget": 90.0,
                    "occasion": "birthday",
                    "preferences": ["healthy", "natural", "self-care"]
                },
                "expected_categories": ["wellness", "fitness"],
                "expected_tools": ["review_analysis", "budget_optimizer"]
            },
            {
                "name": "Artistic friend",
                "profile": {
                    "age": 26,
                    "hobbies": ["art", "design", "photography"],
                    "relationship": "friend",
                    "budget": 75.0,
                    "occasion": "birthday",
                    "preferences": ["creative", "unique", "artistic"]
                },
                "expected_categories": ["art", "books"],
                "expected_tools": ["review_analysis", "trend_analysis"]
            },
            {
                "name": "Outdoor adventure father",
                "profile": {
                    "age": 50,
                    "hobbies": ["outdoor", "fitness", "travel"],
                    "relationship": "father",
                    "budget": 150.0,
                    "occasion": "fathers_day",
                    "preferences": ["active", "quality", "practical"]
                },
                "expected_categories": ["outdoor", "fitness"],
                "expected_tools": ["price_comparison", "review_analysis"]
            },
            {
                "name": "Foodie spouse",
                "profile": {
                    "age": 35,
                    "hobbies": ["cooking", "food", "travel"],
                    "relationship": "spouse",
                    "budget": 130.0,
                    "occasion": "anniversary",
                    "preferences": ["quality", "sophisticated", "gourmet"]
                },
                "expected_categories": ["cooking", "food", "experience"],
                "expected_tools": ["review_analysis", "trend_analysis"]
            },
            {
                "name": "Budget-conscious student",
                "profile": {
                    "age": 20,
                    "hobbies": ["reading", "technology", "music"],
                    "relationship": "friend",
                    "budget": 40.0,
                    "occasion": "birthday",
                    "preferences": ["affordable", "practical", "trendy"]
                },
                "expected_categories": ["books", "technology"],
                "expected_tools": ["budget_optimizer", "price_comparison"]
            },
            {
                "name": "Luxury gift for spouse",
                "profile": {
                    "age": 40,
                    "hobbies": ["wellness", "travel", "food"],
                    "relationship": "spouse",
                    "budget": 250.0,
                    "occasion": "anniversary",
                    "preferences": ["luxury", "sophisticated", "quality"]
                },
                "expected_categories": ["wellness", "experience", "food"],
                "expected_tools": ["review_analysis", "trend_analysis"]
            },
            {
                "name": "Eco-conscious friend",
                "profile": {
                    "age": 29,
                    "hobbies": ["gardening", "wellness", "outdoor"],
                    "relationship": "friend",
                    "budget": 70.0,
                    "occasion": "birthday",
                    "preferences": ["eco-friendly", "sustainable", "natural"]
                },
                "expected_categories": ["gardening", "wellness", "outdoor"],
                "expected_tools": ["review_analysis", "inventory_check"]
            },
            {
                "name": "Gaming enthusiast",
                "profile": {
                    "age": 22,
                    "hobbies": ["gaming", "technology", "music"],
                    "relationship": "brother",
                    "budget": 100.0,
                    "occasion": "christmas",
                    "preferences": ["trendy", "tech-savvy", "fun"]
                },
                "expected_categories": ["gaming", "technology"],
                "expected_tools": ["price_comparison", "review_analysis"]
            }
        ]
        
        return scenarios
    
    def test_scenario(self, scenario: Dict, verbose: bool = True) -> Dict:
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
        env_state = self.env.reset(user)
        
        # Forward pass
        with torch.no_grad():
            carry = self.model.initial_carry({
                "inputs": torch.zeros(1, 10),
                "puzzle_identifiers": torch.zeros(1, 1)
            })
            
            carry, model_outputs, selected_tools = self.model.forward_with_enhancements(
                carry, env_state, self.env.gift_catalog
            )
        
        # Analyze results
        category_scores = model_outputs['category_scores'][0]
        top_categories_idx = torch.topk(category_scores, 5).indices
        predicted_categories = [self.model.gift_categories[idx] for idx in top_categories_idx]
        
        predicted_reward = model_outputs['predicted_rewards'].mean().item()
        
        # Check matches
        expected_categories = set(scenario["expected_categories"])
        actual_categories = set(predicted_categories[:3])  # Top 3
        category_match = len(expected_categories.intersection(actual_categories)) > 0
        
        expected_tools = set(scenario["expected_tools"])
        actual_tools = set(selected_tools)
        tool_match = len(expected_tools.intersection(actual_tools)) > 0
        
        # Calculate match percentage
        category_overlap = len(expected_categories.intersection(actual_categories))
        category_match_pct = category_overlap / len(expected_categories) if expected_categories else 0
        
        tool_overlap = len(expected_tools.intersection(actual_tools))
        tool_match_pct = tool_overlap / len(expected_tools) if expected_tools else 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎯 Scenario: {scenario['name']}")
            print(f"{'='*60}")
            print(f"👤 User: {user.age}y {user.relationship}, ${user.budget:.0f}, {user.occasion}")
            print(f"🎨 Hobbies: {', '.join(user.hobbies)}")
            print(f"💭 Preferences: {', '.join(user.personality_traits)}")
            print()
            print(f"📦 Expected Categories: {', '.join(expected_categories)}")
            print(f"🎁 Predicted Categories (Top 5):")
            for i, cat in enumerate(predicted_categories, 1):
                marker = "✅" if cat in expected_categories else "  "
                print(f"   {i}. {cat} {marker}")
            print(f"📊 Category Match: {category_match_pct:.0%} ({category_overlap}/{len(expected_categories)}) {'✅' if category_match else '❌'}")
            print()
            print(f"🔧 Expected Tools: {', '.join(expected_tools)}")
            print(f"🛠️  Selected Tools: {', '.join(selected_tools)}")
            print(f"📊 Tool Match: {tool_match_pct:.0%} ({tool_overlap}/{len(expected_tools)}) {'✅' if tool_match else '❌'}")
            print()
            print(f"⭐ Predicted Reward: {predicted_reward:.3f}")
        
        return {
            'scenario_name': scenario['name'],
            'category_match': category_match,
            'category_match_pct': category_match_pct,
            'tool_match': tool_match,
            'tool_match_pct': tool_match_pct,
            'predicted_reward': predicted_reward,
            'predicted_categories': predicted_categories,
            'selected_tools': selected_tools,
            'expected_categories': list(expected_categories),
            'expected_tools': list(expected_tools)
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 RUNNING COMPREHENSIVE TEST SUITE")
        print("="*60)
        print()
        
        # Create real-world scenarios
        scenarios = self.create_real_world_scenarios()
        print(f"📋 Testing {len(scenarios)} real-world scenarios\n")
        
        # Test each scenario
        results = []
        for scenario in scenarios:
            result = self.test_scenario(scenario, verbose=True)
            results.append(result)
        
        # Calculate overall statistics
        print("\n" + "="*60)
        print("📊 OVERALL TEST RESULTS")
        print("="*60)
        
        category_match_rate = sum(r['category_match'] for r in results) / len(results)
        avg_category_match_pct = np.mean([r['category_match_pct'] for r in results])
        tool_match_rate = sum(r['tool_match'] for r in results) / len(results)
        avg_tool_match_pct = np.mean([r['tool_match_pct'] for r in results])
        avg_reward = np.mean([r['predicted_reward'] for r in results])
        
        print(f"\n✅ Category Match Rate: {category_match_rate:.1%}")
        print(f"📊 Avg Category Overlap: {avg_category_match_pct:.1%}")
        print(f"✅ Tool Match Rate: {tool_match_rate:.1%}")
        print(f"📊 Avg Tool Overlap: {avg_tool_match_pct:.1%}")
        print(f"⭐ Average Reward: {avg_reward:.3f}")
        print(f"🎯 Quality Score: {(category_match_rate + avg_reward) / 2:.3f}")
        
        # Category distribution analysis
        all_predicted_categories = []
        for r in results:
            all_predicted_categories.extend(r['predicted_categories'][:3])
        
        category_dist = Counter(all_predicted_categories)
        
        print(f"\n📦 Category Distribution (Top 3 predictions):")
        for cat, count in category_dist.most_common(10):
            pct = count / (len(results) * 3) * 100
            print(f"   {cat}: {count} ({pct:.1f}%)")
        
        print(f"\n🎨 Unique Categories Used: {len(category_dist)}/{len(self.model.gift_categories)}")
        
        # Tool distribution analysis
        all_selected_tools = []
        for r in results:
            all_selected_tools.extend(r['selected_tools'])
        
        tool_dist = Counter(all_selected_tools)
        
        print(f"\n🔧 Tool Distribution:")
        for tool, count in tool_dist.most_common():
            pct = count / len(results) * 100
            print(f"   {tool}: {count} ({pct:.1f}%)")
        
        # Best and worst scenarios
        print(f"\n🏆 Best Performing Scenarios:")
        sorted_results = sorted(results, key=lambda x: x['category_match_pct'], reverse=True)
        for r in sorted_results[:3]:
            print(f"   {r['scenario_name']}: {r['category_match_pct']:.0%} category, {r['tool_match_pct']:.0%} tool")
        
        print(f"\n⚠️  Challenging Scenarios:")
        for r in sorted_results[-3:]:
            print(f"   {r['scenario_name']}: {r['category_match_pct']:.0%} category, {r['tool_match_pct']:.0%} tool")
        
        # Save detailed results
        report = {
            'model_checkpoint': 'finetuned_best.pt',
            'test_date': '2025-11-12',
            'num_scenarios': len(results),
            'overall_metrics': {
                'category_match_rate': category_match_rate,
                'avg_category_overlap': avg_category_match_pct,
                'tool_match_rate': tool_match_rate,
                'avg_tool_overlap': avg_tool_match_pct,
                'average_reward': avg_reward,
                'quality_score': (category_match_rate + avg_reward) / 2
            },
            'category_distribution': dict(category_dist),
            'tool_distribution': dict(tool_dist),
            'unique_categories_used': len(category_dist),
            'total_categories': len(self.model.gift_categories),
            'detailed_results': results
        }
        
        os.makedirs("test_results", exist_ok=True)
        with open("test_results/comprehensive_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: test_results/comprehensive_test_report.json")
        print(f"\n🎉 Comprehensive testing completed!")
        
        return report


def main():
    """Main test function"""
    checkpoint_path = "checkpoints/finetuned/finetuned_best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run fine-tuning first: python finetune_category_diversity.py")
        return
    
    # Initialize tester
    tester = ComprehensiveModelTester(checkpoint_path)
    
    # Run comprehensive test
    report = tester.run_comprehensive_test()
    
    # Print summary
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)
    print(f"✅ Model: Fine-tuned (Category Diversity)")
    print(f"📊 Test Scenarios: {report['num_scenarios']} real-world cases")
    print(f"🎯 Category Match: {report['overall_metrics']['category_match_rate']:.1%}")
    print(f"🔧 Tool Match: {report['overall_metrics']['tool_match_rate']:.1%}")
    print(f"⭐ Quality Score: {report['overall_metrics']['quality_score']:.3f}")
    print(f"🎨 Category Diversity: {report['unique_categories_used']}/{report['total_categories']}")
    print("="*60)


if __name__ == "__main__":
    main()
