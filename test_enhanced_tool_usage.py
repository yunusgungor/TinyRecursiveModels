#!/usr/bin/env python3
"""
Test script for enhanced tool usage diversity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from collections import Counter
from models.rl.environment import UserProfile
from models.tools.enhanced_tool_selector import ContextAwareToolSelector


def test_tool_diversity():
    """Test tool selection diversity across different user profiles"""
    print("🛠️ TESTING ENHANCED TOOL USAGE DIVERSITY")
    print("=" * 60)
    
    selector = ContextAwareToolSelector()
    
    # Load realistic user scenarios
    with open("data/realistic_user_scenarios.json", "r") as f:
        scenario_data = json.load(f)
    
    all_tool_selections = []
    scenario_results = []
    
    print("🧪 Testing Tool Selection for Each Scenario:")
    print("-" * 50)
    
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
        print(f"   Age: {user.age}, Budget: ${user.budget}")
        print(f"   Hobbies: {user.hobbies}")
        print(f"   Expected tools: {scenario['expected_tools']}")
        
        # Select tools multiple times to test consistency and diversity
        selected_tools_list = []
        for _ in range(3):  # Test 3 times
            selected_tools = selector.select_tools(user, max_tools=2)
            selected_tools_list.append([tool for tool, _ in selected_tools])
            all_tool_selections.extend([tool for tool, _ in selected_tools])
        
        # Analyze selections
        all_selected = [tool for sublist in selected_tools_list for tool in sublist]
        tool_counts = Counter(all_selected)
        
        print(f"   Selected tools (3 runs):")
        for tool, count in tool_counts.most_common():
            print(f"     • {tool}: {count}/6 times")
        
        # Check if expected tools were selected
        expected_tools = set(scenario["expected_tools"])
        actual_tools = set(all_selected)
        tool_match = len(expected_tools.intersection(actual_tools)) > 0
        
        print(f"   Tool Match: {'✅ Yes' if tool_match else '❌ No'}")
        
        # Get explanations for first selection
        if selected_tools_list:
            explanations = selector.explain_tool_selection(user, 
                [(tool, 0.8) for tool in selected_tools_list[0]])
            print(f"   Explanations:")
            for tool, exp in explanations.items():
                print(f"     • {tool}: {', '.join(exp['reasons'][:2])}")  # Show first 2 reasons
        
        scenario_results.append({
            "scenario": scenario["name"],
            "tool_match": tool_match,
            "selected_tools": list(actual_tools),
            "expected_tools": list(expected_tools)
        })
    
    # Calculate overall statistics
    print(f"\n📊 OVERALL TOOL USAGE STATISTICS:")
    print("=" * 50)
    
    tool_distribution = Counter(all_tool_selections)
    total_selections = len(all_tool_selections)
    
    print(f"Total tool selections: {total_selections}")
    print(f"Tool distribution:")
    for tool, count in tool_distribution.most_common():
        percentage = (count / total_selections) * 100
        print(f"  • {tool}: {count} times ({percentage:.1f}%)")
    
    # Calculate diversity metrics
    num_unique_tools = len(tool_distribution)
    max_possible_tools = 5  # We have 5 tools
    diversity_ratio = num_unique_tools / max_possible_tools
    
    # Calculate evenness (how evenly distributed the usage is)
    if tool_distribution:
        expected_count = total_selections / len(tool_distribution)
        evenness = 1 - (sum(abs(count - expected_count) for count in tool_distribution.values()) / 
                       (2 * total_selections * (1 - 1/len(tool_distribution))))
    else:
        evenness = 0
    
    print(f"\nDiversity Metrics:")
    print(f"  • Tool Diversity Ratio: {diversity_ratio:.2f} ({num_unique_tools}/{max_possible_tools} tools used)")
    print(f"  • Usage Evenness: {evenness:.2f} (1.0 = perfectly even)")
    
    # Check for overused tools
    overused_threshold = 0.4  # 40% threshold
    overused_tools = [tool for tool, count in tool_distribution.items() 
                     if (count / total_selections) > overused_threshold]
    
    if overused_tools:
        print(f"  • Overused tools (>{overused_threshold*100:.0f}%): {overused_tools}")
    else:
        print(f"  • No tools are overused (>{overused_threshold*100:.0f}%)")
    
    # Calculate tool match rate
    tool_matches = sum(1 for result in scenario_results if result["tool_match"])
    tool_match_rate = tool_matches / len(scenario_results)
    
    print(f"\nTool Matching:")
    print(f"  • Tool Match Rate: {tool_match_rate:.1%} ({tool_matches}/{len(scenario_results)} scenarios)")
    
    return {
        "tool_distribution": dict(tool_distribution),
        "diversity_ratio": diversity_ratio,
        "evenness": evenness,
        "tool_match_rate": tool_match_rate,
        "overused_tools": overused_tools
    }


def test_context_sensitivity():
    """Test if tool selection is sensitive to different contexts"""
    print(f"\n🎯 TESTING CONTEXT SENSITIVITY:")
    print("=" * 50)
    
    selector = ContextAwareToolSelector()
    
    # Test different contexts
    test_contexts = [
        {
            "name": "Budget-conscious young tech enthusiast",
            "user": UserProfile(
                age=22, hobbies=["technology", "gaming"], relationship="friend",
                budget=50.0, occasion="birthday", personality_traits=["trendy", "affordable"]
            ),
            "expected_tools": ["budget_optimizer", "price_comparison", "trend_analysis"]
        },
        {
            "name": "Quality-focused mature gardener",
            "user": UserProfile(
                age=55, hobbies=["gardening", "reading"], relationship="mother",
                budget=150.0, occasion="mothers_day", personality_traits=["quality", "practical"]
            ),
            "expected_tools": ["review_analysis", "inventory_check"]
        },
        {
            "name": "Luxury-seeking executive",
            "user": UserProfile(
                age=45, hobbies=["business", "wine"], relationship="boss",
                budget=300.0, occasion="appreciation", personality_traits=["luxury", "sophisticated"]
            ),
            "expected_tools": ["review_analysis", "inventory_check"]
        }
    ]
    
    context_results = []
    
    for context in test_contexts:
        print(f"\n🧪 {context['name']}")
        user = context["user"]
        
        # Select tools multiple times
        all_selections = []
        for _ in range(5):
            selected = selector.select_tools(user, max_tools=2)
            all_selections.extend([tool for tool, _ in selected])
        
        tool_counts = Counter(all_selections)
        print(f"   Selected tools:")
        for tool, count in tool_counts.most_common():
            print(f"     • {tool}: {count}/10 times")
        
        # Check if context-appropriate tools were selected
        expected_set = set(context["expected_tools"])
        actual_set = set(all_selections)
        context_match = len(expected_set.intersection(actual_set)) > 0
        
        print(f"   Context Match: {'✅ Yes' if context_match else '❌ No'}")
        print(f"   Expected: {context['expected_tools']}")
        print(f"   Actual: {list(actual_set)}")
        
        context_results.append({
            "context": context["name"],
            "context_match": context_match,
            "tool_distribution": dict(tool_counts)
        })
    
    # Calculate context sensitivity
    context_matches = sum(1 for result in context_results if result["context_match"])
    context_sensitivity = context_matches / len(context_results)
    
    print(f"\n📈 Context Sensitivity: {context_sensitivity:.1%} ({context_matches}/{len(context_results)} contexts)")
    
    return context_results, context_sensitivity


def compare_with_baseline():
    """Compare enhanced tool selection with baseline (trend_analysis only)"""
    print(f"\n📊 COMPARISON WITH BASELINE:")
    print("=" * 50)
    
    # Baseline: 87.5% trend_analysis usage (from original test)
    baseline_distribution = {
        "trend_analysis": 87.5,
        "price_comparison": 5.0,
        "review_analysis": 5.0,
        "budget_optimizer": 2.5,
        "inventory_check": 0.0
    }
    
    # Run enhanced tool selection test
    enhanced_results = test_tool_diversity()
    
    # Convert to percentages
    total_enhanced = sum(enhanced_results["tool_distribution"].values())
    enhanced_percentages = {
        tool: (count / total_enhanced) * 100 
        for tool, count in enhanced_results["tool_distribution"].items()
    }
    
    print(f"\nComparison:")
    print(f"{'Tool':<20} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 60)
    
    all_tools = set(list(baseline_distribution.keys()) + list(enhanced_percentages.keys()))
    
    for tool in sorted(all_tools):
        baseline_pct = baseline_distribution.get(tool, 0.0)
        enhanced_pct = enhanced_percentages.get(tool, 0.0)
        improvement = enhanced_pct - baseline_pct
        
        print(f"{tool:<20} {baseline_pct:<11.1f}% {enhanced_pct:<11.1f}% {improvement:+.1f}%")
    
    # Calculate improvement metrics
    print(f"\nImprovement Metrics:")
    
    # Diversity improvement
    baseline_diversity = len([tool for tool, pct in baseline_distribution.items() if pct > 5]) / 5
    enhanced_diversity = enhanced_results["diversity_ratio"]
    diversity_improvement = enhanced_diversity - baseline_diversity
    
    print(f"  • Diversity Ratio: {baseline_diversity:.2f} → {enhanced_diversity:.2f} (+{diversity_improvement:.2f})")
    
    # Overuse reduction
    baseline_overuse = max(baseline_distribution.values())
    enhanced_overuse = max(enhanced_percentages.values()) if enhanced_percentages else 0
    overuse_reduction = baseline_overuse - enhanced_overuse
    
    print(f"  • Max Tool Usage: {baseline_overuse:.1f}% → {enhanced_overuse:.1f}% (-{overuse_reduction:.1f}%)")
    
    # Tool match rate (baseline was 50%)
    baseline_tool_match = 50.0
    enhanced_tool_match = enhanced_results["tool_match_rate"] * 100
    tool_match_improvement = enhanced_tool_match - baseline_tool_match
    
    print(f"  • Tool Match Rate: {baseline_tool_match:.1f}% → {enhanced_tool_match:.1f}% (+{tool_match_improvement:.1f}%)")
    
    return {
        "diversity_improvement": diversity_improvement,
        "overuse_reduction": overuse_reduction,
        "tool_match_improvement": tool_match_improvement
    }


def main():
    """Main test function"""
    print("🧪 TESTING ENHANCED TOOL USAGE SYSTEM")
    print("=" * 60)
    
    # Test tool diversity
    diversity_results = test_tool_diversity()
    
    # Test context sensitivity
    context_results, context_sensitivity = test_context_sensitivity()
    
    # Compare with baseline
    comparison_results = compare_with_baseline()
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("=" * 50)
    
    print(f"Enhanced Tool Usage Results:")
    print(f"  • Tool Diversity Ratio: {diversity_results['diversity_ratio']:.2f}")
    print(f"  • Usage Evenness: {diversity_results['evenness']:.2f}")
    print(f"  • Tool Match Rate: {diversity_results['tool_match_rate']:.1%}")
    print(f"  • Context Sensitivity: {context_sensitivity:.1%}")
    
    print(f"\nImprovements over Baseline:")
    print(f"  • Diversity: +{comparison_results['diversity_improvement']:.2f}")
    print(f"  • Overuse Reduction: -{comparison_results['overuse_reduction']:.1f}%")
    print(f"  • Tool Match Rate: +{comparison_results['tool_match_improvement']:.1f}%")
    
    # Success criteria
    success_criteria = [
        diversity_results['diversity_ratio'] >= 0.8,  # Use at least 80% of available tools
        diversity_results['evenness'] >= 0.6,  # Reasonably even distribution
        diversity_results['tool_match_rate'] >= 0.6,  # 60% tool match rate
        context_sensitivity >= 0.7,  # 70% context sensitivity
        not diversity_results['overused_tools']  # No overused tools
    ]
    
    success_count = sum(success_criteria)
    
    if success_count >= 4:
        print(f"\n✅ SUCCESS: Enhanced tool usage shows significant improvement!")
        print(f"   Met {success_count}/5 success criteria")
    elif success_count >= 3:
        print(f"\n⚠️ PARTIAL SUCCESS: Good improvements with room for enhancement")
        print(f"   Met {success_count}/5 success criteria")
    else:
        print(f"\n❌ NEEDS WORK: Tool usage improvements are insufficient")
        print(f"   Met {success_count}/5 success criteria")


if __name__ == "__main__":
    main()