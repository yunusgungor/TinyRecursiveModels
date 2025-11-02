#!/usr/bin/env python3
"""
Real-world testing with realistic gift data
"""
import torch
import json
import numpy as np
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def load_realistic_data():
    """Load realistic gift catalog and user scenarios"""
    
    # Load gift catalog
    with open("data/realistic_gift_catalog.json", "r") as f:
        gift_data = json.load(f)
    
    # Load user scenarios
    with open("data/realistic_user_scenarios.json", "r") as f:
        scenario_data = json.load(f)
    
    return gift_data, scenario_data

def create_realistic_environment(gift_data):
    """Create environment with realistic gift catalog"""
    
    # Convert to expected format (list of gifts)
    realistic_catalog = []
    
    for gift in gift_data["gifts"]:
        realistic_catalog.append({
            "id": gift["id"],
            "name": gift["name"],
            "category": gift["category"],
            "price": gift["price"],
            "rating": gift["rating"],
            "tags": gift["tags"],
            "age_suitability": gift["age_range"],
            "occasion_fit": gift["occasions"],
            "description": f"{gift['name']} - {gift['category']} item with {gift['rating']}/5 rating"
        })
    
    # Save to expected location
    with open("data/realistic_gift_catalog_env.json", "w") as f:
        json.dump(realistic_catalog, f, indent=2)
    
    return GiftRecommendationEnvironment("data/realistic_gift_catalog_env.json")

def run_real_world_test():
    """Run comprehensive real-world test"""
    
    print("🌍 REAL-WORLD TESTING WITH REALISTIC DATA")
    print("=" * 60)
    
    # Load data
    gift_data, scenario_data = load_realistic_data()
    
    print(f"📦 Loaded {gift_data['metadata']['total_gifts']} realistic products")
    print(f"👥 Loaded {scenario_data['metadata']['total_scenarios']} user scenarios")
    print(f"💰 Price range: ${gift_data['metadata']['price_range']['min']:.2f} - ${gift_data['metadata']['price_range']['max']:.2f}")
    
    # Create realistic environment
    env = create_realistic_environment(gift_data)
    
    # Load production model
    print(f"\n🤖 Loading Production Model...")
    
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 64,
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 0,
        "L_layers": 1,
        "num_heads": 1,
        "expansion": 1.5,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "action_space_size": 5,
        "max_recommendations": 1,
        "value_head_hidden": 16,
        "policy_head_hidden": 16,
        "reward_prediction": False,
        "reward_head_hidden": 16,
        "max_tool_calls_per_step": 1,
        "tool_call_threshold": 0.01,
        "tool_result_encoding_dim": 32,
        "tool_selection_method": "confidence",
        "tool_fusion_method": "concatenate",
        "tool_usage_reward_weight": 1.0,
        "forward_dtype": "float32"
    }
    
    checkpoint = torch.load("checkpoints/production_tool_enhanced/production_model.pt", map_location="cpu")
    model = ToolEnhancedTRM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test each realistic scenario
    print(f"\n🧪 Testing Realistic User Scenarios:")
    print("-" * 60)
    
    results = []
    tool_usage_stats = {}
    category_recommendations = {}
    
    for i, scenario in enumerate(scenario_data["scenarios"], 1):
        print(f"\n👤 Scenario {i}: {scenario['name']}")
        
        profile_data = scenario["profile"]
        user = UserProfile(
            age=profile_data["age"],
            hobbies=profile_data["hobbies"],
            relationship=profile_data["relationship"],
            budget=profile_data["budget"],
            occasion=profile_data["occasion"],
            personality_traits=profile_data["preferences"]
        )
        
        print(f"   Age: {user.age}, Budget: ${user.budget}, Occasion: {user.occasion}")
        print(f"   Hobbies: {', '.join(user.hobbies)}")
        print(f"   Preferences: {', '.join(user.personality_traits)}")
        
        try:
            # Reset environment with user
            state = env.reset(user)
            
            # Create carry
            dummy_batch = {
                "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
            }
            carry = model.initial_carry(dummy_batch)
            
            with torch.no_grad():
                # Get model recommendations
                new_carry, rl_output, tool_calls = model.forward_with_tools(
                    carry, state, state.available_gifts, max_tool_calls=1
                )
                
                action = model.select_action(rl_output["action_probs"], state.available_gifts, deterministic=True)
                
                # Get environment feedback
                next_state, reward, done, info = env.step({
                    "recommendations": action["recommendations"],
                    "confidence_scores": action["confidence_scores"]
                })
                
                # Analyze recommendations
                recommended_gifts = []
                for gift_id in action["recommendations"]:
                    # Find gift in catalog (it's a list, not dict)
                    gift_found = None
                    for gift in env.gift_catalog:
                        if gift.id == gift_id:
                            gift_found = gift
                            break
                    
                    if gift_found:
                        recommended_gifts.append({
                            "id": gift_found.id,
                            "name": gift_found.name,
                            "category": gift_found.category,
                            "price": gift_found.price,
                            "rating": gift_found.rating
                        })
                        
                        # Track category recommendations
                        category = gift_found.category
                        category_recommendations[category] = category_recommendations.get(category, 0) + 1
                
                # Track tool usage
                tools_used = []
                for call in tool_calls:
                    if call.success:
                        tools_used.append(call.tool_name)
                        tool_usage_stats[call.tool_name] = tool_usage_stats.get(call.tool_name, 0) + 1
                
                # Evaluate against expectations
                expected_categories = set(scenario["expected_categories"])
                recommended_categories = set(gift["category"] for gift in recommended_gifts)
                category_match = len(expected_categories.intersection(recommended_categories)) > 0
                
                expected_tools = set(scenario["expected_tools"])
                used_tools = set(tools_used)
                tool_match = len(expected_tools.intersection(used_tools)) > 0
                
                # Store results
                result = {
                    "scenario": scenario["name"],
                    "user_profile": profile_data,
                    "reward": reward,
                    "recommended_gifts": recommended_gifts,
                    "tools_used": tools_used,
                    "category_match": category_match,
                    "tool_match": tool_match,
                    "expected_vs_actual": {
                        "expected_categories": list(expected_categories),
                        "recommended_categories": list(recommended_categories),
                        "expected_tools": list(expected_tools),
                        "used_tools": tools_used
                    }
                }
                results.append(result)
                
                # Print results
                print(f"   🎁 Recommendations:")
                for gift in recommended_gifts:
                    print(f"     • {gift['name']} (${gift['price']:.2f}) - {gift['category']}")
                
                print(f"   🛠️ Tools Used: {', '.join(tools_used) if tools_used else 'None'}")
                print(f"   📊 Reward: {reward:.3f}")
                print(f"   ✅ Category Match: {'Yes' if category_match else 'No'}")
                print(f"   🔧 Tool Match: {'Yes' if tool_match else 'No'}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e),
                "reward": 0.0
            })
    
    # Calculate comprehensive metrics
    print(f"\n📊 COMPREHENSIVE REAL-WORLD RESULTS")
    print("=" * 60)
    
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        # Performance metrics
        avg_reward = np.mean([r["reward"] for r in valid_results])
        reward_std = np.std([r["reward"] for r in valid_results])
        
        # Tool usage metrics
        total_tool_calls = sum(len(r["tools_used"]) for r in valid_results)
        tool_activation_rate = sum(1 for r in valid_results if len(r["tools_used"]) > 0) / len(valid_results)
        
        # Relevance metrics
        category_match_rate = sum(1 for r in valid_results if r["category_match"]) / len(valid_results)
        tool_match_rate = sum(1 for r in valid_results if r["tool_match"]) / len(valid_results)
        
        print(f"🎯 Performance Metrics:")
        print(f"  Average Reward: {avg_reward:.3f} ± {reward_std:.3f}")
        print(f"  Success Rate: {sum(1 for r in valid_results if r['reward'] > 0.1) / len(valid_results):.1%}")
        
        print(f"\n🛠️ Tool Usage Metrics:")
        print(f"  Tool Activation Rate: {tool_activation_rate:.1%}")
        print(f"  Total Tool Calls: {total_tool_calls}")
        print(f"  Tool Distribution:")
        for tool, count in sorted(tool_usage_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {tool}: {count} uses ({count/len(valid_results)*100:.1f}%)")
        
        print(f"\n🎯 Relevance Metrics:")
        print(f"  Category Match Rate: {category_match_rate:.1%}")
        print(f"  Tool Match Rate: {tool_match_rate:.1%}")
        
        print(f"\n📦 Category Distribution:")
        for category, count in sorted(category_recommendations.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} recommendations")
        
        # Overall assessment
        print(f"\n🏆 REAL-WORLD ASSESSMENT:")
        
        overall_score = (avg_reward * 0.4 + 
                        tool_activation_rate * 0.2 + 
                        category_match_rate * 0.3 + 
                        tool_match_rate * 0.1)
        
        print(f"  Overall Score: {overall_score:.3f}/1.000")
        
        if overall_score > 0.7:
            print(f"  🌟 EXCELLENT: Model performs very well on real-world data!")
        elif overall_score > 0.5:
            print(f"  ✅ GOOD: Model shows solid real-world performance")
        elif overall_score > 0.3:
            print(f"  ⚠️ FAIR: Model works but needs improvement")
        else:
            print(f"  ❌ POOR: Model needs significant improvement")
        
        # Detailed insights
        print(f"\n🔍 Key Insights:")
        
        if tool_activation_rate > 0.5:
            print(f"  • Tool integration is working well ({tool_activation_rate:.1%} activation)")
        else:
            print(f"  • Tool usage could be improved ({tool_activation_rate:.1%} activation)")
        
        if category_match_rate > 0.6:
            print(f"  • Recommendations are relevant to user interests")
        else:
            print(f"  • Recommendation relevance needs improvement")
        
        most_used_tool = max(tool_usage_stats.items(), key=lambda x: x[1])[0] if tool_usage_stats else None
        if most_used_tool:
            print(f"  • Most effective tool: {most_used_tool}")
        
        return {
            "overall_score": overall_score,
            "avg_reward": avg_reward,
            "tool_activation_rate": tool_activation_rate,
            "category_match_rate": category_match_rate,
            "results": valid_results
        }
    
    return None

if __name__ == "__main__":
    result = run_real_world_test()
    
    if result:
        print(f"\n🎉 Real-world testing completed successfully!")
        print(f"📊 Overall Performance: {result['overall_score']:.3f}/1.000")
    else:
        print(f"\n❌ Real-world testing failed")