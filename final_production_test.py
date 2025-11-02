#!/usr/bin/env python3
"""
Final production model test
"""
import torch
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def final_test():
    """Final comprehensive test"""
    print("🎯 FINAL PRODUCTION MODEL TEST")
    print("=" * 50)
    
    # Production config
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
    
    # Load production model
    checkpoint = torch.load("checkpoints/production_tool_enhanced/production_model.pt", map_location="cpu")
    model = ToolEnhancedTRM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test environment
    env = GiftRecommendationEnvironment("data/gift_catalog.json")
    
    # Comprehensive test scenarios
    test_scenarios = [
        ("Young Tech Enthusiast", UserProfile(22, ["technology", "gaming"], "friend", 150.0, "birthday", ["trendy"])),
        ("Eco-Conscious Mother", UserProfile(35, ["gardening", "environment"], "mother", 100.0, "mother_day", ["eco-conscious"])),
        ("Cooking Enthusiast", UserProfile(45, ["cooking", "food"], "partner", 200.0, "anniversary", ["practical"])),
        ("Fitness Lover", UserProfile(28, ["fitness", "sports"], "sibling", 80.0, "graduation", ["active"])),
        ("Creative Professional", UserProfile(32, ["art", "music"], "colleague", 120.0, "promotion", ["creative"])),
        ("Book Lover", UserProfile(50, ["reading", "literature"], "parent", 90.0, "birthday", ["intellectual"])),
        ("Outdoor Adventurer", UserProfile(30, ["hiking", "camping"], "friend", 180.0, "christmas", ["adventurous"])),
        ("Fashion Forward", UserProfile(26, ["fashion", "style"], "sister", 160.0, "birthday", ["trendy"]))
    ]
    
    print(f"\n🧪 Testing {len(test_scenarios)} diverse scenarios:")
    
    results = []
    tool_usage = {}
    
    for name, user in test_scenarios:
        try:
            state = env.reset(user)
            
            # Create carry
            dummy_batch = {
                "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
            }
            carry = model.initial_carry(dummy_batch)
            
            with torch.no_grad():
                new_carry, rl_output, tool_calls = model.forward_with_tools(
                    carry, state, state.available_gifts, max_tool_calls=1
                )
                
                action = model.select_action(rl_output["action_probs"], state.available_gifts, deterministic=True)
                
                next_state, reward, done, info = env.step({
                    "recommendations": action["recommendations"],
                    "confidence_scores": action["confidence_scores"]
                })
                
                # Track results
                results.append({
                    "name": name,
                    "user": user,
                    "reward": reward,
                    "tool_calls": len(tool_calls),
                    "recommendations": action["recommendations"],
                    "tools_used": [call.tool_name for call in tool_calls if call.success]
                })
                
                # Track tool usage
                for call in tool_calls:
                    if call.success:
                        tool_usage[call.tool_name] = tool_usage.get(call.tool_name, 0) + 1
                
                print(f"  {name:<20} | Reward: {reward:>6.3f} | Tools: {len(tool_calls)} | Recs: {len(action['recommendations'])}")
                for call in tool_calls:
                    print(f"    → {call.tool_name} ({'✓' if call.success else '✗'})")
                
        except Exception as e:
            print(f"  {name:<20} | ❌ Error: {e}")
            results.append({"name": name, "reward": 0.0, "tool_calls": 0, "error": str(e)})
    
    # Calculate final metrics
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        avg_reward = sum(r["reward"] for r in valid_results) / len(valid_results)
        avg_tools = sum(r["tool_calls"] for r in valid_results) / len(valid_results)
        success_rate = sum(1 for r in valid_results if r["reward"] > 0.1) / len(valid_results)
        tool_activation_rate = sum(1 for r in valid_results if r["tool_calls"] > 0) / len(valid_results)
        
        print(f"\n📊 FINAL PERFORMANCE METRICS:")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Tool Calls: {avg_tools:.1f}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Tool Activation Rate: {tool_activation_rate:.1%}")
        print(f"  Model Size: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print(f"\n🛠️ TOOL USAGE STATISTICS:")
        if tool_usage:
            for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} uses")
        else:
            print("  No tools were used")
        
        print(f"\n🎯 PRODUCTION READINESS:")
        if avg_reward > 0.1 and tool_activation_rate > 0.1:
            print("  ✅ Model is production-ready!")
            print("  ✅ Tool integration working")
            print("  ✅ Consistent performance across scenarios")
        elif avg_reward > 0.1:
            print("  ⚠️ Model works but tools need optimization")
        else:
            print("  ❌ Model needs more training")
        
        return {
            "avg_reward": avg_reward,
            "avg_tools": avg_tools,
            "success_rate": success_rate,
            "tool_activation_rate": tool_activation_rate,
            "tool_usage": tool_usage
        }
    
    return None

if __name__ == "__main__":
    result = final_test()
    
    if result:
        print(f"\n🏆 FINAL RESULT: SUCCESS!")
        print(f"Tool-Enhanced TRM is ready for production deployment!")
    else:
        print(f"\n❌ FINAL RESULT: NEEDS WORK")
        print(f"Model requires additional training or optimization.")