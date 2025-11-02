#!/usr/bin/env python3
"""
Test tool usage with different thresholds
"""
import torch
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def test_tool_thresholds():
    """Test different tool call thresholds"""
    print("🛠️ Testing Tool Usage with Different Thresholds")
    
    # Load trained model
    checkpoint = torch.load("checkpoints/tool_enhanced_gift_recommendation/phase3/best_model.pt", map_location="cpu")
    
    # Create model config (extracted from checkpoint)
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 128,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 1,
        "num_heads": 1,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.1,
        "action_space_size": 100,
        "max_recommendations": 2,
        "value_head_hidden": 256,
        "policy_head_hidden": 256,
        "reward_prediction": True,
        "reward_head_hidden": 128,
        "max_tool_calls_per_step": 3,
        "tool_call_threshold": 0.1,  # Much lower threshold
        "tool_result_encoding_dim": 128,
        "tool_selection_method": "confidence",
        "tool_fusion_method": "concatenate",
        "tool_usage_reward_weight": 0.1,
        "forward_dtype": "float32"
    }
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n📊 Testing threshold: {threshold}")
        
        # Update threshold
        config["tool_call_threshold"] = threshold
        
        # Create model
        model = ToolEnhancedTRM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Test environment
        env = GiftRecommendationEnvironment("data/gift_catalog.json")
        
        # Test user
        user = UserProfile(25, ["technology", "gaming"], "friend", 200.0, "christmas", ["trendy"])
        state = env.reset(user)
        
        # Create carry
        dummy_batch = {
            "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
        }
        carry = model.initial_carry(dummy_batch)
        
        # Test multiple steps
        total_tool_calls = 0
        for step in range(3):
            with torch.no_grad():
                new_carry, rl_output, tool_calls = model.forward_with_tools(
                    carry, state, state.available_gifts, max_tool_calls=3
                )
                total_tool_calls += len(tool_calls)
                
                print(f"  Step {step+1}: {len(tool_calls)} tool calls")
                for i, call in enumerate(tool_calls):
                    print(f"    Tool {i+1}: {call.tool_name} (confidence: {call.confidence:.3f})")
                
                carry = new_carry
        
        print(f"  Total tool calls: {total_tool_calls}")
        
        # Get tool usage stats
        stats = model.get_tool_usage_stats()
        print(f"  Success rate: {stats.get('success_rates', {})}")

if __name__ == "__main__":
    test_tool_thresholds()